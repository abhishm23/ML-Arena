"""ML Arena — turning a business question into a target we can actually learn.

This module is the data-science heart of the product. The user has told us *what*
they want to find and *which columns* mean what; everything between that and a
clean modelling table happens here.

Two ideas run through the whole file:

1. **Compare each row with its own expectation.** For "discount leakage" and
   "margin overflow" we do not flag rows just because a number is big. We first
   learn what a normal discount looks like *for that kind of order* — same segment,
   same product, same channel, same size — and only then flag the rows that beat
   their own expectation by a wide margin. A 40% discount on a clearance pallet is
   not leakage; a 40% discount on a full-price repeat order is.

2. **Say what we did, in the user's language.** Every transformation appends a
   sentence to `notes`. Those sentences are streamed to the user's screen while the
   run is happening, so they have to be true, specific and free of jargon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np

from mlarena import config  # noqa: F401  -- env bootstrap before polars
import polars as pl

from mlarena.engine import models as model_zoo
from mlarena.usecases.registry import UseCase

SEED = config.DEFAULT_SEED


# --------------------------------------------------------------------------- #
# TargetPlan (contract §7)
# --------------------------------------------------------------------------- #


@dataclass
class TargetPlan:
    task: str
    frame: pl.DataFrame
    target_col: str | None
    feature_cols: list[str]
    numeric_cols: list[str]
    categorical_cols: list[str]
    label_names: dict | None
    positive_rate: float | None
    impact_col: str | None
    impact_label: str
    score_direction: str
    key_cols: list[str]
    notes: list[str]
    extra: dict = field(default_factory=dict)
    baseline: str | None = None


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

_CURRENCY_JUNK = r"[\$€£¥₹,%\s\u00a0]"


def _note(notes: list[str], text: str) -> None:
    if text and text not in notes:
        notes.append(text)


def _fmt(n) -> str:
    try:
        if isinstance(n, np.ndarray):
            n = float(np.nanmedian(n))
        n = float(n)
    except Exception:
        return str(n)
    return f"{int(n):,}" if n.is_integer() else f"{n:,.2f}"


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _pick(mapping: dict, key: str) -> str | None:
    vals = mapping.get(key) or []
    return vals[0] if vals else None


def _pick_all(mapping: dict, key: str) -> list[str]:
    vals = mapping.get(key) or []
    seen: list[str] = []
    for v in vals:
        if v and v not in seen:
            seen.append(v)
    return seen


def _is_datetime_dtype(dtype) -> bool:
    return dtype in (pl.Date,) or isinstance(dtype, (pl.datatypes.Datetime, pl.datatypes.Date))


def to_float(s: pl.Series) -> pl.Series | None:
    """Best-effort numeric coercion. Handles '$1,234.50', '22%', ' 12 '."""
    try:
        if s.dtype.is_numeric():
            return s.cast(pl.Float64)
        if s.dtype == pl.Boolean:
            return s.cast(pl.Int8).cast(pl.Float64)
        if _is_datetime_dtype(s.dtype) or s.dtype in (pl.Duration, pl.Time):
            return None
        if s.dtype in (pl.String, pl.Categorical, pl.Enum):
            cleaned = s.cast(pl.String).str.replace_all(_CURRENCY_JUNK, "")
            out = cleaned.cast(pl.Float64, strict=False)
            good = out.is_not_null().sum()
            if good == 0 or (good / max(1, s.len())) < 0.6:
                return None
            return out
        return s.cast(pl.Float64, strict=False)
    except Exception:
        return None


def to_datetime(s: pl.Series) -> pl.Series | None:
    """Best-effort date coercion, ISO first then a few common business formats."""
    try:
        if _is_datetime_dtype(s.dtype):
            return s.cast(pl.Datetime("us"))
        if s.dtype.is_numeric():
            return None
        text = s.cast(pl.String).str.strip_chars()
        for fmt in (None, "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
                    "%d %b %Y", "%b %d, %Y", "%Y%m%d", "%Y-%m"):
            try:
                out = (text.str.to_datetime(format=fmt, strict=False, exact=False)
                       if fmt else text.str.to_datetime(strict=False))
            except Exception:
                continue
            good = out.is_not_null().sum()
            if good and (good / max(1, s.len())) >= 0.7:
                return out.cast(pl.Datetime("us"))
        return None
    except Exception:
        return None


def _cap_cardinality(s: pl.Series, top: int = 30) -> tuple[pl.Series, int]:
    """Keep the `top` most common values, fold the long tail into 'Other'."""
    vc = s.value_counts(sort=True)
    keep = vc.head(top).get_column(s.name).to_list()
    out = pl.when(s.is_in(keep)).then(s).otherwise(pl.lit("Other")).alias(s.name)
    n_folded = int(s.n_unique() - len(keep))
    return out, max(0, n_folded)


def _safe_div(a: pl.Series, b: pl.Series) -> pl.Series:
    return pl.when((b.is_null()) | (b == 0)).then(None).otherwise(a / b)


# --------------------------------------------------------------------------- #
# Shared cleaning / feature building
# --------------------------------------------------------------------------- #


def _clean_numeric(frame: pl.DataFrame, col: str, notes: list[str],
                   *, label: str = "number") -> tuple[pl.DataFrame, bool]:
    """Coerce one column to Float64 in place. Returns (frame, kept)."""
    s = to_float(frame.get_column(col))
    if s is None:
        _note(notes, f"We skipped “{col}” because its values are not numbers we could read.")
        return frame, False
    n = frame.height
    nulls = s.null_count()
    if nulls == n:
        _note(notes, f"We skipped “{col}” because it is empty on every row.")
        return frame, False
    if nulls / max(1, n) > 0.7:
        _note(notes, f"We skipped “{col}” because it is blank on more than 7 rows in 10.")
        return frame, False
    if s.n_unique() <= 1:
        _note(notes, f"We skipped “{col}” because it holds the same value on every row, "
                     f"so it cannot explain anything.")
        return frame, False
    if not frame.get_column(col).dtype.is_numeric():
        _note(notes, f"We read “{col}” as a {label} even though it was stored as text.")
    return frame.with_columns(s.alias(col)), True


def _clean_categorical(frame: pl.DataFrame, col: str, notes: list[str],
                       *, top: int = 30) -> tuple[pl.DataFrame, bool]:
    s = frame.get_column(col)
    if _is_datetime_dtype(s.dtype):
        return frame, False
    if s.null_count() == frame.height:
        _note(notes, f"We skipped “{col}” because it is empty on every row.")
        return frame, False
    try:
        s = s.cast(pl.String)
    except Exception:
        return frame, False
    distinct = s.n_unique()
    if distinct <= 1:
        _note(notes, f"We skipped “{col}” because it never changes from row to row.")
        return frame, False
    if distinct > top:
        s, folded = _cap_cardinality(s, top)
        _note(notes, f"“{col}” had {distinct:,} different values. We kept the {top} most "
                     f"common ones and grouped the other {folded:,} together as “Other”, "
                     f"so the model is not distracted by one-off entries.")
    return frame.with_columns(s.alias(col)), True


def _add_date_features(frame: pl.DataFrame, col: str, notes: list[str],
                       *, label: str = "date") -> tuple[pl.DataFrame, list[str], str | None]:
    """Turn one date column into a handful of plain numeric columns."""
    s = to_datetime(frame.get_column(col))
    if s is None:
        _note(notes, f"We could not read “{col}” as {label}s, so we left it out of the analysis.")
        return frame, [], None
    good = s.is_not_null().sum()
    if good < 2 or s.n_unique() <= 1:
        _note(notes, f"We skipped “{col}” because it does not contain a usable range of {label}s.")
        return frame, [], None
    derived = [f"{col} year", f"{col} month", f"{col} day of week", f"{col} day of month",
               f"{col} days since start", f"{col} weekend"]
    epoch_days = s.dt.epoch("d")
    base_day = int(epoch_days.min() or 0)
    frame = frame.with_columns([
        s.dt.year().cast(pl.Float64).alias(derived[0]),
        s.dt.month().cast(pl.Float64).alias(derived[1]),
        s.dt.weekday().cast(pl.Float64).alias(derived[2]),
        s.dt.day().cast(pl.Float64).alias(derived[3]),
        (epoch_days - base_day).cast(pl.Float64).alias(derived[4]),
        s.dt.weekday().is_in([6, 7]).cast(pl.Float64).alias(derived[5]),
    ])
    lo, hi = s.min(), s.max()
    span = (hi - lo).days if lo is not None and hi is not None else 0
    _note(notes, f"We turned “{col}” into six simple numbers (year, month, weekday, day, "
                 f"days since the first record, and a weekend flag) so the models can use "
                 f"the calendar without being confused by dates. Your data covers "
                 f"{_fmt(span)} days.")
    return frame, derived, col


def _drop_leaky(frame: pl.DataFrame, target: pl.Series, num_cols: list[str],
                notes: list[str], *, threshold: float = 0.995) -> list[str]:
    """Drop numeric columns that move in lockstep with the target."""
    keep: list[str] = []
    y = target.to_numpy().astype(float)
    finite = np.isfinite(y)
    for c in num_cols:
        try:
            x = frame.get_column(c).to_numpy().astype(float)
        except Exception:
            continue
        ok = finite & np.isfinite(x)
        if ok.sum() > 30 and np.nanstd(x[ok]) > 0 and np.nanstd(y[ok]) > 0:
            r = abs(float(np.corrcoef(x[ok], y[ok])[0, 1]))
            if r >= threshold:
                _note(notes, f"We left “{c}” out of the prediction because it moves in "
                             f"lockstep with what you asked us to predict — including it "
                             f"would make the model look brilliant and tell you nothing.")
                continue
        keep.append(c)
    return keep


def _build_features(frame: pl.DataFrame, numeric_src: list[str], cat_src: list[str],
                    date_src: list[str], notes: list[str], *, top_card: int = 30
                    ) -> tuple[pl.DataFrame, list[str], list[str], list[str]]:
    """Coerce, clean and de-duplicate the columns we will actually model on."""
    num_final: list[str] = []
    cat_final: list[str] = []
    date_kept: list[str] = []
    seen: set[str] = set()

    for c in numeric_src:
        if c in seen or c not in frame.columns:
            continue
        seen.add(c)
        frame, ok = _clean_numeric(frame, c, notes)
        if ok:
            num_final.append(c)

    for c in cat_src:
        if c in seen or c not in frame.columns:
            continue
        seen.add(c)
        frame, ok = _clean_categorical(frame, c, notes, top=top_card)
        if ok:
            cat_final.append(c)

    for c in date_src:
        if c in seen or c not in frame.columns:
            continue
        seen.add(c)
        frame, derived, kept = _add_date_features(frame, c, notes)
        if kept:
            date_kept.append(kept)
            num_final.extend(derived)

    # a constant derived column is useless too
    for c in list(num_final):
        if c in frame.columns and frame.get_column(c).drop_nulls().n_unique() <= 1:
            num_final.remove(c)
    return frame, num_final, cat_final, date_kept


def _pandas_features(frame: pl.DataFrame, num_cols: list[str], cat_cols: list[str]):
    """Column-subset to pandas — never the whole frame if we can avoid it."""
    cols = list(dict.fromkeys(list(num_cols) + list(cat_cols)))
    if not cols:
        return None
    pdf = frame.select(cols).to_pandas()
    for c in cat_cols:
        if c in pdf.columns:
            pdf[c] = pdf[c].astype("object").where(pdf[c].notna(), None)
    return pdf


def _predict_in_batches(pipe, pdf, batch: int = 150_000) -> np.ndarray:
    if pdf is None or len(pdf) == 0:
        return np.zeros(0)
    if len(pdf) <= batch:
        return np.asarray(pipe.predict(pdf), dtype=float)
    out = np.empty(len(pdf), dtype=float)
    for start in range(0, len(pdf), batch):
        stop = min(start + batch, len(pdf))
        out[start:stop] = np.asarray(pipe.predict(pdf.iloc[start:stop]), dtype=float)
    return out


def _expected_model(seed: int):
    """A small, fast regressor used to work out 'what should this row have looked like?'"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=180, learning_rate=0.08, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.12, n_iter_no_change=12,
        l2_regularization=1.0, random_state=seed,
    )


def _expectation(frame: pl.DataFrame, num_cols: list[str], cat_cols: list[str],
                 y: np.ndarray, seed: int, notes: list[str], *,
                 max_fit: int = 80_000, what: str = "value"
                 ) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit expected-<what> from context. Returns (expected, holdout_mask) or None.

    We fit on a random 80% and measure how noisy the leftovers are on the held-out
    20%. Rows that were used for fitting are never *more* likely to be flagged, so
    the held-out error is the honest yardstick to use.
    """
    n = len(y)
    pdf = _pandas_features(frame, num_cols, cat_cols)
    if pdf is None or not num_cols + cat_cols:
        return None
    if n < 120:
        _note(notes, f"There were only {n:,} rows, which is too few to learn a reliable "
                     f"expected {what} from. We used a simple rule instead.")
        return None

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_fit = min(max_fit, max(50, int(n * 0.8)))
    fit_idx = idx[:n_fit]
    hold_mask = np.ones(n, dtype=bool)
    hold_mask[fit_idx] = False
    if hold_mask.sum() < 30:                     # tiny file: use a real split
        hold_mask = np.zeros(n, dtype=bool)
        hold_mask[idx[n_fit:]] = True
        if hold_mask.sum() < 10:
            hold_mask = np.ones(n, dtype=bool)
    try:
        pipe = model_zoo.make_pipeline(_expected_model(seed), num_cols, cat_cols, False)
        pipe.fit(pdf.iloc[fit_idx], y[fit_idx])
        expected = _predict_in_batches(pipe, pdf)
    except Exception:
        _note(notes, f"We could not learn an expected {what} from the other columns, so we "
                     f"used a simple rule instead.")
        return None
    return expected, hold_mask


# --------------------------------------------------------------------------- #
# Strategy context
# --------------------------------------------------------------------------- #


class _Ctx:
    """Bundles the user's mapping with the notes we are accumulating."""

    def __init__(self, dataset_id: str, uc: UseCase, mapping: dict, options: dict):
        self.dataset_id = dataset_id
        self.uc = uc
        self.mapping = mapping or {}
        self.options = options or {}
        self.notes: list[str] = []
        self.key_cols: list[str] = []
        self.seed = int(self.options.get("seed", SEED))

    # -- column lookup ------------------------------------------------------ #
    def one(self, role: str) -> str | None:
        return _pick(self.mapping, role)

    def many(self, role: str) -> list[str]:
        return _pick_all(self.mapping, role)

    def require(self, role: str, frame: pl.DataFrame) -> str:
        col = self.one(role)
        if not col:
            r = next((x for x in self.uc.roles if x.key == role), None)
            label = r.label if r else role
            raise ValueError(
                f"This question needs a column for “{label}”. Pick one and run it again."
            )
        if col not in frame.columns:
            raise ValueError(
                f"The column “{col}” you chose for “{role}” is not in this dataset any more."
            )
        return col

    def add_keys(self, cols: list[str]) -> None:
        for c in cols:
            if c and c not in self.key_cols:
                self.key_cols.append(c)

    def note(self, text: str) -> None:
        _note(self.notes, text)


def _get_frame(dataset_id: str) -> pl.DataFrame:
    from mlarena.dataio import loader  # imported lazily: agent C owns this module

    return loader.store.get(dataset_id)


# --------------------------------------------------------------------------- #
# Residual framing (shared by discount leakage and margin overflow)
# --------------------------------------------------------------------------- #


def _residual_plan(ctx: _Ctx, frame: pl.DataFrame, *, value_col: str, context_num: list[str],
                   context_cat: list[str], context_date: list[str], money_col: str | None,
                   label_names: dict, subject: str,
                   direction: str, floor_rule=None, value_as_feature: bool = True) -> TargetPlan:
    """Flag rows whose value is far worse than its own context predicts.

    `value_as_feature` is deliberate and important. Knowing that a discount is large
    is not enough to call it leakage — the model has to work out whether it was large
    *for that kind of order*. Leaving the value out entirely would make the label
    (which is defined as "worse than expected") impossible to learn, and the whole
    race would come back at coin-toss accuracy on data where we had already found
    thousands of real problems. So the model sees the value and has to reconstruct
    the expectation itself from the context.
    """
    notes = ctx.notes
    frame, num_cols, cat_cols, date_cols = _build_features(
        frame, context_num, context_cat, context_date, notes)

    # ---- degenerate: nothing to compare against --------------------------- #
    value_series = frame.get_column(value_col)
    n_distinct = value_series.n_unique()
    if n_distinct <= 1:
        _note(notes, f"Every row has the same {subject} ({value_series[0]}), so there is no "
                     f"pattern to learn and nothing stands out. We are reporting zero "
                     f"flagged rows — this file cannot answer this question.")
        frame = frame.with_columns(pl.lit(0, dtype=pl.Int8).alias("_flag"))
        return TargetPlan(
            task="classification", frame=frame, target_col="_flag",
            feature_cols=[], numeric_cols=[], categorical_cols=[], label_names=label_names,
            positive_rate=0.0, impact_col=money_col, impact_label=ctx.uc.impact_label,
            score_direction=direction, key_cols=ctx.key_cols, notes=notes,
            extra={"degenerate": True},
        )

    y = frame.get_column(value_col).to_numpy().astype(float)
    finite = np.isfinite(y)
    if finite.sum() < len(y):
        fill = float(np.nanmedian(y[finite])) if finite.any() else 0.0
        y = np.where(finite, y, fill)
        frame = frame.with_columns(pl.Series(value_col, y))
        _note(notes, f"{int((~finite).sum()):,} rows had a blank or unreadable {subject}. "
                     f"We filled those with the middle value so the maths still works; they "
                     f"are not flagged as problems.")

    iqr = float(np.subtract(*np.percentile(y, [75, 25]))) if len(y) else 0.0

    # ---- can we learn an expectation? ------------------------------------- #
    usable_context = [c for c in num_cols + cat_cols if c != value_col]
    expected = None
    hold_mask = None
    if usable_context:
        got = _expectation(frame, [c for c in num_cols if c != value_col], cat_cols, y,
                           ctx.seed, notes, what=subject)
        if got is not None:
            expected, hold_mask = got

    if expected is None:
        # ---- fallback: the simple rule, said out loud --------------------- #
        cut = float(np.quantile(y, 0.90))
        flag = (y > cut).astype(np.int8)
        _note(notes, f"We could not find other columns that explain the {subject}, so we "
                     f"fell back to a simple rule: flag the largest 10% of {subject} values "
                     f"(above {_fmt(cut)}). Treat this as a shortlist for a human look, not "
                     f"as a proven finding.")
        frame = frame.with_columns(pl.Series("_flag", flag))
        feature_cols = list(dict.fromkeys(num_cols + cat_cols))
        numeric_features = [c for c in num_cols if c != value_col]
        if value_as_feature:
            feature_cols = list(dict.fromkeys(feature_cols + [value_col]))
            numeric_features = numeric_features + [value_col]
        return TargetPlan(
            task="classification", frame=frame, target_col="_flag",
            feature_cols=feature_cols,
            numeric_cols=numeric_features,
            categorical_cols=cat_cols, label_names=label_names,
            positive_rate=float(flag.mean()), impact_col=money_col,
            impact_label=ctx.uc.impact_label, score_direction=direction,
            key_cols=ctx.key_cols, notes=notes,
            extra={"fallback": "top_decile", "threshold": cut},
        )

    residual = y - expected

    if hold_mask is None or hold_mask.sum() < 20:
        hold_mask = np.ones(len(y), dtype=bool)
    resid_std = float(np.std(residual[hold_mask]))
    iqr_guard = 0.15 * iqr
    threshold = max(2.0 * resid_std, iqr_guard)
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = max(1e-9, 0.05 * float(np.std(y)) or 1e-9)
        _note(notes, f"The {subject} barely varies in this file, so we used a very tight "
                     f"tolerance to separate the unusual rows.")

    _note(notes, f"Step 1 — we learned what a normal {subject} looks like for each kind of "
                 f"row, using {len(usable_context)} other columns as evidence.")
    if direction == "higher_is_risk":
        _note(notes, f"Step 2 — we compared every row with its own expectation. A row is flagged "
                     f"only when its {subject} is more than {_fmt(threshold)} above what we "
                     f"expected for a row like it.")
    else:
        _note(notes, f"Step 2 — we compared every row with its own expectation. A row is flagged "
                     f"only when its {subject} is more than {_fmt(threshold)} below what we "
                     f"expected for a row like it.")
    if iqr_guard > 2.0 * resid_std:
        _note(notes, f"The tolerance came from the natural spread of your {subject} values "
                     f"(a quarter of your rows sit within {_fmt(iqr)} of each other), because "
                     f"the pattern was too noisy to trust on its own.")
    else:
        _note(notes, f"The tolerance came from how far our expectations usually miss by "
                     f"(about {_fmt(resid_std)}), with a safety margin of two misses.")

    if direction == "higher_is_risk":
        flag = (residual > threshold).astype(np.int8)
        _note(notes, "Rows flagged: those receiving far more than their situation justifies.")
    else:
        flag = (residual < -threshold).astype(np.int8)
        _note(notes, "Rows flagged: those earning far less than their situation justifies.")

    # ---- optional second rule (e.g. a policy floor) ----------------------- #
    if floor_rule is not None:
        floor, floor_col, floor_desc = floor_rule(frame, y, cat_cols, ctx)
        if floor is not None:
            extra_flag = (y < floor).astype(np.int8)
            before = int(flag.sum())
            flag = np.maximum(flag, extra_flag)
            _note(notes, f"We also flagged {int(flag.sum()) - before:,} rows that fell below "
                         f"the floor of {_fmt(floor)} — {floor_desc}.")

    rate = float(flag.mean()) if len(flag) else 0.0
    if rate == 0.0:
        _note(notes, "Nothing in this file beat its own expectation by a wide enough margin. "
                     "That is a good sign — it means no obvious problem of this kind.")
    elif rate > 0.5:
        _note(notes, f"More than half the rows ({_pct(rate)}) came out unusual, which means "
                     f"the tolerance is too tight to be useful. Widen the picture by adding "
                     f"more context columns (segment, product, channel) and run it again.")

    frame = frame.with_columns(pl.Series("_flag", flag))
    feature_cols = list(dict.fromkeys(num_cols + cat_cols))
    if value_as_feature:
        feature_cols = list(dict.fromkeys(feature_cols + [value_col]))
        _note(notes, f"The models also get to see “{value_col}” itself. That sounds like cheating "
                     f"but it is not: a big {subject} is only a problem if it is big *for that "
                     f"kind of row*, so the model still has to work out what was reasonable "
                     f"before it can tell the difference.")
    numeric_features = [c for c in num_cols if c != value_col]
    if value_as_feature:
        numeric_features = numeric_features + [value_col]
    return TargetPlan(
        task="classification", frame=frame, target_col="_flag",
        feature_cols=feature_cols,
        numeric_cols=numeric_features,
        categorical_cols=cat_cols, label_names=label_names,
        positive_rate=rate, impact_col=money_col, impact_label=ctx.uc.impact_label,
        score_direction=direction, key_cols=ctx.key_cols, notes=notes,
        extra={"threshold": threshold, "residual_std": resid_std, "iqr": iqr,
               "expected_mean": float(np.mean(expected)), "date_cols": date_cols},
    )


# --------------------------------------------------------------------------- #
# 1. discount leakage
# --------------------------------------------------------------------------- #


def _discount_leakage(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    raw = frame

    discount_cols = [c for c in ctx.many("discount") if c in raw.columns]
    if not discount_cols:
        raise ValueError("Pick the discount column you want to police and run it again.")
    # Prefer a percentage-style column when the user gave us both.
    primary = next((c for c in discount_cols if "pct" in c.lower() or "percent" in c.lower()
                    or "rate" in c.lower()), discount_cols[0])
    if len(discount_cols) > 1:
        _note(notes, f"You gave us {len(discount_cols)} discount columns. We watched “{primary}” "
                     f"and kept the others aside, because one discount column can give away "
                     f"the answer to another.")
    if primary != discount_cols[0]:
        _note(notes, f"“{primary}” looks like a percentage, so we used that as the one to police.")

    money_col = ctx.one("money")
    if money_col and money_col not in raw.columns:
        money_col = None

    frame, ok = _clean_numeric(raw, primary, notes, label="discount")
    if not ok:
        raise ValueError(
            f"We could not read “{primary}” as a number, so we cannot police it. Check the "
            f"column and try again."
        )

    # context: everything the user pointed at, except any other discount column
    context_num: list[str] = []
    context_cat: list[str] = []
    for role in ("quantity", "money"):
        for c in ctx.many(role):
            if c in frame.columns and c != primary:
                context_num.append(c)
    for role in ("customer", "segment", "product", "channel"):
        for c in ctx.many(role):
            if c in frame.columns and c != primary:
                context_cat.append(c)
    context_date = [c for c in ctx.many("date") if c in frame.columns]

    # anything else numeric/dimensional the user did not name is still useful context
    extra_num, extra_cat = _auto_context(frame, exclude=set(discount_cols) | {primary})
    context_num.extend(extra_num)
    context_cat.extend(extra_cat)

    if money_col:
        ctx.add_keys([money_col])
    key_roles = ("customer", "segment", "product", "channel")
    key_candidates: list[str] = []
    for r in key_roles:
        key_candidates.extend(ctx.many(r))
    ctx.add_keys(key_candidates[:3])
    ctx.add_keys(context_date[:1])

    _note(notes, f"We are policing “{primary}”. Every row keeps its own identity columns so "
                 f"you can find it again in your system.")
    if not context_num and not context_cat:
        _note(notes, "You did not give us any context columns (segment, product, channel, "
                     "quantity), so we have nothing to compare each order against.")

    return _residual_plan(
        ctx, frame, value_col=primary, context_num=context_num, context_cat=context_cat,
        context_date=context_date, money_col=money_col,
        label_names={0: "Within expectation", 1: "Discount leakage"},
        subject="discount",
        direction="higher_is_risk",
    )


def _auto_context(frame: pl.DataFrame, *, exclude: set[str], limit: int = 24
                  ) -> tuple[list[str], list[str]]:
    """Columns the user did not name but that can still explain the row."""
    num: list[str] = []
    cat: list[str] = []
    for c, dt in zip(frame.columns, frame.dtypes):
        if c in exclude or c.startswith("_"):
            continue
        if dt.is_numeric():
            if c not in num:
                num.append(c)
        elif not _is_datetime_dtype(dt) and dt not in (pl.Duration, pl.Time):
            try:
                if 1 < frame.get_column(c).n_unique() <= 200:
                    cat.append(c)
            except Exception:
                continue
    return num[:limit], cat[:limit]


# --------------------------------------------------------------------------- #
# 2. margin overflow
# --------------------------------------------------------------------------- #


def _margin_overflow(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    raw = frame

    margin_cols = [c for c in ctx.many("margin") if c in raw.columns]
    if not margin_cols:
        raise ValueError("Pick the margin column you want to watch and run it again.")
    primary = next((c for c in margin_cols if "pct" in c.lower() or "percent" in c.lower()
                    or "rate" in c.lower()), margin_cols[0])

    money_col = ctx.one("money")
    if money_col and money_col not in raw.columns:
        money_col = None
    cost_col = ctx.one("cost")
    if cost_col and cost_col not in raw.columns:
        cost_col = None

    frame, ok = _clean_numeric(raw, primary, notes, label="margin")
    if not ok:
        raise ValueError(f"We could not read “{primary}” as a number, so we cannot judge it.")

    # If we have revenue and cost we can sanity-check the margin the user calculated.
    if money_col and cost_col:
        rev = to_float(frame.get_column(money_col))
        cst = to_float(frame.get_column(cost_col))
        if rev is not None and cst is not None:
            derived = _safe_div(rev - cst, rev)
            check = frame.with_columns(derived.alias("_margin_check"))
            got = check.get_column("_margin_check").drop_nulls()
            if got.len() > 10:
                as_pct = got.mean() * 100.0
                stored = frame.get_column(primary)
                scale = 100.0 if float(stored.abs().median() or 0) > 1.5 else 1.0
                diff = float((got.mean() * scale) - float(stored.median()))
                _note(notes, f"We checked “{primary}” against revenue minus cost. They agree "
                             f"to within {abs(diff):.2f} points on a typical row"
                             if abs(diff) < max(2.0, 0.05 * abs(as_pct))
                             else
                             f"Heads up: “{primary}” and (revenue − cost) ÷ revenue disagree "
                             f"by about {abs(diff):.2f} points on a typical row. We used your "
                             f"“{primary}” column as the source of truth.")

    context_num: list[str] = []
    context_cat: list[str] = []
    for role in ("quantity",):
        context_num.extend(c for c in ctx.many(role) if c in frame.columns)
    for role in ("segment", "product", "channel", "customer"):
        context_cat.extend(c for c in ctx.many(role) if c in frame.columns)
    context_date = [c for c in ctx.many("date") if c in frame.columns]

    # revenue and cost are part of the context (a big order earns a different margin),
    # but they are never the answer we are looking for.
    for c in (money_col, cost_col):
        if c and c in frame.columns:
            context_num.append(c)

    extra_num, extra_cat = _auto_context(frame, exclude={primary})
    context_num.extend(extra_num)
    context_cat.extend(extra_cat)

    segment_col = _pick(ctx.mapping, "segment")
    if segment_col and segment_col not in frame.columns:
        segment_col = None

    def _floor_rule(fr: pl.DataFrame, y: np.ndarray, cat_cols: list[str], c: _Ctx):
        """Policy floor = the 25th percentile of margin in the row's own segment."""
        if segment_col and segment_col in fr.columns:
            tmp = fr.select([segment_col]).with_columns(pl.Series("_m", y))
            stats = tmp.group_by(segment_col).agg(
                pl.col("_m").quantile(0.25).alias("floor"),
                pl.len().alias("n"),
            ).filter(pl.col("n") >= 20)
            if stats.height:
                joined = tmp.join(stats.select([segment_col, "floor"]), on=segment_col,
                                  how="left")
                floor_vals = joined.get_column("floor").to_numpy()
                if np.isfinite(floor_vals).sum() >= 20:
                    fill = float(np.nanquantile(y, 0.25))
                    floor_vals = np.where(np.isfinite(floor_vals), floor_vals, fill)
                    return (np.maximum(0.0, floor_vals) if float(np.nanmedian(y)) > 0
                            else floor_vals), None, (
                        f"the typical low-water margin for that row's “{segment_col}” group")
        return max(0.0, float(np.quantile(y, 0.25))), None, (
            "the typical low-water margin across the whole file")

    for c in ([money_col] if money_col else []) + ([segment_col] if segment_col else []):
        ctx.add_keys([c])
    ctx.add_keys(context_date[:1])

    _note(notes, f"We are watching “{primary}”. First we learn the margin a row of this kind "
                 f"should have earned, then we look for rows that fell well short of it.")

    return _residual_plan(
        ctx, frame, value_col=primary, context_num=context_num, context_cat=context_cat,
        context_date=context_date, money_col=money_col,
        label_names={0: "On policy", 1: "Below policy floor"},
        subject="margin",
        direction="lower_is_risk", floor_rule=_floor_rule,
    )


# --------------------------------------------------------------------------- #
# 3. revenue predict
# --------------------------------------------------------------------------- #


def _revenue_predict(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    target = ctx.require("target", frame)
    frame, ok = _clean_numeric(frame, target, notes, label="money column")
    if not ok:
        raise ValueError(f"We could not read “{target}” as a number, so there is nothing to predict.")

    ctx.add_keys(ctx.many("customer")[:1] + ctx.many("product")[:1] + ctx.many("date")[:1])
    _note(notes, f"“{target}” is the number we are trying to predict. Everything else you "
                 f"pointed at becomes evidence, and we will rank that evidence at the end.")

    chosen_num: list[str] = []
    chosen_cat: list[str] = []
    for role in ("drivers",):
        for c in ctx.many(role):
            if c in frame.columns and c != target:
                if frame.get_column(c).dtype.is_numeric():
                    chosen_num.append(c)
                else:
                    chosen_cat.append(c)
    for role in ("product", "segment", "channel", "customer"):
        for c in ctx.many(role):
            if c in frame.columns and c != target and c not in chosen_num + chosen_cat:
                chosen_cat.append(c)
    date_cols = [c for c in ctx.many("date") if c in frame.columns]

    if not chosen_num and not chosen_cat and not date_cols:
        auto_num, auto_cat = _auto_context(frame, exclude={target})
        chosen_num.extend(auto_num)
        chosen_cat.extend(auto_cat)
        _note(notes, f"You did not pick specific evidence columns, so we used all "
                     f"{len(auto_num) + len(auto_cat)} other usable columns in the file.")

    frame, num_cols, cat_cols, _dates = _build_features(
        frame, chosen_num, chosen_cat, date_cols, notes)
    num_cols = [c for c in num_cols if c != target]

    y = frame.get_column(target).to_numpy().astype(float)
    num_cols = _drop_leaky(frame, pl.Series(y), num_cols, notes)
    cat_cols = [c for c in cat_cols if c != target]

    if not num_cols and not cat_cols:
        raise ValueError(
            "There is nothing left to predict from — every other column was empty, constant "
            "or a duplicate of what you asked us to predict. Add a few more columns and retry."
        )

    _note(notes, f"{len(num_cols) + len(cat_cols)} columns went into the race as evidence. "
                 f"We will hold back 1 row in 5 to check whether the model really works.")
    return TargetPlan(
        task="regression", frame=frame, target_col=target,
        feature_cols=list(dict.fromkeys(num_cols + cat_cols)),
        numeric_cols=num_cols, categorical_cols=cat_cols, label_names=None,
        positive_rate=None, impact_col=target, impact_label=ctx.uc.impact_label,
        score_direction="higher_is_good", key_cols=ctx.key_cols, notes=notes,
        extra={},
    )


# --------------------------------------------------------------------------- #
# 4. demand forecast
# --------------------------------------------------------------------------- #


def _season_length(dates: pl.Series) -> tuple[int, str]:
    d = dates.drop_nulls().sort()
    if d.len() < 3:
        return 7, "weeks"
    gaps = d.diff().dt.total_days().drop_nulls()
    med = float(gaps.median() or 1.0)
    if med <= 1.5:
        return 7, "weeks"          # daily data -> weekly rhythm
    if med <= 8:
        return 52, "years"         # weekly data -> yearly rhythm
    if med <= 45:
        return 12, "years"         # monthly data -> yearly rhythm
    return 4, "cycles"


def _demand_forecast(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    target = ctx.require("target", frame)
    date_col = ctx.require("date", frame)
    frame, ok = _clean_numeric(frame, target, notes, label="demand number")
    if not ok:
        raise ValueError(f"We could not read “{target}” as a number, so there is nothing to forecast.")

    dt = to_datetime(frame.get_column(date_col))
    if dt is None or dt.n_unique() <= 1:
        raise ValueError(
            f"We need a usable date column to forecast with, and “{date_col}” does not look "
            f"like one."
        )
    frame = frame.with_columns(dt.alias(date_col))

    series_col = _pick(ctx.mapping, "series")
    if series_col and series_col not in frame.columns:
        series_col = None
    if series_col:
        frame = frame.with_columns(frame.get_column(series_col).cast(pl.String).alias(series_col))

    drivers = [c for c in ctx.many("drivers") if c in frame.columns and c != target]
    if series_col and series_col in drivers:
        drivers.remove(series_col)

    _note(notes, f"We sorted every row by “{date_col}” so the model always looks backwards in "
                 f"time, never forwards.")

    season, unit = _season_length(frame.get_column(date_col))
    frame = frame.sort([series_col, date_col] if series_col else [date_col])

    over = ["over", series_col] if series_col else []
    gap_days = float(frame.get_column(date_col).diff().dt.total_days().median() or 1.0)
    gap_days = max(1.0, gap_days)
    seasonal_lag = max(1, int(season))
    lags = sorted({1, 2, 3, seasonal_lag, 2 * seasonal_lag, 3 * seasonal_lag})
    lags = [l for l in lags if l >= 1]

    exprs = []
    for lag in lags:
        e = pl.col(target).shift(lag)
        exprs.append((e.over(*over) if over else e).alias(f"lag_{lag}"))
    for win in (7, 28):
        e = pl.col(target).rolling_mean(window_size=win, min_periods=1)
        exprs.append((e.over(*over) if over else e).alias(f"average of last {win}"))
        e2 = pl.col(target).rolling_std(window_size=win, min_periods=2)
        exprs.append((e2.over(*over) if over else e2).alias(f"wobble over last {win}"))
    frame = frame.with_columns(exprs)

    # a lag longer than the history we have is just an empty column
    max_nulls = 0.8 * frame.height
    lags = [l for l in lags
            if l == seasonal_lag
            or frame.get_column(f"lag_{l}").null_count() <= max_nulls]
    lag_cols = [f"lag_{l}" for l in lags]
    roll_cols = [f"average of last {w}" for w in (7, 28)] + [f"wobble over last {w}" for w in (7, 28)]
    seasonal_col = f"lag_{seasonal_lag}"

    _note(notes, f"We built {len(lag_cols) + len(roll_cols)} history columns: what the number "
                 f"was {', '.join(str(l) for l in lags[:-1])} and {lags[-1]} periods ago, plus "
                 f"the recent average and how wobbly it has been. That is how a forecast learns "
                 f"that Decembers are busy.")
    _note(notes, f"Your data moves roughly every {int(gap_days)} day(s), so we compare each row "
                 f"with the same point {unit} earlier ({seasonal_lag} rows back).")

    driver_num = [c for c in drivers if frame.get_column(c).dtype.is_numeric()]
    driver_cat = [c for c in drivers if c not in driver_num]
    frame, num_cols, cat_cols, date_cols = _build_features(
        frame, driver_num, driver_cat, [date_col], notes)
    num_cols = list(dict.fromkeys(lag_cols + roll_cols + num_cols))
    num_cols = [c for c in num_cols if c in frame.columns]
    if series_col and series_col in num_cols:
        num_cols.remove(series_col)
    if series_col:
        cat_cols = list(dict.fromkeys([series_col] + cat_cols))

    ctx.add_keys([date_col] + ([series_col] if series_col else []))
    if series_col:
        n_series = frame.get_column(series_col).n_unique()
        _note(notes, f"“{series_col}” splits the data into {n_series:,} separate series. Each "
                     f"one gets its own history, so a quiet region is never judged against a "
                     f"busy one.")
    else:
        _note(notes, "All rows are treated as one single series.")

    if frame.height < 60:
        raise ValueError(
            f"There are only {frame.height:,} rows here. A forecast needs a reasonable run of "
            f"history — at least 60 rows — to learn a seasonal pattern."
        )

    return TargetPlan(
        task="regression", frame=frame, target_col=target,
        feature_cols=list(dict.fromkeys(num_cols + cat_cols)),
        numeric_cols=num_cols, categorical_cols=cat_cols, label_names=None,
        positive_rate=None, impact_col=target, impact_label=ctx.uc.impact_label,
        score_direction="higher_is_good", key_cols=ctx.key_cols, notes=notes,
        extra={"series_col": series_col, "time_col": date_col,
               "season_length": seasonal_lag, "seasonal_lag_col": seasonal_col,
               "lag_cols": lag_cols},
        baseline="seasonal_naive",
    )


# --------------------------------------------------------------------------- #
# 5. churn risk
# --------------------------------------------------------------------------- #


def _churn_risk(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    cust = ctx.require("customer", frame)
    date_col = ctx.require("date", frame)
    money_cols = [c for c in ctx.many("money") if c in frame.columns]
    if not money_cols:
        raise ValueError("Pick the column that holds what each purchase was worth.")
    money = money_cols[0]

    dt = to_datetime(frame.get_column(date_col))
    if dt is None or dt.n_unique() < 3:
        raise ValueError(
            f"We need a date column with a real spread of dates to work out who has gone "
            f"quiet, and “{date_col}” does not give us that."
        )
    frame = frame.with_columns(dt.alias(date_col))
    frame, ok = _clean_numeric(frame, money, notes, label="money column")
    if not ok:
        raise ValueError(f"We could not read “{money}” as a number.")

    qty = next((c for c in ctx.many("quantity") if c in frame.columns), None)
    product = next((c for c in ctx.many("product") if c in frame.columns), None)
    segment = next((c for c in ctx.many("segment") if c in frame.columns), None)
    channel = next((c for c in ctx.many("channel") if c in frame.columns), None)

    window_pct = float(ctx.options.get("churn_window_pct", 25) or 25)
    window_pct = min(60.0, max(5.0, window_pct)) / 100.0

    lo, hi = frame.get_column(date_col).min(), frame.get_column(date_col).max()
    span_days = max(1, (hi - lo).days)
    cutoff = hi - timedelta(days=int(span_days * window_pct))
    _note(notes, f"Your records run from {lo.date()} to {hi.date()} — {_fmt(span_days)} days. "
                 f"We treat the last {int(window_pct * 100)}% of that window (from "
                 f"{cutoff.date()}) as 'the future' and ask who went quiet in it.")
    _note(notes, f"A customer counts as churned if they bought nothing on or after "
                 f"{cutoff.date()}. Everything the model sees is from before that date, so it "
                 f"cannot peek at the answer.")

    history = frame.filter(pl.col(date_col) < cutoff)
    future = frame.filter(pl.col(date_col) >= cutoff)
    if history.height < 30:
        raise ValueError(
            "There is not enough history before the cut-off date to learn from. Try a smaller "
            "churn window (for example 15%)."
        )

    hist_ids = history.get_column(cust).unique()
    future_ids = set(future.get_column(cust).unique().to_list())
    _note(notes, f"{hist_ids.len():,} customers had at least one purchase before the cut-off "
                 f"date and could be scored. {len(future_ids):,} customers bought after it.")

    h = history.sort([cust, date_col])
    h = h.with_columns(
        pl.col(date_col).diff().dt.total_days().over(cust).alias("_gap")
    )

    aggs = [
        pl.col(date_col).max().alias("Last purchase date"),
        pl.col(date_col).min().alias("First purchase date"),
        pl.len().alias("Number of purchases"),
        pl.col(money).sum().alias("Total spend"),
        pl.col(money).mean().alias("Average order value"),
        pl.col(money).max().alias("Biggest order"),
        pl.col(money).std().alias("Spread of order values"),
        pl.col("_gap").drop_nulls().mean().alias("Typical days between purchases"),
        pl.col("_gap").last().alias("Days since the purchase before last"),
        pl.col("_gap").drop_nulls().median().alias("Middle days between purchases"),
    ]
    if qty:
        h, _ = _clean_numeric(h, qty, notes, label="quantity")
        aggs.append(pl.col(qty).sum().alias("Total units"))
        aggs.append(pl.col(qty).mean().alias("Average units per order"))
    if product:
        aggs.append(pl.col(product).cast(pl.String).n_unique().alias("Different products bought"))
    if channel:
        aggs.append(pl.col(channel).cast(pl.String).n_unique().alias("Different channels used"))

    cust_frame = h.group_by(cust).agg(aggs)

    # trend: spend in the last quarter of the history window vs the first quarter
    q = max(1, int(span_days * 0.25))
    first_cut = lo + timedelta(days=q)
    late_start = cutoff - timedelta(days=q)
    early = (history.filter(pl.col(date_col) < first_cut)
             .group_by(cust).agg(pl.col(money).sum().alias("_early"), pl.len().alias("_n_early")))
    late = (history.filter(pl.col(date_col) >= late_start)
            .group_by(cust).agg(pl.col(money).sum().alias("_late"), pl.len().alias("_n_late")))
    cust_frame = cust_frame.join(early, on=cust, how="left").join(late, on=cust, how="left")

    cust_frame = cust_frame.with_columns([
        (pl.lit(hi) - pl.col("Last purchase date")).dt.total_days().alias("Days since last purchase"),
        (pl.col("Last purchase date") - pl.col("First purchase date")).dt.total_days().alias("Days as a customer"),
        _safe_div(pl.col("Total spend"), pl.col("Number of purchases")).alias("Spend per purchase"),
        _safe_div(pl.col("_late").fill_null(0.0), pl.col("_early").fill_null(0.0) + 1.0)
            .alias("Recent spend vs earlier spend"),
        _safe_div(pl.col("_n_late").fill_null(0.0), pl.col("_n_early").fill_null(0.0) + 1.0)
            .alias("Recent visits vs earlier visits"),
    ])
    _note(notes, "For each customer we summarised their whole history into one row: how "
                 "recently they bought, how often, how much in total, what a typical basket "
                 "looks like, how long they have been a customer, and whether their recent "
                 "spending is rising or falling.")

    label = cust_frame.get_column(cust).is_in(pl.Series(list(future_ids)))
    cust_frame = cust_frame.with_columns(
        pl.when(label).then(pl.lit(1, dtype=pl.Int8)).otherwise(pl.lit(0, dtype=pl.Int8))
        .alias("_churn")
    )
    cust_frame = cust_frame.drop(["_early", "_late", "_n_early", "_n_late"])

    drop = [c for c in ("Last purchase date", "First purchase date") if c in cust_frame.columns]
    cust_frame = cust_frame.drop(drop)
    _note(notes, "We then dropped the raw dates: knowing the exact last-purchase date would "
                 "give away the answer, so the model only keeps the number of days.")

    target = "_churn"
    y = cust_frame.get_column(target).to_numpy().astype(float)
    rate = float(y.mean())
    _note(notes, f"{int(y.sum()):,} of {len(y):,} customers ({_pct(rate)}) went quiet in the "
                 f"final window. That is the outcome the models will try to spot in advance.")

    exclude = {cust, target}
    feature_num = [c for c in cust_frame.columns
                   if c not in exclude and cust_frame.get_column(c).dtype.is_numeric()]
    feature_cat = [c for c in cust_frame.columns
                   if c not in exclude and not cust_frame.get_column(c).dtype.is_numeric()]
    feature_num = _drop_leaky(cust_frame, pl.Series(y), feature_num, notes)
    _note(notes, "Everything we know about a customer comes from before the cut-off date, so "
                 "the model never gets a peek at the future it is meant to predict.")

    if not feature_num and not feature_cat:
        raise ValueError(
            "We could not build any usable history for these customers. Check that the date "
            "and money columns are the right way round."
        )

    ctx.add_keys([cust])
    return TargetPlan(
        task="classification", frame=cust_frame, target_col=target,
        feature_cols=list(dict.fromkeys(feature_num + feature_cat)),
        numeric_cols=feature_num, categorical_cols=feature_cat,
        label_names={0: "Still buying", 1: "Gone quiet"},
        positive_rate=rate, impact_col="Total spend",
        impact_label=ctx.uc.impact_label, score_direction="higher_is_risk",
        key_cols=ctx.key_cols, notes=notes,
        extra={"cutoff": str(cutoff.date()), "window_pct": int(window_pct * 100),
               "segment_col": segment, "money_col": money},
    )


# --------------------------------------------------------------------------- #
# 6. credit default
# --------------------------------------------------------------------------- #

_BAD_WORDS = ("default", "late", "delinq", "bad", "writ", "npa", "overdue", "past due",
              "missed", "unpaid", "churn", "loss")
_GOOD_WORDS = ("current", "paid", "good", "ok", "active", "regular", "clear", "settled",
               "on time", "ontime", "no")


def _binary_from_text(s: pl.Series, notes: list[str], col: str) -> tuple[pl.Series, int]:
    """Map a text/binary column onto 0/1 using the words we see in the values."""
    tmp = pl.DataFrame({
        "__v": s.cast(pl.String).str.to_lowercase().str.strip_chars()
    })
    text = pl.col("__v")
    bad = pl.lit(False)
    for w in _BAD_WORDS:
        bad = bad | text.str.contains(w, literal=True)
    good = pl.lit(False)
    for w in _GOOD_WORDS:
        good = good | text.str.contains(w, literal=True)
    out = (pl.when(bad).then(pl.lit(1, dtype=pl.Int8))
             .when(good).then(pl.lit(0, dtype=pl.Int8))
             .otherwise(pl.lit(None, dtype=pl.Int8)))
    res = tmp.select(out.alias("__r")).to_series()
    unknown = int(res.null_count())
    if unknown:
        _note(notes, f"{unknown:,} rows had a “{col}” value we could not classify as good or "
                     f"bad, so they were left out of the training.")
    return res, unknown


def _credit_default(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    money_col = ctx.require("money", frame)
    frame, ok = _clean_numeric(frame, money_col, notes, label="money column")
    if not ok:
        raise ValueError(f"We could not read “{money_col}” as a number.")

    outcome = ctx.one("outcome")
    status = ctx.one("status")
    dpd = ctx.one("days_past_due")
    target = "_default"
    source = ""

    if outcome and outcome in frame.columns:
        out_series = frame.get_column(outcome)
        if out_series.dtype.is_numeric() and out_series.n_unique() <= 2:
            vals = sorted(v for v in out_series.drop_nulls().unique().to_list())
            if vals and set(vals) <= {0, 1}:
                frame = frame.with_columns(out_series.cast(pl.Int8).alias(target))
                _note(notes, f"We used “{outcome}” as the answer: 1 means the bad outcome, "
                             f"0 means everything was fine.")
            else:
                bad_value = vals[-1] if vals else 1
                frame = frame.with_columns(
                    (out_series == bad_value).cast(pl.Int8).alias(target))
                _note(notes, f"“{outcome}” holds two values. We treated {bad_value} as the bad "
                             f"outcome and {vals[0]} as the good one — check that is the right "
                             f"way round.")
            source = f"the outcome column “{outcome}” exactly as you gave it"
        else:
            mapped, _ = _binary_from_text(frame.get_column(outcome), notes, outcome)
            frame = frame.with_columns(mapped.alias(target))
            source = f"the words in “{outcome}”"
            _note(notes, f"We read “{outcome}” and marked the rows whose value sounds like a "
                         f"default or a late payment as the bad outcome.")
    elif status and status in frame.columns:
        mapped, _ = _binary_from_text(frame.get_column(status), notes, status)
        frame = frame.with_columns(mapped.alias(target))
        source = f"the status column “{status}”"
        _note(notes, f"We read “{status}” and treated values that sound like a default, a "
                     f"write-off or a missed payment as the bad outcome.")
    elif dpd and dpd in frame.columns:
        frame, ok = _clean_numeric(frame, dpd, notes, label="days-past-due number")
        if not ok:
            raise ValueError(f"We could not read “{dpd}” as a number.")
        frame = frame.with_columns(
            (pl.col(dpd) > 90).cast(pl.Int8).alias(target))
        source = f"more than 90 days past due in “{dpd}”"
        _note(notes, f"We treated any account more than 90 days past due in “{dpd}” as a "
                     f"default, which is the usual banking rule of thumb.")
    else:
        raise ValueError(
            "We need to know who actually defaulted. Point us at a column that says so — a "
            "yes/no outcome, a status column, or a days-past-due number — and run it again."
        )

    nulls = int(frame.get_column(target).is_null().sum())
    if nulls:
        frame = frame.filter(pl.col(target).is_not_null())
        _note(notes, f"{nulls:,} rows were dropped because we could not tell whether they were "
                     f"good or bad.")

    y = frame.get_column(target).to_numpy().astype(float)
    rate = float(y.mean()) if len(y) else 0.0
    if 0 < rate:
        _note(notes, f"{int(y.sum()):,} of {len(y):,} accounts ({_pct(rate)}) went bad. We "
                     f"learned the pattern from {source}.")
    if len(y) < 100:
        raise ValueError(
            f"Only {len(y):,} usable rows are left after working out who defaulted. That is "
            f"too few to learn a reliable pattern."
        )

    ctx.add_keys(ctx.many("customer")[:1] + [money_col])

    chosen_num: list[str] = []
    chosen_cat: list[str] = []
    for c in ctx.many("drivers"):
        if c in frame.columns and c != target and c not in (outcome, status, dpd, money_col):
            (chosen_num if frame.get_column(c).dtype.is_numeric() else chosen_cat).append(c)
    for role in ("segment", "product", "customer"):
        for c in ctx.many(role):
            if c in frame.columns and c not in chosen_num + chosen_cat:
                chosen_cat.append(c)
    date_cols = [c for c in ctx.many("date") if c in frame.columns]

    if not chosen_num and not chosen_cat and not date_cols:
        auto_num, auto_cat = _auto_context(
            frame, exclude={target, outcome, status, dpd, money_col})
        chosen_num.extend(auto_num)
        chosen_cat.extend(auto_cat)
        _note(notes, f"You did not name any extra evidence, so we used the other "
                     f"{len(auto_num) + len(auto_cat)} usable columns in the file.")

    frame, num_cols, cat_cols, _ = _build_features(frame, chosen_num, chosen_cat, date_cols, notes)
    num_cols = [c for c in num_cols if c not in (target, money_col, dpd)]
    cat_cols = [c for c in cat_cols if c not in (target, outcome, status)]
    num_cols = _drop_leaky(frame, pl.Series(y), num_cols, notes)

    if not num_cols and not cat_cols:
        raise ValueError(
            "Nothing usable is left to predict from. Add a few columns describing the account "
            "— size, tenure, rate, segment — and try again."
        )
    _note(notes, f"{len(num_cols) + len(cat_cols)} columns describe each account. The money "
                 f"owed (“{money_col}”) is deliberately kept out of the prediction so the model "
                 f"cannot shortcut to 'big loans go bad'.")

    return TargetPlan(
        task="classification", frame=frame, target_col=target,
        feature_cols=list(dict.fromkeys(num_cols + cat_cols)),
        numeric_cols=num_cols, categorical_cols=cat_cols,
        label_names={0: "Paid as agreed", 1: "Defaulted or late"},
        positive_rate=rate, impact_col=money_col, impact_label=ctx.uc.impact_label,
        score_direction="higher_is_risk", key_cols=ctx.key_cols, notes=notes,
        extra={"outcome_source": source},
    )


# --------------------------------------------------------------------------- #
# 7. transaction anomaly
# --------------------------------------------------------------------------- #


def _transaction_anomaly(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    measure_cols = [c for c in ctx.many("measures") if c in frame.columns]
    if len(measure_cols) < 1:
        raise ValueError("Pick at least one numeric column to judge the records on.")
    if len(measure_cols) == 1:
        _note(notes, "You gave us one number to judge on. That works, but two or three give "
                     "the models a much better chance of spotting a genuinely odd record.")

    ctx.add_keys(ctx.many("key")[:2] + ctx.many("dimensions")[:2])
    money_col = ctx.one("money")
    if money_col and money_col in frame.columns and money_col not in measure_cols:
        ctx.add_keys([money_col])
    elif money_col not in frame.columns:
        money_col = None

    dim_cols = [c for c in ctx.many("dimensions") if c in frame.columns
                and c not in measure_cols]
    date_cols = [c for c in ctx.many("date") if c in frame.columns]

    frame, num_cols, cat_cols, _ = _build_features(
        frame, measure_cols, dim_cols, date_cols, notes)
    num_cols = [c for c in dict.fromkeys(measure_cols + num_cols) if c in frame.columns]

    if not num_cols:
        raise ValueError(
            "None of the numeric columns you picked could be read as numbers. Check them and "
            "try again."
        )
    _note(notes, f"We are judging each record on {len(num_cols)} number(s) and "
                 f"{len(cat_cols)} category column(s). There is no right answer to learn from "
                 f"here — we simply look for records that do not fit the crowd.")
    return TargetPlan(
        task="anomaly", frame=frame, target_col=None,
        feature_cols=list(dict.fromkeys(num_cols + cat_cols)),
        numeric_cols=num_cols, categorical_cols=cat_cols, label_names=None,
        positive_rate=None, impact_col=money_col if money_col else None,
        impact_label=ctx.uc.impact_label, score_direction="higher_is_risk",
        key_cols=ctx.key_cols, notes=notes, extra={},
    )


# --------------------------------------------------------------------------- #
# 8. segmentation
# --------------------------------------------------------------------------- #


def _segmentation(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    measure_cols = [c for c in ctx.many("measures") if c in frame.columns]
    if len(measure_cols) < 2:
        raise ValueError(
            "Grouping customers needs at least two numbers to compare them on — for example "
            "how often they buy and how much they spend."
        )
    dim_cols = [c for c in ctx.many("dimensions") if c in frame.columns]
    date_cols = [c for c in ctx.many("date") if c in frame.columns]
    cust = ctx.one("customer")

    if cust and cust in frame.columns:
        frame, ok = _clean_numeric(frame, measure_cols[0], notes, label="number")
        if not ok:
            raise ValueError(f"We could not read “{measure_cols[0]}” as a number.")
        agg = [pl.len().alias("Number of records")]
        agg_cols: list[str] = []
        for c in measure_cols:
            tmp, ok = _clean_numeric(frame, c, notes, label="number")
            if not ok:
                continue
            frame = tmp
            avg_name, tot_name = f"Average {c}", f"Total {c}"
            agg.append(pl.col(c).mean().alias(avg_name))
            agg.append(pl.col(c).sum().alias(tot_name))
            agg_cols.extend([avg_name, tot_name])
        if date_cols:
            d = date_cols[0]
            frame = frame.with_columns(to_datetime(frame.get_column(d)).alias(d))
            agg.append(pl.col(d).max().alias("Most recent date"))
        frame = frame.group_by(cust).agg(agg)
        if "Most recent date" in frame.columns:
            latest = frame.get_column("Most recent date").max()
            frame = frame.with_columns(
                (pl.lit(latest) - pl.col("Most recent date"))
                .dt.total_days().alias("Days since their last record")
            ).drop("Most recent date")
            agg_cols.append("Days since their last record")
        _note(notes, f"We rolled {cust} up to one row per customer — {frame.height:,} customers — "
                     f"because segments should describe customers, not individual transactions.")
        ctx.add_keys([cust])
        live_cats = [c for c in dim_cols if c in frame.columns]
        frame, num_cols, cat_cols, _ = _build_features(
            frame, agg_cols, live_cats, [], notes)
        num_cols = [c for c in dict.fromkeys(agg_cols + num_cols) if c in frame.columns]
    else:
        if cust:
            _note(notes, "We could not find the customer column, so we grouped the records "
                         "themselves instead of customers.")
        for c in measure_cols:
            frame, _ = _clean_numeric(frame, c, notes, label="number")
        _note(notes, "We are grouping the records in this file. Add a customer ID next time to "
                     "group customers instead.")
        frame, num_cols, cat_cols, _ = _build_features(frame, measure_cols, dim_cols, [], notes)
        num_cols = [c for c in dict.fromkeys(measure_cols + num_cols) if c in frame.columns]

    if len(num_cols) < 2:
        raise ValueError(
            "We ended up with fewer than two usable numbers to group on. Check the columns "
            "you picked."
        )
    _note(notes, f"{len(num_cols)} number(s) describe each row. The models will look for "
                 f"natural groups without being told what the groups are — we then test how "
                 f"many groups actually make sense.")
    return TargetPlan(
        task="clustering", frame=frame, target_col=None,
        feature_cols=list(dict.fromkeys(num_cols + cat_cols)),
        numeric_cols=num_cols, categorical_cols=cat_cols, label_names=None,
        positive_rate=None, impact_col=None, impact_label=ctx.uc.impact_label,
        score_direction="higher_is_good", key_cols=ctx.key_cols, notes=notes, extra={},
    )


# --------------------------------------------------------------------------- #
# 9. price elasticity
# --------------------------------------------------------------------------- #


def _price_elasticity(ctx: _Ctx, frame: pl.DataFrame) -> TargetPlan:
    notes = ctx.notes
    qty = ctx.require("quantity", frame)
    price = ctx.require("price", frame)

    frame, ok = _clean_numeric(frame, qty, notes, label="demand number")
    if not ok:
        raise ValueError(f"We could not read “{qty}” as a number.")
    frame, ok = _clean_numeric(frame, price, notes, label="price")
    if not ok:
        raise ValueError(f"We could not read “{price}” as a number.")

    is_discount = any(w in price.lower() for w in ("disc", "promo", "rebate", "markdown"))
    _note(notes, f"We are measuring how “{qty}” responds to “{price}”. "
                 + ("Because that column is a discount, we expect demand to go UP as it rises."
                    if is_discount else
                    "We expect demand to go DOWN as the price rises."))

    pv = frame.get_column(price)
    if pv.n_unique() < 5:
        _note(notes, f"“{price}” only takes {pv.n_unique()} different values, so we can only "
                     f"give you a rough sense of the response. More price variation would give "
                     f"a sharper answer.")
    if (pv <= 0).all():
        raise ValueError(f"“{price}” has no positive values, so we cannot read a price from it.")

    chosen_cat = []
    for role in ("product", "segment", "channel", "customer"):
        for c in ctx.many(role):
            if c in frame.columns and c not in (qty, price):
                chosen_cat.append(c)
    date_cols = [c for c in ctx.many("date") if c in frame.columns]

    frame, num_cols, cat_cols, _ = _build_features(frame, [], chosen_cat, date_cols, notes)
    num_cols = list(dict.fromkeys([price] + num_cols))
    y = frame.get_column(qty).to_numpy().astype(float)
    num_cols = _drop_leaky(frame, pl.Series(y), num_cols, notes)
    if price not in num_cols:
        _note(notes, f"“{price}” had to be left out of the prediction because it mirrors "
                     f"“{qty}” almost exactly, so there is no relationship to measure here.")

    ctx.add_keys([c for c in (price, qty) if c] + ctx.many("product")[:1])
    _note(notes, "At the end we will sweep the price up and down to sketch out how much demand "
                 "and revenue you would get at each level, and point at the best spot.")

    return TargetPlan(
        task="regression", frame=frame, target_col=qty,
        feature_cols=list(dict.fromkeys(num_cols + cat_cols)),
        numeric_cols=num_cols, categorical_cols=cat_cols, label_names=None,
        positive_rate=None, impact_col=None, impact_label=ctx.uc.impact_label,
        score_direction="higher_is_good", key_cols=ctx.key_cols, notes=notes,
        extra={"price_col": price, "quantity_col": qty, "is_discount": bool(is_discount)},
    )


# --------------------------------------------------------------------------- #
# Dispatch (contract §7)
# --------------------------------------------------------------------------- #

_STRATEGIES = {
    "discount_leakage": _discount_leakage,
    "margin_overflow": _margin_overflow,
    "revenue_predict": _revenue_predict,
    "demand_forecast": _demand_forecast,
    "churn_risk": _churn_risk,
    "credit_default": _credit_default,
    "transaction_anomaly": _transaction_anomaly,
    "segmentation": _segmentation,
    "price_elasticity": _price_elasticity,
}


def build_target(dataset_id: str, uc: UseCase, mapping: dict[str, list[str]],
                 options: dict | None = None) -> TargetPlan:
    """Turn the user's goal + column choices into a modelling table."""
    options = options or {}
    strategy = _STRATEGIES.get(uc.target)
    if strategy is None:
        raise ValueError(
            f"“{uc.name}” is not something this build knows how to do yet. "
            f"Available: {', '.join(sorted(_STRATEGIES))}."
        )
    frame = _get_frame(dataset_id)
    if frame is None or frame.height == 0:
        raise ValueError("That dataset is empty, so there is nothing to analyse.")

    ctx = _Ctx(dataset_id, uc, mapping, options)
    ctx.note(f"We loaded {frame.height:,} rows and {len(frame.columns)} columns for "
             f"“{uc.name}”.")
    plan = strategy(ctx, frame)

    if plan.frame.height == 0:
        raise ValueError(
            "After cleaning, no rows were left to analyse. Check the columns you pointed at."
        )
    if not plan.feature_cols and plan.task != "classification":
        raise ValueError("We could not find any usable columns to learn from in this file.")
    return plan


__all__ = ["TargetPlan", "build_target", "to_float", "to_datetime"]
