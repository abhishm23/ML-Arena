"""ML Arena — automatic column understanding ("smart column detection").

Contract: `docs/CONTRACT.md` §4.  Public signatures here are FROZEN.

The job of this module is to look at a freshly loaded table and answer, in plain
business language, "what *is* each column?".  A non-technical user should never
have to tell ML Arena that `net_price` is money or that `status` is a category.

Design notes
------------
* Everything is computed with vectorised polars expressions — at most two passes
  over the frame (plus one tiny ``group_by`` per low-cardinality column).
* When the frame is bigger than ``sample_rows`` we profile a systematic sample
  and **say so** in ``quality`` and in the headline.  We never pretend.
* Classification combines four independent signals — the column name, the
  dtype, the number of distinct values and the unique ratio — because any one
  of them alone is easy to fool (a numeric money column and a numeric id column
  look identical to a dtype check).
"""

from __future__ import annotations

from mlarena import config  # noqa: F401  -- sets POLARS_SKIP_CPU_CHECK before polars loads

import math
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import polars as pl

from mlarena.dataio import loader

__all__ = ["ColumnProfile", "DatasetProfile", "profile", "profile_to_dict"]


# -----------------------------------------------------------------------------
# dataclasses (contract §4)
# -----------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    name: str
    kind: str
    dtype: str
    semantic: str
    null_pct: float
    distinct: int
    unique_ratio: float
    samples: list
    stats: dict
    role_hint: str
    confidence: float


@dataclass
class DatasetProfile:
    dataset_id: str
    rows: int
    cols: int
    profiled_ms: float
    columns: list[ColumnProfile]
    measures: list[str]
    dimensions: list[str]
    datetimes: list[str]
    identifiers: list[str]
    quality: dict
    suggestions: list[dict]
    #: Additive extension (contract §4 does not list it): free-text column names,
    #: so the UI can badge them without re-deriving the classification.
    texts: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# name understanding
# -----------------------------------------------------------------------------
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z]+")

_ID_TOKENS = {
    "id", "ids", "uuid", "guid", "key", "keys", "sku", "serial", "ref", "reference",
    "identifier", "invoice", "no", "num", "number", "account", "ticket",
}
_ID_STRONG_TOKENS = {"id", "ids", "uuid", "guid", "sku", "serial", "key", "identifier"}
_CODE_TOKENS = {"code", "codes", "ref", "reference", "invoice", "sku"}
_CONTACT_TOKENS = {"email", "phone", "mobile", "fax", "url", "link", "ip", "mac", "website"}

_MONEY_TOKENS = {
    "amount", "amt", "price", "cost", "revenue", "rev", "sales", "sale", "turnover",
    "margin", "discount", "value", "total", "subtotal", "net", "gross", "profit",
    "balance", "salary", "wage", "wages", "fee", "fees", "charge", "charges",
    "payment", "paid", "spend", "spent", "income", "expense", "expenses", "tax",
    "refund", "credit", "debit", "budget", "asp", "mrr", "arr", "cogs", "basket",
}
_QTY_TOKENS = {
    "qty", "quantity", "quantities", "units", "volume", "stock", "inventory",
    "weight", "mass", "size", "length", "width", "height", "distance", "duration",
    "hours", "minutes", "seconds", "days", "tenure",
}
_COUNT_TOKENS = {
    "count", "cnt", "frequency", "freq", "occurrences", "hits", "visits", "sessions",
    "transactions", "orders", "items", "clicks", "views",
}
_PCT_TOKENS = {
    "pct", "percent", "percentage", "rate", "ratio", "share", "margin", "discount",
    "growth", "conversion", "churn", "uplift", "lift", "score", "probability", "prob",
}
_DATE_TOKENS = {
    "date", "dt", "time", "timestamp", "ts", "month", "year", "yr", "period", "week",
    "quarter", "qtr", "day", "created", "updated", "modified", "at", "on", "when",
}
_FLAG_TOKENS = {
    "is", "has", "was", "were", "flag", "active", "enabled", "disabled", "deleted",
    "churned", "defaulted", "returned", "cancelled", "canceled", "refunded", "paid",
    "valid", "open", "closed", "flagged", "duplicate", "fraud", "anomaly",
}
_TEXT_TOKENS = {
    "comment", "comments", "note", "notes", "description", "desc", "text", "message",
    "msg", "review", "reviews", "feedback", "remark", "remarks", "details", "summary",
    "body", "content", "reason", "explanation", "narrative", "address",
}
_CATEGORY_TOKENS = {
    "category", "cat", "type", "status", "state", "segment", "region", "area",
    "channel", "group", "class", "label", "level", "tier", "brand", "department",
    "country", "city", "town", "product", "service", "vendor", "supplier",
    "method", "mode", "source", "medium", "priority", "stage", "grade", "team",
    "owner", "rep", "agent", "store", "warehouse", "currency", "unit",
}

_DATE_VALUE_RE = re.compile(
    r"^\s*("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"          # 2024-01-31, 2024/01/31
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"       # 31/01/2024
    r"|\d{8}"                                  # 20240131
    r"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"       # ISO timestamp
    r")"
)
_YEAR_VALUE_RE = re.compile(r"^(19|20)\d{2}$")

#: polars `str.to_date` formats we try, cheapest/most common first.
_DATE_FORMATS: tuple[str | None, ...] = (
    None,
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%Y%m%d",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
)


def _tokens(name: str) -> list[str]:
    """`"Unit_List Price"` -> `["unit", "list", "price"]`."""
    spaced = _CAMEL_RE.sub(" ", str(name))
    return [t for t in _NON_WORD_RE.split(spaced.lower()) if t]


def _hits(tokens: list[str], vocab: set[str]) -> bool:
    return any(t in vocab for t in tokens)


def _looks_like_id(tokens: list[str]) -> bool:
    if _hits(tokens, _ID_STRONG_TOKENS):
        return True
    if "id" in tokens or "code" in tokens:
        return True
    # "order_no", "invoice_number", "po_num", ...
    if len(tokens) >= 2 and tokens[-1] in {"no", "num", "number", "ref", "reference"}:
        return True
    return False


def _looks_like_date_name(tokens: list[str]) -> bool:
    if _hits(tokens, _DATE_TOKENS - {"at", "on"}):
        return True
    return bool(tokens) and tokens[-1] in {"at", "on", "dt", "ts"}


# -----------------------------------------------------------------------------
# dtype helpers
# -----------------------------------------------------------------------------
def _is_numeric(dtype: Any) -> bool:
    try:
        return bool(dtype.is_numeric())
    except Exception:
        return str(dtype) in {
            "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32",
            "UInt64", "Float32", "Float64", "Int128", "Decimal",
        }


def _is_temporal(dtype: Any) -> bool:
    try:
        return bool(dtype.is_temporal())
    except Exception:
        text = str(dtype)
        return text.startswith("Date") or text.startswith("Datetime") or text == "Time"


def _is_string(dtype: Any) -> bool:
    text = str(dtype)
    return text in ("String", "Categorical", "Enum") or text.startswith("String")


def _is_integer(dtype: Any) -> bool:
    return str(dtype) in {
        "Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64",
        "Int128",
    }


def _is_bool(dtype: Any) -> bool:
    return str(dtype) == "Boolean"


# -----------------------------------------------------------------------------
# JSON safety
# -----------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert anything numpy/polars/datetime-ish into JSON types."""
    return loader.json_safe(obj)


def _num(value: Any, digits: int = 6) -> float | None:
    """Round a possibly-missing scalar into a JSON-safe float."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return round(out, digits)


# -----------------------------------------------------------------------------
# pass 1 — cheap facts
# -----------------------------------------------------------------------------
def _basic_facts(work: pl.DataFrame) -> tuple[list[int], list[int]]:
    """(null counts, distinct counts) aligned with ``work.columns``."""
    try:
        nulls = [int(v) for v in work.null_count().row(0)]
    except Exception:  # pragma: no cover - defensive
        nulls = []
        for name in work.columns:
            try:
                nulls.append(int(work.select(pl.col(name).is_null().sum()).item() or 0))
            except Exception:
                nulls.append(0)
    try:
        distinct = [int(v) for v in work.select(pl.all().n_unique()).row(0)]
    except Exception:
        distinct = []
        for name in work.columns:
            try:
                distinct.append(int(work.select(pl.col(name).n_unique()).item() or 0))
            except Exception:
                distinct.append(0)
    return nulls, distinct


# -----------------------------------------------------------------------------
# pass 2 — type-driven statistics, one vectorised select
# -----------------------------------------------------------------------------
def _stat_exprs(name: str, dtype: Any) -> tuple[list[str], list[pl.Expr]]:
    col = pl.col(name)
    keys: list[str] = []
    exprs: list[pl.Expr] = []

    def add(key: str, expr: pl.Expr) -> None:
        keys.append(key)
        exprs.append(expr.alias(f"{name}\u0000{key}"))

    if _is_numeric(dtype) and not _is_bool(dtype):
        f = col.cast(pl.Float64)
        add("min", col.min())
        add("max", col.max())
        add("mean", f.mean())
        add("std", f.std())
        add("median", f.median())
        add("p25", f.quantile(0.25))
        add("p75", f.quantile(0.75))
        add("sum", col.sum())
        add("skew", f.skew())
        add("negatives", f.lt(0).sum())
        add("zeros", f.eq(0).sum())
    elif _is_temporal(dtype):
        add("min", col.min())
        add("max", col.max())
    elif _is_string(dtype):
        add("len_mean", col.str.len_chars().mean())
        add("len_max", col.str.len_chars().max())
    return keys, exprs


def _all_stats(work: pl.DataFrame) -> dict[str, dict[str, Any]]:
    plan: dict[str, list[str]] = {}
    exprs: list[pl.Expr] = []
    for name, dtype in zip(work.columns, work.dtypes):
        keys, col_exprs = _stat_exprs(name, dtype)
        plan[name] = keys
        exprs.extend(col_exprs)
    if not exprs:
        return {}
    try:
        row = work.select(exprs).row(0)
        out: dict[str, dict[str, Any]] = {}
        cursor = 0
        for name in work.columns:
            size = len(plan[name])
            out[name] = dict(zip(plan[name], row[cursor:cursor + size]))
            cursor += size
        return out
    except Exception:
        return _all_stats_per_column(work)


def _all_stats_per_column(work: pl.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, dtype in zip(work.columns, work.dtypes):
        keys, exprs = _stat_exprs(name, dtype)
        values: dict[str, Any] = {}
        for key, expr in zip(keys, exprs):
            try:
                values[key] = work.select(expr).item()
            except Exception:
                values[key] = None
        out[name] = values
    return out


def _date_parse_rate(work: pl.DataFrame, name: str) -> float:
    """Fraction of non-null values that parse as a date, best format wins."""
    try:
        series = work.select(pl.col(name)).to_series()
    except Exception:
        return 0.0
    total = int(series.len() - series.null_count())
    if total <= 0:
        return 0.0
    best = 0.0
    for fmt in _DATE_FORMATS:
        try:
            if fmt is None:
                parsed = series.str.to_datetime(strict=False, exact=False)
            else:
                parsed = series.str.to_datetime(format=fmt, strict=False, exact=False)
            got = int(parsed.len() - parsed.null_count())
            best = max(best, got / total)
            if best >= 0.995:
                break
        except Exception:
            continue
    return best


def _values_look_like_dates(work: pl.DataFrame, name: str) -> bool:
    try:
        head = work.select(pl.col(name).drop_nulls().head(20)).to_series().to_list()
    except Exception:
        return False
    if not head:
        return False
    hits = sum(1 for v in head if isinstance(v, str) and _DATE_VALUE_RE.match(v))
    return hits >= max(1, int(0.6 * len(head)))


def _top_values(work: pl.DataFrame, name: str, limit: int = 10) -> tuple[list, float]:
    try:
        grouped = (
            work.select(pl.col(name), pl.lit(1).alias("__one__"))
            .group_by(name)
            .agg(pl.len().alias("__n__"))
            .sort("__n__", descending=True)
            .head(limit)
        )
    except Exception:
        return [], 0.0
    rows = grouped.to_dicts()
    top = [[_json_safe(r.get(name)), int(r.get("__n__") or 0)] for r in rows]
    total = sum(int(r.get("__n__") or 0) for r in rows)
    try:
        non_null = int(work.select(pl.col(name).is_not_null().sum()).item() or 0)
    except Exception:
        non_null = 0
    coverage = (total / non_null) if non_null else 0.0
    return top, round(coverage, 4)


def _samples(work: pl.DataFrame, name: str, dtype: Any, *, spread: bool) -> list:
    try:
        col = pl.col(name).drop_nulls()
        if not spread:
            col = col.unique(maintain_order=True)
        values = work.select(col.head(5)).to_series().to_list()
    except Exception:
        values = []
    return [_json_safe(v) for v in values]


# -----------------------------------------------------------------------------
# classification
# -----------------------------------------------------------------------------
def _semantic_for_measure(tokens: list[str], minimum: Any, maximum: Any) -> tuple[str, float]:
    """(semantic, confidence) for a numeric measure."""
    pct = _hits(tokens, _PCT_TOKENS)
    money = _hits(tokens, _MONEY_TOKENS)
    qty = _hits(tokens, _QTY_TOKENS)
    count = _hits(tokens, _COUNT_TOKENS)

    if pct:
        lo, hi = _num(minimum), _num(maximum)
        if lo is not None and hi is not None and lo >= -1.0 and hi <= 1.0:
            return "percent", 0.92
        if lo is not None and hi is not None and lo >= -1000.0 and hi <= 1000.0:
            return "percent", 0.78
    if money:
        return "currency", 0.9
    if qty:
        return "quantity", 0.85
    if count:
        return "count", 0.85
    return "numeric", 0.7


def _classify(
    *,
    name: str,
    dtype: Any,
    distinct: int,
    non_null: int,
    unique_ratio: float,
    null_pct: float,
    stats: dict,
    parse_rate: float,
    avg_len: float,
) -> tuple[str, str, float, str]:
    """Return ``(kind, semantic, confidence, role_hint)``."""
    tokens = _tokens(name)
    id_hint = _looks_like_id(tokens)
    money_hint = _hits(tokens, _MONEY_TOKENS)
    qty_hint = _hits(tokens, _QTY_TOKENS)
    pct_hint = _hits(tokens, _PCT_TOKENS)
    flag_hint = _hits(tokens, _FLAG_TOKENS)
    date_hint = _looks_like_date_name(tokens)
    text_hint = _hits(tokens, _TEXT_TOKENS)
    contact_hint = _hits(tokens, _CONTACT_TOKENS)
    cat_hint = _hits(tokens, _CATEGORY_TOKENS)

    def dim(semantic: str, conf: float, hint: str) -> tuple[str, str, float, str]:
        return "dimension", semantic, conf, hint

    # ---- real date / time columns -----------------------------------------
    if _is_temporal(dtype):
        if str(dtype) == "Time":
            return dim(
                "category", 0.8, "Looks like a time of day you can group by"
            )
        conf = 0.98 if date_hint else 0.94
        return "datetime", "date", conf, "Looks like a date you can trend over"

    # ---- booleans ----------------------------------------------------------
    if _is_bool(dtype):
        return dim("flag", 0.96, "Looks like a yes/no flag")

    # ---- numbers -----------------------------------------------------------
    if _is_numeric(dtype):
        if distinct <= 1:
            return dim("unknown", 0.6, "This column is always the same value")
        if flag_hint and distinct <= 3:
            return dim("flag", 0.88, "Looks like a yes/no flag")
        if id_hint and not money_hint and not qty_hint and unique_ratio >= 0.6:
            semantic = "code" if _hits(tokens, _CODE_TOKENS) else "id"
            return (
                "identifier",
                semantic,
                0.9,
                "Looks like a unique reference for each row — useful for matching, "
                "not for maths",
            )
        if money_hint or qty_hint or pct_hint:
            semantic, conf = _semantic_for_measure(
                tokens, stats.get("min"), stats.get("max")
            )
            return "measure", semantic, conf, _measure_hint(semantic)
        if distinct <= 25:
            if date_hint:
                return dim(
                    "date",
                    0.7,
                    "Looks like a time period you can group or trend over",
                )
            if id_hint:
                return dim("code", 0.75, "Looks like a short code that groups your data")
            return dim("category", 0.72, "Looks like a label that groups your data")
        if id_hint:
            return (
                "identifier",
                "code" if _hits(tokens, _CODE_TOKENS) else "id",
                0.8,
                "Looks like a unique reference for each row — useful for matching, "
                "not for maths",
            )
        if _is_integer(dtype) and unique_ratio > 0.98 and distinct > 1000 and null_pct == 0.0:
            return (
                "identifier",
                "id",
                0.65,
                "Looks like a row number or reference — useful for matching, not "
                "for maths",
            )
        semantic, conf = _semantic_for_measure(tokens, stats.get("min"), stats.get("max"))
        if semantic == "numeric":
            conf = 0.68 if unique_ratio < 0.05 else 0.75
        return "measure", semantic, conf, _measure_hint(semantic)

    # ---- text-ish columns --------------------------------------------------
    if _is_string(dtype) or str(dtype).startswith(("List", "Array", "Struct", "Object")):
        if distinct <= 0:
            return dim("unknown", 0.5, "This column is completely empty")
        if parse_rate >= 0.8:
            conf = 0.9 if date_hint else 0.78
            return "datetime", "date", conf, "Looks like a date you can trend over"
        if id_hint and unique_ratio >= 0.5:
            return (
                "identifier",
                "code" if _hits(tokens, _CODE_TOKENS) else "id",
                0.88,
                "Looks like a unique reference for each row — useful for matching, "
                "not for maths",
            )
        if contact_hint:
            return (
                "identifier",
                "id",
                0.8,
                "Looks like a unique contact detail — useful for matching, not "
                "for maths",
            )
        if text_hint and distinct > 25:
            return "text", "free_text", 0.85, "Looks like free-form notes or comments"
        if distinct <= 1:
            return dim("unknown", 0.6, "This column is always the same value")
        if distinct <= 25:
            if flag_hint and distinct <= 3:
                return dim("flag", 0.85, "Looks like a yes/no flag")
            if id_hint:
                return dim("code", 0.75, "Looks like a short code that groups your data")
            return dim("category", 0.85, "Looks like a label that groups your data")
        if avg_len > 25 and unique_ratio > 0.6:
            return "text", "free_text", 0.8, "Looks like free-form notes or comments"
        if unique_ratio > 0.95 and distinct > 50:
            return (
                "identifier",
                "id",
                0.7,
                "Almost every row has its own value — useful for matching, not "
                "for maths",
            )
        if cat_hint:
            return dim("category", 0.7, "Looks like a label that groups your data")
        return dim("category", 0.62, "Looks like a label that groups your data")

    # ---- anything else (lists, structs, ...) -------------------------------
    return "text", "unknown", 0.4, "We could not work out what this column is"


def _measure_hint(semantic: str) -> str:
    return {
        "currency": "Looks like a money amount you can analyse",
        "percent": "Looks like a percentage you can compare",
        "quantity": "Looks like a quantity you can add up",
        "count": "Looks like a count of things",
        "numeric": "Looks like a number you can analyse",
    }.get(semantic, "Looks like a number you can analyse")


# -----------------------------------------------------------------------------
# quality
# -----------------------------------------------------------------------------
def _plain_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _quality(
    *,
    profiles: list[ColumnProfile],
    rows: int,
    cols: int,
    total_rows: int,
    sampled: bool,
    duplicate_rows: int,
    null_cells: int,
) -> dict:
    cells = max(1, rows * cols)
    null_cells_pct = round(100.0 * null_cells / cells, 3)

    constant_cols = [
        p.name for p in profiles if p.distinct == 1 and p.null_pct < 99.99
    ]
    empty_cols = [p.name for p in profiles if p.distinct == 0 or p.null_pct >= 99.99]
    high_cardinality = [
        p.name
        for p in profiles
        if p.kind in ("dimension", "text")
        and (p.distinct > 200 or (p.unique_ratio > 0.5 and p.distinct > 25))
    ]

    missing_cols = [p for p in profiles if 0 < p.null_pct < 99.99]

    score = 100.0
    score -= min(30.0, null_cells_pct * 1.5)
    if rows:
        score -= min(15.0, (100.0 * duplicate_rows / rows) * 0.5)
    score -= min(10.0, 3.0 * len(constant_cols))
    score -= min(15.0, 5.0 * len(empty_cols))
    score -= min(10.0, 2.0 * len(high_cardinality))
    if not any(p.kind == "measure" for p in profiles):
        score -= 8.0
    if rows < 50:
        score -= 15.0
    score = int(max(0.0, min(100.0, round(score))))

    n_money = sum(1 for p in profiles if p.semantic == "currency")
    n_cat = sum(1 for p in profiles if p.semantic == "category")
    n_date = sum(1 for p in profiles if p.kind == "datetime")
    n_id = sum(1 for p in profiles if p.kind == "identifier")

    if rows == 0:
        if cols == 0:
            headline = "This file is empty, so there is nothing to analyse yet."
        else:
            headline = (
                "This file has column headings but no rows of data, so there is "
                "nothing to analyse yet."
            )
    else:
        looks: list[str] = []
        if n_money:
            looks.append(f"{n_money} look{'s' if n_money == 1 else ''} like money")
        if n_cat:
            looks.append(f"{n_cat} look{'s' if n_cat == 1 else ''} like categories")
        if n_date:
            looks.append(f"{n_date} look{'s' if n_date == 1 else ''} like dates")
        if n_id:
            looks.append(f"{n_id} look{'s' if n_id == 1 else ''} like ID columns")
        headline = f"{total_rows:,} rows and {cols} columns"
        if looks:
            headline += " — " + _plain_join(looks)
        headline += "."

        if empty_cols:
            headline += (
                f" {len(empty_cols)} column"
                f"{'s are' if len(empty_cols) != 1 else ' is'} completely empty."
            )
        elif missing_cols:
            worst = max(missing_cols, key=lambda p: p.null_pct)
            if null_cells_pct < 1.0:
                headline += (
                    f" {len(missing_cols)} column"
                    f"{'s have' if len(missing_cols) != 1 else ' has'} a few missing "
                    "values (under 1% of all cells)."
                )
            else:
                headline += (
                    f" {len(missing_cols)} column"
                    f"{'s have' if len(missing_cols) != 1 else ' has'} missing values "
                    f"— '{worst.name}' is the emptiest at "
                    f"{worst.null_pct:.0f}%."
                )
        else:
            headline += " Every cell is filled in."
        if constant_cols:
            headline += (
                f" {len(constant_cols)} column"
                f"{'s never' if len(constant_cols) != 1 else ' never'} change"
                " value, so they add nothing to an analysis."
            )
        if duplicate_rows:
            headline += (
                f" {duplicate_rows:,} row"
                f"{'s are' if duplicate_rows != 1 else ' is'} an exact duplicate."
            )
        if sampled:
            headline += (
                f" (Checked {rows:,} rows spread evenly across all {total_rows:,} — "
                "the rest were too many to inspect one by one.)"
            )

    return {
        "duplicate_rows": int(duplicate_rows),
        "null_cells_pct": null_cells_pct,
        "constant_cols": constant_cols,
        "high_cardinality": high_cardinality,
        "empty_cols": empty_cols,
        "score": score,
        "headline": headline,
        # --- additive extensions -------------------------------------------
        "sampled": bool(sampled),
        "sampled_rows": int(rows),
        "total_rows": int(total_rows),
        "missing_cols": [p.name for p in missing_cols],
        "text_dates": [
            p.name for p in profiles if p.kind == "datetime" and p.dtype == "String"
        ],
    }


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
def profile(dataset_id: str, *, sample_rows: int = 200_000) -> DatasetProfile:
    """Understand every column of a loaded dataset. Never raises on messy data."""
    started = time.perf_counter()
    frame = loader.store.get(dataset_id)  # KeyError with a helpful message
    total_rows = int(frame.height)
    cols = int(frame.width)

    sampled = sample_rows > 0 and total_rows > sample_rows
    if sampled:
        step = max(1, total_rows // sample_rows)
        work = frame.gather_every(step).head(sample_rows)
    else:
        work = frame
    rows = int(work.height)

    # --- pass 1: cheap facts ------------------------------------------------
    nulls, distincts = _basic_facts(work)
    stats_by_col = _all_stats(work)

    # --- per-column analysis ------------------------------------------------
    profiles: list[ColumnProfile] = []
    null_cells = 0
    for index, (name, dtype) in enumerate(zip(work.columns, work.dtypes)):
        null_count = nulls[index] if index < len(nulls) else 0
        distinct = distincts[index] if index < len(distincts) else 0
        non_null = max(0, rows - null_count)
        null_pct = round(100.0 * null_count / rows, 3) if rows else 0.0
        unique_ratio = (distinct / non_null) if non_null else 0.0
        null_cells += null_count

        raw_stats = stats_by_col.get(name, {})
        avg_len = _num(raw_stats.get("len_mean")) or 0.0

        parse_rate = 0.0
        if _is_string(dtype):
            tokens = _tokens(name)
            if _looks_like_date_name(tokens) or _values_look_like_dates(work, name):
                parse_rate = _date_parse_rate(work, name)

        kind, semantic, confidence, role_hint = _classify(
            name=name,
            dtype=dtype,
            distinct=distinct,
            non_null=non_null,
            unique_ratio=unique_ratio,
            null_pct=null_pct,
            stats=raw_stats,
            parse_rate=parse_rate,
            avg_len=avg_len,
        )

        # nudge confidence down when the evidence is thin
        if rows < 100:
            confidence = round(max(0.3, confidence - 0.15), 3)
        if null_pct > 40.0:
            confidence = round(max(0.3, confidence - 0.1), 3)
        confidence = round(float(min(0.99, max(0.0, confidence))), 3)

        stats = _build_stats(
            kind=kind,
            semantic=semantic,
            dtype=dtype,
            raw=raw_stats,
            non_null=non_null,
            work=work,
            name=name,
            distinct=distinct,
            avg_len=avg_len,
            parse_rate=parse_rate,
        )

        profiles.append(
            ColumnProfile(
                name=str(name),
                kind=kind,
                dtype=loader._dtype_name(dtype),
                semantic=semantic,
                null_pct=null_pct,
                distinct=int(distinct),
                unique_ratio=round(float(unique_ratio), 6),
                samples=_samples(
                    work,
                    name,
                    dtype,
                    spread=kind in ("measure", "datetime"),
                ),
                stats=stats,
                role_hint=role_hint,
                confidence=confidence,
            )
        )

    # --- duplicates ---------------------------------------------------------
    duplicate_rows = 0
    if rows:
        try:
            duplicate_rows = int(rows - work.unique().height)
        except Exception:
            duplicate_rows = 0

    quality = _quality(
        profiles=profiles,
        rows=rows,
        cols=cols,
        total_rows=total_rows,
        sampled=sampled,
        duplicate_rows=duplicate_rows,
        null_cells=null_cells,
    )

    result = DatasetProfile(
        dataset_id=str(dataset_id),
        rows=total_rows,
        cols=cols,
        profiled_ms=(time.perf_counter() - started) * 1000.0,
        columns=profiles,
        measures=[p.name for p in profiles if p.kind == "measure"],
        dimensions=[p.name for p in profiles if p.kind == "dimension"],
        datetimes=[p.name for p in profiles if p.kind == "datetime"],
        identifiers=[p.name for p in profiles if p.kind == "identifier"],
        quality=quality,
        suggestions=[],
        texts=[p.name for p in profiles if p.kind == "text"],
    )

    # lazy import: derived.py imports loader, so a module-level import would cycle
    try:
        from mlarena.columns import derived as _derived

        result.suggestions = list(_derived.suggest_derived(result))
    except Exception:
        result.suggestions = []

    return result


def _parsed_date_bounds(work: pl.DataFrame, name: str):
    """(min, max) of a text column that is really a date, best format wins."""
    try:
        series = work.get_column(name)
    except Exception:
        return None, None
    for fmt in _DATE_FORMATS:
        try:
            parsed = (
                series.str.to_datetime(strict=False, exact=False)
                if fmt is None
                else series.str.to_datetime(format=fmt, strict=False, exact=False)
            )
            low, high = parsed.min(), parsed.max()
        except Exception:
            continue
        if low is not None and high is not None:
            return low, high
    return None, None


def _build_stats(
    *,
    kind: str,
    semantic: str,
    dtype: Any,
    raw: dict,
    non_null: int,
    work: pl.DataFrame,
    name: str,
    distinct: int,
    avg_len: float,
    parse_rate: float = 0.0,
) -> dict:
    if kind == "measure":
        negatives = raw.get("negatives")
        zeros = raw.get("zeros")
        return {
            "min": _num(raw.get("min")),
            "max": _num(raw.get("max")),
            "mean": _num(raw.get("mean")),
            "std": _num(raw.get("std")),
            "median": _num(raw.get("median")),
            "p25": _num(raw.get("p25")),
            "p75": _num(raw.get("p75")),
            "sum": _num(raw.get("sum"), 4),
            "skew": _num(raw.get("skew"), 4),
            "negatives_pct": (
                round(100.0 * float(negatives) / non_null, 3)
                if non_null and negatives is not None
                else 0.0
            ),
            "zeros_pct": (
                round(100.0 * float(zeros) / non_null, 3)
                if non_null and zeros is not None
                else 0.0
            ),
        }

    if kind == "datetime":
        low, high = raw.get("min"), raw.get("max")
        span_days = None
        if low is None and high is None and _is_string(dtype):
            # stored as text but recognised as a date — parse it so the stats work
            low, high = _parsed_date_bounds(work, name)
        try:
            if low is not None and high is not None:
                span_days = int((high - low).days)
        except Exception:
            span_days = None
        stats = {
            "min": _json_safe(low),
            "max": _json_safe(high),
            "span_days": span_days,
        }
        if _is_string(dtype):
            stats["stored_as"] = "text"
            stats["parse_rate"] = round(float(parse_rate), 3)
        return stats

    if kind == "text":
        return {
            "avg_length": _num(avg_len, 2),
            "max_length": _num(raw.get("len_max"), 2),
        }

    # dimension / identifier / unknown
    stats: dict[str, Any] = {}
    if distinct and distinct <= 200:
        top, coverage = _top_values(work, name)
        stats["top"] = top
        stats["coverage"] = coverage
    else:
        stats["top"] = []
        stats["coverage"] = 0.0
    if semantic in ("date",):
        stats["min"] = _json_safe(raw.get("min"))
        stats["max"] = _json_safe(raw.get("max"))
    return stats


def profile_to_dict(p: DatasetProfile) -> dict:
    """Guaranteed JSON-safe rendering of a :class:`DatasetProfile`."""
    if isinstance(p, dict):
        return _json_safe(p)
    return _json_safe(asdict(p))
