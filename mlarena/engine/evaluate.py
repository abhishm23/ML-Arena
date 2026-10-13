"""ML Arena — honest measurement (contract §10).

Choosing a champion is only as good as the yardstick. This module owns the
yardsticks, and it is deliberately conservative:

* A number we cannot honestly compute is reported as **missing**, not as zero.
  If every row has the same outcome there is no "separation" to measure, and we
  say so instead of printing a fake 0.5.
* A percentage error is never computed on a base that can be zero or negative.
  When that happens we switch to a measure that cannot blow up (WAPE — the total
  error as a share of the total actual) and we tell the user we did.
* For rare-event problems, accuracy is a trap: predicting "nothing ever goes
  wrong" is 99% accurate and completely useless. That is why the headline metric
  for those questions is separation power, and why we report how many of the real
  cases a human would actually catch by reviewing only the top K flagged rows.
"""

from __future__ import annotations

import math

import numpy as np

from mlarena import config  # noqa: F401  -- env bootstrap

_SK = config.get_module("sklearn")

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def confusion_at_threshold(y_true, y_score, threshold: float = 0.5) -> dict:
    """The four numbers behind every classification metric, in plain terms."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) else None)
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "flagged": tp + fp,
        "actual_positives": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(1, len(y_true)),
    }


def classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Everything we report for a classifier. Missing values stay missing."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_score = np.nan_to_num(y_score, nan=0.5, posinf=1.0, neginf=0.0)
    out: dict = {"threshold": float(threshold)}
    out.update(confusion_at_threshold(y_true, y_score, threshold))

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    out["positives"] = n_pos
    out["negatives"] = n_neg
    out["positive_rate"] = n_pos / max(1, len(y_true))

    if _SK is None or n_pos == 0 or n_neg == 0:
        out["roc_auc"] = None
        out["pr_auc"] = None
        out["log_loss"] = None
        out["single_class"] = True
        return out

    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    out["single_class"] = False
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = None
    try:
        out["log_loss"] = float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6),
                                         labels=[0, 1]))
    except Exception:
        out["log_loss"] = None
    try:
        out["brier"] = float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0)))
    except Exception:
        out["brier"] = None
    return out


def best_f1_threshold(y_true, y_score, default: float = 0.5) -> float:
    """Pick the cut-off that best balances catching cases against false alarms.

    0.50 is an arbitrary line. On a rare-event problem it often flags almost
    nothing, which makes the money-impact figure nonsense. So we look at the
    held-back test rows and choose the cut-off that gives the best balance of
    "caught the real ones" and "did not cry wolf".
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) < 20 or len(np.unique(y_true)) < 2:
        return float(default)
    best, best_f1 = float(default), -1.0
    try:
        grid = np.unique(np.quantile(y_score, np.linspace(0.50, 0.995, 60)))
    except Exception:
        return float(default)
    for t in grid:
        pred = y_score >= t
        tp = int(np.sum(pred & (y_true == 1)))
        if tp == 0:
            continue
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        f1 = 2.0 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best = f1, float(t)
    return float(best)


def topk_capture(y_true, y_score, k: int) -> dict:
    """How many of the real problems a human catches by reviewing only the top K.

    This is the number that decides whether the model is worth using: nobody has
    time to review 40,000 rows, but everyone has time for 500.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"k": 0, "captured": 0, "total_positives": 0, "share": None,
                "precision_at_k": None, "lift": None, "share_of_rows": None}
    k = int(max(1, min(k, n)))
    order = np.argsort(-np.nan_to_num(y_score, nan=-np.inf), kind="stable")[:k]
    captured = int(y_true[order].sum())
    total_pos = int(y_true.sum())
    base = total_pos / n if n else 0.0
    precision = captured / k
    return {
        "k": k,
        "captured": captured,
        "total_positives": total_pos,
        "share": (captured / total_pos) if total_pos else None,
        "precision_at_k": precision,
        "lift": (precision / base) if base > 0 else None,
        "share_of_rows": k / n,
    }


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=0.0,
                           posinf=0.0, neginf=0.0)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    n = len(y_true)
    out: dict = {"n": n}
    if n == 0:
        return {"n": 0, "r2": None, "rmse": None, "mae": None, "mape": None,
                "wape": None, "smape": None, "mape_note": None}

    err = y_true - y_pred
    out["mae"] = float(np.mean(np.abs(err)))
    out["rmse"] = float(math.sqrt(float(np.mean(err ** 2))))
    out["mean_actual"] = float(np.mean(y_true))
    out["std_actual"] = float(np.std(y_true))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if n > 1 and denom > _EPS:
        out["r2"] = float(1.0 - float(np.sum(err ** 2)) / denom)
    else:
        out["r2"] = None

    # --- percentage errors, guarded --------------------------------------- #
    abs_actual = np.abs(y_true)
    total_actual = float(np.sum(abs_actual))
    scale = float(np.mean(abs_actual))
    if total_actual > _EPS:
        out["wape"] = float(np.sum(np.abs(err)) / total_actual)
    else:
        out["wape"] = None
    out["smape"] = float(np.mean(2.0 * np.abs(err) / (abs_actual + np.abs(y_pred) + _EPS)))

    safe = (abs_actual > max(1e-9, 0.01 * scale)) if scale > 0 else np.zeros(n, dtype=bool)
    if safe.sum() >= max(10, 0.5 * n) and scale > 0:
        out["mape"] = float(np.mean(np.abs(err[safe]) / abs_actual[safe]))
        out["mape_note"] = None
    else:
        out["mape"] = None
        out["mape_note"] = (
            "Percentage error was not used because this column contains zero or "
            "negative values, which make percentages meaningless. We report the total "
            "error as a share of the total actual instead."
        )
    return out


# --------------------------------------------------------------------------- #
# Clustering / anomaly
# --------------------------------------------------------------------------- #


def clustering_metrics(X, labels) -> dict:
    labels = np.asarray(labels)
    n = len(labels)
    uniq = [v for v in np.unique(labels) if v != -1]
    k = len(uniq)
    noise = int((labels == -1).sum())
    sizes = {int(v): int((labels == v).sum()) for v in uniq}
    out = {
        "n_clusters": k,
        "noise": noise,
        "sizes": sizes,
        "silhouette": None,
        "davies_bouldin": None,
        "largest_share": (max(sizes.values()) / n) if sizes and n else None,
    }
    if _SK is None or not (2 <= k < n):
        out["note"] = (
            "Group tightness could not be measured because the data did not split into "
            "between 2 and " + f"{max(0, n - 1)}" + " groups."
        )
        return out
    try:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

        sample = None
        if n > 20_000:
            rng = np.random.default_rng(config.DEFAULT_SEED)
            sample = rng.choice(n, 20_000, replace=False)
        Xs = X[sample] if sample is not None else X
        ls = labels[sample] if sample is not None else labels
        if len(np.unique(ls[ls != -1])) >= 2:
            out["silhouette"] = float(silhouette_score(Xs, ls))
            out["davies_bouldin"] = float(davies_bouldin_score(Xs, ls))
            out["calinski"] = float(calinski_harabasz_score(Xs, ls))
    except Exception as exc:  # pragma: no cover - defensive
        out["note"] = f"Group tightness could not be measured ({type(exc).__name__})."
    return out


def anomaly_metrics(scores, consensus=None, flags=None) -> dict:
    """For unsupervised outlier hunting there is no answer key.

    So we measure two honest things: how many records a model calls unusual, and
    how closely it agrees with what the whole field of models thinks is unusual.
    """
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    out: dict = {"n": int(len(scores))}
    if flags is None:
        flags = scores > np.quantile(scores, 0.95) if len(scores) else np.zeros(0, bool)
    flags = np.asarray(flags)
    out["flagged"] = int(np.sum(flags == 1)) if flags.dtype != bool else int(flags.sum())
    out["contamination"] = float(out["flagged"] / max(1, len(scores)))
    out["agreement"] = None
    if consensus is not None and len(consensus) == len(scores) and len(scores) > 10:
        c = np.asarray(consensus, dtype=float)
        rs = _rank(scores)
        rc = _rank(c)
        if np.std(rs) > _EPS and np.std(rc) > _EPS:
            out["agreement"] = float(np.corrcoef(rs, rc)[0, 1])
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


# --------------------------------------------------------------------------- #
# Scoring for the leaderboard bars
# --------------------------------------------------------------------------- #


def primary_value(task: str, metrics: dict, primary_metric: str):
    """Pull the headline number out of a metrics dict, or None if unavailable."""
    if primary_metric == "consensus":
        return metrics.get("agreement")
    return metrics.get(primary_metric)


def _absolute_score(task: str, primary_metric: str, value) -> float:
    """Map a metric onto 0..100 in a way a non-technical reader can interpret."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 0.0
    v = float(value)
    if primary_metric == "roc_auc":
        return float(np.clip((v - 0.5) * 200.0, 0.0, 100.0))
    if primary_metric == "r2":
        return float(np.clip(v, 0.0, 1.0) * 100.0)
    if primary_metric == "silhouette":
        return float(np.clip(v, 0.0, 1.0) * 100.0)
    if primary_metric == "consensus":
        return float(np.clip((v + 1.0) / 2.0, 0.0, 1.0) * 100.0)
    if primary_metric in ("accuracy", "f1", "precision", "recall", "pr_auc"):
        return float(np.clip(v, 0.0, 1.0) * 100.0)
    return float(np.clip(v, 0.0, 100.0))


def normalise_scores(task: str, entries: list[dict], primary_metric: str) -> list[dict]:
    """Attach a 0..100 UI score to each leaderboard entry, in place."""
    for e in entries:
        metrics = e.get("metrics") or {}
        value = e.get("primary")
        score = _absolute_score(task, primary_metric, value)
        if value is None:
            # a model we could measure nothing for still gets a bar, from whatever
            # secondary evidence we do have
            for fallback in ("f1", "accuracy", "wape", "davies_bouldin"):
                fv = metrics.get(fallback)
                if fv is not None and math.isfinite(float(fv)):
                    if fallback == "wape":
                        score = float(np.clip(1.0 - float(fv), 0.0, 1.0) * 100.0)
                    elif fallback == "davies_bouldin":
                        score = float(np.clip(1.0 / (1.0 + max(0.0, float(fv))), 0.0, 1.0) * 100.0)
                    else:
                        score = float(np.clip(float(fv), 0.0, 1.0) * 100.0)
                    break
        e["score"] = round(score, 1)
    return entries


def rank_entries(task: str, entries: list[dict], primary_metric: str) -> list[dict]:
    """Sort the field best-first, then number them. Failed models go last."""
    ok = [e for e in entries if e.get("status") == "ok"]
    bad = [e for e in entries if e.get("status") != "ok"]

    def key(e: dict):
        v = e.get("primary")
        missing = v is None or (isinstance(v, float) and not math.isfinite(v))
        return (1 if missing else 0, -(float(v) if not missing else 0.0), e.get("train_ms", 0.0))

    ok.sort(key=key)
    for i, e in enumerate(ok + bad, start=1):
        e["rank"] = i
    return ok + bad


def pick_champion(entries: list[dict]) -> dict | None:
    for e in entries:
        if e.get("status") == "ok" and e.get("primary") is not None:
            return e
    for e in entries:
        if e.get("status") == "ok":
            return e
    return entries[0] if entries else None


# --------------------------------------------------------------------------- #
# Business impact
# --------------------------------------------------------------------------- #


def business_impact(*, task: str, impact_label: str, impact_unit: str,
                    y_true=None, y_pred=None, impact_values=None,
                    flagged_mask=None, topk: dict | None = None,
                    metrics: dict | None = None, n_rows: int = 0) -> dict:
    """Turn a metric into money or records, with the formula spelled out."""
    metrics = metrics or {}
    unit = impact_unit or "records"

    if task == "classification":
        threshold = float(metrics.get("threshold", 0.5))
        if y_pred is None and y_true is not None:
            y_pred = np.zeros(len(y_true), dtype=int)
        y_pred = np.asarray(y_pred if y_pred is not None else [], dtype=int)
        flagged = int((y_pred == 1).sum())
        values = None
        if impact_values is not None:
            v = np.asarray(impact_values, dtype=float)
            if len(v) == len(y_pred):
                values = float(np.nansum(v[y_pred == 1]))
        positives = int(metrics.get("actual_positives", 0) or 0)
        caught = int(metrics.get("tp", 0) or 0)
        if unit == "currency" and values is not None:
            value = values
            formula = ("sum of the money column over the rows we flagged as at risk "
                       f"(score of {threshold:.2f} or more)")
            note = (f"The {flagged:,} rows we flag carry {_money(values)} between them. "
                    + _catch_sentence(topk))
        else:
            value = float(flagged)
            formula = "number of rows flagged as at risk"
            note = (f"We flagged {flagged:,} of {n_rows:,} rows for a human to look at"
                    + (f", and {caught:,} of the {positives:,} real cases are inside that "
                       f"list" if positives else "")
                    + ". " + _catch_sentence(topk)).strip()
        return {"label": impact_label, "value": value, "unit": unit,
                "formula": formula, "note": note}

    if task == "regression":
        mae = metrics.get("mae")
        if mae is None:
            return {"label": impact_label, "value": None, "unit": unit,
                    "formula": "not available", "note": "We could not measure the error."}
        rows = int(n_rows or metrics.get("n", 0) or 0)
        total = float(mae) * max(1, rows)
        return {
            "label": impact_label,
            "value": total,
            "unit": unit,
            "formula": ("average size of the miss on the held-back test rows, multiplied by "
                        "the number of rows in your file"),
            "note": (f"On an average row our prediction is off by about {_money(mae)}. Across "
                     f"your {rows:,} rows that adds up to roughly {_money(total)} of "
                     f"uncertainty to plan around."),
        }

    if task == "anomaly":
        flagged = int(metrics.get("flagged", 0) or 0)
        money = None
        if impact_values is not None and flagged_mask is not None:
            v = np.asarray(impact_values, dtype=float)
            mask = np.asarray(flagged_mask).astype(bool)
            if len(v) == len(mask):
                money = float(np.nansum(v[mask]))
        note = (f"{flagged:,} of {n_rows:,} records look unusual enough to be worth a look.")
        if money is not None:
            note += f" Together they carry {_money(money)}."
        return {"label": impact_label, "value": float(flagged), "unit": "records",
                "formula": "number of records that do not look like the rest",
                "note": note}

    if task == "clustering":
        return {"label": impact_label, "value": float(n_rows), "unit": unit,
                "formula": "number of rows sorted into groups",
                "note": f"Every one of the {n_rows:,} rows was placed in a group."}

    return {"label": impact_label, "value": None, "unit": unit,
            "formula": "not available", "note": ""}


def _catch_sentence(topk: dict | None) -> str:
    if not topk or not topk.get("share"):
        return ""
    return (f"If you review only the top {topk['k']:,} flagged rows you would catch about "
            f"{topk['share'] * 100:.0f}% of them.")


def _money(v: float) -> str:
    try:
        v = float(v)
    except Exception:
        return str(v)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e9:
        return f"{sign}{v / 1e9:.2f} billion"
    if v >= 1e6:
        return f"{sign}{v / 1e6:.2f} million"
    if v >= 1e3:
        return f"{sign}{v / 1e3:.1f} thousand"
    return f"{sign}{v:,.2f}"


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #


def _source_column(output_name: str, numeric_cols: list[str], cat_cols: list[str]):
    for prefix, cols in (("num__", numeric_cols), ("cat__", cat_cols)):
        if output_name.startswith(prefix):
            rest = output_name[len(prefix):]
            for c in sorted(cols, key=len, reverse=True):
                if rest == c or rest.startswith(c + "_"):
                    return c, prefix
            return rest, prefix
    return output_name, ""


def compute_drivers(fitted, X, y_true, numeric_cols: list[str], categorical_cols: list[str],
                    task: str, *, top: int = 12, seed: int = 42) -> list[dict]:
    """Which columns actually moved the prediction, and which way.

    We prefer the model's own importances, aggregate one-hot columns back to the
    business column they came from, and fall back to shuffling each column and
    measuring the damage when the model does not expose any.
    """
    numeric_cols = list(numeric_cols or [])
    categorical_cols = list(categorical_cols or [])
    if X is None or len(X) == 0:
        return []

    importances: dict[str, float] = {}
    signed: dict[str, float] = {}
    source = ""

    est = fitted
    prep = None
    try:
        if hasattr(fitted, "named_steps"):
            est = fitted.named_steps.get("model", fitted)
            prep = fitted.named_steps.get("prepare")
    except Exception:
        est = fitted

    names = None
    if prep is not None:
        try:
            names = list(prep.get_feature_names_out())
        except Exception:
            names = None

    raw = None
    signed_raw = None
    if hasattr(est, "feature_importances_"):
        raw = np.asarray(est.feature_importances_, dtype=float)
        source = "the model's own ranking of what it used most"
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        raw = np.abs(coef)
        signed_raw = coef
        source = "the weights the model learned for each column"

    if raw is not None:
        if names is None or len(names) != len(raw):
            names = numeric_cols + categorical_cols
            if len(names) != len(raw):
                return []
        for i, name in enumerate(names):
            col, kind = _source_column(str(name), numeric_cols, categorical_cols)
            importances[col] = importances.get(col, 0.0) + float(raw[i])
            if signed_raw is not None:
                signed[col] = signed.get(col, 0.0) + float(signed_raw[i])
    else:
        # permutation importance on a sample — slower but always available
        if _SK is None:
            return []
        from sklearn.inspection import permutation_importance

        try:
            n = len(X)
            idx = np.arange(n)
            if n > 5000:
                idx = np.random.default_rng(seed).choice(n, 5000, replace=False)
            Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            ys = np.asarray(y_true)[idx]
            scoring = "roc_auc" if task == "classification" else "r2"
            if task == "classification" and len(np.unique(ys)) < 2:
                scoring = "accuracy"
            r = permutation_importance(fitted, Xs, ys, n_repeats=3, random_state=seed,
                                       scoring=scoring, n_jobs=1)
            cols = list(X.columns) if hasattr(X, "columns") else []
            for i, col in enumerate(cols):
                importances[str(col)] = float(max(0.0, r.importances_mean[i]))
            source = "how much the model's accuracy dropped when we shuffled each column"
        except Exception:
            return []

    if not importances:
        return []

    total = sum(v for v in importances.values() if v > 0) or 1.0
    ranked = sorted(importances.items(), key=lambda kv: -kv[1])[:top]

    out: list[dict] = []
    for name, val in ranked:
        if val <= 0:
            continue
        direction = "neutral"
        if name in signed:
            s = signed[name]
            direction = "up" if s > 0 else ("down" if s < 0 else "neutral")
        else:
            direction = _direction_from_data(X, name, y_true, categorical=name in categorical_cols)
        out.append({
            "feature": name,
            "importance": round(float(val / total) * 100.0, 2),
            "raw_importance": float(val),
            "direction": direction,
            "kind": "category" if name in categorical_cols else "number",
            "source": source,
        })
    return out


def _direction_from_data(X, name: str, y_true, *, categorical: bool) -> str:
    try:
        if name not in getattr(X, "columns", []):
            return "neutral"
        y = np.asarray(y_true, dtype=float)
        col = X[name]
        if categorical:
            # compare the average outcome when this category is present at all
            vals = col.astype(str).to_numpy()
            vc = {}
            for v in np.unique(vals):
                m = vals == v
                if m.sum() >= 20:
                    vc[v] = float(np.nanmean(y[m]))
            if len(vc) < 2:
                return "neutral"
            hi = max(vc, key=vc.get)
            lo = min(vc, key=vc.get)
            if abs(vc[hi] - vc[lo]) < 1e-9:
                return "neutral"
            return f"category:{hi}"
        x = np.asarray(col, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 30 or np.std(x[ok]) < _EPS:
            return "neutral"
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if r > 0.05:
            return "up"
        if r < -0.05:
            return "down"
        return "neutral"
    except Exception:
        return "neutral"


__all__ = [
    "classification_metrics",
    "regression_metrics",
    "clustering_metrics",
    "anomaly_metrics",
    "confusion_at_threshold",
    "best_f1_threshold",
    "topk_capture",
    "normalise_scores",
    "rank_entries",
    "pick_champion",
    "primary_value",
    "business_impact",
    "compute_drivers",
]
