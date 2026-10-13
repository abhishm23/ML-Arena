"""ML Arena — the orchestrator (contract §10).

`run()` is the whole show: it loads, prepares, plans, races the models one at a
time, picks a champion, scores every row and explains what happened. Two
behaviours matter more than anything else in this file:

* **A failing model never kills the run.** Every model is trained inside its own
  try/except. If one blows up, the user sees a friendly warning explaining why,
  that model is recorded in the leaderboard with `status="failed"`, and the race
  carries on. A user who waited two minutes for an answer must always get one.
* **Every event explains itself.** The UI is a live show, and each event carries a
  `detail` sentence written for someone who has never heard the word "model".
"""

from __future__ import annotations

import math
import time
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from mlarena import config  # noqa: F401  -- env bootstrap before polars
import polars as pl

from mlarena.engine import evaluate as ev
from mlarena.engine import models as model_zoo
from mlarena.engine import narration as nar
from mlarena.engine.planner import make_plan
from mlarena.usecases import definitions as defs
from mlarena.usecases import registry

# --------------------------------------------------------------------------- #
# RunResult (contract §10)
# --------------------------------------------------------------------------- #


@dataclass
class RunResult:
    run_id: str
    status: str
    usecase: dict
    dataset: dict
    target: dict
    plan: dict
    leaderboard: list[dict]
    champion: dict
    metrics: dict
    impact: dict
    drivers: list[dict]
    insights: list[dict]
    segments: list[dict]
    distributions: dict
    preview: dict
    scored_path: str
    scored_columns: list[str]
    exports: dict
    narration: list[dict]
    log: list[dict]
    error: str | None
    started_at: str
    finished_at: str
    duration_ms: float
    #: Per-row "why was this one flagged?" — additive to contract §10.
    explanations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            k: getattr(self, k) for k in (
                "run_id", "status", "usecase", "dataset", "target", "plan", "leaderboard",
                "champion", "metrics", "impact", "drivers", "insights", "segments",
                "distributions", "preview", "scored_path", "scored_columns", "exports",
                "narration", "log", "error", "started_at", "finished_at", "duration_ms",
                "explanations",
            )
        }


# --------------------------------------------------------------------------- #
# Progress map — every phase owns a slice of the bar
# --------------------------------------------------------------------------- #

_PHASES = {
    "load": (0.00, 0.05),
    "profile": (0.05, 0.08),
    "prepare": (0.08, 0.16),
    "plan": (0.16, 0.20),
    "train": (0.20, 0.72),
    "select": (0.72, 0.78),
    "score": (0.78, 0.90),
    "explain": (0.90, 0.97),
    "export": (0.97, 0.99),
    "done": (0.99, 1.00),
    "error": (0.00, 0.00),
}


class _Emitter:
    """Collects every event, mirrors it to the caller, and keeps the log."""

    def __init__(self, emit, started: float):
        self._emit = emit
        self.log: list[dict] = []
        self.seq = 0
        self.started = started
        self._phase_seen: dict[str, float] = {}

    def __call__(self, phase: str, level: str, title: str, detail: str = "", *,
                 progress: float | None = None, model: str | None = None,
                 metrics: dict | None = None, payload: dict | None = None) -> dict:
        self.seq += 1
        if progress is None:
            lo, hi = _PHASES.get(phase, (0.0, 0.0))
            frac = min(0.99, self._phase_seen.get(phase, 0.0))
            progress = lo + (hi - lo) * frac
        title = (title or "")[:80]
        event = {
            "seq": self.seq,
            "ts": int(time.time() * 1000),
            "phase": phase,
            "level": level,
            "title": title,
            "detail": detail or "",
            "progress": round(float(progress), 4),
            "model": model,
            "metrics": _json_safe(metrics) if metrics else None,
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000.0, 1),
            "payload": _json_safe(payload) if payload else {},
        }
        self.log.append(event)
        if self._emit is not None:
            try:
                self._emit(event)
            except Exception:
                pass
        return event

    def phase_progress(self, phase: str, frac: float) -> None:
        self._phase_seen[phase] = max(0.0, min(1.0, frac))


# --------------------------------------------------------------------------- #
# JSON safety
# --------------------------------------------------------------------------- #


def _json_safe(v):
    import datetime as _dt

    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if not math.isfinite(v) else round(v, 6)
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    if isinstance(v, np.generic):
        return _json_safe(v.item())
    if isinstance(v, np.ndarray):
        return [_json_safe(x) for x in v.tolist()]
    if isinstance(v, (list, tuple, set)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _json_safe(x) for k, x in v.items()}
    return str(v)


def _finite(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Design matrix, splitting, model fitting
# --------------------------------------------------------------------------- #


def _design(frame: pl.DataFrame, num_cols: list[str], cat_cols: list[str]):
    return defs._pandas_features(frame, num_cols, cat_cols)


def _split_indices(target, plan, y, seed: int):
    """Return (train_idx, test_idx). Unsupervised tasks get no test set."""
    n = len(y)
    if plan.test_strategy == "unsupervised" or n < 40:
        return np.arange(n), np.array([], dtype=int)

    if plan.test_strategy == "time_split" and target.extra.get("time_col"):
        tcol = target.extra["time_col"]
        if tcol in target.frame.columns:
            order = np.argsort(target.frame.get_column(tcol).to_numpy(), kind="stable")
            cut = int(n * (1.0 - plan.test_size))
            return order[:cut], order[cut:]

    rng = np.random.default_rng(seed)
    if plan.test_strategy == "stratified_split" and target.target_col is not None:
        y = np.asarray(y)
        train_parts, test_parts = [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            idx = idx[rng.permutation(len(idx))]
            cut = max(1, int(len(idx) * (1.0 - plan.test_size)))
            train_parts.append(idx[:cut])
            test_parts.append(idx[cut:])
        if test_parts and all(len(p) for p in test_parts):
            return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(test_parts))
    perm = rng.permutation(n)
    cut = int(n * (1.0 - plan.test_size))
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def _stratified_sample(y, n_take: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    take = []
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        k = max(1, int(round(n_take * cnt / len(y))))
        k = min(k, len(idx))
        take.append(rng.choice(idx, k, replace=False))
    out = np.concatenate(take)
    return np.sort(out)


def _score_estimator(pipe, X, task: str) -> np.ndarray:
    """One number per row, always pointing the same way: bigger = more of the thing."""
    if task == "classification":
        if hasattr(pipe, "predict_proba"):
            p = np.asarray(pipe.predict_proba(X))
            return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else np.asarray(p).ravel()
        if hasattr(pipe, "decision_function"):
            d = np.asarray(pipe.decision_function(X), dtype=float).ravel()
            return 1.0 / (1.0 + np.exp(-d))
        return np.asarray(pipe.predict(X), dtype=float)
    if task == "anomaly":
        if hasattr(pipe, "decision_function"):
            return -np.asarray(pipe.decision_function(X), dtype=float).ravel()
        if hasattr(pipe, "score_samples"):
            return -np.asarray(pipe.score_samples(X), dtype=float).ravel()
        return -np.asarray(pipe.predict(X), dtype=float).ravel()
    return np.asarray(pipe.predict(X), dtype=float).ravel()


def _predict_batches(fn, n: int, batch: int = 120_000):
    out = np.empty(n, dtype=float)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        out[start:stop] = fn(start, stop)
    return out


def _choose_k(X, seed: int, k_min: int = 2, k_max: int = 6) -> tuple[int, list[dict]]:
    """How many groups are actually there? Let the data answer."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    if n < 20:
        return 2, []
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= 15_000 else rng.choice(n, 15_000, replace=False)
    Xs = _numeric_matrix(X, idx)
    if Xs is None or len(Xs) < 20:
        return 3, []
    try:
        Xs = StandardScaler().fit_transform(Xs)
    except Exception:
        return 3, []
    kmax = max(k_min, min(k_max, max(2, n // 30)))
    results = []
    for k in range(k_min, kmax + 1):
        try:
            labels = MiniBatchKMeans(n_clusters=k, n_init=5, batch_size=2048,
                                     random_state=seed).fit_predict(Xs)
            if len(np.unique(labels)) < 2:
                continue
            sub = min(5000, len(Xs))
            si = np.arange(len(Xs)) if sub == len(Xs) else rng.choice(len(Xs), sub, replace=False)
            sil = float(silhouette_score(Xs[si], labels[si]))
            results.append({"k": k, "silhouette": round(sil, 4)})
        except Exception:
            continue
    if not results:
        return 3, []
    best = max(results, key=lambda r: r["silhouette"])["k"]
    return int(best), results


def _numeric_matrix(X, idx=None):
    """Numeric-only numpy view of the design matrix (categories already encoded)."""
    try:
        sub = X if idx is None else X.iloc[idx]
        num = sub.select_dtypes(include=[np.number])
        if num.shape[1] == 0:
            return None
        return np.nan_to_num(num.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return None


def _dbscan_eps(X, seed: int, min_samples: int = 10) -> float:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(n) if n <= 8000 else rng.choice(n, 8000, replace=False)
    Xs = _numeric_matrix(X, idx)
    if Xs is None or len(Xs) < 20:
        return 0.5
    try:
        Xs = StandardScaler().fit_transform(Xs)
        nn = NearestNeighbors(n_neighbors=min(min_samples, max(2, len(Xs) - 1))).fit(Xs)
        d, _ = nn.kneighbors(Xs)
        return float(np.quantile(d[:, -1], 0.9))
    except Exception:
        return 0.5


# --------------------------------------------------------------------------- #
# Insights
# --------------------------------------------------------------------------- #


def _insight(title: str, body: str, severity: str = "info", icon: str = "•") -> dict:
    return {"title": title, "body": body, "severity": severity, "icon": icon}


def _money(v) -> str:
    return ev._money(v)


def _build_insights(*, task: str, target, plan, champion: dict, metrics: dict, impact: dict,
                    drivers: list[dict], topk: dict | None,
                    cluster_scores: list[dict] | None) -> list[dict]:
    out: list[dict] = []
    name = champion.get("display", "the winning model")
    primary = champion.get("primary")

    if task == "classification":
        full = metrics.get("all_rows") or {}
        flagged = full.get("flagged") or metrics.get("flagged") or 0
        pos = full.get("actual_positives") or 0
        if primary is not None:
            if primary >= 0.8:
                out.append(_insight(
                    f"{name} separates the two outcomes very clearly",
                    f"A separation score of {primary:.2f} means the model ranks almost every "
                    f"real case above the safe ones. This is a pattern worth acting on.",
                    "good", "✔"))
            elif primary >= 0.65:
                out.append(_insight(
                    f"{name} finds a real but imperfect pattern",
                    f"A separation score of {primary:.2f} is a genuine signal — good enough to "
                    f"prioritise a review list, not good enough to act on automatically.",
                    "info", "•"))
            else:
                out.append(_insight(
                    "The signal here is weak",
                    f"A separation score of {primary:.2f} is only slightly better than guessing "
                    f"(0.50). Treat the ranking as a weak hint, and consider adding columns that "
                    f"describe the situation more directly.",
                    "warn", "!"))
        if topk and topk.get("share"):
            out.append(_insight(
                f"Reviewing the top {topk['k']:,} rows catches "
                f"{topk['share'] * 100:.0f}% of the problem",
                f"{topk['captured']:,} of the {topk['total_positives']:,} real cases sit inside "
                f"the top {topk['k']:,} rows by risk — that is only "
                f"{topk['share_of_rows'] * 100:.1f}% of your file. Start the review there."
                + (f" Flagged rows are {topk['lift']:.1f}× more likely to be a real case than a "
                   f"random row." if topk.get("lift") else ""),
                "good", "★"))
        if pos and flagged:
            cut = metrics.get("threshold")
            cut_txt = f"{cut:.2f}" if isinstance(cut, (int, float)) else "the cut-off"
            out.append(_insight(
                "How the cut-off was set",
                f"Rows scoring {cut_txt} or higher are flagged: {flagged:,} rows across your "
                f"whole file, of which {full.get('tp', 0):,} are real cases and "
                f"{full.get('fp', 0):,} are false alarms. {full.get('fn', 0):,} real cases fall "
                f"below the line. Move the cut-off to trade one against the other.",
                "info", "•"))
        if impact.get("value") and impact.get("unit") == "currency":
            out.append(_insight(
                f"{impact['label']}: {_money(impact['value'])}",
                impact.get("note", ""), "warn", "\u25c6"))

    elif task == "regression":
        r2 = metrics.get("r2")
        if r2 is not None:
            if r2 >= 0.7:
                out.append(_insight(
                    "This is a strong, usable prediction",
                    f"The model reproduces {r2 * 100:.0f}% of the ups and downs in your numbers, "
                    f"which is high enough to plan against.", "good", "✔"))
            elif r2 >= 0.3:
                out.append(_insight(
                    "The prediction captures the main pattern, not the detail",
                    f"{r2 * 100:.0f}% of the variation is explained. Use it to rank and to spot "
                    f"direction, not to promise an exact figure.", "info", "•"))
            else:
                out.append(_insight(
                    "The model explains very little",
                    f"Only {max(0.0, r2) * 100:.0f}% of the variation is explained, so this set "
                    f"of columns does not really determine your number. Try adding columns that "
                    f"describe the situation more directly.", "warn", "!"))
        mae = metrics.get("mae")
        if mae is not None:
            out.append(_insight(
                "How far off a typical prediction is",
                f"On an average row the prediction is about {mae:,.2f} away from the truth "
                f"(in the same units as your column).", "info", "•"))
        if metrics.get("mape_note"):
            out.append(_insight("Why there is no percentage error",
                                metrics["mape_note"], "info", "•"))

    elif task == "anomaly":
        out.append(_insight(
            f"{metrics.get('flagged', 0):,} records stand out",
            f"That is {metrics.get('contamination', 0) * 100:.1f}% of your file. This is a "
            f"shortlist for a human to glance at, ordered by how unusual each record is — "
            f"unusual is not the same as wrong.", "info", "•"))
        if drivers:
            out.append(_insight(
                f"What makes a record unusual: {drivers[0]['feature']}",
                nar.driver_sentence(drivers[0]), "info", "•"))

    elif task == "clustering":
        k = metrics.get("n_clusters")
        if k:
            out.append(_insight(
                f"Your customers fall into {k} natural groups",
                "The grouping was not imposed by us — the model found the arrangement that fits "
                "your data best. See the group profiles below for what makes each one different.",
                "good", "✔"))
        if cluster_scores:
            best = max(cluster_scores, key=lambda r: r["silhouette"])
            out.append(_insight(
                f"Why {best['k']} groups",
                "We tried every sensible number of groups and kept the one where the groups were "
                "tightest and most clearly separated. " + ", ".join(
                    f"{r['k']} groups scored {r['silhouette']:.2f}" for r in cluster_scores[:4])
                + ".", "info", "•"))
        if metrics.get("silhouette") is not None and metrics["silhouette"] < 0.2:
            out.append(_insight(
                "The groups overlap a fair amount",
                "A tightness score below 0.20 means many customers sit between groups. The "
                "grouping is still useful for description, but do not treat it as a hard "
                "classification.", "warn", "!"))

    if drivers:
        d = drivers[0]
        out.append(_insight(
            f"The strongest signal is “{d['feature']}”",
            nar.driver_sentence(d) + f" ({d.get('source', 'ranked by the model')}.)",
            "info", "•"))
    if plan.sample_rows:
        out.append(_insight("We trained on a sample", plan.sample_reason, "info", "•"))
    return out[:6]


# --------------------------------------------------------------------------- #
# Segments / distributions
# --------------------------------------------------------------------------- #


def _cluster_segments(labels, frame, num_cols, cat_cols, key_cols) -> list[dict]:
    labels = np.asarray(labels)
    out = []
    n = len(labels)
    for cid in sorted(set(int(v) for v in np.unique(labels))):
        mask = labels == cid
        size = int(mask.sum())
        if size == 0:
            continue
        profile = []
        for c in num_cols[:6]:
            try:
                vals = frame.get_column(c).to_numpy()[mask]
                vals = vals[np.isfinite(vals.astype(float))]
                if len(vals):
                    profile.append({"feature": c, "mean": round(float(np.mean(vals)), 4),
                                    "overall": round(float(np.nanmean(
                                        frame.get_column(c).to_numpy().astype(float))), 4)})
            except Exception:
                continue
        label = "Noise / outliers" if cid == -1 else f"Group {cid + 1}"
        # name the group by the feature it stands out on most
        distinguishing = None
        if profile:
            def _dev(p):
                o = p["overall"] or 0.0
                return abs(p["mean"] - o) / (abs(o) + 1e-9)
            best = max(profile, key=_dev)
            if _dev(best) > 0.15:
                direction = "higher" if best["mean"] > (best["overall"] or 0) else "lower"
                distinguishing = (f"{direction} {best['feature']} than average "
                                  f"({best['mean']:,.2f} vs {best['overall']:,.2f})")
        out.append({
            "id": cid,
            "label": label,
            "size": size,
            "share": round(size / max(1, n), 4),
            "distinguishing": distinguishing,
            "profile": profile,
            "plain": (f"{size:,} rows ({size / max(1, n) * 100:.0f}% of the file)"
                      + (f", typically {distinguishing}." if distinguishing else ".")),
        })
    return out


def _anomaly_segments(scores, frame, num_cols, key_cols) -> list[dict]:
    n = len(scores)
    if n == 0:
        return []
    q99, q95, q90 = np.quantile(scores, [0.99, 0.95, 0.90])
    bands = [("Most unusual 1%", scores >= q99), ("Next 4%", (scores >= q95) & (scores < q99)),
             ("Next 5%", (scores >= q90) & (scores < q95)), ("The rest", scores < q90)]
    out = []
    for label, mask in bands:
        size = int(mask.sum())
        if not size:
            continue
        profile = []
        for c in num_cols[:5]:
            try:
                vals = frame.get_column(c).to_numpy().astype(float)
                sel = vals[mask]
                sel = sel[np.isfinite(sel)]
                if len(sel):
                    profile.append({"feature": c, "mean": round(float(np.mean(sel)), 4),
                                    "overall": round(float(np.nanmean(vals)), 4)})
            except Exception:
                continue
        out.append({"id": label, "label": label, "size": size,
                    "share": round(size / n, 4), "profile": profile,
                    "plain": f"{size:,} records ({size / n * 100:.1f}% of the file)."})
    return out


def _histogram(values, bins: int = 20) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"bins": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"bins": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts]}


def _distributions(*, task, target, champion, leaderboard, primary_metric, score_all,
                   y_true_all, actual_all, drivers, segments, timeseries,
                   impact_values=None, sample_cap=1500) -> dict:
    out: dict = {}
    if score_all is not None and len(score_all):
        out["score_histogram"] = _histogram(score_all, 20)

    out["leaderboard"] = {
        "labels": [e["display"] for e in leaderboard],
        "values": [float(e["primary"]) if e.get("primary") is not None else 0.0
                   for e in leaderboard],
    }

    if task == "classification" and y_true_all is not None:
        n_pos = int(np.sum(y_true_all == 1))
        n_neg = int(len(y_true_all) - n_pos)
        labels = (target.label_names or {0: "No", 1: "Yes"})
        out["class_balance"] = {
            "labels": [str(labels.get(0, "No")), str(labels.get(1, "Yes"))],
            "values": [n_neg, n_pos],
        }
        rng = np.random.default_rng(config.DEFAULT_SEED)
        n = len(score_all)
        idx = np.arange(n) if n <= sample_cap else rng.choice(n, sample_cap, replace=False)
        x = score_all[idx]
        if impact_values is not None and len(impact_values) == n:
            y = np.asarray(impact_values, dtype=float)[idx]
            y_label = f"{target.impact_col or 'value'} on the row"
        else:
            y = np.arange(len(idx), dtype=float)
            y_label = "Row number"
        out["scatter"] = {
            "x": [round(float(v), 4) for v in x],
            "y": [round(float(v), 4) if np.isfinite(v) else 0.0 for v in y],
            "color": [int(v) for v in y_true_all[idx]],
            "x_label": "Risk score given by the model",
            "y_label": y_label,
        }
        bands = np.array([0, 20, 40, 60, 80, 100], dtype=float)
        idxs = np.clip(np.digitize(score_all * 100.0, bands) - 1, 0, len(bands) - 2)
        counts = [int(np.sum(idxs == i)) for i in range(len(bands) - 1)]
        positives = [int(np.sum(y_true_all[idxs == i] == 1)) for i in range(len(bands) - 1)]
        out["risk_bands"] = {
            "labels": [f"{int(bands[i])}-{int(bands[i + 1])}" for i in range(len(bands) - 1)],
            "values": counts,
            "positives": positives,
        }

    if task == "regression" and actual_all is not None:
        rng = np.random.default_rng(config.DEFAULT_SEED)
        n = len(actual_all)
        idx = np.arange(n) if n <= sample_cap else rng.choice(n, sample_cap, replace=False)
        a = np.asarray(actual_all, dtype=float)[idx]
        p = np.asarray(score_all, dtype=float)[idx]
        err = np.abs(a - p)
        thr = float(np.nanquantile(err, 0.9)) if len(err) else 0.0
        out["scatter"] = {
            "x": [round(float(v), 4) for v in a],
            "y": [round(float(v), 4) for v in p],
            "color": [1 if e > thr else 0 for e in err],
            "x_label": "What actually happened",
            "y_label": "What the model predicted",
        }

    if task == "anomaly" and score_all is not None:
        labels = [s["label"] for s in segments] or ["Unusual", "Normal"]
        values = [s["size"] for s in segments] or [0, 0]
        out["segments"] = {"labels": labels, "values": values}

    if task == "clustering" and segments:
        out["segments"] = {"labels": [s["label"] for s in segments],
                           "values": [s["size"] for s in segments]}

    if timeseries:
        out["timeseries"] = timeseries

    if drivers:
        out["feature_impact"] = {"labels": [d["feature"] for d in drivers[:10]],
                                 "values": [d["importance"] for d in drivers[:10]]}

    if task in ("classification", "regression") and champion.get("metrics"):
        m = champion["metrics"]
        if task == "classification":
            axes = ["Separation", "Rare-case skill", "Balanced score", "Hit rate"]
            vals = [m.get("roc_auc"), m.get("pr_auc"), m.get("f1"), m.get("accuracy")]
        else:
            axes = ["Explained variation", "Error level", "Average miss"]
            r2 = m.get("r2")
            vals = [max(0.0, r2) if r2 is not None else None,
                    1.0 - min(1.0, m["wape"]) if m.get("wape") is not None else None,
                    None]
        series = [{"name": champion["display"],
                   "values": [round(float(v), 4) if v is not None else 0.0 for v in vals]}]
        out["metric_radar"] = {"axes": axes, "series": series}
    return out


# --------------------------------------------------------------------------- #
# Why was THIS row flagged?
# --------------------------------------------------------------------------- #


def _reference_values(X, feats: list[str]) -> dict:
    """The 'typical record' value for each column we are going to probe."""
    import pandas as pd

    refs: dict = {}
    for f in feats:
        col = X[f]
        if pd.api.types.is_numeric_dtype(col):
            values = col.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            refs[f] = float(np.median(values)) if len(values) else 0.0
        else:
            modes = col.astype("object").where(col.notna(), None).dropna()
            refs[f] = modes.mode().iloc[0] if len(modes) else None
    return refs


#: Column-name hints that a number is really a proportion, mirroring the profiler.
_PERCENT_HINTS = ("pct", "percent", "rate", "ratio", "margin", "discount", "share",
                  "yield", "conversion", "utilisation", "utilization")


def _percent_like(name: str, column) -> bool:
    """True when a column is a 0-1 proportion that should be shown as a percentage.

    The profiler already tags these, but the engine works on a feature frame with no
    profile attached, so we re-apply the same two tests it uses: the name hints at a
    rate, and every value sits inside a proportional range.
    """
    low = str(name).lower()
    if not any(h in low for h in _PERCENT_HINTS):
        return False
    try:
        values = np.asarray(column, dtype=float)
    except (TypeError, ValueError):
        return False
    values = values[np.isfinite(values)]
    if not len(values):
        return False
    return float(values.min()) >= -2.0 and float(values.max()) <= 2.0


def _explain_rows(champ_pipe, X_all, target, scores, drivers, task, *,
                  source=None, top_rows: int = 12, top_feats: int = 5) -> list[dict]:
    """Explain the individual records the champion is most worried about.

    Model-agnostic ablation: take the row, swap one column at a time for the value a
    *typical* record has, and watch how far the score moves. Whatever moves it most
    is what drove the decision for that row.

    We do it this way rather than with SHAP because it needs no extra dependency,
    works for every estimator in the zoo including the ones with no coefficients or
    importances, and produces a sentence a business user can verify by eye against
    their own row: "its discount was 34%, where a typical order is 8%".
    """
    if champ_pipe is None or X_all is None or scores is None or len(X_all) == 0:
        return []
    if task not in ("classification", "anomaly"):
        return []

    import pandas as pd

    values = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    if not len(values):
        return []
    n = len(values)
    take = int(min(top_rows, n))
    order = np.argsort(-values)[:take]

    feats = [d.get("feature") for d in (drivers or []) if d.get("feature") in X_all.columns]
    if not feats:
        feats = [c for c in X_all.columns][:top_feats]
    feats = list(dict.fromkeys(feats))[:top_feats]
    if not feats:
        return []

    refs = _reference_values(X_all, feats)
    base = X_all.iloc[order].reset_index(drop=True)

    # One batched prediction for the originals plus every single-column variant.
    blocks = [base]
    position: dict[tuple[str, int], int] = {}
    for i in range(take):
        position[("__orig__", i)] = i
    for f in feats:
        variant = base.copy()
        variant[f] = refs.get(f)
        blocks.append(variant)
        offset = sum(len(b) for b in blocks[:-1])
        for i in range(take):
            position[(f, i)] = offset + i

    big = pd.concat(blocks, ignore_index=True)
    try:
        perturbed = _score_estimator(champ_pipe, big, task)
    except Exception:
        return []
    perturbed = np.nan_to_num(np.asarray(perturbed, dtype=float), nan=0.0)
    if len(perturbed) < len(big):
        return []

    frame = target.frame
    # The feature frame folds rare categories into "Other", so it cannot tell a user
    # which record they are looking at. When the original file still lines up row for
    # row, label from that instead and prefer columns that read like an identifier.
    label_frame = frame
    if source is not None and len(source) == len(frame):
        label_frame = source
    _ID_TOKENS = ("id", "code", "no", "num", "ref", "key", "sku", "invoice",
                  "order", "txn", "transaction", "account", "loan", "policy")
    keys: list[str] = []
    for c in label_frame.columns:
        low = str(c).lower()
        if c != target.impact_col and any(t in low for t in _ID_TOKENS):
            keys.append(c)
    for c in (target.key_cols or []):
        if c in label_frame.columns and c not in keys and c != target.impact_col:
            keys.append(c)
    keys = keys[:2]

    def _label_for(row_idx: int) -> str:
        parts = []
        for c in keys:
            try:
                text = str(label_frame.get_column(c)[int(row_idx)])
            except Exception:
                continue
            if text in ("Other", "None", "null", "", "nan", "NaT"):
                continue
            parts.append(text)
        return " \u00b7 ".join(parts) or f"row {int(row_idx) + 1:,}"

    out: list[dict] = []
    for i, row_idx in enumerate(order):
        original = float(perturbed[position[("__orig__", i)]])
        contributions = []
        for f in feats:
            reference = refs.get(f)
            here = base[f].iloc[i]
            is_category = not pd.api.types.is_numeric_dtype(X_all[f])
            as_percent = (not is_category) and _percent_like(f, X_all[f])
            delta = original - float(perturbed[position[(f, i)]])
            contributions.append({
                "feature": f,
                "value": _json_safe(here),
                "reference": _json_safe(reference),
                "contribution": round(float(delta), 4),
                "kind": "percent" if as_percent else ("category" if is_category else "number"),
                "plain": nar.local_explanation(
                    f, here, reference, delta,
                    is_category=is_category, as_percent=as_percent),
            })
        contributions.sort(key=lambda c: -abs(c["contribution"]))
        rising = [c for c in contributions if c["contribution"] > 0]
        label = _label_for(row_idx)

        out.append({
            "row_index": int(row_idx),
            "label": label,
            "score": round(original, 4),
            "summary": nar.row_summary(original, rising, label),
            "drivers": contributions,
        })
    return out


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def run(dataset_id: str, usecase_key: str, mapping: dict[str, list[str]],
        options: dict | None = None, emit=None, run_id: str | None = None) -> RunResult:
    options = dict(options or {})
    started_perf = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = run_id or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    emit = _Emitter(emit, started_perf)
    seed = int(options.get("seed", config.DEFAULT_SEED))
    narration: list[dict] = []

    result = RunResult(
        run_id=run_id, status="failed", usecase={}, dataset={}, target={}, plan={},
        leaderboard=[], champion={}, metrics={}, impact={}, drivers=[], insights=[],
        segments=[], distributions={}, preview={"columns": [], "rows": []},
        scored_path="", scored_columns=[], exports={}, narration=narration,
        log=emit.log, error=None, started_at=started_at, finished_at=started_at,
        duration_ms=0.0,
    )

    try:
        config.ensure_dirs()

        # ---- 1. load ------------------------------------------------------ #
        uc = registry.get_usecase(usecase_key)
        result.usecase = registry.serialize(uc)
        emit("load", "info", f"Opening your data for “{uc.name}”",
             "Everything below runs on this computer. Nothing is uploaded anywhere.",
             progress=0.01)

        frame = defs._get_frame(dataset_id)
        if frame is None or frame.height == 0:
            raise ValueError("That dataset is empty or no longer open.")
        meta = None
        try:
            from mlarena.dataio import loader
            meta = loader.store.meta(dataset_id)
        except Exception:
            meta = None
        n_rows, n_cols = frame.height, frame.width
        emit("load", "success", f"{n_rows:,} rows and {n_cols} columns ready",
             f"We read your file and found {n_rows:,} rows. A row is one record — one order, "
             f"one customer, one invoice, depending on your file.",
             payload={"rows": n_rows, "cols": n_cols})
        result.dataset = {
            "dataset_id": dataset_id, "rows": int(n_rows), "cols": int(n_cols),
            "engine": getattr(meta, "engine", "polars"),
            "load_ms": float(getattr(meta, "load_ms", 0.0) or 0.0),
            "scored_rows": 0,
        }

        narration.append(nar.opening_beat(uc.name, uc.business_question, n_rows, n_cols))

        # ---- 2. profile --------------------------------------------------- #
        emit.phase_progress("profile", 0.4)
        quality = None
        try:
            from mlarena.dataio import profiler
            quality = profiler.profile(dataset_id)
        except Exception:
            quality = None
        if quality is not None and getattr(quality, "quality", None):
            headline = quality.quality.get("headline")
            if headline:
                emit("profile", "info", "What we noticed about your data", headline)
        if not any(e["phase"] == "profile" for e in emit.log):
            emit("profile", "info", "Checking your data for problems",
                 "We looked for missing values, duplicated rows and columns that never change, "
                 "so the models are not distracted by them.")
        emit.phase_progress("profile", 1.0)

        # ---- 3. prepare the target ---------------------------------------- #
        emit.phase_progress("prepare", 0.05)
        emit("prepare", "info", "Working out exactly what we are looking for",
             f"“{uc.business_question}”", progress=0.09)
        target = defs.build_target(dataset_id, uc, mapping or {}, options)

        notes = list(target.notes)
        for i, note in enumerate(notes):
            emit.phase_progress("prepare", 0.1 + 0.9 * (i + 1) / max(1, len(notes)))
            emit("prepare", "info", "Preparing your data", note)
        narration.append(nar.data_beat(notes))

        result.target = {
            "task": target.task,
            "target_col": target.target_col,
            "label_names": _json_safe(target.label_names),
            "positive_rate": _finite(target.positive_rate),
            "notes": notes,
            "rows_after_preparation": int(target.frame.height),
            "features": len(target.numeric_cols) + len(target.categorical_cols),
        }
        if target.positive_rate is not None:
            emit("prepare", "info",
                 f"{target.positive_rate * 100:.1f}% of rows are the thing we are looking for",
                 f"{int(round(target.positive_rate * target.frame.height)):,} rows carry the "
                 f"outcome. Because it is the smaller group, the models are judged on how well "
                 f"they pick those rows out — not on overall accuracy, which would flatter a "
                 f"model that simply says 'no' every time.")

        # A label column holding a single value cannot be learned from. Running the race
        # anyway would produce a page of failed models and a meaningless champion, so we
        # stop here and say plainly why — that is more useful than a broken scoreboard.
        if target.task == "classification" and target.target_col is not None:
            labels = target.frame.get_column(target.target_col).to_numpy().astype(float)
            labels = labels[np.isfinite(labels)]
            if labels.size and int(np.unique(labels).size) < 2:
                only_value = labels[0]
                only_text = (target.label_names or {}).get(int(only_value), int(only_value))
                emit("error", "error",
                     "There is nothing to tell apart in this column",
                     f"Every row in “{target.target_col}” has the same value ({only_text}). A "
                     f"model can only learn the difference between two outcomes, so there is no "
                     f"pattern to find here. Choose an outcome column that varies — one where "
                     f"some rows are one thing and some are another — and run it again.",
                     progress=1.0)
                result.status = "failed"
                result.error = (f"“{target.target_col}” contains only one value "
                                f"({only_text}), so there is nothing to predict.")
                result.finished_at = datetime.now(timezone.utc).isoformat()
                result.duration_ms = round((time.perf_counter() - started_perf) * 1000.0, 1)
                result.narration = narration
                result.log = emit.log
                return result

        # ---- 4. plan ------------------------------------------------------ #
        budget = float(options.get("time_budget_s", config.DEFAULT_TIME_BUDGET_S) or
                       config.DEFAULT_TIME_BUDGET_S)
        plan = make_plan(uc, target, n_rows=target.frame.height, time_budget_s=budget,
                         options=options)
        for note in plan.notes:
            emit("plan", "info", "Planning the race", note)
        emit.phase_progress("plan", 1.0)
        emit("plan", "success", f"{len(plan.models)} models are ready to race",
             "Each one looks at your data in a different way. Comparing them is what tells us "
             "whether the answer is real or just an accident of one method.",
             progress=0.20, payload={"models": [m.display for m in plan.models]})
        narration.append(nar.plan_beat(plan))

        result.plan = {
            "test_strategy": plan.test_strategy,
            "test_size": plan.test_size,
            "sample_rows": plan.sample_rows,
            "sample_reason": plan.sample_reason,
            "budget_s": plan.budget_s,
            "model_count": len(plan.models),
            "models": [model_zoo.to_dict(m) for m in plan.models],
        }

        # ---- 5. design matrix --------------------------------------------- #
        num_cols, cat_cols = target.numeric_cols, target.categorical_cols
        feature_cols = target.feature_cols
        X_all = _design(target.frame, num_cols, cat_cols)
        if X_all is None or X_all.shape[1] == 0:
            raise ValueError(
                "There are no usable columns left to learn from. Add a few more columns to "
                "your file and run it again."
            )
        emit("prepare", "info",
             f"{X_all.shape[1]} usable evidence columns",
             f"We converted your categories into numbers the models can read (for example "
             f"“Online” becomes its own on/off column) and filled small gaps with the middle "
             f"value of that column.")

        y_all = None
        if target.target_col is not None and target.target_col in target.frame.columns:
            y_all = target.frame.get_column(target.target_col).to_numpy().astype(float)

        train_idx, test_idx = _split_indices(target, plan, y_all if y_all is not None
                                             else np.zeros(target.frame.height), seed)
        if plan.sample_rows and len(train_idx) > plan.sample_rows:
            if y_all is not None and target.task == "classification":
                train_idx = _stratified_sample(y_all[train_idx], plan.sample_rows, seed)
            else:
                rng = np.random.default_rng(seed)
                train_idx = np.sort(rng.choice(train_idx, plan.sample_rows, replace=False))
        X_train = X_all.iloc[train_idx]
        X_test = X_all.iloc[test_idx] if len(test_idx) else None
        y_train = y_all[train_idx] if y_all is not None else None
        y_test = y_all[test_idx] if (y_all is not None and len(test_idx)) else None

        # ---- clustering: how many groups? --------------------------------- #
        cluster_k = None
        cluster_scores: list[dict] = []
        if target.task == "clustering":
            cluster_k, cluster_scores = _choose_k(X_all, seed)
            if cluster_scores:
                emit("plan", "info", f"We will look for {cluster_k} groups",
                     "We tried every sensible number of groups and kept the one where the groups "
                     "came out tightest: "
                     + ", ".join(f"{r['k']} → {r['silhouette']:.2f}" for r in cluster_scores)
                     + ". The score runs from 0 (groups overlapping) to 1 (perfectly separated).",
                     progress=0.21)

        db_eps = _dbscan_eps(X_all, seed) if target.task == "clustering" else None

        # ---- 6. the race --------------------------------------------------- #
        entries: list[dict] = []
        anomaly_scores: dict[str, np.ndarray] = {}
        total_models = len(plan.models)
        deadline = started_perf + plan.budget_s
        stopped_early = False

        for i, spec in enumerate(plan.models, start=1):
            elapsed = time.perf_counter() - started_perf
            if elapsed > plan.budget_s:
                stopped_early = True
                remaining = total_models - i + 1
                emit("train", "warn", "Time is up for the remaining models",
                     f"We have used {elapsed:.0f} of the {plan.budget_s:.0f} seconds allowed. "
                     f"{remaining} model(s) did not get a turn; the winner will be chosen from "
                     f"the {len(entries)} that finished. Raise the time budget to let everyone "
                     f"run.", progress=0.20 + 0.52 * (i - 1) / max(1, total_models))
                narration.append(nar.budget_beat(elapsed, plan.budget_s, len(entries),
                                                 remaining))
                break

            intro = nar.model_intro(spec, i, total_models, target.task)
            emit("train", "info", intro["title"], intro["text"],
                 model=spec.key, progress=0.20 + 0.52 * (i - 1) / max(1, total_models),
                 payload={"persona": spec.persona, "family": spec.family, "cost": spec.cost,
                          "plain": spec.plain})

            entry = {
                "rank": 0, "key": spec.key, "display": spec.display,
                "persona": spec.persona, "family": spec.family, "cost": spec.cost,
                "status": "ok", "error": None, "metrics": {}, "primary": None, "score": 0.0,
                "train_ms": 0.0, "predict_ms": 0.0, "why": "",
            }
            t0 = time.perf_counter()
            # scikit-learn's internals can emit noisy numerical warnings that mean nothing
            # to a business user; the run log is the only place we want messages.
            _quiet = warnings.catch_warnings()
            _quiet.__enter__()
            warnings.simplefilter("ignore")
            try:
                est = spec.build(seed)
                if target.task == "clustering" and cluster_k:
                    try:
                        params = est.get_params()
                        if "n_clusters" in params:
                            est.set_params(n_clusters=int(cluster_k))
                        if "n_components" in params:
                            est.set_params(n_components=int(cluster_k))
                        if spec.key == "dbscan" and db_eps:
                            est.set_params(eps=float(db_eps))
                    except Exception:
                        pass

                if spec.family == "baseline":
                    pipe = est
                    pipe.fit(X_train, y_train)
                    train_ms = (time.perf_counter() - t0) * 1000.0
                    p0 = time.perf_counter()
                    pred_test = pipe.predict(X_test) if X_test is not None else None
                    predict_ms = (time.perf_counter() - p0) * 1000.0
                    metrics = (ev.regression_metrics(y_test, pred_test)
                               if y_test is not None and pred_test is not None else {})
                else:
                    pipe = model_zoo.make_pipeline(est, num_cols, cat_cols, spec.needs_scaling)
                    if target.task == "clustering":
                        labels_all = pipe.fit_predict(X_all)
                        train_ms = (time.perf_counter() - t0) * 1000.0
                        p0 = time.perf_counter()
                        Xs = _numeric_matrix(X_all)
                        if Xs is None:
                            Xs = np.zeros((len(X_all), 1))
                        metrics = ev.clustering_metrics(Xs, labels_all)
                        predict_ms = (time.perf_counter() - p0) * 1000.0
                        entry["_labels"] = labels_all
                    elif target.task == "anomaly":
                        pipe.fit(X_all)
                        train_ms = (time.perf_counter() - t0) * 1000.0
                        p0 = time.perf_counter()
                        scores = _predict_batches(
                            lambda a, b: _score_estimator(pipe, X_all.iloc[a:b], "anomaly"),
                            len(X_all))
                        predict_ms = (time.perf_counter() - p0) * 1000.0
                        anomaly_scores[spec.key] = scores
                        entry["_scores"] = scores
                        metrics = {}
                    else:
                        pipe.fit(X_train, y_train)
                        train_ms = (time.perf_counter() - t0) * 1000.0
                        p0 = time.perf_counter()
                        if X_test is not None:
                            if target.task == "classification":
                                s_test = _score_estimator(pipe, X_test, "classification")
                                metrics = ev.classification_metrics(y_test, s_test)
                            else:
                                pred_test = _score_estimator(pipe, X_test, "regression")
                                metrics = ev.regression_metrics(y_test, pred_test)
                        else:
                            metrics = {}
                        predict_ms = (time.perf_counter() - p0) * 1000.0
                    entry["_pipe"] = pipe

                entry["train_ms"] = round(train_ms, 1)
                entry["predict_ms"] = round(predict_ms, 1)
                entry["metrics"] = _json_safe(metrics) or {}

                if target.task == "anomaly":
                    entry["primary"] = None       # filled in once every model has voted
                else:
                    entry["primary"] = _finite(ev.primary_value(target.task, metrics,
                                                                plan.primary_metric))

                verdict = nar.model_verdict(
                    spec, metrics, entry["primary"],
                    next((e["primary"] for e in entries if e.get("primary") is not None), None),
                    target.task, plan.primary_metric)
                entry["why"] = verdict["text"]
                emit("evaluate", "success", verdict["title"], verdict["text"],
                     model=spec.key,
                     progress=0.20 + 0.52 * i / max(1, total_models),
                     metrics={k: v for k, v in (entry["metrics"] or {}).items()
                              if isinstance(v, (int, float)) and v is not None})
            except Exception as exc:
                entry["status"] = "failed"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["train_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
                msg = nar.model_failed(spec, str(exc))
                entry["why"] = msg["text"]
                emit("train", "warn", msg["title"], msg["text"], model=spec.key,
                     progress=0.20 + 0.52 * i / max(1, total_models),
                     payload={"error": entry["error"]})
            finally:
                _quiet.__exit__(None, None, None)

            entries.append(entry)
            emit.phase_progress("train", i / max(1, total_models))

        ok_entries = [e for e in entries if e["status"] == "ok"]
        if not ok_entries:
            raise ValueError(
                "Every model failed on this data. The most common cause is a column that "
                "contains values the models cannot read. Try removing the columns you are "
                "least sure about and run it again. Technical detail: "
                + (entries[0].get("error") or "unknown"))

        # ---- anomaly: agreement with the crowd ----------------------------- #
        consensus = None
        if target.task == "anomaly" and anomaly_scores:
            stack = []
            for e in ok_entries:
                s = anomaly_scores.get(e["key"])
                if s is None:
                    continue
                stack.append(ev._rank(np.nan_to_num(s, nan=0.0)))
            if stack:
                consensus = np.mean(np.vstack(stack), axis=0)
                for e in ok_entries:
                    s = anomaly_scores.get(e["key"])
                    if s is None:
                        continue
                    m = ev.anomaly_metrics(s, consensus)
                    e["metrics"] = _json_safe(m)
                    e["primary"] = _finite(m.get("agreement"))
                emit("evaluate", "info", "Comparing the models with each other",
                     "There is no answer key for 'unusual', so we measured how much each model "
                     "agrees with what the whole field flags. A model that agrees strongly with "
                     "everyone else is picking up a real pattern rather than its own quirk.",
                     progress=0.72)

        ev.normalise_scores(target.task, entries, plan.primary_metric)
        ordered = ev.rank_entries(target.task, entries, plan.primary_metric)
        champion_entry = ev.pick_champion(ordered)
        if champion_entry is None:
            raise ValueError("We could not choose a winner from the models that finished.")
        champion_entry["primary_metric"] = plan.primary_metric
        champion_entry["primary_metric_label"] = uc.primary_metric_label

        # tidy internal arrays out of the serialisable leaderboard
        public_board = []
        for e in ordered:
            public_board.append({k: _json_safe(v) for k, v in e.items() if not k.startswith("_")})
        result.leaderboard = public_board

        emit.phase_progress("select", 1.0)
        emit("select", "success",
             f"Winner: {champion_entry['display']}",
             nar.champion_story(champion_entry, public_board, target.task,
                                {}).get("why_won", ""),
             model=champion_entry["key"], progress=0.78)

        # ---- 7. score every row with the champion -------------------------- #
        emit.phase_progress("score", 0.1)
        n_all = len(X_all)
        emit("score", "info", f"Scoring all {n_all:,} rows",
             "The winner now goes back over every row in your file — including the rows we held "
             "back during testing — and gives each one a score.",
             model=champion_entry["key"])

        champ_pipe = champion_entry.get("_pipe")
        if target.task == "clustering":
            labels_all = champion_entry.get("_labels")
            if labels_all is None and champ_pipe is not None:
                labels_all = champ_pipe.predict(X_all)
            score_all = np.asarray(labels_all, dtype=float)
            actual_all = None
            y_true_all = None
        elif target.task == "anomaly":
            raw = champion_entry.get("_scores")
            if raw is None and champ_pipe is not None:
                raw = _predict_batches(
                    lambda a, b: _score_estimator(champ_pipe, X_all.iloc[a:b], "anomaly"), n_all)
            raw = np.asarray(raw, dtype=float)
            score_all = ev._rank(raw) / max(1, len(raw) - 1) * 100.0
            actual_all = None
            y_true_all = None
        else:
            if champ_pipe is None:
                raise ValueError("The winning model was not available for scoring.")
            if target.task == "classification":
                score_all = _predict_batches(
                    lambda a, b: _score_estimator(champ_pipe, X_all.iloc[a:b],
                                                  "classification"), n_all)
                y_true_all = y_all
                actual_all = None
            else:
                score_all = _predict_batches(
                    lambda a, b: _score_estimator(champ_pipe, X_all.iloc[a:b], "regression"),
                    n_all)
                y_true_all = None
                actual_all = y_all

        emit.phase_progress("score", 0.6)

        # ---- 8. metrics, impact, drivers ----------------------------------- #
        metrics = dict(champion_entry.get("metrics") or {})
        threshold = float(metrics.get("threshold", 0.5) or 0.5)
        topk = None
        if target.task == "classification" and y_test is not None and len(y_test):
            s_test = _score_estimator(champ_pipe, X_test, "classification")
            n_test = len(s_test)
            # 0.50 is an arbitrary line, and on a rare event it can flag almost nothing.
            # We instead agree a review budget with the user: flag the riskiest slice of
            # the file, which is what a person can actually work through.
            pos_rate = float(metrics.get("positive_rate") or 0.0)
            k = int(min(n_test, max(25, round(0.05 * n_test))))
            if pos_rate <= 0.15:
                thr = float(np.sort(s_test)[-k])
                how = (f"the riskiest {k:,} rows (about {k / n_test * 100:.0f}% of your file)")
            else:
                thr = ev.best_f1_threshold(y_test, s_test, default=threshold)
                if float(np.mean(s_test >= thr)) > 0.6:
                    thr = float(np.quantile(s_test, 0.4))
                how = (f"rows scoring {thr:.2f} or higher, the cut-off that best balances "
                       f"catching cases against false alarms")
            threshold = float(thr)
            metrics["threshold"] = threshold
            # Evaluation scored every model at a default cut-off. Now that we have agreed
            # the cut-off the user will actually work to, the headline numbers must be
            # recomputed at it — otherwise the report quotes one threshold alongside
            # precision/recall figures that belong to a different one.
            test_cm = ev.confusion_at_threshold(y_test, s_test, threshold)
            for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "accuracy"):
                metrics[key] = _json_safe(test_cm.get(key))
            topk = ev.topk_capture(y_test, s_test, k)
            # The honest headline numbers stay the held-back test ones. The full-file
            # view lives entirely under `all_rows` / the `*_all_rows` scalars — it must
            # never be blended into the top-level confusion matrix, or precision and
            # the counts would describe two different populations.
            full = ev.confusion_at_threshold(y_all, score_all, threshold)
            metrics["all_rows"] = {key: _json_safe(full.get(key)) for key in
                                   ("tp", "fp", "tn", "fn", "flagged", "actual_positives",
                                    "precision", "recall", "accuracy")}
            metrics["flagged"] = full["flagged"]
            metrics["flagged_share"] = _finite(
                (full["flagged"] / n_all) if n_all else 0.0)
            metrics["actual_positives_all_rows"] = full["actual_positives"]
            metrics["precision_all_rows"] = _json_safe(full["precision"])
            metrics["recall_all_rows"] = _json_safe(full["recall"])
            metrics["topk"] = _json_safe(topk)
            metrics["review_budget"] = k
            metrics["roc_auc_all_rows"] = _finite(
                ev.classification_metrics(y_all, score_all).get("roc_auc"))
            emit("score", "success",
                 f"All {n_all:,} rows scored",
                 f"We flag {how}: {full['flagged']:,} rows in your file, carrying "
                 f"{full['tp']:,} of the {full['actual_positives']:,} real cases. The cut-off "
                 f"was chosen on the held-back test rows, never on the whole file.",
                 model=champion_entry["key"], progress=0.80)
        elif target.task == "regression":
            all_rows = ev.regression_metrics(actual_all, score_all)
            metrics["r2_all_rows"] = _finite(all_rows.get("r2"))
            metrics["mae_all_rows"] = _finite(all_rows.get("mae"))
            emit("score", "success", f"All {n_all:,} rows scored",
                 f"On an average row the prediction is about "
                 f"{metrics.get('mae', 0) or 0:,.2f} away from what actually happened.",
                 model=champion_entry["key"], progress=0.80)
        else:
            emit("score", "success", f"All {n_all:,} rows scored",
                 "Every row now carries a score from the winning model.",
                 model=champion_entry["key"], progress=0.80)

        champion_entry["metrics"] = _json_safe(metrics)
        champion_entry["primary"] = _finite(
            ev.primary_value(target.task, metrics, plan.primary_metric))
        for e in public_board:
            if e.get("key") == champion_entry.get("key"):
                e["metrics"] = _json_safe(metrics)
                e["primary"] = champion_entry["primary"]
                break

        # impact values for money maths
        impact_values = None
        if target.impact_col and target.impact_col in target.frame.columns:
            try:
                impact_values = target.frame.get_column(target.impact_col).to_numpy().astype(float)
            except Exception:
                impact_values = None

        flagged_mask = None
        if target.task == "classification":
            flagged_mask = (score_all >= threshold)
        elif target.task == "anomaly":
            flagged_mask = score_all >= 95.0

        impact = ev.business_impact(
            task=target.task, impact_label=uc.impact_label, impact_unit=uc.impact_unit,
            y_true=y_true_all, y_pred=(flagged_mask.astype(int) if flagged_mask is not None
                                       else None),
            impact_values=impact_values, flagged_mask=flagged_mask, topk=topk,
            metrics=metrics, n_rows=n_all)
        result.impact = _json_safe(impact)
        if impact.get("note"):
            emit("explain", "champion", impact["label"], impact["note"],
                 progress=0.84)

        # drivers
        drivers: list[dict] = []
        if champ_pipe is not None and target.task in ("classification", "regression"):
            try:
                n_drive = min(len(X_all), 20_000)
                rng = np.random.default_rng(seed)
                idx = (np.arange(len(X_all)) if len(X_all) <= n_drive
                       else rng.choice(len(X_all), n_drive, replace=False))
                Xd = X_all.iloc[idx]
                yd = y_all[idx] if y_all is not None else np.zeros(len(idx))
                drivers = ev.compute_drivers(champ_pipe, Xd, yd, num_cols, cat_cols,
                                             target.task, top=12, seed=seed)
            except Exception:
                drivers = []
        elif target.task in ("anomaly", "clustering"):
            # no target: rank columns by how much they vary relative to the crowd
            drivers = []
            try:
                for c in num_cols[:12]:
                    v = target.frame.get_column(c).to_numpy().astype(float)
                    v = v[np.isfinite(v)]
                    if len(v) < 10:
                        continue
                    mad = float(np.median(np.abs(v - np.median(v)))) * 1.4826
                    drivers.append({
                        "feature": c,
                        "importance": round(float(np.std(v) / (abs(np.mean(v)) + 1e-9)) * 100, 2),
                        "raw_importance": float(np.std(v)),
                        "direction": "neutral", "kind": "number",
                        "source": "how much this column varies across your records",
                    })
                total = sum(d["raw_importance"] for d in drivers) or 1.0
                for d in drivers:
                    d["importance"] = round(d["raw_importance"] / total * 100, 2)
                drivers.sort(key=lambda d: -d["importance"])
                drivers = drivers[:12]
            except Exception:
                drivers = []

        for d in drivers:
            d["plain"] = nar.driver_sentence(d)
        result.drivers = _json_safe(drivers)
        # emitted now so the live console stays in sync, but appended to the story
        # after the champion beat so the written story reads in the right order
        pending_driver_beats = nar.driver_beats(drivers, limit=3)
        for b in pending_driver_beats:
            emit("explain", "info", b["title"], b["text"], progress=0.88)

        # segments + distributions
        segments: list[dict] = []
        if target.task == "clustering":
            segments = _cluster_segments(champion_entry.get("_labels"), target.frame,
                                         num_cols, cat_cols, target.key_cols)
            for s in segments[:4]:
                emit("explain", "info", f"{s['label']} — {s['size']:,} rows", s["plain"],
                     progress=0.90)
        elif target.task == "anomaly":
            segments = _anomaly_segments(score_all, target.frame, num_cols, target.key_cols)
        result.segments = _json_safe(segments)

        timeseries = None
        if target.task == "regression" and target.extra.get("time_col") and actual_all is not None:
            tcol = target.extra["time_col"]
            try:
                tvals = target.frame.get_column(tcol).to_numpy()
                order = np.argsort(tvals, kind="stable")
                cap = 400
                step = max(1, len(order) // cap)
                sel = order[::step]
                timeseries = {
                    "x": [str(np.datetime_as_string(tvals[i], unit="D"))
                          if isinstance(tvals[i], np.datetime64)
                          else str(tvals[i])[:10] for i in sel],
                    "actual": [round(float(actual_all[i]), 4) for i in sel],
                    "predicted": [round(float(score_all[i]), 4) for i in sel],
                }
            except Exception:
                timeseries = None

        result.distributions = _json_safe(_distributions(
            task=target.task, target=target, champion=champion_entry,
            leaderboard=public_board, primary_metric=plan.primary_metric,
            score_all=score_all, y_true_all=y_true_all, actual_all=actual_all,
            drivers=drivers, segments=segments, timeseries=timeseries,
            impact_values=impact_values))

        # price elasticity curve
        elasticity = None
        if uc.target == "price_elasticity" and champ_pipe is not None:
            champ_r2 = metrics.get("r2")
            if champ_r2 is not None and champ_r2 > 0.02:
                elasticity = _elasticity_curve(target, champ_pipe, num_cols, cat_cols, X_all,
                                               seed)
            else:
                emit("explain", "warn", "Price does not explain demand in this data",
                     f"The best model could only explain {max(0.0, champ_r2 or 0.0) * 100:.0f}% "
                     f"of the ups and downs in “{target.target_col}”. That means price and "
                     f"demand are not connected in a consistent way here, so we will not "
                     f"recommend a price from it — the honest answer is that this file cannot "
                     f"answer the question. More price variation in the data would help.",
                     progress=0.92)
                narration.append(nar.beat(
                    "elasticity", "No reliable price response found",
                    "Price does not explain demand consistently in this file, so we are not "
                    "recommending a price. Collect more variation in prices and run it again."))
                result.impact = _json_safe({
                    "label": "No reliable price recommendation",
                    "value": 0.0,
                    "unit": uc.impact_unit,
                    "formula": "not applicable — the model found no dependable link between "
                               "price and demand",
                    "note": (f"We will not put a money figure on a price change here. The best "
                             f"model explained only {max(0.0, champ_r2 or 0.0) * 100:.0f}% of "
                             f"the movement in “{target.target_col}”, so any 'best price' we "
                             f"quoted would be noise dressed up as advice. If you can supply "
                             f"a period where prices actually varied — a promotion, a price "
                             f"test — the answer would be real."),
                })
                emit("explain", "champion", "No reliable price recommendation",
                     result.impact["note"], progress=0.93)
            if elasticity:
                result.distributions["elasticity"] = _json_safe(elasticity["chart"])
                emit("explain", "info", "The best price we can see in your history",
                     elasticity["sentence"], progress=0.92)
                narration.append(nar.beat("elasticity", "The best price we can see",
                                          elasticity["sentence"]))
                # the money headline for this question is the upside, not the average miss
                best = elasticity["chart"]["best"]
                cur = elasticity["chart"]["current"]
                cur_i = int(np.argmin(np.abs(np.array(elasticity["chart"]["x"]) - cur)))
                upside = float(best["revenue"] - elasticity["chart"]["revenue"][cur_i])
                value = upside * n_all
                result.impact = _json_safe({
                    "label": uc.impact_label,
                    "value": value,
                    "unit": uc.impact_unit,
                    "formula": ("average revenue per row at the best price in the sweep, minus "
                                "average revenue per row at your typical price, multiplied by "
                                "the number of rows"),
                    "note": (f"At your typical price the model expects about "
                             f"{elasticity['chart']['revenue'][cur_i]:,.2f} of revenue per row; "
                             f"at {best['x']:,.2f} it expects about {best['revenue']:,.2f}. "
                             f"Scaled to your {n_all:,} rows that is roughly "
                             f"{_money(abs(value))} of "
                             f"{'upside' if value > 0 else 'downside'} — a pattern from your "
                             f"past data, not a promise."),
                })
                emit("explain", "champion", uc.impact_label, result.impact["note"],
                     progress=0.93)

        impact = result.impact or impact

        # ---- 9. champion story, insights, narration ------------------------ #
        story = nar.champion_story(champion_entry, public_board, target.task, impact)
        result.champion = _json_safe({
            **{k: v for k, v in champion_entry.items() if not k.startswith("_")},
            "why_won": story["why_won"],
            "runner_up": story["runner_up"],
            "tradeoff": story["tradeoff"],
            "plain_summary": story["plain_summary"],
        })
        result.metrics = _json_safe(metrics)
        emit("select", "champion", story["title"], story["text"], progress=0.94)
        narration.append(nar.champion_beat(champion_entry, story))
        if impact.get("note"):
            narration.append(nar.impact_beat(impact))
        narration.extend(pending_driver_beats)

        insights = _build_insights(
            task=target.task, target=target, plan=plan, champion=champion_entry,
            metrics=metrics, impact=impact, drivers=drivers, topk=topk,
            cluster_scores=cluster_scores)
        result.insights = _json_safe(insights)
        for ins in insights[:4]:
            emit("explain", "info", ins["title"], ins["body"], progress=0.95)

        # ---- 10. write the scored file ------------------------------------- #
        # NOTE: for churn and segmentation the modelling table is one row per customer,
        # so the scored file has to come from the same frame the scores were made for.
        scored_frame, scored_cols = _build_scored_frame(
            target, target.frame, score_all, y_all, y_true_all, actual_all, threshold)
        path = _write_scored(run_id, scored_frame)
        result.scored_path = path
        result.scored_columns = scored_cols
        result.dataset["scored_rows"] = int(scored_frame.height)
        emit("export", "success", f"Scored file written",
             f"Every one of the {scored_frame.height:,} rows, with its score, is saved as "
             f"{'a Parquet file' if path.endswith('.parquet') else 'a CSV file'} ready to "
             f"download and open in Excel.", progress=0.98)

        # ---- 11. preview ---------------------------------------------------- #
        preview_cols = scored_cols[:12]
        head = scored_frame.select(preview_cols).head(50)
        result.preview = {
            "columns": [str(c) for c in preview_cols],
            "rows": [[_json_safe(v) for v in row] for row in head.rows()],
        }

        # ---- 12. why each of the worst rows was flagged ---------------------- #
        try:
            result.explanations = _json_safe(_explain_rows(
                champ_pipe, X_all, target, score_all, drivers, target.task, source=frame))
        except Exception:
            result.explanations = []
        if result.explanations:
            emit("explain", "info",
                 f"Why the top {len(result.explanations)} rows were picked",
                 "Open any row in the results below to see the handful of things about it "
                 "that pushed it up the list. Each one is a plain difference you can check "
                 "against the record yourself.",
                 progress=0.985)

        result.champion["primary_metric"] = plan.primary_metric
        result.champion["primary_metric_label"] = uc.primary_metric_label
        narration.append(nar.closing_beat(champion_entry["display"]))
        result.status = "succeeded"

        finished = time.perf_counter()
        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.duration_ms = round((finished - started_perf) * 1000.0, 1)
        emit("done", "success", "Analysis complete",
             f"Finished in {result.duration_ms / 1000:.1f} seconds using only this computer. "
             f"{champion_entry['display']} won, and {champion_entry.get('why') or ''}".strip(),
             progress=1.0)
        return result

    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.status = "failed"
        emit("error", "error", "We could not finish this analysis", str(exc))
        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.duration_ms = round((time.perf_counter() - started_perf) * 1000.0, 1)
        result.narration = narration
        result.log = emit.log
        return result


# --------------------------------------------------------------------------- #
# Scored output
# --------------------------------------------------------------------------- #


def _build_scored_frame(target, source: pl.DataFrame, score_all, y_all, y_true_all,
                        actual_all, threshold: float):
    cols: list[str] = []
    exprs = []
    for c in target.key_cols:
        if c in source.columns and c not in cols:
            cols.append(c)
            exprs.append(pl.col(c))

    if target.task == "regression":
        exprs.append(pl.Series("score", np.asarray(score_all, dtype=float)))
        exprs.append(pl.Series("predicted", np.asarray(score_all, dtype=float)))
        cols.extend(["score", "predicted"])
        if actual_all is not None:
            exprs.append(pl.Series("actual", np.asarray(actual_all, dtype=float)))
            exprs.append(pl.Series("difference",
                                   np.asarray(actual_all, dtype=float) -
                                   np.asarray(score_all, dtype=float)))
            cols.extend(["actual", "difference"])
    elif target.task == "classification":
        exprs.append(pl.Series("score", np.asarray(score_all, dtype=float)))
        exprs.append(pl.Series("probability", np.asarray(score_all, dtype=float)))
        cols.extend(["score", "probability"])
        labels = target.label_names or {0: "No", 1: "Yes"}
        names = [str(labels.get(0, "No")), str(labels.get(1, "Yes"))]
        pred = np.where(np.asarray(score_all) >= threshold, 1, 0)
        exprs.append(pl.Series("predicted", [names[int(p)] for p in pred]))
        cols.append("predicted")
        if y_all is not None:
            actual_names = [names[int(v)] if np.isfinite(v) else "Unknown" for v in y_all]
            exprs.append(pl.Series("actual", actual_names))
            cols.append("actual")
    elif target.task == "anomaly":
        exprs.append(pl.Series("score", np.asarray(score_all, dtype=float)))
        exprs.append(pl.Series("is_unusual",
                               (np.asarray(score_all) >= 95.0).astype(int)))
        cols.extend(["score", "is_unusual"])
    else:
        exprs.append(pl.Series("score", np.asarray(score_all, dtype=float)))
        cols.append("score")

    if target.impact_col and target.impact_col in source.columns \
            and target.impact_col not in cols:
        exprs.append(pl.col(target.impact_col))
        cols.append(target.impact_col)

    # a little context so the file is readable on its own
    for c in target.numeric_cols[:2] + target.categorical_cols[:2]:
        if c in source.columns and c not in cols:
            exprs.append(pl.col(c))
            cols.append(c)

    out = source.select(exprs)
    order = ["score", "predicted", "probability", "actual", "difference", "is_unusual"]
    if out.height and target.task == "classification":
        try:
            out = out.sort("score", descending=True)
        except Exception:
            pass
    elif out.height and target.task == "anomaly":
        try:
            out = out.sort("score", descending=True)
        except Exception:
            pass
    ordered = [c for c in order if c in out.columns] + [c for c in out.columns
                                                        if c not in order]
    return out.select(ordered), ordered


def _write_scored(run_id: str, frame: pl.DataFrame) -> str:
    run_dir = config.RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    parquet = run_dir / "scored.parquet"
    try:
        frame.write_parquet(parquet, compression="zstd")
        return str(parquet)
    except Exception:
        csv = run_dir / "scored.csv"
        frame.write_csv(csv)
        return str(csv)


# --------------------------------------------------------------------------- #
# Price elasticity curve
# --------------------------------------------------------------------------- #


def _elasticity_curve(target, pipe, num_cols, cat_cols, X_all, seed) -> dict | None:
    """Sweep the price and see what demand and revenue would have looked like."""
    price_col = target.extra.get("price_col")
    if not price_col or price_col not in target.frame.columns:
        return None
    prices = target.frame.get_column(price_col).to_numpy().astype(float)
    prices = prices[np.isfinite(prices)]
    if len(prices) < 30:
        return None
    lo, hi = np.quantile(prices, [0.05, 0.95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    grid = np.linspace(lo, hi, 25)

    base = X_all.iloc[np.arange(0, len(X_all), max(1, len(X_all) // 4000))]
    xs, ys, revs = [], [], []
    for p in grid:
        Xp = base.copy()
        if price_col in Xp.columns:
            Xp[price_col] = float(p)
        pred = _score_estimator(pipe, Xp, "regression")
        demand = float(np.mean(np.clip(pred, 0, None)))
        xs.append(round(float(p), 4))
        ys.append(round(demand, 4))
        revs.append(round(demand * float(p), 4))

    if not xs:
        return None
    best_i = int(np.argmax(revs))
    current = float(np.mean(prices))
    is_discount = bool(target.extra.get("is_discount"))
    noun = "discount" if is_discount else "price"
    best_x = xs[best_i]
    move = "raise" if best_x > current else "lower"
    sentence = (
        f"Your history centres on a {noun} of about {current:,.2f}. The highest total revenue in "
        f"this sweep comes at about {best_x:,.2f} — which suggests you could {move} the {noun} "
        f"from where it typically sits. This is a pattern in past data, not a guarantee: it "
        f"assumes customers respond the way they did before."
    )
    return {
        "chart": {
            "x": xs, "demand": ys, "revenue": revs,
            "x_label": f"{noun.capitalize()}",
            "best": {"x": best_x, "revenue": revs[best_i]},
            "current": round(current, 4),
        },
        "sentence": sentence,
    }


__all__ = ["run", "RunResult"]
