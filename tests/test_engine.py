"""ML Arena — engine end-to-end test (contract §16).

Self-contained on purpose. The engine is owned by one agent but depends on the
data layer owned by another, so this test builds its own synthetic business data
in memory and registers it through the real loader when that module exists,
falling back to a tiny in-memory shim when it does not. Either way the test runs.

Run it directly:
    "C:\\ML Arena\\.venv\\Scripts\\python.exe" "C:\\ML Arena\\tests\\test_engine.py"

or with pytest:
    "C:\\ML Arena\\.venv\\Scripts\\python.exe" -m pytest tests/test_engine.py -q
"""

from __future__ import annotations

import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from mlarena import config  # noqa: E402  -- env bootstrap before polars
import polars as pl  # noqa: E402

from mlarena.engine import runner  # noqa: E402
from mlarena.usecases import registry  # noqa: E402

DATASET_ID = "test_orders"
ROWS = 6000
SEED = 7


# --------------------------------------------------------------------------- #
# A store that works whether or not agent C's loader is on disk yet
# --------------------------------------------------------------------------- #


def _install_store() -> str:
    """Return a label describing which store we are using."""
    try:
        from mlarena.dataio import loader as _real  # noqa: F401
        if hasattr(_real, "register_frame") and hasattr(_real, "store"):
            return "real"
    except Exception:
        pass

    shim = types.ModuleType("mlarena.dataio.loader")

    class _ShimStore:
        def __init__(self) -> None:
            self._frames: dict[str, pl.DataFrame] = {}
            self._meta: dict[str, object] = {}

        def get(self, dataset_id):
            if dataset_id not in self._frames:
                raise KeyError(f"No dataset called {dataset_id!r} is open.")
            return self._frames[dataset_id]

        def meta(self, dataset_id):
            return self._meta.get(dataset_id)

        def has(self, dataset_id):
            return dataset_id in self._frames

        def drop(self, dataset_id):
            self._frames.pop(dataset_id, None)

        def list(self):
            return list(self._meta.values())

        def to_pandas(self, dataset_id, columns=None):
            df = self.get(dataset_id)
            if columns:
                df = df.select(columns)
            return df.to_pandas()

        def _put(self, meta, frame):
            self._frames[meta.dataset_id] = frame
            self._meta[meta.dataset_id] = meta

    store = _ShimStore()

    def register_frame(dataset_id, df, *, path="", fmt="memory", derived_from=None,
                       notes=None):
        result = types.SimpleNamespace(
            dataset_id=dataset_id, path=path, fmt=fmt, rows=df.height, cols=df.width,
            size_bytes=0, load_ms=0.0, engine="polars",
            columns=list(df.columns), dtypes={c: str(d) for c, d in zip(df.columns, df.dtypes)},
            derived_from=derived_from, notes=list(notes or []),
        )
        store._put(result, df)
        return result

    shim.store = store
    shim.register_frame = register_frame
    shim.load_dataset = lambda *a, **k: (_ for _ in ()).throw(
        NotImplementedError("shim loader cannot read files"))
    shim.frame_pandas = lambda dataset_id, columns=None: store.to_pandas(dataset_id, columns)
    shim.dataset_ids = lambda: store.ids() if hasattr(store, "ids") else list(store._frames)

    sys.modules["mlarena.dataio.loader"] = shim
    try:
        import mlarena.dataio as _pkg
        _pkg.loader = shim
    except Exception:
        pass
    return "shim"


# --------------------------------------------------------------------------- #
# Synthetic business data
# --------------------------------------------------------------------------- #


def make_orders() -> pl.DataFrame:
    """A plausible orders file: some columns drive the money, a few rows are odd."""
    rng = np.random.default_rng(SEED)
    n = ROWS

    segment = rng.choice(["Retail", "Wholesale", "Government"], n, p=[0.55, 0.32, 0.13])
    channel = rng.choice(["Online", "Store", "Partner"], n, p=[0.5, 0.35, 0.15])
    region = rng.choice(["North", "South", "East", "West"], n)
    product = rng.choice([f"SKU-{i:03d}" for i in range(1, 26)], n)

    quantity = rng.integers(1, 80, n).astype(float)
    unit_price = np.clip(rng.normal(48, 9, n), 8, 120).round(2)
    revenue = (quantity * unit_price).round(2)

    base_discount = np.select(
        [segment == "Retail", segment == "Wholesale", segment == "Government"],
        [0.05, 0.19, 0.11],
    )
    discount = (base_discount + 0.0022 * quantity + rng.normal(0, 0.018, n)).clip(0, 0.55)
    leaky = rng.choice(n, 90, replace=False)
    discount[leaky] += rng.uniform(0.28, 0.45, len(leaky))
    discount = discount.clip(0, 0.9).round(4)

    margin = (0.42 - 0.55 * discount + rng.normal(0, 0.035, n)).clip(0.02, 0.75).round(4)
    cost = (revenue * (1.0 - margin)).round(2)

    days = rng.integers(0, 540, n)
    order_date = np.datetime64("2023-06-01") + days.astype("timedelta64[D]")

    # a handful of records that simply do not look like the rest
    odd = rng.choice(n, 45, replace=False)
    quantity[odd[:20]] = rng.uniform(700, 1500, 20)
    revenue[odd[:20]] = (quantity[odd[:20]] * unit_price[odd[:20]]).round(2)
    unit_price[odd[20:]] = rng.uniform(400, 900, len(odd) - 20)

    return pl.DataFrame({
        "order_id": [f"O{i:07d}" for i in range(n)],
        "order_date": pl.Series(order_date).cast(pl.Datetime("us")),
        "segment": segment,
        "channel": channel,
        "region": region,
        "product": product,
        "quantity": quantity,
        "unit_price": unit_price,
        "revenue": revenue,
        "cost": cost,
        "discount_pct": discount,
        "margin_pct": margin,
    })


# --------------------------------------------------------------------------- #
# The three runs
# --------------------------------------------------------------------------- #

CASES = [
    (
        "discount_leakage",
        {
            "discount": ["discount_pct"],
            "money": ["revenue"],
            "segment": ["segment"],
            "product": ["product"],
            "channel": ["channel"],
            "quantity": ["quantity"],
            "date": ["order_date"],
        },
    ),
    (
        "revenue_predict",
        {
            "target": ["revenue"],
            "drivers": ["quantity", "unit_price", "segment", "channel", "discount_pct"],
            "date": ["order_date"],
        },
    ),
    (
        "transaction_anomaly",
        {
            "measures": ["revenue", "quantity", "unit_price", "margin_pct"],
            "dimensions": ["segment", "channel"],
            "key": ["order_id"],
            "money": ["revenue"],
            "date": ["order_date"],
        },
    ),
]

REQUIRED_PHASES = {"load", "profile", "prepare", "plan", "train", "evaluate", "select",
                   "score", "explain", "export", "done"}

BANNED_IN_DETAILS = ("hyperparameter", "roc-auc", "one-hot encoding")


def collect_events():
    events: list[dict] = []

    def emit(event):
        events.append(event)

    return events, emit


def run_case(usecase_key: str, mapping: dict) -> dict:
    events, emit = collect_events()
    started = time.perf_counter()
    result = runner.run(DATASET_ID, usecase_key, mapping,
                        {"time_budget_s": 90, "seed": SEED}, emit)
    elapsed = time.perf_counter() - started
    return {"key": usecase_key, "result": result, "events": events, "seconds": elapsed}


def check_case(case: dict) -> None:
    key = case["key"]
    result = case["result"]
    events = case["events"]

    assert result.status == "succeeded", (
        f"{key}: run failed — {result.error}")

    # --- events ---------------------------------------------------------- #
    assert events, f"{key}: no events were emitted at all"
    phases = {e["phase"] for e in events}
    missing = REQUIRED_PHASES - phases
    assert not missing, f"{key}: no events for phase(s) {sorted(missing)}"
    for e in events:
        assert isinstance(e.get("seq"), int) and e["seq"] > 0, f"{key}: bad seq in {e}"
        assert e.get("title"), f"{key}: an event had no title"
        assert e.get("detail") is not None, f"{key}: an event had no detail"
        assert len(e["title"]) <= 80, f"{key}: title too long: {e['title']!r}"
        assert 0.0 <= float(e["progress"]) <= 1.0, f"{key}: bad progress {e['progress']}"
        low = (e.get("detail") or "").lower()
        for banned in BANNED_IN_DETAILS:
            assert banned not in low, (
                f"{key}: event detail used unglossed jargon {banned!r}: {e['detail']!r}")
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs), f"{key}: events are out of order"

    # --- leaderboard ------------------------------------------------------ #
    board = result.leaderboard
    assert board, f"{key}: the leaderboard is empty"
    assert len(board) >= 5, f"{key}: only {len(board)} models raced"
    ok_models = [e for e in board if e["status"] == "ok"]
    assert ok_models, f"{key}: every model failed"
    for entry in board:
        assert entry.get("why"), f"{key}: {entry['key']} has no plain-language 'why'"
        assert entry.get("display") and entry.get("persona"), f"{key}: entry lacks naming"
        assert entry.get("rank", 0) >= 1, f"{key}: entry has no rank"
        assert "score" in entry and 0.0 <= float(entry["score"]) <= 100.0, (
            f"{key}: bad UI score for {entry['key']}")
        if entry["status"] == "failed":
            assert entry.get("error"), f"{key}: failed entry without an error message"
    ranks = [e["rank"] for e in board]
    assert ranks == sorted(ranks), f"{key}: ranks are not in order"

    # --- champion --------------------------------------------------------- #
    champ = result.champion
    assert champ and champ.get("key"), f"{key}: no champion was chosen"
    assert champ.get("why_won"), f"{key}: champion has no 'why_won'"
    assert champ.get("plain_summary") is not None, f"{key}: champion has no summary"

    # --- impact ----------------------------------------------------------- #
    impact = result.impact
    assert impact, f"{key}: no impact block"
    assert impact.get("value") is not None, f"{key}: impact value is empty"
    assert impact.get("note"), f"{key}: impact has no plain-language note"

    # --- scored file ------------------------------------------------------ #
    assert result.scored_path, f"{key}: no scored_path"
    assert os.path.exists(result.scored_path), (
        f"{key}: scored file does not exist on disk: {result.scored_path}")
    assert os.path.getsize(result.scored_path) > 0, f"{key}: scored file is empty"
    assert "score" in result.scored_columns, f"{key}: scored file has no score column"

    # --- the rest of the payload ------------------------------------------ #
    assert result.drivers, f"{key}: no drivers were produced"
    for d in result.drivers:
        assert d.get("plain"), f"{key}: driver {d.get('feature')} has no plain sentence"
    assert 3 <= len(result.insights) <= 6, f"{key}: expected 3-6 insights, got {len(result.insights)}"
    for ins in result.insights:
        assert ins.get("severity") in ("info", "good", "warn", "bad"), f"{key}: bad severity"
        assert ins.get("title") and ins.get("body"), f"{key}: incomplete insight"
    assert result.narration, f"{key}: no narration beats"
    for b in result.narration:
        assert b.get("beat") and b.get("title") and b.get("text"), f"{key}: bad beat {b}"
    assert result.distributions.get("score_histogram"), f"{key}: no score histogram"
    assert result.preview.get("columns") and result.preview.get("rows"), f"{key}: no preview"
    assert len(result.preview["rows"]) <= 50, f"{key}: preview is too big"
    assert result.log, f"{key}: the log is empty"
    assert result.exports == {}, f"{key}: exports should be left for the API layer"
    assert result.duration_ms > 0 and result.started_at and result.finished_at


def report(case: dict) -> None:
    r = case["result"]
    champ = r.champion
    primary = champ.get("primary")
    metric = champ.get("primary_metric", "?")
    print(f"\n=== {case['key']} ===")
    print(f"  status        : {r.status}")
    print(f"  models raced  : {len(r.leaderboard)} "
          f"({sum(1 for e in r.leaderboard if e['status'] == 'ok')} finished, "
          f"{sum(1 for e in r.leaderboard if e['status'] == 'failed')} failed)")
    print(f"  champion      : {champ.get('display')} ({champ.get('persona')})")
    print(f"  {metric:<14}: {primary if primary is None else round(float(primary), 4)}")
    print(f"  impact        : {r.impact.get('label')} = "
          f"{r.impact.get('value')} {r.impact.get('unit')}")
    print(f"  impact note   : {r.impact.get('note')}")
    print(f"  rows scored   : {r.dataset.get('scored_rows')}")
    print(f"  scored file   : {r.scored_path}")
    print(f"  run time      : {case['seconds']:.1f}s "
          f"(engine reported {r.duration_ms / 1000:.1f}s)")


def main() -> int:
    store_kind = _install_store()
    print(f"ML Arena engine test — dataset store: {store_kind}")
    config.ensure_dirs()

    from mlarena.dataio import loader
    frame = make_orders()
    registered = loader.register_frame(DATASET_ID, frame)
    dataset_id = getattr(registered, "dataset_id", DATASET_ID)
    print(f"synthetic data: {frame.height:,} rows x {frame.width} columns "
          f"(registered as {dataset_id!r})")

    # a quick contract check on the catalogue before we start the heavy work
    assert len(registry.catalog()) == 9, "the catalogue must expose exactly 9 use cases"
    for uc in registry.catalog():
        assert uc["roles"], f"{uc['key']} has no roles"
        assert len(uc["tagline"].split()) <= 8, f"{uc['key']} tagline is too long"
        for role in uc["roles"]:
            assert set(role) >= {"key", "label", "help", "accepts", "required", "multi",
                                 "min_count"}, f"{uc['key']} role is missing keys"

    failures = 0
    total = 0.0
    for key, mapping in CASES:
        case = run_case(key, mapping)
        case["result"].usecase  # touch it so a broken serialisation shows up here
        if case["result"].status != "succeeded":
            print(f"\n!!! {key} FAILED: {case['result'].error}")
            failures += 1
            continue
        check_case(case)
        report(case)
        total += case["seconds"]

    print(f"\ntotal engine time for all three use cases: {total:.1f}s")
    if failures:
        print(f"{failures} use case(s) failed")
        return 1
    print("ALL ENGINE CHECKS PASSED")
    return 0


# --------------------------------------------------------------------------- #
# pytest entry points
# --------------------------------------------------------------------------- #


def test_catalogue_contract():
    assert len(registry.catalog()) == 9
    for uc in registry.catalog():
        assert len(uc["tagline"].split()) <= 8
        assert uc["primary_metric"] and uc["primary_metric_label"]
        assert uc["impact_unit"] in ("currency", "records", "customers", "none")
        assert len(uc["roles"]) >= 3


def test_engine_end_to_end():
    """The three headline use cases, end to end, through the real runner."""
    _install_store()
    config.ensure_dirs()
    from mlarena.dataio import loader

    registered = loader.register_frame(DATASET_ID, make_orders())
    dataset_id = getattr(registered, "dataset_id", DATASET_ID)

    for key, mapping in CASES:
        events, emit = collect_events()
        result = runner.run(dataset_id, key, mapping,
                            {"time_budget_s": 90, "seed": SEED}, emit)
        check_case({"key": key, "result": result, "events": events, "seconds": 0.0})


if __name__ == "__main__":
    raise SystemExit(main())
