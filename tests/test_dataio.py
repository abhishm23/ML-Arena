"""Tests for `mlarena.dataio.loader` and `mlarena.dataio.profiler`.

Run either way:

    "C:\\ML Arena\\.venv\\Scripts\\python.exe" "C:\\ML Arena\\tests\\test_dataio.py"
    "C:\\ML Arena\\.venv\\Scripts\\python.exe" -m pytest tests/test_dataio.py -q
"""

from __future__ import annotations

import csv
import atexit
import json
import shutil
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlarena import config  # noqa: E402,F401  -- must precede polars

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from mlarena.dataio import loader, profiler  # noqa: E402

# -----------------------------------------------------------------------------
# fixtures built once per process
# -----------------------------------------------------------------------------
MESSY_ROWS = 12_000
BIG_ROWS = 1_000_000

STATUSES = ["new", "paid", "shipped", "refunded", "cancelled"]
CUSTOMERS = [
    "Acme Ltd", "Borealis Trading", "Cinder & Co", "Delta Works", "Evergreen",
    "Fathom Group", "Granite Retail", "Harbour Supply", "Ionic Partners",
    "Juniper Foods", "Kestrel Logistics", "Lumen Systems",
]
NOTES = [
    "Delivered on the second attempt after a failed drop at the gate",
    "Customer asked for the invoice to be re-issued with a purchase order number",
    "Split shipment: two parcels left the warehouse a day apart",
    "Left with the neighbour as requested in the delivery instructions",
    "Address corrected by the courier before the first delivery attempt",
    "Fragile item, packed with extra protection at no charge",
]
MEMO = ["ok", "1234", "rush order", "N/A", "42", "check with finance", "n/a", "0"]

# header on purpose: whitespace, a duplicate name and a blank cell
MESSY_HEADER = (
    "  order_id ,Customer Name,order_date,status,list_price,net_price,"
    "quantity,notes,order_id ,,memo,mixed_vals\n"
)

_TMPDIR: Path | None = None
_FILES: dict[str, Path] = {}
TIMINGS: dict[str, float] = {}


def _workdir() -> Path:
    global _TMPDIR
    if _TMPDIR is None:
        _TMPDIR = Path(tempfile.mkdtemp(prefix="mlarena_test_"))
        atexit.register(shutil.rmtree, _TMPDIR, ignore_errors=True)
    return _TMPDIR


def _build_messy_csv(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(MESSY_HEADER)
        writer = csv.writer(fh)
        for i in range(MESSY_ROWS):
            order_id = 100_000 + i
            list_price = round(50.0 + (i % 300) * 1.5, 2)
            if i % 25 == 0:                      # a money column with negatives
                net_price = round(-list_price * 0.1, 2)
            else:
                net_price = round(list_price * (0.6 + (i % 7) * 0.05), 2)
            writer.writerow(
                [
                    order_id,
                    CUSTOMERS[i % len(CUSTOMERS)],
                    "" if i % 97 == 0 else (date(2024, 1, 1) + timedelta(days=i % 400)).isoformat(),
                    STATUSES[i % len(STATUSES)],
                    list_price,
                    net_price,
                    "" if i % 53 == 0 else 1 + (i % 10),
                    f"{NOTES[i % len(NOTES)]} (ref {i})",
                    order_id,                        # duplicate header name
                    "x" if i % 3 else "",
                    MEMO[i % len(MEMO)],
                    i if i < 10_000 else "unknown",  # mixed types further down
                ]
            )


def _small_frame() -> pl.DataFrame:
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(400)]
    return pl.DataFrame(
        {
            "order_id": list(range(1, 401)),
            "order_date": days,
            "status": [STATUSES[i % 5] for i in range(400)],
            "amount": [round(10.0 + i * 1.25, 2) for i in range(400)],
            "quantity": [1 + (i % 9) for i in range(400)],
            "region": [CUSTOMERS[i % 8] for i in range(400)],
            "discount_pct": [round(0.01 * (i % 40), 4) for i in range(400)],
        }
    )


def _build_big_csv(path: Path) -> None:
    rng = np.random.default_rng(7)
    frame = pl.DataFrame(
        {
            "order_id": np.arange(BIG_ROWS, dtype=np.int64),
            "order_date": (
                (pl.Series(np.arange(BIG_ROWS, dtype=np.int32)) % 2000) + 19723
            ).cast(pl.Date),
            "status": np.array(STATUSES)[rng.integers(0, len(STATUSES), BIG_ROWS)],
            "region": np.array(CUSTOMERS)[rng.integers(0, len(CUSTOMERS), BIG_ROWS)],
            "amount": np.round(rng.normal(250.0, 80.0, BIG_ROWS), 2),
            "quantity": rng.integers(1, 12, BIG_ROWS).astype(np.int64),
            "discount_pct": np.round(rng.uniform(0.0, 0.4, BIG_ROWS), 4),
            "margin_pct": np.round(rng.normal(0.25, 0.08, BIG_ROWS), 4),
        }
    )
    frame.write_csv(path)


def _ensure_files() -> dict[str, Path]:
    if _FILES:
        return _FILES
    base = _workdir()
    messy = base / "messy_orders.csv"
    _build_messy_csv(messy)
    small = _small_frame()
    xlsx = base / "orders.xlsx"
    parquet = base / "orders.parquet"
    small.write_parquet(parquet)
    small.to_pandas().to_excel(xlsx, index=False, engine="openpyxl")
    big = base / "big_orders.csv"
    _build_big_csv(big)
    _FILES.update({"messy": messy, "xlsx": xlsx, "parquet": parquet, "big": big})
    return _FILES


def _fresh_store() -> None:
    loader.store.clear()


# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------
def test_messy_csv_loads() -> None:
    files = _ensure_files()
    _fresh_store()

    started = time.perf_counter()
    result = loader.load_dataset(str(files["messy"]), "messy")
    elapsed = time.perf_counter() - started
    TIMINGS["messy_load_s"] = elapsed

    print(
        f"[messy csv] {result.rows:,} rows x {result.cols} cols  "
        f"{result.size_bytes / 1e6:.1f} MB  engine={result.engine}  "
        f"load={elapsed:.3f}s"
    )
    for note in result.notes:
        print(f"   note: {note}")

    assert result.rows == MESSY_ROWS, result.rows
    assert result.cols == 12, result.cols
    assert result.engine in ("polars", "pandas")
    assert result.size_bytes > 0

    columns = set(result.columns)
    # whitespace trimmed, blank header named, duplicate renamed
    assert "order_id" in columns
    assert "order_id_1" in columns
    assert "column" in columns
    assert "Customer Name" in columns
    assert all(c == c.strip() for c in result.columns)
    assert len(result.columns) == len(set(result.columns))

    frame = loader.store.get("messy")
    assert frame.height == MESSY_ROWS
    # the mixed-type column was detected, not silently blanked
    assert frame.schema["mixed_vals"] == pl.String
    assert any("every row was checked" in n for n in result.notes), result.notes

    # json-safe metadata
    json.dumps(result.to_dict())


def test_xlsx_and_parquet_load() -> None:
    files = _ensure_files()
    _fresh_store()

    for fmt, key in (("xlsx", "xlsx"), ("parquet", "parquet")):
        result = loader.load_dataset(str(files[key]), f"small_{fmt}")
        print(
            f"[{fmt}] {result.rows} rows x {result.cols} cols  "
            f"engine={result.engine}  load={result.load_ms / 1000:.3f}s"
        )
        assert result.rows == 400, (fmt, result.rows)
        assert result.cols == 7, (fmt, result.cols)
        assert result.fmt == fmt
        assert result.dataset_id == f"small_{fmt}"
        json.dumps(result.to_dict())

    assert loader.store.meta("small_xlsx").engine == "pandas"
    assert loader.store.meta("small_parquet").engine == "polars"


def test_profiler_classification() -> None:
    files = _ensure_files()
    _fresh_store()
    loader.load_dataset(str(files["messy"]), "messy_prof")

    started = time.perf_counter()
    profile = profiler.profile("messy_prof")
    elapsed = time.perf_counter() - started
    TIMINGS["messy_profile_s"] = elapsed

    kinds = {c.name: c.kind for c in profile.columns}
    semantics = {c.name: c.semantic for c in profile.columns}
    print(f"[messy profile] {elapsed:.3f}s")
    for column in profile.columns:
        print(
            f"   {column.name:14s} -> {column.kind:11s} {column.semantic:10s} "
            f"distinct={column.distinct:<6d} conf={column.confidence}"
        )
    print(f"   headline: {profile.quality['headline']}")

    assert kinds["order_id"] == "identifier"
    assert kinds["order_id_1"] == "identifier"
    assert kinds["status"] == "dimension"
    assert semantics["status"] == "category"
    assert kinds["order_date"] == "datetime"
    assert kinds["list_price"] == "measure"
    assert semantics["list_price"] == "currency"
    assert kinds["net_price"] == "measure"
    assert semantics["net_price"] == "currency"
    assert kinds["Customer Name"] == "dimension"
    assert kinds["notes"] == "text"
    assert kinds["quantity"] == "measure"

    assert "list_price" in profile.measures
    assert "status" in profile.dimensions
    assert "order_date" in profile.datetimes
    assert "order_id" in profile.identifiers

    net_price = next(c for c in profile.columns if c.name == "net_price")
    assert net_price.stats["negatives_pct"] > 0, net_price.stats
    assert net_price.stats["min"] < 0

    status = next(c for c in profile.columns if c.name == "status")
    assert len(status.stats["top"]) == len(STATUSES)
    assert status.stats["coverage"] > 0.99

    order_date = next(c for c in profile.columns if c.name == "order_date")
    assert order_date.stats["span_days"] > 0
    assert order_date.stats["min"] < order_date.stats["max"]

    assert isinstance(profile.quality["score"], int)
    assert 0 <= profile.quality["score"] <= 100
    assert profile.quality["headline"]
    assert isinstance(profile.suggestions, list)

    assert profile.rows == MESSY_ROWS
    assert profile.cols == 12


def test_profile_to_dict_is_json_safe() -> None:
    files = _ensure_files()
    _fresh_store()
    loader.load_dataset(str(files["messy"]), "messy_json")
    profile = profiler.profile("messy_json")
    payload = profiler.profile_to_dict(profile)
    text = json.dumps(payload, allow_nan=False)
    assert len(text) > 100
    assert payload["dataset_id"] == "messy_json"
    assert payload["rows"] == MESSY_ROWS
    assert isinstance(payload["columns"], list)
    assert payload["quality"]["headline"]


def test_big_file_speed() -> None:
    files = _ensure_files()
    _fresh_store()

    started = time.perf_counter()
    result = loader.load_dataset(str(files["big"]), "big")
    load_s = time.perf_counter() - started

    started = time.perf_counter()
    profile = profiler.profile("big")
    sample_profile_s = time.perf_counter() - started

    started = time.perf_counter()
    full_profile = profiler.profile("big", sample_rows=0)
    full_profile_s = time.perf_counter() - started

    TIMINGS.update(
        {
            "big_load_s": load_s,
            "big_profile_sampled_s": sample_profile_s,
            "big_profile_full_s": full_profile_s,
            "big_size_mb": result.size_bytes / 1e6,
        }
    )
    print(
        f"[1M csv] {result.rows:,} rows x {result.cols} cols  "
        f"{result.size_bytes / 1e6:.1f} MB  engine={result.engine}  "
        f"load={load_s:.2f}s  profile(sampled)={sample_profile_s:.2f}s  "
        f"profile(full 1M)={full_profile_s:.2f}s"
    )
    print(f"   headline: {profile.quality['headline']}")

    assert result.rows == BIG_ROWS
    assert result.cols == 8
    assert load_s < 30.0, f"1M-row CSV load took {load_s:.1f}s"
    assert sample_profile_s < 15.0, f"sampled profile took {sample_profile_s:.1f}s"
    assert full_profile_s < 15.0, f"full 1M profile took {full_profile_s:.1f}s"
    assert full_profile.quality["sampled"] is False
    assert profile.quality["sampled"] is True
    assert profile.quality["total_rows"] == BIG_ROWS

    kinds = {c.name: c.kind for c in full_profile.columns}
    assert kinds["amount"] == "measure"
    assert kinds["order_id"] == "identifier"
    assert kinds["order_date"] == "datetime"
    assert kinds["status"] == "dimension"
    assert kinds["region"] == "dimension"
    assert kinds["discount_pct"] == "measure"


def test_text_stored_dates_are_promoted() -> None:
    base = _workdir()
    # polars can sniff these itself — no help needed
    clean = base / "uk_dates.csv"
    clean.write_text(
        "Order Date,created_at,amount\n"
        "05/01/2023,2023-01-05 08:30:00,10\n"
        "06/02/2023,2023-01-06 09:31:00,20\n"
        "07/03/2023,2023-01-07 10:32:00,30\n"
        "08/04/2023,2023-01-08 11:33:00,40\n",
        encoding="utf-8",
    )
    _fresh_store()
    result = loader.load_dataset(str(clean), "uk_dates")
    assert result.dtypes["Order Date"] == "Date"
    assert result.dtypes["created_at"] == "Datetime"

    # one junk value makes polars give up on the whole column — we rescue it
    dirty = base / "dirty_dates.csv"
    dirty.write_text(
        "order_date,amount\n"
        "2023-01-05,10\n"
        "not a date,20\n"
        "2023-03-07,30\n"
        "2023-04-01,40\n"
        "2023-05-01,50\n",
        encoding="utf-8",
    )
    _fresh_store()
    rescued = loader.load_dataset(str(dirty), "dirty_dates")
    assert rescued.dtypes["order_date"] == "Date", rescued.dtypes
    assert any("stored dates as text" in n for n in rescued.notes), rescued.notes
    assert any("left blank" in n for n in rescued.notes), rescued.notes

    profile = profiler.profile("dirty_dates")
    order_date = next(c for c in profile.columns if c.name == "order_date")
    assert order_date.kind == "datetime"
    assert order_date.stats["span_days"] > 0
    assert order_date.stats["min"] == "2023-01-05"
    assert profile.quality["text_dates"] == []


def test_time_of_day_is_not_a_date() -> None:
    base = _workdir()
    path = base / "times.csv"
    path.write_text(
        "time,v\n12:30,1\n13:30,2\n14:30,3\n15:30,4\n",
        encoding="utf-8",
    )
    _fresh_store()
    loader.load_dataset(str(path), "times")
    profile = profiler.profile("times")
    kinds = {c.name: c.kind for c in profile.columns}
    assert kinds["time"] == "dimension", kinds


def test_store_helpers() -> None:
    files = _ensure_files()
    _fresh_store()
    loader.load_dataset(str(files["messy"]), "store_a")

    assert loader.dataset_ids() == ["store_a"]
    assert loader.store.has("store_a")
    assert not loader.store.has("nope")

    try:
        loader.store.get("nope")
    except KeyError as exc:
        assert "nope" in str(exc)
        assert "store_a" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("store.get() on an unknown id must raise KeyError")

    meta = loader.store.meta("store_a")
    assert meta.rows == MESSY_ROWS
    assert [m.dataset_id for m in loader.store.list()] == ["store_a"]

    pdf = loader.frame_pandas("store_a", columns=["order_id", "status"])
    assert list(pdf.columns) == ["order_id", "status"]
    assert len(pdf) == MESSY_ROWS

    # a colliding id is renamed, never silently overwritten
    again = loader.load_dataset(str(files["messy"]), "store_a")
    assert again.dataset_id == "store_a_2"
    assert any("already open" in n for n in again.notes)

    loader.store.drop("store_a_2")
    loader.store.drop("store_a_2")  # idempotent
    assert not loader.store.has("store_a_2")

    derived = loader.register_frame(
        "store_derived",
        loader.store.get("store_a").head(10),
        derived_from="store_a",
        notes=["Made a small table."],
    )
    assert derived.derived_from == "store_a"
    assert derived.rows == 10
    assert derived.fmt == "memory"
    json.dumps(derived.to_dict())


def test_tsv_and_json_load() -> None:
    base = _workdir()
    tsv = base / "orders.tsv"
    jsonl = base / "orders.ndjson"
    _small_frame().write_csv(tsv, separator="\t")
    _small_frame().write_ndjson(jsonl)
    _fresh_store()

    tsv_result = loader.load_dataset(str(tsv), "tsv_orders")
    assert tsv_result.rows == 400
    assert tsv_result.cols == 7
    assert tsv_result.fmt == "tsv"
    assert any("separated" in n for n in tsv_result.notes)

    json_result = loader.load_dataset(str(jsonl), "json_orders")
    assert json_result.rows == 400
    assert json_result.cols == 7
    assert json_result.fmt == "json"


def test_never_crashes_on_junk() -> None:
    base = _workdir()
    empty = base / "empty.csv"
    empty.write_text("", encoding="utf-8")
    header_only = base / "header_only.csv"
    header_only.write_text("a,b,c\n", encoding="utf-8")
    ragged = base / "ragged.csv"
    ragged.write_text("a,b\n1,2\n3,4,5,6\n7\n", encoding="utf-8")
    _fresh_store()

    empty_result = loader.load_dataset(str(empty), "empty")
    assert empty_result.rows == 0
    assert any("empty" in n.lower() for n in empty_result.notes)

    header_result = loader.load_dataset(str(header_only), "header_only")
    assert header_result.rows == 0
    assert header_result.cols == 3

    ragged_result = loader.load_dataset(str(ragged), "ragged")
    assert ragged_result.rows == 3
    assert ragged_result.cols == 2

    try:
        loader.load_dataset(str(base / "does_not_exist.csv"), "ghost")
    except FileNotFoundError as exc:
        assert "does_not_exist.csv" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a missing file must raise FileNotFoundError")


def test_profiler_sample_is_honest() -> None:
    files = _ensure_files()
    _fresh_store()
    loader.load_dataset(str(files["big"]), "sample_check")
    sampled = profiler.profile("sample_check", sample_rows=25_000)
    assert sampled.quality["sampled"] is True
    assert sampled.quality["sampled_rows"] == 25_000
    assert sampled.quality["total_rows"] == BIG_ROWS
    assert sampled.rows == BIG_ROWS
    assert "too many to inspect" in sampled.quality["headline"]


# -----------------------------------------------------------------------------
# runner
# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        test_messy_csv_loads,
        test_xlsx_and_parquet_load,
        test_tsv_and_json_load,
        test_never_crashes_on_junk,
        test_text_stored_dates_are_promoted,
        test_time_of_day_is_not_a_date,
        test_profiler_classification,
        test_profile_to_dict_is_json_safe,
        test_profiler_sample_is_honest,
        test_big_file_speed,
        test_store_helpers,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback

            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            print(f"ok   {test.__name__}")

    print("\n--- measured timings ---")
    for key, value in TIMINGS.items():
        print(f"{key:24s} {value:.3f}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
