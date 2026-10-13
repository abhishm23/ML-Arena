#!/usr/bin/env python
"""Tests for ``mlarena.export.exporters`` (contract section 13).

Run directly:

    "C:\\ML Arena\\.venv\\Scripts\\python.exe" "C:\\ML Arena\\tests\\test_export.py"

or under pytest::

    "C:\\ML Arena\\.venv\\Scripts\\python.exe" -m pytest tests/test_export.py -q

The test builds a realistic *fake* ``RunResult`` by hand (a discount-leakage
classification run with six models, one of which failed) and checks that all four
exporters produce usable artefacts.  It then repeats the whole thing with the
result wrapped in a dataclass to prove the ``getattr`` / ``dict.get`` duality
required by the contract.
"""

from __future__ import annotations

import dataclasses
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlarena import config  # noqa: E402  -- must precede any polars import

from mlarena.export import exporters  # noqa: E402

CHAMPION_NAME = "Gradient Boosting"
IMPACT_LABEL = "Estimated margin at risk"
RUN_ID = "run-20260918-1432-demo"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _write_scored_csv(path: Path, rows: int = 150) -> str:
    """A small but realistic scored-rows file, with a score column."""
    segments = ["Enterprise", "Mid-Market", "SMB", "Government"]
    regions = ["North", "South", "East", "West", "Central"]
    reps = ["Priya Nair", "Ananya Iyer", "Neha Gupta", "Arjun Rao", "Varun Kapoor"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "order_id", "order_date", "customer_segment", "region", "sales_rep",
            "discount_pct", "revenue", "expected_discount", "leakage_score", "flagged",
        ])
        for i in range(rows):
            flagged = 1 if i % 7 == 0 else 0
            score = 0.94 - (i % 7) * 0.07 - (i % 11) * 0.01
            writer.writerow([
                f"ORD-2025-{i + 1:06d}",
                f"2025-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                segments[i % len(segments)],
                regions[i % len(regions)],
                reps[i % len(reps)],
                round(0.10 + (i % 30) * 0.012, 6),
                round(180000 + (i % 97) * 41500.75, 2),
                round(0.14 + (i % 9) * 0.005, 6),
                round(max(0.0, min(1.0, score)), 6),
                flagged,
            ])
    return str(path)


def _fake_result(scored_path: str) -> dict:
    """A hand-built, classification-flavoured RunResult."""
    return {
        "run_id": RUN_ID,
        "status": "succeeded",
        "usecase": {
            "key": "discount_leakage",
            "name": "Discount leakage",
            "emoji": "\U0001f4b8",
            "tagline": "Find discounts that went too far",
            "story": (
                "We learn what a normal discount looks like for each order, then surface the "
                "orders that blew past it. That is money that quietly left the business."
            ),
            "business_question": "Which orders were discounted far more than their context justifies?",
            "task": "classification",
            "primary_metric": "roc_auc",
            "primary_metric_label": "Separation power",
            "impact_unit": "currency",
            "impact_label": IMPACT_LABEL,
            "sector": "Sales",
        },
        "dataset": {
            "dataset_id": "ds_orders_discount_leakage",
            "name": "orders_discount_leakage.csv",
            "rows": 60042,
            "cols": 20,
            "engine": "polars",
            "load_ms": 143.7,
            "scored_rows": 60042,
        },
        "target": {
            "task": "classification",
            "target_col": "leakage_flag",
            "label_names": {"0": "Normal", "1": "Leakage"},
            "positive_rate": 0.1043,
            "notes": [
                "We trained a model to predict the expected discount from context only.",
                "Orders more than two standard deviations above expectation were flagged.",
            ],
        },
        "plan": {
            "test_strategy": "stratified_split",
            "test_size": 0.2,
            "sample_rows": None,
            "sample_reason": "",
            "budget_s": 90.0,
            "model_count": 6,
        },
        "leaderboard": [
            {
                "rank": 1, "key": "hist_gradient_boosting", "display": CHAMPION_NAME,
                "persona": "The Step-by-Step Improver", "family": "boosting", "cost": "medium",
                "status": "ok", "error": None,
                "metrics": {"roc_auc": 0.9412, "accuracy": 0.9128, "precision": 0.7412,
                            "recall": 0.6884, "f1": 0.7139, "pr_auc": 0.7621, "log_loss": 0.2214},
                "primary": 0.9412, "score": 100.0, "train_ms": 1841.6, "predict_ms": 96.2,
                "why": "Learned the discount pattern fastest and separated the flagged orders most cleanly.",
            },
            {
                "rank": 2, "key": "random_forest", "display": "Random Forest",
                "persona": "The Committee of Experts", "family": "ensemble", "cost": "medium",
                "status": "ok", "error": None,
                "metrics": {"roc_auc": 0.9338, "accuracy": 0.9071, "precision": 0.7288,
                            "recall": 0.6742, "f1": 0.7005, "pr_auc": 0.7488, "log_loss": 0.2381},
                "primary": 0.9338, "score": 96.4, "train_ms": 2410.3, "predict_ms": 141.9,
                "why": "Nearly as accurate and very stable, but slower to train.",
            },
            {
                "rank": 3, "key": "xgboost", "display": "XGBoost",
                "persona": "The Relentless Optimiser", "family": "boosting", "cost": "heavy",
                "status": "ok", "error": None,
                "metrics": {"roc_auc": 0.9301, "accuracy": 0.9044, "precision": 0.7205,
                            "recall": 0.6689, "f1": 0.6937, "pr_auc": 0.7410, "log_loss": 0.2449},
                "primary": 0.9301, "score": 94.9, "train_ms": 3902.8, "predict_ms": 118.4,
                "why": "Strong result, but it took twice as long as the winner for no extra accuracy.",
            },
            {
                "rank": 4, "key": "logistic_regression", "display": "Logistic Regression",
                "persona": "The Straight-Line Thinker", "family": "linear", "cost": "fast",
                "status": "ok", "error": None,
                "metrics": {"roc_auc": 0.8817, "accuracy": 0.8702, "precision": 0.6114,
                            "recall": 0.5531, "f1": 0.5807, "pr_auc": 0.6022, "log_loss": 0.3120},
                "primary": 0.8817, "score": 72.6, "train_ms": 118.4, "predict_ms": 12.7,
                "why": "Blazing fast and easy to explain, but it misses the curved parts of the pattern.",
            },
            {
                "rank": 5, "key": "decision_tree", "display": "Decision Tree",
                "persona": "The Twenty Questions Player", "family": "tree", "cost": "fast",
                "status": "ok", "error": None,
                "metrics": {"roc_auc": 0.8604, "accuracy": 0.8533, "precision": 0.5902,
                            "recall": 0.5218, "f1": 0.5538, "pr_auc": 0.5671, "log_loss": 0.3402},
                "primary": 0.8604, "score": 63.1, "train_ms": 74.9, "predict_ms": 5.1,
                "why": "Simple enough to read off a page, but it over-commits to early splits.",
            },
            {
                "rank": 6, "key": "mlp_classifier", "display": "Neural Network (MLP)",
                "persona": "The Pattern Hunter", "family": "neural", "cost": "heavy",
                "status": "failed", "error": "ConvergenceWarning: solver stopped early after 200 iterations",
                "metrics": {}, "primary": None, "score": 0.0, "train_ms": 15020.4, "predict_ms": 0.0,
                "why": "This model did not finish cleanly, so it was left out of the race.",
            },
        ],
        "champion": {
            "rank": 1, "key": "hist_gradient_boosting", "display": CHAMPION_NAME,
            "persona": "The Step-by-Step Improver", "family": "boosting", "cost": "medium",
            "status": "ok", "error": None,
            "metrics": {"roc_auc": 0.9412, "accuracy": 0.9128, "precision": 0.7412,
                        "recall": 0.6884, "f1": 0.7139, "pr_auc": 0.7621, "log_loss": 0.2214},
            "primary": 0.9412, "score": 100.0, "train_ms": 1841.6, "predict_ms": 96.2,
            "why": "It separated over-discounted orders from normal ones more cleanly than anything else.",
            "why_won": (
                "It won because it built hundreds of small correction steps, so it could capture the "
                "curved relationship between quantity, segment and the discount that followed."
            ),
            "runner_up": "Random Forest came second, only 0.7 points behind, and would be the safer pick if you re-run on different data.",
            "tradeoff": "You give up some of the 'why' - a boosting model is harder to read than a single decision tree.",
            "plain_summary": (
                "Think of it as a team of small rules that keep correcting each other until the "
                "prediction is sharp."
            ),
        },
        "metrics": {
            "roc_auc": 0.9412, "accuracy": 0.9128, "precision": 0.7412, "recall": 0.6884,
            "f1": 0.7139, "pr_auc": 0.7621, "log_loss": 0.2214,
            "confusion_matrix": {"tn": 10422, "fp": 288, "fn": 401, "tp": 889},
        },
        "impact": {
            "label": IMPACT_LABEL,
            "value": 184_620_500.0,
            "unit": "currency",
            "formula": "Sum of revenue on orders flagged as over-discounted",
            "note": "If you review the top 500 flagged orders you would catch about 68% of the leakage.",
        },
        "drivers": [
            {"feature": "sales_rep", "importance": 0.2841, "direction": "higher_is_risk",
             "plain": "A handful of reps consistently discount well beyond what the deal justifies."},
            {"feature": "customer_segment", "importance": 0.1932, "direction": "higher_is_risk",
             "plain": "Enterprise deals carry the deepest discounts, so they absorb most of the leakage."},
            {"feature": "quantity", "importance": 0.1504, "direction": "higher_is_risk",
             "plain": "Bigger orders attract bigger discounts, which is expected - until it goes too far."},
            {"feature": "region", "importance": 0.1211, "direction": "higher_is_risk",
             "plain": "Two regions account for most of the excess, which points at a local approval habit."},
            {"feature": "product_category", "importance": 0.0988, "direction": "higher_is_risk",
             "plain": "Software and cloud deals are discounted hardest because the margin can take it."},
            {"feature": "channel", "importance": 0.0712, "direction": "higher_is_risk",
             "plain": "Partner and distributor deals give away more than direct ones."},
            {"feature": "order_month", "importance": 0.0410, "direction": "higher_is_risk",
             "plain": "Leakage spikes at quarter end, when deals are pushed over the line."},
        ],
        "insights": [
            {"title": "7 reps account for most of the leakage", "severity": "high", "icon": "\U0001f6a8",
             "body": "Seven sales reps carry roughly 62% of the excess discount. Reviewing their approval limits would address the bulk of the problem."},
            {"title": "Quarter-end is the riskiest week", "severity": "medium", "icon": "\U0001f4c5",
             "body": "Orders placed in the last five days of a quarter are flagged 2.3x more often than average."},
            {"title": "Hardware is rarely the problem", "severity": "low", "icon": "\u2705",
             "body": "Hardware margins are too thin to absorb deep discounts, so reps already hold the line there."},
        ],
        "segments": [],
        "distributions": {
            "score_histogram": {
                "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "counts": [41_820, 6_140, 2_980, 1_760, 1_120, 880, 1_410, 2_240, 3_120, 5_572],
            },
            "class_balance": {"labels": ["Normal", "Leakage"], "values": [53_780, 6_262]},
            "leaderboard": {
                "labels": [CHAMPION_NAME, "Random Forest", "XGBoost", "Logistic Regression",
                           "Decision Tree", "Neural Network (MLP)"],
                "values": [0.9412, 0.9338, 0.9301, 0.8817, 0.8604, 0.0],
            },
        },
        "preview": {
            "columns": ["order_id", "customer_segment", "discount_pct", "revenue", "leakage_score"],
            "rows": [
                ["ORD-2025-000001", "Enterprise", 0.41, 4_820_500.0, 0.97],
                ["ORD-2025-000002", "Mid-Market", 0.33, 1_240_000.0, 0.91],
            ],
        },
        "scored_path": scored_path,
        "scored_columns": [
            "order_id", "order_date", "customer_segment", "region", "sales_rep",
            "discount_pct", "revenue", "expected_discount", "leakage_score", "flagged",
        ],
        "exports": {},
        "narration": [
            {"beat": "load", "title": "We read your file",
             "text": "60,042 orders and 20 columns arrived in 0.14 seconds."},
            {"beat": "profile", "title": "We looked at what is inside",
             "text": "Three columns look like money, five look like categories, and a handful of rows are duplicated."},
            {"beat": "prepare", "title": "We built the question",
             "text": "We worked out what a normal discount looks like for every kind of order."},
            {"beat": "plan", "title": "We chose six models to race",
             "text": "Each one attacks the problem differently, so you can see which idea wins on your data."},
            {"beat": "select", "title": "We picked a winner",
             "text": "Gradient Boosting separated over-discounted orders from normal ones more cleanly than the rest."},
            {"beat": "explain", "title": "We found out why",
             "text": "Seven reps and two regions carry most of the excess discount."},
            {"beat": "done", "title": "Your answer is ready",
             "text": "Roughly \u20b918.46 Cr of margin is at risk, concentrated in a small number of hands."},
        ],
        "log": [
            {"seq": i, "phase": "train", "level": "info", "title": f"event {i}", "detail": "", "progress": i / 40}
            for i in range(40)
        ],
        "error": None,
        "started_at": "2026-09-18T14:32:07.114000",
        "finished_at": "2026-09-18T14:32:52.918000",
        "duration_ms": 45_804.0,
    }


@dataclasses.dataclass
class FakeRunResult:
    """A ``types.SimpleNamespace``-style dataclass mirroring ``RunResult``."""

    run_id: str
    status: str
    usecase: dict
    dataset: dict
    target: dict
    plan: dict
    leaderboard: list
    champion: dict
    metrics: dict
    impact: dict
    drivers: list
    insights: list
    segments: list
    distributions: dict
    preview: dict
    scored_path: str
    scored_columns: list
    exports: dict
    narration: list
    log: list
    error: str | None
    started_at: str
    finished_at: str
    duration_ms: float


def _as_dataclass(payload: dict) -> FakeRunResult:
    return FakeRunResult(**payload)


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------
def _assert_file(path: str, label: str, min_bytes: int = 1) -> Path:
    assert path, f"{label}: exporter returned an empty path"
    out = Path(path)
    assert out.is_absolute(), f"{label}: expected an absolute path, got {path!r}"
    assert out.exists(), f"{label}: file was not created at {path!r}"
    size = out.stat().st_size
    assert size >= min_bytes, f"{label}: file is only {size} bytes"
    return out


def _check_html(path: Path, label: str) -> int:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "<!doctype html>" in lowered, f"{label}: missing doctype"
    assert "<style>" in lowered, f"{label}: CSS is not inlined"
    assert "<script" not in lowered, f"{label}: report must not contain any script tag"
    assert "<link" not in lowered, f"{label}: report must not link to anything"
    assert "@import" not in lowered, f"{label}: report must not import external CSS"
    assert "http://" not in lowered, f"{label}: report references http://"
    assert "https://" not in lowered, f"{label}: report references https://"
    assert "cdn" not in lowered, f"{label}: report mentions a CDN"
    assert "<script src=" not in lowered, f"{label}: report loads an external script"

    assert CHAMPION_NAME in text, f"{label}: champion name {CHAMPION_NAME!r} missing"
    assert IMPACT_LABEL in text, f"{label}: impact label {IMPACT_LABEL!r} missing"
    assert "18.46 Cr" in text, f"{label}: impact value not rendered in a readable form"
    assert "Executive summary" in text, f"{label}: missing executive summary"
    assert "The business question" in text, f"{label}: missing business question"
    assert "Model leaderboard" in text.lower() or "model leaderboard" in lowered, f"{label}: missing leaderboard"
    assert "<svg" in lowered, f"{label}: no inline SVG charts were rendered"
    assert "xmlns" not in lowered, f"{label}: SVG should not need an xmlns namespace"
    assert "Random Forest" in text, f"{label}: leaderboard rows missing"
    assert "did not finish" in text, f"{label}: failed model not marked in the leaderboard"
    assert "ORD-2025-000001" in text, f"{label}: top scored rows missing"
    assert "Why we picked this model" in text, f"{label}: champion card missing"
    assert "We read your file" in text, f"{label}: story beats missing"
    assert "7 reps account for most of the leakage" in text, f"{label}: insights missing"
    return len(text)


def _check_csv(path: Path, label: str) -> None:
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), f"{label}: CSV does not start with a UTF-8 BOM"
    text = raw.decode("utf-8-sig")
    header = text.splitlines()[0]
    assert header.startswith("order_id"), f"{label}: unexpected header {header!r}"
    assert "leakage_score" in header, f"{label}: score column missing from CSV"
    assert len(text.splitlines()) == 151, f"{label}: expected 151 lines, got {len(text.splitlines())}"
    assert "Priya Nair" in text, f"{label}: scored rows look empty"


def _check_xlsx(path: Path, label: str) -> None:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=False)
    try:
        names = wb.sheetnames
        for required in ("Summary", "Model Leaderboard", "Key Drivers", "Insights", "Scored Data"):
            assert required in names, f"{label}: sheet {required!r} missing (found {names})"

        summary = wb["Summary"]
        summary_text = "\n".join(str(v) for row in summary.iter_rows(values_only=True) for v in row)
        assert "Champion model" in summary_text, f"{label}: Summary sheet lacks the champion row"
        assert CHAMPION_NAME in summary_text, f"{label}: Summary sheet lacks the champion name"

        leaderboard = wb["Model Leaderboard"]
        assert leaderboard.max_row == 7, f"{label}: expected 6 models + header, got {leaderboard.max_row}"
        assert leaderboard.freeze_panes == "A2", f"{label}: leaderboard header is not frozen"

        drivers = wb["Key Drivers"]
        assert drivers.max_row == 8, f"{label}: expected 7 drivers + header, got {drivers.max_row}"

        insights = wb["Insights"]
        assert insights.max_row >= 4, f"{label}: insights sheet looks empty"

        scored = wb["Scored Data"]
        assert scored.max_row == 151, f"{label}: expected 150 scored rows + header, got {scored.max_row}"
    finally:
        wb.close()


def _check_json(path: Path, label: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == RUN_ID, f"{label}: run_id round-trip failed"
    assert "log" not in payload, f"{label}: the raw log must not be dumped into the summary"
    assert payload["log_event_count"] == 40, f"{label}: log count missing/incorrect"
    assert payload["champion"]["display"] == CHAMPION_NAME, f"{label}: champion missing"
    assert payload["impact"]["value"] == 184_620_500.0, f"{label}: impact value changed"
    assert payload["leaderboard"][5]["status"] == "failed", f"{label}: failed model lost"


def _run_suite(payload: dict, label: str, tmp: Path) -> int:
    html_path = exporters.export_html(payload, str(tmp / f"report_{label}.html"))
    csv_path = exporters.export_csv(payload, str(tmp / f"scored_{label}.csv"))
    xlsx_path = exporters.export_xlsx(payload, str(tmp / f"workbook_{label}.xlsx"))
    json_path = exporters.export_summary_json(payload, str(tmp / f"summary_{label}.json"))

    html_file = _assert_file(html_path, f"{label}/html", min_bytes=5000)
    csv_file = _assert_file(csv_path, f"{label}/csv", min_bytes=1000)
    xlsx_file = _assert_file(xlsx_path, f"{label}/xlsx", min_bytes=3000)
    json_file = _assert_file(json_path, f"{label}/json", min_bytes=500)

    html_size = _check_html(html_file, f"{label}/html")
    _check_csv(csv_file, f"{label}/csv")
    _check_xlsx(xlsx_file, f"{label}/xlsx")
    _check_json(json_file, f"{label}/json")
    return html_size


def test_exporters_dict() -> None:
    """The dict path: the API layer serialises RunResult before exporting."""
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        scored = _write_scored_csv(tmp / "scored_rows.csv")
        payload = _fake_result(scored)
        size = _run_suite(payload, "dict", tmp)
        print(f"  dict      -> HTML report {size:,} bytes ({size / 1024:.1f} KB)")


def test_exporters_dataclass() -> None:
    """The dataclass path: exporters must work through getattr too."""
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        scored = _write_scored_csv(tmp / "scored_rows.csv")
        payload = _as_dataclass(_fake_result(scored))
        size = _run_suite(payload, "dataclass", tmp)
        print(f"  dataclass -> HTML report {size:,} bytes ({size / 1024:.1f} KB)")


def test_missing_fields_do_not_crash() -> None:
    """An almost-empty result must still export something rather than explode."""
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        empty = {"run_id": "run-empty", "status": "failed", "error": "No models produced a result."}
        for exporter, name, suffix in (
            (exporters.export_html, "html", ".html"),
            (exporters.export_csv, "csv", ".csv"),
            (exporters.export_xlsx, "xlsx", ".xlsx"),
            (exporters.export_summary_json, "json", ".json"),
        ):
            path = exporter(empty, str(tmp / f"empty.{suffix}"))
            _assert_file(path, f"empty/{name}", min_bytes=1)
        print("  empty result -> all four exporters produced a file")


def test_parent_directory_is_created() -> None:
    """Every exporter must create its own parent directory."""
    with tempfile.TemporaryDirectory() as raw_tmp:
        tmp = Path(raw_tmp)
        scored = _write_scored_csv(tmp / "scored_rows.csv")
        payload = _fake_result(scored)
        nested = tmp / "a" / "b" / "c"
        assert not nested.exists(), "the nested directory already exists - test is not meaningful"
        path = exporters.export_html(payload, str(nested / "report.html"))
        assert Path(path).exists(), f"exporter did not write {path!r}"
        assert Path(path).parent.resolve() == nested.resolve(), (
            f"parent directory is {Path(path).parent!s}, expected {nested!s}"
        )
        print("  nested output directory was created automatically")


def main() -> int:
    print("ML Arena - export module tests")
    print(f"  python : {sys.version.split()[0]}")
    print(f"  samples: {config.SAMPLES_DIR}")
    failures = 0
    for fn in (
        test_exporters_dict,
        test_exporters_dataclass,
        test_missing_fields_do_not_crash,
        test_parent_directory_is_created,
    ):
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # pragma: no cover
            failures += 1
            import traceback

            print(f"ERROR {fn.__name__}: {exc}")
            traceback.print_exc()
    print()
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("All export tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
