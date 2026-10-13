"""Tests for `mlarena.columns.derived` — the safe custom-column engine.

Run either way:

    "C:\\ML Arena\\.venv\\Scripts\\python.exe" "C:\\ML Arena\\tests\\test_derived.py"
    "C:\\ML Arena\\.venv\\Scripts\\python.exe" -m pytest tests/test_derived.py -q
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mlarena import config  # noqa: E402,F401  -- must precede polars

import polars as pl  # noqa: E402

from mlarena.columns import derived  # noqa: E402
from mlarena.dataio import loader, profiler  # noqa: E402

# Expressions that MUST be rejected, with the reason we expect to see named.
REJECT = [
    "__import__('os').system('dir')",
    "open('x')",
    "a.b",
    "a[0]",
    "lambda: 1",
    "(1).__class__",
    "eval('1+1')",
    "[x for x in a]",
    "{'k': a}",
    "a if True else b.__class__",
    "getattr(a, 'x')",
    "round(a, 2).__class__",
    "a and __import__('os')",
    "'abc'.upper()",
    "(lambda: 1)()",
]

# Expressions that MUST be accepted when the named columns exist.
ACCEPT = [
    "(list_price - net_price) / list_price",
    "quantity * unit_price",
    "ifelse(margin < 0, 1, 0)",
    "coalesce(a, b)",
    "round(x, 2)",
    "abs(profit)",
    "clip(margin_pct, 0, 1)",
    "days_between(end_date, start_date)",
    "year(order_date)",
    "isnull(discount_pct)",
    "not is_flag",
    "a > 0 and b < 10",
    "a if b > 0 else 0",
    "log(revenue) + 1",
    "sqrt(area) ** 2",
]

AVAILABLE = [
    "a", "b", "x", "list_price", "net_price", "quantity", "unit_price", "margin",
    "profit", "margin_pct", "end_date", "start_date", "order_date",
    "discount_pct", "is_flag", "revenue", "area",
]


def _sample_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "order_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(6)],
            "ship_date": [date(2024, 1, 3) + timedelta(days=i) for i in range(6)],
            "status": ["new", "paid", "paid", "shipped", "new", "paid"],
            "list_price": [100.0, 200.0, 50.0, 80.0, 120.0, 400.0],
            "net_price": [80.0, 150.0, 50.0, 60.0, 90.0, 300.0],
            "quantity": [2, 1, 5, 3, 2, 4],
            "unit_price": [10.0, 20.0, 5.0, 8.0, 12.0, 40.0],
            "revenue": [500.0, 800.0, 200.0, 300.0, 450.0, 900.0],
            "cost": [300.0, 500.0, 180.0, 250.0, 400.0, 700.0],
            "margin": [-0.1, 0.2, 0.1, -0.3, 0.05, 0.4],
        }
    )


def _seed(dataset_id: str = "derived_src") -> str:
    loader.store.clear()
    loader.register_frame(dataset_id, _sample_frame())
    return dataset_id


# -----------------------------------------------------------------------------
# AST whitelist
# -----------------------------------------------------------------------------
def test_validate_rejects_dangerous_expressions() -> None:
    for expression in REJECT:
        ok, message = derived.validate_expression(expression, AVAILABLE)
        assert ok is False, f"should have been rejected: {expression!r}"
        assert isinstance(message, str) and message.strip(), expression
        assert "not allowed" in message or "is not" in message or "cannot" in message, (
            expression,
            message,
        )
        print(f"   rejected {expression!r}\n      -> {message}")


def test_validate_accepts_safe_expressions() -> None:
    for expression in ACCEPT:
        ok, message = derived.validate_expression(expression, AVAILABLE)
        assert ok is True, f"should have been accepted: {expression!r} -> {message}"
    print(f"   accepted {len(ACCEPT)} safe expressions")


def test_validate_rejects_unknown_names_and_arity() -> None:
    ok, message = derived.validate_expression("nope + 1", AVAILABLE)
    assert ok is False
    assert "nope" in message

    ok, message = derived.validate_expression("round(x, 2, 3)", AVAILABLE)
    assert ok is False
    assert "round" in message

    ok, message = derived.validate_expression("", AVAILABLE)
    assert ok is False

    ok, message = derived.validate_expression("(x + ", AVAILABLE)
    assert ok is False
    assert "typo" in message.lower()

    # name matching is tolerant of case and underscores
    ok, message = derived.validate_expression("LIST_PRICE - net_price", ["List Price", "Net Price"])
    assert ok is True, message

    # a name containing a space must be written in backticks
    ok, message = derived.validate_expression("`List Price` - `Net Price`", ["List Price", "Net Price"])
    assert ok is True, message
    ok, message = derived.validate_expression("`Missing Column` - 1", ["List Price"])
    assert ok is False
    assert "Missing Column" in message


def test_expression_is_built_without_python_eval() -> None:
    """The module must never reach for eval/exec; only ast + polars expressions."""
    import re

    source = Path(derived.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(?<![.\w])eval\s*\(", source), "bare eval() found"
    assert not re.search(r"(?<![.\w])exec\s*\(", source), "bare exec() found"
    assert not re.search(r"(?<![.\w])compile\s*\(", source), "bare compile() found"
    assert "builtins" not in source
    assert "ast.parse(" in source
    assert "mode=\"eval\"" in source


# -----------------------------------------------------------------------------
# preview
# -----------------------------------------------------------------------------
def test_preview_returns_ok_and_values() -> None:
    _seed()
    result = derived.preview("derived_src", "discount_pct", "(list_price - net_price) / list_price")
    assert result["ok"] is True, result["error"]
    assert result["error"] is None
    assert result["dtype"] == "Float64"
    assert result["row_count"] == 6
    assert len(result["values"]) == 6
    assert abs(result["values"][0] - 0.2) < 1e-9
    assert result["stats"]["nulls"] == 0
    assert abs(result["stats"]["max"] - 0.25) < 1e-9
    json.dumps(result, allow_nan=False)


def test_preview_never_raises_on_bad_input() -> None:
    _seed()
    for expression in REJECT + ["nope + 1", "quantity / 0 if False else nope"]:
        result = derived.preview("derived_src", "x", expression)
        assert result["ok"] is False, expression
        assert isinstance(result["error"], str) and result["error"]
        json.dumps(result, allow_nan=False)

    missing = derived.preview("no_such_dataset", "x", "1 + 1")
    assert missing["ok"] is False
    assert "no_such_dataset" in missing["error"]

    # type mismatch is reported, not raised
    mismatch = derived.preview("derived_src", "x", "year(status)")
    assert mismatch["ok"] is False
    assert mismatch["error"]


def test_division_by_zero_yields_null() -> None:
    _seed()
    zero_column = derived.preview("derived_src", "r", "list_price / 0")
    assert zero_column["ok"] is True, zero_column["error"]
    assert zero_column["stats"]["nulls"] == 6
    assert all(v is None for v in zero_column["values"])

    literal = derived.preview("derived_src", "r", "100 / (net_price - net_price)")
    assert literal["ok"] is True
    assert literal["stats"]["nulls"] == 6

    mod = derived.preview("derived_src", "r", "list_price % 0")
    assert mod["ok"] is True
    assert mod["stats"]["nulls"] == 6


def test_preview_functions_behave() -> None:
    _seed()
    cases = {
        "quantity * unit_price": [20.0, 20.0, 25.0, 24.0, 24.0, 160.0],
        "ifelse(margin < 0, 1, 0)": [1, 0, 0, 1, 0, 0],
        "round(list_price / 3, 2)": [33.33, 66.67, 16.67, 26.67, 40.0, 133.33],
        "coalesce(margin, 0)": [-0.1, 0.2, 0.1, -0.3, 0.05, 0.4],
        "abs(margin)": [0.1, 0.2, 0.1, 0.3, 0.05, 0.4],
        "clip(margin, 0, 1)": [0.0, 0.2, 0.1, 0.0, 0.05, 0.4],
        "days_between(ship_date, order_date)": [2, 2, 2, 2, 2, 2],
        "year(order_date)": [2024] * 6,
        "isnull(margin)": [False] * 6,
    }
    for expression, expected in cases.items():
        result = derived.preview("derived_src", "out", expression)
        assert result["ok"] is True, (expression, result["error"])
        got = result["values"]
        assert len(got) == len(expected), expression
        for actual, want in zip(got, expected):
            if isinstance(want, float):
                assert actual is not None and abs(actual - want) < 1e-6, (expression, actual, want)
            else:
                assert actual == want, (expression, actual, want)


def test_function_catalog() -> None:
    catalog = derived.function_catalog()
    names = {entry["name"] for entry in catalog}
    for required in (
        "abs", "round", "min", "max", "sqrt", "log", "log10", "exp", "clip",
        "floor", "ceil", "ifelse", "coalesce", "year", "month", "day",
        "days_between", "isnull",
    ):
        assert required in names, required
    for entry in catalog:
        assert set(entry) == {"name", "signature", "description", "example"}
        assert all(isinstance(entry[k], str) and entry[k] for k in entry)
    json.dumps(catalog, allow_nan=False)


# -----------------------------------------------------------------------------
# materialize
# -----------------------------------------------------------------------------
def test_materialize_creates_new_dataset_and_leaves_source_alone() -> None:
    source_id = _seed()
    before = list(loader.store.get(source_id).columns)
    before_meta = loader.store.meta(source_id)

    new_id = derived.materialize(
        source_id,
        [
            derived.DerivedSpec("discount_pct", "(list_price - net_price) / list_price"),
            derived.DerivedSpec("line_value", "quantity * unit_price"),
        ],
    )

    assert new_id != source_id
    assert loader.store.has(new_id)
    assert loader.store.get(source_id).columns == before, "source frame was mutated"
    assert loader.store.meta(source_id).columns == before
    assert loader.store.meta(source_id).rows == before_meta.rows

    derived_meta = loader.store.meta(new_id)
    assert derived_meta.derived_from == source_id
    assert derived_meta.columns[: len(before)] == before
    assert "discount_pct" in derived_meta.columns
    assert "line_value" in derived_meta.columns
    assert len(derived_meta.columns) == len(before) + 2
    assert derived_meta.notes
    assert any("discount_pct" in n for n in derived_meta.notes)

    frame = loader.store.get(new_id)
    assert frame.height == 6
    assert abs(frame["discount_pct"][0] - 0.2) < 1e-9
    assert frame["line_value"].to_list() == [20.0, 20.0, 25.0, 24.0, 24.0, 160.0]
    json.dumps(derived_meta.to_dict())


def test_materialize_refuses_to_overwrite() -> None:
    source_id = _seed()
    try:
        derived.materialize(source_id, [derived.DerivedSpec("list_price", "net_price * 2")])
    except ValueError as exc:
        assert "list_price" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("overwriting an existing column must raise")

    try:
        derived.materialize(
            source_id,
            [
                derived.DerivedSpec("new_col", "net_price * 2"),
                derived.DerivedSpec("new_col", "net_price * 3"),
            ],
        )
    except ValueError as exc:
        assert "new_col" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate derived names must raise")

    try:
        derived.materialize(source_id, [derived.DerivedSpec("x", "nope * 2")])
    except ValueError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("a bad formula must raise")

    try:
        derived.materialize(source_id, [])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an empty spec list must raise")

    assert len(loader.store.meta(source_id).columns) == 11


def test_materialize_result_can_be_profiled_and_derived_again() -> None:
    source_id = _seed()
    new_id = derived.materialize(
        source_id, [derived.DerivedSpec("discount_pct", "(list_price - net_price) / list_price")]
    )
    profile = profiler.profile(new_id)
    kinds = {c.name: c.kind for c in profile.columns}
    assert kinds["discount_pct"] == "measure"
    assert profile.quality["headline"]
    json.dumps(profiler.profile_to_dict(profile), allow_nan=False)


def test_materialize_with_quoted_column_names() -> None:
    loader.store.clear()
    loader.register_frame(
        "quoted_src",
        pl.DataFrame(
            {
                "Order Qty": [2, 3],
                "Unit Price": [10.0, 20.0],
                "list_price": [100.0, 200.0],
            }
        ),
    )
    new_id = derived.materialize(
        "quoted_src",
        [derived.DerivedSpec("line_value", "`Order Qty` * `Unit Price`")],
    )
    frame = loader.store.get(new_id)
    assert frame["line_value"].to_list() == [20.0, 60.0]

    # tolerant matching also lets an underscore name find a spaced column
    tolerant = derived.preview("quoted_src", "v", "order_qty * 2")
    assert tolerant["ok"] is True, tolerant["error"]
    assert tolerant["values"] == [4, 6]

    assert derived.quote_column("plain") == "plain"
    assert derived.quote_column("Unit Price") == "`Unit Price`"


# -----------------------------------------------------------------------------
# suggestions
# -----------------------------------------------------------------------------
def test_suggest_derived_discount_pct() -> None:
    loader.store.clear()
    loader.register_frame(
        "suggest_src",
        pl.DataFrame(
            {
                "order_id": list(range(1, 21)),
                "list_price": [100.0 + i for i in range(20)],
                "net_price": [80.0 + i for i in range(20)],
            }
        ),
    )
    profile = profiler.profile("suggest_src")
    suggestions = derived.suggest_derived(profile)

    by_name = {s["name"]: s for s in suggestions}
    assert "discount_pct" in by_name, [s["name"] for s in suggestions]
    entry = by_name["discount_pct"]
    assert entry["expression"] == "(list_price - net_price) / list_price"
    assert isinstance(entry["reason"], str) and len(entry["reason"]) > 20
    assert 0.0 <= entry["confidence"] <= 1.0
    assert "discount_amount" in by_name
    json.dumps(suggestions, allow_nan=False)

    # every suggested formula must actually validate against the real columns
    available = loader.store.get("suggest_src").columns
    for suggestion in suggestions:
        ok, message = derived.validate_expression(suggestion["expression"], available)
        assert ok is True, (suggestion, message)


def test_suggest_derived_all_contract_patterns() -> None:
    loader.store.clear()
    loader.register_frame(
        "suggest_all",
        pl.DataFrame(
            {
                "order_id": list(range(1, 31)),
                "order_date": [date(2024, 1, 1) + timedelta(days=i) for i in range(30)],
                "ship_date": [date(2024, 1, 4) + timedelta(days=i) for i in range(30)],
                "list_price": [100.0 + i for i in range(30)],
                "net_price": [80.0 + i for i in range(30)],
                "revenue": [500.0 + i for i in range(30)],
                "cost": [300.0 + i for i in range(30)],
                "quantity": [1 + (i % 9) for i in range(30)],
                "unit_price": [10.0 + (i % 4) for i in range(30)],
            }
        ),
    )
    profile = profiler.profile("suggest_all")
    suggestions = derived.suggest_derived(profile)
    by_name = {s["name"]: s for s in suggestions}
    print(f"   suggestions: {sorted(by_name)}")

    for required in (
        "discount_pct", "discount_amount", "gross_margin_pct", "gross_profit",
        "extended_value", "duration_days",
    ):
        assert required in by_name, (required, sorted(by_name))

    assert by_name["gross_margin_pct"]["expression"] == "(revenue - cost) / revenue"
    assert by_name["gross_profit"]["expression"] == "revenue - cost"
    assert by_name["extended_value"]["expression"] == "quantity * unit_price"
    assert by_name["duration_days"]["expression"] == "days_between(ship_date, order_date)"

    available = loader.store.get("suggest_all").columns
    for suggestion in suggestions:
        ok, message = derived.validate_expression(suggestion["expression"], available)
        assert ok is True, (suggestion, message)
        assert not derived.preview("suggest_all", suggestion["name"], suggestion["expression"])["error"]


def test_suggest_derived_matches_messy_names_and_dicts() -> None:
    loader.store.clear()
    loader.register_frame(
        "suggest_names",
        pl.DataFrame(
            {
                "Unit List Price": [100.0, 120.0, 90.0],
                "NET_PRICE": [80.0, 100.0, 90.0],
                "Order Qty": [1, 2, 3],
                "Price Per Unit": [10.0, 12.0, 9.0],
            }
        ),
    )
    profile = profiler.profile("suggest_names")
    suggestions = derived.suggest_derived(profile)
    by_name = {s["name"]: s for s in suggestions}
    assert "discount_pct" in by_name
    assert "extended_value" in by_name
    assert by_name["discount_pct"]["expression"] == (
        "(`Unit List Price` - NET_PRICE) / `Unit List Price`"
    )

    # names with spaces are quoted, so the suggestion is a real, runnable formula
    available = loader.store.get("suggest_names").columns
    for suggestion in suggestions:
        ok, message = derived.validate_expression(suggestion["expression"], available)
        assert ok is True, (suggestion, message)
        result = derived.preview(
            "suggest_names", suggestion["name"], suggestion["expression"]
        )
        assert result["error"] is None, (suggestion, result["error"])

    # the dict form of a profile works just as well
    as_dict = profiler.profile_to_dict(profile)
    assert [s["name"] for s in derived.suggest_derived(as_dict)] == [
        s["name"] for s in suggestions
    ]

    assert derived.suggest_derived(None) == []
    assert derived.suggest_derived({"columns": []}) == []


def test_suggest_derived_skips_existing_and_missing_columns() -> None:
    loader.store.clear()
    loader.register_frame(
        "suggest_existing",
        pl.DataFrame(
            {
                "list_price": [100.0, 110.0],
                "net_price": [90.0, 100.0],
                "discount_pct": [0.1, 0.09],
            }
        ),
    )
    suggestions = derived.suggest_derived(profiler.profile("suggest_existing"))
    names = {s["name"] for s in suggestions}
    assert "discount_pct" not in names
    assert "discount_amount" in names

    loader.store.clear()
    loader.register_frame("suggest_lonely", pl.DataFrame({"list_price": [1.0, 2.0]}))
    assert derived.suggest_derived(profiler.profile("suggest_lonely")) == []


# -----------------------------------------------------------------------------
# runner
# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        test_validate_rejects_dangerous_expressions,
        test_validate_accepts_safe_expressions,
        test_validate_rejects_unknown_names_and_arity,
        test_expression_is_built_without_python_eval,
        test_preview_returns_ok_and_values,
        test_preview_never_raises_on_bad_input,
        test_division_by_zero_yields_null,
        test_preview_functions_behave,
        test_function_catalog,
        test_materialize_creates_new_dataset_and_leaves_source_alone,
        test_materialize_refuses_to_overwrite,
        test_materialize_result_can_be_profiled_and_derived_again,
        test_materialize_with_quoted_column_names,
        test_suggest_derived_discount_pct,
        test_suggest_derived_all_contract_patterns,
        test_suggest_derived_matches_messy_names_and_dicts,
        test_suggest_derived_skips_existing_and_missing_columns,
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

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
