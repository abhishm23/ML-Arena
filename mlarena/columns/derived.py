"""ML Arena — safe custom-column engine.

Contract: `docs/CONTRACT.md` §5.  Public signatures here are FROZEN.

A non-technical user can type a formula such as ``(list_price - net_price) /
list_price`` and get a real new column.  That is a code-execution surface, so
this module **never** calls ``eval`` or ``exec``:

1. ``ast.parse(expression, mode="eval")`` parses the text into a syntax tree.
2. Every node is walked against a strict whitelist.  Attribute access,
   subscripts, comprehensions, lambdas, imports, starred args, keyword args and
   the walrus operator are all rejected *by name*, with a message that says
   which part of the formula was the problem.
3. Only then is the tree rebuilt as a **polars expression**, so evaluation is
   vectorised and never runs arbitrary Python.
"""

from __future__ import annotations

from mlarena import config  # noqa: F401  -- sets POLARS_SKIP_CPU_CHECK before polars loads

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import polars as pl

from mlarena.dataio import loader

__all__ = [
    "DerivedSpec",
    "validate_expression",
    "preview",
    "materialize",
    "suggest_derived",
    "function_catalog",
    "quote_column",
    "FUNCTION_NAMES",
    "OPERATORS",
]


@dataclass
class DerivedSpec:
    name: str
    expression: str


# -----------------------------------------------------------------------------
# whitelist
# -----------------------------------------------------------------------------
_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Call,
    ast.Compare,
    ast.BoolOp,
    ast.IfExp,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_BIN_OPS: dict[type, str] = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
    ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
}
_CMP_OPS: dict[type, str] = {
    ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
    ast.Gt: ">", ast.GtE: ">=",
}

#: name -> (min args, max args, one-line description, example)
_FUNCTION_SPECS: dict[str, tuple[int, int, str, str]] = {
    "abs": (1, 1, "The size of a number, ignoring any minus sign.", "abs(profit)"),
    "round": (1, 2, "Round to a number of decimal places (default 0).", "round(margin_pct, 2)"),
    "min": (1, 99, "With one value: the smallest in the column. With several: the smallest of each row.", "min(list_price, net_price)"),
    "max": (1, 99, "With one value: the largest in the column. With several: the largest of each row.", "max(list_price, net_price)"),
    "sqrt": (1, 1, "Square root.", "sqrt(area)"),
    "log": (1, 1, "Natural logarithm (used to tame very skewed numbers).", "log(revenue)"),
    "log10": (1, 1, "Base-10 logarithm.", "log10(revenue)"),
    "exp": (1, 1, "e raised to the power of the value.", "exp(score)"),
    "clip": (3, 3, "Keep values inside a floor and a ceiling.", "clip(margin_pct, 0, 1)"),
    "floor": (1, 1, "Round down to a whole number.", "floor(quantity)"),
    "ceil": (1, 1, "Round up to a whole number.", "ceil(quantity)"),
    "ifelse": (3, 3, "Pick one value when a condition is true, another when it is false.", "ifelse(margin < 0, 1, 0)"),
    "coalesce": (2, 99, "Use the first value that is not empty.", "coalesce(discount_pct, 0)"),
    "year": (1, 1, "The year part of a date.", "year(order_date)"),
    "month": (1, 1, "The month number (1-12) of a date.", "month(order_date)"),
    "day": (1, 1, "The day of the month of a date.", "day(order_date)"),
    "days_between": (2, 2, "Whole days from the second date to the first.", "days_between(ship_date, order_date)"),
    "isnull": (1, 1, "True when the value is missing.", "isnull(discount_pct)"),
}

FUNCTION_NAMES: list[str] = sorted(_FUNCTION_SPECS)

#: Additive extension for the UI's operator cheatsheet (contract §14 returns
#: `functions` and `operators` from `/api/usecases`; `function_catalog()` covers
#: the first, this covers the second).
OPERATORS: list[dict] = [
    {"symbol": "+", "description": "Add two values together.", "example": "net_price + tax"},
    {"symbol": "-", "description": "Take one value away from another.", "example": "revenue - cost"},
    {"symbol": "*", "description": "Multiply two values.", "example": "quantity * unit_price"},
    {"symbol": "/", "description": "Divide one value by another (a zero bottom gives a blank, never an error).", "example": "profit / revenue"},
    {"symbol": "//", "description": "Divide and throw away the remainder.", "example": "items // pack_size"},
    {"symbol": "%", "description": "The remainder after dividing.", "example": "order_no % 10"},
    {"symbol": "**", "description": "Raise to a power.", "example": "growth ** 2"},
    {"symbol": "==", "description": "True when two values are the same.", "example": "status == \"paid\""},
    {"symbol": "!=", "description": "True when two values differ.", "example": "status != \"cancelled\""},
    {"symbol": ">", "description": "True when the left value is larger.", "example": "amount > 1000"},
    {"symbol": ">=", "description": "True when the left value is larger or equal.", "example": "days_late >= 30"},
    {"symbol": "<", "description": "True when the left value is smaller.", "example": "margin < 0"},
    {"symbol": "<=", "description": "True when the left value is smaller or equal.", "example": "discount <= 0.5"},
    {"symbol": "and", "description": "True only when both conditions hold.", "example": "amount > 1000 and status == \"paid\""},
    {"symbol": "or", "description": "True when either condition holds.", "example": "region == \"North\" or region == \"South\""},
    {"symbol": "not", "description": "Flips true to false and back.", "example": "not is_churned"},
]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Case / space / underscore insensitive key: 'List Price' -> 'listprice'."""
    return re.sub(r"[^0-9a-z]+", "", str(text).lower())


def _src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return type(node).__name__


def _resolve(name: str, available: Sequence[str], aliases: dict[str, str] | None = None) -> str | None:
    target = aliases.get(name, name) if aliases else name
    if target in available:
        return target
    key = _norm(target)
    for column in available:
        if _norm(column) == key:
            return column
    return None


def _display_name(name: str, aliases: dict[str, str] | None) -> str:
    return aliases.get(name, name) if aliases else name


_BACKTICK_RE = re.compile(r"`([^`]+)`")
_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _prepare(expression: str, available: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Swap ``\\`List Price\\``` for a parseable placeholder.

    Column names with spaces cannot be written directly in a formula, so the UI
    (and our own suggestions) can wrap them in backticks.  The placeholder is
    mapped back to the real column before the expression is built.
    """
    aliases: dict[str, str] = {}
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        raw = match.group(1).strip()
        real = _resolve(raw, available) or raw
        key = f"mlarena_col_{counter}"
        counter += 1
        aliases[key] = real
        return key

    return _BACKTICK_RE.sub(replace, str(expression)), aliases


def quote_column(name: str) -> str:
    """Render a column name so it is safe inside a formula."""
    text = str(name)
    return text if _SAFE_IDENT_RE.match(text) else f"`{text}`"


def _reject_message(node: ast.AST) -> str:
    snippet = _src(node)
    if isinstance(node, ast.Attribute):
        return (
            f"'{snippet}' is not allowed — a formula cannot reach inside an object "
            "with a dot. Use a function such as abs(x), or the '+' '-' '*' '/' "
            "operators."
        )
    if isinstance(node, ast.Subscript):
        return (
            f"'{snippet}' is not allowed — formulas work on a whole column at a "
            "time, not on one row at a time."
        )
    if isinstance(node, ast.Lambda):
        return "A formula cannot define its own function (no 'lambda')."
    if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "A formula cannot loop over values."
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        return f"'{snippet}' is not allowed — a formula works with single numbers and columns."
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "A formula cannot import anything."
    if isinstance(node, ast.NamedExpr):
        return "The ':=' assignment inside a formula is not allowed."
    if isinstance(node, ast.Starred):
        return "'*args' is not allowed — pass each value separately."
    if isinstance(node, ast.keyword):
        return f"Named arguments like '{node.arg}=' are not allowed — pass the values in order."
    if isinstance(node, ast.JoinedStr):
        return "Text templates (f-strings) are not allowed in a formula."
    if isinstance(node, ast.Await):
        return "'await' is not allowed in a formula."
    if isinstance(node, getattr(ast, "Slice", ())):
        return "Slices are not allowed in a formula."
    return f"'{snippet}' is not something a formula can use."


def _check(
    node: ast.AST,
    available: Sequence[str],
    aliases: dict[str, str] | None = None,
) -> str | None:
    """Walk the tree in source order and return the first problem, or None."""
    if isinstance(node, ast.Expression):
        return _check(node.body, available, aliases)

    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return None
        return f"The value '{node.value!r}' is not allowed in a formula."

    if isinstance(node, ast.Name):
        if _resolve(node.id, available, aliases) is None:
            shown = _display_name(node.id, aliases)
            hint = ", ".join(sorted(available)[:12])
            more = "" if len(available) <= 12 else f" … ({len(available)} in total)"
            return (
                f"There is no column called '{shown}'. Columns you can use: "
                f"{hint}{more}."
            )
        return None

    if isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            return f"The operator '{_src(node.op)}' is not allowed in a formula."
        return _check(node.left, available, aliases) or _check(node.right, available, aliases)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            return f"The operator '{_src(node.op)}' is not allowed in a formula."
        return _check(node.operand, available, aliases)

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            return "Only 'and' and 'or' can join conditions."
        for value in node.values:
            error = _check(value, available, aliases)
            if error:
                return error
        return None

    if isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _CMP_OPS:
                return f"The comparison '{_src(op)}' is not allowed in a formula."
        error = _check(node.left, available, aliases)
        if error:
            return error
        for comparator in node.comparators:
            error = _check(comparator, available, aliases)
            if error:
                return error
        return None

    if isinstance(node, ast.IfExp):
        for part in (node.test, node.body, node.orelse):
            error = _check(part, available, aliases)
            if error:
                return error
        return None

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            return (
                f"'{_src(node.func)}' cannot be called — only the built-in formula "
                f"functions can. Try one of: {', '.join(FUNCTION_NAMES)}."
            )
        fname = node.func.id
        if fname not in _FUNCTION_SPECS:
            return (
                f"'{fname}' is not one of the formula functions. Available: "
                f"{', '.join(FUNCTION_NAMES)}."
            )
        if node.keywords:
            return _reject_message(node.keywords[0])
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                return _reject_message(arg)
            error = _check(arg, available, aliases)
            if error:
                return error
        low, high, _, _ = _FUNCTION_SPECS[fname]
        count = len(node.args)
        if count < low or count > high:
            if low == high:
                wanted = f"exactly {low}"
            elif high >= 99:
                wanted = f"at least {low}"
            else:
                wanted = f"between {low} and {high}"
            return (
                f"'{fname}' needs {wanted} value{'s' if low != 1 else ''}, "
                f"but {count} {'were' if count != 1 else 'was'} given."
            )
        return None

    if isinstance(node, ast.Load):
        return None

    return _reject_message(node)


def validate_expression(expression: str, available: list[str]) -> tuple[bool, str]:
    """``(ok, message)``.  Rejects anything outside the whitelist."""
    if expression is None or not str(expression).strip():
        return False, "Type a formula first — for example (revenue - cost) / revenue."
    if not isinstance(available, (list, tuple)):
        available = list(available or [])
    available = [str(c) for c in available]

    text, aliases = _prepare(expression, available)
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        detail = exc.msg or "unexpected symbol"
        return False, f"There is a typo in the formula near position {exc.offset or 0}: {detail}."
    except (ValueError, MemoryError, RecursionError) as exc:
        return False, f"This formula could not be read: {exc}."

    # cheap global sweep first, so exotic nodes are always caught
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False, _reject_message(node)

    error = _check(tree, available, aliases)
    if error:
        return False, error
    return True, "Looks good."


# -----------------------------------------------------------------------------
# polars expression builder
# -----------------------------------------------------------------------------
def _lit(value: Any) -> pl.Expr:
    if isinstance(value, bool):
        return pl.lit(value, dtype=pl.Boolean)
    return pl.lit(value)


def _safe_ratio(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    """Division that yields null on a zero denominator instead of blowing up."""
    return pl.when(right.cast(pl.Float64) == 0).then(None).otherwise(left / right)


def _safe_floor_div(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return pl.when(right.cast(pl.Float64) == 0).then(None).otherwise(left // right)


def _safe_mod(left: pl.Expr, right: pl.Expr) -> pl.Expr:
    return pl.when(right.cast(pl.Float64) == 0).then(None).otherwise(left % right)


def _build_binop(op: type, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if op is ast.Add:
        return left + right
    if op is ast.Sub:
        return left - right
    if op is ast.Mult:
        return left * right
    if op is ast.Div:
        return _safe_ratio(left, right)
    if op is ast.FloorDiv:
        return _safe_floor_div(left, right)
    if op is ast.Mod:
        return _safe_mod(left, right)
    if op is ast.Pow:
        return left ** right
    raise ValueError(f"Unsupported operator: {_src(op)}")


def _build_cmp(op: type, left: pl.Expr, right: pl.Expr) -> pl.Expr:
    if op is ast.Eq:
        return left == right
    if op is ast.NotEq:
        return left != right
    if op is ast.Lt:
        return left < right
    if op is ast.LtE:
        return left <= right
    if op is ast.Gt:
        return left > right
    if op is ast.GtE:
        return left >= right
    raise ValueError(f"Unsupported comparison: {_src(op)}")


def _fn_min(args: list[pl.Expr]) -> pl.Expr:
    return args[0].min() if len(args) == 1 else pl.min_horizontal(*args)


def _fn_max(args: list[pl.Expr]) -> pl.Expr:
    return args[0].max() if len(args) == 1 else pl.max_horizontal(*args)


def _fn_round(args: list[pl.Expr]) -> pl.Expr:
    if len(args) == 1:
        return args[0].round(0)
    return args[0].round(args[1])


def _fn_round_from(
    node: ast.Call,
    available: Sequence[str],
    aliases: dict[str, str] | None = None,
) -> pl.Expr:
    """`round(x, 2)` — polars wants a plain number of decimals, not a column."""
    value = _to_expr(node.args[0], available, aliases)
    if len(node.args) == 1:
        return value.round(0)
    decimals = node.args[1]
    if isinstance(decimals, ast.Constant) and isinstance(decimals.value, int):
        return value.round(int(decimals.value))
    return value.round(_to_expr(decimals, available, aliases))


def _fn_clip(args: list[pl.Expr]) -> pl.Expr:
    return args[0].clip(args[1], args[2])


def _fn_ifelse(args: list[pl.Expr]) -> pl.Expr:
    return pl.when(args[0]).then(args[1]).otherwise(args[2])


def _fn_days_between(args: list[pl.Expr]) -> pl.Expr:
    return (args[0] - args[1]).dt.total_days()


_FUNCTIONS: dict[str, Callable[[list[pl.Expr]], pl.Expr]] = {
    "abs": lambda a: a[0].abs(),
    "round": _fn_round,
    "min": _fn_min,
    "max": _fn_max,
    "sqrt": lambda a: a[0].sqrt(),
    "log": lambda a: a[0].log(),
    "log10": lambda a: a[0].log(10),
    "exp": lambda a: a[0].exp(),
    "clip": _fn_clip,
    "floor": lambda a: a[0].floor(),
    "ceil": lambda a: a[0].ceil(),
    "ifelse": _fn_ifelse,
    "coalesce": lambda a: pl.coalesce(a),
    "year": lambda a: a[0].dt.year(),
    "month": lambda a: a[0].dt.month(),
    "day": lambda a: a[0].dt.day(),
    "days_between": _fn_days_between,
    "isnull": lambda a: a[0].is_null(),
}


def _to_expr(
    node: ast.AST,
    available: Sequence[str],
    aliases: dict[str, str] | None = None,
) -> pl.Expr:
    if isinstance(node, ast.Expression):
        return _to_expr(node.body, available, aliases)

    if isinstance(node, ast.Constant):
        return _lit(node.value)

    if isinstance(node, ast.Name):
        column = _resolve(node.id, available, aliases)
        if column is None:  # pragma: no cover - validate_expression catches this
            raise ValueError(
                f"There is no column called '{_display_name(node.id, aliases)}'."
            )
        return pl.col(column)

    if isinstance(node, ast.BinOp):
        return _build_binop(
            type(node.op),
            _to_expr(node.left, available, aliases),
            _to_expr(node.right, available, aliases),
        )

    if isinstance(node, ast.UnaryOp):
        operand = _to_expr(node.operand, available, aliases)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        return operand.not_()

    if isinstance(node, ast.BoolOp):
        values = [_to_expr(v, available, aliases) for v in node.values]
        if isinstance(node.op, ast.And):
            out = values[0]
            for value in values[1:]:
                out = out & value
            return out
        out = values[0]
        for value in values[1:]:
            out = out | value
        return out

    if isinstance(node, ast.Compare):
        pieces = [_to_expr(node.left, available, aliases)]
        pieces.extend(_to_expr(c, available, aliases) for c in node.comparators)
        result = _build_cmp(type(node.ops[0]), pieces[0], pieces[1])
        for index in range(1, len(node.ops)):
            result = result & _build_cmp(
                type(node.ops[index]), pieces[index], pieces[index + 1]
            )
        return result

    if isinstance(node, ast.IfExp):
        return (
            pl.when(_to_expr(node.test, available, aliases))
            .then(_to_expr(node.body, available, aliases))
            .otherwise(_to_expr(node.orelse, available, aliases))
        )

    if isinstance(node, ast.Call):
        fname = node.func.id  # type: ignore[union-attr]
        if fname == "round":
            return _fn_round_from(node, available, aliases)
        args = [_to_expr(a, available, aliases) for a in node.args]
        return _FUNCTIONS[fname](args)

    raise ValueError(f"Unsupported formula element: {_src(node)}")


def _expression_to_polars(expression: str, available: Sequence[str]) -> pl.Expr:
    available = [str(c) for c in available]
    text, aliases = _prepare(expression, available)
    tree = ast.parse(text.strip(), mode="eval")
    return _to_expr(tree, available, aliases)


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------
def function_catalog() -> list[dict]:
    """For the UI's formula helper."""
    return [
        {
            "name": name,
            "signature": _signature(name),
            "description": spec[2],
            "example": spec[3],
        }
        for name, spec in _FUNCTION_SPECS.items()
    ]


def _signature(name: str) -> str:
    return {
        "abs": "abs(x)",
        "round": "round(x, decimals)",
        "min": "min(a, b, ...)",
        "max": "max(a, b, ...)",
        "sqrt": "sqrt(x)",
        "log": "log(x)",
        "log10": "log10(x)",
        "exp": "exp(x)",
        "clip": "clip(x, low, high)",
        "floor": "floor(x)",
        "ceil": "ceil(x)",
        "ifelse": "ifelse(condition, a, b)",
        "coalesce": "coalesce(a, b, ...)",
        "year": "year(date)",
        "month": "month(date)",
        "day": "day(date)",
        "days_between": "days_between(end_date, start_date)",
        "isnull": "isnull(x)",
    }[name]


def preview(dataset_id: str, name: str, expression: str, *, limit: int = 200) -> dict:
    """Try a formula and show the first few results.  Never raises."""
    def failure(message: str, row_count: int = 0) -> dict:
        return {
            "ok": False,
            "error": message,
            "values": [],
            "dtype": "Unknown",
            "stats": {"min": None, "max": None, "mean": None, "nulls": 0},
            "row_count": int(row_count),
        }

    try:
        frame = loader.store.get(dataset_id)
    except KeyError as exc:
        return failure(str(exc.args[0] if exc.args else exc))
    except Exception as exc:  # pragma: no cover - defensive
        return failure(f"Could not open the dataset: {exc}")

    available = [str(c) for c in frame.columns]
    ok, message = validate_expression(expression, available)
    if not ok:
        return failure(message, frame.height)

    label = (name or "result").strip() or "result"
    try:
        series = frame.select(
            _expression_to_polars(expression, available).alias(label)
        ).to_series()
    except Exception as exc:
        return failure(
            "That formula is allowed, but the values in your columns do not fit "
            "it. Check that you are using dates where dates are needed and "
            f"numbers where numbers are needed. ({type(exc).__name__})",
            frame.height,
        )

    dtype = loader._dtype_name(series.dtype)
    try:
        values = [loader.json_safe(v) for v in series.head(limit).to_list()]
    except Exception:  # pragma: no cover - defensive
        values = []

    stats: dict[str, Any] = {"min": None, "max": None, "mean": None, "nulls": 0}
    try:
        stats["nulls"] = int(series.null_count())
    except Exception:
        pass
    if series.dtype.is_numeric():
        try:
            stats["min"] = loader.json_safe(series.min())
            stats["max"] = loader.json_safe(series.max())
            stats["mean"] = loader.json_safe(series.mean())
        except Exception:
            pass
    elif series.dtype.is_temporal():
        try:
            stats["min"] = loader.json_safe(series.min())
            stats["max"] = loader.json_safe(series.max())
        except Exception:
            pass

    return {
        "ok": True,
        "error": None,
        "values": values,
        "dtype": dtype,
        "stats": stats,
        "row_count": int(frame.height),
    }


def materialize(dataset_id: str, specs: list[DerivedSpec]) -> str:
    """Return a NEW dataset_id with the original columns plus the derived ones."""
    frame = loader.store.get(dataset_id)
    if not specs:
        raise ValueError(
            "There are no custom columns to add — write a formula first."
        )

    existing = [str(c) for c in frame.columns]
    seen: set[str] = set()
    exprs: list[pl.Expr] = []
    notes: list[str] = []

    for spec in specs:
        name = str(getattr(spec, "name", "") or "").strip()
        expression = str(getattr(spec, "expression", "") or "").strip()
        if not name:
            raise ValueError("Every custom column needs a name.")
        if name in existing:
            raise ValueError(
                f"There is already a column called '{name}'. Please choose a "
                "different name for the new column — ML Arena will not overwrite "
                "your data."
            )
        if name in seen:
            raise ValueError(
                f"Two custom columns are both called '{name}'. Please give each "
                "one a unique name."
            )
        seen.add(name)

        ok, message = validate_expression(expression, existing)
        if not ok:
            raise ValueError(f"The formula for '{name}' has a problem: {message}")

        try:
            expr = _expression_to_polars(expression, existing).alias(name)
        except Exception as exc:
            raise ValueError(
                f"The formula for '{name}' could not be prepared: {exc}"
            ) from exc
        exprs.append(expr)
        notes.append(f"Created the custom column '{name}' = {expression}.")

    try:
        built = frame.with_columns(exprs)
    except Exception as exc:
        raise ValueError(
            "Those formulas are allowed, but the values in your columns do not "
            f"fit them (for example using a date where a number is needed). "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    notes.append(
        f"This new table keeps all {len(existing)} original columns and adds "
        f"{len(exprs)} custom column{'s' if len(exprs) != 1 else ''}."
    )
    result = loader.register_frame(
        dataset_id=f"{dataset_id}__custom",
        df=built,
        fmt="memory",
        derived_from=dataset_id,
        notes=notes,
    )
    return result.dataset_id


# -----------------------------------------------------------------------------
# suggestions
# -----------------------------------------------------------------------------
def _profile_columns(profile: Any) -> list[tuple[str, str, str]]:
    """Normalise a DatasetProfile (or its dict form) into (name, kind, semantic)."""
    if profile is None:
        return []
    if isinstance(profile, dict):
        raw = profile.get("columns") or []
    else:
        raw = getattr(profile, "columns", None) or []
    out: list[tuple[str, str, str]] = []
    for column in raw:
        if isinstance(column, dict):
            name = column.get("name")
            kind = column.get("kind") or ""
            semantic = column.get("semantic") or ""
        else:
            name = getattr(column, "name", None)
            kind = getattr(column, "kind", "") or ""
            semantic = getattr(column, "semantic", "") or ""
        if name:
            out.append((str(name), str(kind), str(semantic)))
    return out


def _alias_hit(norm: str, tokens: str, alias: str) -> bool:
    """Short aliases must match a whole token; long ones may match inside a word."""
    if len(alias) >= 7:
        return alias in norm
    return f"_{alias}_" in f"_{tokens}_"


def _find(
    columns: Sequence[tuple[str, str, str]],
    aliases: Sequence[str],
    kinds: Sequence[str] | None = None,
    taken: set[str] | None = None,
) -> str | None:
    best: tuple[int, str] | None = None
    for name, kind, _semantic in columns:
        if taken and name in taken:
            continue
        if kinds and kind not in kinds:
            continue
        norm = _norm(name)
        tokens = "_".join(_tokenize(name))
        for alias in aliases:
            if _alias_hit(norm, tokens, alias):
                score = len(alias)
                if best is None or score > best[0]:
                    best = (score, name)
    return best[1] if best else None


def _tokenize(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [t for t in re.split(r"[^0-9A-Za-z]+", spaced.lower()) if t]


_LIST_PRICE = ("listprice", "msrp", "retailprice", "rrp", "originalprice",
               "catalogprice", "standardprice", "grossprice", "pricelist")
_NET_PRICE = ("netprice", "paidprice", "finalprice", "sellingprice", "actualprice",
              "invoiceprice", "saleprice", "netunitprice")
_REVENUE = ("revenue", "netsales", "grosssales", "totalrevenue", "salesamount",
            "revenueamount", "turnover", "sales")
_COST = ("totalcost", "unitcost", "costamount", "cogs", "cost", "costs")
_QUANTITY = ("quantity", "qtyordered", "quantitysold", "numunits", "units", "qty")
_UNIT_PRICE = ("unitprice", "priceperunit", "priceeach", "unitrate", "unitcostprice")
_DISCOUNT_PCT = ("discountpct", "discountpercent", "discountrate",
                 "discountpercentage", "pctdiscount", "discount")
_MARGIN = ("grossmarginpct", "marginpct", "profitmargin", "marginpercent",
           "grossmargin", "margin")
_START = ("startdate", "start", "begindate", "begin", "fromdate", "opendate",
          "createdat", "createddate", "orderdate", "purchasedate", "firstseen")
_END = ("enddate", "end", "finishdate", "finish", "todate", "closedate",
        "completedat", "completeddate", "shipdate", "deliverydate", "duedate",
        "lastseen")


def _add(
    out: list[dict],
    *,
    name: str,
    expression: str,
    reason: str,
    confidence: float,
    taken: set[str],
) -> None:
    if name in taken:
        return
    if any(item["expression"] == expression for item in out):
        return
    taken.add(name)
    out.append(
        {
            "name": name,
            "expression": expression,
            "reason": reason,
            "confidence": round(float(confidence), 2),
        }
    )


def suggest_derived(profile: Any) -> list[dict]:
    """Auto-suggest useful derived columns from column names."""
    columns = _profile_columns(profile)
    if not columns:
        return []

    out: list[dict] = []
    taken: set[str] = {name for name, _k, _s in columns}

    list_price = _find(columns, _LIST_PRICE)
    net_price = _find(columns, _NET_PRICE)
    revenue = _find(columns, _REVENUE)
    cost = _find(columns, _COST)
    quantity = _find(columns, _QUANTITY)
    unit_price = _find(columns, _UNIT_PRICE)
    if unit_price is None and not list_price and not net_price:
        unit_price = _find(columns, ("price",))
    discount_pct = _find(columns, _DISCOUNT_PCT)
    margin = _find(columns, _MARGIN)

    # --- 1 & 2: list price vs net price -------------------------------------
    if list_price and net_price and list_price != net_price:
        _add(
            out,
            name="discount_pct",
            expression=(f"({quote_column(list_price)} - {quote_column(net_price)}) "
                f"/ {quote_column(list_price)}"),
            reason=(
                f"Compares what customers actually paid ({net_price}) with the full "
                f"list price ({list_price}). A high number means a big discount, "
                "which is the fastest way to spot money leaking away."
            ),
            confidence=0.95,
            taken=taken,
        )
        _add(
            out,
            name="discount_amount",
            expression=f"{quote_column(list_price)} - {quote_column(net_price)}",
            reason=(
                f"The money taken off each line — {list_price} minus {net_price}. "
                "Adding this up tells you how much discount you gave away in total."
            ),
            confidence=0.9,
            taken=taken,
        )

    # --- 3 & 4: revenue vs cost ---------------------------------------------
    if revenue and cost and revenue != cost:
        _add(
            out,
            name="gross_margin_pct",
            expression=(f"({quote_column(revenue)} - {quote_column(cost)}) "
                f"/ {quote_column(revenue)}"),
            reason=(
                f"The share of each sale you keep after paying the direct cost — "
                f"({revenue} - {cost}) divided by {revenue}. Low values are where "
                "you are selling at little or no profit."
            ),
            confidence=0.93,
            taken=taken,
        )
        _add(
            out,
            name="gross_profit",
            expression=f"{quote_column(revenue)} - {quote_column(cost)}",
            reason=(
                f"The cash left from each sale after the direct cost ({revenue} "
                f"minus {cost})."
            ),
            confidence=0.9,
            taken=taken,
        )

    # --- 5: quantity x unit price -------------------------------------------
    if quantity and unit_price and quantity != unit_price:
        _add(
            out,
            name="extended_value",
            expression=f"{quote_column(quantity)} * {quote_column(unit_price)}",
            reason=(
                f"Multiplies the number of items ({quantity}) by the price of one "
                f"({unit_price}) to give the total value of the line."
            ),
            confidence=0.88,
            taken=taken,
        )

    # --- 6: discount percent x revenue --------------------------------------
    if discount_pct and revenue and discount_pct != revenue:
        _add(
            out,
            name="discount_value",
            expression=f"{quote_column(discount_pct)} * {quote_column(revenue)}",
            reason=(
                f"Turns the discount percentage ({discount_pct}) into the money it "
                f"cost you, by applying it to {revenue}."
            ),
            confidence=0.8,
            taken=taken,
        )

    # --- 7: a pair of dates --------------------------------------------------
    date_columns = [
        (name, kind, semantic)
        for name, kind, semantic in columns
        if kind == "datetime" or semantic == "date"
    ]
    if len(date_columns) >= 2:
        start = _find(date_columns, _START) or date_columns[0][0]
        end = _find(
            [c for c in date_columns if c[0] != start], _END
        )
        if end is None:
            end = next((c[0] for c in date_columns if c[0] != start), None)
        if start and end and start != end:
            _add(
                out,
                name="duration_days",
                expression=f"days_between({quote_column(end)}, {quote_column(start)})",
                reason=(
                    f"Counts the days between {start} and {end}. Long gaps often "
                    "point at slow approvals, late deliveries or delays worth "
                    "investigating."
                ),
                confidence=0.85,
                taken=taken,
            )

    # --- 8: generic ratios between measures ---------------------------------
    measure_columns = [
        (name, kind, semantic)
        for name, kind, semantic in columns
        if kind == "measure" and semantic in ("currency", "quantity", "count", "numeric")
    ]
    pairs = 0
    for index, (first, _k, _s) in enumerate(measure_columns):
        if pairs >= 4:
            break
        for second, _k2, _s2 in measure_columns[index + 1:]:
            if pairs >= 4:
                break
            base = _norm(first)
            other = _norm(second)
            if not base or not other or base == other:
                continue
            label = f"{_safe_name(first)}_per_{_safe_name(second)}"
            _add(
                out,
                name=label,
                expression=f"{quote_column(first)} / {quote_column(second)}",
                reason=(
                    f"Divides {first} by {second} so you can compare rows fairly, "
                    "whatever their size."
                ),
                confidence=0.45,
                taken=taken,
            )
            pairs += 1

    return out


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    return cleaned or "value"
