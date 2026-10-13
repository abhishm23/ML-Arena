"""ML Arena — dataset loading and the in-memory dataset store.

Contract: `docs/CONTRACT.md` §3.  Public signatures here are FROZEN.

Design goals
------------
* **Fast** — polars columnar reads with a lazy scan for very large files.
* **Never crash on a messy file** — every reader strategy is wrapped and we
  cascade polars -> polars(all-string) -> pandas -> a clear human sentence.
* **Honest** — the engine actually used, the real row count and any sampling are
  reported in the returned :class:`LoadResult` and in plain-language ``notes``.
"""

from __future__ import annotations

from mlarena import config  # noqa: F401  -- sets POLARS_SKIP_CPU_CHECK before polars loads

import csv
import math
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Sequence

import polars as pl

__all__ = [
    "LoadResult",
    "load_dataset",
    "register_frame",
    "DatasetStore",
    "store",
    "frame_pandas",
    "dataset_ids",
    "json_safe",
]

# --- tunables -----------------------------------------------------------------
#: Above this size we never try to hold the whole file in memory.
LARGE_FILE_BYTES = int(1.5 * 1024 ** 3)
#: Above this size we prefer a lazy scan + collect over a direct read.
LAZY_SCAN_BYTES = 200 * 1024 * 1024
#: How many rows we keep when a file is too large to hold entirely.
LARGE_SAMPLE_ROWS = 200_000
#: Bytes used for delimiter / encoding sniffing.
SNIFF_BYTES = 64 * 1024

_EXT_FORMAT = {
    ".csv": "csv",
    ".txt": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".xls": "xlsx",
    ".json": "json",
    ".ndjson": "json",
    ".jsonl": "json",
}

_MEMORY_ERROR_HINTS = ("memory", "allocation", "out of memory", "oom")


# -----------------------------------------------------------------------------
# Result object
# -----------------------------------------------------------------------------
@dataclass
class LoadResult:
    """Everything the UI and the other agents need to know about a dataset."""

    dataset_id: str
    path: str
    fmt: str
    rows: int
    cols: int
    size_bytes: int
    load_ms: float
    engine: str
    columns: list[str]
    dtypes: dict[str, str]
    derived_from: str | None = None
    notes: list[str] = field(default_factory=list)

    # -- convenience ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (rounds the timing, keeps every contract field)."""
        return {
            "dataset_id": str(self.dataset_id),
            "path": str(self.path),
            "fmt": str(self.fmt),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "size_bytes": int(self.size_bytes),
            "load_ms": round(float(self.load_ms), 1),
            "engine": str(self.engine),
            "columns": [str(c) for c in self.columns],
            "dtypes": {str(k): str(v) for k, v in self.dtypes.items()},
            "derived_from": None if self.derived_from is None else str(self.derived_from),
            "notes": [str(n) for n in self.notes],
        }

    def __repr__(self) -> str:  # keeps log lines readable
        return (
            f"LoadResult(id={self.dataset_id!r}, {self.rows:,}x{self.cols}, "
            f"fmt={self.fmt!r}, engine={self.engine!r}, {self.load_ms:.0f}ms)"
        )


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
def _human_bytes(n: int) -> str:
    step = 1024.0
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < step or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= step
    return f"{value:.1f} TB"


def json_safe(obj: Any) -> Any:
    """Recursively convert numpy / polars / datetime values into JSON types.

    NaN and infinities become ``None`` (JSON has no way to spell them), numpy
    booleans become real booleans, and dates become ISO strings.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, timedelta):
        return obj.total_seconds()
    if isinstance(obj, Decimal):
        value = float(obj)
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pl.Series):
        return [json_safe(v) for v in obj.to_list()]
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return json_safe(item())
        except Exception:
            pass
    try:
        import numpy as np

        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.generic):
            return json_safe(obj.item())
        if isinstance(obj, np.ndarray):
            return [json_safe(v) for v in obj.tolist()]
    except Exception:
        pass
    text = str(obj)
    return text if text else None


def _dtype_name(dtype: Any) -> str:
    """Map a polars dtype onto the contract's short vocabulary."""
    text = str(dtype)
    if text.startswith("Datetime"):
        return "Datetime"
    if text == "Date":
        return "Date"
    if text == "Time":
        return "Time"
    if text in ("Boolean",):
        return "Boolean"
    if text in ("String", "Categorical", "Enum"):
        return "String"
    if text in ("Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"):
        return "Int64"
    if text in ("Float32", "Float64", "Decimal", "Int128"):
        return "Float64"
    if text.startswith("List") or text.startswith("Array") or text.startswith("Struct"):
        return "String"
    if text.startswith("Null"):
        return "String"
    return text


def _schema_dict(df: pl.DataFrame) -> dict[str, str]:
    return {name: _dtype_name(dtype) for name, dtype in zip(df.columns, df.dtypes)}


def _looks_like_memory_error(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in text for hint in _MEMORY_ERROR_HINTS)


def _read_head(path: Path) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(SNIFF_BYTES)
    except OSError:
        return b""


def _sniff_encoding(head: bytes) -> str:
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def _sniff_delimiter(head: bytes, default: str = ",") -> str:
    try:
        text = head.decode("utf-8", errors="replace")
    except Exception:
        return default
    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if not lines:
        return default
    sample = "\n".join(lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        if dialect.delimiter:
            return dialect.delimiter
    except Exception:
        pass
    counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
    best = max(counts, key=lambda k: counts[k])
    return best if counts[best] > 0 else default


def _polars_encoding(encoding: str) -> str:
    """polars only understands utf8 / utf8-lossy; anything else needs pandas."""
    if encoding in ("utf-8", "utf8", "ascii", "utf-8-sig"):
        return "utf8"
    return "utf8-lossy"


def _dedupe_names(names: Sequence[str]) -> list[str]:
    """`["a", "a", "a_1", "a"]` -> `["a", "a_1", "a_1_1", "a_2"]` (stable)."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for raw in names:
        base = str(raw).strip() or "column"
        if base not in seen:
            seen[base] = 0
            out.append(base)
            continue
        seen[base] += 1
        candidate = f"{base}_{seen[base]}"
        while candidate in seen:
            seen[base] += 1
            candidate = f"{base}_{seen[base]}"
        seen[candidate] = 0
        out.append(candidate)
    return out


def _clean_columns(df: pl.DataFrame, notes: list[str]) -> pl.DataFrame:
    """Trim header whitespace and resolve duplicate header names."""
    if df.width == 0:
        return df
    original = [str(c) for c in df.columns]
    fixed = _dedupe_names(original)
    trimmed_names = [c.strip() or "column" for c in original]

    trimmed = sum(1 for a in original if a != a.strip())
    blank = sum(1 for a in original if not a.strip())
    duplicates = len(set(trimmed_names)) != len(trimmed_names)

    if fixed != original:
        try:
            df = df.rename(dict(zip(original, fixed)))
        except Exception:
            # duplicate source names make a mapping ambiguous — rename by position
            df = df.clone()
            df.columns = fixed

    if trimmed:
        notes.append(
            f"Tidied {trimmed} column name{'s' if trimmed != 1 else ''} "
            "by removing extra spaces."
        )
    if blank:
        notes.append(
            f"{blank} column{'s' if blank != 1 else ''} had no header, "
            f"so {'they were' if blank != 1 else 'it was'} named 'column'."
        )
    if duplicates:
        notes.append(
            "Two or more columns shared a name; the extras were renamed "
            "(for example 'price' and 'price_1') so nothing gets mixed up."
        )
    return df


def _coerce_string_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Best-effort typing for a frame we deliberately read as all-text."""
    exprs: list[pl.Expr] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if dtype != pl.String:
            exprs.append(pl.col(name))
            continue
        cleaned = pl.when(pl.col(name).str.strip_chars() == "").then(None).otherwise(
            pl.col(name).str.strip_chars()
        )
        base = int(df.select(cleaned.is_null().sum().alias("n")).item() or 0)
        choice: pl.Expr = cleaned
        for caster in (
            lambda e: e.cast(pl.Int64, strict=False),
            lambda e: e.cast(pl.Float64, strict=False),
            lambda e: e.cast(pl.Date, strict=False),
        ):
            candidate = caster(cleaned)
            nulls = int(df.select(candidate.is_null().sum().alias("n")).item() or 0)
            if nulls == base:
                choice = candidate
                break
        exprs.append(choice.alias(name))
    try:
        return df.select(exprs)
    except Exception:
        return df


# --- text-stored dates --------------------------------------------------------
_DATE_NAME_TOKENS = frozenset(
    {
        "date", "dates", "dt", "day", "time", "timestamp", "ts", "at", "on", "when",
        "month", "year", "period", "week", "quarter", "created", "updated",
        "modified", "start", "end", "from", "to", "begin", "finish", "opened",
        "closed", "due", "shipped", "delivered", "purchased", "ordered",
    }
)
_DATE_LIKE_RE = re.compile(
    r"^\s*("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    r")"
)
_TEXT_DATE_FORMATS: tuple[str | None, ...] = (
    None, "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y",
    "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S",
)


def _name_tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    return [t for t in re.split(r"[^0-9A-Za-z]+", spaced.lower()) if t]


def _looks_like_date_values(series: pl.Series) -> bool:
    try:
        head = [v for v in series.head(50).to_list() if isinstance(v, str)]
    except Exception:
        return False
    if not head:
        return False
    hits = sum(1 for v in head if _DATE_LIKE_RE.match(v))
    return hits >= max(1, int(0.8 * len(head)))


def _promote_text_dates(df: pl.DataFrame, notes: list[str]) -> pl.DataFrame:
    """Turn a date-looking *text* column into a real Date/Datetime column.

    polars' own `try_parse_dates` gives up on a whole column when a single value
    is unparseable.  A business file with one bad date would then reach the
    modelling engine as text, so we promote it here and say so in the notes.
    """
    promoted: list[str] = []
    exprs: list[pl.Expr] = []
    for name, dtype in zip(df.columns, df.dtypes):
        if dtype != pl.String:
            exprs.append(pl.col(name))
            continue
        if not (set(_name_tokens(name)) & _DATE_NAME_TOKENS):
            exprs.append(pl.col(name))
            continue
        series = df.get_column(name)
        non_null = int(series.len() - series.null_count())
        if non_null < 2 or not _looks_like_date_values(series):
            exprs.append(pl.col(name))
            continue
        parsed = _parse_text_dates(series, non_null)
        if parsed is None:
            exprs.append(pl.col(name))
            continue
        exprs.append(parsed.alias(name))
        blanks = int(parsed.null_count()) - int(series.null_count())
        promoted.append((name, max(0, blanks)))

    if not promoted:
        return df
    try:
        out = df.select(exprs)
    except Exception:  # pragma: no cover - defensive
        return df
    names = [n for n, _ in promoted]
    shown = ", ".join(f"'{n}'" for n in names[:4])
    extra = "" if len(names) <= 4 else f" and {len(names) - 4} more"
    notes.append(
        f"{shown}{extra} stored dates as text, so they were converted into real "
        "dates — that lets ML Arena sort and trend over them properly."
    )
    blanks = sum(b for _, b in promoted)
    if blanks:
        notes.append(
            f"{blanks:,} value{'s' if blanks != 1 else ''} in those columns did not "
            "look like a date and were left blank."
        )
    return out


def _parse_text_dates(series: pl.Series, non_null: int) -> pl.Series | None:
    for fmt in _TEXT_DATE_FORMATS:
        for kind in ("date", "datetime"):
            try:
                if kind == "date":
                    parsed = (
                        series.str.to_date(strict=False, exact=False)
                        if fmt is None
                        else series.str.to_date(format=fmt, strict=False, exact=False)
                    )
                else:
                    parsed = (
                        series.str.to_datetime(strict=False, exact=False)
                        if fmt is None
                        else series.str.to_datetime(
                            format=fmt, strict=False, exact=False
                        )
                    )
            except Exception:
                continue
            try:
                got = int(parsed.len() - parsed.null_count())
            except Exception:
                continue
            # same 80% rule the profiler uses to call a text column a date
            if got >= 0.80 * non_null:
                return parsed
    return None


# -----------------------------------------------------------------------------
# CSV / TSV reading
# -----------------------------------------------------------------------------
def _csv_kwargs(separator: str, encoding: str, *, strict: bool) -> dict[str, Any]:
    return {
        "separator": separator,
        "encoding": _polars_encoding(encoding),
        "try_parse_dates": True,
        "ignore_errors": not strict,
        "truncate_ragged_lines": True,
        "raise_if_empty": False,
        "null_values": ["", "NA", "N/A", "na", "n/a", "NULL", "null", "None"],
    }


def _read_csv(
    path: Path,
    *,
    fmt: str,
    size_bytes: int,
    notes: list[str],
) -> tuple[pl.DataFrame, str, bool]:
    """Return ``(frame, engine, sampled)`` for a delimited text file.

    Strict reads are attempted first: a column that starts out numeric and turns
    into text halfway down the file must be *detected*, not silently blanked.
    """
    head = _read_head(path)
    encoding = _sniff_encoding(head)
    if fmt == "tsv":
        separator = "\t"
    else:
        separator = _sniff_delimiter(head, default=",")

    errors: list[str] = []
    strict = _csv_kwargs(separator, encoding, strict=True)
    lenient = _csv_kwargs(separator, encoding, strict=False)

    if encoding not in ("utf-8", "utf8", "ascii", "utf-8-sig"):
        notes.append(
            f"The file is not saved as plain UTF-8 text, so it was read as "
            f"{encoding}."
        )
    if separator != ",":
        label = "tab" if separator == "\t" else f"'{separator}'"
        notes.append(f"Detected a {label}-separated file.")

    # --- strategy 0: too big to hold in memory -> lazy scan + bounded sample --
    if size_bytes > LARGE_FILE_BYTES:
        try:
            lf = pl.scan_csv(path, **strict)
            sample = lf.head(LARGE_SAMPLE_ROWS).collect()
            try:
                total = int(lf.select(pl.len()).collect().item())
            except Exception:
                total = sample.height
            notes.append(
                f"This file is {_human_bytes(size_bytes)}, which is larger than "
                "this computer can comfortably hold in memory. ML Arena read the "
                f"first {LARGE_SAMPLE_ROWS:,} rows so you can still see and analyse "
                "the data — results describe that sample, not every row."
            )
            return sample, "polars", total > sample.height
        except Exception as exc:  # pragma: no cover - depends on machine memory
            errors.append(f"lazy scan: {exc}")

    # --- strategy 1: fast columnar read (strict) -----------------------------
    try:
        if size_bytes > LAZY_SCAN_BYTES:
            frame = pl.scan_csv(path, **strict).collect()
        else:
            frame = pl.read_csv(path, **strict)
        return frame, "polars", False
    except Exception as exc:
        if _looks_like_memory_error(exc):
            try:
                lf = pl.scan_csv(path, **lenient)
                sample = lf.head(LARGE_SAMPLE_ROWS).collect()
                try:
                    total = int(lf.select(pl.len()).collect().item())
                except Exception:
                    total = sample.height
                notes.append(
                    "There was not enough memory to read every row, so ML Arena "
                    f"read the first {LARGE_SAMPLE_ROWS:,} rows instead. Anything "
                    "you see is based on that sample."
                )
                return sample, "polars", total > sample.height
            except Exception as inner:
                errors.append(f"sampled read: {inner}")
        errors.append(f"fast read: {type(exc).__name__}: {exc}")

    # --- strategy 2: full-scan schema inference (the first rows lied) --------
    try:
        frame = pl.read_csv(path, **{**strict, "infer_schema_length": None})
        notes.append(
            "Some rows did not match the first rows of the file, so every row was "
            "checked before deciding the column types."
        )
        return frame, "polars", False
    except Exception as exc:
        errors.append(f"full-scan inference: {type(exc).__name__}: {exc}")

    # --- strategy 3: lenient columnar read (skip unparseable values) ---------
    try:
        frame = pl.read_csv(path, **{**lenient, "infer_schema_length": 10_000})
        notes.append(
            "Some values did not fit their column, so they were left blank rather "
            "than stopping the read."
        )
        return frame, "polars", False
    except Exception as exc:
        errors.append(f"lenient read: {type(exc).__name__}: {exc}")

    # --- strategy 4: read everything as text, then re-type -------------------
    try:
        frame = pl.read_csv(path, **{**lenient, "infer_schema_length": 0})
        frame = _coerce_string_frame(frame)
        notes.append(
            "The file mixes different kinds of values in the same column, so "
            "ML Arena read it as text first and then worked out the types."
        )
        return frame, "polars", False
    except Exception as exc:
        errors.append(f"text-first read: {type(exc).__name__}: {exc}")

    # --- strategy 5: pandas (tolerates far messier files) --------------------
    try:
        frame = _read_csv_pandas(path, separator, encoding)
        notes.append(
            "This file needed the slower compatibility reader, which skips rows "
            "it cannot make sense of."
        )
        return frame, "pandas", False
    except Exception as exc:
        errors.append(f"compatibility read: {type(exc).__name__}: {exc}")

    sep_label = "tab" if separator == "\t" else "comma"
    raise RuntimeError(
        f"ML Arena could not read '{path.name}'. It was tried as a "
        f"{sep_label}-separated text file with the fast reader, the text-first "
        "reader and the compatibility reader, and all three failed. The most "
        "likely causes are that the file is not really a CSV/TSV file, or it is "
        f"damaged. Technical detail: {'; '.join(errors[:3])}"
    )


def _read_csv_pandas(path: Path, separator: str, encoding: str) -> pl.DataFrame:
    import pandas as pd

    last: Exception | None = None
    for enc in (encoding, "utf-8", "cp1252", "latin-1"):
        try:
            pdf = pd.read_csv(
                path,
                sep=separator,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
                skip_blank_lines=True,
                skipinitialspace=True,
            )
            return pl.from_pandas(pdf, include_index=False)
        except Exception as exc:  # try the next encoding
            last = exc
    raise last if last else RuntimeError("pandas could not read the file")


# -----------------------------------------------------------------------------
# other formats
# -----------------------------------------------------------------------------
def _read_parquet(path: Path, notes: list[str]) -> tuple[pl.DataFrame, str]:
    try:
        return pl.read_parquet(path), "polars"
    except Exception as exc:
        errors = [f"polars: {type(exc).__name__}: {exc}"]
    try:
        import pandas as pd

        pdf = pd.read_parquet(path)
        notes.append("This Parquet file needed the compatibility reader.")
        return pl.from_pandas(pdf, include_index=False), "pandas"
    except Exception as exc:
        errors.append(f"compatibility: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"ML Arena could not read the Parquet file '{path.name}'. It may be "
        "damaged or written with an unusual compression codec. Technical detail: "
        + "; ".join(errors)
    )


def _read_excel(path: Path, notes: list[str]) -> tuple[pl.DataFrame, str]:
    import pandas as pd

    errors: list[str] = []
    for kwargs in ({"engine": "openpyxl"}, {}):
        try:
            pdf = pd.read_excel(path, **kwargs)
            if pdf.empty:
                notes.append("This spreadsheet has no rows below its header.")
            return pl.from_pandas(pdf, include_index=False), "pandas"
        except Exception as exc:
            errors.append(f"{kwargs.get('engine', 'default')}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"ML Arena could not read the spreadsheet '{path.name}'. The usual reason "
        "is that it is an old .xls file that has been renamed to .xlsx, or it is "
        "password protected. Technical detail: " + "; ".join(errors)
    )


def _read_json(path: Path, fmt_ext: str, notes: list[str]) -> tuple[pl.DataFrame, str]:
    errors: list[str] = []
    is_nd = fmt_ext in (".ndjson", ".jsonl")
    attempts: list[tuple[str, Callable[[], pl.DataFrame]]] = []
    if is_nd:
        attempts.append(("polars", lambda: pl.read_ndjson(path)))
    else:
        attempts.append(("polars", lambda: pl.read_json(path)))
        attempts.append(("polars", lambda: pl.read_ndjson(path)))
    for label, fn in attempts:
        try:
            frame = fn()
            if frame.width:
                return frame, label
            errors.append(f"{label}: no columns found")
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    try:
        import pandas as pd

        pdf = pd.read_json(path, lines=is_nd)
        notes.append("This JSON file needed the compatibility reader.")
        return pl.from_pandas(pdf, include_index=False), "pandas"
    except Exception as exc:
        errors.append(f"compatibility: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"ML Arena could not read the JSON file '{path.name}'. It should be a list "
        "of records (one object per row), or one object per line. Technical "
        "detail: " + "; ".join(errors[:3])
    )


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame()


# -----------------------------------------------------------------------------
# public API — load
# -----------------------------------------------------------------------------
def load_dataset(src_path: str, dataset_id: str) -> LoadResult:
    """Read a file from disk as fast as possible and register it in the store."""
    started = time.perf_counter()
    path = Path(src_path)
    if not path.exists():
        raise FileNotFoundError(f"There is no file at '{src_path}'.")
    if path.is_dir():
        raise IsADirectoryError(
            f"'{src_path}' is a folder, not a file. Please choose a single data file."
        )

    size_bytes = path.stat().st_size
    ext = path.suffix.lower()
    fmt = _EXT_FORMAT.get(ext, "csv")
    notes: list[str] = []
    sampled = False

    if size_bytes == 0:
        notes.append("This file is empty, so there is nothing to analyse yet.")
        frame, engine = _empty_frame(), "polars"
    elif fmt in ("csv", "tsv"):
        frame, engine, sampled = _read_csv(
            path, fmt=fmt, size_bytes=size_bytes, notes=notes
        )
    elif fmt == "parquet":
        frame, engine = _read_parquet(path, notes)
    elif fmt == "xlsx":
        frame, engine = _read_excel(path, notes)
    elif fmt == "json":
        frame, engine = _read_json(path, ext, notes)
    else:  # pragma: no cover - _EXT_FORMAT covers every branch above
        frame, engine, sampled = _read_csv(
            path, fmt="csv", size_bytes=size_bytes, notes=notes
        )

    if frame.width and frame.height == 0:
        notes.append("This file has column headings but no data rows.")

    frame = _promote_text_dates(frame, notes)
    frame = _clean_columns(frame, notes)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    kind = "fast columnar engine" if engine == "polars" else "compatibility engine"
    if size_bytes == 0:
        head_note = "Nothing to read — the file is empty."
    elif sampled:
        head_note = (
            f"Read the first {frame.height:,} rows of a {_human_bytes(size_bytes)} "
            f"file in {elapsed_ms / 1000.0:.1f} s using the {kind}."
        )
    else:
        head_note = (
            f"Read {frame.height:,} rows and {frame.width:,} columns in "
            f"{elapsed_ms / 1000.0:.1f} s using the {kind} "
            f"({_human_bytes(size_bytes)})."
        )
    notes.insert(0, head_note)

    return _publish(
        dataset_id=dataset_id,
        frame=frame,
        path=str(path.resolve()),
        fmt=fmt,
        size_bytes=size_bytes,
        load_ms=elapsed_ms,
        engine=engine,
        derived_from=None,
        notes=notes,
    )


def register_frame(
    dataset_id: str,
    df: pl.DataFrame,
    *,
    path: str = "",
    fmt: str = "memory",
    derived_from: str | None = None,
    notes: list[str] | None = None,
) -> LoadResult:
    """Register an already-built frame (used by the derived-column engine)."""
    started = time.perf_counter()
    if df is None:
        raise ValueError("register_frame() needs a data frame; got None.")
    if not isinstance(df, pl.DataFrame):
        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                df = pl.from_pandas(df, include_index=False)
            else:  # pragma: no cover - defensive
                df = pl.DataFrame(df)
        except Exception as exc:
            raise TypeError(
                "register_frame() expects a polars DataFrame (or a pandas one); "
                f"got {type(df).__name__}."
            ) from exc

    log: list[str] = list(notes or [])
    frame = _clean_columns(df, log)
    frame = _promote_text_dates(frame, log)
    size_bytes = 0
    if path:
        try:
            candidate = Path(path)
            if candidate.exists():
                size_bytes = candidate.stat().st_size
        except OSError:
            size_bytes = 0
    return _publish(
        dataset_id=dataset_id,
        frame=frame,
        path=path,
        fmt=fmt,
        size_bytes=size_bytes,
        load_ms=(time.perf_counter() - started) * 1000.0,
        engine="polars",
        derived_from=derived_from,
        notes=log,
    )


def _unique_id(dataset_id: str, notes: list[str]) -> str:
    """Never silently overwrite an existing dataset."""
    wanted = str(dataset_id).strip() or "dataset"
    if not store.has(wanted):
        return wanted
    suffix = 2
    while store.has(f"{wanted}_{suffix}"):
        suffix += 1
    fresh = f"{wanted}_{suffix}"
    notes.append(
        f"A dataset called '{wanted}' was already open, so this one was stored as "
        f"'{fresh}' to keep both."
    )
    return fresh


def _publish(
    *,
    dataset_id: str,
    frame: pl.DataFrame,
    path: str,
    fmt: str,
    size_bytes: int,
    load_ms: float,
    engine: str,
    derived_from: str | None,
    notes: list[str],
) -> LoadResult:
    final_id = _unique_id(dataset_id, notes)
    if not path:
        config.ensure_dirs()
        path = str(Path(config.DATA_DIR) / f"{final_id}.{fmt}")
    result = LoadResult(
        dataset_id=final_id,
        path=path,
        fmt=fmt,
        rows=int(frame.height),
        cols=int(frame.width),
        size_bytes=int(size_bytes),
        load_ms=float(load_ms),
        engine=engine,
        columns=[str(c) for c in frame.columns],
        dtypes=_schema_dict(frame),
        derived_from=derived_from,
        notes=list(notes),
    )
    store._put(result, frame)
    return result


# -----------------------------------------------------------------------------
# dataset store
# -----------------------------------------------------------------------------
class DatasetStore:
    """In-memory registry of loaded frames, keyed by dataset id.

    Thread-safe: the API layer serves uploads and previews from several request
    threads at once.
    """

    def __init__(self) -> None:
        self._frames: dict[str, pl.DataFrame] = {}
        self._meta: dict[str, LoadResult] = {}
        self._lock = threading.RLock()

    # -- internals -----------------------------------------------------------
    def _put(self, meta: LoadResult, frame: pl.DataFrame) -> None:
        with self._lock:
            self._frames[meta.dataset_id] = frame
            self._meta[meta.dataset_id] = meta

    def _unknown(self, dataset_id: str) -> KeyError:
        known = self.ids()
        if known:
            listing = ", ".join(repr(k) for k in known[:8])
            if len(known) > 8:
                listing += f", … ({len(known)} in total)"
            hint = f"Datasets currently open: {listing}."
        else:
            hint = "No datasets are open yet — load a file first."
        return KeyError(
            f"There is no dataset called '{dataset_id}'. It may have been removed. {hint}"
        )

    # -- public --------------------------------------------------------------
    def ids(self) -> list[str]:
        with self._lock:
            return list(self._meta.keys())

    def get(self, dataset_id: str) -> pl.DataFrame:
        with self._lock:
            if dataset_id not in self._frames:
                raise self._unknown(dataset_id)
            return self._frames[dataset_id]

    def meta(self, dataset_id: str) -> LoadResult:
        with self._lock:
            if dataset_id not in self._meta:
                raise self._unknown(dataset_id)
            return self._meta[dataset_id]

    def has(self, dataset_id: str) -> bool:
        with self._lock:
            return dataset_id in self._frames

    def drop(self, dataset_id: str) -> None:
        """Forget a dataset. Dropping something that is already gone is fine."""
        with self._lock:
            self._frames.pop(dataset_id, None)
            self._meta.pop(dataset_id, None)

    def list(self) -> list[LoadResult]:
        with self._lock:
            return [self._meta[k] for k in self._meta]

    def to_pandas(self, dataset_id: str, columns: list[str] | None = None):
        frame = self.get(dataset_id)
        if columns:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise KeyError(
                    "These columns are not in the dataset: " + ", ".join(missing)
                )
            frame = frame.select(list(columns))
        return frame.to_pandas()

    def clear(self) -> None:
        """Drop everything (used by tests and 'Start over')."""
        with self._lock:
            self._frames.clear()
            self._meta.clear()


#: Module-level singleton used by the rest of the application.
store: DatasetStore = DatasetStore()


def frame_pandas(dataset_id: str, columns: list[str] | None = None):
    """Pandas view of a stored dataset (the modelling engine expects pandas)."""
    return store.to_pandas(dataset_id, columns)


def dataset_ids() -> list[str]:
    """Ids of every dataset currently in memory, in insertion order."""
    return store.ids()
