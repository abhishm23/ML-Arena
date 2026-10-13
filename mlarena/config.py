"""ML Arena — shared configuration and environment bootstrap.

IMPORTANT: import this module *before* importing polars anywhere in the codebase.
The polars wheel performs a CPU feature-flag check at import time which raises on
some Windows CPUs (`RuntimeError: unknown feature flag: 'sse3'`). Setting
POLARS_SKIP_CPU_CHECK before the first polars import avoids a hard crash on those
machines, which matters because this is a tool meant to run on *any* user's laptop.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# --- environment bootstrap (must happen before any polars import) -------------
os.environ.setdefault("POLARS_SKIP_CPU_CHECK", "1")
os.environ.setdefault("POLARS_MAX_THREADS", str(max(1, (os.cpu_count() or 4))))
# keep BLAS/OpenMP from oversubscribing the machine on small laptops
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4))))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# --- paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
SAMPLES_DIR = ROOT / "samples"
DOCS_DIR = ROOT / "docs"
ASSETS_DIR = ROOT / "assets"

APP_NAME = "ML Arena"
APP_TAGLINE = "Where models compete. You just pick the goal."
APP_VERSION = "1.0.0"

MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024          # 4 GB guard
DEFAULT_TIME_BUDGET_S = 75.0                        # per run, all models together
TRAIN_SAMPLE_THRESHOLD = 400_000                    # rows above which we train on a sample
DEFAULT_SEED = 42


def ensure_dirs() -> None:
    """Create every runtime directory. Safe to call repeatedly."""
    for path in (DATA_DIR, RUNS_DIR, SAMPLES_DIR, ASSETS_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --- optional dependency probing ---------------------------------------------
_MODULE_CACHE: dict[str, object | None] = {}


def get_module(name: str):
    """Import a module once, returning None when it is unavailable."""
    if name not in _MODULE_CACHE:
        try:
            _MODULE_CACHE[name] = importlib.import_module(name)
        except Exception:
            _MODULE_CACHE[name] = None
    return _MODULE_CACHE[name]


def have(name: str) -> bool:
    """True when an optional dependency is importable."""
    return get_module(name) is not None


def get_polars():
    """Return the polars module, or None if it could not be imported."""
    return get_module("polars")


def get_pandas():
    return get_module("pandas")


def cpu_count() -> int:
    return os.cpu_count() or 4


def memory_gb() -> float:
    """Best-effort physical memory in GB (used for planning decisions)."""
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    try:  # Windows fallback without psutil
        if sys.platform == "win32":
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 8.0


def gpu_available() -> bool:
    """We target CPU-only laptops; report honestly so the planner can adapt."""
    return False


def capability_report() -> dict:
    """Small JSON-safe summary shown in the UI footer / diagnostics."""
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "python": sys.version.split()[0],
        "cpu_count": cpu_count(),
        "memory_gb": round(memory_gb(), 1),
        "gpu": gpu_available(),
        "polars": getattr(get_polars(), "__version__", None),
        "pyarrow": getattr(get_module("pyarrow"), "__version__", None),
        "sklearn": getattr(get_module("sklearn"), "__version__", None),
        "xgboost": getattr(get_module("xgboost"), "__version__", None),
        "lightgbm": getattr(get_module("lightgbm"), "__version__", None),
        "pandas": getattr(get_module("pandas"), "__version__", None),
    }
