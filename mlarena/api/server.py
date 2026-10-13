"""ML Arena — HTTP + WebSocket API and static host.

Everything the browser needs is served from this one process, on the user's own
machine. There is no outbound network traffic anywhere in this application.

Run it with ``python run.py``.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from mlarena import config

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from mlarena.api.jobs import jobs
from mlarena.columns import derived
from mlarena.dataio import loader, profiler
from mlarena.engine import narration, runner
from mlarena.export import exporters
from mlarena.usecases import registry

app = FastAPI(title=config.APP_NAME, version=config.APP_VERSION, docs_url="/api/docs")

# Profiles are cheap to compute but not free; cache them per dataset.
_PROFILE_CACHE: dict[str, profiler.DatasetProfile] = {}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: str, fallback: str = "dataset") -> str:
    stem = Path(text).stem.lower()
    cleaned = _SAFE_NAME.sub("_", stem).strip("._-")
    return (cleaned or fallback)[:48]


def _new_dataset_id(label: str) -> str:
    return f"{_slug(label)}-{uuid.uuid4().hex[:6]}"


def _profile_for(dataset_id: str) -> profiler.DatasetProfile:
    cached = _PROFILE_CACHE.get(dataset_id)
    if cached is not None:
        return cached
    try:
        prof = profiler.profile(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"We could not read the columns of this file: {exc}",
        ) from exc
    _PROFILE_CACHE[dataset_id] = prof
    return prof


def _dataset_payload(dataset_id: str) -> dict:
    try:
        meta = loader.store.meta(dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "dataset_id": dataset_id,
        "dataset": meta.to_dict(),
        "profile": profiler.profile_to_dict(_profile_for(dataset_id)),
    }


def _result_for(run_id: str):
    """The finished run as a RunResult (in memory) or a dict (reloaded from disk)."""
    job = jobs.get(run_id)
    if job is not None and job.result is not None:
        return job.result
    path = Path(config.RUNS_DIR) / run_id / "result.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("result")
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# health & capabilities
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "version": config.APP_VERSION,
        "app": config.APP_NAME,
        "tagline": config.APP_TAGLINE,
        **config.capability_report(),
    }


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def _store_upload(upload: UploadFile) -> tuple[str, Path]:
    config.ensure_dirs()
    original = Path(upload.filename or "upload.csv").name
    suffix = Path(original).suffix or ".csv"
    dataset_id = _new_dataset_id(original)
    dest = Path(config.DATA_DIR) / f"{dataset_id}{suffix}"
    total = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = upload.file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > config.MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "That file is larger than 4 GB. Try exporting just the "
                            "columns you need, or use a Parquet/CSV subset."
                        ),
                    )
                handle.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save the upload: {exc}") from exc
    return dataset_id, dest


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    dataset_id, dest = _store_upload(file)
    try:
        loader.load_dataset(str(dest), dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"We could not open that file: {exc}",
        ) from exc
    _PROFILE_CACHE.pop(dataset_id, None)
    return _dataset_payload(dataset_id)


@app.get("/api/datasets")
def list_datasets() -> dict:
    return {"datasets": [m.to_dict() for m in loader.store.list()]}


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    return _dataset_payload(dataset_id)


@app.get("/api/datasets/{dataset_id}/suggestions")
def get_suggestions(dataset_id: str) -> dict:
    prof = _profile_for(dataset_id)
    return {"suggestions": prof.suggestions or []}


@app.post("/api/datasets/{dataset_id}/columns/preview")
def preview_column(dataset_id: str, payload: dict = Body(...)) -> dict:
    name = str(payload.get("name") or "new_column")
    expression = str(payload.get("expression") or "")
    if not expression.strip():
        return {"ok": False, "error": "Type a formula first, for example: list_price - net_price"}
    try:
        return derived.preview(dataset_id, name, expression, limit=int(payload.get("limit", 200)))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"That formula could not be worked out: {exc}"}


@app.post("/api/datasets/{dataset_id}/columns/apply")
def apply_columns(dataset_id: str, payload: dict = Body(...)) -> dict:
    raw = payload.get("columns") or []
    if not isinstance(raw, list) or not raw:
        raise HTTPException(status_code=422, detail="Add at least one new column before applying.")
    specs = []
    for item in raw:
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="Each new column needs a name and a formula.")
        name = str(item.get("name") or "").strip()
        expression = str(item.get("expression") or "").strip()
        if not name or not expression:
            raise HTTPException(
                status_code=422, detail="Every new column needs both a name and a formula."
            )
        specs.append(derived.DerivedSpec(name=name, expression=expression))
    try:
        new_id = derived.materialize(dataset_id, specs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"That column could not be created: {exc}") from exc
    _PROFILE_CACHE.pop(new_id, None)
    return _dataset_payload(new_id)


# --------------------------------------------------------------------------- #
# use cases
# --------------------------------------------------------------------------- #
@app.get("/api/usecases")
def list_usecases() -> dict:
    glossary: dict = {}
    try:
        glossary = narration.metric_glossary()
    except Exception:
        glossary = {}
    return {
        "usecases": registry.catalog(),
        "functions": derived.function_catalog(),
        "operators": getattr(derived, "OPERATORS", []),
        "metrics": glossary,
    }


@app.get("/api/usecases/{key}")
def get_usecase(key: str) -> dict:
    try:
        return {"usecase": registry.serialize(registry.get_usecase(key))}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# runs
# --------------------------------------------------------------------------- #
@app.post("/api/runs")
async def start_run(payload: dict = Body(...)) -> dict:
    dataset_id = str(payload.get("dataset_id") or "")
    usecase = str(payload.get("usecase") or "")
    if not dataset_id or not loader.store.has(dataset_id):
        raise HTTPException(status_code=404, detail="That dataset is no longer open. Please re-upload it.")
    try:
        uc = registry.get_usecase(usecase)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    mapping = payload.get("mapping") or {}
    if not isinstance(mapping, dict):
        raise HTTPException(status_code=422, detail="Column choices were not in the expected format.")
    mapping = {str(k): [str(c) for c in (v or [])] for k, v in mapping.items()}

    missing = [
        role.label
        for role in uc.roles
        if role.required and len(mapping.get(role.key) or []) < max(1, role.min_count)
    ]
    if missing:
        raise HTTPException(
            status_code=422,
            detail="Please choose a column for: " + ", ".join(missing) + ".",
        )

    options = payload.get("options") or {}
    if not isinstance(options, dict):
        options = {}

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - defensive
        loop = None

    job = jobs.start(dataset_id, usecase, mapping, options, loop=loop, label=uc.name)
    return {"run_id": job.run_id, "status": job.status, "usecase": registry.serialize(uc)}


def _running_run_dict(job) -> dict:
    return {
        "run_id": job.run_id,
        "status": "running",
        "usecase": registry.serialize(registry.get_usecase(job.usecase)),
        "dataset": {"dataset_id": job.dataset_id},
        "target": {},
        "plan": {},
        "leaderboard": [],
        "champion": {},
        "metrics": {},
        "impact": {},
        "drivers": [],
        "insights": [],
        "segments": [],
        "distributions": {},
        "preview": {"columns": [], "rows": []},
        "scored_path": "",
        "scored_columns": [],
        "exports": {},
        "narration": [],
        "log": job.snapshot(),
        "error": None,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(job.started_at)),
        "finished_at": None,
        "duration_ms": job.elapsed_ms,
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    job = jobs.get(run_id)
    if job is None:
        reloaded = _result_for(run_id)
        if reloaded is None:
            raise HTTPException(
                status_code=404,
                detail=f"No analysis called {run_id} is available. It may belong to an earlier session.",
            )
        data = reloaded if isinstance(reloaded, dict) else reloaded.to_dict()
        return {"status": data.get("status", "succeeded"), "run": data, "events": data.get("log", [])}

    finished = job.status != "running"
    data = job.result_dict() if finished and job.result is not None else None
    if data is None:
        data = _running_run_dict(job)
    return {
        "status": job.status,
        "run": data,
        "summary": job.summary(),
        "events": job.snapshot(),
        "error": job.error,
    }


@app.get("/api/runs")
def list_runs() -> dict:
    return {"runs": jobs.list()}


_EXPORTERS = {
    "html": ("report.html", "text/html"),
    "csv": ("scored_rows.csv", "text/csv"),
    "xlsx": ("report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    "json": ("summary.json", "application/json"),
}


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str, fmt: str = Query("html")) -> FileResponse:
    fmt = (fmt or "html").lower().strip()
    if fmt not in _EXPORTERS:
        raise HTTPException(
            status_code=422, detail=f"Unknown format {fmt!r}. Use html, csv, xlsx or json."
        )
    result = _result_for(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="That analysis result is not available yet.")

    filename, media = _EXPORTERS[fmt]
    out_dir = Path(config.RUNS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    if not out_path.exists():
        try:
            if fmt == "html":
                exporters.export_html(result, str(out_path))
            elif fmt == "csv":
                exporters.export_csv(result, str(out_path))
            elif fmt == "xlsx":
                exporters.export_xlsx(result, str(out_path))
            else:
                exporters.export_summary_json(result, str(out_path))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=500, detail=f"We could not build the {fmt} file: {exc}"
            ) from exc

    if not out_path.exists():
        raise HTTPException(status_code=500, detail=f"The {fmt} file was not created.")

    uc = ""
    try:
        data = result if isinstance(result, dict) else result.to_dict()
        uc = (data.get("usecase") or {}).get("name") or ""
    except Exception:
        uc = ""
    download_name = f"ml-arena-{_slug(uc or run_id, 'report')}-{run_id.split('-')[-1]}.{fmt}"

    return FileResponse(str(out_path), media_type=media, filename=download_name)


# --------------------------------------------------------------------------- #
# bundled samples
# --------------------------------------------------------------------------- #
def _sample_usecase(name: str) -> str:
    stem = name.lower()
    explicit = {
        "ml_arena_large_benchmark": "discount_leakage",
        "sales_demand_series": "demand_forecast",
        "customers_churn": "churn_risk",
        "loan_credit_default": "credit_default",
        "transactions_anomaly": "transaction_anomaly",
    }
    for key, usecase in explicit.items():
        if key in stem:
            return usecase
    for uc in registry.all_usecases():
        if uc.key in stem:
            return uc.key
    return "revenue_predict"


@app.get("/api/samples")
def list_samples() -> dict:
    out: list[dict] = []
    if Path(config.SAMPLES_DIR).exists():
        for path in sorted(Path(config.SAMPLES_DIR).glob("*.csv")):
            usecase = _sample_usecase(path.name)
            try:
                label = registry.get_usecase(usecase).name
            except Exception:
                label = usecase.replace("_", " ").title()
            out.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "usecase": usecase,
                    "usecase_label": label,
                }
            )
    out.sort(key=lambda s: s["size_bytes"])
    return {"samples": out}


@app.post("/api/samples/{name}/load")
def load_sample(name: str) -> dict:
    path = Path(config.SAMPLES_DIR) / Path(name).name
    if not path.exists() or path.suffix.lower() != ".csv":
        raise HTTPException(status_code=404, detail=f"There is no bundled sample called {name!r}.")
    dataset_id = _new_dataset_id(path.name)
    try:
        loader.load_dataset(str(path), dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not open that sample: {exc}") from exc
    _PROFILE_CACHE.pop(dataset_id, None)
    return _dataset_payload(dataset_id)


# --------------------------------------------------------------------------- #
# websocket: live event stream
# --------------------------------------------------------------------------- #
_HEARTBEAT = {
    "phase": "heartbeat",
    "level": "info",
    "title": "Still working",
    "detail": "Large files take a moment to chew through — this is normal.",
    "progress": None,
    "model": None,
    "metrics": None,
    "payload": {},
}


@app.websocket("/ws/runs/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    job = jobs.get(run_id)
    if job is None:
        await websocket.send_json(
            {
                "phase": "error",
                "level": "error",
                "title": "We lost track of that analysis",
                "detail": "It may belong to an earlier session. Please start the analysis again.",
                "progress": 1.0,
            }
        )
        await websocket.close()
        return

    job.attach_loop(asyncio.get_running_loop())
    queue = job.subscribe()
    last_seq = 0
    last_send = time.monotonic()
    try:
        # Subscribe before snapshotting so no event can slip through the gap,
        # then drop anything the snapshot already covered.
        for event in job.snapshot():
            await websocket.send_json(event)
            last_seq = max(last_seq, int(event.get("seq") or 0))
        last_send = time.monotonic()

        if job.finished.is_set():
            await _finish_stream(websocket, job)
            return

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
            except asyncio.TimeoutError:
                # The worker thread may have finished without the terminal event
                # reaching this queue (or it was emitted before we subscribed).
                if job.finished.is_set() and queue.empty():
                    await _finish_stream(websocket, job)
                    return
                if time.monotonic() - last_send >= 20.0:
                    await websocket.send_json(_HEARTBEAT)
                    last_send = time.monotonic()
                continue

            seq = int(event.get("seq") or 0)
            if seq and seq <= last_seq:
                continue
            last_seq = max(last_seq, seq)
            await websocket.send_json(event)
            last_send = time.monotonic()
            if event.get("phase") == "done":
                # The engine emits "done" from its own thread a moment before the job
                # manager has stored the finished result. Wait for that hand-off so the
                # client's immediate GET /api/runs/{id} returns the complete payload
                # instead of an empty "still running" shell.
                await _await_finalisation(job)
                return
    except (WebSocketDisconnect, RuntimeError):
        pass
    except Exception:
        pass
    finally:
        job.unsubscribe(queue)
        try:
            await websocket.close()
        except Exception:
            pass


async def _await_finalisation(job, timeout: float = 30.0) -> None:
    """Block (off the event loop) until the worker thread has stored its result."""
    if job.finished.is_set():
        return
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, job.finished.wait, timeout)
    except Exception:
        pass


async def _finish_stream(websocket: WebSocket, job) -> None:
    """Make sure the client always sees a terminal `done` event before we close."""
    last = job.last_event() or {}
    if last.get("phase") != "done":
        await websocket.send_json(
            {
                "phase": "done",
                "level": "success" if job.status == "succeeded" else "error",
                "title": "All done" if job.status == "succeeded" else "Stopped early",
                "detail": job.error or "",
                "progress": 1.0,
                "payload": {"status": job.status},
            }
        )


# --------------------------------------------------------------------------- #
# static frontend (registered last so /api and /ws win the match)
# --------------------------------------------------------------------------- #
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon = Path(config.WEB_DIR) / "favicon.svg"
    if icon.exists():
        return FileResponse(str(icon), media_type="image/svg+xml")
    return HTMLResponse(status_code=204)


@app.get("/", include_in_schema=False)
def index():
    page = Path(config.WEB_DIR) / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<h1>ML Arena</h1><p>The interface files are missing from <code>web/</code>.</p>",
            status_code=500,
        )
    return FileResponse(str(page), media_type="text/html")


if Path(config.WEB_DIR).exists():
    app.mount("/", StaticFiles(directory=str(config.WEB_DIR), html=True), name="web")
