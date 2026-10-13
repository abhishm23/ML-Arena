"""Job manager for ML Arena.

The analysis engine (`mlarena.engine.runner`) is deliberately synchronous: it is
plain CPU-bound Python that streams progress through an ``emit`` callback. The web
layer is asynchronous. This module is the bridge.

A run executes on a worker thread; every event it emits is appended to a
thread-safe list *and* fanned out to any number of :class:`asyncio.Queue`
subscribers on the server's event loop. Because events are retained, a browser that
connects a few hundred milliseconds late still receives the whole story from the
beginning instead of starting mid-sentence.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from mlarena import config
from mlarena.engine import runner

# How many events we keep for replay. A full run emits well under this.
_EVENT_BUFFER = 4000
# Per-subscriber queue depth. Large enough that a slow browser cannot stall the run.
_QUEUE_DEPTH = 2000


def _safe_put(queue: asyncio.Queue, item: Any) -> None:
    """Put without ever raising — a full queue must not break the worker thread."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        try:
            queue.get_nowait()  # drop the oldest event and retry once
            queue.put_nowait(item)
        except Exception:
            pass
    except Exception:
        pass


class RunJob:
    """A single analysis run: its events, subscribers and final result."""

    def __init__(
        self,
        run_id: str,
        dataset_id: str,
        usecase: str,
        mapping: dict[str, list[str]],
        options: dict[str, Any],
        label: str = "",
    ) -> None:
        self.run_id = run_id
        self.dataset_id = dataset_id
        self.usecase = usecase
        self.mapping = mapping
        self.options = options
        self.label = label

        self.status = "running"          # running | succeeded | failed
        self.error: str | None = None
        self.result: runner.RunResult | None = None

        self.events: list[dict] = []
        self._lock = threading.Lock()
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._terminal_event: dict | None = None

        self.finished = threading.Event()
        self.started_at = time.time()
        self.finished_at: float | None = None

    # -- event plumbing ------------------------------------------------------
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _fanout(self, event: dict, subscribers: list[asyncio.Queue]) -> None:
        loop = self._loop
        if loop is None or not subscribers:
            return
        for queue in subscribers:
            try:
                loop.call_soon_threadsafe(_safe_put, queue, event)
            except RuntimeError:
                pass  # loop already closed; the run is over anyway

    def publish(self, event: dict) -> None:
        """Called from the worker thread for every engine event.

        A terminal ``done`` event is *buffered*, not broadcast. The engine emits it
        from its own thread a moment before the job manager has stored the finished
        result; broadcasting it immediately would let a client react to ``done`` and
        fetch a result that does not exist yet, rendering an empty page. It is
        released by :meth:`finish_stream` once the result is committed.
        """
        is_terminal = event.get("phase") == "done"
        with self._lock:
            self.events.append(event)
            if len(self.events) > _EVENT_BUFFER:
                del self.events[: len(self.events) - _EVENT_BUFFER]
            if is_terminal:
                self._terminal_event = event
                return
            subscribers = list(self._subscribers)
        self._fanout(event, subscribers)

    def finish_stream(self, status: str) -> None:
        """Release the buffered terminal event, or synthesise one, and broadcast it."""
        with self._lock:
            terminal = self._terminal_event
            self._terminal_event = None
            if terminal is None:
                terminal = {
                    "seq": len(self.events) + 1,
                    "ts": int(time.time() * 1000),
                    "phase": "done",
                    "level": "success" if status == "succeeded" else "error",
                    "title": "All done" if status == "succeeded" else "Stopped early",
                    "detail": self.error or "",
                    "progress": 1.0,
                    "model": None,
                    "metrics": None,
                    "elapsed_ms": self.elapsed_ms,
                    "payload": {"status": status},
                }
                self.events.append(terminal)
            subscribers = list(self._subscribers)
        self._fanout(terminal, subscribers)

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_DEPTH)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.events)

    def last_event(self) -> dict | None:
        with self._lock:
            return self.events[-1] if self.events else None

    @property
    def elapsed_ms(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.time()
        return round((end - self.started_at) * 1000.0, 1)

    # -- lifecycle -----------------------------------------------------------
    def mark_finished(self, status: str, error: str | None = None) -> None:
        self.status = status
        self.error = error
        self.finished_at = time.time()
        self.finished.set()

    def summary(self) -> dict:
        """Small status payload for polling clients."""
        last = self.last_event() or {}
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "usecase": self.usecase,
            "dataset_id": self.dataset_id,
            "event_count": len(self.events),
            "elapsed_ms": self.elapsed_ms,
            "progress": last.get("progress", 0.0),
            "phase": last.get("phase", "queued"),
            "title": last.get("title", ""),
            "detail": last.get("detail", ""),
        }

    def result_dict(self) -> dict | None:
        return self.result.to_dict() if self.result is not None else None


class JobManager:
    """Owns every run in this server process."""

    def __init__(self) -> None:
        self._jobs: dict[str, RunJob] = {}
        self._lock = threading.Lock()

    # -- introspection -------------------------------------------------------
    def get(self, run_id: str) -> RunJob | None:
        with self._lock:
            return self._jobs.get(run_id)

    def require(self, run_id: str) -> RunJob:
        job = self.get(run_id)
        if job is None:
            raise KeyError(
                f"No analysis run called {run_id!r}. It may have finished before "
                f"this server started."
            )
        return job

    def list(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [j.summary() for j in sorted(jobs, key=lambda j: j.started_at, reverse=True)]

    # -- execution -----------------------------------------------------------
    def start(
        self,
        dataset_id: str,
        usecase: str,
        mapping: dict[str, list[str]],
        options: dict[str, Any] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        label: str = "",
    ) -> RunJob:
        options = dict(options or {})
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        job = RunJob(run_id, dataset_id, usecase, mapping, options, label=label)
        if loop is not None:
            job.attach_loop(loop)

        with self._lock:
            self._jobs[run_id] = job

        thread = threading.Thread(
            target=self._execute, args=(job,), name=f"mlarena-run-{run_id}", daemon=True
        )
        thread.start()
        return job

    def _execute(self, job: RunJob) -> None:
        status, error = "succeeded", None
        try:
            result = runner.run(
                job.dataset_id,
                job.usecase,
                job.mapping,
                job.options,
                emit=job.publish,
                run_id=job.run_id,
            )
            job.result = result
            if result.status == "succeeded":
                status, error = "succeeded", None
            else:
                status, error = "failed", result.error or "The analysis did not complete."
        except Exception as exc:  # noqa: BLE001 - a bad run must never kill the server
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            job.publish(
                {
                    "seq": len(job.events) + 1,
                    "ts": int(time.time() * 1000),
                    "phase": "error",
                    "level": "error",
                    "title": "We could not finish this analysis",
                    "detail": (
                        f"{error}. Nothing was uploaded anywhere — you can adjust your "
                        f"column choices and try again."
                    ),
                    "progress": 1.0,
                    "model": None,
                    "metrics": None,
                    "elapsed_ms": job.elapsed_ms,
                    "payload": {"traceback": traceback.format_exc()[-4000:]},
                }
            )
        finally:
            # Order matters: commit the result and persist it *before* the terminal
            # event is released, so a client that reacts to `done` immediately finds
            # a complete result waiting for it.
            job.mark_finished(status, error)
            self._persist(job)
            job.finish_stream(status)

    def _persist(self, job: RunJob) -> None:
        """Write the finished run to disk so exports and reloads keep working."""
        try:
            config.ensure_dirs()
            run_dir = Path(config.RUNS_DIR) / job.run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": job.run_id,
                "status": job.status,
                "error": job.error,
                "usecase": job.usecase,
                "dataset_id": job.dataset_id,
                "mapping": job.mapping,
                "options": job.options,
                "elapsed_ms": job.elapsed_ms,
                "result": job.result_dict(),
                "events": job.events,
            }
            (run_dir / "result.json").write_text(
                json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8"
            )
        except Exception:
            pass  # persistence is a convenience, never a hard requirement


# Module-level singleton used by the API layer.
jobs = JobManager()
