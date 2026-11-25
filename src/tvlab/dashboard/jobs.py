"""Background jobs for the dashboard.

A 500-name NSE scan is 1,500 HTTP requests and takes 80-140 seconds. Doing that
inside the request that renders a page means a blank browser for two minutes, and
a user who cannot tell a slow scan from a hung one. So the dashboard starts a job,
returns an id immediately, and the page polls for progress.

This is deliberately small - a single-worker executor and a dict - because the
constraint that matters is not throughput but *politeness*: two universes being
scanned at once against NSE and Yahoo is how a working dashboard starts getting
403s. One scan at a time, and a second request for the same kind of work joins the
one already running rather than stacking another behind it.

Job state lives in the process. A restart loses the list, which is correct: these
are reads of the market, not records of it. The durable record is the daily
journal, written by ``tvlab.screener.daily``.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

#: ``queued`` -> ``running`` -> ``done`` | ``error``
TERMINAL = ("done", "error")


@dataclass
class Job:
    id: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    state: str = "queued"
    message: str = ""
    total: int = 0
    done: int = 0
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    error: Optional[str] = None
    result: Any = None

    @property
    def elapsed(self) -> float:
        end = self.finished if self.finished is not None else time.time()
        return max(0.0, end - (self.started or self.created))

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "state": self.state,
            "message": self.message,
            "total": int(self.total),
            "done": int(self.done),
            "fraction": (self.done / self.total) if self.total else None,
            "elapsed_seconds": round(self.elapsed, 1),
            "error": self.error,
            "params": dict(self.params),
        }
        if include_result:
            out["result"] = self.result
        return out


class Progress:
    """The handle a job body updates so the browser can show where it is.

    Thread-safe because ``scan_universe(workers=8)`` calls it from eight threads.
    Every method is best-effort: a display update must never be the reason a scan
    of 500 names fails.
    """

    def __init__(self, job: Job, lock: threading.Lock):
        self._job = job
        self._lock = lock

    def note(self, text: str) -> None:
        with self._lock:
            self._job.message = str(text)

    def set_total(self, total: int) -> None:
        with self._lock:
            self._job.total = int(total)
            self._job.done = 0

    def advance(self, done: int, total: Optional[int] = None) -> None:
        with self._lock:
            if total is not None:
                self._job.total = int(total)
            self._job.done = int(done)


class JobRunner:
    """One executor, a bounded history, and no duplicate work of the same kind."""

    def __init__(self, max_workers: int = 1, keep: int = 20):
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="tvlab-job")
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._keep = keep

    # ------------------------------------------------------------------ submit
    def submit(self, kind: str, fn: Callable[[Progress], Any],
               params: Optional[Dict[str, Any]] = None) -> Job:
        """Start ``fn(progress)`` in the background and return its job."""
        with self._lock:
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=dict(params or {}))
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim()
        progress = Progress(job, self._lock)
        self._pool.submit(self._run, job, fn, progress)
        return job

    def _run(self, job: Job, fn: Callable[[Progress], Any], progress: Progress) -> None:
        with self._lock:
            if job.state == "queued":
                job.state = "running"
                job.started = time.time()
        try:
            result = fn(progress)
        except Exception as exc:  # noqa: BLE001 - the error becomes the job's answer
            log.warning("job %s (%s) failed: %s", job.id, job.kind, exc)
            with self._lock:
                job.state = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = time.time()
            return
        with self._lock:
            job.result = result
            job.state = "done"
            job.finished = time.time()
            if not job.message:
                job.message = "done"

    # ------------------------------------------------------------------- query
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self, kind: Optional[str] = None) -> Optional[Job]:
        """The queued/running job of this kind, if any.

        A dashboard that submits a scan every time a button is pressed would
        otherwise queue five identical scans behind each other, and the fifth
        would describe a market fifteen minutes stale by the time it printed.
        """
        with self._lock:
            for jid in reversed(self._order):
                job = self._jobs.get(jid)
                if job is None or job.state in TERMINAL:
                    continue
                if kind is None or job.kind == kind:
                    return job
        return None

    def recent(self, limit: int = 10) -> List[Job]:
        with self._lock:
            return [self._jobs[j] for j in reversed(self._order[-limit:]) if j in self._jobs]

    def _trim(self) -> None:
        """Drop old *finished* jobs; never drop a running one."""
        while len(self._order) > self._keep:
            for i, jid in enumerate(self._order):
                job = self._jobs.get(jid)
                if job is not None and job.state not in TERMINAL:
                    continue
                self._order.pop(i)
                self._jobs.pop(jid, None)
                break
            else:                      # everything left is still running
                return

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
