"""In-memory job store and thread pool runner.

Jobs are stored in a module-level dict — restart loses them. Fine for MVP;
swap for Redis/Celery when persistence is required.
"""
from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from queue import Queue
from typing import Any, Callable


_store: dict[str, "JobRecord"] = {}
_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=2)  # cap concurrent GPU jobs


@dataclass
class JobRecord:
    job_id: str
    status: str = "pending"          # pending | running | done | failed
    current_stage: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    partner_id: str | None = None
    error: str | None = None
    result: dict | None = None
    _events: Queue = field(default_factory=Queue)  # SSE events; sentinel = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "partner_id": self.partner_id,
            "error": self.error,
            "result": self.result,
        }


def create_job(partner_id: str | None = None) -> JobRecord:
    record = JobRecord(job_id=str(uuid.uuid4()), partner_id=partner_id)
    with _lock:
        _store[record.job_id] = record
    return record


def get_job(job_id: str) -> JobRecord | None:
    with _lock:
        return _store.get(job_id)


def submit(record: JobRecord, fn: Callable, **kwargs: Any) -> None:
    _pool.submit(_run, record, fn, kwargs)


def _run(record: JobRecord, fn: Callable, kwargs: dict) -> None:
    record.status = "running"
    record._events.put({"event": "status", "data": "running"})
    try:
        result = fn(record, **kwargs)
        record.status = "done"
        record.result = result
        record.completed_at = datetime.now(timezone.utc).isoformat()
        record._events.put({"event": "done", "data": result})
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        record.completed_at = datetime.now(timezone.utc).isoformat()
        record._events.put({"event": "error", "data": str(exc)})
    finally:
        record._events.put(None)  # sentinel — tells the SSE stream to close
