from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

_DEFAULT_TIMEOUT_SECONDS = 300.0
_MAX_TIMEOUT_SECONDS = 600.0
_ANSWER_RETENTION_SECONDS = 60.0
_PAIR8_RE = re.compile(r"^[A-Za-z0-9]{8}$")


def _logs_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def _control_bridge_debug_on() -> bool:
    """Enabled by EITHER the env var JARVIS_CONTROL_BRIDGE_DEBUG=1 OR the presence of a
    flag file sidecar/logs/.control_bridge_debug -- the file path works regardless of
    how the sidecar process inherits env (launcher-independent)."""
    if os.environ.get("JARVIS_CONTROL_BRIDGE_DEBUG"):
        return True
    try:
        return os.path.exists(os.path.join(_logs_dir(), ".control_bridge_debug"))
    except Exception:
        return False


def _debug_log(event: str, **fields: Any) -> None:
    """Gated control-bridge tracer (see _control_bridge_debug_on). Zero overhead when
    off. Captures enqueue / poll / resolve WITH pair8 so a live ask_user repro shows
    exactly where the request and the poll diverge -- pair8 mismatch (enqueue to X,
    poll for Y), request not visible (process/dict mismatch), or no poll at all."""
    if not _control_bridge_debug_on():
        return
    try:
        log_dir = _logs_dir()
        os.makedirs(log_dir, exist_ok=True)
        rec = {"ts": time.time(), "pid": os.getpid(), "event": event, **fields}
        with open(os.path.join(log_dir, "control_bridge_debug.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # pragma: no cover - tracing must never break the bridge
        pass


@dataclass
class _ControlRequest:
    id: str
    kind: str
    to_window: str
    payload: dict[str, Any]
    created_at: float
    deadline_at: float
    status: str = "pending"
    result: dict[str, Any] | None = None
    answered_at: float | None = None
    event: threading.Event = field(default_factory=threading.Event)


_requests: dict[str, _ControlRequest] = {}
_lock = threading.Lock()


def _coerce_pair8(value: str | None) -> str:
    pair8 = str(value or "").strip()[:8]
    if not _PAIR8_RE.match(pair8):
        raise ValueError("valid target window pair8 is required")
    return pair8


def _coerce_timeout(value: float | int | None) -> float:
    try:
        timeout = float(value if value is not None else _DEFAULT_TIMEOUT_SECONDS)
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_SECONDS
    if timeout <= 0:
        return _DEFAULT_TIMEOUT_SECONDS
    return min(timeout, _MAX_TIMEOUT_SECONDS)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


def _summary(req: _ControlRequest) -> dict[str, Any]:
    return {
        "id": req.id,
        "kind": req.kind,
        "to_window": req.to_window,
        "payload": req.payload,
        "created_at": _iso(req.created_at),
        "deadline_at": _iso(req.deadline_at),
    }


def _cleanup_locked(now: float) -> None:
    for request_id, req in list(_requests.items()):
        if req.status == "pending" and now >= req.deadline_at:
            req.status = "expired"
            req.result = {
                "ok": False,
                "error": f"{req.kind} control request timed out",
                "request_id": req.id,
            }
            req.answered_at = now
            req.event.set()
        if req.status != "pending" and req.answered_at is not None:
            if now - req.answered_at >= _ANSWER_RETENTION_SECONDS:
                _requests.pop(request_id, None)


def submit_request(
    *,
    kind: str,
    to_window: str,
    payload: dict[str, Any] | None,
    timeout_seconds: float | int | None = None,
) -> dict[str, Any]:
    clean_kind = str(kind or "").strip()
    if not clean_kind:
        raise ValueError("control request kind is required")
    target = _coerce_pair8(to_window)
    request_payload = dict(payload or {})
    timeout = _coerce_timeout(timeout_seconds)
    now = time.monotonic()
    req = _ControlRequest(
        id=uuid.uuid4().hex[:16],
        kind=clean_kind,
        to_window=target,
        payload=request_payload,
        created_at=now,
        deadline_at=now + timeout,
    )
    with _lock:
        _cleanup_locked(now)
        _requests[req.id] = req
    _debug_log("enqueue", id=req.id, kind=clean_kind, to_window=target, timeout=timeout)

    answered = req.event.wait(timeout)
    if not answered:
        with _lock:
            current = _requests.get(req.id)
            if current is not None and current.status == "pending":
                current.status = "expired"
                current.result = {
                    "ok": False,
                    "error": f"{clean_kind} control request timed out",
                    "request_id": req.id,
                }
                current.answered_at = time.monotonic()
                current.event.set()
            result = current.result if current is not None else None
    else:
        with _lock:
            current = _requests.get(req.id)
            result = current.result if current is not None else None

    _debug_log(
        "resolve",
        id=req.id,
        kind=clean_kind,
        answered=answered,
        result_ok=(result or {}).get("ok") if isinstance(result, dict) else None,
        result_err=(result or {}).get("error") if isinstance(result, dict) else None,
    )
    if isinstance(result, dict):
        return result
    return {"ok": False, "error": f"{clean_kind} control request failed", "request_id": req.id}


def pending_requests(*, to_window: str, limit: int = 10) -> list[dict[str, Any]]:
    try:
        target = _coerce_pair8(to_window)
    except ValueError:
        _debug_log("poll_bad_pair8", raw_to_window=to_window)
        raise
    safe_limit = max(1, min(int(limit or 10), 50))
    now = time.monotonic()
    with _lock:
        _cleanup_locked(now)
        all_pending = [
            req
            for req in sorted(_requests.values(), key=lambda item: item.created_at)
            if req.status == "pending"
        ]
        pending = [_summary(req) for req in all_pending if req.to_window == target]
    # Killer diagnostic: the poll's query pair8 vs EVERY pending request's to_window.
    # If an ask_user is pending under a DIFFERENT pair8 than the poll queries -> the
    # pairing mismatch is proven; if none are pending -> the enqueue never reached this
    # process; if matched>0 but no modal -> the break is pi-side (runAskUserDialog).
    _debug_log(
        "poll",
        query_to_window=target,
        matched=len(pending),
        all_pending_to_windows=[req.to_window for req in all_pending],
        all_pending_kinds=[req.kind for req in all_pending],
    )
    return pending[:safe_limit]


def answer_request(*, request_id: str, to_window: str, result: dict[str, Any] | None) -> dict[str, Any]:
    target = _coerce_pair8(to_window)
    clean_id = str(request_id or "").strip()
    if not clean_id:
        raise ValueError("request_id is required")
    now = time.monotonic()
    with _lock:
        _cleanup_locked(now)
        req = _requests.get(clean_id)
        if req is None:
            raise KeyError(clean_id)
        if req.to_window != target:
            raise PermissionError("control request belongs to another window")
        if req.status != "pending":
            return {"ok": False, "error": f"control request is already {req.status}", "request_id": req.id}
        req.status = "answered"
        req.result = dict(result or {})
        req.answered_at = now
        req.event.set()
    return {"ok": True, "request_id": clean_id}


def reset_for_tests() -> None:
    with _lock:
        for req in _requests.values():
            if req.status == "pending":
                req.status = "expired"
                req.result = {"ok": False, "error": "control bridge reset", "request_id": req.id}
                req.answered_at = time.monotonic()
                req.event.set()
        _requests.clear()
