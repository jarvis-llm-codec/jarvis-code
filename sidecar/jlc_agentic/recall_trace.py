"""Env-gated recall/preflight trace sink.

Set JARVIS_RECALL_TRACE to a JSONL path to enable. The helpers are intentionally
best-effort: tracing must never break chat, recall, or provider calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_TRACE_LOCK = threading.Lock()
_FALSE_VALUES = {"", "0", "false", "off", "no"}


def trace_path() -> Path | None:
    raw = os.environ.get("JARVIS_RECALL_TRACE")
    if raw is None:
        return None
    value = raw.strip()
    if value.lower() in _FALSE_VALUES:
        return None
    if value == "1":
        return Path.cwd() / "jarvis_recall_trace.jsonl"
    return Path(value).expanduser()


def enabled() -> bool:
    return trace_path() is not None


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def preview_text(text: str, limit: int = 240) -> str:
    clean = " ".join(str(text or "").replace("\r", " ").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def emit(event: str, **payload: Any) -> None:
    path = trace_path()
    if path is None:
        return
    record: dict[str, Any] = {
        "event": event,
        "ts": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    record.update(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with _TRACE_LOCK:
            with path.open("a", encoding="utf-8", newline="") as fh:
                fh.write(line)
    except Exception:
        return


def query_fields(query: str, *, limit: int = 240) -> dict[str, str]:
    return {
        "query_sha256": sha256_text(query),
        "query_preview": preview_text(query, limit=limit),
    }
