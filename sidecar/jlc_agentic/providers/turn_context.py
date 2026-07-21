"""Per-turn context for adapters that need turn-scoped state.

2026-06-15. ``get_llm(role)`` builds and caches a provider adapter keyed only by
(role, config_path); it never receives the per-turn conv_id / active project /
host retriever. Most adapters do not need them, but the Agent SDK adapter
(anthropic_agent_sdk) does — to set the SDK ``cwd`` and to back the bridged
``recall_turns`` memory tool from the same JHB store the host writes to.

ChatTurn.run sets this just before invoking the AgenticLoop and clears it after,
on the same worker thread the adapter's ``stream_chat_completions`` runs on.
threading.local (not a contextvar) because the loop runs synchronously on a
worker thread; the adapter reads it on that same thread, then captures the values
into closures (it must NOT be read from the SDK's own loop thread, where the tls
would be empty).
"""
from __future__ import annotations

import threading
from typing import Any

_tls = threading.local()


def set(  # noqa: A001 - module-level verb mirrors threading.local idiom
    *,
    conv_id: str | None = None,
    project_root: str | None = None,
    retriever: Any = None,
    reasoning_effort: str | None = None,
    second_eyes_phase: str | None = None,
    stream_text_deltas: bool | None = None,
    suppress_widget_tool_calls: bool | None = None,
) -> None:
    _tls.value = {
        "conv_id": conv_id,
        "project_root": project_root,
        "retriever": retriever,
        "reasoning_effort": reasoning_effort,
        "second_eyes_phase": second_eyes_phase,
        "stream_text_deltas": stream_text_deltas,
        "suppress_widget_tool_calls": suppress_widget_tool_calls,
    }


def get() -> dict[str, Any]:
    return getattr(_tls, "value", None) or {}


def clear() -> None:
    if hasattr(_tls, "value"):
        _tls.value = {}
