"""Provider payload preflight for the agentic chat path."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from jlc_agentic import recall_trace

_tiktoken_enc = None
_RECALL_START = "[Recalled context]\n"
_RECALL_END = "\n---RECALL_END---\n"
_RECENT_RE = re.compile(r"<recent_window>.*?</recent_window>", re.DOTALL)


class ProviderPreflightError(RuntimeError):
    """Raised before provider I/O when the payload remains over budget."""


@dataclass
class PreflightResult:
    messages: list[dict[str, Any]]
    estimated_input_tokens: int
    budget_tokens: int
    action: str
    section_tokens: dict[str, int]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _get_tiktoken_enc():
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken

            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = False
    return _tiktoken_enc


def count_tokens(text: str) -> int:
    enc = _get_tiktoken_enc()
    if enc:
        try:
            return len(enc.encode(text or ""))
        except Exception:
            pass
    return len((text or "").split())


def trim_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    enc = _get_tiktoken_enc()
    if enc:
        try:
            toks = enc.encode(text or "")
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens]).rstrip()
        except Exception:
            pass
    # Conservative fallback: English-ish chars/token approximation.
    limit = max(0, max_tokens * 4)
    if len(text or "") <= limit:
        return text
    return (text or "")[:limit].rstrip()


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _message_text(message: dict[str, Any]) -> str:
    return _content_text(message.get("content")) if isinstance(message, dict) else ""


def _tool_schema_tokens(tools: list[dict[str, Any]] | None) -> int:
    if not tools:
        return 0
    try:
        return count_tokens(json.dumps(tools, ensure_ascii=False, sort_keys=True, default=str))
    except Exception:
        return count_tokens(str(tools))


def _messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(count_tokens(_message_text(message)) for message in messages or [])


def _model_context_window(model: str | None) -> int:
    override = _env_int("JARVIS_PROVIDER_CONTEXT_WINDOW", 0)
    if override > 0:
        return override
    name = str(model or "").casefold()
    if "gpt-5.5" in name or "gpt-5.4" in name or "openai-codex" in name:
        return 267_000
    if "gpt-5" in name:
        return 400_000
    if "claude" in name:
        return 200_000
    return 267_000


def _budget_tokens(model: str | None) -> int:
    override = _env_int("JARVIS_PROVIDER_PREFLIGHT_MAX_TOKENS", 0)
    if override > 0:
        return override
    fraction = _env_float("JARVIS_PROVIDER_PREFLIGHT_CONTEXT_FRACTION", 0.70)
    return max(1, int(_model_context_window(model) * fraction))


def _extract_recall_block(text: str) -> tuple[str, str, str] | None:
    start = text.find(_RECALL_START)
    if start < 0:
        return None
    content_start = start + len(_RECALL_START)
    end = text.find(_RECALL_END, content_start)
    if end < 0:
        return None
    prefix = text[:start]
    recall = text[content_start:end]
    suffix = text[end + len(_RECALL_END) :]
    return prefix, recall, suffix


def _replace_recall_block(message: dict[str, Any], recall_text: str) -> dict[str, Any]:
    text = _message_text(message)
    split = _extract_recall_block(text)
    if split is None:
        return message
    prefix, _old, suffix = split
    if recall_text:
        marker = (
            _RECALL_START
            + recall_text.rstrip()
            + "\n[preflight: recalled context trimmed to fit provider budget]"
            + _RECALL_END
        )
    else:
        marker = (
            _RECALL_START
            + "[preflight: recalled context omitted to fit provider budget]"
            + _RECALL_END
        )
    return {**message, "content": prefix + marker + suffix}


def _trim_recall(messages: list[dict[str, Any]], max_recall_tokens: int) -> tuple[list[dict[str, Any]], str]:
    action = "allow"
    out: list[dict[str, Any]] = []
    for message in messages:
        text = _message_text(message)
        split = _extract_recall_block(text)
        if split is None:
            out.append(message)
            continue
        _prefix, recall, _suffix = split
        recall_tokens = count_tokens(recall)
        if recall_tokens <= max_recall_tokens:
            out.append(message)
            continue
        trimmed = trim_to_tokens(recall, max_recall_tokens)
        out.append(_replace_recall_block(message, trimmed))
        action = "trim_recall"
    return out, action


def _drop_recall(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    dropped = False
    out: list[dict[str, Any]] = []
    for message in messages:
        text = _message_text(message)
        if _extract_recall_block(text) is None:
            out.append(message)
            continue
        out.append(_replace_recall_block(message, ""))
        dropped = True
    return out, dropped


def section_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> dict[str, int]:
    sections = {
        "system": 0,
        "jhb_project_system": 0,
        "retrieved_prior_turns": 0,
        "recent_window": 0,
        "tool_results": 0,
        "assistant_history": 0,
        "current_user": 0,
        "other_user": 0,
        "tools": _tool_schema_tokens(tools),
    }
    user_indices = [i for i, message in enumerate(messages or []) if message.get("role") == "user"]
    last_user = user_indices[-1] if user_indices else None
    for i, message in enumerate(messages or []):
        role = str(message.get("role") or "")
        text = _message_text(message)
        toks = count_tokens(text)
        if role == "system":
            sections["system"] += toks
            if "---JHB_END---" in text or "---PROJECT_END---" in text:
                sections["jhb_project_system"] += toks
        elif role == "tool":
            sections["tool_results"] += toks
        elif role == "assistant":
            sections["assistant_history"] += toks
        elif role == "user" and i == last_user:
            sections["current_user"] += toks
        elif role == "user":
            sections["other_user"] += toks
        split = _extract_recall_block(text)
        if split is not None:
            sections["retrieved_prior_turns"] += count_tokens(split[1])
        for match in _RECENT_RE.finditer(text):
            sections["recent_window"] += count_tokens(match.group(0))
    return sections


def preflight_messages(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    model: str | None = None,
    provider_call_index: int | None = None,
    surface: str = "agentic_loop",
) -> PreflightResult:
    if os.environ.get("JARVIS_PROVIDER_PREFLIGHT", "1").strip().lower() in {"0", "false", "off", "no"}:
        total = _messages_tokens(messages) + _tool_schema_tokens(tools)
        return PreflightResult(
            messages=list(messages),
            estimated_input_tokens=total,
            budget_tokens=_budget_tokens(model),
            action="disabled",
            section_tokens=section_tokens(messages, tools),
        )

    budget = _budget_tokens(model)
    evidence_budget = _env_int("JARVIS_RETRIEVED_EVIDENCE_MAX_TOKENS", 40_000)
    candidate = list(messages)
    action = "allow"

    sections = section_tokens(candidate, tools)
    total = _messages_tokens(candidate) + sections["tools"]
    if sections["retrieved_prior_turns"] > evidence_budget:
        candidate, action = _trim_recall(candidate, evidence_budget)
        sections = section_tokens(candidate, tools)
        total = _messages_tokens(candidate) + sections["tools"]

    if total > budget:
        candidate, dropped = _drop_recall(candidate)
        if dropped:
            action = "drop_recall"
            sections = section_tokens(candidate, tools)
            total = _messages_tokens(candidate) + sections["tools"]

    if total > budget:
        action = "block_before_provider"
        recall_trace.emit(
            "provider_preflight",
            surface=surface,
            provider_call_index=provider_call_index,
            model=model,
            estimated_input_tokens=total,
            budget_tokens=budget,
            section_tokens=sections,
            action=action,
        )
        raise ProviderPreflightError(
            "JARVIS provider preflight blocked an over-budget request before provider I/O "
            f"(estimated_input_tokens={total}, budget_tokens={budget})"
        )

    recall_trace.emit(
        "provider_preflight",
        surface=surface,
        provider_call_index=provider_call_index,
        model=model,
        estimated_input_tokens=total,
        budget_tokens=budget,
        section_tokens=sections,
        action=action,
    )
    return PreflightResult(
        messages=candidate,
        estimated_input_tokens=total,
        budget_tokens=budget,
        action=action,
        section_tokens=sections,
    )
