"""Source-side caps for recalled evidence served to models."""
from __future__ import annotations

import os
from typing import Any

from jlc_agentic.agentic.preflight import count_tokens, trim_to_tokens

DEFAULT_EVIDENCE_TOKENS = 40_000
DEFAULT_FRAGMENT_TOKENS = 4_000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def evidence_token_budget(default: int = DEFAULT_EVIDENCE_TOKENS) -> int:
    return _env_int("JARVIS_RETRIEVED_EVIDENCE_MAX_TOKENS", default)


def fragment_token_budget(total_budget: int | None = None) -> int:
    total = evidence_token_budget() if total_budget is None else max(0, int(total_budget))
    configured = _env_int("JARVIS_RECALL_FRAGMENT_MAX_TOKENS", DEFAULT_FRAGMENT_TOKENS)
    if total <= 0:
        return 0
    return max(1, min(configured, total))


def cap_text(text: str, max_tokens: int) -> tuple[str, int, int, bool]:
    original_tokens = count_tokens(text or "")
    if max_tokens <= 0:
        return "", original_tokens, 0, original_tokens > 0
    if original_tokens <= max_tokens:
        return text or "", original_tokens, original_tokens, False
    target = max_tokens
    trimmed = ""
    trimmed_tokens = 0
    while target > 0:
        trimmed = trim_to_tokens(text or "", target).rstrip()
        trimmed_tokens = count_tokens(trimmed)
        if trimmed_tokens <= max_tokens:
            break
        target = max(0, target - max(1, trimmed_tokens - max_tokens))
    if target <= 0 and trimmed_tokens > max_tokens:
        trimmed = ""
        trimmed_tokens = 0
    return trimmed, original_tokens, trimmed_tokens, True


def _cap_field(value: Any, max_tokens: int) -> tuple[str, dict[str, Any]]:
    capped, original, served, truncated = cap_text(str(value or ""), max_tokens)
    return capped, {
        "original_tokens": original,
        "served_tokens": served,
        "truncated": truncated,
    }


def cap_fragment_fields(
    fragment: dict[str, Any],
    *,
    remaining_tokens: int,
    per_fragment_tokens: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    budget = max(0, min(remaining_tokens, per_fragment_tokens))
    original_user = str(fragment.get("user", "") or "")
    original_assistant = str(fragment.get("assistant", "") or "")
    original_tokens = count_tokens(original_user) + count_tokens(original_assistant)
    if budget <= 0:
        return None, {
            "turn": fragment.get("turn"),
            "original_tokens": original_tokens,
            "served_tokens": 0,
            "truncated": original_tokens > 0,
            "dropped": True,
        }

    # Reserve half for each field first, then give the assistant any leftover.
    user_budget = budget // 2
    assistant_budget = budget - user_budget
    capped_user, user_meta = _cap_field(original_user, user_budget)
    leftover = max(0, user_budget - user_meta["served_tokens"])
    capped_assistant, assistant_meta = _cap_field(original_assistant, assistant_budget + leftover)

    served_tokens = int(user_meta["served_tokens"]) + int(assistant_meta["served_tokens"])
    capped = dict(fragment)
    capped["user"] = capped_user
    capped["assistant"] = capped_assistant
    if "snippet" in capped and not original_assistant:
        capped["snippet"] = capped_user
    truncated = bool(user_meta["truncated"] or assistant_meta["truncated"] or original_tokens > served_tokens)
    if truncated:
        capped["capped"] = True
        capped["original_tokens_est"] = original_tokens
        capped["served_tokens_est"] = served_tokens
    return capped, {
        "turn": fragment.get("turn"),
        "original_tokens": original_tokens,
        "served_tokens": served_tokens,
        "truncated": truncated,
        "dropped": False,
    }


def cap_fragments(
    fragments: list[dict[str, Any]],
    *,
    total_budget_tokens: int | None = None,
    per_fragment_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_budget = evidence_token_budget() if total_budget_tokens is None else max(0, int(total_budget_tokens))
    fragment_budget = fragment_token_budget(total_budget) if per_fragment_tokens is None else max(0, int(per_fragment_tokens))
    remaining = total_budget
    capped: list[dict[str, Any]] = []
    fragment_meta: list[dict[str, Any]] = []
    original_tokens = 0
    served_tokens = 0
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        capped_fragment, meta = cap_fragment_fields(
            fragment,
            remaining_tokens=remaining,
            per_fragment_tokens=fragment_budget,
        )
        fragment_meta.append(meta)
        original_tokens += int(meta.get("original_tokens") or 0)
        served_tokens += int(meta.get("served_tokens") or 0)
        remaining = max(0, remaining - int(meta.get("served_tokens") or 0))
        if capped_fragment is not None:
            capped.append(capped_fragment)
    truncated = bool(
        original_tokens > served_tokens
        or len(capped) < len([fragment for fragment in fragments if isinstance(fragment, dict)])
        or any(meta.get("truncated") or meta.get("dropped") for meta in fragment_meta)
    )
    return capped, {
        "served_policy": "capped" if truncated else "full",
        "budget_tokens": total_budget,
        "per_fragment_budget_tokens": fragment_budget,
        "original_tokens": original_tokens,
        "served_tokens": served_tokens,
        "truncated_tokens": max(0, original_tokens - served_tokens),
        "truncated": truncated,
        "fragments": fragment_meta,
    }


def cap_block(text: str, *, total_budget_tokens: int | None = None) -> tuple[str, dict[str, Any]]:
    budget = evidence_token_budget() if total_budget_tokens is None else max(0, int(total_budget_tokens))
    capped, original, served, truncated = cap_text(text or "", budget)
    return capped, {
        "served_policy": "capped" if truncated else "full",
        "budget_tokens": budget,
        "original_tokens": original,
        "served_tokens": served,
        "truncated_tokens": max(0, original - served),
        "truncated": truncated,
    }
