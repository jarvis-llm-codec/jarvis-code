"""Prompt assets for jlc_agentic — system prompts and shared policies."""

from __future__ import annotations

from pathlib import Path

from .env_directive import get_env_directive
from .reasoning_policy import POLICY_ENCODER, POLICY_USER_FACING

_CONSTITUTION_PATH = Path(__file__).with_name("jarvis_code_constitution.md")
_CONSTITUTION_CACHE: str | None = None


def get_constitution() -> str:
    """Return the jarvis-code constitution (Principles 0–5).

    The constitution is the shared behavioral contract for chat, subagent,
    and encoder layers. W2.9.16 origin (2026-05-08) — closes the
    fake-CoT-evidence hallucination loop observed in W2.9.15 live.
    Cached after first read; static text is prompt-cache safe.
    """
    global _CONSTITUTION_CACHE
    if _CONSTITUTION_CACHE is None:
        _CONSTITUTION_CACHE = _CONSTITUTION_PATH.read_text(encoding="utf-8")
    return _CONSTITUTION_CACHE


__all__ = [
    "POLICY_USER_FACING",
    "POLICY_ENCODER",
    "get_env_directive",
    "get_constitution",
]
