"""LLM client shim that wraps a ProviderRouter to expose the
stream_chat_completions interface the AgenticLoop / ChatTurn / subagent paths
expect.

Originally lived inline in jlc_agentic.agentic.chat_turn; moved here in W2.6
so providers.get_llm() can return a router-backed adapter for any role and
every caller (chat, subagent, encoder, bench, jlc_agentic_coder) gets OAuth /
KeyPool / fallback for free.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterator
from typing import Any


class LLMRouterAdapter:
    """Wraps ProviderRouter.call(alias, messages) so AgenticLoop sees the
    same stream_chat_completions contract as the legacy OpenAICompatible
    adapter. router.call() is non-streaming; we synthesize a single chunk."""

    def __init__(self, router: Any, alias: str, extra_kwargs: dict[str, Any] | None = None) -> None:
        self.router = router
        self.alias = alias
        self.llm_meta: dict[str, Any] | None = None
        # Optional kwargs forwarded to every router stream_call / call (e.g.
        # `{"reasoning_effort": "none"}` for the encoder so reasoning models
        # like nemotron-3-nano:30b run in instruct mode).
        self.extra_kwargs = dict(extra_kwargs) if extra_kwargs else {}

    async def close(self) -> None:
        return None

    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        on_chunk: Any = None,
        **kwargs: Any,
    ) -> str:
        """Encoder-facing compatibility shim.

        JLCEncoder historically called LLMClient.chat(system=..., user=...).
        The router path exposes stream_chat_completions(), so adapt the call
        without changing encoder encode/compression logic.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        parts: list[str] = []
        if on_chunk is not None:
            kwargs.setdefault("allow_partial_stream", True)
        use_stream = on_chunk is not None or _requires_stream_transport(self.alias)
        try:
            for chunk in self.stream_chat_completions(
                messages=messages,
                stream=use_stream,
                max_tokens=max_tokens,
                **kwargs,
            ):
                delta = _field(_first_choice(chunk), "delta") or {}
                reasoning = _normalize_reasoning(
                    _field(delta, "reasoning_content") or _field(delta, "reasoning")
                )
                content = _field(delta, "content") or ""
                if reasoning and on_chunk is not None:
                    self._emit_chunk(on_chunk, str(reasoning), "reasoning")
                if content:
                    text = str(content)
                    parts.append(text)
                    if on_chunk is not None:
                        self._emit_chunk(on_chunk, text, "content")
        except Exception as exc:
            if parts and _is_stream_interrupted(exc):
                print(
                    f"[jlc:encoder-stream] upstream stream ended early after {len(parts)} chunks; using partial encoder output",
                    file=sys.stderr,
                )
            else:
                raise
        return _strip_markdown_fences("".join(parts))

    @staticmethod
    def _emit_chunk(on_chunk: Any, text: str, kind: str) -> None:
        try:
            on_chunk(text, kind)
        except TypeError:
            try:
                on_chunk(text)
            except Exception:
                pass
        except Exception:
            pass

    def stream_chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        parallel_tool_calls: bool = True,
        stream: bool = True,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        call_kwargs = dict(self.extra_kwargs)
        call_kwargs.update(kwargs)
        if tools:
            call_kwargs["tools"] = tools
            call_kwargs["parallel_tool_calls"] = parallel_tool_calls

        if stream:
            stream_call = getattr(self.router, "stream_call", None)
            if callable(stream_call):
                for chunk in stream_call(self.alias, messages, **call_kwargs):
                    yield chunk
                self.llm_meta = getattr(self.router, "last_stream_meta", None)
                return
            # Router lacks stream_call (e.g., legacy mocks in tests) -- fall
            # through to the single-shot replay path so behavior degrades
            # gracefully instead of erroring.

        call_kwargs["stream"] = False
        result = self.router.call(self.alias, messages, **call_kwargs)
        self.llm_meta = result.get("llm_meta")
        yield _response_to_chunk(result.get("response"))


def _response_to_chunk(response: Any) -> dict[str, Any]:
    choice = _first_choice(response)
    message = _field(choice, "message") or {}
    delta: dict[str, Any] = {}
    content = _field(message, "content")
    if content is not None:
        delta["content"] = content
    # Ollama Cloud reasoning models (gpt-oss:120b, deepseek-v4-pro, qwen3-next:80b)
    # surface internal reasoning via message.reasoning while content is empty.
    # AgenticLoop reads delta.reasoning_content/reasoning and promotes it to
    # final via the final_from_reasoning fallback when content stays silent.
    reasoning = _normalize_reasoning(
        _field(message, "reasoning_content") or _field(message, "reasoning")
    )
    if reasoning:
        delta["reasoning"] = reasoning
    tool_calls = _field(message, "tool_calls")
    if tool_calls:
        delta["tool_calls"] = [_tool_call_delta(i, call) for i, call in enumerate(tool_calls)]
    return {"choices": [{"delta": delta, "finish_reason": _field(choice, "finish_reason")}]}


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$", stripped)
    if m:
        return m.group(1).strip()
    return stripped


def _requires_stream_transport(alias: str) -> bool:
    """Some transports require SSE even when callers do not need live chunks."""
    return alias.startswith("openai-codex-")


def _is_stream_interrupted(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "stream interrupted" in text
        or "incomplete chunked read" in text
        or "peer closed connection" in text
    )


def _normalize_reasoning(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "")
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(value)


def _first_choice(response: Any) -> Any:
    choices = _field(response, "choices") or []
    return choices[0] if choices else {}


def _tool_call_delta(index: int, call: Any) -> dict[str, Any]:
    fn = _field(call, "function") or {}
    return {
        "index": index,
        "id": _field(call, "id"),
        "type": _field(call, "type") or "function",
        "function": {
            "name": _field(fn, "name") or "",
            "arguments": _field(fn, "arguments") or "",
        },
    }


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
