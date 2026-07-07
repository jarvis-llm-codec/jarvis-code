"""Main internal agentic loop."""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Callable

from jlc_agentic import recall_trace

from .preflight import preflight_messages
from .schema import ALL_TOOLS


_tiktoken_enc = None
_RECALL_TOOL_NAMES = {"recall_turn", "recall_turns"}
_RECALL_CONTEXT_MARKERS = (
    "[Recalled context",
    "## Retrieved Prior Turns",
    "[Auto raw recall",
    "<auto_recall",
)
_DENIAL_AUDIT_PATTERNS = (
    "answer: no record",
    "no record",
    "no records",
    "not on record",
    "not recorded",
    "no stored record",
    "no stored evidence",
    "nothing found",
    "nothing logged",
    "cannot find",
    "can't find",
    "could not find",
    "don't have any record",
    "do not have any record",
    "have no record",
    "came up empty",
)
_RECENT_WINDOW_RE = re.compile(r"<recent_window>.*?</recent_window>", re.DOTALL)


def _get_tiktoken_enc():
    """Lazy-load + cache the cl100k_base encoder. get_encoding is several
    hundred ms cold; we only want to pay it once per process."""
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            import tiktoken
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _tiktoken_enc = False  # mark as failed so we stop retrying
    return _tiktoken_enc


def _count_tokens(text: str) -> int:
    """Best-effort tiktoken count; falls back to whitespace split."""
    enc = _get_tiktoken_enc()
    if enc:
        try:
            return len(enc.encode(text or ""))
        except Exception:
            pass
    return len((text or "").split())


def _msg_text(m: dict[str, Any]) -> str:
    c = m.get("content") if isinstance(m, dict) else None
    if isinstance(c, list):
        return "".join(
            p.get("text", "") for p in c
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(c or "")


def _count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_count_tokens(_msg_text(m)) for m in (messages or []))


def _strip_runtime_context(text: str) -> str:
    value = text or ""
    recall_end = "\n---RECALL_END---\n"
    if "[Recalled context]\n" in value and recall_end in value:
        value = value.split(recall_end, 1)[1]
    value = _RECENT_WINDOW_RE.sub("", value)
    return value.strip()


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if isinstance(message, dict) and message.get("role") == "user":
            return _strip_runtime_context(_msg_text(message))
    return ""


def _history_has_recall_context(messages: list[dict[str, Any]]) -> bool:
    for message in messages or []:
        text = _msg_text(message)
        if any(marker in text for marker in _RECALL_CONTEXT_MARKERS):
            return True
    return False


def _looks_like_denial(text: str) -> bool:
    lowered = (text or "").lower()[:1000]
    return any(pattern in lowered for pattern in _DENIAL_AUDIT_PATTERNS)


class AgenticLoop:
    """Run an internal tool-calling loop until final answer or max iterations."""

    def __init__(
        self,
        llm_client: Any,
        dispatcher: Any,
        max_iter: int = 20,
        on_step: Callable[[dict[str, Any]], None] | None = None,
        timeout_sec: float = 300.0,
        silence_threshold_sec: float = 30.0,
        on_silence: Callable[[float], None] | None = None,
        on_timeout: Callable[[dict[str, Any]], str] | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_result: Callable[[str, dict[str, Any]], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.dispatcher = dispatcher
        self.max_iter = max_iter
        self.on_step = on_step
        self.timeout_sec = timeout_sec
        self.silence_threshold_sec = silence_threshold_sec
        self.on_silence = on_silence
        self.on_timeout = on_timeout
        self.tools = tools if tools is not None else ALL_TOOLS
        self.on_token = on_token
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        self.should_cancel = should_cancel
        # Per-run cumulative counters across ALL internal LLM re-calls.
        # Reset at the start of every run() so each user-turn shows its own total.
        self.cumulative_in_tokens = 0
        self.cumulative_out_tokens = 0
        self.cumulative_think_tokens = 0
        self.cumulative_chat_seconds = 0.0
        self.cumulative_subagent_in_tokens = 0
        self.cumulative_subagent_out_tokens = 0
        self.cumulative_subagent_think_tokens = 0
        self.cumulative_subagent_seconds = 0.0
        # Per-call token deltas (most recent _call_llm_stream invocation).
        # Surfaced through on_step so the UI can show a per-turn cost line.
        self.last_call_in_tokens = 0
        self.last_call_out_tokens = 0
        self.last_call_think_tokens = 0
        self._provider_call_index = 0

    def run(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run loop and return final message with execution stats."""
        history = list(messages)
        tool_calls_made = 0
        start = time.monotonic()
        timeout_extended = False
        self.cumulative_in_tokens = 0
        self.cumulative_out_tokens = 0
        self.cumulative_think_tokens = 0
        self.cumulative_chat_seconds = 0.0
        self.cumulative_subagent_in_tokens = 0
        self.cumulative_subagent_out_tokens = 0
        self.cumulative_subagent_think_tokens = 0
        self.cumulative_subagent_seconds = 0.0
        self.last_call_in_tokens = 0
        self.last_call_out_tokens = 0
        self.last_call_think_tokens = 0
        self._provider_call_index = 0
        recall_tool_called = False
        auto_recall_present = _history_has_recall_context(history)
        # 2026-05-04 R3: capture user-facing content emitted alongside
        # tool_calls in earlier iterations. OpenAI/Anthropic spec allows an
        # assistant message to carry BOTH content and tool_calls (model says
        # something while dispatching a tool). Prior code only used the LAST
        # iteration's content as `final`; if the model said its piece in
        # iter=1 and iter=2 only thought (think_tokens>0, content=None), the
        # iter=1 content was lost and the turn was tagged silent_with_reasoning.
        # Recovery: at terminal branches, if final is empty but earlier
        # iterations emitted content, fall back to the accumulated parts.
        accumulated_content: list[str] = []
        # R3g (2026-05-05): kimi-k2.5 in long-context + synthesis-retry routes
        # the user-facing answer through delta.reasoning instead of delta.content.
        # Empirical evidence: t429 in fix_r3f_run4 produced TWO visible reply
        # blocks on terminal (via on_token reasoning emit) but out=0 think=47
        # → assistant_msg.content was None both times. R3e ("reasoning carries
        # thoughts only") was correct for tool-calling iters but wrong for
        # synthesis retry. Capture reasoning per call so terminal branches can
        # surface it as final when content channel stayed empty.
        accumulated_reasoning: list[str] = []
        for it in range(1, self.max_iter + 1):
            elapsed = time.monotonic() - start
            if elapsed >= self.timeout_sec:
                if self.on_timeout is not None:
                    action = self.on_timeout(
                        {
                            "iter": it,
                            "elapsed_sec": elapsed,
                            "timeout_sec": self.timeout_sec,
                            "tool_calls_made": tool_calls_made,
                            "max_iter": self.max_iter,
                        }
                    )
                    if action == "continue" and not timeout_extended:
                        start = time.monotonic()
                        timeout_extended = True
                    elif action == "swap":
                        return {
                            "final_message": "timeout reached; model swap requested",
                            "tool_calls_made": tool_calls_made,
                            "iters": it,
                            "halt_reason": "timeout_swap_requested",
                            "cumulative_in_tokens": self.cumulative_in_tokens,
                            "cumulative_out_tokens": self.cumulative_out_tokens,
                            "cumulative_think_tokens": self.cumulative_think_tokens,
                            "cumulative_chat_seconds": self.cumulative_chat_seconds,
                        }
                    else:
                        return {
                            "final_message": "timeout reached",
                            "tool_calls_made": tool_calls_made,
                            "iters": it,
                            "halt_reason": "timeout",
                            "cumulative_in_tokens": self.cumulative_in_tokens,
                            "cumulative_out_tokens": self.cumulative_out_tokens,
                            "cumulative_think_tokens": self.cumulative_think_tokens,
                            "cumulative_chat_seconds": self.cumulative_chat_seconds,
                        }
                else:
                    return {
                        "final_message": "timeout reached",
                        "tool_calls_made": tool_calls_made,
                        "iters": it,
                        "halt_reason": "timeout",
                        "cumulative_in_tokens": self.cumulative_in_tokens,
                        "cumulative_out_tokens": self.cumulative_out_tokens,
                        "cumulative_think_tokens": self.cumulative_think_tokens,
                        "cumulative_chat_seconds": self.cumulative_chat_seconds,
                    }
            assistant_msg = self._call_llm_stream(history)
            if assistant_msg.get("_cancelled"):
                final = "".join(accumulated_content).strip() or "[cancelled by user]"
                return {
                    "final_message": final,
                    "tool_calls_made": tool_calls_made,
                    "iters": it,
                    "halt_reason": "cancelled",
                    "cumulative_in_tokens": self.cumulative_in_tokens,
                    "cumulative_out_tokens": self.cumulative_out_tokens,
                    "cumulative_think_tokens": self.cumulative_think_tokens,
                    "cumulative_chat_seconds": self.cumulative_chat_seconds,
                }
            tool_calls = assistant_msg.get("tool_calls") or []
            # R3: accumulate any content this iteration emitted (regardless of
            # tool_calls presence) so terminal branches can recover lost replies.
            iter_content = assistant_msg.get("content")
            if isinstance(iter_content, str) and iter_content.strip():
                accumulated_content.append(iter_content)
            # R3g: also accumulate reasoning-channel emissions in case the
            # answer landed there (kimi synthesis-retry pattern).
            iter_reasoning = assistant_msg.get("reasoning")
            if isinstance(iter_reasoning, str) and iter_reasoning.strip():
                accumulated_reasoning.append(iter_reasoning)
            if not tool_calls:
                final = assistant_msg.get("content") or ""
                # Forced synthesis pass — when the LLM ends a tool-using turn
                # without a user-facing reply (delta.content empty, tool_calls
                # also empty), inject one explicit synthesis instruction and
                # call once more. Catches the kimi_run2 pattern where 22/1000
                # turns ended at this branch with `final=""` after recall_turn /
                # write_file / web_search returned but the model never
                # narrated. Only fires when prior tools were called this turn —
                # a true zero-tool zero-content turn (genuine LLM silence) is
                # left untouched and tagged halt_reason="silent".
                # E1 (2026-05-04): thinking models like kimi-k2.5:cloud sometimes
                # stream the user-facing reply via delta.reasoning_content rather
                # than delta.content. think_tokens > 0 means the model already
                # responded — retry would be false-positive AND the dangling
                # assistant message (content=null + tool_calls=null) breaks
                # OpenAI spec → HTTP 400 from Ollama Cloud. Skip retry; tag
                # halt_reason="silent_with_reasoning" so the analysis layer can
                # filter cleanly.
                # R3e (2026-05-04): E1's gate (skip retry if think_tokens>0)
                # was based on the assumption that thinking models route the
                # user-facing reply via delta.reasoning_content. Raw stream
                # dumps from kimi-k2.5 disproved this — reasoning_content
                # carries thoughts only, content is always 0 when final="".
                # E2 sanitize handles the OpenAI spec issue (HTTP 400) so
                # synthesis retry is now safe regardless of reasoning state.
                if not final and tool_calls_made > 0:
                    # Preserve original call's per-step token deltas so on_step
                    # below reports them, not the retry's. cumulative_* still
                    # accumulates correctly (the retry call updates those).
                    orig_in = self.last_call_in_tokens
                    orig_out = self.last_call_out_tokens
                    orig_think = self.last_call_think_tokens
                    # E2 (2026-05-04): OpenAI spec requires assistant messages
                    # to have content OR tool_calls non-null. Defensive sanitize
                    # before history.append so providers like Ollama Cloud don't
                    # reject the retry request with HTTP 400.
                    sanitized_msg = dict(assistant_msg)
                    if sanitized_msg.get("content") is None and not sanitized_msg.get("tool_calls"):
                        sanitized_msg["content"] = ""
                    sanitized_msg.pop("reasoning", None)  # R3g: not part of OpenAI spec
                    history.append(sanitized_msg)
                    history.append({
                        "role": "system",
                        "content": (
                            "Synthesis pass: the previous tool results (which may "
                            "include errors, refusals, or partial data) are all you "
                            "have. Write a brief natural-language reply to the user "
                            "now. If the tools failed or returned nothing useful, "
                            "acknowledge that conversationally and move on. Do NOT "
                            "call any more tools."
                        ),
                    })
                    retry_msg = self._call_llm_stream(history)
                    final = (retry_msg.get("content") or "")
                    if retry_msg.get("tool_calls"):
                        sys.stderr.write(
                            "[loop:synthesis] retry returned tool_calls — dropped "
                            f"(model ignored 'no more tools'). content_len={len(final)}\n"
                        )
                    if not final:
                        sys.stderr.write(
                            f"[loop:synthesis] retry produced empty content after "
                            f"{tool_calls_made} prior tool calls; halting silent\n"
                        )
                    halt_reason = "final" if final else "silent_after_synthesis"
                    # Restore original per-step deltas for on_step reporting.
                    self.last_call_in_tokens = orig_in
                    self.last_call_out_tokens = orig_out
                    self.last_call_think_tokens = orig_think
                    # R3: if synthesis retry also failed but earlier iters had
                    # content, recover from accumulated.
                    if not final and accumulated_content:
                        final = "\n".join(accumulated_content)
                        halt_reason = "final_from_accumulated_after_synthesis"
                        sys.stderr.write(
                            f"[loop:recover] silent_after_synthesis recovered "
                            f"from accumulated content (parts={len(accumulated_content)} "
                            f"len={len(final)})\n"
                        )
                    # R3g (2026-05-05): final fallback to reasoning channel.
                    # kimi-k2.5 in long-context + synthesis-retry routes the
                    # user-facing answer through delta.reasoning. Empirical
                    # evidence: t429 fix_r3f_run4 produced TWO visible reply
                    # blocks on terminal but out=0. Surface reasoning as final
                    # when both content and accumulated_content empty.
                    if not final and accumulated_reasoning:
                        final = "\n".join(accumulated_reasoning)
                        halt_reason = "final_from_reasoning_after_synthesis"
                        sys.stderr.write(
                            f"[loop:recover] silent_after_synthesis recovered "
                            f"from reasoning channel (parts={len(accumulated_reasoning)} "
                            f"len={len(final)})\n"
                        )
                else:
                    if final:
                        halt_reason = "final"
                    elif accumulated_content:
                        final = "\n".join(accumulated_content)
                        halt_reason = "final_from_accumulated"
                        sys.stderr.write(
                            f"[loop:recover] silent recovered from "
                            f"accumulated content (parts={len(accumulated_content)} "
                            f"len={len(final)})\n"
                        )
                    elif accumulated_reasoning:
                        # R3g: pure-silent (iter=1, tools=0) — surface reasoning
                        # if model emitted user-facing text via reasoning channel.
                        final = "\n".join(accumulated_reasoning)
                        halt_reason = "final_from_reasoning"
                        sys.stderr.write(
                            f"[loop:recover] silent recovered from "
                            f"reasoning channel (parts={len(accumulated_reasoning)} "
                            f"len={len(final)})\n"
                        )
                    else:
                        halt_reason = "silent"
                if self.on_step is not None:
                    self.on_step({
                        "iter": it,
                        "tool_calls": 0,
                        "halt": halt_reason,
                        "in_tokens": self.last_call_in_tokens,
                        "out_tokens": self.last_call_out_tokens,
                        "think_tokens": self.last_call_think_tokens,
                    })
                self._emit_deny_gate_audit(
                    final=final,
                    halt_reason=halt_reason,
                    tool_calls_made=tool_calls_made,
                    recall_tool_called=recall_tool_called,
                    auto_recall_present=auto_recall_present,
                    history=history,
                )
                return {
                    "final_message": final,
                    "tool_calls_made": tool_calls_made,
                    "iters": it,
                    "halt_reason": halt_reason,
                    "cumulative_in_tokens": self.cumulative_in_tokens,
                    "cumulative_out_tokens": self.cumulative_out_tokens,
                    "cumulative_think_tokens": self.cumulative_think_tokens,
                    "cumulative_chat_seconds": self.cumulative_chat_seconds,
                }

            # R3g: strip reasoning before appending — JLC-internal field only.
            wire_msg = {k: v for k, v in assistant_msg.items() if k != "reasoning"}
            history.append(wire_msg)
            normalized_calls = []
            for call in tool_calls:
                fn = call.get("function") or {}
                args_raw = fn.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw)
                except Exception:
                    parsed_args = {}
                normalized_calls.append(
                    {
                        "id": call.get("id", ""),
                        "name": fn.get("name", ""),
                        "args": parsed_args,
                    }
                )
            if any(call.get("name") in _RECALL_TOOL_NAMES for call in normalized_calls):
                recall_tool_called = True
            if self.on_tool_call is not None:
                for call in normalized_calls:
                    try:
                        self.on_tool_call(call["name"], call.get("args", {}))
                    except Exception:
                        pass
            exec_results = self.dispatcher.execute_all(normalized_calls)
            self._accumulate_subagent_meter(normalized_calls, exec_results)
            for call, result in zip(normalized_calls, exec_results):
                if self.on_tool_result is not None:
                    try:
                        self.on_tool_result(call["name"], result)
                    except Exception:
                        pass
                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            tool_calls_made += len(tool_calls)
            if self.on_step is not None:
                self.on_step({
                    "iter": it,
                    "tool_calls": len(tool_calls),
                    "halt": None,
                    "in_tokens": self.last_call_in_tokens,
                    "out_tokens": self.last_call_out_tokens,
                    "think_tokens": self.last_call_think_tokens,
                })

        limit_msg = "loop limit reached"
        return {
            "final_message": limit_msg,
            "tool_calls_made": tool_calls_made,
            "iters": self.max_iter,
            "halt_reason": "max_iter",
            "cumulative_in_tokens": self.cumulative_in_tokens,
            "cumulative_out_tokens": self.cumulative_out_tokens,
            "cumulative_think_tokens": self.cumulative_think_tokens,
            "cumulative_chat_seconds": self.cumulative_chat_seconds,
        }

    def _emit_deny_gate_audit(
        self,
        *,
        final: str,
        halt_reason: str,
        tool_calls_made: int,
        recall_tool_called: bool,
        auto_recall_present: bool,
        history: list[dict[str, Any]],
    ) -> None:
        final_is_denial = _looks_like_denial(final)
        recall_observed = bool(recall_tool_called or auto_recall_present)
        query = _last_user_text(history)
        fields = recall_trace.query_fields(query) if query else {}
        recall_trace.emit(
            "deny_gate_audit",
            provider_call_index=self._provider_call_index,
            halt_reason=halt_reason,
            tool_calls_made=tool_calls_made,
            final_is_denial=final_is_denial,
            recall_tool_called=recall_tool_called,
            auto_recall_present=auto_recall_present,
            recall_observed=recall_observed,
            deny_gate_observed=bool(final_is_denial and recall_observed),
            final_preview=recall_trace.preview_text(final),
            **fields,
        )

    def _call_llm_stream(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        self._provider_call_index += 1
        model_name = str(getattr(self.llm_client, "model", "") or "")
        preflight = preflight_messages(
            messages,
            tools=self.tools,
            model=model_name,
            provider_call_index=self._provider_call_index,
            surface="agentic_loop",
        )
        messages = preflight.messages
        # Charge the FULL prompt sent to the API (system + history + tool
        # results so far) to the per-run counter — this is what the provider
        # actually bills, and is what `chat[in]` should reflect across N
        # internal turns instead of just the first call.
        in_delta = preflight.estimated_input_tokens
        self.last_call_in_tokens = in_delta
        self.cumulative_in_tokens += in_delta
        import os as _os
        debug_agentic = _os.environ.get("JARVIS_AGENTIC_DEBUG") == "1"
        if debug_agentic:
            try:
                tool_names = [
                    ((tool.get("function") or {}).get("name") or tool.get("name") or "")
                    for tool in (self.tools or [])
                    if isinstance(tool, dict)
                ]
                tool_details = []
                for tool in (self.tools or []):
                    if not isinstance(tool, dict):
                        continue
                    fn = tool.get("function") or tool
                    params = fn.get("parameters") if isinstance(fn, dict) else {}
                    props = params.get("properties", {}) if isinstance(params, dict) else {}
                    tool_details.append({
                        "name": fn.get("name") if isinstance(fn, dict) else "",
                        "param_keys": sorted(props.keys()) if isinstance(props, dict) else [],
                    })
                system_preview = next(
                    (
                        str(message.get("content") or "")[:3000]
                        for message in messages
                        if message.get("role") == "system"
                    ),
                    "",
                ).replace("\n", "\\n")
                sys.stderr.write(
                    f"[agentic-debug] messages={len(messages)} in_tokens={in_delta} "
                    f"tools={tool_names}\n"
                )
                sys.stderr.write(
                    "[agentic-debug] tool_details="
                    f"{json.dumps(tool_details, ensure_ascii=False, default=str)}\n"
                )
                sys.stderr.write(f"[agentic-debug] system_preview={system_preview}\n")
                sys.stderr.flush()
            except Exception as exc:
                sys.stderr.write(f"[agentic-debug] prompt dump failed: {type(exc).__name__}: {exc}\n")
                sys.stderr.flush()
        call_t0 = time.monotonic()
        try:
            stream = self.llm_client.stream_chat_completions(
                messages=messages,
                tools=self.tools,
                parallel_tool_calls=True,
                stream=True,
            )
        except Exception:
            try:
                self.cumulative_chat_seconds += time.monotonic() - call_t0
            except Exception:
                pass
            raise
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tools_by_index: dict[int, dict[str, Any]] = {}
        last_chunk_at = time.monotonic()
        # R3d (2026-05-04) ROOT CAUSE PROBE: capture every raw chunk when
        # JLC_SILENT_DUMP_DIR env is set. After stream ends, if the assembled
        # message has empty content (silent_with_reasoning trigger), dump the
        # full raw chunk sequence so we can see exactly what the API returned —
        # finish_reason, channel routing, abrupt termination, etc.
        _dump_dir = _os.environ.get("JLC_SILENT_DUMP_DIR")
        raw_chunks: list[dict[str, Any]] = [] if _dump_dir else []  # always allocate; cheap
        capture_raw = bool(_dump_dir)

        cancelled_mid_stream = False
        for chunk in stream:
            if self.should_cancel is not None:
                try:
                    if self.should_cancel():
                        cancelled_mid_stream = True
                        try:
                            stream.close()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        break
                except Exception:
                    pass
            if capture_raw:
                try:
                    raw_chunks.append(chunk)
                except Exception:
                    pass
            now = time.monotonic()
            silent_for = now - last_chunk_at
            if self.on_silence is not None and silent_for >= self.silence_threshold_sec:
                self.on_silence(silent_for)
            last_chunk_at = now
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            # DashScope qwen3 / glm reasoning models stream their internal
            # reasoning trace separately as delta.reasoning_content. Surface
            # it through on_token with kind="reasoning" so the UI can color it
            # differently from the final answer.
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)
                if self.on_token is not None:
                    try:
                        self._emit_token(reasoning, "reasoning")
                    except Exception as exc:
                        # 2026-05-03 silent-swallow probe: reasoning tokens
                        # accumulate (think_tokens > 0) but live BLUE never
                        # rendered. Surface the swallowed cause to stderr —
                        # if Windows console UnicodeEncodeError is the
                        # culprit, this exposes it instead of hiding it.
                        sys.stderr.write(f"[loop:reasoning emit fail] {type(exc).__name__}: {exc}\n")
                        sys.stderr.flush()

            text = delta.get("content")
            if isinstance(text, str):
                content_parts.append(text)
                if self.on_token is not None and text:
                    try:
                        self._emit_token(text, "content")
                    except Exception as exc:
                        sys.stderr.write(f"[loop:content emit fail] {type(exc).__name__}: {exc}\n")
                        sys.stderr.flush()
            for tc in delta.get("tool_calls") or []:
                if debug_agentic:
                    try:
                        sys.stderr.write(
                            "[agentic-debug] tool_call_delta="
                            f"{json.dumps(tc, ensure_ascii=False, default=str)}\n"
                        )
                        sys.stderr.flush()
                    except Exception:
                        pass
                idx = int(tc.get("index", 0))
                item = tools_by_index.setdefault(
                    idx,
                    {"id": tc.get("id"), "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    item["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    existing = item["function"]["name"]
                    new_name = fn["name"]
                    # Chat Completions streaming sends name in fragments
                    # ("re" + "ad" → "read"). /responses streaming sends the
                    # full name in every chunk ("recall_turn" × N). Detect:
                    # - empty existing or identical → assign new
                    # - new starts with existing → replace (full name arrived)
                    # - else → append (fragment extends existing)
                    if not existing or new_name == existing:
                        item["function"]["name"] = new_name
                    elif new_name.startswith(existing):
                        item["function"]["name"] = new_name
                    else:
                        item["function"]["name"] = existing + new_name
                if fn.get("arguments"):
                    item["function"]["arguments"] += fn["arguments"]

        try:
            self.cumulative_chat_seconds += time.monotonic() - call_t0
        except Exception:
            pass

        tool_calls = [tools_by_index[k] for k in sorted(tools_by_index)]
        content = "".join(content_parts).strip() or None
        # Charge this call's output: visible content + tool-call args also
        # cost output tokens at the provider, but we keep them in `out` for
        # transparency. Reasoning is paid separately as `think`.
        out_text = "".join(content_parts)
        for tc in tools_by_index.values():
            fn = tc.get("function") or {}
            out_text += fn.get("name") or ""
            out_text += fn.get("arguments") or ""
        out_delta = _count_tokens(out_text)
        think_delta = _count_tokens("".join(reasoning_parts))
        self.last_call_out_tokens = out_delta
        self.last_call_think_tokens = think_delta
        self.cumulative_out_tokens += out_delta
        self.cumulative_think_tokens += think_delta
        # R3d ROOT CAUSE PROBE: dump raw stream when content is None but
        # reasoning happened. Captures exactly what the API returned —
        # finish_reason, channel routing, tool_call interplay, abrupt cut.
        # R3g (2026-05-05) widening: also dump on (a) content_parts non-empty
        # but stripped to empty (whitespace-only emission) and (b) any case
        # where finish_reason=stop arrived with content=None — these are the
        # exact synthesis-retry silent cases we're hunting.
        last_finish = None
        if raw_chunks:
            try:
                _last = raw_chunks[-1]
                _ch = (_last.get("choices") or [{}])[0]
                last_finish = _ch.get("finish_reason")
            except Exception:
                pass
        should_dump = capture_raw and (
            (content is None and reasoning_parts)
            or (content is None and last_finish == "stop")
            or (content_parts and not content)  # whitespace-only emission
        )
        if should_dump:
            try:
                import datetime as _dt
                _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                _path = _os.path.join(_dump_dir, f"silent_dump_{_ts}.jsonl")
                with open(_path, "w", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "_meta": {
                            "ts": _dt.datetime.now().isoformat(),
                            "messages_in_count": len(messages),
                            "last_user_msg": next(
                                (m.get("content","")[:200] for m in reversed(messages)
                                 if m.get("role")=="user"), ""),
                            "reasoning_total_chars": sum(len(r) for r in reasoning_parts),
                            "reasoning_first_500": "".join(reasoning_parts)[:500],
                            "reasoning_last_500": "".join(reasoning_parts)[-500:],
                            "content_parts_count": len(content_parts),
                            "content_parts_total_chars": sum(len(c) for c in content_parts),
                            "content_first_500": "".join(content_parts)[:500],
                            "tool_calls_assembled": len(tool_calls),
                            "tool_call_names": [
                                (tc.get("function") or {}).get("name","")
                                for tc in tool_calls
                            ],
                            "last_finish_reason": last_finish,
                            "raw_chunks_count": len(raw_chunks),
                            "synthesis_retry": bool(messages and any(
                                m.get("role") == "system"
                                and "Synthesis pass" in (m.get("content") or "")
                                for m in messages
                            )),
                        }
                    }, ensure_ascii=False) + "\n")
                    for _c in raw_chunks:
                        _f.write(json.dumps(_c, ensure_ascii=False, default=str) + "\n")
                sys.stderr.write(
                    f"[loop:silent_dump] content=None reasoning={think_delta}tok "
                    f"tools={len(tool_calls)} chunks={len(raw_chunks)} → {_path}\n"
                )
                sys.stderr.flush()
            except Exception as _exc:
                sys.stderr.write(f"[loop:silent_dump fail] {type(_exc).__name__}: {_exc}\n")
                sys.stderr.flush()
        # R3X (2026-05-05): Channel unification — final answer always lives in
        # content. Some models (kimi-k2.5, qwen3-thinking, GLM-5, ...) emit the
        # user-facing answer via the reasoning channel when finish_reason=stop.
        # This bypassed our content-based pipeline (R3f, R3g salvages) and
        # caused channel-priority collisions (t676: stale ORANGE 174c overrode
        # BLUE 550c). r3g_run1 evidence: 5/5 fallback turns had content=0 with
        # reasoning carrying the actual answer.
        #
        # Rule: terminal iter (finish=stop, no tool_calls) with empty content
        # but non-empty reasoning → promote reasoning AS the final content.
        # Tool-calling iters keep reasoning separate (legitimate scratchpad).
        # Placed after silent_dump so dump still records the pre-promotion
        # state for forensic visibility.
        if last_finish == "stop" and not tool_calls and not content and reasoning_parts:
            promoted = "".join(reasoning_parts).strip() or None
            if promoted:
                content = promoted
                sys.stderr.write(
                    f"[loop:R3X] reasoning→content promoted "
                    f"({len(promoted)}c, {think_delta}tok)\n"
                )
                sys.stderr.flush()
        # R3g: surface reasoning so loop can fall back to it when content empty.
        # Note: reasoning is NOT part of OpenAI assistant-message spec — it's a
        # JLC-internal field used only by run() for fallback. Strip before any
        # history.append() to keep the wire format clean.
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls or None,
            "reasoning": "".join(reasoning_parts) or None,
            "_cancelled": cancelled_mid_stream,
        }

    def _accumulate_subagent_meter(
        self,
        calls: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> None:
        for call, envelope in zip(calls, results):
            if call.get("name") != "delegate_subagent":
                continue
            try:
                payload = envelope.get("result") if isinstance(envelope, dict) else None
                if not isinstance(payload, dict):
                    continue
                self.cumulative_subagent_in_tokens += int(payload.get("in_tokens") or 0)
                self.cumulative_subagent_out_tokens += int(payload.get("out_tokens") or 0)
                self.cumulative_subagent_think_tokens += int(payload.get("think_tokens") or 0)
                self.cumulative_subagent_seconds += float(payload.get("elapsed_sec") or 0.0)
            except Exception:
                continue

    def _emit_token(self, text: str, kind: str) -> None:
        """Forward a token to on_token, passing kind when the callback supports it.

        Backwards-compatible with single-arg callbacks that ignore kind.
        """
        cb = self.on_token
        if cb is None or not text:
            return
        try:
            cb(text, kind)
        except TypeError:
            cb(text)
