"""Chat turn entry - wraps AgenticLoop with one-shot jhb prepend per chat turn."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
import threading
from typing import Any

from jlc_agentic.providers import get_llm, turn_context

from .loop import AgenticLoop, _count_messages_tokens, _count_tokens, _msg_text

# W2.9.28 (2026-05-25): server-side recall hook. 1만턴 벤치 W2.9.27 결과
# miss_with_denial=96 — chat LLM이 회상 질문에서 jlc_recall 자율 호출 안 하고
# 그냥 denial로 빠지는 패턴 잔존. reasoning_policy.py [Tools]/[Recall+JHB]
# HARD 태그가 tool description을 직접 모순(architect 진단). 정책 수술 +
# 결정적 regex hook으로 LLM 자율성 우회. recall_block 인프라는
# prepend_two_tier에 이미 배관됨 — 추가 인프라 0.
_RECALL_TRIGGER_RE = re.compile(
    r"(예전에|아까|그때|이전에|기억나|기억해|적어놨|적어둔|"
    r"말했었|말한 적|얘기했|얘기한|얘기했었|뭐였|뭐였더라|"
    r"했었지|볼트에|썼던|적었|이전 대화|previous|previously|"
    r"earlier|we discussed|remember when|you said|I said|"
    r"before you|last time)",
    re.IGNORECASE,
)
# Disable knob for ablation / debugging. ON by default.
_AUTO_RECALL_ENABLED = os.environ.get("JLC_AUTO_RECALL", "1") != "0"

# Only completed turns persist into JHB. timeout/swap/cancelled = incomplete,
# encoder skipped so the JHB is not contaminated by partial state.
# R3 (2026-05-04): final_from_accumulated[_after_synthesis] = recovered turns
# where the user-facing reply was emitted in an earlier iteration alongside
# tool_calls; treat as completed for encoding purposes.
# W2 C-r3 (2026-05-06): final_from_reasoning[_after_synthesis] = Ollama-style
# reasoning models that route the user-facing answer through message.reasoning
# while content stays empty. R3X promotion handles this when raw_chunk capture
# is enabled, otherwise the loop falls back to halt=final_from_reasoning. Both
# are valid completed turns from JHB's perspective.
_ENCODE_HALT_REASONS = {
    "final",
    "max_iter",
    "final_from_accumulated",
    "final_from_accumulated_after_synthesis",
    "final_from_reasoning",
    "final_from_reasoning_after_synthesis",
}


class _LazyChatLLM:
    """Defers ``get_llm("chat")`` to the first ``stream_chat_completions``
    call. ChatTurn.__init__ used to resolve the LLM eagerly, which forced
    config + ProviderRouter resolution before the aider boot sequence had
    a chance to register the router (W2.6 race observed when ChatTurn was
    constructed in tests / library use without aider main).
    Forwards llm_meta from the underlying adapter so ChatTurn.run can lift
    it onto the result dict, matching what the previous router-swap code
    did.
    """

    def __init__(self) -> None:
        self._inner: Any = None
        self.llm_meta: dict[str, Any] | None = None
        self._resolve_lock = threading.Lock()

    def _resolve(self) -> Any:
        # Double-checked locking: the outer check is the fast path (no lock
        # acquisition once resolved); the inner re-check inside the lock
        # prevents two threads from each calling get_llm and clobbering
        # _inner. Acceptable cost: one Lock acquisition per first-turn
        # resolution per ChatTurn instance.
        if self._inner is None:
            with self._resolve_lock:
                if self._inner is None:
                    self._inner = get_llm("chat")
        return self._inner

    def stream_chat_completions(self, *args: Any, **kwargs: Any) -> Any:
        inner = self._resolve()
        result = inner.stream_chat_completions(*args, **kwargs)
        # Mirror llm_meta from the resolved adapter (LLMRouterAdapter sets
        # this in stream_chat_completions; legacy adapters leave it None).
        self.llm_meta = getattr(inner, "llm_meta", None)
        return result


class ChatTurn:
    """Single chat turn entrypoint with one-shot jhb prepend + one-shot encode."""

    def __init__(
        self,
        slim: Any,
        llm_client: Any | None = None,
        dispatcher: Any = None,
        max_iter: int = 20,
        on_step=None,
        on_token=None,
        on_tool_call=None,
        on_tool_result=None,
        on_encode_token=None,
        on_encode_start=None,
        on_encode_done=None,
        should_cancel=None,
    ) -> None:
        self.slim = slim
        self.on_encode_token = on_encode_token
        self.on_encode_start = on_encode_start
        self.on_encode_done = on_encode_done
        effective_llm = llm_client if llm_client is not None else _LazyChatLLM()
        self.loop = AgenticLoop(
            llm_client=effective_llm,
            dispatcher=dispatcher,
            max_iter=max_iter,
            on_step=on_step,
            on_token=on_token,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            should_cancel=should_cancel,
        )

    def run(
        self,
        user_message: str,
        conv_id: str = "conversation",
        project_path: str | None = None,
        recall_block: str = "",
        prior_history: list[dict[str, Any]] | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        """Run one chat turn.

        prior_history contains prior assistant/tool/system history from older turns.
        reasoning_effort is the user-selected chat effort. It is carried on
        turn_context so the provider can spend the requested amount of thinking for
        this turn. None leaves the provider/SDK default untouched.
        jhb prepend is applied once at the chat-turn boundary before loop execution.
        Encoder is invoked at chat-turn end ONLY when the turn completed cleanly
        (halt_reason in {final, max_iter}). timeout/swap/cancelled skips encoder
        to avoid contaminating JHB with partial state.
        Prior encode for the same conv_id is awaited so prepend_two_tier sees fresh JHB.
        """
        try:
            turn_chat_in_before = int(getattr(self.loop, "cumulative_in_tokens", 0) or 0)
            turn_chat_out_before = int(getattr(self.loop, "cumulative_out_tokens", 0) or 0)
            turn_chat_think_before = int(getattr(self.loop, "cumulative_think_tokens", 0) or 0)
            turn_chat_seconds_before = float(getattr(self.loop, "cumulative_chat_seconds", 0.0) or 0.0)
            turn_subagent_in_before = int(getattr(self.loop, "cumulative_subagent_in_tokens", 0) or 0)
            turn_subagent_out_before = int(getattr(self.loop, "cumulative_subagent_out_tokens", 0) or 0)
            turn_subagent_think_before = int(getattr(self.loop, "cumulative_subagent_think_tokens", 0) or 0)
            turn_subagent_seconds_before = float(getattr(self.loop, "cumulative_subagent_seconds", 0.0) or 0.0)
            # idea #12 step 2 + 4: stale-tolerant entry with backlog throttle.
            # Path B's raw_recent_window covers up to 4 turns of encoder lag,
            # so the chat side does NOT block the next turn waiting for the
            # encoder to finish. Instead, it only throttles when the in-flight
            # count exceeds the absorber depth (>= 3 active encodes), in which
            # case a tier-1 wait gives the encoder a chance to drain. Tier 2
            # is a safety net so lag cannot grow without bound under retry
            # storms or cold-start cascades.
            # W2.9.23: backpressure ensure window coverage
            # W2.9.24 (2026-05-11): JLC_DISABLE_BACKPRESSURE=1 = Recipe B drain-on-fire,
            # chat does NOT wait for encoder. raw_recent_window covers lag.
            blocked_s = 0.0
            if os.environ.get("JLC_DISABLE_BACKPRESSURE") == "1":
                pass  # drain-on-fire mode
            elif hasattr(self.slim, "ensure_window_coverage"):
                try:
                    blocked_s = self.slim.ensure_window_coverage(conv_id)
                except Exception as exc:
                    sys.stderr.write(f"[chat_turn] ensure_window_coverage failed: {exc}\n")
            else:
                # Fallback to legacy throttle if slim not updated
                inflight_probe = getattr(self.slim, "encode_in_flight", None)
                wait_pending = getattr(self.slim, "wait_for_pending_encode", None)
                if callable(inflight_probe) and callable(wait_pending):
                    try:
                        if inflight_probe(conv_id) >= 3:
                            t0 = time.monotonic()
                            wait_pending(conv_id, timeout=1.0)
                            if inflight_probe(conv_id) >= 3:
                                wait_pending(conv_id, timeout=5.0)
                            blocked_s = time.monotonic() - t0
                    except Exception as exc:
                        sys.stderr.write(f"[chat_turn] backlog throttle failed: {exc}\n")
                elif callable(wait_pending):
                    try:
                        t0 = time.monotonic()
                        wait_pending(conv_id)
                        blocked_s = time.monotonic() - t0
                    except Exception as exc:
                        sys.stderr.write(f"[chat_turn] wait_for_pending_encode failed: {exc}\n")

            # Projectless/chat turns keep the prompt compact: JHB already
            # carries the durable conversation silhouette, so the raw
            # recent_window only adds token bloat here.
            recent_window = self._build_recent_window(conv_id) if project_path else ""
            effective_user_msg = recent_window + user_message if recent_window else user_message

            messages: list[dict[str, Any]] = list(prior_history or [])
            messages.append({"role": "user", "content": effective_user_msg})

            # W2.9.28: server-side recall hook — see module header comment.
            if _AUTO_RECALL_ENABLED:
                try:
                    auto_block = self._maybe_auto_recall(user_message, conv_id)
                    if auto_block:
                        recall_block = (
                            (recall_block + "\n\n" + auto_block) if recall_block else auto_block
                        )
                except Exception as exc:
                    sys.stderr.write(f"[chat_turn] auto_recall hook failed: {exc}\n")

            prepared = self.slim.prepend_two_tier(
                aider_messages=messages,
                project_path=project_path,
                conv_id=conv_id,
                recall_block=recall_block,
            )
            # Per-turn context for the Agent SDK adapter (anthropic-agent-sdk):
            # set on THIS worker thread so the adapter (running inside loop.run on
            # the same thread) can read cwd/conv_id/retriever and capture them into
            # its MCP closures. No-op for every other provider. Cleared in finally
            # so it never leaks across turns. (2026-06-15)
            turn_context.set(
                conv_id=conv_id,
                project_root=project_path,
                retriever=getattr(self.slim, "retriever", None),
                reasoning_effort=reasoning_effort,
            )
            try:
                result = self.loop.run(prepared)
            finally:
                turn_context.clear()
            result["turn_chat_in_tokens"] = max(
                0,
                int(result.get("cumulative_in_tokens") or getattr(self.loop, "cumulative_in_tokens", 0) or 0) - turn_chat_in_before,
            )
            result["turn_chat_out_tokens"] = max(
                0,
                int(result.get("cumulative_out_tokens") or getattr(self.loop, "cumulative_out_tokens", 0) or 0) - turn_chat_out_before,
            )
            result["turn_chat_think_tokens"] = max(
                0,
                int(result.get("cumulative_think_tokens") or getattr(self.loop, "cumulative_think_tokens", 0) or 0) - turn_chat_think_before,
            )
            result["turn_chat_seconds"] = max(
                0.0,
                float(result.get("cumulative_chat_seconds") or getattr(self.loop, "cumulative_chat_seconds", 0.0) or 0.0) - turn_chat_seconds_before,
            )
            result["turn_subagent_in_tokens"] = max(
                0,
                int(getattr(self.loop, "cumulative_subagent_in_tokens", 0) or 0) - turn_subagent_in_before,
            )
            result["turn_subagent_out_tokens"] = max(
                0,
                int(getattr(self.loop, "cumulative_subagent_out_tokens", 0) or 0) - turn_subagent_out_before,
            )
            result["turn_subagent_think_tokens"] = max(
                0,
                int(getattr(self.loop, "cumulative_subagent_think_tokens", 0) or 0) - turn_subagent_think_before,
            )
            result["turn_subagent_seconds"] = max(
                0.0,
                float(getattr(self.loop, "cumulative_subagent_seconds", 0.0) or 0.0) - turn_subagent_seconds_before,
            )
            # W2.9.23: track backpressure time in result dict
            if blocked_s > 0:
                result["enc_blocking_s"] = blocked_s
            # W2.6: get_llm("chat") already returns LLMRouterAdapter when the
            # ProviderRouter is registered, so a separate router-swap path
            # in ChatTurn is no longer needed. Just lift llm_meta off the
            # adapter (router or lazy proxy) onto the result.
            llm_meta = getattr(self.loop.llm_client, "llm_meta", None)
            if llm_meta is not None:
                result["llm_meta"] = llm_meta
                usage_meta = llm_meta.get("usage") if isinstance(llm_meta, dict) else None
                if isinstance(usage_meta, dict):
                    reasoning_tokens = int(
                        usage_meta.get("reasoningTokens")
                        or usage_meta.get("reasoning_tokens")
                        or usage_meta.get("thought")
                        or 0
                    )
                    if reasoning_tokens > 0 and int(result.get("turn_chat_think_tokens") or 0) <= 0:
                        turn_chat_out = int(result.get("turn_chat_out_tokens") or 0)
                        result["turn_chat_think_tokens"] = reasoning_tokens
                        result["turn_chat_out_tokens"] = max(0, turn_chat_out - reasoning_tokens)
        except KeyboardInterrupt:
            return {
                "final_message": "[interrupted by user]",
                "tool_calls_made": 0,
                "iters": 0,
                "halt_reason": "cancelled",
            }

        if result.get("halt_reason") in _ENCODE_HALT_REASONS:
            self._inject_chat_meter(
                blocked_s=blocked_s,
                prepared=prepared,
                user_msg=user_message,
                assistant_msg=result.get("final_message", "") or "",
                result=result,
            )
            self._schedule_encode(
                project_path=project_path,
                conv_id=conv_id,
                user_msg=user_message,
                assistant_msg=result.get("final_message", "") or "",
                llm_meta=result.get("llm_meta"),
            )
        return result

    def _maybe_auto_recall(self, user_message: str, conv_id: str) -> str:
        """W2.9.28 server-side recall hook. If the user prompt contains a
        strong past-reference marker, fire retriever.hybrid_search
        deterministically and return an `<auto_recall>` block for
        prepend_two_tier to inject. Bypasses chat LLM's tool-call judgment
        which proved unreliable at the 10k benchmark (miss_with_denial=96
        even after the W2.9.27 graceful-packaging fix).
        """
        if not user_message or not _RECALL_TRIGGER_RE.search(user_message):
            return ""
        retriever = getattr(self.slim, "retriever", None)
        if retriever is None:
            return ""
        try:
            result = asyncio.run(
                retriever.hybrid_search(
                    query=user_message,
                    top_k=5,
                    session_id=conv_id,
                )
            )
        except RuntimeError as exc:
            # asyncio.run inside a live event loop → bail (caller can later
            # promote this hook to async if needed). Logged so we notice.
            sys.stderr.write(f"[chat_turn] auto_recall asyncio.run blocked: {exc}\n")
            return ""
        except Exception as exc:
            sys.stderr.write(f"[chat_turn] auto_recall hybrid_search failed: {exc}\n")
            return ""
        if not isinstance(result, dict):
            return ""
        fragments = result.get("fragments") or []
        if not fragments:
            return ""
        try:
            payload = json.dumps(
                {"query": user_message[:300], "fragments": fragments},
                ensure_ascii=False,
                indent=2,
            )
        except Exception:
            return ""
        return (
            "<auto_recall>\n"
            "Server-side pre-fetched recall fragments for the past-reference\n"
            "query. Treat as authoritative — answer from these and do NOT\n"
            "re-issue jlc_recall for the same query.\n"
            + payload
            + "\n</auto_recall>"
        )

    def _stash_recent_window(self, text: str) -> None:
        """Stash the rendered recent_window text on slim so the chat-meter
        breakdown can attribute the tokens to a dedicated bucket. Always
        update on every code path of _build_recent_window — including early
        returns — so a later turn cannot read a stale value from a prior
        turn (which would double-count tokens into recent_window and
        under-count tool_loop by the same amount).
        """
        try:
            self.slim.last_recent_window_text = text
        except Exception:
            pass

    def _build_recent_window(self, conv_id: str) -> str:
        """Path B (2026-05-09): inject the last N raw turns as `<recent_window>`
        on top of the user message. Replaces the encoder-emitted Conversation
        Tail, which was hallucinating turn IDs under weaker instruction
        followers like devstral-small-2:24b. Trades ~1k chat input tokens for
        zero hallucination + encoder-model flexibility.
        """
        try:
            slim = self.slim
            cfg = getattr(getattr(slim, "config", None), "conversation_tail", None)
            if cfg is not None and not getattr(cfg, "enabled", True):
                self._stash_recent_window("")
                return ""
            count = int(getattr(cfg, "count", 3)) if cfg else 3
            if count <= 0:
                self._stash_recent_window("")
                return ""
            retriever = getattr(slim, "retriever", None)
            if retriever is None:
                self._stash_recent_window("")
                return ""
            turns = list(retriever.load_turns(conv_id) or [])
            # W2.9.21 §4.2 follow-up: merge in-memory batch buffer so the raw
            # window stays gap-free between fires. retriever.save_turn only
            # commits to disk when the batch fires, leaving up to BATCH_SIZE-1
            # most recent turns invisible to a disk-only window. Pull them
            # from slim._batch_buffer and synthesize sequential turn ids that
            # continue past the disk tail (display-only labels).
            buf_turns: list[dict[str, Any]] = []
            buffer = getattr(slim, "_batch_buffer", None)
            guard = getattr(slim, "_batch_buffer_guard", None)
            if isinstance(buffer, dict):
                try:
                    if guard is not None:
                        with guard:
                            buf_copy = list(buffer.get(conv_id, []))
                    else:
                        buf_copy = list(buffer.get(conv_id, []))
                    last_disk_turn = 0
                    if turns:
                        try:
                            last_disk_turn = int(turns[-1].get("turn") or 0)
                        except (TypeError, ValueError):
                            last_disk_turn = 0
                    for i, b in enumerate(buf_copy, start=1):
                        if not isinstance(b, dict):
                            continue
                        buf_turns.append({
                            "turn": last_disk_turn + i,
                            "user": b.get("user", "") or "",
                            "assistant": b.get("assistant", "") or "",
                        })
                except Exception as exc:
                    sys.stderr.write(f"[chat_turn] buffer merge failed: {exc}\n")
            combined = turns + buf_turns
            recent = combined[-count:] if combined else []
            if not recent:
                self._stash_recent_window("")
                return ""
            lines = ["<recent_window>"]
            for t in recent:
                if not isinstance(t, dict):
                    continue
                tid = t.get("turn", t.get("turn_id", "?"))
                u = (t.get("user") or "").strip()
                a = (t.get("assistant") or "").strip()
                if len(u) > 600:
                    u = u[:600] + "…"
                if len(a) > 1500:
                    a = a[:1500] + "…"
                u = u.replace("\r", " ").replace("\n", " ")
                a = a.replace("\r", " ").replace("\n", " ")
                lines.append(f"turn {tid}:")
                lines.append(f"  USER: {u}")
                lines.append(f"  ASSISTANT: {a}")
            lines.append("</recent_window>")
            lines.append("")
            rendered = "\n".join(lines)
            self._stash_recent_window(rendered)
            return rendered
        except Exception as exc:
            sys.stderr.write(f"[chat_turn] _build_recent_window failed: {exc}\n")
            self._stash_recent_window("")
            return ""

    def _inject_chat_meter(
        self,
        prepared: list[dict[str, Any]],
        user_msg: str,
        assistant_msg: str,
        result: dict[str, Any],
        blocked_s: float = 0.0,
    ) -> None:
        """Stash sidecar chat token counts before encode prints the meter line."""
        try:
            encoder = getattr(self.slim, "encoder", None)
            if encoder is None:
                return

            jlc_head_tok = _count_tokens(getattr(self.slim, "last_jlc_head_text", "") or "")

            system_tok = 0
            first_non_system = 0
            if prepared and prepared[0].get("role") == "system":
                system_tok = max(0, _count_tokens(_msg_text(prepared[0])) - jlc_head_tok)
                first_non_system = 1

            user_tok = 0
            last_user_idx = -1
            for idx in range(len(prepared) - 1, -1, -1):
                if prepared[idx].get("role") == "user":
                    last_user_idx = idx
                    user_tok = _count_tokens(_msg_text(prepared[idx]))
                    break
            if not user_tok:
                user_tok = _count_tokens(user_msg)

            history_messages = [
                msg
                for idx, msg in enumerate(prepared[first_non_system:])
                if idx + first_non_system != last_user_idx
            ]
            history_tok = _count_messages_tokens(history_messages)

            cum_in = result.get("cumulative_in_tokens") or getattr(self.loop, "cumulative_in_tokens", 0)
            cum_out = result.get("cumulative_out_tokens") or getattr(self.loop, "cumulative_out_tokens", 0)
            cum_think = result.get("cumulative_think_tokens") or getattr(self.loop, "cumulative_think_tokens", 0)
            cum_seconds = result.get("cumulative_chat_seconds")
            if cum_seconds is None:
                cum_seconds = getattr(self.loop, "cumulative_chat_seconds", 0.0)

            first_call_in = _count_messages_tokens(prepared)
            if not cum_in:
                cum_in = first_call_in
            if not cum_out:
                cum_out = _count_tokens(assistant_msg)
            if not cum_think:
                cum_think = 0

            encoder.last_chat_in = int(cum_in)
            encoder.last_chat_in_user = int(user_tok)
            encoder.last_chat_think = int(cum_think)
            encoder.last_chat_out = int(cum_out)
            encoder.last_chat_cache_read = 0
            encoder.last_chat_cache_write = 0
            encoder.last_chat_seconds = float(cum_seconds or 0.0)
            encoder.last_enc_blocking_s = float(blocked_s)
            encoder.last_subagent_in = int(getattr(self.loop, "cumulative_subagent_in_tokens", 0) or 0)
            encoder.last_subagent_out = int(getattr(self.loop, "cumulative_subagent_out_tokens", 0) or 0)
            encoder.last_subagent_think = int(getattr(self.loop, "cumulative_subagent_think_tokens", 0) or 0)
            encoder.last_subagent_seconds = float(getattr(self.loop, "cumulative_subagent_seconds", 0.0) or 0.0)
            encoder.last_chat_turn_in = int(result.get("turn_chat_in_tokens") or 0)
            encoder.last_chat_turn_out = int(result.get("turn_chat_out_tokens") or 0)
            encoder.last_chat_turn_cache_read = 0
            encoder.last_chat_turn_cache_write = 0
            encoder.last_chat_turn_think = int(result.get("turn_chat_think_tokens") or 0)
            encoder.last_chat_turn_seconds = float(result.get("turn_chat_seconds") or 0.0)
            encoder.last_subagent_turn_in = int(result.get("turn_subagent_in_tokens") or 0)
            encoder.last_subagent_turn_out = int(result.get("turn_subagent_out_tokens") or 0)
            encoder.last_subagent_turn_think = int(result.get("turn_subagent_think_tokens") or 0)
            encoder.last_subagent_turn_seconds = float(result.get("turn_subagent_seconds") or 0.0)
            recent_window_tok = _count_tokens(getattr(self.slim, "last_recent_window_text", "") or "")
            head_bd = getattr(self.slim, "last_jlc_head_breakdown", {}) or {}
            encoder.last_chat_in_breakdown = {
                "jlc_head": int(jlc_head_tok),
                "head_constitution": int(head_bd.get("constitution", 0) or 0),
                "head_lang": int(head_bd.get("lang", 0) or 0),
                "head_tool_channel": int(head_bd.get("tool_channel", 0) or 0),
                "head_reasoning": int(head_bd.get("reasoning", 0) or 0),
                "head_env": int(head_bd.get("env", 0) or 0),
                "head_retrieval": int(head_bd.get("retrieval", 0) or 0),
                "head_jhb": int(head_bd.get("jhb", 0) or 0),
                "head_project_md": int(head_bd.get("project_md", 0) or 0),
                "aider_system": int(system_tok),
                "history": int(history_tok),
                "user_msg": int(user_tok),
                "repo_map": 0,
                "recent_window": int(recent_window_tok),
                "tool_loop": int(max(0, int(cum_in) - first_call_in)),
                "subagent": int(getattr(self.loop, "cumulative_subagent_in_tokens", 0) or 0),
            }
        except Exception as exc:
            sys.stderr.write(f"[jlc:meter] chat inject failed: {exc}\n")

    def _schedule_encode(
        self,
        project_path: str | None,
        conv_id: str,
        user_msg: str,
        assistant_msg: str,
        llm_meta: dict[str, Any] | None = None,
    ) -> None:
        """Fire-and-forget background encode. Compatible with slim.encode_and_save_async
        signature (which itself dispatches to a thread) and with simpler mocks that
        accept the same kwargs synchronously."""
        encode = getattr(self.slim, "encode_and_save_async", None)
        if not callable(encode):
            return
        try:
            kwargs = {
                "project_path": project_path,
                "conv_id": conv_id,
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
            }
            if llm_meta is not None:
                kwargs["llm_meta"] = llm_meta
            if self.on_encode_token is not None:
                kwargs["on_token"] = self.on_encode_token
            if self.on_encode_done is not None:
                kwargs["on_done"] = self.on_encode_done
            if self.on_encode_start is not None:
                try:
                    self.on_encode_start()
                except Exception:
                    pass
            encode(**kwargs)
        except Exception as exc:
            sys.stderr.write(f"[chat_turn] encode_and_save_async failed: {exc}\n")
