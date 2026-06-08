"""WebSocket endpoint for the local Jarvis web UI."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
import traceback
import uuid
from typing import Any, Callable

from fastapi import WebSocket, WebSocketDisconnect

from jlc_agentic import JarvisAgentic, get_slim
from jlc_agentic.agentic.chat_turn import ChatTurn
from jlc_agentic.agentic.schema import get_dispatcher
from jlc_agentic.providers import get_llm

from .messages import Inbound, Outbound
from .state import parse_jhb_blocks, refresh_mixer_tokens, save_mixer, ui_state


class ConnectionManager:
    async def send(self, websocket: WebSocket, payload: Outbound) -> None:
        await websocket.send_text(json.dumps(payload, ensure_ascii=False))


manager = ConnectionManager()


class _ToolArgContentFilter:
    """Suppress plaintext JSON tool-argument objects from the UI stream.

    Cmd output remains raw because _cmd_emit runs before this filter. This is a
    defensive UI boundary for providers that leak function-call args as content
    instead of structured tool_calls.
    """

    _PRIMARY_KEYS = {
        "query",
        "limit",
        "path",
        "pattern",
        "command",
        "project_path",
        "old_string",
        "new_string",
        "top_k",
        "turns_back",
        "n_results",
    }
    _SHADOW_RE = re.compile(
        r"(?i)(?<![\w])(?:analysis\s+to=|final\s+to=|assistant\s+to=|code\s+to=|to=)"
        r"[A-Za-z_][\w_]*|is\s+not\s+a\s+valid\s+code\s+block"
    )
    _SHADOW_RECOVERY_RE = re.compile(
        r"(?i)is\s+not\s+a\s+valid\s+code\s+block|let\s+me\s+retry|"
        r"i\s+cannot\s+(?:use|call|execute|parse)"
    )
    # Hangul run ≥6 chars signals user-facing Korean answer emerging from
    # shadow noise. Used for shadow recovery so the answer is preserved even
    # when the model concatenates it on the same line as tool-call shadow.
    _HANGUL_RUN_RE = re.compile(r"[가-힯ᄀ-ᇿ]{6,}")
    # Lookahead tail held at end of each feed so a shadow pattern split across
    # streaming chunks (e.g., "to=" + "recall_turn") is still detected on the
    # next feed. Must exceed the longest SHADOW_RE prefix (~"is not a valid
    # code block" = 24 chars).
    _PLAIN_LOOKAHEAD = 32

    def __init__(self) -> None:
        self._buf: list[str] | None = None
        self._shadow: str | None = None
        self._depth = 0
        self._in_string = False
        self._escape = False
        self._plain_tail = ""

    def feed(self, text: str) -> list[str]:
        out: list[str] = []
        # Prepend lookahead tail held from the previous feed so SHADOW_RE
        # patterns that straddle chunk boundaries are still detected.
        if self._plain_tail and self._buf is None and self._shadow is None:
            text = self._plain_tail + text
            self._plain_tail = ""
        plain: list[str] = []
        index = 0
        while index < len(text):
            if self._shadow is not None:
                remainder = self._feed_shadow(text[index:])
                if remainder:
                    out.extend(self.feed(remainder))
                return out

            ch = text[index]
            if self._buf is None:
                if ch == "{":
                    if plain:
                        out.append("".join(plain))
                        plain = []
                    self._buf = ["{"]
                    self._depth = 1
                    self._in_string = False
                    self._escape = False
                else:
                    plain.append(ch)
                    shadow_match = self._SHADOW_RE.search("".join(plain))
                    if shadow_match:
                        before = "".join(plain[: shadow_match.start()])
                        if before:
                            out.append(before)
                        self._shadow = "".join(plain[shadow_match.start():])
                        plain = []
                index += 1
                continue

            self._buf.append(ch)
            self._advance(ch)
            if self._depth == 0:
                candidate = "".join(self._buf)
                self._buf = None
                if not self._looks_like_tool_args(candidate):
                    out.append(candidate)
            elif len(self._buf) > 4096:
                out.append("".join(self._buf))
                self._buf = None
                self._depth = 0
                self._in_string = False
                self._escape = False
            index += 1
        if plain:
            plain_str = "".join(plain)
            safe, hold = self._split_safe_plain(plain_str)
            if safe:
                out.append(safe)
            self._plain_tail = hold
        return out

    def flush(self) -> list[str]:
        """Emit any held plain tail. Call at turn end so partial-prefix tail
        does not stay buffered indefinitely. Drops shadow/buf state so the
        next turn starts fresh."""
        out: list[str] = []
        if self._plain_tail:
            out.append(self._plain_tail)
            self._plain_tail = ""
        self._shadow = None
        self._buf = None
        self._depth = 0
        self._in_string = False
        self._escape = False
        return out

    def _split_safe_plain(self, plain_str: str) -> tuple[str, str]:
        """Return (emit_now, hold_for_next_feed).

        Hold trailing chars only when they look like the START of a SHADOW_RE
        pattern that could be completed by the next chunk. Specifically:
        - trailing `=` (suggestive of `to=` continuation), or
        - trailing `to` at a word boundary (could become `to=identifier`).
        Other tokens (Korean answers, `c0|`, `One`, etc.) flush immediately
        so streaming granularity is preserved."""
        n = len(plain_str)
        if n == 0:
            return "", ""
        if plain_str.endswith("="):
            return plain_str[:-1], "="
        if n >= 2 and plain_str[-2:].lower() == "to":
            prev_alnum = n >= 3 and (plain_str[-3].isalnum() or plain_str[-3] == "_")
            if not prev_alnum:
                return plain_str[:-2], plain_str[-2:]
        return plain_str, ""

    def _feed_shadow(self, text: str) -> str:
        self._shadow = (self._shadow or "") + text
        shadow = self._shadow

        phrase = self._SHADOW_RECOVERY_RE.search(shadow)
        if phrase:
            newline = shadow.find("\n", phrase.end())
            if newline >= 0:
                remainder = shadow[newline + 1 :]
                self._shadow = None
                return remainder
            return ""

        json_end = self._tool_json_end(shadow)
        if json_end is not None:
            after = shadow[json_end:].lstrip()
            if after and not self._SHADOW_RE.search(after[:80]):
                self._shadow = None
                return after

        # Korean answer often resumes on the SAME line as shadow garbage —
        # recover from the first sustained Hangul run so the user-facing
        # reply is preserved even without a newline separator.
        hangul = self._HANGUL_RUN_RE.search(shadow)
        if hangul:
            self._shadow = None
            return shadow[hangul.start() :]

        if "{" not in shadow:
            newline = shadow.find("\n")
            if newline >= 0:
                remainder = shadow[newline + 1 :]
                self._shadow = None
                return remainder

        if len(shadow) > 4096:
            self._shadow = None
        return ""

    def _tool_json_end(self, text: str) -> int | None:
        start = text.find("{")
        while start >= 0:
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(text)):
                ch = text[idx]
                if escape:
                    escape = False
                    continue
                if ch == "\\" and in_string:
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        if self._looks_like_tool_args(candidate):
                            return idx + 1
                        break
            start = text.find("{", start + 1)
        return None

    def _advance(self, ch: str) -> None:
        if self._escape:
            self._escape = False
            return
        if ch == "\\" and self._in_string:
            self._escape = True
            return
        if ch == '"':
            self._in_string = not self._in_string
            return
        if self._in_string:
            return
        if ch == "{":
            self._depth += 1
        elif ch == "}":
            self._depth -= 1

    def _looks_like_tool_args(self, candidate: str) -> bool:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            return False
        if not isinstance(data, dict):
            return False
        keys = set(data)
        # Any JSON object carrying a tool-arg primary key is treated as shadow.
        # The previous subset constraint (every key must be allowed) was too
        # narrow — `top_k`/`turns_back` slipped through. Trade-off: a legitimate
        # answer that happens to render `{"query": ...}` as an example will be
        # suppressed, which Phase 5 self-critique already accepted.
        return bool(keys & self._PRIMARY_KEYS)


class _ThinkTagParser:
    """Split a streaming content channel into reasoning (inside <think>...
    </think>) and user-facing content (outside). Some chat models (e.g.
    minimax-m2.5) inline reasoning inside content with <think> tags rather
    than emitting a separate `delta.reasoning_content` field, so without this
    split the UI never sees a `kind="reasoning"` chunk and ReasoningCard
    stays hidden. Robust to chunks that split a tag across stream boundaries.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _OPEN_TAIL = len(_OPEN) - 1  # max partial-prefix bytes to hold
    _CLOSE_TAIL = len(_CLOSE) - 1

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        """Return [(emit_text, kind)] where kind ∈ {'reasoning', 'content'}."""
        if not text:
            return []
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            if self._in_think:
                idx = self._buf.find(self._CLOSE)
                if idx >= 0:
                    if idx > 0:
                        out.append((self._buf[:idx], "reasoning"))
                    self._buf = self._buf[idx + len(self._CLOSE):]
                    self._in_think = False
                    continue
                # Hold the trailing CLOSE_TAIL chars in case the close tag
                # straddles the next chunk.
                if len(self._buf) > self._CLOSE_TAIL:
                    out.append((self._buf[:-self._CLOSE_TAIL], "reasoning"))
                    self._buf = self._buf[-self._CLOSE_TAIL:]
                break
            idx = self._buf.find(self._OPEN)
            if idx >= 0:
                if idx > 0:
                    out.append((self._buf[:idx], "content"))
                self._buf = self._buf[idx + len(self._OPEN):]
                self._in_think = True
                continue
            if len(self._buf) > self._OPEN_TAIL:
                out.append((self._buf[:-self._OPEN_TAIL], "content"))
                self._buf = self._buf[-self._OPEN_TAIL:]
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        """Drain held bytes at turn end. Anything still buffered did not
        complete a tag, so emit it as the last channel we were in."""
        if not self._buf:
            return []
        kind = "reasoning" if self._in_think else "content"
        out = [(self._buf, kind)]
        self._buf = ""
        self._in_think = False
        return out


def _mixer_system_message() -> str:
    parts: list[str] = []
    if ui_state.mixer.get("english_only"):
        parts.append(
            "Web UI Context Mixer: answer in English only for this turn. "
            "Do not include Korean, Japanese, or Chinese in reasoning or the final answer."
        )
    if ui_state.mixer.get("custom_note_enabled") and ui_state.mixer.get("custom_note"):
        parts.append(f"Web UI Custom Note:\n{ui_state.mixer['custom_note']}")
    return "\n\n".join(parts)


UI_TOOL_SYSTEM_MESSAGE = """[Jarvis UI tool use]
You have structured tools available in this chat: `recall_turn`, `web_search`,
`delegate_subagent`, `jre_search`, `read`, `grep`, `edit`, `write_file`, `bash`,
and `register_project`.

Use tools actively when they are relevant. For any memory-reference question
about prior turns, user facts, family, location, body metrics, names, earlier
decisions, or wording such as "예전에", "그때", "했었지", "뭐였지", call
`recall_turn` before answering. For current external facts, news, weather, or
prices, call `web_search`. For complex multi-step investigation, call
`delegate_subagent`.

Do not stop after saying you will check. Make the structured tool call, read the
result, then answer naturally in English only.

Tool calls must use the structured function-calling channel only. Never print
tool names, function names, `to=...`, `analysis to=...`, `code`, `final`, or
argument JSON in user-facing content."""


def _preview_for_log(value: Any, limit: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    text = text.replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def _build_chat_turn(slim: JarvisAgentic, emit: Callable[[Outbound], None], conv_id: str, project_path: str | None, content_filter: _ToolArgContentFilter, think_parser: _ThinkTagParser, should_cancel: Callable[[], bool] | None = None) -> ChatTurn:
    dispatcher = get_dispatcher(conv_id=conv_id, storage_root=str(slim.jhb_root), project_root=project_path, retriever=slim.retriever)
    llm = get_llm("chat")

    def _emit_chat_chunk(text: str, kind: str) -> None:
        # Sync — these callbacks fire from AgenticLoop / encoder worker threads
        # via cb(text, kind) without await (loop.py:696, encoder.py:293,
        # chat_turn.py:340, slim.py:692). The `emit` parameter is in fact
        # `emit_from_thread` (ws.py:481-498), a sync wrapper around
        # asyncio.run_coroutine_threadsafe — making this async leaks coroutines
        # and silently drops every token (verified live 2026-05-10).
        if not text:
            return
        if os.environ.get("JLC_CHAT_CMD_MIRROR") == "1":
            color = "\033[34m" if kind == "reasoning" else "\033[33m"
            sys.stderr.write(f"{color}{text}\033[0m")
            sys.stderr.flush()
        if kind == "reasoning":
            emit({"type": "assistant_reasoning_token", "text": text})
        elif kind == "content":
            for piece in content_filter.feed(text):
                if piece:
                    emit({"type": "assistant_token", "text": piece})

    def on_token(text: str, kind: str = "content") -> None:
        if not text:
            return
        if kind == "reasoning":
            _emit_chat_chunk(text, "reasoning")
            return
        # `kind == "content"` — feed through the think-tag parser so models
        # that inline <think>...</think> in content (e.g. minimax-m2.5) still
        # surface the reasoning trace on the ReasoningCard channel.
        for piece, piece_kind in think_parser.feed(text):
            _emit_chat_chunk(piece, piece_kind)

    def on_encode_start() -> None:
        emit({"type": "jhb_block_started", "block_id": "encoding-live", "title": "JHB Update", "priority": "P2"})

    def on_encode_token(text: str, kind: str = "content") -> None:
        if kind == "content" and text:
            emit({"type": "jhb_block_token", "block_id": "encoding-live", "text": text})
        elif kind == "reasoning" and text:
            emit({"type": "encoder_reasoning_token", "block_id": "encoding-live", "text": text})

    def on_encode_done(updated_jhb: str) -> None:
        blocks = parse_jhb_blocks(updated_jhb)
        block = blocks[-1] if blocks else {"id": "encoding-live", "title": "JHB Update", "priority": "P2", "bullets": [], "token": 0}
        emit({"type": "jhb_block_done", "block_id": "encoding-live", "block": block})
        emit({"type": "jhb_token_update", "total_token": sum(b["token"] for b in blocks)})

    def on_tool_call(name: str, args: dict[str, Any]) -> None:
        sys.stderr.write(f"[jarvis-ui] tool_call name={name} args={_preview_for_log(args)}\n")
        sys.stderr.flush()

    def on_tool_result(name: str, result: dict[str, Any]) -> None:
        sys.stderr.write(f"[jarvis-ui] tool_result name={name} result={_preview_for_log(result)}\n")
        sys.stderr.flush()

    return ChatTurn(
        slim=slim,
        llm_client=llm,
        dispatcher=dispatcher,
        max_iter=20,
        on_token=on_token,
        on_tool_call=on_tool_call,
        on_tool_result=on_tool_result,
        on_encode_start=on_encode_start,
        on_encode_token=None,
        on_encode_done=on_encode_done,
        should_cancel=should_cancel,
    )


def _validate_inbound(data: Any) -> Inbound:
    if not isinstance(data, dict):
        raise ValueError("inbound websocket payload must be an object")
    msg_type = data.get("type")
    if msg_type not in {"user_input", "mixer_toggle", "mixer_set_custom_note", "cancel"}:
        return {"type": str(msg_type or "")}
    if msg_type == "cancel":
        return {"type": "cancel"}
    if msg_type == "user_input":
        return {
            "type": "user_input",
            "text": str(data.get("text") or ""),
            "turn_id": str(data.get("turn_id") or uuid.uuid4().hex),
        }
    if msg_type == "mixer_toggle":
        return {
            "type": "mixer_toggle",
            "key": str(data.get("key") or ""),
            "enabled": bool(data.get("enabled")),
        }
    return {"type": "mixer_set_custom_note", "text": str(data.get("text") or "")}


async def _send_error(emit: Callable[[Outbound], Any], exc: BaseException) -> None:
    traceback.print_exc(file=sys.stderr)
    try:
        result = emit({
            "type": "error_log",
            "level": "error",
            "text": f"{type(exc).__name__}: {exc}",
        })
        if hasattr(result, "__await__"):
            await result
    except Exception:
        sys.stderr.write("[jarvis-ui] failed to enqueue error_log\n")
        sys.stderr.flush()


async def _drain_send_queue(websocket: WebSocket, queue: asyncio.Queue[Outbound | None]) -> None:
    seq = 0
    debug = os.environ.get("JARVIS_UI_WS_DEBUG") == "1"
    while True:
        payload = await queue.get()
        if payload is None:
            return
        try:
            seq += 1
            payload["seq"] = seq
            if debug:
                text = str(payload.get("text", ""))
                preview = repr(text[:120] + ("..." if len(text) > 120 else ""))
                sys.stderr.write(f"[ws-send] seq={seq} type={payload.get('type')} text={preview}\n")
                sys.stderr.flush()
            await manager.send(websocket, payload)
            await asyncio.sleep(0)
        except Exception as exc:
            sys.stderr.write(f"[jarvis-ui] websocket send failed: {type(exc).__name__}: {exc}\n")
            sys.stderr.flush()
            raise


async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    conn = ui_state.connect()
    # JLC_UI_CONV_ID lets the launcher pin a fresh conv_id for clean tests
    # (no JHB contamination from prior runs). Default keeps the legacy
    # "conversation" so existing storage paths still resolve.
    conv_id = os.environ.get("JLC_UI_CONV_ID") or "conversation"
    project_path = None
    slim: JarvisAgentic | None = None
    send_queue: asyncio.Queue[Outbound | None] = asyncio.Queue()
    drain_task = asyncio.create_task(_drain_send_queue(websocket, send_queue))
    content_filter = _ToolArgContentFilter()
    think_parser = _ThinkTagParser()
    cancel_event = threading.Event()
    turn_task: asyncio.Task[None] | None = None

    async def emit(payload: Outbound) -> None:
        await send_queue.put(payload)

    try:
        slim = get_slim()
        current_jhb = slim.load_jhb(conv_id)
        blocks = parse_jhb_blocks(current_jhb)
        jhb_token = sum(block["token"] for block in blocks)
        refresh_mixer_tokens(ui_state.mixer, jhb_token=jhb_token)
        await emit({"type": "init", "jhb_blocks": blocks, "mixer": ui_state.mixer, "turn": ui_state.turn, "jhb_token": jhb_token})

        loop = asyncio.get_running_loop()

        def emit_from_thread(payload: Outbound) -> None:
            def log_enqueue_error(fut: asyncio.Future[None]) -> None:
                try:
                    exc = fut.exception()
                except Exception as err:
                    exc = err
                if exc is not None:
                    sys.stderr.write(f"[jarvis-ui] websocket enqueue failed: {exc}\n")
                    sys.stderr.flush()

            try:
                future = asyncio.run_coroutine_threadsafe(send_queue.put(payload), loop)
                future.add_done_callback(log_enqueue_error)
            except Exception as exc:
                sys.stderr.write(f"[jarvis-ui] websocket enqueue failed: {type(exc).__name__}: {exc}\n")
                sys.stderr.flush()

        chat_turn = _build_chat_turn(slim, emit_from_thread, conv_id, project_path, content_filter, think_parser, should_cancel=cancel_event.is_set)

        async def _execute_turn(text: str, turn_id: str) -> None:
            try:
                conn.messages.append({"role": "user", "content": text})
                refresh_mixer_tokens(ui_state.mixer, message_text=text, jhb_token=jhb_token)
                await emit({"type": "user_echo", "text": text, "turn_id": turn_id})

                def run_turn() -> dict[str, Any]:
                    prior_history = [{"role": "system", "content": UI_TOOL_SYSTEM_MESSAGE}]
                    mixer_note = _mixer_system_message()
                    if mixer_note:
                        prior_history.append({"role": "system", "content": mixer_note})
                    return chat_turn.run(text, conv_id=conv_id, project_path=project_path, prior_history=prior_history)

                result = await asyncio.to_thread(run_turn)
                final = str(result.get("final_message") or "")
                conn.messages.append({"role": "assistant", "content": final})
                for piece, piece_kind in think_parser.flush():
                    if not piece:
                        continue
                    if piece_kind == "reasoning":
                        await emit({"type": "assistant_reasoning_token", "text": piece})
                    else:
                        for sub in content_filter.feed(piece):
                            if sub:
                                await emit({"type": "assistant_token", "text": sub})
                for piece in content_filter.flush():
                    if piece:
                        await emit({"type": "assistant_token", "text": piece})
                if result.get("halt_reason") == "cancelled":
                    await emit({"type": "error_log", "level": "warn", "text": "Cancelled by user"})
                current_turn = ui_state.next_turn()
                await emit({"type": "assistant_done", "turn_id": turn_id, "turn": current_turn})
            except Exception as exc:
                await _send_error(emit, exc)
                try:
                    await emit({"type": "assistant_done", "turn_id": turn_id, "turn": ui_state.turn})
                except Exception:
                    pass

        while True:
            try:
                data = _validate_inbound(await websocket.receive_json())
                msg_type = data.get("type")
                if msg_type == "mixer_toggle":
                    key = str(data.get("key") or "")
                    if key in {"english_only", "custom_note_enabled"}:
                        ui_state.mixer[key] = bool(data.get("enabled"))  # type: ignore[literal-required]
                        refresh_mixer_tokens(ui_state.mixer, jhb_token=jhb_token)
                        save_mixer(ui_state.mixer)
                        await emit({"type": "init", "jhb_blocks": blocks, "mixer": ui_state.mixer, "turn": ui_state.turn, "jhb_token": jhb_token})
                    continue
                if msg_type == "mixer_set_custom_note":
                    ui_state.mixer["custom_note"] = str(data.get("text") or "")
                    ui_state.mixer["custom_note_enabled"] = bool(ui_state.mixer["custom_note"])
                    refresh_mixer_tokens(ui_state.mixer, jhb_token=jhb_token)
                    save_mixer(ui_state.mixer)
                    await emit({"type": "init", "jhb_blocks": blocks, "mixer": ui_state.mixer, "turn": ui_state.turn, "jhb_token": jhb_token})
                    continue
                if msg_type == "cancel":
                    if turn_task and not turn_task.done():
                        cancel_event.set()
                    continue
                if msg_type != "user_input":
                    await emit({"type": "error_log", "level": "warn", "text": f"unknown message type: {msg_type}"})
                    continue

                text = str(data.get("text") or "")
                if not text.strip():
                    continue
                if turn_task and not turn_task.done():
                    await emit({"type": "error_log", "level": "warn", "text": "Previous turn still running"})
                    continue
                turn_id = str(data.get("turn_id") or uuid.uuid4().hex)
                cancel_event.clear()
                turn_task = asyncio.create_task(_execute_turn(text, turn_id))
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                await _send_error(emit, exc)
                continue
    except WebSocketDisconnect:
        ui_state.disconnect(conn.ws_id)
    except Exception as exc:
        await _send_error(emit, exc)
    finally:
        ui_state.disconnect(conn.ws_id)
        if turn_task is not None and not turn_task.done():
            cancel_event.set()
            try:
                await asyncio.wait_for(turn_task, timeout=5)
            except Exception:
                turn_task.cancel()
        try:
            await send_queue.put(None)
            await asyncio.wait_for(drain_task, timeout=5)
        except Exception:
            drain_task.cancel()
        if slim is not None:
            await slim.close()
