"""JLC mini-jhb encoder."""
from __future__ import annotations

import os
import re
import sys
import threading
import unicodedata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Force line-buffered stderr so background-thread emissions (encoder progress
# markers + the per-turn `[jlc:meter]` line) surface immediately instead of
# sitting in the pipe buffer until the next stdin event flushes them.
# Without this, harnesses that capture child stdout/stderr via subprocess.PIPE
# see meter lines appear in the NEXT turn's raw region (verified in
# 1000bench dryrun_view 2026-05-03). PYTHONUNBUFFERED=1 doesn't fix this in
# Python 3 because stderr buffering of redirected pipes is decided by the
# TextIOWrapper, not the env var.
try:
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass


def _is_verbose_encoder() -> bool:
    """Idea #12 step 7 console hygiene — verbose mode opens up encoder
    progress markers + raw token stream to stderr. Default is QUIET so
    background-encoder output does not interleave with the live chat
    stream and confuse the user. Opt-in for dev/debug:
        $env:JLC_ENCODER_VERBOSE = "1"
    """
    return os.environ.get("JLC_ENCODER_VERBOSE") == "1"


def _compact_tokens(value: int) -> str:
    value = max(0, int(value))
    if value < 1000:
        return str(value)
    return f"{value / 1000:.1f}k"


# REASONING BUDGET ↔ EXPOSURE toggle. The constitution ships with the
# BUDGET block (fast, deterministic) wrapped in HTML markers; setting
# `JLC_ENCODER_REASONING_EXPOSE=1` swaps it for the EXPOSURE block so the
# encoder LLM emits a `<think>...</think>` trace (slower, dev/debug only).
# Either way the markers themselves are stripped before the prompt reaches
# the model.
_REASONING_EXPOSURE_BLOCK = """REASONING EXPOSURE

Expose your reasoning in a `<think>...</think>` block before emitting the
output delimiters. Spend whatever reasoning you need to make a correct
delta patch — quality of the JHB silhouette matters more than speed at
this stage. If uncertain, preserve prev_jhb rather than fabricating."""

_REASONING_MODE_BLOCK_RE = re.compile(
    r"<!-- REASONING_MODE:BUDGET[^>]*-->\n?(?P<body>.*?)\n?<!-- /REASONING_MODE -->",
    re.DOTALL,
)


def _is_reasoning_expose() -> bool:
    return os.environ.get("JLC_ENCODER_REASONING_EXPOSE", "0") == "1"


def _apply_reasoning_mode(text: str) -> str:
    if _is_reasoning_expose():
        return _REASONING_MODE_BLOCK_RE.sub(_REASONING_EXPOSURE_BLOCK, text, count=1)
    return _REASONING_MODE_BLOCK_RE.sub(lambda m: m.group("body"), text, count=1)


def _emit_progress(msg: str) -> None:
    """Encoder progress markers (`encoding attempt N/5`, `succeeded`,
    BEGIN/END STREAM, etc.). Verbose-only on the console — silent by
    default so the user sees only the chat answer + 1-line red meter.

    Always mirrors to encoder.log when JLC_ENCODER_LOG is set so the
    UI / analysis still has the full trail.
    """
    if _is_verbose_encoder():
        print(msg, file=sys.stderr)
    _mirror_to_log(msg)


def _log_post_encode(msg: str) -> None:
    """Idea #12 step 6 — emit a post-encode meter line.

    Always prints to stderr in ANSI red so the 1-liner stands out from
    the chat answer's stream and the user can spot it as the per-turn
    encode boundary. When JLC_ENCODER_LOG is set, also mirrors plain
    text (no color codes) to the log file so analysis is grep-clean.

    JLC_ENCODER_LOG values:
      unset  → stderr only (default).
      "1"    → also append to ~/.jarvis-code/encoder.log.
      <path> → also append to <path>.
    """
    # ANSI bright red wrap for terminal visibility; plain text for the
    # file mirror.
    print(f"\x1b[91m{msg}\x1b[0m", file=sys.stderr)
    _mirror_to_log(msg)


def _emit_encoder_raw_dump(raw_output: str, *, attempt: int, streamed_chunks: int) -> None:
    if os.environ.get("JLC_ENCODER_SIDECAR_OUTPUT", "1") == "0":
        return
    print(
        f"[jlc:enc:content] ---BEGIN RAW attempt={attempt} chunks={streamed_chunks}---",
        file=sys.stderr,
        flush=True,
    )
    print(raw_output, file=sys.stderr, flush=True)
    print("[jlc:enc:content] ---END RAW---", file=sys.stderr, flush=True)
    _mirror_to_log(f"[jlc:enc:content] ---BEGIN RAW attempt={attempt} chunks={streamed_chunks}---")
    _mirror_to_log(raw_output)
    _mirror_to_log("[jlc:enc:content] ---END RAW---")


def _mirror_to_log(msg: str) -> None:
    """Best-effort file mirror shared by _emit_progress and
    _log_post_encode. File IO failures cannot wedge an encode.
    """
    flag = os.environ.get("JLC_ENCODER_LOG")
    if not flag:
        return
    try:
        if flag == "1":
            target = Path.home() / ".jarvis-code" / "encoder.log"
        else:
            target = Path(flag)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(msg)
            fh.write("\n")
            fh.flush()
    except Exception:
        pass

try:
    import tiktoken
except ImportError:  # tiktoken is optional; count_tokens falls back to whitespace split
    tiktoken = None  # type: ignore[assignment]

from .llm_client import LLMClient


class _EncoderThinkTagParser:
    """Split a streaming content channel into reasoning (inside <think>...
    </think>) and user-facing content (outside). Mirrors ws.py's parser so
    the encoder cmd_mirror can colorize reasoning even when providers
    (Ollama Cloud GLM-5, minimax-m2.5, etc.) inline reasoning as <think>
    tags inside delta.content rather than emitting a separate
    reasoning_content field. Robust to chunks that split a tag across
    stream boundaries.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"
    _OPEN_TAIL = len(_OPEN) - 1
    _CLOSE_TAIL = len(_CLOSE) - 1

    def __init__(self) -> None:
        self._in_think = False
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
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
        if not self._buf:
            return []
        kind = "reasoning" if self._in_think else "content"
        out = [(self._buf, kind)]
        self._buf = ""
        self._in_think = False
        return out


@dataclass
class TailEntry:
    turn_id: str
    summary: str
    token_count: int
    created_at: float


RECENT_TAIL_SYSTEM_PROMPT = """You compress the most recent N conversation turns in chronological order. The goal is to keep the next turn from losing the immediately preceding flow.

Priority (high -> low):
1. Open questions the assistant put to the user
2. Promises, decisions, conditions ("if X then I'll do Y", "proceed after confirming Z")
3. Coding thread — in-flight file paths, functions/symbols, errors hit, fixes applied, pending edits, next actions
4. Emotional nuance (frustration, urgency, satisfaction)
5. One or two key facts (only when load-bearing for the next turn)

Drop entirely:
- Greetings and acknowledgements ("OK", "yep", "thanks", "got it")
- Restatements of facts already in the JHB
- Generic encouragement or praise
- Raw tool-call args (keep only the name and a result summary)

Format:
- Output exactly one line per input turn.
- Allowed output is only N lines of `{turn_id}: {one-line prose}`.
- No extra explanation, markdown, lists, or code blocks.
- Preserve the input order in the output.

Preserve a little more next-turn-load-bearing detail for the most recent turns; older turns may be compressed harder. Mirror the user's language; keep code tokens verbatim."""


class JLCEncoder:
    """Compress conversation turns into an updated mini jhb."""

    _tiktoken_enc: Any = None
    _tiktoken_init_lock = threading.Lock()

    def __init__(self, llm: LLMClient, prompt_path: Path | None = None, target_tokens: int = 2000) -> None:
        self.llm = llm
        self.prompt_path = prompt_path
        self.target_tokens = target_tokens
        self._cached_system_prompt: str | None = None
        # Token-meter handoff — populated each encode(); chat counts are
        # injected before encode runs so the meter line printed
        # at encode-end shows chat + encoder side by side.
        self.last_enc_in: int = 0
        self.last_enc_out: int = 0
        self.last_enc_think: int = 0  # paid encoder reasoning tokens
        self.last_enc_seconds: float = 0.0
        # idea #12 step 4: chat-turn id whose encode produced these last_*
        # values. Set by slim._encode_and_save_locked after encode returns
        # so the bench meter can attribute lagged encodes correctly.
        self.last_enc_turn_id: int | None = None
        self.last_tail_in: int = 0
        self.last_tail_out: int = 0
        self.last_tail_think: int = 0
        self.last_tail_seconds: float = 0.0
        self.last_jhb_tokens: int = 0
        self.last_jhb_target: int = target_tokens
        self.last_jhb_delta: int = 0
        self.last_retries: int = 0
        self.last_chat_in: int = 0
        self.last_chat_in_user: int = 0
        self.last_chat_out: int = 0
        self.last_chat_cache_read: int = 0
        self.last_chat_cache_write: int = 0
        self.last_chat_think: int = 0  # paid reasoning tokens (chat LLM)
        self.last_chat_seconds: float = 0.0
        self.last_enc_blocking_s: float = 0.0
        self.last_chat_in_breakdown: dict = {}  # per-field chat[in] breakdown
        self.last_chat_prompt_context: dict = {}
        self.last_subagent_in: int = 0
        self.last_subagent_out: int = 0
        self.last_subagent_think: int = 0
        self.last_subagent_seconds: float = 0.0
        self.last_chat_turn_in: int = 0
        self.last_chat_turn_out: int = 0
        self.last_chat_turn_cache_read: int = 0
        self.last_chat_turn_cache_write: int = 0
        self.last_chat_turn_think: int = 0
        self.last_chat_turn_seconds: float = 0.0
        self.last_subagent_turn_in: int = 0
        self.last_subagent_turn_out: int = 0
        self.last_subagent_turn_think: int = 0
        self.last_subagent_turn_seconds: float = 0.0
        self.last_error: str | None = None

    def format_post_encode_meter_line(self) -> str:
        """Return the canonical per-turn meter line without emitting it."""
        delta_sign = "+" if self.last_jhb_delta >= 0 else ""
        chat_cache = self.last_chat_turn_cache_read + self.last_chat_turn_cache_write
        chat_io = self.last_chat_turn_in + self.last_chat_turn_out
        enc_total = self.last_enc_in + self.last_enc_think + self.last_enc_out
        return (
            f"[jlc:meter] chat[in={_compact_tokens(self.last_chat_turn_in)} "
            f"out={_compact_tokens(self.last_chat_turn_out)} in+out={_compact_tokens(chat_io)} "
            f"cached={_compact_tokens(chat_cache)} "
            f"{self.last_chat_turn_seconds:.1f}s] | "
            f"encoder[in={_compact_tokens(self.last_enc_in)} out={_compact_tokens(self.last_enc_out)} "
            f"total={_compact_tokens(enc_total)} {self.last_enc_seconds:.1f}s] | "
            f"jhb={self.last_jhb_tokens}/{self.last_jhb_target} ({delta_sign}{self.last_jhb_delta})"
        )

    def _record_encode_meter(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        raw_output: str,
        think_parts: list[str],
        started_at: float,
        jhb: str,
        prev_jhb: str,
        retries: int,
        emit: bool = True,
    ) -> None:
        self.last_enc_in = self.count_tokens(system_prompt) + self.count_tokens(user_prompt)
        self.last_enc_out = self.count_tokens(raw_output)
        self.last_enc_think = self.count_tokens("".join(think_parts))
        try:
            self.last_enc_seconds = time.time() - started_at
        except Exception:
            self.last_enc_seconds = 0.0
        self.last_jhb_tokens = self.count_tokens(jhb)
        self.last_jhb_delta = self.last_jhb_tokens - self.count_tokens(prev_jhb)
        self.last_retries = retries

        if emit:
            _log_post_encode(self.format_post_encode_meter_line())

    def _load_system_prompt(self) -> str:
        # NOTE: not cached — `{{TODAY}}` must be re-rendered every call so
        # the encoder LLM sees the actual session date instead of a stale
        # value pinned at first import.
        if self.prompt_path is not None:
            text = Path(self.prompt_path).read_text(encoding="utf-8")
        else:
            from importlib.resources import files
            text = (files("jlc_agentic") / "prompts" / "encoder_system.md").read_text(encoding="utf-8")
        text = _apply_reasoning_mode(text)
        from datetime import datetime
        today_iso = datetime.now().strftime("%Y-%m-%d")
        return text.replace("{{TODAY}}", today_iso)

    async def encode(
        self,
        prev_jhb: str,
        user_msg: str,
        assistant_msg: str,
        prev_project_md: str = "",
        project_active: bool = False,
        target_tokens: int | None = None,
        on_token: Callable[[str], None] | Callable[[str, str], None] | None = None,
        batch_turns: list[dict] | None = None,
        current_turn: int | None = None,
        origin: str = "user",
        origin_window: str | None = None,
        origin_window_label: str | None = None,
    ) -> tuple[str, str, int]:
        """Return (updated mini-jhb markdown, previous project markdown, retry_count).

        The encoder updates only conversation-level JHB memory. Project
        memory updates are handled by the explicit project-memory path, so this
        method always returns the previous project markdown unchanged.

        target_tokens overrides self.target_tokens for this call only — used by
        JarvisAgentic to expand the JHB budget when a project is active (coding mode)
        and shrink it for plain conversation. None = use the instance default.
        """
        import asyncio

        t0 = time.time()
        active_target = target_tokens if (target_tokens and target_tokens > 0) else self.target_tokens
        self.last_jhb_target = active_target
        self.last_error = None
        try:
            from .diff import normalize_stored_jhb  # noqa: PLC0415

            prev_jhb = normalize_stored_jhb(prev_jhb or "")
        except Exception:
            prev_jhb = prev_jhb or ""
        last_good = prev_jhb or ""
        last_good_project = prev_project_md or ""

        try:
            system_prompt = self._load_system_prompt()
        except Exception as exc:  # noqa: BLE001
            self.last_error = str(exc)
            print(f"[jlc:enc:bg] prompt load failed: {exc}", file=sys.stderr)
            return last_good, last_good_project, 0

        # ENGLISH-ONLY constraint for benchmark sessions. JLC_BENCH_PORT
        # is set by the bench harness; when present, prefix the encoder
        # system prompt with a hard language constraint so JHB updates never
        # leak Korean/Japanese/Chinese tokens that would corrupt the run's
        # transcript.
        if os.environ.get("JLC_BENCH_PORT"):
            system_prompt = (
                "CRITICAL LANGUAGE CONSTRAINT — non-negotiable for this benchmark.\n"
                "Write the JHB block in ENGLISH ONLY.\n"
                "- NO Korean (한국어), Japanese (日本語), Chinese (中文).\n"
                "- NO mixed-language phrasing even briefly.\n"
                "- If the source user/assistant text contains non-English words "
                "or proper nouns, paraphrase or transliterate so the output "
                "remains entirely English.\n"
                "- Applies to all output: section headings, bullets, summaries, "
                "and any reasoning the model emits.\n\n"
                + system_prompt
            )

        if batch_turns:
            user_prompt = self._build_batch_user_prompt(
                prev_jhb, batch_turns, prev_project_md, project_active, current_turn,
            )
        else:
            user_prompt = self._build_user_prompt(
                prev_jhb,
                user_msg,
                assistant_msg,
                prev_project_md,
                project_active,
                current_turn,
                origin=origin,
                origin_window=origin_window,
                origin_window_label=origin_window_label,
            )

        # Aggressive cap: encoder runs under per-conv lock that blocks the next
        # turn. Old 60→600s × 10 = 71min. New 30→120s × 5 ≈ 6min worst-case.
        # LLMClient does its own retry on top, so this is the OUTER bound.
        backoff = 5
        max_backoff = 30
        max_attempts = 5
        attempt = 0
        raw_output = ""
        # Encoder output is not user-facing. Keep the logical encoder call
        # non-streaming regardless of environment; provider adapters may still
        # use an internal stream transport when a backend requires it.
        stream_enabled = False
        # Disable colors on dumb terminals or when explicitly turned off.
        color_enabled = (
            stream_enabled
            and os.environ.get("JLC_ENCODER_COLOR", "1") != "0"
            and os.environ.get("NO_COLOR") in (None, "")
        )
        # 256-color ANSI — soft purple for <think>, soft gold for the answer
        # block. Picked for a black terminal background; both have ~70-80%
        # luminance so they don't fatigue at long stretches.
        _C_THINK = "\x1b[38;5;177m"  # encoder thinking
        _C_ANS = "\x1b[38;5;221m"    # encoder answer
        _C_RESET = "\x1b[0m"

        # Accumulate the encoder's reasoning stream so we can bill it honestly
        # in the meter (encoder thinking is a paid output stream too).
        think_parts: list[str] = []
        streamed_chunk_count = 0

        # The encoder is a background cache writer, so never mirror chunks to
        # the sidecar console. Summary/meter lines remain visible.
        cmd_mirror = False

        # Inline <think>...</think> splitter for providers that don't emit a
        # separate delta.reasoning_content channel (Ollama Cloud GLM-5,
        # minimax-m2.5, etc.). Re-created per encode() call so retries don't
        # carry stale buffer state.
        think_parser = _EncoderThinkTagParser()

        def _emit_piece(piece_text: str, piece_kind: str) -> None:
            if not piece_text:
                return
            if piece_kind == "reasoning":
                think_parts.append(piece_text)
            if on_token is not None:
                try:
                    on_token(piece_text, piece_kind)  # type: ignore[misc]
                except TypeError:
                    on_token(piece_text)  # type: ignore[misc]
            if cmd_mirror:
                if color_enabled:
                    color = _C_THINK if piece_kind == "reasoning" else _C_ANS
                    sys.stderr.write(f"{color}{piece_text}{_C_RESET}")
                else:
                    sys.stderr.write(piece_text)
                sys.stderr.flush()

        def _on_chunk(text: str, kind: str = "content") -> None:
            nonlocal streamed_chunk_count
            try:
                streamed_chunk_count += 1
                if kind == "reasoning":
                    # Provider emitted a dedicated reasoning channel — trust
                    # it and bypass the inline <think> splitter.
                    _emit_piece(text, "reasoning")
                else:
                    # delta.content — may carry inline <think>...</think>.
                    for piece_text, piece_kind in think_parser.feed(text):
                        _emit_piece(piece_text, piece_kind)
            except Exception:
                pass

        def _flush_think_parser() -> None:
            try:
                for piece_text, piece_kind in think_parser.flush():
                    _emit_piece(piece_text, piece_kind)
            except Exception:
                pass

        while attempt < max_attempts:
            attempt += 1
            try:
                _emit_progress(f"[jlc:enc:bg] encoding attempt {attempt}/{max_attempts}...")
                if stream_enabled:
                    _emit_progress("[jlc:enc-raw] ---BEGIN STREAM---")
                    raw_output = await self.llm.chat(
                        system=system_prompt,
                        user=user_prompt,
                        max_tokens=4096,
                        on_chunk=_on_chunk,
                    )
                    _flush_think_parser()
                    _emit_progress("\n[jlc:enc-raw] ---END STREAM---")
                else:
                    raw_output = await self.llm.chat(
                        system=system_prompt,
                        user=user_prompt,
                        max_tokens=4096,
                    )
            except Exception as exc:  # noqa: BLE001
                self.last_error = str(exc)
                print(f"[jlc:enc-retry] attempt {attempt}/{max_attempts} failed: {exc}, waiting {backoff}s...", file=sys.stderr)
                if attempt >= max_attempts:
                    print(f"[jlc:enc:bg] GIVE UP after {max_attempts} attempts, keeping previous jhb", file=sys.stderr)
                    self._record_encode_meter(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_output=raw_output,
                        think_parts=think_parts,
                        started_at=t0,
                        jhb=last_good,
                        prev_jhb=prev_jhb,
                        retries=attempt - 1,
                    )
                    return last_good, last_good_project, attempt - 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            raw_output = unicodedata.normalize("NFC", raw_output).strip()
            if raw_output and streamed_chunk_count == 0:
                _emit_encoder_raw_dump(raw_output, attempt=attempt, streamed_chunks=streamed_chunk_count)
            if not raw_output:
                self.last_error = "encoder returned empty output"
                print(f"[jlc:enc-retry] attempt {attempt}/{max_attempts} empty output, waiting {backoff}s...", file=sys.stderr)
                if attempt >= max_attempts:
                    print(f"[jlc:enc:bg] GIVE UP after {max_attempts} attempts (empty), keeping previous jhb", file=sys.stderr)
                    self._record_encode_meter(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        raw_output=raw_output,
                        think_parts=think_parts,
                        started_at=t0,
                        jhb=last_good,
                        prev_jhb=prev_jhb,
                        retries=attempt - 1,
                    )
                    return last_good, last_good_project, attempt - 1
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue

            break
        else:
            print("[jlc:enc:bg] loop exhausted, keeping previous jhb", file=sys.stderr)
            self._record_encode_meter(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                raw_output=raw_output,
                think_parts=think_parts,
                started_at=t0,
                jhb=last_good,
                prev_jhb=prev_jhb,
                retries=max_attempts - 1,
            )
            return last_good, last_good_project, max_attempts - 1

        self.last_error = None
        _emit_progress(f"[jlc:enc:bg] encoding succeeded on attempt {attempt}")

        # Skip the post-hoc dump when streaming is on (already shown live).
        # Default OFF: cmd window stays clean during UI mode; the JHB content
        # surfaces via on_token callbacks → UI's JHBPreview panel. Set
        # JLC_ENCODER_VERBOSE=1 to debug encoder output in the cmd window.
        if not stream_enabled and os.environ.get("JLC_ENCODER_VERBOSE", "0") == "1":
            print(f"[jlc:enc-raw] ---BEGIN RESPONSE (attempt {attempt})---", file=sys.stderr)
            print(raw_output, file=sys.stderr)
            print("[jlc:enc-raw] ---END RESPONSE---", file=sys.stderr)

        new_jhb_raw = self._parse_jhb_output(raw_output)
        new_project_md = last_good_project
        if not new_jhb_raw:
            print("[jlc:enc:bg] no JHB block parsed; keeping previous jhb", file=sys.stderr)
            new_jhb = last_good
        else:
            new_jhb = self._apply_diff_or_fallback(prev_jhb, new_jhb_raw)
            new_jhb = self._auto_keep_missing_sections(prev_jhb, new_jhb)  # W2.9.20
            new_jhb = self._enforce_priority_fifo(new_jhb)

        tok_count = self.count_tokens(new_jhb)

        if tok_count > active_target:
            print(f"[jlc:enc:bg] over budget ({tok_count}>{active_target}), enforcing...", file=sys.stderr)
            new_jhb = await self._enforce_budget(new_jhb, system_prompt, target=active_target)
            tok_count = self.count_tokens(new_jhb)
            print(f"[jlc:enc:bg] post-enforcement: {tok_count} tokens", file=sys.stderr)

        section_count = len(re.findall(r"^## ", new_jhb, re.MULTILINE))
        if section_count > 20:
            print(
                f"[jlc:enc:bg] section count over soft cap ({section_count}>20); keeping encoder result",
                file=sys.stderr,
            )

        self._record_encode_meter(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            think_parts=think_parts,
            started_at=t0,
            jhb=new_jhb,
            prev_jhb=prev_jhb,
            retries=attempt - 1,
        )

        return new_jhb, new_project_md, attempt - 1

    def _apply_diff_or_fallback(self, prev_jhb: str, jhb_block: str) -> str:
        """Apply the encoder's JHB diff DSL.

        The encoder is now delta-only for JHB: malformed patches keep the
        previous JHB instead of treating the block as a full rewrite. This
        preserves the existing P0-P3/FIFO policy while removing the slow and
        risky whole-file regeneration path.
        """
        from .diff import DiffDriftError, DiffParseError, apply_diff

        try:
            applied = apply_diff(prev_jhb or "", jhb_block)
            return applied
        except DiffDriftError as exc:
            print(f"[jlc:enc:bg] diff drift, keeping previous jhb: {exc}", file=sys.stderr)
            return prev_jhb or ""
        except DiffParseError as exc:
            print(f"[jlc:enc:bg] diff parse failed, keeping previous jhb: {exc}", file=sys.stderr)
            return prev_jhb or ""

    def _auto_keep_missing_sections(self, prev_jhb: str, new_jhb: str) -> str:
        """Defensive guard against encoder silent evict — P0/P1 protected only.

        Compute the set of `## H2` section titles in prev_jhb (excluding the
        reserved `## Conversation Tail`) and in new_jhb. A prev-section is
        considered protected only if its priority tag is P0 or P1; missing
        protected sections are re-appended (silent-evict block). P2/P3 (and
        untagged) sections may be evicted by encoder judgment — the retriever
        recovers their detail from the turn log on demand.

        Constitution §3 reaffirmed: encoders compress, do not arbitrate.
        P2/P3 eviction is compression (a normal part of the lifecycle), not
        the silent overwrite or chat-hallucination laundering that §3
        forbids. P0/P1 are the active surface — drops there ARE silent-evict
        bugs and are blocked here. v1 paper §3.5 explicitly classifies
        evicted sections as "recoverable via retriever".
        """
        from .diff import parse_jhb_sections

        prev_sections = parse_jhb_sections(prev_jhb or "")
        new_sections = parse_jhb_sections(new_jhb or "")

        # Extract priority from each prev section's first heading line.
        # Only P0 and P1 sections are protected from silent evict.
        protected_titles: set[str] = set()
        for title, markdown in prev_sections:
            if title.strip() == "Conversation Tail":
                continue
            first_line = markdown.split("\n", 1)[0]
            prio_match = re.search(r"\[(P[01])\]\s*$", first_line)
            if prio_match:
                protected_titles.add(title)

        new_titles = {title for title, _ in new_sections}
        missing = protected_titles - new_titles
        if not missing:
            return new_jhb

        salvaged = [markdown.rstrip(" \t\n") for title, markdown in prev_sections if title in missing]
        if not salvaged:
            return new_jhb

        salvage_block = "\n\n".join(salvaged)
        tail_match = re.search(r"^## Conversation Tail\s*$", new_jhb, re.MULTILINE)
        if tail_match:
            before_tail = new_jhb[: tail_match.start()].rstrip(" \t\n")
            tail = new_jhb[tail_match.start() :].lstrip("\n")
            rendered = f"{before_tail}\n\n{salvage_block}\n\n{tail}".strip(" \t\n") + "\n"
        else:
            trimmed_jhb = new_jhb.rstrip(" \t\n")
            rendered = f"{trimmed_jhb}\n\n{salvage_block}\n"

        print(
            f"[jlc:enc:bg] auto-keep: salvaged {len(missing)} silent-evicted section(s): "
            f"{sorted(missing)}",
            file=sys.stderr,
        )
        return rendered

    _PRIORITY_CAPS = {"P0": 20, "P1": 10, "P2": 10, "P3": 10}
    _PRIORITY_NEXT = {"P0": "P1", "P1": "P2", "P2": "P3", "P3": None}
    _PRIORITY_RE = re.compile(r"\[(P[0-3])\]\s*$")
    _TURN_TAG_RE = re.compile(r"\(t(\d+)\)")
    _BULLET_RE = re.compile(r"^\s*-\s+")

    def _enforce_priority_fifo(self, jhb: str) -> str:
        sections = self._parse_priority_sections(jhb)
        if not sections:
            return jhb
        moved = 0
        dropped = 0
        empty_removed = any(section.get("priority") and not section.get("items") for section in sections)
        for level in ("P0", "P1", "P2", "P3"):
            cap = self._PRIORITY_CAPS[level]
            while True:
                bullets = self._priority_bullets(sections, level)
                if len(bullets) <= cap:
                    break
                section_idx, item_idx, _item = bullets[0]
                item = sections[section_idx]["items"].pop(item_idx)
                next_level = self._PRIORITY_NEXT[level]
                if next_level is None:
                    dropped += 1
                    continue
                target_idx = self._priority_target_section(sections, next_level)
                if target_idx is None:
                    sections[section_idx]["items"].insert(item_idx, item)
                    break
                sections[target_idx]["items"].append(item)
                moved += 1
        if not moved and not dropped and not empty_removed:
            return jhb
        print(
            f"[jlc:enc:bg] priority FIFO normalized: moved={moved} dropped={dropped} empty_removed={int(empty_removed)}",
            file=sys.stderr,
        )
        return self._render_priority_sections(sections)

    def _parse_priority_sections(self, jhb: str) -> list[dict[str, Any]]:
        from .diff import parse_jhb_sections

        parsed = parse_jhb_sections(jhb or "")
        sections: list[dict[str, Any]] = []
        order = 0
        for _title, markdown in parsed:
            lines = markdown.rstrip(" \t\n").splitlines()
            if not lines:
                continue
            heading = lines[0]
            match = self._PRIORITY_RE.search(heading)
            priority = match.group(1) if match else None
            items: list[dict[str, Any]] = []
            current_bullet: dict[str, Any] | None = None
            for line in lines[1:]:
                if self._BULLET_RE.match(line):
                    current_bullet = {"kind": "bullet", "lines": [line], "order": order}
                    order += 1
                    items.append(current_bullet)
                    continue
                if current_bullet is not None and (line.startswith(" ") or line.startswith("\t") or not line.strip()):
                    current_bullet["lines"].append(line)
                    continue
                current_bullet = None
                items.append({"kind": "other", "lines": [line], "order": order})
                order += 1
            sections.append({"heading": heading, "priority": priority, "items": items})
        return sections

    def _priority_bullets(self, sections: list[dict[str, Any]], level: str) -> list[tuple[int, int, dict[str, Any]]]:
        bullets: list[tuple[int, int, dict[str, Any]]] = []
        for section_idx, section in enumerate(sections):
            if section.get("priority") != level:
                continue
            for item_idx, item in enumerate(section.get("items") or []):
                if item.get("kind") == "bullet":
                    bullets.append((section_idx, item_idx, item))
        bullets.sort(key=lambda entry: (self._bullet_turn(entry[2]), int(entry[2].get("order") or 0)))
        return bullets

    def _bullet_turn(self, item: dict[str, Any]) -> int:
        text = "\n".join(str(line) for line in item.get("lines") or [])
        match = self._TURN_TAG_RE.search(text)
        if not match:
            return -1
        try:
            return int(match.group(1))
        except ValueError:
            return -1

    def _priority_target_section(self, sections: list[dict[str, Any]], level: str) -> int | None:
        for idx, section in enumerate(sections):
            if section.get("priority") == level:
                return idx
        if len(sections) >= 20:
            return None
        sections.append({"heading": f"## FIFO {level} [{level}]", "priority": level, "items": []})
        return len(sections) - 1

    @staticmethod
    def _render_priority_sections(sections: list[dict[str, Any]]) -> str:
        rendered_sections: list[str] = []
        for section in sections:
            if section.get("priority") and not section.get("items"):
                continue
            lines = [str(section["heading"]).rstrip()]
            for item in section.get("items") or []:
                lines.extend(str(line).rstrip() for line in item.get("lines") or [])
            rendered_sections.append("\n".join(lines).rstrip(" \t\n"))
        return "\n\n".join(part for part in rendered_sections if part.strip()).strip(" \t\n") + "\n"

    async def compress_recent_turns(
        self,
        turns: list[dict[str, Any]],
        *,
        max_tokens_per_turn: int = 220,
    ) -> list[TailEntry]:
        """Thread-aware compression for the recent conversation tail.

        This is intentionally separate from fact-mode JHB encoding: it keeps
        open questions, promises, in-flight coding context, and pending next
        actions even when they would be too transient for the durable JHB.
        """
        prepared: list[dict[str, str]] = []
        for item in turns:
            if not isinstance(item, dict):
                continue
            turn_id = str(item.get("turn", item.get("turn_id", ""))).strip()
            if not turn_id:
                continue
            safe_user = self._truncate(
                self._neutralize_delimiters(unicodedata.normalize("NFC", str(item.get("user") or ""))),
                3000,
            )
            safe_assistant = self._truncate(
                self._neutralize_delimiters(unicodedata.normalize("NFC", str(item.get("assistant") or ""))),
                5000,
            )
            prepared.append({"turn_id": turn_id, "user": safe_user, "assistant": safe_assistant})
        if not prepared:
            return []

        max_tokens_per_turn = max(1, int(max_tokens_per_turn or 220))
        blocks = []
        for item in prepared:
            blocks.append(
                "TURN "
                f"{item['turn_id']}\n"
                "USER:\n"
                f"{item['user']}\n"
                "ASSISTANT:\n"
                f"{item['assistant']}"
            )
        prompt = (
            "These are the most recent turns to compress. Output one line per TURN, in input order.\n\n"
            + "\n\n---\n\n".join(blocks)
            + f"\n\nKeep each line's summary under {min(max_tokens_per_turn, 220)} tokens."
        )
        self.last_tail_in = self.count_tokens(RECENT_TAIL_SYSTEM_PROMPT) + self.count_tokens(prompt)
        self.last_tail_out = 0
        self.last_tail_think = 0
        self.last_tail_seconds = 0.0
        raw = ""
        tail_t0 = time.time()
        try:
            raw = await self.llm.chat(
                system=RECENT_TAIL_SYSTEM_PROMPT,
                user=prompt,
                max_tokens=max(256, max_tokens_per_turn * len(prepared) * 2),
            )
            try:
                self.last_tail_seconds = time.time() - tail_t0
            except Exception:
                self.last_tail_seconds = 0.0
            self.last_tail_out = self.count_tokens(raw)
            summaries = self._parse_recent_tail_output(raw, [item["turn_id"] for item in prepared])
        except Exception as exc:  # noqa: BLE001
            try:
                self.last_tail_seconds = time.time() - tail_t0
            except Exception:
                self.last_tail_seconds = 0.0
            print(f"[jlc:tail] compress_recent_turns LLM failed, using heuristic: {exc}", file=sys.stderr)
            summaries = {}
        tail_tps = (
            f"{self.last_tail_out / self.last_tail_seconds:.0f} tps"
            if self.last_tail_seconds > 0.001
            else "tps=n/a"
        )
        _log_post_encode(
            f"[jlc:meter] tail[in={self.last_tail_in} out={self.last_tail_out} "
            f"think={self.last_tail_think}, {self.last_tail_seconds:.1f}s, {tail_tps}]"
        )

        entries: list[TailEntry] = []
        now = time.time()
        for item in prepared:
            summary = summaries.get(item["turn_id"], "")
            if not summary:
                summary = self._heuristic_tail_summary(item["user"], item["assistant"])
            summary = self._cap_summary_tokens(summary, max_tokens=max_tokens_per_turn)
            if not summary.strip():
                summary = "특이사항 없음."
            if summary.strip():
                entries.append(
                    TailEntry(
                        turn_id=item["turn_id"],
                        summary=summary,
                        token_count=self.count_tokens(summary),
                        created_at=now,
                    )
                )
        return entries

    def _parse_recent_tail_output(self, raw: str, turn_ids: list[str]) -> dict[str, str]:
        text = unicodedata.normalize("NFC", raw or "").strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text).strip()
        parsed: dict[str, str] = {}
        expected = set(turn_ids)
        fallback_lines: list[str] = []
        for line in text.splitlines():
            clean = re.sub(r"^\s*[-*]\s+", "", line).strip()
            if not clean:
                continue
            match = re.match(r"^\s*(?:TURN\s*)?([^:\s]+)\s*:\s*(.+?)\s*$", clean, re.I)
            if match and match.group(1) in expected:
                parsed[match.group(1)] = self._clean_tail_summary(match.group(2))
            else:
                fallback_lines.append(clean)
        if len(parsed) == len(turn_ids):
            return parsed
        for turn_id, line in zip((tid for tid in turn_ids if tid not in parsed), fallback_lines, strict=False):
            parsed[turn_id] = self._clean_tail_summary(line)
        return parsed

    @staticmethod
    def _summarize_tool_events(tool_events: list[dict[str, Any]]) -> str:
        if not tool_events:
            return ""
        parts: list[str] = []
        for ev in tool_events[:8]:
            if not isinstance(ev, dict):
                continue
            name = str(ev.get("name") or ev.get("tool") or ev.get("type") or "tool")
            status = str(ev.get("status") or ev.get("result") or ev.get("summary") or "")
            status = re.sub(r"\s+", " ", status).strip()[:160]
            parts.append(f"{name}: {status}" if status else name)
        return "; ".join(parts)

    @staticmethod
    def _clean_tail_summary(raw: str) -> str:
        text = unicodedata.normalize("NFC", raw or "").strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", "", text).strip()
        lines = [re.sub(r"^\s*[-*]\s+", "", ln).strip() for ln in text.splitlines()]
        text = " ".join(ln for ln in lines if ln)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _cap_summary_tokens(self, summary: str, max_tokens: int) -> str:
        summary = self._clean_tail_summary(summary)
        max_tokens = max(1, max_tokens)
        if self.count_tokens(summary) <= max_tokens:
            return summary
        words = summary.split()
        if not words:
            return ""
        lo, hi = 1, len(words)
        best = words[:1]
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = " ".join(words[:mid])
            if self.count_tokens(candidate) <= max_tokens:
                best = words[:mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return " ".join(best).rstrip(" ,;:")

    def _heuristic_tail_summary(self, user_msg: str, assistant_msg: str, tools_summary: str = "") -> str:
        text = f"{user_msg}\n{assistant_msg}"
        # Drop pure filler turns hard; they do not help restore thread state.
        filler = re.compile(r"^\s*(ok|okay|thanks|thank you|고마워|감사|ㅇㅇ|응|넵|네|오케이|알겠어)[.!?\s]*$", re.I)
        if filler.fullmatch(user_msg or "") and filler.fullmatch(assistant_msg or ""):
            return ""

        paths = re.findall(r"(?:[A-Za-z]:\\[^\s:]+|[\w./-]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json))", text)
        funcs = re.findall(r"\b(?:def|class|function)\s+([A-Za-z_][\w]*)|`([A-Za-z_][\w.]*)`", text)
        symbols = [a or b for a, b in funcs if (a or b)]
        questions = re.findall(r"([^.!?\n]{3,120}\?)", assistant_msg or "")
        errors = re.findall(r"([A-Za-z_]*Error:?[^\n]{0,160}|failed[^\n]{0,160}|실패[^\n]{0,160}|에러[^\n]{0,160})", text, re.I)

        parts: list[str] = []
        if questions:
            parts.append(f"assistant open question: {questions[-1].strip()}")
        if paths:
            parts.append("coding context files: " + ", ".join(dict.fromkeys(paths[:5])))
        if symbols:
            parts.append("symbols/functions: " + ", ".join(dict.fromkeys(symbols[:5])))
        if errors:
            parts.append("current error/thread: " + errors[-1].strip())
        if tools_summary:
            parts.append("tool summary: " + tools_summary)
        if not parts:
            user_one = re.sub(r"\s+", " ", (user_msg or "").strip())[:240]
            assistant_one = re.sub(r"\s+", " ", (assistant_msg or "").strip())[:320]
            parts.append(f"user asked: {user_one}; assistant replied: {assistant_one}")
        return "; ".join(p for p in parts if p).strip()

    async def _enforce_budget(self, jhb: str, system_prompt: str, target: int | None = None) -> str:
        budget = target if (target and target > 0) else self.target_tokens
        jhb = self._enforce_priority_fifo(jhb)
        jhb = self._drop_priority_bullets_until_budget(jhb, "P3", budget)
        if self.count_tokens(jhb) <= budget:
            print("[jlc:enc:bg] P3 FIFO drop succeeded", file=sys.stderr)
            return jhb

        jhb = self._drop_priority_bullets_until_budget(jhb, "P2", budget)
        if self.count_tokens(jhb) <= budget:
            print("[jlc:enc:bg] P2 FIFO drop succeeded", file=sys.stderr)
            return jhb

        prev_tokens = self.count_tokens(jhb)
        try:
            recompress_prompt = (
                f"The following jhb exceeds the {budget}-token budget. "
                f"Compress it to under {budget} tokens by:\n"
                "1. Compressing older P1 bullets to terse keywords if still over budget\n"
                "2. Keeping P0 content intact unless it is redundant\n"
                "3. NEVER remove all P0 content\n\n"
                "Output the compressed jhb in markdown only.\n\n"
                f"───── JHB TO COMPRESS ─────\n{jhb}\n───── END ─────"
            )
            compressed = await self.llm.chat(system=system_prompt, user=recompress_prompt, max_tokens=2048)
            compressed = unicodedata.normalize("NFC", compressed).strip()
            compressed_tokens = self.count_tokens(compressed) if compressed else 0
            if compressed and compressed_tokens < prev_tokens * 0.3:
                print(
                    f"[jlc:enc:bg] recompress collapse rejected ({compressed_tokens} < {prev_tokens * 0.3:.0f})",
                    file=sys.stderr,
                )
                compressed = ""
            if compressed and not self._contains_p0_section(compressed):
                print("[jlc:enc:bg] recompress lost all P0 sections, rejected", file=sys.stderr)
                compressed = ""
            if compressed and compressed_tokens <= budget:
                print("[jlc:enc:bg] recompress succeeded", file=sys.stderr)
                return compressed
            if compressed and compressed_tokens < prev_tokens:
                jhb = self._enforce_priority_fifo(compressed)
                print("[jlc:enc:bg] recompress partial, using as base for mechanical strip", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"[jlc:enc:bg] recompress failed: {exc}", file=sys.stderr)

        tok = self.count_tokens(jhb)
        if tok > budget:
            jhb = self._drop_priority_bullets_until_budget(jhb, "P1", budget)
            tok = self.count_tokens(jhb)
            if tok <= budget:
                print("[jlc:enc:bg] P1 FIFO drop succeeded", file=sys.stderr)
                return jhb
        if tok > budget:
            p0_budget = min(budget, 1500)
            jhb = self._drop_priority_bullets_until_budget(jhb, "P0", p0_budget, min_keep=1)
            tok = self.count_tokens(jhb)
            if tok <= p0_budget:
                print(f"[jlc:enc:bg] P0 FIFO drop succeeded ({tok}<={p0_budget})", file=sys.stderr)
                return jhb
        if tok > budget:
            print(f"[jlc:enc:bg] WARNING: still over budget after P3/P2/P1/P0 FIFO drop ({tok}>{budget})", file=sys.stderr)
        else:
            print("[jlc:enc:bg] P1 recompress succeeded", file=sys.stderr)
        return jhb

    @staticmethod
    def _contains_p0_section(text: str) -> bool:
        return bool(re.search(r"\[P0\]", text))

    @staticmethod
    def _strip_sections_by_level(jhb: str, level: str) -> str:
        pattern = re.compile(rf"^## .+\[{level}\]\s*$([\s\S]*?)(?=^## |\Z)", flags=re.MULTILINE)
        result = pattern.sub("", jhb)
        if result == jhb:
            return jhb
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result.strip()

    def _drop_priority_bullets_until_budget(self, jhb: str, level: str, budget: int, *, min_keep: int = 0) -> str:
        sections = self._parse_priority_sections(jhb)
        if not sections:
            return jhb
        dropped = 0
        while self.count_tokens(self._render_priority_sections(sections)) > budget:
            bullets = self._priority_bullets(sections, level)
            if len(bullets) <= min_keep:
                break
            section_idx, item_idx, _item = bullets[0]
            sections[section_idx]["items"].pop(item_idx)
            dropped += 1
        if not dropped:
            return jhb
        print(f"[jlc:enc:bg] dropped {dropped} oldest {level} bullet(s) under budget pressure", file=sys.stderr)
        return self._render_priority_sections(sections)

    def count_tokens(self, text: str) -> int:
        # Double-checked locking — first thread initializes, others reuse.
        # Without the lock two threads could race and one would replace the
        # other's encoder mid-call.
        if tiktoken is None:
            return len(text.split())
        if JLCEncoder._tiktoken_enc is None:
            with JLCEncoder._tiktoken_init_lock:
                if JLCEncoder._tiktoken_enc is None:
                    try:
                        JLCEncoder._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
                    except Exception:  # noqa: BLE001
                        return len(text.split())
        try:
            return len(JLCEncoder._tiktoken_enc.encode(text))
        except Exception:  # noqa: BLE001
            return len(text.split())

    @staticmethod
    def _truncate(text: str, max_chars: int = 4000) -> str:
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit("\n", 1)[0]
        return cut + "\n...(truncated)"

    _DENIAL_HEAD_CHARS = 240
    _DENIAL_PHRASES = (
        "no record",
        "no records",
        "no such record",
        "no stored record",
        "no matching record",
        "nothing found",
        "nothing logged",
        "came up empty",
        "don't have any record",
        "do not have any record",
        "have no record",
        "can't confirm",
        "cannot confirm",
        "unknown",
    )
    _DENIAL_REPLACEMENT = (
        "[DENIAL RESPONSE OMITTED FROM DURABLE MEMORY: the assistant reported no record "
        "or insufficient evidence. Do not add this denial as a JHB fact.]"
    )

    @staticmethod
    def _is_denial_reply(text: str) -> bool:
        head = unicodedata.normalize("NFC", text or "").strip()[:JLCEncoder._DENIAL_HEAD_CHARS]
        head = re.sub(r"^[`'\"*_\s>:-]+", "", head)
        head = re.sub(r"(?i)^answer\s*[:：]\s*", "", head).strip()
        folded = head.casefold()
        if re.match(r"^(?:no records?|no such record|none|unknown)\b", folded):
            return True
        return any(phrase in folded for phrase in JLCEncoder._DENIAL_PHRASES)

    @staticmethod
    def _assistant_for_jhb_prompt(assistant_msg: str) -> str:
        normalized = unicodedata.normalize("NFC", assistant_msg.strip())
        neutralized = JLCEncoder._neutralize_delimiters(normalized)
        if JLCEncoder._is_denial_reply(neutralized):
            return JLCEncoder._DENIAL_REPLACEMENT
        return neutralized

    @staticmethod
    def _build_user_prompt(
        prev_jhb: str,
        user_msg: str,
        assistant_msg: str,
        prev_project_md: str = "",
        project_active: bool = False,
        current_turn: int | None = None,
        origin: str = "user",
        origin_window: str | None = None,
        origin_window_label: str | None = None,
    ) -> str:
        prev = prev_jhb.strip() if prev_jhb.strip() else "(none)"
        safe_user = unicodedata.normalize("NFC", user_msg)
        safe_user = JLCEncoder._escape_markdown_headings(safe_user)
        safe_user = JLCEncoder._neutralize_delimiters(safe_user)
        trimmed_assistant = JLCEncoder._assistant_for_jhb_prompt(assistant_msg)
        turn_line = f"current_turn=t{current_turn}\n" if current_turn is not None else ""
        origin_line = f"ORIGIN: {origin or 'user'}\n"
        if origin_window:
            origin_line += f"ORIGIN_WINDOW: {origin_window}\n"
        if origin_window_label:
            origin_line += f"ORIGIN_WINDOW_LABEL: {origin_window_label}\n"
        return (
            turn_line +
            "PREVIOUS JHB:\n"
            f"{prev}\n"
            "\nNEW MESSAGE METADATA:\n"
            f"{origin_line}"
            "\nNEW USER MESSAGE:\n"
            f"{safe_user}\n"
            "\nNEW ASSISTANT REPLY:\n"
            f"{trimmed_assistant}\n"
        )

    @staticmethod
    def _build_batch_user_prompt(
        prev_jhb: str,
        batch_turns: list[dict],
        prev_project_md: str = "",
        project_active: bool = False,
        current_turn: int | None = None,
    ) -> str:
        prev = prev_jhb.strip() if prev_jhb.strip() else "(none)"
        body_parts = []
        for i, t in enumerate(batch_turns, start=1):
            safe_user = unicodedata.normalize("NFC", t.get("user", "") or "")
            safe_user = JLCEncoder._escape_markdown_headings(safe_user)
            safe_user = JLCEncoder._neutralize_delimiters(safe_user)
            trimmed_assistant = JLCEncoder._assistant_for_jhb_prompt(t.get("assistant", "") or "")
            origin = str(t.get("origin") or "user").strip() or "user"
            origin_window = str(t.get("origin_window") or "").strip()
            origin_window_label = str(t.get("origin_window_label") or "").strip()
            metadata = f"ORIGIN: {origin}\n"
            if origin_window:
                metadata += f"ORIGIN_WINDOW: {origin_window}\n"
            if origin_window_label:
                metadata += f"ORIGIN_WINDOW_LABEL: {origin_window_label}\n"
            body_parts.append(
                f"METADATA:\n{metadata}USER:\n{safe_user}\nASSISTANT:\n{trimmed_assistant}\n"
            )
        body = "\n".join(body_parts)
        turn_line = f"current_turn=t{current_turn}\n" if current_turn is not None else ""
        return (
            turn_line +
            "PREVIOUS JHB:\n"
            f"{prev}\n"
            "\nNEW USER MESSAGE / ASSISTANT REPLY:\n"
            f"{body}"
        )

    @staticmethod
    def _escape_markdown_headings(text: str) -> str:
        # Insert ZWSP after EACH '#' char so markdown parsers don't see a heading.
        # # → #​, ## → #​#​, ### → #​#​#​ (each # followed by ZWSP).
        # Avoids visible backslash and double-escape on re-encode round-trip.
        return re.sub(r"(?m)^(#+)", lambda m: "​".join(m.group(1)) + "​", text)

    _DELIM_TOKENS = ("<<<JHB>>>", "<<<END_JHB>>>", "<<<JARVIS_MD>>>", "<<<END_JARVIS_MD>>>")

    @staticmethod
    def _neutralize_delimiters(text: str) -> str:
        for tok in JLCEncoder._DELIM_TOKENS:
            text = text.replace(tok, tok.replace("<<<", "<<​<").replace(">>>", ">​>>"))
        return text

    _MAX_ENCODER_OUTPUT_BYTES = 128 * 1024  # JHB-only output ceiling

    @staticmethod
    def _parse_jhb_output(raw: str) -> str:
        """Extract JHB from the encoder's JHB-only output.

        Uses last-match semantics so a stray delimiter inside content does not
        truncate the real block. Legacy JARVIS_MD blocks are tolerated but
        ignored so older in-flight encoder responses do not break the runtime.

        Also strips ZWSP that may have been injected by _escape_markdown_headings
        in previous round-trips — ensures headings parse correctly.
        """
        # DoS guard — a malicious / runaway encoder LLM could emit megabytes.
        # Truncate before any further work.
        if len(raw) > JLCEncoder._MAX_ENCODER_OUTPUT_BYTES:
            print(
                f"[jlc:enc:bg] encoder output {len(raw)} bytes exceeds cap "
                f"{JLCEncoder._MAX_ENCODER_OUTPUT_BYTES}; truncating",
                file=sys.stderr,
            )
            raw = raw[: JLCEncoder._MAX_ENCODER_OUTPUT_BYTES]
        # Remove ZWSP (Zero-Width Space U+200B) that may survive from prior
        # _escape_markdown_headings calls or LLM output.
        raw = raw.replace("​", "")
        # LLMs occasionally emit a markdown horizontal rule (or stray dash /
        # em-dash / equals) directly adjacent to the JHB delimiters
        # (e.g. `─────<<<END_JHB>>>`, `---<<<JHB>>>`, `—<<<END_JHB>>>`),
        # which then leaks into block content or breaks last-pair anchoring.
        # Strip ANY run (1+) of rule characters and surrounding whitespace
        # immediately preceding any delimiter token. Anchored to delimiters so
        # this never touches normal block body text — _neutralize_delimiters
        # already escapes literal `<<<JHB>>>` mentions inside content.
        raw = re.sub(r"[\s\-─═—=*_+~]+(<<<(?:END_)?(?:JHB|JARVIS_MD)>>>)", r"\n\1", raw)

        def _last_pair(start_tok: str, end_tok: str) -> str | None:
            start_positions = [m.start() for m in re.finditer(re.escape(start_tok), raw)]
            end_positions = [m.end() for m in re.finditer(re.escape(end_tok), raw)]
            if not start_positions or not end_positions:
                return None
            start = start_positions[0] + len(start_tok)
            end = end_positions[-1] - len(end_tok)
            if end <= start:
                return None
            return raw[start:end].strip()

        jhb = _last_pair("<<<JHB>>>", "<<<END_JHB>>>") or ""
        if jhb:
            from .diff import strip_jhb_wrappers  # noqa: PLC0415

            jhb = strip_jhb_wrappers(jhb)
        return jhb
