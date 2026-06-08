"""JLC slim middleware core for Aider hooks."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MAX_CACHED_ENCODE_LOCKS = 1000
_SESSION_ID = "jarvis_session"
_PROMPT_TAG = "brief-chat-v3"


def _normalize_session_id(session_id: str | None) -> str:
    raw = str(session_id or "").strip()
    return raw or _SESSION_ID


def _atomic_write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text via temp file + os.replace so a crash mid-write cannot corrupt
    an existing file. Caller ensures parent dir exists.
    """
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    # Owner-only (0o600) on POSIX. Windows lacks os.fchmod; ACLs handle this.
    try:
        os.fchmod(fd, 0o600)
    except (AttributeError, OSError):
        pass
    try:
        with os.fdopen(fd, "w", encoding=encoding, errors="replace", newline="") as fh:
            fh.write(content)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise

from .config import JLCConfig, load_config
from .embedder import LocalEmbedder
from .encoder import JLCEncoder, TailEntry
from .git_manager import GitManager
from .graph import JLCGraph
from .jre import JREEngine
from .providers import clear_cache, get_llm


def _read_encoder_model_spec(config_path: str | None) -> str:
    """Best-effort fetch of `roles.encoder` (e.g. `ollama-cloud/glm-5.1`)
    from the active config yaml so the encoder meter dict can surface
    provider/model on the pi footer. Quietly returns "" on any failure."""
    try:
        import os
        import yaml
        candidates: list[str] = []
        if config_path:
            candidates.append(config_path)
        env_path = os.environ.get("JARVIS_CODE_CONFIG")
        if env_path:
            candidates.append(env_path)
        candidates.append(os.path.expanduser("~/.jarvis-code/config.yaml"))
        for path in candidates:
            if not path or not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            spec = str((data.get("roles") or {}).get("encoder") or "")
            if spec:
                return spec
        return ""
    except Exception:
        return ""
from .retriever import JLCRetriever
from .tagger import JLCTagger
from .turn_logger import JLCTurnLogger
try:
    from jarvis_sidecar.raw_store import extract_turn_numbers
except Exception:  # pragma: no cover - sidecar package unavailable in standalone mode
    extract_turn_numbers = None
try:
    from jarvis_sidecar.raw_store import extract_local_dates
except Exception:  # pragma: no cover - sidecar package unavailable in standalone mode
    extract_local_dates = None
try:
    from jarvis_sidecar.raw_store import (
        append_encoder_turn as append_pi_sidecar_encoder_turn,
        append_meter_turn as append_pi_sidecar_meter_turn,
    )
except Exception:  # pragma: no cover - optional sidecar bridge
    append_pi_sidecar_encoder_turn = None
    append_pi_sidecar_meter_turn = None

JHB_DELIM = "\n---JHB_END---\n"
PROJ_DELIM = "\n---PROJECT_END---\n"
RECALL_DELIM = "\n---RECALL_END---\n"


def _paperlog_path(session_id: str) -> Path:
    root = Path(os.environ.get("JARVIS_RAW_STORE", "~/.jarvis-code/pi-sidecar")).expanduser()
    safe = session_id.strip() or _SESSION_ID
    invalid = '<>:"/\\|?*'
    table = str.maketrans({ch: "_" for ch in invalid})
    safe = safe.translate(table).replace("..", "_")
    if safe == _SESSION_ID:
        return root / "jarvis_session.paperlog"
    bench_root = os.environ.get("JARVIS_RAW_BENCH_STORE")
    if bench_root:
        return Path(bench_root).expanduser() / f"{safe}.paperlog"
    return root.parent / "conversation_bench_archive" / f"{safe}.paperlog"


def _one_line(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def _append_paperlog_row(
    *,
    session_id: str,
    turn_id: int,
    user_message: str,
    assistant_message: str,
    meter_line: str,
) -> None:
    path = _paperlog_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).replace(microsecond=0).isoformat()
    line = (
        f"{ts} | turn={turn_id} | "
        f"question={_one_line(user_message)} | "
        f"answer={_one_line(assistant_message)} | "
        f"prompt_tag={_PROMPT_TAG} | "
        f"{meter_line}"
    )
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.write("\n")


class JarvisAgentic:
    """Slim JLC core: two-tier prompt prepend + async encode/save pipeline."""

    def __init__(self, config_path: str | Path | None = None, completion_client: Any | None = None) -> None:
        self.config: JLCConfig = load_config(config_path)
        self._jhb_root = Path(self.config.jhb.storage_path).expanduser()
        # Public read-only accessor for the JHB storage root. External wiring
        # (e.g. dispatcher closure binding for recall_turn) should depend on
        # this attribute name instead of poking at `_jhb_root` directly so the
        # internal layout can change without breaking callers.
        self.jhb_root = self._jhb_root
        self._encoder_config_path = str(config_path) if config_path else None
        self._encoder_llm = get_llm("encoder", config_path=self._encoder_config_path)
        # prompt_path=None → JLCEncoder uses importlib.resources (works in dev + wheel/zip).
        self.encoder = JLCEncoder(llm=self._encoder_llm, prompt_path=None, target_tokens=self.config.jhb.target_tokens)
        self.last_jlc_head_breakdown: dict[str, int] = {}
        # Per-conversation locks to serialize jhb/JARVIS.md writes across thread fallbacks.
        # threading.Lock (process-wide) — asyncio.Lock would bind to first loop and break
        # when thread fallback creates a fresh loop per call. Plain Lock (not RLock) is
        # required because _encode_and_save acquires in a worker thread (via
        # asyncio.to_thread) and releases in the event loop thread, which RLock would
        # reject as cross-thread release.
        # OrderedDict + LRU eviction caps memory at _MAX_CACHED_ENCODE_LOCKS so a process
        # that touches thousands of conv_ids does not leak a Lock per id.
        self._encode_locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._encode_locks_guard = threading.Lock()
        # idea #12 step 4: per-conv in-flight encode counter for backlog
        # throttle. Guarded by _encode_locks_guard since increments/decrements
        # are O(1) and we want lock-coherence with _encode_locks.
        self._encode_in_flight: dict[str, int] = {}
        # W2.9.21 §4.2: per-conv batch buffer. encode_and_save_async appends
        # each completed bench turn here and only fires the encoder when the
        # buffer reaches BATCH_SIZE. Collapses N per-turn fires into 1 per
        # 5 turns, removing the agentic-loop multi-fire that drove the
        # backlog throttle's tier-1+tier-2 trips.
        self._batch_buffer: dict[str, list[dict[str, Any]]] = {}
        self._batch_buffer_guard = threading.Lock()
        # `/turn` is a sync FastAPI handler, so production chat normally
        # reaches the thread fallback in `_dispatch_batch_encode_async`.
        # Reuse one background worker instead of creating a daemon thread
        # and fresh asyncio loop for every completed turn. Multi-thousand
        # turn runs on Windows otherwise accumulate thread/handle/commit
        # pressure even though per-conversation encode locks serialize the
        # useful work already.
        self._background_encode_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="jlc-encode",
        )
        # Per-conv "has the first encode fired yet?" flag. Cold-start convs
        # fire on turn 1 (single-turn batch) so the very next turn already
        # sees a non-empty JHB. After the first fire, future fires happen
        # every BATCH_SIZE turns. In-memory only — process resume re-fires
        # once on the next turn, which is harmless (load_jhb still loads
        # the persisted JHB so prev_jhb stays correct).
        self._batch_first_fired: dict[str, bool] = {}
        # W2.9.25 minimal (Jun 2026-05-12): BATCH_SIZE=1 — encoder fires
        # every turn. Rationale: with recent_window=0 the JHB IS the chat's
        # only memory of the past; encoder lag means stale silhouette and
        # mode drift (v4 1648-turn evidence: enc_lag mean 88, cataloging
        # mode collapse). Per-turn encode keeps the silhouette current.
        # devstral-small-2:24b dense is fast enough to keep up with chat.
        self.BATCH_SIZE = 1

        self.git_manager = GitManager(self._jhb_root)
        self.git_manager.init_repo()
        # All attributes the warmup daemon may touch (retriever, jre, _embedder)
        # are constructed BELOW; the warmup thread is spawned at the END of
        # __init__ to avoid races against attribute-not-yet-set.
        self._embedder: LocalEmbedder | None = None
        self._embedder_warmed = False
        self.tagger = JLCTagger(custom_patterns=self.config.tagger.custom_patterns, max_tags_per_turn=self.config.tagger.max_tags_per_turn)
        self.graph = JLCGraph(storage_root=self._jhb_root, batch_interval=self.config.graph.batch_interval, max_nodes=self.config.graph.max_nodes, max_edges=self.config.graph.max_edges, prune_stale_turns=self.config.graph.prune_stale_turns)
        self.turn_logger = JLCTurnLogger(storage_root=self._jhb_root)
        self.retriever = JLCRetriever(storage_root=self._jhb_root, embedder=None)
        self.jre = JREEngine(storage_root=self._jhb_root, embedder=None)
        # bge-m3 cold start is ~30s. Run synchronously so the "Loading weights:
        # 391/391" progress bar surfaces before the chat UI accepts input —
        # users (and recall_turn) need the embedder warm before the first turn.
        self._warmup_embedder()

    def reload_encoder_llm(self) -> None:
        """Re-fetch the encoder LLM after /llmsetting/apply rewrote
        roles.encoder. Without this the encoder keeps using the LLM bound at
        sidecar boot even after config.yaml changes. Chat uses _LazyChatLLM
        so module-level clear_cache() alone is enough for chat; encoder is
        bound to a long-lived JLCEncoder instance and needs an explicit swap.
        """
        clear_cache()
        self._encoder_llm = get_llm("encoder", config_path=self._encoder_config_path)
        self.encoder.llm = self._encoder_llm

    def _get_embedder(self) -> LocalEmbedder:
        """Lazy-load embedder on first use to avoid blocking Aider startup."""
        if self._embedder is None:
            self._embedder = LocalEmbedder(
                model_name=self.config.embedder.model_name,
                cache_dir=self.config.embedder.cache_dir,
            )
            # Inject embedder into retriever/jre that were created with None
            self.retriever._embedder = self._embedder
            self.jre._embedder = self._embedder
        return self._embedder

    def prepend_two_tier(
        self,
        aider_messages: list[dict[str, Any]],
        project_path: str | None,
        recall_block: str = "",
    ) -> list[dict[str, Any]]:
        conv_id = _SESSION_ID
        conv_jhb = self.render_jhb()
        project_md = self._load_project_md(project_path)
        # Language directive sits ahead of JHB so it survives even if downstream
        # reminders are pruned.
        lang_directive = (
            "[Language]\n"
            "Answer in English only in BOTH reasoning/thinking and final answer. "
            "Do not translate English into another language, and do not mirror "
            "the user's language if it is not English.\n\n"
        )
        tool_channel_directive = (
            "[Tool use and channel discipline]\n"
            "Route every tool invocation through the structured function-calling "
            "channel. Use `recall_turn` when a user request depends on a "
            "specific prior conversation fact that is not explicit in JHB, "
            "or when your answer would otherwise be 'I do not know', "
            "'not sure', or 'not on record'. Do not use recall for brand-new "
            "questions or facts already clear in JHB. For current external "
            "facts, call `web_search`. For complex multi-step investigation, call "
            "`delegate_subagent`.\n"
            "Keep the user-facing content stream to a natural-language reply in "
            "English only. Do not include raw tool-call shadows, JSON "
            "argument dumps, channel labels, `to=<name>`, `analysis to=...code`, "
            "`code`, or `final`.\n\n"
        )
        # Reasoning depth policy — let the LLM decide how deep to think per
        # request instead of hard-coded heuristics. Single source in
        # jlc_agentic.prompts.reasoning_policy.
        from jlc_agentic.prompts import (
            POLICY_USER_FACING,
            get_constitution,
            get_env_directive,
        )

        # W2.9.16 (2026-05-08): jarvis-code constitution sits AHEAD of every
        # other directive so the truthfulness-over-fluency and retrieval-first
        # rules are the first instructions the chat model sees.
        constitution_directive = (
            "[jarvis-code Constitution — applies to chat, subagent, encoder]\n"
            + get_constitution()
            + "\n\n"
        )

        reasoning_directive = POLICY_USER_FACING + "\n"
        # Host-OS bash hint — Windows lacks head/tail/wc; tells the LLM
        # to use PowerShell equivalents instead. Static, prompt-cache safe.
        env_directive = get_env_directive()
        # Retrieval policy — JHB is a lossy summary. When the user asks about
        # specific past content (quotes, decisions, snippets) call recall_turn
        # BEFORE answering "I don't know" or asking a clarifying question.
        retrieval_directive = (
            "[Memory access]\n"
            "Treat the JHB above as a compressed summary of past turns, not the "
            "full transcript. JHB is a silhouette, not ground truth.\n\n"
            "Use JHB as the fast path only when it clearly and explicitly "
            "contains the answer. If the user asks for a specific prior "
            "conversation fact and JHB is missing, vague, stale, or ambiguous, "
            "call `recall_turn` before answering.\n\n"
            "Recall-worthy prior facts include names, family/people, places, "
            "dates, preferences, decisions, numbers, previous errors, exact "
            "wording, code/project details, and anything the user reasonably "
            "expects you may remember from earlier turns.\n\n"
            "If you would otherwise say 'I do not remember', 'I am not sure', "
            "'not on record', or ask the user to repeat a prior fact, make one "
            "`recall_turn` attempt first. This applies in casual chat too.\n\n"
            "If JHB already contains the needed recent context clearly enough, "
            "answer directly without recall.\n\n"
            "Do not call `recall_turn` for brand-new information requests, "
            "general knowledge, ordinary brainstorming, one-word fillers, or "
            "acknowledgements.\n\n"
            "After recall, distinguish between partial mention and truly no "
            "mention. Do not over-infer from vague JHB hints.\n\n"
            "[Short replies]\n"
            "Short confirmations, denials, or acknowledgements are usually "
            "answers to the previous assistant question. Read the recent "
            "window/JHB and continue the pending action instead of treating "
            "them as a new casual greeting.\n\n"
            "[Recent window]\n"
            "When prior turns exist, the latest user message is prefixed with a "
            "`<recent_window>...</recent_window>` block holding raw user/assistant "
            "text from the last N turns (oldest first). It is the immediate "
            "conversation flow, not part of the user's question. Read it for "
            "continuity, then answer the actual user message that follows.\n\n"
            "[New project bootstrap]\n"
            "If the user asks to create, start, set up, build, or register a "
            "project and the target name/path is clear, treat that as explicit "
            "consent to call `register_project`; do not ask for another "
            "confirmation merely because registration is involved. Ask only "
            "when the target name/path is ambiguous, missing, or unsafe.\n"
            "Register folders before relying on router memory. The auto-router "
            "registers only folders mentioned by absolute path in the user's "
            "utterance. Folders you create yourself, and existing folders the "
            "user references only by nickname, remain invisible until "
            "registered.\n"
            "For a new folder you just created, immediately call "
            "`register_project(path=\"<absolute>\")` before writing any other "
            "file in that folder. This writes starter JARVIS.md and adds the "
            "folder to the registry.\n"
            "For an existing unregistered folder whose absolute path appears in "
            "JHB memory or `recall_turn`, call "
            "`register_project(path=\"<absolute>\")` once, then proceed. "
            "`register_project` is idempotent. Use it on already-registered "
            "paths when registration state is unclear.\n\n"
            "[Project work execution]\n"
            "For coding, debugging, page edits, styling changes, asset "
            "insertion, and small bug fixes, reason through the whole turn "
            "before using tools. Choose compact batches: one reconnaissance "
            "batch for relevant reads/searches, one edit batch for the actual "
            "change, and one verification batch when practical. Do not "
            "alternate tiny reads and tiny edits across many model rounds. "
            "Once the relevant files, assets, or web evidence are clear enough, "
            "stop exploring and make the change. If the work is not converging "
            "after a few rounds, stop tool use, report what changed and what "
            "remains uncertain, and ask for the next instruction instead of "
            "continuing an open-ended loop.\n\n"
        )
        # System head stays static (lang_directive + reasoning_directive +
        # retrieval_directive + JHB + project_md) so Anthropic prompt-cache hits
        # across turns. recall_block is per-turn dynamic context — prepend it
        # onto the last user message instead so the cacheable prefix is undisturbed.
        head = f"{constitution_directive}{lang_directive}{tool_channel_directive}{reasoning_directive}{env_directive}{retrieval_directive}{conv_jhb}{JHB_DELIM}{project_md}{PROJ_DELIM}"
        self.last_jlc_head_breakdown = {
            "constitution": self.encoder.count_tokens(constitution_directive),
            "lang": self.encoder.count_tokens(lang_directive),
            "tool_channel": self.encoder.count_tokens(tool_channel_directive),
            "reasoning": self.encoder.count_tokens(reasoning_directive),
            "env": self.encoder.count_tokens(env_directive),
            "retrieval": self.encoder.count_tokens(retrieval_directive),
            "jhb": self.encoder.count_tokens(conv_jhb),
            "project_md": self.encoder.count_tokens(project_md),
        }
        # Stash for the Pi extension to break down chat[in] in the meter.
        self.last_jlc_head_text = head

        merged = list(aider_messages)
        if merged and merged[0].get("role") == "system":
            old = str(merged[0].get("content", ""))
            merged[0] = {**merged[0], "content": head + old}
        else:
            merged = [{"role": "system", "content": head}] + merged

        if recall_block:
            for i in range(len(merged) - 1, -1, -1):
                if merged[i].get("role") == "user":
                    user_text = str(merged[i].get("content", ""))
                    recall_prefix = f"[Recalled context]\n{recall_block}{RECALL_DELIM}"
                    merged[i] = {**merged[i], "content": recall_prefix + user_text}
                    break
        return merged

    _CONFIDENCE_RANK = {"LOW": 0, "MID": 1, "HIGH": 2}

    # Class-level lock to prevent concurrent warmup from multiple threads
    _warmup_lock = threading.Lock()

    def _warmup_embedder(self) -> None:
        """Synchronous warmup so the first recall probe doesn't trip the timeout
        and the bge-m3 progress bar surfaces before the chat UI is ready.
        Idempotent — safe if hybrid_search has already loaded the model.
        Thread-safe: uses lock to prevent race condition where two threads
        both see _embedder_warmed=False and both load weights.
        """
        # Fast path without lock
        if self._embedder_warmed:
            return
        # Slow path: acquire lock to prevent concurrent warmup
        with self._warmup_lock:
            # Double-check after acquiring lock
            if self._embedder_warmed:
                return
            import os
            import sys
        if os.environ.get("JARVIS_SKIP_EMBEDDER_WARMUP") == "1":
            print("[jlc:embed] warmup skipped (JARVIS_SKIP_EMBEDDER_WARMUP=1)", file=sys.stderr, flush=True)
            return
        try:
            print("[jlc:embed] Loading bge-m3 weights...", file=sys.stderr, flush=True)
            embedder = self._get_embedder()  # build wrapper (cheap; weights load on first embed)
            embedder.embed(["warmup"])  # force SentenceTransformer load before chat starts; without this, the first turn's retriever and the encoder's JRE race on _ensure_model and BOTH load weights
            self._embedder_warmed = True
            print(
                f"[jlc:embed] Embedder ready (dim={embedder.dim}).",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:
            print(f"[jlc:warmup] embedder warmup deferred: {exc}", file=sys.stderr, flush=True)
        except BaseException as exc:  # noqa: BLE001
            # SentenceTransformer load can SystemExit / segfault via torch DLL conflicts
            # (e.g., after partial SWE-bench install). Catch BaseException so the sidecar
            # survives in light-memory mode instead of dying silently at startup.
            print(f"[jlc:warmup] embedder warmup crashed (degraded mode): {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def recall_for_query(
        self,
        query: str,
        top_k: int = 3,
        timeout: float = 5.0,
        min_confidence: str = "MID",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Sync wrapper for retriever.hybrid_search.

        Returns a dict with keys:
          - text (str): formatted recall block for backward compat (empty if
            nothing useful).
          - fragments (list[dict]): structured turn data with full user/assistant
            text (no truncation). Empty list when nothing found.
          - confidence (str): "HIGH" | "MID" | "LOW".

        Used by `/recall` (caller passes min_confidence="LOW" to always show
        what the index has) and by the Pi extension on every chat turn.

        Bounded by `timeout` so a slow embedder can't stall the chat LLM call.
        """
        _empty: dict[str, Any] = {"text": "", "fragments": [], "confidence": "LOW"}
        if not query or not query.strip():
            return _empty
        if callable(extract_turn_numbers):
            turn_numbers = set(extract_turn_numbers(query, max_turns=max(1, top_k)))
            if turn_numbers:
                conv_id = _normalize_session_id(session_id)
                fragments = self.retriever.load_turns_by_number(turn_numbers, session_id=conv_id)
                if fragments:
                    fragments.sort(key=lambda frag: int(frag.get("turn") or 0))
                    return self._format_recall_fragments(
                        fragments=fragments,
                        confidence="HIGH",
                        source="turn_number",
                    )
        if callable(extract_local_dates):
            local_dates = extract_local_dates(query, max_dates=max(1, top_k))
            if local_dates:
                conv_id = _normalize_session_id(session_id)
                fragments = self.retriever.load_turns_by_local_dates(local_dates, session_id=conv_id, limit=top_k)
                if fragments:
                    return self._format_recall_fragments(
                        fragments=fragments,
                        confidence="HIGH",
                        source="date",
                    )
        result_holder: list[Any] = []

        def runner() -> None:
            try:
                conv_id = _normalize_session_id(session_id)
                result_holder.append(asyncio.run(
                    self.retriever.hybrid_search(query, top_k=top_k, session_id=conv_id)
                ))
            except Exception as exc:
                import sys
                print(f"[jlc:recall] hybrid_search failed: {exc}", file=sys.stderr)
                result_holder.append(None)

        # Always run on a fresh thread + fresh event loop. Aider's call site
        # may or may not have a running loop; this avoids both the "no current
        # loop" and "loop already running" failure modes.
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive() or not result_holder:
            return _empty
        result = result_holder[0]
        if not result:
            return _empty
        confidence = result.get("confidence", "LOW")
        fragments = result.get("fragments", [])
        if not fragments:
            return _empty
        cur_rank = self._CONFIDENCE_RANK.get(confidence, 0)
        min_rank = self._CONFIDENCE_RANK.get(min_confidence, 1)
        if cur_rank < min_rank:
            return _empty

        return self._format_recall_fragments(fragments=fragments, confidence=confidence)

    def _format_recall_fragments(
        self,
        *,
        fragments: list[dict[str, Any]],
        confidence: str,
        source: str | None = None,
    ) -> dict[str, Any]:
        label = f"[Recalled context — confidence={confidence}]"
        if source:
            label = f"[Recalled context — confidence={confidence}, source={source}]"
        lines = [label]
        for frag in fragments:
            turn = frag.get("turn", "?")
            score = frag.get("score", 0.0)
            local_date = frag.get("local_date")
            user = frag.get("user") or ""
            assistant = frag.get("assistant") or ""
            suffix = f" | {local_date}" if local_date else ""
            lines.append(f"--- turn {turn}{suffix} (score={score}) ---")
            lines.append(f"Q: {user}")
            lines.append(f"A: {assistant}")
            lines.append("")  # blank line between turns
        lines.append(
            "Use these recalled turns when answering. Cite by turn number."
        )
        text = "\n".join(lines) + "\n"
        result: dict[str, Any] = {"text": text, "fragments": fragments, "confidence": confidence}
        if source:
            result["source"] = source
        return result


    def encode_and_save_async(
        self,
        project_path: str | None,
        user_msg: str,
        assistant_msg: str,
        llm_meta: dict[str, Any] | None = None,
        on_token: Any | None = None,
        on_done: Any | None = None,
        session_id: str | None = None,
    ) -> int | None:
        """Fire-and-forget encode + persist for this turn's JHB.

        Background dispatch lets the chat LLM keep streaming additional turns
        (Aider reflection / auto-lint) without interleaving encoder logs into
        the assistant output. Staleness is handled by `wait_for_pending_encode`
        in inject_pre_call: the next LLM call blocks until any in-flight encode
        for this conv_id finishes, so prev_jhb is always current at inject time.

        Race fix (Phase 1.6): the lock is acquired synchronously here in the
        caller's thread BEFORE dispatching the worker. Without this, a rapid
        wait_for_pending_encode call could win the lock-free window between
        thread.start() and the worker's own lock.acquire(), letting the next
        turn proceed against a stale JHB.
        """
        # W2.9.21 §4.2: buffer the turn; only fire the encoder once the
        # buffer reaches BATCH_SIZE turns. Returns early when the buffer is
        # still filling, so the agentic loop's per-LLM-call invocations
        # collapse into a single encode per 5 bench turns instead of
        # 1.6+/turn (the prior pattern that drove throttle trips).
        conv_id = _normalize_session_id(session_id)
        with self._batch_buffer_guard:
            buf = self._batch_buffer.setdefault(conv_id, [])
            buf.append({
                "user": user_msg,
                "assistant": assistant_msg,
                "llm_meta": llm_meta,
            })
            # Cold-start exception: fire on turn 1 of a fresh conv so JHB is
            # seeded before turns 2-4 inject. After the first fire, batch
            # every BATCH_SIZE turns.
            fire_threshold = (
                1 if not self._batch_first_fired.get(conv_id) else self.BATCH_SIZE
            )
            if len(buf) < fire_threshold:
                return None
            batch = buf[:fire_threshold]
            del buf[:fire_threshold]
            self._batch_first_fired[conv_id] = True

            # Race fix (2026-05-10 follow-up to W2.9.21 §4.2): assign turn ids
            # and persist raw turns to the retriever JSONL synchronously here,
            # under the same buffer guard that already serializes fire
            # ordering. Previously this work lived inside
            # _encode_batch_save_locked, so the chat thread's next
            # _build_recent_window read could land while the batch was queued
            # behind a still-running prior fire — disk would still hold only
            # the previous batch, collapsing recent_window for one turn per
            # boundary. Heavy work (encoder LLM call + retriever.index_turn
            # embeddings) stays inside the encode lock; what moves out is
            # only the fast file append + counter increment.
            self._fire_batch_locked(conv_id, batch)

        # Hand off to the batch dispatcher (mirrors the original async/thread
        # split so behavior under both event-loop and thread contexts is
        # preserved).
        self._dispatch_batch_encode_async(
            project_path,
            conv_id,
            batch,
            on_token=on_token,
            on_done=on_done,
        )
        last_turn = batch[-1].get("_turn_id") if batch else None
        return int(last_turn) if isinstance(last_turn, int) else None

    def force_flush_batch(self, conv_id: str) -> None:
        """Backpressure helper for W2.9.23. Drain whatever is in
        _batch_buffer right now even if BATCH_SIZE not reached, so
        _ensure_window_coverage can make forward progress when chat
        has outpaced encoder by more than count turns.
        """
        with self._batch_buffer_guard:
            buf = self._batch_buffer.get(conv_id, [])
            if not buf:
                return
            batch = buf[:]
            self._batch_buffer[conv_id] = []
            self._fire_batch_locked(conv_id, batch)

        self._dispatch_batch_encode_async(
            None,
            conv_id,
            batch,
        )

    def _fire_batch_locked(self, conv_id: str, batch: list[dict[str, Any]]) -> None:
        """Assign TIDs and persist raw turns synchronously.
        Caller MUST hold _batch_buffer_guard.
        """
        import sys as _sys
        for entry in batch:
            tid = self._increment_turn(conv_id)
            entry["_turn_id"] = tid
            try:
                self.retriever.save_turn(
                    tid,
                    entry.get("user", "") or "",
                    entry.get("assistant", "") or "",
                    conv_id,
                )
            except Exception as exc:
                print(
                    f"[jlc:slim] save_turn failed conv={conv_id} turn={tid}: {exc}",
                    file=_sys.stderr,
                )

    def _dispatch_batch_encode_async(
        self,
        project_path: str | None,
        conv_id: str,
        batch: list[dict[str, Any]],
        on_token: Any | None = None,
        on_done: Any | None = None,
    ) -> None:
        """W2.9.21 §4.2: fire-and-forget batch encode dispatch.

        Mirrors the encode_and_save_async dispatch pattern (per-conv lock +
        inflight counter + asyncio/thread split) but routes through
        _encode_batch_save_locked which calls encoder.encode with
        batch_turns and iterates per-turn side effects (retriever.save_turn /
        index_turn) so recall stays per-turn even though the encoder sees
        the 5-turn narrative as one stretch.
        """
        lock = self._get_encode_lock(conv_id)
        self._inflight_inc(conv_id)
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if running:
            loop = asyncio.get_running_loop()

            async def _wait_then_encode():
                await asyncio.to_thread(lock.acquire)
                try:
                    await self._encode_batch_save_locked(
                        project_path,
                        conv_id,
                        batch,
                        on_token=on_token,
                        on_done=on_done,
                    )
                finally:
                    try:
                        lock.release()
                    except Exception:
                        pass

            task = loop.create_task(_wait_then_encode())

            def _on_done(_t, _c=conv_id, _self=self):
                try:
                    _self._inflight_dec(_c)
                except Exception:
                    pass

            task.add_done_callback(_on_done)
        else:
            def runner(_lock=lock, _c=conv_id, _self=self):
                try:
                    _lock.acquire()
                    asyncio.run(
                        _self._encode_batch_save_locked(
                            project_path,
                            _c,
                            batch,
                            on_token=on_token,
                            on_done=on_done,
                        )
                    )
                finally:
                    try:
                        _lock.release()
                    except Exception:
                        pass
                    try:
                        _self._inflight_dec(_c)
                    except Exception:
                        pass

            self._background_encode_executor.submit(runner)
        return

    def _encode_and_save_async_legacy(
        self,
        project_path: str | None,
        conv_id: str,
        user_msg: str,
        assistant_msg: str,
        llm_meta: dict[str, Any] | None = None,
        on_token: Any | None = None,
        on_done: Any | None = None,
    ) -> None:
        """W2.9.21 §4.2: legacy single-turn dispatch path. Retained for
        tests and standalone callers that bypass the buffer (e.g. _poc
        scripts). Production chat path goes through encode_and_save_async
        which now buffers 5 turns before firing the encoder.
        """
        lock = self._get_encode_lock(conv_id)
        # idea #12 step 2 (UI fix 2026-05-10): bump in-flight counter BEFORE
        # the worker is dispatched so the backlog throttle sees the queued
        # encode, then move lock.acquire() INTO the worker. The previous
        # design held lock.acquire() in the caller thread, which serialized
        # turn N's chat reply on turn N-1's encoder finishing — manifesting
        # as a UI send-button lock on every turn after the first. The Phase
        # 1.6 race the caller-side acquire was guarding (wait_for_pending
        # winning the lock-free start window) is now covered by the inflight
        # counter, which the chat throttle reads directly.
        self._inflight_inc(conv_id)
        try:
            asyncio.get_running_loop()
            running = True
        except RuntimeError:
            running = False

        if running:
            loop = asyncio.get_running_loop()

            async def _wait_then_encode():
                # Acquire the per-conv lock off the event loop so a long
                # in-flight encode does not stall coroutine scheduling for
                # other connections.
                await asyncio.to_thread(lock.acquire)
                try:
                    await self._encode_and_save_locked(
                        project_path,
                        conv_id,
                        user_msg,
                        assistant_msg,
                        llm_meta=llm_meta,
                        on_token=on_token,
                        on_done=on_done,
                    )
                finally:
                    try:
                        lock.release()
                    except Exception:
                        pass

            task = loop.create_task(_wait_then_encode())

            def _on_done(_t, _c=conv_id, _self=self):
                # Bulletproof: inflight decrement MUST run on success AND
                # failure. Lock release is paired with acquire inside the
                # task itself so a cancelled coroutine before acquire does
                # not double-release.
                try:
                    _self._inflight_dec(_c)
                except Exception:
                    pass

            task.add_done_callback(_on_done)
        else:
            def runner(_lock=lock, _c=conv_id, _self=self):
                acquired = False
                try:
                    _lock.acquire()
                    acquired = True
                    asyncio.run(
                        _self._encode_and_save_locked(
                            project_path,
                            _c,
                            user_msg,
                            assistant_msg,
                            llm_meta=llm_meta,
                            on_token=on_token,
                            on_done=on_done,
                        )
                    )
                finally:
                    try:
                        _self._inflight_dec(_c)
                    finally:
                        if acquired:
                            try:
                                _lock.release()
                            except Exception:
                                pass
            self._background_encode_executor.submit(runner)


    def _current_turn_id(self, conv_id: str) -> int:
        """ID of the most recently assigned turn (disk or pending batch)."""
        meta = self._load_meta(conv_id)
        last_tid = int(meta.get("turn", 0))
        with self._batch_buffer_guard:
            buf_len = len(self._batch_buffer.get(conv_id, []))
        return last_tid + buf_len

    def ensure_window_coverage(self, conv_id: str) -> float:
        """W2.9.23: backpressure. Returns total seconds blocked."""
        cfg = getattr(self.config, "conversation_tail", None)
        count = int(getattr(cfg, "count", 5)) if cfg else 5
        if count <= 0:
            return 0.0

        blocked_s = 0.0
        max_iters = 60
        import time
        import sys as _sys
        for i in range(max_iters):
            cur = self._current_turn_id(conv_id)
            head = getattr(self.encoder, "last_enc_turn_id", 0) or 0
            if (cur - head) <= count:
                return blocked_s
            
            t0 = time.monotonic()
            if self.encode_in_flight(conv_id) > 0:
                self.wait_for_pending_encode(timeout=10.0, session_id=conv_id)
            else:
                self.force_flush_batch(conv_id)
            blocked_s += time.monotonic() - t0
        
        _sys.stderr.write(
            f"[jlc:backpressure] gap not closed after {max_iters} iters "
            f"conv={conv_id} cur={cur} head={head} count={count}\n"
        )
        return blocked_s

    def wait_for_pending_encode(self, timeout: float = 600.0, session_id: str | None = None) -> None:
        """Block until any in-flight encode for this conv_id finishes.

        Called from inject_pre_call so the next LLM call sees a fresh JHB.
        Acquire + release the per-conv encode lock — if encoder is mid-flight
        we wait for it; if it's already done, this returns instantly.

        timeout: maximum seconds to wait. On expiry, log a stderr warning
        and return WITHOUT releasing — the holder still owns the lock and
        will release on its own. Intended for cold-start races on turn 1
        of a 100-turn run, where the encoder LLM warm-up + bge-m3 load can
        take longer than a chat turn would otherwise tolerate. Default 10
        minutes is generous enough for any normal encode + safe enough that
        a truly stuck encoder no longer wedges chat indefinitely.
        """
        conv_id = _normalize_session_id(session_id)
        lock = self._get_encode_lock(conv_id)
        if lock.acquire(timeout=timeout):
            lock.release()
        else:
            import sys
            sys.stderr.write(
                f"[wait_for_pending_encode] timed out after {timeout}s "
                f"for conv_id={conv_id!r}, proceeding with possibly-stale JHB\n"
            )

    def load_conversation_tail(self, conv_id: str) -> list[TailEntry]:
        path = self._tail_path(conv_id)
        if not path.exists():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        entries: list[TailEntry] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                entries.append(
                    TailEntry(
                        turn_id=str(item.get("turn_id", "")),
                        summary=str(item.get("summary", "")),
                        token_count=int(item.get("token_count", 0)),
                        created_at=float(item.get("created_at", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                continue
        return entries

    def save_conversation_tail(self, conv_id: str, entries: list[TailEntry]) -> None:
        path = self._tail_path(conv_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(entry) for entry in entries if entry.summary.strip()]
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

    async def regenerate_conversation_tail(self, conv_id: str) -> list[TailEntry]:
        cfg = self.config.conversation_tail
        count = int(getattr(cfg, "count", 5) or 0)
        if not getattr(cfg, "enabled", True) or count <= 0:
            return self.load_conversation_tail(conv_id)
        try:
            turns = self.retriever.load_turns(conv_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[jlc:tail] retriever.load_turns failed conv={conv_id}: {exc}", file=__import__('sys').stderr)
            return self.load_conversation_tail(conv_id)
        recent = turns[-count:] if turns else []
        if not recent:
            entries: list[TailEntry] = []
            self.save_conversation_tail(conv_id, entries)
            current = self.load_jhb(conv_id)
            self.save_jhb(conv_id, self._strip_tail_section(current))
            return entries

        entries = await self.encoder.compress_recent_turns(
            recent,
            max_tokens_per_turn=int(getattr(cfg, "max_tokens_per_turn", 220) or 220),
        )
        self.save_conversation_tail(conv_id, entries)
        current = self.load_jhb(conv_id)
        self.save_jhb(conv_id, self._strip_tail_section(current))
        return entries

    _TAIL_HEADER = "## Conversation Tail"
    _TAIL_META = "_(직전 N턴 압축 맥락 — 시간순. chat LLM은 이 섹션으로 최근 흐름 파악)_"

    def _strip_tail_section(self, text: str) -> str:
        # Anything from `## Conversation Tail` to EOF is regenerated from
        # retriever_turns.jsonl on each reroll; encoder echoes are discarded.
        idx = text.find(self._TAIL_HEADER)
        if idx == -1:
            return text
        return text[:idx].rstrip()

    def _format_tail_section(self, entries: list[TailEntry]) -> str:
        if not entries:
            return ""
        meta = self._TAIL_META.replace("N턴", f"{len(entries)}턴")
        lines = [self._TAIL_HEADER, "", meta, ""]
        for entry in entries:
            turn = entry.turn_id.strip() or "turn"
            summary = entry.summary.strip()
            if summary:
                lines.append(f"{turn}: {summary}")
        return "\n".join(lines).strip()

    def render_jhb(self, session_id: str | None = None) -> str:
        # jhb.md now persists with the tail section merged in (see save_jhb),
        # so render is just a load. Kept for back-compat with callers.
        return self.load_jhb(_normalize_session_id(session_id)).strip()

    def _inflight_inc(self, conv_id: str) -> None:
        """idea #12 step 4 helper — increment per-conv encode in-flight count."""
        with self._encode_locks_guard:
            self._encode_in_flight[conv_id] = self._encode_in_flight.get(conv_id, 0) + 1

    def _inflight_dec(self, conv_id: str) -> None:
        """idea #12 step 4 helper — decrement per-conv encode in-flight count.

        Floors at zero and removes empty entries so the dict cannot grow
        without bound across many short-lived conv_ids.
        """
        with self._encode_locks_guard:
            cur = self._encode_in_flight.get(conv_id, 0)
            if cur > 1:
                self._encode_in_flight[conv_id] = cur - 1
            else:
                self._encode_in_flight.pop(conv_id, None)

    def encode_in_flight(self, conv_id: str) -> int:
        """Public probe: how many encodes currently hold (or are about to
        hold) the JHB lock for this conv. Chat-side backlog throttle reads
        this before deciding whether to wait or proceed with stale JHB.
        """
        with self._encode_locks_guard:
            return self._encode_in_flight.get(conv_id, 0)

    def _get_encode_lock(self, conv_id: str) -> threading.Lock:
        with self._encode_locks_guard:
            existing = self._encode_locks.get(conv_id)
            if existing is not None:
                self._encode_locks.move_to_end(conv_id)
                return existing
            if len(self._encode_locks) >= _MAX_CACHED_ENCODE_LOCKS:
                # Evict oldest, but never an actively-held lock — evicting a
                # held lock would let a second caller mint a fresh lock for
                # the same conv_id and break serialization while the holder
                # is still inside the critical section.
                oldest_key, oldest_lock = self._encode_locks.popitem(last=False)
                if oldest_lock.locked():
                    self._encode_locks[oldest_key] = oldest_lock
                    self._encode_locks.move_to_end(oldest_key, last=False)
            lock = threading.Lock()
            self._encode_locks[conv_id] = lock
            return lock

    async def _encode_and_save(
        self,
        project_path: str | None,
        conv_id: str,
        user_msg: str,
        assistant_msg: str,
        llm_meta: dict[str, Any] | None = None,
        on_token: Any | None = None,
        on_done: Any | None = None,
    ) -> None:
        lock = self._get_encode_lock(conv_id)
        await asyncio.to_thread(lock.acquire)
        try:
            await self._encode_and_save_locked(
                project_path,
                conv_id,
                user_msg,
                assistant_msg,
                llm_meta=llm_meta,
                on_token=on_token,
                on_done=on_done,
            )
        finally:
            lock.release()

    async def _encode_and_save_locked(
        self,
        project_path: str | None,
        conv_id: str,
        user_msg: str,
        assistant_msg: str,
        llm_meta: dict[str, Any] | None = None,
        on_token: Any | None = None,
        on_done: Any | None = None,
    ) -> None:
        prev_jhb = self.load_jhb(conv_id)
        # Path B (2026-05-09): legacy `## Conversation Tail` blocks left over
        # from patch-4-era runs are stripped before the encoder sees them, so
        # the encoder is not tempted to re-emit a tail section that we no
        # longer want in durable JHB. Fresh conv_ids are unaffected.
        prev_jhb = self._strip_tail_section(prev_jhb)
        prev_project_md = self._load_project_md(project_path)
        prev_sha = self._content_sha1(prev_jhb)

        turn = self._increment_turn(conv_id)

        try:
            self.retriever.save_turn(turn, user_msg, assistant_msg, conv_id)
        except Exception as exc:
            print(f"[jlc:slim] retriever.save_turn failed conv={conv_id}: {exc}", file=__import__('sys').stderr)
        try:
            await self.retriever.index_turn({"turn": turn, "user": user_msg, "assistant": assistant_msg}, conv_id)
        except Exception as exc:
            print(f"[jlc:slim] retriever.index_turn failed conv={conv_id}: {exc}", file=__import__('sys').stderr)

        # JHB-Coding mode: when a project is active, give the JHB more headroom
        # so code/diff/file paths can stay in P0 longer than chat-mode allows.
        # Plain chat keeps the conservative 2K target.
        coding_target = self.config.jhb.target_tokens
        try:
            encode_kwargs = {
                "prev_jhb": prev_jhb,
                "user_msg": user_msg,
                "assistant_msg": assistant_msg,
                "prev_project_md": prev_project_md,
                "project_active": bool(project_path),
                "target_tokens": coding_target,
                "current_turn": turn,
            }
            if on_token is not None:
                encode_kwargs["on_token"] = on_token
            updated_jhb, updated_project_md, retry_count = await self.encoder.encode(
                **encode_kwargs,
            )
        except Exception as exc:
            print(f"[jlc:slim] encoder failed: {exc}", file=__import__('sys').stderr)
            self.encoder.last_error = str(exc)
            updated_jhb, updated_project_md, retry_count = prev_jhb, prev_project_md, -1

        # idea #12 step 4: stash the chat-turn id this encode just finished.
        # Bench meter reads encoder.last_enc_turn_id to compute enc_lag_turns
        # against the next chat turn's id, so analysis joins lag-correctly.
        try:
            self.encoder.last_enc_turn_id = turn
        except Exception:
            pass

        # idea #12 step 4: append a per-encode row to turns_enc.jsonl so
        # analysis can reconstruct the encoder timeline independently of
        # the chat-side meter (which carries last-completed values that may
        # belong to an earlier turn under background mode). Best-effort —
        # never block the encode path on file IO.
        try:
            self._append_turns_enc_row(
                conv_id=conv_id,
                turn_id=turn,
                retry_count=retry_count,
            )
        except Exception as exc:
            print(
                f"[jlc:slim] turns_enc append failed conv={conv_id} turn={turn}: {exc}",
                file=__import__('sys').stderr,
            )

        if callable(append_pi_sidecar_encoder_turn):
            try:
                append_pi_sidecar_encoder_turn(
                    turn_id=turn,
                    project_path=project_path,
                    encoder_meta={
                        "enc_in": getattr(self.encoder, "last_enc_in", 0),
                        "enc_think": getattr(self.encoder, "last_enc_think", 0),
                        "enc_out": getattr(self.encoder, "last_enc_out", 0),
                        "enc_seconds": getattr(self.encoder, "last_enc_seconds", 0.0),
                        "jhb_tokens": getattr(self.encoder, "last_jhb_tokens", 0),
                        "jhb_delta": getattr(self.encoder, "last_jhb_delta", 0),
                        "jhb_delta_tokens": getattr(self.encoder, "last_jhb_delta_tokens", 0),
                        "jhb_delta_chars": getattr(self.encoder, "last_jhb_delta_chars", 0),
                        "jhb_diff_added": getattr(self.encoder, "last_jhb_diff_added", 0),
                        "jhb_diff_removed": getattr(self.encoder, "last_jhb_diff_removed", 0),
                        "failure_mode": getattr(self.encoder, "last_failure_mode", "skipped_empty_input"),
                        "encoder_retries": retry_count,
                        "enc_model_spec": _read_encoder_model_spec(self._encoder_config_path),
                        "enc_reasoning_effort": (__import__("os").environ.get("JLC_ENCODER_REASONING_EFFORT", "none").strip().lower() or "none"),
                    },
                )
            except Exception as exc:
                print(
                    f"[jlc:slim] pi-sidecar encoder append failed conv={conv_id} turn={turn}: {exc}",
                    file=__import__('sys').stderr,
                )
        if callable(append_pi_sidecar_meter_turn):
            try:
                append_pi_sidecar_meter_turn(
                    turn_id=turn,
                    project_path=project_path,
                    meter_line=self.encoder.format_post_encode_meter_line(),
                    session_id=conv_id,
                )
            except Exception as exc:
                print(
                    f"[jlc:slim] pi-sidecar meter append failed conv={conv_id} turn={turn}: {exc}",
                    file=__import__('sys').stderr,
                )
        try:
            _append_paperlog_row(
                session_id=conv_id,
                turn_id=turn,
                user_message=user_msg,
                assistant_message=assistant_msg,
                meter_line=self.encoder.format_post_encode_meter_line(),
            )
        except Exception as exc:
            print(
                f"[jlc:slim] paperlog append failed conv={conv_id} turn={turn}: {exc}",
                file=__import__('sys').stderr,
            )

        self.save_jhb(conv_id, updated_jhb)
        if callable(on_done):
            try:
                on_done(updated_jhb)
            except Exception:
                pass
        if project_path and updated_project_md != prev_project_md:
            self._save_project_md(project_path, updated_project_md)

        # Path B (2026-05-09): the conversation tail is no longer the
        # encoder's responsibility. Raw last-N turns are injected at chat
        # time (see ChatTurn._build_recent_window). This isolates the
        # encoder to durable JHB compression, where weaker
        # instruction-followers (e.g. devstral-small-2:24b) stay accurate
        # — they were hallucinating turn IDs in the encoder-emitted tail
        # under patch 4.

        try:
            self.jre.record_changes(turn, prev_jhb, updated_jhb, session_id=conv_id)
        except Exception as exc:
            print(f"[jlc:slim] jre.record_changes failed conv={conv_id}: {exc}", file=__import__('sys').stderr)

        try:
            tags = self.tagger.extract(user_msg, assistant_msg)
        except Exception as exc:
            print(f"[jlc:slim] tagger.extract failed: {exc}", file=__import__('sys').stderr)
            tags = []

        try:
            user_tokens = self.encoder.count_tokens(user_msg)
            assistant_tokens = self.encoder.count_tokens(assistant_msg)
            jhb_tokens = self.encoder.count_tokens(updated_jhb)
            entry = {
                "turn": turn,
                "user": user_msg,
                "assistant": assistant_msg,
                "user_tokens": user_tokens,
                "assistant_tokens": assistant_tokens,
                "jhb_tokens": jhb_tokens,
                "jhb_sha1_before": prev_sha,
                "jhb_sha1_after": self._content_sha1(updated_jhb),
                "tags": tags,
                "encoder_retry_count": retry_count,
                # Phase 1.9 mandatory (MiniMax review): persist cwd so a
                # global JHB recall can still tell which project a turn
                # belonged to. Without this, "그때 수정한 거?" is ambiguous
                # across mixed-project history.
                "project_path": project_path or "",
            }
            if llm_meta:
                entry["llm_meta"] = llm_meta
            self.turn_logger.append(entry, session_id=conv_id)
        except Exception:
            pass

        try:
            self._update_tags_index(conv_id, turn, tags)
        except Exception as exc:
            print(f"[jlc:slim] tags_index update failed: {exc}", file=__import__('sys').stderr)

        try:
            self.graph.accumulate(turn, tags, session_id=conv_id)
        except Exception as exc:
            print(f"[jlc:slim] graph.accumulate failed: {exc}", file=__import__('sys').stderr)

        try:
            self.git_manager.auto_commit(turn, session_id=conv_id)
        except Exception as exc:
            print(f"[jlc:slim] git_manager.auto_commit failed: {exc}", file=__import__('sys').stderr)

    async def _encode_batch_save_locked(
        self,
        project_path: str | None,
        conv_id: str,
        batch: list[dict[str, Any]],
        on_token: Any | None = None,
        on_done: Any | None = None,
    ) -> None:
        """W2.9.21 §4.2: 5-turn batch encode + per-turn retriever indexing.

        Side effects (retriever.save_turn / index_turn / jre.record_changes /
        tagger / log entry / git_manager) happen for each individual turn so
        recall, graph, and audit trails stay per-turn — only the encoder LLM
        call collapses into one batch invocation. JHB is rewritten once at
        the end with the encoder's batch-aware output.
        """
        import sys as _sys

        prev_jhb = self.load_jhb(conv_id)
        prev_jhb = self._strip_tail_section(prev_jhb)
        prev_project_md = self._load_project_md(project_path)
        prev_sha = self._content_sha1(prev_jhb)

        # Race fix (2026-05-10): turn-id assignment and retriever.save_turn
        # already ran synchronously in encode_and_save_async before dispatch,
        # so each batch entry carries its assigned `_turn_id`. Only the slow
        # embedding index_turn stays inside the encode lock here. Falling
        # back to _increment_turn keeps us robust if a non-batched legacy
        # caller dispatches a raw batch (e.g. _poc scripts).
        turn_ids: list[int] = []
        for t in batch:
            turn_id = t.get("_turn_id")
            if turn_id is None:
                turn_id = self._increment_turn(conv_id)
                try:
                    self.retriever.save_turn(turn_id, t.get("user", "") or "", t.get("assistant", "") or "", conv_id)
                except Exception as exc:
                    print(f"[jlc:slim] retriever.save_turn failed conv={conv_id} turn={turn_id}: {exc}", file=_sys.stderr)
            turn_ids.append(turn_id)
            try:
                await self.retriever.index_turn(
                    {"turn": turn_id, "user": t.get("user", "") or "", "assistant": t.get("assistant", "") or ""},
                    conv_id,
                )
            except Exception as exc:
                print(f"[jlc:slim] retriever.index_turn failed conv={conv_id} turn={turn_id}: {exc}", file=_sys.stderr)

        coding_target = self.config.jhb.target_tokens
        try:
            encode_kwargs: dict[str, Any] = {
                "prev_jhb": prev_jhb,
                "user_msg": "",
                "assistant_msg": "",
                "prev_project_md": prev_project_md,
                "project_active": bool(project_path),
                "target_tokens": coding_target,
                "batch_turns": batch,
                "current_turn": turn_ids[-1] if turn_ids else None,
            }
            if on_token is not None:
                encode_kwargs["on_token"] = on_token
            updated_jhb, updated_project_md, retry_count = await self.encoder.encode(**encode_kwargs)
        except Exception as exc:
            print(f"[jlc:slim] batch encoder failed: {exc}", file=_sys.stderr)
            self.encoder.last_error = str(exc)
            updated_jhb, updated_project_md, retry_count = prev_jhb, prev_project_md, -1

        last_turn = turn_ids[-1] if turn_ids else 0
        try:
            self.encoder.last_enc_turn_id = last_turn
        except Exception:
            pass

        try:
            self._append_turns_enc_row(conv_id=conv_id, turn_id=last_turn, retry_count=retry_count)
        except Exception as exc:
            print(f"[jlc:slim] turns_enc append failed conv={conv_id} turn={last_turn}: {exc}", file=_sys.stderr)

        if callable(append_pi_sidecar_encoder_turn):
            try:
                append_pi_sidecar_encoder_turn(
                    turn_id=last_turn,
                    project_path=project_path,
                    encoder_meta={
                        "enc_in": getattr(self.encoder, "last_enc_in", 0),
                        "enc_think": getattr(self.encoder, "last_enc_think", 0),
                        "enc_out": getattr(self.encoder, "last_enc_out", 0),
                        "enc_seconds": getattr(self.encoder, "last_enc_seconds", 0.0),
                        "jhb_tokens": getattr(self.encoder, "last_jhb_tokens", 0),
                        "jhb_delta": getattr(self.encoder, "last_jhb_delta", 0),
                        "jhb_delta_tokens": getattr(self.encoder, "last_jhb_delta_tokens", 0),
                        "jhb_delta_chars": getattr(self.encoder, "last_jhb_delta_chars", 0),
                        "jhb_diff_added": getattr(self.encoder, "last_jhb_diff_added", 0),
                        "jhb_diff_removed": getattr(self.encoder, "last_jhb_diff_removed", 0),
                        "failure_mode": getattr(self.encoder, "last_failure_mode", "skipped_empty_input"),
                        "encoder_retries": retry_count,
                        "enc_model_spec": _read_encoder_model_spec(self._encoder_config_path),
                        "enc_reasoning_effort": (__import__("os").environ.get("JLC_ENCODER_REASONING_EFFORT", "none").strip().lower() or "none"),
                    },
                )
            except Exception as exc:
                print(f"[jlc:slim] pi-sidecar encoder append failed conv={conv_id} turn={last_turn}: {exc}", file=_sys.stderr)
        if callable(append_pi_sidecar_meter_turn):
            try:
                append_pi_sidecar_meter_turn(
                    turn_id=last_turn,
                    project_path=project_path,
                    meter_line=self.encoder.format_post_encode_meter_line(),
                    session_id=conv_id,
                )
            except Exception as exc:
                print(f"[jlc:slim] pi-sidecar meter append failed conv={conv_id} turn={last_turn}: {exc}", file=_sys.stderr)
        try:
            last_user_msg = ""
            last_assistant_msg = ""
            if batch:
                last_item = batch[-1] if isinstance(batch[-1], dict) else {}
                last_user_msg = str(last_item.get("user", "") or "")
                last_assistant_msg = str(last_item.get("assistant", "") or "")
            _append_paperlog_row(
                session_id=conv_id,
                turn_id=last_turn,
                user_message=last_user_msg,
                assistant_message=last_assistant_msg,
                meter_line=self.encoder.format_post_encode_meter_line(),
            )
        except Exception as exc:
            print(f"[jlc:slim] paperlog append failed conv={conv_id} turn={last_turn}: {exc}", file=_sys.stderr)

        self.save_jhb(conv_id, updated_jhb)
        if callable(on_done):
            try:
                on_done(updated_jhb)
            except Exception:
                pass
        if project_path and updated_project_md != prev_project_md:
            self._save_project_md(project_path, updated_project_md)

        try:
            self.jre.record_changes(last_turn, prev_jhb, updated_jhb, session_id=conv_id)
        except Exception as exc:
            print(f"[jlc:slim] jre.record_changes failed conv={conv_id}: {exc}", file=_sys.stderr)

        try:
            self.git_manager.auto_commit(last_turn, session_id=conv_id)
        except Exception as exc:
            print(f"[jlc:slim] git_manager.auto_commit failed: {exc}", file=_sys.stderr)

    def load_jhb(self, conv_id: str) -> str:
        path = self._jhb_path(conv_id)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def save_jhb(self, conv_id: str, jhb: str) -> bool:
        # Path B (2026-05-09): JHB is durable-only; the conversation tail
        # is injected raw at chat time, not stored here. Strip any tail
        # block the encoder may still emit out of habit before persisting.
        jhb = self._strip_tail_section(jhb)
        merged = unicodedata.normalize("NFC", jhb).strip()
        new_sha = self._content_sha1(merged)
        meta = self._load_meta(conv_id)
        path = self._jhb_path(conv_id)
        if meta.get("jhb_sha1") == new_sha and path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, merged)
        meta["jhb_sha1"] = new_sha
        self._save_meta(conv_id, meta)
        return True

    async def close(self) -> None:
        executor = getattr(self, "_background_encode_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)
        self.graph.flush_all()
        await self._encoder_llm.close()
        await self.retriever.close()
        await self.jre.close()

    def _load_project_md(self, project_path: str | None) -> str:
        if not project_path:
            return ""
        try:
            path = Path(project_path) / "JARVIS.md"
        except TypeError:
            return ""
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _save_project_md(self, project_path: str | None, content: str) -> None:
        if not project_path:
            return
        try:
            path = Path(project_path) / "JARVIS.md"
        except TypeError:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, content or "")

    def _increment_turn(self, conv_id: str) -> int:
        # Private: only called from _encode_and_save_locked which already holds
        # the encode lock for conv_id, so the read-modify-write is serialized
        # without acquiring the lock here (re-acquire would deadlock the plain
        # Lock cross-thread acquire/release pattern used by encode_and_save).
        meta = self._load_meta(conv_id)
        turn = int(meta.get("turn", 0)) + 1
        meta["turn"] = turn
        meta["last_updated"] = datetime.now(UTC).replace(microsecond=0).isoformat()
        self._save_meta(conv_id, meta)
        return turn

    def _load_meta(self, conv_id: str) -> dict[str, Any]:
        path = self._meta_path(conv_id)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _save_meta(self, conv_id: str, meta: dict[str, Any]) -> None:
        path = self._meta_path(conv_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(meta, ensure_ascii=False, indent=2))

    def _update_tags_index(self, conv_id: str, turn: int, tags: list[str]) -> None:
        if not tags:
            return
        tags_path = self._jhb_root / self._sanitize_conv_id(conv_id) / "tags.json"
        if tags_path.exists():
            try:
                tags_data = json.loads(tags_path.read_text(encoding="utf-8"))
            except Exception:
                tags_data = {"tags": {}}
        else:
            tags_data = {"tags": {}}

        tags_data = self.tagger.update_index(tags_data, turn, tags)
        tags_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(tags_path, json.dumps(tags_data, ensure_ascii=False, indent=2))

    def _jhb_path(self, conv_id: str) -> Path:
        return self._jhb_root / self._sanitize_conv_id(conv_id) / "jhb.md"

    def _meta_path(self, conv_id: str) -> Path:
        return self._jhb_root / self._sanitize_conv_id(conv_id) / "meta.json"

    def _tail_path(self, conv_id: str) -> Path:
        return self._jhb_root / self._sanitize_conv_id(conv_id) / "conversation_tail.json"

    def _turns_enc_path(self, conv_id: str) -> Path:
        """idea #12 step 4 — per-conv encoder timeline. One JSONL row per
        successful encode, written from _encode_and_save_locked AFTER the
        encoder has populated last_enc_*. Lives next to jhb.md so analysis
        can join chat turns.jsonl with encoder turns_enc.jsonl on turn_id.
        """
        return self._jhb_root / self._sanitize_conv_id(conv_id) / "turns_enc.jsonl"

    def _append_turns_enc_row(
        self,
        conv_id: str,
        turn_id: int,
        retry_count: int,
    ) -> None:
        """idea #12 step 4 helper. Best-effort; caller already wraps in try."""
        enc = self.encoder
        row = {
            "turn_id": turn_id,
            "completed_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "enc_in": getattr(enc, "last_enc_in", 0),
            "enc_think": getattr(enc, "last_enc_think", 0),
            "enc_out": getattr(enc, "last_enc_out", 0),
            "enc_seconds": getattr(enc, "last_enc_seconds", 0.0),
            "jhb_tokens": getattr(enc, "last_jhb_tokens", 0),
            "jhb_delta": getattr(enc, "last_jhb_delta", 0),
            "jhb_delta_tokens": getattr(enc, "last_jhb_delta_tokens", 0),
            "jhb_delta_chars": getattr(enc, "last_jhb_delta_chars", 0),
            "jhb_diff_added": getattr(enc, "last_jhb_diff_added", 0),
            "jhb_diff_removed": getattr(enc, "last_jhb_diff_removed", 0),
            "failure_mode": getattr(enc, "last_failure_mode", "skipped_empty_input"),
            "encoder_retries": retry_count,
        }
        path = self._turns_enc_path(conv_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False) + "\n"
        # Append-only; jhb.md uses atomic replace, but turns_enc.jsonl is a
        # log so a normal append is fine. The encode lock serializes writers
        # for a given conv_id, so no cross-process races to worry about.
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    _WINDOWS_RESERVED = frozenset({
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    })

    @staticmethod
    def _sanitize_conv_id(conv_id: str) -> str:
        raw = conv_id.strip()
        if not raw:
            return "default"
        # Windows reserved name guard — folder creation fails on `CON`, `NUL`, etc.
        base = raw.split(".", 1)[0].upper()
        if base in JarvisAgentic._WINDOWS_RESERVED:
            raw = f"_{raw}_"
        invalid = '<>:"/\\|?*'
        table = str.maketrans({ch: "_" for ch in invalid})
        return raw.translate(table).replace("..", "_")

    @staticmethod
    def _content_sha1(content: str) -> str:
        normalized = unicodedata.normalize("NFC", content)
        return hashlib.sha1(normalized.encode("utf-8", errors="replace")).hexdigest()

