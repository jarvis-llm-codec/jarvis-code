"""JLC Retriever — raw turn storage + semantic search for context recall."""
from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import threading
import time
import unicodedata
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from . import recall_snippets, recall_trace
from .embedder import LocalEmbedder

try:
    from jarvis_sidecar.file_locks import cross_process_file_lock, locked_atomic_write_text
    from jarvis_sidecar.raw_store import (
        normalize_origin_window,
        normalize_turn_origin,
        _timestamp_local_date as timestamp_local_date,
    )
except Exception:  # pragma: no cover - sidecar package unavailable in standalone mode
    cross_process_file_lock = None
    locked_atomic_write_text = None
    normalize_origin_window = None
    normalize_turn_origin = None
    timestamp_local_date = None

_MAX_CACHED_LOCKS = 1000
_MAX_BM25_CACHE = 500
_MAX_LITERAL_INDEX_CACHE = 128
_LITERAL_INDEX_VERSION = 3
_SESSION_ID = "jarvis_session"


def _normalize_session_id(session_id: str | None) -> str:
    raw = str(session_id or "").strip()
    return raw or _SESSION_ID


def _normalize_origin(origin: str | None) -> str:
    if callable(normalize_turn_origin):
        return normalize_turn_origin(origin)
    value = str(origin or "").strip()
    return value if value in {"user", "monologue_directive", "monologue_report"} else "user"


def _normalize_origin_window(origin_window: str | None) -> str | None:
    if callable(normalize_origin_window):
        return normalize_origin_window(origin_window)
    value = str(origin_window or "").strip()
    return value[:64] if value else None


def _normalize_origin_window_label(label: str | None) -> str | None:
    text = "".join(ch for ch in str(label or "") if ord(ch) >= 32 and ord(ch) != 127).strip()
    return text[:32] if text else None


def _origin_fields(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "origin": _normalize_origin(source.get("origin")),
        "origin_window": _normalize_origin_window(source.get("origin_window")),
        "origin_window_label": _normalize_origin_window_label(source.get("origin_window_label")),
    }


def _fragment_for_turn(turn: dict[str, Any], *, score: float = 0.0) -> dict[str, Any]:
    return {
        "turn": turn.get("turn", 0),
        "score": round(score, 4),
        "user": turn.get("user", ""),
        "assistant": turn.get("assistant", ""),
        "ts": turn.get("ts", ""),
        **_origin_fields(turn),
    }


def _missing_lock_error(name: str) -> RuntimeError:
    message = (
        f"jarvis_sidecar.file_locks.{name} unavailable; refusing shared retriever write without a lock"
    )
    print(f"[jlc:ret] {message}", file=sys.stderr)
    return RuntimeError(message)


def _require_cross_process_file_lock():
    if not callable(cross_process_file_lock):
        raise _missing_lock_error("cross_process_file_lock")
    return cross_process_file_lock


def _require_locked_atomic_write_text():
    if not callable(locked_atomic_write_text):
        raise _missing_lock_error("locked_atomic_write_text")
    return locked_atomic_write_text


def _atomic_write_text(path: Path, content: str) -> None:
    _require_locked_atomic_write_text()(path, content)

_tokenize_re = re.compile(r"[\w]+", re.UNICODE)
_literal_token_re = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_-]*|[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+",
    re.UNICODE,
)
_literal_entity_re = re.compile(
    r"\b(?:project|artifact|quantity|coordinator|checkpoint|identifier|codename|code\s+name|"
    r"function|class|method|variable|property|slot|tag|flag|key|token)\s+"
    r"([A-Za-z0-9][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)

_LITERAL_STOPWORDS = {
    "about",
    "again",
    "answer",
    "access",
    "archive",
    "artifact",
    "asked",
    "audit",
    "bead",
    "being",
    "budget",
    "check",
    "checkpoint",
    "code",
    "coordinator",
    "crate",
    "did",
    "does",
    "exactly",
    "final",
    "flag",
    "form",
    "from",
    "fallback",
    "graduation",
    "have",
    "handoff",
    "key",
    "label",
    "lane",
    "line",
    "marker",
    "merge",
    "no",
    "not",
    "other",
    "please",
    "project",
    "quantity",
    "record",
    "release",
    "reply",
    "review",
    "risk",
    "routing",
    "signal",
    "single",
    "slot",
    "specifically",
    "that",
    "this",
    "tag",
    "token",
    "use",
    "value",
    "values",
    "what",
    "when",
    "where",
    "which",
    "with",
}

_LITERAL_NOISE_SUBSTRINGS = (
    "answer:",
    "no record",
    "reply with",
    "single final",
    "exactly this form",
    "being asked",
    "graduation check",
)

# Denial phrases used by chat when it cannot find evidence. A turn whose
# assistant text contains one of these carries no usable evidence — keeping
# it in the top-k starves a real planting turn (e.g. "Parking note: X equals
# Y") of its slot and lets chat echo its own past denial.
#
# Patterns are anchored to the response opening (first 200 chars) so a
# normal answer that happens to mention "no record" deep in the body is not
# misclassified as a denial.
_DENIAL_PATTERNS = (
    "no record",
    "no records",
    "nothing found",
    "nothing logged",
    "nothing on any",
    "came up empty",
    "blank across the board",
    "didn't come up",
    "haven't come up",
    "don't have any record",
    "have no record",
    "can't confirm",
    "cannot confirm",
    "all four came up blank",
    "all blanks",
)

_DENIAL_HEAD_CHARS = 200


def _is_denial_response(assistant_text: str) -> bool:
    head = (assistant_text or "").lower()[:_DENIAL_HEAD_CHARS]
    return any(p in head for p in _DENIAL_PATTERNS)


def _tokenize_for_bm25(text: str) -> list[str]:
    """Lowercased word-token split. Identifiers like `checkCollision` survive
    as one token — code-domain probes ("did we use checkCollision?") match
    exactly, which is the BM25 advantage over embeddings.
    """
    return [t.lower() for t in _tokenize_re.findall(text or "")]


def _literal_casefold(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text or "")).casefold()


def _tokenize_for_literals(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _literal_token_re.finditer(text or ""):
        token = _literal_casefold(match.group(0)).strip()
        if not token or token in _LITERAL_STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
        if "-" in token or "_" in token:
            tokens.extend(part for part in re.split(r"[-_]+", token) if len(part) > 1)
    return tokens


def _literal_entity_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    for match in _literal_entity_re.finditer(query or ""):
        tokens.update(_tokenize_for_literals(match.group(1)))
    return tokens


def _literal_signature_for_path(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {"name": path.name, "exists": False, "mtime_ns": 0, "size": 0}
    return {
        "name": path.name,
        "exists": True,
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _literal_rare_threshold(turn_count: int) -> int:
    return max(3, min(20, int(max(1, turn_count) * 0.08)))


def _literal_idf(df: int, turn_count: int) -> float:
    return math.log(1.0 + (max(1, turn_count) + 1.0) / (max(1, df) + 0.5))


def _is_noise_literal_anchor(anchor: str) -> bool:
    folded = _literal_casefold(anchor)
    if not folded or len(folded) < 3:
        return True
    if "<" in folded or ">" in folded:
        return True
    return any(part in folded for part in _LITERAL_NOISE_SUBSTRINGS)


class RetrieverSearchResult(TypedDict):
    confidence: Literal["HIGH", "MID", "LOW"]
    fragments: list[dict[str, Any]]


class JLCRetriever:
    """Store raw conversation turns and retrieve relevant ones via embedding search."""

    def __init__(
        self,
        storage_root: Path,
        embedder: LocalEmbedder | None = None,
        embed_latency_hook: Callable[[int], None] | None = None,
    ) -> None:
        self._root = storage_root
        # Allow None (lazy injection from JarvisAgentic._get_embedder) — will be set later
        self._embedder = embedder
        # threading.Lock — survives across thread fallback's fresh event loops
        # (asyncio.Lock binds to the loop it was created on, so each thread
        # would have ended up with a fresh non-shared lock).
        # OrderedDict + LRU eviction caps memory at _MAX_CACHED_LOCKS so a
        # process touching thousands of conv_ids does not leak a Lock per id.
        self._index_locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._save_locks: OrderedDict[str, threading.Lock] = OrderedDict()
        self._locks_guard = threading.Lock()
        self._embed_latency_ms_total = 0
        self._embed_latency_hook = embed_latency_hook
        # BM25 corpus cache — [(turn_index, tokenized_text), ...]
        # Keyed by (conv_id, turn_count) so new turns auto-invalidate.
        # OrderedDict + LRU caps at _MAX_BM25_CACHE entries.
        self._bm25_corpus_cache: OrderedDict[str, tuple[int, list[list[str]]]] = OrderedDict()
        self._bm25_cache_guard = threading.Lock()
        self._literal_index_cache: OrderedDict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = OrderedDict()
        self._literal_index_cache_guard = threading.Lock()

    def _index_lock(self, conv_id: str = _SESSION_ID) -> threading.Lock:
        with self._locks_guard:
            existing = self._index_locks.get(conv_id)
            if existing is not None:
                self._index_locks.move_to_end(conv_id)
                return existing
            if len(self._index_locks) >= _MAX_CACHED_LOCKS:
                # Never evict an actively-held lock — see _get_encode_lock.
                oldest_key, oldest_lock = self._index_locks.popitem(last=False)
                if oldest_lock.locked():
                    self._index_locks[oldest_key] = oldest_lock
                    self._index_locks.move_to_end(oldest_key, last=False)
            lock = threading.Lock()
            self._index_locks[conv_id] = lock
            return lock

    def _save_lock(self, conv_id: str = _SESSION_ID) -> threading.Lock:
        with self._locks_guard:
            existing = self._save_locks.get(conv_id)
            if existing is not None:
                self._save_locks.move_to_end(conv_id)
                return existing
            if len(self._save_locks) >= _MAX_CACHED_LOCKS:
                # Never evict an actively-held lock — see _get_encode_lock.
                oldest_key, oldest_lock = self._save_locks.popitem(last=False)
                if oldest_lock.locked():
                    self._save_locks[oldest_key] = oldest_lock
                    self._save_locks.move_to_end(oldest_key, last=False)
            lock = threading.Lock()
            self._save_locks[conv_id] = lock
            return lock

    async def close(self) -> None:
        return None

    # ── Store ──

    def save_turn(
        self,
        turn: int,
        user_msg: str,
        assistant_msg: str,
        session_id: str | None = None,
        *,
        origin: str = "user",
        origin_window: str | None = None,
        origin_window_label: str | None = None,
    ) -> Path:
        """Append a turn to the conversation's JSONL file.

        Per-conv threading.Lock prevents line interleaving when multiple
        background encode threads fire on the same conv (e.g. rapid turns).
        """
        conv_id = _normalize_session_id(session_id)
        conv_dir = self._conv_dir(conv_id)
        conv_dir.mkdir(parents=True, exist_ok=True)

        turns_file = conv_dir / "retriever_turns.jsonl"
        entry = {
            "turn": turn,
            "ts": datetime.now(UTC).isoformat(),
            "user": user_msg.strip(),
            "assistant": assistant_msg.strip(),
            "origin": _normalize_origin(origin),
            "origin_window": _normalize_origin_window(origin_window),
            "origin_window_label": _normalize_origin_window_label(origin_window_label),
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._save_lock(conv_id):
            file_lock = _require_cross_process_file_lock()
            with file_lock(turns_file):
                with turns_file.open("a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    try:
                        os.fsync(f.fileno())
                    except OSError:
                        pass

        # Invalidate BM25 corpus cache for this conversation.
        with self._bm25_cache_guard:
            self._bm25_corpus_cache.pop(conv_id, None)
        with self._literal_index_cache_guard:
            self._literal_index_cache.pop(conv_id, None)
        return turns_file

    def load_turns(self, session_id: str | None = None) -> list[dict[str, Any]]:
        """Load all turns for a conversation."""
        turns_file = self._conv_dir(_normalize_session_id(session_id)) / "retriever_turns.jsonl"
        if not turns_file.exists():
            return []
        turns = []
        for line in turns_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return turns

    def load_turns_by_number(self, turn_numbers: set[int], session_id: str | None = None) -> list[dict[str, Any]]:
        """Load specific turn ids from retriever_turns.jsonl."""
        if not turn_numbers:
            return []
        turns_file = self._conv_dir(_normalize_session_id(session_id)) / "retriever_turns.jsonl"
        if not turns_file.exists():
            return []
        remaining = set(turn_numbers)
        results: list[dict[str, Any]] = []
        for line in turns_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            turn = record.get("turn")
            if turn in remaining:
                results.append(record)
                remaining.discard(turn)
                if not remaining:
                    break
        return results

    def load_turns_by_local_dates(self, local_dates: list[str], session_id: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Load turns whose timestamp falls on one of the given KST dates."""
        if not local_dates or not callable(timestamp_local_date):
            return []
        wanted = set(local_dates)
        rank = {value: idx for idx, value in enumerate(local_dates)}
        turns_file = self._conv_dir(_normalize_session_id(session_id)) / "retriever_turns.jsonl"
        if not turns_file.exists():
            return []
        results: list[dict[str, Any]] = []
        for line in turns_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            local_date = timestamp_local_date(record.get("ts") or record.get("timestamp"))
            if local_date in wanted:
                record["local_date"] = local_date
                results.append(record)
        results.sort(key=lambda item: (rank.get(str(item.get("local_date")), 9999), str(item.get("ts") or ""), int(item.get("turn") or 0)))
        return results[:limit]

    # ── Embed ──

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from local fastembed model."""
        # Lazy-load embedder if not injected yet
        if self._embedder is None:
            from .embedder import LocalEmbedder

            self._embedder = LocalEmbedder()
        t0 = time.monotonic()
        try:
            embeddings = await asyncio.to_thread(self._embedder.embed, texts)
            if not embeddings:
                return []
            return embeddings
        except Exception as exc:
            print(f"[jlc:ret] embed failed: {exc}", file=sys.stderr)
            return []
        finally:
            elapsed_ms = int(round((time.monotonic() - t0) * 1000))
            self._embed_latency_ms_total += elapsed_ms
            if self._embed_latency_hook is not None:
                try:
                    self._embed_latency_hook(elapsed_ms)
                except Exception:
                    pass

    def pop_embed_latency_ms(self) -> int:
        val = self._embed_latency_ms_total
        self._embed_latency_ms_total = 0
        return val

    # ── Index ──

    def _index_jsonl_path(self, conv_id: str) -> Path:
        return self._conv_dir(conv_id) / "index.jsonl"

    def _literal_index_json_path(self, conv_id: str) -> Path:
        return self._conv_dir(conv_id) / "literal_index.v3.json"

    def _literal_index_signature(self, conv_id: str) -> list[dict[str, Any]]:
        conv_dir = self._conv_dir(conv_id)
        return [
            _literal_signature_for_path(conv_dir / "retriever_turns.jsonl"),
            _literal_signature_for_path(self._index_jsonl_path(conv_id)),
        ]

    @staticmethod
    def _build_literal_index_data(turns: list[dict[str, Any]]) -> dict[str, Any]:
        postings: dict[str, set[int]] = {}
        for turn_idx, turn in enumerate(turns):
            tokens = set(_tokenize_for_literals(JLCRetriever._turn_to_text(turn)))
            for token in tokens:
                postings.setdefault(token, set()).add(turn_idx)
        return {
            "version": _LITERAL_INDEX_VERSION,
            "turn_count": len(turns),
            "tokens": {
                token: sorted(indices)
                for token, indices in sorted(postings.items())
            },
        }

    def _load_literal_index_file(
        self,
        conv_id: str,
        *,
        signature: list[dict[str, Any]],
        turn_count: int,
    ) -> dict[str, Any] | None:
        path = self._literal_index_json_path(conv_id)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("version") != _LITERAL_INDEX_VERSION:
            return None
        if payload.get("sources") != signature or payload.get("turn_count") != turn_count:
            return None
        tokens = payload.get("tokens")
        if not isinstance(tokens, dict):
            return None
        return payload

    def _save_literal_index_file(
        self,
        conv_id: str,
        *,
        signature: list[dict[str, Any]],
        data: dict[str, Any],
    ) -> None:
        path = self._literal_index_json_path(conv_id)
        try:
            payload = dict(data)
            payload["sources"] = signature
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True))
        except Exception as exc:
            print(f"[jlc:ret] literal index cache write skipped for {conv_id}: {exc}", file=sys.stderr)

    def _literal_index_data(self, conv_id: str, turns: list[dict[str, Any]]) -> dict[str, Any]:
        signature = self._literal_index_signature(conv_id)
        with self._literal_index_cache_guard:
            cached = self._literal_index_cache.get(conv_id)
            if cached is not None and cached[0] == signature:
                self._literal_index_cache.move_to_end(conv_id)
                return cached[1]

        data = self._load_literal_index_file(conv_id, signature=signature, turn_count=len(turns))
        if data is None:
            data = self._build_literal_index_data(turns)
            self._save_literal_index_file(conv_id, signature=signature, data=data)

        with self._literal_index_cache_guard:
            if len(self._literal_index_cache) >= _MAX_LITERAL_INDEX_CACHE:
                self._literal_index_cache.popitem(last=False)
            self._literal_index_cache[conv_id] = (signature, data)
        return data

    def _legacy_index_json_path(self, conv_id: str) -> Path:
        return self._conv_dir(conv_id) / "index.json"

    @staticmethod
    def _index_jsonl_text(index_data: list[dict[str, Any]]) -> str:
        if not index_data:
            return ""
        return "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in index_data)

    @staticmethod
    def _next_migrated_index_path(legacy_path: Path) -> Path:
        preferred = legacy_path.with_name(f"{legacy_path.name}.migrated")
        if not preferred.exists():
            return preferred
        for suffix in range(1, 1000):
            candidate = legacy_path.with_name(f"{legacy_path.name}.migrated.{suffix}")
            if not candidate.exists():
                return candidate
        return legacy_path.with_name(f"{legacy_path.name}.migrated.{int(time.time())}")

    def _migrate_legacy_index(self, conv_id: str) -> None:
        legacy_path = self._legacy_index_json_path(conv_id)
        if not legacy_path.exists():
            return

        index_data: list[dict[str, Any]] = []
        try:
            raw_index = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(raw_index, list):
                index_data = [entry for entry in raw_index if isinstance(entry, dict)]
            else:
                print(f"[jlc:ret] legacy index.json for {conv_id} not a list, migrating as empty", file=sys.stderr)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[jlc:ret] legacy index.json migration failed for {conv_id}: {exc}", file=sys.stderr)

        index_path = self._index_jsonl_path(conv_id)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if not index_path.exists():
            _atomic_write_text(index_path, self._index_jsonl_text(index_data))

        legacy_path.rename(self._next_migrated_index_path(legacy_path))

    def _load_index_data(self, conv_id: str) -> list[dict[str, Any]] | None:
        self._migrate_legacy_index(conv_id)
        index_path = self._index_jsonl_path(conv_id)
        if not index_path.exists():
            return None

        index_data: list[dict[str, Any]] = []
        for line_number, line in enumerate(index_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(
                    f"[jlc:ret] index.jsonl parse failed for {conv_id} line {line_number}: {exc}",
                    file=sys.stderr,
                )
                continue
            if isinstance(entry, dict):
                index_data.append(entry)
        return index_data

    async def build_index(self, session_id: str | None = None) -> int:
        """Build/rebuild embedding index for a conversation. Returns indexed count."""
        conv_id = _normalize_session_id(session_id)
        turns = self.load_turns(conv_id)
        if not turns:
            return 0

        texts = [self._turn_to_text(t) for t in turns]
        embeddings = await self._embed(texts)
        if not embeddings or len(embeddings) != len(turns):
            return 0

        index_data = []
        for turn, emb in zip(turns, embeddings):
            index_data.append({"turn": turn.get("turn", 0), "embedding": emb})

        lock = self._index_lock(conv_id)
        await asyncio.to_thread(lock.acquire)
        try:
            self._migrate_legacy_index(conv_id)
            index_path = self._index_jsonl_path(conv_id)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(index_path, self._index_jsonl_text(index_data))
        finally:
            lock.release()
        return len(index_data)

    async def index_turn(self, turn_data: dict[str, Any], session_id: str | None = None) -> bool:
        """Incrementally index a single turn (append to existing index).

        Serialized per-conversation to prevent read-modify-write races that can
        silently drop entries when multiple turns are indexed concurrently.
        """
        conv_id = _normalize_session_id(session_id)
        text = self._turn_to_text(turn_data)
        embeddings = await self._embed([text])
        if not embeddings:
            print(
                f"[jlc:ret] index_turn {conv_id} turn={turn_data.get('turn')}: empty embedding, skipped",
                file=sys.stderr,
            )
            return False

        # threading.Lock acquired via to_thread so we don't block the event loop
        lock = self._index_lock(conv_id)
        await asyncio.to_thread(lock.acquire)
        try:
            self._migrate_legacy_index(conv_id)
            index_path = self._index_jsonl_path(conv_id)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"turn": turn_data.get("turn", 0), "embedding": embeddings[0]}, ensure_ascii=False) + "\n"
            file_lock = _require_cross_process_file_lock()
            with file_lock(index_path):
                with index_path.open("a", encoding="utf-8", newline="") as fh:
                    fh.write(line)
                    fh.flush()
                    try:
                        os.fsync(fh.fileno())
                    except OSError:
                        pass
        finally:
            lock.release()
        return True

    # ── Search ──

    @staticmethod
    def _confidence_from_top1(top1: float) -> Literal["HIGH", "MID", "LOW"]:
        if top1 >= 0.85:
            return "HIGH"
        if top1 >= 0.70:
            return "MID"
        return "LOW"

    @staticmethod
    def _query_literal_anchors(query: str, literal_index: dict[str, Any]) -> list[tuple[str, list[str]]]:
        token_index = literal_index.get("tokens") if isinstance(literal_index, dict) else {}
        if not isinstance(token_index, dict):
            token_index = {}
        turn_count = int(literal_index.get("turn_count") or 0) if isinstance(literal_index, dict) else 0
        rare_threshold = _literal_rare_threshold(turn_count)
        entity_tokens = _literal_entity_tokens(query)

        raw_anchors: list[str] = []
        try:
            raw_anchors.extend(str(anchor) for anchor in recall_snippets._query_anchors(query))
        except Exception:
            raw_anchors.extend(_tokenize_for_literals(query))
        raw_anchors.extend(_tokenize_for_literals(query))

        out: list[tuple[str, list[str]]] = []
        seen: set[tuple[str, ...]] = set()
        for anchor in raw_anchors:
            if _is_noise_literal_anchor(anchor):
                continue
            tokens = list(dict.fromkeys(_tokenize_for_literals(anchor)))
            if not tokens:
                continue
            if entity_tokens and not (set(tokens) & entity_tokens):
                continue
            postings = [token_index.get(token) for token in tokens]
            if not any(isinstance(items, list) and items for items in postings):
                continue
            dfs = [
                len(items)
                for items in postings
                if isinstance(items, list) and items
            ]
            rare = any(df <= rare_threshold for df in dfs)
            identifierish = any(
                len(token) >= 5
                and any(ch.isalpha() for ch in token)
                and len(token_index.get(token, [])) <= max(rare_threshold * 4, int(max(1, turn_count) * 0.25))
                for token in tokens
            )
            if not rare and not (len(tokens) >= 2 and identifierish):
                continue
            key = tuple(tokens)
            if key in seen:
                continue
            seen.add(key)
            out.append((anchor, tokens))
        out.sort(key=lambda item: (len(item[1]), len(item[0])), reverse=True)
        return out[:80]

    @staticmethod
    def _literal_candidate_score(tokens: list[str], literal_index: dict[str, Any]) -> float:
        token_index = literal_index.get("tokens") if isinstance(literal_index, dict) else {}
        if not isinstance(token_index, dict):
            return 0.0
        turn_count = int(literal_index.get("turn_count") or 0)
        rare_threshold = _literal_rare_threshold(turn_count)
        score = 0.0
        for token in dict.fromkeys(tokens):
            df = len(token_index.get(token, []))
            if df <= 0:
                continue
            idf = _literal_idf(df, turn_count)
            score += idf * (1.4 if df <= rare_threshold else 0.25)
        if len(tokens) > 1:
            score += min(0.6, 0.12 * len(tokens))
        return score

    def _literal_exact_candidates(
        self,
        *,
        query: str,
        turns: list[dict[str, Any]],
        conv_id: str,
    ) -> tuple[list[tuple[int, float, dict[str, Any]]], set[str], dict[str, Any]]:
        literal_index = self._literal_index_data(conv_id, turns)
        token_index = literal_index.get("tokens")
        if not isinstance(token_index, dict) or not token_index:
            return [], set(), literal_index

        anchors = self._query_literal_anchors(query, literal_index)
        if not anchors:
            return [], set(), literal_index

        max_turn = max((int(turn.get("turn") or 0) for turn in turns), default=1) or 1
        by_idx: dict[int, dict[str, Any]] = {}
        rare_query_tokens: set[str] = set()
        entity_tokens = _literal_entity_tokens(query)
        rare_threshold = _literal_rare_threshold(int(literal_index.get("turn_count") or 0))
        for _anchor, tokens in anchors:
            for token in tokens:
                df = len(token_index.get(token, []))
                if 0 < df <= rare_threshold:
                    rare_query_tokens.add(token)

        for anchor, tokens in anchors:
            postings: list[set[int]] = []
            for token in dict.fromkeys(tokens):
                raw = token_index.get(token)
                if not isinstance(raw, list) or not raw:
                    postings = []
                    break
                postings.append(set(int(item) for item in raw))
            if not postings:
                continue
            candidate_indices = set.intersection(*postings)
            if not candidate_indices:
                continue
            anchor_score = self._literal_candidate_score(tokens, literal_index)
            if anchor_score <= 0:
                continue
            for turn_idx in candidate_indices:
                if turn_idx < 0 or turn_idx >= len(turns):
                    continue
                turn = turns[turn_idx]
                if _is_denial_response(str(turn.get("assistant", ""))):
                    continue
                entry = by_idx.setdefault(
                    turn_idx,
                    {"score": 0.0, "anchors": [], "tokens": set()},
                )
                entry["score"] = float(entry["score"]) + anchor_score
                if len(entry["anchors"]) < 8:
                    entry["anchors"].append(anchor)
                entry["tokens"].update(tokens)

        raw_candidates: list[tuple[int, float, dict[str, Any]]] = []
        for turn_idx, meta in by_idx.items():
            turn_num = int(turns[turn_idx].get("turn") or 0)
            recency = (turn_num / max_turn) * 0.2
            score = 100.0 + min(50.0, float(meta["score"])) + recency
            tokens = sorted(str(token) for token in meta["tokens"])
            signature = sorted((set(tokens) & entity_tokens) or (set(tokens) & rare_query_tokens) or set(tokens))
            raw_candidates.append((
                turn_idx,
                score,
                {
                    "literal_score": round(score, 4),
                    "literal_anchors": list(meta["anchors"]),
                    "literal_tokens": tokens[:20],
                    "literal_signature": signature[:20],
                    "exact_literal": True,
                },
            ))
        best_by_signature: dict[tuple[str, ...], tuple[int, float, dict[str, Any]]] = {}
        for candidate in raw_candidates:
            turn_idx, score, meta = candidate
            signature = tuple(str(token) for token in meta.get("literal_signature") or [])
            if not signature:
                signature = (f"turn:{turn_idx}",)
            existing = best_by_signature.get(signature)
            if existing is None:
                best_by_signature[signature] = candidate
                continue
            existing_turn = int(turns[existing[0]].get("turn") or 0)
            current_turn = int(turns[turn_idx].get("turn") or 0)
            if (current_turn, score) > (existing_turn, existing[1]):
                best_by_signature[signature] = candidate
        candidates = list(best_by_signature.values())
        candidates.sort(
            key=lambda item: (
                item[1],
                int(turns[item[0]].get("turn") or 0),
            ),
            reverse=True,
        )
        return candidates, rare_query_tokens, literal_index

    @staticmethod
    def _apply_literal_candidates(
        scored: list[tuple[int, float]],
        literal_candidates: list[tuple[int, float, dict[str, Any]]],
        score_trace: dict[int, dict[str, Any]],
    ) -> list[tuple[int, float]]:
        if not literal_candidates:
            return scored
        merged: dict[int, float] = {idx: score for idx, score in scored}
        for idx, literal_score, meta in literal_candidates:
            merged[idx] = max(merged.get(idx, float("-inf")), literal_score)
            trace = score_trace.setdefault(idx, {})
            trace.update(meta)
        return sorted(merged.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _apply_rare_literal_boost(
        turns: list[dict[str, Any]],
        scored: list[tuple[int, float]],
        *,
        rare_query_tokens: set[str],
        literal_index: dict[str, Any],
        score_trace: dict[int, dict[str, Any]],
    ) -> list[tuple[int, float]]:
        if not scored or not rare_query_tokens:
            return scored
        token_index = literal_index.get("tokens") if isinstance(literal_index, dict) else {}
        if not isinstance(token_index, dict):
            return scored
        turn_count = int(literal_index.get("turn_count") or len(turns))
        out: list[tuple[int, float]] = []
        for idx, score in scored:
            doc_tokens = set(_tokenize_for_literals(JLCRetriever._turn_to_text(turns[idx])))
            matched = sorted(doc_tokens & rare_query_tokens)
            if not matched:
                out.append((idx, score))
                continue
            boost = min(
                2.0,
                sum(_literal_idf(len(token_index.get(token, [])), turn_count) for token in matched) * 0.08,
            )
            trace = score_trace.setdefault(idx, {})
            trace["literal_boost"] = round(float(boost), 4)
            trace["literal_boost_tokens"] = matched[:20]
            out.append((idx, score + boost))
        return sorted(out, key=lambda item: item[1], reverse=True)

    @staticmethod
    def _mmr_rerank(
        turns: list[dict[str, Any]],
        scored: list[tuple[int, float]],
        query_tokens: list[str],
        score_trace: dict[int, dict[str, Any]],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if len(scored) <= top_k or top_k <= 1:
            return scored
        q_set = set(query_tokens)
        selected: list[tuple[int, float]] = []
        deferred: list[tuple[int, float]] = []
        seen_signatures: set[frozenset[str]] = set()
        for idx, score in scored:
            doc_tokens = set(_tokenize_for_bm25(JLCRetriever._turn_to_text(turns[idx])))
            signature = frozenset(sorted(q_set & doc_tokens))
            exact_literal = bool(score_trace.get(idx, {}).get("exact_literal"))
            if signature and signature in seen_signatures and not exact_literal:
                trace = score_trace.setdefault(idx, {})
                trace["mmr_deferred"] = True
                deferred.append((idx, score))
                continue
            selected.append((idx, score))
            if signature:
                seen_signatures.add(signature)
            if len(selected) >= top_k:
                break
        selected_ids = {idx for idx, _score in selected}
        remainder = [item for item in scored if item[0] not in selected_ids]
        ranked = selected + [item for item in deferred if item[0] not in selected_ids]
        ranked_ids = {idx for idx, _score in ranked}
        ranked.extend(item for item in remainder if item[0] not in ranked_ids)
        return ranked

    async def search_within(
        self, query: str, scope: set[int], top_k: int = 5, session_id: str | None = None,
    ) -> RetrieverSearchResult:
        """Search ONLY within a set of turn numbers (JRE-narrowed scope)."""
        conv_id = _normalize_session_id(session_id)
        try:
            index_data = self._load_index_data(conv_id)
        except OSError as exc:
            print(f"[jlc:ret] search_within: index.jsonl load failed for {conv_id}: {exc}", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}
        if index_data is None:
            print(f"[jlc:ret] search_within: {conv_id} has no index.jsonl", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}

        if not index_data:
            print(f"[jlc:ret] search_within: {conv_id} index.jsonl is empty", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}

        query_emb = await self._embed([query])
        if not query_emb:
            print(f"[jlc:ret] search_within: query embedding failed for {conv_id}", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}

        q_vec = query_emb[0]

        # Only score turns within the JRE scope
        scored = []
        for item in index_data:
            emb = item.get("embedding")
            turn_num = item.get("turn")
            if emb is None or turn_num is None:
                continue
            if turn_num not in scope:
                continue
            sim = self._cosine_sim(q_vec, emb)
            scored.append((sim, turn_num))

        scored.sort(reverse=True)

        turns = self.load_turns(conv_id)
        turn_map = {t["turn"]: t for t in turns}
        available_turns = set(turn_map)

        selected = scored[:top_k]
        top1 = selected[0][0] if selected else 0.0
        confidence = self._confidence_from_top1(top1)
        score_by_turn = {turn_num: round(sim, 4) for sim, turn_num in scored}
        window_turns: set[int] = set()
        for _, center_turn in selected:
            for turn_num in range(center_turn - 2, center_turn + 3):
                if turn_num in available_turns:
                    window_turns.add(turn_num)

        fragments = []
        for turn_num in sorted(window_turns):
            if turn_num in turn_map:
                t = turn_map[turn_num]
                fragments.append(_fragment_for_turn(t, score=score_by_turn.get(turn_num, 0.0)))

        return {"confidence": confidence, "fragments": fragments}

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        bm25_pool: int = 50,
        recency_bonus: float = 0.05,
        correction_bonus: float = 0.10,
        session_id: str | None = None,
    ) -> RetrieverSearchResult:
        """JRE-style hybrid: BM25 narrow → bge-m3 cosine rerank → meta bonuses.

        Pipeline (matches `C:/codex/JRE.md` spec):
          1. BM25 over all stored turns, take top `bm25_pool` (default 50).
          2. Cosine-rerank that pool against bge-m3 query embedding.
          3. Apply recency bonus (later turn = small boost) + correction bonus
             (later turn touching the same identifier as an earlier one wins).
          4. Return original turn text — no LLM summarization.

        Falls back gracefully:
          - rank-bm25 missing → degrades to keyword scoring.
          - embedder degraded → returns BM25 top_k unchanged.
        """
        conv_id = _normalize_session_id(session_id)
        turns = self.load_turns(conv_id)
        if not turns:
            return {"confidence": "LOW", "fragments": []}

        query_tokens = _tokenize_for_bm25(query)
        if not query_tokens:
            return {"confidence": "LOW", "fragments": []}

        literal_candidates, rare_query_tokens, literal_index = self._literal_exact_candidates(
            query=query,
            turns=turns,
            conv_id=conv_id,
        )
        bm25_scored = self._bm25_score_with_cache(turns, query_tokens, top_n=bm25_pool, conv_id=conv_id)
        if not bm25_scored and not literal_candidates:
            recall_trace.emit(
                "recall_trace",
                surface="retriever.hybrid_search",
                conv_id=conv_id,
                top_k=top_k,
                bm25_pool=bm25_pool,
                **recall_trace.query_fields(query),
                candidates=[],
                served_turns=[],
                confidence="LOW",
            )
            return {"confidence": "LOW", "fragments": []}
        score_trace: dict[int, dict[str, Any]] = {
            idx: {"bm25": round(float(score), 4)}
            for idx, score in bm25_scored
        }

        # Embedding rerank — only if embedder is healthy
        if self._embedder is None:
            from .embedder import LocalEmbedder

            self._embedder = LocalEmbedder()
        if bm25_scored and not self._embedder.is_degraded:
            try:
                # Load cached embeddings from index.jsonl (built at index_turn time)
                index_lookup: dict[int, list[float]] = {}
                try:
                    raw_index = self._load_index_data(conv_id)
                    for entry in raw_index or []:
                        if "turn" in entry and "embedding" in entry:
                            index_lookup[entry["turn"]] = entry["embedding"]
                except OSError as exc:
                    print(f"[jlc:ret] index.jsonl load failed for {conv_id}: {exc}", file=sys.stderr)

                # Embed query (always 1 call)
                q_emb = await self._embed([query])
                if not q_emb:
                    raise RuntimeError("query embedding returned empty")
                q_vec = q_emb[0]

                # Resolve pool embeddings: use cached where available, collect misses
                pool_embeddings: list[list[float] | None] = [None] * len(bm25_scored)
                miss_indices: list[int] = []  # indices into bm25_scored
                miss_texts: list[str] = []

                for k, (turn_idx, _bm25_score) in enumerate(bm25_scored):
                    turn_number = turns[turn_idx].get("turn", -1)
                    cached = index_lookup.get(turn_number)
                    if cached and len(cached) == len(q_vec):
                        pool_embeddings[k] = cached
                    else:
                        miss_indices.append(k)
                        miss_texts.append(self._turn_to_text(turns[turn_idx]))

                # Batch-embed cache misses (0 calls when fully cached)
                if miss_texts:
                    miss_embs = await self._embed(miss_texts)
                    if miss_embs and len(miss_embs) == len(miss_texts):
                        for j, mi in enumerate(miss_indices):
                            pool_embeddings[mi] = miss_embs[j]

                # Cosine rerank with available embeddings
                rerank: list[tuple[float, int, float]] = []
                for k, (turn_idx, bm25_score) in enumerate(bm25_scored):
                    emb = pool_embeddings[k]
                    if emb is not None and len(emb) == len(q_vec):
                        sim = self._cosine_sim(q_vec, emb)
                    else:
                        # Fallback: use BM25 score normalized as sim proxy
                        sim = bm25_score * 0.5
                    score_trace.setdefault(turn_idx, {})["cosine"] = round(float(sim), 4)
                    rerank.append((sim, turn_idx, bm25_score))
                rerank.sort(key=lambda x: x[0], reverse=True)
                bm25_scored = [(idx, sim) for sim, idx, _ in rerank]
            except Exception as exc:
                print(f"[jlc:ret] hybrid rerank failed: {exc}", file=sys.stderr)

        bm25_scored = self._apply_literal_candidates(bm25_scored, literal_candidates, score_trace)
        bm25_scored = self._apply_rare_literal_boost(
            turns,
            bm25_scored,
            rare_query_tokens=rare_query_tokens,
            literal_index=literal_index,
            score_trace=score_trace,
        )

        # Meta bonuses — recency + correction
        bonus_adjusted = self._apply_meta_bonuses(
            turns, bm25_scored, query_tokens,
            recency_bonus=recency_bonus,
            correction_bonus=correction_bonus,
        )
        for idx, score in bonus_adjusted:
            score_trace.setdefault(idx, {})["after_meta"] = round(float(score), 4)

        # Denial penalty — demote turns whose assistant response is a "no
        # record found" denial. Without this, self-echo loops form: chat
        # denies at turn N, user re-asks same anchor later, retriever
        # returns turn N's denial as top-1 (high cosine on razor tokens),
        # chat sees its own denial and reinforces it.
        before_denial = {idx: score for idx, score in bonus_adjusted}
        bonus_adjusted = self._apply_denial_penalty(turns, bonus_adjusted)
        for idx, score in bonus_adjusted:
            trace = score_trace.setdefault(idx, {})
            trace["final"] = round(float(score), 4)
            trace["denial_penalty"] = bool(score < before_denial.get(idx, score))

        ranked = self._mmr_rerank(turns, bonus_adjusted, query_tokens, score_trace, top_k=top_k)

        # Top-k after all scoring
        top = ranked[:top_k]
        if not top:
            recall_trace.emit(
                "recall_trace",
                surface="retriever.hybrid_search",
                conv_id=conv_id,
                top_k=top_k,
                bm25_pool=bm25_pool,
                **recall_trace.query_fields(query),
                candidates=[],
                served_turns=[],
                confidence="LOW",
            )
            return {"confidence": "LOW", "fragments": []}

        top_score = top[0][1]
        if any(bool(score_trace.get(idx, {}).get("exact_literal")) for idx, _score in top):
            confidence = "HIGH"
        else:
            confidence = self._confidence_from_top1(top_score)
        fragments = []
        for turn_idx, score in top:
            t = turns[turn_idx]
            exact_literal = bool(score_trace.get(turn_idx, {}).get("exact_literal"))
            fragment = _fragment_for_turn(t, score=1.0 if exact_literal else score)
            if exact_literal:
                fragment["exact_literal"] = True
                fragment["rank_score"] = round(float(score), 4)
            fragments.append(fragment)
        q_set = set(query_tokens)
        candidates: list[dict[str, Any]] = []
        for rank, (turn_idx, score) in enumerate(ranked[: max(top_k, 10)], start=1):
            turn = turns[turn_idx]
            doc_tokens = set(_tokenize_for_bm25(self._turn_to_text(turn)))
            trace = score_trace.get(turn_idx, {})
            candidates.append({
                "rank": rank,
                "turn": turn.get("turn", 0),
                "score": round(float(score), 4),
                "bm25": trace.get("bm25"),
                "cosine": trace.get("cosine"),
                "after_meta": trace.get("after_meta"),
                "denial_penalty": bool(trace.get("denial_penalty", False)),
                "exact_literal": bool(trace.get("exact_literal", False)),
                "literal_score": trace.get("literal_score"),
                "literal_anchors": trace.get("literal_anchors", []),
                "literal_tokens": trace.get("literal_tokens", []),
                "literal_signature": trace.get("literal_signature", []),
                "literal_boost": trace.get("literal_boost"),
                "literal_boost_tokens": trace.get("literal_boost_tokens", []),
                "mmr_deferred": bool(trace.get("mmr_deferred", False)),
                "final_score": trace.get("final", round(float(score), 4)),
                "user_chars": len(str(turn.get("user", ""))),
                "assistant_chars": len(str(turn.get("assistant", ""))),
                "matched_terms": sorted(q_set & doc_tokens)[:20],
            })
        recall_trace.emit(
            "recall_trace",
            surface="retriever.hybrid_search",
            conv_id=conv_id,
            top_k=top_k,
            bm25_pool=bm25_pool,
            **recall_trace.query_fields(query),
            candidates=candidates,
            served_turns=[fragment.get("turn") for fragment in fragments],
            served_chars=sum(
                len(str(fragment.get("user", ""))) + len(str(fragment.get("assistant", "")))
                for fragment in fragments
            ),
            confidence=confidence,
        )
        return {"confidence": confidence, "fragments": fragments}

    @staticmethod
    def _bm25_score(
        turns: list[dict[str, Any]],
        query_tokens: list[str],
        top_n: int = 50,
    ) -> list[tuple[int, float]]:
        """Return [(turn_index, bm25_score), ...] sorted descending.

        Falls back to a simple token-overlap score if rank_bm25 isn't available
        — keeps the pipeline functional in degraded environments.
        """
        return JLCRetriever._bm25_score_with_cache(turns, query_tokens, top_n, None)

    def _bm25_score_with_cache(
        self,
        turns: list[dict[str, Any]],
        query_tokens: list[str],
        top_n: int = 50,
        conv_id: str | None = None,
    ) -> list[tuple[int, float]]:
        """BM25 scoring with optional corpus cache for the given conv_id."""
        # Try to use cached corpus if conv_id is provided and cache is valid.
        cache_key = conv_id or ""
        with self._bm25_cache_guard:
            cached = self._bm25_corpus_cache.get(cache_key)
            if cached is not None and len(cached[1]) == len(turns):
                corpus = cached[1]
                self._bm25_corpus_cache.move_to_end(cache_key)
            else:
                corpus = [_tokenize_for_bm25(JLCRetriever._turn_to_text(t)) for t in turns]
                if conv_id:
                    if len(self._bm25_corpus_cache) >= _MAX_BM25_CACHE:
                        self._bm25_corpus_cache.popitem(last=False)
                    self._bm25_corpus_cache[conv_id] = (len(turns), corpus)

        if not corpus:
            return []
        try:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi(corpus)
            raw = bm25.get_scores(query_tokens)
            # Keep all docs that contain at least one query token (overlap > 0).
            # Don't filter on `bm25 > 0` — when a query term has IDF≈0 (e.g.
            # appears in every doc) BM25 returns 0 even though the doc still
            # matches. Use overlap as the gate, BM25 score for ranking.
            q_set = set(query_tokens)
            scored = []
            for i, s in enumerate(raw):
                if q_set & set(corpus[i]):
                    scored.append((i, float(s)))
        except Exception as exc:
            # Fallback: token-overlap ratio (no IDF). Keeps recall positive.
            print(f"[jlc:ret] rank_bm25 unavailable ({exc}); using token overlap", file=sys.stderr)
            q_set = set(query_tokens)
            scored = []
            for i, doc in enumerate(corpus):
                if not doc:
                    continue
                overlap = len(q_set.intersection(doc))
                if overlap:
                    scored.append((i, overlap / max(len(q_set), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def _apply_meta_bonuses(
        turns: list[dict[str, Any]],
        scored: list[tuple[int, float]],
        query_tokens: list[str],
        recency_bonus: float,
        correction_bonus: float,
    ) -> list[tuple[int, float]]:
        """Boost later turns slightly + boost a later turn that re-touches the
        same identifier as an earlier turn (heuristic for "this fixes the
        previous answer" patterns).
        """
        if not scored:
            return []
        max_turn = max(turns[i].get("turn", 0) for i, _ in scored) or 1
        # Pre-compute identifier overlap per turn vs query
        q_set = set(query_tokens)
        token_overlap_by_idx: dict[int, set[str]] = {}
        for i, _ in scored:
            doc_tokens = set(_tokenize_for_bm25(JLCRetriever._turn_to_text(turns[i])))
            token_overlap_by_idx[i] = doc_tokens & q_set

        # Identifiers seen earlier (lower turn) — later turn touching the same
        # identifier earns the correction bonus.
        scored_by_turn = sorted(scored, key=lambda x: turns[x[0]].get("turn", 0))
        seen_overlaps: set[str] = set()
        bonused: dict[int, float] = {}
        for idx, base in scored_by_turn:
            turn_num = turns[idx].get("turn", 0)
            current = token_overlap_by_idx[idx]
            recency = (turn_num / max_turn) * recency_bonus
            correction = correction_bonus if (current & seen_overlaps) else 0.0
            bonused[idx] = base + recency + correction
            seen_overlaps |= current

        # Re-sort with bonuses applied
        return sorted(((idx, bonused[idx]) for idx, _ in scored), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _apply_denial_penalty(
        turns: list[dict[str, Any]],
        scored: list[tuple[int, float]],
        penalty: float = 0.01,
    ) -> list[tuple[int, float]]:
        """Strongly demote turns whose assistant text is a denial.

        A denial turn ("I don't have any record of X") carries no evidence,
        only a refusal — keeping it in the top-k starves the slot from a
        planting turn ("Parking note: X equals Y"). Penalty (not removal)
        so a denial can still surface when nothing else matches.
        """
        if not scored:
            return scored
        out: list[tuple[int, float]] = []
        for idx, score in scored:
            if _is_denial_response(turns[idx].get("assistant", "")):
                out.append((idx, score * penalty))
            else:
                out.append((idx, score))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    async def search(self, query: str, top_k: int = 5, session_id: str | None = None) -> RetrieverSearchResult:
        """Search for relevant turns by semantic similarity."""
        conv_id = _normalize_session_id(session_id)
        try:
            index_data = self._load_index_data(conv_id)
        except OSError as exc:
            print(f"[jlc:ret] search: index.jsonl load failed for {conv_id}: {exc}", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}
        if index_data is None:
            print(f"[jlc:ret] search: {conv_id} has no index.jsonl", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}

        if not index_data:
            print(f"[jlc:ret] search: {conv_id} index.jsonl is empty", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}

        query_emb = await self._embed([query])
        if not query_emb:
            print(f"[jlc:ret] search: query embedding failed for {conv_id}", file=sys.stderr)
            return {"confidence": "LOW", "fragments": []}

        q_vec = query_emb[0]

        scored = []
        for item in index_data:
            emb = item.get("embedding")
            turn_num = item.get("turn")
            if emb is None or turn_num is None:
                continue
            sim = self._cosine_sim(q_vec, emb)
            scored.append((sim, turn_num))

        scored.sort(reverse=True)

        turns = self.load_turns(conv_id)
        turn_map = {t["turn"]: t for t in turns}
        available_turns = set(turn_map)

        selected = scored[:top_k]
        top1 = selected[0][0] if selected else 0.0
        confidence = self._confidence_from_top1(top1)
        score_by_turn = {turn_num: round(sim, 4) for sim, turn_num in scored}
        window_turns: set[int] = set()
        for _, center_turn in selected:
            for turn_num in range(center_turn - 2, center_turn + 3):
                if turn_num in available_turns:
                    window_turns.add(turn_num)

        fragments = []
        for turn_num in sorted(window_turns):
            if turn_num in turn_map:
                t = turn_map[turn_num]
                fragments.append(_fragment_for_turn(t, score=score_by_turn.get(turn_num, 0.0)))

        return {"confidence": confidence, "fragments": fragments}

    async def recall(
        self, query: str, llm: Any, top_k: int = 3, max_tokens: int = 150,
    ) -> dict[str, Any]:
        """Search + LLM summarize. Returns {"ok": bool, "text": str, "warning": str|None}.

        Quality-first: if LLM fails, returns warning instead of degraded fallback.
        """
        search_result = await self.search(query, top_k=top_k)
        results = search_result.get("fragments", [])
        if not results:
            results = self.search_keyword(query, max_results=top_k)
        if not results:
            return {"ok": True, "text": "", "warning": None}

        raw_parts = []
        for r in results:
            raw_parts.append(f"[Turn {r['turn']}] User: {r['user']}\nAssistant: {r['assistant']}")
        raw_context = "\n---\n".join(raw_parts)

        system = (
            "Summarize into 2-3 key facts. English only. Max 150 tokens. Plain text."
        )
        try:
            summary = await llm.chat(system=system, user=f"Query: {query}\n\n{raw_context}", max_tokens=max_tokens)
            return {"ok": True, "text": summary.strip(), "warning": None}
        except Exception as exc:
            print(f"[jlc:ret] recall failed — encoder LLM down: {exc}", file=sys.stderr)
            return {
                "ok": False,
                "text": "",
                "warning": "Recall unavailable: encoder LLM is down. Use Claude directly for this query.",
            }

    def search_keyword(self, keyword: str, max_results: int = 5, session_id: str | None = None) -> list[dict[str, Any]]:
        """Fallback: keyword search when embedding server is down.

        Two-stage match so natural-language probes ("what did we discuss about X?")
        still hit turns containing "X":
          1. Whole-string substring — score 1.0 (exact intent).
          2. Token OR match — score = matched_terms / total_terms, ranked.
        """
        turns = self.load_turns(_normalize_session_id(session_id))
        kw_lower = keyword.lower()
        results: list[dict[str, Any]] = []

        for t in turns:
            text = (t.get("user", "") + " " + t.get("assistant", "")).lower()
            if kw_lower in text:
                results.append(_fragment_for_turn(t, score=1.0))
                if len(results) >= max_results:
                    return results

        if results:
            return results

        stopwords = {
            "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "of", "in", "on", "at", "to", "for", "with", "about", "as", "by",
            "that", "this", "these", "those", "it", "its", "we", "you", "i",
            "what", "when", "where", "who", "how", "why", "which",
            "last", "week", "day", "did", "can", "could", "should", "would",
        }
        tokens = [w for w in _tokenize_re.findall(kw_lower) if w not in stopwords and len(w) > 1]
        if not tokens:
            return []

        scored: list[tuple[float, dict[str, Any]]] = []
        for t in turns:
            text = (t.get("user", "") + " " + t.get("assistant", "")).lower()
            hits = sum(1 for tok in tokens if tok in text)
            if hits == 0:
                continue
            score = hits / len(tokens)
            scored.append((score, _fragment_for_turn(t, score=score)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    # ── Tag Search ──

    def search_by_tag(self, tag: str) -> list[dict[str, Any]]:
        """Find turns containing a specific tag (reads from turns.jsonl)."""
        turns = self.load_turns()
        tag_lower = tag.lower()
        results = []
        for t in turns:
            turn_tags = [tg.lower() for tg in t.get("tags", [])]
            if tag_lower in turn_tags:
                results.append({
                    "turn": t.get("turn", 0),
                    "ts": t.get("ts", ""),
                    "tags": t.get("tags", []),
                    "user": t.get("user", ""),
                    "assistant": t.get("assistant", ""),
                    **_origin_fields(t),
                })
        return results

    # ── Helpers ──

    def _conv_dir(self, session_id: str | None = None) -> Path:
        safe = _normalize_session_id(session_id).strip()
        if not safe:
            safe = "default"
        invalid = '<>:"/\\|?*'
        table = str.maketrans({ch: "_" for ch in invalid})
        safe = safe.translate(table).replace("..", "_")
        return self._root / safe

    @staticmethod
    def _turn_to_text(turn: dict[str, Any]) -> str:
        """Combine user + assistant into a single searchable string."""
        u = turn.get("user", "")
        a = turn.get("assistant", "")
        origin = _normalize_origin(turn.get("origin"))
        origin_window = _normalize_origin_window(turn.get("origin_window")) or ""
        origin_window_label = _normalize_origin_window_label(turn.get("origin_window_label")) or ""
        origin_line = ""
        if origin != "user" or origin_window or origin_window_label:
            origin_line = f"Origin: {origin}"
            if origin_window:
                origin_line += f" window={origin_window}"
            if origin_window_label:
                origin_line += f" label={origin_window_label}"
            origin_line += "\n"
        return f"{origin_line}User: {u}\nAssistant: {a}"

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        # Guard against NaN/Inf from a degraded embedder (e.g. OOM-poisoned
        # output). NaN propagates through dot/norm and would silently corrupt
        # rerank ordering.
        if norm_a != norm_a or norm_b != norm_b:
            return 0.0
        result = dot / (norm_a * norm_b)
        if result != result:
            return 0.0
        return result

