"""JLC Retriever — raw turn storage + semantic search for context recall."""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, TypedDict

from .embedder import LocalEmbedder

try:
    from jarvis_sidecar.raw_store import _timestamp_local_date as timestamp_local_date
except Exception:  # pragma: no cover - sidecar package unavailable in standalone mode
    timestamp_local_date = None

_MAX_CACHED_LOCKS = 1000
_MAX_BM25_CACHE = 500
_SESSION_ID = "jarvis_session"


def _normalize_session_id(session_id: str | None) -> str:
    raw = str(session_id or "").strip()
    return raw or _SESSION_ID


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    # Owner-only (0o600) on POSIX. Windows lacks os.fchmod; ACLs handle this.
    try:
        os.fchmod(fd, 0o600)
    except (AttributeError, OSError):
        pass
    try:
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace", newline="") as fh:
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

_tokenize_re = re.compile(r"[\w]+", re.UNICODE)

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

    def save_turn(self, turn: int, user_msg: str, assistant_msg: str, session_id: str | None = None) -> Path:
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
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._save_lock(conv_id):
            with turns_file.open("a", encoding="utf-8") as f:
                f.write(line)

        # Invalidate BM25 corpus cache for this conversation.
        with self._bm25_cache_guard:
            self._bm25_corpus_cache.pop(conv_id, None)
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
                fragments.append({
                    "turn": turn_num,
                    "score": score_by_turn.get(turn_num, 0.0),
                    "user": t.get("user", ""),
                    "assistant": t.get("assistant", ""),
                    "ts": t.get("ts", ""),
                })

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

        bm25_scored = self._bm25_score_with_cache(turns, query_tokens, top_n=bm25_pool, conv_id=conv_id)
        if not bm25_scored:
            return {"confidence": "LOW", "fragments": []}

        # Embedding rerank — only if embedder is healthy
        if self._embedder is None:
            from .embedder import LocalEmbedder

            self._embedder = LocalEmbedder()
        if not self._embedder.is_degraded:
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
                    rerank.append((sim, turn_idx, bm25_score))
                rerank.sort(key=lambda x: x[0], reverse=True)
                bm25_scored = [(idx, sim) for sim, idx, _ in rerank]
            except Exception as exc:
                print(f"[jlc:ret] hybrid rerank failed: {exc}", file=sys.stderr)

        # Meta bonuses — recency + correction
        bonus_adjusted = self._apply_meta_bonuses(
            turns, bm25_scored, query_tokens,
            recency_bonus=recency_bonus,
            correction_bonus=correction_bonus,
        )

        # Denial penalty — demote turns whose assistant response is a "no
        # record found" denial. Without this, self-echo loops form: chat
        # denies at turn N, user re-asks same anchor later, retriever
        # returns turn N's denial as top-1 (high cosine on razor tokens),
        # chat sees its own denial and reinforces it.
        bonus_adjusted = self._apply_denial_penalty(turns, bonus_adjusted)

        # Top-k after all scoring
        top = bonus_adjusted[:top_k]
        if not top:
            return {"confidence": "LOW", "fragments": []}

        top_score = top[0][1]
        confidence = self._confidence_from_top1(top_score)
        fragments = []
        for turn_idx, score in top:
            t = turns[turn_idx]
            fragments.append({
                "turn": t.get("turn", 0),
                "score": round(score, 4),
                "user": t.get("user", ""),
                "assistant": t.get("assistant", ""),
                "ts": t.get("ts", ""),
            })
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
        penalty: float = 0.5,
    ) -> list[tuple[int, float]]:
        """Halve the score of turns whose assistant text is a denial.

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
                fragments.append({
                    "turn": turn_num,
                    "score": score_by_turn.get(turn_num, 0.0),
                    "user": t.get("user", ""),
                    "assistant": t.get("assistant", ""),
                    "ts": t.get("ts", ""),
                })

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
                results.append({
                    "turn": t["turn"],
                    "score": 1.0,
                    "user": t.get("user", ""),
                    "assistant": t.get("assistant", ""),
                    "ts": t.get("ts", ""),
                })
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
            scored.append((hits / len(tokens), {
                "turn": t["turn"],
                "score": round(hits / len(tokens), 4),
                "user": t.get("user", ""),
                "assistant": t.get("assistant", ""),
                "ts": t.get("ts", ""),
            }))

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
        return f"User: {u}\nAssistant: {a}"

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

