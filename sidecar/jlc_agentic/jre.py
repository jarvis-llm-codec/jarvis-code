"""JRE — JARVIS Recall Engine.

JHB-routed context retrieval: records section→turn mappings during encoding,
retrieves specific turns via jhb routing instead of brute-force scanning.
"""
from __future__ import annotations

import json
import re
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncio

from .embedder import LocalEmbedder

_SESSION_ID = "jarvis_session"

# ── Stop words for keyword matching ──
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it",
    "its", "my", "your", "our", "their", "how", "when", "where", "why",
    "not", "no", "nor", "and", "or", "but", "if", "then", "so",
    "i", "me", "we", "you", "he", "she", "they", "them",
})

_SECTION_RE = re.compile(r"^## (.+?)\s*\[(P[0-3])\]\s*$", re.MULTILINE)


class JREEngine:
    """Records section→turn mappings during encoding; retrieves turns via jhb routing."""

    def __init__(self, storage_root: Path, embedder: LocalEmbedder | None = None) -> None:
        self._root = storage_root
        # Allow None for lazy injection from JarvisAgentic._get_embedder()
        self._embedder: LocalEmbedder | None = embedder
        self._map_cache: dict[str, dict[str, list[int]]] = {}  # conv_id → {section: [turns]}
        self._map_cache_mtime: dict[str, float] = {}  # conv_id → context_map.jsonl mtime at cache time
        # Per-conv file lock — prevents JSONL line interleaving when two
        # background tasks call record_changes for the same conv concurrently.
        self._ctxmap_locks: dict[str, threading.Lock] = {}
        self._ctxmap_locks_guard = threading.Lock()

    def _ensure_embedder(self) -> LocalEmbedder:
        """Lazy-load embedder if not injected yet."""
        if self._embedder is None:
            self._embedder = LocalEmbedder()
        return self._embedder

    async def close(self) -> None:
        return None

    # ══════════════════════════════════════════════
    # Recording (called from _encode_and_save)
    # ══════════════════════════════════════════════

    def record_changes(
        self, turn: int, prev_jhb: str, new_jhb: str, session_id: str | None = None,
    ) -> list[str]:
        """Diff prev/new jhb, append changed sections to context_map.jsonl.

        Returns list of changed section names (for logging).
        """
        conv_id = self._sanitize_session_id(session_id)
        prev_sections = self.parse_jhb_sections(prev_jhb)
        new_sections = self.parse_jhb_sections(new_jhb)

        changed: list[str] = []
        ts = datetime.now(UTC).isoformat()
        entries: list[dict[str, Any]] = []

        for name, (priority, body) in new_sections.items():
            prev = prev_sections.get(name)
            if prev is None or prev[1] != body:
                changed.append(name)
                entries.append({
                    "section": name,
                    "priority": priority,
                    "turn": turn,
                    "ts": ts,
                })

        if entries:
            self._append_context_map(entries, session_id=conv_id)
            # Update in-memory cache incrementally (no file re-read)
            cache = self._map_cache.get(conv_id)
            if cache is not None:
                for e in entries:
                    cache.setdefault(e["section"], []).append(e["turn"])
            # Update cached mtime so same-process re-reads are not forced to reload
            try:
                self._map_cache_mtime[conv_id] = self._context_map_path(conv_id).stat().st_mtime
            except OSError:
                self._map_cache_mtime.pop(conv_id, None)

        return changed

    # ══════════════════════════════════════════════
    # Querying
    # ══════════════════════════════════════════════

    async def narrow(self, query: str, jhb: str, session_id: str | None = None) -> set[int]:
        """Phase 1: JHB-routed scope narrowing.

        Returns set of candidate turn numbers (JRE's job ends here).
        The retriever then searches WITHIN this set for precision.
        """
        conv_id = self._sanitize_session_id(session_id)
        sections = self.parse_jhb_sections(jhb)
        if not sections:
            return set()

        # Tier 1: keyword matching (<1ms)
        matched = self._match_sections_keyword(query, sections)

        # Tier 2: embedding fallback if keyword match found nothing
        if not matched:
            try:
                matched = await self._match_sections_embedding(query, sections)
            except Exception as exc:
                print(f"[jlc:jre] embedding match failed ({type(exc).__name__}): {exc}", file=sys.stderr)
                matched = []

        if not matched:
            return set()

        # Look up context_map for matched sections
        context_map = self._load_context_map(conv_id)
        turn_numbers: set[int] = set()
        for section_name, _score in matched:
            turns = context_map.get(section_name, [])
            turn_numbers.update(turns)

        if turn_numbers:
            sec_names = [s for s, _ in matched]
            print(
                f"[jlc:jre] narrow: {len(turn_numbers)} candidates from sections {sec_names}",
                file=sys.stderr,
            )

        return turn_numbers

    async def recall(
        self,
        query: str,
        jhb: str,
        top_k: int = 5,
        max_turns: int = 8,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full recall (narrow + load). Used when retriever is unavailable.

        Returns list of turn dicts: [{"turn": int, "user": str, "assistant": str}, ...]
        """
        conv_id = self._sanitize_session_id(session_id)
        turn_numbers = await self.narrow(query, jhb, session_id=conv_id)
        if not turn_numbers:
            return []

        # Cap total turns — evenly sample across time range (first, last, spread)
        if len(turn_numbers) > max_turns:
            sorted_turns = sorted(turn_numbers)
            if max_turns >= 3:
                step = max(1, (len(sorted_turns) - 1) / (max_turns - 1))
                sampled = {sorted_turns[round(i * step)] for i in range(max_turns)}
                sampled.add(sorted_turns[0])
                sampled.add(sorted_turns[-1])
                turn_numbers = sampled
            else:
                turn_numbers = set(sorted_turns[:max_turns])

        results = self._load_turns_by_number(conv_id, turn_numbers)
        results.sort(key=lambda r: r.get("turn", 0))

        if results:
            print(f"[jlc:jre] recall: {len(results)} turns loaded", file=sys.stderr)

        return results

    def is_ready(self, min_entries: int = 10, session_id: str | None = None) -> bool:
        """Return True if context_map has enough data (default: >= 10 entries)."""
        path = self._context_map_path(session_id)
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as f:
                count = sum(1 for _ in f)
            return count >= min_entries
        except Exception:
            return False

    # ══════════════════════════════════════════════
    # JHB parsing
    # ══════════════════════════════════════════════

    @staticmethod
    def parse_jhb_sections(jhb: str) -> dict[str, tuple[str, str]]:
        """Parse '## Name [Px]' sections into {name: (priority, body)}.

        Example:
            {"Current Task": ("P1", "- Refactoring auth module...")}
        """
        if not jhb:
            return {}

        sections: dict[str, tuple[str, str]] = {}
        matches = list(_SECTION_RE.finditer(jhb))
        for i, m in enumerate(matches):
            name = m.group(1).strip()
            priority = m.group(2)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(jhb)
            body = jhb[start:end].strip()
            sections[name] = (priority, body)
        return sections

    # ══════════════════════════════════════════════
    # Section matching — Tier 1: keyword
    # ══════════════════════════════════════════════

    def _match_sections_keyword(
        self,
        query: str,
        sections: dict[str, tuple[str, str]],
        min_overlap: int = 1,
    ) -> list[tuple[str, float]]:
        """Fast keyword matching against section headers + body preview.

        Returns [(section_name, score), ...] sorted descending.
        """
        query_tokens = {
            w for w in re.findall(r"\w+", query.lower())
            if w not in _STOP_WORDS and len(w) > 1
        }
        if not query_tokens:
            return []

        scored: list[tuple[str, float]] = []
        for name, (priority, body) in sections.items():
            section_text = f"{name} {body[:200]}"
            section_tokens = {
                w for w in re.findall(r"\w+", section_text.lower())
                if w not in _STOP_WORDS and len(w) > 1
            }
            overlap = len(query_tokens & section_tokens)
            if overlap >= min_overlap:
                boost = 1.2 if priority == "P0" else 1.0
                scored.append((name, overlap * boost))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:5]

    # ══════════════════════════════════════════════
    # Section matching — Tier 2: embedding
    # ══════════════════════════════════════════════

    async def _match_sections_embedding(
        self,
        query: str,
        sections: dict[str, tuple[str, str]],
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Embed query, cosine sim against section headers + body preview.

        Returns [(name, score), ...] for sections above threshold.
        """
        section_names = list(sections.keys())
        if not section_names:
            return []

        # Build texts: header + first 200 chars of body
        section_texts = [
            f"{name}: {sections[name][1][:200]}" for name in section_names
        ]

        # Embed query + all sections in one call
        all_texts = [query] + section_texts
        embeddings = await self._embed(all_texts)
        if not embeddings or len(embeddings) != len(all_texts):
            return []

        q_vec = embeddings[0]
        scored: list[tuple[str, float]] = []
        for i, name in enumerate(section_names):
            sim = self._cosine_sim(q_vec, embeddings[i + 1])
            if sim >= threshold:
                scored.append((name, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:5]

    # ══════════════════════════════════════════════
    # context_map I/O
    # ══════════════════════════════════════════════

    def _load_context_map(self, session_id: str | None = None) -> dict[str, list[int]]:
        """Read context_map.jsonl, aggregate into {section_name: [turn_numbers]}.

        Uses in-memory cache with mtime invalidation — disk is re-read when
        context_map.jsonl has changed since last cache write. Protects the MCP
        process from serving stale narrowing results when the middleware
        process appends new entries.
        """
        conv_id = self._sanitize_session_id(session_id)
        path = self._context_map_path(conv_id)
        current_mtime: float | None
        try:
            current_mtime = path.stat().st_mtime
        except OSError:
            current_mtime = None

        if conv_id in self._map_cache and current_mtime is not None:
            cached_mtime = self._map_cache_mtime.get(conv_id)
            if cached_mtime is not None and cached_mtime >= current_mtime:
                return self._map_cache[conv_id]

        if not path.exists():
            self._map_cache[conv_id] = {}
            self._map_cache_mtime.pop(conv_id, None)
            return {}

        mapping: dict[str, list[int]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    section = entry.get("section", "")
                    turn = entry.get("turn")
                    if section and turn is not None:
                        mapping.setdefault(section, []).append(turn)
                except json.JSONDecodeError:
                    continue

        self._map_cache[conv_id] = mapping
        if current_mtime is not None:
            self._map_cache_mtime[conv_id] = current_mtime
        return mapping

    def _append_context_map(self, entries: list[dict[str, Any]], session_id: str | None = None) -> None:
        """Append entries to context_map.jsonl (thread-safe)."""
        conv_id = self._sanitize_session_id(session_id)
        with self._ctxmap_locks_guard:
            lock = self._ctxmap_locks.get(conv_id)
            if lock is None:
                lock = threading.Lock()
                self._ctxmap_locks[conv_id] = lock
        path = self._context_map_path(conv_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            with path.open("a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ══════════════════════════════════════════════
    # Turn loading — targeted read
    # ══════════════════════════════════════════════

    def _load_turns_by_number(self, turn_numbers: set[int], session_id: str | None = None) -> list[dict[str, Any]]:
        """Load specific turns from retriever_turns.jsonl by turn number.

        Reads file line-by-line and only parses matching turns.
        """
        turns_file = self._conv_dir(session_id) / "retriever_turns.jsonl"
        if not turns_file.exists():
            return []

        results: list[dict[str, Any]] = []
        remaining = set(turn_numbers)
        with turns_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("turn") in remaining:
                        results.append(data)
                        remaining.discard(data["turn"])
                        if not remaining:
                            break
                except json.JSONDecodeError:
                    continue
        return results

    # ══════════════════════════════════════════════
    # Embedding helper
    # ══════════════════════════════════════════════

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from local fastembed model."""
        embedder = self._ensure_embedder()
        try:
            embeddings = await asyncio.to_thread(embedder.embed, texts)
            if not embeddings:
                return []
            return embeddings
        except Exception as exc:
            print(f"[jlc:jre] embed failed: {exc}", file=sys.stderr)
            return []

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ══════════════════════════════════════════════
    # Path helpers
    # ══════════════════════════════════════════════

    def _conv_dir(self, session_id: str | None = None) -> Path:
        return self._root / self._sanitize_session_id(session_id)

    def _context_map_path(self, session_id: str | None = None) -> Path:
        return self._conv_dir(session_id) / "context_map.jsonl"

    @staticmethod
    def _sanitize_session_id(session_id: str | None = None) -> str:
        raw = str(session_id or _SESSION_ID).strip() or _SESSION_ID
        invalid = '<>:"/\\|?*'
        table = str.maketrans({ch: "_" for ch in invalid})
        return raw.translate(table).replace("..", "_")

