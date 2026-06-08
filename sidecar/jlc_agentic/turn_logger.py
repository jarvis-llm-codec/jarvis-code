"""JLC Turn Logger — full-metadata turn storage."""
from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class JLCTurnLogger:
    """Append-only turn logger with full metadata."""

    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root
        # In-memory cache: {conv_id: [turn_data, ...]}
        self._cache: dict[str, list[dict[str, Any]]] = {}
        # Per-conv write lock — threading.Lock so sync `append` works in any
        # call context (asyncio.Lock would have required `async with`, which
        # `append` cannot use since callers may run from sync code).
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _get_lock(self, conv_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(conv_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[conv_id] = lock
            return lock

    def append(self, conv_id: str, entry: dict[str, Any]) -> None:
        """Append a turn entry to turns.jsonl (thread-safe)."""
        entry = dict(entry)
        if entry.get("llm_meta") is None:
            entry.pop("llm_meta", None)
        if "ts" not in entry:
            entry["ts"] = datetime.now(UTC).isoformat()

        turns_path = self._turns_path(conv_id)
        turns_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(entry, ensure_ascii=False) + "\n"
        lock = self._get_lock(conv_id)
        with lock:
            with open(turns_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            if conv_id in self._cache:
                self._cache[conv_id].append(entry)

    def load_all(self, conv_id: str) -> list[dict[str, Any]]:
        """Load all turns (cached in memory after first read)."""
        if conv_id in self._cache:
            return self._cache[conv_id]

        turns_path = self._turns_path(conv_id)
        if not turns_path.exists():
            self._cache[conv_id] = []
            return []

        turns: list[dict[str, Any]] = []
        for line in turns_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    turns.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        self._cache[conv_id] = turns
        return turns

    def load_turn(self, conv_id: str, turn_num: int) -> dict[str, Any] | None:
        """Load a specific turn by number."""
        turns = self.load_all(conv_id)
        for t in turns:
            if t.get("turn") == turn_num:
                return t
        return None

    def search_by_tag(self, conv_id: str, tag: str) -> list[dict[str, Any]]:
        """Find turns containing a specific tag. Returns full user/assistant
        text (no truncation) to preserve raw-data fidelity for paper appendix
        and reviewer reproducibility. Token-budget truncation, if needed,
        belongs in the LLM-context formatting layer (retriever.py), not in
        the storage-layer query API."""
        turns = self.load_all(conv_id)
        tag_lower = tag.lower()
        return [
            {
                "turn": t["turn"],
                "ts": t.get("ts", ""),
                "tags": t.get("tags", []),
                "user": t.get("user", ""),
                "assistant": t.get("assistant", ""),
            }
            for t in turns
            if tag_lower in [tg.lower() for tg in t.get("tags", [])]
        ]

    def invalidate_cache(self, conv_id: str) -> None:
        """Clear in-memory cache for a conversation."""
        self._cache.pop(conv_id, None)

    def _turns_path(self, conv_id: str) -> Path:
        safe = self._sanitize(conv_id)
        return self._root / safe / "turns.jsonl"

    @staticmethod
    def _sanitize(conv_id: str) -> str:
        raw = conv_id.strip() or "default"
        invalid = '<>:"/\\|?*'
        table = str.maketrans({ch: "_" for ch in invalid})
        safe = raw.translate(table)
        return safe.replace("..", "_")
