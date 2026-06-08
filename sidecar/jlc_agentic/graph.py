"""JLC Relation Graph — co-occurrence based, zero LLM calls."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger("jlc.graph")


class JLCGraph:
    """Tag co-occurrence graph. Updated every N turns (batch)."""

    def __init__(
        self,
        storage_root: Path,
        batch_interval: int = 5,
        max_nodes: int = 500,
        max_edges: int = 2000,
        prune_stale_turns: int = 200,
    ) -> None:
        self._root = storage_root
        self._batch_interval = batch_interval
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self._prune_stale = prune_stale_turns
        # Accumulate tags between flushes: {conv_id: [(turn, tags), ...]}
        self._buffer: dict[str, list[tuple[int, list[str]]]] = {}

    def accumulate(self, conv_id: str, turn: int, tags: list[str]) -> bool:
        """Buffer tags for batch update. Returns True if flush triggered."""
        if not tags:
            return False
        buf = self._buffer.setdefault(conv_id, [])
        buf.append((turn, tags))
        if len(buf) >= self._batch_interval:
            self.flush(conv_id, turn)
            return True
        return False

    def flush(self, conv_id: str, current_turn: int) -> None:
        """Write buffered tags to _graph.json with file lock."""
        buf = self._buffer.pop(conv_id, [])
        if not buf:
            return

        graph_path = self._graph_path(conv_id)
        graph = self._load(graph_path)

        for turn, tags in buf:
            # Update nodes
            for tag in tags:
                node = graph["nodes"].setdefault(tag, {"count": 0, "last_turn": 0})
                node["count"] += 1
                node["last_turn"] = turn
            # Update edges (all pairs)
            for i, a in enumerate(tags):
                for b in tags[i + 1 :]:
                    key = "||".join(sorted([a, b]))
                    edge = graph["edges"].setdefault(key, {"weight": 0, "last_turn": 0})
                    edge["weight"] += 1
                    edge["last_turn"] = turn

        # Prune stale edges: weight=1 and older than prune_stale turns
        cutoff = current_turn - self._prune_stale
        stale_keys = [
            k for k, v in graph["edges"].items()
            if v["weight"] <= 1 and v["last_turn"] < cutoff
        ]
        for k in stale_keys:
            del graph["edges"][k]

        # Cap nodes by count (keep highest)
        if len(graph["nodes"]) > self._max_nodes:
            sorted_nodes = sorted(
                graph["nodes"].items(), key=lambda x: x[1]["count"], reverse=True,
            )
            graph["nodes"] = dict(sorted_nodes[: self._max_nodes])

        # Cap edges by weight (keep highest)
        if len(graph["edges"]) > self._max_edges:
            sorted_edges = sorted(
                graph["edges"].items(), key=lambda x: x[1]["weight"], reverse=True,
            )
            graph["edges"] = dict(sorted_edges[: self._max_edges])

        self._save(graph_path, graph)

    def flush_all(self) -> None:
        """Flush all buffered conversations (e.g., on shutdown)."""
        for conv_id in list(self._buffer.keys()):
            buf = self._buffer.get(conv_id, [])
            last_turn = buf[-1][0] if buf else 0
            self.flush(conv_id, last_turn)

    def load(self, conv_id: str) -> dict[str, Any]:
        """Load graph for API access."""
        return self._load(self._graph_path(conv_id))

    def neighbors(self, conv_id: str, tag: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Find top-k related tags by edge weight."""
        graph = self.load(conv_id)
        results: list[dict[str, Any]] = []
        for key, edge in graph["edges"].items():
            parts = key.split("||")
            if len(parts) != 2:
                continue
            if tag in parts:
                other = parts[0] if parts[1] == tag else parts[1]
                results.append({"tag": other, "weight": edge["weight"], "last_turn": edge["last_turn"]})
        results.sort(key=lambda x: x["weight"], reverse=True)
        return results[:top_k]

    # -- Internal --

    def _graph_path(self, conv_id: str) -> Path:
        safe = self._sanitize(conv_id)
        return self._root / safe / "_graph.json"

    @staticmethod
    def _sanitize(conv_id: str) -> str:
        raw = conv_id.strip() or "default"
        invalid = '<>:"/\\|?*.~'
        table = str.maketrans({ch: "_" for ch in invalid})
        safe = raw.translate(table)
        # Block path traversal
        return safe.replace("..", "_")

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"nodes": {}, "edges": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("graph load failed for %s: %s", path, exc)
            return {"nodes": {}, "edges": {}}

    @staticmethod
    def _save(path: Path, graph: dict[str, Any]) -> None:
        """Crash-safe write: unique tempfile + fsync + os.replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(graph, ensure_ascii=False, separators=(",", ":"))
        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            try:
                os.fchmod(fd, 0o600)
            except (OSError, AttributeError):
                pass
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
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
