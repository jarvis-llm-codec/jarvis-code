"""Read file tool."""
from __future__ import annotations

from pathlib import Path

CHUNK_BYTES = 50 * 1024

SCHEMA = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file (entire or by line range).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "default": 0},
                "end": {"type": "integer", "default": -1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


from . import resolve_under_project_root as _resolve_path


def handler(path: str, start: int = 0, end: int = -1, project_root: str | None = None) -> dict:
    """Read file content with optional line slicing."""
    target = _resolve_path(path, project_root)
    text = target.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    sliced = lines[start:] if end == -1 else lines[start:end]
    content = "".join(sliced)
    raw = content.encode("utf-8")
    if len(raw) <= CHUNK_BYTES:
        return {"path": str(target), "start": start, "end": end, "chunked": False, "content": content}
    chunks = []
    offset = 0
    while offset < len(raw):
        part = raw[offset : offset + CHUNK_BYTES].decode("utf-8", errors="replace")
        chunks.append(part)
        offset += CHUNK_BYTES
    return {
        "path": str(target),
        "start": start,
        "end": end,
        "chunked": True,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
