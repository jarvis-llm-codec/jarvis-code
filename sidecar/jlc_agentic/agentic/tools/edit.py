"""Edit tool."""
from __future__ import annotations

from pathlib import Path

SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Replace text in a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        },
    },
}


from . import resolve_under_project_root as _resolve_path


def handler(
    path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    project_root: str | None = None,
) -> dict:
    """Replace text once or all occurrences."""
    fp = _resolve_path(path, project_root)
    text = fp.read_text(encoding="utf-8", errors="replace")
    if old_string not in text:
        raise ValueError("old_string not found")
    if replace_all:
        updated = text.replace(old_string, new_string)
        count = text.count(old_string)
    else:
        updated = text.replace(old_string, new_string, 1)
        count = 1
    fp.write_text(updated, encoding="utf-8")
    return {"path": str(fp), "replacements": count}
