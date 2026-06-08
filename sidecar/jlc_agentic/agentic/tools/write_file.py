"""Write file tool ? create or fully overwrite a file with content."""
from __future__ import annotations

from pathlib import Path

SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": (
            "Create a new file or fully overwrite an existing file with the given content. "
            "Parent directories are created automatically. "
            "Prefer this over `bash` for writing multi-line files (heredoc/escape-free). "
            "Use `edit` only for partial replacements in existing files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path."},
                "content": {"type": "string", "description": "Full file content (UTF-8)."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
}


from . import resolve_under_project_root as _resolve_path


def handler(path: str, content: str, project_root: str | None = None) -> dict:
    fp = _resolve_path(path, project_root)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content, encoding="utf-8")
    return {"path": str(fp), "bytes": len(content.encode("utf-8"))}
