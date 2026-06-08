"""Agentic tool handlers."""
from __future__ import annotations

from pathlib import Path


def resolve_under_project_root(path: str, project_root: str | None) -> Path:
    """Resolve `path` and reject any traversal that escapes `project_root`.

    Absolute paths are returned as-is (LLM is allowed to reach outside the
    active project — handoff scenario covers this). Relative paths join under
    `project_root` (cwd when None) and are then verified to live inside the
    root after `..` normalization. Escapes raise ValueError so the dispatcher
    surfaces a tool error instead of silently reading the wrong file.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    if not project_root:
        return p
    root = Path(project_root).resolve()
    resolved = (root / p).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes project_root: {path}") from exc
    return resolved
