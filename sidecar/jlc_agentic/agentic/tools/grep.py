"""Grep tool."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

SCHEMA = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "Search text in files using ripgrep. Pass `path` as narrowly as "
            "possible (a specific subdirectory or file) ? searching from a "
            "repo root with vendored sources or large fork trees can hit the "
            "timeout. Use the `glob` parameter to filter by extension "
            "(e.g. '*.py'). .git / node_modules / __pycache__ / virtualenvs "
            "and other common build/cache trees are excluded by default."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "glob": {"type": "string"},
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
}

_DEFAULT_EXCLUDES = (
    "!**/.git/**",
    "!**/node_modules/**",
    "!**/__pycache__/**",
    "!**/.venv/**",
    "!**/venv/**",
    "!**/.mypy_cache/**",
    "!**/.pytest_cache/**",
    "!**/.ruff_cache/**",
    "!**/dist/**",
    "!**/build/**",
    "!**/.tox/**",
    "!**/*.egg-info/**",
    "!**/.aider.*",
    "!**/.aider.chat.history.md",
    "!**/.aider.input.history",
    "!**/.aider.tags.cache.v*/**",
)
_FALLBACK_PATH_BLOCKLIST = (
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".egg-info",
    ".aider.tags.cache",
    ".aider.chat.history",
    ".aider.input.history",
)
_FALLBACK_SUFFIX_BLOCKLIST = (
    ".db", ".sqlite", ".pyc", ".pyo", ".so", ".dll", ".exe",
    ".pdf", ".zip", ".gz", ".tar", ".whl", ".bin",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".mp3", ".mp4", ".wav", ".ogg",
    ".woff", ".woff2", ".ttf", ".otf",
)
_TIMEOUT_SEC = 8


def _resolve_search_path(path: str, project_root: str | None) -> str:
    if path == ".":
        return project_root if project_root else path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    if not project_root:
        return path
    root = Path(project_root).resolve()
    resolved = (root / p).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes project_root: {path}") from exc
    return str(resolved)


def handler(pattern: str, path: str = ".", glob: str | None = None, project_root: str | None = None) -> dict:
    """Search for regex pattern and return up to 200 matches."""
    effective_path = _resolve_search_path(path, project_root)
    cmd = ["rg", "-n", "--max-count", "200"]
    for exc in _DEFAULT_EXCLUDES:
        cmd.extend(["-g", exc])
    if glob:
        cmd.extend(["-g", glob])
    cmd.extend([pattern, effective_path])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=_TIMEOUT_SEC)
        if proc.returncode in (0, 1):
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()][:200]
            return {"engine": "rg", "matches": lines, "count": len(lines)}
    except subprocess.TimeoutExpired:
        return {
            "engine": "rg",
            "matches": [],
            "count": 0,
            "error": (
                f"ripgrep timeout {_TIMEOUT_SEC}s ? narrow `path` to a "
                "subdirectory or specific file."
            ),
        }
    except FileNotFoundError:
        pass

    regex = re.compile(pattern)
    matches: list[str] = []
    root = Path(effective_path)
    files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    deadline = time.monotonic() + _TIMEOUT_SEC
    for fp in files:
        if time.monotonic() > deadline:
            break
        fp_str = str(fp).replace("\\", "/")
        if any(token in fp_str for token in _FALLBACK_PATH_BLOCKLIST):
            continue
        if fp.suffix.lower() in _FALLBACK_SUFFIX_BLOCKLIST:
            continue
        if glob and not fp.match(glob):
            continue
        try:
            for idx, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                if regex.search(line):
                    matches.append(f"{fp}:{idx}:{line}")
                    if len(matches) >= 200:
                        return {"engine": "python_re", "matches": matches, "count": len(matches)}
        except Exception:
            continue
    return {"engine": "python_re", "matches": matches, "count": len(matches)}
