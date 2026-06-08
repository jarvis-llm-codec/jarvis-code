"""Bash tool."""
from __future__ import annotations

import subprocess

SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run shell command and capture stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "cwd": {"type": "string", "default": "."},
                "timeout": {"type": "integer", "default": 60},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


def handler(command: str, cwd: str = ".", timeout: int = 60, project_root: str | None = None) -> dict:
    """Execute shell command with timeout."""
    effective_cwd = project_root if cwd == "." and project_root else cwd
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=effective_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": effective_cwd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": effective_cwd,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nTimeoutExpired",
        }
