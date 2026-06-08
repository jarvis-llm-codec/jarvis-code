"""Environment hint directive — OS-specific bash tool usage hints.

Injected into chat / subagent system prompts so the LLM avoids Unix-only
invocations on Windows hosts (`head`, `tail`, `wc`, `awk`, `sed`).
"""
from __future__ import annotations

import platform


def get_env_directive() -> str:
    """Return host-OS-specific bash hint, or empty string for POSIX."""
    if platform.system() == "Windows":
        return (
            "[Environment]\n"
            "Host OS = Windows. The `bash` tool runs through cmd.exe by "
            "default — Unix utilities like `head`, `tail`, `wc`, `awk`, "
            "`sed` are NOT available. Substitute:\n"
            "- `head -N file` -> `powershell -NoProfile -Command "
            "\"Get-Content -Path file -TotalCount N\"`\n"
            "- `tail -N file` -> `powershell -NoProfile -Command "
            "\"Get-Content -Path file -Tail N\"`\n"
            "- `wc -l file`   -> `powershell -NoProfile -Command "
            "\"(Get-Content file | Measure-Object -Line).Lines\"`\n"
            "- Or use the `read_file` tool and slice in your own logic, "
            "or `git log --oneline -N`, `findstr /n` for git/grep cases.\n\n"
        )
    return ""
