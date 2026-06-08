"""Register a project folder so subsequent turns can route to it by name.

The router (`_auto_route_active_project`) auto-registers a folder only when the
USER's utterance contains its absolute path (Tier 1). Folders the assistant
creates itself, or pre-existing folders the user only nicknames ("테트리스"),
stay invisible to the router. This tool gives the assistant an explicit way to
register such folders so the next turn's `registry.match` finds them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# System directories we refuse to register — registering these would seed
# JARVIS.md inside sensitive paths and pollute the registry. Match by string
# prefix on the resolved path; both Windows and POSIX patterns are covered.
_SYSTEM_DENY_PREFIXES: tuple[str, ...] = (
    # Windows OS roots only — `C:\Users\Public`/`Default`/`All Users` are NOT
    # listed because some IME/installer setups (Hangul ESTsoft IME observed
    # 2026-05-03) put pytest's tmp_path under `C:\Users\Public\Documents\...`
    # and a blanket deny would break every test that lands there. The
    # _SENSITIVE_SUFFIXES list below still catches sensitive subfolder names.
    "C:\\Windows", "C:\\Program Files", "C:\\Program Files (x86)",
    "C:\\ProgramData", "C:\\$Recycle.Bin",
    # Non-standard System32/SysWOW64 spellings — the canonical paths
    # under C:\Windows\... are already covered above, but a malicious or
    # mistaken absolute reference to bare C:\System32 should still be
    # rejected.
    "C:\\System32", "C:\\SysWOW64",
    # POSIX system roots only — /home /mnt /media stay ALLOWED (normal
    # project locations).
    "/tmp", "/var", "/usr", "/etc", "/proc", "/sys", "/dev", "/bin", "/sbin",
    "/Library", "/System", "/Applications", "/run",
    # Additional POSIX system roots flagged by parity review (2026-05-03).
    "/root", "/opt", "/boot", "/srv",
)

# Sensitive user-folder suffixes — never register these even when they live
# under the user's home directory (they're config / secret stores, not
# projects).
_SENSITIVE_SUFFIXES: tuple[str, ...] = (
    ".ssh", ".gnupg", ".aws", ".kube", ".docker", ".config",
    ".cache", ".npm", ".venv", ".local", ".git",
    "AppData", "Library",
    # Parity-review additions (2026-05-03): GnuPG variant + JS dep cache +
    # credential/env folders accidentally pointed at as a project root.
    ".gpg", "node_modules", ".npmrc", ".env",
)


def _looks_sensitive(resolved: str) -> bool:
    norm = resolved.replace("/", os.sep)
    upper = norm.upper()
    for prefix in _SYSTEM_DENY_PREFIXES:
        pref_norm = prefix.replace("/", os.sep).upper()
        if upper == pref_norm or upper.startswith(pref_norm + os.sep):
            return True
    tail = Path(resolved).name
    if tail in _SENSITIVE_SUFFIXES:
        return True
    return False


def safe_resolve_for_register(p_in: Path) -> tuple[bool, Path | None, str | None]:
    """Shared guard for any code path that registers a project folder.

    Phase 1.13 wired these checks into `register_project.handler` only.
    Phase 1.14.B extracts them so the Tier 1 auto-register branch in
    `_auto_route_active_project` (utterance-driven absolute-path routing
    in `jlc_agentic_coder.py`) gets the same protection — without that
    parity an attacker can sidestep the explicit-tool guards by simply
    pasting an absolute path into chat.

    Returns:
        (ok, resolved_path, error)
        · ok=True  → resolved_path is a strict-resolved Path safe to register
        · ok=False → error is a short reason; resolved_path is None

    Checks (in order):
      1. Reject symlinks / Windows junctions (POSIX symlink + reparse point).
      2. `resolve(strict=True)` — must exist on disk; defeats relative
         escapes and dangling links.
      3. Reject system roots (Windows OS / Program Files / POSIX /etc /tmp
         …) and sensitive suffix folders (.ssh / .gnupg / AppData / …).
    """
    try:
        if p_in.is_symlink() or os.path.islink(str(p_in)):
            return False, None, f"refusing to register a symlink/junction: {p_in}"
    except OSError as exc:
        return False, None, f"link check failed: {exc}"

    try:
        resolved_path = p_in.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        return False, None, f"resolve failed: {exc}"
    resolved = str(resolved_path)

    if _looks_sensitive(resolved):
        return False, None, f"refusing to register a system / sensitive path: {resolved}"

    return True, resolved_path, None


SCHEMA = {
    "type": "function",
    "function": {
        "name": "register_project",
        "description": (
            "Register an existing folder as a JLC project so future turns can "
            "route to it by its name in the user's utterance. Call this "
            "(a) immediately after creating a NEW project folder via "
            "`bash`/`write_file`, or (b) when the user nicknames a folder that "
            "is not yet registered (use the absolute path you can see in JHB "
            "memory or have just created). Idempotent — re-registering an "
            "already-registered path returns the existing entry without error. "
            "Also seeds JARVIS.md when missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path of the existing folder to register.",
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Optional override for the registered name. Defaults to "
                        "the folder's basename. Use this when the folder name "
                        "is generic ('app', 'src') and a clearer alias helps "
                        "future routing."
                    ),
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}


def handler(path: str, name: str | None = None, project_root: str | None = None) -> dict:
    """Register `path` in the Registry. `project_root` is dispatcher-injected
    and ignored here (this tool intentionally operates outside the active
    project root so brand-new sibling projects can be registered).

    Hardened in GAN review (2026-05-03):
      · Symlinks rejected — would let an attacker register `/etc` via
        `/tmp/atk -> /etc` (MiniMax Critical).
      · System directories (Windows / Program Files / /tmp / /etc / .ssh /
        AppData …) refused — auto-seeding JARVIS.md inside these is
        polluting (Kimi/Qwen High).
      · `p` resolved at the top so the idempotency check and registry.add
        agree on a single canonical string (Qwen Mid).
    """
    from jlc_agentic.registration import initialize_jarvis_md, scan_project
    from jlc_agentic.registry import Registry

    p_in = Path(path).expanduser()
    if not p_in.exists():
        return {"ok": False, "error": f"path does not exist: {p_in}"}
    if not p_in.is_dir():
        return {"ok": False, "error": f"path is not a directory: {p_in}"}

    ok, resolved_path, err = safe_resolve_for_register(p_in)
    if not ok:
        return {"ok": False, "error": err}
    resolved = str(resolved_path)
    p = resolved_path  # use canonical path everywhere downstream

    registry = Registry()
    existing = None
    for entry in registry.all():
        try:
            entry_resolved = str(Path(entry.path).expanduser().resolve())
        except OSError as exc:
            print(
                f"[jlc:register_project] WARNING: cannot resolve registered "
                f"path '{entry.path}' ({exc}); skipping for dedup",
                file=sys.stderr,
            )
            continue
        if entry_resolved == resolved:
            existing = entry
            break
    if existing is not None:
        return {
            "ok": True,
            "already_registered": True,
            "project_id": existing.project_id,
            "name": existing.name,
            "path": existing.path,
        }

    try:
        scan = scan_project(p, name=name)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"scan failed: {exc}"}

    jarvis_md_state = "existing"
    try:
        initialize_jarvis_md(p, scan)
        jarvis_md_state = "created"
    except FileExistsError:
        pass
    except Exception as exc:  # noqa: BLE001
        jarvis_md_state = f"skipped ({exc})"

    entry_name = name or scan.name
    entry = registry.add(name=entry_name, path=str(p))
    return {
        "ok": True,
        "already_registered": False,
        "project_id": entry.project_id,
        "name": entry.name,
        "path": entry.path,
        "jarvis_md": jarvis_md_state,
        "languages": scan.primary_languages,
        "file_count": scan.file_count,
        "markers": scan.markers,
    }
