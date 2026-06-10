from __future__ import annotations

import ctypes
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PAIR_MISMATCH_MESSAGE = "pair mismatch: this sidecar belongs to another JARVIS window"
MEMORY_WRITE_DISABLED_MESSAGE = "memory write gate retired — shared memory writes are enabled"
MEMORY_WRITE_ENABLED_MESSAGE = "memory write enabled — shared memory writes are active"

_lock_path: Path | None = None
_lock_owned = False
_memory_write_enabled = True
_memory_write_owner: dict[str, Any] | None = None


def current_pair_id() -> str:
    # Priority (a): env var
    pair = str(os.environ.get("JARVIS_PAIR_ID") or "").strip()
    if pair:
        return pair

    # Priority (b): JARVIS_SIDECAR_RUNTIME env → runtime JSON pair_id
    # Fallback: the --sidecar-window single-window launch restarts the sidecar
    # without injecting JARVIS_PAIR_ID, so the Agent-SDK bridge (which runs in
    # this process) sees an empty pair and skips ask_user. Read pair_id from the
    # runtime JSON that jarvis.ps1/jarvis.sh always write (JARVIS_SIDECAR_RUNTIME).
    try:
        runtime_path = str(os.environ.get("JARVIS_SIDECAR_RUNTIME") or "").strip()
        if runtime_path and os.path.isfile(runtime_path):
            with open(runtime_path, encoding="utf-8") as fh:
                data = json.load(fh)
            pair = str(data.get("pair_id") or "").strip()
            if pair:
                return pair
    except Exception:
        pass

    # No env-provided pairing: return empty so pair enforcement stays OFF when
    # there is no real pairing info. The Agent-SDK control bridge then falls back
    # to its shared sentinel bucket (_CONTROL_FALLBACK_PAIR8="selfself"), so
    # ask_user still works without a real pair_id.
    #
    # Do NOT re-add a blind disk scan here. A previous "scan sidecar-runtime-*.json,
    # match by pid else newest-mtime" fallback returned dead windows' pair_ids in
    # env-less/headless contexts (tests, fresh sidecar), which spuriously flipped
    # pair_enforced() True and made pair_matches(None) False -> /route_turn and
    # pair-gated writes rejected every header-less request. The sentinel makes that
    # recovery unnecessary; recovering a stale pair is strictly worse than empty.
    return ""


def pair_enforced() -> bool:
    return bool(current_pair_id())


def pair_matches(header_value: str | None) -> bool:
    pair_id = current_pair_id()
    return not pair_id or str(header_value or "").strip() == pair_id


def conversation_root() -> Path:
    try:
        from jlc_agentic.config import load_config

        return Path(load_config().jhb.storage_path).expanduser()
    except Exception:
        return Path("~/.jarvis-code/conversation").expanduser()


def memory_write_enabled() -> bool:
    return True


def memory_write_owner() -> dict[str, Any] | None:
    return None


def acquire_memory_write_lock(root: Path | None = None) -> bool:
    """M5.5: memory writes are shared; instance.lock is retained only as legacy debris."""
    global _lock_path, _lock_owned, _memory_write_enabled, _memory_write_owner
    _lock_path = None
    _lock_owned = False
    _memory_write_enabled = True
    _memory_write_owner = None
    return True


def release_memory_write_lock() -> None:
    global _lock_path, _lock_owned, _memory_write_enabled, _memory_write_owner
    _lock_path = None
    _lock_owned = False
    _memory_write_enabled = True
    _memory_write_owner = None


def _read_lock(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _owner_pid(owner: dict[str, Any] | None) -> int | None:
    try:
        pid = int((owner or {}).get("pid") or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False

    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)
