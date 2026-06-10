from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from . import pairing

# Closing a console window with the X button kills the wrapper without
# running its finally block, so the shutdown choreography never reaches the
# sidecar (live incident, 2026-06-11: zombie sidecars piled up per closed
# window). The sidecar watches its wrapper pid and exits on its own; the
# watchdog is a job of the same wrapper, so it can never race a respawn.

_POLL_SECONDS = 5.0


def wrapper_pid_from_env() -> int:
    try:
        pid = int(str(os.environ.get("JARVIS_WRAPPER_PID") or "0").strip())
    except ValueError:
        return 0
    return pid if pid > 0 else 0


def cleanup_own_runtime_file() -> None:
    raw = str(os.environ.get("JARVIS_SIDECAR_RUNTIME") or "").strip()
    if not raw:
        return
    runtime_path = Path(raw)
    try:
        runtime_path.unlink(missing_ok=True)
    except OSError:
        pass
    # The dead wrapper's watchdog sentinel has no other sweeper; it would
    # accumulate forever in data/ after X-closed windows.
    wrapper_pid = wrapper_pid_from_env()
    if wrapper_pid:
        try:
            (runtime_path.parent / f"sidecar-watchdog-{wrapper_pid}.run").unlink(missing_ok=True)
        except OSError:
            pass


def watch_wrapper_once(wrapper_pid: int) -> bool:
    """Returns True while the wrapper is alive."""
    return wrapper_pid > 0 and pairing._pid_alive(wrapper_pid)


def _watch_loop(wrapper_pid: int, poll_seconds: float) -> None:
    while True:
        time.sleep(poll_seconds)
        if not watch_wrapper_once(wrapper_pid):
            cleanup_own_runtime_file()
            os._exit(0)


def start_wrapper_watch(poll_seconds: float = _POLL_SECONDS) -> threading.Thread | None:
    wrapper_pid = wrapper_pid_from_env()
    if not wrapper_pid:
        return None
    thread = threading.Thread(
        target=_watch_loop,
        args=(wrapper_pid, poll_seconds),
        name="jarvis-wrapper-watch",
        daemon=True,
    )
    thread.start()
    return thread
