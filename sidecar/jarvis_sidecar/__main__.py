from __future__ import annotations

import atexit
import copy
import os
from pathlib import Path
import socket as _sock
import sys
import faulthandler
import ctypes

import uvicorn
from uvicorn.config import LOGGING_CONFIG


class _Tee:
    def __init__(self, *streams):
        self._streams = tuple(stream for stream in streams if stream is not None)

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self) -> int:
        for stream in self._streams:
            try:
                return stream.fileno()
            except Exception:
                pass
        raise OSError("no fileno available")

    @property
    def encoding(self) -> str:
        for stream in self._streams:
            encoding = getattr(stream, "encoding", None)
            if encoding:
                return encoding
        return "utf-8"

    def write(self, text: str) -> int:
        for stream in self._streams:
            try:
                stream.write(text)
                stream.flush()
            except Exception:
                pass
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            try:
                stream.flush()
            except Exception:
                pass


def _install_process_log():
    _enable_windows_ansi()
    repo_root = Path(__file__).resolve().parents[2]
    log_dir = repo_root / "data"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "sidecar-process.log", "a", encoding="utf-8", buffering=1)
    console_out = _open_console_stream(sys.stdout)
    console_err = _open_console_stream(sys.stderr)
    sys.stdout = _Tee(sys.stdout, console_out, log_file)
    sys.stderr = _Tee(sys.stderr, console_err, log_file)
    faulthandler.enable(file=log_file, all_threads=True)
    print(f"[jarvis-sidecar] process start pid={os.getpid()}", flush=True)
    atexit.register(lambda: print(f"[jarvis-sidecar] process exit pid={os.getpid()}", flush=True))


def _open_console_stream(reference):
    if os.name != "nt":
        return None
    if getattr(reference, "isatty", lambda: False)():
        return None
    try:
        return open(
            "CONOUT$",
            "w",
            encoding=getattr(reference, "encoding", None) or "utf-8",
            errors="replace",
            buffering=1,
        )
    except OSError:
        return None


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        enable_vt = 0x0004
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            if not handle or handle == ctypes.c_void_p(-1).value:
                continue
            mode = ctypes.c_uint32()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                continue
            kernel32.SetConsoleMode(handle, mode.value | enable_vt)
    except Exception:
        pass


def main() -> None:
    _install_process_log()
    host = os.environ.get("JARVIS_SIDECAR_HOST", "127.0.0.1")
    port = int(os.environ.get("JARVIS_SIDECAR_PORT", "8765"))
    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as _s:
        if _s.connect_ex((host, port)) == 0:
            print(f"[jarvis-sidecar] port {port} already in use; exiting to avoid duplicate", flush=True)
            return
    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["loggers"]["uvicorn.access"]["handlers"] = []
    log_config["loggers"]["uvicorn.access"]["propagate"] = False
    uvicorn.run(
        "jarvis_sidecar.app:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
