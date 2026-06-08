"""Launcher helpers for the local Jarvis Web UI."""
from __future__ import annotations

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"
WS_PORT_START = 7150
WS_PORT_END = 7250
VITE_PORT = 5173
RUNTIME_PATH = Path.home() / ".jarvis-code" / "runtime.json"

_BOOTED = False
_WS_PORT: int | None = None
_PROCS: list[subprocess.Popen[Any]] = []


def _port_open(port: int, host: str = HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _find_free_port() -> int:
    for port in range(WS_PORT_START, WS_PORT_END + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free Jarvis UI sidecar port in {WS_PORT_START}-{WS_PORT_END}")


def _wait_for_port(port: int, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_open(port):
            return True
        time.sleep(0.2)
    return False


def _write_runtime(ws_port: int) -> None:
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ws_port": ws_port,
        "vite_port": VITE_PORT,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    RUNTIME_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cleanup_runtime() -> None:
    if _WS_PORT is None or not RUNTIME_PATH.exists():
        return
    try:
        current = json.loads(RUNTIME_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    if current.get("ws_port") == _WS_PORT:
        try:
            RUNTIME_PATH.unlink()
        except OSError:
            pass


def _cleanup_processes() -> None:
    for proc in _PROCS:
        if proc.poll() is not None:
            continue
        try:
            proc.terminate()
        except OSError:
            pass


def _start_sidecar(ws_port: int, repo_root: Path) -> None:
    # Run the sidecar uvicorn server in an in-process daemon thread so the
    # JarvisAgentic singleton (jlc_agentic.get_slim) is shared with Aider's
    # coder, the bench TCP listener, and the WS endpoint. Previously this
    # was a subprocess.Popen, which gave each process its own LocalEmbedder
    # and double-loaded bge-m3 weights every session.
    from .app import start_sidecar_once

    start_sidecar_once(host=HOST, port=ws_port)
    if not _wait_for_port(ws_port):
        raise RuntimeError(f"sidecar did not become ready on port {ws_port}")
    sys.stderr.write(f"[jarvis-ui] sidecar 띄움 (in-process thread) ws://localhost:{ws_port}/ws\n")
    sys.stderr.flush()


def _start_vite(repo_root: Path) -> bool:
    if _port_open(VITE_PORT):
        sys.stderr.write(f"[jarvis-ui] vite dev server :{VITE_PORT} ready\n")
        sys.stderr.flush()
        return True

    ui_dir = repo_root / "ui"
    if not (ui_dir / "package.json").exists():
        sys.stderr.write("[jarvis-ui] ui/package.json 없음 - Vite 자동 부팅 skip\n")
        sys.stderr.flush()
        return False
    if not (ui_dir / "node_modules").exists():
        sys.stderr.write("[jarvis-ui] ui dependencies 없음. cmd 창에서 'cd ui && npm install' 실행 필요\n")
        sys.stderr.flush()
        return False
    npm = shutil.which("npm")
    if npm is None:
        sys.stderr.write("[jarvis-ui] npm 실행 파일을 찾지 못함 - Vite 자동 부팅 skip\n")
        sys.stderr.flush()
        return False

    proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(ui_dir),
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    _PROCS.append(proc)
    if not _wait_for_port(VITE_PORT, timeout=30.0):
        sys.stderr.write(f"[jarvis-ui] vite dev server :{VITE_PORT} ready 대기 실패\n")
        sys.stderr.flush()
        return False
    sys.stderr.write(f"[jarvis-ui] vite dev server :{VITE_PORT} ready\n")
    sys.stderr.flush()
    return True


def boot_web_ui(repo_root: str | Path | None = None, open_browser: bool = True) -> int:
    """Start sidecar + Vite once for this aider process and return the WS port."""
    global _BOOTED, _WS_PORT
    if _BOOTED and _WS_PORT is not None:
        return _WS_PORT

    root = Path(repo_root or Path.cwd()).resolve()
    ws_port = _find_free_port()
    _WS_PORT = ws_port
    _write_runtime(ws_port)
    atexit.register(_cleanup_runtime)
    atexit.register(_cleanup_processes)

    _start_sidecar(ws_port, root)
    vite_ready = _start_vite(root)
    if vite_ready:
        url = f"http://localhost:{VITE_PORT}/?ws_port={ws_port}"
        if open_browser and os.environ.get("JARVIS_UI_NO_BROWSER") != "1" and "--no-browser" not in sys.argv:
            webbrowser.open(url)
            sys.stderr.write(f"[jarvis-ui] browser open {url}\n")
        else:
            sys.stderr.write(f"[jarvis-ui] browser skip {url}\n")
        sys.stderr.flush()

    _BOOTED = True
    return ws_port
