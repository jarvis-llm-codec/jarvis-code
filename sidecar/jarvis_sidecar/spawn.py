from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import pairing
from .window_labels import (
    data_dir,
    pair8_from_pair_id,
    read_runtime,
    runtime_files,
    sanitize_window_label,
    set_runtime_label,
)

SPAWN_ENV_CUT_KEYS = frozenset({
    "JARVIS_PAIR_ID",
    "JARVIS_SIDECAR_URL",
    "JARVIS_DEFAULT_PROVIDER",
    "JARVIS_DEFAULT_MODEL",
    # An inherited label would duplicate the parent's name onto the child
    # and trip the unique-live-label resolver with ambiguity errors.
    "JARVIS_WINDOW_LABEL",
    # Spawned children break away from the parent's Job Object, but the
    # inherited flag makes their jarvis.ps1 skip creating its own job —
    # X-closing the child then leaves its dev servers alive.
    "JARVIS_IN_PROCESS_JOB",
})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def jarvis_script_path() -> Path:
    return repo_root() / "jarvis.ps1"


def child_spawn_env(
    parent: dict[str, str] | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, str]:
    env = dict(parent or os.environ)
    for key in list(env):
        if key.upper() in SPAWN_ENV_CUT_KEYS:
            env.pop(key, None)
    if provider and model:
        env["JARVIS_DEFAULT_PROVIDER"] = provider
        env["JARVIS_DEFAULT_MODEL"] = model
    # Mark the child as a spawned worker so its sidecar stores JHB under a
    # bounded _windows/worker<N> slot instead of the main window's durable
    # conversation home. The main window (launched directly) never sets this.
    env["JARVIS_SPAWNED"] = "1"
    return env


def _runtime_files() -> set[Path]:
    return runtime_files()


def _read_runtime(path: Path) -> dict[str, Any] | None:
    return read_runtime(path)


def _candidate_runtime(path: Path, *, current_pair8: str | None, known_paths: set[Path]) -> dict[str, Any] | None:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if resolved in known_paths:
        return None
    record = _read_runtime(path)
    if not record:
        return None
    pair_id = str(record.get("pair_id") or "")
    pair8 = pair8_from_pair_id(pair_id)
    if not pair8 or pair8 == current_pair8:
        return None
    try:
        pid = int(record.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or not pairing._pid_alive(pid):
        return None
    return {
        "pair8": pair8,
        "pair_id": pair_id,
        "url": str(record.get("url") or ""),
        "port": record.get("port"),
        "pid": pid,
        "runtime_path": str(path),
    }


# Live evidence (2026-06-11): a child window's wrapper can take 2+ minutes to
# reach the runtime-file write on a cold machine (PowerShell + wt startup under
# load). The poll returns as soon as the file appears, so a long cap only
# costs time in the genuinely-slow case it exists to cover.
DEFAULT_SPAWN_TIMEOUT_SECONDS = 150.0


def wait_for_spawned_runtime(
    *,
    current_pair8: str | None,
    known_paths: set[Path],
    timeout_seconds: float = DEFAULT_SPAWN_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        for path in sorted(data_dir().glob("sidecar-runtime-*.json")):
            candidate = _candidate_runtime(path, current_pair8=current_pair8, known_paths=known_paths)
            if candidate is not None:
                return candidate
        time.sleep(0.25)
    return None


def _powershell_exe() -> str:
    return shutil.which("powershell.exe") or shutil.which("powershell") or "powershell.exe"


def build_launch_argv(
    *,
    wt: str | None,
    powershell: str,
    root: str,
    script: str,
    provider: str | None = None,
    model: str | None = None,
    label: str | None = None,
) -> list[str]:
    shell_args = [powershell, "-NoExit", "-ExecutionPolicy", "Bypass", "-File", script]
    if provider and model:
        shell_args.extend(["--provider", provider, "--model", model])
    if label:
        shell_args.extend(["--window-label", label])
    if wt:
        # wt has no new-window subcommand; a new window is "-w new" plus an
        # explicit new-tab. The tab may be hosted by an existing
        # WindowsTerminal process with its own environment, so the cut env
        # block is not guaranteed to reach the child — that leak is absorbed
        # by the wrapper's inherited-pair guard (live owner -> regenerate)
        # and by the wrapper overwriting JARVIS_SIDECAR_URL itself.
        return [wt, "-w", "new", "nt", "-d", root, *shell_args]
    return shell_args


def _current_platform() -> str:
    """Coarse OS bucket for spawn dispatch. Kept as a tiny seam so tests can
    drive the Unix/macOS backends on a Windows dev host (monkeypatch this)."""
    if os.name == "nt":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "unix"


def jarvis_launch_target() -> Path:
    """The launch entrypoint a spawned worker should exec.

    Windows opens ``jarvis.ps1`` (PowerShell wrapper); every Unix host opens
    ``jarvis.sh`` (the POSIX shim that resolves the repo and delegates to
    ``scripts/jarvis-launcher.py``). The two must never be crossed — ``jarvis.ps1``
    under a Unix shell dies immediately, which is exactly the ``0x80070002`` a
    Linux worker spawn hit before this dispatcher existed.
    """
    if _current_platform() == "windows":
        return repo_root() / "jarvis.ps1"
    return repo_root() / "jarvis.sh"


# Linux desktop terminal emulators, in preference order, with tmux as the
# headless fallback so a desktopless host (CI, plain SSH, WSL without WSLg) can
# still bring up a worker session. The first binary found on PATH wins.
UNIX_TERMINAL_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("gnome-terminal", "gnome-terminal"),
    ("konsole", "konsole"),
    ("xterm", "xterm"),
    ("tmux", "tmux"),
)

# GUI terminals are useless without a display server, yet their binary may still
# be on PATH on a headless host (SSH, CI, WSL without WSLg). Launching one there
# fails to open a window, no runtime file is ever written, and the spawn waiter
# then blocks the full timeout. Skip them when no display is advertised so the
# search falls through to the headless tmux backend.
GUI_UNIX_TERMINALS = frozenset({"gnome-terminal", "konsole", "xterm"})

# Operational, non-secret env vars forwarded to a tmux worker session (see the
# tmux branch in build_unix_launch_argv). Deliberately excludes credentials:
# forwarded values land on the tmux command line (world-readable via
# /proc/<pid>/cmdline), and the worker's launcher loads its API keys from
# credentials.yaml itself.
TMUX_FORWARD_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "USER", "LOGNAME")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def detect_unix_terminal() -> tuple[str, str] | None:
    """Return ``(kind, abs_path)`` for the first usable terminal, else None.

    GUI terminals are only eligible when a display is available; otherwise the
    search falls through to the headless tmux backend.
    """
    has_display = _has_display()
    for kind, exe in UNIX_TERMINAL_CANDIDATES:
        if kind in GUI_UNIX_TERMINALS and not has_display:
            continue
        found = shutil.which(exe)
        if found:
            return kind, found
    return None


def jarvis_worker_command(
    launch_target: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    label: str | None = None,
) -> list[str]:
    """The ``jarvis.sh ...`` argv a worker terminal should execute.

    Provider/model/label ride as explicit CLI flags rather than only as env:
    ``jarvis-launcher.py`` forwards ``--provider``/``--model`` to Pi and consumes
    ``--window-label`` itself, and a tmux server reuses its own (stale) env, so
    the flag path is the one that survives every Unix launch backend.
    """
    command = [launch_target]
    if provider and model:
        command.extend(["--provider", provider, "--model", model])
    if label:
        command.extend(["--window-label", label])
    return command


def build_unix_launch_argv(
    *,
    terminal: str,
    term_path: str,
    root: str,
    command: list[str],
    session_hint: str | None = None,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Wrap the jarvis worker command in the chosen terminal's launch syntax.

    ``env`` is consumed only by the tmux backend; the GUI terminals are launched
    as fresh processes and inherit our env through ``Popen(env=...)``.
    """
    if terminal == "gnome-terminal":
        return [term_path, "--working-directory", root, "--", *command]
    if terminal == "konsole":
        return [term_path, "--workdir", root, "-e", *command]
    if terminal == "xterm":
        # xterm has no working-directory flag; Popen(cwd=root) sets the process
        # cwd, and jarvis.sh re-resolves the repo root regardless.
        return [term_path, "-e", *command]
    if terminal == "tmux":
        # Name the session by the worker label (unique live slot) so a second
        # spawn does not collide on the parent's pair id.
        session = f"jarvis-{session_hint}" if session_hint else "jarvis"
        # A fresh tmux server (the common case: our Popen starts it) inherits our
        # env, but a pre-existing SERVER hands the new session its OWN, possibly
        # stale, environment — so PATH/HOME for python3+node can be wrong and
        # jarvis.sh fails to boot. tmux's own -e is unreliable for PATH (verified
        # on tmux 3.x: -e HOME took but -e PATH did not), so prefix the command
        # with `env KEY=VALUE` to set the worker's environment deterministically
        # right before exec. Only non-secret operational vars are forwarded.
        source = env or {}
        env_prefix = [f"{key}={source[key]}" for key in TMUX_FORWARD_ENV_KEYS if source.get(key)]
        wrapped = ["env", *env_prefix, *command] if env_prefix else list(command)
        return [term_path, "new-session", "-d", "-s", session, *wrapped]
    raise ValueError(f"unsupported unix terminal backend: {terminal!r}")


def build_macos_launch_argv(*, root: str, command: list[str]) -> list[str]:
    """osascript ``do script`` argv that opens the worker in Terminal.app.

    UNVERIFIED: built but never exercised on a real macOS host (no runtime
    available, 2026-06-14). Each token is POSIX-quoted for the inner shell, then
    the whole line is embedded as an AppleScript string literal (escape backslash
    and double-quote). ``exec`` keeps the shell from lingering after jarvis exits.
    """
    inner = "cd " + shlex.quote(root) + " && exec " + " ".join(shlex.quote(c) for c in command)
    applescript = 'tell application "Terminal" to do script "%s"' % (
        inner.replace("\\", "\\\\").replace('"', '\\"')
    )
    return ["osascript", "-e", applescript]


def _launch_windows_window(
    env: dict[str, str],
    *,
    root: str,
    provider: str | None,
    model: str | None,
    label: str | None,
) -> subprocess.Popen[Any]:
    script = str(jarvis_script_path())
    wt = shutil.which("wt.exe") or shutil.which("wt")
    argv = build_launch_argv(
        wt=wt,
        powershell=_powershell_exe(),
        root=root,
        script=script,
        provider=provider,
        model=model,
        label=label,
    )
    breakaway = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
    creationflags = breakaway if wt else (getattr(subprocess, "CREATE_NEW_CONSOLE", 0) | breakaway)
    return subprocess.Popen(argv, cwd=root, env=env, creationflags=creationflags)


def _launch_unix_window(
    env: dict[str, str],
    *,
    platform: str,
    root: str,
    provider: str | None,
    model: str | None,
    label: str | None,
) -> subprocess.Popen[Any]:
    command = jarvis_worker_command(
        str(jarvis_launch_target()), provider=provider, model=model, label=label
    )
    if platform == "macos":
        argv = build_macos_launch_argv(root=root, command=command)
    else:
        terminal = detect_unix_terminal()
        if terminal is None:
            raise RuntimeError(
                "no usable terminal to open a JARVIS worker window: install one of "
                "gnome-terminal, konsole, xterm, or tmux (headless fallback)"
            )
        kind, term_path = terminal
        argv = build_unix_launch_argv(
            terminal=kind, term_path=term_path, root=root, command=command, session_hint=label, env=env
        )
    # start_new_session=True is the Unix analog of CREATE_BREAKAWAY_FROM_JOB: the
    # launched terminal (or tmux server) leads its own session and process group,
    # so killing the parent's group never reaches a sibling worker window.
    return subprocess.Popen(argv, cwd=root, env=env, start_new_session=True)


def launch_visible_jarvis_window(
    env: dict[str, str],
    *,
    provider: str | None = None,
    model: str | None = None,
    label: str | None = None,
) -> subprocess.Popen[Any]:
    root = str(repo_root())
    platform = _current_platform()
    if platform == "windows":
        return _launch_windows_window(
            env, root=root, provider=provider, model=model, label=label
        )
    return _launch_unix_window(
        env, platform=platform, root=root, provider=provider, model=model, label=label
    )


def next_worker_label() -> str:
    """Default name for a spawned window: worker1, worker2, ... (smallest free
    slot among known runtimes). Model-invented descriptive labels grew too
    long to address; short sequential names keep "send this to worker1"
    natural for the user.
    """
    used: set[int] = set()
    for path in _runtime_files():
        record = read_runtime(path) or {}
        text = str(record.get("label") or "").strip().lower()
        if not (text.startswith("worker") and text[6:].isdigit()):
            continue
        # Only a LIVE window reserves its slot. A worker that closed without
        # cleanup (X-close / crash) leaves a stale runtime file behind; counting
        # it would push every future worker's number up by one (a ghost worker1
        # makes the next spawn "worker2"). Mirror the liveness gate that
        # _candidate_runtime already uses so dead windows free their slot.
        try:
            pid = int(record.get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if pid <= 0 or not pairing._pid_alive(pid):
            continue
        used.add(int(text[6:]))
    n = 1
    while n in used:
        n += 1
    return f"worker{n}"


def spawn_window(
    *,
    timeout_seconds: float = DEFAULT_SPAWN_TIMEOUT_SECONDS,
    provider: str | None = None,
    model: str | None = None,
    label: str | None = None,
    launcher: Callable[[dict[str, str]], Any] | None = None,
    waiter: Callable[[str | None, set[Path], float], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    current_pair8 = pairing.current_pair_id()[:8] or None
    known_paths = _runtime_files()
    label = sanitize_window_label(label)
    env = child_spawn_env(provider=provider, model=model)
    launched = launcher(env) if launcher else launch_visible_jarvis_window(
        env,
        provider=provider,
        model=model,
        label=label,
    )
    wait = waiter or (lambda current, known, timeout: wait_for_spawned_runtime(
        current_pair8=current,
        known_paths=known,
        timeout_seconds=timeout,
    ))
    runtime = wait(current_pair8, known_paths, timeout_seconds)
    if runtime is None:
        pid = getattr(launched, "pid", None)
        raise TimeoutError(
            f"spawned JARVIS window but no new runtime file appeared within {timeout_seconds:.1f}s (pid={pid}); "
            "the window may still be booting and register late — run list_windows before spawning again, do not double-spawn"
        )
    if label:
        runtime.update(set_runtime_label(str(runtime["pair8"]), label))
    return runtime
