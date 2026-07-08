"""Interactive selector for JARVIS Code chat + encoder models.

Reads scripts/llm_catalog.yaml manifest, fetches live model lists from each
provider that has an API key in env, asks the user (via arrow keys + Enter)
to pick chat and encoder, then writes:
    - active config.yaml   -> roles.chat, roles.subagent, roles.encoder
    - pi-agent/models.json -> provider/model entries (so Pi's resolver finds them)

Invoked via scripts/llmsetting.ps1, but works standalone too. All the
non-UI logic (catalog load, model fetch, config write) lives in
sidecar/jarvis_sidecar/llm_setting.py so Pi's /model-setting slash command
can call the same code via the sidecar.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sidecar"))

try:
    from jarvis_sidecar.llm_setting import apply_picks, fetch_models, load_catalog, provider_supports_model_setting
    from jarvis_sidecar.config import load_credentials_into_env
except ImportError as e:
    print(f"ERROR: cannot import jarvis_sidecar.llm_setting ({e}). Run sidecar bootstrap first.", file=sys.stderr)
    sys.exit(2)

_IS_WINDOWS = platform.system() == "Windows"

DIM = "\x1b[2m"
INVERSE = "\x1b[7m"
RESET = "\x1b[0m"
CURSOR_HOME = "\x1b[H"
CLEAR_EOL = "\x1b[K"
CLEAR_EOS = "\x1b[J"
PAGE_SIZE = 15  # viewport height for arrow_select when the list is longer
BACK = "__back__"  # sentinel returned by selectors when user navigates back


def _enable_ansi() -> None:
    """Best-effort enable ANSI escape processing on Windows console.
    Modern Windows Terminal supports VT by default; classic conhost needs this."""
    if not _IS_WINDOWS:
        return
    try:
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except OSError:
        pass


def _clear_screen() -> None:
    os.system("cls" if _IS_WINDOWS else "clear")


def _read_key() -> str:
    """Return one of: 'up', 'down', 'left', 'right', 'enter', 'esc', 'other'."""
    if _IS_WINDOWS:
        import msvcrt  # noqa: PLC0415
        ch = msvcrt.getch()
        if ch in (b'\xe0', b'\x00'):
            ch2 = msvcrt.getch()
            if ch2 == b'H':
                return 'up'
            if ch2 == b'P':
                return 'down'
            if ch2 == b'K':
                return 'left'
            if ch2 == b'M':
                return 'right'
            return 'other'
        if ch == b'\r':
            return 'enter'
        if ch in (b'\x1b', b'\x03'):
            return 'esc'
        return 'other'

    # POSIX (Linux/macOS): put the terminal in raw mode, read one keystroke,
    # and decode ANSI arrow sequences (ESC '[' 'A'..'D'). Without this the
    # menu is unusable on Linux/macOS — every prompt would auto-confirm.
    import select  # noqa: PLC0415
    import termios  # noqa: PLC0415
    import tty  # noqa: PLC0415

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = os.read(fd, 1)
        if ch == b'\x1b':
            # Distinguish a lone ESC (cancel) from an arrow escape sequence.
            # Arrow bytes arrive together; a bare ESC has nothing following it.
            ready, _, _ = select.select([fd], [], [], 0.05)
            if not ready:
                return 'esc'
            if os.read(fd, 1) != b'[':
                return 'esc'
            code = os.read(fd, 1)
            return {b'A': 'up', b'B': 'down', b'C': 'right', b'D': 'left'}.get(code, 'other')
        if ch in (b'\r', b'\n'):
            return 'enter'
        if ch == b'\x03':  # Ctrl-C
            return 'esc'
        return 'other'
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def arrow_select(title: str, items: list[tuple[str, bool]], start_idx: int = 0, allow_back: bool = False) -> int | None:
    """Arrow-key menu. Returns chosen index >= 0, -1 if user pressed Left
    (back), or None on Esc (cancel). Arrow navigation skips unselectable items.

    Long lists scroll within a fixed-height viewport (PAGE_SIZE rows). The
    current row is highlighted with inverse video so the cursor is obvious
    even when the screen redraws."""
    selectable_idxs = [i for i, (_, ok) in enumerate(items) if ok]
    if not selectable_idxs:
        print(title)
        for label, _ in items:
            print(f"    {label}")
        print("  (no selectable items)")
        return None

    cur = start_idx if start_idx in selectable_idxs else selectable_idxs[0]
    page = min(PAGE_SIZE, len(items))
    viewport_top = 0
    last_line_count = 0
    first_render = True

    def adjust_viewport() -> None:
        nonlocal viewport_top
        if cur < viewport_top:
            viewport_top = cur
        elif cur >= viewport_top + page:
            viewport_top = cur - page + 1
        max_top = max(0, len(items) - page)
        viewport_top = max(0, min(viewport_top, max_top))

    def render() -> None:
        nonlocal last_line_count, first_render
        if first_render:
            _clear_screen()
            first_render = False
        else:
            sys.stdout.write(CURSOR_HOME)

        lines: list[str] = []
        for sub in title.splitlines() or [""]:
            lines.append(sub)
        if allow_back:
            lines.append("  (Up/Down to move, Enter to pick, Left to go back, Esc to cancel)")
        else:
            lines.append("  (Up/Down to move, Enter to pick, Esc to cancel)")
        lines.append("")

        end = min(viewport_top + page, len(items))
        if viewport_top > 0:
            lines.append(f"  {DIM}^ {viewport_top} more above{RESET}")
        else:
            lines.append("")

        for i in range(viewport_top, end):
            label, ok = items[i]
            marker = ">" if i == cur else " "
            if i == cur:
                lines.append(f"{INVERSE}  {marker} {label}{RESET}")
            elif ok:
                lines.append(f"  {marker} {label}")
            else:
                lines.append(f"  {marker} {DIM}{label}{RESET}")

        remaining_below = len(items) - end
        if remaining_below > 0:
            lines.append(f"  {DIM}v {remaining_below} more below{RESET}")
        else:
            lines.append("")

        for line in lines:
            sys.stdout.write(line + CLEAR_EOL + "\n")
        for _ in range(max(0, last_line_count - len(lines))):
            sys.stdout.write(CLEAR_EOL + "\n")
        sys.stdout.write(CLEAR_EOS)
        sys.stdout.flush()
        last_line_count = len(lines)

    adjust_viewport()
    render()

    while True:
        key = _read_key()
        if key == 'up':
            j = selectable_idxs.index(cur)
            cur = selectable_idxs[(j - 1) % len(selectable_idxs)]
            adjust_viewport()
            render()
        elif key == 'down':
            j = selectable_idxs.index(cur)
            cur = selectable_idxs[(j + 1) % len(selectable_idxs)]
            adjust_viewport()
            render()
        elif key == 'left' and allow_back:
            return -1
        elif key == 'enter':
            return cur
        elif key == 'esc':
            return None


def _ordered_providers(catalog: dict[str, Any], role: str) -> list[str]:
    providers = list(catalog["providers"].keys())
    rec_provider = catalog.get("recommended", {}).get(role, {}).get("provider")
    if rec_provider and rec_provider in providers:
        providers.remove(rec_provider)
        providers.insert(0, rec_provider)
    return providers


def select_provider(catalog: dict[str, Any], fetched: dict[str, list[str] | None], role: str,
                    *, allow_back: bool = False, start_provider: str | None = None) -> str | None:
    """Returns provider id, BACK sentinel, or None for cancel."""
    providers = catalog["providers"]
    rec_provider = catalog.get("recommended", {}).get(role, {}).get("provider")
    ordered = _ordered_providers(catalog, role)

    items: list[tuple[str, bool]] = []
    for pid in ordered:
        cfg = providers[pid]
        label = cfg.get("label", pid)
        if cfg.get("enabled") is False:
            note = cfg.get("note", "not available")
            items.append((f"x {label}  ({note})", False))
        elif fetched.get(pid) is None:
            items.append((f"x {label}  (no API key — set {cfg.get('auth_env', '?')} in env)", False))
        elif pid == rec_provider:
            items.append((f"* {label}  (recommended)", True))
        else:
            items.append((f"  {label}", True))

    start_idx = ordered.index(start_provider) if start_provider in ordered else 0
    idx = arrow_select(
        f"\n=== Select {role.upper()} provider ===",
        items, start_idx=start_idx, allow_back=allow_back,
    )
    if idx is None:
        return None
    if idx == -1:
        return BACK
    return ordered[idx]


def select_model(provider_id: str, models: list[str], recommended_model: str | None,
                 *, allow_back: bool = True, start_model: str | None = None) -> str | None:
    """Returns model id, BACK sentinel, or None for cancel."""
    ordered = list(models)
    if recommended_model and recommended_model in ordered:
        ordered.remove(recommended_model)
        ordered.insert(0, recommended_model)

    items: list[tuple[str, bool]] = []
    for m in ordered:
        if m == recommended_model:
            items.append((f"* {m}  (recommended)", True))
        else:
            items.append((f"  {m}", True))

    start_idx = ordered.index(start_model) if start_model in ordered else 0
    idx = arrow_select(
        f"\n--- Models on {provider_id} ({len(ordered)} available) ---",
        items, start_idx=start_idx, allow_back=allow_back,
    )
    if idx is None:
        return None
    if idx == -1:
        return BACK
    return ordered[idx]


def _rec_model(catalog: dict[str, Any], role: str, provider: str) -> str | None:
    rec = catalog.get("recommended", {}).get(role, {})
    return rec.get("model") if rec.get("provider") == provider else None


def main() -> int:
    _enable_ansi()
    _clear_screen()
    print("== JARVIS Code LLM setting ==")
    catalog = load_catalog()
    catalog["providers"] = {
        pid: cfg for pid, cfg in catalog["providers"].items()
        if provider_supports_model_setting(cfg)
    }
    load_credentials_into_env()

    print("\nFetching available models from providers with keys present...")
    fetched: dict[str, list[str] | None] = {}
    for pid, cfg in catalog["providers"].items():
        if cfg.get("enabled") is False:
            fetched[pid] = None
            continue
        models = fetch_models(pid, cfg)
        fetched[pid] = models
        if models:
            print(f"  {pid}: {len(models)} models")

    chat_p: str | None = None
    chat_m: str | None = None
    enc_p: str | None = None
    enc_m: str | None = None

    state = "chat_provider"
    while state != "done":
        if state == "chat_provider":
            res = select_provider(catalog, fetched, "chat", allow_back=False, start_provider=chat_p)
            if res is None:
                print("Cancelled.")
                return 1
            chat_p = res
            state = "chat_model"
        elif state == "chat_model":
            assert chat_p is not None
            res = select_model(chat_p, fetched[chat_p] or [], _rec_model(catalog, "chat", chat_p),
                               allow_back=True, start_model=chat_m)
            if res is None:
                print("Cancelled.")
                return 1
            if res == BACK:
                state = "chat_provider"
                continue
            chat_m = res
            state = "encoder_provider"
        elif state == "encoder_provider":
            res = select_provider(catalog, fetched, "encoder", allow_back=True, start_provider=enc_p)
            if res is None:
                print("Cancelled.")
                return 1
            if res == BACK:
                state = "chat_model"
                continue
            enc_p = res
            state = "encoder_model"
        elif state == "encoder_model":
            assert enc_p is not None
            res = select_model(enc_p, fetched[enc_p] or [], _rec_model(catalog, "encoder", enc_p),
                               allow_back=True, start_model=enc_m)
            if res is None:
                print("Cancelled.")
                return 1
            if res == BACK:
                state = "encoder_provider"
                continue
            enc_m = res
            state = "done"

    assert chat_p and chat_m and enc_p and enc_m
    print("\n=== Writing JARVIS Code config ===")
    paths = apply_picks((chat_p, chat_m), (enc_p, enc_m), catalog=catalog)
    print(f"  routing  -> {paths['config_path']}")
    print(f"  models   -> {paths['models_json_path']}")

    print("\nDone.")
    print(f"  chat    = {chat_p}/{chat_m}")
    print(f"  encoder = {enc_p}/{enc_m}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
