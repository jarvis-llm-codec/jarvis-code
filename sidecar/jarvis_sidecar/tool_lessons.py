"""Cross-window memory for repeated tool failures ("lesson packs").

Phase 1 wires the bash domain: every failed shell command is fingerprinted
(executable + error signature + inner command head), counted in a
machine-global store, and the first success in the same turn after a failure
becomes that fingerprint's known working alternative — deterministically, no
LLM in the loop. Nothing is ever injected into prompts up front; the
/tool_lesson/observe endpoint returns a short hint only when a *failure*
matches a known lesson, so the chat model stops retrying shapes that always
fail on this machine while staying free to ignore the advice.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

MAX_LESSONS = 200
CMD_HEAD_LEN = 160
LOCK_STALE_SECONDS = 10.0
LOCK_TIMEOUT_SECONDS = 2.0

# Turn-scoped state (per sidecar process == per window): failures awaiting a
# working alternative, and fingerprints already hinted this turn.
_pending_failures: dict[str, list[str]] = {}
_hinted: dict[str, set[str]] = {}
_turn_state_cap = 8  # keep only the most recent turns' scratch state


def lessons_path() -> Path:
    override = os.environ.get("JARVIS_TOOL_LESSONS_PATH")
    if override:
        return Path(override).expanduser()
    return Path("~/.jarvis-code/tool-lessons.json").expanduser()


_ERROR_CLASS_PATTERNS: list[tuple[re.Pattern[str], str | None]] = [
    (re.compile(r"FullyQualifiedErrorId\s*:\s*([\w.+-]+)", re.IGNORECASE), None),
    (re.compile(r"ParserError", re.IGNORECASE), "parser_error"),
    (re.compile(r"command not found", re.IGNORECASE), "command_not_found"),
    (re.compile(r"is not recognized as", re.IGNORECASE), "not_recognized"),
    (re.compile(r"No such file or directory", re.IGNORECASE), "no_such_file"),
    (re.compile(r"Permission denied", re.IGNORECASE), "permission_denied"),
    (re.compile(r"SyntaxError", re.IGNORECASE), "syntax_error"),
]


def error_class(output: str | None) -> str:
    text = str(output or "")
    for pattern, label in _ERROR_CLASS_PATTERNS:
        match = pattern.search(text)
        if match:
            return label if label is not None else match.group(1)
    return "nonzero_exit"


def executable_head(command: str) -> str:
    text = str(command or "").strip().strip("\"'")
    if not text:
        return "unknown"
    head = re.split(r"\s+", text, maxsplit=1)[0]
    head = head.strip("\"'").replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    return head.lower() or "unknown"


_INNER_COMMAND_RE = re.compile(r"-Command\s+([\"'])(.+?)\1", re.IGNORECASE | re.DOTALL)


def inner_head(command: str) -> str | None:
    """First word inside powershell -Command "..." — the cmdlet that actually
    ran, which separates Select-String lessons from Get-Content lessons."""
    match = _INNER_COMMAND_RE.search(str(command or ""))
    if not match:
        return None
    inner = match.group(2).strip()
    if not inner:
        return None
    return re.split(r"[\s({|;]+", inner, maxsplit=1)[0].lower() or None


def fingerprint(command: str, output: str | None) -> str:
    parts = [executable_head(command), error_class(output)]
    inner = inner_head(command)
    if inner:
        parts.append(inner)
    return "|".join(parts)


def _cmd_head(command: str) -> str:
    flat = re.sub(r"\s+", " ", str(command or "").strip())
    return flat[:CMD_HEAD_LEN]


def _acquire_lock(path: Path) -> Path | None:
    lock = path.with_suffix(path.suffix + ".lock")
    deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                    lock.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)
        except OSError:
            return None


def _release_lock(lock: Path | None) -> None:
    if lock is None:
        return
    try:
        lock.unlink(missing_ok=True)
    except OSError:
        pass


def _load_store(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _save_store(path: Path, store: dict[str, Any]) -> None:
    if len(store) > MAX_LESSONS:
        # Size-capped by design: evict the least recently seen lessons.
        ordered = sorted(store.items(), key=lambda item: item[1].get("last_seen") or 0)
        for key, _record in ordered[: len(store) - MAX_LESSONS]:
            store.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=0), encoding="utf-8")
    os.replace(tmp, path)


def _turn_key(pair8: str | None, turn_id: Any) -> str:
    return f"{pair8 or 'pair'}::{turn_id if turn_id not in (None, '') else 'turn'}"


def _prune_turn_state(active_key: str) -> None:
    for state in (_pending_failures, _hinted):
        if active_key not in state and len(state) >= _turn_state_cap:
            for stale in list(state)[: len(state) - _turn_state_cap + 1]:
                state.pop(stale, None)


def _hint_text(record: dict[str, Any]) -> str:
    shape = str(record.get("fingerprint") or "").replace("|", " · ")
    count = int(record.get("count") or 0)
    alternative = str(record.get("alternative") or "").strip()
    base = f"[lesson x{count}] this command shape has failed before on this machine ({shape})."
    if alternative:
        return f"{base} Known working alternative: {alternative}"
    return f"{base} No known working alternative yet — change the approach instead of retrying the same shape."


def observe(
    *,
    tool: str,
    command: str,
    is_error: bool,
    output_head: str | None = None,
    pair8: str | None = None,
    turn_id: Any = None,
) -> dict[str, Any]:
    """Record one tool outcome; on a repeat failure return an advisory hint."""
    command = str(command or "")
    if not command.strip():
        return {"ok": True}
    key = _turn_key(pair8, turn_id)
    _prune_turn_state(key)
    path = lessons_path()

    if not is_error:
        pending = _pending_failures.pop(key, [])
        if not pending:
            return {"ok": True}
        lock = _acquire_lock(path)
        try:
            store = _load_store(path)
            paired = 0
            for fp in pending:
                record = store.get(fp)
                if isinstance(record, dict) and not record.get("alternative"):
                    record["alternative"] = _cmd_head(command)
                    record["alternative_recorded_at"] = int(time.time())
                    paired += 1
            if paired:
                _save_store(path, store)
        finally:
            _release_lock(lock)
        return {"ok": True, "paired": paired}

    fp = fingerprint(command, output_head)
    lock = _acquire_lock(path)
    try:
        store = _load_store(path)
        record = store.get(fp)
        if not isinstance(record, dict):
            record = {
                "tool": str(tool or "bash"),
                "fingerprint": fp,
                "count": 0,
                "first_seen": int(time.time()),
                "alternative": None,
            }
        prior_count = int(record.get("count") or 0)
        record["count"] = prior_count + 1
        record["last_seen"] = int(time.time())
        record["cmd_head"] = _cmd_head(command)
        store[fp] = record
        _save_store(path, store)
    finally:
        _release_lock(lock)

    _pending_failures.setdefault(key, [])
    if fp not in _pending_failures[key]:
        _pending_failures[key].append(fp)

    if prior_count < 1:
        return {"ok": True, "fingerprint": fp, "count": record["count"]}
    hinted = _hinted.setdefault(key, set())
    if fp in hinted:
        return {"ok": True, "fingerprint": fp, "count": record["count"]}
    hinted.add(fp)
    return {"ok": True, "fingerprint": fp, "count": record["count"], "hint": _hint_text(record)}
