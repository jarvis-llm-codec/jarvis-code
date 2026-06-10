from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .file_locks import cross_process_file_lock

MAX_WINDOW_LABEL_CHARS = 32
PAIR8_RE = re.compile(r"^[A-Za-z0-9]{8}$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return repo_root() / "data"


def normalize_pair8(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text if PAIR8_RE.fullmatch(text) else None


def sanitize_window_label(value: str | None) -> str | None:
    text = "".join(ch for ch in str(value or "") if ord(ch) >= 32 and ord(ch) != 127).strip()
    if not text:
        return None
    return text[:MAX_WINDOW_LABEL_CHARS]


def pair8_from_pair_id(pair_id: str | None) -> str | None:
    normalized = "".join(ch for ch in str(pair_id or "") if ch.isalnum())
    return normalized[:8] if len(normalized) >= 8 else None


def runtime_path_for_pair8(pair8: str) -> Path:
    return data_dir() / f"sidecar-runtime-{pair8}.json"


def read_runtime(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def runtime_files() -> set[Path]:
    root = data_dir()
    if not root.exists():
        return set()
    return {path.resolve() for path in root.glob("sidecar-runtime-*.json") if path.is_file()}


def runtime_record_for_pair8(pair8: str | None) -> dict[str, Any] | None:
    normalized = normalize_pair8(pair8)
    if not normalized:
        return None
    path = runtime_path_for_pair8(normalized)
    record = read_runtime(path)
    if record is None:
        return None
    record_pair8 = pair8_from_pair_id(str(record.get("pair_id") or ""))
    if record_pair8 and record_pair8 != normalized:
        return None
    return record


def runtime_label_for_pair8(pair8: str | None) -> str | None:
    record = runtime_record_for_pair8(pair8)
    if record is None:
        return None
    return sanitize_window_label(record.get("label"))


def set_runtime_label(pair8: str, label: str | None) -> dict[str, Any]:
    normalized = normalize_pair8(pair8)
    if not normalized:
        raise ValueError("pair8 is required")
    path = runtime_path_for_pair8(normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    next_label = sanitize_window_label(label)
    with cross_process_file_lock(path):
        record = read_runtime(path) or {}
        old_label = sanitize_window_label(record.get("label"))
        record["label"] = next_label
        _atomic_write_text_unlocked(path, json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "pair8": normalized,
        "old_label": old_label,
        "label": next_label,
        "runtime_path": str(path),
    }


def _atomic_write_text_unlocked(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        try:
            os.fchmod(fd, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace", newline="") as fh:
            fh.write(content)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_name, str(path))
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def resolve_live_label(label: str, windows: list[dict[str, Any]]) -> str | None:
    target = sanitize_window_label(label)
    if not target:
        raise ValueError("to_window is required")
    matches = [
        str(window.get("pair8") or "")
        for window in windows
        if window.get("alive") and sanitize_window_label(window.get("label")) == target
    ]
    matches = [pair for pair in matches if normalize_pair8(pair)]
    if len(matches) > 1:
        raise ValueError(f"ambiguous window label {target!r}: {', '.join(matches)}")
    return matches[0] if matches else None
