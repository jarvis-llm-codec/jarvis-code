from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REF_RE = re.compile(r"^[0-9a-f]{24}$")
DEFAULT_SESSION_ID = "default"
DEFAULT_SESSION_CAP_BYTES = 100 * 1024 * 1024
DEFAULT_GLOBAL_CAP_BYTES = 512 * 1024 * 1024
GC_WRITE_INTERVAL = 50
GC_YOUNG_ENTRY_GRACE = timedelta(hours=1)

_lock = threading.Lock()
_write_count = 0


class EvidenceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def store_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    original_text = payload.get("original_text")
    if not isinstance(original_text, str):
        raise EvidenceError("original_text must be a string", 400)

    session_id = normalize_session_id(payload.get("session_id"))
    original_bytes = original_text.encode("utf-8")
    full_sha = hashlib.sha256(original_bytes).hexdigest()
    ref = ref_for_bytes(original_bytes)
    now = utc_now()
    root = evidence_root()
    session_dir = root / "sessions" / session_id
    blob_path = blob_path_for_ref(session_dir, ref)
    index_path = index_path_for_session(session_dir)

    with _lock:
        session_dir.mkdir(parents=True, exist_ok=True)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        entries = read_index(index_path)
        existing = find_entry(entries, ref)
        if existing:
            if existing.get("full_sha256") == full_sha and int(existing.get("original_bytes") or -1) == len(original_bytes):
                return {"ok": True, "ref": ref, "stored": True, "dedup": True}
            return {
                "ok": False,
                "ref": ref,
                "stored": False,
                "error": "evidence ref collision",
            }

        blob_path.write_text(original_text, encoding="utf-8", newline="\n")
        entry = build_entry(payload, session_id, ref, full_sha, len(original_bytes), now, blob_path)
        append_index_entry(index_path, entry)
        maybe_run_write_gc(root, now)
        return {"ok": True, "ref": ref, "stored": True, "dedup": False}


def retrieve_evidence(ref: str, start_line: int | None = None, end_line: int | None = None) -> dict[str, Any]:
    validate_ref(ref)
    validate_line_range(start_line, end_line)
    root = evidence_root()

    with _lock:
        found = find_ref(root, ref)
        if not found:
            raise EvidenceError("evidence ref not found", 404)
        index_path, entry = found
        session_dir = index_path.parent
        blob_path = blob_path_from_entry(session_dir, entry)
        if not blob_path.is_file():
            raise EvidenceError("evidence blob not found", 404)
        # newline="" disables universal-newline translation so CRLF originals
        # round-trip byte-exact (acceptance: retrieve must restore the original).
        with blob_path.open("r", encoding="utf-8", newline="") as handle:
            content = handle.read()
        ranged = slice_lines(content, start_line, end_line)
        entry["last_retrieved_at"] = utc_now().isoformat()
        rewrite_index_entry(index_path, entry)
        return {
            "ok": True,
            "ref": ref,
            "metadata": entry,
            "content": ranged,
            "encoding": entry.get("encoding", "utf-8"),
            "had_bom": bool(entry.get("had_bom", False)),
        }


def run_evidence_gc(
    *,
    root: Path | None = None,
    now: datetime | None = None,
    session_cap_bytes: int = DEFAULT_SESSION_CAP_BYTES,
    global_cap_bytes: int = DEFAULT_GLOBAL_CAP_BYTES,
    active_session_id: str | None = None,
) -> dict[str, Any]:
    root = root or evidence_root()
    now = now or utc_now()
    if not root.exists():
        return {"ok": True, "deleted": 0, "bytes_deleted": 0}

    deleted = 0
    bytes_deleted = 0
    for session_dir in sessions_root(root).glob("*"):
        if not session_dir.is_dir():
            continue
        result = trim_session(session_dir, now, session_cap_bytes, active_session_id)
        deleted += result["deleted"]
        bytes_deleted += result["bytes_deleted"]

    result = trim_global(root, now, global_cap_bytes, active_session_id)
    deleted += result["deleted"]
    bytes_deleted += result["bytes_deleted"]
    return {"ok": True, "deleted": deleted, "bytes_deleted": bytes_deleted}


def evidence_root() -> Path:
    configured = os.environ.get("JARVIS_EVIDENCE_STORE")
    if configured and configured.strip():
        return Path(configured).expanduser()
    return Path.home() / ".jarvis-code" / "evidence-store"


def normalize_session_id(value: Any) -> str:
    raw = str(value or "").strip() or DEFAULT_SESSION_ID
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._-")
    return (safe or DEFAULT_SESSION_ID)[:120]


def ref_for_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def validate_ref(ref: str) -> None:
    if not REF_RE.fullmatch(ref):
        raise EvidenceError("invalid evidence ref", 400)


def validate_line_range(start_line: int | None, end_line: int | None) -> None:
    if start_line is not None and start_line < 1:
        raise EvidenceError("start_line must be >= 1", 400)
    if end_line is not None and end_line < 1:
        raise EvidenceError("end_line must be >= 1", 400)
    if start_line is not None and end_line is not None and end_line < start_line:
        raise EvidenceError("end_line must be >= start_line", 400)


def build_entry(
    payload: dict[str, Any],
    session_id: str,
    ref: str,
    full_sha: str,
    original_bytes: int,
    now: datetime,
    blob_path: Path,
) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    original_text = payload.get("original_text") if isinstance(payload.get("original_text"), str) else ""
    compressed_text = payload.get("compressed_text") if isinstance(payload.get("compressed_text"), str) else ""
    expires_at = payload.get("expires_at")
    return {
        "ref": ref,
        "full_sha256": full_sha,
        "hash_algo": "sha256",
        "created_at": now.isoformat(),
        "expires_at": expires_at if isinstance(expires_at, str) and expires_at.strip() else None,
        "last_retrieved_at": None,
        "session_id": session_id,
        "conversation_id": as_optional_str(payload.get("conversation_id")),
        "turn_key": as_optional_str(payload.get("turn_key")),
        "provider_call_id": as_optional_str(payload.get("provider_call_id")),
        "tool_call_id": as_optional_str(payload.get("tool_call_id")),
        "tool_name": as_optional_str(payload.get("tool_name")),
        "kind": as_optional_str(payload.get("kind")) or "unknown",
        "cwd": as_optional_str(metadata.get("cwd")),
        "command": as_optional_str(metadata.get("command")),
        "exit_code": metadata.get("exit_code"),
        "source_path": as_optional_str(metadata.get("source_path")),
        "source_paths": as_optional_str_list(metadata.get("source_paths")),
        "original_bytes": original_bytes,
        "original_lines": line_count(original_text),
        "original_tokens_est": as_optional_int(payload.get("original_tokens_est")),
        "compressed_bytes": len(compressed_text.encode("utf-8")) if compressed_text else 0,
        "compressed_tokens_est": as_optional_int(payload.get("compressed_tokens_est")),
        "kept_count": as_optional_int(payload.get("kept_count")) or 0,
        "dropped_count": as_optional_int(payload.get("dropped_count")) or 0,
        "encoding": "utf-8",
        "had_bom": False,
        "blob_path": str(blob_path.relative_to(blob_path.parents[1])),
    }


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def as_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_optional_str_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    items = [text for item in value if (text := as_optional_str(item))]
    return items or None


def line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines()) or 1


def sessions_root(root: Path) -> Path:
    return root / "sessions"


def index_path_for_session(session_dir: Path) -> Path:
    return session_dir / "index.jsonl"


def blob_path_for_ref(session_dir: Path, ref: str) -> Path:
    validate_ref(ref)
    return session_dir / "blobs" / f"{ref}.txt"


def blob_path_from_entry(session_dir: Path, entry: dict[str, Any]) -> Path:
    ref = str(entry.get("ref") or "")
    return blob_path_for_ref(session_dir, ref)


def read_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def write_index(index_path: Path, entries: list[dict[str, Any]]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{json.dumps(entry, ensure_ascii=False, sort_keys=True)}\n" for entry in entries)
    index_path.write_text(text, encoding="utf-8", newline="\n")


def append_index_entry(index_path: Path, entry: dict[str, Any]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{json.dumps(entry, ensure_ascii=False, sort_keys=True)}\n")


def rewrite_index_entry(index_path: Path, entry: dict[str, Any]) -> None:
    entries = read_index(index_path)
    ref = entry.get("ref")
    updated = False
    for index, existing in enumerate(entries):
        if existing.get("ref") == ref:
            entries[index] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)
    write_index(index_path, entries)


def find_entry(entries: list[dict[str, Any]], ref: str) -> dict[str, Any] | None:
    for entry in entries:
        if entry.get("ref") == ref:
            return entry
    return None


def find_ref(root: Path, ref: str) -> tuple[Path, dict[str, Any]] | None:
    sessions = sessions_root(root)
    if not sessions.exists():
        return None
    for session_dir in sorted(path for path in sessions.glob("*") if path.is_dir()):
        index_path = index_path_for_session(session_dir)
        entry = find_entry(read_index(index_path), ref)
        if entry:
            return index_path, entry
    return None


def slice_lines(content: str, start_line: int | None, end_line: int | None) -> str:
    if start_line is None and end_line is None:
        return content
    lines = content.splitlines(keepends=True)
    start = max(0, (start_line or 1) - 1)
    end = end_line if end_line is not None else len(lines)
    selected = lines[start:end]
    if selected:
        selected[-1] = selected[-1].rstrip("\r\n")
    return "".join(selected)


def maybe_run_write_gc(root: Path, now: datetime) -> None:
    global _write_count
    _write_count += 1
    if _write_count % GC_WRITE_INTERVAL == 0:
        run_evidence_gc(root=root, now=now)


def trim_session(
    session_dir: Path,
    now: datetime,
    session_cap_bytes: int,
    active_session_id: str | None,
) -> dict[str, int]:
    entries = read_index(index_path_for_session(session_dir))
    total = sum(entry_size(session_dir, entry) for entry in entries)
    if total <= session_cap_bytes:
        return {"deleted": 0, "bytes_deleted": 0}
    return delete_until_under_cap(session_dir, entries, total, session_cap_bytes, now, active_session_id)


def trim_global(
    root: Path,
    now: datetime,
    global_cap_bytes: int,
    active_session_id: str | None,
) -> dict[str, int]:
    all_entries: list[tuple[Path, dict[str, Any]]] = []
    total = 0
    for session_dir in sorted(path for path in sessions_root(root).glob("*") if path.is_dir()):
        for entry in read_index(index_path_for_session(session_dir)):
            all_entries.append((session_dir, entry))
            total += entry_size(session_dir, entry)
    if total <= global_cap_bytes:
        return {"deleted": 0, "bytes_deleted": 0}

    deleted = 0
    bytes_deleted = 0
    for session_dir, entry in sorted_gc_candidates(all_entries, now, active_session_id):
        if total <= global_cap_bytes:
            break
        size = delete_entry(session_dir, entry)
        if size <= 0:
            continue
        total -= size
        deleted += 1
        bytes_deleted += size
    return {"deleted": deleted, "bytes_deleted": bytes_deleted}


def delete_until_under_cap(
    session_dir: Path,
    entries: list[dict[str, Any]],
    total: int,
    cap: int,
    now: datetime,
    active_session_id: str | None,
) -> dict[str, int]:
    deleted = 0
    bytes_deleted = 0
    candidates = [(session_dir, entry) for entry in entries]
    for _, entry in sorted_gc_candidates(candidates, now, active_session_id):
        if total <= cap:
            break
        size = delete_entry(session_dir, entry)
        if size <= 0:
            continue
        total -= size
        deleted += 1
        bytes_deleted += size
    return {"deleted": deleted, "bytes_deleted": bytes_deleted}


def sorted_gc_candidates(
    candidates: list[tuple[Path, dict[str, Any]]],
    now: datetime,
    active_session_id: str | None,
) -> list[tuple[Path, dict[str, Any]]]:
    old = [(session_dir, entry) for session_dir, entry in candidates if not is_young_entry(entry, now)]
    return sorted(
        old,
        key=lambda item: (
            0 if is_expired(item[1], now) else 1,
            1 if active_session_id and item[1].get("session_id") == active_session_id else 0,
            timestamp_sort_value(item[1].get("last_retrieved_at") or item[1].get("created_at")),
        ),
    )


def entry_size(session_dir: Path, entry: dict[str, Any]) -> int:
    path = blob_path_from_entry(session_dir, entry)
    try:
        return path.stat().st_size
    except OSError:
        return int(entry.get("original_bytes") or 0)


def delete_entry(session_dir: Path, entry: dict[str, Any]) -> int:
    index_path = index_path_for_session(session_dir)
    entries = [item for item in read_index(index_path) if item.get("ref") != entry.get("ref")]
    path = blob_path_from_entry(session_dir, entry)
    size = entry_size(session_dir, entry)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return 0
    write_index(index_path, entries)
    return size


def is_young_entry(entry: dict[str, Any], now: datetime) -> bool:
    created = parse_time(entry.get("created_at"))
    return created is not None and now - created < GC_YOUNG_ENTRY_GRACE


def is_expired(entry: dict[str, Any], now: datetime) -> bool:
    expires = parse_time(entry.get("expires_at"))
    return expires is not None and expires <= now


def timestamp_sort_value(value: Any) -> float:
    parsed = parse_time(value)
    return parsed.timestamp() if parsed else 0.0


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)
