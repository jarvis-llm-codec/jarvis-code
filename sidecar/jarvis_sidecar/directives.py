from __future__ import annotations

import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from . import pairing
from .file_locks import cross_process_file_lock, locked_atomic_write_text
from .raw_store import _storage_root, normalize_origin_window
from .window_labels import runtime_label_for_pair8

DirectiveKind = Literal["directive", "report"]
VALID_DIRECTIVE_KINDS: set[str] = {"directive", "report"}
GAN_ROUND_CAP = 3
GAN_ID_RE = re.compile(r"^g_[0-9a-f]{8}$")
GAN_TERMINAL_STATUSES = {"agreed", "escalated"}
JOB_ID_RE = re.compile(r"^j_[0-9a-f]{8}$")
JOB_TERMINAL_STATUSES = {"done", "escalated"}
CRITIC_REVIEW_MARKERS = ("[CRITIC_REVIEW]", "[SECOND_EYES_REVIEW]")
CRITIC_MAIN_MARKERS = ("[CRITIC_MAIN]", "[SECOND_EYES_MAIN]")
CRITIC_HEAVY_MARKERS = ("[CRITIC_HEAVY]", "[SECOND_EYES_HEAVY]")
CRITIC_PLAN_READY_MARKERS = ("[CRITIC_PLAN_READY]", "[SECOND_EYES_PLAN_READY]")


class GANDirectiveError(ValueError):
    """Protocol-level GAN rejection that should surface as HTTP 409."""


class JobDirectiveError(ValueError):
    """Protocol-level job rejection that should surface as HTTP 409."""


def directives_path() -> Path:
    return _storage_root() / "directives.jsonl"


def directives_cursor_root() -> Path:
    return _storage_root() / "directives-cursors"


def normalize_directive_kind(kind: str | None) -> DirectiveKind:
    value = str(kind or "directive").strip().lower()
    if value not in VALID_DIRECTIVE_KINDS:
        raise ValueError(f"invalid directive kind: {kind!r}")
    return value  # type: ignore[return-value]


def normalize_directive_window(value: str | None) -> str | None:
    window = normalize_origin_window(value)
    if not window:
        return None
    return window


def job_cycle_cap() -> int:
    raw = str(os.environ.get("JARVIS_JOB_CYCLE_CAP", "3") or "").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 3
    return max(1, parsed)


def _cursor_path(to_window: str, kind: DirectiveKind | None) -> Path:
    suffix = kind or "all"
    return directives_cursor_root() / f"{to_window}.{suffix}.cursor"


def _legacy_cursor_path(to_window: str, kind: DirectiveKind | None) -> Path:
    suffix = f".{kind}" if kind else ""
    return pairing.conversation_root() / "_windows" / f"jhb-{to_window}" / f"directives{suffix}.cursor"


def _migrate_legacy_cursor(to_window: str, kind: DirectiveKind | None) -> Path:
    target = _cursor_path(to_window, kind)
    legacy = _legacy_cursor_path(to_window, kind)
    if target.exists() or not legacy.exists():
        return target
    try:
        value = _read_cursor(legacy)
        _write_cursor(target, value)
    except OSError:
        pass
    return target


def _read_cursor(path: Path) -> int:
    try:
        value = int(path.read_text(encoding="utf-8").strip() or "0")
    except Exception:
        return 0
    return max(0, value)


def _write_cursor(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    locked_atomic_write_text(path, f"{max(0, int(offset))}\n")


def _queue_stat(path: Path) -> tuple[int, int]:
    try:
        stat = path.stat()
    except OSError:
        return 0, 0
    return int(stat.st_size), int(getattr(stat, "st_mtime_ns", 0))


def append_directive(
    *,
    kind: str = "directive",
    from_window: str = "external",
    to_window: str | None = None,
    body: str,
    gan_target: str | None = None,
    issues_open: int | None = None,
    gan_status: str | None = None,
    job_target: str | None = None,
    job_status: str | None = None,
) -> dict[str, Any]:
    normalized_kind = normalize_directive_kind(kind)
    normalized_to = normalize_directive_window(to_window)
    normalized_from = normalize_directive_window(from_window) or "external"
    has_gan = bool(str(gan_target or "").strip())
    has_job = bool(str(job_target or "").strip())
    if has_gan and has_job:
        raise ValueError("directive cannot target both gan and job")
    target = directives_path()
    with cross_process_file_lock(target):
        if has_job:
            record = _build_job_record_locked(
                target,
                kind=normalized_kind,
                from_window=normalized_from,
                to_window=normalized_to,
                body=body,
                job_target=job_target or "",
                job_status=job_status,
            )
        elif has_gan:
            record = _build_gan_record_locked(
                target,
                kind=normalized_kind,
                from_window=normalized_from,
                to_window=normalized_to,
                body=body,
                gan_target=gan_target or "",
                issues_open=issues_open,
                gan_status=gan_status,
            )
        else:
            if not normalized_to:
                raise ValueError("to_window is required")
            record = _base_record(
                kind=normalized_kind,
                from_window=normalized_from,
                to_window=normalized_to,
                body=body,
            )
        _append_record_locked(target, record)
    return record


def _base_record(
    *,
    kind: DirectiveKind,
    from_window: str,
    to_window: str,
    body: str,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "ts": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "kind": kind,
        "from_window": from_window,
        "to_window": to_window,
        "body": str(body or ""),
    }


def _append_record_locked(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="") as fh:
        fh.write(line)
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def _new_gan_id() -> str:
    return f"g_{uuid.uuid4().hex[:8]}"


def _new_job_id() -> str:
    return f"j_{uuid.uuid4().hex[:8]}"


def _normalize_gan_id(value: str | None) -> str:
    gan_id = str(value or "").strip().lower()
    if not GAN_ID_RE.match(gan_id):
        raise ValueError(f"invalid gan_id: {value!r}")
    return gan_id


def _normalize_job_id(value: str | None) -> str:
    job_id = str(value or "").strip().lower()
    if not JOB_ID_RE.match(job_id):
        raise ValueError(f"invalid job_id: {value!r}")
    return job_id


def _normalize_issues_open(value: int | None) -> int:
    if value is None:
        raise GANDirectiveError("issues_open is required for gan_send")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GANDirectiveError("issues_open must be an integer") from exc
    if parsed < 0:
        raise GANDirectiveError("issues_open must be >= 0")
    return parsed


def _normalize_terminal_status(value: str | None) -> str:
    status = str(value or "").strip().lower()
    if status not in GAN_TERMINAL_STATUSES:
        raise GANDirectiveError("gan_close requires status agreed or escalated")
    return status


def _normalize_job_status(value: str | None) -> str:
    status = str(value or "").strip().lower()
    if status not in JOB_TERMINAL_STATUSES:
        raise JobDirectiveError("job_close requires status done or escalated")
    return status


def _read_gan_rows_locked(path: Path, gan_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("rb") as fh:
        for line in fh:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            gan = record.get("gan")
            if isinstance(gan, dict) and str(gan.get("gan_id") or "").strip().lower() == gan_id:
                rows.append(record)
    return rows


def _read_job_rows_locked(path: Path, job_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("rb") as fh:
        for line in fh:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            job = record.get("job")
            if isinstance(job, dict) and str(job.get("job_id") or "").strip().lower() == job_id:
                rows.append(record)
    return rows


def _read_directive_rows_locked(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("rb") as fh:
        for line in fh:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(record, dict):
                rows.append(record)
    return rows


def _gan_is_terminal(rows: list[dict[str, Any]]) -> bool:
    return any(str((row.get("gan") or {}).get("status") or "").lower() in GAN_TERMINAL_STATUSES for row in rows)


def _job_is_terminal(rows: list[dict[str, Any]]) -> bool:
    return any(str((row.get("job") or {}).get("status") or "").lower() in JOB_TERMINAL_STATUSES for row in rows)


def _open_gan_between_locked(path: Path, window_a: str, window_b: str) -> str | None:
    """Return the gan_id of an open GAN between the two windows, if any.

    The recipient model must echo gan_id to continue a session; without this
    guard a forgetful model silently forks the debate into a second GAN and
    the round cap / convergence pressure never bind.
    """
    if not path.exists():
        return None
    first_by_id: dict[str, set[str]] = {}
    terminal: set[str] = set()
    with path.open("rb") as fh:
        for line in fh:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            gan = record.get("gan")
            if not isinstance(gan, dict):
                continue
            gan_id = str(gan.get("gan_id") or "").strip().lower()
            if not GAN_ID_RE.match(gan_id):
                continue
            if gan_id not in first_by_id:
                first_by_id[gan_id] = {
                    normalize_directive_window(record.get("from_window")) or "",
                    normalize_directive_window(record.get("to_window")) or "",
                }
            if str(gan.get("status") or "").lower() in GAN_TERMINAL_STATUSES:
                terminal.add(gan_id)
    pair = {window_a, window_b}
    for gan_id, participants in first_by_id.items():
        if gan_id not in terminal and participants == pair:
            return gan_id
    return None


def _open_job_between_locked(path: Path, window_a: str, window_b: str) -> str | None:
    """Return the job_id of an open job between the two windows, if any."""
    if not path.exists():
        return None
    first_by_id: dict[str, set[str]] = {}
    terminal: set[str] = set()
    with path.open("rb") as fh:
        for line in fh:
            try:
                record = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            job = record.get("job")
            if not isinstance(job, dict):
                continue
            job_id = str(job.get("job_id") or "").strip().lower()
            if not JOB_ID_RE.match(job_id):
                continue
            if job_id not in first_by_id:
                first_by_id[job_id] = {
                    normalize_directive_window(record.get("from_window")) or "",
                    normalize_directive_window(record.get("to_window")) or "",
                }
            if str(job.get("status") or "").lower() in JOB_TERMINAL_STATUSES:
                terminal.add(job_id)
    pair = {window_a, window_b}
    for job_id, participants in first_by_id.items():
        if job_id not in terminal and participants == pair:
            return job_id
    return None


def _gan_directive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("kind") or "") == "directive"
        and str((row.get("gan") or {}).get("status") or "open").lower() == "open"
    ]


def _gan_last_round(rows: list[dict[str, Any]]) -> int:
    rounds: list[int] = []
    for row in rows:
        try:
            rounds.append(int((row.get("gan") or {}).get("round") or 0))
        except (TypeError, ValueError):
            continue
    return max(rounds) if rounds else 1


def _job_dispatch_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("kind") or "") == "directive"
        and str((row.get("job") or {}).get("phase") or "").lower() == "dispatch"
        and str((row.get("job") or {}).get("status") or "open").lower() == "open"
    ]


def _job_last_cycle(rows: list[dict[str, Any]]) -> int:
    cycles: list[int] = []
    for row in rows:
        try:
            cycles.append(int((row.get("job") or {}).get("cycle") or 0))
        except (TypeError, ValueError):
            continue
    return max(cycles) if cycles else 1


def _gan_last_issues_open(rows: list[dict[str, Any]]) -> int:
    directives = _gan_directive_rows(rows)
    if not directives:
        return 0
    try:
        return int((directives[-1].get("gan") or {}).get("issues_open") or 0)
    except (TypeError, ValueError):
        return 0


def _gan_participants(rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not rows:
        raise GANDirectiveError("unknown gan_id")
    first = rows[0]
    worker = normalize_directive_window(first.get("from_window")) or ""
    destroyer = normalize_directive_window(first.get("to_window")) or ""
    if not worker or not destroyer:
        raise GANDirectiveError("gan participants are invalid")
    return worker, destroyer


def _job_participants(rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not rows:
        raise JobDirectiveError("unknown job_id")
    first = rows[0]
    orchestrator = normalize_directive_window(first.get("from_window")) or ""
    worker = normalize_directive_window(first.get("to_window")) or ""
    if not orchestrator or not worker:
        raise JobDirectiveError("job participants are invalid")
    return orchestrator, worker


def _gan_role_for_recipient(to_window: str, participants: tuple[str, str]) -> str:
    worker, destroyer = participants
    if to_window == worker:
        return "worker"
    if to_window == destroyer:
        return "destroyer"
    raise GANDirectiveError("gan participants are fixed")


def _job_role_for_recipient(to_window: str, participants: tuple[str, str]) -> str:
    orchestrator, worker = participants
    if to_window == orchestrator:
        return "orchestrator"
    if to_window == worker:
        return "worker"
    raise JobDirectiveError("job participants are fixed")


def _job_phase_for_sender(from_window: str, participants: tuple[str, str]) -> str:
    orchestrator, worker = participants
    if from_window == orchestrator:
        return "dispatch"
    if from_window == worker:
        return "review"
    raise JobDirectiveError("job participants are fixed")


def _infer_gan_close_target(from_window: str, participants: tuple[str, str]) -> str:
    worker, destroyer = participants
    if from_window == worker:
        return destroyer
    if from_window == destroyer:
        return worker
    raise GANDirectiveError("gan participants are fixed")


def _infer_job_close_target(from_window: str, participants: tuple[str, str]) -> str:
    orchestrator, worker = participants
    if from_window == orchestrator:
        return worker
    if from_window == worker:
        return orchestrator
    raise JobDirectiveError("job participants are fixed")


def _build_gan_record_locked(
    path: Path,
    *,
    kind: DirectiveKind,
    from_window: str,
    to_window: str | None,
    body: str,
    gan_target: str,
    issues_open: int | None,
    gan_status: str | None,
) -> dict[str, Any]:
    target = str(gan_target or "").strip().lower()
    if target == "new":
        if kind != "directive":
            raise GANDirectiveError("new gan must start with a directive")
        if not to_window:
            raise ValueError("to_window is required")
        if from_window == "external":
            raise GANDirectiveError("gan requires a source window")
        if to_window == from_window:
            raise GANDirectiveError("gan requires two distinct windows")
        existing = _open_gan_between_locked(path, from_window, to_window)
        if existing:
            raise GANDirectiveError(
                f"open gan {existing} already exists between these windows; "
                f"continue it by passing gan_id={existing} or close it first"
            )
        issue_count = _normalize_issues_open(issues_open)
        record = _base_record(kind=kind, from_window=from_window, to_window=to_window, body=body)
        record["gan"] = {
            "gan_id": _new_gan_id(),
            "round": 1,
            "role": "destroyer",
            "issues_open": issue_count,
            "status": "open",
        }
        return record

    gan_id = _normalize_gan_id(target)
    rows = _read_gan_rows_locked(path, gan_id)
    if not rows:
        raise GANDirectiveError("unknown gan_id")
    if _gan_is_terminal(rows):
        raise GANDirectiveError("gan is already terminal")
    participants = _gan_participants(rows)
    if from_window not in participants:
        raise GANDirectiveError("gan participants are fixed")

    if kind == "report":
        status = _normalize_terminal_status(gan_status)
        close_to = to_window or _infer_gan_close_target(from_window, participants)
        role = _gan_role_for_recipient(close_to, participants)
        record = _base_record(kind=kind, from_window=from_window, to_window=close_to, body=body)
        record["gan"] = {
            "gan_id": gan_id,
            "round": _gan_last_round(rows),
            "role": role,
            "issues_open": 0 if status == "agreed" else _gan_last_issues_open(rows),
            "status": status,
        }
        return record

    if not to_window:
        raise ValueError("to_window is required")
    role = _gan_role_for_recipient(to_window, participants)
    previous_directives = _gan_directive_rows(rows)
    round_number = len(previous_directives) + 1
    if round_number > GAN_ROUND_CAP:
        raise GANDirectiveError("round cap reached; close with agreed or escalated")
    issue_count = _normalize_issues_open(issues_open)
    # Round 1 is the hand-off, round 2 is the first verdict (sets the issue
    # baseline); convergence pressure binds from round 3 so a 0-issue summon
    # cannot brick the destroyer's verdict.
    if round_number >= 3 and issue_count >= _gan_last_issues_open(rows):
        raise GANDirectiveError("issues must converge; close or escalate")
    record = _base_record(kind=kind, from_window=from_window, to_window=to_window, body=body)
    record["gan"] = {
        "gan_id": gan_id,
        "round": round_number,
        "role": role,
        "issues_open": issue_count,
        "status": "open",
    }
    return record


def _build_job_record_locked(
    path: Path,
    *,
    kind: DirectiveKind,
    from_window: str,
    to_window: str | None,
    body: str,
    job_target: str,
    job_status: str | None,
) -> dict[str, Any]:
    target = str(job_target or "").strip().lower()
    if target == "new":
        if kind != "directive":
            raise JobDirectiveError("new job must start with a directive")
        if not to_window:
            raise ValueError("to_window is required")
        if from_window == "external":
            raise JobDirectiveError("job requires a source window")
        if to_window == from_window:
            raise JobDirectiveError("job requires two distinct windows")
        existing = _open_job_between_locked(path, from_window, to_window)
        if existing:
            raise JobDirectiveError(
                f"open job {existing} already exists between these windows; "
                f"continue it by passing job_id={existing} or close it first"
            )
        record = _base_record(kind=kind, from_window=from_window, to_window=to_window, body=body)
        record["job"] = {
            "job_id": _new_job_id(),
            "cycle": 1,
            "role": "worker",
            "phase": "dispatch",
            "status": "open",
        }
        return record

    job_id = _normalize_job_id(target)
    rows = _read_job_rows_locked(path, job_id)
    if not rows:
        raise JobDirectiveError("unknown job_id")
    if _job_is_terminal(rows):
        raise JobDirectiveError("job is already terminal")
    participants = _job_participants(rows)
    if from_window not in participants:
        raise JobDirectiveError("job participants are fixed")

    if kind == "report":
        status = _normalize_job_status(job_status)
        close_to = to_window or _infer_job_close_target(from_window, participants)
        orchestrator, worker = participants
        # Only the orchestrator (cycle-1 dispatcher) terminally closes a job.
        # A worker's done/escalated is a completion *claim*, not the verdict —
        # convert it into a waking review handback (kind=directive, open) so the
        # orchestrator gets its judgment turn. That turn is the whole reason a
        # job differs from a one-shot directive; without it the orchestrator's
        # "you review" never fires. Live 2026-06-12: a worker self-closed done
        # and the user's review never happened.
        if from_window == worker:
            handback_to = orchestrator
            record = _base_record(
                kind="directive", from_window=from_window, to_window=handback_to, body=body
            )
            record["job"] = {
                "job_id": job_id,
                "cycle": _job_last_cycle(rows),
                "role": "orchestrator",
                "phase": "review",
                "status": "open",
                "worker_claimed_status": status,
            }
            return record
        role = _job_role_for_recipient(close_to, participants)
        phase = _job_phase_for_sender(from_window, participants)
        record = _base_record(kind=kind, from_window=from_window, to_window=close_to, body=body)
        record["job"] = {
            "job_id": job_id,
            "cycle": _job_last_cycle(rows),
            "role": role,
            "phase": phase,
            "status": status,
        }
        return record

    if not to_window:
        raise ValueError("to_window is required")
    if to_window == from_window:
        raise JobDirectiveError("job handback requires the counterpart window")
    role = _job_role_for_recipient(to_window, participants)
    phase = _job_phase_for_sender(from_window, participants)
    if phase == "dispatch":
        cycle = len(_job_dispatch_rows(rows)) + 1
        if cycle > job_cycle_cap():
            raise JobDirectiveError("cycle cap reached; close with done or escalated")
    else:
        cycle = _job_last_cycle(rows)
    record = _base_record(kind=kind, from_window=from_window, to_window=to_window, body=body)
    record["job"] = {
        "job_id": job_id,
        "cycle": cycle,
        "role": role,
        "phase": phase,
        "status": "open",
    }
    return record


def get_pending(
    *,
    to_window: str,
    kind: str | None = None,
    consume: bool = True,
    limit: int = 50,
    known_mtime_ns: int | None = None,
    known_size: int | None = None,
) -> dict[str, Any]:
    normalized_to = normalize_directive_window(to_window)
    if not normalized_to:
        raise ValueError("to_window is required")
    normalized_kind = normalize_directive_kind(kind) if kind else None
    queue_path = directives_path()
    cursor_path = _migrate_legacy_cursor(normalized_to, normalized_kind)
    max_items = max(1, min(int(limit or 50), 100))

    with cross_process_file_lock(queue_path):
        size, mtime_ns = _queue_stat(queue_path)
        cursor = min(_read_cursor(cursor_path), size)
        if (
            known_mtime_ns is not None
            and known_size is not None
            and int(known_mtime_ns) == mtime_ns
            and int(known_size) == size
            and cursor >= size
        ):
            return {
                "ok": True,
                "items": [],
                "unchanged": True,
                "queue_mtime_ns": mtime_ns,
                "queue_size": size,
                "cursor": cursor,
                "cursor_at_end": True,
            }
        if size <= 0 or not queue_path.exists():
            if consume and cursor != 0:
                _write_cursor(cursor_path, 0)
            return {
                "ok": True,
                "items": [],
                "unchanged": False,
                "queue_mtime_ns": mtime_ns,
                "queue_size": size,
                "cursor": 0,
                "cursor_at_end": True,
            }

        items: list[dict[str, Any]] = []
        next_cursor = cursor
        with queue_path.open("rb") as fh:
            fh.seek(cursor)
            while True:
                line = fh.readline()
                if not line:
                    next_cursor = fh.tell()
                    break
                line_end = fh.tell()
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    next_cursor = line_end
                    continue
                if not isinstance(record, dict):
                    next_cursor = line_end
                    continue
                row_kind = str(record.get("kind") or "")
                if row_kind not in VALID_DIRECTIVE_KINDS:
                    next_cursor = line_end
                    continue
                row_to = normalize_directive_window(record.get("to_window"))
                if row_to == normalized_to and (normalized_kind is None or row_kind == normalized_kind):
                    items.append(_normalize_row(record))
                    next_cursor = line_end
                    if len(items) >= max_items:
                        break
                else:
                    next_cursor = line_end
        if consume:
            _write_cursor(cursor_path, next_cursor)
        return {
            "ok": True,
            "items": items,
            "unchanged": False,
            "queue_mtime_ns": mtime_ns,
            "queue_size": size,
            "cursor": next_cursor,
            "cursor_at_end": next_cursor >= size,
        }


def _normalize_row(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "id": str(record.get("id") or ""),
        "ts": str(record.get("ts") or ""),
        "kind": normalize_directive_kind(str(record.get("kind") or "directive")),
        "from_window": normalize_directive_window(record.get("from_window")) or "external",
        "to_window": normalize_directive_window(record.get("to_window")) or "",
        "body": str(record.get("body") or ""),
    }
    gan = _normalize_gan(record.get("gan"))
    if gan:
        normalized["gan"] = gan
    job = _normalize_job(record.get("job"))
    if job:
        normalized["job"] = job
    return normalized


def _normalize_gan(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        gan_id = _normalize_gan_id(str(value.get("gan_id") or ""))
    except ValueError:
        return None
    try:
        round_number = int(value.get("round") or 0)
    except (TypeError, ValueError):
        round_number = 0
    role = str(value.get("role") or "").strip().lower()
    if role not in {"worker", "destroyer"}:
        role = ""
    try:
        issue_count = int(value.get("issues_open") or 0)
    except (TypeError, ValueError):
        issue_count = 0
    status = str(value.get("status") or "open").strip().lower()
    if status not in {"open", *GAN_TERMINAL_STATUSES}:
        status = "open"
    return {
        "gan_id": gan_id,
        "round": max(1, round_number),
        "role": role,
        "issues_open": max(0, issue_count),
        "status": status,
    }


def _normalize_job(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    try:
        job_id = _normalize_job_id(str(value.get("job_id") or ""))
    except ValueError:
        return None
    try:
        cycle = int(value.get("cycle") or 0)
    except (TypeError, ValueError):
        cycle = 0
    role = str(value.get("role") or "").strip().lower()
    if role not in {"orchestrator", "worker"}:
        role = ""
    phase = str(value.get("phase") or "").strip().lower()
    if phase not in {"dispatch", "review"}:
        phase = ""
    status = str(value.get("status") or "open").strip().lower()
    if status not in {"open", *JOB_TERMINAL_STATUSES}:
        status = "open"
    return {
        "job_id": job_id,
        "cycle": max(1, cycle),
        "role": role,
        "phase": phase,
        "status": status,
    }


def get_gan_history(gan_id: str) -> dict[str, Any]:
    normalized_id = _normalize_gan_id(gan_id)
    queue_path = directives_path()
    with cross_process_file_lock(queue_path):
        rows = [_normalize_row(row) for row in _read_gan_rows_locked(queue_path, normalized_id)]
    if not rows:
        raise GANDirectiveError("unknown gan_id")
    statuses = [
        str((row.get("gan") or {}).get("status") or "open")
        for row in rows
        if isinstance(row.get("gan"), dict)
    ]
    status = next((item for item in reversed(statuses) if item in GAN_TERMINAL_STATUSES), "open")
    rounds = [
        int((row.get("gan") or {}).get("round") or 0)
        for row in rows
        if isinstance(row.get("gan"), dict)
    ]
    return {
        "ok": True,
        "gan_id": normalized_id,
        "status": status,
        "round": max(rounds) if rounds else 0,
        "round_cap": GAN_ROUND_CAP,
        "items": rows,
    }


def get_job_history(job_id: str) -> dict[str, Any]:
    normalized_id = _normalize_job_id(job_id)
    queue_path = directives_path()
    with cross_process_file_lock(queue_path):
        rows = [_normalize_row(row) for row in _read_job_rows_locked(queue_path, normalized_id)]
    if not rows:
        raise JobDirectiveError("unknown job_id")
    statuses = [
        str((row.get("job") or {}).get("status") or "open")
        for row in rows
        if isinstance(row.get("job"), dict)
    ]
    status = next((item for item in reversed(statuses) if item in JOB_TERMINAL_STATUSES), "open")
    cycles = [
        int((row.get("job") or {}).get("cycle") or 0)
        for row in rows
        if isinstance(row.get("job"), dict)
    ]
    return {
        "ok": True,
        "job_id": normalized_id,
        "status": status,
        "cycle": max(cycles) if cycles else 0,
        "cycle_cap": job_cycle_cap(),
        "items": rows,
    }


def _window_rank(window: dict[str, Any]) -> tuple[bool, bool, str]:
    # Higher is more authoritative: the current window beats any other; a live
    # window beats a dead one; among equals, the newest created_at wins.
    return (
        bool(window.get("current")),
        bool(window.get("alive")),
        str(window.get("created_at") or ""),
    )


def _body_has_any_marker(body: Any, markers: tuple[str, ...]) -> bool:
    text = str(body or "")
    return any(marker in text for marker in markers)


def _job_contract(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        body = row.get("body")
        if _body_has_any_marker(body, CRITIC_REVIEW_MARKERS) or _body_has_any_marker(body, CRITIC_MAIN_MARKERS):
            return "critic"
    return "builder"


def _latest_job_dispatch_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    dispatches = _job_dispatch_rows(rows)
    return dispatches[-1] if dispatches else None


def _critic_stage(rows: list[dict[str, Any]]) -> str | None:
    dispatch = _latest_job_dispatch_row(rows)
    if not dispatch:
        return "artifact_review"
    if _body_has_any_marker(dispatch.get("body"), CRITIC_PLAN_READY_MARKERS):
        return "plan_review"
    try:
        cycle = int((dispatch.get("job") or {}).get("cycle") or 0)
    except (TypeError, ValueError):
        cycle = 0
    return "fix_review" if cycle >= 3 else "artifact_review"


def _job_window_status(pair8: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "idle"
    latest = rows[-1]
    if normalize_directive_window(latest.get("to_window")) == pair8:
        return "queued"
    if normalize_directive_window(latest.get("from_window")) == pair8:
        return "waiting"
    return "idle"


def _job_window_metadata() -> dict[str, dict[str, Any]]:
    target = directives_path()
    if not target.exists():
        return {}
    with cross_process_file_lock(target):
        rows = _read_directive_rows_locked(target)

    jobs: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        job = row.get("job")
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or "").strip().lower()
        if not JOB_ID_RE.match(job_id):
            continue
        jobs.setdefault(job_id, []).append(row)

    metadata: dict[str, dict[str, Any]] = {}
    for job_id, job_rows in jobs.items():
        if not job_rows or _job_is_terminal(job_rows):
            continue
        try:
            orchestrator, worker = _job_participants(job_rows)
        except JobDirectiveError:
            continue
        contract = _job_contract(job_rows)
        cycle = _job_last_cycle(job_rows)
        latest = job_rows[-1]
        latest_job = latest.get("job") if isinstance(latest.get("job"), dict) else {}
        stage = _critic_stage(job_rows) if contract == "critic" else None
        for pair8, role in ((orchestrator, "orchestrator"), (worker, "critic" if contract == "critic" else "builder")):
            entry = {
                "role": role,
                "status": _job_window_status(pair8, job_rows),
                "contract": contract,
                "active_job_id": job_id,
                "active_job_cycle": cycle,
                "active_job_phase": str(latest_job.get("phase") or ""),
                "active_job_role": str(latest_job.get("role") or ""),
                "job_cycle_cap": job_cycle_cap(),
                "counterpart_window": worker if pair8 == orchestrator else orchestrator,
            }
            if stage:
                entry["stage"] = stage
            metadata[pair8] = entry
    return metadata


def list_windows() -> list[dict[str, Any]]:
    current = pairing.current_pair_id()[:8] or None
    windows_root = pairing.conversation_root() / "_windows"
    best: dict[str, dict[str, Any]] = {}
    job_metadata = _job_window_metadata()
    if windows_root.exists():
        for candidate in sorted(windows_root.iterdir()):
            if not candidate.is_dir():
                continue
            name = candidate.name
            owner = _read_owner(candidate / "owner.json")
            # Reusable `worker*` slots and `overflow-*` dirs carry pair8 in
            # owner.json (the dir name is just a storage slot, no longer the
            # window identity); legacy per-GUID `jhb-<pair8>` dirs carry it in
            # the folder name. owner.json is authoritative, name is the fallback.
            pair8 = ""
            owner_pair8 = owner.get("pair8")
            if isinstance(owner_pair8, str):
                pair8 = owner_pair8.strip()[:8]
            if not pair8 and name.startswith("jhb-"):
                pair8 = name.removeprefix("jhb-")
            if not pair8:
                continue
            pid = _owner_pid(owner)
            alive = pairing._pid_alive(pid) if pid is not None else False
            entry = {
                "pair8": pair8,
                "label": runtime_label_for_pair8(pair8),
                "pid": pid,
                "alive": alive,
                "current": bool(current and pair8 == current),
                "path": str(candidate),
                "created_at": owner.get("created_at") if isinstance(owner.get("created_at"), str) else None,
            }
            # Freshest-slot reuse can leave the same pair8 in an old DEAD slot
            # and the current LIVE slot. Keep the most authoritative per pair8
            # (current > live > newest) so a dead duplicate never shadows the
            # live window and breaks label routing.
            incumbent = best.get(pair8)
            if incumbent is None or _window_rank(entry) > _window_rank(incumbent):
                best[pair8] = entry
    # The main window stores its JHB at the conversation home (conversation/),
    # not under _windows. Surface it from conversation/owner.json so workers can
    # address it (e.g. report back) and it appears in the window list.
    home_owner = _read_owner(pairing.conversation_root() / "owner.json")
    home_owner_pair8 = home_owner.get("pair8")
    if isinstance(home_owner_pair8, str) and home_owner_pair8.strip():
        home_pair8 = home_owner_pair8.strip()[:8]
        pid = _owner_pid(home_owner)
        alive = pairing._pid_alive(pid) if pid is not None else False
        entry = {
            "pair8": home_pair8,
            "label": runtime_label_for_pair8(home_pair8),
            "pid": pid,
            "alive": alive,
            "current": bool(current and home_pair8 == current),
            "path": str(pairing.conversation_root()),
            "created_at": home_owner.get("created_at") if isinstance(home_owner.get("created_at"), str) else None,
        }
        incumbent = best.get(home_pair8)
        if incumbent is None or _window_rank(entry) > _window_rank(incumbent):
            best[home_pair8] = entry
    if current and current not in best:
        best[current] = {
            "pair8": current,
            "label": runtime_label_for_pair8(current),
            "pid": os.getpid(),
            "alive": True,
            "current": True,
            "path": str(windows_root),
            "created_at": None,
        }
    for pair8, entry in best.items():
        meta = job_metadata.get(pair8)
        if meta:
            entry.update(meta)
        else:
            entry.setdefault("role", "unknown")
            entry.setdefault("status", "idle")
            entry.setdefault("contract", "passive")
        if not bool(entry.get("alive")):
            entry["status"] = "stale"
    windows = list(best.values())
    windows.sort(key=lambda item: (not bool(item.get("alive")), str(item.get("pair8") or "")))
    return windows


def _read_owner(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _owner_pid(owner: dict[str, Any]) -> int | None:
    try:
        pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None
