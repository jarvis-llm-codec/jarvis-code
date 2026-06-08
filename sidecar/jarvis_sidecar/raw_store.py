from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, date, datetime, timedelta, timezone
from collections import OrderedDict
from pathlib import Path
from typing import Any

_SESSION_ID = "jarvis_session"
_SESSION_FILE = "jarvis_session.jsonl"
_SESSION_PAPERLOG_FILE = "jarvis_session.paperlog"
_KST = timezone(timedelta(hours=9), name="KST")


def _storage_root() -> Path:
    configured = os.environ.get("JARVIS_RAW_STORE")
    if configured:
        return Path(configured).expanduser()
    # Phase 2: default to repo-local <repo>/data/raw-store. parents[2] from
    # sidecar/jarvis_sidecar/raw_store.py is the repo root.
    return Path(__file__).resolve().parents[2] / "data" / "raw-store"


def _bench_storage_root() -> Path:
    configured = os.environ.get("JARVIS_RAW_BENCH_STORE")
    if configured:
        return Path(configured).expanduser()
    return _storage_root().parent / "conversation_bench_archive"


def _sanitize_session_id(session_id: str | None) -> str:
    raw = str(session_id or _SESSION_ID).strip() or _SESSION_ID
    invalid = '<>:"/\\|?*'
    table = str.maketrans({ch: "_" for ch in invalid})
    return raw.translate(table).replace("..", "_")


def _session_path(session_id: str | None) -> Path:
    safe = _sanitize_session_id(session_id)
    if safe == _SESSION_ID:
        return _storage_root() / _SESSION_FILE
    return _bench_storage_root() / f"{safe}.jsonl"


def _paperlog_path(session_id: str | None) -> Path:
    safe = _sanitize_session_id(session_id)
    if safe == _SESSION_ID:
        return _storage_root() / _SESSION_PAPERLOG_FILE
    return _bench_storage_root() / f"{safe}.paperlog"


_path_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_path_locks_guard = threading.Lock()
_MAX_PATH_LOCKS = 100


def _get_path_lock(path_key: str) -> threading.Lock:
    with _path_locks_guard:
        lock = _path_locks.get(path_key)
        if lock is not None:
            _path_locks.move_to_end(path_key)
            return lock
        if len(_path_locks) >= _MAX_PATH_LOCKS:
            _path_locks.popitem(last=False)
        lock = threading.Lock()
        _path_locks[path_key] = lock
        return lock


def append_raw_turn(
    *,
    project_path: str | None,
    user_message: str,
    assistant_message: str,
    tool_events: list[dict[str, Any]],
    llm_meta: dict[str, Any],
    session_id: str = _SESSION_ID,
) -> Path:
    path = _session_path(session_id)
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "conv_id": _sanitize_session_id(session_id),
        "project_path": project_path,
        "user": user_message,
        "assistant": assistant_message,
        "tool_events": tool_events,
        "llm_meta": llm_meta,
    }
    lock_key = str(path)
    with _get_path_lock(lock_key):
        _append_jsonl_row(path, record)
    _append_project_raw_pointer(project_path, path, record)
    return path


def append_encoder_turn(
    *,
    turn_id: int,
    project_path: str | None,
    encoder_meta: dict[str, Any],
    session_id: str = _SESSION_ID,
) -> Path:
    path = _session_path(session_id)
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    if _encoder_turn_already_written(path, turn_id):
        return path
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "conv_id": _sanitize_session_id(session_id),
        "project_path": project_path,
        "turn": turn_id,
        "kind": "encoder",
        "encoder_meta": encoder_meta,
    }
    lock_key = str(path)
    with _get_path_lock(lock_key):
        _append_jsonl_row(path, record)
    return path


def append_meter_turn(
    *,
    turn_id: int | None,
    project_path: str | None,
    meter_line: str,
    session_id: str = _SESSION_ID,
) -> Path:
    path = _session_path(session_id)
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "conv_id": _sanitize_session_id(session_id),
        "project_path": project_path,
        "turn": turn_id,
        "kind": "meter",
        "meter_line": meter_line,
    }
    lock_key = str(path)
    with _get_path_lock(lock_key):
        _append_jsonl_row(path, record)
    return path


def append_paper_turn(
    *,
    turn_id: int | None,
    project_path: str | None,
    user_message: str,
    assistant_message: str,
    meter_line: str,
    session_id: str = _SESSION_ID,
) -> Path:
    path = _paperlog_path(session_id)
    root = path.parent
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "turn": turn_id,
        "question": user_message,
        "answer": assistant_message,
        "meter": meter_line,
        "project_path": project_path,
    }
    line = (
        f"{record['timestamp']} | turn={record['turn']} | "
        f"question={_one_line(user_message)} | "
        f"answer={_one_line(assistant_message)} | "
        f"{meter_line}"
    )
    lock_key = str(path)
    with _get_path_lock(lock_key):
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
            fh.write("\n")
    return path


def recent_turns(limit: int = 3, session_id: str = _SESSION_ID) -> list[dict[str, Any]]:
    """Return the most recent N conversation turns (user/assistant pairs) in chronological order.

    Skips encoder/meter rows. Used to back-stop stale JHB when the encoder is mid-flight.
    limit<=0 disables injection entirely (returns [])."""
    if limit <= 0:
        return []
    path = _session_path(session_id)
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    collected: list[dict[str, Any]] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") in {"encoder", "meter"}:
            continue
        if "user" not in record and "assistant" not in record:
            continue
        collected.append(record)
        if len(collected) >= limit:
            break
    collected.reverse()
    return collected


def recall_raw(query: str, top_k: int = 5, session_id: str = _SESSION_ID) -> list[dict[str, Any]]:
    turn_hits = recall_raw_turn_numbers(query, top_k=top_k, session_id=session_id)
    if turn_hits:
        return turn_hits
    date_hits = recall_raw_dates(query, top_k=top_k, session_id=session_id)
    if date_hits:
        return date_hits
    path = _session_path(session_id)
    paperlog_path = _paperlog_path(session_id)
    if not path.exists() and not paperlog_path.exists():
        return []
    terms = _recall_terms(query)
    hits: list[tuple[int, int, dict[str, Any]]] = []
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") in {"encoder", "meter"}:
                continue
            score = _raw_recall_score(terms, str(record.get("user", "")), str(record.get("assistant", "")))
            if score > 0 or not terms:
                hits.append((score, line_no, {"line": line_no, **record}))
    if paperlog_path.exists():
        for line_no, line in enumerate(paperlog_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            parsed = _parse_paperlog_line(line)
            if parsed is None:
                continue
            user, assistant = parsed
            score = _raw_recall_score(terms, user, assistant)
            if score > 0 or not terms:
                hits.append((
                    score,
                    line_no,
                    {
                        "line": f"paperlog:{line_no}",
                        "timestamp": line.split(" | ", 1)[0].strip(),
                        "project_path": None,
                        "user": user,
                        "assistant": assistant,
                        "source": "paperlog",
                    },
                ))
    hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _score, _line_no, record in hits[:top_k]]


_TURN_REF_RE = re.compile(
    # keyword-first: turn 5, turn #5, turn no. 5, turn number 5, 턴 5, 턴 5번
    r"(?:\bturns?\s*(?:#|number|no\.?)?\s*|턴\s*#?\s*)(\d+)"
    # number-first (Korean word order): 5번 턴, 5턴, 5번째 턴
    r"|(\d+)\s*(?:번째|번)?\s*턴"
    # bare hash: #5
    r"|#\s*(\d+)",
    re.IGNORECASE,
)


def extract_turn_numbers(query: str, *, max_turns: int = 20) -> list[int]:
    """Extract explicit turn references such as 'turn 3817', '턴 3817', or '5번 턴'."""
    numbers: list[int] = []
    seen: set[int] = set()
    for match in _TURN_REF_RE.finditer(query or ""):
        raw = match.group(1) or match.group(2) or match.group(3)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value <= 0 or value in seen:
            continue
        seen.add(value)
        numbers.append(value)
        if len(numbers) >= max_turns:
            break
    return numbers


def recall_raw_turn_numbers(query: str, top_k: int = 5, session_id: str = _SESSION_ID) -> list[dict[str, Any]]:
    turn_numbers = extract_turn_numbers(query, max_turns=max(1, top_k))
    if not turn_numbers:
        return []
    wanted = set(turn_numbers)
    hits: list[dict[str, Any]] = []
    path = _session_path(session_id)
    if path.exists():
        ordinal_turn = 0
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") in {"encoder", "meter"}:
                continue
            if "user" not in record and "assistant" not in record:
                continue
            ordinal_turn += 1
            explicit_turn = record.get("turn")
            candidate_turns = {ordinal_turn}
            if isinstance(explicit_turn, int):
                candidate_turns.add(explicit_turn)
            if candidate_turns & wanted:
                matched_turn = next(turn for turn in turn_numbers if turn in candidate_turns)
                hits.append({"line": line_no, "turn": matched_turn, **record})
    paperlog_path = _paperlog_path(session_id)
    if paperlog_path.exists():
        for line_no, line in enumerate(paperlog_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            parsed_turn = _parse_paperlog_turn(line)
            if parsed_turn not in wanted:
                continue
            parsed = _parse_paperlog_line(line)
            if parsed is None:
                continue
            user, assistant = parsed
            hits.append({
                "line": f"paperlog:{line_no}",
                "turn": parsed_turn,
                "timestamp": line.split(" | ", 1)[0].strip(),
                "project_path": None,
                "user": user,
                "assistant": assistant,
                "source": "paperlog",
            })
    rank = {turn: idx for idx, turn in enumerate(turn_numbers)}
    hits.sort(key=lambda hit: (rank.get(int(hit.get("turn") or 0), 9999), str(hit.get("line"))))
    return hits[:top_k]


_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)(\s*(?:쯤|경|께|무렵|around|about))?", re.IGNORECASE)
_KR_MONTH_DAY_RE = re.compile(r"(?:(\d{4})년\s*)?(\d{1,2})월\s*(\d{1,2})일(\s*(?:쯤|경|께|무렵))?")
_KR_DAY_RE = re.compile(r"(?<!월\s)(?<!\d)(\d{1,2})일(\s*(?:쯤|경|께|무렵))?")


def extract_local_dates(query: str, *, now: date | None = None, max_dates: int = 10) -> list[str]:
    """Extract KST calendar dates from natural date references."""
    text = query or ""
    today = now or datetime.now(_KST).date()
    dates: list[date] = []
    seen: set[date] = set()

    def add(center: date, around: bool = False) -> None:
        candidates = [center]
        if around:
            candidates = [center, center - timedelta(days=1), center + timedelta(days=1)]
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            dates.append(candidate)
            if len(dates) >= max_dates:
                return

    if "오늘" in text:
        add(today)
    if "어제" in text:
        add(today - timedelta(days=1))
    if "그제" in text or "그저께" in text:
        add(today - timedelta(days=2))

    consumed_spans: list[tuple[int, int]] = []
    for match in _ISO_DATE_RE.finditer(text):
        try:
            center = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
        add(center, around=bool(match.group(4)))
        consumed_spans.append(match.span())

    for match in _KR_MONTH_DAY_RE.finditer(text):
        try:
            year = int(match.group(1)) if match.group(1) else today.year
            center = date(year, int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
        add(center, around=bool(match.group(4)))
        consumed_spans.append(match.span())

    def consumed(start: int, end: int) -> bool:
        return any(start >= span_start and end <= span_end for span_start, span_end in consumed_spans)

    for match in _KR_DAY_RE.finditer(text):
        if consumed(*match.span()):
            continue
        try:
            day = int(match.group(1))
            center = date(today.year, today.month, day)
        except ValueError:
            continue
        # If the day is far in the future, infer the previous month. This
        # keeps "25일" natural near month boundaries without guessing years.
        if center > today + timedelta(days=3):
            month = today.month - 1 or 12
            year = today.year - 1 if today.month == 1 else today.year
            try:
                center = date(year, month, day)
            except ValueError:
                continue
        add(center, around=bool(match.group(2)))
        if len(dates) >= max_dates:
            break

    return [value.isoformat() for value in dates[:max_dates]]


def recall_raw_dates(query: str, top_k: int = 5, session_id: str = _SESSION_ID) -> list[dict[str, Any]]:
    local_dates = extract_local_dates(query, max_dates=max(1, top_k))
    if not local_dates:
        return []
    wanted = set(local_dates)
    rank = {value: idx for idx, value in enumerate(local_dates)}
    hits: list[dict[str, Any]] = []
    path = _session_path(session_id)
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("kind") in {"encoder", "meter"}:
                continue
            if "user" not in record and "assistant" not in record:
                continue
            local_date = _timestamp_local_date(record.get("timestamp") or record.get("ts"))
            if local_date in wanted:
                hits.append({"line": line_no, "local_date": local_date, **record})
    paperlog_path = _paperlog_path(session_id)
    if paperlog_path.exists():
        for line_no, line in enumerate(paperlog_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            timestamp = line.split(" | ", 1)[0].strip()
            local_date = _timestamp_local_date(timestamp)
            if local_date not in wanted:
                continue
            parsed = _parse_paperlog_line(line)
            if parsed is None:
                continue
            user, assistant = parsed
            hits.append({
                "line": f"paperlog:{line_no}",
                "turn": _parse_paperlog_turn(line),
                "timestamp": timestamp,
                "local_date": local_date,
                "project_path": None,
                "user": user,
                "assistant": assistant,
                "source": "paperlog",
            })
    hits.sort(key=lambda hit: (rank.get(str(hit.get("local_date")), 9999), str(hit.get("timestamp") or ""), str(hit.get("line"))))
    return hits[:top_k]


def _timestamp_local_date(value: Any) -> str | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(_KST).date().isoformat()


_RECALL_TOKEN_RE = re.compile(r"[\w가-힣]+", re.UNICODE)
_RAW_DENIAL_PATTERNS = (
    "기록된 게 없",
    "기록에 없",
    "기억에 없",
    "아직 안 알려",
    "아직 알려주",
    "안 알려주",
    "없어요",
    "없네요",
    "no record",
    "not recorded",
    "don't have",
)


def _recall_terms(query: str) -> list[str]:
    terms = [term.casefold() for term in _RECALL_TOKEN_RE.findall(query or "") if len(term) >= 2]
    expanded = list(terms)
    if any(term in {"와이프", "아내", "부인", "배우자"} for term in terms):
        expanded.extend(["와이프", "아내", "부인", "배우자"])
    if any(term.startswith("이름") or term == "성함" for term in terms):
        expanded.extend(["이름", "성함"])
    return list(dict.fromkeys(expanded))


def _raw_recall_score(terms: list[str], user: str, assistant: str) -> int:
    haystack = f"{user}\n{assistant}".casefold()
    score = sum(haystack.count(term) for term in terms)
    if score <= 0 and terms:
        return 0
    assistant_head = assistant.casefold()[:220]
    if any(pattern in assistant_head for pattern in _RAW_DENIAL_PATTERNS):
        score -= 4
    if "와이프" in haystack or "아내" in haystack or "부인" in haystack:
        score += 2
    return score


def _parse_paperlog_line(line: str) -> tuple[str, str] | None:
    if " | question=" not in line or " | answer=" not in line:
        return None
    try:
        question_part = line.split(" | question=", 1)[1]
        question, answer_part = question_part.split(" | answer=", 1)
        answer = answer_part.split(" | prompt_tag=", 1)[0].split(" | [jlc:meter]", 1)[0]
    except ValueError:
        return None
    return question.strip(), answer.strip()


def _parse_paperlog_turn(line: str) -> int | None:
    match = re.search(r"(?:^|\s\|\s)turn=(\d+)", line)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def recent_failure_modes(limit: int = 50, session_id: str = _SESSION_ID) -> dict[str, int]:
    path = _session_path(session_id)
    if not path.exists():
        return {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    counts: dict[str, int] = {}
    seen = 0
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("kind") != "encoder":
            continue
        meta = record.get("encoder_meta")
        if not isinstance(meta, dict):
            continue
        mode = str(meta.get("failure_mode") or "").strip()
        if not mode:
            continue
        counts[mode] = counts.get(mode, 0) + 1
        seen += 1
        if seen >= limit:
            break
    return counts


def _append_project_raw_pointer(project_path: str | None, log_path: Path, record: dict[str, Any]) -> None:
    if not project_path:
        return
    raw_md = Path(project_path).expanduser() / "jarvis" / "RAW.md"
    if not raw_md.parent.exists():
        return
    user = str(record.get("user", "")).replace("\n", " ").strip()[:160]
    timestamp = str(record.get("timestamp", ""))
    pointer = f"\n- {timestamp}: raw turn stored at `{log_path}`; user: {user}\n"
    with _get_path_lock(str(raw_md)):
        with raw_md.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(pointer)


def _append_jsonl_row(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def _one_line(text: str) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())


def _encoder_turn_already_written(path: Path, turn_id: int) -> bool:
    if not path.exists():
        return False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return False
        return record.get("kind") == "encoder" and record.get("turn") == turn_id
    return False
