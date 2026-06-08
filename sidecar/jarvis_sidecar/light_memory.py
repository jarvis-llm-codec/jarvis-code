from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .memory_files import MEMORY_FILES

_SECTION_TO_FILE = {
    "rule": "LAW.md",
    "rules": "LAW.md",
    "규칙": "LAW.md",
    "ban": "BAN.md",
    "bans": "BAN.md",
    "금지": "BAN.md",
    "habit": "HABIT.md",
    "habits": "HABIT.md",
    "습관": "HABIT.md",
    "map": "MAP.md",
    "지도": "MAP.md",
    "why": "WHY.md",
    "decisions": "WHY.md",
    "decision": "WHY.md",
    "이유": "WHY.md",
    "결정": "WHY.md",
}

_HEADING_RE = re.compile(r"^#{2,3}\s+(.+?)\s*(?:\[P\d+\])?\s*$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_BAN_RULE_RE = re.compile(r"^(?:ban|forbidden|do not|don't|금지|하지 마|하지마)\s*[:：\-–—]\s*\S", re.IGNORECASE)
_LAW_RULE_RE = re.compile(r"^(?:rule|law|must|always|규칙|원칙|반드시)\s*[:：\-–—]\s*\S", re.IGNORECASE)


def update_light_memory(
    *,
    memory_project_path: str | None,
    user_message: str,
    assistant_message: str,
    raw_path: Path,
    tool_events: list[dict[str, Any]],
    jhb_delta_text: str = "",
) -> dict[str, Any]:
    if not memory_project_path:
        return {"updated": []}

    root = Path(memory_project_path).expanduser()
    jarvis_dir = root / "jarvis"
    if not jarvis_dir.exists():
        return {"updated": []}

    updated: list[str] = []
    now_path = jarvis_dir / "NOW.md"
    now_content = _render_now(root, user_message, assistant_message, raw_path)
    if _write_if_changed(now_path, now_content):
        updated.append("jarvis/NOW.md")

    # OMM auto-append removed (2026-05-17): generic "tool/command failed"
    # bullets polluted OMM.md without recording WHAT failed or WHY. The
    # LLM now owns OMM entries via the TURN END guide in MODE_PROMPT, so
    # each line is a concrete fact (what failed + why + correction).
    # tool_events are still surfaced through raw_path / JHB for the LLM
    # to consult when writing the entry.

    rule_updates = _extract_explicit_rules(user_message)
    for filename, rules in rule_updates.items():
        if _append_memory_lines(jarvis_dir / filename, filename, rules):
            updated.append(f"jarvis/{filename}")

    classified = classify_razor_to_files(jhb_delta_text)
    for filename, lines in classified.items():
        if _append_memory_lines(jarvis_dir / filename, filename, lines):
            updated.append(f"jarvis/{filename}")

    return {"updated": sorted(set(updated)), "mode": "light"}


def classify_razor_to_files(jhb_delta_text: str) -> dict[str, list[str]]:
    sections = _parse_sections(jhb_delta_text)
    grouped: dict[str, list[str]] = {}
    for heading, lines in sections.items():
        filename = _SECTION_TO_FILE.get(_normalize_heading_key(heading))
        if not filename:
            continue
        for line in lines:
            grouped.setdefault(filename, [])
            if line not in grouped[filename]:
                grouped[filename].append(line)
    return grouped


def build_jhb_delta_text(prev_jhb: str, new_jhb: str) -> str:
    prev_sections = _parse_sections(prev_jhb)
    new_sections = _parse_sections(new_jhb)
    blocks: list[str] = []
    for heading, lines in new_sections.items():
        prev_lines = {_normalize_bullet(line) for line in prev_sections.get(heading, [])}
        added = [line for line in lines if _normalize_bullet(line) not in prev_lines]
        if not added:
            continue
        blocks.append(f"### {heading}")
        blocks.extend(added)
        blocks.append("")
    return "\n".join(blocks).strip()


def _parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_heading: str | None = None
    for raw_line in (text or "").splitlines():
        heading_match = _HEADING_RE.match(raw_line.strip())
        if heading_match:
            current_heading = _clean_heading(heading_match.group(1))
            sections.setdefault(current_heading, [])
            continue
        if current_heading is None:
            continue
        bullet_match = _BULLET_RE.match(raw_line)
        if bullet_match:
            bullet = f"- {_one_line(bullet_match.group(1), 400)}"
            sections[current_heading].append(bullet)
    return {heading: lines for heading, lines in sections.items() if lines}


def _clean_heading(heading: str) -> str:
    return re.sub(r"\s+", " ", heading).strip()


def _normalize_heading_key(heading: str) -> str:
    normalized = heading.casefold()
    normalized = re.sub(r"^[^0-9a-zA-Z가-힣]+", "", normalized)
    normalized = re.sub(r"[^0-9a-zA-Z가-힣 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _append_memory_lines(path: Path, filename: str, lines: list[str]) -> bool:
    clean_lines = [line for line in (_normalize_appended_line(line) for line in lines) if line]
    if not clean_lines:
        return False

    existing = path.read_text(encoding="utf-8") if path.exists() else MEMORY_FILES.get(filename, "")
    existing_lines = [_normalize_bullet(line) for line in existing.splitlines()[-50:]]
    pending = [line for line in clean_lines if _normalize_bullet(line) not in existing_lines]
    if not pending:
        return False

    placeholder = MEMORY_FILES.get(filename, "").strip()
    body_lines = existing.splitlines()
    if existing.strip() == placeholder:
        title = body_lines[0] if body_lines else f"# {Path(filename).stem}"
        path.write_text(title + "\n\n" + "\n".join(pending) + "\n", encoding="utf-8")
        return True

    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for line in pending:
            fh.write(line + "\n")
    return True


def _normalize_appended_line(line: str) -> str:
    text = _one_line(line, 400).strip()
    if not text:
        return ""
    if text.startswith("- "):
        return text
    if text.startswith("-"):
        return "- " + text[1:].strip()
    return "- " + text


def _normalize_bullet(line: str) -> str:
    return _one_line(line, 400).casefold()


def _render_now(root: Path, user_message: str, assistant_message: str, raw_path: Path) -> str:
    return f"""# NOW

Updated: {_ts()}

## Current Thread
Project: {root.name}

User: {_one_line(user_message, 500)}

Assistant: {_one_line(assistant_message, 700)}

## Next Action
- Continue from the current thread unless the user switches projects or gives a newer priority.

## Evidence
- Raw turn: `{raw_path}`
"""


def _write_if_changed(path: Path, content: str) -> bool:
    existing = path.read_text(encoding="utf-8") if path.exists() else None
    if existing == content:
        return False
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)
    return True


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)


def _event_is_error(event: dict[str, Any]) -> bool:
    for result in event.get("toolResults", []) or []:
        if isinstance(result, dict) and result.get("isError"):
            return True
    return False


def _extract_explicit_rules(text: str) -> dict[str, list[str]]:
    law: list[str] = []
    ban: list[str] = []
    for raw_line in (text or "").splitlines():
        line = _one_line(raw_line, 280).strip()
        if not line:
            continue
        normalized = line.lstrip("-•* \t")
        if _matches_prefixed_rule(normalized, _BAN_RULE_RE):
            ban.append(f"- {normalized}")
            continue
        if _matches_prefixed_rule(normalized, _LAW_RULE_RE):
            law.append(f"- {normalized}")
    return {"LAW.md": law, "BAN.md": ban}


def _matches_prefixed_rule(line: str, pattern: re.Pattern[str]) -> bool:
    return bool(pattern.match(line))


def _one_line(text: str, limit: int) -> str:
    return " ".join((text or "").split())[:limit]


def _ts() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
