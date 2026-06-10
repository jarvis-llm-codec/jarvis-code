from __future__ import annotations

import hashlib
import re


class DiffParseError(ValueError):
    """DSL output is structurally malformed."""


class DiffDriftError(ValueError):
    """KEEP target title does not exist in prev_jhb."""


_PRIORITY_SUFFIX_RE = re.compile(r"\s*\[P\d\]\s*$")
_PRIORITY_SUFFIX_CAPTURE_RE = re.compile(r"\s*\[(P[0-3])\]\s*$")
_KEEP_RE = re.compile(r'^KEEP "([^"]+)"\s*$')
_APPEND_RE = re.compile(r'^APPEND "([^"]+)"(?:\s+\[(P[0-3])\])?\s*$')
_UPDATE_BY_TURN_RE = re.compile(r'^UPDATE_BY_TURN "([^"]+)"\s+t(\d+)\s*$')
_SECTION_HEADING_RE = re.compile(r"^## .*$", re.MULTILINE)
_JHB_START = "<<<JHB>>>"
_JHB_END = "<<<END_JHB>>>"


def section_title(heading_line: str) -> str:
    title = heading_line.rstrip("\r\n")
    if title.startswith("##"):
        title = title[2:]
        if title.startswith(" "):
            title = title[1:]
    title = _PRIORITY_SUFFIX_RE.sub("", title)
    return title.strip()


def parse_diff_dsl(diff_output: str) -> tuple[list[tuple[str, object]], list[str]]:
    normalized = diff_output.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    sections_in_order: list[tuple[str, object]] = []
    kept_titles: list[str] = []
    seen_keep_titles: set[str] = set()
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        if line.strip().upper() == "PASSTHROUGH":
            sections_in_order.append(("passthrough", ""))
            index += 1
            continue

        if line.startswith("KEEP"):
            match = _KEEP_RE.fullmatch(line)
            if not match:
                raise DiffParseError(f"Malformed KEEP marker: {line!r}")
            title = section_title(match.group(1))
            if not title:
                raise DiffParseError(f"KEEP marker has empty title: {line!r}")
            if title in seen_keep_titles:
                raise DiffParseError(f"Duplicate KEEP title: {title!r}")
            seen_keep_titles.add(title)
            sections_in_order.append(("keep", title))
            kept_titles.append(title)
            index += 1
            continue

        if line.startswith("APPEND"):
            match = _APPEND_RE.fullmatch(line)
            if not match:
                raise DiffParseError(f"Malformed APPEND marker: {line!r}")
            raw_title = match.group(1)
            priority = match.group(2)
            if priority is None:
                priority_match = _PRIORITY_SUFFIX_CAPTURE_RE.search(raw_title)
                if priority_match is None:
                    raise DiffParseError(f"Malformed APPEND marker: {line!r}")
                priority = priority_match.group(1)
            title = section_title(raw_title)
            if not title:
                raise DiffParseError(f"APPEND marker has empty title: {line!r}")
            body_lines, index = _collect_command_body(lines, index + 1)
            if not body_lines or not any(_is_bullet_line(ln) for ln in body_lines):
                raise DiffParseError(f"APPEND {title!r} must include at least one bullet")
            sections_in_order.append(("append", {"title": title, "priority": priority, "body": _clean_body(body_lines)}))
            continue

        if line.startswith("UPDATE_BY_TURN"):
            match = _UPDATE_BY_TURN_RE.fullmatch(line)
            if not match:
                raise DiffParseError(f"Malformed UPDATE_BY_TURN marker: {line!r}")
            title = section_title(match.group(1))
            turn = int(match.group(2))
            if not title:
                raise DiffParseError(f"UPDATE_BY_TURN marker has empty title: {line!r}")
            body_lines, index = _collect_command_body(lines, index + 1)
            if not body_lines or not any(_is_bullet_line(ln) for ln in body_lines):
                raise DiffParseError(f"UPDATE_BY_TURN {title!r} t{turn} must include a replacement bullet")
            sections_in_order.append(("update_turn", {"title": title, "turn": turn, "body": _clean_body(body_lines)}))
            continue

        if line.startswith("## "):
            title = section_title(line)
            if not title:
                raise DiffParseError(f"Section heading has empty title: {line!r}")
            section_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index]
                if _is_command_line(next_line):
                    break
                section_lines.append(next_line)
                index += 1
            section_markdown = "\n".join(section_lines).rstrip(" \t\n") + "\n"
            sections_in_order.append(("changed", section_markdown))
            continue

        raise DiffParseError(f"Unexpected DSL line: {line!r}")

    if not sections_in_order:
        raise DiffParseError("Diff DSL output contains no sections or KEEP markers")

    return sections_in_order, kept_titles


def apply_diff(prev_jhb: str, diff_output: str) -> str:
    prev_sections = _parse_jhb_sections(prev_jhb)
    prev_by_title = {title: markdown for title, markdown in prev_sections}
    sections_in_order, _ = parse_diff_dsl(diff_output)
    if all(kind == "passthrough" for kind, _payload in sections_in_order):
        return prev_jhb.rstrip(" \t\n") + ("\n" if prev_jhb.strip() else "")
    if any(kind in {"append", "update_turn"} for kind, _payload in sections_in_order):
        return _apply_delta_ops(prev_sections, sections_in_order)

    changed_titles: set[str] = set()
    rendered_sections: list[str] = []

    for kind, payload in sections_in_order:
        if kind == "keep":
            payload = str(payload)
            if payload not in prev_by_title:
                raise DiffDriftError(f"KEEP target title not found in prev_jhb: {payload!r}")
            rendered_sections.append(prev_by_title[payload].rstrip(" \t\n"))
            continue
        if kind == "passthrough":
            continue

        payload = str(payload)
        title = section_title(payload.split("\n", 1)[0])
        if title in changed_titles:
            raise DiffParseError(f"Duplicate changed section title: {title!r}")
        changed_titles.add(title)
        rendered_sections.append(payload.rstrip(" \t\n"))

    return "\n\n".join(rendered_sections) + "\n"


def _apply_delta_ops(
    prev_sections: list[tuple[str, str]],
    ops: list[tuple[str, object]],
) -> str:
    sections: list[dict[str, object]] = [
        {"title": title, "markdown": markdown.rstrip(" \t\n")}
        for title, markdown in prev_sections
    ]
    index_by_title = {str(section["title"]): idx for idx, section in enumerate(sections)}
    for kind, payload_obj in ops:
        if kind == "keep":
            title = str(payload_obj)
            if title not in index_by_title:
                raise DiffDriftError(f"KEEP target title not found in prev_jhb: {title!r}")
            continue
        if kind == "passthrough":
            continue
        if kind == "changed":
            raise DiffParseError("Full section rewrite is not allowed in APPEND/UPDATE delta mode")
        if not isinstance(payload_obj, dict):
            raise DiffParseError(f"Malformed delta payload for {kind}")
        title = str(payload_obj.get("title") or "")
        if not title:
            raise DiffParseError(f"{kind} has empty title")
        if kind == "append":
            priority = str(payload_obj.get("priority") or "P1")
            body = str(payload_obj.get("body") or "").strip()
            if title not in index_by_title:
                sections.append({"title": title, "markdown": f"## {title} [{priority}]"})
                index_by_title[title] = len(sections) - 1
            idx = index_by_title[title]
            markdown = str(sections[idx]["markdown"]).rstrip(" \t\n")
            sections[idx]["markdown"] = f"{markdown}\n{body}".rstrip(" \t\n")
            continue
        if kind == "update_turn":
            turn = int(payload_obj.get("turn") or 0)
            if title not in index_by_title:
                raise DiffDriftError(f"UPDATE_BY_TURN target section not found: {title!r}")
            idx = index_by_title[title]
            body = str(payload_obj.get("body") or "").strip()
            if not re.search(r"\(t\d+\)(?!\d)", body):
                raise DiffParseError("UPDATE_BY_TURN replacement must include a turn tag")
            markdown = str(sections[idx]["markdown"])
            stripped = markdown.rstrip(" \t\n")
            if body not in stripped:
                sections[idx]["markdown"] = f"{stripped}\n{body}".rstrip(" \t\n")
            continue
        raise DiffParseError(f"Unknown diff operation: {kind!r}")

    rendered = [str(section["markdown"]).rstrip(" \t\n") for section in sections]
    return "\n\n".join(section for section in rendered if section.strip()) + "\n"


def _is_command_line(line: str) -> bool:
    return (
        line.startswith("## ")
        or _KEEP_RE.fullmatch(line) is not None
        or _APPEND_RE.fullmatch(line) is not None
        or _UPDATE_BY_TURN_RE.fullmatch(line) is not None
    )


def _collect_command_body(lines: list[str], index: int) -> tuple[list[str], int]:
    body: list[str] = []
    while index < len(lines):
        line = lines[index]
        if _is_command_line(line):
            break
        body.append(line)
        index += 1
    return body, index


def _clean_body(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return "\n".join(lines).rstrip(" \t\n")


def _is_bullet_line(line: str) -> bool:
    return bool(re.match(r"^\s*-\s+", line))


def strip_jhb_wrappers(text: str) -> str:
    """Remove accidental code fences and JHB delimiters from an encoder block."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    normalized = _strip_outer_code_fence(normalized)
    start = normalized.find(_JHB_START)
    end = normalized.rfind(_JHB_END)
    if start != -1 and end != -1 and end > start:
        normalized = normalized[start + len(_JHB_START) : end].strip()
        normalized = _strip_outer_code_fence(normalized)
    return normalized.strip()


def normalize_stored_jhb(text: str) -> str:
    """Convert accidentally persisted encoder DSL/wrappers into canonical JHB markdown.

    Normal on-disk JHB is markdown sections (`## Title [Pn]` plus bullets).
    This helper only rewrites obvious corrupted/raw forms: fenced files, JHB
    delimiter blocks, or files whose first meaningful line is diff DSL.
    """

    if not text.strip():
        return ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    body = strip_jhb_wrappers(normalized)
    first_line = _first_meaningful_line(body)
    needs_normalization = (
        _JHB_START in normalized
        or _JHB_END in normalized
        or _is_outer_code_fence(normalized.strip())
        or first_line == "PASSTHROUGH"
        or first_line.startswith("APPEND")
        or first_line.startswith("KEEP")
        or first_line.startswith("UPDATE_BY_TURN")
    )
    if not needs_normalization:
        return text
    try:
        return apply_diff("", body)
    except (DiffDriftError, DiffParseError):
        if _parse_jhb_sections(body):
            return body.rstrip(" \t\n") + "\n"
        return text


def _first_meaningful_line(text: str) -> str:
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_outer_code_fence(text: str) -> bool:
    lines = text.strip().split("\n")
    return len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip().startswith("```")


def _strip_outer_code_fence(text: str) -> str:
    if not _is_outer_code_fence(text):
        return text
    lines = text.strip().split("\n")
    return "\n".join(lines[1:-1]).strip()


def compute_jhb_sha(jhb: str) -> str:
    normalized = jhb.replace("\r\n", "\n").replace("\r", "\n")
    tuples: list[tuple[str, str]] = []
    for title, markdown in _parse_jhb_sections(normalized):
        lines = markdown.split("\n")
        body_lines = lines[1:]
        body = _normalize_body("\n".join(body_lines))
        tuples.append((title, body))
    serialized = repr(tuples).encode("utf-8", errors="replace")
    return hashlib.sha256(serialized).hexdigest()[:16]


def _parse_jhb_sections(jhb: str) -> list[tuple[str, str]]:
    normalized = jhb.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_SECTION_HEADING_RE.finditer(normalized))
    sections: list[tuple[str, str]] = []

    for position, match in enumerate(matches):
        heading = match.group(0)
        title = section_title(heading)
        if not title:
            raise DiffParseError(f"Section heading has empty title: {heading!r}")
        end = matches[position + 1].start() if position + 1 < len(matches) else len(normalized)
        markdown = normalized[match.start() : end].rstrip(" \t\n") + "\n"
        sections.append((title, markdown))

    return sections


parse_jhb_sections = _parse_jhb_sections


def _normalize_body(body: str) -> str:
    normalized_lines = [line.rstrip() for line in body.split("\n")]
    collapsed_lines: list[str] = []
    blank_pending = False

    for line in normalized_lines:
        if line == "":
            blank_pending = True
            continue
        if blank_pending and collapsed_lines:
            collapsed_lines.append("")
        collapsed_lines.append(line)
        blank_pending = False

    return "\n".join(collapsed_lines).strip()
