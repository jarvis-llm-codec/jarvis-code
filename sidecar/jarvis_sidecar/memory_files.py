from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any
from datetime import UTC, datetime

from .file_locks import locked_atomic_write_text

# CANONICAL — DO NOT MODIFY. Per Jun's original epistemic design.
# Each section answers a *different* question, not a temporal slice.
# Adding/removing/renaming canonical sections silently breaks update_jarvis_md
# routing, the encoder pipeline, and re-entry semantics.
CANONICAL_SECTIONS = ("NOW", "MAP", "LAW", "BAN", "HABIT", "WHY", "OMM", "RAW")
DESIGN_BRIEF_SECTION = "DESIGN_BRIEF"
VALID_SECTIONS = (*CANONICAL_SECTIONS, DESIGN_BRIEF_SECTION)
INTERRUPT_CHECKPOINT_BEGIN = "<!-- JARVIS_INTERRUPT_CHECKPOINT_BEGIN -->"
INTERRUPT_CHECKPOINT_END = "<!-- JARVIS_INTERRUPT_CHECKPOINT_END -->"

SECTION_HEADERS = {
    "NOW":   "## NOW — Current Active Task",
    "MAP":   "## MAP — Project Map and Symbol Index",
    "LAW":   "## LAW — Learned Agent Warnings",
    "BAN":   "## BAN — Forbidden Actions",
    "HABIT": "## HABIT — User and Project Preferences",
    "WHY":   "## WHY — Why History Yells (Decision Rationale)",
    "OMM":   "## OMM — Oh My Mistake (Failure Retrospectives)",
    "RAW":   "## RAW — Raw Evidence Pointers",
    DESIGN_BRIEF_SECTION: "## Design Brief",
}

SECTION_STARTERS = {
    "NOW": (
        "- Status: no active task yet.\n"
        "- Last verified: not yet.\n"
        "Next: wait for a concrete project request."
    ),
    "MAP": (
        "- Keep only stable files, symbols, entry points, tests, and runtime commands.\n"
        "- Prefer paths plus purpose; remove stale implementation trivia."
    ),
    "LAW": (
        "- Format: `LAW-001: Trigger -> Rule -> Verify`.\n"
        "- Use for hard project invariants that must stay true on future edits."
    ),
    "BAN": (
        "- Format: `BAN-001: Never <action>; because <failure>; verify <check>`.\n"
        "- Use for known-dangerous actions, not generic caution."
    ),
    "HABIT": (
        "- Format: `HABIT-001: When <situation>, prefer <style/workflow>`.\n"
        "- Use for user/project preferences that affect future choices."
    ),
    "WHY": (
        "- Record decision rationale only: `Decision -> Why -> Tradeoff`.\n"
        "- Do not duplicate changelog, NOW, or RAW evidence."
    ),
    "OMM": (
        "OMM entries are operational mistake-prevention rules, not apologies.\n"
        "Use this exact shape:\n"
        "### OMM-001: Short title\n"
        "- Trigger: When this rule must be recalled.\n"
        "- Mistake: What failed before, concretely.\n"
        "- Rule: What must/never happen next time.\n"
        "- Required action: What to inspect or change before proceeding.\n"
        "- Verify: Command, test, log, or observable check."
    ),
    "RAW": (
        "- Evidence pointers only: date, request, files changed, commands run, test result, turn id if known.\n"
        "- Do not paste transcripts or long explanations here."
    ),
}

def _default_jarvis_md(name: str) -> str:
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    sections = "\n\n".join(f"{SECTION_HEADERS[k]}\n{SECTION_STARTERS[k]}" for k in CANONICAL_SECTIONS)
    return f"""---
project: {name}
updated: {now}
---
# JARVIS.md — {name}

{sections}

"""

def ensure_workspace_memory(project_path: str | None) -> dict[str, str]:
    if not project_path:
        return {}
    root = Path(project_path).expanduser()
    # 2026-05-19: previously returned {} when the dir did not exist, which made
    # the caller in app.py fall back to "existing" via `.get(..., "existing")`,
    # lying to the LLM. Live snake-game test: register_project responded
    # `jarvis_md: existing` for a brand-new project whose folder did not yet
    # exist, LLM tried to read JARVIS.md → ENOENT. Create the folder when
    # missing so a real JARVIS.md skeleton always lands.
    if not root.exists():
        try:
            root.mkdir(parents=True, exist_ok=True)
        except Exception:
            return {"JARVIS.md": "missing"}
    if not root.is_dir():
        return {"JARVIS.md": "missing"}

    jarvis_md = root / "JARVIS.md"
    if jarvis_md.exists():
        return {"JARVIS.md": "existing"}

    locked_atomic_write_text(jarvis_md, _default_jarvis_md(root.name or "project"))
    return {"JARVIS.md": "created"}

def read_project_memory(project_path: str | None, max_chars: int = 60000) -> tuple[str, list[str]]:
    if not project_path:
        return "", []
    root = Path(project_path).expanduser()
    jarvis_md = root / "JARVIS.md"
    warnings: list[str] = []
    if not jarvis_md.exists():
        return "", []
    try:
        content = jarvis_md.read_text(encoding="utf-8")
        if len(content) > max_chars:
            warnings.append("project memory truncated")
            content = content[:max_chars].rstrip() + "\n...[truncated]"
        return content, warnings
    except Exception as exc:
        warnings.append(f"failed to read JARVIS.md: {exc}")
        return "", warnings

def update_jarvis_md(project_path: str | None, *, field: str, value: Any) -> dict[str, Any]:
    """Patch one section of <project_path>/JARVIS.md.

    Replaces the body between `## <FIELD>` and the next `## ` header (or EOF)
    with `value`. Other sections are untouched. Creates the file from the
    default skeleton if missing. Appends the section if not present.
    Updates the `updated:` frontmatter line when present.
    """
    result = update_jarvis_md_batch(project_path, updates=[{"field": field, "value": value}])
    if not result.get("ok"):
        return result
    fields = result.get("fields")
    return {
        "ok": True,
        "field": fields[0] if isinstance(fields, list) and fields else str(field).strip().upper(),
        "path": result.get("path"),
        "bytes": result.get("bytes"),
    }


def update_jarvis_md_batch(project_path: str | None, *, updates: list[dict[str, Any]]) -> dict[str, Any]:
    """Patch one or more sections of <project_path>/JARVIS.md in one write."""
    if not project_path:
        return {"ok": False, "error": "project_path is required"}
    normalized: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for item in updates:
        field = item.get("field") if isinstance(item, dict) else None
        field_upper = _normalize_section_field(field)
        if field_upper not in VALID_SECTIONS:
            return {
                "ok": False,
                "error": f"field must be one of {list(VALID_SECTIONS)}; got {field!r}",
            }
        if field_upper in seen:
            return {"ok": False, "error": f"duplicate field in updates: {field_upper}"}
        seen.add(field_upper)
        normalized.append((field_upper, item.get("value") if isinstance(item, dict) else ""))
    if not normalized:
        return {"ok": False, "error": "updates must not be empty"}
    root = Path(project_path).expanduser()
    if not root.exists() or not root.is_dir():
        return {"ok": False, "error": f"project_path does not exist: {root}"}
    p = root / "JARVIS.md"
    if not p.exists():
        locked_atomic_write_text(p, _default_jarvis_md(root.name or "project"))

    content = p.read_text(encoding="utf-8")
    new_content = content
    for field_upper, value in normalized:
        new_body = str(value).rstrip("\n")

        # Replace the named section's body. A section runs from its `## NAME`
        # header line (with optional `— Description` suffix) to the next `## `
        # header or end of file.
        section_re = _section_regex(field_upper)
        if section_re.search(new_content):
            # lambda replacement so backslashes inside `value` (e.g. Windows
            # paths like C:\jarvis_workspace\tetris) are not interpreted as
            # regex backreferences.
            new_content = section_re.sub(
                lambda m, body=new_body: f"{m.group(1)}{body}\n\n", new_content, count=1
            )
        else:
            if not new_content.endswith("\n"):
                new_content += "\n"
            header = SECTION_HEADERS.get(field_upper, f"## {field_upper}")
            new_content = new_content + f"\n{header}\n{new_body}\n"

    now_iso = datetime.now(UTC).replace(microsecond=0).isoformat()
    new_content = re.sub(
        r"^updated:[^\n]*$",
        f"updated: {now_iso}",
        new_content,
        count=1,
        flags=re.MULTILINE,
    )

    locked_atomic_write_text(p, new_content)
    return {
        "ok": True,
        "fields": [field for field, _value in normalized],
        "path": str(p),
        "bytes": len(new_content),
    }

def write_interrupt_checkpoint(
    project_path: str | None,
    *,
    user_message: str = "",
    assistant_message: str = "",
    tool_events: list[dict[str, Any]] | None = None,
    subturn_log: str = "",
    mode: str = "",
    cwd: str | None = None,
    reason: str = "escape_interrupt",
) -> dict[str, Any]:
    """Persist an interrupted coding turn into JARVIS.md NOW without LLM help.

    This is the hard safety path for ESC/cancel. It intentionally avoids an
    LLM summarizer so a user interrupt always writes a recoverable checkpoint
    before the aborted turn is discarded.
    """
    if not project_path:
        return {"ok": False, "error": "project_path is required"}
    root = Path(project_path).expanduser()
    if not root.exists() or not root.is_dir():
        return {"ok": False, "error": f"project_path does not exist: {root}"}
    ensure_workspace_memory(str(root))
    p = root / "JARVIS.md"
    content = p.read_text(encoding="utf-8")
    previous_now = _section_body(content, "NOW")
    previous_now = _remove_interrupt_checkpoint(previous_now).strip()
    checkpoint = _format_interrupt_checkpoint(
        user_message=user_message,
        assistant_message=assistant_message,
        tool_events=tool_events or [],
        subturn_log=subturn_log,
        mode=mode,
        cwd=cwd,
        reason=reason,
    )
    new_now = checkpoint if not previous_now else f"{checkpoint}\n\n{previous_now}"
    return update_jarvis_md(str(root), field="NOW", value=new_now)

def clear_interrupt_checkpoint(project_path: str | None) -> dict[str, Any]:
    """Remove the transient interrupted-turn block from JARVIS.md NOW."""
    if not project_path:
        return {"ok": False, "error": "project_path is required"}
    root = Path(project_path).expanduser()
    if not root.exists() or not root.is_dir():
        return {"ok": False, "error": f"project_path does not exist: {root}"}
    ensure_workspace_memory(str(root))
    p = root / "JARVIS.md"
    content = p.read_text(encoding="utf-8")
    previous_now = _section_body(content, "NOW")
    if INTERRUPT_CHECKPOINT_BEGIN not in previous_now:
        return {
            "ok": True,
            "fields": [],
            "path": str(p),
            "bytes": len(content),
            "unchanged": True,
        }
    cleaned_now = _remove_interrupt_checkpoint(previous_now).strip()
    return update_jarvis_md(str(root), field="NOW", value=cleaned_now)

def _section_body(content: str, field: str) -> str:
    field_upper = _normalize_section_field(field)
    heading = _section_heading_pattern(field_upper)
    section_re = re.compile(
        rf"^## {heading}(?:[ \t]+—[^\n]*)?[ \t]*\n([\s\S]*?)(?=^## |\Z)",
        re.MULTILINE | re.IGNORECASE,
    )
    match = section_re.search(content)
    return match.group(1) if match else ""


def _normalize_section_field(field: Any) -> str:
    return re.sub(r"[\s-]+", "_", str(field).strip().upper())


def _section_heading_pattern(field_upper: str) -> str:
    if field_upper == DESIGN_BRIEF_SECTION:
        return r"Design[ \t_-]+Brief"
    return re.escape(field_upper)


def _section_regex(field_upper: str) -> re.Pattern[str]:
    heading = _section_heading_pattern(field_upper)
    return re.compile(
        rf"(^## {heading}(?:[ \t]+—[^\n]*)?[ \t]*\n)([\s\S]*?)(?=^## |\Z)",
        re.MULTILINE | re.IGNORECASE,
    )

def _remove_interrupt_checkpoint(text: str) -> str:
    checkpoint_re = re.compile(
        rf"{re.escape(INTERRUPT_CHECKPOINT_BEGIN)}[\s\S]*?{re.escape(INTERRUPT_CHECKPOINT_END)}\s*",
        re.MULTILINE,
    )
    return checkpoint_re.sub("", text)

def _format_interrupt_checkpoint(
    *,
    user_message: str,
    assistant_message: str,
    tool_events: list[dict[str, Any]],
    subturn_log: str,
    mode: str,
    cwd: str | None,
    reason: str,
) -> str:
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    lines = [
        INTERRUPT_CHECKPOINT_BEGIN,
        "### Interrupted Turn Checkpoint",
        "",
        "Status: interrupted by user ESC. Current work state was saved automatically.",
        f"Saved at: {now}",
        f"Reason: {_one_line(reason, 120)}",
    ]
    if mode:
        lines.append(f"Mode: {_one_line(mode, 80)}")
    if cwd:
        lines.append(f"CWD: {_one_line(cwd, 240)}")
    if user_message.strip():
        lines.extend(["", "User request:", _block(user_message, 1600)])
    if assistant_message.strip():
        lines.extend(["", "Assistant partial work/state:", _block(assistant_message, 4000)])
    tool_lines = _format_tool_events(tool_events)
    if tool_lines:
        lines.extend(["", "Tool work observed:", *tool_lines])
    if subturn_log.strip():
        lines.extend(["", "Current turn subturn log:", _block(subturn_log, 8000)])
    lines.extend(
        [
            "",
            "Resume guidance:",
            "- Treat this as the latest interrupted working state.",
            "- Continue from here unless the user gives a newer direction.",
            INTERRUPT_CHECKPOINT_END,
        ]
    )
    return "\n".join(lines).rstrip()

def _format_tool_events(tool_events: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for event in tool_events[-8:]:
        results = event.get("toolResults")
        if not isinstance(results, list):
            continue
        for result in results[-8:]:
            if not isinstance(result, dict):
                continue
            name = _one_line(str(result.get("toolName") or "tool"), 80)
            state = "error" if result.get("isError") else "ok"
            text = _one_line(str(result.get("text") or ""), 260)
            out.append(f"- {name}: {state}{f' - {text}' if text else ''}")
    return out[-12:]

def _one_line(value: str, limit: int) -> str:
    text = " ".join(str(value).replace("\r", "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."

def _block(value: str, limit: int) -> str:
    text = str(value).replace("\r", "").strip()
    if len(text) > limit:
        text = text[: max(0, limit - 20)].rstrip() + "\n...[truncated]"
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())
