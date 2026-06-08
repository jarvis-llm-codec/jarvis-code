"""Switch the active project so subsequent turns route memory/context to it.

Use this when the user's utterance refers to a project that is NOT the current
`active_memory_project_path` shown in the JARVIS Code Memory block. Typical
trigger: user says "let's continue the landing page" while the sticky active
is still "tetris" from a previous session — call this tool to flip active to
the matching registered project, then proceed with the user's request.

Idempotent. If `slug_or_name` is already active, this is a no-op that returns
the same project info. Pass `auto_create=True` only when you have explicit
user intent to register a NEW project — otherwise let `register_project`
own creation and use this tool to switch among already-known projects.
"""
from __future__ import annotations

from typing import Any


SCHEMA = {
    "type": "function",
    "function": {
        "name": "switch_project",
        "description": (
            "Flip the active project for subsequent turns when the user's "
            "current utterance refers to a different registered project than "
            "the one currently shown as `active_memory_project_path`. "
            "Idempotent. Returns the resolved project info. Use this for "
            "switching among already-registered projects — call "
            "`register_project` first if the target project is not yet "
            "registered."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slug_or_name": {
                    "type": "string",
                    "description": (
                        "The registered project's slug (e.g. 'cosmetics-landing') "
                        "or human name. Matched case-insensitively against "
                        "project_id, slug, and name in the workspace registry."
                    ),
                },
                "code_path": {
                    "type": "string",
                    "description": (
                        "Optional. Update the project's code_path when switching. "
                        "Use this if the project's code folder moved or was "
                        "initially registered without a code_path."
                    ),
                },
                "auto_create": {
                    "type": "boolean",
                    "description": (
                        "Default false. Set true ONLY when the user has explicitly "
                        "asked to start a new project and you have not already "
                        "called register_project for it."
                    ),
                },
            },
            "required": ["slug_or_name"],
            "additionalProperties": False,
        },
    },
}


def handler(
    slug_or_name: str,
    code_path: str | None = None,
    auto_create: bool = False,
    project_root: str | None = None,
) -> dict[str, Any]:
    """Switch active project via sidecar's in-memory ProjectRouter so the
    change is visible to the very next /context call (no process restart).

    `project_root` is dispatcher-injected and ignored — this tool intentionally
    operates above any single project root.
    """
    try:
        from jarvis_sidecar.app import router as sidecar_router
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"sidecar router unavailable: {exc}"}

    if not isinstance(slug_or_name, str) or not slug_or_name.strip():
        return {"ok": False, "error": "slug_or_name must be a non-empty string"}

    project, warnings = sidecar_router.switch_project(
        slug_or_name.strip(),
        code_path=code_path,
        auto_create=bool(auto_create),
    )
    if project is None:
        return {
            "ok": False,
            "error": f"unknown project: {slug_or_name}",
            "warnings": warnings,
            "hint": "Call register_project first, or pass auto_create=true with explicit user intent.",
        }

    try:
        from jarvis_sidecar.memory_files import ensure_workspace_memory

        memory_files = ensure_workspace_memory(project.path)
    except Exception:
        memory_files = {}

    return {
        "ok": True,
        "project_id": project.project_id,
        "name": project.name,
        "slug": project.slug,
        "memory_path": project.path,
        "code_path": project.code_path,
        "warnings": warnings,
        "memory_files": memory_files,
    }
