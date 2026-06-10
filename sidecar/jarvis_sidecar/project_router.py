from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from .file_locks import locked_atomic_write_text
from .workspace import (
    InvalidProjectNameError,
    RegistryCorruptError,
    WorkspaceProject,
    WorkspaceRegistry,
    parse_project_switch_command,
    workspace_root,
)


class ProjectRouter:
    def __init__(self) -> None:
        self.registry = WorkspaceRegistry()
        self._active_path = workspace_root() / "active_project.json"
        self._legacy_active_path = workspace_root() / "active_per_conv.json"
        self._active_updated_at: datetime | None = self._load_active_updated_at()
        self._active_project_id: str | None = self._load_active()
        self._last_resolved_from = "fallback"

    def _load_active(self) -> str | None:
        legacy = self._load_legacy_active()
        if legacy is not None:
            self._save_active(legacy)
            try:
                self._legacy_active_path.rename(self._legacy_active_path.with_suffix(".json.bak"))
            except Exception:
                pass
            return legacy
        if not self._active_path.exists():
            return None
        try:
            raw = json.loads(self._active_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        project_id = raw.get("project_id")
        if not isinstance(project_id, str):
            return None
        return project_id if self.registry.get_by_id(project_id) is not None else None

    def _load_active_updated_at(self) -> datetime | None:
        if not self._active_path.exists():
            return None
        try:
            raw = json.loads(self._active_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict):
            return None
        value = raw.get("updated_at")
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
        except ValueError:
            return None

    def _load_legacy_active(self) -> str | None:
        if not self._legacy_active_path.exists():
            return None
        try:
            raw = json.loads(self._legacy_active_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(raw, dict) or not raw:
            return None
        for project_id in reversed(list(raw.values())):
            if isinstance(project_id, str) and self.registry.get_by_id(project_id) is not None:
                return project_id
        return None

    def _save_active(self, project_id: str | None) -> None:
        self._active_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {}
        if project_id:
            payload["project_id"] = project_id
        if self._active_updated_at is not None:
            payload["updated_at"] = self._active_updated_at.isoformat()
        locked_atomic_write_text(self._active_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _remember_active(self, project: WorkspaceProject) -> None:
        self._active_updated_at = datetime.now(UTC)
        if self._active_project_id == project.project_id:
            self._save_active(project.project_id)
            return
        self._active_project_id = project.project_id
        self._save_active(project.project_id)

    def _clear_active(self) -> None:
        if self._active_project_id is None and self._active_updated_at is None:
            return
        self._active_project_id = None
        self._active_updated_at = None
        self._save_active(None)

    @property
    def active_project_id(self) -> str | None:
        return self._active_project_id

    def clear_active_project(self) -> None:
        self._clear_active()

    def clear_active_project_if(self, project_id: str | None) -> bool:
        if not project_id or self._active_project_id != project_id:
            return False
        self._clear_active()
        return True

    def resolve_by_path(self, candidate_path: str) -> WorkspaceProject | None:
        if not candidate_path:
            return None

        abs_candidate = _normalize_path(candidate_path)
        if not abs_candidate:
            return None

        best: WorkspaceProject | None = None
        best_len = -1
        tied = False
        for project in self.registry.all():
            abs_code = _normalize_path(project.code_path)
            if not abs_code:
                continue
            if not _is_path_within(abs_candidate, abs_code):
                continue
            prefix_len = len(abs_code)
            if prefix_len > best_len:
                best = project
                best_len = prefix_len
                tied = False
            elif prefix_len == best_len:
                tied = True
        return None if tied else best

    def select(
        self,
        *,
        user_message: str,
        cwd_hint: str | None,
        active_project_path: str | None,
        mode: str = "auto",
    ) -> tuple[WorkspaceProject | None, list[str], dict[str, Any]]:
        warnings: list[str] = []
        trace: dict[str, Any] = {"source": "none"}
        chat_mode = mode == "chat"
        if self.registry.is_corrupt:
            warnings.append(f"workspace registry corrupt: {self.registry.load_error}")
            trace.update({"source": "registry_corrupt"})
            self._last_resolved_from = "registry_corrupt"
            return None, warnings, trace

        explicit_switch = parse_project_switch_command(user_message)
        if explicit_switch:
            project, switch_warnings = self.switch_project(explicit_switch, auto_create=False)
            warnings.extend(switch_warnings)
            if project is not None:
                trace.update({"source": "explicit_switch", "path": project.path, "project_id": project.project_id})
                self._last_resolved_from = "explicit_switch"
                return project, warnings, trace
            trace.update({"source": "explicit_switch_unknown", "target": explicit_switch})
            self._last_resolved_from = "fallback"
            return None, warnings, trace

        matches = self.registry.match(user_message)
        if len(matches) == 1:
            project = matches[0]
            self._remember_active(project)
            trace.update({"source": "workspace_registry_match", "path": project.path, "project_id": project.project_id})
            self._last_resolved_from = "workspace_registry_match"
            return project, warnings, trace
        if len(matches) > 1:
            hinted = self._disambiguate_matches_by_hint(matches, active_project_path, cwd_hint)
            if hinted is not None:
                self._remember_active(hinted)
                trace.update(
                    {
                        "source": "ambiguous_registry_match_hint",
                        "path": hinted.path,
                        "project_id": hinted.project_id,
                    }
                )
                self._last_resolved_from = "ambiguous_registry_match_hint"
                return hinted, warnings, trace
            warnings.append("ambiguous project mention; asking for clarification")
            trace.update(
                {
                    "source": "ambiguous_registry_match",
                    "candidates": [
                        {
                            "project_id": m.project_id,
                            "name": m.name,
                            "slug": m.slug,
                            "path": m.path,
                        }
                        for m in matches
                    ],
                }
            )
            self._last_resolved_from = "ambiguous_registry_match"
            return None, warnings, trace

        fallback_path = active_project_path or cwd_hint
        if fallback_path:
            project = self.registry.get_by_path(fallback_path)
            if project:
                self._remember_active(project)
                trace.update({"source": "extension_cwd_fallback", "path": project.path, "project_id": project.project_id})
                self._last_resolved_from = "extension_cwd_fallback"
                return project, warnings, trace
            warnings.append(f"active project path not registered: {fallback_path}")
            trace.update({"source": "fallback", "active_project_path": fallback_path})
            self._last_resolved_from = "fallback"
            return None, warnings, trace

        if chat_mode:
            trace.update({"source": "chat_no_project"})
            self._last_resolved_from = "chat_no_project"
            return None, warnings, trace

        sticky_id = self._active_project_id
        if sticky_id:
            sticky = self.registry.get_by_id(sticky_id)
            if sticky:
                # Sticky is a coding/deepdive fallback only. Chat contexts return
                # before this branch unless the user explicitly mentioned a project.
                trace.update({"source": "sticky_active_project", "path": sticky.path, "project_id": sticky.project_id})
                self._last_resolved_from = "sticky_active_project"
                return sticky, warnings, trace
            self._active_project_id = None
            self._active_updated_at = None
            self._save_active(None)

        trace.update({"source": "fallback"})
        self._last_resolved_from = "fallback"
        return None, warnings, trace

    def _disambiguate_matches_by_hint(
        self,
        matches: list[WorkspaceProject],
        active_project_path: str | None,
        cwd_hint: str | None,
    ) -> WorkspaceProject | None:
        code_paths = {
            normalized
            for normalized in (_normalize_path(project.code_path) for project in matches)
            if normalized is not None
        }
        if len(code_paths) > 1:
            return None

        by_id = {project.project_id: project for project in matches}

        for hint in (active_project_path, cwd_hint):
            hinted = self.registry.get_by_path(hint)
            if hinted is not None and hinted.project_id in by_id:
                return by_id[hinted.project_id]

        if self._active_project_id in by_id:
            return by_id[self._active_project_id]

        for hint in (cwd_hint, active_project_path):
            normalized_hint = _normalize_path(hint)
            if not normalized_hint:
                continue
            candidates: list[WorkspaceProject] = []
            for project in matches:
                code_path = _normalize_path(project.code_path)
                memory_path = _normalize_path(project.path)
                if (code_path and _is_path_within(normalized_hint, code_path)) or (
                    memory_path and _is_path_within(normalized_hint, memory_path)
                ):
                    candidates.append(project)
            if len(candidates) == 1:
                return candidates[0]

        return None

    def create_project(self, name: str, *, code_path: str | None = None) -> WorkspaceProject:
        project = self.registry.create_or_get(name, code_path=code_path)
        return project

    def switch_project(
        self,
        slug_or_name: str,
        *,
        code_path: str | None = None,
        auto_create: bool = False,
    ) -> tuple[WorkspaceProject | None, list[str]]:
        if self.registry.is_corrupt:
            raise RegistryCorruptError(
                f"workspace registry is corrupt; refusing project switch: {self.registry.load_error}"
            )
        warnings: list[str] = []
        exact = self.registry.get_by_slug_or_name(slug_or_name)
        if len(exact) == 1:
            project = exact[0]
            if code_path and project.code_path != code_path:
                project = self.registry.create_or_get(project.name, code_path=code_path)
            self._remember_active(project)
            self._last_resolved_from = "explicit_switch"
            return project, warnings
        if len(exact) > 1:
            warnings.append(f"multiple projects match explicit switch target: {slug_or_name}")
            return None, warnings

        fuzzy = self.registry.match(slug_or_name)
        if len(fuzzy) == 1:
            project = fuzzy[0]
            if code_path and project.code_path != code_path:
                project = self.registry.create_or_get(project.name, code_path=code_path)
            self._remember_active(project)
            self._last_resolved_from = "explicit_switch"
            return project, warnings
        if len(fuzzy) > 1:
            warnings.append(f"multiple projects match explicit switch target: {slug_or_name}")
            return None, warnings

        if not auto_create:
            warnings.append(f"unknown project: {slug_or_name}. Use auto_create=True to register.")
            return None, warnings

        try:
            project = self.registry.create_or_get(slug_or_name, code_path=code_path)
        except InvalidProjectNameError as exc:
            warnings.append(str(exc))
            return None, warnings
        self._remember_active(project)
        self._last_resolved_from = "explicit_switch"
        return project, warnings

    def status_fields(self) -> dict[str, str | None]:
        project = self.registry.get_by_id(self._active_project_id)
        return {
            "active_project_id": project.project_id if project is not None else None,
            "active_project_path": project.path if project is not None else None,
            "active_project_resolved_from": self._last_resolved_from,
        }


def _normalize_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(path)))
    except OSError:
        return None


def _is_path_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False

