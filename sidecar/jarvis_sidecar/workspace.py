from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import get_effective_project_root, internal_memory_root, is_protected_path
from .file_locks import cross_process_file_lock


@dataclass
class WorkspaceProject:
    project_id: str
    name: str
    slug: str
    path: str
    code_path: str | None = None


class RegistryCorruptError(RuntimeError):
    pass


class InvalidProjectNameError(ValueError):
    pass


def workspace_root() -> Path:
    return internal_memory_root()


def registry_path() -> Path:
    return workspace_root() / "workspace_registry.json"


class WorkspaceRegistry:
    def __init__(self) -> None:
        self.root = workspace_root()
        self.path = registry_path()
        self._data: dict[str, dict[str, Any]] = {}
        self._load_error: str | None = None
        self._stat_sig: tuple[int, int] | None = None
        self._load()

    def _current_stat_sig(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _load(self) -> None:
        if not self.path.exists():
            self._data = {}
            self._load_error = None
            self._stat_sig = None
            return
        try:
            # utf-8-sig: PowerShell-created files carry a BOM by default on
            # Windows; strict utf-8 made the registry unreadable and blocked
            # register_project until manual repair (live, 2026-06-12).
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("registry root must be a JSON object")
            self._data = self._dedupe_loaded_data(raw)
            self._load_error = None
        except Exception as exc:
            self._data = {}
            self._load_error = f"{type(exc).__name__}: {exc}"
        self._stat_sig = self._current_stat_sig()

    def _dedupe_loaded_data(self, raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
        deduped: dict[str, dict[str, Any]] = {}
        index: dict[tuple[str, str], str] = {}
        project_keys: dict[str, set[tuple[str, str]]] = {}
        for project_id, payload in raw.items():
            if not isinstance(project_id, str) or not isinstance(payload, dict):
                continue
            project = self._project_from_raw(project_id, payload)
            if project is None:
                continue
            keys = _dedupe_keys(project)
            for key in keys:
                existing_id = index.get(key)
                if not existing_id or existing_id == project_id:
                    continue
                deduped.pop(existing_id, None)
                for existing_key in project_keys.pop(existing_id, set()):
                    if index.get(existing_key) == existing_id:
                        index.pop(existing_key, None)
            deduped[project_id] = asdict(project)
            project_keys[project_id] = keys
            for key in keys:
                index[key] = project_id
        return deduped

    def _maybe_reload(self) -> None:
        if self._current_stat_sig() != self._stat_sig:
            self._load()

    @property
    def is_corrupt(self) -> bool:
        return self._load_error is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def status_fields(self) -> dict[str, Any]:
        self._maybe_reload()
        return {
            "registry_ok": not self.is_corrupt,
            "registry_path": str(self.path),
            "registry_error": self._load_error,
            "registry_project_count": 0 if self.is_corrupt else len(self._data),
        }

    def save(self) -> None:
        with cross_process_file_lock(self.path):
            if self._current_stat_sig() != self._stat_sig:
                self._load()
            self._raise_if_corrupt()
            self._save_unlocked()

    def _save_unlocked(self) -> None:
        text = json.dumps(self._data, ensure_ascii=False, indent=2)
        json.loads(text)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", errors="replace", newline="") as fh:
                fh.write(text + "\n")
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp_name, str(self.path))
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self._stat_sig = self._current_stat_sig()

    def _raise_if_corrupt(self) -> None:
        if self.is_corrupt:
            raise RegistryCorruptError(
                f"workspace registry is corrupt; refusing to overwrite {self.path}: {self._load_error}"
            )

    def _assert_writable(self) -> None:
        self._maybe_reload()
        self._raise_if_corrupt()

    def remove_project(self, project_id: str) -> bool:
        with cross_process_file_lock(self.path):
            self._load()
            self._raise_if_corrupt()
            if project_id not in self._data:
                return False
            self._data.pop(project_id, None)
            self._save_unlocked()
            return True

    def all(self) -> list[WorkspaceProject]:
        self._maybe_reload()
        projects: list[WorkspaceProject] = []
        for pid, raw in self._data.items():
            try:
                projects.append(
                    WorkspaceProject(
                        project_id=pid,
                        name=str(raw.get("name") or pid),
                        slug=str(raw.get("slug") or pid),
                        path=str(raw.get("path") or ""),
                        code_path=raw.get("code_path"),
                    )
                )
            except Exception:
                continue
        return projects

    def get_by_path(self, path: str | None) -> WorkspaceProject | None:
        self._maybe_reload()
        if not path:
            return None
        resolved = _resolve_path_str(path)
        if resolved is None:
            return None
        for project in self.all():
            project_path = _resolve_path_str(project.path)
            code_path = _resolve_path_str(project.code_path)
            if resolved == project_path or resolved == code_path:
                return project
        return None

    def get_by_id(self, project_id: str | None) -> WorkspaceProject | None:
        self._maybe_reload()
        if not project_id:
            return None
        raw = self._data.get(project_id)
        if not isinstance(raw, dict):
            return None
        try:
            return WorkspaceProject(
                project_id=project_id,
                name=str(raw.get("name") or project_id),
                slug=str(raw.get("slug") or project_id),
                path=str(raw.get("path") or ""),
                code_path=raw.get("code_path"),
            )
        except Exception:
            return None

    def get_by_slug_or_name(self, value: str | None) -> list[WorkspaceProject]:
        needle = (value or "").strip().casefold()
        if not needle:
            return []
        hits: list[WorkspaceProject] = []
        for project in self.all():
            if (
                project.project_id.casefold() == needle
                or project.slug.casefold() == needle
                or project.name.casefold() == needle
            ):
                hits.append(project)
        return hits

    def match(self, utterance: str) -> list[WorkspaceProject]:
        # Word-boundary matching: "tetris" must not match inside "3d-tetris",
        # "workspace" must not match inside "jarvis_workspace". `re.ASCII`
        # keeps `\w` to [A-Za-z0-9_] so adjacent Korean ("tetris를") still
        # counts as a boundary; `-` is added explicitly because compound
        # slugs like `3d-tetris` must stay atomic.
        text = (utterance or "").casefold()
        hits: list[WorkspaceProject] = []
        for project in self.all():
            name = project.name.casefold()
            slug = project.slug.casefold()
            matched = project.project_id.casefold() in text
            if not matched and _is_matchable_project_token(slug):
                slug_pat = rf"(?<![\w-]){re.escape(slug)}(?![\w-])"
                matched = re.search(slug_pat, text, re.ASCII) is not None
            if not matched and name != slug and _is_matchable_project_token(name):
                name_pat = rf"(?<![\w-]){re.escape(name)}(?![\w-])"
                matched = re.search(name_pat, text, re.ASCII) is not None
            if matched:
                hits.append(project)
        return hits

    def create_or_get(self, name: str, *, code_path: str | None = None) -> WorkspaceProject:
        with cross_process_file_lock(self.path):
            self._load()
            self._raise_if_corrupt()
            slug = slugify(name)
            _raise_if_degenerate_project_slug(slug, name)
            resolved_code_path = code_path
            if resolved_code_path is None:
                resolved_code_path, _warnings = resolve_code_path(name)
            elif is_protected_path(resolved_code_path):
                redirected, _warnings = resolve_code_path(name, explicit=resolved_code_path)
                resolved_code_path = redirected
            existing_project_id, existing_project = self._find_existing_by_slug_or_path(slug, resolved_code_path)
            if existing_project is not None and existing_project_id is not None:
                if resolved_code_path and not existing_project.code_path:
                    existing_project.path = resolved_code_path
                    existing_project.code_path = resolved_code_path
                    self._data[existing_project_id] = asdict(existing_project)
                    self._save_unlocked()
                return existing_project
            project_id = self._project_id_for(slug, resolved_code_path)
            existing = self._data.get(project_id)
            if existing:
                project = self._project_from_raw(project_id, existing)
                if project is None:
                    self._data.pop(project_id, None)
                else:
                    if resolved_code_path and project.code_path != resolved_code_path:
                        project.code_path = resolved_code_path
                        self._data[project_id] = asdict(project)
                        self._save_unlocked()
                    return project

            if resolved_code_path:
                Path(resolved_code_path).mkdir(parents=True, exist_ok=True)
            project = WorkspaceProject(
                project_id=project_id,
                name=name,
                slug=slug,
                path=resolved_code_path or str(self.root / slug),
                code_path=resolved_code_path,
            )
            self._data[project_id] = asdict(project)
            self._save_unlocked()
            return project

    def _find_existing_by_slug_or_path(
        self,
        slug: str,
        code_path: str | None,
    ) -> tuple[str | None, WorkspaceProject | None]:
        requested_path = _resolve_path_str(code_path)
        slug_key = slug.casefold()
        for project_id, raw in list(self._data.items()):
            if not isinstance(raw, dict):
                continue
            project = self._project_from_raw(project_id, raw)
            if project is None:
                continue
            if project.slug.casefold() == slug_key:
                return project_id, project
            if requested_path and requested_path in {
                _resolve_path_str(project.code_path),
                _resolve_path_str(project.path),
            }:
                return project_id, project
        return None, None

    def _project_from_raw(self, project_id: str, raw: dict[str, Any]) -> WorkspaceProject | None:
        try:
            return WorkspaceProject(
                project_id=project_id,
                name=str(raw.get("name") or project_id),
                slug=str(raw.get("slug") or project_id),
                path=str(raw.get("path") or ""),
                code_path=raw.get("code_path"),
            )
        except Exception:
            return None

    def _project_id_for(self, slug: str, code_path: str | None) -> str:
        base = f"{slug}-{hashlib.sha1(slug.encode('utf-8')).hexdigest()[:6]}"
        existing = self._data.get(base)
        if existing is None:
            return base
        existing_code = _resolve_path_str(existing.get("code_path"))
        requested_code = _resolve_path_str(code_path)
        if existing_code == requested_code or not requested_code:
            return base
        return f"{slug}-{hashlib.sha1(requested_code.encode('utf-8')).hexdigest()[:6]}"


def resolve_code_path(name: str, *, explicit: str | None = None) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    slug = slugify(name)
    default_root = get_effective_project_root()

    if explicit:
        explicit_resolved = _resolve_path_str(explicit)
        if explicit_resolved and not is_protected_path(explicit_resolved):
            return explicit_resolved, warnings
        if default_root:
            redirected = str(Path(default_root) / slug)
            warnings.append(
                f"requested path is inside protected_roots; redirected to default_project_root: {redirected}"
            )
            return redirected, warnings
        warnings.append("no safe project root is available for protected path redirection")
        return None, warnings

    if not default_root:
        warnings.append("no project root is available for new project folder creation")
        return None, warnings
    return str(Path(default_root) / slug), warnings


def parse_project_switch_command(utterance: str) -> str | None:
    request = parse_project_switch_request(utterance)
    return request["slug_or_name"] if request else None


def parse_project_switch_request(utterance: str) -> dict[str, str | bool] | None:
    text = (utterance or "").strip()
    if not text:
        return None
    match = re.match(r"^/project\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    if not value:
        return None
    auto_create = False
    code_path: str | None = None

    code_path_match = re.search(r"\s+--code-path\s+(.+?)\s*$", value, flags=re.IGNORECASE)
    if code_path_match:
        code_path = code_path_match.group(1).strip()
        value = value[: code_path_match.start()].strip()
        quoted_code_path = re.fullmatch(r"[\"'“‘](.+?)[\"'”’]", code_path)
        if quoted_code_path:
            code_path = quoted_code_path.group(1).strip()

    if re.search(r"\s+--new\s*$", value, flags=re.IGNORECASE):
        auto_create = True
        value = re.sub(r"\s+--new\s*$", "", value, flags=re.IGNORECASE).strip()

    quoted = re.fullmatch(r"[\"'“‘](.+?)[\"'”’]", value)
    if quoted:
        value = quoted.group(1).strip()
    if not value:
        return None
    payload: dict[str, str | bool] = {"slug_or_name": value[:120], "auto_create": auto_create}
    if code_path:
        payload["code_path"] = code_path
    return payload


def parse_setup_default_root_command(utterance: str) -> str | None:
    text = (utterance or "").strip()
    if not text:
        return None
    match = re.match(r"^/setup-default-root\s+(.+?)\s*$", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = match.group(1).strip()
    quoted = re.fullmatch(r"[\"'“‘](.+?)[\"'”’]", value)
    if quoted:
        value = quoted.group(1).strip()
    return value or None


def slugify(name: str) -> str:
    ascii_name = name.encode("ascii", errors="ignore").decode("ascii")
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", ascii_name.casefold()).strip("-")
    if slug:
        return slug[:60]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"project-{digest}"


_COMMON_DEGENERATE_SLUGS = frozenset({
    "app",
    "web",
})


def _is_degenerate_project_slug(slug: str) -> bool:
    value = str(slug or "").strip().casefold()
    if not value:
        return True
    if value in _COMMON_DEGENERATE_SLUGS:
        return True
    if value.isdigit():
        return True
    if len(value) < 3:
        return True
    return False


def _raise_if_degenerate_project_slug(slug: str, name: str) -> None:
    if _is_degenerate_project_slug(slug):
        raise InvalidProjectNameError(
            f"project name {name!r} is too ambiguous; use a more specific project name"
        )


def _is_matchable_project_token(value: str) -> bool:
    token = str(value or "").strip().casefold()
    if _is_degenerate_project_slug(token):
        return False
    # Single-token generic names create false project matches in normal prose.
    # Compound slugs like "web-game" or "3d-tetris" remain matchable.
    if re.fullmatch(r"[a-z]+", token) and token in _COMMON_DEGENERATE_SLUGS:
        return False
    return True


def _dedupe_keys(project: WorkspaceProject) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    slug = str(project.slug or "").strip().casefold()
    if slug:
        keys.add(("slug", slug))
    for path in (project.code_path, project.path):
        resolved = _resolve_path_str(path)
        if resolved:
            keys.add(("path", resolved))
    return keys


def _resolve_path_str(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return None
