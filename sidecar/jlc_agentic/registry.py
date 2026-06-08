"""Project registry for jarvis-code.

Single conversation + N projects. Project identity = `<folder>-<sha1[:6]>`.
Registry stores `{project_id: {"name": str, "path": str}}` mappings.
Matching is utterance-only (no cwd auto-detection — JARVIS philosophy:
the chat is the companion, the project follows the words).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _is_link_or_junction(p: Path) -> bool:
    """Detect both POSIX symlinks and Windows directory junctions.

    pathlib.is_symlink() returns False for Windows junctions even though they
    are reparse points. We need both because Jun runs on Windows.
    """
    try:
        if p.is_symlink():
            return True
    except OSError:
        return False
    if os.name == "nt":
        try:
            attrs = p.lstat().st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except (OSError, AttributeError):
            return False
    return False


def _atomic_write_text(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    # Owner-only (0o600) on POSIX. Windows lacks os.fchmod; ACLs handle this.
    try:
        os.fchmod(fd, 0o600)
    except (AttributeError, OSError):
        pass
    try:
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


@dataclass
class ProjectEntry:
    project_id: str
    name: str
    path: str


class Registry:
    """`~/.jarvis-code/registry.json` reader/writer + utterance matcher."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path).expanduser() if path else Path("~/.jarvis-code/registry.json").expanduser()
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            raw_text = self._path.read_text(encoding="utf-8")
            raw = json.loads(raw_text)
            self._data = raw if isinstance(raw, dict) else {}
            # Snapshot a backup of every successfully loaded registry so a
            # corrupt write later can be recovered from.
            self._snapshot_backup(raw_text)
        except Exception as exc:
            recovered = self._try_recover_from_backup()
            if recovered:
                print(
                    f"[jlc:registry] WARNING: registry.json corrupt ({exc}); "
                    f"recovered {len(recovered)} entries from backup",
                    file=sys.stderr,
                )
                self._data = recovered
            else:
                print(
                    f"[jlc:registry] ERROR: registry.json corrupt ({exc}); "
                    f"no backup available — starting empty",
                    file=sys.stderr,
                )
                self._data = {}

    def _backup_dir(self) -> Path:
        return self._path.parent / ".registry_backups"

    def _snapshot_backup(self, raw_text: str, keep: int = 5) -> None:
        try:
            backup_dir = self._backup_dir()
            backup_dir.mkdir(parents=True, exist_ok=True)
            from datetime import UTC, datetime
            ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            (backup_dir / f"registry_{ts}.json").write_text(raw_text, encoding="utf-8")
            backups = sorted(backup_dir.glob("registry_*.json"), reverse=True)
            for old in backups[keep:]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except OSError:
            pass

    def _try_recover_from_backup(self) -> dict[str, dict[str, str]]:
        backup_dir = self._backup_dir()
        if not backup_dir.exists():
            return {}
        for bp in sorted(backup_dir.glob("registry_*.json"), reverse=True):
            try:
                raw = json.loads(bp.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
            except Exception:
                continue
        return {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self._path, json.dumps(self._data, ensure_ascii=False, indent=2))

    def all(self) -> list[ProjectEntry]:
        return [
            ProjectEntry(pid, v.get("name", pid), v.get("path", ""))
            for pid, v in self._data.items()
        ]

    def get(self, project_id: str) -> ProjectEntry | None:
        v = self._data.get(project_id)
        if not v:
            return None
        return ProjectEntry(project_id, v.get("name", project_id), v.get("path", ""))

    def add(self, name: str, path: str | Path, allow_overwrite: bool = False) -> ProjectEntry:
        raw_path = Path(path).expanduser()
        if _is_link_or_junction(raw_path):
            print(
                f"[jlc:registry] WARNING: '{raw_path}' is a symlink or junction; "
                f"storing as-is without resolving target",
                file=sys.stderr,
            )
            path_str = str(raw_path.absolute())
        else:
            path_str = str(raw_path.resolve())
        project_id = self._mint_id(name)
        if project_id in self._data and not allow_overwrite:
            existing = self._data[project_id]
            if existing.get("path") != path_str or existing.get("name") != name:
                print(
                    f"[jlc:registry] WARNING: project_id collision for '{name}' "
                    f"(existing path={existing.get('path')!r}); pass allow_overwrite=True to replace",
                    file=sys.stderr,
                )
                return ProjectEntry(project_id, existing.get("name", name), existing.get("path", ""))
        self._data[project_id] = {"name": name, "path": path_str}
        self.save()
        return ProjectEntry(project_id, name, path_str)

    def remove(self, project_id: str) -> bool:
        if project_id in self._data:
            del self._data[project_id]
            self.save()
            return True
        return False

    _TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)

    @classmethod
    def tokenize(cls, text: str, min_len: int = 2) -> set[str]:
        """Public tokenization helper — same regex `match()` uses internally,
        exposed so callers (Tier 3.5 disjoint guard) don't reach into the
        private `_TOKEN_RE`. `min_len` defaults to 2 (Tier 3.5) but `match()`
        keeps its strict 3 internally to preserve existing semantics."""
        if not text:
            return set()
        return {t.lower() for t in cls._TOKEN_RE.findall(text) if len(t) >= min_len}

    def match(self, utterance: str) -> list[ProjectEntry]:
        """Return projects whose name (token-level) or project_id appears in the utterance.

        Token-level word-boundary match — avoids "ai" matching "I'm using AI" type
        false positives. Names with <2 chars never match (too generic).
        Returns 0/1/N candidates — caller decides how to respond.
        """
        if not utterance:
            return []
        utt_tokens = {t.lower() for t in self._TOKEN_RE.findall(utterance) if len(t) >= 3}
        text_lower = utterance.lower()
        hits: list[ProjectEntry] = []
        for pid, v in self._data.items():
            name = v.get("name", "")
            if not name or len(name) < 3:
                continue
            name_tokens = [t.lower() for t in self._TOKEN_RE.findall(name) if len(t) >= 3]
            name_lower = name.lower()
            matched = False
            # Tier 1: all-tokens token-level match — strongest, avoids "ai" type
            # false positives in English.
            if name_tokens and utt_tokens and all(t in utt_tokens for t in name_tokens):
                matched = True
            # Tier 2: project_id substring match.
            elif pid.lower() in text_lower:
                matched = True
            # Tier 3: substring fallback for languages whose token boundaries
            # don't map cleanly onto `\w+` (Korean, Chinese, Japanese). Keeps
            # the all-tokens path strict for English while letting "테트리스"
            # match a project named "테트리스 게임".
            elif name_lower in text_lower:
                matched = True
            else:
                # Tier 4: any name-token whose length>=3 appears literally — keeps
                # multi-word Korean names matchable with one keyword.
                for t in name_tokens:
                    if t in text_lower:
                        matched = True
                        break
            if matched:
                hits.append(ProjectEntry(pid, name, v.get("path", "")))
        return hits

    @staticmethod
    def _mint_id(name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "project"
        digest = hashlib.sha1(name.encode("utf-8", errors="replace")).hexdigest()[:6]
        return f"{slug}-{digest}"
