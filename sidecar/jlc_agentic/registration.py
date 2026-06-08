"""First-registration flow for jarvis-code.

When the user mentions a project that registry.match() returns 0 hits for,
this module:
  1. Asks for the path (caller decides how — chat / prompt / arg).
  2. Scans the folder for language hints + top-level structure.
  3. Drafts a JARVIS.md MAP skeleton — other axes reserved for the
     first encoder turn.
  4. Returns a `ScanReport` so the caller can confirm with Jun before
     committing to registry.add() + JARVIS.md write.

This is heuristic-only — no LLM call. The encoder fills the remaining axes
lazily once real conversation turns arrive. Intentionally cheap so registry
creation feels instant.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# (extension → label) — order matters for primary-language inference.
_LANG_EXT_MAP: list[tuple[str, str]] = [
    (".py", "Python"),
    (".ts", "TypeScript"),
    (".tsx", "TypeScript"),
    (".js", "JavaScript"),
    (".jsx", "JavaScript"),
    (".rs", "Rust"),
    (".go", "Go"),
    (".java", "Java"),
    (".kt", "Kotlin"),
    (".swift", "Swift"),
    (".c", "C"),
    (".h", "C"),
    (".cpp", "C++"),
    (".cc", "C++"),
    (".cs", "C#"),
    (".rb", "Ruby"),
    (".php", "PHP"),
    (".lua", "Lua"),
    (".sh", "Shell"),
    (".ps1", "PowerShell"),
    (".sql", "SQL"),
]

_MARKER_FILES = {
    "package.json": "Node.js",
    "pnpm-lock.yaml": "Node.js (pnpm)",
    "yarn.lock": "Node.js (yarn)",
    "pyproject.toml": "Python (PEP-621)",
    "requirements.txt": "Python",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "Gemfile": "Ruby",
    "composer.json": "PHP",
    "Dockerfile": "Docker",
    "docker-compose.yml": "Docker Compose",
    ".gitignore": "git",
    "Makefile": "Make",
    "CMakeLists.txt": "CMake",
    "tsconfig.json": "TypeScript config",
    "next.config.js": "Next.js",
    "next.config.mjs": "Next.js",
    "vite.config.ts": "Vite",
    "vite.config.js": "Vite",
}

# Directories we never descend into — vendored, generated, or huge.
_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", "target", ".next", ".cache", ".idea", ".vscode",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "coverage",
    "out", ".tox", "vendor",
}

_MAX_SCAN_FILES = 5000


def _is_symlink_loop(entry: Path) -> bool:
    """Detect if entry is a symlink/junction that points back to an ancestor."""
    try:
        if not entry.is_dir():
            return False
        # Resolve to absolute and check if it's under the original root
        resolved = entry.resolve()
        # Simple ancestor check — if resolved path starts with a previously-seen
        # prefix that is not the immediate parent, it's likely a loop.
        # For full correctness we'd track visited inodes, but this catches
        # the common "symlink to parent" case.
        if resolved == entry.parent.resolve():
            return True  # Points to parent → loop
        return False
    except OSError:
        return False  # Permission denied or broken link — skip safely


@dataclass
class ScanReport:
    name: str
    path: str
    primary_languages: list[str] = field(default_factory=list)
    file_count: int = 0
    extension_top: list[tuple[str, int]] = field(default_factory=list)
    top_dirs: list[str] = field(default_factory=list)
    markers: list[str] = field(default_factory=list)
    truncated: bool = False

    def to_jarvis_md(self) -> str:
        """Render MAP skeleton + empty canonical JARVIS.md axes."""
        langs = ", ".join(self.primary_languages) or "(unknown)"
        markers = ", ".join(self.markers) or "(none)"
        ext_lines = "\n".join(
            f"  - `{ext}`: {count}" for ext, count in self.extension_top
        ) or "  - (none)"
        dir_lines = "\n".join(f"  - `{d}/`" for d in self.top_dirs) or "  - (none)"

        return (
            f"# JARVIS.md — {self.name}\n\n"
            "_Auto-drafted by jarvis-code first-registration. Keep sections compact, operational, and trigger-based._\n\n"
            "## NOW — Current Active Task\n"
            "- Status: no active task yet.\n"
            "- Last verified: not yet.\n"
            "Next: wait for a concrete project request.\n\n"
            "## MAP — Project Map and Symbol Index\n"
            f"- **Path**: `{self.path}`\n"
            f"- **Primary languages**: {langs}\n"
            f"- **File count (scanned)**: {self.file_count}"
            f"{' (truncated)' if self.truncated else ''}\n"
            f"- **Top extensions**:\n{ext_lines}\n"
            f"- **Top-level directories**:\n{dir_lines}\n"
            f"- **Markers**: {markers}\n\n"
            "## LAW — Learned Agent Warnings\n\n"
            "- Format: `LAW-001: Trigger -> Rule -> Verify`.\n"
            "- Use for hard project invariants that must stay true on future edits.\n\n"
            "## BAN — Forbidden Actions\n\n"
            "- Format: `BAN-001: Never <action>; because <failure>; verify <check>`.\n"
            "- Use for known-dangerous actions, not generic caution.\n\n"
            "## HABIT — User and Project Preferences\n\n"
            "- Format: `HABIT-001: When <situation>, prefer <style/workflow>`.\n"
            "- Use for user/project preferences that affect future choices.\n\n"
            "## WHY — Why History Yells (Decision Rationale)\n\n"
            "- Record decision rationale only: `Decision -> Why -> Tradeoff`.\n"
            "- Do not duplicate changelog, NOW, or RAW evidence.\n\n"
            "## OMM — Oh My Mistake (Failure Retrospectives)\n\n"
            "OMM entries are operational mistake-prevention rules, not apologies.\n"
            "Use this exact shape:\n"
            "### OMM-001: Short title\n"
            "- Trigger: When this rule must be recalled.\n"
            "- Mistake: What failed before, concretely.\n"
            "- Rule: What must/never happen next time.\n"
            "- Required action: What to inspect or change before proceeding.\n"
            "- Verify: Command, test, log, or observable check.\n\n"
            "## RAW — Raw Evidence Pointers\n"
            "- Evidence pointers only: date, request, files changed, commands run, test result, turn id if known.\n"
            "- Do not paste transcripts or long explanations here.\n"
        )


def scan_project(path: str | Path, name: str | None = None) -> ScanReport:
    """Walk `path` (skipping vendored/generated dirs) and return a ScanReport.

    Cheap heuristic — no LLM. Stops after _MAX_SCAN_FILES to keep registration
    feeling instant on monorepos.
    """
    root = Path(path).expanduser().resolve()
    inferred_name = name or root.name or "project"

    if not root.exists() or not root.is_dir():
        return ScanReport(name=inferred_name, path=str(root))

    ext_counter: Counter[str] = Counter()
    top_dirs: list[str] = []
    markers: list[str] = []
    file_count = 0
    truncated = False

    try:
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and entry.name not in _SKIP_DIRS and not entry.name.startswith("."):
                top_dirs.append(entry.name)
        top_dirs = top_dirs[:12]

        for candidate, label in _MARKER_FILES.items():
            if (root / candidate).exists():
                markers.append(label)
    except OSError:
        pass

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        # Filter skip dirs BEFORE os.walk descends — prevents symlink loops
        # and reduces filesystem traversal.
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not d.startswith(".")
            and not _is_symlink_loop(Path(dirpath) / d)
        ]
        for fname in filenames:
            file_count += 1
            if file_count > _MAX_SCAN_FILES:
                truncated = True
                break
            ext = Path(fname).suffix.lower()
            if ext:
                ext_counter[ext] += 1
        if truncated:
            break

    primary_languages = _infer_languages(ext_counter)

    return ScanReport(
        name=inferred_name,
        path=str(root),
        primary_languages=primary_languages,
        file_count=min(file_count, _MAX_SCAN_FILES),
        extension_top=ext_counter.most_common(8),
        top_dirs=top_dirs,
        markers=markers,
        truncated=truncated,
    )


def _infer_languages(ext_counter: Counter[str]) -> list[str]:
    """Return up to 3 distinct language labels ranked by file count."""
    label_counts: Counter[str] = Counter()
    for ext, count in ext_counter.items():
        for known_ext, label in _LANG_EXT_MAP:
            if ext == known_ext:
                label_counts[label] += count
                break
    return [label for label, _ in label_counts.most_common(3)]


def initialize_jarvis_md(project_path: str | Path, scan: ScanReport) -> Path:
    """Write the JARVIS.md skeleton to <project_path>/JARVIS.md.

    Refuses to overwrite an existing JARVIS.md — caller must decide whether to
    delete it first or keep the existing context.
    """
    target = Path(project_path).expanduser() / "JARVIS.md"
    if target.exists():
        raise FileExistsError(f"{target} already exists; refusing to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(scan.to_jarvis_md(), encoding="utf-8")
    return target


__all__ = ["ScanReport", "scan_project", "initialize_jarvis_md"]
