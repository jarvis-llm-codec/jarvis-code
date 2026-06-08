from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jarvis_sidecar.workspace import WorkspaceRegistry, _resolve_path_str, workspace_root

BLOCKLIST = {
    "session",
    "auto-prompts",
    "timestamp",
    "default",
    "test",
    "temp",
    "tmp",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or apply JARVIS workspace registry cleanup.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Show cleanup candidates without changing files.")
    mode.add_argument("--apply", action="store_true", help="Remove candidates and write quarantine backup.")
    args = parser.parse_args()

    registry = WorkspaceRegistry()
    candidates: list[dict[str, Any]] = []
    for project in registry.all():
        record = project_to_record(project)
        if is_cleanup_candidate(record):
            candidates.append(record)

    if not candidates:
        print("No cleanup candidates.")
        return 0

    print(f"Cleanup candidates: {len(candidates)}")
    for item in candidates:
        reason_text = ", ".join(item["cleanup_reasons"])
        print(f"- {item['project_id']} | {item['name']} | {item.get('code_path') or item['path']} | {reason_text}")

    if not args.apply:
        print("Dry run only. Re-run with --apply to quarantine and remove these entries.")
        return 0

    quarantine_path = workspace_root() / "workspace_registry_quarantine.json"
    existing_quarantine = load_json(quarantine_path, default=[])
    if not isinstance(existing_quarantine, list):
        existing_quarantine = []

    timestamp = datetime.now(UTC).isoformat()
    existing_quarantine.append({"timestamp": timestamp, "entries": candidates})
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.write_text(json.dumps(existing_quarantine, ensure_ascii=False, indent=2), encoding="utf-8")

    for item in candidates:
        registry.remove_project(item["project_id"])
    print(f"Applied cleanup. Quarantine backup: {quarantine_path}")
    return 0


def project_to_record(project: Any) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "name": project.name,
        "slug": project.slug,
        "path": project.path,
        "code_path": project.code_path,
    }


def is_cleanup_candidate(project: dict[str, Any]) -> bool:
    reasons = cleanup_reasons(project)
    project["cleanup_reasons"] = reasons
    return bool(reasons)


def cleanup_reasons(project: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    name = str(project.get("name") or "").strip()
    if name.casefold() in {item.casefold() for item in BLOCKLIST}:
        reasons.append("blocklisted_name")

    for field in ("path", "code_path"):
        value = _resolve_path_str(project.get(field))
        if not value:
            continue
        resolved = Path(value)
        if not resolved.exists():
            reasons.append(f"{field}_missing")
        if is_repo_internal_path(resolved):
            reasons.append(f"{field}_inside_jarvis_repo")

    return sorted(set(reasons))


def is_repo_internal_path(path: Path) -> bool:
    repo_root = Path(__file__).resolve().parents[2]
    internal_roots = (repo_root, repo_root / "pi", repo_root / "sidecar")
    return any(is_within(path, root) for root in internal_roots)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


if __name__ == "__main__":
    raise SystemExit(main())
