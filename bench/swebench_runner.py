#!/usr/bin/env python3
"""Generate SWE-bench Verified predictions with JARVIS Code.

Methodology:
    This runner performs one agentic attempt per instance on local Windows.
    It does not run project test suites during generation because benchmark
    repositories need project-specific Linux environments. Scoring is left to
    the official SWE-bench tooling, for example sb-cli/cloud evaluation.

    The output predictions JSONL uses the official row shape:
    {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}

    Keep sb-cli/API credentials local. Do not commit them or bake them into
    CI.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any

import polyglot_runner as agent_exec


DEFAULT_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_TIMEOUT_SECONDS = 1200
DEFAULT_GIT_TIMEOUT_SECONDS = 900
DEFAULT_STALL_TIMEOUT_MS = "1200000"


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Ignoring invalid JSONL line {line_no} in {path}", file=sys.stderr)
                continue
            if isinstance(raw, dict):
                records.append(raw)
    return records


def load_completed_records(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    for record in read_jsonl_records(path):
        instance_id = record.get("instance_id")
        if isinstance(instance_id, str):
            completed[instance_id] = record
    return completed


def parse_instance_filter(raw: str | None) -> list[str] | None:
    return agent_exec.parse_task_filter(raw)


def sanitize_id(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in value)


def bench_conv_id(instance_id: str) -> str:
    return f"swe_{sanitize_id(instance_id)}"


def repo_cache_name(repo: str) -> str:
    return f"{sanitize_id(repo.replace('/', '__'))}.git"


def load_dataset_instances(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise agent_exec.RunnerError(
            "datasets is required for SWE-bench generation. Install it with: "
            "python -m pip install --upgrade datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    instances: list[dict[str, Any]] = []
    for row in dataset:
        if not isinstance(row, dict):
            continue
        missing = [field for field in ("instance_id", "repo", "base_commit", "problem_statement") if field not in row]
        if missing:
            raise agent_exec.RunnerError(
                f"Dataset row is missing required field(s) {', '.join(missing)}: {row!r}"
            )
        instances.append(dict(row))
    return instances


def select_instances(
    instances: list[dict[str, Any]],
    requested: list[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    by_id = {str(item["instance_id"]): item for item in instances}
    if requested is None:
        selected = list(instances)
    else:
        missing = [instance_id for instance_id in requested if instance_id not in by_id]
        if missing:
            raise agent_exec.RunnerError(f"Unknown instance(s): {', '.join(missing)}")
        selected = [by_id[instance_id] for instance_id in requested]

    if limit is not None:
        if limit < 0:
            raise agent_exec.RunnerError("--limit must be non-negative")
        selected = selected[:limit]
    return selected


def remove_tree(path: Path) -> None:
    """rmtree that survives read-only files (git objects on Windows)."""

    def _onexc(func: Any, p: Any, exc: BaseException) -> None:
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onexc=_onexc)


def remove_tree_best_effort(path: Path) -> None:
    try:
        remove_tree(path)
    except OSError:
        pass


def run_git(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = DEFAULT_GIT_TIMEOUT_SECONDS,
    input_text: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        "git",
        "-c",
        "core.longpaths=true",
        "-c",
        "core.autocrlf=false",
        *args,
    ]
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        timeout=timeout,
        check=False,
    )
    if check and completed.returncode != 0:
        location = f" in {cwd}" if cwd else ""
        raise agent_exec.RunnerError(
            f"git {' '.join(args)} failed{location} with exit code {completed.returncode}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed


def ensure_bare_repo(repo: str, repos_cache: Path, timeout: int) -> Path:
    repos_cache.mkdir(parents=True, exist_ok=True)
    bare_path = repos_cache / repo_cache_name(repo)
    if bare_path.exists():
        run_git(["config", "core.longpaths", "true"], cwd=bare_path, timeout=timeout)
        run_git(["config", "core.autocrlf", "false"], cwd=bare_path, timeout=timeout)
        return bare_path

    url = f"https://github.com/{repo}.git"
    run_git(["clone", "--bare", url, str(bare_path)], timeout=timeout)
    run_git(["config", "core.longpaths", "true"], cwd=bare_path, timeout=timeout)
    run_git(["config", "core.autocrlf", "false"], cwd=bare_path, timeout=timeout)
    return bare_path


def ensure_commit_available(bare_repo: Path, repo: str, base_commit: str, timeout: int) -> None:
    found = run_git(["cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=bare_repo, timeout=timeout, check=False)
    if found.returncode == 0:
        return
    url = f"https://github.com/{repo}.git"
    run_git(["fetch", url, base_commit, "--depth=1"], cwd=bare_repo, timeout=timeout)


def fresh_checkout(
    repo: str,
    base_commit: str,
    repos_cache: Path,
    work_root: Path,
    instance_id: str,
    timeout: int,
) -> Path:
    bare_repo = ensure_bare_repo(repo, repos_cache, timeout)
    ensure_commit_available(bare_repo, repo, base_commit, timeout)

    workdir = work_root / sanitize_id(instance_id)
    agent_exec.ensure_safe_task_dest(work_root, workdir)
    if workdir.exists():
        remove_tree(workdir)

    work_root.mkdir(parents=True, exist_ok=True)
    run_git(["clone", "--shared", str(bare_repo), str(workdir)], timeout=timeout)
    run_git(["config", "core.longpaths", "true"], cwd=workdir, timeout=timeout)
    run_git(["config", "core.autocrlf", "false"], cwd=workdir, timeout=timeout)
    run_git(["checkout", "--detach", base_commit], cwd=workdir, timeout=timeout)
    return workdir


def write_problem_file(workdir: Path, instance: dict[str, Any]) -> Path:
    path = workdir / "PROBLEM.md"
    text = f"""# {instance["instance_id"]}

Repository: {instance["repo"]}
Base commit: {instance["base_commit"]}

## Issue

{instance["problem_statement"]}

## Runner instructions

Fix the issue by editing repository source files. Do not modify test files.
Do not create scratch files inside the repository. This Windows generation
runner does not provide project test environments, so reason from the code
instead of relying on the project's full test suite.
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_task_prompt(workdir: Path) -> Path:
    root = str(workdir.resolve()).replace("\\", "/")
    prompt = (
        f"Work ONLY inside {root}. "
        f"Read {root}/PROBLEM.md (a real GitHub issue) and fix it by editing the repository source. "
        "Do NOT modify test files. "
        "Repo test environments are not installed on this machine, so reason from the code instead of relying on running the project's test suite. "
        "Do NOT create scratch files inside the repo. "
        "When the fix is complete, stop."
    )
    prompt = " ".join(prompt.split())
    path = workdir / "task.txt"
    path.write_text(prompt + "\n", encoding="utf-8", newline="\n")
    return path


def build_jarvis_command(args: argparse.Namespace, prompt_path: Path, instance_id: str) -> list[str]:
    command = [
        *agent_exec.resolve_jarvis_command(args.jarvis_cmd),
        "--yolo",
        "--recent-turns",
        "0",
        "--bench-conv",
        bench_conv_id(instance_id),
        "--auto-prompts",
        str(prompt_path.resolve()),
    ]
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.model:
        command.extend(["--model", args.model])
    return command


def git_changed_paths(workdir: Path, base_commit: str, timeout: int) -> list[str]:
    completed = run_git(
        ["diff", "--cached", "--name-only", "-z", base_commit],
        cwd=workdir,
        timeout=timeout,
    )
    if not completed.stdout:
        return []
    return [item for item in completed.stdout.split("\0") if item]


def is_excluded_patch_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    basename = normalized.rsplit("/", 1)[-1]
    if normalized in {"PROBLEM.md", "task.txt", "jarvis.stdout.log", "jarvis.stderr.log"}:
        return True
    # jarvis engine runtime artifacts (auto-prompts progress, project memory, maps)
    if basename in {".auto_progress.json", "JARVIS.md"}:
        return True
    if normalized.startswith(".jarvis") or "/.jarvis" in normalized:
        return True
    if normalized.startswith("tests/") or "/tests/" in normalized:
        return True
    if basename.startswith("test_") or "_test." in basename:
        return True
    return False


def unstage_paths(workdir: Path, base_commit: str, paths: list[str], timeout: int) -> None:
    for start in range(0, len(paths), 100):
        chunk = paths[start : start + 100]
        run_git(["reset", "-q", base_commit, "--", *chunk], cwd=workdir, timeout=timeout)


def extract_model_patch(workdir: Path, base_commit: str, timeout: int) -> tuple[str, list[str]]:
    run_git(["add", "-A"], cwd=workdir, timeout=timeout)
    paths = git_changed_paths(workdir, base_commit, timeout)
    excluded = [path for path in paths if is_excluded_patch_path(path)]
    if excluded:
        unstage_paths(workdir, base_commit, excluded, timeout)

    completed = run_git(
        ["diff", "--cached", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/", base_commit],
        cwd=workdir,
        timeout=timeout,
    )
    return completed.stdout, excluded


def apply_check_patch(
    repo: str,
    base_commit: str,
    repos_cache: Path,
    apply_root: Path,
    instance_id: str,
    patch: str,
    timeout: int,
) -> tuple[bool, str | None]:
    if not patch.strip():
        return False, "empty_patch"

    check_dir = fresh_checkout(
        repo,
        base_commit,
        repos_cache,
        apply_root,
        f"{sanitize_id(instance_id)}__apply_check",
        timeout,
    )
    patch_path = check_dir / ".swebench_candidate.patch"
    patch_path.write_text(patch, encoding="utf-8", newline="\n")
    completed = run_git(["apply", "--check", str(patch_path)], cwd=check_dir, timeout=timeout, check=False)
    ok = completed.returncode == 0
    error = None if ok else (completed.stderr.strip() or completed.stdout.strip() or "git apply --check failed")
    return ok, error


def run_instance(
    args: argparse.Namespace,
    instance: dict[str, Any],
    repos_cache: Path,
    work_root: Path,
    apply_root: Path,
) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    repo = str(instance["repo"])
    base_commit = str(instance["base_commit"])

    started = time.monotonic()
    workdir = fresh_checkout(repo, base_commit, repos_cache, work_root, instance_id, args.git_timeout)
    apply_dir: Path | None = None
    try:
        write_problem_file(workdir, instance)
        prompt_path = write_task_prompt(workdir)
        command = build_jarvis_command(args, prompt_path, instance_id)
        print(f"[run] {instance_id}: {' '.join(command)}")
        exit_code, timed_out, jarvis_duration = agent_exec.run_jarvis(command, workdir, args.timeout)

        patch, excluded_paths = extract_model_patch(workdir, base_commit, args.git_timeout)
        apply_dir = apply_root / f"{sanitize_id(instance_id)}__apply_check"
        apply_ok, apply_error = apply_check_patch(
            repo,
            base_commit,
            repos_cache,
            apply_root,
            instance_id,
            patch,
            args.git_timeout,
        )
        empty_patch = not bool(patch.strip())
        duration_s = time.monotonic() - started
        return {
            "instance_id": instance_id,
            "repo": repo,
            "base_commit": base_commit,
            "duration_s": round(duration_s, 3),
            "jarvis_duration_s": round(jarvis_duration, 3),
            "timed_out": timed_out,
            "exit_code": exit_code,
            "patch_bytes": len(patch.encode("utf-8")),
            "empty_patch": empty_patch,
            "apply_ok": apply_ok,
            "apply_error": apply_error,
            "excluded_paths": excluded_paths,
            "bench_conv": bench_conv_id(instance_id),
            "model_patch": patch,
            "ts": agent_exec.utc_now_iso(),
            "methodology": "single_attempt_agentic_windows_generation_official_swebench_scoring",
        }
    finally:
        if apply_dir is not None and apply_dir.exists():
            remove_tree_best_effort(apply_dir)
        if not args.keep_workdirs and workdir.exists():
            remove_tree_best_effort(workdir)


def print_summary(records: dict[str, dict[str, Any]]) -> None:
    completed = len(records)
    apply_ok = sum(1 for record in records.values() if record.get("apply_ok") is True)
    empty = sorted(instance_id for instance_id, record in records.items() if record.get("empty_patch") is True)
    timed_out = sorted(instance_id for instance_id, record in records.items() if record.get("timed_out") is True)

    print("")
    print("Summary")
    print(f"  completed:   {completed}")
    print(f"  apply_ok:    {apply_ok}")
    print(f"  empty_patch: {', '.join(empty) if empty else '-'}")
    print(f"  timeouts:    {', '.join(timed_out) if timed_out else '-'}")


def cmd_generate(args: argparse.Namespace) -> int:
    repos_cache = agent_exec.resolve_path(args.repos_cache)
    work_root = agent_exec.resolve_path(args.workdir)
    apply_root = work_root / "_apply_check"
    results_path = agent_exec.resolve_path(args.results)
    if args.timeout <= 0:
        raise agent_exec.RunnerError("--timeout must be positive")
    if args.git_timeout <= 0:
        raise agent_exec.RunnerError("--git-timeout must be positive")

    os.environ.setdefault("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS", DEFAULT_STALL_TIMEOUT_MS)

    instances = load_dataset_instances(args.dataset, args.split)
    selected = select_instances(instances, parse_instance_filter(args.instances), args.limit)
    completed = load_completed_records(results_path)
    repos_cache.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    apply_root.mkdir(parents=True, exist_ok=True)

    print(f"dataset:     {args.dataset} [{args.split}] ({len(instances)} instances)")
    print(f"repos_cache: {repos_cache}")
    print(f"workdir:     {work_root}")
    print(f"results:     {results_path}")
    print(f"selected:    {len(selected)} instance(s)")

    for instance in selected:
        instance_id = str(instance["instance_id"])
        if instance_id in completed:
            print(f"[skip] {instance_id}: already recorded")
            continue

        record = run_instance(args, instance, repos_cache, work_root, apply_root)
        agent_exec.jsonl_append(results_path, record)
        completed[instance_id] = record
        status = "APPLY_OK" if record["apply_ok"] else "CHECK"
        timeout_note = " timeout" if record["timed_out"] else ""
        empty_note = " empty_patch" if record["empty_patch"] else ""
        print(f"[done] {instance_id}: {status}{timeout_note}{empty_note}")

    print_summary(load_completed_records(results_path))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    results_path = agent_exec.resolve_path(args.results)
    predictions_path = agent_exec.resolve_path(args.predictions)
    records = load_completed_records(results_path)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    missing_patch = 0
    with predictions_path.open("w", encoding="utf-8", newline="\n") as fh:
        for instance_id in sorted(records):
            record = records[instance_id]
            patch = record.get("model_patch")
            if not isinstance(patch, str):
                patch = ""
                missing_patch += 1
            row = {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name,
                "model_patch": patch,
            }
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"predictions:   {predictions_path}")
    print(f"rows:          {len(records)}")
    print(f"missing_patch: {missing_patch}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate SWE-bench predictions with JARVIS.")
    generate.add_argument("--dataset", default=DEFAULT_DATASET, help=f"HF dataset name. Default: {DEFAULT_DATASET}.")
    generate.add_argument("--split", default=DEFAULT_SPLIT, help=f"Dataset split. Default: {DEFAULT_SPLIT}.")
    generate.add_argument("--repos-cache", required=True, help="Directory for per-repo bare clone cache.")
    generate.add_argument("--workdir", required=True, help="Directory for fresh per-instance working copies.")
    generate.add_argument("--results", default="results.jsonl", help="JSONL generation results path.")
    generate.add_argument("--instances", help="Comma-separated SWE-bench instance IDs to run.")
    generate.add_argument("--limit", type=int, help="Run at most N selected instances.")
    generate.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-instance JARVIS timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    generate.add_argument(
        "--git-timeout",
        type=int,
        default=DEFAULT_GIT_TIMEOUT_SECONDS,
        help=f"Per git operation timeout in seconds. Default: {DEFAULT_GIT_TIMEOUT_SECONDS}.",
    )
    generate.add_argument("--provider", help="Optional JARVIS --provider pass-through.")
    generate.add_argument("--model", help="Optional JARVIS --model pass-through.")
    generate.add_argument(
        "--jarvis-cmd",
        default="jarvis",
        help="JARVIS command to execute. Defaults to jarvis, with repo-local jarvis.ps1 fallback.",
    )
    generate.add_argument(
        "--keep-workdirs",
        action="store_true",
        help="Keep per-instance working copies after each run. Default removes them.",
    )
    generate.set_defaults(func=cmd_generate)

    collect = subparsers.add_parser("collect", help="Collect results into official predictions JSONL.")
    collect.add_argument("--results", default="results.jsonl", help="JSONL generation results path.")
    collect.add_argument("--predictions", default="predictions.jsonl", help="Output predictions JSONL path.")
    collect.add_argument(
        "--model-name",
        required=True,
        help='Value for model_name_or_path, e.g. "jarvis-code+gpt-5.5(subscription)".',
    )
    collect.set_defaults(func=cmd_collect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except agent_exec.RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: command or file not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
