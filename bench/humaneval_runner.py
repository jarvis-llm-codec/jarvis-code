#!/usr/bin/env python3
"""Run HumanEval+ generation with JARVIS Code and collect EvalPlus samples.

Methodology:
    This runner performs one agentic attempt per task. The agent may write its
    own scratch tests, but the HumanEval/EvalPlus tests and canonical solutions
    are never written into the task directory. Scoring is intentionally left to
    the official EvalPlus harness:

        evalplus.evaluate --dataset humaneval --samples samples.jsonl

    EvalPlus can be fragile on native Windows because its evaluator executes
    generated code and uses Unix-oriented process controls. If local evaluation
    fails on Windows, keep the generated samples.jsonl and run the same command
    in WSL, Docker, or a Linux machine. Publish results with the model, date,
    subscription/API route, and the single-attempt agentic methodology.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import polyglot_runner as agent_exec


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_STALL_TIMEOUT_MS = "900000"


def task_dir_name(task_id: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_-")
    normalized = task_id.lower().replace("/", "_")
    return "".join(ch if ch in allowed else "_" for ch in normalized)


def task_bench_id(task_id: str) -> str:
    return task_dir_name(task_id)


def task_sort_key(task_id: str) -> tuple[str, int | str]:
    prefix, sep, suffix = task_id.partition("/")
    if sep and suffix.isdigit():
        return prefix, int(suffix)
    return prefix, suffix


def load_humaneval_plus() -> dict[str, dict[str, Any]]:
    try:
        from evalplus.data import get_human_eval_plus
    except ModuleNotFoundError as exc:
        raise agent_exec.RunnerError(
            "evalplus is required for generate. Install it in the bench venv with: "
            "python -m pip install --upgrade evalplus"
        ) from exc

    problems = get_human_eval_plus()
    if not isinstance(problems, dict):
        raise agent_exec.RunnerError("evalplus.data.get_human_eval_plus() did not return a dict")
    return problems


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
        task = record.get("task")
        if isinstance(task, str):
            completed[task] = record
    return completed


def parse_task_filter(raw: str | None) -> list[str] | None:
    return agent_exec.parse_task_filter(raw)


def select_tasks(
    all_tasks: list[str],
    requested: list[str] | None,
    limit: int | None,
) -> list[str]:
    if requested is None:
        selected = list(all_tasks)
    else:
        available = set(all_tasks)
        missing = [task for task in requested if task not in available]
        if missing:
            raise agent_exec.RunnerError(f"Unknown task(s): {', '.join(missing)}")
        selected = list(requested)

    if limit is not None:
        if limit < 0:
            raise agent_exec.RunnerError("--limit must be non-negative")
        selected = selected[:limit]
    return selected


def write_solution_seed(task_dir: Path, prompt: str) -> Path:
    path = task_dir / "solution.py"
    text = prompt if prompt.endswith("\n") else prompt + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_problem(task_dir: Path, task_id: str, entry_point: str) -> Path:
    path = task_dir / "PROBLEM.md"
    text = f"""# {task_id}

Entry point: `{entry_point}`

Complete `solution.py`. The file already contains the original prompt,
signature, and docstring. Keep the function signature intact and keep the
final implementation self-contained.

You may create temporary scratch tests, but do not install or import EvalPlus,
do not look up canonical solutions, and do not add hidden benchmark data.
"""
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_task_prompt(task_dir: Path) -> Path:
    root = str(task_dir.resolve()).replace("\\", "/")
    prompt = (
        f"Work ONLY inside {root}. "
        f"Read {root}/PROBLEM.md and complete the function in {root}/solution.py. "
        "solution.py must stay self-contained (keep the signature). "
        f"You may write your own tests in {root}/scratch_test.py and run them with python. "
        "Do NOT install or import evalplus, and do NOT look up canonical solutions. "
        "When your implementation is solid, stop."
    )
    prompt = " ".join(prompt.split())
    path = task_dir / "task.txt"
    path.write_text(prompt + "\n", encoding="utf-8", newline="\n")
    return path


def reset_task_dir(workdir: Path, task_id: str) -> Path:
    task_dir = workdir / task_dir_name(task_id)
    agent_exec.ensure_safe_task_dest(workdir, task_dir)
    if task_dir.exists():
        import shutil

        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)
    return task_dir


def build_jarvis_command(args: argparse.Namespace, prompt_path: Path, task_id: str) -> list[str]:
    command = [
        *agent_exec.resolve_jarvis_command(args.jarvis_cmd),
        "--yolo",
        "--recent-turns",
        "0",
    ]
    if not args.no_bench_conv:
        command.extend(["--bench-conv", task_bench_id(task_id)])
    command.extend(["--auto-prompts", str(prompt_path.resolve())])
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.model:
        command.extend(["--model", args.model])
    return command


def validate_solution(solution_path: Path, entry_point: str) -> tuple[bool, bool, str | None]:
    try:
        source = solution_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        return False, False, f"{exc.__class__.__name__}: {exc}"

    try:
        tree = ast.parse(source, filename=str(solution_path))
    except SyntaxError as exc:
        return False, False, f"SyntaxError: {exc.msg} at line {exc.lineno}"

    entry_point_ok = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point
        for node in tree.body
    )
    return True, entry_point_ok, None


def has_cheat_flag(solution_path: Path) -> bool:
    try:
        source = solution_path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False
    lowered = source.lower()
    return "evalplus" in lowered


def run_task(
    args: argparse.Namespace,
    workdir: Path,
    task_id: str,
    problem: dict[str, Any],
) -> dict[str, Any]:
    prompt = problem.get("prompt")
    entry_point = problem.get("entry_point")
    if not isinstance(prompt, str) or not prompt:
        raise agent_exec.RunnerError(f"{task_id} has no prompt string")
    if not isinstance(entry_point, str) or not entry_point:
        raise agent_exec.RunnerError(f"{task_id} has no entry_point string")

    started = time.monotonic()
    task_dir = reset_task_dir(workdir, task_id)
    solution_path = write_solution_seed(task_dir, prompt)
    write_problem(task_dir, task_id, entry_point)
    prompt_path = write_task_prompt(task_dir)

    command = build_jarvis_command(args, prompt_path, task_id)
    print(f"[run] {task_id}: {' '.join(command)}")
    exit_code, timed_out, jarvis_duration = agent_exec.run_jarvis(command, task_dir, args.timeout)

    syntax_ok, entry_point_ok, syntax_error = validate_solution(solution_path, entry_point)
    cheat_flag = has_cheat_flag(solution_path)
    duration_s = time.monotonic() - started

    return {
        "task": task_id,
        "duration_s": round(duration_s, 3),
        "jarvis_duration_s": round(jarvis_duration, 3),
        "timed_out": timed_out,
        "exit_code": exit_code,
        "syntax_ok": syntax_ok,
        "entry_point_ok": entry_point_ok,
        "syntax_error": syntax_error,
        "cheat_flag": cheat_flag,
        "entry_point": entry_point,
        "workdir": str(task_dir),
        "solution_path": str(solution_path),
        "bench_conv": task_bench_id(task_id),
        "ts": agent_exec.utc_now_iso(),
        "methodology": "single_attempt_agent_self_test_evalplus_official_scoring",
    }


def print_generate_summary(records: dict[str, dict[str, Any]]) -> None:
    total = len(records)
    syntax_ok = sum(1 for record in records.values() if record.get("syntax_ok") is True)
    entry_ok = sum(1 for record in records.values() if record.get("entry_point_ok") is True)
    timed_out = sorted(task for task, record in records.items() if record.get("timed_out") is True)
    cheat_flags = sorted(task for task, record in records.items() if record.get("cheat_flag") is True)

    print("")
    print("Summary")
    print(f"  completed:      {total}")
    print(f"  syntax_ok:      {syntax_ok}")
    print(f"  entry_point_ok: {entry_ok}")
    print(f"  timeouts:       {', '.join(timed_out) if timed_out else '-'}")
    print(f"  cheat_flags:    {', '.join(cheat_flags) if cheat_flags else '-'}")


def cmd_generate(args: argparse.Namespace) -> int:
    problems = load_humaneval_plus()
    all_tasks = sorted(problems, key=task_sort_key)
    selected = select_tasks(all_tasks, parse_task_filter(args.tasks), args.limit)

    workdir = agent_exec.resolve_path(args.workdir)
    results_path = agent_exec.resolve_path(args.results)
    if args.timeout <= 0:
        raise agent_exec.RunnerError("--timeout must be positive")

    os.environ.setdefault("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS", DEFAULT_STALL_TIMEOUT_MS)
    workdir.mkdir(parents=True, exist_ok=True)
    completed = load_completed_records(results_path)

    print(f"dataset:  HumanEval+ ({len(all_tasks)} tasks)")
    print(f"workdir:  {workdir}")
    print(f"results:  {results_path}")
    print(f"selected: {len(selected)} task(s)")

    for task_id in selected:
        if task_id in completed:
            print(f"[skip] {task_id}: already recorded")
            continue

        record = run_task(args, workdir, task_id, problems[task_id])
        agent_exec.jsonl_append(results_path, record)
        completed[task_id] = record

        status = "OK" if record["syntax_ok"] and record["entry_point_ok"] else "CHECK"
        timeout_note = " timeout" if record["timed_out"] else ""
        cheat_note = " cheat_flag" if record["cheat_flag"] else ""
        print(f"[done] {task_id}: {status}{timeout_note}{cheat_note}")

    print_generate_summary(load_completed_records(results_path))
    return 0


def solution_path_for_record(record: dict[str, Any], workdir: Path) -> Path:
    raw_path = record.get("solution_path")
    if isinstance(raw_path, str) and raw_path:
        return Path(raw_path)
    task = record.get("task")
    if not isinstance(task, str):
        raise agent_exec.RunnerError("Result record is missing task")
    return workdir / task_dir_name(task) / "solution.py"


def cmd_collect(args: argparse.Namespace) -> int:
    workdir = agent_exec.resolve_path(args.workdir)
    results_path = agent_exec.resolve_path(args.results)
    samples_path = agent_exec.resolve_path(args.samples)
    completed = load_completed_records(results_path)

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with samples_path.open("w", encoding="utf-8", newline="\n") as fh:
        for task_id in sorted(completed, key=task_sort_key):
            solution_path = solution_path_for_record(completed[task_id], workdir)
            try:
                solution = solution_path.read_text(encoding="utf-8-sig")
            except OSError:
                solution = ""
                missing += 1
            row = {"task_id": task_id, "solution": solution}
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"samples: {samples_path}")
    print(f"rows:    {len(completed)}")
    print(f"missing: {missing}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    samples_path = agent_exec.resolve_path(args.samples)
    command = [
        sys.executable,
        "-m",
        "evalplus.evaluate",
        "--dataset",
        "humaneval",
        "--samples",
        str(samples_path),
    ]
    if args.i_just_wanna_run:
        command.append("--i-just-wanna-run")
    if args.base_only:
        command.append("--base-only")
    if args.parallel is not None:
        command.extend(["--parallel", str(args.parallel)])

    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0 and os.name == "nt":
        print(
            "EvalPlus evaluation failed on Windows. Keep samples.jsonl and run the same "
            "command in WSL, Docker, or Linux for official scoring.",
            file=sys.stderr,
        )
    return completed.returncode


def add_common_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workdir", required=True, help="Per-task working directory root.")
    parser.add_argument("--results", default="results.jsonl", help="JSONL generation results path.")
    parser.add_argument("--limit", type=int, help="Run at most N selected tasks.")
    parser.add_argument("--tasks", help="Comma-separated task IDs, e.g. HumanEval/0,HumanEval/1.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-task JARVIS timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument("--provider", help="Optional JARVIS --provider pass-through.")
    parser.add_argument("--model", help="Optional JARVIS --model pass-through.")
    parser.add_argument(
        "--jarvis-cmd",
        default="jarvis",
        help="JARVIS command to execute. Defaults to jarvis, with repo-local jarvis.ps1 fallback.",
    )
    parser.add_argument(
        "--no-bench-conv",
        action="store_true",
        help="Skip --bench-conv isolation flag for diagnostics.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate HumanEval+ solutions with JARVIS.")
    add_common_generate_args(generate)
    generate.set_defaults(func=cmd_generate)

    collect = subparsers.add_parser("collect", help="Collect solution.py files into samples.jsonl.")
    collect.add_argument("--workdir", required=True, help="Per-task working directory root.")
    collect.add_argument("--results", default="results.jsonl", help="JSONL generation results path.")
    collect.add_argument("--samples", default="samples.jsonl", help="Output EvalPlus samples JSONL path.")
    collect.set_defaults(func=cmd_collect)

    evaluate = subparsers.add_parser("evaluate", help="Run the official EvalPlus evaluator.")
    evaluate.add_argument("--samples", default="samples.jsonl", help="EvalPlus samples JSONL path.")
    evaluate.add_argument("--parallel", type=int, help="Pass-through EvalPlus --parallel value.")
    evaluate.add_argument("--base-only", action="store_true", help="Pass through EvalPlus --base-only.")
    evaluate.add_argument(
        "--i-just-wanna-run",
        action="store_true",
        help="Pass through EvalPlus --i-just-wanna-run to force re-evaluation.",
    )
    evaluate.set_defaults(func=cmd_evaluate)

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
