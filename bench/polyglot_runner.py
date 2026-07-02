#!/usr/bin/env python3
"""Run Aider polyglot benchmark subsets with JARVIS Code.

Examples:
    python bench/polyglot_runner.py --language python \
        --exercises C:/bench/polyglot-benchmark/python/exercises/practice \
        --workdir C:/jarvis_workspace/polyglot_python_work

    python bench/polyglot_runner.py --language go \
        --exercises C:/bench/polyglot-benchmark/go/exercises/practice \
        --workdir C:/jarvis_workspace/polyglot_go_work

    python bench/polyglot_runner.py --language javascript \
        --exercises C:/bench/polyglot-benchmark/javascript/exercises/practice \
        --workdir C:/jarvis_workspace/polyglot_js_work

Methodology:
    This runner gives the agent a single attempt per task and lets the agent
    run tests on its own. The final score is still decided by this runner with
    a fresh language-appropriate test command after the agent exits. This differs from
    the aider leaderboard harness, which uses two attempts and feeds test
    errors back after the first failure; publish results with that caveat plus
    the model, date, and subscription/API route used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 900
DEFAULT_PYTEST_TIMEOUT_SECONDS = 120
SUPPORTED_LANGUAGES = ("python", "go", "javascript")


class RunnerError(RuntimeError):
    """Raised for benchmark setup or per-task runner failures."""


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def safe_child_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RunnerError(f"Task file path must be relative: {relative}")
    resolved = (root / candidate).resolve(strict=False)
    if not is_relative_to(resolved, root.resolve()):
        raise RunnerError(f"Task file path escapes task root: {relative}")
    return resolved


def ensure_safe_task_dest(workdir: Path, dest: Path) -> None:
    resolved_workdir = workdir.resolve()
    resolved_dest = dest.resolve(strict=False)
    if resolved_dest == resolved_workdir or not is_relative_to(resolved_dest, resolved_workdir):
        raise RunnerError(f"Refusing to delete/copy outside workdir: {dest}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise RunnerError(f"Expected JSON object in {path}")
    return raw


def completed_record_key(record: dict[str, Any], language: str) -> str | None:
    task = record.get("task")
    if not isinstance(task, str):
        return None
    if language == "python":
        if record.get("language", "python") != "python":
            return None
        return task
    record_language = record.get("language")
    if record_language != language:
        return None
    return f"{language}:{task}"


def selected_task_key(task: str, language: str) -> str:
    if language == "python":
        return task
    return f"{language}:{task}"


def load_completed_records(path: Path, language: str = "python") -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}

    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] Ignoring invalid JSONL line {line_no} in {path}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                continue
            key = completed_record_key(record, language)
            if key is not None and "pass" in record:
                completed[key] = record
    return completed


def as_string_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    raise RunnerError(f"Expected {label} to be a string or list of strings")


def load_task_files(task_dir: Path) -> tuple[list[str], list[str]]:
    config_path = task_dir / ".meta" / "config.json"
    config = read_json(config_path)
    files = config.get("files")
    if not isinstance(files, dict):
        raise RunnerError(f"Missing files object in {config_path}")

    solution_files = as_string_list(files.get("solution"), f"{config_path}: files.solution")
    test_files = as_string_list(files.get("test"), f"{config_path}: files.test")
    if not solution_files:
        raise RunnerError(f"No solution files listed in {config_path}")
    if not test_files:
        raise RunnerError(f"No test files listed in {config_path}")
    return solution_files, test_files


def discover_tasks(exercises: Path) -> list[str]:
    if not exercises.exists():
        raise RunnerError(f"Exercises path does not exist: {exercises}")
    if not exercises.is_dir():
        raise RunnerError(f"Exercises path is not a directory: {exercises}")

    tasks: list[str] = []
    for child in sorted(exercises.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        if (child / ".meta" / "config.json").exists() and (child / ".docs" / "instructions.md").exists():
            tasks.append(child.name)
    return tasks


def parse_task_filter(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    tasks = [item.strip() for item in raw.split(",") if item.strip()]
    return tasks or None


def task_bench_id(task: str, language: str = "python") -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    safe = "".join(ch if ch in allowed else "_" for ch in task)
    if language == "python":
        # Preserve the baseline Python conv-id format used for published runs.
        return f"polyglot_py_{safe}"
    if language not in SUPPORTED_LANGUAGES:
        raise RunnerError(f"Unsupported language: {language}")
    return f"{language}_{safe}"


def test_command_label(language: str) -> str:
    if language == "python":
        return "pytest"
    if language == "go":
        return "go test ./..."
    if language == "javascript":
        return "npm test"
    raise RunnerError(f"Unsupported language: {language}")


def resolve_npm_command() -> str:
    if os.name == "nt":
        command = shutil.which("npm.cmd")
        if command:
            return command
        raise RunnerError("npm.cmd was not found on PATH")
    command = shutil.which("npm")
    if command:
        return command
    raise RunnerError("npm was not found on PATH")


def final_test_command(language: str) -> list[str]:
    if language == "python":
        return [sys.executable, "-m", "pytest", "-x", "-q"]
    if language == "go":
        return ["go", "test", "./..."]
    if language == "javascript":
        return [resolve_npm_command(), "test"]
    raise RunnerError(f"Unsupported language: {language}")


def package_json_hash(task_dir: Path) -> str:
    package_json = task_dir / "package.json"
    if not package_json.exists():
        raise RunnerError(f"JavaScript task is missing package.json: {package_json}")
    # Hash only the dependency sections: exercise package.json files differ in
    # per-task fields like "name", which would defeat the cache entirely.
    try:
        raw = json.loads(package_json.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return sha256_file(package_json)
    if not isinstance(raw, dict):
        return sha256_file(package_json)
    deps = {
        key: raw[key]
        for key in ("dependencies", "devDependencies")
        if isinstance(raw.get(key), dict)
    }
    if not deps:
        return sha256_file(package_json)
    blob = json.dumps(deps, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def copy_node_modules(source: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def run_npm_install(task_dir: Path, timeout_seconds: int) -> None:
    log_path = task_dir / "npm.install.log"
    command = [resolve_npm_command(), "install"]
    with log_path.open("wb") as log_fh:
        try:
            completed = subprocess.run(
                command,
                cwd=str(task_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"npm install timed out after {timeout_seconds}s; see {log_path}") from exc
    if completed.returncode != 0:
        raise RunnerError(f"npm install failed with exit code {completed.returncode}; see {log_path}")


def prepare_javascript_dependencies(task_dir: Path, workdir: Path, timeout_seconds: int) -> str:
    cache_root = workdir / "_node_modules_cache"
    cache_entry = cache_root / package_json_hash(task_dir)
    cached_node_modules = cache_entry / "node_modules"
    task_node_modules = task_dir / "node_modules"

    if cached_node_modules.exists():
        copy_node_modules(cached_node_modules, task_node_modules)
        return "hit"

    run_npm_install(task_dir, timeout_seconds)
    if not task_node_modules.exists():
        raise RunnerError(f"npm install completed but node_modules is missing: {task_node_modules}")

    cache_entry.mkdir(parents=True, exist_ok=True)
    copy_node_modules(task_node_modules, cached_node_modules)
    return "miss"


def prepare_language_dependencies(
    language: str,
    task_dir: Path,
    workdir: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if language == "javascript":
        return {"node_modules_cache": prepare_javascript_dependencies(task_dir, workdir, timeout_seconds)}
    if language in {"python", "go"}:
        return {}
    raise RunnerError(f"Unsupported language: {language}")


def build_test_instruction(language: str, root: str) -> str:
    if language == "python":
        return f"Run pytest in {root} yourself and iterate until all tests pass. "
    if language == "go":
        return f"Run go test ./... in {root} yourself and iterate until all tests pass. "
    if language == "javascript":
        return f"Run npm test in {root} yourself and iterate until all tests pass. "
    raise RunnerError(f"Unsupported language: {language}")


def write_task_prompt(
    task_dir: Path,
    solution_files: list[str],
    test_files: list[str],
    language: str = "python",
) -> Path:
    # Absolute paths on purpose: without them the project router can route
    # the turn into a previously active project dir instead of this cwd
    # (observed 2026-07-03: agent worked in jarvis_workspace/hello_py).
    root = str(task_dir.resolve()).replace("\\", "/")
    solution_text = ", ".join(f"{root}/{name}" for name in solution_files)
    test_text = ", ".join(f"{root}/{name}" for name in test_files)
    prompt = (
        f"Work ONLY inside {root} for this task. "
        f"Read {root}/.docs/instructions.md and implement the solution in {solution_text}. "
        f"{build_test_instruction(language, root)}"
        f"Do NOT modify the test files ({test_text}). "
        "When all tests pass, stop."
    )
    prompt = " ".join(prompt.split())
    prompt_path = task_dir / "task.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8", newline="\n")
    return prompt_path


def reset_task_dir(source: Path, dest: Path, workdir: Path) -> None:
    ensure_safe_task_dest(workdir, dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)


def capture_initial_test_hashes(task_dir: Path, test_files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in test_files:
        path = safe_child_path(task_dir, relative)
        if not path.exists():
            raise RunnerError(f"Listed test file does not exist: {relative}")
        hashes[relative] = sha256_file(path)
    return hashes


def restore_modified_tests(
    source_task_dir: Path,
    work_task_dir: Path,
    original_hashes: dict[str, str],
) -> list[str]:
    modified: list[str] = []
    for relative, original_hash in original_hashes.items():
        work_path = safe_child_path(work_task_dir, relative)
        source_path = safe_child_path(source_task_dir, relative)

        changed = True
        if work_path.exists() and work_path.is_file():
            changed = sha256_file(work_path) != original_hash
        if not changed:
            continue

        modified.append(relative)
        if not source_path.exists():
            raise RunnerError(f"Cannot restore missing source test file: {relative}")
        work_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, work_path)

    return modified


def resolve_jarvis_command(command: str) -> list[str]:
    found = shutil.which(command)
    if found is None and command == "jarvis":
        repo_root = Path(__file__).resolve().parents[1]
        local_wrapper = repo_root / "jarvis.ps1"
        if local_wrapper.exists():
            found = str(local_wrapper)

    executable = Path(found or command).expanduser()
    if executable.suffix.lower() == ".ps1":
        shell = "powershell" if os.name == "nt" else "pwsh"
        return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(executable)]
    return [str(executable) if found else command]


def build_jarvis_command(args: argparse.Namespace, prompt_path: Path, task: str) -> list[str]:
    command = [
        *resolve_jarvis_command(args.jarvis_cmd),
        "--yolo",
        "--recent-turns",
        "0",
    ]
    if not args.no_bench_conv:
        command.extend(["--bench-conv", task_bench_id(task, args.language)])
    command.extend([
        "--auto-prompts",
        str(prompt_path.resolve()),
    ])
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.model:
        command.extend(["--model", args.model])
    return command


def kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    if proc.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        proc.terminate()
        return

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()


def run_jarvis_conpty(
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int | None, bool, float]:
    """Run jarvis under a Windows ConPTY so the pi TUI sees a real terminal.

    Plain Popen with redirected stdout gives the TUI no TTY and the agent
    turn never executes (observed 2026-07-03: prompt fires, zero tool
    activity, stall watchdog aborts). ConPTY via pywinpty fixes that in any
    session (SSH, scheduled task) without needing an interactive desktop.
    """
    import threading

    import winpty  # pywinpty

    stdout_path = cwd / "jarvis.stdout.log"
    start = time.monotonic()
    timed_out = False

    cmdline = subprocess.list2cmdline(command)
    proc = winpty.PtyProcess.spawn(cmdline, cwd=str(cwd), dimensions=(40, 120))

    def drain() -> None:
        # ConPTY buffer must be drained continuously or the child blocks.
        with stdout_path.open("ab") as log_fh:
            while True:
                try:
                    data = proc.read(4096)
                except (EOFError, OSError):
                    return
                if data:
                    log_fh.write(data.encode("utf-8", "replace"))

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    while proc.isalive():
        if time.monotonic() - start > timeout_seconds:
            timed_out = True
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            try:
                proc.terminate(force=True)
            except Exception:
                pass
            break
        time.sleep(1.0)

    reader.join(timeout=10)
    exit_code: int | None = None
    try:
        exit_code = proc.exitstatus
    except Exception:
        exit_code = None
    return exit_code, timed_out, time.monotonic() - start


def run_jarvis(
    command: list[str],
    cwd: Path,
    timeout_seconds: int,
) -> tuple[int | None, bool, float]:
    if os.name == "nt" and os.environ.get("JLC_BENCH_NO_PTY") != "1":
        try:
            return run_jarvis_conpty(command, cwd, timeout_seconds)
        except ImportError:
            print("[warn] pywinpty not installed; falling back to plain pipes", file=sys.stderr)

    stdout_path = cwd / "jarvis.stdout.log"
    stderr_path = cwd / "jarvis.stderr.log"
    start = time.monotonic()
    timed_out = False
    exit_code: int | None = None

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    popen_kwargs: dict[str, Any] = {
        "cwd": str(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": None,
        "stderr": None,
        "creationflags": creationflags,
    }
    if os.name != "nt":
        popen_kwargs["start_new_session"] = True

    with stdout_path.open("wb") as stdout_fh, stderr_path.open("wb") as stderr_fh:
        popen_kwargs["stdout"] = stdout_fh
        popen_kwargs["stderr"] = stderr_fh
        proc = subprocess.Popen(command, **popen_kwargs)
        try:
            exit_code = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_tree(proc)
            try:
                exit_code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_code = proc.wait(timeout=10)
        except KeyboardInterrupt:
            kill_process_tree(proc)
            raise

    return exit_code, timed_out, time.monotonic() - start


def run_final_pytest(task_dir: Path, timeout_seconds: int) -> tuple[int | None, bool]:
    log_path = task_dir / "pytest.final.log"
    command = [sys.executable, "-m", "pytest", "-x", "-q"]
    with log_path.open("wb") as log_fh:
        try:
            completed = subprocess.run(
                command,
                cwd=str(task_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_fh.write(f"\n[polyglot_runner] pytest timed out after {timeout_seconds}s\n".encode("utf-8"))
            return None, True
    return completed.returncode, False


def run_final_tests(task_dir: Path, language: str, timeout_seconds: int) -> tuple[int | None, bool]:
    if language == "python":
        return run_final_pytest(task_dir, timeout_seconds)

    log_path = task_dir / "test.final.log"
    command = final_test_command(language)
    with log_path.open("wb") as log_fh:
        try:
            completed = subprocess.run(
                command,
                cwd=str(task_dir),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_fh.write(
                f"\n[polyglot_runner] {test_command_label(language)} timed out after {timeout_seconds}s\n".encode(
                    "utf-8"
                )
            )
            return None, True
    return completed.returncode, False


def run_task(args: argparse.Namespace, exercises: Path, workdir: Path, task: str) -> dict[str, Any]:
    source_task_dir = exercises / task
    work_task_dir = workdir / task

    task_started = time.monotonic()
    reset_task_dir(source_task_dir, work_task_dir, workdir)
    solution_files, test_files = load_task_files(work_task_dir)
    original_hashes = capture_initial_test_hashes(work_task_dir, test_files)
    dependency_meta = prepare_language_dependencies(args.language, work_task_dir, workdir, args.pytest_timeout)
    prompt_path = write_task_prompt(work_task_dir, solution_files, test_files, args.language)

    command = build_jarvis_command(args, prompt_path, task)
    print(f"[run] {task}: {' '.join(command)}")
    jarvis_exit_code, timed_out, jarvis_duration = run_jarvis(command, work_task_dir, args.timeout)

    modified_tests = restore_modified_tests(source_task_dir, work_task_dir, original_hashes)
    test_exit_code, test_timed_out = run_final_tests(work_task_dir, args.language, args.pytest_timeout)

    total_duration = time.monotonic() - task_started
    passed = test_exit_code == 0 and not test_timed_out
    record = {
        "task": task,
        "pass": passed,
        "duration_s": round(total_duration, 3),
        "jarvis_duration_s": round(jarvis_duration, 3),
        "timed_out": timed_out,
        "exit_code": jarvis_exit_code,
        "pytest_exit_code": test_exit_code,
        "pytest_timed_out": test_timed_out,
        "test_modified": bool(modified_tests),
        "test_modified_files": modified_tests,
        "solution_files": solution_files,
        "test_files": test_files,
        "bench_conv": task_bench_id(task, args.language),
        "ts": utc_now_iso(),
        "methodology": "single_attempt_agent_self_test_runner_verified",
    }
    if args.language != "python":
        record.update(
            {
                "language": args.language,
                "test_command": test_command_label(args.language),
                "test_exit_code": test_exit_code,
                "test_timed_out": test_timed_out,
                **dependency_meta,
            }
        )
    return record


def print_summary(records: dict[str, dict[str, Any]]) -> None:
    completed = len(records)
    passed = sum(1 for record in records.values() if record.get("pass") is True)
    pass_rate = (passed / completed * 100.0) if completed else 0.0
    timed_out = sorted(task for task, record in records.items() if record.get("timed_out") is True)
    modified_tests = sorted(task for task, record in records.items() if record.get("test_modified") is True)

    print("")
    print("Summary")
    print(f"  completed: {completed}")
    print(f"  passed:    {passed}")
    print(f"  pass_rate: {pass_rate:.1f}%")
    print(f"  timeouts:  {', '.join(timed_out) if timed_out else '-'}")
    print(f"  test edits:{' ' + ', '.join(modified_tests) if modified_tests else ' -'}")


def select_tasks(all_tasks: list[str], requested: list[str] | None, limit: int | None) -> list[str]:
    if requested is None:
        selected = list(all_tasks)
    else:
        available = set(all_tasks)
        missing = [task for task in requested if task not in available]
        if missing:
            raise RunnerError(f"Unknown task(s): {', '.join(missing)}")
        selected = list(requested)

    if limit is not None:
        if limit < 0:
            raise RunnerError("--limit must be non-negative")
        selected = selected[:limit]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=SUPPORTED_LANGUAGES,
        default="python",
        help="Benchmark language subset. Default: python.",
    )
    parser.add_argument(
        "--exercises",
        required=True,
        help="Path to polyglot-benchmark/<language>/exercises/practice.",
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Directory where fresh per-task working copies are created.",
    )
    parser.add_argument(
        "--results",
        default="results.jsonl",
        help="JSONL results path. Defaults to ./results.jsonl.",
    )
    parser.add_argument("--limit", type=int, help="Run at most N selected tasks.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-task JARVIS timeout in seconds. Default: {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--pytest-timeout",
        type=int,
        default=DEFAULT_PYTEST_TIMEOUT_SECONDS,
        help=(
            "Final test timeout in seconds. Reused for pytest, go test, npm test, "
            f"and npm install cache misses. Default: {DEFAULT_PYTEST_TIMEOUT_SECONDS}."
        ),
    )
    parser.add_argument("--tasks", help="Comma-separated task names to run in that order.")
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
        help="Skip --bench-conv isolation flag (diagnostic: bench-conv session-replacement suspect).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        exercises = resolve_path(args.exercises)
        workdir = resolve_path(args.workdir)
        results_path = resolve_path(args.results)
        if args.timeout <= 0:
            raise RunnerError("--timeout must be positive")
        if args.pytest_timeout <= 0:
            raise RunnerError("--pytest-timeout must be positive")

        # Give slow reasoning models room before the auto-prompts stall
        # watchdog aborts the turn (default is 180s; bench tasks need more).
        os.environ.setdefault("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS", "900000")

        workdir.mkdir(parents=True, exist_ok=True)
        all_tasks = discover_tasks(exercises)
        selected_tasks = select_tasks(all_tasks, parse_task_filter(args.tasks), args.limit)
        completed = load_completed_records(results_path, args.language)

        if args.language != "python":
            print(f"language:  {args.language}")
        print(f"exercises: {exercises}")
        print(f"workdir:   {workdir}")
        print(f"results:   {results_path}")
        print(f"selected:  {len(selected_tasks)} task(s)")

        for task in selected_tasks:
            key = selected_task_key(task, args.language)
            if key in completed:
                print(f"[skip] {task}: already recorded")
                continue

            record = run_task(args, exercises, workdir, task)
            jsonl_append(results_path, record)
            completed[key] = record
            status = "PASS" if record["pass"] else "FAIL"
            timeout_note = " timeout" if record["timed_out"] else ""
            test_note = " test_modified" if record["test_modified"] else ""
            print(f"[done] {task}: {status}{timeout_note}{test_note}")

        print_summary(load_completed_records(results_path, args.language))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except RunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: command or file not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
