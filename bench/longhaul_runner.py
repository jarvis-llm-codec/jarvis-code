#!/usr/bin/env python3
"""Prepare, drive, and collect LongHaul-Bench runs for JARVIS Code.

This runner is intentionally outside the public longhaul-bench harness.  It
bridges the benchmark script format into JARVIS Code's existing auto-prompts
interface, then reconstructs the LongHaul evidence bundle from the JARVIS bench
conversation archive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable

try:
    import polyglot_runner as agent_exec
except ImportError:  # pragma: no cover - defensive for unusual invocation paths.
    sys.path.append(str(Path(__file__).resolve().parent))
    import polyglot_runner as agent_exec


RunnerError = agent_exec.RunnerError
jsonl_append = agent_exec.jsonl_append
utc_now_iso = agent_exec.utc_now_iso

ALLOWED_KINDS = {"plant", "mutate", "filler", "probe", "trap", "exam"}
ALLOWED_TIERS = {100, 300, 500, 1000}
LEDGER_EVENTS = {"compaction_observed", "retry", "rate_limit_stall", "session_restart", "format_violation"}
TRANSCRIPT_EVENTS = LEDGER_EVENTS | {"timeout"}

META_FILENAME = "longhaul_runner_meta.json"
RUN_STATE_FILENAME = "longhaul_runner_run_state.json"
EVENTS_FILENAME = "longhaul_runner_events.jsonl"
PROMPT_MAP_FILENAME = "prompt_map.jsonl"
SCRIPT_COPY_FILENAME = "script.jsonl"
DEFAULT_MODEL_LABEL = "gpt-5.5 (openai-codex subscription)"
DRIVE_CONTRACT = "external:jarvis-code/bench/longhaul_runner.py"
AUTO_PROMPT_ENGINE_EVIDENCE = (
    "pi/packages/coding-agent/examples/extensions/jarvis-jlc.ts:"
    "loadAutoPromptState splits the file on CR/LF and trims nonempty lines"
)


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise RunnerError(f"Expected JSON object in {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(record, dict):
                raise RunnerError(f"{path}:{line_no}: expected JSON object")
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_for_line_auto_prompts(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text).strip()


def validate_script(records: list[dict[str, Any]]) -> None:
    if not records:
        raise RunnerError("script.jsonl has no records")
    for index, record in enumerate(records, start=1):
        missing = {"turn", "kind", "text"} - set(record)
        if missing:
            raise RunnerError(f"script:{index}: missing keys: {', '.join(sorted(missing))}")
        if record["turn"] != index:
            raise RunnerError(f"script:{index}: expected turn {index}, got {record.get('turn')!r}")
        if record["kind"] not in ALLOWED_KINDS:
            raise RunnerError(f"script:{index}: unexpected kind {record.get('kind')!r}")
        if not isinstance(record["text"], str):
            raise RunnerError(f"script:{index}: text must be string")


def bench_conv_id(seed: int, tier: int) -> str:
    return f"longhaul-s{seed}-t{tier}"


def prompt_dir(workdir: Path, label: str) -> Path:
    return workdir / "prompts" / label


def prepare_command(args: argparse.Namespace) -> int:
    script_path = resolve_path(args.script)
    workdir = resolve_path(args.out)
    if not script_path.exists():
        raise RunnerError(f"script not found: {script_path}")

    script = read_jsonl(script_path)
    validate_script(script)

    tier = int(args.tier) if args.tier is not None else len(script)
    seed = int(args.seed)
    if tier not in ALLOWED_TIERS:
        print(f"[warn] tier {tier} is outside LongHaul's published tiers {sorted(ALLOWED_TIERS)}", file=sys.stderr)

    workdir.mkdir(parents=True, exist_ok=True)
    script_copy = workdir / SCRIPT_COPY_FILENAME
    if script_path.resolve(strict=False) != script_copy.resolve(strict=False):
        shutil.copy2(script_path, script_copy)

    prompt_records: list[dict[str, Any]] = []
    delivered_prompts: list[str] = []
    any_flattened = False
    for record in script:
        original = record["text"]
        delivered = flatten_for_line_auto_prompts(original)
        if not delivered:
            raise RunnerError(f"script:{record['turn']}: prompt becomes empty after auto-prompts flattening")
        flattened = delivered != original.strip()
        any_flattened = any_flattened or flattened
        delivered_prompts.append(delivered)
        prompt_map: dict[str, Any] = {
            "delivered_sha256": sha256_text(delivered),
            "flattened": flattened,
            "kind": record["kind"],
            "original_sha256": sha256_text(original),
            "prompt_sha256": sha256_text(delivered),
            "turn": record["turn"],
        }
        if "refs" in record:
            prompt_map["refs"] = record["refs"]
        prompt_records.append(prompt_map)

    full_prompt_dir = prompt_dir(workdir, "full")
    full_prompt_file = full_prompt_dir / "auto_prompts.txt"
    full_prompt_dir.mkdir(parents=True, exist_ok=True)
    full_prompt_file.write_text("\n".join(delivered_prompts) + "\n", encoding="utf-8", newline="\n")
    write_jsonl(workdir / PROMPT_MAP_FILENAME, prompt_records)

    metadata = {
        "auto_prompt_engine_evidence": AUTO_PROMPT_ENGINE_EVIDENCE,
        "auto_prompt_mode": "line_file",
        "auto_prompts_path": str(full_prompt_file),
        "bench_conv": bench_conv_id(seed, tier),
        "created_at": utc_now_iso(),
        "hidden_answer_key_policy": "script_only; answer_key.jsonl and manifest.json are not copied by this runner",
        "prepare_mode": "flattened_line_auto_prompts",
        "prompt_count": len(script),
        "prompt_flattening": any_flattened,
        "prompt_map_path": str(workdir / PROMPT_MAP_FILENAME),
        "public_longhaul_repo_touched": False,
        "script_copy_path": str(script_copy),
        "script_sha256": sha256_file(script_copy),
        "seed": seed,
        "selected_path": "3_flatten_newlines_to_spaces",
        "selected_path_reason": (
            "JARVIS auto-prompts currently accepts one trimmed prompt per line; "
            "runner flattening avoids Pi engine surgery for this pilot."
        ),
        "tier": tier,
        "workdir": str(workdir),
    }
    write_json(workdir / META_FILENAME, metadata)

    print(
        f"prepared {len(script)} prompts for {metadata['bench_conv']} "
        f"at {workdir} (prompt_flattening={any_flattened})"
    )
    print(f"auto_prompts={full_prompt_file}")
    return 0


def load_prepare_meta(workdir: Path) -> dict[str, Any]:
    path = workdir / META_FILENAME
    if not path.exists():
        raise RunnerError(f"missing {META_FILENAME}; run prepare first: {path}")
    meta = read_json(path)
    for key in ("bench_conv", "prompt_count", "seed", "tier"):
        if key not in meta:
            raise RunnerError(f"{path}: missing {key}")
    return meta


def load_prompt_map(workdir: Path) -> list[dict[str, Any]]:
    path = workdir / PROMPT_MAP_FILENAME
    if not path.exists():
        raise RunnerError(f"missing {PROMPT_MAP_FILENAME}: {path}")
    records = read_jsonl(path)
    for index, record in enumerate(records, start=1):
        if record.get("turn") != index:
            raise RunnerError(f"{path}:{index}: expected turn {index}, got {record.get('turn')!r}")
        if not isinstance(record.get("delivered_sha256"), str):
            raise RunnerError(f"{path}:{index}: missing delivered_sha256")
    return records


def active_prompt_file_for_run(workdir: Path, prompt_map: list[dict[str, Any]], limit: int | None) -> tuple[str, Path, int]:
    full_file = prompt_dir(workdir, "full") / "auto_prompts.txt"
    if limit is None:
        if not full_file.exists():
            raise RunnerError(f"missing auto prompts file: {full_file}")
        return "full", full_file, len(prompt_map)
    if limit <= 0:
        raise RunnerError("--limit must be positive")
    target = min(limit, len(prompt_map))
    source_lines = full_file.read_text(encoding="utf-8").splitlines()
    if len(source_lines) < target:
        raise RunnerError(f"{full_file}: expected at least {target} prompt lines, got {len(source_lines)}")
    label = f"limit_{target}"
    limited_dir = prompt_dir(workdir, label)
    limited_file = limited_dir / "auto_prompts.txt"
    limited_dir.mkdir(parents=True, exist_ok=True)
    limited_file.write_text("\n".join(source_lines[:target]) + "\n", encoding="utf-8", newline="\n")
    return label, limited_file, target


def read_progress(progress_file: Path) -> int:
    if not progress_file.exists():
        return 0
    try:
        value = read_json(progress_file)
    except Exception:
        return 0
    idx = value.get("idx")
    if not isinstance(idx, (int, float)):
        return 0
    return max(0, int(idx))


def build_jarvis_command(args: argparse.Namespace, meta: dict[str, Any], prompt_file: Path) -> list[str]:
    command = [
        *agent_exec.resolve_jarvis_command(args.jarvis_cmd),
        "--yolo",
        "--recent-turns",
        "0",
        "--bench-conv",
        str(meta["bench_conv"]),
        "--auto-prompts",
        str(prompt_file.resolve()),
    ]
    if args.provider:
        command.extend(["--provider", args.provider])
    if args.model:
        command.extend(["--model", args.model])
    return command


def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("JARVIS_AUTO_RESET_EVERY", "0")
    env.setdefault("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS", "0")
    return env


def spawn_with_temp_env(cmdline: str, cwd: Path, env: dict[str, str]) -> Any:
    import winpty  # pywinpty

    overrides = {
        "JARVIS_AUTO_RESET_EVERY": env.get("JARVIS_AUTO_RESET_EVERY", "0"),
        "JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS": env.get("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS", "0"),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        return winpty.PtyProcess.spawn(cmdline, cwd=str(cwd), dimensions=(40, 120))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def monitor_until_exit(
    *,
    is_alive: Callable[[], bool],
    progress_file: Path,
    target: int,
    events_path: Path,
    stall_warn_seconds: int,
) -> None:
    last_idx = read_progress(progress_file)
    last_progress_at = time.monotonic()
    warned_at_idx: int | None = None
    print(f"[progress] {last_idx}/{target}")
    while is_alive():
        idx = min(read_progress(progress_file), target)
        if idx != last_idx:
            last_idx = idx
            last_progress_at = time.monotonic()
            warned_at_idx = None
            print(f"[progress] {last_idx}/{target}")
        if (
            stall_warn_seconds > 0
            and warned_at_idx != last_idx
            and time.monotonic() - last_progress_at >= stall_warn_seconds
        ):
            warned_at_idx = last_idx
            print(
                f"[warn] no auto-prompts progress for {stall_warn_seconds}s "
                f"at {last_idx}/{target}; leaving JARVIS running",
                file=sys.stderr,
            )
            jsonl_append(
                events_path,
                {
                    "event": "rate_limit_stall",
                    "idx": last_idx,
                    "seconds": stall_warn_seconds,
                    "ts": utc_now_iso(),
                },
            )
        time.sleep(2.0)


def run_process_observed(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    progress_file: Path,
    target: int,
    events_path: Path,
    stall_warn_seconds: int,
) -> tuple[int | None, float]:
    start = time.monotonic()
    stdout_path = cwd / "jarvis.stdout.log"
    if os.name == "nt" and os.environ.get("JLC_BENCH_NO_PTY") != "1":
        try:
            cmdline = subprocess.list2cmdline(command)
            proc = spawn_with_temp_env(cmdline, cwd, env)

            def drain() -> None:
                with stdout_path.open("ab") as log_fh:
                    while True:
                        try:
                            data = proc.read(4096)
                        except (EOFError, OSError):
                            return
                        if data:
                            log_fh.write(data.encode("utf-8", "replace"))
                            log_fh.flush()

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            monitor_until_exit(
                is_alive=proc.isalive,
                progress_file=progress_file,
                target=target,
                events_path=events_path,
                stall_warn_seconds=stall_warn_seconds,
            )
            reader.join(timeout=10)
            try:
                exit_code: int | None = proc.exitstatus
            except Exception:
                exit_code = None
            return exit_code, time.monotonic() - start
        except ImportError:
            print("[warn] pywinpty not installed; falling back to plain pipes", file=sys.stderr)

    stderr_path = cwd / "jarvis.stderr.log"
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    with stdout_path.open("ab") as out_fh, stderr_path.open("ab") as err_fh:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=out_fh,
            stderr=err_fh,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        monitor_until_exit(
            is_alive=lambda: proc.poll() is None,
            progress_file=progress_file,
            target=target,
            events_path=events_path,
            stall_warn_seconds=stall_warn_seconds,
        )
        return proc.poll(), time.monotonic() - start


def load_run_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def run_command(args: argparse.Namespace) -> int:
    workdir = resolve_path(args.workdir)
    meta = load_prepare_meta(workdir)
    prompt_map = load_prompt_map(workdir)
    label, prompt_file, target = active_prompt_file_for_run(workdir, prompt_map, args.limit)
    progress_file = prompt_file.parent / ".auto_progress.json"
    events_path = workdir / EVENTS_FILENAME
    state_path = workdir / RUN_STATE_FILENAME

    command = build_jarvis_command(args, meta, prompt_file)
    env = child_env()
    existing_state = load_run_state(state_path)
    same_run = (
        existing_state is not None
        and existing_state.get("active_prompts_path") == str(prompt_file)
        and existing_state.get("target_turns") == target
    )
    started_at = str(existing_state.get("started_at")) if same_run and existing_state.get("started_at") else utc_now_iso()
    restarted = bool(existing_state.get("resumed")) if same_run else False
    launches = int(existing_state.get("launches", 0)) if same_run else 0
    session_restarts = int(existing_state.get("session_restarts", 0)) if same_run else 0
    max_restarts = int(args.max_restarts)

    write_json(
        state_path,
        {
            "active_label": label,
            "active_prompts_path": str(prompt_file),
            "bench_conv": meta["bench_conv"],
            "command": command,
            "completed_idx": read_progress(progress_file),
            "env_contract": {
                "JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS": env.get("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS"),
                "JARVIS_AUTO_RESET_EVERY": env.get("JARVIS_AUTO_RESET_EVERY"),
            },
            "launches": launches,
            "limit": args.limit,
            "resumed": restarted,
            "session_restarts": session_restarts,
            "started_at": started_at,
            "target_turns": target,
            "updated_at": utc_now_iso(),
        },
    )

    print("command=" + subprocess.list2cmdline(command))
    print(f"bench_conv={meta['bench_conv']} prompts={prompt_file} target={target}")
    while True:
        before_idx = min(read_progress(progress_file), target)
        if before_idx >= target:
            print(f"completed {before_idx}/{target}; nothing to launch")
            break

        launches += 1
        print(f"launch {launches}: starting at cursor {before_idx}/{target}")
        exit_code, wall_seconds = run_process_observed(
            command,
            cwd=workdir,
            env=env,
            progress_file=progress_file,
            target=target,
            events_path=events_path,
            stall_warn_seconds=int(args.stall_warn_seconds),
        )
        after_idx = min(read_progress(progress_file), target)
        jsonl_append(
            events_path,
            {
                "duration_seconds": round(wall_seconds, 3),
                "event": "process_exit",
                "exit_code": exit_code,
                "idx_after": after_idx,
                "idx_before": before_idx,
                "launch": launches,
                "ts": utc_now_iso(),
            },
        )
        if after_idx >= target:
            print(f"completed {after_idx}/{target}")
            break

        session_restarts += 1
        restarted = True
        jsonl_append(
            events_path,
            {
                "event": "session_restart",
                "exit_code": exit_code,
                "idx_after": after_idx,
                "idx_before": before_idx,
                "launch": launches,
                "ts": utc_now_iso(),
            },
        )
        write_json(
            state_path,
            {
                "active_label": label,
                "active_prompts_path": str(prompt_file),
                "bench_conv": meta["bench_conv"],
                "command": command,
                "completed_idx": after_idx,
                "env_contract": {
                    "JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS": env.get("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS"),
                    "JARVIS_AUTO_RESET_EVERY": env.get("JARVIS_AUTO_RESET_EVERY"),
                },
                "launches": launches,
                "limit": args.limit,
                "resumed": restarted,
                "session_restarts": session_restarts,
                "started_at": started_at,
                "target_turns": target,
                "updated_at": utc_now_iso(),
            },
        )
        if session_restarts > max_restarts:
            raise RunnerError(
                f"JARVIS exited before completing prompts after {session_restarts} restarts "
                f"({after_idx}/{target}); see {events_path}"
            )
        print(f"[restart] process ended before completion; relaunching from cursor {after_idx}/{target}")

    completed_idx = min(read_progress(progress_file), target)
    write_json(
        state_path,
        {
            "active_label": label,
            "active_prompts_path": str(prompt_file),
            "bench_conv": meta["bench_conv"],
            "command": command,
            "completed_idx": completed_idx,
            "env_contract": {
                "JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS": env.get("JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS"),
                "JARVIS_AUTO_RESET_EVERY": env.get("JARVIS_AUTO_RESET_EVERY"),
            },
            "finished_at": utc_now_iso(),
            "launches": launches,
            "limit": args.limit,
            "resumed": restarted,
            "session_restarts": session_restarts,
            "started_at": started_at,
            "target_turns": target,
            "updated_at": utc_now_iso(),
        },
    )
    return 0


def parse_iso(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def include_timestamp(record_ts: Any, since: dt.datetime | None) -> bool:
    if since is None:
        return True
    parsed = parse_iso(record_ts)
    return parsed is not None and parsed >= since


def bench_store_root(args: argparse.Namespace) -> Path:
    if args.raw_bench_store:
        return resolve_path(args.raw_bench_store)
    configured = os.environ.get("JARVIS_RAW_BENCH_STORE")
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    return (Path("~/.jarvis-code/raw-store").expanduser().parent / "conversation_bench_archive").resolve(strict=False)


def load_archive(archive_path: Path, since: dt.datetime | None) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
    turns: list[tuple[int, dict[str, Any]]] = []
    meters: list[tuple[int, dict[str, Any]]] = []
    if not archive_path.exists():
        return turns, meters
    with archive_path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or not include_timestamp(record.get("timestamp"), since):
                continue
            if record.get("kind") == "meter":
                meters.append((line_no, record))
            elif record.get("kind") in {"encoder"}:
                continue
            elif "user" in record or "assistant" in record:
                turns.append((line_no, record))
    return turns, meters


def parse_paperlog_line(line: str) -> dict[str, Any] | None:
    if " | question=" not in line or " | answer=" not in line:
        return None
    try:
        timestamp, rest = line.split(" | ", 1)
        question_part = rest.split(" | question=", 1)[1]
        question, answer_part = question_part.split(" | answer=", 1)
        answer, tail = answer_part, ""
        if " | [jlc:meter]" in answer_part:
            answer, tail = answer_part.split(" | [jlc:meter]", 1)
            tail = "[jlc:meter]" + tail
        elif " | prompt_tag=" in answer_part:
            answer = answer_part.split(" | prompt_tag=", 1)[0]
    except ValueError:
        return None
    turn_match = re.search(r"(?:^|\s\|\s)turn=(\d+)", line)
    return {
        "answer": answer.strip(),
        "meter_line": tail.strip() or None,
        "question": question.strip(),
        "timestamp": timestamp.strip(),
        "turn": int(turn_match.group(1)) if turn_match else None,
    }


def load_paperlog(path: Path, since: dt.datetime | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parsed = parse_paperlog_line(line)
        if parsed is None or not include_timestamp(parsed.get("timestamp"), since):
            continue
        records.append(parsed)
    return records


def parse_compact_count(raw: str) -> int | None:
    value = raw.strip().replace(",", "")
    if not value:
        return None
    multiplier = 1
    suffix = value[-1].lower()
    if suffix == "k":
        multiplier = 1_000
        value = value[:-1]
    elif suffix == "m":
        multiplier = 1_000_000
        value = value[:-1]
    try:
        return max(0, int(round(float(value) * multiplier)))
    except ValueError:
        return None


def parse_meter_usage(meter_line: str | None) -> tuple[int | None, int | None]:
    if not meter_line:
        return None, None
    match = re.search(r"chat\[in=([^\s\]]+)\s+out=([^\s\]]+)", meter_line)
    if not match:
        return None, None
    return parse_compact_count(match.group(1)), parse_compact_count(match.group(2))


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def load_runner_events(path: Path, since: dt.datetime | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and include_timestamp(record.get("ts"), since):
            events.append(record)
    return events


def event_turn(event: dict[str, Any], target: int) -> int | None:
    idx = event.get("idx_after", event.get("idx"))
    if not isinstance(idx, (int, float)):
        return None
    turn = int(idx) + 1
    if target <= 0:
        return None
    return max(1, min(turn, target))


def events_by_turn(events: list[dict[str, Any]], target: int) -> dict[int, list[str]]:
    mapped: dict[int, list[str]] = {}
    for event in events:
        raw_name = event.get("event")
        if raw_name not in LEDGER_EVENTS:
            continue
        turn = event_turn(event, target)
        if turn is None:
            continue
        bucket = mapped.setdefault(turn, [])
        if raw_name not in bucket:
            bucket.append(str(raw_name))
    return mapped


def choose_meter_for_turn(
    turn_index: int,
    meters: list[tuple[int, dict[str, Any]]],
    paperlog: list[dict[str, Any]],
) -> tuple[str | None, str]:
    explicit = [record for _line_no, record in meters if record.get("turn") == turn_index]
    if explicit:
        return str(explicit[-1].get("meter_line") or ""), "raw_meter_explicit_turn"
    if len(meters) >= turn_index:
        return str(meters[turn_index - 1][1].get("meter_line") or ""), "raw_meter_order"
    paper_explicit = [record for record in paperlog if record.get("turn") == turn_index and record.get("meter_line")]
    if paper_explicit:
        return str(paper_explicit[-1].get("meter_line") or ""), "paperlog_explicit_turn"
    if len(paperlog) >= turn_index and paperlog[turn_index - 1].get("meter_line"):
        return str(paperlog[turn_index - 1].get("meter_line") or ""), "paperlog_order"
    return None, "none"


def collect_command(args: argparse.Namespace) -> int:
    workdir = resolve_path(args.workdir)
    out_dir = resolve_path(args.out)
    meta = load_prepare_meta(workdir)
    prompt_map = load_prompt_map(workdir)
    run_state = load_run_state(workdir / RUN_STATE_FILENAME) or {}
    target = int(run_state.get("target_turns") or len(prompt_map))
    target = min(target, len(prompt_map))
    since = parse_iso(run_state.get("started_at"))
    prompt_subset = prompt_map[:target]

    store_root = bench_store_root(args)
    bench_conv = str(meta["bench_conv"])
    archive_path = store_root / f"{bench_conv}.jsonl"
    paperlog_path = store_root / f"{bench_conv}.paperlog"
    turns, meters = load_archive(archive_path, since)
    paperlog = load_paperlog(paperlog_path, since)
    runner_events = load_runner_events(workdir / EVENTS_FILENAME, since)
    turn_events = events_by_turn(runner_events, target)

    transcript: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    warnings: list[str] = []
    usable_turns = turns[:target]

    for index, prompt_record in enumerate(prompt_subset, start=1):
        if len(usable_turns) < index:
            break
        line_no, turn_record = usable_turns[index - 1]
        response = str(turn_record.get("assistant") or "")
        user_text = str(turn_record.get("user") or "")
        delivered_sha = str(prompt_record["delivered_sha256"])
        if sha256_text(user_text) != delivered_sha:
            warnings.append(
                f"archive line {line_no} did not match delivered prompt sha for turn {index}; "
                "using script order mapping"
            )
        events = turn_events.get(index, [])
        started_at = str(turn_record.get("timestamp") or run_state.get("started_at") or "")
        finished_at = started_at
        transcript.append(
            {
                "events": [event for event in events if event in TRANSCRIPT_EVENTS],
                "finished_at": finished_at,
                "prompt_sha256": delivered_sha,
                "response_text": response,
                "started_at": started_at,
                "turn": index,
            }
        )

        meter_line, meter_source = choose_meter_for_turn(index, meters, paperlog)
        agent_in, agent_out = parse_meter_usage(meter_line)
        notes = (
            "provider=null; provider_usage_not_confirmed_by_runner; "
            f"agent_submitted_source={meter_source}; harness_estimate_method=chars_div_4_fallback"
        )
        ledger.append(
            {
                "events": [event for event in events if event in LEDGER_EVENTS],
                "input_tokens": {
                    "agent_submitted": agent_in,
                    "harness_estimate": estimate_tokens(user_text),
                    "provider": None,
                },
                "notes": notes,
                "output_tokens": {
                    "agent_submitted": agent_out,
                    "harness_estimate": estimate_tokens(response),
                    "provider": None,
                },
                "turn": index,
                "wall_ms": None,
            }
        )

    turns_completed = len(transcript)
    tier = int(meta["tier"])
    limited_run = run_state.get("limit") is not None
    dnf = turns_completed < tier
    dnf_reason = None
    if dnf:
        if limited_run:
            dnf_reason = f"partial_limit_{run_state.get('limit')}: completed {turns_completed}/{tier}"
        else:
            dnf_reason = f"incomplete_archive: completed {turns_completed}/{tier}"
    started_at = transcript[0]["started_at"] if transcript else str(run_state.get("started_at") or utc_now_iso())
    finished_at = transcript[-1]["finished_at"] if transcript else str(run_state.get("finished_at") or utc_now_iso())
    run_meta = {
        "agent": {
            "drive_contract": DRIVE_CONTRACT,
            "model": str(args.model_label or DEFAULT_MODEL_LABEL),
            "name": "jarvis-code",
        },
        "auto_prompt_mode": meta.get("auto_prompt_mode"),
        "benchmark": "longhaul-bench",
        "bench_conv": bench_conv,
        "collect_warnings": warnings,
        "dnf": dnf,
        "dnf_reason": dnf_reason,
        "estimator": {
            "input": "delivered_prompt",
            "method": "chars_div_4_fallback",
            "output": "captured_response",
        },
        "finished_at": finished_at,
        "hidden_answer_key_policy": meta.get("hidden_answer_key_policy"),
        "prompt_flattening": bool(meta.get("prompt_flattening")),
        "prompt_flattening_reason": meta.get("selected_path_reason"),
        "provider_token_source": None,
        "resumed": bool(run_state.get("resumed") or run_state.get("session_restarts")),
        "run_state": {
            "active_label": run_state.get("active_label"),
            "active_prompts_path": run_state.get("active_prompts_path"),
            "env_contract": run_state.get("env_contract", {}),
            "launches": run_state.get("launches"),
            "session_restarts": run_state.get("session_restarts", 0),
            "target_turns": target,
        },
        "seed": int(meta["seed"]),
        "started_at": started_at,
        "tier": tier,
        "token_ledger_policy": (
            "provider is null unless runner can independently confirm provider/API usage; "
            "JLC meter is recorded under agent_submitted only"
        ),
        "turns_completed": turns_completed,
    }

    validate_transcript(transcript)
    validate_ledger(ledger)
    validate_run_meta(run_meta)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "transcript.jsonl", transcript)
    write_jsonl(out_dir / "ledger.jsonl", ledger)
    write_json(out_dir / "run_meta.json", run_meta)
    write_json(
        out_dir / "collect_summary.json",
        {
            "archive_path": str(archive_path),
            "ledger_count": len(ledger),
            "paperlog_path": str(paperlog_path),
            "run_meta_path": str(out_dir / "run_meta.json"),
            "transcript_count": len(transcript),
            "warnings": warnings,
        },
    )

    print(
        f"collected transcript={len(transcript)} ledger={len(ledger)} "
        f"dnf={dnf} out={out_dir}"
    )
    if warnings:
        print(f"[warn] {len(warnings)} mapping warning(s); see run_meta.json")
    return 0


def validate_count3(record: Any, path: str) -> None:
    if not isinstance(record, dict):
        raise RunnerError(f"{path}: must be object")
    if set(record) != {"agent_submitted", "harness_estimate", "provider"}:
        raise RunnerError(f"{path}: expected provider/harness_estimate/agent_submitted")
    for key, value in record.items():
        if value is not None and (not isinstance(value, int) or value < 0):
            raise RunnerError(f"{path}.{key}: expected nonnegative integer or null")


def validate_transcript(records: list[dict[str, Any]]) -> None:
    allowed = {"events", "finished_at", "prompt_sha256", "response_text", "started_at", "turn"}
    for index, record in enumerate(records, start=1):
        if not {"turn", "response_text"} <= set(record):
            raise RunnerError(f"transcript:{index}: missing required fields")
        if set(record) - allowed:
            raise RunnerError(f"transcript:{index}: unknown fields {sorted(set(record) - allowed)}")
        if not isinstance(record["turn"], int) or record["turn"] < 1:
            raise RunnerError(f"transcript:{index}: bad turn")
        if not isinstance(record["response_text"], str):
            raise RunnerError(f"transcript:{index}: response_text must be string")
        prompt_sha = record.get("prompt_sha256")
        if prompt_sha is not None and (not isinstance(prompt_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", prompt_sha)):
            raise RunnerError(f"transcript:{index}: bad prompt_sha256")
        events = record.get("events", [])
        if not isinstance(events, list) or any(event not in TRANSCRIPT_EVENTS for event in events):
            raise RunnerError(f"transcript:{index}: bad events")


def validate_ledger(records: list[dict[str, Any]]) -> None:
    allowed = {"events", "input_tokens", "notes", "output_tokens", "turn", "wall_ms"}
    for index, record in enumerate(records, start=1):
        if not {"turn", "input_tokens", "output_tokens"} <= set(record):
            raise RunnerError(f"ledger:{index}: missing required fields")
        if set(record) - allowed:
            raise RunnerError(f"ledger:{index}: unknown fields {sorted(set(record) - allowed)}")
        if not isinstance(record["turn"], int) or record["turn"] < 1:
            raise RunnerError(f"ledger:{index}: bad turn")
        validate_count3(record["input_tokens"], f"ledger:{index}.input_tokens")
        validate_count3(record["output_tokens"], f"ledger:{index}.output_tokens")
        wall_ms = record.get("wall_ms")
        if wall_ms is not None and (not isinstance(wall_ms, int) or wall_ms < 0):
            raise RunnerError(f"ledger:{index}: wall_ms must be nonnegative integer or null")
        events = record.get("events", [])
        if not isinstance(events, list) or any(event not in LEDGER_EVENTS for event in events):
            raise RunnerError(f"ledger:{index}: bad events")
        if "notes" in record and not isinstance(record["notes"], str):
            raise RunnerError(f"ledger:{index}: notes must be string")


def validate_run_meta(record: dict[str, Any]) -> None:
    required = {"benchmark", "seed", "tier", "agent", "turns_completed", "dnf"}
    missing = required - set(record)
    if missing:
        raise RunnerError(f"run_meta: missing {', '.join(sorted(missing))}")
    if record["benchmark"] != "longhaul-bench":
        raise RunnerError("run_meta: benchmark must be longhaul-bench")
    if not isinstance(record["seed"], int):
        raise RunnerError("run_meta: seed must be integer")
    if record["tier"] not in ALLOWED_TIERS:
        raise RunnerError(f"run_meta: tier must be one of {sorted(ALLOWED_TIERS)}")
    agent = record.get("agent")
    if not isinstance(agent, dict) or not {"name", "model", "drive_contract"} <= set(agent):
        raise RunnerError("run_meta: agent must include name/model/drive_contract")
    if not isinstance(record["turns_completed"], int) or record["turns_completed"] < 0:
        raise RunnerError("run_meta: turns_completed must be nonnegative integer")
    if not isinstance(record["dnf"], bool):
        raise RunnerError("run_meta: dnf must be boolean")
    estimator = record.get("estimator")
    if estimator is not None and (not isinstance(estimator, dict) or not isinstance(estimator.get("method"), str)):
        raise RunnerError("run_meta: estimator.method must be string")
    if "resumed" in record and not isinstance(record["resumed"], bool):
        raise RunnerError("run_meta: resumed must be boolean")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="convert script.jsonl to JARVIS auto-prompts input")
    prepare.add_argument("--script", required=True, help="LongHaul script.jsonl path")
    prepare.add_argument("--out", required=True, help="runner work directory")
    prepare.add_argument("--seed", type=int, default=42, help="LongHaul seed for bench-conv id")
    prepare.add_argument("--tier", type=int, default=None, help="LongHaul tier; defaults to script length")
    prepare.set_defaults(func=prepare_command)

    run = subparsers.add_parser("run", help="drive JARVIS through the prepared prompts")
    run.add_argument("--workdir", required=True, help="directory produced by prepare")
    run.add_argument("--limit", type=int, default=None, help="smoke-run first N prompts in an isolated prompt dir")
    run.add_argument("--jarvis-cmd", default="jarvis", help="jarvis executable or wrapper")
    run.add_argument("--provider", default=None, help="optional provider passed through to jarvis")
    run.add_argument("--model", default=None, help="optional model passed through to jarvis")
    run.add_argument("--stall-warn-seconds", type=int, default=20 * 60, help="warn after this many seconds with no progress")
    run.add_argument("--max-restarts", type=int, default=3, help="max process relaunches after premature exits")
    run.set_defaults(func=run_command)

    collect = subparsers.add_parser("collect", help="collect LongHaul transcript/ledger/run_meta artifacts")
    collect.add_argument("--workdir", required=True, help="directory produced by prepare/run")
    collect.add_argument("--out", required=True, help="artifact output directory")
    collect.add_argument("--raw-bench-store", default=None, help="override JARVIS_RAW_BENCH_STORE")
    collect.add_argument("--model-label", default=DEFAULT_MODEL_LABEL, help="run_meta agent.model label")
    collect.set_defaults(func=collect_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except RunnerError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
