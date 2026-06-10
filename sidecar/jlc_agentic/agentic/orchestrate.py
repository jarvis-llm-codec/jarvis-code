from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from jarvis_sidecar.raw_store import _storage_root

from .dispatcher import LEAN_READ_ONLY_TOOLS
from .schema import get_subagent_dispatcher
from .subagent import Subagent


class OrchestrationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ERROR = "error"


@dataclass
class OrchestrationBudget:
    max_calls: int | None = None
    max_tokens: int | None = None
    max_wallclock_sec: float | None = None


@dataclass
class OrchestrationSpec:
    task: str
    dimensions: list[str]
    # Both finders and the verifier are subagents: None => get_llm("subagent"),
    # i.e. they honor the configured roles.subagent (/model-setting). No hardcoded
    # per-use model. Set roles.subagent to a cheap model for cheap fan-out.
    worker_model: str | None = None
    verifier_model: str | None = None
    max_concurrency: int = 3
    budget: OrchestrationBudget = field(default_factory=OrchestrationBudget)
    project_root: str | None = None
    conv_id: str | None = None


@dataclass
class FinderOutcome:
    dimension: str
    ran: bool
    summary: str
    halt_reason: str
    in_tokens: int
    out_tokens: int
    elapsed_sec: float
    error: str | None = None


@dataclass
class OrchestrationResult:
    orchestration_id: str
    state: OrchestrationState
    summary: str
    finders_total: int
    finders_ran: int
    stop_reason: str | None
    finders: list[FinderOutcome]
    in_tokens: int
    out_tokens: int
    elapsed_sec: float
    event_log_path: str


FINDER_PROMPT = (
    "You are a finder examining the following task:\n{task}\n\n"
    "Focus EXCLUSIVELY on the '{dimension}' dimension. Use read and grep to ground "
    "EVERY claim in actual code with file:line. No baseless claims. If you find nothing "
    "in your dimension, say so explicitly. Return concise, grounded findings as your final message."
)

VERIFIER_PROMPT = (
    "You are the verifier and synthesizer for an orchestration of the task:\n{task}\n\n"
    "Finder dimensions: {dimensions}. Below are their raw findings:\n\n{findings_block}\n\n"
    "Adversarially verify each finding: DROP any not grounded in real code (re-check with "
    "read/grep if needed). Then synthesize a deduplicated list of CONFIRMED findings with "
    "file:line. Output a concise summary followed by the confirmed findings."
)


_CANCEL_EVENTS: dict[str, threading.Event] = {}
_CANCEL_LOCK = threading.Lock()
_LOG_LOCKS: dict[str, threading.Lock] = {}
_LOG_LOCKS_LOCK = threading.Lock()
_JOB_RESULTS: OrderedDict[str, OrchestrationResult] = OrderedDict()
_JOB_RUNNING: set[str] = set()
_JOB_LOCK = threading.Lock()
_JOB_RESULT_CAP = 64


def cancel(orchestration_id: str) -> bool:
    clean_id = str(orchestration_id or "").strip()
    if not clean_id:
        return False
    with _CANCEL_LOCK:
        ev = _CANCEL_EVENTS.get(clean_id)
    if ev is None:
        return False
    ev.set()
    return True


def run_orchestration(
    spec: OrchestrationSpec,
    *,
    orchestration_id: str | None = None,
    on_event: Callable[[dict], None] | None = None,
) -> OrchestrationResult:
    orchestration_id = str(orchestration_id or uuid.uuid4().hex[:12]).strip()
    ev = threading.Event()
    started = time.monotonic()
    event_log_path = _event_log_path(orchestration_id)
    with _CANCEL_LOCK:
        _CANCEL_EVENTS[orchestration_id] = ev

    dimensions = [str(dimension) for dimension in spec.dimensions]
    outcomes: list[FinderOutcome] = []
    calls_used = 0
    tokens_used = 0
    total_in_tokens = 0
    total_out_tokens = 0
    stop_reason: str | None = None
    state = OrchestrationState.RUNNING
    budget_exhausted = False

    def emit(event: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "orchestration_id": orchestration_id,
            "event": event,
            **payload,
        }
        _append_event(event_log_path, record)
        if on_event is not None:
            on_event(dict(record))

    def mark_budget_exhausted(which: str, running: int) -> str:
        nonlocal budget_exhausted
        if not budget_exhausted:
            budget_exhausted = True
            emit(
                "budget_exhausted",
                which=which,
                ran=len(outcomes),
                total=len(dimensions),
                in_flight=running,
            )
        return "budget_exhausted"

    def gate_stop_reason(running: int) -> str | None:
        nonlocal calls_used, tokens_used
        if ev.is_set():
            return "budget_exhausted" if budget_exhausted else "cancelled"
        budget = spec.budget
        if budget.max_wallclock_sec is not None:
            if time.monotonic() - started >= float(budget.max_wallclock_sec):
                ev.set()
                return mark_budget_exhausted("wallclock_sec", running)
        if budget.max_calls is not None:
            # Each submitted finder reserves one provider call. Actual completed
            # call counts may overshoot this soft cap by already in-flight work.
            if calls_used + running >= int(budget.max_calls):
                return mark_budget_exhausted("calls", running)
        if budget.max_tokens is not None and tokens_used >= int(budget.max_tokens):
            return mark_budget_exhausted("tokens", running)
        return None

    def run_finder(dimension: str) -> tuple[FinderOutcome, int]:
        try:
            result = Subagent(
                name="finder",
                system_prompt=FINDER_PROMPT.format(task=spec.task, dimension=dimension),
                model=spec.worker_model,
                read_only=True,
                allowed_tools=LEAN_READ_ONLY_TOOLS,
                should_cancel=lambda: ev.is_set(),
                dispatcher=_lean_dispatcher(spec),
                project_root=spec.project_root,
                conv_id=spec.conv_id,
                retriever=None,
                on_raw=lambda _line: None,
            ).run(spec.task)
            return (
                FinderOutcome(
                    dimension=dimension,
                    ran=True,
                    summary=result.summary,
                    halt_reason=result.halt_reason,
                    in_tokens=result.in_tokens,
                    out_tokens=result.out_tokens,
                    elapsed_sec=result.elapsed_sec,
                ),
                max(1, int(result.iters or 0)),
            )
        except Exception as exc:  # noqa: BLE001 - one finder must not kill fan-in
            return (
                FinderOutcome(
                    dimension=dimension,
                    ran=True,
                    summary="",
                    halt_reason="error",
                    in_tokens=0,
                    out_tokens=0,
                    elapsed_sec=0.0,
                    error=str(exc),
                ),
                1,
            )

    emit(
        "orchestration_start",
        task=spec.task,
        dimensions=dimensions,
        budget=asdict(spec.budget),
    )
    try:
        next_index = 0
        max_workers = max(1, int(spec.max_concurrency or 1))
        futures: dict[Any, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while next_index < len(dimensions) or futures:
                while (
                    stop_reason is None
                    and next_index < len(dimensions)
                    and len(futures) < max_workers
                ):
                    stop_reason = gate_stop_reason(len(futures))
                    if stop_reason is not None:
                        break
                    dimension = dimensions[next_index]
                    next_index += 1
                    emit("finder_start", dimension=dimension)
                    futures[pool.submit(run_finder, dimension)] = dimension

                if not futures:
                    break

                done, _pending = wait(
                    futures,
                    timeout=0.05,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    current_stop = gate_stop_reason(len(futures))
                    if stop_reason is None and current_stop is not None:
                        stop_reason = current_stop
                    continue

                for future in done:
                    dimension = futures.pop(future)
                    outcome, provider_calls = future.result()
                    outcomes.append(outcome)
                    calls_used += provider_calls
                    tokens_used += int(outcome.in_tokens or 0) + int(outcome.out_tokens or 0)
                    total_in_tokens += int(outcome.in_tokens or 0)
                    total_out_tokens += int(outcome.out_tokens or 0)
                    if outcome.error:
                        emit("finder_error", dimension=dimension, error=outcome.error)
                    else:
                        emit(
                            "finder_done",
                            dimension=dimension,
                            halt_reason=outcome.halt_reason,
                            in_tokens=outcome.in_tokens,
                            out_tokens=outcome.out_tokens,
                        )
                    if outcome.halt_reason == "cancelled" and ev.is_set() and stop_reason is None:
                        stop_reason = "cancelled"
                    current_stop = gate_stop_reason(len(futures))
                    if stop_reason is None and current_stop is not None:
                        stop_reason = current_stop

        finders_ran = sum(1 for outcome in outcomes if outcome.ran)
        summary = ""
        if finders_ran == 0 or ev.is_set() or stop_reason == "cancelled":
            if stop_reason is None and ev.is_set():
                stop_reason = "cancelled"
            state = _state_for_stop(stop_reason)
            summary = (
                f"(verification skipped: {stop_reason or 'none'}; "
                f"finders ran {finders_ran}/{len(dimensions)})"
            )
        else:
            emit("verify_start")
            try:
                verifier_result = Subagent(
                    name="verifier",
                    system_prompt=VERIFIER_PROMPT.format(
                        task=spec.task,
                        dimensions=", ".join(dimensions),
                        findings_block=_findings_block(outcomes),
                    ),
                    model=spec.verifier_model,
                    read_only=True,
                    allowed_tools=LEAN_READ_ONLY_TOOLS,
                    should_cancel=lambda: ev.is_set(),
                    dispatcher=_lean_dispatcher(spec),
                    project_root=spec.project_root,
                    conv_id=spec.conv_id,
                    retriever=None,
                    on_raw=lambda _line: None,
                ).run(_verify_task(spec.task, dimensions, outcomes))
                total_in_tokens += int(verifier_result.in_tokens or 0)
                total_out_tokens += int(verifier_result.out_tokens or 0)
                emit(
                    "verify_done",
                    in_tokens=verifier_result.in_tokens,
                    out_tokens=verifier_result.out_tokens,
                )
                summary = verifier_result.summary
                if not str(summary or "").strip():
                    # Verifier returned no synthesis (e.g. a flaky agent-sdk
                    # subagent emitting zero output). Don't claim a clean done with
                    # an empty summary or lose the finders' work (DNA #1: no silent
                    # failure) — fall back to the raw finder findings.
                    emit("verify_empty_fallback", finders_ran=finders_ran)
                    summary = (
                        "(verifier produced no synthesis — raw finder findings below)\n\n"
                        + _findings_block(outcomes)
                    )
                if verifier_result.halt_reason == "cancelled":
                    stop_reason = "cancelled"
                    state = OrchestrationState.CANCELLED
                elif stop_reason == "budget_exhausted":
                    state = OrchestrationState.BUDGET_EXHAUSTED
                else:
                    state = OrchestrationState.DONE
            except Exception as exc:  # noqa: BLE001
                summary = f"[error] verifier failed: {exc}"
                state = OrchestrationState.ERROR

        if stop_reason == "budget_exhausted" and state == OrchestrationState.RUNNING:
            state = OrchestrationState.BUDGET_EXHAUSTED
        elif stop_reason == "cancelled" and state == OrchestrationState.RUNNING:
            state = OrchestrationState.CANCELLED
        elif state == OrchestrationState.RUNNING:
            state = OrchestrationState.DONE

        result = OrchestrationResult(
            orchestration_id=orchestration_id,
            state=state,
            summary=summary,
            finders_total=len(dimensions),
            finders_ran=finders_ran,
            stop_reason=stop_reason,
            finders=outcomes,
            in_tokens=total_in_tokens,
            out_tokens=total_out_tokens,
            elapsed_sec=time.monotonic() - started,
            event_log_path=str(event_log_path),
        )
        emit(
            "orchestration_done",
            state=result.state.value,
            finders_ran=result.finders_ran,
            finders_total=result.finders_total,
            stop_reason=result.stop_reason,
        )
        return result
    finally:
        with _CANCEL_LOCK:
            _CANCEL_EVENTS.pop(orchestration_id, None)


def start_orchestration_job(
    spec: OrchestrationSpec,
    *,
    orchestration_id: str | None = None,
    on_event: Callable[[dict], None] | None = None,
    on_complete: Callable[[OrchestrationResult], None] | None = None,
) -> str:
    clean_id = str(orchestration_id or uuid.uuid4().hex[:12]).strip()
    if not clean_id:
        clean_id = uuid.uuid4().hex[:12]
    with _JOB_LOCK:
        if clean_id in _JOB_RUNNING:
            raise ValueError(f"orchestration already running: {clean_id}")
        _JOB_RUNNING.add(clean_id)

    def worker() -> None:
        started = time.monotonic()
        try:
            result = run_orchestration(spec, orchestration_id=clean_id, on_event=on_event)
        except Exception as exc:  # noqa: BLE001 - detached jobs must surface errors through result lookup
            result = OrchestrationResult(
                orchestration_id=clean_id,
                state=OrchestrationState.ERROR,
                summary=f"[error] orchestration failed: {exc}",
                finders_total=len(spec.dimensions),
                finders_ran=0,
                stop_reason="error",
                finders=[],
                in_tokens=0,
                out_tokens=0,
                elapsed_sec=time.monotonic() - started,
                event_log_path=str(_event_log_path(clean_id)),
            )
        with _JOB_LOCK:
            _JOB_RUNNING.discard(clean_id)
            _JOB_RESULTS[clean_id] = result
            _JOB_RESULTS.move_to_end(clean_id)
            while len(_JOB_RESULTS) > _JOB_RESULT_CAP:
                _JOB_RESULTS.popitem(last=False)
        if on_complete is not None:
            try:
                on_complete(result)
            except Exception:
                pass

    thread = threading.Thread(target=worker, name=f"orchestration-{clean_id}", daemon=True)
    thread.start()
    return clean_id


def get_orchestration_status(orchestration_id: str) -> dict[str, Any]:
    clean_id = str(orchestration_id or "").strip()
    if not clean_id:
        return {"orchestration_id": clean_id, "state": "unknown"}
    with _JOB_LOCK:
        if clean_id in _JOB_RUNNING:
            return {"orchestration_id": clean_id, "state": "running"}
        result = _JOB_RESULTS.get(clean_id)
        if result is not None:
            _JOB_RESULTS.move_to_end(clean_id)
    if result is None:
        return {"orchestration_id": clean_id, "state": "unknown"}
    return {
        "orchestration_id": clean_id,
        "state": result.state.value,
        "finders_ran": result.finders_ran,
        "finders_total": result.finders_total,
        "stop_reason": result.stop_reason,
    }


def get_orchestration_result(orchestration_id: str) -> OrchestrationResult | None:
    clean_id = str(orchestration_id or "").strip()
    if not clean_id:
        return None
    with _JOB_LOCK:
        result = _JOB_RESULTS.get(clean_id)
        if result is not None:
            _JOB_RESULTS.move_to_end(clean_id)
        return result


def _reset_jobs_for_tests() -> None:
    with _JOB_LOCK:
        _JOB_RESULTS.clear()
        _JOB_RUNNING.clear()


def _state_for_stop(stop_reason: str | None) -> OrchestrationState:
    if stop_reason == "cancelled":
        return OrchestrationState.CANCELLED
    if stop_reason == "budget_exhausted":
        return OrchestrationState.BUDGET_EXHAUSTED
    return OrchestrationState.DONE


def _event_log_path(orchestration_id: str) -> Path:
    return _storage_root() / "orchestrations" / f"{orchestration_id}.jsonl"


def _lean_dispatcher(spec: OrchestrationSpec) -> Any:
    return get_subagent_dispatcher(
        conv_id=spec.conv_id,
        project_root=spec.project_root,
        retriever=None,
        read_only=True,
        allowed_tools=LEAN_READ_ONLY_TOOLS,
    )


def _append_event(path: Path, record: dict[str, Any]) -> None:
    lock = _log_lock(str(path))
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8", newline="") as fh:
            fh.write(line)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass


def _log_lock(key: str) -> threading.Lock:
    with _LOG_LOCKS_LOCK:
        lock = _LOG_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOG_LOCKS[key] = lock
        return lock


def _findings_block(outcomes: list[FinderOutcome]) -> str:
    parts: list[str] = []
    for outcome in outcomes:
        status = outcome.halt_reason
        if outcome.error:
            status = f"error: {outcome.error}"
        parts.append(
            f"## {outcome.dimension}\n"
            f"status: {status}\n"
            f"{outcome.summary or '(no summary)'}"
        )
    return "\n\n".join(parts)


def _verify_task(
    task: str,
    dimensions: list[str],
    outcomes: list[FinderOutcome],
) -> str:
    return (
        "Verify and synthesize the finder results for this task:\n"
        f"{task}\n\n"
        f"Dimensions: {', '.join(dimensions)}\n\n"
        f"{_findings_block(outcomes)}"
    )
