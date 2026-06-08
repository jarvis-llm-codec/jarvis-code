"""Subagent - clean-context delegation for heavy tasks.

Contract: subagent runs in clean context (no parent jhb head),
returns only a summary, mirrors raw stream to caller-supplied sink.
With an llm_client passed in, runs a real agentic loop using a
sub-safe dispatcher (no delegate_subagent -> depth=1 hard cap).
Without llm_client, falls back to a stub for fast unit tests.
"""
from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from jlc_agentic.providers import get_llm

ALLOWED_NAMES = {"deep_research", "codebase_explore", "multi_file_refactor"}

SUBAGENT_SYSTEM_PROMPTS = {
    "deep_research": (
        "You are a deep-research subagent. Investigate the task using read-only "
        "tools (read, grep, web_search, recall_turn, jre_search). Return a concise "
        "summary of findings as your final message. Do not write files."
    ),
    "codebase_explore": (
        "You are a codebase exploration subagent. Map relevant modules, call sites, "
        "and dependencies using read and grep. Return a structured summary "
        "(files, symbols, relationships). Do not modify files."
    ),
    "multi_file_refactor": (
        "You are a refactor subagent. Plan and apply edits across multiple files "
        "using read, grep, edit, and bash. Return a summary of changes made and any "
        "files left untouched."
    ),
}


@dataclass
class SubagentResult:
    name: str
    summary: str
    iters: int = 0
    halt_reason: str = "stub"
    raw_log_path: str | None = None
    elapsed_sec: float = 0.0
    sub_id: str = ""
    in_tokens: int = 0
    out_tokens: int = 0
    think_tokens: int = 0


class Subagent:
    """Run a heavy task in a clean context and return only a summary."""

    def __init__(
        self,
        name: str,
        max_iter: int = 10,
        model: str | None = None,
        on_raw: Callable[[str], None] | None = None,
        on_token: Callable[..., None] | None = None,
        timeout_sec: float = 300.0,
        silence_threshold_sec: float = 30.0,
        on_silence: Callable[[float], None] | None = None,
        dispatcher: Any | None = None,
        llm_client: Any | None = None,
        system_prompt: str | None = None,
    ) -> None:
        if name not in ALLOWED_NAMES:
            raise ValueError(
                f"unknown subagent: {name}; allowed: {sorted(ALLOWED_NAMES)}"
            )
        self.name = name
        self.max_iter = max_iter
        self.model = model
        self.timeout_sec = timeout_sec
        self.silence_threshold_sec = silence_threshold_sec
        self.on_silence = on_silence
        if on_raw is None:

            def default(line: str, _name=name) -> None:
                sys.stderr.write(f"[sub:{_name}] {line}\n")
                sys.stderr.flush()

            self.on_raw = default
        else:
            self.on_raw = on_raw
        if dispatcher is None:
            from .schema import get_subagent_dispatcher

            dispatcher = get_subagent_dispatcher()
        tools = getattr(dispatcher, "tools", None)
        if not isinstance(tools, dict):
            raise ValueError(
                f"subagent dispatcher.tools must be a dict, got {type(tools).__name__}"
            )
        if "delegate_subagent" in tools:
            raise ValueError(
                "subagent dispatcher must not expose delegate_subagent "
                "(depth=1 hard cap)"
            )
        self.dispatcher = dispatcher
        self.llm_client = llm_client if llm_client is not None else get_llm("subagent")
        # Reasoning depth policy injected after the role-specific prompt so
        # the subagent decides how deep to think per task. Single source in
        # jlc_agentic.prompts.reasoning_policy.
        from jlc_agentic.prompts import (
            POLICY_USER_FACING,
            get_constitution,
            get_env_directive,
        )

        # W2.9.16: constitution leads. Subagents follow Principle 1 (retrieval
        # before fact answers) and Principle 4 (same as chat, smaller context).
        constitution_block = (
            "[jarvis-code Constitution — applies to chat, subagent, encoder]\n"
            + get_constitution()
            + "\n\n"
        )
        base_prompt = system_prompt or SUBAGENT_SYSTEM_PROMPTS.get(name, "")
        directives = get_env_directive() + POLICY_USER_FACING
        role_block = (
            (base_prompt + "\n\n" + directives) if base_prompt else directives
        )
        self.system_prompt = constitution_block + role_block
        # Decision #29 (2026-05-02): subagent's reasoning/content stream
        # mirrors to the main terminal raw, in real time. None = silent.
        self.on_token = on_token

    def run(self, task: str, **kwargs: Any) -> SubagentResult:
        sub_id = uuid.uuid4().hex[:8]
        started = time.monotonic()
        self.on_raw(f"=== sub start id={sub_id} task={task[:80]!r} ===")

        return self._run_llm(task=task, started=started, sub_id=sub_id)

    def _run_llm(self, task: str, started: float, sub_id: str) -> SubagentResult:
        from .loop import AgenticLoop
        from .schema import SUBAGENT_TOOLS

        def step_to_raw(info: dict[str, Any]) -> None:
            self.on_raw(
                f"step iter={info.get('iter')} tool_calls={info.get('tool_calls', 0)}"
                + (f" halt={info.get('halt')}" if info.get("halt") else "")
            )

        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task})

        loop = AgenticLoop(
            llm_client=self.llm_client,
            dispatcher=self.dispatcher,
            max_iter=self.max_iter,
            on_step=step_to_raw,
            on_token=self.on_token,
            timeout_sec=self.timeout_sec,
            silence_threshold_sec=self.silence_threshold_sec,
            on_silence=self.on_silence,
            tools=SUBAGENT_TOOLS,
        )
        try:
            result = loop.run(messages)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            self.on_raw(f"=== sub error id={sub_id} err={exc!r} ===")
            return SubagentResult(
                name=self.name,
                summary=f"[error] {self.name} failed: {exc}",
                iters=0,
                halt_reason="error",
                raw_log_path=None,
                elapsed_sec=elapsed,
                sub_id=sub_id,
                in_tokens=0,
                out_tokens=0,
                think_tokens=0,
            )

        elapsed = time.monotonic() - started
        self.on_raw(
            f"=== sub end id={sub_id} halt={result.get('halt_reason')} "
            f"iters={result.get('iters')} ==="
        )
        return SubagentResult(
            name=self.name,
            summary=result.get("final_message") or "",
            iters=int(result.get("iters", 0)),
            halt_reason=str(result.get("halt_reason", "unknown")),
            raw_log_path=None,
            elapsed_sec=elapsed,
            sub_id=sub_id,
            in_tokens=int(getattr(loop, "cumulative_in_tokens", 0) or 0),
            out_tokens=int(getattr(loop, "cumulative_out_tokens", 0) or 0),
            think_tokens=int(getattr(loop, "cumulative_think_tokens", 0) or 0),
        )


SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_subagent",
        "description": (
            "Delegate a heavy task to a clean-context subagent. "
            "Returns only a summary; the subagent's raw stream is mirrored to stderr. "
            "Use for deep_research / codebase_explore / multi_file_refactor."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": sorted(ALLOWED_NAMES),
                },
                "task": {"type": "string"},
                "model": {"type": "string"},
            },
            "required": ["name", "task"],
            "additionalProperties": False,
        },
    },
}


def make_handler(
    llm_client: Any | None = None,
    on_token: Callable[..., None] | None = None,
    conv_id: str | None = None,
    storage_root: str | None = None,
    project_root: str | None = None,
    retriever: Any | None = None,
) -> Callable[..., dict]:
    """Build a delegate_subagent handler bound to a specific LLM client.

    on_token (optional): callback invoked for every reasoning/content
    chunk produced by the subagent's internal loop. Lets the caller mirror
    the subagent's raw stream to the user's terminal in real time
    (decision #29). Without it, the subagent runs silently until summary.

    conv_id / storage_root (optional): forwarded to the subagent's
    dispatcher so recall_turn inside the subagent reads from the same
    JHB store the host writes to.

    retriever (optional): forwarded so recall_turn inside the subagent
    reuses the host's warm singleton instead of cold-loading bge-m3.
    """

    def _handler(name: str, task: str, model: str | None = None, project_root: str | None = None) -> dict:
        from .schema import get_subagent_dispatcher

        sub = Subagent(
            name=name,
            model=model,
            llm_client=llm_client,
            on_token=on_token,
            dispatcher=get_subagent_dispatcher(
                conv_id=conv_id,
                storage_root=storage_root,
                project_root=project_root if project_root is not None else None,
                retriever=retriever,
            ),
        )
        result = sub.run(task=task)
        return {
            "subagent": result.name,
            "summary": result.summary,
            "iters": result.iters,
            "halt_reason": result.halt_reason,
            "elapsed_sec": round(result.elapsed_sec, 4),
            "in_tokens": result.in_tokens,
            "out_tokens": result.out_tokens,
            "think_tokens": result.think_tokens,
            "sub_id": result.sub_id,
        }

    return _handler


handler = make_handler(llm_client=None)
