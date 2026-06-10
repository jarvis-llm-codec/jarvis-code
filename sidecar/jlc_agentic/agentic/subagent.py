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
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from jlc_agentic.providers import build_adapter_for_spec, get_llm, turn_context

SUBAGENT_PRESETS: dict[str, dict[str, Any]] = {
    "deep_research": {
        "system_prompt": (
            "You are a deep-research subagent. Investigate the task using read-only "
            "tools (read, grep, web_search, recall_turns, jre_search). Return a concise "
            "summary of findings as your final message. Do not write files."
        ),
        "read_only": True,
    },
    "codebase_explore": {
        "system_prompt": (
            "You are a codebase exploration subagent. Map relevant modules, call sites, "
            "and dependencies using read and grep. Return a structured summary "
            "(files, symbols, relationships). Do not modify files."
        ),
        "read_only": True,
    },
    "multi_file_refactor": {
        "system_prompt": (
            "You are a refactor subagent. Plan and apply edits across multiple files "
            "using read, grep, edit, and bash. Return a summary of changes made and any "
            "files left untouched."
        ),
        "read_only": False,
    },
    "destroyer": {
        "system_prompt": (
            "You are an adversarial code-destroyer subagent. Read the target file(s) "
            "directly and tear the code apart: defects, edge cases, broken assumptions, "
            "race conditions, security holes. Cite exact file:line. No baseless nitpicks "
            "- every claim grounded in code you read. Return the critique as your final "
            "message. You cannot write files."
        ),
        "read_only": True,
    },
}

SUBAGENT_SYSTEM_PROMPTS = {
    name: str(preset["system_prompt"])
    for name, preset in SUBAGENT_PRESETS.items()
}

# Session-local resume store. Promote this to raw_store if cross-process
# persistence becomes required.
MAX_SUBAGENT_HISTORIES = 64
_SUBAGENT_HISTORIES: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_SUBAGENT_HISTORY_LOCK = Lock()


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
    messages: list[dict[str, Any]] | None = None


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
        read_only: bool | None = None,
        conv_id: str | None = None,
        project_root: str | None = None,
        retriever: Any | None = None,
        reasoning_effort: str | None = None,
        allowed_tools: set[str] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        preset = SUBAGENT_PRESETS.get(name)
        if preset is None and not system_prompt:
            raise ValueError(
                f"unknown subagent: {name}; pass system_prompt for an ad-hoc "
                f"subagent or use a preset: {sorted(SUBAGENT_PRESETS)}"
            )
        self.name = name
        self.max_iter = max_iter
        # Subagent model = the picker-selected role (roles.subagent, which falls
        # back to chat) or an explicit `model=` from an internal caller. There is
        # no per-preset model differentiation — a subagent is not a separate kind
        # of model, it reuses the chat-tier model. See _select_llm_client.
        self.model = model
        if read_only is not None:
            self.read_only = bool(read_only)
        elif preset is not None:
            self.read_only = bool(preset.get("read_only", True))
        else:
            self.read_only = True
        self.conv_id = conv_id
        self.project_root = project_root
        self.retriever = retriever
        self.allowed_tools = set(allowed_tools) if allowed_tools is not None else None
        self.should_cancel = should_cancel
        self.reasoning_effort = (
            str(reasoning_effort).strip() if reasoning_effort else None
        ) or "medium"
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

            dispatcher = get_subagent_dispatcher(
                read_only=self.read_only,
                allowed_tools=self.allowed_tools,
            )
        elif self.read_only or self.allowed_tools is not None:
            from .dispatcher import READ_ONLY_TOOLS, ToolDispatcher

            tools = getattr(dispatcher, "tools", None)
            if isinstance(tools, dict):
                allowed = set(tools)
                if self.read_only:
                    allowed &= READ_ONLY_TOOLS
                if self.allowed_tools is not None:
                    allowed &= self.allowed_tools
                dispatcher = ToolDispatcher(
                    {
                        name: handler
                        for name, handler in tools.items()
                        if name in allowed
                    },
                    active_project_path=getattr(dispatcher, "active_project_path", None),
                    on_external_write=getattr(dispatcher, "on_external_write", None),
                )
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
        self.llm_client = self._select_llm_client(llm_client, self.model)
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
        base_prompt = system_prompt or str(preset.get("system_prompt") if preset else "")
        directives = get_env_directive() + POLICY_USER_FACING
        role_block = (
            (base_prompt + "\n\n" + directives) if base_prompt else directives
        )
        self.system_prompt = constitution_block + role_block
        # Decision #29 (2026-05-02): subagent's reasoning/content stream
        # mirrors to the main terminal raw, in real time. None = silent.
        self.on_token = on_token

    @staticmethod
    def _select_llm_client(llm_client: Any | None, model_spec: str | None) -> Any:
        if model_spec:
            return build_adapter_for_spec(model_spec)
        if llm_client is not None:
            return llm_client
        return get_llm("subagent")

    def run(
        self,
        task: str,
        prior_history: list[dict[str, Any]] | None = None,
        sub_id: str | None = None,
        **kwargs: Any,
    ) -> SubagentResult:
        sub_id = sub_id or uuid.uuid4().hex[:8]
        started = time.monotonic()
        self.on_raw(f"=== sub start id={sub_id} task={task[:80]!r} ===")

        return self._run_llm(
            task=task,
            started=started,
            sub_id=sub_id,
            prior_history=prior_history,
        )

    def _run_llm(
        self,
        task: str,
        started: float,
        sub_id: str,
        prior_history: list[dict[str, Any]] | None = None,
    ) -> SubagentResult:
        from .loop import AgenticLoop
        from .schema import READ_ONLY_SUBAGENT_TOOLS, SUBAGENT_TOOLS

        def step_to_raw(info: dict[str, Any]) -> None:
            self.on_raw(
                f"step iter={info.get('iter')} tool_calls={info.get('tool_calls', 0)}"
                + (f" halt={info.get('halt')}" if info.get("halt") else "")
            )

        messages: list[dict[str, Any]] = list(prior_history or [])
        if not messages and self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": task})
        tool_schemas = READ_ONLY_SUBAGENT_TOOLS if self.read_only else SUBAGENT_TOOLS
        if self.allowed_tools is not None:
            tool_schemas = [
                schema
                for schema in tool_schemas
                if (schema.get("function") or {}).get("name") in self.allowed_tools
            ]

        loop = AgenticLoop(
            llm_client=self.llm_client,
            dispatcher=self.dispatcher,
            max_iter=self.max_iter,
            on_step=step_to_raw,
            on_token=self.on_token,
            timeout_sec=self.timeout_sec,
            silence_threshold_sec=self.silence_threshold_sec,
            on_silence=self.on_silence,
            tools=tool_schemas,
            should_cancel=self.should_cancel,
        )
        previous_turn_context = dict(turn_context.get())
        try:
            turn_context.set(
                conv_id=self.conv_id,
                project_root=self.project_root,
                retriever=self.retriever,
                reasoning_effort=self.reasoning_effort,
                stream_text_deltas=True,
                suppress_widget_tool_calls=True,
            )
            result = loop.run(messages)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            summary = f"[error] {self.name} failed: {exc}"
            self.on_raw(f"=== sub error id={sub_id} err={exc!r} ===")
            return SubagentResult(
                name=self.name,
                summary=summary,
                iters=0,
                halt_reason="error",
                raw_log_path=None,
                elapsed_sec=elapsed,
                sub_id=sub_id,
                in_tokens=int(getattr(loop, "cumulative_in_tokens", 0) or 0),
                out_tokens=int(getattr(loop, "cumulative_out_tokens", 0) or 0),
                think_tokens=int(getattr(loop, "cumulative_think_tokens", 0) or 0),
                messages=self._history_with_final(messages, summary),
            )
        finally:
            if previous_turn_context:
                turn_context.set(
                    conv_id=previous_turn_context.get("conv_id"),
                    project_root=previous_turn_context.get("project_root"),
                    retriever=previous_turn_context.get("retriever"),
                    reasoning_effort=previous_turn_context.get("reasoning_effort"),
                    second_eyes_phase=previous_turn_context.get("second_eyes_phase"),
                    stream_text_deltas=previous_turn_context.get("stream_text_deltas"),
                    suppress_widget_tool_calls=previous_turn_context.get(
                        "suppress_widget_tool_calls"
                    ),
                )
            else:
                turn_context.clear()

        elapsed = time.monotonic() - started
        final_message = result.get("final_message") or ""
        self.on_raw(
            f"=== sub end id={sub_id} halt={result.get('halt_reason')} "
            f"iters={result.get('iters')} ==="
        )
        return SubagentResult(
            name=self.name,
            summary=final_message,
            iters=int(result.get("iters", 0)),
            halt_reason=str(result.get("halt_reason", "unknown")),
            raw_log_path=None,
            elapsed_sec=elapsed,
            sub_id=sub_id,
            in_tokens=int(getattr(loop, "cumulative_in_tokens", 0) or 0),
            out_tokens=int(getattr(loop, "cumulative_out_tokens", 0) or 0),
            think_tokens=int(getattr(loop, "cumulative_think_tokens", 0) or 0),
            messages=self._history_with_final(messages, final_message),
        )

    @staticmethod
    def _history_with_final(
        messages: list[dict[str, Any]],
        final_message: str,
    ) -> list[dict[str, Any]]:
        history = [dict(message) for message in messages]
        history.append({"role": "assistant", "content": final_message})
        return history


def _get_subagent_history(sub_id: str | None) -> list[dict[str, Any]] | None:
    if not sub_id:
        return None
    with _SUBAGENT_HISTORY_LOCK:
        stored = _SUBAGENT_HISTORIES.get(sub_id)
        if stored is not None:
            _SUBAGENT_HISTORIES.move_to_end(sub_id)
        return [dict(message) for message in stored] if stored is not None else None


def _put_subagent_history(sub_id: str, messages: list[dict[str, Any]]) -> None:
    with _SUBAGENT_HISTORY_LOCK:
        _SUBAGENT_HISTORIES[sub_id] = [dict(message) for message in messages]
        _SUBAGENT_HISTORIES.move_to_end(sub_id)
        while len(_SUBAGENT_HISTORIES) > MAX_SUBAGENT_HISTORIES:
            _SUBAGENT_HISTORIES.popitem(last=False)


def _resume_system_prompt(
    prior_history: list[dict[str, Any]] | None,
) -> str | None:
    if not prior_history:
        return None
    for message in prior_history:
        if message.get("role") == "system":
            content = message.get("content")
            return str(content) if content else None
    return None


SCHEMA = {
    "type": "function",
    "function": {
        "name": "delegate_subagent",
        "description": (
            "Delegate a heavy task to a clean-context subagent. "
            "Returns only a summary; the subagent's raw stream is mirrored to stderr. "
            "Use presets such as deep_research, codebase_explore, "
            "multi_file_refactor, or destroyer; ad-hoc names require system_prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Preset name or ad-hoc subagent name.",
                },
                "task": {"type": "string"},
                "read_only": {
                    "type": "boolean",
                    "description": "Expose only read-only tools for this subagent run.",
                },
                "system_prompt": {
                    "type": "string",
                    "description": "Required for ad-hoc names that are not built-in presets.",
                },
                "sub_id": {
                    "type": "string",
                    "description": "Existing subagent id to resume by caller request.",
                },
            },
            "required": ["name", "task"],
            "additionalProperties": False,
        },
    },
}


def make_handler(
    llm_client: Any | None = None,
    on_token: Callable[..., None] | None = None,
    on_raw: Callable[[str], None] | None = None,
    conv_id: str | None = None,
    storage_root: str | None = None,
    project_root: str | None = None,
    retriever: Any | None = None,
    reasoning_effort: str | None = "medium",
) -> Callable[..., dict]:
    """Build a delegate_subagent handler bound to a specific LLM client.

    on_token (optional): callback invoked for every reasoning/content
    chunk produced by the subagent's internal loop. Lets the caller mirror
    the subagent's raw stream to the user's terminal in real time
    (decision #29). Without it, the subagent runs silently until summary.

    on_raw (optional): callback invoked for subagent lifecycle/step lines.
    Without it, those lines are written to stderr.

    conv_id / storage_root (optional): forwarded to the subagent's
    dispatcher so recall_turns inside the subagent reads from the same
    JHB store the host writes to.

    retriever (optional): forwarded so recall_turns inside the subagent
    reuses the host's warm singleton instead of cold-loading bge-m3.

    reasoning_effort (optional): parked on turn_context while the subagent
    loop runs so Agent SDK providers do not fall back to their high default.
    """
    bound_project_root = project_root

    def _handler(
        name: str,
        task: str,
        model: str | None = None,
        project_root: str | None = None,
        read_only: bool | None = None,
        system_prompt: str | None = None,
        sub_id: str | None = None,
    ) -> dict:
        from .schema import get_subagent_dispatcher

        prior_history = _get_subagent_history(sub_id) if sub_id else None
        if sub_id and prior_history is None:
            raise ValueError(f"unknown subagent sub_id: {sub_id}")
        effective_project_root = (
            project_root if project_root is not None else bound_project_root
        )
        sub = Subagent(
            name=name,
            model=model,
            llm_client=llm_client,
            on_raw=on_raw,
            on_token=on_token,
            system_prompt=system_prompt or _resume_system_prompt(prior_history),
            read_only=read_only,
            conv_id=conv_id,
            project_root=effective_project_root,
            retriever=retriever,
            reasoning_effort=reasoning_effort,
            dispatcher=get_subagent_dispatcher(
                conv_id=conv_id,
                storage_root=storage_root,
                project_root=effective_project_root,
                retriever=retriever,
                read_only=bool(read_only),
            ),
        )
        result = sub.run(task=task, prior_history=prior_history, sub_id=sub_id)
        if result.messages is not None:
            _put_subagent_history(result.sub_id, result.messages)
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
