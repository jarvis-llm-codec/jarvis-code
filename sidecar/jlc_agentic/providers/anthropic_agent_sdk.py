"""Anthropic Agent SDK adapter — Claude subscription credit backend.

2026-06-15. Lets the user's Claude subscription "Agent SDK credit" pool power
the jarvis-code ``chat`` role. The 6/15 Anthropic plan change made it legal for
a third-party app that authenticates with the user's subscription *through the
Agent SDK* to draw on a separate monthly Agent SDK credit pool (Help Center
15036540). raw Messages API + subscription OAuth does NOT draw on that pool and
is gray-area, so we MUST route through the ``claude-agent-sdk`` library.

Architecture = **seam-level turn delegation** (NOT a normal completion provider):

  JLC owns the OUTER loop (memory carry + JARVIS persona, already baked into the
  incoming ``messages``). For one chat turn we delegate the INNER agentic loop
  to the Agent SDK: ``query()`` runs Claude with its native tools (Read/Edit/
  Bash/Glob/Grep/WebSearch/...) plus a bridged ``recall_turns`` memory tool, and
  we translate the resulting message stream back into the OpenAI-style chunks the
  ``AgenticLoop`` expects. Crucially we return **no tool_calls** — the SDK already
  executed them — so JLC's loop terminates after a single iteration. Thesis holds:
  JLC = soul/memory, Agent SDK = muscle.

Why this lives behind the ProviderAdapter seam (vs a new turn-level hook): the
adapter is a drop-in for ``LLMRouterAdapter`` so memory carry, persona system
messages, UI token streaming, turn logging and JHB encoding all keep working
unchanged — the turn's compressed context arrives in ``messages`` and the final
answer leaves as content chunks.

v1 scope / known limits (see plan robust-snuggling-toast.md):
  - chat/subagent via the agentic path (stream_chat_completions). The JHB encoder
    may also run here, but ONLY on haiku-class models (chat() / _is_encoder_safe_model)
    — it fires every turn, so a heavier model would drain the subscription rate
    limit. This lets a subscription-only user run the encoder with no API key. (2026-06-16)
  - block-level streaming (the Python SDK exposes message-level events, not token
    deltas): thinking/tool activity streams as reasoning; assistant text blocks are
    buffered and the terminal ResultMessage becomes content. This keeps the visible
    tool timeline above the final answer instead of letting prose jump ahead of it.
  - auto-approve (permission_mode=bypassPermissions) with a JARVIS phase-aware
    PreToolUse hook backstop for Critic Mode / plan-draft restrictions.
  - requires the Claude Code CLI on PATH + ``claude setup-token`` ->
    ``CLAUDE_CODE_OAUTH_TOKEN``. Fail loud if either is missing.
  - jre_search is a stub upstream ("JRE not yet wired") so it is NOT bridged.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import queue
import re
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from . import turn_context

# Sentinel pushed onto the bridge queue when the async producer is done.
_DONE = object()

# _drive_async hang guard (D4, 2026-06-22): the consumer must never block on an
# unbounded q.get(). If the SDK query() stalls without yielding or raising, the
# worker thread used to block forever (documented 80-92s+ hang). We poll the
# queue on a short interval and abort with a TimeoutError once the overall
# deadline passes without ANY new item. Defaults are generous so normal long
# turns (thinking / tool runs that still emit periodic SDK messages) are
# unaffected; env overrides allow tuning without a code change.
try:
    _DRIVE_ASYNC_OVERALL_TIMEOUT_SEC = float(
        os.environ.get("JARVIS_AGENT_SDK_STREAM_TIMEOUT_SEC", "120")
    )
except ValueError:
    _DRIVE_ASYNC_OVERALL_TIMEOUT_SEC = 120.0
try:
    _DRIVE_ASYNC_POLL_INTERVAL_SEC = float(
        os.environ.get("JARVIS_AGENT_SDK_STREAM_POLL_SEC", "1.0")
    )
except ValueError:
    _DRIVE_ASYNC_POLL_INTERVAL_SEC = 1.0

_INSTALL_HINT = (
    "claude-agent-sdk is not installed in the sidecar venv. Install it with "
    "`pip install claude-agent-sdk` (and ensure the Claude Code CLI is on PATH). "
    "Refusing silent fallback."
)
_TOKEN_HINT = (
    "No Claude subscription auth found. Either log in interactively (`claude`, "
    "which writes ~/.claude/.credentials.json) or run `claude setup-token` and "
    "export CLAUDE_CODE_OAUTH_TOKEN (headless). If ~/.claude/.credentials.json "
    "exists but its access token is expired and has no refresh token, Claude Code "
    "will still fail with authentication_failed. Refusing silent fallback."
)

# Valid ClaudeAgentOptions.effort values (claude-agent-sdk Literal). The route
# judge maps an intent to one of these via turn_context.route_to_effort; we
# only forward a value the SDK recognizes and otherwise leave effort=None (the
# SDK default), so an unknown/absent route never overrides behavior. Claude
# accepts the full JLC vocabulary up to "xhigh"/"max" — no clamp needed here
# (unlike the OpenAI Responses path, which tops out at "high"). (2026-06-16)
_SDK_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


def _coerce_sdk_effort(value: Any) -> str | None:
    """Return a valid SDK effort string, or None to leave the SDK default."""
    if not value:
        return None
    eff = str(value).strip().lower()
    return eff if eff in _SDK_EFFORTS else None


# Native Agent SDK tools we expose for a daily coding/chat driver, plus the
# bridged JLC memory tool. Listing them explicitly keeps the surface deterministic.
_NATIVE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "TodoWrite",
    "Task",
]
_MEMORY_SERVER_NAME = "jlc_memory"
_RECALL_TOOL = f"mcp__{_MEMORY_SERVER_NAME}__recall_turns"
_CONTROL_SERVER_NAME = "jarvis_control"
# Sentinel pair8 used when no real pair_id is resolvable (single-process sidecar
# with Agent-SDK regime). Must match the same constant in jarvis_sidecar/app.py so
# enqueue (here) and dequeue (app.py control/pending + control/{id}/answer) agree.
# Must be 8 alphanumerics: control_bridge._coerce_pair8 validates ^[A-Za-z0-9]{8}$
# and rejects underscores, so the older "__self__" sentinel was silently dropped
# on both enqueue and dequeue. "s"/"l" are non-hex, so this cannot collide with a
# real (hex-derived) pair_id truncated to 8 chars.
_CONTROL_FALLBACK_PAIR8 = "selfself"
_ASK_USER_TOOL_NAME = "ask_user"
_SPAWN_WINDOW_TOOL_NAME = "spawn_window"
_LIST_WINDOWS_TOOL_NAME = "list_windows"
_SEND_DIRECTIVE_TOOL_NAME = "send_directive"
_JOB_SEND_TOOL_NAME = "job_send"
_JOB_CLOSE_TOOL_NAME = "job_close"
_SET_CHAT_MODEL_TOOL_NAME = "set_chat_model"
_SET_ENCODER_MODEL_TOOL_NAME = "set_encoder_model"
_SET_WINDOW_LABEL_TOOL_NAME = "set_window_label"
# GAN/map deepdive tools (item 2, 2026-06-25). gan_send/gan_close drive the GAN
# consensus bus and are bridged SIDECAR-DIRECT (Pattern S) via append_directive,
# exactly like send_directive/job_send. map_create/feature_verdict MUTATE pi's
# in-process activeMapRun ledger and are bridged PI-ROUTED (Pattern P) via the
# control-bridge submit_request -> pi handleControlBridgeRequest, exactly like
# spawn_window, so pi (and only pi) owns its own map ledger.
_GAN_SEND_TOOL_NAME = "gan_send"
_GAN_CLOSE_TOOL_NAME = "gan_close"
_MAP_CREATE_TOOL_NAME = "map_create"
_FEATURE_VERDICT_TOOL_NAME = "feature_verdict"
_ORCHESTRATE_START_TOOL_NAME = "ultracode"
_ORCHESTRATE_STATUS_TOOL_NAME = "ultracode_status"
_ORCHESTRATE_RESULT_TOOL_NAME = "ultracode_result"
_ORCHESTRATE_CANCEL_TOOL_NAME = "ultracode_cancel"
_BRIDGED_CONTROL_TOOL_NAMES = frozenset(
    {
        _ASK_USER_TOOL_NAME,
        _SPAWN_WINDOW_TOOL_NAME,
        _LIST_WINDOWS_TOOL_NAME,
        _SEND_DIRECTIVE_TOOL_NAME,
        _JOB_SEND_TOOL_NAME,
        _JOB_CLOSE_TOOL_NAME,
        _SET_CHAT_MODEL_TOOL_NAME,
        _SET_ENCODER_MODEL_TOOL_NAME,
        _SET_WINDOW_LABEL_TOOL_NAME,
        _GAN_SEND_TOOL_NAME,
        _GAN_CLOSE_TOOL_NAME,
        _MAP_CREATE_TOOL_NAME,
        _FEATURE_VERDICT_TOOL_NAME,
        _ORCHESTRATE_START_TOOL_NAME,
        _ORCHESTRATE_STATUS_TOOL_NAME,
        _ORCHESTRATE_RESULT_TOOL_NAME,
        _ORCHESTRATE_CANCEL_TOOL_NAME,
    }
)
_CONTROL_TOOL_ORDER = (
    _ASK_USER_TOOL_NAME,
    _SPAWN_WINDOW_TOOL_NAME,
    _LIST_WINDOWS_TOOL_NAME,
    _SEND_DIRECTIVE_TOOL_NAME,
    _JOB_SEND_TOOL_NAME,
    _JOB_CLOSE_TOOL_NAME,
    _SET_CHAT_MODEL_TOOL_NAME,
    _SET_ENCODER_MODEL_TOOL_NAME,
    _SET_WINDOW_LABEL_TOOL_NAME,
    _GAN_SEND_TOOL_NAME,
    _GAN_CLOSE_TOOL_NAME,
    _MAP_CREATE_TOOL_NAME,
    _FEATURE_VERDICT_TOOL_NAME,
    _ORCHESTRATE_START_TOOL_NAME,
    _ORCHESTRATE_STATUS_TOOL_NAME,
    _ORCHESTRATE_RESULT_TOOL_NAME,
    _ORCHESTRATE_CANCEL_TOOL_NAME,
)
# Turn-taking interlock (regime B): GAN/job HANDOFF tools hand the conversation turn
# to a counterpart window. Because the SDK runs its own agentic loop to natural
# completion (max_turns=None), the sender does NOT end its turn after a handoff -- it
# lingers (extra thinking/text) while the counterpart wakes via pi's idle poll, so
# two windows run at once (the regime-A pi idle-gate that blocks this is bypassed by
# the SDK inner loop). The _generate loop force-ends the SDK turn once a handoff tool
# result arrives successfully. These are the MCP-prefixed names as they appear in
# ToolUseBlock.name (mcp__jarvis_control__<tool>). gan_send/gan_close are strict
# 2-window ping-pong; job_send/job_close are handbacks/handoffs. spawn_window is
# intentionally EXCLUDED: parallel worker fan-out spawns windows (each carries its own
# initial job) and must be able to issue several spawns in one turn.
_TURN_ENDING_HANDOFF_TOOL_NAMES = frozenset(
    f"mcp__{_CONTROL_SERVER_NAME}__{name}"
    for name in (
        _GAN_SEND_TOOL_NAME,
        _GAN_CLOSE_TOOL_NAME,
        _JOB_SEND_TOOL_NAME,
        _JOB_CLOSE_TOOL_NAME,
        _ORCHESTRATE_START_TOOL_NAME,
    )
)

# The Agent SDK executes Claude-native tools internally and never returns JLC
# tool_calls. That is fine for native-equivalent coding tools (Read/Bash/Edit),
# but impossible for Pi/JARVIS control tools unless they are bridged through the
# outer harness. Keep this bridge narrow: Claude authors payloads, while the
# JARVIS window that owns the UI/control surface performs the side effect.
_JLC_CONTROL_TOOL_NAMES = frozenset(
    {
        "ask_user",
        "spawn_window",
        "list_windows",
        "send_directive",
        "job_send",
        "job_close",
        "gan_send",
        "gan_close",
        "map_create",
        "feature_verdict",
        "ultracode",
        "ultracode_status",
        "ultracode_result",
        "ultracode_cancel",
        "managed_process",
        "retrieve_output",
        "switch_project",
        "register_project",
        "unregister_project",
        "update_jarvis_md",
        "set_chat_model",
        "set_encoder_model",
        "set_window_label",
        "generate_image",
        "edit_image",
    }
)
_SECOND_EYES_PLAN_DRAFT_PHASE = "plan_draft"
_SECOND_EYES_REVIEW_PHASE = "review"
_SECOND_EYES_IMPLEMENT_PHASE = "implement"
_SECOND_EYES_PHASES = frozenset(
    {_SECOND_EYES_PLAN_DRAFT_PHASE, _SECOND_EYES_REVIEW_PHASE, _SECOND_EYES_IMPLEMENT_PHASE}
)
_SECOND_EYES_PLAN_DRAFT_NATIVE_TOOLS = ["Read", "Glob", "Grep", "WebSearch", "WebFetch"]
_SECOND_EYES_REVIEW_NATIVE_TOOLS = ["Read", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"]
_SECOND_EYES_PLAN_DRAFT_BLOCKED_TOOLS = frozenset(
    {"Write", "Edit", "MultiEdit", "Bash", "Task", "TodoWrite"}
)
_SECOND_EYES_REVIEW_BLOCKED_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "Task", "TodoWrite"})
_SECOND_EYES_REVIEW_MARKER = "[CRITIC_REVIEW]"
_SECOND_EYES_MAIN_MARKER = "[CRITIC_MAIN]"
_SECOND_EYES_REQUEST_MARKER = "[CRITIC MODE REQUESTED]"
_SECOND_EYES_REMINDER_MARKER = "JARVIS_CRITIC_REVIEW_REQUIRED"
_LEGACY_SECOND_EYES_REVIEW_MARKER = "[SECOND_EYES_REVIEW]"
_LEGACY_SECOND_EYES_MAIN_MARKER = "[SECOND_EYES_MAIN]"
_SECOND_EYES_REVIEW_MARKERS = (_SECOND_EYES_REVIEW_MARKER, _LEGACY_SECOND_EYES_REVIEW_MARKER)
_SECOND_EYES_MAIN_MARKERS = (_SECOND_EYES_MAIN_MARKER, _LEGACY_SECOND_EYES_MAIN_MARKER)

# New-artifact ask_user gate (Part B backstop, 2026-06-24). pi marks a NEW
# user-facing-artifact turn with this marker in the system prompt; the PreToolUse
# hook then denies file/shell tools until the model calls ask_user, so the
# autonomous Agent-SDK loop cannot build before clarifying (mirrors codex, where pi
# owns the loop and pauses on the ask_user tool_call). Read/Glob/Grep + control/
# recall tools stay allowed. Tool names below are the _normalize_tool_schema_name
# OUTPUT form (lowercase); the capitalized Critic-Mode blocked sets above predate
# that normalizer and are a separate latent casing issue (not touched here).
_NEW_ARTIFACT_GATE_MARKER = "[JLC:NEW_ARTIFACT_ASK_USER_GATE]"
_NEW_ARTIFACT_GATED_TOOLS = frozenset({"write", "edit", "multi_edit", "bash"})

# Claude-native harness-orchestration tools that launch DETACHED background work
# (a nested Claude Code workflow / agent runner). permission_mode="bypassPermissions"
# lets the model reach these even though they are not in our curated allow_tools set.
# Inside JARVIS's embedded one-shot Agent SDK turn there is no persistent Claude Code
# harness to actually RUN that background work and no path for its results to return:
# the model calls it, gets "launched in background", then waits on results that never
# arrive -> the turn goes silent and the stream-idle guard aborts it (120s hang).
# Self-contained native tools (Read/Write/Bash/WebSearch/Task/...) run and finish
# inside the turn and are left to flow through untouched -- this set is ONLY the
# orchestration class, redirected to JARVIS's own bridged equivalents (ultracode /
# delegate_subagent) which run in-process and stream results back. (2026-07-01)
_EMBEDDED_UNSUPPORTED_NATIVE_TOOLS = frozenset({"workflow"})

# Disable the bundled Claude CLI's ToolSearch. On a first-party Anthropic host
# (OAuth via ~/.claude/.credentials.json) with a supported model, the CLI
# auto-enables tool-search and DEFERS MCP tools (mcp__jarvis_control__ask_user,
# spawn_window) behind a tool_search/select gate -- so even a forced ask_user call
# renders only a text label, never the modal, until the model first selects it.
# Turning tool-search off presents all tools eagerly (a few k tokens of schema),
# so ask_user is directly invocable and the bridge renders the modal. The SDK
# merges this onto the inherited environment (subprocess_cli.py), so the OAuth
# credentials / CLAUDE_CONFIG_DIR the CLI needs are preserved.
_AGENT_SDK_SUBPROCESS_ENV = {"ENABLE_TOOL_SEARCH": "0"}
_TOOL_NAME_CANONICAL_ALIASES = {
    "askuserquestion": "ask_user",
    "askuser": "ask_user",
    "jobsend": "job_send",
    "jobid": "job_id",
    "listwindows": "list_windows",
    "listwindow": "list_windows",
    "senddirective": "send_directive",
    "jobclose": "job_close",
    "spawnwindow": "spawn_window",
    "spawnwin": "spawn_window",
    "switchproject": "switch_project",
    "registerproject": "register_project",
    "unregisterproject": "unregister_project",
    "setchatmodel": "set_chat_model",
    "setencodermodel": "set_encoder_model",
    "setwindowlabel": "set_window_label",
    "generateimage": "generate_image",
    "editimage": "edit_image",
    "recallturns": "recall_turns",
    "mapcreate": "map_create",
    "updatejarvismd": "update_jarvis_md",
    "managedprocess": "managed_process",
    "featureverdict": "feature_verdict",
    "gansend": "gan_send",
    "ganclose": "gan_close",
    "ultracode": "ultracode",
    "ultracodestatus": "ultracode_status",
    "ultracoderesult": "ultracode_result",
    "ultracodecancel": "ultracode_cancel",
    "jobsendtool": "job_send",
    "askuserq": "ask_user",
    "askusershortquestion": "ask_user",
    "askuserv2": "ask_user",
}
_TOOL_NAME_CANONICAL_TO_HINT = {
    "ask_user": (),
    "bash": ("command",),
    "edit": ("file_path", "path"),
    "glob": ("pattern", "path"),
    "grep": ("pattern", "query", "path"),
    "ls": ("path",),
    "job_send": (),
    "job_close": (),
    "list_windows": (),
    "managed_process": ("command",),
    "map_create": (),
    "ultracode": ("task",),
    "ultracode_status": ("orchestration_id",),
    "ultracode_result": ("orchestration_id",),
    "ultracode_cancel": ("orchestration_id",),
    "mcp__jarvis_control__": (),
    "multi_edit": ("file_path", "path"),
    "notebook_edit": ("notebook_path", "path"),
    "read": ("file_path", "path"),
    "register_project": (),
    "send_directive": (),
    "set_chat_model": ("model",),
    "set_encoder_model": ("model",),
    "set_window_label": ("label",),
    "spawn_window": (),
    "switch_project": (),
    "task": ("description",),
    "todo_write": (),
    "unregister_project": (),
    "update_jarvis_md": (),
}


def _subscription_auth_available() -> bool:
    """True if the Agent SDK can authenticate with the Claude subscription.

    Two valid paths (verified live 2026-06-15): a headless ``CLAUDE_CODE_OAUTH_TOKEN``
    (from ``claude setup-token``), OR an interactive login whose credentials the
    Claude Code CLI stored at ``~/.claude/.credentials.json`` — the SDK spawns the
    CLI, which reuses that login. We do NOT count ``ANTHROPIC_API_KEY`` here: a
    console key bills pay-as-you-go, not the subscription's Agent SDK credit pool.
    """
    from jlc_agentic.claude_auth import agent_sdk_auth_available  # noqa: PLC0415

    return agent_sdk_auth_available()


def _normalize_tool_schema_name(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    value = raw.strip()
    if not value:
        return ""

    # MCP tool names commonly show as mcp__server__tool. Keep the tool suffix
    # only so routing and filtering can match the canonical tool symbol.
    if "__" in value:
        value = value.rsplit("__", 1)[-1]

    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_").lower()
    if not value:
        return ""
    if value in _TOOL_NAME_CANONICAL_ALIASES:
        return _TOOL_NAME_CANONICAL_ALIASES[value]
    collapsed = value.replace("_", "")
    if collapsed in _TOOL_NAME_CANONICAL_ALIASES:
        return _TOOL_NAME_CANONICAL_ALIASES[collapsed]
    return value


# The raw name of the bridged control tool as the model sees it, e.g.
# ``mcp__jarvis_control__ask_user``. Anything that normalizes to ``ask_user`` but
# does NOT start with this prefix is the Claude-native ``AskUserQuestion`` (or a
# variant), which cannot render a question UI in the embedded one-shot SDK turn.
_BRIDGED_CONTROL_TOOL_PREFIX = f"mcp__{_CONTROL_SERVER_NAME}__"
_EDIT_PARAM_ALIASES = {"old_text": "old_string", "new_text": "new_string"}


def _is_native_ask_user_leak(raw_name: Any, normalized_name: str) -> bool:
    """True when the model invoked Claude's NATIVE ``AskUserQuestion`` instead of
    the bridged ``mcp__jarvis_control__ask_user``.

    Both names collapse to ``ask_user`` through the alias table (see
    ``_TOOL_NAME_CANONICAL_ALIASES``), so the normalized name alone cannot tell
    them apart. We inspect the RAW name: a genuinely bridged call starts with the
    control-server MCP prefix; anything else that still normalizes to ``ask_user``
    is the native leak. ``bypassPermissions`` lets the model reach the native tool
    even though it is not in our curated ``allowed_tools`` -- and inside this
    embedded turn it has no question UI, so it returns ``is_error`` immediately and
    the user's answer never arrives. Callers deny it and redirect to the bridge.
    """
    if normalized_name != _ASK_USER_TOOL_NAME:
        return False
    if not isinstance(raw_name, str):
        return True
    return not raw_name.strip().lower().startswith(_BRIDGED_CONTROL_TOOL_PREFIX)


def _remap_edit_params(input_data: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    remapped = dict(input_data)
    changed = False
    for alias, canonical in _EDIT_PARAM_ALIASES.items():
        if alias in remapped and canonical not in remapped:
            remapped[canonical] = remapped.pop(alias)
            changed = True
    return remapped, changed


def _remap_edit_param_slips(tool_name: str, tool_input: Any) -> dict[str, Any] | None:
    """Map pi edit arg slips (old_text/new_text) to Claude native names.

    Returns None when no correction is needed. Official keys are never overwritten:
    if a model sends both an alias and the native key, the native key wins and that
    pair is left untouched.
    """
    try:
        if not isinstance(tool_input, dict):
            return None
        normalized_name = _normalize_tool_schema_name(tool_name)
        if normalized_name == "edit":
            remapped, changed = _remap_edit_params(tool_input)
            return remapped if changed else None
        if normalized_name != "multi_edit":
            return None
        edits = tool_input.get("edits")
        if not isinstance(edits, list):
            return None
        remapped_edits: list[Any] = []
        changed = False
        for edit in edits:
            if isinstance(edit, dict):
                remapped_edit, edit_changed = _remap_edit_params(edit)
                remapped_edits.append(remapped_edit if edit_changed else edit)
                changed = changed or edit_changed
            else:
                remapped_edits.append(edit)
        if not changed:
            return None
        remapped_input = dict(tool_input)
        remapped_input["edits"] = remapped_edits
        return remapped_input
    except Exception:  # pragma: no cover - hook correction must never block tools
        return None


def _is_encoder_safe_model(model: str) -> bool:
    """Only haiku-class models may run the every-turn JHB encoder here.

    The encoder fires on every turn, so a heavier model (opus/sonnet) routed to
    it would drain the Claude subscription rate limit fast. Haiku is cheap enough
    that a subscription-only user can run the memory encoder with no second
    provider and no API key — the whole point of allowing it. (2026-06-16)
    """
    return "haiku" in (model or "").lower()


class AnthropicAgentSDKAdapter:
    """ProviderAdapter that delegates a whole chat turn to the Claude Agent SDK.

    Satisfies the same ``stream_chat_completions`` contract as ``LLMRouterAdapter``
    (yields OpenAI-style chunks, exposes ``llm_meta``) so the AgenticLoop / ChatTurn
    path treats it like any other provider.
    """

    def __init__(self, model: str, config: dict[str, Any] | None = None) -> None:
        self.model = model
        self.config = dict(config or {})
        self.llm_meta: dict[str, Any] | None = None

    async def close(self) -> None:  # parity with LLMRouterAdapter
        return None

    async def chat(
        self,
        system: str,
        user: str,
        max_tokens: int = 2048,
        on_chunk: Any = None,
        **kwargs: Any,
    ) -> str:
        """Non-agentic completion for the JHB encoder (text -> text).

        This is what lets a Claude-subscription-only user run jarvis-code's memory
        encoder with no second provider and no API key. The encoder fires every
        turn, so it is gated to haiku-class models (a heavier model would drain the
        subscription rate limit). Unlike ``stream_chat_completions`` (the agentic
        chat path), this runs a single tool-less ``query()`` — no native tools, no
        memory MCP, no agentic loop — and returns the concatenated assistant text.
        (2026-06-16)"""
        if not _is_encoder_safe_model(self.model):
            raise RuntimeError(
                f"anthropic-agent-sdk may run the JHB encoder only on haiku-class "
                f"models (it fires every turn; a heavier model would drain your "
                f"Claude rate limit). Got {self.model!r}. Set roles.encoder to "
                f"anthropic-agent-sdk/claude-haiku-4-5, another provider, or a local model."
            )

        try:
            from claude_agent_sdk import (  # noqa: PLC0415
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                ThinkingBlock,
                query,
            )
        except Exception as exc:  # pragma: no cover - exercised via monkeypatch path
            raise RuntimeError(_INSTALL_HINT) from exc

        if not _subscription_auth_available():
            raise RuntimeError(_TOKEN_HINT)

        options = ClaudeAgentOptions(
            system_prompt=system or None,
            # Pure summarizer: no native tools, no memory bridge, no settings
            # inheritance, single turn. Keeps it fast and cheap on the rate limit.
            setting_sources=[],
            permission_mode="bypassPermissions",
            cwd=os.getcwd(),
            env=_AGENT_SDK_SUBPROCESS_ENV,
            model=self.model,
            allowed_tools=[],
            max_turns=1,
            # The JHB encoder is a mechanical summarizer — run it non-reasoning and
            # fastest. ClaudeAgentOptions.effort defaults to "high" (deep reasoning),
            # which makes the every-turn encoder slow and wasteful. Mirror the codex
            # encoder's reasoning_effort="none". (2026-06-16)
            effort="low",
            thinking={"type": "disabled"},
        )
        cli_path = self.config.get("cli_path")
        if cli_path:
            options.cli_path = cli_path

        t0 = time.monotonic()
        parts: list[str] = []
        # chat() is awaited inside the encoder's event loop, so we can drive the
        # async generator directly (the sync stream path needs _drive_async; this
        # one does not).
        async for msg in query(prompt=user, options=options):
            if isinstance(msg, AssistantMessage):
                err = getattr(msg, "error", None)
                if err:
                    raise RuntimeError(f"Agent SDK encoder error: {err}")
                for block in getattr(msg, "content", None) or []:
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "") or ""
                        if text:
                            parts.append(text)
                            if on_chunk:
                                on_chunk(text, "content")
                    elif isinstance(block, ThinkingBlock):
                        thinking = getattr(block, "thinking", "") or ""
                        if thinking and on_chunk:
                            on_chunk(thinking, "reasoning")
            elif isinstance(msg, ResultMessage):
                if getattr(msg, "is_error", False):
                    detail = getattr(msg, "errors", None) or getattr(msg, "subtype", "error")
                    status = getattr(msg, "api_error_status", None)
                    raise RuntimeError(
                        f"Agent SDK encoder result error (status={status}): {detail}"
                    )
                self.llm_meta = _result_llm_meta(
                    msg, self.model, int((time.monotonic() - t0) * 1000)
                )
        return "".join(parts)

    # ------------------------------------------------------------------ seam
    def stream_chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        parallel_tool_calls: bool = True,
        stream: bool = True,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        ctx = turn_context.get()
        second_eyes_phase = _coerce_second_eyes_phase(
            ctx.get("critic_phase")
            or ctx.get("second_eyes_phase")
            or kwargs.get("jarvis_critic_phase")
            or kwargs.get("critic_phase")
            or kwargs.get("jarvis_second_eyes_phase")
            or kwargs.get("second_eyes_phase")
            or _infer_second_eyes_phase_from_messages(messages)
        )
        requested_tool_names = _requested_tool_schema_names(tools)
        bridged_control_tools = _bridgeable_control_tool_names(requested_tool_names)
        unsupported_control_only = _jlc_control_only_tool_names(requested_tool_names)
        if unsupported_control_only and second_eyes_phase != _SECOND_EYES_PLAN_DRAFT_PHASE:
            names = ", ".join(unsupported_control_only)
            raise RuntimeError(
                "anthropic-agent-sdk cannot execute JARVIS control tools from "
                f"the Pi tool surface ({names}). This backend delegates to "
                "Claude native tools and returns no JLC tool_calls. Run this "
                "control phase through a Pi-native tool-calling model, or expose "
                "that control through the JARVIS MCP bridge before calling the model."
            )

        # Import lazily + fail loud: keeps the package optional and gives a clear
        # message instead of an ImportError deep in the loop.
        try:
            from claude_agent_sdk import (  # noqa: PLC0415
                AssistantMessage,
                ClaudeAgentOptions,
                HookEventMessage,
                HookMatcher,
                ResultMessage,
                StreamEvent,
                SystemMessage,
                TextBlock,
                ThinkingBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
                query,
            )
        except Exception as exc:  # pragma: no cover - exercised via monkeypatch path
            raise RuntimeError(_INSTALL_HINT) from exc

        if not _subscription_auth_available():
            raise RuntimeError(_TOKEN_HINT)

        system_prompt, prompt = _split_messages(messages)

        # Read per-turn context on THIS (worker) thread, then capture into the MCP
        # closures below — threading.local does not propagate to the SDK's loop
        # thread / to_thread workers, so we must carry it by closure, not by tls.
        cwd = ctx.get("project_root") or os.getcwd()
        conv_id = ctx.get("conv_id") or "conversation"
        retriever = ctx.get("retriever")
        # Route-derived reasoning effort for this turn (chat=low ... heavy=xhigh).
        # None leaves the SDK default ("high") so an untagged turn is unchanged.
        effort = _coerce_sdk_effort(
            ctx.get("reasoning_effort")
            or kwargs.get("reasoning_effort")
            or kwargs.get("reasoning")
        )
        stream_answer_text = bool(
            ctx.get("stream_text_deltas") or kwargs.get("stream_text_deltas")
        )
        suppress_widget_tool_calls = bool(
            ctx.get("suppress_widget_tool_calls")
            or kwargs.get("suppress_widget_tool_calls")
        )

        memory_server = _build_memory_server(conv_id, retriever)
        control_server = (
            _build_control_server(bridged_control_tools, second_eyes_phase)
            if bridged_control_tools
            else None
        )
        native_tools = _second_eyes_native_tools(second_eyes_phase)
        control_allowed_tools = _control_allowed_tool_names(bridged_control_tools)
        allowed_tools = [
            *(native_tools if native_tools is not None else _NATIVE_TOOLS),
            _RECALL_TOOL,
            *control_allowed_tools,
        ]
        new_artifact_gate = _NEW_ARTIFACT_GATE_MARKER in (system_prompt or "")
        hooks = _build_agent_sdk_hooks(second_eyes_phase, HookMatcher, new_artifact_gate=new_artifact_gate)
        # Fold AFTER the marker scan above — folding moves system text into the
        # prompt body, so any system_prompt scans must run first.
        system_prompt, prompt = _fold_oversized_system_prompt(system_prompt, prompt)
        # For plan_draft, `tools` already restricts Claude native tools to a
        # read-only set. Passing deny rules for tools outside that set can make
        # the Agent SDK abort with "Permission deny rule ... matches no known
        # tool", so keep disallowed_tools empty here.
        disallowed_tools: list[str] = []
        mcp_servers = {_MEMORY_SERVER_NAME: memory_server}
        if control_server is not None:
            mcp_servers[_CONTROL_SERVER_NAME] = control_server

        max_turns = self.config.get("max_turns")
        options = ClaudeAgentOptions(
            system_prompt=system_prompt or None,
            # Do not inherit the user's CLAUDE.md / project / local settings — JLC
            # supplies the entire persona + memory via system_prompt/prompt.
            setting_sources=[],
            permission_mode="bypassPermissions",
            cwd=str(cwd),
            env=_AGENT_SDK_SUBPROCESS_ENV,
            model=self.model,
            mcp_servers=mcp_servers,
            tools=native_tools,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            hooks=hooks,
            include_hook_events=True,
            # Optional safety cap on internal agentic iterations (providers.yaml
            # `max_turns`). None = let Claude run to natural completion.
            max_turns=int(max_turns) if max_turns else None,
            # effort=None is the SDK default; only override when the route gave us
            # a recognized level.
            effort=effort,
            # True token-level streaming (2026-06-25). Verified against the
            # installed claude_agent_sdk 0.2.101: this bool (types.py:1831) appends
            # the CLI flag --include-partial-messages (subprocess_cli.py:334-335),
            # next to the already-working --include-hook-events. It is ORTHOGONAL to
            # env / permission_mode / max_turns and does NOT touch process_env
            # (subprocess_cli.py:431-436), so ENABLE_TOOL_SEARCH=0 + the inherited
            # OAuth creds / CLAUDE_CONFIG_DIR are untouched. It makes the CLI emit
            # type:"stream_event" messages -> StreamEvent (message_parser.py:304-315)
            # carrying the RAW Anthropic SSE delta dict, which _translate forwards as
            # fine-grained reasoning_content so pi animates the live "돌아가는 컨텐츠"
            # (text/thinking/tool-input typed out char-by-char). Whole-block paths
            # (Part A/B/C) stay as the fallback for results that arrive only as a
            # completed block.
            include_partial_messages=True,
        )
        cli_path = self.config.get("cli_path")
        if cli_path:
            options.cli_path = cli_path

        block_types = {
            "text": TextBlock,
            "thinking": ThinkingBlock,
            "tool_use": ToolUseBlock,
            "tool_result": ToolResultBlock,
        }
        msg_types = {
            "assistant": AssistantMessage,
            "user": UserMessage,
            "result": ResultMessage,
            "system": SystemMessage,
            "hook_event": HookEventMessage,
            "stream_event": StreamEvent,
        }

        t0 = time.monotonic()
        pending_text_parts: list[str] = []
        stream_state = _AgentSDKStreamState()
        emitted_content = False
        result_seen = False
        # Turn-taking interlock (regime B): ids of GAN/job handoff tool calls seen
        # this turn; a successful result for any of them force-ends the SDK turn.
        pending_handoff_ids: set[str] = set()

        for msg in _drive_async(lambda: query(prompt=prompt, options=options)):
            _debug_log_raw_message(msg)
            for chunk, is_content in _translate(
                msg,
                block_types,
                msg_types,
                text_buffer=pending_text_parts,
                stream_state=stream_state,
                stream_preamble=new_artifact_gate,
                stream_answer_text=stream_answer_text,
                suppress_widget_tool_calls=suppress_widget_tool_calls,
            ):
                if is_content:
                    emitted_content = True
                yield chunk
            # Turn-taking interlock: end the SDK turn after a successful GAN/job
            # handoff so the sender goes idle immediately (see
            # _TURN_ENDING_HANDOFF_TOOL_NAMES). Scan AFTER _translate has emitted the
            # '[done: <tool>]' label. The whole message is scanned before breaking, so
            # parallel fan-out (several handoffs in one assistant message -> several
            # results in one user message) is preserved -- every dispatch already
            # happened by the time the results arrive. Breaking closes the query
            # generator, which cancels the SDK subprocess (_drive_async finally).
            handoff_turn_end = False
            content = getattr(msg, "content", None)
            if isinstance(content, (list, tuple)):
                for block in content:
                    if isinstance(block, block_types["tool_use"]):
                        if getattr(block, "name", "") in _TURN_ENDING_HANDOFF_TOOL_NAMES:
                            handoff_id = str(getattr(block, "id", "") or "")
                            if handoff_id:
                                pending_handoff_ids.add(handoff_id)
                                _debug_note(
                                    "handoff_seen",
                                    tool=getattr(block, "name", ""),
                                    id=handoff_id,
                                )
                    elif isinstance(block, block_types["tool_result"]):
                        handoff_id = str(getattr(block, "tool_use_id", "") or "")
                        if handoff_id and handoff_id in pending_handoff_ids:
                            rbody = getattr(block, "content", None)
                            if rbody in (None, "", [], {}):
                                rbody = getattr(msg, "tool_use_result", None)
                            is_err = bool(getattr(block, "is_error", False))
                            ok = _handoff_result_succeeded(is_err, rbody)
                            # A terminal-state failure (job/GAN already closed) also
                            # ends the turn: there is no valid retry, so lingering
                            # only re-pokes the dead job and keeps both windows live.
                            terminal = (not ok) and _handoff_result_is_terminal(rbody)
                            _debug_note(
                                "handoff_result",
                                id=handoff_id,
                                success=ok,
                                terminal=terminal,
                                is_error=is_err,
                            )
                            if ok or terminal:
                                handoff_turn_end = True
            if handoff_turn_end:
                _debug_note("handoff_force_end", ids=sorted(pending_handoff_ids))
                sentinel = _tool_activity_sentinel_chunk(stream_state.tool_activity)
                if sentinel is not None:
                    yield sentinel
                yield _finish_chunk("stop", tool_activity=stream_state.tool_activity)
                # Terminal chunk emitted; mark so the result-less fallback below does
                # not emit a SECOND finish chunk after we break.
                result_seen = True
                break
            if isinstance(msg, msg_types["result"]):
                result_seen = True
                final_text = _final_text_from_result_or_buffer(msg, pending_text_parts)
                if stream_answer_text and stream_state.partial_answer_text_parts:
                    streamed_text = "".join(stream_state.partial_answer_text_parts)
                    if final_text.startswith(streamed_text):
                        final_text = final_text[len(streamed_text) :]
                    elif streamed_text.startswith(final_text):
                        final_text = ""
                if final_text:
                    yield _content_chunk(final_text)
                    emitted_content = True
                self.llm_meta = _result_llm_meta(
                    msg, self.model, int((time.monotonic() - t0) * 1000)
                )
                # LIVE transport (item 1): fold the observed-work trailer into the
                # FINAL assistant TEXT as a sentinel line BEFORE the finish chunk, so
                # it survives pi-ai's openai-completions materializer (which keeps
                # delta.content but drops unknown top-level chunk keys). The consumer
                # parses+strips it. The top-level key on the finish chunk is kept for
                # a future metadata-carry path.
                sentinel = _tool_activity_sentinel_chunk(stream_state.tool_activity)
                if sentinel is not None:
                    yield sentinel
                yield _finish_chunk(
                    getattr(msg, "stop_reason", None) or "stop",
                    tool_activity=stream_state.tool_activity,
                )

        if not result_seen:
            if pending_text_parts and not emitted_content:
                yield _content_chunk("".join(pending_text_parts))
            # Stream ended without a ResultMessage — emit a terminal chunk so the
            # loop does not hang waiting for finish_reason. Same sentinel transport as
            # the ResultMessage path so a result-less turn still feeds pi's sensor.
            sentinel = _tool_activity_sentinel_chunk(stream_state.tool_activity)
            if sentinel is not None:
                yield sentinel
            yield _finish_chunk("stop", tool_activity=stream_state.tool_activity)


# ----------------------------------------------------------------- helpers


# The SDK passes system_prompt as a `--system-prompt <text>` CLI argument
# (subprocess_cli._build_command), and Windows CreateProcess caps the whole
# command line at 32,767 chars. An oversized system prompt therefore dies AT
# SPAWN with WinError 206, which python maps to ENOENT and the SDK misreports as
# CLINotFoundError "Claude Code not found" (live: ultracode verifier carrying 5
# finders' findings, run 60f6f78b3b3a, 2026-07-02). Fold oversized system
# prompts into the stdin-borne prompt, which has no such limit. POSIX argv
# limits are ~2MB, so the guard only engages on Windows.
_SYSTEM_PROMPT_ARG_SAFE_CHARS = 24_000


def _fold_oversized_system_prompt(
    system_prompt: str | None,
    prompt: str,
    *,
    limit: int = _SYSTEM_PROMPT_ARG_SAFE_CHARS,
    platform: str | None = None,
) -> tuple[str | None, str]:
    if not system_prompt:
        return system_prompt, prompt
    if (platform if platform is not None else sys.platform) != "win32":
        return system_prompt, prompt
    if len(system_prompt) <= limit:
        return system_prompt, prompt
    folded = (
        "<system-instructions>\n"
        f"{system_prompt}\n"
        "</system-instructions>\n\n"
        f"{prompt}"
    )
    return None, folded


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, str]:
    """Fold JLC's OpenAI-style messages into (system_prompt, user prompt).

    System messages (persona + carried context JLC already compressed) become the
    replace-mode system_prompt. The remaining conversation is serialized with role
    labels into a single prompt string; JLC already bounds this via memory carry so
    a single prompt is faithful and O(n).
    """
    system_parts: list[str] = []
    convo_parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if content is None:
            continue
        text = content if isinstance(content, str) else _coerce_content(content)
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            convo_parts.append(f"[assistant]\n{text}")
        elif role == "tool":
            convo_parts.append(f"[tool result]\n{text}")
        else:  # user / unknown
            convo_parts.append(f"[user]\n{text}")
    return "\n\n".join(system_parts), "\n\n".join(convo_parts)


def _coerce_content(content: Any) -> str:
    """Best-effort flatten of list/multimodal content into text."""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _current_instruction_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    last_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = _coerce_content(message.get("content"))
        if role in {"system", "developer"}:
            parts.append(content)
        elif role == "user":
            last_user = content
    if last_user:
        parts.append(last_user)
    return "\n".join(part for part in parts if part)


def _infer_second_eyes_phase_from_messages(messages: list[dict[str, Any]]) -> str | None:
    text = _current_instruction_text(messages)
    if _SECOND_EYES_REVIEW_MARKER in text or _LEGACY_SECOND_EYES_REVIEW_MARKER in text:
        return _SECOND_EYES_REVIEW_PHASE
    if _SECOND_EYES_MAIN_MARKER in text or _LEGACY_SECOND_EYES_MAIN_MARKER in text:
        return _SECOND_EYES_IMPLEMENT_PHASE
    if _SECOND_EYES_REQUEST_MARKER in text or _SECOND_EYES_REMINDER_MARKER in text:
        return _SECOND_EYES_PLAN_DRAFT_PHASE
    return None


def _coerce_second_eyes_phase(value: Any) -> str | None:
    if not value:
        return None
    phase = str(value).strip().lower()
    return phase if phase in _SECOND_EYES_PHASES else None


def _second_eyes_native_tools(phase: str | None) -> list[str] | None:
    if phase == _SECOND_EYES_PLAN_DRAFT_PHASE:
        return _SECOND_EYES_PLAN_DRAFT_NATIVE_TOOLS
    if phase == _SECOND_EYES_REVIEW_PHASE:
        return _SECOND_EYES_REVIEW_NATIVE_TOOLS
    return None


def _second_eyes_blocked_native_tools(phase: str | None) -> frozenset[str]:
    if phase == _SECOND_EYES_PLAN_DRAFT_PHASE:
        return _SECOND_EYES_PLAN_DRAFT_BLOCKED_TOOLS
    if phase == _SECOND_EYES_REVIEW_PHASE:
        return _SECOND_EYES_REVIEW_BLOCKED_TOOLS
    return frozenset()


def _second_eyes_permission_message(phase: str, tool_name: str) -> str:
    if phase == _SECOND_EYES_PLAN_DRAFT_PHASE:
        return (
            f"Critic Mode plan draft cannot use {tool_name}; gather choices/recon, "
            "draft the main plan, then send a review-only job before implementation."
        )
    return f"Critic Mode reviewer cannot use {tool_name}; return review-only findings to the main window."


def _build_agent_sdk_hooks(
    phase: str | None, hook_matcher_cls: Any, new_artifact_gate: bool = False
) -> dict[str, list[Any]]:
    blocked = _second_eyes_blocked_native_tools(phase)
    # Per-call mutable: flips True once the model calls ask_user this turn. A fresh
    # closure is built per request (_build_agent_sdk_hooks is called per call), so
    # this state is correctly turn-scoped.
    ask_user_seen = {"seen": False}

    async def _pre_tool_use(input_data: Any, _tool_use_id: str | None, _context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(input_data, dict):
            raw_tool_name = input_data.get("tool_name") or input_data.get("toolName") or ""
        else:
            raw_tool_name = getattr(input_data, "tool_name", "") or getattr(input_data, "toolName", "")
        tool_name = _normalize_tool_schema_name(raw_tool_name)
        # Native AskUserQuestion (or any non-bridged variant) has no question UI in
        # this embedded one-shot SDK turn: it fails with is_error immediately and the
        # model never receives the answer. Redirect to the bridged ask_user. This MUST
        # run BEFORE the new-artifact gate below -- otherwise the native call would set
        # ask_user_seen and silently disable the create-file gate without ever asking.
        if _is_native_ask_user_leak(raw_tool_name, tool_name):
            reason = (
                "[JLC] The native AskUserQuestion tool cannot show a question UI in "
                "this embedded agent-sdk turn: it fails immediately with an error and "
                "you never receive the user's answer. Call the bridged `ask_user` tool "
                "instead -- it opens a modal in the JARVIS UI and returns the user's reply."
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
                "reason": reason,
            }
        if new_artifact_gate:
            if tool_name == _ASK_USER_TOOL_NAME:
                ask_user_seen["seen"] = True
            elif not ask_user_seen["seen"] and tool_name in _NEW_ARTIFACT_GATED_TOOLS:
                reason = (
                    "[JLC] Call ask_user first to confirm design / scope / features "
                    "before creating files -- this is a new-artifact turn."
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    },
                    "reason": reason,
                }
        if tool_name in _EMBEDDED_UNSUPPORTED_NATIVE_TOOLS:
            reason = (
                "[JLC] The native Workflow tool is unavailable here: its background "
                "workflow has no runner in this environment and cannot return results, "
                "so it would hang. Use the `ultracode` tool for parallel multi-agent "
                "orchestration, or `delegate_subagent` for a single subagent -- both "
                "run inside JARVIS and stream results back."
            )
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
                "reason": reason,
            }
        if phase and tool_name in blocked:
            reason = _second_eyes_permission_message(phase, tool_name)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                },
                "reason": reason,
            }
        if isinstance(input_data, dict):
            tool_input = input_data.get("tool_input")
            if tool_input is None:
                tool_input = input_data.get("toolInput")
        else:
            tool_input = getattr(input_data, "tool_input", None)
            if tool_input is None:
                tool_input = getattr(input_data, "toolInput", None)
        updated_input = _remap_edit_param_slips(tool_name, tool_input)
        if updated_input is not None:
            _debug_note("edit_param_remap", tool_name=tool_name)
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": updated_input,
                }
            }
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
        }

    async def _post_tool_use(_input_data: Any, _tool_use_id: str | None, _context: dict[str, Any]) -> dict[str, Any]:
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse"}}

    async def _post_tool_use_failure(
        _input_data: Any, _tool_use_id: str | None, _context: dict[str, Any]
    ) -> dict[str, Any]:
        return {"hookSpecificOutput": {"hookEventName": "PostToolUseFailure"}}

    return {
        "PreToolUse": [hook_matcher_cls(matcher=None, hooks=[_pre_tool_use])],
        "PostToolUse": [hook_matcher_cls(matcher=None, hooks=[_post_tool_use])],
        "PostToolUseFailure": [hook_matcher_cls(matcher=None, hooks=[_post_tool_use_failure])],
    }


def _tool_schema_name(tool: Any) -> str | None:
    if not isinstance(tool, dict):
        return None
    direct = tool.get("name")
    if isinstance(direct, str) and direct:
        return _normalize_tool_schema_name(direct)
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str) and name:
            return _normalize_tool_schema_name(name)
    definition = tool.get("definition")
    if isinstance(definition, dict):
        name = definition.get("name")
        if isinstance(name, str) and name:
            return _normalize_tool_schema_name(name)
    return None


def _requested_tool_schema_names(tools: list[dict[str, Any]] | None) -> list[str]:
    if not tools:
        return []
    names = [_tool_schema_name(tool) for tool in tools]
    present = [name for name in names if name]
    present = [name for idx, name in enumerate(present) if name and name not in present[:idx]]
    return present


def _bridgeable_control_tool_names(names: list[str]) -> set[str]:
    normalized = {_normalize_tool_schema_name(name) for name in names}
    return {name for name in normalized if name in _BRIDGED_CONTROL_TOOL_NAMES}


def _control_allowed_tool_names(names: set[str]) -> list[str]:
    return [
        f"mcp__{_CONTROL_SERVER_NAME}__{name}"
        for name in _CONTROL_TOOL_ORDER
        if _normalize_tool_schema_name(name) in names
    ]


def _jlc_control_only_tool_names(names: list[str]) -> list[str]:
    if not names:
        return []
    normalized = {_normalize_tool_schema_name(name) for name in names}
    unsupported = [
        name
        for name in normalized
        if name in _JLC_CONTROL_TOOL_NAMES and name not in _BRIDGED_CONTROL_TOOL_NAMES
    ]
    if unsupported and all(name in _JLC_CONTROL_TOOL_NAMES for name in normalized):
        return unsupported
    return []


def _build_memory_server(conv_id: str, retriever: Any):
    """Expose JLC's recall_turns as an in-process MCP tool, bound to this turn's
    conv_id/retriever via closure. The handler runs ``asyncio.run`` internally, so
    we offload it to a worker thread (asyncio.to_thread) to avoid the "cannot call
    asyncio.run from a running event loop" error inside the SDK's loop."""
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: PLC0415

    from jlc_agentic.agentic.tools import recall_turn  # noqa: PLC0415

    recall_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in past JARVIS conversations.",
            },
            "top_k": {"type": "integer", "description": "How many fragments to return."},
        },
        "required": ["query"],
    }

    @tool(
        "recall_turns",
        "Search JARVIS's long-term conversation memory (JHB) for relevant past "
        "turns. Use when the user refers to earlier discussions or asks what was "
        "decided/done before.",
        recall_schema,
    )
    async def _recall(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            recall_turn.handler,
            query=str(args.get("query") or ""),
            top_k=int(args.get("top_k") or 5),
            conv_id=conv_id,
            retriever=retriever,
            trace_surface="sdk_mcp",
        )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    return create_sdk_mcp_server(name=_MEMORY_SERVER_NAME, tools=[_recall])


def _current_control_pair8() -> str:
    """Return the 8-char pair identifier for control-bridge routing.

    Always returns a non-empty string.  When no real pair_id is available
    (Agent-SDK regime, no JARVIS_PAIR_ID set), returns the ``__self__``
    sentinel so the control-bridge enqueue still writes to a valid bucket.
    The dequeue side (app.py /control/pending and /control/{id}/answer)
    uses the same sentinel, so both sides agree without a real pair_id.
    """
    try:
        from jarvis_sidecar import pairing  # noqa: PLC0415

        pair_id = pairing.current_pair_id()
    except Exception:  # pragma: no cover - import only fails outside sidecar runtime
        pair_id = os.environ.get("JARVIS_PAIR_ID")
    pair8 = str(pair_id or "").strip()[:8]
    return pair8 or _CONTROL_FALLBACK_PAIR8


def _build_control_server(tool_names: set[str], second_eyes_phase: str | None):
    """Expose JARVIS control tools to Claude Agent SDK as MCP tools.

    The model authors the tool payload, but the outer JARVIS/Pi harness executes
    UI/control side effects. We bridge only the narrow tools whose payload must be
    model-authored but whose side effects belong to the harness.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool  # noqa: PLC0415

    from jarvis_sidecar.control_bridge import submit_request  # noqa: PLC0415

    ask_user_schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "text": {"type": "string"},
                        "prompt": {"type": "string"},
                        "title": {"type": "string"},
                        "label": {"type": "string"},
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 6,
                        },
                        "recommended": {"type": "string"},
                        "allow_custom": {"type": "boolean"},
                    },
                },
            },
            "timeout_seconds": {"type": "number"},
        },
        "required": ["questions"],
    }

    @tool(
        _ASK_USER_TOOL_NAME,
        "Ask the human user 1-6 short planning questions through the JARVIS UI. "
        "Use this when user-facing choices are unresolved. Before calling, FIRST write "
        "ONE short, warm sentence in the USER'S OWN LANGUAGE letting them know you have "
        "a few quick questions to get it right (do NOT restate the questions or options "
        "in that line -- the interactive UI shows those). The harness streams that one "
        "line live, just before the question UI appears. Return after the user answers; "
        "do not call other tools in the same internal step.",
        ask_user_schema,
    )
    async def _ask_user(args: dict[str, Any]) -> dict[str, Any]:
        target = _current_control_pair8()
        if not target:
            result = {
                "ok": False,
                "error": "JARVIS_PAIR_ID is required for bridged ask_user",
            }
        else:
            timeout_seconds = args.get("timeout_seconds")
            result = await asyncio.to_thread(
                submit_request,
                kind=_ASK_USER_TOOL_NAME,
                to_window=target,
                payload=args,
                timeout_seconds=timeout_seconds,
            )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    spawn_window_schema = {
        "type": "object",
        "properties": {
            "initial_directive": {
                "type": "string",
                "description": "Task or review directive to send to the new JARVIS window.",
            },
            "model": {
                "type": "string",
                "description": "Optional provider/model or bare model name for the worker.",
            },
            "label": {
                "type": "string",
                "description": "Optional user-facing window label; omit for worker1, worker2, ...",
            },
            "timeout_seconds": {"type": "number"},
            "gan": {"type": "boolean"},
            "job": {
                "type": "boolean",
                "description": "Defaults to true when initial_directive is present; pass false only for explicit passive one-way notices.",
            },
            "issues_open": {
                "type": "number",
                "description": "Required when gan is true.",
            },
            "feature_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional map feature ids; requires job=true.",
            },
        },
    }

    @tool(
        _SPAWN_WINDOW_TOOL_NAME,
        "Open a new JARVIS Code worker window through the owning Pi harness. "
        "Use only when the user explicitly asks for a worker, Critic Mode, "
        "or separate delegated window. Worker tasks default to job handback. "
        "The MCP tool only submits the request; "
        "the outer JARVIS harness performs the actual UI/window side effect.",
        spawn_window_schema,
    )
    async def _spawn_window(args: dict[str, Any]) -> dict[str, Any]:
        target = _current_control_pair8()
        if not target:
            result = {
                "ok": False,
                "error": "JARVIS_PAIR_ID is required for bridged spawn_window",
            }
        else:
            timeout_seconds = args.get("timeout_seconds")
            result = await asyncio.to_thread(
                submit_request,
                kind=_SPAWN_WINDOW_TOOL_NAME,
                to_window=target,
                payload=args,
                timeout_seconds=timeout_seconds,
            )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    list_windows_schema = {"type": "object", "properties": {}}

    @tool(
        _LIST_WINDOWS_TOOL_NAME,
        "List active JARVIS windows and labels before sending a message to an existing worker.",
        list_windows_schema,
    )
    async def _list_windows(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_list_windows)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    send_directive_schema = {
        "type": "object",
        "properties": {
            "to_window": {
                "type": "string",
                "description": "Target JARVIS window pair8 address or unique label, such as worker1.",
            },
            "message": {
                "type": "string",
                "description": "Message or directive body to deliver.",
            },
        },
        "required": ["to_window", "message"],
    }

    @tool(
        _SEND_DIRECTIVE_TOOL_NAME,
        "Low-level passive one-shot message to an existing JARVIS window. "
        "Prefer job_send for worker tasks or any reply that should wake the main window.",
        send_directive_schema,
    )
    async def _send_directive(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_send_directive, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    job_send_schema = {
        "type": "object",
        "properties": {
            "to_window": {
                "type": "string",
                "description": "Target JARVIS window pair8 address or unique label.",
            },
            "message": {
                "type": "string",
                "description": "Dispatch, review request, progress handback, or next-cycle instruction.",
            },
            "job_id": {
                "type": "string",
                "description": "Existing job_id from the turn header; omit only to start a new job.",
            },
        },
        "required": ["to_window", "message"],
    }

    @tool(
        _JOB_SEND_TOOL_NAME,
        "Send a structured JARVIS job handoff or handback to another window. "
        "Use this for Critic Mode plan/review handbacks and job-loop dispatches. "
        "When this is a worker handback, include the current job_id and send it "
        "to the orchestrator window from the job header.",
        job_send_schema,
    )
    async def _job_send(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_job_send, args, second_eyes_phase)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    job_close_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Existing server-issued job_id."},
            "status": {
                "type": "string",
                "enum": ["done", "escalated"],
                "description": "done when complete, escalated when blocked or cycle-capped.",
            },
            "summary": {
                "type": "string",
                "description": "Terminal completion or escalation summary.",
            },
        },
        "required": ["job_id", "status", "summary"],
    }

    @tool(
        _JOB_CLOSE_TOOL_NAME,
        "Close a JARVIS job with a terminal report. Workers should normally use "
        "job_send for review handbacks; if a worker closes, the server converts "
        "that into a review handback so the orchestrator keeps final judgment.",
        job_close_schema,
    )
    async def _job_close(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_job_close, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    set_chat_model_schema = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Chat model as provider/model. Omit to fetch model catalog.",
            },
            "force": {
                "type": "boolean",
                "description": "Skip catalog validation and save as-is.",
            },
        },
    }

    @tool(
        _SET_CHAT_MODEL_TOOL_NAME,
        "List available models and active chat role, or set chat model immediately. "
        "Call without a model argument to fetch the model catalog.",
        set_chat_model_schema,
    )
    async def _set_chat_model(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            _submit_control_set_llm_model,
            "chat",
            args,
        )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    set_encoder_model_schema = {
        "type": "object",
        "properties": {
            "model": {
                "type": "string",
                "description": "Encoder model as provider/model. Omit to fetch model catalog.",
            },
            "force": {
                "type": "boolean",
                "description": "Skip catalog validation and save as-is.",
            },
        },
    }

    @tool(
        _SET_ENCODER_MODEL_TOOL_NAME,
        "List available models and active encoder role, or set encoder role immediately. "
        "Call without a model argument to fetch the model catalog.",
        set_encoder_model_schema,
    )
    async def _set_encoder_model(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(
            _submit_control_set_llm_model,
            "encoder",
            args,
        )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    set_window_label_schema = {
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "New short display label for this window.",
            },
        },
        "required": ["label"],
    }

    @tool(
        _SET_WINDOW_LABEL_TOOL_NAME,
        "Rename this JARVIS window. Use only when user explicitly asks for a label change.",
        set_window_label_schema,
    )
    async def _set_window_label(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_set_window_label, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    gan_send_schema = {
        "type": "object",
        "properties": {
            "to_window": {
                "type": "string",
                "description": "Target JARVIS window pair8 address or unique label",
            },
            "message": {
                "type": "string",
                "description": "Verdict, acceptance, or rebuttal body with enumerated open issues",
            },
            "issues_open": {
                "type": "number",
                "description": "Number of open issues in this round",
            },
            "gan_id": {
                "type": "string",
                "description": "Existing gan_id; omit to start a new GAN",
            },
        },
        "required": ["to_window", "message", "issues_open"],
    }

    @tool(
        _GAN_SEND_TOOL_NAME,
        "Send a structured GAN consensus round to another JARVIS window. Use this only "
        "when the user explicitly asks to run GAN/consensus. If gan_id is omitted, this "
        "starts round 1 and the server issues gan_id. If gan_id is provided, the server "
        "stamps the next round. When replying inside an existing GAN (the turn header "
        "shows a gan_id), you MUST pass that gan_id; the server rejects a second open GAN "
        "between the same two windows. Protocol: round 1 hands the work to the destroyer, "
        "round 2 is the destroyer's verdict (sets the issue baseline), round 3 is the "
        "rebuttal or acceptance; maximum 3 send rounds; enumerate open issues and pass "
        "issues_open. From round 3 onward issues_open must strictly decrease. This tool "
        "does not spawn windows; use spawn_window separately. The counterpart's round "
        "arrives automatically as a new turn once this window is idle.",
        gan_send_schema,
    )
    async def _gan_send(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_gan_send, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    gan_close_schema = {
        "type": "object",
        "properties": {
            "gan_id": {
                "type": "string",
                "description": "Existing server-issued gan_id",
            },
            "status": {
                "type": "string",
                "enum": ["agreed", "escalated"],
                "description": "agreed when remaining issues are resolved; escalated when issues remain.",
            },
            "summary": {
                "type": "string",
                "description": "Terminal consensus or escalation summary",
            },
        },
        "required": ["gan_id", "status", "summary"],
    }

    @tool(
        _GAN_CLOSE_TOOL_NAME,
        "Close a GAN consensus session with a terminal report. status must be agreed or "
        "escalated. Use agreed only when the remaining issues are resolved. Use escalated "
        "when issues remain after the tie-break rules or the round cap; the summary must "
        "include remaining issues and both sides' arguments. After close, the server "
        "rejects any further append for that gan_id.",
        gan_close_schema,
    )
    async def _gan_close(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_gan_close, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    ultracode_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Specific review, audit, or investigation task including target files or scope.",
            },
            "dimensions": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
                "description": "Focused finder dimensions such as correctness, security, or performance.",
            },
            "max_concurrency": {
                "type": "number",
                "description": "Maximum concurrent finder calls.",
            },
            "max_calls": {
                "type": "number",
                "description": "Soft provider-call budget.",
            },
            "max_wallclock_sec": {
                "type": "number",
                "description": "Wallclock budget in seconds; expiry cancels running finders.",
            },
            "project_root": {
                "type": "string",
                "description": "Optional project root for read/grep grounding.",
            },
        },
        "required": ["task", "dimensions"],
    }

    @tool(
        _ORCHESTRATE_START_TOOL_NAME,
        "Start a background multi-angle orchestration for thorough review/audit. "
        "This returns immediately with an orchestration_id and ends the current turn; "
        "JARVIS will wake this window with the synthesized result when the job completes. "
        "Do not pass model names.",
        ultracode_schema,
    )
    async def _ultracode(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_ultracode, args)
        return {"content": [{"type": "text", "text": _ultracode_result_text(result)}]}

    orchestration_id_schema = {
        "type": "object",
        "properties": {
            "orchestration_id": {
                "type": "string",
                "description": "Background orchestration id returned by ultracode.",
            }
        },
        "required": ["orchestration_id"],
    }

    @tool(
        _ORCHESTRATE_STATUS_TOOL_NAME,
        "Check whether a background orchestration is still running, done, or unknown.",
        orchestration_id_schema,
    )
    async def _ultracode_status(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_ultracode_status, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    @tool(
        _ORCHESTRATE_RESULT_TOOL_NAME,
        "Fetch the full result for a completed background orchestration.",
        orchestration_id_schema,
    )
    async def _ultracode_result(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_ultracode_result, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    @tool(
        _ORCHESTRATE_CANCEL_TOOL_NAME,
        "Cancel a running background orchestration by id.",
        orchestration_id_schema,
    )
    async def _ultracode_cancel(args: dict[str, Any]) -> dict[str, Any]:
        result = await asyncio.to_thread(_submit_control_ultracode_cancel, args)
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    map_create_schema = {
        "type": "object",
        "properties": {
            "project_path": {
                "type": "string",
                "description": "Absolute project root; artifacts live in <project>/.jarvis-map/",
            },
            "title": {
                "type": "string",
                "description": "Map title; defaults to the project folder name",
            },
            "append": {
                "type": "boolean",
                "description": "Add features to the open map run",
            },
            "replace": {
                "type": "boolean",
                "description": "Abandon the open map run and start a new one",
            },
            "features": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Feature name in the user's vocabulary",
                        },
                        "summary": {
                            "type": "string",
                            "description": "One-line scope note",
                        },
                        "zone": {
                            "type": "string",
                            "enum": ["feature", "skeleton"],
                            "description": "skeleton = global scaffolding/refactor zone; the worker owns the method there",
                        },
                        "acceptance": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Checkable acceptance criteria; checkpoints verify these verbatim",
                        },
                    },
                    "required": ["title", "acceptance"],
                },
            },
        },
        "required": ["project_path", "features"],
    }

    @tool(
        _MAP_CREATE_TOOL_NAME,
        "Persist the feature map of a delegated build that is too large for one worker "
        "context, before the first dispatch. Use ONLY when a delegated build will not fit "
        "a single worker context; whole delegation to one worker is the default. Record "
        "every feature with its acceptance criteria, then dispatch features via "
        "job_send/spawn_window with feature_ids. The map lives in <project>/.jarvis-map/ "
        "and survives restarts. Use append:true to add features to the open map; "
        "replace:true abandons the open map and starts fresh.",
        map_create_schema,
    )
    async def _map_create(args: dict[str, Any]) -> dict[str, Any]:
        target = _current_control_pair8()
        result = await asyncio.to_thread(
            submit_request,
            kind=_MAP_CREATE_TOOL_NAME,
            to_window=target,
            payload=args,
            timeout_seconds=args.get("timeout_seconds"),
        )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    feature_verdict_schema = {
        "type": "object",
        "properties": {
            "feature_id": {
                "type": "string",
                "description": "Map feature id from MAP STATUS (e.g. f2)",
            },
            "verdict": {
                "type": "string",
                "enum": ["pass", "reject"],
            },
            "reason": {
                "type": "string",
                "description": "Required on reject: what is wrong and what right looks like",
            },
            "evidence": {
                "type": "string",
                "description": "Required on pass: the runnable check you executed",
            },
        },
        "required": ["feature_id", "verdict"],
    }

    @tool(
        _FEATURE_VERDICT_TOOL_NAME,
        "Record the checkpoint verdict for one map feature. pass requires evidence (the "
        "runnable check you executed); reject requires a concrete reason — it rides the "
        "next dispatch ticket as the intent channel to the worker. The ledger, rejection "
        "cap, and escalate ladder advance automatically; the result tells you the exact "
        "next step.",
        feature_verdict_schema,
    )
    async def _feature_verdict(args: dict[str, Any]) -> dict[str, Any]:
        target = _current_control_pair8()
        result = await asyncio.to_thread(
            submit_request,
            kind=_FEATURE_VERDICT_TOOL_NAME,
            to_window=target,
            payload=args,
            timeout_seconds=args.get("timeout_seconds"),
        )
        return {"content": [{"type": "text", "text": _to_text(result)}]}

    all_tools = {
        _ASK_USER_TOOL_NAME: _ask_user,
        _SPAWN_WINDOW_TOOL_NAME: _spawn_window,
        _LIST_WINDOWS_TOOL_NAME: _list_windows,
        _SEND_DIRECTIVE_TOOL_NAME: _send_directive,
        _JOB_SEND_TOOL_NAME: _job_send,
        _JOB_CLOSE_TOOL_NAME: _job_close,
        _SET_CHAT_MODEL_TOOL_NAME: _set_chat_model,
        _SET_ENCODER_MODEL_TOOL_NAME: _set_encoder_model,
        _SET_WINDOW_LABEL_TOOL_NAME: _set_window_label,
        _GAN_SEND_TOOL_NAME: _gan_send,
        _GAN_CLOSE_TOOL_NAME: _gan_close,
        _MAP_CREATE_TOOL_NAME: _map_create,
        _FEATURE_VERDICT_TOOL_NAME: _feature_verdict,
        _ORCHESTRATE_START_TOOL_NAME: _ultracode,
        _ORCHESTRATE_STATUS_TOOL_NAME: _ultracode_status,
        _ORCHESTRATE_RESULT_TOOL_NAME: _ultracode_result,
        _ORCHESTRATE_CANCEL_TOOL_NAME: _ultracode_cancel,
    }
    tools = [all_tools[name] for name in _CONTROL_TOOL_ORDER if name in tool_names]
    return create_sdk_mcp_server(name=_CONTROL_SERVER_NAME, tools=tools)


def _resolve_control_window(value: Any) -> str:
    from jarvis_sidecar.directives import list_windows  # noqa: PLC0415
    from jarvis_sidecar.window_labels import normalize_pair8, resolve_live_label  # noqa: PLC0415

    raw = str(value or "").strip()
    if not raw:
        raise ValueError("to_window is required")
    direct = normalize_pair8(raw)
    windows = list_windows()
    if direct:
        direct_matches = [window for window in windows if str(window.get("pair8") or "") == direct]
        if any(bool(window.get("alive")) for window in direct_matches):
            return direct
        if direct_matches:
            raise ValueError(f"window is not live: {raw!r}")
    resolved = resolve_live_label(raw, windows)
    if resolved:
        return resolved
    if direct:
        return direct
    raise ValueError(f"unknown window: {raw!r}")


def _submit_control_list_windows() -> dict[str, Any]:
    from jarvis_sidecar.directives import list_windows  # noqa: PLC0415

    return {"ok": True, "windows": list_windows()}


def _optional_int(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _optional_float(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc


def _submit_control_send_directive(args: dict[str, Any]) -> dict[str, Any]:
    from jarvis_sidecar.directives import (  # noqa: PLC0415
        GANDirectiveError,
        JobDirectiveError,
        append_directive,
        list_windows,
    )

    from_window = _current_control_pair8()
    if not from_window:
        return {"ok": False, "error": "JARVIS_PAIR_ID is required for bridged send_directive"}
    try:
        to_window = _resolve_control_window(args.get("to_window"))
        message = str(args.get("message") or "").strip()
        if not message:
            raise ValueError("message is required")
        item = append_directive(
            kind="directive",
            from_window=from_window,
            to_window=to_window,
            body=message,
        )
        return {"ok": True, "item": item, "windows": list_windows()}
    except (GANDirectiveError, JobDirectiveError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _submit_control_gan_send(args: dict[str, Any]) -> dict[str, Any]:
    """Sidecar-direct bridge for gan_send (Pattern S).

    Mirrors pi's regime-A gan_send execute (jarvis-jlc.ts), which only POSTs to
    /directives with gan_target/issues_open and touches ZERO pi in-process state.
    We call append_directive in-process (the same function /directives calls),
    replicating the 409 self-heal: a new GAN whose open session already exists
    between the two windows is continued once with the existing gan_id.
    """
    from jarvis_sidecar.directives import (  # noqa: PLC0415
        GANDirectiveError,
        JobDirectiveError,
        append_directive,
        list_windows,
    )

    from_window = _current_control_pair8()
    if not from_window:
        return {"ok": False, "error": "JARVIS_PAIR_ID is required for bridged gan_send"}
    try:
        to_window = _resolve_control_window(args.get("to_window"))
        message = str(args.get("message") or "").strip()
        if not message:
            raise ValueError("message is required")
        try:
            issues_open = int(args.get("issues_open"))
        except (TypeError, ValueError) as exc:
            raise ValueError("issues_open must be an integer") from exc
        gan_id = str(args.get("gan_id") or "").strip()
        try:
            item = append_directive(
                kind="directive",
                from_window=from_window,
                to_window=to_window,
                body=message,
                gan_target=gan_id or "new",
                issues_open=issues_open,
            )
        except GANDirectiveError as exc:
            existing = (
                re.search(r"open gan (g_[0-9a-f]{8}) already exists", str(exc))
                if not gan_id
                else None
            )
            if not existing:
                raise
            item = append_directive(
                kind="directive",
                from_window=from_window,
                to_window=to_window,
                body=message,
                gan_target=existing.group(1),
                issues_open=issues_open,
            )
        return {"ok": True, "item": item, "windows": list_windows()}
    except (GANDirectiveError, JobDirectiveError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _submit_control_gan_close(args: dict[str, Any]) -> dict[str, Any]:
    """Sidecar-direct bridge for gan_close (Pattern S).

    Mirrors pi's regime-A gan_close execute (jarvis-jlc.ts), which only POSTs to
    /directives with a terminal report (gan_target/gan_status). The cosmetic
    appendSubturnLedger note pi adds is local-only (not the map ledger) and is
    deliberately not reproduced, exactly like job_close's cosmetic bits. The GAN
    report branch (directives._build_gan_record_locked) does NOT require
    to_window, so we pass None.
    """
    from jarvis_sidecar.directives import (  # noqa: PLC0415
        GANDirectiveError,
        JobDirectiveError,
        append_directive,
        list_windows,
    )

    from_window = _current_control_pair8()
    if not from_window:
        return {"ok": False, "error": "JARVIS_PAIR_ID is required for bridged gan_close"}
    try:
        gan_id = str(args.get("gan_id") or "").strip()
        status = str(args.get("status") or "").strip()
        summary = str(args.get("summary") or "").strip()
        if not gan_id or not summary or status not in {"agreed", "escalated"}:
            raise ValueError("gan_id, status agreed|escalated, and summary are required")
        item = append_directive(
            kind="report",
            from_window=from_window,
            to_window=None,
            body=summary,
            gan_target=gan_id,
            gan_status=status,
        )
        return {"ok": True, "item": item, "windows": list_windows()}
    except (GANDirectiveError, JobDirectiveError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _orchestration_result_payload(result: Any) -> dict[str, Any]:
    payload = asdict(result)
    state = payload.get("state")
    if hasattr(state, "value"):
        payload["state"] = state.value
    return payload


def _format_orchestrate_done(result: Any) -> str:
    state = getattr(result, "state", "")
    state_value = state.value if hasattr(state, "value") else str(state or "")
    summary = re.sub(r"\s+", " ", str(getattr(result, "summary", "") or "")).strip()
    if len(summary) > 1500:
        summary = f"{summary[:1497]}..."
    orchestration_id = str(getattr(result, "orchestration_id", "") or "")
    return (
        f"[ULTRACODE DONE {orchestration_id}] "
        f"state={state_value} "
        f"finders={getattr(result, 'finders_ran', 0)}/{getattr(result, 'finders_total', 0)} "
        f"stop_reason={getattr(result, 'stop_reason', None)}. "
        f"{summary} "
        f"Relay this synthesized result to the user. "
        f'Call ultracode_result("{orchestration_id}") for full findings if needed. '
        f"Do NOT start another orchestration in response to this report."
    )


def _submit_control_ultracode(args: dict[str, Any]) -> dict[str, Any]:
    from jarvis_sidecar.directives import append_directive  # noqa: PLC0415
    from jlc_agentic.agentic import orchestrate  # noqa: PLC0415

    from_window = _current_control_pair8()
    if not from_window:
        return {"ok": False, "error": "JARVIS_PAIR_ID is required for bridged ultracode"}
    try:
        task = str(args.get("task") or "").strip()
        if not task:
            raise ValueError("task is required")
        raw_dimensions = args.get("dimensions")
        if not isinstance(raw_dimensions, list):
            raise ValueError("dimensions must be an array of strings")
        dimensions = [str(dimension).strip() for dimension in raw_dimensions if str(dimension).strip()]
        if not dimensions:
            raise ValueError("dimensions is required")
        max_concurrency = _optional_int(args.get("max_concurrency"), "max_concurrency")
        max_calls = _optional_int(args.get("max_calls"), "max_calls")
        max_tokens = _optional_int(args.get("max_tokens"), "max_tokens")
        max_wallclock_sec = _optional_float(args.get("max_wallclock_sec"), "max_wallclock_sec")
        project_root = str(args.get("project_root") or "").strip() or None
        spec = orchestrate.OrchestrationSpec(
            task=task,
            dimensions=dimensions,
            max_concurrency=max_concurrency or 3,
            budget=orchestrate.OrchestrationBudget(
                max_calls=max_calls,
                max_tokens=max_tokens,
                max_wallclock_sec=max_wallclock_sec,
            ),
            project_root=project_root,
            conv_id=from_window,
        )

        def on_complete(result: Any) -> None:
            try:
                append_directive(
                    kind="directive",
                    from_window=from_window,
                    to_window=from_window,
                    body=_format_orchestrate_done(result),
                )
            except Exception:
                pass

        orchestration_id = orchestrate.start_orchestration_job(spec, on_complete=on_complete)
        return {
            "ok": True,
            "orchestration_id": orchestration_id,
            "message": (
                f"Ultracode {orchestration_id} started in the background; "
                "you will be re-engaged with the result when it completes. "
                "The turn will now end."
            ),
        }
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}


def _submit_control_ultracode_status(args: dict[str, Any]) -> dict[str, Any]:
    from jlc_agentic.agentic import orchestrate  # noqa: PLC0415

    orchestration_id = str(args.get("orchestration_id") or "").strip()
    if not orchestration_id:
        return {"ok": False, "error": "orchestration_id is required"}
    return {"ok": True, **orchestrate.get_orchestration_status(orchestration_id)}


def _submit_control_ultracode_result(args: dict[str, Any]) -> dict[str, Any]:
    from jlc_agentic.agentic import orchestrate  # noqa: PLC0415

    orchestration_id = str(args.get("orchestration_id") or "").strip()
    if not orchestration_id:
        return {"ok": False, "error": "orchestration_id is required"}
    result = orchestrate.get_orchestration_result(orchestration_id)
    if result is None:
        return {"ok": False, "error": "not ready or unknown id", "orchestration_id": orchestration_id}
    return {"ok": True, "result": _orchestration_result_payload(result)}


def _submit_control_ultracode_cancel(args: dict[str, Any]) -> dict[str, Any]:
    from jlc_agentic.agentic import orchestrate  # noqa: PLC0415

    orchestration_id = str(args.get("orchestration_id") or "").strip()
    if not orchestration_id:
        return {"ok": False, "error": "orchestration_id is required"}
    return {"ok": orchestrate.cancel(orchestration_id), "orchestration_id": orchestration_id}


def _submit_control_job_send(args: dict[str, Any], second_eyes_phase: str | None = None) -> dict[str, Any]:
    from jarvis_sidecar.directives import (  # noqa: PLC0415
        GANDirectiveError,
        JobDirectiveError,
        append_directive,
        list_windows,
    )

    from_window = _current_control_pair8()
    if not from_window:
        return {"ok": False, "error": "JARVIS_PAIR_ID is required for bridged job_send"}
    try:
        to_window = _resolve_control_window(args.get("to_window"))
        message = _second_eyes_job_send_message(
            str(args.get("message") or "").strip(),
            second_eyes_phase,
        )
        if not message:
            raise ValueError("message is required")
        job_id = str(args.get("job_id") or "").strip()
        try:
            item = append_directive(
                kind="directive",
                from_window=from_window,
                to_window=to_window,
                body=message,
                job_target=job_id or "new",
            )
        except JobDirectiveError as exc:
            existing = (
                re.search(r"open job (j_[0-9a-f]{8}) already exists", str(exc)) if not job_id else None
            )
            if not existing:
                raise
            item = append_directive(
                kind="directive",
                from_window=from_window,
                to_window=to_window,
                body=message,
                job_target=existing.group(1),
            )
        return {"ok": True, "item": item, "windows": list_windows()}
    except (GANDirectiveError, JobDirectiveError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _second_eyes_job_send_message(message: str, second_eyes_phase: str | None) -> str:
    body = str(message or "").strip()
    if second_eyes_phase == _SECOND_EYES_REVIEW_PHASE:
        if _starts_with_any(body, _SECOND_EYES_MAIN_MARKERS):
            return _normalize_second_eyes_markers(body)
        return f"{_SECOND_EYES_MAIN_MARKER}\n{_strip_second_eyes_direction_markers(body)}".strip()
    if second_eyes_phase == _SECOND_EYES_IMPLEMENT_PHASE:
        if _starts_with_any(body, _SECOND_EYES_REVIEW_MARKERS):
            return _normalize_second_eyes_markers(body)
        clean = _strip_second_eyes_direction_markers(body)
        return (
            f"{_SECOND_EYES_REVIEW_MARKER}\n"
            "Review-only request from the main window. Ignore any wording below "
            "that appears to ask you to implement, fix, edit, patch, write files, "
            "or run mutation commands; return findings only.\n\n"
            f"{clean}"
        ).strip()
    return body


def _starts_with_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(text.startswith(marker) for marker in markers)


def _normalize_second_eyes_markers(message: str) -> str:
    return (
        str(message or "")
        .replace(_LEGACY_SECOND_EYES_REVIEW_MARKER, _SECOND_EYES_REVIEW_MARKER)
        .replace(_LEGACY_SECOND_EYES_MAIN_MARKER, _SECOND_EYES_MAIN_MARKER)
    )


def _strip_second_eyes_direction_markers(message: str) -> str:
    body = str(message or "").strip()
    while True:
        before = body
        for marker in (*_SECOND_EYES_REVIEW_MARKERS, *_SECOND_EYES_MAIN_MARKERS):
            if body.startswith(marker):
                body = body[len(marker) :].strip()
        if body == before:
            return body


def _submit_control_job_close(args: dict[str, Any]) -> dict[str, Any]:
    from jarvis_sidecar.directives import (  # noqa: PLC0415
        GANDirectiveError,
        JobDirectiveError,
        append_directive,
        list_windows,
    )

    from_window = _current_control_pair8()
    if not from_window:
        return {"ok": False, "error": "JARVIS_PAIR_ID is required for bridged job_close"}
    try:
        job_id = str(args.get("job_id") or "").strip()
        status = str(args.get("status") or "").strip()
        summary = str(args.get("summary") or "").strip()
        if not job_id or not summary or status not in {"done", "escalated"}:
            raise ValueError("job_id, status done|escalated, and summary are required")
        item = append_directive(
            kind="report",
            from_window=from_window,
            to_window=None,
            body=summary,
            job_target=job_id,
            job_status=status,
        )
        return {"ok": True, "item": item, "windows": list_windows()}
    except (GANDirectiveError, JobDirectiveError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


def _submit_control_set_llm_model(role: str, args: dict[str, Any]) -> dict[str, Any]:
    if role not in {"chat", "encoder"}:
        return {"ok": False, "error": f"unsupported llm role: {role!r}"}
    from jarvis_sidecar.app import LLMSettingApplyRequest, llmsetting_apply, llmsetting_catalog  # noqa: PLC0415

    model = str(args.get("model") or "").strip()
    force = bool(args.get("force"))
    if not model:
        return llmsetting_catalog()
    if "/" not in model:
        return {"ok": False, "error": "model must be provider/model"}
    payload = {"force": force}
    if role == "chat":
        payload["chat"] = model
    else:
        payload["encoder"] = model
    try:
        request = LLMSettingApplyRequest(**payload)
        return llmsetting_apply(request)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _submit_control_set_window_label(args: dict[str, Any]) -> dict[str, Any]:
    from jarvis_sidecar.app import WindowLabelRequest, set_window_label  # noqa: PLC0415

    label = str(args.get("label") or "").strip()
    if not label:
        return {"ok": False, "error": "label is required"}
    try:
        request = WindowLabelRequest(label=label)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    try:
        return set_window_label(request)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _to_text(obj: Any) -> str:
    import json  # noqa: PLC0415

    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def _ultracode_result_text(result: Any) -> str:
    if isinstance(result, dict) and result.get("ok") and result.get("message"):
        return str(result["message"])
    return _to_text(result)


def _handoff_result_succeeded(is_error: bool, body: Any) -> bool:
    """Did a GAN/job handoff tool result succeed (i.e. is it turn-ending)?

    The bridged control tools return a JSON dict text -- a REJECTED handoff is
    ``{"ok": false, "error": ...}`` (e.g. the "gan and job cannot both be true"
    guard). A rejected handoff must NOT end the turn, so the model can recover and
    retry. ``is_error`` covers a tool that raised. A body we cannot parse defaults
    to success: the handoff side effect (the directive enqueue) already happened, so
    we do not keep the sender's turn alive on a parse gap (that gap is the overlap
    bug we are fixing)."""
    if is_error:
        return False
    text = _coerce_content(body).strip() if body is not None else ""
    if not text:
        return True
    import json  # noqa: PLC0415

    try:
        obj = json.loads(text)
    except Exception:
        return '"ok":false' not in text.replace(" ", "")
    if isinstance(obj, dict) and obj.get("ok") is False:
        return False
    return True


def _handoff_result_is_terminal(body: Any) -> bool:
    """Did a handoff FAIL because the target job/GAN is already terminal/closed?

    A rejected handoff is normally RECOVERABLE -- the model should fix the call and
    retry (e.g. "gan and job cannot both be true") -- so it does NOT end the turn.
    But a failure that means the channel itself is OVER ("job is already terminal",
    an already-closed/agreed GAN, an append rejected after close) has no valid
    retry: the model should STOP, not keep poking the dead job. Treat those as
    turn-ending so the sender does not livelock re-sending to a closed job -- the
    post-completion "both windows keep running" symptom."""
    text = _coerce_content(body).strip().lower() if body is not None else ""
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "already terminal",
            "already closed",
            "no longer active",
            "further append",
            "is terminal",
            "already agreed",
            "already escalated",
        )
    )


def _drive_async(make_agen):
    """Drive an async generator from a sync generator without a running loop.

    ``stream_chat_completions`` is sync and is called inside a worker thread
    (server/ws.py runs the turn via asyncio.to_thread → no running loop here).
    We run the async ``query()`` in a dedicated daemon thread with its own event
    loop, pushing each message onto a thread-safe queue that this generator drains.
    A stop Event lets us abandon the producer promptly if the consumer is closed
    (e.g. JLC cancels the turn and calls stream.close()).
    """
    q: queue.Queue[Any] = queue.Queue()
    stop_evt = threading.Event()
    err: dict[str, BaseException] = {}
    state: dict[str, Any] = {}
    ready = threading.Event()

    async def _consume() -> None:
        agen = make_agen()
        try:
            async for msg in agen:
                q.put(msg)
                if stop_evt.is_set():
                    break
        except asyncio.CancelledError:
            # Expected when the outer synchronous stream is closed early.
            pass
        except BaseException as exc:  # noqa: BLE001 - forwarded to consumer thread
            err["exc"] = exc
        finally:
            with contextlib.suppress(BaseException):
                await agen.aclose()
            q.put(_DONE)

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        task = loop.create_task(_consume())
        state["loop"] = loop
        state["task"] = task
        ready.set()
        try:
            loop.run_until_complete(task)
            loop.run_until_complete(loop.shutdown_asyncgens())
        except asyncio.CancelledError:
            pass
        except BaseException as exc:  # noqa: BLE001 - surfaced below
            err["exc"] = exc
            q.put(_DONE)
        finally:
            with contextlib.suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            with contextlib.suppress(Exception):
                loop.close()

    worker = threading.Thread(target=_runner, name="agent-sdk-bridge", daemon=True)
    worker.start()
    timed_out = False
    overall = max(0.0, _DRIVE_ASYNC_OVERALL_TIMEOUT_SEC)
    poll = max(0.05, _DRIVE_ASYNC_POLL_INTERVAL_SEC)
    try:
        # Heartbeat-style deadline: any item from the producer (thinking/tool
        # activity or content) resets the clock, so only a fully stalled producer
        # trips the guard. An overall<=0 disables the bound (back to blocking get).
        deadline = (time.monotonic() + overall) if overall > 0 else None
        while True:
            if deadline is None:
                item = q.get()
            else:
                try:
                    item = q.get(timeout=poll)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    continue
                deadline = time.monotonic() + overall
            if item is _DONE:
                break
            yield item
    finally:
        stop_evt.set()
        if ready.wait(timeout=0.5):
            loop = state.get("loop")
            task = state.get("task")
            if loop is not None and task is not None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(task.cancel)
        worker.join(timeout=5.0)
    if "exc" in err:
        raise err["exc"]
    if timed_out:
        raise TimeoutError(
            "anthropic-agent-sdk stream produced no output for "
            f"{overall:.0f}s; aborting the hung turn (see "
            "JARVIS_AGENT_SDK_STREAM_TIMEOUT_SEC). The producer was cancelled."
        )


@dataclass
class _AgentSDKStreamState:
    hook_tool_starts: set[str] = field(default_factory=set)
    hook_tool_finishes: set[str] = field(default_factory=set)
    # tool_use_id -> label (e.g. "shell", "write") so the matching '[done: ...]'
    # result line can name the tool it finished, even across the hook/_translate
    # dedup split (only one path fires per tool).
    tool_labels: dict[str, str] = field(default_factory=dict)
    # tool_use_id -> set, so a result body delivered through BOTH the UserMessage
    # path AND an echoed AssistantMessage tool_result is emitted once.
    result_finishes: set[str] = field(default_factory=set)
    # Track 1: result-bearing native tools (read/bash/edit/grep/glob/ls) render as a
    # STRUCTURED pre-resolved tool_call (regime-A widget) instead of a [done:] label.
    # pending_widget_tool_calls buffers the CALL (name+args) until the matching
    # result; widget_tool_ids marks them so any label path suppresses its label (no
    # double-render); tool_call_index gives each emitted tool_call delta a stable
    # stream index.
    pending_widget_tool_calls: dict[str, dict[str, str]] = field(default_factory=dict)
    widget_tool_ids: set[str] = field(default_factory=set)
    tool_call_index: int = 0
    # Track 2: streamable-input tools (write) open their tool_call at
    # content_block_start and stream the model-generated content INTO the card via
    # input_json_delta -> arg deltas (instead of an atomic card on completion).
    #   streamed_tool_call_index: tool_use_id -> the OpenAI tool_call stream index, so
    #     the open / arg deltas / result-attach all land on the SAME card. Survives
    #     across messages (open in AssistantMessage, result in a later UserMessage);
    #     popped on result-attach.
    #   partial_tool_ids: content_block index -> tool_use_id, so an input_json_delta on
    #     that block routes its fragment to the right tool_call. Per-message (cleared on
    #     message_start with the other block bookkeeping).
    #   streamed_widget_ids: ids that opened a streamed tool_call, so the whole-block
    #     ToolUseBlock path suppresses the atomic widget for them (no double-render).
    streamed_tool_call_index: dict[str, int] = field(default_factory=dict)
    partial_tool_ids: dict[int, str] = field(default_factory=dict)
    streamed_widget_ids: set[str] = field(default_factory=set)
    # --- partial-message (StreamEvent) bookkeeping (include_partial_messages) ---
    # When True, fine-grained content_block_delta events already streamed the
    # ThinkingBlock / preamble TextBlock token-by-token, so the whole-block paths
    # in the AssistantMessage branch must NOT re-emit the same text (avoid dupes).
    partial_active: bool = False
    # content_block index -> block kind ('thinking'|'text'|'tool_use') started by a
    # content_block_start event in the CURRENT assistant turn, so deltas can be
    # routed and the matching block can be suppressed when the AssistantMessage
    # arrives. Reset on message_start.
    partial_block_kinds: dict[int, str] = field(default_factory=dict)
    # content_block index -> the tool name of a streaming tool_use block, so an
    # input_json_delta on that index can be labeled (e.g. '[ask user: ...]').
    partial_tool_names: dict[int, str] = field(default_factory=dict)
    # True once ANY thinking/text/tool delta streamed in this assistant turn (used
    # to decide whether the whole-block AssistantMessage echo is a duplicate).
    partial_streamed_thinking: bool = False
    partial_streamed_text: bool = False
    partial_streamed_answer_text: bool = False
    partial_answer_text_parts: list[str] = field(default_factory=list)
    # --- live-stream cumulative byte budgets (DNA #1: every output path has a HARD
    # ceiling). Each delta path accumulates the REAL UTF-8 bytes it has forwarded so
    # far this turn and STOPS once its budget is reached. Without these, a long
    # thinking block / gate preamble / multi-fragment tool input would stream without
    # bound (each fragment was only re-bounded individually, never accumulated).
    # Reset on message_start.
    partial_thinking_bytes: int = 0  # cumulative thinking_delta bytes forwarded
    partial_text_bytes: int = 0  # cumulative gate-turn text_delta bytes forwarded
    # content_block index -> cumulative input_json_delta bytes forwarded for that
    # tool_use block. Reset on message_start AND on each content_block_start.
    partial_input_bytes: dict[int, int] = field(default_factory=dict)
    # Indices/flags that have already emitted their one-time '...' truncation marker,
    # so the marker is emitted exactly once per capped stream (no marker spam).
    partial_thinking_capped: bool = False
    partial_text_capped: bool = False
    partial_input_capped: set[int] = field(default_factory=set)
    # --- regime-B observed-work SENSOR (item 1, 2026-06-25) --------------------
    # The Agent SDK runs the agentic loop and executes tools internally, so pi's
    # tool_execution_end never fires in regime B and turnSuccessfulFileMutations
    # stays empty -> all post-turn memory orchestration silently no-ops. We tap the
    # SAME ToolUseBlock/ToolResultBlock the streaming labels already iterate and
    # accumulate a thin, normalized per-tool activity record. It rides out on the
    # FINAL chunk as `jarvis_tool_activity`; pi (regime B only) reconstructs
    # turnSuccessfulFileMutations + toolEvents from it and feeds its existing
    # single pipeline. We DUPLICATE no orchestration here -- only PRODUCE the signal.
    #   tool_calls: tool_use_id -> partial record {tool, tool_use_id, abs_path,
    #     mutation_kind, command} captured at the ToolUseBlock; finalized
    #     (success/result_preview) and appended to tool_activity when its
    #     ToolResultBlock arrives. Keyed by id so call/result correlate.
    tool_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    #   tool_activity: completed per-tool records = the trailer carried on the
    #     finish chunk. Bounded to _TOOL_ACTIVITY_MAX_RECORDS (DNA #1).
    tool_activity: list[dict[str, Any]] = field(default_factory=list)


def _translate(
    msg: Any,
    block_types: dict,
    msg_types: dict,
    *,
    text_buffer: list[str] | None = None,
    stream_state: _AgentSDKStreamState | None = None,
    stream_preamble: bool = False,
    stream_answer_text: bool = False,
    suppress_widget_tool_calls: bool = False,
) -> Iterator[tuple[dict, bool]]:
    """Yield (openai_chunk, is_content) tuples for one SDK message.

    stream_preamble: when True (ask_user / new-artifact gate turn), assistant
    TextBlock prose is ALSO streamed live as reasoning_content -- so the model's
    "I have a few questions.." preamble is visible before the modal pops -- while
    still being buffered for the final answer (Part C). Buffering-until-Result
    semantics are unchanged; this only ADDS a live mirror.

    Message types handled (all source-verified against claude_agent_sdk 0.2.101):
      - StreamEvent (include_partial_messages=True): RAW Anthropic SSE deltas ->
        fine-grained reasoning_content so pi animates thinking / ask_user questions
        token-by-token (the "돌아가는 컨텐츠"). Whole-block paths below stay as a
        fallback and self-suppress when partials already streamed the same text.
      - AssistantMessage: TextBlock (buffer + Part C preamble mirror), ThinkingBlock,
        ToolUseBlock activity label (Part A), and the ECHOED tool_result block.
      - UserMessage: the REAL tool-result delivery path -- a ToolResultBlock arrives
        inside UserMessage.content (message_parser.py:81-115), NOT an AssistantMessage.
        This is the #1 Part B fix: emit '[done: <tool>]' + bounded body here.
      - HookEventMessage / ResultMessage: unchanged.
    """
    state = stream_state or _AgentSDKStreamState()
    stream_event_t = msg_types.get("stream_event")
    if stream_event_t is not None and isinstance(msg, stream_event_t):
        yield from _translate_stream_event(
            msg,
            state,
            stream_preamble=stream_preamble,
            stream_answer_text=stream_answer_text,
            suppress_widget_tool_calls=suppress_widget_tool_calls,
        )
        return
    if isinstance(msg, msg_types.get("hook_event", ())):
        label = _hook_event_activity_label(msg, state)
        if label:
            yield _reasoning_chunk(f"\n{label}\n"), False
        return
    if isinstance(msg, msg_types["assistant"]):
        # Surface a hard provider error (auth/billing/rate-limit) loudly.
        err = getattr(msg, "error", None)
        if err:
            raise RuntimeError(f"Agent SDK assistant error: {err}")
        for block in getattr(msg, "content", None) or []:
            if isinstance(block, block_types["text"]):
                text = getattr(block, "text", "") or ""
                if text:
                    if text_buffer is not None:
                        text_buffer.append(text)
                        # Part C: on an ask_user / new-artifact gate turn, mirror the
                        # assistant preamble live as reasoning so it is visible before
                        # the modal pops -- the buffer still feeds the final answer.
                        # If partial text_delta already streamed this preamble live,
                        # skip the whole-block echo (avoid a double render) but KEEP
                        # the buffer append above so the final answer is intact.
                        if stream_preamble and not state.partial_streamed_text:
                            preamble = _trim_multiline_hint(
                                text, _PREAMBLE_LIMIT, max_lines=_PREAMBLE_MAX_LINES
                            )
                            if preamble:
                                yield _reasoning_chunk(f"{preamble}\n"), False
                    else:
                        yield _content_chunk(text), True
            elif isinstance(block, block_types["thinking"]):
                thinking = getattr(block, "thinking", "") or ""
                # If thinking already streamed token-by-token via thinking_delta,
                # the whole-block echo is a duplicate -- skip it.
                if thinking and not state.partial_streamed_thinking:
                    yield _reasoning_chunk(thinking), False
            elif isinstance(block, block_types["tool_use"]):
                name = getattr(block, "name", "?")
                tool_id = str(getattr(block, "id", "") or "")
                if tool_id:
                    state.tool_labels[tool_id] = _agent_tool_short_label(name)
                raw_input = getattr(block, "input", None)
                # Regime-B sensor: capture the tool CALL (name + abs path) BEFORE the
                # hook-dedup early-continue, so a hook-pathed tool still gets sensed.
                _sensor_record_tool_use(state, name, tool_id, raw_input)
                if tool_id and tool_id in state.hook_tool_starts:
                    continue
                # Track 1: result-bearing native tools render as a structured
                # pre-resolved tool_call (regime-A widget). Buffer the CALL now; the
                # structured chunk is emitted on its result (echo/UserMessage path).
                # Suppress the call label so the tool is not double-rendered.
                if tool_id and _is_widget_tool(name):
                    if suppress_widget_tool_calls:
                        yield (
                            _reasoning_chunk(
                                f"\n{_agent_tool_activity_label(name, raw_input)}\n"
                            ),
                            False,
                        )
                        continue
                    # Track 2: a write whose input already streamed live (content_block
                    # _start opened the tool_call) attaches its result via the streamed
                    # path -- do NOT also buffer an atomic widget here (double-render).
                    if tool_id in state.streamed_widget_ids:
                        continue
                    state.widget_tool_ids.add(tool_id)
                    state.pending_widget_tool_calls[tool_id] = {
                        "name": _widget_tool_display_name(name),
                        "arguments": _tool_args_json(_normalize_widget_tool_args(name, raw_input)),
                    }
                    continue
                yield _reasoning_chunk(f"\n{_agent_tool_activity_label(name, raw_input)}\n"), False
            elif isinstance(block, block_types["tool_result"]):
                # Echoed assistant-side tool_result (message_parser.py:149-156). Real
                # tool execution does NOT use this path -- it arrives via UserMessage
                # (handled below) -- but keep it for the echo case, deduped.
                tool_id = str(getattr(block, "tool_use_id", "") or "")
                body = getattr(block, "content", None)
                # Regime-B sensor finalize (idempotent via pop) -- before the dedup
                # early-continue so an echoed result still finalizes the record once.
                _sensor_finalize_tool_result(
                    state, tool_id, getattr(block, "is_error", False), body
                )
                # Track 2: a streamed write (input arrived live) attaches its result to
                # the already-open tool_call instead of a fresh atomic widget chunk.
                if not suppress_widget_tool_calls:
                    stream_chunk = _maybe_streamed_result_chunk(
                        state, tool_id, body, getattr(block, "is_error", False)
                    )
                    if stream_chunk is not None:
                        yield stream_chunk, False
                        continue
                # Track 1: a buffered widget tool emits its structured pre-resolved
                # tool_call here (call + result) instead of a [done:] label -- BEFORE
                # the label dedup, so the widget renders exactly once regardless of
                # the hook/echo dedup state.
                if not suppress_widget_tool_calls:
                    widget_chunk = _maybe_pre_resolved_widget_chunk(
                        state, tool_id, body, getattr(block, "is_error", False)
                    )
                    if widget_chunk is not None:
                        yield widget_chunk, False
                        continue
                if tool_id and (
                    tool_id in state.hook_tool_finishes
                    or tool_id in state.result_finishes
                ):
                    continue
                if tool_id:
                    state.result_finishes.add(tool_id)
                tool_label = state.tool_labels.get(tool_id) if tool_id else None
                yield _reasoning_chunk(_tool_done_label(tool_label, body)), False
        # An assistant message closes the current partial turn's block bookkeeping.
        state.partial_block_kinds.clear()
        state.partial_tool_names.clear()
        state.partial_streamed_thinking = False
        state.partial_streamed_text = False
        return
    user_t = msg_types.get("user")
    if user_t is not None and isinstance(msg, user_t):
        # PART B #1 FIX: the REAL tool-result delivery path. A finished tool's
        # stdout/result arrives as a ToolResultBlock inside UserMessage.content
        # (message_parser.py:102-115). The old loop dropped UserMessage entirely,
        # so '[done: shell]' never fired live. Emit it here.
        content = getattr(msg, "content", None)
        if isinstance(content, (list, tuple)):
            tool_result_t = block_types.get("tool_result")
            for block in content:
                if tool_result_t is None or not isinstance(block, tool_result_t):
                    continue
                tool_id = str(getattr(block, "tool_use_id", "") or "")
                body = getattr(block, "content", None)
                if body in (None, "", [], {}):
                    # Fall back to the parallel raw copy on the UserMessage when the
                    # block content is empty (types.py:1021, parser 84/114).
                    body = getattr(msg, "tool_use_result", None)
                # Regime-B sensor finalize (idempotent via pop) -- this UserMessage
                # path is the REAL result delivery (ground truth), so it is where the
                # sensor record is normally completed. Placed before the dedup
                # early-continue so it still finalizes when the live label was already
                # emitted by the hook/echo path.
                _sensor_finalize_tool_result(
                    state, tool_id, getattr(block, "is_error", False), body
                )
                # Track 2: a streamed write (input arrived live) attaches its result to
                # the already-open tool_call instead of a fresh atomic widget chunk --
                # the REAL result path, BEFORE the dedup.
                if not suppress_widget_tool_calls:
                    stream_chunk = _maybe_streamed_result_chunk(
                        state, tool_id, body, getattr(block, "is_error", False)
                    )
                    if stream_chunk is not None:
                        yield stream_chunk, False
                        continue
                # Track 1: a buffered widget tool emits its structured pre-resolved
                # tool_call here (call + result) instead of a [done:] label -- the
                # REAL result path, BEFORE the dedup, so the widget renders once.
                if not suppress_widget_tool_calls:
                    widget_chunk = _maybe_pre_resolved_widget_chunk(
                        state, tool_id, body, getattr(block, "is_error", False)
                    )
                    if widget_chunk is not None:
                        yield widget_chunk, False
                        continue
                if tool_id and (
                    tool_id in state.hook_tool_finishes
                    or tool_id in state.result_finishes
                ):
                    continue
                if tool_id:
                    state.result_finishes.add(tool_id)
                tool_label = state.tool_labels.get(tool_id) if tool_id else None
                yield _reasoning_chunk(_tool_done_label(tool_label, body)), False
        return
    if isinstance(msg, msg_types["result"]):
        if getattr(msg, "is_error", False):
            detail = getattr(msg, "errors", None) or getattr(msg, "subtype", "error")
            status = getattr(msg, "api_error_status", None)
            raise RuntimeError(f"Agent SDK result error (status={status}): {detail}")
        return  # finish/usage handled by the caller
    # SystemMessage / RateLimitEvent — nothing to emit. (StreamEvent + UserMessage
    # are handled above when their msg_types entries are registered.)
    return


# Anthropic SSE delta event types carried verbatim in StreamEvent.event (the SDK is
# pass-through: types.py:1231 "raw Anthropic API stream event"). Verified there are
# no delta-name string literals in the installed package, so we read them here.
_DELTA_TYPE_THINKING = "thinking_delta"
_DELTA_TYPE_TEXT = "text_delta"
_DELTA_TYPE_INPUT_JSON = "input_json_delta"
# TOTAL (cumulative, per content_block) UTF-8 byte budget for streamed tool-input
# fragments (ask_user questions stream char-by-char as input_json_delta). Once the
# running sum of forwarded fragment bytes for a block reaches this, we STOP
# forwarding that block and emit one '...' marker. This is a cumulative ceiling, NOT
# a per-fragment cap -- the bounded whole-block label still lands later.
_PARTIAL_TOOL_INPUT_LIVE_TOTAL_LIMIT = 600
# TOTAL (cumulative, per assistant turn) UTF-8 byte budget for streamed thinking_delta
# tokens. A summarized thinking block can be long; cap the live forward so it cannot
# flood the panel. The whole-block ThinkingBlock echo self-suppresses once any
# thinking streamed, so the live stream owns its own ceiling.
_STREAM_THINKING_LIMIT = 2000
# One-time marker emitted when a live delta stream hits its cumulative budget.
_STREAM_CAP_MARKER = "..."


@dataclass
class _StreamEmit:
    """Result of bounding one live-stream piece against a cumulative budget.

    text: the (possibly empty) string to forward now. When the budget is hit on
        this piece, ``text`` is the byte-bounded remainder of the budget plus a
        one-time ``...`` marker; when the budget was already hit, ``text`` is "".
    consumed: REAL UTF-8 bytes of forwarded CONTENT (excluding the marker) to add
        to the running total, so the accumulator measures true wire size.
    hit_cap: True iff this piece reached/exceeded the budget AND emitted the
        one-time marker -- the caller flips its ``*_capped`` flag so later pieces
        forward nothing and the marker is never repeated.
    """

    text: str
    consumed: int
    hit_cap: bool


def _bounded_stream_piece(
    piece: str, already: int, budget: int, *, capped: bool
) -> _StreamEmit:
    """Bound a single live-stream delta piece against a cumulative byte budget.

    DNA #1: the live delta path owns a HARD ceiling. ``already`` is the real UTF-8
    bytes forwarded for this stream so far; ``budget`` is the total ceiling. We
    measure the piece in bytes (not code points, so Korean/CJK stays within budget
    instead of ~3x over) and forward at most ``budget - already`` content bytes,
    slicing on a UTF-8 boundary. Once the budget is reached we append ONE ``...``
    marker and signal ``hit_cap`` so the caller stops forwarding and never repeats
    the marker.
    """
    if capped or budget <= 0:
        return _StreamEmit("", 0, False)
    remaining = budget - already
    if remaining <= 0:
        # Budget already consumed but marker not yet emitted (defensive): cap now.
        return _StreamEmit(_STREAM_CAP_MARKER, 0, True)
    raw = piece.encode("utf-8")
    if len(raw) <= remaining:
        # Fits whole: forward as-is; no marker, budget not yet hit.
        return _StreamEmit(piece, len(raw), False)
    # Overflow: forward the remainder of the budget (byte-bounded) + one marker.
    head = raw[:remaining].decode("utf-8", "ignore")
    return _StreamEmit(head + _STREAM_CAP_MARKER, len(head.encode("utf-8")), True)


def _translate_stream_event(
    msg: Any,
    state: _AgentSDKStreamState,
    *,
    stream_preamble: bool,
    stream_answer_text: bool = False,
    suppress_widget_tool_calls: bool = False,
) -> Iterator[tuple[dict, bool]]:
    """Forward RAW Anthropic SSE deltas as fine-grained reasoning_content.

    Only active when ClaudeAgentOptions.include_partial_messages=True. ``msg.event``
    is the raw API stream event dict (StreamEvent.event, types.py:1231). We surface:
      - thinking_delta -> reasoning_content (live thinking)
      - input_json_delta on an ask_user tool_use block -> reasoning_content (the
        actual questions, typed out -- Part C live form)
      - text_delta -> reasoning_content ONLY on a gate/ask_user turn (Part C preamble
        mirror); off-gate the answer text is the FINAL ANSWER and must stay buffered
        so prose does not jump above the tool timeline.
      - text_delta -> content when the subagent turn_context asks for live answer
        text. This keeps delegate_subagent cards alive during long Agent SDK answer
        blocks; chat does not set this flag and keeps the existing buffered order.
    These set state.partial_streamed_* so the AssistantMessage whole-block echo
    self-suppresses (no double render). Result bodies are NOT in the SSE delta
    stream -- they arrive only as a completed UserMessage/ToolResultBlock.
    """
    event = getattr(msg, "event", None)
    if not isinstance(event, dict):
        return
    etype = event.get("type")
    if etype == "message_start":
        # New assistant turn: reset per-turn partial bookkeeping.
        state.partial_active = True
        state.partial_block_kinds.clear()
        state.partial_tool_names.clear()
        # partial_tool_ids is per-message (block indices reset per turn); the
        # streamed_tool_call_index / streamed_widget_ids it feeds PERSIST across
        # messages (open here, result attaches in a later UserMessage), so only this
        # block->id map is cleared on message_start.
        state.partial_tool_ids.clear()
        state.partial_streamed_thinking = False
        state.partial_streamed_text = False
        # Reset cumulative live-stream byte budgets (DNA #1 ceilings) for the turn.
        state.partial_thinking_bytes = 0
        state.partial_text_bytes = 0
        state.partial_input_bytes.clear()
        state.partial_thinking_capped = False
        state.partial_text_capped = False
        state.partial_input_capped.clear()
        return
    if etype == "content_block_start":
        state.partial_active = True
        index = event.get("index")
        block = event.get("content_block")
        if not isinstance(block, dict) or index is None:
            return
        btype = block.get("type")
        if btype == "thinking":
            state.partial_block_kinds[index] = "thinking"
        elif btype == "text":
            state.partial_block_kinds[index] = "text"
        elif btype == "tool_use":
            state.partial_block_kinds[index] = "tool_use"
            name = block.get("name")
            normalized = _normalize_tool_schema_name(name) if isinstance(name, str) else None
            if isinstance(name, str):
                state.partial_tool_names[index] = normalized or name
            # Track 2: a streamable-input tool (write) OPENS its tool_call now so pi
            # shows the card immediately; the content fills in via input_json_delta
            # below, and the whole-block ToolUseBlock path then suppresses the atomic
            # widget for this id (no double-render).
            tool_id = str(block.get("id") or "")
            if (
                not suppress_widget_tool_calls
                and tool_id
                and normalized in _INPUT_STREAM_TOOL_NAMES
            ):
                oai_index = state.tool_call_index
                state.tool_call_index += 1
                state.streamed_tool_call_index[tool_id] = oai_index
                state.partial_tool_ids[index] = tool_id
                state.streamed_widget_ids.add(tool_id)
                yield _tool_call_open_chunk(oai_index, tool_id, normalized or "write"), False
        # Fresh per-block input budget: a new block at this index starts at 0 bytes.
        state.partial_input_bytes[index] = 0
        state.partial_input_capped.discard(index)
        return
    if etype == "content_block_delta":
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return
        dtype = delta.get("type")
        index = event.get("index")
        if dtype == _DELTA_TYPE_THINKING:
            piece = delta.get("thinking") or ""
            if piece:
                # Mark streamed even when over budget so the whole-block ThinkingBlock
                # echo stays suppressed (no double render of the already-shown head).
                state.partial_streamed_thinking = True
                emit = _bounded_stream_piece(
                    piece,
                    state.partial_thinking_bytes,
                    _STREAM_THINKING_LIMIT,
                    capped=state.partial_thinking_capped,
                )
                if emit.text:
                    state.partial_thinking_bytes += emit.consumed
                    if emit.hit_cap:
                        state.partial_thinking_capped = True
                    yield _reasoning_chunk(emit.text), False
            return
        if dtype == _DELTA_TYPE_TEXT:
            piece = delta.get("text") or ""
            if piece and stream_answer_text:
                state.partial_streamed_answer_text = True
                state.partial_answer_text_parts.append(piece)
                yield _content_chunk(piece), True
                return
            # Only mirror text live on a gate/ask_user turn (Part C). Off-gate this
            # is the final answer -- keep it buffered so it lands AFTER the tool
            # timeline (whole-block path owns it).
            if piece and stream_preamble:
                # Mark streamed even when over budget so the bounded whole-block
                # preamble echo stays suppressed (it would otherwise re-render the
                # head we already streamed).
                state.partial_streamed_text = True
                emit = _bounded_stream_piece(
                    piece,
                    state.partial_text_bytes,
                    _PREAMBLE_LIMIT,
                    capped=state.partial_text_capped,
                )
                if emit.text:
                    state.partial_text_bytes += emit.consumed
                    if emit.hit_cap:
                        state.partial_text_capped = True
                    yield _reasoning_chunk(emit.text), False
            return
        if dtype == _DELTA_TYPE_INPUT_JSON:
            # Track 2: stream a write's model-generated content INTO its open tool_call
            # so the card fills in char-by-char (the input IS the file body). No byte
            # budget here -- the card bounds the DISPLAY itself (truncate + ctrl+o to
            # expand), and pi does NOT re-send these display args to the model (regime
            # B: the SDK owns the loop and the turn terminates on the pre-resolved
            # result), so there is no token cost to forwarding the whole content.
            stream_tool_id = state.partial_tool_ids.get(index)
            if stream_tool_id is not None and stream_tool_id in state.streamed_tool_call_index:
                frag = delta.get("partial_json") or ""
                if frag:
                    yield (
                        _tool_call_arg_delta_chunk(
                            state.streamed_tool_call_index[stream_tool_id], frag
                        ),
                        False,
                    )
                return
            # ask_user input is NOT streamed as reasoning anymore: the bridged modal
            # (runAskUserDialog) is the real interactive surface and shows the
            # questions + options, so dumping the raw questions JSON here was just
            # redundant blue noise. (The pre-modal era streamed it as a fallback
            # surface; with the modal live, suppress it.)
            return
        return
    if etype in ("content_block_stop", "message_delta", "message_stop"):
        return
    return


def _debug_log_raw_message(msg: Any) -> None:
    """Env-gated raw-shape logger (zero overhead when unset).

    Set JARVIS_AGENT_SDK_DEBUG=1 to append one JSON line per SDK message to
    sidecar/logs/agent_sdk_raw_shapes.jsonl, capturing the real message/block
    types + salient attrs. This is the ground-truth net for the next live run if
    anything is still off (the prior round's miss came from guessing shapes). Off
    by default; the env check short-circuits before any work.
    """
    if not os.environ.get("JARVIS_AGENT_SDK_DEBUG"):
        return
    try:
        import json  # noqa: PLC0415

        rec: dict[str, Any] = {
            "ts": time.time(),
            # pid = window (separate pi process); tid = one SDK query (its bridge
            # thread). Two tids alive in one pid => that window is running two turns
            # at once (the residual concurrency we are attributing).
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "msg_type": type(msg).__name__,
        }
        event = getattr(msg, "event", None)
        if isinstance(event, dict):
            rec["event_type"] = event.get("type")
            delta = event.get("delta")
            if isinstance(delta, dict):
                rec["delta_type"] = delta.get("type")
            rec["block_index"] = event.get("index")
        subtype = getattr(msg, "subtype", None)
        if subtype is not None:
            rec["subtype"] = subtype
        hook_event = getattr(msg, "hook_event_name", None)
        if hook_event is not None:
            rec["hook_event_name"] = hook_event
        tool_use_result = getattr(msg, "tool_use_result", None)
        if tool_use_result is not None:
            rec["has_tool_use_result"] = True
        content = getattr(msg, "content", None)
        if isinstance(content, (list, tuple)):
            blocks: list[dict[str, Any]] = []
            for block in content:
                brec: dict[str, Any] = {"block_type": type(block).__name__}
                for attr in ("name", "id", "tool_use_id", "is_error"):
                    val = getattr(block, attr, None)
                    if val is not None:
                        brec[attr] = val if isinstance(val, (str, bool)) else str(val)
                tval = getattr(block, "text", None)
                if isinstance(tval, str):
                    brec["text_len"] = len(tval)
                ival = getattr(block, "input", None)
                if isinstance(ival, dict):
                    brec["input_keys"] = sorted(ival.keys())
                cval = getattr(block, "content", None)
                if cval is not None:
                    brec["content_kind"] = type(cval).__name__
                    # DEBUG: short preview of tool-result bodies so a live run shows
                    # handoff ok:true/false (the turn-taking interlock turns on this)
                    # without dumping full payloads.
                    try:
                        preview = _coerce_content(cval).strip()
                    except Exception:  # pragma: no cover - debug net must never break
                        preview = ""
                    if preview:
                        brec["content_preview"] = preview[:240]
                blocks.append(brec)
            rec["blocks"] = blocks
        elif isinstance(content, str):
            rec["content_kind"] = "str"
            rec["content_len"] = len(content)
        # sidecar root = providers/ -> jlc_agentic/ -> sidecar/
        sidecar_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        log_dir = os.path.join(sidecar_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "agent_sdk_raw_shapes.jsonl")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # pragma: no cover - debug net must never break the turn
        return


def _debug_note(event: str, **fields: Any) -> None:
    """Env-gated structured marker into the same raw-shapes log (zero overhead off).

    Used to prove control-flow decisions (e.g. the turn-taking force-end actually
    firing) during a live run, instead of inferring them from message shapes.
    """
    if not os.environ.get("JARVIS_AGENT_SDK_DEBUG"):
        return
    try:
        import json  # noqa: PLC0415

        rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "tid": threading.get_ident(),
            "msg_type": "_note",
            "event": event,
            **fields,
        }
        sidecar_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        log_dir = os.path.join(sidecar_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "agent_sdk_raw_shapes.jsonl")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:  # pragma: no cover - debug net must never break the turn
        return


def _final_text_from_result_or_buffer(result_msg: Any, text_buffer: list[str]) -> str:
    final_text = getattr(result_msg, "result", None)
    if final_text:
        return _user_facing_brand_text(str(final_text))
    return _user_facing_brand_text("".join(text_buffer))


def _user_facing_brand_text(text: str) -> str:
    return (
        text.replace("Second Eyes", "Critic Mode")
        .replace("second eyes", "critic mode")
        .replace("SECOND EYES", "CRITIC MODE")
        .replace(_LEGACY_SECOND_EYES_REVIEW_MARKER, _SECOND_EYES_REVIEW_MARKER)
        .replace(_LEGACY_SECOND_EYES_MAIN_MARKER, _SECOND_EYES_MAIN_MARKER)
    )


_AGENT_TOOL_LABELS = {
    "ask_user": "ask user",
    "bash": "shell",
    "edit": "edit",
    "glob": "glob",
    "grep": "grep",
    "ls": "list",
    "multi_edit": "edit",
    "notebook_edit": "notebook edit",
    "read": "read",
    "spawn_window": "spawn window",
    "list_windows": "list windows",
    "send_directive": "send directive",
    "job_send": "job send",
    "job_close": "job close",
    "set_chat_model": "set chat model",
    "set_encoder_model": "set encoder model",
    "set_window_label": "set window label",
    "task": "agent task",
    "todo_write": "todo",
    "web_fetch": "web fetch",
    "web_search": "web search",
    "write": "write",
}

_AGENT_TOOL_HINT_KEYS = {
    "bash": ("command",),
    "edit": ("file_path", "path"),
    "glob": ("pattern", "path"),
    "grep": ("pattern", "query", "path"),
    "ls": ("path",),
    "multi_edit": ("file_path", "path"),
    "notebook_edit": ("notebook_path", "path"),
    "read": ("file_path", "path"),
    "task": ("description",),
    "web_fetch": ("url",),
    "web_search": ("query",),
    "write": ("file_path", "path"),
}

# Tools whose preview is a composite (path + content / old->new diff) rendered by a
# per-tool formatter rather than the single-scalar _trim_tool_hint path. Adding the
# content keys to _AGENT_TOOL_HINT_KEYS would not help: _agent_tool_input_hint returns
# only the FIRST non-None key, so composite rendering needs dedicated formatters.
_COMPOSITE_HINT_TOOLS = frozenset({"bash", "write", "edit", "multi_edit"})

_FALLBACK_TOOL_HINT_KEYS = ("query", "url", "file_path", "path", "command", "pattern")
_JARVIS_CONTROL_TOOL_PREFIX = "mcp__jarvis_control__"
_TOOL_HINT_LIMIT = 96

# Multi-line preview bounds (CLAUDE.md DNA #1 "bound everything by size"). Every
# preview MUST be byte+line bounded before it reaches _reasoning_chunk so a verbose
# write/grep body cannot flood the pi thinking panel and shove out the final answer.
_BASH_HINT_LIMIT = 400  # shell command preview
_WRITE_BODY_LIMIT = 400  # write content preview (after the file_path + line-count header)
_WRITE_BODY_MAX_LINES = 10
_EDIT_SIDE_LIMIT = 120  # each of old_string / new_string in an edit summary
# Tool-result body preview. Large bounded (CLAUDE.md DNA #1: never unbounded, but
# big enough to stream a real file body live instead of an 8-line stub). Path A:
# the body rides reasoning_content (thinking panel), so a generous bound here +
# pi keeping the body lines in the collapsed view = the file content streams
# without polluting assistant content / session history / memory.
_RESULT_BODY_LIMIT = 16384  # ~16 KB tool-result body preview
_RESULT_BODY_MAX_LINES = 400
_PREAMBLE_LIMIT = 600  # ask_user / gate-turn preamble live mirror (Part C)
_PREAMBLE_MAX_LINES = 8
# Keys carrying a result body inside a hook PostToolUse `data` dict (symmetric with
# _hook_tool_input, which reads the INPUT keys). First dict/str/list wins.
_HOOK_RESULT_KEYS = (
    "tool_response",
    "toolResult",
    "tool_result",
    "result",
    "output",
    "content",
)


def _hook_event_activity_label(msg: Any, state: _AgentSDKStreamState) -> str | None:
    event_name = str(getattr(msg, "hook_event_name", "") or "").strip()
    subtype = str(getattr(msg, "subtype", "") or "").strip()
    data = getattr(msg, "data", None)
    if not isinstance(data, dict):
        data = {}
    if event_name == "PreToolUse" and subtype == "hook_started":
        tool_name = _hook_tool_name(data)
        if not tool_name:
            return None
        tool_id = _hook_tool_use_id(data)
        if tool_id:
            state.hook_tool_starts.add(tool_id)
            state.tool_labels[tool_id] = _agent_tool_short_label(tool_name)
        # Track 1: a widget tool is rendered as a structured tool_call, so suppress
        # its hook call label too (no double-render). Mark the id so the matching
        # PostToolUse suppresses its done label as well.
        if _is_widget_tool(tool_name):
            if tool_id:
                state.widget_tool_ids.add(tool_id)
            return None
        return _agent_tool_activity_label(tool_name, _hook_tool_input(data))
    if event_name == "PostToolUse" and subtype == "hook_started":
        tool_id = _hook_tool_use_id(data)
        tool_label = None
        hook_result = _hook_tool_result(data)
        if tool_id:
            state.hook_tool_finishes.add(tool_id)
            tool_label = state.tool_labels.get(tool_id)
            # Regime-B sensor: when the result is delivered through the hook path
            # (not the UserMessage/echo block), finalize the record here too.
            _sensor_finalize_tool_result(state, tool_id, False, hook_result)
        # Track 1: widget tools render as a structured tool_call -> suppress the
        # hook done label (no double-render).
        if tool_id and tool_id in state.widget_tool_ids:
            return None
        return _tool_done_label(tool_label, hook_result).rstrip("\n")
    if event_name == "PostToolUseFailure" and subtype == "hook_started":
        tool_name = _hook_tool_name(data)
        tool_id = _hook_tool_use_id(data)
        if tool_id:
            state.hook_tool_finishes.add(tool_id)
            _sensor_finalize_tool_result(state, tool_id, True, _hook_tool_result(data))
        # Track 1: widget tools render as a structured tool_call (jarvis_is_error
        # carries the failure) -> suppress the hook failed label (no double-render).
        if _is_widget_tool(tool_name) or (tool_id and tool_id in state.widget_tool_ids):
            return None
        if tool_name:
            return f"[failed: {_AGENT_TOOL_LABELS.get(tool_name, tool_name)}]"
        return "[failed]"
    return None


def _hook_tool_name(data: dict[str, Any]) -> str:
    for key in ("tool_name", "toolName", "tool"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_tool_schema_name(value)
    return ""


def _hook_tool_input(data: dict[str, Any]) -> Any:
    for key in ("tool_input", "toolInput", "input"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


def _hook_tool_result(data: dict[str, Any]) -> Any:
    """Read the tool RESULT body from hook PostToolUse data (symmetric with
    _hook_tool_input). Returns the first non-empty str/dict/list value."""
    for key in _HOOK_RESULT_KEYS:
        value = data.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _tool_done_label(tool_label: str | None, body: Any) -> str:
    """Render the result line for a finished tool (Part B).

    Leads with a recognized '[done...]' bracket token (so it persists in pi's
    collapsed view) and appends a bounded body preview. tool_label names the tool
    when known (e.g. '[done: shell]'); falls back to '[done]'.
    """
    head = f"[done: {tool_label}]" if tool_label else "[done]"
    preview = _result_body_preview(body)
    if preview:
        return f"{head}\n{preview}\n"
    return f"{head}\n"


def _result_body_preview(body: Any) -> str:
    """Bounded multi-line preview of a tool-result body (Part B).

    Shared by the _translate ToolResultBlock path (body = block.content) and the
    hook PostToolUse path (body = msg.data result keys). Returns '' when there is
    no meaningful body so the caller can fall back to the bare '[done]' label.
    The salient summary lands on the same activity line so it survives the pi
    collapsed view (continuation lines are display-only-while-live).
    """
    text = _coerce_block_text(body)
    if not text.strip():
        return ""
    return _trim_multiline_hint(
        text, _RESULT_BODY_LIMIT, max_lines=_RESULT_BODY_MAX_LINES
    )


# --- Track 1: regime-B native tool -> regime-A tool widget (pre-resolved) --------
# read/bash/edit/grep/glob/ls/write render as a STRUCTURED tool_call carrying the
# result inline, so pi shows its real tool widget WITHOUT executing. write joins as
# an ATOMIC card: its full content rides the tool_call arguments and the card lands
# on completion (write does not blue-stream during generation, so there is no live
# text to double-render). Streaming the content INTO the card char-by-char (true
# Track 2 = input_json_delta -> tool_call arg deltas) is a further enhancement on top
# of this atomic baseline. MCP/control tools never qualify.
_WIDGET_TOOL_NAMES = frozenset({"read", "bash", "edit", "grep", "glob", "ls", "write"})


def _is_widget_tool(name: Any) -> bool:
    return _normalize_tool_schema_name(name) in _WIDGET_TOOL_NAMES


def _widget_tool_display_name(name: Any) -> str:
    return _normalize_tool_schema_name(name) or (str(name or "").strip() or "tool")


# Track 2: tools whose MODEL-GENERATED INPUT (not the result) is the payload worth
# streaming INTO the card char-by-char -- write's file content. A subset of the
# widget tools. read/bash/edit/grep/glob/ls have tiny inputs (a path/command) and an
# ATOMIC result, so they stay on the Track 1 atomic path; only write streams.
_INPUT_STREAM_TOOL_NAMES = frozenset({"write"})


def _is_input_stream_tool(name: Any) -> bool:
    return _normalize_tool_schema_name(name) in _INPUT_STREAM_TOOL_NAMES


def _normalize_widget_tool_args(name: Any, raw_input: Any) -> Any:
    """Map the SDK's native Edit input keys (old_string/new_string) onto the keys pi's
    edit tool renderer expects (oldText/newText), so the regime-B edit card shows the
    actual diff instead of a bare title. pi already accepts file_path/path. Other tools
    pass through unchanged."""
    if _normalize_tool_schema_name(name) != "edit" or not isinstance(raw_input, dict):
        return raw_input
    out = dict(raw_input)
    if "old_string" in out and "oldText" not in out:
        out["oldText"] = out.pop("old_string")
    if "new_string" in out and "newText" not in out:
        out["newText"] = out.pop("new_string")
    return out


def _tool_args_json(raw_input: Any) -> str:
    import json  # noqa: PLC0415

    if isinstance(raw_input, dict):
        try:
            return json.dumps(raw_input, ensure_ascii=False, default=str)
        except Exception:  # pragma: no cover - defensive
            return "{}"
    return "{}"


def _tool_call_chunk(
    *,
    index: int,
    tool_id: str,
    name: str,
    arguments: str,
    result: str,
    is_error: bool,
) -> dict[str, Any]:
    """One OpenAI tool_calls streaming delta carrying a PRE-RESOLVED native tool
    (call + result) so pi renders its regime-A tool widget WITHOUT executing.
    jarvis_result/jarvis_is_error are sidecar-only sibling fields (regime-A streams
    never carry them); pi's openai-completions reads them onto the toolCall block and
    the agent loop short-circuits execution + terminates the turn on them."""
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                            "jarvis_result": result,
                            "jarvis_is_error": bool(is_error),
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }


def _maybe_pre_resolved_widget_chunk(
    state: _AgentSDKStreamState, tool_id: str, body: Any, is_error: bool
) -> dict[str, Any] | None:
    """If tool_id is a buffered widget call, pop it and return its pre-resolved
    tool_call chunk (call + result); else None. Idempotent via result_finishes so a
    result delivered through BOTH the echo and UserMessage paths emits the widget
    exactly once (no double-render)."""
    if not tool_id:
        return None
    pending = state.pending_widget_tool_calls.get(tool_id)
    if pending is None:
        return None
    state.pending_widget_tool_calls.pop(tool_id, None)
    if tool_id in state.result_finishes:
        return None  # already emitted via the other result path
    state.result_finishes.add(tool_id)
    index = state.tool_call_index
    state.tool_call_index += 1
    return _tool_call_chunk(
        index=index,
        tool_id=tool_id,
        name=pending["name"],
        arguments=pending["arguments"],
        result=_result_body_preview(body),
        is_error=bool(is_error),
    )


def _tool_call_open_chunk(index: int, tool_id: str, name: str) -> dict[str, Any]:
    """Track 2: OPEN a streaming tool_call (no result yet) so pi creates the card
    immediately; the model-generated content then fills in via arg deltas."""
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": name, "arguments": ""},
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }


def _tool_call_arg_delta_chunk(index: int, fragment: str) -> dict[str, Any]:
    """Track 2: one streaming tool_call ARGUMENT fragment (a write's input_json_delta),
    appended to the SAME stream index so pi grows partialArgs and the card fills in."""
    return {
        "choices": [
            {
                "delta": {"tool_calls": [{"index": index, "function": {"arguments": fragment}}]},
                "finish_reason": None,
            }
        ]
    }


def _tool_call_result_attach_chunk(
    *, index: int, tool_id: str, result: str, is_error: bool
) -> dict[str, Any]:
    """Track 2: ATTACH the pre-resolved result to an already-streamed tool_call (args
    arrived live) so pi's agent loop short-circuits execution + terminates the turn.
    The empty arguments string appends nothing -- the content already streamed."""
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"arguments": ""},
                            "jarvis_result": result,
                            "jarvis_is_error": bool(is_error),
                        }
                    ]
                },
                "finish_reason": None,
            }
        ]
    }


def _maybe_streamed_result_chunk(
    state: "_AgentSDKStreamState", tool_id: str, body: Any, is_error: bool
) -> dict[str, Any] | None:
    """If tool_id streamed its input live (Track 2), attach the result to that open
    tool_call (idempotent via result_finishes) and return the chunk; else None."""
    if not tool_id:
        return None
    index = state.streamed_tool_call_index.get(tool_id)
    if index is None:
        return None
    if tool_id in state.result_finishes:
        return None
    state.result_finishes.add(tool_id)
    state.streamed_tool_call_index.pop(tool_id, None)
    return _tool_call_result_attach_chunk(
        index=index, tool_id=tool_id, result=_result_body_preview(body), is_error=is_error
    )


def _coerce_block_text(body: Any) -> str:
    """Flatten an SDK content body (str | list[block] | dict) into preview text."""
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        for key in ("text", "content", "output", "result"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    if isinstance(body, (list, tuple)):
        parts: list[str] = []
        for item in body:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
            else:
                value = getattr(item, "text", None)
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(p for p in parts if p)
    value = getattr(body, "text", None)
    if isinstance(value, str):
        return value
    return ""


def _hook_tool_use_id(data: dict[str, Any]) -> str:
    for key in ("tool_use_id", "toolUseID", "toolUseId", "id"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _agent_tool_short_label(name: Any) -> str:
    """The bare tool label (e.g. 'shell', 'write') with no brackets/hint, used to
    name the matching '[done: ...]' result line."""
    clean_name = str(name or "?").strip() or "?"
    normalized_name = _normalize_tool_schema_name(clean_name)
    if normalized_name:
        clean_name = normalized_name
    label = _AGENT_TOOL_LABELS.get(clean_name)
    if label is None:
        if clean_name.startswith(_JARVIS_CONTROL_TOOL_PREFIX):
            label = clean_name.replace(_JARVIS_CONTROL_TOOL_PREFIX, "").replace("_", " ").strip()
        else:
            label = clean_name
    return label or clean_name


def _agent_tool_activity_label(name: Any, raw_input: Any = None) -> str:
    clean_name = str(name or "?").strip() or "?"
    normalized_name = _normalize_tool_schema_name(clean_name)
    if normalized_name:
        clean_name = normalized_name
    label = _AGENT_TOOL_LABELS.get(clean_name)
    if label is None:
        if clean_name.startswith(_JARVIS_CONTROL_TOOL_PREFIX):
            label = clean_name.replace(_JARVIS_CONTROL_TOOL_PREFIX, "").replace("_", " ").strip()
        else:
            label = f"agent tool: {clean_name}"
    hint = _agent_tool_input_hint(clean_name, raw_input)
    if hint:
        return f"[{label}: {hint}]"
    return f"[{label}]"


def _agent_tool_input_hint(name: str, raw_input: Any) -> str:
    normalized_name = _normalize_tool_schema_name(name)
    if not normalized_name:
        return ""
    # Part C (2026-06-25): ask_user is a control tool, but its INPUT carries the
    # actual planning questions the user must see. Render them instead of blanking
    # (the new-artifact gate forces ask_user-first, so there is no preamble
    # TextBlock to mirror -- this question render is the load-bearing surface).
    if normalized_name == "ask_user":
        # The bridged modal (runAskUserDialog) renders the questions + options
        # interactively, so the activity line stays a bare [ask user] marker instead
        # of dumping every question (which duplicated the modal as blue noise).
        return ""
    if normalized_name in _JLC_CONTROL_TOOL_NAMES:
        return ""
    if not isinstance(raw_input, dict):
        return ""
    if normalized_name in _COMPOSITE_HINT_TOOLS:
        composite = _composite_tool_hint(normalized_name, raw_input)
        if composite:
            return composite
        # fall through to the scalar path (e.g. write with only a file_path).
    keys = _AGENT_TOOL_HINT_KEYS.get(normalized_name, _FALLBACK_TOOL_HINT_KEYS)
    for key in keys:
        value = raw_input.get(key)
        if value is None:
            continue
        hint = _trim_tool_hint(value)
        if hint:
            return hint
    return ""


# ask_user question text lives under one of these keys per question object
# (ask_user_schema, _build_control_server: question/text/prompt/title/label).
_ASK_USER_QUESTION_KEYS = ("question", "text", "prompt", "title", "label")
_ASK_USER_QUESTIONS_LIMIT = 600  # bounded multi-question preview (Part C)
_ASK_USER_QUESTIONS_MAX_LINES = 8


def _ask_user_questions_hint(raw_input: Any) -> str:
    """Bounded multi-line preview of the ask_user planning questions (Part C).

    Pulls each question's text (question/text/prompt/title/label) and a compact
    options list out of ``input['questions']`` so the '[ask user: ...]' activity
    line shows what is actually being asked instead of a bare '[ask user]'.
    Returns '' when no question text is present (caller falls back to the bare
    label). Byte+line bounded like every other preview (DNA #1).
    """
    if not isinstance(raw_input, dict):
        return ""
    questions = raw_input.get("questions")
    items: list[Any]
    if isinstance(questions, (list, tuple)):
        items = list(questions)
    elif isinstance(questions, dict):
        items = [questions]
    else:
        # Fall back to a flat single-question shape (question/prompt at top level).
        items = [raw_input]
    rendered: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            if isinstance(item, str) and item.strip():
                rendered.append(item.strip())
            continue
        text = _first_str(item, _ASK_USER_QUESTION_KEYS)
        if not text:
            continue
        options = item.get("options")
        if isinstance(options, (list, tuple)):
            opts = [str(o).strip() for o in options if str(o).strip()]
            if opts:
                text = f"{text} ({' / '.join(opts)})"
        rendered.append(text)
    if not rendered:
        return ""
    body = "\n".join(f"- {q}" for q in rendered)
    return _trim_multiline_hint(
        body, _ASK_USER_QUESTIONS_LIMIT, max_lines=_ASK_USER_QUESTIONS_MAX_LINES
    )


def _composite_tool_hint(name: str, raw_input: dict[str, Any]) -> str:
    """Per-tool rich preview that shows command/content/diff (not just a path).

    Every branch is byte+line bounded and PRESERVES newlines so the pi panel can
    render multi-line content without flooding (see _BASH_HINT_LIMIT etc.).
    """
    if name == "bash":
        command = raw_input.get("command")
        if command is None:
            return ""
        return _trim_multiline_hint(command, _BASH_HINT_LIMIT)
    if name == "write":
        path = _trim_tool_hint(_first_str(raw_input, ("file_path", "path")))
        content = raw_input.get("content")
        if content is None:
            return path
        body = str(content)
        line_count = body.count("\n") + 1 if body else 0
        header = f"{path or '?'} ({line_count} line{'s' if line_count != 1 else ''})"
        preview = _trim_multiline_hint(
            body, _WRITE_BODY_LIMIT, max_lines=_WRITE_BODY_MAX_LINES
        )
        return f"{header}\n{preview}" if preview else header
    if name in ("edit", "multi_edit"):
        path = _trim_tool_hint(_first_str(raw_input, ("file_path", "path")))
        old = raw_input.get("old_string")
        new = raw_input.get("new_string")
        if old is None and new is None:
            return path
        old_p = _trim_multiline_hint(old, _EDIT_SIDE_LIMIT) if old is not None else ""
        new_p = _trim_multiline_hint(new, _EDIT_SIDE_LIMIT) if new is not None else ""
        diff = f"- {old_p}\n+ {new_p}"
        return f"{path}\n{diff}" if path else diff
    return ""


def _first_str(raw_input: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# --- regime-B observed-work SENSOR helpers (item 1, 2026-06-25) --------------
# Map the SDK's canonical (normalized) tool name onto pi's mutation vocabulary so
# the reconstructed record passes recordJarvisTurnToolOutcome's isJarvisMutationTool
# gate (pi {edit,write,write_file,apply_patch}). MultiEdit is a multi-file mutation
# that is NOT in pi's 4-set as "multi_edit", so it is remapped to "write_file".
# Non-mutation tools (bash/read/...) keep a distinct name and are carried for
# toolEvents only -- pi's sourcePath gate self-skips them for mutations.
_SENSOR_TOOL_PI_VOCAB = {
    "write": "write",
    "edit": "edit",
    "multi_edit": "write_file",
    "notebook_edit": "edit",
    "bash": "bash",
}
# Mutation kind per pi-vocab tool: write creates, edit/write_file/notebook_edit
# edit, everything else is non-mutating ("none").
_SENSOR_MUTATION_KIND = {
    "write": "create",
    "edit": "edit",
    "write_file": "edit",
}
# Hard ceilings (CLAUDE.md DNA #1) so a runaway agentic turn cannot bloat the
# finish chunk. Mirrors pi's own 12-entry bound on turnSuccessfulFileMutations.
_TOOL_ACTIVITY_MAX_RECORDS = 64
_SENSOR_RESULT_PREVIEW_LIMIT = 400  # UTF-8 bytes per result_preview
_SENSOR_COMMAND_LIMIT = 200  # UTF-8 bytes per carried bash command

# Sentinel marker for the regime-B observed-work trailer carried in the FINAL
# assistant TEXT. This is the LIVE transport: pi-ai's openai-completions provider
# materializes delta.content but DROPS unknown top-level chunk keys, so a top-level
# `jarvis_tool_activity` chunk field never reaches the AssistantMessage. The
# consumer (jarvis-jlc.ts JARVIS_SDK_TOOL_TRAILER_MARKER / _RE) parses this exact
# line out of the assistant text and strips it before display/persist. Producer and
# consumer MUST stay pinned to the same marker string.
_TOOL_ACTIVITY_SENTINEL_MARKER = "[[JARVIS_TOOL_ACTIVITY]]"


def _sensor_record_tool_use(
    state: _AgentSDKStreamState, name: Any, tool_id: str, raw_input: Any
) -> None:
    """Capture a tool CALL as a partial sensor record, keyed by tool_use_id.

    Tapped from the SAME ToolUseBlock the streaming label path already reads -- no
    new SDK plumbing. Finalized (success/result_preview) when the matching
    ToolResultBlock arrives. The absolute path comes from input.file_path||path,
    which the SDK reports absolute (live ground truth: it was a full C:\\... path).
    """
    if not tool_id:
        return
    canonical = _normalize_tool_schema_name(name)
    mapped = _SENSOR_TOOL_PI_VOCAB.get(canonical)
    if mapped is None:
        # Non-(mutation|bash) tools (read/glob/grep/web_*) carry no abs_path and are
        # not needed for memory orchestration; skip to keep the trailer lean.
        return
    raw = raw_input if isinstance(raw_input, dict) else {}
    # NotebookEdit supplies its path under `notebook_path` (the module's own hint
    # table _TOOL_NAME_CANONICAL_TO_HINT['notebook_edit'] = ('notebook_path','path')).
    # Read it here too, else abs_path=null and pi's sourcePath gate self-skips a real
    # mutation.
    abs_path = _first_str(raw, ("file_path", "path", "notebook_path")) or None
    command = None
    if mapped == "bash":
        command = _first_str(raw, ("command",)) or None
        if command:
            command = _truncate_to_utf8_bytes(command, _SENSOR_COMMAND_LIMIT)
    state.tool_calls[tool_id] = {
        "tool": mapped,
        "tool_use_id": tool_id,
        "abs_path": abs_path,
        "mutation_kind": _SENSOR_MUTATION_KIND.get(mapped, "none"),
        "success": True,
        "command": command,
        "result_preview": "",
    }


def _sensor_finalize_tool_result(
    state: _AgentSDKStreamState, tool_id: str, is_error: Any, body: Any
) -> None:
    """Finalize a captured tool record from its ToolResultBlock and append it.

    Dedup is guaranteed by the caller's existing result_finishes set, so each tool
    appends exactly once. result_preview is byte-bounded (DNA #1)."""
    rec = state.tool_calls.pop(tool_id, None)
    if rec is None:
        return
    rec["success"] = not bool(is_error)
    preview = _trim_multiline_hint(_coerce_block_text(body), _SENSOR_RESULT_PREVIEW_LIMIT)
    rec["result_preview"] = preview
    if len(state.tool_activity) >= _TOOL_ACTIVITY_MAX_RECORDS:
        # Drop the oldest so a pathological many-file turn cannot bloat the chunk.
        state.tool_activity.pop(0)
    state.tool_activity.append(rec)


def _truncate_to_utf8_bytes(text: str, byte_limit: int, *, suffix: str = "...") -> str:
    """Hard-cap ``text`` so ``len(result.encode('utf-8')) <= byte_limit``.

    CLAUDE.md DNA #1: bounds are a HARD ceiling on the REAL wire size. ``len()``
    counts code points, so a Korean preview (3 UTF-8 bytes/char) used to run ~3x
    the named byte budget. We measure bytes, reserve room for ``suffix`` WITHIN
    the limit, slice on a byte boundary, and decode with ``errors='ignore'`` so a
    multibyte char is never split mid-sequence. The returned string (suffix
    included) is guaranteed to encode to <= ``byte_limit`` bytes.
    """
    if byte_limit <= 0:
        return ""
    raw = text.encode("utf-8")
    if len(raw) <= byte_limit:
        return text
    suffix_bytes = suffix.encode("utf-8")
    if len(suffix_bytes) >= byte_limit:
        # Degenerate: no room for content; emit as much of the suffix as fits.
        return suffix_bytes[:byte_limit].decode("utf-8", "ignore")
    keep = byte_limit - len(suffix_bytes)
    return raw[:keep].decode("utf-8", "ignore") + suffix


def _trim_tool_hint(value: Any) -> str:
    text = " ".join(str(value).split())
    # _TOOL_HINT_LIMIT is a REAL UTF-8 byte ceiling (DNA #1), so Korean/CJK single
    # scalars stay within budget instead of running ~3x over.
    return _truncate_to_utf8_bytes(text, _TOOL_HINT_LIMIT)


def _trim_multiline_hint(
    value: Any, byte_limit: int, *, max_lines: int | None = None
) -> str:
    """Bound a preview by BOTH bytes and lines while PRESERVING newlines.

    Unlike _trim_tool_hint (which collapses all whitespace and caps at 96), this
    keeps line structure for content/diff previews. Appends a ' ...(+N lines)' /
    '...' marker when truncated.

    ``byte_limit`` is a HARD UTF-8 BYTE ceiling (DNA #1): the marker is counted
    WITHIN the limit, so ``len(result.encode('utf-8')) <= byte_limit`` always
    holds (the trailing ``...(+N more lines)`` note is an additive, bounded tail).
    Previously the size check used ``len()`` (code points), letting a Korean
    preview run ~3x the named byte budget.
    """
    if value is None:
        return ""
    text = str(value).rstrip("\n")
    if not text:
        return ""
    lines = text.split("\n")
    dropped_lines = 0
    if max_lines is not None and len(lines) > max_lines:
        dropped_lines = len(lines) - max_lines
        lines = lines[:max_lines]
    text = "\n".join(lines)
    text = _truncate_to_utf8_bytes(text, byte_limit)
    if dropped_lines:
        text = f"{text}\n...(+{dropped_lines} more line{'s' if dropped_lines != 1 else ''})"
    return text


def _content_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}, "finish_reason": None}]}


def _reasoning_chunk(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"reasoning_content": text}, "finish_reason": None}]}


def _finish_chunk(
    stop_reason: str, *, tool_activity: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    chunk: dict[str, Any] = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    # Regime-B observed-work trailer (item 1): a top-level array SIBLING to
    # `choices`, attached ONLY when non-empty. app.py's _openai_proxy_chunk does
    # dict(chunk)+setdefaults and never strips unknown top-level keys, so this
    # rides verbatim through the streaming /v1 SSE path to pi, which (regime B
    # only) reconstructs turnSuccessfulFileMutations + toolEvents from it.
    if tool_activity:
        chunk["jarvis_tool_activity"] = tool_activity
    return chunk


def _tool_activity_sentinel_chunk(
    tool_activity: list[dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Build the LIVE-transport content chunk carrying the trailer sentinel.

    pi-ai's openai-completions provider materializes ``delta.content`` but drops
    unknown top-level chunk keys, so the top-level ``jarvis_tool_activity`` on the
    finish chunk never reaches pi. We additionally fold the trailer into the FINAL
    assistant TEXT as a single ``[[JARVIS_TOOL_ACTIVITY]] {json}`` line, which the
    consumer parses (parseJarvisSdkToolTrailerFromText) and strips
    (stripJarvisSdkToolTrailer) before display/persist. The per-record fields are
    already byte-bounded (result_preview<=400, command<=200) and the array is capped
    at 64 records, so this line cannot bloat (DNA #1). Returns None when no tools ran
    (no sentinel emitted, so a no-tool turn's text is untouched).
    """
    if not tool_activity:
        return None
    import json  # noqa: PLC0415

    payload = json.dumps({"jarvis_tool_activity": tool_activity}, ensure_ascii=False)
    # Newline-delimited so the consumer's anchored, multiline regex matches the line
    # cleanly regardless of any preceding assistant text.
    return _content_chunk(f"\n{_TOOL_ACTIVITY_SENTINEL_MARKER} {payload}\n")


def _result_llm_meta(result_msg: Any, model: str, latency_ms: int) -> dict[str, Any]:
    """Map ResultMessage usage/cost into the llm_meta shape JLC's turn logger and
    cost accounting expect (matches provider_router._llm_meta)."""
    usage = getattr(result_msg, "usage", None) or {}
    tokens_in = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    tokens_out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    cache_read = int(
        usage.get("cache_read_input_tokens") or usage.get("cached_tokens") or 0
    )
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    total = int(usage.get("total_tokens") or (tokens_in + tokens_out + cache_read + cache_write))
    cost = getattr(result_msg, "total_cost_usd", None)
    return {
        "alias": f"anthropic-agent-sdk/{model}",
        "provider": "anthropic-agent-sdk",
        "litellm_id": model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": 0,
        "total_tokens": total,
        "usage": usage,
        "cost_usd": float(cost) if cost is not None else 0.0,
        "latency_ms": latency_ms,
        "key_idx": None,
        "fallback_attempts": 0,
    }
