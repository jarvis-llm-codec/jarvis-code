from __future__ import annotations

import html as html_lib
import json
import os
import re
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from jlc_agentic.agentic.tools.web_search import handler as brave_web_search
from jlc_agentic.providers import get_llm
from jlc_agentic.providers import load_config as load_runtime_config
from jlc_agentic.providers import clear_cache
from jlc_agentic.slim import JarvisAgentic

from .config import (
    credentials_path,
    ensure_sidecar_config,
    get_default_project_root,
    get_effective_project_root,
    get_project_root_source,
    get_protected_roots,
    internal_memory_root,
    load_credentials_into_env,
    providers_path,
    save_credential_env,
    setup_default_project_root,
)
from .evidence import EvidenceError, retrieve_evidence, run_evidence_gc, store_evidence
from .llm_setting import apply_picks as llm_apply_picks
from .llm_setting import current_roles as llm_current_roles
from .llm_setting import fetch_all as llm_fetch_all
from .llm_setting import load_catalog as llm_load_catalog
from .memory_files import (
    clear_interrupt_checkpoint,
    ensure_workspace_memory,
    read_project_memory,
    update_jarvis_md_batch as update_project_jarvis_md_batch,
    update_jarvis_md as update_project_jarvis_md,
    write_interrupt_checkpoint,
)
from .project_router import ProjectRouter
from .raw_store import (
    append_encoder_turn,
    append_raw_turn,
    extract_local_dates,
    extract_turn_numbers,
    recall_raw,
    recent_failure_modes,
    recent_turns,
)
from .provider_router_holder import set_provider_router as set_sidecar_provider_router
from .web_tools import docs_search as docs_search_tool
from .web_tools import package_info as package_info_tool
from .web_tools import web_fetch as web_fetch_tool
from .workspace import RegistryCorruptError, parse_setup_default_root_command

_SESSION_ID = "jarvis_session"

# In-memory cache for background encoding results, keyed by session_id.
# Populated by the on_done callback wired into encode_and_save_async (/turn),
# consumed by pi-agent via GET /encoding_status polling so the chat surface
# can render an "encoded: NNN tok" line once the background encode completes.
_encoding_results: dict[str, dict[str, Any]] = {}
_encoding_results_lock = threading.Lock()
_encoder_stream_print_lock = threading.Lock()
_encoder_stream_print_kind: str | None = None
_project_memory_load_log_keys: set[str] = set()
_ANSI_PURPLE = "\x1b[95m"
_ANSI_RESET = "\x1b[0m"


def _print_project_memory_loaded(*, project_name: str | None, project_path: str, tokens: int) -> None:
    label = project_name or os.path.basename(os.path.normpath(project_path)) or "project"
    print(
        f"{_ANSI_PURPLE}[jarvis-sidecar] JARVIS.md loaded project={label} "
        f"tokens~={tokens} path={project_path}{_ANSI_RESET}",
        file=sys.stderr,
        flush=True,
    )


def _print_project_memory_updated(*, project_path: str | None, result: dict[str, Any]) -> None:
    if not result.get("ok"):
        return
    if result.get("unchanged"):
        return
    path = str(result.get("path") or "")
    label_path = project_path or (str(Path(path).parent) if path else "")
    label = os.path.basename(os.path.normpath(label_path)) if label_path else "project"
    fields = result.get("fields") or result.get("field") or []
    if isinstance(fields, str):
        field_text = fields
    elif isinstance(fields, list):
        field_text = ",".join(str(field) for field in fields)
    else:
        field_text = ""
    field_part = f" fields={field_text}" if field_text else ""
    bytes_part = f" bytes={result.get('bytes')}" if result.get("bytes") is not None else ""
    print(
        f"{_ANSI_PURPLE}[jarvis-sidecar] JARVIS.md 업데이트 project={label}"
        f"{field_part}{bytes_part} path={path or label_path}{_ANSI_RESET}",
        file=sys.stderr,
        flush=True,
    )


def _project_payload(project: Any) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "name": project.name,
        "slug": project.slug,
        "path": project.path,
        "memory_path": project.path,
        "code_path": project.code_path,
    }


def _remaining_project_payload() -> list[dict[str, Any]]:
    if not router.registry.status_fields()["registry_ok"]:
        return []
    return [
        {
            "project_id": project.project_id,
            "name": project.name,
            "path": project.path,
        }
        for project in router.registry.all()
    ]


def _registered_memory_project(project_path: str | None) -> Any | None:
    if not project_path:
        return None
    project = router.registry.get_by_path(project_path)
    if project is None:
        router.clear_active_project()
        return None
    return project


def _guard_memory_write_path(project_path: str | None) -> str | None:
    if not project_path:
        return None
    project = _registered_memory_project(project_path)
    return project.path if project is not None else None


def _path_key(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return str(Path(value).expanduser().resolve()).casefold()
    except OSError:
        return None


def _limited_workspace_line(prefix: str, entries: list[str], hidden_count: int, *, max_bytes: int) -> str:
    visible = list(entries)
    hidden = max(0, hidden_count)

    def render() -> str:
        if visible:
            value = ", ".join(visible)
            suffix = f" (+{hidden} more)" if hidden else ""
            return f"{prefix}: {value}{suffix}"
        if hidden:
            return f"{prefix}: (+{hidden} more)"
        return f"{prefix}: (none)"

    while visible and len(render().encode("utf-8")) > max_bytes:
        visible.pop()
        hidden += 1
    return render()


def _limited_workspace_value(prefix: str, value: str, *, max_bytes: int) -> str:
    marker = "..."
    line = f"{prefix}: {value}"
    if len(line.encode("utf-8")) <= max_bytes:
        return line
    budget = max_bytes - len(f"{prefix}: {marker}".encode("utf-8"))
    if budget <= 0:
        return f"{prefix}: {marker}"
    raw = value.encode("utf-8")[:budget]
    while raw:
        try:
            trimmed = raw.decode("utf-8")
            return f"{prefix}: {trimmed}{marker}"
        except UnicodeDecodeError:
            raw = raw[:-1]
    return f"{prefix}: {marker}"


def _workspace_project_entry(project: Any) -> str:
    label = str(project.name or project.slug or project.project_id)
    project_path = str(project.code_path or project.path or "")
    if len(project_path) > 96:
        project_path = f"...{project_path[-93:]}"
    return f"{label} ({project_path})" if project_path else label


def _build_workspace_block(active_project: Any | None = None) -> str:
    projects = [] if router.registry.is_corrupt else sorted(
        router.registry.all(),
        key=lambda project: (
            str(project.name or "").casefold(),
            str(project.slug or "").casefold(),
            str(project.code_path or project.path or "").casefold(),
        ),
    )
    project_limit = 12
    folder_limit = 8
    registered_entries = [_workspace_project_entry(project) for project in projects[:project_limit]]
    registered_hidden = max(0, len(projects) - len(registered_entries))

    active = active_project or router.registry.get_by_id(router.active_project_id)
    active_label = str(active.name or active.slug or active.project_id) if active is not None else "(none)"

    registered_paths = {
        key
        for project in projects
        for key in (_path_key(project.path), _path_key(project.code_path))
        if key is not None
    }
    unregistered_entries: list[str] = []
    default_root = get_effective_project_root()
    if default_root:
        try:
            for child in sorted(Path(default_root).iterdir(), key=lambda item: item.name.casefold()):
                if not child.is_dir():
                    continue
                key = _path_key(str(child))
                if key is None or key in registered_paths:
                    continue
                unregistered_entries.append(child.name)
        except OSError:
            unregistered_entries = []
    visible_unregistered = unregistered_entries[:folder_limit]
    unregistered_hidden = max(0, len(unregistered_entries) - len(visible_unregistered))

    lines = [
        "[WORKSPACE]",
        _limited_workspace_line("registered", registered_entries, registered_hidden, max_bytes=360),
        _limited_workspace_value("active", active_label, max_bytes=120),
        _limited_workspace_line(
            "unregistered folders in workspace",
            visible_unregistered,
            unregistered_hidden,
            max_bytes=160,
        ),
    ]
    block = "\n".join(lines)
    while len(block.encode("utf-8")) > 650 and registered_entries:
        registered_entries = registered_entries[:-1]
        registered_hidden += 1
        lines[1] = _limited_workspace_line("registered", registered_entries, registered_hidden, max_bytes=320)
        block = "\n".join(lines)
    return block


def _print_encoder_token(text: str, kind: str = "content") -> None:
    """Force encoder streaming and mirror chunks in the visible sidecar."""
    if not text or os.environ.get("JLC_ENCODER_SIDECAR_OUTPUT", "1") == "0":
        return
    label = "reasoning" if kind == "reasoning" else "content"
    global _encoder_stream_print_kind
    with _encoder_stream_print_lock:
        try:
            if _encoder_stream_print_kind != label:
                if _encoder_stream_print_kind is not None:
                    sys.stderr.write(f"\n[jlc:enc:{_encoder_stream_print_kind}] ---END---\n")
                sys.stderr.write(f"[jlc:enc:{label}] ---BEGIN---\n")
                _encoder_stream_print_kind = label
            sys.stderr.write(text)
            sys.stderr.flush()
        except Exception:
            pass


def _finish_encoder_token_print() -> None:
    global _encoder_stream_print_kind
    with _encoder_stream_print_lock:
        if _encoder_stream_print_kind is None:
            return
        sys.stderr.write(f"\n[jlc:enc:{_encoder_stream_print_kind}] ---END---\n")
        sys.stderr.flush()
        _encoder_stream_print_kind = None


class ContextRequest(BaseModel):
    project_path: str | None = None
    active_project_path: str | None = None
    cwd_hint: str | None = None
    user_message: str = ""
    mode: str = "coding"
    hints: dict[str, Any] = Field(default_factory=dict)
    cwd: str | None = None
    bench_conv_id: str | None = None
    context_turn_key: str | None = None


class TurnRequest(BaseModel):
    project_path: str | None = None
    user_message: str = ""
    assistant_message: str = ""
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    llm_meta: dict[str, Any] = Field(default_factory=dict)
    bench_conv_id: str | None = None


class RecallRequest(BaseModel):
    queries: list[str] = Field(default_factory=list)
    top_k: int = 5
    bench_conv_id: str | None = None


class RegisterProjectRequest(BaseModel):
    path: str
    name: str | None = None


class UnregisterProjectRequest(BaseModel):
    project_id: str | None = None
    slug_or_name: str | None = None
    path: str | None = None


class SwitchRequest(BaseModel):
    slug_or_name: str
    code_path: str | None = None
    auto_create: bool = False


class ResolvePathRequest(BaseModel):
    path: str


class RouteTurnRequest(BaseModel):
    user_message: str = ""
    cwd_hint: str | None = None
    active_project_path: str | None = None
    recent_messages: list[dict[str, str]] = Field(default_factory=list)
    pending_project: dict[str, Any] | None = None
    bench_conv_id: str | None = None


class SetupRequest(BaseModel):
    default_project_root: str


class UpdateJarvisMdRequest(BaseModel):
    project_path: str
    field: str | None = None
    value: Any = None
    updates: list[dict[str, Any]] | None = None


class InterruptCheckpointRequest(BaseModel):
    project_path: str
    user_message: str = ""
    assistant_message: str = ""
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    subturn_log: str = ""
    mode: str = ""
    cwd: str | None = None
    reason: str = "escape_interrupt"

class ClearInterruptCheckpointRequest(BaseModel):
    project_path: str | None = None


class EvidenceStoreRequest(BaseModel):
    session_id: str | None = None
    conversation_id: str | None = None
    turn_key: str | None = None
    provider_call_id: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    kind: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    original_text: str
    compressed_text: str | None = None
    original_tokens_est: int | None = None
    compressed_tokens_est: int | None = None
    kept_count: int | None = None
    dropped_count: int | None = None
    expires_at: str | None = None


class SubturnMeterRequest(BaseModel):
    reason: str = ""
    user_turn_key: str | None = None
    call_input_tokens: int = 0
    turn_input_tokens: int = 0
    output_tokens: int = 0
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    usage_cache_read_tokens: int | None = None
    usage_cache_write_tokens: int | None = None
    usage_total_tokens: int | None = None
    usage_reasoning_tokens: int | None = None
    provider_cache_read_tokens: int | None = None
    provider_cache_write_tokens: int | None = None
    cache_meter: str | None = None
    cache_hit_pct: float | None = None
    stable_prefix_hash: str | None = None
    stable_prefix_tokens_est: int | None = None
    live_tokens_est: int | None = None
    compressed_tool_outputs: int = 0
    compression_saved_tokens_est: int = 0
    compression_skips: dict[str, int] | None = None
    provider_calls: int = 0
    elapsed_seconds: float = 0.0
    payload_summary: str = ""
    payload_preview: str = ""
    subturn_summary: str = ""


class SubturnObserveRequest(BaseModel):
    source: str = "jlc"
    event: str = ""
    session_id: str | None = None
    user_turn_key: str | None = None
    legacy: dict[str, Any] = Field(default_factory=dict)
    candidate: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class WebSearchRequest(BaseModel):
    query: str
    top_k: int = 5


class WebFetchRequest(BaseModel):
    url: str
    max_chars: int = 12000
    timeout_sec: float = 10.0


class DocsSearchRequest(BaseModel):
    query: str
    domains: list[str] = Field(default_factory=list)
    top_k: int = 5
    fetch_top: int = 0
    max_chars: int = 4000


class PackageInfoRequest(BaseModel):
    ecosystem: Literal["npm", "pypi", "github"] = "npm"
    package: str
    include_release_notes: bool = False


def _compact_tokens(value: int) -> str:
    value = max(0, int(value))
    if value < 1000:
        return str(value)
    return f"{value / 1000:.1f}k"


class TranslateInputRequest(BaseModel):
    text: str
    target_language: str = "English"


class LLMSettingApplyRequest(BaseModel):
    chat: str  # "provider/model"
    encoder: str  # "provider/model"


class CredentialSetRequest(BaseModel):
    env_name: str
    value: str
    do_validate: bool = Field(default=True, alias="validate")


router = ProjectRouter()
_agent: JarvisAgentic | None = None
_agent_last_error: str | None = None
_agent_last_error_type: str | None = None
_agent_last_error_repr: str | None = None
_agent_last_error_filename: str | None = None
_agent_last_error_traceback: str | None = None
_agent_last_error_ts: datetime | None = None
_agent_retry_interval = 60.0
_agent_guard = threading.Lock()
_last_bench_conv_id: str | None = None
_subturn_debug_lock = threading.Lock()
_subturn_debug_events: list[dict[str, Any]] = []
_subturn_debug_next_id = 0
_SUBTURN_DEBUG_MAX_EVENTS = 200
_SUBTURN_DEBUG_TEXT_MAX_CHARS = 20000


def _init_provider_router() -> None:
    try:
        from jlc_agentic.bootstrap import init_provider_router
        ensure_sidecar_config()
        set_sidecar_provider_router(None)
        router = init_provider_router(SimpleNamespace(providers_config=str(providers_path())))
        set_sidecar_provider_router(router)
        clear_cache()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        set_sidecar_provider_router(None)
        clear_cache()
        print(f"[jarvis-sidecar] provider router init skipped: {exc}", file=sys.stderr)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        _init_provider_router()
        run_evidence_gc()
        yield
    finally:
        await close_agent()


app = FastAPI(title="JARVIS Code JLC Sidecar", version="1.01.0", lifespan=lifespan)


_SUBTURN_DEBUG_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JARVIS Subturn Observe</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f1419;
      --panel: #151c23;
      --panel-2: #101820;
      --text: #e8edf2;
      --muted: #95a3b3;
      --line: #2a3541;
      --orange: #ff8a3d;
      --blue: #57b6ff;
      --green: #70d38b;
      --red: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: var(--panel-2);
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .brand { color: var(--orange); }
    .actions { display: flex; gap: 8px; align-items: center; }
    button {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
    }
    button:hover { border-color: var(--blue); }
    main { padding: 18px 22px 28px; }
    .cards {
      display: grid;
      grid-template-columns: repeat(5, minmax(140px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }
    .card {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px;
      min-height: 76px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 5px;
    }
    .value {
      font-size: 20px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .subvalue {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
      overflow-wrap: anywhere;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr);
      gap: 14px;
    }
    section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      min-width: 0;
      overflow: hidden;
    }
    section h2 {
      margin: 0;
      padding: 11px 13px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }
    .events { max-height: calc(100vh - 238px); overflow: auto; }
    .event {
      width: 100%;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      background: transparent;
      text-align: left;
      padding: 10px 12px;
    }
    .event.active { background: rgba(87, 182, 255, 0.12); }
    .event-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--text);
      font-weight: 600;
    }
    .event-meta {
      color: var(--muted);
      font-size: 12px;
      margin-top: 3px;
      overflow-wrap: anywhere;
    }
    .details { padding: 12px; }
    .summary {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 12px;
    }
    .mini {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 6px;
      padding: 9px;
      min-width: 0;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
      color: #dce8f7;
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px;
      max-height: calc(100vh - 380px);
      overflow: auto;
    }
    .empty {
      color: var(--muted);
      padding: 18px;
    }
    .ok { color: var(--green); }
    .warn { color: var(--orange); }
    @media (max-width: 980px) {
      .cards { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
      .layout { grid-template-columns: 1fr; }
      .events { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <h1><span class="brand">JARVIS</span> Subturn Observe</h1>
    <div class="actions">
      <span id="status" class="subvalue">loading</span>
      <button id="refresh">Refresh</button>
      <button id="clear">Clear</button>
    </div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Calls / Events</div><div id="count" class="value">0</div></div>
      <div class="card"><div class="label">Latest Call</div><div id="calls" class="value">-</div></div>
      <div class="card"><div class="label">Actual Input</div><div id="callInput" class="value">-</div></div>
      <div class="card"><div class="label">Actual Total</div><div id="actualTotal" class="value">-</div></div>
      <div class="card"><div class="label">Actual Saved</div><div id="saved" class="value">-</div></div>
      <div class="card">
        <div class="label">Cache / Compression</div>
        <div id="cacheSummary" class="value">__INITIAL_CACHE_SUMMARY__</div>
        <div id="compressSummary" class="subvalue">__INITIAL_COMPRESS_SUMMARY__</div>
      </div>
    </div>
    <div class="layout">
      <section>
        <h2>Provider Calls</h2>
        <div id="events" class="events"><div class="empty">No subturn observations yet.</div></div>
      </section>
      <section>
        <h2>Selected Call</h2>
        <div class="details">
          <div id="summary" class="summary"></div>
          <pre id="detail">No event selected.</pre>
        </div>
      </section>
    </div>
  </main>
  <script id="initial-state" type="application/json">__INITIAL_STATE_JSON__</script>
  <script>
    const stateUrl = "/debug/subturn/state";
    const clearUrl = "/debug/subturn/clear";
    let current = null;
    let selectedId = null;

    function fmt(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return value.toLocaleString();
      return String(value);
    }

    function tokenValue(event, key) {
      return event && event.tokens ? event.tokens[key] : undefined;
    }

    function usageValue(event, key) {
      return event && event.usage ? event.usage[key] : undefined;
    }

    function pct(value) {
      return value === null || value === undefined ? "-" : `${value}%`;
    }

    function compactTokens(value) {
      const numeric = Number(value || 0);
      if (!Number.isFinite(numeric) || numeric <= 0) return "0";
      if (numeric < 1000) return String(Math.round(numeric));
      const compact = Math.round((numeric / 1000) * 10) / 10;
      return `${Number.isInteger(compact) ? compact.toFixed(0) : compact.toFixed(1)}k`;
    }

    function cachePct(value) {
      if (typeof value !== "number" || !Number.isFinite(value)) return "";
      const percent = Math.round(value * 1000) / 10;
      return Number.isInteger(percent) ? percent.toFixed(0) : percent.toFixed(1);
    }

    function meterEventForCall(state, call) {
      return call ? eventById(state, call.meter_event_id) : undefined;
    }

    function previousCallForPrefix(state, call) {
      const callNo = Number(call && call.call);
      if (!Number.isFinite(callNo)) return undefined;
      return (state.calls || [])
        .filter((item) => {
          if (item === call) return false;
          if (call.user_turn_key && item.user_turn_key && item.user_turn_key !== call.user_turn_key) return false;
          const itemCall = Number(item.call);
          return Number.isFinite(itemCall) && itemCall < callNo;
        })
        .sort((left, right) => Number(right.call) - Number(left.call))[0];
    }

    function prefixMark(state, call, probe) {
      const hash = probe && typeof probe.stable_prefix_hash === "string" ? probe.stable_prefix_hash : "";
      if (!hash) return { text: "", changed: false };
      const previous = previousCallForPrefix(state, call);
      const previousProbe = previous && meterEventForCall(state, previous)?.cache_probe;
      const previousHash =
        previousProbe && typeof previousProbe.stable_prefix_hash === "string" ? previousProbe.stable_prefix_hash : "";
      if (!previousHash) return { text: "", changed: false };
      return previousHash === hash ? { text: "=", changed: false } : { text: "≠", changed: true };
    }

    function cacheSummary(probe, mark = "") {
      if (!probe) return "cache: -";
      const meter = String(probe.cache_meter || "").trim();
      if (meter === "unreported") return "cache: unreported";
      if (!meter) return "cache: -";
      const pctText = cachePct(probe.cache_hit_pct);
      const hash = typeof probe.stable_prefix_hash === "string" ? probe.stable_prefix_hash.slice(0, 8) : "";
      return `cache: ${meter}${pctText ? ` ${pctText}%` : ""}${hash ? ` | hash ${hash}${mark}` : ""}`;
    }

    function compressionSummary(compression) {
      if (!compression) return "compress: —";
      const outputs = Number(compression.compressed_tool_outputs || 0);
      const saved = Number(compression.compression_saved_tokens_est || 0);
      if (!Number.isFinite(outputs) || outputs <= 0) return "compress: —";
      return `compress: ${outputs} ${outputs === 1 ? "output" : "outputs"}, ~${compactTokens(saved)} saved`;
    }

    function callMetrics(state, call) {
      const meterEvent = meterEventForCall(state, call);
      const probe = meterEvent && meterEvent.cache_probe;
      const mark = prefixMark(state, call, probe);
      return {
        cacheText: cacheSummary(probe, mark.text),
        compressText: compressionSummary(meterEvent && meterEvent.compression),
        cacheChanged: mark.changed,
      };
    }

    function eventMetrics(event) {
      return {
        cacheText: cacheSummary(event && event.cache_probe),
        compressText: compressionSummary(event && event.compression),
        cacheChanged: false,
      };
    }

    function savedEstimate(event) {
      const legacy = event && event.legacy ? event.legacy.tokens : undefined;
      const candidate = event && event.candidate ? event.candidate.tokens : undefined;
      if (!legacy || !candidate || !legacy.total || !candidate.total) return "-";
      const saved = Math.max(0, legacy.total - candidate.total);
      return `${Math.round((saved / legacy.total) * 1000) / 10}%`;
    }

    function mini(label, value) {
      const box = document.createElement("div");
      box.className = "mini";
      const l = document.createElement("div");
      l.className = "label";
      l.textContent = label;
      const v = document.createElement("div");
      v.className = "value";
      v.textContent = fmt(value);
      box.append(l, v);
      return box;
    }

    function breakdownBrief(breakdown) {
      if (!breakdown || !breakdown.tokens) return "-";
      return Object.entries(breakdown.tokens)
        .filter(([, value]) => typeof value === "number" && value > 0)
        .sort((left, right) => right[1] - left[1])
        .slice(0, 5)
        .map(([key, value]) => `${key}:${fmt(value)}`)
        .join(" | ") || "-";
    }

    function renderCards(state) {
      const latest = state.latest_call;
      const metrics = latest ? callMetrics(state, latest) : eventMetrics(state.latest);
      document.getElementById("count").textContent = `${fmt((state.calls || []).length)} / ${fmt(state.count)}`;
      document.getElementById("calls").textContent = fmt(latest && latest.call);
      document.getElementById("callInput").textContent = fmt(latest && latest.actual_message_tokens);
      document.getElementById("actualTotal").textContent = fmt(latest && latest.actual_total_tokens);
      document.getElementById("saved").textContent = pct(latest && latest.actual_saved_pct);
      const cacheSummaryNode = document.getElementById("cacheSummary");
      cacheSummaryNode.textContent = metrics.cacheText;
      cacheSummaryNode.className = "value" + (metrics.cacheChanged ? " warn" : "");
      document.getElementById("compressSummary").textContent = metrics.compressText;
    }

    function eventById(state, id) {
      return (state.events || []).find((event) => event.id === id);
    }

    function validSelection(state) {
      if (!selectedId) return false;
      if (String(selectedId).startsWith("call:")) {
        return (state.calls || []).some((call) => `call:${call.key}` === selectedId);
      }
      return (state.events || []).some((event) => `event:${event.id}` === selectedId);
    }

    function renderEvents(state) {
      const root = document.getElementById("events");
      root.textContent = "";
      const calls = state.calls || [];
      if (!calls.length && (!state.events || state.events.length === 0)) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No subturn observations yet.";
        root.append(empty);
        return;
      }
      if (!validSelection(state)) selectedId = calls.length ? `call:${calls[0].key}` : `event:${state.events[0].id}`;
      for (const call of calls) {
        const metrics = callMetrics(state, call);
        const button = document.createElement("button");
        const key = `call:${call.key}`;
        button.className = "event" + (key === selectedId ? " active" : "");
        button.onclick = () => {
          selectedId = key;
          render(current);
        };
        const title = document.createElement("div");
        title.className = "event-title";
        const left = document.createElement("span");
        left.textContent = `call ${call.call} ${call.route || call.mode || "unknown"}`;
        const right = document.createElement("span");
        right.textContent = pct(call.saved_pct);
        title.append(left, right);
        const meta = document.createElement("div");
        meta.className = "event-meta";
        const legacy = fmt(call.legacy_tokens);
        const actual = fmt(call.actual_message_tokens);
        const candidate = fmt(call.candidate_tokens);
        meta.textContent = `${call.timestamp || "-"} | ${legacy} -> ${actual} -> ${candidate} | ${metrics.cacheText} | ${metrics.compressText} | ${call.actual_summary || ""}`;
        if (metrics.cacheChanged) meta.classList.add("warn");
        button.append(title, meta);
        root.append(button);
      }
      if (calls.length) return;
      for (const event of state.events) {
        const metrics = eventMetrics(event);
        const button = document.createElement("button");
        const key = `event:${event.id}`;
        button.className = "event" + (key === selectedId ? " active" : "");
        button.onclick = () => {
          selectedId = key;
          render(current);
        };
        const title = document.createElement("div");
        title.className = "event-title";
        const left = document.createElement("span");
        left.textContent = `#${event.provider_calls ?? event.id} ${event.reason || event.event || event.source}`;
        const right = document.createElement("span");
        right.textContent = `${fmt(event.elapsed_seconds)}s`;
        title.append(left, right);
        const meta = document.createElement("div");
        meta.className = "event-meta";
        meta.textContent = `${event.timestamp} | ${event.source || "unknown"} | ${metrics.cacheText} | ${metrics.compressText} | ${event.actual_summary || ""}`;
        button.append(title, meta);
        root.append(button);
      }
    }

    function renderDetail(state) {
      const summary = document.getElementById("summary");
      const detail = document.getElementById("detail");
      summary.textContent = "";
      if (selectedId && String(selectedId).startsWith("call:")) {
        const callKey = String(selectedId).slice("call:".length);
        const call = (state.calls || []).find((item) => item.key === callKey) || state.latest_call;
        if (!call) {
          detail.textContent = "No provider call selected.";
          return;
        }
        const metrics = callMetrics(state, call);
        summary.append(
          mini("Route", call.route || call.mode || "-"),
          mini("Provider Call", call.call),
          mini("Legacy Tokens", call.legacy_tokens),
          mini("Actual Tokens", call.actual_message_tokens),
          mini("Candidate Tokens", call.candidate_tokens),
          mini("Input Breakdown", breakdownBrief(call.actual_breakdown)),
          mini("Actual Total", call.actual_total_tokens),
          mini("Actual Saved", pct(call.actual_saved_pct)),
          mini("Candidate Estimate", pct(call.saved_pct)),
          mini("Cache", metrics.cacheText),
          mini("Compression", metrics.compressText)
        );
        detail.textContent = JSON.stringify(
          {
            call,
            meter_event: eventById(state, call.meter_event_id),
            candidate_event: eventById(state, call.candidate_event_id),
          },
          null,
          2
        );
        return;
      }
      const eventId = selectedId ? Number(String(selectedId).replace(/^event:/, "")) : undefined;
      const event = (state.events || []).find((item) => item.id === eventId) || state.latest;
      if (!event) {
        detail.textContent = "No event selected.";
        return;
      }
      const metrics = eventMetrics(event);
      summary.append(
        mini("Reason", event.reason || event.event || "-"),
        mini("Provider Calls", event.provider_calls),
        mini("Call Input", tokenValue(event, "call_input")),
        mini("Turn Input", tokenValue(event, "turn_input")),
        mini("Output", tokenValue(event, "output")),
        mini("Cache Read", usageValue(event, "cache_read")),
        mini("Actual Total", usageValue(event, "total")),
        mini("Saved Estimate", savedEstimate(event)),
        mini("Cache", metrics.cacheText),
        mini("Compression", metrics.compressText)
      );
      detail.textContent = JSON.stringify(event, null, 2);
    }

    function render(state) {
      current = state;
      renderCards(state);
      renderEvents(state);
      renderDetail(state);
      document.getElementById("status").textContent = `updated ${new Date().toLocaleTimeString()}`;
    }

    async function refresh() {
      const response = await fetch(stateUrl, { cache: "no-store" });
      render(await response.json());
    }

    async function clearEvents() {
      await fetch(clearUrl, { method: "POST" });
      selectedId = null;
      await refresh();
    }

    document.getElementById("refresh").onclick = refresh;
    document.getElementById("clear").onclick = clearEvents;
    try {
      const initialText = document.getElementById("initial-state")?.textContent || "";
      if (initialText.trim()) render(JSON.parse(initialText));
    } catch (error) {
      document.getElementById("status").textContent = `initial state failed: ${error}`;
    }
    refresh().catch((error) => {
      document.getElementById("status").textContent = `failed: ${error}`;
    });
    setInterval(refresh, 3000);
  </script>
</body>
</html>
"""


def _truncate_debug_text(value: Any, max_chars: int = _SUBTURN_DEBUG_TEXT_MAX_CHARS) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 15)].rstrip()}...[truncated]"


def _compact_debug_dict(value: dict[str, Any], max_chars: int = _SUBTURN_DEBUG_TEXT_MAX_CHARS) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        compact[key] = _compact_debug_value(item, max_chars)
    return compact


def _compact_debug_value(value: Any, max_chars: int = _SUBTURN_DEBUG_TEXT_MAX_CHARS) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _truncate_debug_text(value, max_chars) if isinstance(value, str) else value
    if isinstance(value, list):
        compact = [_compact_debug_value(child, 4000) for child in value[:50]]
        if len(value) > 50:
            compact.append({"truncated_count": len(value) - 50})
        return compact
    if isinstance(value, dict):
        return _compact_debug_dict(value, 4000)
    return _truncate_debug_text(value, max_chars)


def _record_subturn_debug_event(event: dict[str, Any]) -> dict[str, Any]:
    global _subturn_debug_next_id
    now = datetime.now(UTC).isoformat(timespec="seconds")
    with _subturn_debug_lock:
        _subturn_debug_next_id += 1
        record = {
            "id": _subturn_debug_next_id,
            "timestamp": now,
            **event,
        }
        _subturn_debug_events.append(record)
        if len(_subturn_debug_events) > _SUBTURN_DEBUG_MAX_EVENTS:
            del _subturn_debug_events[: len(_subturn_debug_events) - _SUBTURN_DEBUG_MAX_EVENTS]
        return dict(record)


def _debug_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _debug_number_at(value: Any, path: tuple[str, ...]) -> int | float | None:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, (int, float)):
        return current
    parsed = _debug_int(current)
    return parsed


def _subturn_event_provider_call(event: dict[str, Any]) -> int | None:
    if event.get("source") == "subturn_meter":
        return _debug_int(event.get("provider_calls"))
    data = event.get("data")
    if isinstance(data, dict):
        call = _debug_int(data.get("provider_call"))
        if call is not None:
            return call
        call = _debug_int(data.get("provider_calls"))
        if call is not None:
            return call
    return _debug_int(event.get("provider_calls"))


def _subturn_call_group_key(event: dict[str, Any], provider_call: int) -> str:
    user_turn_key = event.get("user_turn_key")
    if isinstance(user_turn_key, str) and user_turn_key.strip():
        return f"{user_turn_key.strip()}:{provider_call}"
    data = event.get("data")
    if isinstance(data, dict):
        data_turn_key = data.get("user_turn_key")
        if isinstance(data_turn_key, str) and data_turn_key.strip():
            return f"{data_turn_key.strip()}:{provider_call}"
    return f"call:{provider_call}"


def _subturn_saved_pct(legacy_tokens: int | float | None, candidate_tokens: int | float | None) -> float | None:
    if legacy_tokens is None or candidate_tokens is None:
        return None
    if legacy_tokens <= 0:
        return None
    saved = max(0.0, float(legacy_tokens) - float(candidate_tokens))
    return round((saved / float(legacy_tokens)) * 1000.0) / 10.0


def _subturn_debug_calls(events_chronological: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for event in events_chronological:
        provider_call = _subturn_event_provider_call(event)
        if provider_call is None or provider_call <= 0:
            continue
        key = _subturn_call_group_key(event, provider_call)
        group = groups.setdefault(
            key,
            {
                "key": key,
                "call": provider_call,
                "event_ids": [],
                "route": None,
                "mode": None,
                "legacy_tokens": None,
                "actual_message_tokens": None,
                "candidate_tokens": None,
                "legacy_messages": None,
                "actual_messages": None,
                "candidate_messages": None,
                "actual_total_tokens": None,
                "actual_breakdown": None,
                "actual_estimated_total": None,
                "elapsed_seconds": None,
                "actual_summary": "",
            },
        )
        group["event_ids"].append(event.get("id"))
        group["latest_event_id"] = event.get("id")
        group["timestamp"] = event.get("timestamp")
        user_turn_key = event.get("user_turn_key")
        if isinstance(user_turn_key, str) and user_turn_key.strip():
            group["user_turn_key"] = user_turn_key.strip()

        source = event.get("source")
        if source == "subturn_meter":
            group["meter_event_id"] = event.get("id")
            group["actual_message_tokens"] = _debug_number_at(event, ("tokens", "call_input")) or group["actual_message_tokens"]
            group["actual_total_tokens"] = _debug_number_at(event, ("usage", "total")) or group["actual_total_tokens"]
            group["elapsed_seconds"] = event.get("elapsed_seconds")
            group["actual_summary"] = event.get("actual_summary", "")
            continue

        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        candidate = event.get("candidate") if isinstance(event.get("candidate"), dict) else {}
        legacy = event.get("legacy") if isinstance(event.get("legacy"), dict) else {}
        if event.get("event") in {"candidate_state", "candidate_payload"}:
            group["candidate_event_id"] = event.get("id")
            group["route"] = data.get("route") or group["route"]
            group["mode"] = data.get("mode") or group["mode"]
            group["legacy_messages"] = legacy.get("messages") if isinstance(legacy, dict) else group["legacy_messages"]
            group["candidate_messages"] = (
                candidate.get("messages") if isinstance(candidate, dict) else group["candidate_messages"]
            )
            legacy_after = _debug_number_at(legacy, ("tokens", "after_messages"))
            legacy_total = _debug_number_at(legacy, ("tokens", "total"))
            actual_tokens = _debug_number_at(data, ("actual_message_tokens",))
            candidate_total = _debug_number_at(candidate, ("tokens", "total"))
            group["legacy_tokens"] = legacy_after or legacy_total or group["legacy_tokens"]
            group["actual_message_tokens"] = actual_tokens or group["actual_message_tokens"]
            group["state_carry_enabled"] = data.get("subturn_state_carry_enabled")
            group["state_carry_applied"] = data.get("state_carry_applied")
            group["state_carry_recent_messages"] = data.get("subturn_state_carry_recent_messages")
            group["candidate_tokens"] = candidate_total or group["candidate_tokens"]
            actual_breakdown = data.get("actual_breakdown")
            if isinstance(actual_breakdown, dict):
                group["actual_breakdown"] = actual_breakdown
                group["actual_estimated_total"] = _debug_number_at(actual_breakdown, ("totals", "estimated_total"))

    calls = list(groups.values())
    for group in calls:
        group["saved_pct"] = _subturn_saved_pct(group.get("legacy_tokens"), group.get("candidate_tokens"))
        group["actual_saved_pct"] = _subturn_saved_pct(group.get("legacy_tokens"), group.get("actual_message_tokens"))
    calls.sort(key=lambda item: (_debug_int(item.get("latest_event_id")) or 0), reverse=True)
    return calls


def _subturn_debug_state() -> dict[str, Any]:
    with _subturn_debug_lock:
        events_chronological = [dict(event) for event in _subturn_debug_events]
    events = [dict(event) for event in reversed(events_chronological)]
    calls = _subturn_debug_calls(events_chronological)
    latest = events[0] if events else None
    return {
        "ok": True,
        "count": len(events),
        "max_events": _SUBTURN_DEBUG_MAX_EVENTS,
        "latest": latest,
        "latest_call": calls[0] if calls else None,
        "calls": calls,
        "events": events,
    }


def _render_subturn_debug_html() -> str:
    state = _subturn_debug_state()
    cache_text, compress_text = _subturn_debug_initial_metric_text(state)
    initial_state = json.dumps(state, ensure_ascii=False, default=str).replace("</", "<\\/")
    return (
        _SUBTURN_DEBUG_HTML.replace("__INITIAL_CACHE_SUMMARY__", html_lib.escape(cache_text, quote=False))
        .replace("__INITIAL_COMPRESS_SUMMARY__", html_lib.escape(compress_text, quote=False))
        .replace("__INITIAL_STATE_JSON__", initial_state)
    )


def _subturn_debug_initial_metric_text(state: dict[str, Any]) -> tuple[str, str]:
    latest_call = state.get("latest_call")
    if isinstance(latest_call, dict):
        return _subturn_debug_call_metric_text(state, latest_call)
    latest = state.get("latest")
    if isinstance(latest, dict):
        return (
            _subturn_debug_cache_text(latest.get("cache_probe") if isinstance(latest.get("cache_probe"), dict) else None),
            _subturn_debug_compression_text(
                latest.get("compression") if isinstance(latest.get("compression"), dict) else None
            ),
        )
    return "cache: -", "compress: —"


def _subturn_debug_call_metric_text(state: dict[str, Any], call: dict[str, Any]) -> tuple[str, str]:
    meter_event = _subturn_debug_meter_event_for_call(state, call)
    cache_probe = meter_event.get("cache_probe") if isinstance(meter_event.get("cache_probe"), dict) else None
    compression = meter_event.get("compression") if isinstance(meter_event.get("compression"), dict) else None
    return (
        _subturn_debug_cache_text(cache_probe, _subturn_debug_prefix_mark(state, call, cache_probe)),
        _subturn_debug_compression_text(compression),
    )


def _subturn_debug_meter_event_for_call(state: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    meter_event_id = _debug_int(call.get("meter_event_id"))
    if meter_event_id is None:
        return {}
    for event in state.get("events") or []:
        if isinstance(event, dict) and _debug_int(event.get("id")) == meter_event_id:
            return event
    return {}


def _subturn_debug_prefix_mark(
    state: dict[str, Any],
    call: dict[str, Any],
    cache_probe: dict[str, Any] | None,
) -> str:
    hash_value = cache_probe.get("stable_prefix_hash") if cache_probe else None
    if not isinstance(hash_value, str) or not hash_value:
        return ""
    previous = _subturn_debug_previous_call_for_prefix(state, call)
    if not previous:
        return ""
    previous_event = _subturn_debug_meter_event_for_call(state, previous)
    previous_probe = previous_event.get("cache_probe") if isinstance(previous_event.get("cache_probe"), dict) else None
    previous_hash = previous_probe.get("stable_prefix_hash") if previous_probe else None
    if not isinstance(previous_hash, str) or not previous_hash:
        return ""
    return "=" if previous_hash == hash_value else "≠"


def _subturn_debug_previous_call_for_prefix(state: dict[str, Any], call: dict[str, Any]) -> dict[str, Any] | None:
    current_call = _debug_int(call.get("call"))
    if current_call is None:
        return None
    current_turn = call.get("user_turn_key")
    candidates = []
    for item in state.get("calls") or []:
        if not isinstance(item, dict) or item is call:
            continue
        item_call = _debug_int(item.get("call"))
        if item_call is None or item_call >= current_call:
            continue
        item_turn = item.get("user_turn_key")
        if isinstance(current_turn, str) and current_turn and isinstance(item_turn, str) and item_turn != current_turn:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: _debug_int(item.get("call")) or 0, reverse=True)[0]


def _subturn_debug_cache_text(cache_probe: dict[str, Any] | None, prefix_mark: str = "") -> str:
    if not cache_probe:
        return "cache: -"
    meter = str(cache_probe.get("cache_meter") or "").strip()
    if meter == "unreported":
        return "cache: unreported"
    if not meter:
        return "cache: -"
    hit_pct = _subturn_debug_cache_pct(cache_probe.get("cache_hit_pct"))
    text = f"cache: {meter}{f' {hit_pct}%' if hit_pct else ''}"
    hash_value = cache_probe.get("stable_prefix_hash")
    if isinstance(hash_value, str) and hash_value:
        text += f" | hash {hash_value[:8]}{prefix_mark}"
    return text


def _subturn_debug_cache_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    percent = round(float(value) * 1000.0) / 10.0
    return str(int(percent)) if percent.is_integer() else f"{percent:.1f}"


def _subturn_debug_compression_text(compression: dict[str, Any] | None) -> str:
    outputs = _debug_int(compression.get("compressed_tool_outputs") if compression else None) or 0
    if outputs <= 0:
        return "compress: —"
    saved = _debug_int(compression.get("compression_saved_tokens_est") if compression else None) or 0
    noun = "output" if outputs == 1 else "outputs"
    return f"compress: {outputs} {noun}, ~{_compact_tokens(saved)} saved"


def get_agent() -> JarvisAgentic | None:
    global _agent
    global _agent_last_error
    global _agent_last_error_type
    global _agent_last_error_repr
    global _agent_last_error_filename
    global _agent_last_error_traceback
    global _agent_last_error_ts
    if _agent is not None:
        return _agent

    now = datetime.now(UTC)
    with _agent_guard:
        if _agent is not None:
            return _agent
        next_retry_at = _agent_next_retry_at()
        if next_retry_at is not None and now < next_retry_at:
            return None
        try:
            _agent = JarvisAgentic()
        except Exception as exc:  # noqa: BLE001
            _agent_last_error = str(exc)
            _agent_last_error_type = type(exc).__name__
            _agent_last_error_repr = repr(exc)
            _agent_last_error_filename = str(getattr(exc, "filename", "") or "") or None
            _agent_last_error_traceback = "".join(traceback.format_exception(exc))[-4000:]
            _agent_last_error_ts = now
            return None
        _agent_last_error = None
        _agent_last_error_type = None
        _agent_last_error_repr = None
        _agent_last_error_filename = None
        _agent_last_error_traceback = None
        _agent_last_error_ts = None
        return _agent


async def close_agent() -> None:
    global _agent
    with _agent_guard:
        agent = _agent
        _agent = None
    if agent is not None:
        await agent.close()


def token_estimate(text: str) -> int:
    return max(0, len(text) // 4)


def _runtime_now() -> datetime:
    timezone_name = os.environ.get("JARVIS_TIMEZONE") or os.environ.get("TZ") or "Asia/Seoul"
    try:
        return datetime.now(ZoneInfo(timezone_name))
    except Exception:
        return datetime.now().astimezone()


def _agent_next_retry_at() -> datetime | None:
    if _agent_last_error_ts is None:
        return None
    return _agent_last_error_ts + timedelta(seconds=_agent_retry_interval)


def _agent_status_fields() -> dict[str, str | None]:
    next_retry_at = _agent_next_retry_at()
    return {
        "last_agent_error": _agent_last_error,
        "last_agent_error_type": _agent_last_error_type,
        "last_agent_error_repr": _agent_last_error_repr,
        "last_agent_error_filename": _agent_last_error_filename,
        "last_agent_error_traceback_tail": _agent_last_error_traceback,
        "next_retry_at": next_retry_at.isoformat() if next_retry_at is not None else None,
    }


def _effective_session_id(bench_conv_id: str | None) -> str:
    candidate = str(bench_conv_id or "").strip()
    return candidate or _SESSION_ID


def _note_session_mode(session_id: str) -> None:
    global _last_bench_conv_id
    _last_bench_conv_id = session_id if session_id != _SESSION_ID else None


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "service": "jarvis-jlc-sidecar", "agent_loaded": _agent is not None}


@app.get("/status")
def status() -> dict[str, Any]:
    ensure_sidecar_config()
    config = load_runtime_config()
    default_project_root = get_effective_project_root()
    protected_roots = get_protected_roots()
    roles = {
        role: _summarize_role(role, (config.get("roles") or {}).get(role))
        for role in ("chat", "subagent", "encoder")
    }
    return {
        "ok": True,
        "service": "jarvis-jlc-sidecar",
        "agent_loaded": _agent is not None,
        "roles": roles,
        "mode": "bench" if _last_bench_conv_id else "default",
        "bench_conv_id": _last_bench_conv_id,
        "default_project_root": default_project_root,
        "configured_project_root": get_default_project_root(),
        "project_root_source": get_project_root_source(),
        "internal_memory_root": str(internal_memory_root().resolve()),
        "sidecar_app_file": str(Path(__file__).resolve()),
        "python_executable": sys.executable,
        "process_id": os.getpid(),
        "process_cwd": os.getcwd(),
        "protected_roots": protected_roots,
        "setup_required": not bool(default_project_root),
        **router.status_fields(),
        **router.registry.status_fields(),
        **_agent_status_fields(),
    }


@app.post("/context")
def context(req: ContextRequest) -> dict[str, Any]:
    session_id = _effective_session_id(req.bench_conv_id)
    _note_session_mode(session_id)
    print(
        f"[jarvis-sidecar] /context session={session_id} mode={req.mode} "
        f"user_len={len(req.user_message or '')}",
        file=sys.stderr,
        flush=True,
    )
    setup_command_path = parse_setup_default_root_command(req.user_message)
    if setup_command_path:
        setup_default_project_root(setup_command_path)
    cwd_hint = req.cwd_hint or req.cwd or req.hints.get("cwd")
    active_project_hint = None if req.mode == "chat" else (req.active_project_path or req.project_path)
    selected, warnings, trace = router.select(
        user_message=req.user_message,
        cwd_hint=str(cwd_hint) if cwd_hint else None,
        active_project_path=str(active_project_hint) if active_project_hint else None,
        mode=req.mode,
    )
    memory_project_path = selected.path if selected else (None if req.mode == "chat" else req.project_path)
    if memory_project_path:
        guarded_path = _guard_memory_write_path(memory_project_path)
        if guarded_path is None:
            warnings.append(f"active project path not registered: {memory_project_path}")
            memory_project_path = None
        else:
            memory_project_path = guarded_path

    file_states = ensure_workspace_memory(memory_project_path)
    project_memory, project_warnings = read_project_memory(memory_project_path, max_chars=60000 if selected else 12000)
    warnings.extend(project_warnings)
    if (
        req.mode != "chat"
        and memory_project_path
        and project_memory.strip()
    ):
        turn_key = req.context_turn_key or req.user_message.strip() or "unknown-turn"
        log_key = f"{session_id}\0{turn_key}\0{memory_project_path}"
        if log_key not in _project_memory_load_log_keys:
            _project_memory_load_log_keys.add(log_key)
            if len(_project_memory_load_log_keys) > 2000:
                _project_memory_load_log_keys.clear()
                _project_memory_load_log_keys.add(log_key)
            _print_project_memory_loaded(
                project_name=selected.name if selected else None,
                project_path=memory_project_path,
                tokens=token_estimate(project_memory),
            )

    jhb = ""
    recall_block = ""
    memory_mode = "light"
    try:
        agent = get_agent()
        if agent is None:
            detail = f": {_agent_last_error}" if _agent_last_error else ""
            raise RuntimeError(f"JLC agent unavailable{detail}")
        agent.wait_for_pending_encode(timeout=2.0, session_id=session_id)
        jhb = agent.render_jhb(session_id=session_id)
        memory_mode = "full"
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"JLC context degraded: {exc}")

    # Pre-inject recall so the main LLM usually does not need a second
    # internal turn just to call recall_turns. Prefer the same hybrid
    # retriever used by /recall; raw keyword recall is only a degraded
    # fallback. Keeping /context and recall_turns on the same retrieval path is
    # important for personal facts that live in the conversation store rather
    # than the repo-local raw-store.
    try:
        _recall_top_k = max(0, int(os.environ.get("JARVIS_AUTO_RECALL_TOP_K", "10")))
    except ValueError:
        _recall_top_k = 10
    auto_recall_enabled = str(req.mode or "").strip().lower() == "chat"
    if auto_recall_enabled and _recall_top_k > 0 and req.user_message.strip():
        try:
            agent = get_agent()
            if agent is None:
                detail = f": {_agent_last_error}" if _agent_last_error else ""
                raise RuntimeError(f"JLC agent unavailable{detail}")
            recall_result = agent.recall_for_query(
                req.user_message,
                top_k=_recall_top_k,
                timeout=4.0,
                min_confidence="LOW",
                session_id=session_id,
            )
            recall_block = str(recall_result.get("text") or "").strip()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"JLC auto recall degraded: {exc}")
            recall_block = ""
        raw_recall_block = _format_raw_hits(recall_raw(req.user_message, top_k=_recall_top_k, session_id=session_id))
        if raw_recall_block:
            recall_block = (
                f"{raw_recall_block}\n\n{recall_block}"
                if recall_block and raw_recall_block not in recall_block
                else raw_recall_block
            )

    # Backstop for stale JHB: include the last verbatim turn so the LLM
    # can answer immediate follow-ups even when the encoder is mid-flight.
    # Controlled by JARVIS_RECENT_TURNS env (default 1 for normal chat).
    # jarvis.ps1 exposes this via `--recent-turns N`.
    try:
        _recent_limit = max(0, int(os.environ.get("JARVIS_RECENT_TURNS", "1")))
    except ValueError:
        _recent_limit = 1
    recent_raw_block = _format_recent_turns(recent_turns(limit=_recent_limit, session_id=session_id))
    workspace_block = _build_workspace_block(selected)

    context_block = _build_context_block(
        project_path=memory_project_path,
        code_path=selected.code_path if selected else None,
        jhb=jhb,
        project_memory=project_memory,
        recall_block=recall_block,
        recent_raw=recent_raw_block,
        trace=trace,
        warnings=warnings,
        memory_mode=memory_mode,
    )
    return {
        "context": context_block,
        "workspace_block": workspace_block,
        "recall_block": recall_block,
        "recent_raw_block": recent_raw_block,
        "active_project_path": memory_project_path,
        "project_id": selected.project_id if selected else None,
        "project_name": selected.name if selected else None,
        "code_path": selected.code_path if selected else None,
        "memory_mode": memory_mode,
        "context_tokens": token_estimate(context_block),
        "jhb_tokens": token_estimate(jhb),
        "project_tokens": token_estimate(project_memory),
        "recall_tokens": token_estimate(recall_block),
        "recent_raw_tokens": token_estimate(recent_raw_block),
        "warnings": warnings,
        "trace": trace,
        "memory_files": file_states,
        "default_project_root": get_effective_project_root(),
        "configured_project_root": get_default_project_root(),
        "project_root_source": get_project_root_source(),
        "internal_memory_root": str(internal_memory_root().resolve()),
        "protected_roots": get_protected_roots(),
        "setup_required": not bool(get_effective_project_root()),
    }


@app.post("/resolve_project_by_path")
def resolve_project_by_path(req: ResolvePathRequest) -> dict[str, Any]:
    project = router.resolve_by_path(req.path)
    if project is None:
        return {"ok": True, "project": None}
    return {
        "ok": True,
        "project": {
            "project_id": project.project_id,
            "name": project.name,
            "slug": project.slug,
            "path": project.path,
            "code_path": project.code_path,
        },
    }


@app.get("/projects")
def list_projects() -> dict[str, Any]:
    registry_status = router.registry.status_fields()
    if not registry_status["registry_ok"]:
        return {"ok": False, "projects": [], **registry_status}
    projects = []
    for project in router.registry.all():
        projects.append(
            {
                "project_id": project.project_id,
                "name": project.name,
                "slug": project.slug,
                "path": project.path,
                "code_path": project.code_path,
            }
        )
    return {"ok": True, "projects": projects, **registry_status}


@app.post("/route_turn")
async def route_turn(req: RouteTurnRequest) -> dict[str, Any]:
    """Use the configured chat model as the first-pass JARVIS route judge."""
    user_message = (req.user_message or "").strip()
    if not user_message:
        return _route_turn_fallback("chat", "empty user message")
    projects = _route_project_summaries()
    system = _route_turn_system_prompt()
    user = _route_turn_user_prompt(req, projects)
    try:
        llm = get_llm("chat")
        raw = await llm.chat(
            system=system,
            user=user,
            max_tokens=900,
            reasoning_effort="none",
        )
        parsed = _extract_json_object(raw)
        decision = _normalize_route_decision(parsed, raw=raw)
        decision["ok"] = True
        return decision
    except Exception as exc:  # noqa: BLE001
        return {
            **_route_turn_fallback("chat", f"chat LLM router unavailable: {exc}"),
            "ok": False,
            "error": str(exc),
        }


@app.post("/turn")
def turn(req: TurnRequest) -> dict[str, Any]:
    session_id = _effective_session_id(req.bench_conv_id)
    _note_session_mode(session_id)
    print(
        f"[jarvis-sidecar] /turn session={session_id} "
        f"user_len={len(req.user_message or '')} assistant_len={len(req.assistant_message or '')}",
        file=sys.stderr,
        flush=True,
    )
    project_path = _guard_memory_write_path(req.project_path)
    ensure_workspace_memory(project_path)
    raw_path = append_raw_turn(
        project_path=project_path,
        user_message=req.user_message,
        assistant_message=req.assistant_message,
        tool_events=req.tool_events,
        llm_meta=req.llm_meta,
        session_id=session_id,
    )
    light_result = {"updated": [], "mode": "light"}
    try:
        agent = get_agent()
        if agent is None:
            detail = f": {_agent_last_error}" if _agent_last_error else ""
            raise RuntimeError(f"JLC agent unavailable{detail}")
        prev_jhb = agent.render_jhb(session_id=session_id)

        # Inject chat token counts into encoder before encode
        encoder = getattr(agent, "encoder", None)
        slim = getattr(agent, "slim", None)
        usage = (req.llm_meta or {}).get("usage", {})
        if encoder is not None and usage:
            reasoning_tokens = usage.get("reasoningTokens") or usage.get("reasoning_tokens") or usage.get("thought") or 0
            chat_seconds = (req.llm_meta or {}).get("chat_seconds")
            encoder.last_chat_in = usage.get("input", 0)
            encoder.last_chat_out = usage.get("output", 0)
            encoder.last_chat_cache_read = usage.get("cacheRead", 0)
            encoder.last_chat_cache_write = usage.get("cacheWrite", 0)
            encoder.last_chat_think = int(reasoning_tokens or 0)
            encoder.last_chat_turn_in = encoder.last_chat_in
            encoder.last_chat_turn_out = encoder.last_chat_out
            encoder.last_chat_turn_cache_read = encoder.last_chat_cache_read
            encoder.last_chat_turn_cache_write = encoder.last_chat_cache_write
            encoder.last_chat_turn_think = encoder.last_chat_think
            encoder.last_chat_turn_seconds = float(chat_seconds or 0.0)
            encoder.last_chat_seconds = encoder.last_chat_turn_seconds
            # Note: chat_seconds and breakdown not available from client
            prompt_context = (req.llm_meta or {}).get("prompt_context")
            if isinstance(prompt_context, dict):
                encoder.last_chat_prompt_context = {
                    "context_tokens": int(prompt_context.get("context_tokens", 0) or 0),
                    "jhb_tokens": int(prompt_context.get("jhb_tokens", 0) or 0),
                    "project_tokens": int(prompt_context.get("project_tokens", 0) or 0),
                    "recall_tokens": int(prompt_context.get("recall_tokens", 0) or 0),
                }
            else:
                encoder.last_chat_prompt_context = {}
            prompt_breakdown = (req.llm_meta or {}).get("prompt_breakdown")
            if isinstance(prompt_breakdown, dict):
                encoder.last_chat_in_breakdown = {
                    str(key): int(value or 0)
                    for key, value in prompt_breakdown.items()
                    if isinstance(key, str)
                }
            else:
                encoder.last_chat_in_breakdown = {}

        encode_done = threading.Event()
        encoder_summary: dict[str, Any] | None = None

        def _on_encode_done(_updated_jhb: str) -> None:
            nonlocal encoder_summary
            try:
                encoder = getattr(agent, "encoder", None)
                turn_id = getattr(encoder, "last_enc_turn_id", None)
                if isinstance(turn_id, int) and turn_id > 0:
                    encoder_meta = {
                        "enc_in": getattr(encoder, "last_enc_in", 0),
                        "enc_think": getattr(encoder, "last_enc_think", 0),
                        "enc_out": getattr(encoder, "last_enc_out", 0),
                        "enc_seconds": getattr(encoder, "last_enc_seconds", 0.0),
                        "jhb_tokens": getattr(encoder, "last_jhb_tokens", 0),
                        "jhb_delta": getattr(encoder, "last_jhb_delta", 0),
                        "jhb_delta_tokens": getattr(encoder, "last_jhb_delta_tokens", 0),
                        "jhb_delta_chars": getattr(encoder, "last_jhb_delta_chars", 0),
                        "jhb_diff_added": getattr(encoder, "last_jhb_diff_added", 0),
                        "jhb_diff_removed": getattr(encoder, "last_jhb_diff_removed", 0),
                        "failure_mode": getattr(encoder, "last_failure_mode", "not_reported"),
                        "encoder_retries": getattr(encoder, "last_retries", 0),
                    }
                    append_encoder_turn(
                        turn_id=turn_id,
                        project_path=req.project_path,
                        encoder_meta=encoder_meta,
                        session_id=session_id,
                    )
                    encoder_summary = {
                        "turn": turn_id,
                        **encoder_meta,
                    }

                    # PI polling cache
                import time
                record = {
                    "turn": int(turn_id) if isinstance(turn_id, int) else 0,
                    "enc_out": int(getattr(encoder, "last_enc_out", 0) or 0) if encoder else 0,
                    "enc_seconds": float(getattr(encoder, "last_enc_seconds", 0) or 0) if encoder else 0,
                    "jhb_tokens": int(getattr(encoder, "last_jhb_tokens", 0) or 0) if encoder else 0,
                    "jhb_delta": int(getattr(encoder, "last_jhb_delta", 0) or 0) if encoder else 0,
                    "error": str(getattr(encoder, "last_error", "") or "") if encoder else "",
                    "ts": time.time(),
                }
                with _encoding_results_lock:
                    _encoding_results[session_id] = record
            except Exception as exc:  # noqa: BLE001
                print(f"[jarvis-sidecar] encoder raw append failed: {exc}", file=sys.stderr)
            finally:
                _finish_encoder_token_print()
                encode_done.set()

        scheduled_turn = agent.encode_and_save_async(
            project_path=req.project_path,
            user_msg=req.user_message,
            assistant_msg=req.assistant_message,
            llm_meta={**req.llm_meta, "tool_events": req.tool_events},
            on_done=_on_encode_done,
            session_id=session_id,
        )
        print(
            f"[jarvis-sidecar] /turn scheduled encoder session={session_id}",
            file=sys.stderr,
            flush=True,
        )

    except Exception as exc:  # noqa: BLE001
        print(f"[jarvis-sidecar] /turn encoder schedule failed: {exc}", file=sys.stderr, flush=True)
        return {
            "ok": True,
            "memory_mode": "light",
            "scheduled_encode": False,
            "raw_saved": True,
            "raw_path": str(raw_path),
            "light_memory": light_result,
            
            "warning": str(exc),
        }
    return {
        "ok": True,
        "memory_mode": "full",
        "scheduled_encode": True,
        "scheduled_turn": scheduled_turn,
        "raw_saved": True,
        "raw_path": str(raw_path),
        "light_memory": light_result,
        
    }


@app.get("/encoding_status")
def encoding_status(conv_id: str, clear: bool = False, min_turn: int | None = None) -> dict[str, Any]:
    session_id = _effective_session_id(conv_id)
    with _encoding_results_lock:
        record = _encoding_results.get(session_id)
        if record is not None and min_turn is not None:
            record_turn = int(record.get("turn") or 0)
            if record_turn < min_turn:
                if clear:
                    _encoding_results.pop(session_id, None)
                return {"ok": True, "ready": False, "stale_turn": record_turn}
        if record is not None and clear:
            _encoding_results.pop(session_id, None)
    if record is None:
        return {"ok": True, "ready": False}
    # Surface encoder identity on every poll so the pi footer can mirror the
    # chat-model badge ("(provider) model effort") for the encoder role.
    # Sourced directly from the active config + env so this works even when
    # the record dict was assembled before these keys existed.
    try:
        from jlc_agentic.slim import _read_encoder_model_spec
        enc_model_spec = _read_encoder_model_spec(None)
    except Exception:
        enc_model_spec = ""
    # Encoder requests are always sent with reasoning disabled.
    enc_reasoning_effort = "none"
    return {
        "ok": True,
        "ready": True,
        "enc_out": record["enc_out"],
        "enc_seconds": record["enc_seconds"],
        "jhb_tokens": record["jhb_tokens"],
        "jhb_delta": record["jhb_delta"],
        "error": record.get("error") or None,
        "enc_model_spec": enc_model_spec,
        "enc_reasoning_effort": enc_reasoning_effort,
        "ts": record["ts"],
    }


@app.post("/recall")
def recall(req: RecallRequest) -> dict[str, Any]:
    session_id = _effective_session_id(req.bench_conv_id)
    _note_session_mode(session_id)
    results = []
    warnings = []
    for query in req.queries:
        if _is_explicit_raw_recall_query(query):
            raw_hits = recall_raw(query, top_k=req.top_k, session_id=session_id)
            results.append({
                "query": query,
                "text": _format_raw_hits(raw_hits),
                "fragments": _raw_hits_to_fragments(raw_hits),
                "confidence": "HIGH" if raw_hits else "LOW",
                "source": "raw_explicit",
            })
            continue
        try:
            agent = get_agent()
            if agent is None:
                raise RuntimeError("JLC agent unavailable")
            recall_result = agent.recall_for_query(
                query,
                top_k=req.top_k,
                timeout=8.0,
                min_confidence="LOW",
                session_id=session_id,
            )
            text = recall_result.get("text", "")
            fragments = recall_result.get("fragments", [])
            confidence = recall_result.get("confidence", "LOW")
            if not text and not fragments:
                raw_hits = recall_raw(query, top_k=req.top_k, session_id=session_id)
                text = _format_raw_hits(raw_hits)
                fragments = _raw_hits_to_fragments(raw_hits)
                confidence = "LOW"
                if raw_hits:
                    results.append({
                        "query": query, "text": text,
                        "fragments": fragments, "confidence": confidence,
                        "source": "raw_fallback",
                    })
                else:
                    results.append({"query": query, "text": "", "fragments": [], "confidence": "LOW"})
            else:
                results.append({
                    "query": query, "text": text,
                    "fragments": fragments, "confidence": confidence,
                })
        except Exception as exc:  # noqa: BLE001
            raw_hits = recall_raw(query, top_k=req.top_k, session_id=session_id)
            if raw_hits:
                results.append({
                    "query": query,
                    "text": _format_raw_hits(raw_hits),
                    "fragments": _raw_hits_to_fragments(raw_hits),
                    "confidence": "LOW",
                    "source": "raw_fallback",
                })
                warnings.append(f"JLC recall degraded for query {query!r}: {exc}")
            else:
                warnings.append(f"recall failed for query {query!r}: {exc}")
    return {"ok": bool(results) or not warnings, "results": results, "warnings": warnings}


@app.post("/web_search")
def web_search(req: WebSearchRequest) -> dict[str, Any]:
    top_k = max(1, min(int(req.top_k), 20))
    return brave_web_search(req.query, top_k=top_k)


@app.post("/web_fetch")
def web_fetch(req: WebFetchRequest) -> dict[str, Any]:
    return web_fetch_tool(req.url, max_chars=req.max_chars, timeout_sec=req.timeout_sec)


@app.post("/docs_search")
def docs_search(req: DocsSearchRequest) -> dict[str, Any]:
    return docs_search_tool(
        req.query,
        search_handler=brave_web_search,
        domains=req.domains,
        top_k=req.top_k,
        fetch_top=req.fetch_top,
        max_chars=req.max_chars,
    )


@app.post("/package_info")
def package_info(req: PackageInfoRequest) -> dict[str, Any]:
    return package_info_tool(req.ecosystem, req.package, include_release_notes=req.include_release_notes)


@app.get("/credentials/catalog")
def credentials_catalog() -> dict[str, Any]:
    """Credential targets used by the Pi /api-key command."""
    load_credentials_into_env()
    catalog = llm_load_catalog()
    targets: dict[str, dict[str, Any]] = {}
    for pid, cfg in catalog.get("providers", {}).items():
        env_name = cfg.get("auth_env")
        if not isinstance(env_name, str) or not env_name.strip():
            continue
        targets[pid] = {
            "label": cfg.get("label", pid),
            "env_name": env_name,
            "kind": "llm",
            "configured": bool(os.environ.get(env_name, "").strip()),
        }
    targets["brave-search"] = {
        "label": "Brave Search",
        "env_name": "BRAVE_SEARCH_API_KEY",
        "kind": "web_search",
        "configured": bool(os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()),
    }
    return {"ok": True, "targets": targets, "credentials_path": str(credentials_path())}


@app.post("/credentials/set")
def credentials_set(req: CredentialSetRequest) -> dict[str, Any]:
    env_name = req.env_name.strip()
    value = req.value.strip()
    if not env_name or not value:
        return {"ok": False, "error": "env_name and value are required"}
    try:
        path = save_credential_env(env_name, value)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    validation = _validate_credential_env(env_name) if req.do_validate else {"ok": True, "skipped": True}
    try:
        clear_cache()
        _init_provider_router()
    except Exception:
        pass
    return {
        "ok": bool(validation.get("ok")),
        "env_name": env_name,
        "credentials_path": str(path),
        "validation": validation,
    }


def _validate_credential_env(env_name: str) -> dict[str, Any]:
    if env_name == "BRAVE_SEARCH_API_KEY":
        result = brave_web_search("jarvis code", top_k=1)
        if result.get("ok"):
            return {"ok": True, "provider": "brave-search"}
        return {"ok": False, "error": result.get("error", "Brave validation failed")}

    catalog = llm_load_catalog()
    matched = [
        pid for pid, cfg in catalog.get("providers", {}).items()
        if cfg.get("auth_env") == env_name
    ]
    if not matched:
        return {"ok": True, "warning": "saved; no live validator for this env"}
    for pid in matched:
        models = llm_fetch_all({"providers": {pid: catalog["providers"][pid]}}).get(pid)
        if models:
            return {"ok": True, "provider": pid, "models": len(models)}
    return {"ok": False, "error": f"saved, but validation failed for: {', '.join(matched)}"}


@app.post("/register_project")
def register_project(req: RegisterProjectRequest) -> dict[str, Any]:
    name = req.name or req.path.rstrip("/\\").split("\\")[-1].split("/")[-1] or "project"
    try:
        project = router.create_project(name, code_path=req.path)
        project, warnings = router.switch_project(project.slug, code_path=project.code_path, auto_create=False)
    except RegistryCorruptError as exc:
        return {"ok": False, "error": str(exc), **router.registry.status_fields()}
    if project is None:
        return {"ok": False, "warnings": warnings}
    guarded_project = _registered_memory_project(project.path)
    if guarded_project is None:
        return {"ok": False, "error": "registered project disappeared before memory seed"}
    project = guarded_project
    file_states = ensure_workspace_memory(project.path)
    return {
        "ok": True,
        **_project_payload(project),
        "warnings": warnings
        + ([] if project.code_path == req.path else ["requested path redirected to a safe location"]),
        "jarvis_md": file_states.get("JARVIS.md", "missing"),
        "memory_files": file_states,
    }


@app.post("/unregister_project")
def unregister_project(req: UnregisterProjectRequest) -> dict[str, Any]:
    registry_status = router.registry.status_fields()
    if not registry_status["registry_ok"]:
        return {
            "ok": False,
            "error": f"workspace registry is corrupt: {router.registry.load_error}",
            **registry_status,
        }

    project = router.registry.get_by_id(req.project_id)
    if project is None:
        project = router.registry.get_by_path(req.path)
    if project is None and req.slug_or_name:
        matches = router.registry.get_by_slug_or_name(req.slug_or_name)
        if len(matches) > 1:
            return {
                "ok": False,
                "error": f"multiple projects match unregister target: {req.slug_or_name}",
                "candidates": [_project_payload(candidate) for candidate in matches],
                "remaining": _remaining_project_payload(),
            }
        project = matches[0] if matches else None

    if project is None:
        return {
            "ok": True,
            "removed": False,
            "warning": "project registration not found",
            "remaining": _remaining_project_payload(),
        }

    try:
        removed = router.registry.remove_project(project.project_id)
    except RegistryCorruptError as exc:
        return {"ok": False, "error": str(exc), **router.registry.status_fields()}
    if removed:
        router.clear_active_project_if(project.project_id)
    return {
        "ok": True,
        "removed": removed,
        "remaining": _remaining_project_payload(),
        **_project_payload(project),
        **router.status_fields(),
        **router.registry.status_fields(),
    }


@app.post("/switch_project")
def switch_project(req: SwitchRequest) -> dict[str, Any]:
    try:
        project, warnings = router.switch_project(req.slug_or_name, code_path=req.code_path, auto_create=req.auto_create)
    except RegistryCorruptError as exc:
        return {"ok": False, "error": str(exc), **router.registry.status_fields()}
    if project is None:
        return {"ok": False, "warnings": warnings}
    guarded_project = _registered_memory_project(project.path)
    if guarded_project is None:
        return {"ok": False, "error": "selected project disappeared before memory seed"}
    project = guarded_project
    file_states = ensure_workspace_memory(project.path)
    return {
        "ok": True,
        **_project_payload(project),
        "warnings": warnings,
        "memory_files": file_states,
    }


@app.post("/setup")
def setup(req: SetupRequest) -> dict[str, Any]:
    raw = setup_default_project_root(req.default_project_root)
    return {
        "ok": True,
        "default_project_root": get_effective_project_root(),
        "configured_project_root": get_default_project_root(),
        "project_root_source": get_project_root_source(),
        "internal_memory_root": str(internal_memory_root().resolve()),
        "protected_roots": raw.get("protected_roots", []),
        "setup_required": not bool(get_effective_project_root()),
    }


@app.post("/update_jarvis_md")
def update_jarvis_md(req: UpdateJarvisMdRequest) -> dict[str, Any]:
    project = _registered_memory_project(req.project_path)
    if project is None:
        return {
            "ok": False,
            "error": "project path is not registered",
            "project_path": req.project_path,
            "unregistered": True,
        }
    if req.updates is not None:
        result = update_project_jarvis_md_batch(project.path, updates=req.updates)
        _print_project_memory_updated(project_path=project.path, result=result)
        return result
    if req.field is None:
        return {"ok": False, "error": "field is required when updates is not provided"}
    result = update_project_jarvis_md(project.path, field=req.field, value=req.value)
    _print_project_memory_updated(project_path=project.path, result=result)
    return result


@app.post("/interrupt_checkpoint")
def interrupt_checkpoint(req: InterruptCheckpointRequest) -> dict[str, Any]:
    result = write_interrupt_checkpoint(
        req.project_path,
        user_message=req.user_message,
        assistant_message=req.assistant_message,
        tool_events=req.tool_events,
        subturn_log=req.subturn_log,
        mode=req.mode,
        cwd=req.cwd,
        reason=req.reason,
    )
    _print_project_memory_updated(project_path=req.project_path, result=result)
    return result


@app.post("/interrupt_checkpoint/clear")
def clear_interrupt_checkpoint_endpoint(req: ClearInterruptCheckpointRequest) -> dict[str, Any]:
    result = clear_interrupt_checkpoint(req.project_path)
    _print_project_memory_updated(project_path=req.project_path, result=result)
    return result


@app.post("/evidence/store")
def evidence_store_endpoint(req: EvidenceStoreRequest) -> dict[str, Any]:
    try:
        payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        result = store_evidence(payload)
    except EvidenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    if result.get("ok") is False:
        raise HTTPException(status_code=409, detail=result.get("error") or "evidence store failed")
    return result


@app.get("/evidence/{ref}")
def evidence_retrieve_endpoint(
    ref: str,
    start_line: int | None = Query(default=None),
    end_line: int | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return retrieve_evidence(ref, start_line=start_line, end_line=end_line)
    except EvidenceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/subturn_meter")
def subturn_meter(req: SubturnMeterRequest) -> dict[str, Any]:
    subturn_no = max(0, int(req.provider_calls))
    done = "done " if req.reason == "assistant_end" else ""
    actual_parts = []
    if req.usage_input_tokens is not None:
        actual_parts.append(f"in={_compact_tokens(max(0, int(req.usage_input_tokens)))}")
    if req.usage_output_tokens is not None:
        actual_parts.append(f"out={_compact_tokens(max(0, int(req.usage_output_tokens)))}")
    if req.usage_cache_read_tokens is not None:
        actual_parts.append(f"cached={_compact_tokens(max(0, int(req.usage_cache_read_tokens)))}")
    if req.usage_cache_write_tokens is not None:
        actual_parts.append(f"cache_write={_compact_tokens(max(0, int(req.usage_cache_write_tokens)))}")
    if req.usage_reasoning_tokens is not None:
        actual_parts.append(f"reasoning={_compact_tokens(max(0, int(req.usage_reasoning_tokens)))}")
    if req.usage_total_tokens is not None:
        actual_parts.append(f"total={_compact_tokens(max(0, int(req.usage_total_tokens)))}")
    actual_summary = f"actual[{', '.join(actual_parts)}]" if actual_parts else "actual[pending]"
    record = _record_subturn_debug_event(
        {
            "source": "subturn_meter",
            "reason": req.reason,
            "user_turn_key": _truncate_debug_text(req.user_turn_key, 200),
            "provider_calls": subturn_no,
            "elapsed_seconds": max(0.0, float(req.elapsed_seconds)),
            "actual_summary": actual_summary,
            "tokens": {
                "call_input": max(0, int(req.call_input_tokens)),
                "turn_input": max(0, int(req.turn_input_tokens)),
                "output": max(0, int(req.output_tokens)),
            },
            "usage": {
                "input": req.usage_input_tokens,
                "output": req.usage_output_tokens,
                "cache_read": req.usage_cache_read_tokens,
                "cache_write": req.usage_cache_write_tokens,
                "reasoning": req.usage_reasoning_tokens,
                "total": req.usage_total_tokens,
            },
            "cache_probe": {
                "provider_cache_read_tokens": req.provider_cache_read_tokens,
                "provider_cache_write_tokens": req.provider_cache_write_tokens,
                "cache_meter": req.cache_meter,
                "cache_hit_pct": req.cache_hit_pct,
                "stable_prefix_hash": _truncate_debug_text(req.stable_prefix_hash, 64),
                "stable_prefix_tokens_est": req.stable_prefix_tokens_est,
                "live_tokens_est": req.live_tokens_est,
            },
            "compression": {
                "compressed_tool_outputs": max(0, int(req.compressed_tool_outputs)),
                "compression_saved_tokens_est": max(0, int(req.compression_saved_tokens_est)),
                "compression_skips": req.compression_skips or None,
            },
            "payload_summary": _truncate_debug_text(req.payload_summary, 4000),
            "payload_preview": _truncate_debug_text(req.payload_preview),
            "subturn_summary": _truncate_debug_text(req.subturn_summary, 8000),
        }
    )
    trace_enabled = os.environ.get("JLC_PAYLOAD_TRACE_STDERR") == "1"
    meter_extras = ""
    comp_outputs = max(0, int(req.compressed_tool_outputs))
    comp_saved = max(0, int(req.compression_saved_tokens_est))
    if comp_outputs > 0 or comp_saved > 0:
        meter_extras += f" | saved={_compact_tokens(comp_saved)} ({comp_outputs} compressed)"
    if req.cache_meter == "actual" and req.cache_hit_pct is not None:
        meter_extras += f" | cache={req.cache_hit_pct * 100:.0f}%"
    elif req.cache_meter == "unreported":
        meter_extras += " | cache=unreported"
    print(
        f"\x1b[31m[jlc:subturn] #{subturn_no} {done}"
        f"{actual_summary} "
        f"{max(0.0, float(req.elapsed_seconds)):.1f}s"
        f"{meter_extras}"
        f"\x1b[0m",
        file=sys.stderr,
        flush=True,
    )
    if trace_enabled and req.payload_preview:
        print(
            f"\x1b[36m[jlc:payload] #{subturn_no}\n{req.payload_preview}\x1b[0m",
            file=sys.stderr,
            flush=True,
        )
    return {"ok": True, "debug_event_id": record["id"]}


@app.get("/debug/subturn", response_class=HTMLResponse)
def subturn_debug_page() -> HTMLResponse:
    return HTMLResponse(_render_subturn_debug_html())


@app.get("/debug/subturn/state")
def subturn_debug_state() -> dict[str, Any]:
    return _subturn_debug_state()


@app.post("/debug/subturn/clear")
def clear_subturn_debug_state() -> dict[str, Any]:
    with _subturn_debug_lock:
        _subturn_debug_events.clear()
    return {"ok": True, "count": 0}


@app.post("/debug/subturn/observe")
def observe_subturn_debug(req: SubturnObserveRequest) -> dict[str, Any]:
    record = _record_subturn_debug_event(
        {
            "source": _truncate_debug_text(req.source, 200),
            "event": _truncate_debug_text(req.event, 200),
            "session_id": _truncate_debug_text(req.session_id, 200),
            "user_turn_key": _truncate_debug_text(req.user_turn_key, 200),
            "legacy": _compact_debug_dict(req.legacy),
            "candidate": _compact_debug_dict(req.candidate),
            "data": _compact_debug_dict(req.data),
            "notes": [_truncate_debug_text(note, 1000) for note in req.notes[:20]],
        }
    )
    return {"ok": True, "debug_event_id": record["id"]}


@app.post("/translate_input")
async def translate_input(req: TranslateInputRequest) -> dict[str, Any]:
    text = req.text
    if not text.strip():
        return {"ok": True, "text": ""}

    system = (
        "You translate draft chat input. Output only the translated text. "
        "Preserve intent, tone, punctuation, markdown, code spans, file paths, "
        "commands, URLs, and proper nouns. Do not answer the message."
    )
    user = (
        f"Translate this text to {req.target_language}. "
        "Return only the translation, no quotes and no explanation.\n\n"
        f"{text}"
    )
    try:
        llm = get_llm("chat")
        translated = await llm.chat(
            system=system,
            user=user,
            max_tokens=max(256, min(2048, len(text) * 3)),
            reasoning_effort="none",
        )
        translated = _clean_translation_output(translated)
        return {"ok": True, "text": translated}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/llmsetting/catalog")
def llmsetting_catalog() -> dict[str, Any]:
    """Return the LLM provider catalog + live model lists + currently-active
    roles. Consumed by the Pi /model-setting slash command."""
    load_credentials_into_env()
    catalog = llm_load_catalog()
    fetched = llm_fetch_all(catalog)
    providers_out: dict[str, dict[str, Any]] = {}
    for pid, cfg in catalog["providers"].items():
        models = fetched.get(pid)
        if cfg.get("enabled") is False:
            available = False
            reason = cfg.get("note") or "disabled in catalog"
        elif models is None:
            available = False
            reason = (
                "not logged in (run /gpt-login)"
                if cfg.get("auth_kind") == "oauth"
                else f"no API key (set {cfg.get('auth_env', '?')})"
            )
        else:
            available = True
            reason = None
        providers_out[pid] = {
            "label": cfg.get("label", pid),
            "enabled": cfg.get("enabled", True),
            "available": available,
            "reason": reason,
            "models": list(models) if models else [],
            "auth_env": cfg.get("auth_env"),
            "auth_kind": cfg.get("auth_kind"),
        }
    return {
        "ok": True,
        "providers": providers_out,
        "recommended": catalog.get("recommended", {}),
        "current": llm_current_roles(),
    }


@app.post("/llmsetting/apply")
def llmsetting_apply(req: LLMSettingApplyRequest) -> dict[str, Any]:
    """Write the picked chat/encoder roles to data/config.yaml and register
    the matching entries in pi-agent/models.json."""
    def _split(value: str) -> tuple[str, str] | None:
        if "/" not in value:
            return None
        provider, model = value.split("/", 1)
        provider = provider.strip()
        model = model.strip()
        if not provider or not model:
            return None
        return provider, model

    chat = _split(req.chat)
    encoder = _split(req.encoder)
    if not chat or not encoder:
        return {"ok": False, "error": "chat and encoder must be 'provider/model' strings"}
    try:
        paths = llm_apply_picks(chat, encoder)
    except (OSError, KeyError, ValueError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    # Refresh in-process caches so the running sidecar picks up the new roles
    # without needing a restart. Chat goes through _LazyChatLLM so the module
    # cache invalidation inside reload_encoder_llm() covers it too; encoder is
    # bound to the long-lived slim/JLCEncoder instance and must be swapped.
    global _agent_last_error
    global _agent_last_error_ts
    reload_warning: str | None = None
    try:
        with _agent_guard:
            agent = _agent
            if agent is None:
                # A previous invalid encoder may have placed construction in
                # retry backoff. The user just changed its configuration, so
                # the old failure must not delay or describe the new choice.
                _agent_last_error = None
                _agent_last_error_ts = None
        if agent is not None and hasattr(agent, "reload_encoder_llm"):
            agent.reload_encoder_llm()
        else:
            clear_cache()
            if get_agent() is None and _agent_last_error:
                reload_warning = _agent_last_error
    except Exception as exc:  # noqa: BLE001
        reload_warning = f"{type(exc).__name__}: {exc}"
        print(f"[jarvis-sidecar] /llmsetting/apply reload failed: {exc}", file=sys.stderr)
    result = {
        "ok": True,
        "chat": f"{chat[0]}/{chat[1]}",
        "encoder": f"{encoder[0]}/{encoder[1]}",
        "config_path": paths["config_path"],
        "providers_path": paths["providers_path"],
        "models_json_path": paths["models_json_path"],
    }
    if reload_warning:
        result["reload_warning"] = reload_warning
    return result


def _build_context_block(
    *,
    project_path: str | None,
    code_path: str | None,
    jhb: str,
    project_memory: str,
    recall_block: str,
    recent_raw: str,
    trace: dict[str, Any],
    warnings: list[str],
    memory_mode: str,
) -> str:
    runtime_now = _runtime_now()
    parts = [
        "[JARVIS Code Memory]",
        "Use this block as durable project memory. It is not the user's request.",
        "Use ## Runtime Clock for all relative time words. Do not guess current date/time from model memory.",
        "Short confirmations, denials, or acknowledgements are usually answers to the previous assistant question. Check ## Recent Turns before treating them as a new casual topic.",
        "When present, ## Retrieved Prior Turns is query-specific evidence for the current user message. If it conflicts with JHB or Recent Turns, trust Retrieved Prior Turns and treat the conflicting summary/recent answer as stale or mistaken.",
        "JHB is a compact lossy summary, not the full transcript. ## Retrieved Prior Turns may be prefilled from the current chat query; prefer it over calling recall_turns.",
        "Use recall_turns only as a last-resort fallback when JHB, ## Recent Turns, and ## Retrieved Prior Turns still lack the exact prior fact needed. Do not call it for ordinary follow-ups, brand-new questions, or when the injected context already answers clearly.",
        "Daily chat keeps compact JHB memory. Coding projects use the unified JARVIS.md file at the project's code path (see ## Memory Project Files below).",
        "Each project keeps a single JARVIS.md at its code path. There is no separate workspace folder; do not invent or reference one.",
        "active_memory_project_path equals active_code_project_path under the unified-JARVIS.md policy. Both point at the project's code directory; JARVIS.md lives directly inside it.",
        "If the user asks to create, start, set up, build, or register a project and the target name/path is clear, treat that as explicit consent to call register_project; ask only when the target is ambiguous, missing, or unsafe.",
        "If the user's utterance clearly targets a different project, call switch_project before editing memory. If the target is ambiguous across multiple projects, ask a clarifying question first.",
        "cwd is not project identity. Project identity comes from user utterance and the JARVIS memory registry.",
        "Before invoking write or edit on a non-trivial file, output one short assistant line stating intent and target (e.g. 'Editing JARVIS.md: NOW section' or 'Editing app.py: <change>'). Current providers do not stream tool-call arguments incrementally, so this single line is what the user sees while you generate the tool input. Skip the line only for tiny writes or trivial edits.",
        f"memory_mode: {memory_mode}",
        f"active_memory_project_path: {project_path or 'none'}",
        f"active_code_project_path: {code_path or 'none'}",
        f"project_selection_source: {trace.get('source', 'none')}",
        f"default_project_root: {get_effective_project_root() or 'unset'}  # source of truth for 'project folder / 워크스페이스 폴더' questions — cite this verbatim, do not guess",
        "",
        "## Runtime Clock",
        f"current_datetime: {runtime_now.isoformat(timespec='seconds')}",
        f"timezone: {runtime_now.tzname() or 'local'}",
        f"today: {runtime_now.date().isoformat()}",
    ]
    if warnings:
        parts.append("warnings: " + "; ".join(warnings))
    if recall_block.strip():
        parts.extend(["", "## Retrieved Prior Turns", recall_block.strip()])
    if recent_raw.strip():
        parts.extend(["", "## Recent Turns (verbatim)", recent_raw.strip()])
    if jhb.strip():
        parts.extend(["", "## JHB", jhb.strip()])
    if project_memory.strip():
        parts.extend(["", "## Memory Project Files", project_memory.strip()])
    return "\n".join(parts).strip()


_ROUTE_VALUES = {"chat", "unregistered_coding", "deepdive", "heavy_deepdive"}


def _route_project_summaries() -> list[dict[str, str]]:
    projects: list[dict[str, str]] = []
    try:
        for project in router.registry.all():
            projects.append(
                {
                    "project_id": project.project_id,
                    "name": project.name,
                    "slug": project.slug,
                    "path": project.path,
                    "code_path": project.code_path or "",
                }
            )
    except Exception:  # noqa: BLE001
        return []
    return projects


def _route_turn_system_prompt() -> str:
    return (
        "You are the JARVIS Code chat-LLM router. Decide the user's intent before "
        "the main coding agent runs. Return only one JSON object, with no markdown.\n\n"
        "Routes:\n"
        "- chat: ordinary conversation, memory recall, questions, confirmations, or discussion that does not need file/tool work.\n"
        "- unregistered_coding: code/file analysis or edits for an explicit external/unregistered path or material, without JARVIS project memory.\n"
        "- deepdive: registered workspace project work, focused coding/debugging, or clear creation/registration of a new project.\n"
        "- heavy_deepdive: registered workspace project work needing root-cause analysis, broad structure review, multi-file refactor, regression/performance work, or explicit full tests.\n\n"
        "Project rules:\n"
        "- You understand natural language in any language. Do not require exact slash commands or full paths.\n"
        "- If the user clearly names a registered project by name, slug, nickname, translation, or prior context, set target_project_hint.\n"
        "- If multiple registered projects could match, set needs_clarification=true and route=chat.\n"
        "- If the user asks to create/start/register a new project and the target is clear, set create_project=true, register_project=true, route=deepdive or heavy_deepdive, and provide an ASCII project_slug.\n"
        "- For non-English project targets, produce a concise filesystem-safe ASCII slug by transliterating or translating the intent. Do not hardcode user-specific aliases.\n"
        "- If the user asks to edit an external absolute path without registering it, choose unregistered_coding.\n"
        "- If the user explicitly says not to register, never set register_project.\n\n"
        "JSON schema:\n"
        "{"
        "\"route\":\"chat|unregistered_coding|deepdive|heavy_deepdive\","
        "\"confidence\":\"high|medium|low\","
        "\"target_project_hint\":string|null,"
        "\"project_slug\":string|null,"
        "\"code_path_hint\":string|null,"
        "\"create_project\":boolean,"
        "\"register_project\":boolean,"
        "\"needs_clarification\":boolean,"
        "\"clarification\":string|null,"
        "\"reason\":string"
        "}"
    )


def _route_turn_user_prompt(req: RouteTurnRequest, projects: list[dict[str, str]]) -> str:
    payload = {
        "user_message": req.user_message,
        "cwd_hint": req.cwd_hint,
        "active_project_path": req.active_project_path,
        "registered_projects": projects[:80],
        "pending_project": req.pending_project,
        "recent_messages": req.recent_messages[-8:],
        "default_project_root": get_effective_project_root(),
        "protected_roots": get_protected_roots(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _route_turn_fallback(route: str, reason: str) -> dict[str, Any]:
    return {
        "ok": True,
        "route": route if route in _ROUTE_VALUES else "chat",
        "confidence": "low",
        "target_project_hint": None,
        "project_slug": None,
        "code_path_hint": None,
        "create_project": False,
        "register_project": False,
        "needs_clarification": False,
        "clarification": None,
        "reason": reason,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _route_str(value: Any, *, max_len: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def _route_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    return bool(value)


def _normalize_route_decision(data: dict[str, Any], *, raw: str) -> dict[str, Any]:
    route = _route_str(data.get("route")) or "chat"
    if route not in _ROUTE_VALUES:
        route = "chat"
    confidence = (_route_str(data.get("confidence"), max_len=20) or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    needs_clarification = _route_bool(data.get("needs_clarification"))
    if needs_clarification:
        route = "chat"
    return {
        "route": route,
        "confidence": confidence,
        "target_project_hint": _route_str(data.get("target_project_hint")),
        "project_slug": _route_str(data.get("project_slug"), max_len=120),
        "code_path_hint": _route_str(data.get("code_path_hint"), max_len=500),
        "create_project": _route_bool(data.get("create_project")),
        "register_project": _route_bool(data.get("register_project")),
        "needs_clarification": needs_clarification,
        "clarification": _route_str(data.get("clarification"), max_len=500),
        "reason": _route_str(data.get("reason"), max_len=500) or "chat LLM router decision",
        "raw_text": raw[:2000],
    }


def _format_recent_turns(turns: list[dict[str, Any]]) -> str:
    """Recent raw turns as plain user/assistant pairs (no timestamps,
    no headers, no preamble). Full text — no per-turn truncation, so
    follow-up resolution sees the actual exchange."""
    if not turns:
        return ""
    lines: list[str] = []
    for turn in turns:
        user = str(turn.get("user", "")).strip().replace("\r", "")
        assistant = str(turn.get("assistant", "")).strip().replace("\r", "")
        if user:
            lines.append(f"user: {user}")
        if assistant:
            lines.append(f"assistant: {assistant}")
    return "\n".join(lines)


def _format_raw_hits(hits: list[dict[str, Any]]) -> str:
    """Format raw JSONL hits into a bounded multi-line recall block.

    No LLM summarization is used; long fields are clipped deterministically so
    automatic recall cannot erase the token savings from reducing recent turns.
    """
    if not hits:
        return ""
    try:
        max_chars_per_field = max(0, int(os.environ.get("JARVIS_AUTO_RECALL_MAX_FIELD_CHARS", "1200")))
    except ValueError:
        max_chars_per_field = 1200
    lines = ["[Auto raw recall: current user query, top hits]"]
    for hit in hits:
        user = _clip_recall_field(str(hit.get("user", "")), max_chars_per_field)
        assistant = _clip_recall_field(str(hit.get("assistant", "")), max_chars_per_field)
        line_num = hit.get("line", "?")
        local_date = hit.get("local_date")
        suffix = f" | {local_date}" if local_date else ""
        lines.append(f"--- line {line_num}{suffix} ---")
        lines.append(f"Q: {user}")
        lines.append(f"A: {assistant}")
        lines.append("")
    return "\n".join(lines)


def _is_explicit_raw_recall_query(query: str) -> bool:
    return bool(extract_turn_numbers(query, max_turns=1) or extract_local_dates(query, max_dates=1))


def _clip_recall_field(text: str, max_chars: int) -> str:
    clean = text.strip().replace("\r", "")
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip() + " [clipped]"


def _raw_hits_to_fragments(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw JSONL hits into structured fragment dicts."""
    fragments = []
    for hit in hits:
        fragments.append({
            "turn": hit.get("line", 0),
            "score": 0.0,
            "user": str(hit.get("user", "")),
            "assistant": str(hit.get("assistant", "")),
            "ts": str(hit.get("ts") or hit.get("timestamp") or ""),
            "local_date": str(hit.get("local_date", "")),
        })
    return fragments


def _clean_translation_output(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()
    if (
        len(cleaned) >= 2
        and cleaned[0] == cleaned[-1]
        and cleaned[0] in {"'", '"'}
        and "\n" not in cleaned
    ):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _summarize_role(role: str, value: Any) -> dict[str, Any]:
    provider: str | None = None
    model: str | None = None
    configured: str | None = None

    if isinstance(value, dict):
        provider = str(value.get("provider") or "").strip() or None
        model = str(value.get("model") or "").strip() or None
        if provider and model:
            configured = f"{provider}/{model}"
    elif isinstance(value, str):
        configured = value.strip() or None
        if configured and "/" in configured:
            provider, model = configured.split("/", 1)
            provider = provider.strip() or None
            model = model.strip() or None
        elif configured:
            model = configured

    summary: dict[str, Any] = {
        "configured": configured,
        "provider": provider,
        "model": model,
        "display": configured or "unconfigured",
    }
    if not configured:
        summary["display"] = "unconfigured"
    summary["role"] = role
    return summary
