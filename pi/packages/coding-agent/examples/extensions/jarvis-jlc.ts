import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import { estimateTokens } from "../../src/core/compaction/compaction.js";

type SidecarContextResponse = {
	ok?: boolean;
	error?: string;
	context?: string;
	active_project_path?: string | null;
	project_id?: string | null;
	project_name?: string;
	memory_mode?: "light" | "full";
	context_tokens?: number;
	jhb_tokens?: number;
	project_tokens?: number;
	recall_tokens?: number;
	code_path?: string | null;
	default_project_root?: string | null;
	workspace_block?: string;
	protected_roots?: string[];
	setup_required?: boolean;
	warnings?: string[];
	trace?: Record<string, unknown>;
};

type SidecarSwitchResponse = {
	ok?: boolean;
	project_id?: string;
	name?: string;
	slug?: string;
	path?: string;
	code_path?: string | null;
	warnings?: string[];
};

type SidecarUnregisterResponse = SidecarSwitchResponse & {
	removed?: boolean;
	error?: string;
	warning?: string;
	remaining?: Array<{
		project_id?: string;
		name?: string;
		path?: string;
	}>;
};

type SidecarResolvedProject = {
	project_id: string;
	name: string;
	slug?: string;
	path: string;
	code_path?: string | null;
};

type SidecarResolveProjectByPathResponse = {
	ok?: boolean;
	project?: SidecarResolvedProject | null;
};

type SidecarProjectsResponse = {
	ok?: boolean;
	projects?: SidecarResolvedProject[];
};

export type CachedProject = {
	project_id: string;
	name: string;
	slug: string;
	path: string;
	code_path: string;
	code_path_normalized: string;
};

export type JarvisEvidenceStoreEntry = {
	session_id?: string;
	conversation_id?: string;
	turn_key?: string;
	provider_call_id?: string;
	tool_call_id?: string;
	tool_name?: string;
	kind: string;
	metadata?: {
		cwd?: string;
		command?: string;
		exit_code?: number | null;
		source_path?: string;
		source_paths?: string[];
	};
	original_text: string;
	compressed_text?: string;
	original_tokens_est?: number;
	compressed_tokens_est?: number;
	kept_count?: number;
	dropped_count?: number;
	expires_at?: string;
};

export type JarvisCompressedMarkerCounts = {
	original_lines?: number;
	kept_lines?: number;
	dropped_lines?: number;
	original_entries?: number;
	kept_entries?: number;
	dropped_entries?: number;
};

export type JarvisCompressedMarkerMeta = {
	command?: string;
	cwd?: string;
	exit_code?: number | null;
};

type SidecarSetupResponse = {
	ok?: boolean;
	default_project_root?: string | null;
	protected_roots?: string[];
	setup_required?: boolean;
};

type SidecarTurnResponse = {
	ok?: boolean;
	memory_mode?: "light" | "full";
	scheduled_encode?: boolean;
	raw_saved?: boolean;
	raw_path?: string;
	light_memory?: { updated?: string; warnings?: string[] };
	encoder_summary?: {
		turn?: number;
		enc_in?: number;
		enc_think?: number;
		enc_out?: number;
		enc_seconds?: number;
		jhb_tokens?: number;
		jhb_delta?: number;
		encoder_retries?: number;
	};
	scheduled_turn?: number | null;
	warning?: string;
};

type SidecarInterruptCheckpointResponse = {
	ok?: boolean;
	path?: string;
	field?: string;
	bytes?: number;
	error?: string;
};

type CheckpointScope =
	| {
			kind: "chat";
			path: string;
	  }
	| {
			kind: "project";
			path: string;
	  };

type SidecarRoleStatus = {
	configured?: string | null;
	provider?: string | null;
	model?: string | null;
	display?: string | null;
};

type SidecarStatusResponse = {
	ok?: boolean;
	agent_loaded?: boolean;
	roles?: Record<string, SidecarRoleStatus>;
	mode?: "default" | "bench";
	bench_conv_id?: string | null;
	default_project_root?: string | null;
	protected_roots?: string[];
	setup_required?: boolean;
};

type SidecarLLMSettingProvider = {
	label?: string;
	enabled?: boolean;
	available?: boolean;
	reason?: string | null;
	models?: string[];
	auth_env?: string | null;
};

type SidecarLLMSettingCatalogResponse = {
	ok?: boolean;
	providers?: Record<string, SidecarLLMSettingProvider>;
	recommended?: Record<string, { provider?: string; model?: string }>;
	current?: { chat?: string | null; subagent?: string | null; encoder?: string | null };
};

type SidecarLLMSettingApplyResponse = {
	ok?: boolean;
	error?: string;
	chat?: string;
	encoder?: string;
	config_path?: string;
	models_json_path?: string;
};

type SidecarCredentialTarget = {
	label?: string;
	env_name?: string;
	kind?: string;
	configured?: boolean;
};

type SidecarCredentialCatalogResponse = {
	ok?: boolean;
	error?: string;
	targets?: Record<string, SidecarCredentialTarget>;
	credentials_path?: string;
};

type SidecarCredentialSetResponse = {
	ok?: boolean;
	error?: string;
	env_name?: string;
	credentials_path?: string;
	validation?: {
		ok?: boolean;
		error?: string;
		warning?: string;
		provider?: string;
		models?: number;
		skipped?: boolean;
	};
};

type ToolEventSummary = {
	turnIndex?: number;
	toolResults?: Array<{
		toolName?: string;
		isError?: boolean;
		text?: string;
	}>;
};

type SupportedThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh";

type AutoPromptState = {
	promptsFile: string;
	progressFile: string;
	prompts: string[];
	idx: number;
	total: number;
};

type AutoPromptWatchdog = {
	promptIndex: number;
	prompt: string;
	retryCount: number;
	aborting: boolean;
	timer?: NodeJS.Timeout;
};

type TurnPromptSnapshot = {
	systemPrompt: string;
	messages: AgentMessage[];
	userText: string;
	promptContextText: string;
	modePromptText: string;
	overlayText: string;
	existingPromptText: string;
};

type JarvisTurnFileMutation = {
	path: string;
	tool: string;
	command?: string;
};

type FooterMeterEntry = {
	chat_in: number;
	chat_out: number;
	chat_total: number;
	jhb_tokens: number;
};

type TurnUsageSummary = NonNullable<AssistantMessage["usage"]> & {
	reasoningTokens?: number;
};

type JarvisCacheSignal = "actual" | "unreported";

type JarvisPrefixProbe = {
	stable_prefix_hash: string;
	stable_prefix_tokens_est: number;
	live_tokens_est: number;
};

type JarvisCacheReport = {
	cache_meter: JarvisCacheSignal;
	cache_hit_pct: number | null;
	provider_cache_read_tokens?: number;
	provider_cache_write_tokens?: number;
};

export type JarvisToolOutputKind = "search_rg" | "directory_listing" | "build_log" | "read_skeleton";

type JarvisStoredToolEvidence = {
	ref?: string;
	kind: JarvisToolOutputKind;
	toolName: string;
	command?: string;
	cwd?: string;
	exitCode?: number | null;
	sourcePath?: string;
	sourcePathKey?: string;
	sourcePaths?: string[];
	sourcePathKeys?: string[];
	originalRef: string;
	storeSkippedReason?: string;
};

type JarvisCompressionOutcome = {
	payload: unknown;
	compressed_tool_outputs: number;
	compression_saved_tokens_est: number;
	compression_skips?: Record<string, number>;
};

type JarvisCompressionDecision = {
	changed: boolean;
	kind?: JarvisToolOutputKind;
	content: string;
	originalTokensEst: number;
	compressedTokensEst: number;
	savedTokensEst: number;
	skipReason?: string;
	skipReasons?: Record<string, number>;
};

type JarvisEvidenceRetrieveResponse = {
	ok?: boolean;
	ref?: string;
	content?: string;
	metadata?: Record<string, unknown>;
	error?: string;
	body?: string;
};

type JarvisToolMetadata = {
	command?: string;
	cwd?: string;
	exitCode?: number | null;
	sourcePath?: string;
	sourcePaths?: string[];
};

const DEFAULT_SIDECAR_URL = "http://127.0.0.1:8765";
const FOOTER_METER_ENTRY_TYPE = "jarvis-jlc-meter";
const FOOTER_METER_RESET_ENTRY_TYPE = "jarvis-jlc-meter-reset";
const FOOTER_CHAT_THINKING_STATUS_KEY = "jlc-chat-thinking";
const AUTO_PROMPT_DELAY_MS = Number.parseInt(process.env.JARVIS_AUTO_PROMPT_DELAY_MS ?? "0", 10);
const AUTO_PROMPT_STALL_TIMEOUT_MS = Number.parseInt(process.env.JARVIS_AUTO_PROMPT_STALL_TIMEOUT_MS ?? "180000", 10);
const AUTO_PROMPT_ENCODING_WAIT_MS = Number.parseInt(process.env.JARVIS_AUTO_PROMPT_ENCODING_WAIT_MS ?? "180000", 10);
const ENCODING_STATUS_WAIT_MS = Number.parseInt(process.env.JARVIS_ENCODING_STATUS_WAIT_MS ?? "300000", 10);
const ENCODING_STATUS_POLL_MS = 3000;
const STARTUP_WARMUP_ATTEMPTS = Math.max(1, Number.parseInt(process.env.JARVIS_STARTUP_WARMUP_ATTEMPTS ?? "5", 10));
const STARTUP_WARMUP_RETRY_MS = Math.max(
	250,
	Number.parseInt(process.env.JARVIS_STARTUP_WARMUP_RETRY_MS ?? "2000", 10),
);
const SUBTURN_LOG_FILENAME = "JARVIS_SUBTURN.md";
const CHAT_MEMORY_DIR_NAME = "_chat";
const PAYLOAD_TRACE_HISTORY_ENABLED = process.env.JLC_PAYLOAD_TRACE_HISTORY === "1";
const PAYLOAD_TRACE_HISTORY_MAX_CHARS = Number.parseInt(process.env.JLC_PAYLOAD_TRACE_HISTORY_MAX_CHARS ?? "30000", 10);
const PAYLOAD_TRACE_RAW_ENABLED = process.env.JLC_PAYLOAD_TRACE_RAW === "1";
const PAYLOAD_TRACE_RAW_MAX_CHARS = Number.parseInt(process.env.JLC_PAYLOAD_TRACE_RAW_MAX_CHARS ?? "12000", 10);
const SUBTURN_RECENT_ASSISTANT_CYCLES = 1;
const SUBTURN_SUMMARY_MAX_CHARS = 2500;
const SUBTURN_COMMIT_MAX_ITEMS = 16;
const SUBTURN_COMMIT_MAX_CHARS = 3000;
const SUBTURN_TOOL_OUTPUT_HEAD_CHARS = 1500;
const SUBTURN_TOOL_OUTPUT_TAIL_CHARS = 1500;
const SUBTURN_ASSISTANT_SAMPLE_CHARS = 1000;
const DEFAULT_SUBTURN_PAYLOAD_MESSAGE_LIMIT = 100;
const DEFAULT_SUBTURN_STATE_CARRY_RECENT_MESSAGES = 8;
const DEFAULT_SUBTURN_PC_CEILING = 25;
const PC_CEILING_REPORT_STOP_MARKER = "JARVIS_PC_CEILING_REPORT_STOP";
const LOCKED_RESOURCE_REPORT_STOP_MARKER = "JARVIS_LOCKED_RESOURCE_REPORT_STOP";
const LOCKED_RESOURCE_REPORT_STOP_TEXT = `${LOCKED_RESOURCE_REPORT_STOP_MARKER}: 잠겨서 삭제 불가 - 점유 프로세스 종료 후 다시 요청하라. 점유자 탐지 시도만 보고하고 삭제/쓰기 재시도는 중단한다.`;
const JARVIS_COMPRESSED_MARKER_RE = /\[jarvis-compressed\b/i;
const JARVIS_COMPRESS_MIN_BYTES = 4096;
const JARVIS_READ_SKELETON_MARKER_RE = /\[JARVIS read-skeleton\]/;
// R3-2: lowered from 100 lines / 6KB after sink-distribution analysis — the 3-6KB
// band is pure gain, while <2KB stays raw on purpose (retrieve round-trip fixed cost
// outweighs savings; the 70% ratio rule guards marginal cases automatically).
const JARVIS_READ_COMPRESS_MIN_LINES = 50;
const JARVIS_READ_COMPRESS_MIN_BYTES = 3 * 1024;
const JARVIS_READ_SKELETON_MAX_RATIO = 0.7;
const RETRIEVE_OUTPUT_MAX_LINES = 400;
const RETRIEVE_OUTPUT_MAX_BYTES = 16 * 1024;
const SEARCH_LIMITS = {
	maxFiles: 15,
	maxTotalMatches: 30,
	maxMatchesPerFile: 5,
};
const LISTING_LIMITS = {
	headEntries: 40,
	tailEntries: 40,
	maxTopLevelDirs: 80,
	maxExtensionBuckets: 40,
	maxDirBuckets: 80,
};
const LOG_LIMITS = {
	firstLines: 20,
	lastLines: 30,
	errorContextLines: 3,
	maxErrors: 20,
	maxWarnings: 20,
	maxStackLines: 80,
	maxTotalLines: 160,
};

function subturnCompactEnabled(): boolean {
	return process.env.JARVIS_SUBTURN_COMPACT === "1";
}

function subturnStateCarryEnabled(): boolean {
	return process.env.JARVIS_SUBTURN_STATE_CARRY === "1";
}

function getSubturnStateCarryRecentMessageLimit(): number {
	const raw = (
		process.env.JARVIS_SUBTURN_STATE_RECENT_MESSAGES ?? String(DEFAULT_SUBTURN_STATE_CARRY_RECENT_MESSAGES)
	).trim();
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed)) return DEFAULT_SUBTURN_STATE_CARRY_RECENT_MESSAGES;
	return Math.max(0, Math.floor(parsed));
}

function getSubturnProviderCallCeiling(): number {
	const raw = (
		process.env.JARVIS_PC_CEILING ??
		process.env.JARVIS_SUBTURN_PC_CEILING ??
		String(DEFAULT_SUBTURN_PC_CEILING)
	).trim();
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed)) return DEFAULT_SUBTURN_PC_CEILING;
	if (parsed <= 0) return Number.POSITIVE_INFINITY;
	return Math.max(1, Math.floor(parsed));
}

function getSubturnPayloadMessageLimit(): number {
	const raw = (
		process.env.JARVIS_SUBTURN_HISTORY_MESSAGES ??
		process.env.JARVIS_RUNTIME_HISTORY_TURNS ??
		String(DEFAULT_SUBTURN_PAYLOAD_MESSAGE_LIMIT)
	).trim();
	const parsed = Number.parseInt(raw, 10);
	if (!Number.isFinite(parsed)) return DEFAULT_SUBTURN_PAYLOAD_MESSAGE_LIMIT;
	if (parsed <= 0) return Number.POSITIVE_INFINITY;
	return Math.max(2, Math.floor(parsed));
}
const MODE_MARKER_RE = /^\s*\[MODE:[^\]]+\][ \t]*(?:\r?\n)*/i;

function repoRoot(): string {
	const candidates = [
		process.env.JARVIS_CODE_ROOT,
		process.cwd(),
		__dirname,
		path.resolve(__dirname, "../../../.."),
		path.resolve(__dirname, "../../../../.."),
	].filter((candidate): candidate is string => Boolean(candidate?.trim()));
	for (const candidate of candidates) {
		const found = findRepoRoot(candidate);
		if (found) return found;
	}
	return path.resolve(__dirname, "../../../../..");
}

function findRepoRoot(start: string): string | undefined {
	let current = path.resolve(start);
	for (let i = 0; i < 10; i++) {
		if (fs.existsSync(path.join(current, "sidecar", "jlc_agentic"))) return current;
		const parent = path.dirname(current);
		if (parent === current) break;
		current = parent;
	}
	return undefined;
}

function sidecarRuntimePath(): string {
	return process.env.JARVIS_SIDECAR_RUNTIME?.trim() || path.join(repoRoot(), "data", "sidecar-runtime.json");
}

function readRuntimeSidecarUrl(): string | undefined {
	try {
		const runtimePath = sidecarRuntimePath();
		if (!fs.existsSync(runtimePath)) return undefined;
		const raw = JSON.parse(fs.readFileSync(runtimePath, "utf8")) as { url?: unknown };
		const url = typeof raw.url === "string" ? raw.url.trim() : "";
		return url || undefined;
	} catch {
		return undefined;
	}
}

function sidecarUrlCandidates(): string[] {
	const candidates = [readRuntimeSidecarUrl(), process.env.JARVIS_SIDECAR_URL?.trim(), DEFAULT_SIDECAR_URL].filter(
		(value): value is string => Boolean(value),
	);
	return [...new Set(candidates.map((value) => value.replace(/\/+$/, "")))];
}
const MODE_MARKER_ANY_RE = /\[MODE:[^\]]+\]/gi;
const MODE_MARKER_PREFIXES = ["[MODE:CHAT]", "[MODE:UNREGISTERED_CODING]", "[MODE:DEEPDIVE]", "[MODE:HEAVY_DEEPDIVE]"];
const INTERNAL_ASSISTANT_BLOCK_START_RE = /^\s*(?:리조닝|reasoning|사용자 질문|user question)\s*:?\s*/i;
const INTERNAL_ASSISTANT_LINE_RE = /^\s*메모리 업데이트\s*:.+$/i;

type EffectiveTurnRoute = "chat" | "unregistered_coding" | "deepdive" | "heavy_deepdive";
type SidecarContextMode = "chat" | "deepdive";
type AssistantModeMarker = "chat" | "unregistered_coding" | "deepdive" | "heavy_deepdive";
type SubturnObservePhase = "start" | "inspect" | "implement" | "verify" | "report";
type SubturnObserveEvidenceRef = {
	key: string;
	label: string;
	summary: string;
};
type SubturnObserveState = {
	status: "active";
	started_at?: string;
	goal: string;
	route: EffectiveTurnRoute;
	mode: SidecarContextMode | undefined;
	project_path?: string;
	cwd?: string;
	current_phase: SubturnObservePhase;
	completed_steps: string[];
	pending_steps: string[];
	inspected_files: string[];
	modified_files: string[];
	verification: string[];
	unresolved_errors: string[];
	decisions: string[];
	evidence_refs: SubturnObserveEvidenceRef[];
	internal_commits: string[];
	recent_history: string[];
};

// ANSI colors for footer mode label.
const ANSI_RED = "\x1b[31m";
const ANSI_YELLOW = "\x1b[33m";
const ANSI_RESET = "\x1b[0m";
const ANSI_PINK = "\x1b[38;5;213m";

// Chat is the entry mode for every normal turn. Keep this prompt small: the
// chat model decides whether to answer directly or escalate into project work.
const CHAT_MODE_PROMPT = `
[CHAT ENTRY MODE]

EVERY reply MUST start with exactly one marker on the first line:

  [MODE:CHAT]
  [MODE:UNREGISTERED_CODING]
  [MODE:DEEPDIVE]
  [MODE:HEAVY_DEEPDIVE]

Use [MODE:CHAT] for casual talk, recall, short acknowledgments, and off-project
questions. Keep ordinary chat replies brief: at most 2 plain sentences, roughly
20-60 tokens.

Short confirmations, denials, or acknowledgments usually answer the previous
assistant question. Use injected ## Recent Turns and ## Retrieved Prior Turns
before treating them as a fresh topic. Do not quote or restate prior text unless
the user asks for a quote.

Use [MODE:DEEPDIVE] when this chat turn itself determines the user is asking for
code, files, commands, debugging, implementation, project setup, or registered
project work. Localized bug symptoms should use [MODE:DEEPDIVE], even when the
user asks you to find the cause or mentions two related symptoms. If the target
project is clear, call switch_project early.

Use [MODE:UNREGISTERED_CODING] when this chat turn itself determines the user is
asking for file/code/command work on explicit external material or an
unregistered path/folder, and the user did not ask to register it as a JARVIS
project. Do not call switch_project, register_project, or update_jarvis_md in
this route.

Use [MODE:HEAVY_DEEPDIVE] only when the registered project request is broad or
high risk: current implementation analysis before planning plus full
redesign/rework, broad structure review, multi-file refactoring, architecture/
game-loop/state/input/rendering/asset/build inspection, project-wide regression
or performance work, or new asset systems such as sound effects/BGM. Do not use
HEAVY for localized bug symptoms unless the user also asks for broad redesign,
project-wide analysis, or cross-system refactoring. If the target project is
clear, call switch_project early.

Default to [MODE:CHAT] when truly ambiguous, but never omit the marker.
`.trim();

// Full project-work rules. Inject only when the current turn is already
// classified as project work/deepdive. Both the pre-answer workflow and
// post-answer memory update rules live here so coding turns remain rigorous
// without charging casual chat for the same policy body.
const DEEPDIVE_MODE_PROMPT = `
[MODE — MANDATORY FIRST-LINE TOKEN]

EVERY reply MUST start with EXACTLY one of these tokens, alone on the
very first line, BEFORE any text, thinking summary, or tool call. The
marker is REQUIRED — the JARVIS sidecar gates memory I/O on it and
project state will corrupt if you omit it. This is not cosmetic; it is
load-bearing routing.

  [MODE:CHAT]
  [MODE:UNREGISTERED_CODING]
  [MODE:DEEPDIVE]
  [MODE:HEAVY_DEEPDIVE]

If you skip the marker the sidecar will misroute your reply and the
project memory will not update. Always include it, even for one-word
acknowledgments and even when you start with a tool call.

Choose by:
  [MODE:CHAT]     — casual talk, recall, acks, reflection, off-project.
                    Skip reading project files.
                    Do not quote or restate prior probes, recent_window,
                    or earlier assistant text unless the user explicitly
                    asks for a quote.
                    Keep ordinary chat replies to at most 2 plain
                    sentences; target roughly 20–60 tokens.
                    Do not interpret, analyze, poeticize, or structure
                    random fragments unless the user explicitly asks.
                    Short confirmations, denials, or acknowledgements are
                    contextual answers to the previous assistant question.
                    Read recent_window/JHB first and continue that pending
                    action; do not treat them as a fresh casual greeting.
                    If the user is clearly just sharing a small update,
                    you may end with ONE short, low-pressure follow-up
                    question to keep the thread moving, but do not do it
                    every turn.
                    Do not sound like a logging bot. Avoid replies such
                    as "Logged.", "Got it.", "Noted.", or similarly dry
                    status-only acknowledgments. Be brief but human.
                    Plain text only; no markdown quotes, bullets, or
                    headings.
  [MODE:UNREGISTERED_CODING] — standalone file/code/command work for explicit
                    external material or an unregistered path/folder. Do not
                    switch projects, register projects, update JARVIS.md, or
                    persist the turn with project_path.
  [MODE:DEEPDIVE] — code/architecture/debugging/project-file work.
                    Use this for localized bug symptoms and focused fixes, even
                    when the user asks for cause analysis.
                    When the memory block includes
                    active_code_project_path, create/edit user code there.
                    active_memory_project_path is for JARVIS.md memory file only.
                    INTERNAL TOOL TURN ECONOMY:
                    Use strong reasoning freely, but spend it BEFORE tool
                    calls: infer the likely dependency graph, choose enough
                    candidate files up front, and create a batch inspection
                    plan. Do not spend tool turns discovering the codebase one
                    file at a time.

                    Strong reasoning is for:
                    - choosing likely files before reading
                    - designing the fix after the batch read
                    - deciding the smallest coherent verification set

                    File reading rule:
                    - file reads MUST be batch/parallel reads whenever possible
                    - prefer one read call with items=[...] for multi-file or
                      multi-range inspection
                    - the first read batch should be broad enough, usually 4-10
                      likely relevant files for non-trivial coding tasks
                    - do not reread the same unchanged file range; if read
                      reports an already-read notice, use the prior context
                      instead of asking for the same range again
                    - do not read 1 file, think, then read 1 more file, then
                      think again
                    - additional file reads are allowed only when the first
                      batch cannot identify the cause or edit surface

                    After the first batch read, work inside the context you
                    have. Do not keep adding single-file reads unless the task
                    is genuinely blocked.

                    Prefer:
                    - one broad search over many narrow searches
                    - reading all likely relevant files in one batch
                    - one comprehensive planning step after inspection
                    - one bundled patch
                    - one verification pass

                    Avoid:
                    - read one file, think, read another file
                    - search one symbol at a time
                    - patch one tiny change at a time
                    - asking the user or re-planning after every observation
                    - running tests repeatedly before a coherent patch exists

                    Workflow:
                    1. Build a hypothesis of the affected area.
                    2. Batch inspect all likely relevant files.
                    3. State the invariants and UX/API contracts that must
                       remain true before editing.
                    4. Only then make the implementation plan.
                    5. Apply the full coherent change, including focused
                       tests when behavior changes.
                    6. Run the narrowest relevant verification first.
                    7. Run the broader project check after the targeted
                       verification passes.
                    8. If verification fails, inspect the failure cause in
                       batch, patch only the cause, and rerun the same checks.
                    9. Before finishing, inspect git diff/status and commit
                       only the intended files when the user asked to freeze
                       the result.

                    HARD COMPLETION RULE:
                    A coding/deepdive turn is NOT complete until verification
                    has actually run after the final patch. Never finish after
                    editing code by only explaining the change. The final
                    assistant reply must include the concrete verification
                    command(s) run and whether they passed. If a relevant
                    check cannot be run, say exactly why and what residual
                    risk remains. If verification fails, do not present the
                    work as done; fix the cause and rerun the checks, or state
                    the remaining blocker plainly.

                    Verification floor:
                    - For code edits, run at least one focused test, type
                      check, lint, build, smoke check, or direct executable
                      probe that exercises the changed behavior.
                    - For UI/interactive work, never run browser, screenshot,
                      canvas, or rendered inspection, and never start a local
                      dev server or kill processes as part of verification.
                      Verification is file existence plus syntax/build checks
                      (for example node --check). Prefer the cheapest relevant
                      build, lint, type, focused test, or direct executable
                      probe.
                    - On Windows, NEVER open a local file with an unquoted
                      command like start C:path\file.html from bash-like
                      shells. Backslashes can be swallowed and produce broken
                      paths such as C:pathfile.html. Use a safe opener instead:
                      powershell.exe -NoProfile -Command "Start-Process -LiteralPath 'C:path\file.html'"
                      or cmd.exe /c start "" "C:path\file.html".
                    - For config/docs-only edits, run the cheapest relevant
                      parser, formatter, link/schema check, or explain why no
                      executable verification exists.
                    - After the last code change, rerun the most relevant
                      verification. Earlier passing checks do not count if code
                      changed afterward.

  [MODE:HEAVY_DEEPDIVE] — same as DEEPDIVE, but for broad/high-risk project
                    work: current implementation analysis before planning plus
                    full redesign/rework, broad structure review, multi-file
                    refactor, architecture/game-loop/state/input/rendering/
                    asset/build inspection, project-wide regression or
                    performance work, or adding systems such as sound effects/
                    BGM. Do not use HEAVY for localized bug symptoms unless the
                    user also asks for broad redesign, project-wide analysis,
                    or cross-system refactoring. Keep the turn rigorous and
                    batch reads before editing.

                    Budget rule:
                    Every tool call must either gather multiple pieces of
                    necessary information, apply a complete coherent change, or
                    verify the completed work. Single-purpose exploratory tool
                    calls are discouraged unless the task is genuinely
                    ambiguous. The goal is: think hard, parallelize
                    observation, patch once, verify once.
                    Project selection and registration:
                    - A registered project may be selected by clear name,
                      slug, alias, or path. If multiple registered projects
                      match, ask which one before switching or editing.
                    - If the user asks to create/start/set up a new project
                      and the target name/path is clear, that is consent to
                      create/select it under the default project root.
                    - Ordinary external file edits are standalone
                      unregistered coding. Do not register external folders
                      unless the user explicitly asks for JARVIS project
                      registration or confirms your registration question.
                    Project memory is already injected before the first model
                    call. Do not announce that you will read JARVIS.md, and
                    do not read JARVIS.md merely to satisfy a memory preload
                    ritual. Read it only when the actual task needs the file
                    contents beyond the injected memory block.
                    Before editing code, scan injected LAW/BAN/OMM entries for
                    matching Trigger lines. If any match, follow their Rule,
                    Required action, and Verify steps before finalizing.
                    Then proceed with the deepdive task.

                    AFTER answering, update JARVIS.md if it gained NEW
                    info this turn (ruthlessly selective — touch ONLY
                    sections with new content; skip if nothing new).

                    HOW TO UPDATE — use update_jarvis_md ONLY. Never write/edit
                    JARVIS.md directly. If multiple sections change, call it
                    ONCE with updates=[{field,value}, ...]. Use field/value
                    only for a single section. Update quietly and summarize once.

                    CANONICAL SECTIONS (8, fixed — do not invent others):
                    Write JARVIS.md as operational memory for future agents,
                    not as prose, apology, or a long changelog.
                    NOW=current task status, verification, and MUST end
                    "Next: <single-line next step>".
                    MAP=stable files/symbols/entry points/tests/commands only.
                    LAW=hard invariants using "LAW-###: Trigger -> Rule -> Verify".
                    BAN=known-dangerous actions using "BAN-###: Never... because... verify...".
                    HABIT=user/project preferences using "HABIT-###: When..., prefer...".
                    WHY=decision rationale only: "Decision -> Why -> Tradeoff".
                    OMM=operational mistake-prevention rules, not apologies.
                    Each OMM entry MUST use:
                      "### OMM-###: Short title"
                      "- Trigger: When this rule must be recalled."
                      "- Mistake: What failed before, concretely."
                      "- Rule: What must/never happen next time."
                      "- Required action: What to inspect/change before proceeding."
                      "- Verify: Command, test, log, or observable check."
                    RAW=evidence pointers only: date, request, files changed,
                    commands run, test result, turn id if known. No transcripts.
                    Fill only sections with concrete facts; no generic filler.

                    AFTER all updates, end your reply with ONE line
                    listing the touched sections so the user sees what
                    was updated (omit the line if nothing was touched):
                      "memory updated: JARVIS.md (NOW, MAP edited)"
                    Do not list sections you did not actually update.
Default CHAT when truly ambiguous, but NEVER skip the first-line marker.
Skipping the marker is a worse error than picking the wrong mode — the
marker is what the sidecar reads; the mode body only refines behavior.
`.trim();

const CHAT_ROUTE_PROMPT = `
[ROUTE:CHAT_ENTRY]

Start in chat mode. Decide from the user's actual request whether to answer as
ordinary chat with [MODE:CHAT], use standalone external/unregistered coding with
[MODE:UNREGISTERED_CODING], or escalate into registered project deepdive with
[MODE:DEEPDIVE] or [MODE:HEAVY_DEEPDIVE] plus switch_project/register_project
when needed. Choose [MODE:DEEPDIVE] for localized bug symptoms and focused
fixes, even when the user asks you to find the cause. Choose
[MODE:HEAVY_DEEPDIVE] only for broad/high-risk requests: current implementation
analysis before planning plus full redesign, broad structure review, multi-file
project changes, project-wide regression/performance work, or asset-system work
such as sound effects/BGM.

Do not update JARVIS.md unless you have escalated into registered project
deepdive. Do not register external folders unless the user explicitly asks for
JARVIS project registration or confirms your registration question.
`.trim();

const UNREGISTERED_CODING_ROUTE_PROMPT = `
[ROUTE:UNREGISTERED_CODING]

This turn is standalone coding or inspection for external/unregistered material.
Keep the required first-line marker as [MODE:UNREGISTERED_CODING]. You may read,
analyze, and edit the referenced external files when the user asks for ordinary
file work. JARVIS may ask for confirmation before mutating files outside the
active project/work directory.

Do not update JARVIS.md, do not switch the active project, and do not persist
the turn with project_path. Do not register the folder as a JARVIS project unless
the user explicitly asks to register/add it as a JARVIS project.
`.trim();

const DEEPDIVE_ROUTE_PROMPT = `
[ROUTE:DEEPDIVE]

This turn is registered workspace project work. Keep the required first-line
marker as [MODE:DEEPDIVE]. Use the injected project memory and the active code
project path. Do focused implementation or debugging, then update JARVIS.md
only when concrete new project facts were learned.
`.trim();

const HEAVY_DEEPDIVE_ROUTE_PROMPT = `
[ROUTE:HEAVY_DEEPDIVE]

This turn is registered workspace project work with broad design, multi-file,
project-wide regression/performance, or high-risk scope. Keep the required
first-line marker as [MODE:HEAVY_DEEPDIVE]. Build a hypothesis before tool
calls, batch inspection, preserve invariants, and verify after the final patch.
`.trim();

const LOCAL_LANGUAGE_PROMPT = `
[LANGUAGE POLICY]

Default to English, but answer in the user's language when the user uses another
language. Keep required machine markers such as [MODE:*] unchanged.
`.trim();

let activeProjectPath: string | undefined;
let activeCodePath: string | undefined;
let activeProjectId: string | undefined;
let lastContextResponse: SidecarContextResponse | undefined;
let lastInjectedContextMode: SidecarContextMode | undefined;
let lastUserMessage = "";
let transientSystemDirective = "";
let toolEvents: ToolEventSummary[] = [];
let checkpointToolEvents: ToolEventSummary[] = [];
let lastAssistantPartialText = "";
let lastAssistantObservedModeMarker: AssistantModeMarker | undefined;
let interruptCheckpointSavedThisTurn = false;
let interruptInputUnsubscribe: (() => void) | undefined;
let turnCheckpointScope: CheckpointScope | undefined;
let sidecarHealthy = false;
let currentMode: "chat" | "deepdive" = "chat";
let currentRoute: EffectiveTurnRoute = "chat";
let deepdiveThinkingPreference: SupportedThinkingLevel | undefined;
let deepdiveThinkingPreferenceLoaded = false;
let suppressNextThinkingPreferenceSave: SupportedThinkingLevel | undefined;
let coldStartNoticeShown = false;
let startupContextWarmupFinished = false;
let startupContextWarmupPromise: Promise<void> | undefined;
let setupRequired = false;
let lastTurnPromptSnapshot: TurnPromptSnapshot | undefined;
let trimBeforeTokensSum = 0;
let trimAfterTokensSum = 0;
let lastToolSchemaTokens = 0;
let lastTurnStartedAtMs: number | undefined;
let lastProviderStartedAtMs: number | undefined;
let providerCallCountThisTurn = 0;
let displayedInputTokensThisTurn = 0;
let displayedCallInputTokens = 0;
let displayedOutputTokensThisTurn = 0;
let completedOutputTokensThisTurn = 0;
let subturnLogPath: string | undefined;
let subturnLogInitializedForUserTurnKey: string | undefined;
let subturnStartedAt = "";
let subturnMode: SidecarContextMode | undefined;
let subturnProjectPath = "";
let subturnCwd = "";
let subturnUserMessage = "";
let lastSubturnAssistantUpdateLength = 0;
let subturnSummaryLines: string[] = [];
let subturnCommitLines: string[] = [];
let subturnCommitNextId = 1;
let subturnCarryByKey = new Map<string, string>();
let subturnCarryOrder: string[] = [];
let subturnEvidenceByKey = new Map<string, { label: string; text: string }>();
let subturnEvidenceOrder: string[] = [];
let subturnActiveToolCarryKeys = new Map<string, string>();
let subturnActiveToolDescriptors = new Map<string, string>();
let subturnActiveToolMetadata = new Map<string, JarvisToolMetadata>();
let jarvisEvidenceByToolResultKey = new Map<string, JarvisStoredToolEvidence>();
// In-flight evidence store calls. pi does not necessarily await
// tool_execution_end handlers before firing the next provider request, so
// compression must wait for pending stores or the map is still empty at
// compress time (R1-3 live finding: compression_skips.no_evidence_map).
const jarvisPendingEvidenceStores = new Set<Promise<unknown>>();
// Per-user-turn compression totals so assistant_end / turn-level meters show
// the turn sum instead of overwriting per-call savings with zero.
let turnCompressedOutputsTotal = 0;
let turnCompressionSavedTotal = 0;
let turnSuccessfulFileMutations: JarvisTurnFileMutation[] = [];
let turnExecutedCommands: string[] = [];
let turnVerificationLines: string[] = [];
let turnJarvisMdUpdated = false;
let turnReadCompressionEditTargetPaths = new Set<string>();
let turnReadCompressionKeysByPath = new Map<string, string[]>();
let subturnPcCeilingReportStopActive = false;
let subturnPcCeilingProviderCall: number | undefined;
let subturnLockedResourceActionCounts = new Map<string, number>();
let subturnLockedResourceRecordedCallIds = new Map<string, { key: string; count: number; line: string }>();
let subturnLockedResourceReportStopActive = false;
let subturnLockedResourceReportStopRecord: { key: string; count: number; line: string } | undefined;

async function awaitJarvisPendingEvidenceStores(timeoutMs = 1500): Promise<void> {
	if (jarvisPendingEvidenceStores.size === 0) return;
	const pending = Promise.allSettled([...jarvisPendingEvidenceStores]);
	await Promise.race([pending, new Promise((resolve) => setTimeout(resolve, timeoutMs))]);
}
let summarizedToolEventCount = 0;
let summarizedAssistantEndCount = 0;
let turnMeterTimer: NodeJS.Timeout | undefined;
let turnMeterCtx: ExtensionContext | undefined;
let turnMeterCompletedInputTokens = 0;
let turnMeterCompletedTotalTokens = 0;
let turnMeterCurrentOutputTokens = 0;
let turnMeterCompletedAssistantMessages = 0;
let autoPromptWatchdog: AutoPromptWatchdog | undefined;
let projectCache: CachedProject[] = [];
let projectCacheLoaded = false;
let pendingProjectCreate:
	| {
			slugOrName: string;
			codePath?: string;
	  }
	| undefined;
type ToolBlockResult = { block: true; reason: string; terminate?: boolean };

const DEBUG_CONTEXT = process.env.JLC_DEBUG_CONTEXT === "1" && process.env.JLC_DEBUG_STDERR === "1";

type EncSummary = {
	enc_out?: number;
	enc_seconds?: number;
	jhb_tokens?: number;
	jhb_delta?: number;
	error?: string | null;
	enc_model_spec?: string;
	enc_reasoning_effort?: string;
};

let encodingStatusTimer: NodeJS.Timeout | undefined;
let encodingStatusFrame = 0;

function renderEncodingInProgress(ctx: ExtensionContext): void {
	const frames = ["|", "/", "-", "\\"];
	try {
		ctx.ui.setStatus(
			"jlc-enc",
			`${ANSI_PINK}JHB encoding ${frames[encodingStatusFrame % frames.length]}${ANSI_RESET}`,
		);
		encodingStatusFrame += 1;
	} catch {
		// ignore
	}
}

function startEncodingStatus(ctx: ExtensionContext): void {
	if (encodingStatusTimer) {
		clearInterval(encodingStatusTimer);
		encodingStatusTimer = undefined;
	}
	encodingStatusFrame = 0;
	renderEncodingInProgress(ctx);
	encodingStatusTimer = setInterval(() => renderEncodingInProgress(ctx), 400);
}

function stopEncodingStatus(): void {
	if (encodingStatusTimer) {
		clearInterval(encodingStatusTimer);
		encodingStatusTimer = undefined;
	}
}

function renderEncBadge(ctx: ExtensionContext, s: EncSummary): void {
	stopEncodingStatus();
	const tok = s.enc_out ?? 0;
	const sec = s.enc_seconds ?? 0;
	const tokStr = tok >= 1000 ? `${(tok / 1000).toFixed(1)}K` : String(tok);
	const label = `enc:${tokStr}t/${sec.toFixed(1)}s`;
	// Mirror the chat footer shape "(provider) model" on a second
	// status key so footer.ts can right-align it under the chat model line
	// while the meter label stays left-aligned. Encoder reasoning is always
	// disabled operationally, so do not render effort text here.
	const spec = (s.enc_model_spec ?? "").trim();
	const slash = spec.indexOf("/");
	const provider = slash > 0 ? spec.slice(0, slash) : "";
	const model = slash > 0 ? spec.slice(slash + 1) : spec;
	const modelLabel = model ? (provider ? `(${provider}) ${model}` : model) : "";
	try {
		ctx.ui.setStatus("jlc-enc", `${ANSI_PINK}${label}${ANSI_RESET}`);
		ctx.ui.setStatus("jlc-enc-model", modelLabel ? `${ANSI_PINK}${modelLabel}${ANSI_RESET}` : undefined);
	} catch {
		// ignore
	}
}

function setWorkStatus(ctx: ExtensionContext, text: string | undefined): void {
	try {
		ctx.ui.setStatus("jlc-work", text);
	} catch {
		// ignore
	}
}

function setAutoPromptStatus(ctx: ExtensionContext, state: AutoPromptState | undefined): void {
	try {
		ctx.ui.setStatus("jlc-auto", state ? `AUTO ${Math.min(state.idx + 1, state.total)}/${state.total}` : undefined);
	} catch {
		// ignore
	}
}

function notifyWork(ctx: ExtensionContext, text: string): void {
	void ctx;
	void text;
}

function sendJarvisChatNotice(pi: ExtensionAPI, text: string): void {
	try {
		pi.sendMessage({
			customType: "jarvis-status",
			content: text,
			display: true,
		});
	} catch {
		// ignore
	}
}

function formatUnregisterProjectStateLine(data: SidecarUnregisterResponse | undefined): string {
	const removed = data?.removed ? (data.project_id ?? "unknown") : "none";
	const remainingRaw = data?.remaining;
	if (!Array.isArray(remainingRaw)) return `removed: ${removed} / remaining: unknown`;
	const remaining = remainingRaw
		.map((project) => String(project.project_id ?? project.name ?? project.path ?? "").trim())
		.filter(Boolean);
	return `removed: ${removed} / remaining: ${remaining.length > 0 ? remaining.join(", ") : "(none)"}`;
}

function firstWarmupWarning(warnings: string[] | undefined): string | undefined {
	return warnings?.find(
		(warning) => warning.startsWith("JLC context degraded:") || warning.startsWith("JLC auto recall degraded:"),
	);
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function warmStartupContext(_ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	if (startupContextWarmupPromise) return startupContextWarmupPromise;
	startupContextWarmupFinished = false;
	startupContextWarmupPromise = (async () => {
		try {
			// 웜업은 침묵 (Jun 결정 2026-06-07) — 백그라운드로 충분, 실패/degraded만 알린다.
			const requestBody = {
				mode: "chat",
				user_message: "startup warmup",
				context_turn_key: "__jarvis_startup_warmup__",
				bench_conv_id: benchConvId(pi),
				hints: { startup_warmup: true },
			};
			let response: SidecarContextResponse | undefined;
			for (let attempt = 1; attempt <= STARTUP_WARMUP_ATTEMPTS; attempt++) {
				response = await postSidecar<SidecarContextResponse>("/context", requestBody, "POST", 90000);
				if (response || attempt === STARTUP_WARMUP_ATTEMPTS) break;
				await sleep(STARTUP_WARMUP_RETRY_MS);
			}
			if (!response || !isOkSidecarResponse(response)) {
				sendJarvisChatNotice(pi, `JARVIS memory warmup failed: ${response?.error ?? "sidecar unavailable"}`);
				return;
			}
			const warning = firstWarmupWarning(response.warnings);
			if (warning) {
				sendJarvisChatNotice(pi, `JARVIS memory warmup degraded: ${warning}`);
				return;
			}
		} catch (error) {
			sendJarvisChatNotice(pi, `JARVIS memory warmup failed: ${String(error)}`);
		} finally {
			startupContextWarmupFinished = true;
		}
	})();
	return startupContextWarmupPromise;
}

async function waitForStartupContextWarmup(_ctx: ExtensionContext): Promise<void> {
	if (!startupContextWarmupPromise || startupContextWarmupFinished) return;
	// 사용자가 웜업 중에 입력하면 조용히 완료를 기다렸다가 이어간다.
	await startupContextWarmupPromise;
}

function formatDuration(ms: number): string {
	const totalSeconds = Math.max(0, Math.floor(ms / 1000));
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = totalSeconds % 60;
	return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

function formatPlainTurnTokenCount(tokens: number): string {
	const rounded = Math.max(0, Math.round(tokens));
	return rounded.toLocaleString("en-US");
}

function formatTurnMeter(startedAtMs: number, totalTokens: number): string {
	const elapsed = formatDuration(Date.now() - startedAtMs);
	return `JCL working (${elapsed} · ${formatPlainTurnTokenCount(totalTokens)})`;
}

function logSubturnTokenUsage(
	reason: string,
	payloadSummary = "",
	payloadPreview = "",
	usage?: TurnUsageSummary,
	cacheReport?: JarvisCacheReport,
	prefixProbe?: JarvisPrefixProbe,
	compressionOutcome?: JarvisCompressionOutcome,
): void {
	const startedAtMs =
		reason === "assistant_end" ? (lastProviderStartedAtMs ?? lastTurnStartedAtMs) : lastTurnStartedAtMs;
	const elapsedSeconds = startedAtMs === undefined ? 0 : (Date.now() - startedAtMs) / 1000;
	void postSidecar("/subturn_meter", {
		reason,
		user_turn_key: subturnLogInitializedForUserTurnKey,
		call_input_tokens: displayedCallInputTokens,
		turn_input_tokens: displayedInputTokensThisTurn,
		output_tokens: displayedOutputTokensThisTurn,
		usage_input_tokens: usage?.input,
		usage_output_tokens: usage?.output,
		usage_cache_read_tokens: usage?.cacheRead,
		usage_cache_write_tokens: usage?.cacheWrite,
		usage_total_tokens: usage?.totalTokens,
		usage_reasoning_tokens: usage?.reasoningTokens,
		provider_cache_read_tokens: cacheReport?.provider_cache_read_tokens,
		provider_cache_write_tokens: cacheReport?.provider_cache_write_tokens,
		cache_meter: cacheReport?.cache_meter,
		cache_hit_pct: cacheReport?.cache_hit_pct,
		stable_prefix_hash: prefixProbe?.stable_prefix_hash,
		stable_prefix_tokens_est: prefixProbe?.stable_prefix_tokens_est,
		live_tokens_est: prefixProbe?.live_tokens_est,
		compressed_tool_outputs: compressionOutcome?.compressed_tool_outputs ?? 0,
		compression_saved_tokens_est: compressionOutcome?.compression_saved_tokens_est ?? 0,
		compression_skips: compressionOutcome?.compression_skips,
		provider_calls: providerCallCountThisTurn,
		elapsed_seconds: elapsedSeconds,
		payload_summary: payloadSummary,
		payload_preview: payloadPreview,
	});
}

function logSubturnObserveCandidate(
	providerCall: number,
	beforeMetrics: { message_tokens: number; tool_schema_tokens: number },
	legacyMetrics: { message_tokens: number; tool_schema_tokens: number },
	actualMetrics: { message_tokens: number; tool_schema_tokens: number },
	payloadSummary: string,
	payload: unknown,
): void {
	const candidateState = buildSubturnObserveState(payload);
	const candidateText = renderSubturnObserveState(candidateState);
	const candidateStateTokens = estimateTextTokenCount(candidateText);
	const stateCarryOn = subturnStateCarryEnabled();
	const actualBreakdown = buildProviderPayloadBreakdown(payload);
	void postSidecar("/debug/subturn/observe", {
		source: "jlc",
		event: "candidate_state",
		user_turn_key: subturnLogInitializedForUserTurnKey,
		legacy: {
			messages: providerPayloadMessageCount(payload),
			tokens: {
				before_messages: beforeMetrics.message_tokens,
				after_messages: legacyMetrics.message_tokens,
				tool_schema: legacyMetrics.tool_schema_tokens,
				trimmed_messages: Math.max(0, beforeMetrics.message_tokens - legacyMetrics.message_tokens),
				total: legacyMetrics.message_tokens,
			},
			summary: payloadSummary,
		},
		candidate: {
			messages: 1,
			tokens: {
				state: candidateStateTokens,
				total: candidateStateTokens,
			},
			state: candidateState,
			text: candidateText,
		},
		data: {
			provider_call: providerCall,
			user_turn_key: subturnLogInitializedForUserTurnKey,
			route: currentRoute,
			mode: currentMode,
			subturn_compact_enabled: subturnCompactEnabled(),
			subturn_state_carry_enabled: stateCarryOn,
			subturn_state_carry_recent_messages: getSubturnStateCarryRecentMessageLimit(),
			state_carry_applied: stateCarryOn && actualMetrics.message_tokens < legacyMetrics.message_tokens,
			actual_message_tokens: actualMetrics.message_tokens,
			actual_tool_schema_tokens: actualMetrics.tool_schema_tokens,
			actual_breakdown: actualBreakdown,
		},
		notes: [
			stateCarryOn ? "Actual provider payload uses opt-in state carry." : "Actual provider payload unchanged.",
			"Candidate total counts sticky subturn state only; fixed prompt/tool schema tokens are not included.",
		],
	});
}

function updateTurnMeter(): void {
	// Token meter footer display suppressed; tracking continues via the
	// completed/current counters used elsewhere.
}

function startTurnMeter(ctx: ExtensionContext): void {
	clearTurnMeter(false);
	turnMeterCtx = ctx;
	trimBeforeTokensSum = 0;
	trimAfterTokensSum = 0;
	lastToolSchemaTokens = 0;
	displayedInputTokensThisTurn = 0;
	displayedCallInputTokens = 0;
	displayedOutputTokensThisTurn = 0;
	completedOutputTokensThisTurn = 0;
	turnMeterCompletedInputTokens = 0;
	turnMeterCompletedTotalTokens = 0;
	turnMeterCurrentOutputTokens = 0;
	turnMeterCompletedAssistantMessages = 0;
	// NOTE: jarvisEvidenceByToolResultKey / subturnActiveToolMetadata are NOT
	// reset here. before_agent_start fires again on mid-turn route switches
	// (chat -> deepdive), and wiping the evidence map there made wire
	// compression permanently empty (R1-3 live finding, 2026-06-07). Their
	// lifecycle is the user turn — see initializeSubturnLog.
	// No periodic footer update — display is suppressed (see updateTurnMeter).
}

function updateTurnMeterFromText(ctx: ExtensionContext, text: string): void {
	turnMeterCtx = ctx;
	turnMeterCurrentOutputTokens = estimateTextTokenCount(sanitizeAssistantText(text, true));
	displayedOutputTokensThisTurn = completedOutputTokensThisTurn + turnMeterCurrentOutputTokens;
	updateTurnMeter();
}

function clearTurnMeter(clearStatus: boolean): void {
	if (turnMeterTimer) {
		clearInterval(turnMeterTimer);
		turnMeterTimer = undefined;
	}
	if (clearStatus && turnMeterCtx) {
		setWorkStatus(turnMeterCtx, undefined);
	}
	turnMeterCtx = undefined;
	turnMeterCompletedInputTokens = 0;
	turnMeterCompletedTotalTokens = 0;
	turnMeterCurrentOutputTokens = 0;
	turnMeterCompletedAssistantMessages = 0;
	lastProviderStartedAtMs = undefined;
}

function finishTurnMeter(ctx: ExtensionContext, assistantMessage: AssistantMessage | undefined): void {
	const startedAt = lastTurnStartedAtMs;
	if (startedAt === undefined) {
		clearTurnMeter(true);
		return;
	}
	if (assistantMessage?.usage && turnMeterCompletedAssistantMessages === 0) {
		recordCompletedTurnMeterUsage(assistantMessage);
	}
	const currentInputTokens = Math.max(0, trimAfterTokensSum - turnMeterCompletedInputTokens);
	const totalTokens = Math.max(0, turnMeterCompletedTotalTokens + currentInputTokens + turnMeterCurrentOutputTokens);
	const label = formatTurnMeter(startedAt, totalTokens);
	if (turnMeterTimer) {
		clearInterval(turnMeterTimer);
		turnMeterTimer = undefined;
	}
	// Footer display suppressed — label computed for notifyWork only.
	notifyWork(ctx, label);
	turnMeterCtx = undefined;
	turnMeterCurrentOutputTokens = 0;
}

function recordCompletedTurnMeterUsage(message: AssistantMessage): void {
	const usage = message.usage;
	if (!usage) return;
	turnMeterCompletedAssistantMessages += 1;
	turnMeterCompletedInputTokens += Math.max(0, usage.input ?? 0);
	completedOutputTokensThisTurn += Math.max(0, usage.output ?? 0);
	displayedOutputTokensThisTurn = completedOutputTokensThisTurn;
	turnMeterCompletedTotalTokens += Math.max(
		0,
		usage.totalTokens ??
			Math.max(0, usage.input ?? 0) +
				Math.max(0, usage.output ?? 0) +
				Math.max(0, usage.cacheRead ?? 0) +
				Math.max(0, usage.cacheWrite ?? 0),
	);
	turnMeterCurrentOutputTokens = 0;
}

function summarizeAssistantUsage(messages: AgentMessage[]): TurnUsageSummary | undefined {
	const summary: TurnUsageSummary = {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	let seen = false;
	let reasoningTokens = 0;
	for (const message of messages) {
		if (message.role !== "assistant") continue;
		const usage = (message as AssistantMessage).usage;
		if (!usage) continue;
		seen = true;
		const input = Math.max(0, usage.input ?? 0);
		const output = Math.max(0, usage.output ?? 0);
		const cacheRead = Math.max(0, usage.cacheRead ?? 0);
		const cacheWrite = Math.max(0, usage.cacheWrite ?? 0);
		summary.input += input;
		summary.output += output;
		summary.cacheRead += cacheRead;
		summary.cacheWrite += cacheWrite;
		summary.totalTokens += Math.max(0, usage.totalTokens ?? input + output + cacheRead + cacheWrite);
		summary.cost.input += Math.max(0, usage.cost?.input ?? 0);
		summary.cost.output += Math.max(0, usage.cost?.output ?? 0);
		summary.cost.cacheRead += Math.max(0, usage.cost?.cacheRead ?? 0);
		summary.cost.cacheWrite += Math.max(0, usage.cost?.cacheWrite ?? 0);
		summary.cost.total += Math.max(0, usage.cost?.total ?? 0);
		reasoningTokens += Math.max(0, (usage as { reasoningTokens?: number }).reasoningTokens ?? 0);
	}
	if (!seen) return undefined;
	if (reasoningTokens > 0) summary.reasoningTokens = reasoningTokens;
	return summary;
}

function appendTransientSystemDirective(text: string): void {
	const trimmed = text.trim();
	if (!trimmed) return;
	transientSystemDirective = transientSystemDirective.trim()
		? `${transientSystemDirective.trim()}\n${trimmed}`
		: trimmed;
}

function reportEncoderFailure(pi: ExtensionAPI, error: string | null | undefined): void {
	const detail = (error ?? "").trim();
	if (!detail) return;
	sendJarvisChatNotice(pi, `JLC 인코더 오류: ${detail}\n이전 메모리를 유지합니다.`);
}

function isRetryableProviderErrorMessage(message: string): boolean {
	return /overloaded|provider.?returned.?error|rate.?limit|too many requests|429|500|502|503|504|service.?unavailable|server.?error|internal.?error|network.?error|connection.?error|connection.?refused|connection.?lost|websocket.?closed|websocket.?error|other side closed|fetch failed|upstream.?connect|reset before headers|socket hang up|ended without|stream ended before message_stop|http2 request did not get a response|timed? out|timeout|terminated|retry delay/i.test(
		message,
	);
}

function isRetryableAssistantError(message: AssistantMessage | undefined): boolean {
	if (!message || message.stopReason !== "error") return false;
	const detail = [message.errorMessage ?? "", contentToText(message.content)].filter(Boolean).join("\n");
	return isRetryableProviderErrorMessage(detail);
}

function encodingStatusPath(convId: string, minTurn?: number | null): string {
	const minTurnParam = minTurn ? `&min_turn=${encodeURIComponent(String(minTurn))}` : "";
	return `/encoding_status?conv_id=${encodeURIComponent(convId)}&clear=true${minTurnParam}`;
}

async function pollEncodingStatus(
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	convId: string,
	minTurn?: number | null,
): Promise<void> {
	const deadline = Date.now() + Math.max(ENCODING_STATUS_WAIT_MS, ENCODING_STATUS_POLL_MS);
	try {
		while (Date.now() < deadline) {
			await new Promise((r) => setTimeout(r, ENCODING_STATUS_POLL_MS));
			try {
				const data = await postSidecar<{
					ok?: boolean;
					ready?: boolean;
					enc_out?: number;
					enc_seconds?: number;
					jhb_tokens?: number;
					jhb_delta?: number;
					error?: string | null;
				}>(encodingStatusPath(convId, minTurn), undefined, "GET");
				if (!data?.ready) continue;
				renderEncBadge(ctx, data);
				reportEncoderFailure(pi, data.error);
				return;
			} catch {
				// transient — retry
			}
		}
	} finally {
		stopEncodingStatus();
	}
}

async function waitForEncodingStatus(
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	convId: string,
	minTurn?: number | null,
): Promise<boolean> {
	if (!convId.trim() || AUTO_PROMPT_ENCODING_WAIT_MS <= 0) return false;
	const deadline = Date.now() + AUTO_PROMPT_ENCODING_WAIT_MS;
	while (Date.now() < deadline) {
		try {
			const data = await postSidecar<{
				ok?: boolean;
				ready?: boolean;
				enc_out?: number;
				enc_seconds?: number;
				jhb_tokens?: number;
				jhb_delta?: number;
				error?: string | null;
			}>(encodingStatusPath(convId, minTurn), undefined, "GET");
			if (data?.ready) {
				renderEncBadge(ctx, data);
				reportEncoderFailure(pi, data.error);
				return true;
			}
		} catch {
			// transient — retry
		}
		await new Promise((r) => setTimeout(r, ENCODING_STATUS_POLL_MS));
	}
	return false;
}

function jlcLabel(state: "checking" | "down" | "degraded" | "ok", projectName?: string): string {
	if (state === "checking") return `${ANSI_YELLOW}JLC checking${ANSI_RESET}`;
	if (state === "down") return `${ANSI_RED}JLC down${ANSI_RESET}`;
	if (state === "degraded") return `${ANSI_RED}JLC degraded${ANSI_RESET}`;
	const color = isProjectRoute(currentRoute) ? ANSI_RED : ANSI_YELLOW;
	const label = `JLC:${routeStatusLabel(currentRoute)}`;
	const body = isProjectRoute(currentRoute) && projectName ? `${label}:${projectName}` : label;
	return `${color}${body}${ANSI_RESET}`;
}

function isProjectRoute(route: EffectiveTurnRoute): boolean {
	return route === "deepdive" || route === "heavy_deepdive";
}

function modeForRoute(route: EffectiveTurnRoute): SidecarContextMode {
	return isProjectRoute(route) ? "deepdive" : "chat";
}

function routeStatusLabel(route: EffectiveTurnRoute): string {
	if (route === "heavy_deepdive") return "heavy deepdive";
	if (route === "deepdive") return "deepdive";
	if (route === "unregistered_coding") return "unregistered coding";
	return "chat mode";
}

function setEffectiveRoute(route: EffectiveTurnRoute, reasoningLevel?: SupportedThinkingLevel): void {
	currentRoute = route;
	currentMode = modeForRoute(route);
	if (reasoningLevel) {
		saveDeepdiveThinkingPreference(reasoningLevel);
	}
}

function isSupportedThinkingLevel(value: unknown): value is SupportedThinkingLevel {
	return (
		value === "off" ||
		value === "minimal" ||
		value === "low" ||
		value === "medium" ||
		value === "high" ||
		value === "xhigh"
	);
}

function normalizeThinkingLevel(value: string | undefined): SupportedThinkingLevel | undefined {
	const normalized = value?.trim().toLowerCase();
	if (!normalized) return undefined;
	if (normalized === "max" || normalized === "maximum") {
		return "xhigh";
	}
	return normalized === "medium" || normalized === "high" || normalized === "xhigh" ? normalized : undefined;
}

function jarvisUiStatePath(): string {
	return process.env.JARVIS_UI_STATE_PATH ?? path.resolve(process.cwd(), "..", "data", "jarvis-ui-state.json");
}

function loadDeepdiveThinkingPreference(): SupportedThinkingLevel | undefined {
	if (deepdiveThinkingPreferenceLoaded) return deepdiveThinkingPreference;
	deepdiveThinkingPreferenceLoaded = true;
	try {
		const statePath = jarvisUiStatePath();
		if (!fs.existsSync(statePath)) return undefined;
		const parsed = JSON.parse(fs.readFileSync(statePath, "utf-8")) as {
			deepdiveThinkingLevel?: unknown;
		};
		const parsedLevel = normalizeThinkingLevel(String(parsed.deepdiveThinkingLevel ?? ""));
		if (parsedLevel) {
			deepdiveThinkingPreference = parsedLevel;
		}
	} catch {
		// Corrupt or inaccessible state should not break startup.
	}
	return deepdiveThinkingPreference;
}

function saveDeepdiveThinkingPreference(level: SupportedThinkingLevel): void {
	deepdiveThinkingPreferenceLoaded = true;
	deepdiveThinkingPreference = level;
	try {
		const statePath = jarvisUiStatePath();
		let state: Record<string, unknown> = {};
		if (fs.existsSync(statePath)) {
			const parsed = JSON.parse(fs.readFileSync(statePath, "utf-8")) as unknown;
			if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
				state = parsed as Record<string, unknown>;
			}
		}
		state.deepdiveThinkingLevel = level;
		fs.mkdirSync(path.dirname(statePath), { recursive: true });
		fs.writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf-8");
	} catch {
		// Preference is still kept for the current process.
	}
}

function suppressThinkingPreferenceSaveOnce(level: SupportedThinkingLevel): void {
	suppressNextThinkingPreferenceSave = level;
	setTimeout(() => {
		if (suppressNextThinkingPreferenceSave === level) {
			suppressNextThinkingPreferenceSave = undefined;
		}
	}, 0);
}

function parseDeepdiveReasoningUtterance(text: string): SupportedThinkingLevel | undefined {
	const normalized = text.trim().toLowerCase();
	const hasDeepdiveCue = /\bdeep\s*dive\b|\bdeepdive\b/i.test(normalized);
	const hasReasoningCue = /\breasoning\b|\bthinking\b/i.test(normalized);
	const hasLevelCue = /\b(?:xhigh|max|maximum|high|medium|fast|quick|light)\b/i.test(normalized);
	if (!hasDeepdiveCue && !(hasReasoningCue && hasLevelCue)) {
		return undefined;
	}
	if (/\b(?:xhigh|max|maximum)\b/i.test(normalized)) return "xhigh";
	if (/\bhigh\b/i.test(normalized)) return "high";
	if (/\bmedium\b|\bfast\b|\bquick\b|\blight\b/i.test(normalized)) return "medium";
	return undefined;
}

function stripTrailingPathPunctuation(value: string): string {
	return value
		.replace(/\s+(?:을|를|에|에서|으로|로)\s+.*$/i, "")
		.replace(/\s+(?:을|를|에|에서|으로|로)\s*$/i, "")
		.replace(/[.,;:!?]+$/g, "")
		.replace(/[)\]}]+$/g, "")
		.trim();
}

function isAbsolutePathCandidate(value: string): boolean {
	return /^[A-Za-z]:[\\/]/.test(value) || /^\/[^/\s]+\/.+/.test(value);
}

function extractAbsolutePathsFromText(text: string): string[] {
	const paths: string[] = [];
	const quoted = /["'“‘]([A-Za-z]:[\\/][^"'“”‘’\r\n]+|\/[^"'“”‘’\r\n]+)["'”’]/g;
	for (const match of text.matchAll(quoted)) {
		const value = stripTrailingPathPunctuation(match[1] ?? "");
		if (value && isAbsolutePathCandidate(value)) paths.push(value);
	}
	const unquoted = /(?:^|[\s(])([A-Za-z]:[\\/][^\s)\]}>,;]+|\/[^\s)\]}>,;]+)/g;
	for (const match of text.matchAll(unquoted)) {
		const value = stripTrailingPathPunctuation(match[1] ?? "");
		if (value && isAbsolutePathCandidate(value)) paths.push(value);
	}
	return [...new Set(paths.map((value) => path.resolve(value)))];
}

function routePromptForRoute(route: EffectiveTurnRoute): string {
	if (route === "unregistered_coding") return UNREGISTERED_CODING_ROUTE_PROMPT;
	if (route === "heavy_deepdive") return HEAVY_DEEPDIVE_ROUTE_PROMPT;
	if (route === "deepdive") return DEEPDIVE_ROUTE_PROMPT;
	return CHAT_ROUTE_PROMPT;
}

const ASSISTANT_PROJECT_ROUTE_TOOL_NAMES = new Set([
	"switch_project",
	"register_project",
	"unregister_project",
	"update_jarvis_md",
]);
const CHAT_ROUTE_ONLY_TOOL_NAMES = new Set(["web_search", "web_fetch", "docs_search", "package_info"]);

function leadingModeMarker(text: string): AssistantModeMarker | undefined {
	const match = text.match(/^\s*\[MODE:(CHAT|UNREGISTERED_CODING|DEEPDIVE|HEAVY_DEEPDIVE)\]/i);
	if (!match) return undefined;
	const marker = match[1]?.toLowerCase();
	if (marker === "heavy_deepdive") return "heavy_deepdive";
	if (marker === "unregistered_coding") return "unregistered_coding";
	return marker === "deepdive" ? "deepdive" : "chat";
}

function modeMarkerFromPrefix(marker: string): AssistantModeMarker {
	if (marker === "[MODE:HEAVY_DEEPDIVE]") return "heavy_deepdive";
	if (marker === "[MODE:UNREGISTERED_CODING]") return "unregistered_coding";
	if (marker === "[MODE:DEEPDIVE]") return "deepdive";
	return "chat";
}

function partialLeadingModeMarker(text: string): AssistantModeMarker | undefined {
	const leadingWhitespace = text.match(/^\s*/)?.[0] ?? "";
	const body = text.slice(leadingWhitespace.length).trimEnd().toLowerCase();
	if (!body || body.includes("\n")) return undefined;
	const matches = MODE_MARKER_PREFIXES.filter((marker) => marker.toLowerCase().startsWith(body));
	return matches.length === 1 ? modeMarkerFromPrefix(matches[0]) : undefined;
}

function assistantToolNames(message: AssistantMessage): string[] {
	const names: string[] = [];
	const rawToolCalls = (message as unknown as { tool_calls?: unknown }).tool_calls;
	if (Array.isArray(rawToolCalls)) {
		for (const call of rawToolCalls) {
			if (!call || typeof call !== "object") continue;
			const record = call as Record<string, unknown>;
			const fn =
				record.function && typeof record.function === "object"
					? (record.function as Record<string, unknown>)
					: undefined;
			const name = typeof record.name === "string" ? record.name : typeof fn?.name === "string" ? fn.name : "";
			if (name) names.push(name.toLowerCase());
		}
	}

	const content = (message as unknown as { content?: unknown }).content;
	if (Array.isArray(content)) {
		for (const part of content) {
			if (!part || typeof part !== "object") continue;
			const record = part as Record<string, unknown>;
			const type = String(record.type ?? "").toLowerCase();
			if (type !== "toolcall" && type !== "tool_call" && type !== "function_call") continue;
			const fn =
				record.function && typeof record.function === "object"
					? (record.function as Record<string, unknown>)
					: undefined;
			const name = typeof record.name === "string" ? record.name : typeof fn?.name === "string" ? fn.name : "";
			if (name) names.push(name.toLowerCase());
		}
	}
	return [...new Set(names)];
}

function isSlashCommand(text: string, command: string): boolean {
	const normalized = text.trim().toLowerCase();
	return normalized === command || normalized.startsWith(`${command} `);
}

function hasExplicitProjectRegistrationIntent(text: string): boolean {
	const normalized = text.trim().toLowerCase();
	if (!normalized) return false;
	if (/(등록\s*안|등록하지|등록\s*없이|등록없이|without\s+register|do\s+not\s+register)/i.test(text)) {
		return false;
	}
	if (isSlashCommand(normalized, "/jarvis-register")) return true;
	if (/\b(?:register|add)\b.*\b(?:jarvis\s+)?project\b/i.test(normalized)) return true;
	if (/\bjarvis\s+project\b.*\b(?:register|add)\b/i.test(normalized)) return true;
	if (/(jarvis\s*프로젝트|자비스\s*프로젝트).*(등록|추가)/i.test(text)) return true;
	if (/(등록|추가).*(jarvis\s*프로젝트|자비스\s*프로젝트)/i.test(text)) return true;
	if (/프로젝트\s*등록|등록\s*프로젝트/i.test(text)) return true;
	return false;
}

function canonicalProjectText(value: string): string {
	return value
		.normalize("NFKD")
		.replace(/[\u0300-\u036f]/g, "")
		.toLowerCase()
		.replace(/[_./-]+/g, " ")
		.replace(/[^a-z0-9가-힣]+/g, " ")
		.replace(/\s+/g, " ")
		.trim();
}

function enterProjectWork(reasoningLevel?: SupportedThinkingLevel): void {
	setEffectiveRoute("deepdive", reasoningLevel);
}

function enterProjectWorkPreservingHeavy(reasoningLevel?: SupportedThinkingLevel): void {
	if (currentRoute === "heavy_deepdive") {
		if (reasoningLevel) {
			saveDeepdiveThinkingPreference(reasoningLevel);
		}
		return;
	}
	enterProjectWork(reasoningLevel);
}

function enterHeavyProjectWork(reasoningLevel?: SupportedThinkingLevel): void {
	setEffectiveRoute("heavy_deepdive", reasoningLevel);
}

function enterUnregisteredCoding(): void {
	setEffectiveRoute("unregistered_coding");
}

function activateRouteFromAssistantSignal(
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	routeHint?: "unregistered_coding" | "deepdive" | "heavy_deepdive",
): void {
	if (routeHint === "unregistered_coding") {
		enterUnregisteredCoding();
	} else if (routeHint === "heavy_deepdive") {
		enterHeavyProjectWork();
	} else if (!isProjectRoute(currentRoute)) {
		enterProjectWorkPreservingHeavy();
	}
	try {
		if (isProjectRoute(currentRoute)) {
			const level = applyRouteThinkingLevel(currentRoute, ctx, pi);
			ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, level);
		} else {
			ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, undefined);
		}
		ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down", lastContextResponse?.project_name));
	} catch {
		/* ctx/pi may be stale */
	}
}

function applyChatDefaultThinking(ctx: ExtensionContext, pi: ExtensionAPI): void {
	if (pi.getThinkingLevel() !== "medium") {
		suppressThinkingPreferenceSaveOnce("medium");
		pi.setThinkingLevel("medium");
	}
	ctx.ui.setHiddenThinkingLabel(" ");
}

function resetToChatMode(ctx: ExtensionContext, pi: ExtensionAPI, options: { updateFooter?: boolean } = {}): void {
	setEffectiveRoute("chat");
	try {
		applyChatDefaultThinking(ctx, pi);
		ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, undefined);
		if (options.updateFooter !== false) {
			ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down"));
		}
	} catch {
		/* pi or ctx may be stale */
	}
}

function resetToChatModeAfterEncoding(ctx: ExtensionContext, pi: ExtensionAPI): void {
	setTimeout(() => resetToChatMode(ctx, pi), 1200);
}

function refreshTurnCheckpointScope(): void {
	if (isProjectRoute(currentRoute) && activeProjectPath) {
		turnCheckpointScope = { kind: "project", path: activeProjectPath };
		return;
	}
	turnCheckpointScope = { kind: "chat", path: ensureChatMemoryRoot() };
}

async function clearInterruptCheckpointForScope(scope: CheckpointScope | undefined): Promise<void> {
	if (!scope?.path) return;
	await postSidecar<SidecarInterruptCheckpointResponse>("/interrupt_checkpoint/clear", {
		project_path: scope.path,
	});
}

function selectDeepdiveThinkingLevel(ctx: ExtensionContext): SupportedThinkingLevel {
	const preferred = loadDeepdiveThinkingPreference();
	if (preferred) return preferred;
	const model = ctx.model;
	if (!model?.reasoning) return "medium";

	const levelMap = model.thinkingLevelMap;
	const ordered: SupportedThinkingLevel[] = ["xhigh", "high", "medium"];
	if (levelMap) {
		for (const level of ordered) {
			if (levelMap[level] !== null) return level;
		}
	}

	return "xhigh";
}

export default function jarvisJlc(pi: ExtensionAPI) {
	pi.registerFlag("auto-prompts", {
		type: "string",
		description: "Path to newline-separated prompts file. Auto-fires each turn after agent_end.",
	});
	pi.registerFlag("bench-conv", {
		type: "string",
		description: "Bench conv_id. Isolate JHB/raw storage for benchmark runs only.",
	});

	let autoPromptState: AutoPromptState | undefined;
	const CODE_TOOL_NAMES = new Set(["edit", "write", "write_file"]);
	const READ_TOOL_NAMES = new Set(["read", "ls", "grep", "find"]);
	let lastObservedUserTurnKey = "";
	let lastInjectedContextTurnKey = "";
	let pendingProjectSwitchContextRefresh = false;
	let safetyConfirmedKeys = new Set<string>();

	pi.on("session_start", async (_event, ctx) => {
		recordFooterMeterReset(pi);
		clearAutoPromptWatchdog();
		loadDeepdiveThinkingPreference();
		lastObservedUserTurnKey = "";
		lastInjectedContextTurnKey = "";
		pendingProjectSwitchContextRefresh = false;
		safetyConfirmedKeys = new Set<string>();
		try {
			autoPromptState = loadAutoPromptState(pi.getFlag("auto-prompts"));
		} catch {
			/* pi stale on reload — keep existing state */
		}

		setEffectiveRoute("chat");
		try {
			ctx.ui.setStatus("jarvis", jlcLabel("checking"));
		} catch {
			/* stale */
		}
		sidecarHealthy = await checkHealth();
		try {
			ctx.ui.setStatus("jarvis", sidecarHealthy ? jlcLabel("ok") : jlcLabel("down"));
			ctx.ui.setHiddenThinkingLabel(" ");
		} catch {
			/* stale */
		}
		if (sidecarHealthy) {
			await refreshProjectCache();
			const sidecarStatus = await postSidecar<SidecarStatusResponse>("/status", undefined, "GET");
			setupRequired = sidecarStatus?.setup_required === true;
			if (setupRequired) {
				try {
					ctx.ui.notify(
						"JARVIS setup required: reply with an absolute path or use /setup-default-root <path>.",
						"warning",
					);
				} catch {
					/* stale */
				}
			}
			if (!coldStartNoticeShown) {
				coldStartNoticeShown = true;
				void warmStartupContext(ctx, pi);
			}
		} else {
			projectCache = [];
			projectCacheLoaded = false;
		}
		if (autoPromptState) {
			await fireAutoPrompt(autoPromptState, ctx, pi);
		}
	});

	pi.on("thinking_level_select", async (event, ctx) => {
		const level = normalizeThinkingLevel(isSupportedThinkingLevel(event.level) ? event.level : undefined);
		if (level && suppressNextThinkingPreferenceSave === level) {
			suppressNextThinkingPreferenceSave = undefined;
		} else if (level) {
			saveDeepdiveThinkingPreference(level);
			if (isProjectRoute(currentRoute)) {
				try {
					ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, level);
				} catch {
					/* stale */
				}
			}
		}
		try {
			const label = isProjectRoute(currentRoute) && event.level !== "off" ? `Thinking ${event.level}` : " ";
			ctx.ui.setHiddenThinkingLabel(label);
		} catch {
			/* stale */
		}
	});

	pi.on("session_shutdown", async () => {
		clearInterruptInputCheckpointHook();
	});

	pi.on("context", async (event, ctx) => {
		if (DEBUG_CONTEXT) console.error("[jlc:debug-context-handler] ENTER");
		const userText = stripJarvisMemoryBlock(latestUserText(event.messages));
		if (!userText.trim()) return;
		const userTurnKey = latestUserTurnKey(event.messages, userText);
		const isNewUserTurn = userTurnKey !== lastObservedUserTurnKey;
		if (isNewUserTurn) {
			lastObservedUserTurnKey = userTurnKey;
			lastInjectedContextTurnKey = "";
			pendingProjectSwitchContextRefresh = false;
			providerCallCountThisTurn = 0;
			resetProviderCallCeilingState();
			lastTurnStartedAtMs = Date.now();
			lastProviderStartedAtMs = undefined;
			setEffectiveRoute("chat");
			checkpointToolEvents = [];
			lastAssistantPartialText = "";
			lastAssistantObservedModeMarker = undefined;
			interruptCheckpointSavedThisTurn = false;
			turnCheckpointScope = undefined;
			resetSubturnLogState();
			resetJarvisTurnChoreographyState();
		}
		if (
			!isNewUserTurn &&
			providerCallCountThisTurn > 0 &&
			lastInjectedContextTurnKey &&
			!pendingProjectSwitchContextRefresh
		) {
			return undefined;
		}
		if (!isNewUserTurn && lastInjectedContextTurnKey === userTurnKey && !pendingProjectSwitchContextRefresh) {
			return undefined;
		}
		lastUserMessage = userText;
		const normalizedUser = userText.trim().toLowerCase();
		const explicitChat = isSlashCommand(normalizedUser, "/chat");
		const explicitDeepdive = isSlashCommand(normalizedUser, "/deepdive");
		const utteredDeepdiveLevel = explicitDeepdive ? parseDeepdiveReasoningUtterance(userText) : undefined;
		if (explicitChat) {
			clearActiveProjectState();
			setEffectiveRoute("chat");
		}
		// Resolve cwd defensively: ctx.cwd throws if the extension runtime is
		// stale (session replacement or reload between turns). Fall back to
		// process.cwd() so the sidecar call still succeeds.
		let cwdHint: string;
		try {
			cwdHint = ctx.cwd;
		} catch {
			cwdHint = process.cwd();
		}

		try {
			await maybeHandleSetupFlow(userText, ctx);
		} catch {
			/* ctx stale — skip setup */
		}
		try {
			if (await maybeHandlePendingProjectCreation(userText, ctx)) {
				pendingProjectSwitchContextRefresh = true;
			}
		} catch {
			/* ctx stale — skip pending */
		}
		if (!explicitChat) {
			try {
				if (await maybeSwitchProjectFromUserMessage(userText, ctx)) {
					pendingProjectSwitchContextRefresh = true;
				}
			} catch {
				/* ctx stale — skip switch */
			}
		}
		if (!explicitChat && explicitDeepdive) {
			enterHeavyProjectWork(utteredDeepdiveLevel);
		}
		if (isProjectRoute(currentRoute) && !activeProjectPath && explicitDeepdive) {
			appendTransientSystemDirective(
				[
					"[Project clarification]",
					"Deepdive needs a registered workspace project. If the target project is clear, call switch_project first; otherwise ask which project to use before editing files or updating JARVIS.md.",
				].join("\n"),
			);
		}
		const turnMode: SidecarContextMode = modeForRoute(currentRoute);

		await waitForStartupContextWarmup(ctx);
		const response = await postSidecar<SidecarContextResponse>("/context", {
			cwd_hint: cwdHint,
			mode: turnMode,
			user_message: userText,
			active_project_path: turnMode === "deepdive" ? currentActiveProjectHint() : undefined,
			bench_conv_id: benchConvId(pi),
			context_turn_key: userTurnKey,
			hints: {
				cwd: cwdHint,
			},
		});

		if (isNewUserTurn) {
			const degradation = response?.warnings?.find((warning) => warning.startsWith("JLC context degraded:"));
			if (degradation) {
				sendJarvisChatNotice(pi, `JLC 초기화 오류: ${degradation.slice("JLC context degraded:".length).trim()}`);
			}
		}

		if (!response?.context) {
			try {
				ctx.ui.setStatus("jarvis", jlcLabel("degraded"));
			} catch {
				/* stale */
			}
			return;
		}

		sidecarHealthy = true;
		if (isProjectRoute(currentRoute)) {
			applyContextProjectState(response, turnMode);
		}
		if (explicitChat) {
			clearActiveProjectState();
		}
		lastContextResponse = response;
		lastInjectedContextMode = turnMode;
		const selectedProjectRoot = isProjectRoute(currentRoute) ? activeProjectPath : undefined;
		refreshTurnCheckpointScope();
		const subturnRoot = selectedProjectRoot ?? ensureChatMemoryRoot();
		const subturnMode = modeForRoute(currentRoute);
		initializeSubturnLog(subturnRoot, userText, userTurnKey, cwdHint, subturnMode);
		setupRequired = response.setup_required === true;
		if (isAmbiguousProjectSelection(response.trace)) {
			const candidates = formatAmbiguousProjectCandidates(response.trace);
			appendTransientSystemDirective(
				[
					"[Ambiguous project selection]",
					candidates
						? `The user's utterance matches multiple registered projects: ${candidates}.`
						: "The user's utterance matches multiple registered projects.",
					"Ask which project they mean before switching or writing memory.",
				].join("\n"),
			);
			try {
				ctx.ui.notify("프로젝트가 애매합니다. 먼저 확인 질문을 하겠습니다.", "warning");
			} catch {
				/* stale */
			}
		}
		try {
			if (isProjectRoute(currentRoute)) {
				const level = applyRouteThinkingLevel(currentRoute, ctx, pi);
				ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, level);
			} else {
				ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, undefined);
			}
			ctx.ui.setStatus("jarvis", jlcLabel("ok", response.project_name));
		} catch {
			/* stale */
		}
		lastInjectedContextTurnKey = userTurnKey;
		pendingProjectSwitchContextRefresh = false;
		return { messages: injectMemoryIntoLatestUser(event.messages, response.context) };
	});

	pi.on("before_agent_start", async (event, ctx) => {
		installInterruptInputCheckpointHook(ctx, pi);
		const messages = (event as { messages?: AgentMessage[] }).messages ?? [];
		const promptText = (event as { prompt?: string }).prompt ?? "";
		const userText = promptText.trim() || lastUserMessage;
		const normalizedUser = userText.trim().toLowerCase();
		const explicitChat = isSlashCommand(normalizedUser, "/chat");
		const explicitDeepdive = isSlashCommand(normalizedUser, "/deepdive");
		const utteredDeepdiveLevel = explicitDeepdive ? parseDeepdiveReasoningUtterance(userText) : undefined;
		if (explicitChat) {
			clearActiveProjectState();
			setEffectiveRoute("chat");
		} else if (explicitDeepdive) {
			enterHeavyProjectWork(utteredDeepdiveLevel);
		}

		const activeProjectForPreflight = isProjectRoute(currentRoute)
			? (lastContextResponse?.active_project_path ?? currentActiveProjectHint())
			: undefined;
		refreshTurnCheckpointScope();
		const preflight = activeProjectForPreflight
			? "[P1] project memory is already injected. Do not announce a JARVIS.md read."
			: "";
		const overlay = transientSystemDirective.trim();
		transientSystemDirective = "";
		const existingPrompt = (event as { systemPrompt?: string }).systemPrompt ?? "";
		const modePrompt = isProjectRoute(currentRoute) ? DEEPDIVE_MODE_PROMPT : CHAT_MODE_PROMPT;
		const routePrompt = routePromptForRoute(currentRoute);
		const workspaceBlock = (lastContextResponse?.workspace_block ?? "").trim();
		const parts: string[] = [];
		if (existingPrompt.trim()) parts.push(existingPrompt);
		parts.push(LOCAL_LANGUAGE_PROMPT);
		parts.push(modePrompt);
		parts.push(routePrompt);
		if (workspaceBlock) parts.push(workspaceBlock);
		if (preflight) parts.push(preflight);
		if (overlay) parts.push(overlay);

		try {
			setAutoPromptStatus(ctx, autoPromptState);
			const level = isProjectRoute(currentRoute) ? applyRouteThinkingLevel(currentRoute, ctx, pi) : undefined;
			if (!isProjectRoute(currentRoute)) {
				applyChatDefaultThinking(ctx, pi);
			}
			ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, level);
			ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down"));
			if (isProjectRoute(currentRoute)) {
				setWorkStatus(ctx, undefined);
				notifyWork(
					ctx,
					`JLC: ${routeStatusLabel(currentRoute)} started. Thinking ${level}; waiting for the model/tool calls.`,
				);
			} else {
				setWorkStatus(ctx, undefined);
			}
		} catch {
			// ignore
		}

		const systemPrompt = parts.join("\n\n");
		if (DEBUG_CONTEXT) {
			const roleCounts = new Map<string, number>();
			let historyTokens = 0;
			for (const message of messages) {
				const role = String(message.role ?? "unknown");
				roleCounts.set(role, (roleCounts.get(role) ?? 0) + 1);
				historyTokens += estimateMessageTokenCount(message);
			}
			const promptContextTokens = estimateTextTokenCount(buildPromptContextText(lastContextResponse?.context));
			const roleSummary = Array.from(roleCounts.entries())
				.map(([role, count]) => `${role}=${count}`)
				.join(" ");
			console.error(
				`[jlc:debug-context] messages=${messages.length} ${roleSummary} history_tokens~=${historyTokens} prompt_ctx~=${promptContextTokens} latest_user~=${estimateTextTokenCount(userText)} system_prompt~=${estimateTextTokenCount(systemPrompt)}`,
			);
			console.error(
				`[jlc:debug-system-parts] mode_prompt~=${estimateTextTokenCount(modePrompt)} preflight~=${estimateTextTokenCount(preflight)} overlay~=${estimateTextTokenCount(overlay)} existing_prompt~=${estimateTextTokenCount(existingPrompt)}`,
			);
			if (existingPrompt) {
				const lines = existingPrompt.split("\n").length;
				const head = existingPrompt.slice(0, 120).replace(/\n/g, " ⏎ ");
				const tail = existingPrompt.slice(-120).replace(/\n/g, " ⏎ ");
				console.error(
					`[jlc:debug-existing-prompt] lines=${lines} chars=${existingPrompt.length} head="${head}" tail="${tail}"`,
				);
			}
		}
		startTurnMeter(ctx);
		lastProviderStartedAtMs = Date.now();
		providerCallCountThisTurn = 0;
		resetProviderCallCeilingState();
		lastTurnPromptSnapshot = {
			systemPrompt,
			messages,
			userText,
			promptContextText: buildPromptContextText(lastContextResponse?.context),
			modePromptText: `${modePrompt}\n\n${routePrompt}`,
			overlayText: overlay,
			existingPromptText: existingPrompt,
		};

		return { systemPrompt };
	});

	pi.on("tool_call", async (event, ctx) => {
		const toolName = String(event.toolName ?? "").toLowerCase();
		const lockedResourceBlock = maybeBlockLockedResourceToolCall(toolName, event.input);
		if (lockedResourceBlock) return lockedResourceBlock;
		const pcCeilingBlock = maybeBlockProviderCallCeilingToolCall(toolName, event.input);
		if (pcCeilingBlock) return pcCeilingBlock;
		if (toolName === "bash") {
			return maybeConfirmRiskyBashToolCall(event.input, ctx, safetyConfirmedKeys);
		}
		const isRead = READ_TOOL_NAMES.has(toolName);
		const isCode = CODE_TOOL_NAMES.has(toolName);
		if (!isRead && !isCode) return undefined;

		const rawPath = extractToolPath(event.input);
		if (!rawPath) return undefined;
		if (isProjectRoute(currentRoute)) {
			const shortPath = rawPath.length > 80 ? `...${rawPath.slice(-77)}` : rawPath;
			notifyWork(ctx, `JLC: running ${toolName} ${shortPath}`);
		}
		const absPath = resolveToolPath(rawPath, ctx);
		if (!absPath) return undefined;

		const project = await resolveRegisteredProjectForPath(absPath);
		if (!project) {
			if (isCode) {
				const safetyBlock = await maybeConfirmExternalMutationToolCall(toolName, absPath, ctx, safetyConfirmedKeys);
				if (safetyBlock) return safetyBlock;
			}
			if (!isProjectRoute(currentRoute)) {
				enterUnregisteredCoding();
				try {
					ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, undefined);
					ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down"));
				} catch {
					/* stale */
				}
			}
			return undefined;
		}

		if (!sameProjectPath(activeProjectPath, project.path)) {
			const data = await postSidecar<SidecarSwitchResponse>("/switch_project", {
				slug_or_name: project.slug || project.name,
				code_path: project.code_path,
				auto_create: false,
			});
			if (!data?.ok || !data.path) return undefined;
			patchProjectCache(data);
			pendingProjectSwitchContextRefresh = shouldRefreshContextForSelectedProject(data.path);
		}
		activeProjectPath = project.path;
		activeCodePath = project.code_path;
		activeProjectId = project.project_id;
		enterProjectWorkPreservingHeavy();
		initializeSubturnLog(project.path, lastUserMessage, lastObservedUserTurnKey, ctx.cwd, "deepdive");
		try {
			applyRouteThinkingLevel(currentRoute, ctx, pi);
			ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down", project.name));
		} catch {
			/* stale */
		}

		if (DEBUG_CONTEXT) {
			console.error(
				`[jlc:p1] toolName=${toolName} absPath=${absPath} cache.size=${projectCache.length} active=${activeProjectPath ?? ""}`,
			);
		}
		return undefined;
	});

	pi.on("before_provider_request", async (event) => {
		// trim-before/after measurement for chat_in trace (cumulative across provider calls in this turn)
		if (providerCallCountThisTurn === 0) {
			trimBeforeTokensSum = 0;
			trimAfterTokensSum = 0;
			lastToolSchemaTokens = 0;
		}
		const beforeMetrics = extractPayloadTokens(event.payload);
		trimBeforeTokensSum += beforeMetrics.message_tokens;
		lastToolSchemaTokens = beforeMetrics.tool_schema_tokens;
		const beforeTrim = DEBUG_CONTEXT ? summarizeProviderPayload(event.payload) : "";
		const payloadWithContext = ensureContextInProviderPayload(event.payload, lastContextResponse?.context);
		const legacyPayload = trimPayloadToCurrentJarvisTurn(payloadWithContext, { stateCarry: false });
		const nextPayload = trimPayloadToCurrentJarvisTurn(payloadWithContext);
		const legacyMetrics = extractPayloadTokens(legacyPayload);
		const uncompressedAfterMetrics = extractPayloadTokens(nextPayload);
		const nextProviderCall = providerCallCountThisTurn + 1;
		// Evidence stores fired at tool_execution_end may still be in flight
		// (pi does not await that handler before the next provider call).
		await awaitJarvisPendingEvidenceStores();
		const compressionOutcome = jarvisCompressProviderPayload(nextPayload);
		turnCompressedOutputsTotal += compressionOutcome.compressed_tool_outputs;
		turnCompressionSavedTotal += compressionOutcome.compression_saved_tokens_est;
		const reportStopPayload = applyLockedResourceReportStop(compressionOutcome.payload);
		const providerPayload = filterChatRouteOnlyTools(applyProviderCallCeiling(reportStopPayload, nextProviderCall));
		const afterMetrics = extractPayloadTokens(providerPayload);
		const prefixProbe = measureJarvisPrefixProbe(providerPayload);
		providerCallCountThisTurn = nextProviderCall;
		trimAfterTokensSum += afterMetrics.message_tokens;
		displayedCallInputTokens = afterMetrics.message_tokens;
		displayedInputTokensThisTurn = trimAfterTokensSum;
		const legacySummary = summarizeProviderPayload(legacyPayload);
		const afterSummary = summarizeProviderPayload(providerPayload);
		logSubturnTokenUsage(
			`provider_call=${providerCallCountThisTurn}`,
			afterSummary,
			providerPayloadTracePreview(providerPayload),
			undefined,
			undefined,
			prefixProbe,
			compressionOutcome,
		);
		logSubturnObserveCandidate(
			providerCallCountThisTurn,
			beforeMetrics,
			legacyMetrics,
			afterMetrics,
			legacySummary,
			providerPayload,
		);
		appendSubturnEvent("provider_request", {
			provider_call: providerCallCountThisTurn,
			before_message_tokens: beforeMetrics.message_tokens,
			legacy_message_tokens: legacyMetrics.message_tokens,
			after_message_tokens: afterMetrics.message_tokens,
			state_carry_trimmed_message_tokens: Math.max(
				0,
				legacyMetrics.message_tokens - uncompressedAfterMetrics.message_tokens,
			),
			trimmed_message_tokens: Math.max(0, beforeMetrics.message_tokens - legacyMetrics.message_tokens),
			tool_schema_tokens: afterMetrics.tool_schema_tokens,
			stable_prefix_hash: prefixProbe.stable_prefix_hash,
			stable_prefix_tokens_est: prefixProbe.stable_prefix_tokens_est,
			live_tokens_est: prefixProbe.live_tokens_est,
			compressed_tool_outputs: compressionOutcome.compressed_tool_outputs,
			compression_saved_tokens_est: compressionOutcome.compression_saved_tokens_est,
			compression_skips: compressionOutcome.compression_skips,
			payload_summary: afterSummary,
		});
		if (!DEBUG_CONTEXT) return providerPayload;
		if (beforeTrim) {
			console.error(`[jlc:debug-trim-before] call=${providerCallCountThisTurn} mode=${currentMode} ${beforeTrim}`);
		}
		console.error(
			`[jlc:debug-provider] call=${providerCallCountThisTurn} ${summarizeProviderPayload(providerPayload)}`,
		);
		return providerPayload;
	});

	pi.on("turn_end", async (event, ctx) => {
		if (DEBUG_CONTEXT && event.toolResults.length > 0) {
			console.error(
				`[jlc:debug-tools] turn_end tools=${event.toolResults.map((result) => result.toolName).join(",")}`,
			);
		}
		if (isProjectRoute(currentRoute) && event.toolResults.length > 0) {
			const names = event.toolResults
				.map((result) => result.toolName)
				.filter(Boolean)
				.join(", ");
			notifyWork(ctx, `JLC: tools completed${names ? ` (${names})` : ""}`);
		}
		toolEvents.push({
			turnIndex: event.turnIndex,
			toolResults: event.toolResults.map((result) => ({
				toolName: result.toolName,
				isError: result.isError,
				text: contentToText(result.content).slice(0, 2000),
			})),
		});
		refreshSubturnRollingSummary();
		appendSubturnEvent("turn_end", {
			turn_index: event.turnIndex,
			tool_results: event.toolResults.map((result) => ({
				tool: result.toolName,
				status: result.isError ? "error" : "ok",
				output: summarizeSubturnText(contentToLogText(result.content)),
			})),
		});
	});

	pi.on("tool_execution_start", async (event, ctx) => {
		const args = truncateForCheckpoint(JSON.stringify(event.args ?? {}), 1200);
		const descriptor = summarizeToolDescriptor(event.toolName, event.args);
		const carryKind = carryKindForTool(event.toolName, descriptor.text);
		const activeKey = `${carryKind}:${descriptor.key}`;
		const eventKey = subturnToolEventKey(event.toolCallId, event.toolName);
		const startMetadata = metadataForJarvisToolEvent(event as unknown as Record<string, unknown>, undefined);
		subturnActiveToolCarryKeys.set(eventKey, activeKey);
		subturnActiveToolDescriptors.set(eventKey, descriptor.text);
		subturnActiveToolMetadata.set(eventKey, startMetadata);
		if (isJarvisReadCompressionEditTargetTool(event.toolName) && startMetadata.sourcePath) {
			const pathKey = jarvisReadSourcePathKey(startMetadata.sourcePath, startMetadata.cwd, ctx);
			if (pathKey) turnReadCompressionEditTargetPaths.add(pathKey);
		}
		checkpointToolEvents.push({
			toolResults: [
				{
					toolName: event.toolName,
					isError: false,
					text: `started args=${truncateForCheckpoint(JSON.stringify(event.args ?? {}), 600)}`,
				},
			],
		});
		appendSubturnSummaryLines([
			formatSubturnSummaryLine("tool_start", `${event.toolName ?? "tool"} args=${oneLineForSummary(args, 220)}`),
		]);
		upsertSubturnCarry(activeKey, `${carryKind}: ${event.toolName ?? "tool"} pending ${descriptor.text}`);
		appendSubturnEvent("tool_start", {
			tool: event.toolName,
			args,
		});
	});

	pi.on("tool_execution_update", async (event) => {
		checkpointToolEvents.push({
			toolResults: [
				{
					toolName: event.toolName,
					isError: false,
					text: `update ${truncateForCheckpoint(contentToText(event.partialResult?.content), 800)}`,
				},
			],
		});
		appendSubturnEvent("tool_update", {
			tool: event.toolName,
			output: summarizeSubturnText(contentToLogText(event.partialResult?.content), 800, 800),
		});
	});

	pi.on("tool_execution_end", async (event, ctx) => {
		const output = contentToLogText(event.result?.content);
		const tool = String(event.toolName ?? "tool");
		const eventKey = subturnToolEventKey(event.toolCallId, event.toolName);
		const activeKey = subturnActiveToolCarryKeys.get(eventKey) ?? `tool:last:${tool}`;
		const descriptor = subturnActiveToolDescriptors.get(eventKey) ?? "";
		const carryKind = carryKindForTool(event.toolName, descriptor);
		const status = event.isError ? "error" : "ok";
		checkpointToolEvents.push({
			toolResults: [
				{
					toolName: event.toolName,
					isError: event.isError,
					text: truncateForCheckpoint(contentToText(event.result?.content), 1200),
				},
			],
		});
		appendSubturnSummaryLines([
			formatSubturnSummaryLine(
				"tool_end",
				`${event.toolName ?? "tool"} ${event.isError ? "error" : "ok"} ${oneLineForSummary(output, 260)}`,
			),
		]);
		appendSubturnCommit(
			carryKind,
			`${tool} ${status}${descriptor ? ` ${descriptor}` : ""} => ${oneLineForSummary(output, 180)}`,
		);
		upsertSubturnCarry(
			activeKey,
			`${carryKind}: ${tool} ${status}${descriptor ? ` ${descriptor}` : ""} => ${oneLineForSummary(output, 260)}`,
		);
		if (!event.isError && shouldKeepSubturnEvidence(event.toolName, output)) {
			upsertSubturnEvidence(activeKey, `${tool}${descriptor ? ` ${descriptor}` : ""}`, output);
		}
		const metadata = mergeJarvisToolMetadata(
			subturnActiveToolMetadata.get(eventKey),
			metadataForJarvisToolEvent(event as unknown as Record<string, unknown>, event.isError ? 1 : 0),
		);
		const lockedResourceRecord = recordLockedResourceToolOutcome({
			toolCallId: event.toolCallId,
			toolName: event.toolName,
			input: {
				command: metadata.command,
				path: metadata.sourcePath,
				descriptor,
			},
			isError: event.isError,
			outputText: output,
		});
		recordJarvisTurnToolOutcome(event.toolName, event.isError, output, metadata, ctx);
		const evidenceKey = jarvisEvidenceToolResultKey(event.toolCallId, tool);
		const isReadTool = String(event.toolName ?? "").toLowerCase() === "read";
		const readPathKey =
			isReadTool && metadata.sourcePath
				? jarvisReadSourcePathKey(metadata.sourcePath, metadata.cwd, ctx)
				: undefined;
		const readSourcePathKeys =
			isReadTool && metadata.sourcePaths?.length
				? metadata.sourcePaths
						.map((sourcePath) => jarvisReadSourcePathKey(sourcePath, metadata.cwd, ctx))
						.filter((key): key is string => !!key)
				: undefined;
		if (!event.isError && readPathKey) recordJarvisReadCompressionKey(readPathKey, evidenceKey);
		const kind = !event.isError
			? detectJarvisToolOutputKind(
					event.toolName,
					metadata.command,
					output,
					metadata.sourcePath,
					metadata.sourcePaths,
				)
			: undefined;
		if (
			!event.isError &&
			isReadTool &&
			(metadata.sourcePath || metadata.sourcePaths?.length) &&
			!jarvisReadOutputMeetsCompressionThreshold(output)
		) {
			jarvisEvidenceByToolResultKey.set(evidenceKey, {
				kind: "read_skeleton",
				toolName: tool,
				cwd: metadata.cwd,
				sourcePath: metadata.sourcePath,
				sourcePathKey: readPathKey,
				sourcePaths: metadata.sourcePaths,
				sourcePathKeys: readSourcePathKeys,
				originalRef: sha256Hex24(output),
				storeSkippedReason: "read_below_threshold",
			});
		}
		if (kind && jarvisToolOutputMeetsCompressionGuard(kind, output)) {
			const storePromise = (async () => {
				const ref = await storeJarvisEvidence({
					session_id: subturnLogInitializedForUserTurnKey ?? "default",
					turn_key: subturnLogInitializedForUserTurnKey,
					tool_call_id: typeof event.toolCallId === "string" ? event.toolCallId : undefined,
					tool_name: tool,
					kind,
					metadata: {
						cwd: metadata.cwd,
						command: metadata.command,
						exit_code: metadata.exitCode,
						source_path: metadata.sourcePath,
						source_paths: metadata.sourcePaths,
					},
					original_text: output,
				});
				if (ref?.ref) {
					jarvisEvidenceByToolResultKey.set(evidenceKey, {
						ref: ref.ref,
						kind,
						toolName: tool,
						command: metadata.command,
						cwd: metadata.cwd,
						exitCode: metadata.exitCode,
						sourcePath: metadata.sourcePath,
						sourcePathKey: readPathKey,
						sourcePaths: metadata.sourcePaths,
						sourcePathKeys: readSourcePathKeys,
						originalRef: sha256Hex24(output),
					});
				}
			})();
			jarvisPendingEvidenceStores.add(storePromise);
			storePromise.finally(() => jarvisPendingEvidenceStores.delete(storePromise));
			await storePromise;
		}
		subturnActiveToolCarryKeys.delete(eventKey);
		subturnActiveToolDescriptors.delete(eventKey);
		subturnActiveToolMetadata.delete(eventKey);
		appendSubturnEvent("tool_end", {
			tool: event.toolName,
			status: event.isError ? "error" : "ok",
			output: summarizeSubturnText(output),
			...(lockedResourceRecord
				? {
						locked_resource_key: lockedResourceRecord.key,
						locked_resource_attempts: lockedResourceRecord.count,
						report_stop_active: subturnLockedResourceReportStopActive,
					}
				: {}),
		});
	});

	pi.on("message_update", async (event, ctx) => {
		if (event.message.role !== "assistant") return;
		const text = contentToText((event.message as { content: unknown }).content);
		const updateMarker = leadingModeMarker(text) ?? partialLeadingModeMarker(text);
		if (updateMarker) {
			lastAssistantObservedModeMarker = updateMarker;
		}
		if (updateMarker === "unregistered_coding") {
			activateRouteFromAssistantSignal(ctx, pi, "unregistered_coding");
		} else if (updateMarker === "heavy_deepdive") {
			activateRouteFromAssistantSignal(ctx, pi, "heavy_deepdive");
		} else if (updateMarker === "deepdive") {
			activateRouteFromAssistantSignal(ctx, pi, "deepdive");
		}
		lastAssistantPartialText = sanitizeAssistantText(text);
		if (DEBUG_CONTEXT && /\[MODE:[^\]]+\]/i.test(text)) {
			console.error(`[jlc:mode-marker] observed update current=${currentMode}`);
		}
		sanitizeAssistantMessageInPlace(event.message as AssistantMessage, true);
		updateTurnMeterFromText(ctx, contentToText((event.message as { content: unknown }).content));
		if (lastAssistantPartialText.length - lastSubturnAssistantUpdateLength >= SUBTURN_ASSISTANT_SAMPLE_CHARS) {
			lastSubturnAssistantUpdateLength = lastAssistantPartialText.length;
			appendSubturnEvent("assistant_update", {
				text: truncateForCheckpoint(lastAssistantPartialText, SUBTURN_ASSISTANT_SAMPLE_CHARS),
			});
		}
	});

	pi.on("message_end", async (event, ctx) => {
		if (event.message.role !== "assistant") return undefined;
		const assistantMessage = event.message as AssistantMessage;
		recordCompletedTurnMeterUsage(assistantMessage);
		logSubturnTokenUsage(
			"assistant_end",
			"",
			"",
			assistantMessage.usage as TurnUsageSummary | undefined,
			buildJarvisCacheReport(assistantMessage.usage as TurnUsageSummary | undefined, {
				provider: assistantMessage.provider ?? ctx.model?.provider,
				api: assistantMessage.api,
				model: assistantMessage.responseModel ?? assistantMessage.model ?? ctx.model?.id,
			}),
			undefined,
			{
				payload: undefined,
				compressed_tool_outputs: turnCompressedOutputsTotal,
				compression_saved_tokens_est: turnCompressionSavedTotal,
			},
		);
		const rawText = contentToText((event.message as { content: unknown }).content);
		const rawMarker = leadingModeMarker(rawText || lastAssistantPartialText) ?? lastAssistantObservedModeMarker;
		const toolNames = assistantToolNames(assistantMessage);
		const markerRoute =
			rawMarker === "unregistered_coding" || rawMarker === "heavy_deepdive" || rawMarker === "deepdive"
				? rawMarker
				: undefined;
		if (markerRoute || toolNames.some((name) => ASSISTANT_PROJECT_ROUTE_TOOL_NAMES.has(name))) {
			activateRouteFromAssistantSignal(ctx, pi, markerRoute);
		}
		const assistantText = sanitizeAssistantText(rawText || lastAssistantPartialText);
		if (DEBUG_CONTEXT) {
			const hasMarker = /\[MODE:[^\]]+\]/i.test(rawText);
			const head = rawText.slice(0, 80).replace(/\r?\n/g, "\\n");
			console.error(`[jlc:mode-marker] observed=${hasMarker ? "yes" : "no"} current=${currentMode} head="${head}"`);
		}
		if (assistantText.trim()) {
			upsertSubturnCarry("assistant:last", `assistant: ${oneLineForSummary(assistantText, 260)}`);
			appendSubturnCommit("assistant", oneLineForSummary(assistantText, 180));
		}
		appendSubturnSummaryLines([formatSubturnSummaryLine("assistant", oneLineForSummary(assistantText, 260))]);
		refreshSubturnRollingSummary();
		appendSubturnEvent("assistant_end", {
			stop_reason: assistantMessage.stopReason ?? "",
			tools: toolNames,
			text: truncateForCheckpoint(assistantText, SUBTURN_ASSISTANT_SAMPLE_CHARS),
		});
		lastSubturnAssistantUpdateLength = 0;
		if (assistantMessage.stopReason === "aborted") {
			await saveInterruptCheckpoint(ctx, pi, assistantText);
		}
		const message = sanitizeAssistantMessage(assistantMessage, false);
		return message === event.message ? undefined : { message };
	});

	pi.on("agent_end", async (event, ctx) => {
		if (autoPromptState && autoPromptWatchdog?.aborting) {
			autoPromptWatchdog.aborting = false;
			clearAutoPromptWatchdog();
			clearTurnMeter(true);
			toolEvents = [];
			lastTurnPromptSnapshot = undefined;
			providerCallCountThisTurn = 0;
			resetProviderCallCeilingState();
			turnCheckpointScope = undefined;
			resetJarvisTurnChoreographyState();
			resetToChatMode(ctx, pi);
			return;
		}

		const assistantMessage = latestAssistantMessage(event.messages);
		const turnUsage = summarizeAssistantUsage(event.messages);
		finishTurnMeter(ctx, assistantMessage);
		const assistantTextRaw = assistantMessage ? contentToText(assistantMessage.content) : "";
		const assistantText = sanitizeAssistantText(assistantTextRaw);
		if (isRetryableAssistantError(assistantMessage)) {
			const errorText = assistantMessage?.errorMessage || assistantText || "retryable provider error";
			appendSubturnEvent("provider_retryable_error", {
				error: truncateForCheckpoint(errorText, SUBTURN_ASSISTANT_SAMPLE_CHARS),
				current_route: currentRoute,
			});
			try {
				ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down", lastContextResponse?.project_name));
				if (isProjectRoute(currentRoute)) {
					setWorkStatus(ctx, `JLC: ${routeStatusLabel(currentRoute)} kept after provider retryable error.`);
				}
			} catch {
				/* ctx may be stale */
			}
			return;
		}
		const assistantPendingProjectCreate = parseAssistantProjectCreationPrompt(assistantText);
		if (assistantPendingProjectCreate) {
			pendingProjectCreate = assistantPendingProjectCreate;
		}
		if (DEBUG_CONTEXT && turnUsage) {
			console.error(
				`[jlc:debug-usage] input=${turnUsage.input ?? 0} output=${turnUsage.output ?? 0} total=${turnUsage.totalTokens ?? 0} cache_read=${turnUsage.cacheRead ?? 0} cache_write=${turnUsage.cacheWrite ?? 0}`,
			);
		}
		// O(N) regression trace — appends per-turn chat_in + breakdown to jsonl.
		// Read C:\Users\TopIt\.jarvis-code\chat_in_trace.jsonl after 5+ turns
		// to see whether chat_in grows linearly and which sub-field carries it.
		try {
			const snap = lastTurnPromptSnapshot;
			const traceLine = {
				ts: new Date().toISOString(),
				user: lastUserMessage.slice(0, 60),
				chat_in: turnUsage?.input ?? 0,
				chat_out: turnUsage?.output ?? 0,
				provider_calls: providerCallCountThisTurn,
				system_prompt_tokens: snap ? estimateTextTokenCount(snap.systemPrompt) : 0,
				mode_prompt_tokens: snap ? estimateTextTokenCount(snap.modePromptText) : 0,
				existing_prompt_tokens: snap ? estimateTextTokenCount(snap.existingPromptText) : 0,
				prompt_ctx_tokens: snap ? estimateTextTokenCount(snap.promptContextText) : 0,
				user_text_tokens: snap ? estimateTextTokenCount(snap.userText) : 0,
				jhb_tokens: lastContextResponse?.jhb_tokens ?? 0,
				context_tokens: lastContextResponse?.context_tokens ?? 0,
				message_count: event.messages.length,
				history_assistant_count: event.messages.filter((m) => m.role === "assistant").length,
				history_user_count: event.messages.filter((m) => m.role === "user").length,
				has_tool_in_history: event.messages.some((m) => {
					const r = String((m as { role?: unknown }).role ?? "");
					return r === "tool" || r === "toolResult";
				}),
				current_mode: currentMode,
				current_route: currentRoute,
				trim_before_tokens: trimBeforeTokensSum,
				trim_after_tokens: trimAfterTokensSum,
				tool_schema_tokens: lastToolSchemaTokens,
				cache_read_tokens: turnUsage?.cacheRead ?? 0,
				cache_write_tokens: turnUsage?.cacheWrite ?? 0,
			};
			const traceDir = path.join(process.env.USERPROFILE ?? "", ".jarvis-code");
			fs.mkdirSync(traceDir, { recursive: true });
			fs.appendFileSync(path.join(traceDir, "chat_in_trace.jsonl"), `${JSON.stringify(traceLine)}\n`, "utf8");
		} catch {
			// ignore trace failures
		}
		try {
			ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down"));
		} catch {
			/* ctx may be stale */
		}

		if (!lastUserMessage.trim() || !assistantText.trim()) {
			completeSubturnLog("idle", assistantText);
			toolEvents = [];
			turnCheckpointScope = undefined;
			resetJarvisTurnChoreographyState();
			resetToChatMode(ctx, pi);
			return;
		}

		await patchJarvisMemoryFromObservedWork(pi);
		await registerWorkspaceFolderFromObservedWrite(ctx, pi);
		await unregisterMissingWorkspaceProjectsBackstop(ctx, pi);

		let projectPath = isProjectRoute(currentRoute) ? activeProjectPath : undefined;
		if (isProjectRoute(currentRoute) && !projectPath) {
			try {
				projectPath = await resolveActiveProjectPath(ctx, pi);
			} catch {
				/* stale */
			}
		}
		const response = await postSidecar<SidecarTurnResponse>("/turn", {
			project_path: projectPath,
			user_message: lastUserMessage,
			assistant_message: assistantText,
			tool_events: toolEvents,
			llm_meta: buildTurnLlmMeta(
				assistantMessage,
				ctx,
				lastContextResponse,
				lastTurnPromptSnapshot,
				toolEvents,
				turnUsage,
			),
			bench_conv_id: benchConvId(pi),
		});
		recordFooterMeterEntry(pi, assistantMessage, lastContextResponse, turnUsage);
		if (response?.warning) {
			sendJarvisChatNotice(pi, `JLC 인코더 시작 오류: ${response.warning}`);
		}
		let footerResetDeferred = false;
		if (response?.encoder_summary) {
			const s = response.encoder_summary;
			renderEncBadge(ctx, {
				enc_out: s.enc_out,
				enc_seconds: s.enc_seconds,
				jhb_tokens: s.jhb_tokens,
				jhb_delta: s.jhb_delta,
			});
		} else if (response?.scheduled_encode) {
			startEncodingStatus(ctx);
			if (autoPromptState) {
				const encodingConvId = autoPromptEncodingConvId(pi);
				if (encodingConvId) {
					const encodingReady = await waitForEncodingStatus(ctx, pi, encodingConvId, response.scheduled_turn);
					if (!encodingReady) {
						stopEncodingStatus();
						console.error(
							`[jarvis:auto-prompts] encoding did not complete within ${AUTO_PROMPT_ENCODING_WAIT_MS}ms; continuing`,
						);
					}
				}
			} else {
				// Encoder runs async — poll /encoding_status until ready so the pink
				// footer badge + chat notify fire after deferred encode completes.
				footerResetDeferred = true;
				void pollEncodingStatus(ctx, pi, benchConvId(pi) ?? "", response.scheduled_turn).finally(() =>
					resetToChatModeAfterEncoding(ctx, pi),
				);
			}
		}
		completeSubturnLog("completed", assistantText);
		if (!interruptCheckpointSavedThisTurn) {
			try {
				await clearInterruptCheckpointForScope(turnCheckpointScope);
			} catch {
				/* best-effort cleanup */
			}
		}
		toolEvents = [];
		checkpointToolEvents = [];
		lastAssistantPartialText = "";
		lastAssistantObservedModeMarker = undefined;
		interruptCheckpointSavedThisTurn = false;
		turnCheckpointScope = undefined;
		clearInterruptInputCheckpointHook();
		lastTurnPromptSnapshot = undefined;
		providerCallCountThisTurn = 0;
		resetProviderCallCeilingState();
		resetJarvisTurnChoreographyState();

		// Project work is a one-turn transaction. The next user turn starts as
		// chat unless it explicitly enters deepdive or resolves to a project.
		resetToChatMode(ctx, pi, { updateFooter: !footerResetDeferred });

		if (autoPromptState) {
			await handleAutoPromptTurn(autoPromptState, ctx, pi);
		}
	});

	pi.registerTool({
		name: "recall_turns",
		label: "Recall Turns",
		description: "Search raw turns by turn/date/keywords when /context lacks exact prior wording.",
		parameters: Type.Object({
			queries: Type.Array(Type.String()),
			top_k: Type.Optional(Type.Number()),
		}),
		async execute(_toolCallId, params) {
			const data = await postSidecar("/recall", {
				queries: params.queries,
				top_k: params.top_k ?? 5,
				bench_conv_id: benchConvId(pi),
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			const text = JSON.stringify(details ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},

		renderCall(args, theme, _context) {
			const queries = args.queries as string[];
			const summary = queries.length === 1 ? `"${queries[0]}"` : `${queries.length} queries`;
			return new Text(theme.fg("toolTitle", theme.bold("recall ")) + theme.fg("accent", summary), 0, 0);
		},

		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) {
				return new Text(theme.fg("warning", "Searching..."), 0, 0);
			}

			type RecallFragment = {
				turn: number;
				score: number;
				user: string;
				assistant: string;
				ts: string;
			};
			type RecallResultItem = {
				query?: string;
				text?: string;
				fragments?: RecallFragment[];
				confidence?: "HIGH" | "MID" | "LOW";
				source?: string;
			};
			const data = result.details as {
				ok?: boolean;
				results?: RecallResultItem[];
				warnings?: string[];
			};

			if (!data?.ok) {
				return new Text(theme.fg("error", "Sidecar unavailable"), 0, 0);
			}

			const resultCount = data.results?.length ?? 0;
			if (!expanded) {
				let text = theme.fg("success", `${resultCount} result${resultCount !== 1 ? "s" : ""}`);
				if (data.warnings?.length) {
					text += theme.fg(
						"warning",
						` (${data.warnings.length} warning${data.warnings.length !== 1 ? "s" : ""})`,
					);
				}
				return new Text(text, 0, 0);
			}

			const lines: string[] = [];
			for (const resultItem of data.results ?? []) {
				const item = resultItem as RecallResultItem;
				if (item.query) {
					lines.push(theme.fg("accent", `Query: ${item.query}`));
				}
				if (item.text) {
					lines.push(theme.fg("dim", item.text));
				}
				lines.push("");
			}
			if (data.warnings?.length) {
				lines.push(
					theme.fg("warning", `--- ${data.warnings.length} warning${data.warnings.length !== 1 ? "s" : ""} ---`),
				);
				for (const warning of data.warnings) {
					const truncated = warning.length > 200 ? `${warning.slice(0, 197)}...` : warning;
					lines.push(theme.fg("muted", truncated));
				}
			}

			return new Text(lines.join("\n"), 0, 0);
		},
	});

	pi.registerTool({
		name: "retrieve_output",
		label: "Retrieve Output",
		description: "Retrieve ref; optional 1-based line range.",
		parameters: Type.Object({
			ref: Type.String({ description: "ref" }),
			start_line: Type.Optional(Type.Number({ description: "start" })),
			end_line: Type.Optional(Type.Number({ description: "end" })),
		}),
		async execute(_toolCallId, params) {
			const ref = String(params.ref ?? "").trim();
			const query = new URLSearchParams();
			if (typeof params.start_line === "number" && Number.isFinite(params.start_line)) {
				query.set("start_line", String(Math.max(1, Math.floor(params.start_line))));
			}
			if (typeof params.end_line === "number" && Number.isFinite(params.end_line)) {
				query.set("end_line", String(Math.max(1, Math.floor(params.end_line))));
			}
			const suffix = query.toString() ? `?${query.toString()}` : "";
			const data = await postSidecar<JarvisEvidenceRetrieveResponse>(
				`/evidence/${encodeURIComponent(ref)}${suffix}`,
				undefined,
				"GET",
			);
			if (!data?.ok || typeof data.content !== "string") {
				const text = "ref not found — 원문이 이미 사용 가능하거나 만료됨. 파일 경로를 알면 read를 써라.";
				return {
					content: [{ type: "text", text }],
					details: {
						ok: false,
						ref,
						truncated: false,
						lines_returned: 0,
						bytes_returned: 0,
						metadata: undefined,
					},
				};
			}
			const limited = limitRetrieveOutputContent(data.content);
			return {
				content: [{ type: "text", text: limited.text }],
				details: {
					ok: true,
					ref,
					truncated: limited.truncated,
					lines_returned: limited.linesReturned,
					bytes_returned: Buffer.byteLength(limited.text, "utf8"),
					metadata: data.metadata,
				},
			};
		},
	});

	pi.registerTool({
		name: "update_jarvis_md",
		label: "Update JARVIS.md sections",
		description:
			"Patch active JARVIS.md sections; batch 2+ updates; after code changes refresh NOW/MAP/RAW; never write/edit whole JARVIS.md.",
		parameters: Type.Object({
			field: Type.Optional(Type.String({ description: "NOW|MAP|LAW|BAN|HABIT|WHY|OMM|RAW" })),
			value: Type.Optional(Type.String()),
			updates: Type.Optional(
				Type.Array(
					Type.Object({
						field: Type.String({ description: "NOW|MAP|LAW|BAN|HABIT|WHY|OMM|RAW" }),
						value: Type.String(),
					}),
				),
			),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			turnJarvisMdUpdated = true;
			if (!isProjectRoute(currentRoute)) {
				const text = JSON.stringify(
					{
						ok: false,
						error: "update_jarvis_md requires a registered project route (deepdive or heavy_deepdive). Chat/unregistered coding turns must not mutate JARVIS.md.",
						current_mode: currentMode,
						current_route: currentRoute,
					},
					null,
					2,
				);
				return { content: [{ type: "text", text }], details: { ok: false, gated: currentRoute } };
			}
			// Resolve the project defensively inside registered project routes.
			// Multiple fallbacks because activeProjectPath has been observed to
			// disappear mid-turn (live snake-game test: register succeeded, write
			// succeeded, update_jarvis_md fired with activeProjectPath undefined).
			// lastContextResponse is set once per user turn and not cleared by
			// the same code paths that touch activeProjectPath, so it survives.
			let projectPath: string | undefined =
				activeProjectPath ??
				lastContextResponse?.active_project_path ??
				lastContextResponse?.code_path ??
				undefined;
			if (!projectPath) {
				projectPath = await resolveActiveProjectPath(ctx, pi);
			}
			if (!projectPath) {
				// Last resort: ask sidecar via /context with the registered-project
				// hint cleared so the router falls back to its persisted active.
				try {
					const response = await postSidecar<SidecarContextResponse>("/context", {
						cwd_hint: ctx.cwd,
						mode: modeForRoute(currentRoute),
						user_message: "",
						active_project_path: isProjectRoute(currentRoute) ? currentActiveProjectHint() : undefined,
						hints: { cwd: ctx.cwd },
					});
					projectPath = response?.active_project_path ?? response?.code_path ?? undefined;
					if (response) {
						applyContextProjectState(response, modeForRoute(currentRoute));
						lastContextResponse = response;
					}
				} catch {
					/* swallow */
				}
			}
			if (DEBUG_CONTEXT) {
				console.error(
					`[jlc:update_jarvis_md] resolved=${projectPath ?? "<none>"} active=${activeProjectPath ?? ""} lastCtx=${lastContextResponse?.active_project_path ?? ""} mode=${currentMode} route=${currentRoute}`,
				);
			}
			if (!projectPath) {
				const text = JSON.stringify(
					{
						ok: false,
						error: "update_jarvis_md needs an active project. Call register_project (for a new project) or switch_project (for an existing one) first.",
						current_mode: currentMode,
						current_route: currentRoute,
						debug: {
							active_project_path: activeProjectPath ?? null,
							last_context_active_path: lastContextResponse?.active_project_path ?? null,
							last_context_code_path: lastContextResponse?.code_path ?? null,
						},
					},
					null,
					2,
				);
				return { content: [{ type: "text", text }], details: { ok: false, gated: "no_active_project" } };
			}
			const updates = Array.isArray((params as { updates?: unknown }).updates)
				? (params as { updates?: Array<{ field: string; value: string }> }).updates
				: undefined;
			const data = await postSidecar("/update_jarvis_md", {
				project_path: projectPath,
				...(updates ? { updates } : { field: params.field, value: params.value }),
			});
			const text = JSON.stringify(data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details: data };
		},
	});

	pi.registerTool({
		name: "register_project",
		label: "Register Project",
		description: "Register explicit/confirmed project path in JARVIS memory; not for edit-only external folders.",
		parameters: Type.Object({
			path: Type.String(),
			name: Type.Optional(Type.String()),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const allowRegistration =
				isProjectRoute(currentRoute) ||
				hasExplicitProjectRegistrationIntent(lastUserMessage) ||
				(isAffirmative(lastUserMessage) && pendingProjectCreate !== undefined);
			if (!allowRegistration) {
				const text = JSON.stringify(
					{
						ok: false,
						error: "register_project requires explicit JARVIS project registration, clear new-project creation, or confirmation of a registration question. For external file edits, continue as unregistered coding without registering.",
						current_route: currentRoute,
						last_user_message: lastUserMessage,
					},
					null,
					2,
				);
				return {
					content: [{ type: "text", text }],
					details: { ok: false, gated: "explicit_registration_required" },
				};
			}
			const data = await postSidecar<SidecarSwitchResponse>("/register_project", params);
			if (data?.ok && data.path) {
				const needsContextRefresh = shouldRefreshContextForSelectedProject(data.path);
				patchProjectCache(data);
				activeProjectPath = data.path;
				activeCodePath = data.code_path ?? activeCodePath;
				activeProjectId = data.project_id ?? activeProjectId;
				pendingProjectSwitchContextRefresh = needsContextRefresh;
				enterProjectWorkPreservingHeavy();
				try {
					ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down", data.name));
					applyRouteThinkingLevel(currentRoute, ctx, pi);
				} catch {
					/* pi stale */
				}
			}
			const text = JSON.stringify(data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			const guidance =
				data?.ok && data.path
					? `\n\n[P1] project registered.\n` +
						`code edit at <code_path>/...\n` +
						`memory update: edit sections inside <code_path>/JARVIS.md ` +
						`(NOW/MAP/RAW required when something changed; LAW/WHY/HABIT/BAN/OMM optional)`
					: "";
			return { content: [{ type: "text", text: text + guidance }], details: data };
		},
	});

	pi.registerTool({
		name: "unregister_project",
		label: "Unregister Project",
		description: "Remove registration; do not delete files.",
		promptSnippet:
			"unregister_project: remove registrations, including bulk cleanup/reset-all; prefer this over registry file edits.",
		parameters: Type.Object({
			project_id: Type.Optional(Type.String()),
			slug_or_name: Type.Optional(Type.String()),
			path: Type.Optional(Type.String()),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			// Explicit target (slug/path/id) must never be hijacked by the active-project
			// fallback: sidecar resolves project_id before slug_or_name, so injecting
			// activeProjectId here unregistered the active project instead of the asked one
			// (D6 root cause — 4 runs misdiagnosed as model imprecision).
			const hasExplicitTarget = Boolean(params.project_id || params.slug_or_name || params.path);
			const data = await postSidecar<SidecarUnregisterResponse>("/unregister_project", {
				project_id: params.project_id ?? (hasExplicitTarget ? undefined : activeProjectId),
				slug_or_name: params.slug_or_name,
				path: params.path,
			});
			if (data?.removed && data.project_id) {
				projectCache = projectCache.filter((project) => project.project_id !== data.project_id);
				const activeMatches =
					activeProjectId === data.project_id ||
					sameProjectPath(activeProjectPath, data.path) ||
					sameProjectPath(activeCodePath, data.code_path);
				if (activeMatches) {
					clearActiveProjectState();
					setEffectiveRoute("chat");
					try {
						ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down"));
					} catch {
						/* stale */
					}
				}
			}
			const text = `${formatUnregisterProjectStateLine(data)}\n${JSON.stringify(
				data ?? { ok: false, error: "JARVIS sidecar unavailable" },
				null,
				2,
			)}`;
			return { content: [{ type: "text", text }], details: data };
		},
		renderResult(result, { isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Unregistering..."), 0, 0);
			const data = result.details as SidecarUnregisterResponse | undefined;
			const line = formatUnregisterProjectStateLine(data);
			return new Text(data?.ok === false ? theme.fg("error", line) : theme.fg("success", line), 0, 0);
		},
	});

	pi.registerTool({
		name: "switch_project",
		label: "Switch Project",
		description: "Select active registered project when target is clear; ask if ambiguous.",
		parameters: Type.Object({
			slug_or_name: Type.String(),
			code_path: Type.Optional(Type.String()),
			auto_create: Type.Optional(Type.Boolean()),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const data = await postSidecar<SidecarSwitchResponse>("/switch_project", {
				slug_or_name: params.slug_or_name,
				code_path: params.code_path,
				auto_create: params.auto_create ?? false,
			});
			if (data?.ok && data.path) {
				const needsContextRefresh = shouldRefreshContextForSelectedProject(data.path);
				patchProjectCache(data);
				activeProjectPath = data.path;
				activeCodePath = data.code_path ?? activeCodePath;
				activeProjectId = data.project_id ?? activeProjectId;
				pendingProjectSwitchContextRefresh = needsContextRefresh;
				enterProjectWorkPreservingHeavy();
				try {
					applyRouteThinkingLevel(currentRoute, ctx, pi);
					ctx.ui.setStatus("jarvis", jlcLabel("ok", data.name));
				} catch {
					/* stale */
				}
			}
			const text = JSON.stringify(data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details: data };
		},
	});

	pi.registerTool({
		name: "web_search",
		label: "Web Search",
		description:
			"Search the public web through the JARVIS sidecar using Brave Search. Use for current, external, or web-only facts.",
		promptSnippet: "web_search: search Brave web results for current or external facts.",
		promptGuidelines: [
			"Use web_search for current, external, news, pricing, release, version, or web-only facts before answering.",
			"After finding a specific source URL that matters, call web_fetch to read the page before relying on it.",
			"Summarize relevant findings and include source URLs from the tool result.",
		],
		parameters: Type.Object({
			query: Type.String(),
			top_k: Type.Optional(Type.Number()),
		}),
		async execute(_toolCallId, params) {
			const data = await postSidecar("/web_search", {
				query: params.query,
				top_k: params.top_k ?? 5,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			const text = JSON.stringify(details ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
	});

	pi.registerTool({
		name: "web_fetch",
		label: "Web Fetch",
		description: "Fetch and extract readable text from a specific public URL.",
		promptSnippet: "web_fetch: read a specific public URL after search or when the user gives a link.",
		promptGuidelines: [
			"Use after web_search when an exact source matters, or when the user gives a URL.",
			"Do not use for localhost/private URLs.",
		],
		parameters: Type.Object({
			url: Type.String(),
			max_chars: Type.Optional(Type.Number()),
			timeout_sec: Type.Optional(Type.Number()),
		}),
		async execute(_toolCallId, params) {
			const data = await postSidecar("/web_fetch", {
				url: params.url,
				max_chars: params.max_chars ?? 12000,
				timeout_sec: params.timeout_sec ?? 10,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			const text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
	});

	pi.registerTool({
		name: "docs_search",
		label: "Docs Search",
		description: "Search official documentation domains and optionally fetch the top result.",
		promptSnippet: "docs_search: official docs search; optionally fetch top results.",
		promptGuidelines: [
			"Use for API/library/framework usage questions before relying on general web snippets.",
			"Pass domains when the official domain is known, e.g. react.dev, docs.python.org, playwright.dev.",
		],
		parameters: Type.Object({
			query: Type.String(),
			domains: Type.Optional(Type.Array(Type.String())),
			top_k: Type.Optional(Type.Number()),
			fetch_top: Type.Optional(Type.Number()),
			max_chars: Type.Optional(Type.Number()),
		}),
		async execute(_toolCallId, params) {
			const data = await postSidecar("/docs_search", {
				query: params.query,
				domains: params.domains ?? [],
				top_k: params.top_k ?? 5,
				fetch_top: params.fetch_top ?? 0,
				max_chars: params.max_chars ?? 4000,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			const text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
	});

	pi.registerTool({
		name: "package_info",
		label: "Package Info",
		description: "Look up structured npm, PyPI, or GitHub package/release metadata.",
		promptSnippet: "package_info: structured npm/PyPI/GitHub version and release metadata.",
		promptGuidelines: [
			"Use for latest package versions, release dates, package URLs, and GitHub latest release metadata.",
			"Prefer package_info over broad web_search when the user asks about a package version.",
		],
		parameters: Type.Object({
			ecosystem: Type.Union([Type.Literal("npm"), Type.Literal("pypi"), Type.Literal("github")]),
			package: Type.String(),
			include_release_notes: Type.Optional(Type.Boolean()),
		}),
		async execute(_toolCallId, params) {
			const data = await postSidecar("/package_info", {
				ecosystem: params.ecosystem,
				package: params.package,
				include_release_notes: params.include_release_notes ?? false,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			const text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
	});

	pi.registerCommand("jarvis-status", {
		description: "Show JARVIS sidecar and active project status.",
		handler: async (_args, ctx) => {
			const healthy = await checkHealth();
			sidecarHealthy = healthy;
			const status = healthy ? await postSidecar<SidecarStatusResponse>("/status", undefined, "GET") : undefined;
			const chat = formatRoleStatus(status?.roles?.chat);
			const encoder = formatRoleStatus(status?.roles?.encoder);
			const roleSummary = [chat ? `chat: ${chat}` : "", encoder ? `encoder: ${encoder}` : ""]
				.filter(Boolean)
				.join(" | ");
			ctx.ui.notify(
				`JARVIS sidecar: ${healthy ? "ok" : "down"}${activeProjectPath ? ` | memory: ${activeProjectPath}` : ""}${activeCodePath ? ` | code: ${activeCodePath}` : ""}${roleSummary ? ` | ${roleSummary}` : ""}`,
				healthy ? "info" : "warning",
			);
			ctx.ui.setStatus("jarvis", healthy ? jlcLabel("ok") : jlcLabel("down"));
		},
	});

	pi.registerCommand("jarvis-register", {
		description: "Register the current cwd or supplied path with JARVIS memory.",
		handler: async (args, ctx) => {
			const target = args.trim() || ctx.cwd;
			const data = await postSidecar<SidecarSwitchResponse & { error?: string }>("/register_project", {
				path: target,
			});
			if (data?.ok) {
				patchProjectCache(data);
				activeProjectPath = data.path ?? activeProjectPath;
				activeCodePath = data.code_path ?? activeCodePath;
				activeProjectId = data.project_id ?? activeProjectId;
				ctx.ui.notify(`JARVIS project registered: ${data.name ?? data.path}`, "info");
				ctx.ui.setStatus("jarvis", jlcLabel("ok", data.name));
			} else {
				ctx.ui.notify(`JARVIS register failed: ${data?.error ?? "sidecar unavailable"}`, "error");
			}
		},
	});

	pi.registerCommand("deepdive-reasoning", {
		description: "Set the saved JARVIS deepdive reasoning level: medium, high, xhigh.",
		handler: async (args, ctx) => {
			const level = normalizeThinkingLevel(args);
			if (!level) {
				ctx.ui.notify("Usage: /deepdive-reasoning medium|high|xhigh", "warning");
				return;
			}
			saveDeepdiveThinkingPreference(level);
			if (currentRoute === "heavy_deepdive") {
				try {
					suppressThinkingPreferenceSaveOnce(level);
					pi.setThinkingLevel(level);
					ctx.ui.setStatus(FOOTER_CHAT_THINKING_STATUS_KEY, level);
				} catch {
					/* pi may be stale */
				}
			}
			ctx.ui.notify(
				`JARVIS heavy deepdive reasoning set to ${level}${currentRoute !== "heavy_deepdive" ? " (will apply in heavy deepdive)" : ""}`,
				"info",
			);
		},
	});

	pi.registerCommand("model-setting", {
		description: "Pick JARVIS chat + encoder models (writes data/config.yaml).",
		handler: async (_args, ctx) => {
			await runModelSetting(pi, ctx);
		},
	});

	pi.registerCommand("api-key", {
		description: "Save a JARVIS provider API key (OpenAI, Anthropic, Brave Search, etc.).",
		handler: async (_args, ctx) => {
			await runApiKeySetting(ctx);
		},
	});

	pi.registerCommand("gpt-login", {
		description: "Login to ChatGPT/OpenAI subscription OAuth for JARVIS.",
		handler: async (_args, ctx) => {
			await runGptOAuthCommand(ctx, ["login", "chatgpt"], "GPT OAuth login");
		},
	});

	pi.registerCommand("gpt-login-device", {
		description: "Login to ChatGPT/OpenAI subscription OAuth with device code.",
		handler: async (_args, ctx) => {
			await runGptOAuthCommand(ctx, ["login", "--device-code", "chatgpt"], "GPT OAuth device login");
		},
	});

	pi.registerCommand("gpt-logout", {
		description: "Logout JARVIS ChatGPT/OpenAI subscription OAuth.",
		handler: async (_args, ctx) => {
			await runGptOAuthCommand(ctx, ["logout", "chatgpt"], "GPT OAuth logout");
		},
	});

	pi.registerCommand("gpt-auth-status", {
		description: "Show JARVIS ChatGPT/OpenAI subscription OAuth status.",
		handler: async (_args, ctx) => {
			await runGptOAuthCommand(ctx, ["status"], "GPT OAuth status");
		},
	});
}

async function runGptOAuthCommand(ctx: ExtensionContext, args: string[], label: string): Promise<void> {
	ctx.ui.notify(`${label}: starting...`, "info");
	const result = await runPythonLoginCli(args, 180000);
	const text = [result.stdout.trim(), result.stderr.trim()].filter(Boolean).join("\n").trim();
	const summary = oneLineForSummary(text || `exit ${result.code}`, 420);
	if (result.code === 0) {
		if (args[0] === "login") {
			const synced = syncPythonChatGptAuthToPiAuth();
			if (!synced.ok) {
				ctx.ui.notify(`${label}: login succeeded, but Pi auth sync failed: ${synced.error}`, "warning");
				return;
			}
			refreshPiRuntimeAuth(ctx);
		}
		if (args[0] === "logout") {
			removePiOpenAICodexAuth();
			refreshPiRuntimeAuth(ctx);
		}
		ctx.ui.notify(`${label}: ${summary}`, "info");
		return;
	}
	ctx.ui.notify(`${label} failed (${result.code}): ${summary}`, "error");
}

function refreshPiRuntimeAuth(ctx: ExtensionContext): void {
	try {
		ctx.modelRegistry.authStorage.reload();
		ctx.modelRegistry.refresh();
	} catch {
		/* Pi may be running with a stale command context. */
	}
}

function syncPythonChatGptAuthToPiAuth(): { ok: true } | { ok: false; error: string } {
	try {
		const pythonAuthPath = path.join(os.homedir(), ".jarvis-code", "auth.json");
		if (!fs.existsSync(pythonAuthPath)) return { ok: false, error: `${pythonAuthPath} not found` };
		const raw = JSON.parse(fs.readFileSync(pythonAuthPath, "utf8")) as {
			access_token?: string;
			refresh_token?: string;
			expires_at_unix?: number;
			account_id?: string | null;
		};
		if (!raw.access_token || !raw.refresh_token || !raw.expires_at_unix) {
			return { ok: false, error: "Python auth file is missing token fields" };
		}
		const piAuthPath = path.join(piAgentDir(), "auth.json");
		fs.mkdirSync(path.dirname(piAuthPath), { recursive: true });
		const existing = fs.existsSync(piAuthPath) ? JSON.parse(fs.readFileSync(piAuthPath, "utf8")) : {};
		existing["openai-codex"] = {
			type: "oauth",
			access: raw.access_token,
			refresh: raw.refresh_token,
			expires: raw.expires_at_unix * 1000,
			accountId: raw.account_id ?? undefined,
		};
		fs.writeFileSync(piAuthPath, `${JSON.stringify(existing, null, 2)}\n`, "utf8");
		return { ok: true };
	} catch (error) {
		return { ok: false, error: String((error as Error)?.message ?? error) };
	}
}

function removePiOpenAICodexAuth(): void {
	try {
		const piAuthPath = path.join(piAgentDir(), "auth.json");
		if (!fs.existsSync(piAuthPath)) return;
		const existing = JSON.parse(fs.readFileSync(piAuthPath, "utf8"));
		delete existing["openai-codex"];
		fs.writeFileSync(piAuthPath, `${JSON.stringify(existing, null, 2)}\n`, "utf8");
	} catch {
		/* ignore */
	}
}

function piAgentDir(): string {
	return (
		process.env.JARVIS_CODE_CODING_AGENT_DIR?.trim() ||
		process.env.PI_CODING_AGENT_DIR?.trim() ||
		path.join(repoRoot(), "pi-agent")
	);
}

function runPythonLoginCli(
	args: string[],
	timeoutMs: number,
): Promise<{ code: number; stdout: string; stderr: string }> {
	return new Promise((resolve) => {
		const python = process.env.JARVIS_PYTHON?.trim() || "python";
		const sidecarPath = path.join(repoRoot(), "sidecar");
		const env = {
			...process.env,
			PYTHONPATH: [sidecarPath, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
		};
		let stdout = "";
		let stderr = "";
		let settled = false;
		const child = spawn(python, ["-m", "jlc_agentic.cli.login", ...args], {
			cwd: repoRoot(),
			env,
			windowsHide: false,
			stdio: ["ignore", "pipe", "pipe"],
		});
		const finish = (code: number) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolve({ code, stdout, stderr });
		};
		const timer = setTimeout(() => {
			try {
				child.kill();
			} catch {
				/* ignore */
			}
			finish(124);
		}, timeoutMs);
		child.stdout?.on("data", (chunk) => {
			stdout += String(chunk);
		});
		child.stderr?.on("data", (chunk) => {
			stderr += String(chunk);
		});
		child.on("error", (error) => {
			stderr += String(error?.message ?? error);
			finish(127);
		});
		child.on("close", (code) => finish(code ?? 0));
	});
}

async function runApiKeySetting(ctx: ExtensionContext): Promise<void> {
	const catalog = await postSidecar<SidecarCredentialCatalogResponse>("/credentials/catalog", undefined, "GET");
	if (!catalog?.ok || !catalog.targets) {
		ctx.ui.notify(`JARVIS api-key: credential catalog failed (${catalog?.error ?? "sidecar unavailable"})`, "error");
		return;
	}
	const entries = Object.entries(catalog.targets).sort((a, b) => {
		const ac = a[1].configured ? 0 : 1;
		const bc = b[1].configured ? 0 : 1;
		if (ac !== bc) return ac - bc;
		return (a[1].label ?? a[0]).localeCompare(b[1].label ?? b[0]);
	});
	const labels = entries.map(([id, target]) => {
		const status = target.configured ? "configured" : "missing";
		const envName = target.env_name ? ` ${target.env_name}` : "";
		return `${target.label ?? id} (${status})${envName}`;
	});
	const picked = await ctx.ui.select("Save API key for", labels);
	if (picked === undefined) return;
	const idx = labels.indexOf(picked);
	if (idx < 0) return;
	const [, target] = entries[idx];
	const envName = target.env_name;
	if (!envName) {
		ctx.ui.notify("Selected provider has no API-key environment variable.", "warning");
		return;
	}
	const key = await ctx.ui.input(`Enter ${target.label ?? envName} API key`, envName);
	if (key === undefined) return;
	if (!key.trim()) {
		ctx.ui.notify("API key was empty; nothing saved.", "warning");
		return;
	}
	const result = await postSidecar<SidecarCredentialSetResponse>("/credentials/set", {
		env_name: envName,
		value: key.trim(),
		validate: true,
	});
	if (!result?.ok) {
		const detail = result?.validation?.error ?? result?.error ?? "validation failed";
		ctx.ui.notify(`JARVIS saved ${envName}, but validation failed: ${detail}`, "warning");
		return;
	}
	const validation = result.validation;
	const suffix = validation?.models ? ` (${validation.models} models)` : "";
	ctx.ui.notify(`JARVIS saved ${envName}${suffix}.`, "info");
}

async function runModelSetting(pi: ExtensionAPI, ctx: ExtensionContext): Promise<void> {
	const catalog = await postSidecar<SidecarLLMSettingCatalogResponse>("/llmsetting/catalog", undefined, "GET");
	if (!catalog?.ok || !catalog.providers) {
		ctx.ui.notify(
			`JARVIS model-setting: sidecar catalog fetch failed (${(catalog as { error?: string } | undefined)?.error ?? "no response"})`,
			"error",
		);
		return;
	}
	const providers = catalog.providers;
	const recommended = catalog.recommended ?? {};
	const current = catalog.current ?? {};
	const splitRole = (value: string | null | undefined): { provider?: string; model?: string } => {
		if (!value || !value.includes("/")) return {};
		const [p, m] = value.split("/", 2);
		return { provider: p, model: m };
	};
	const currentChat = splitRole(current.chat);
	const currentEncoder = splitRole(current.encoder);

	const rankWeight = (isCurrent: boolean, isRecommended: boolean): number => {
		if (isCurrent) return 0;
		if (isRecommended) return 1;
		return 2;
	};

	const labelWithMarkers = (name: string, isCurrent: boolean, isRecommended: boolean): string => {
		const marker = isCurrent ? "*" : isRecommended ? "+" : " ";
		const tags: string[] = [];
		if (isCurrent) tags.push("current");
		if (isRecommended) tags.push("recommended");
		const suffix = tags.length ? `  (${tags.join(", ")})` : "";
		return `${marker} ${name}${suffix}`;
	};

	const pickProvider = async (role: "chat" | "encoder"): Promise<string | undefined> => {
		const rec = recommended[role]?.provider;
		const curProvider = role === "chat" ? currentChat.provider : currentEncoder.provider;
		const ordered = Object.keys(providers)
			.slice()
			.sort((a, b) => {
				const wa = rankWeight(a === curProvider, a === rec);
				const wb = rankWeight(b === curProvider, b === rec);
				return wa - wb;
			});
		const labels = ordered.map((pid) => {
			const p = providers[pid];
			const name = p.label ?? pid;
			if (!p.available) return `x ${name}  (${p.reason ?? "unavailable"})`;
			return labelWithMarkers(name, pid === curProvider, pid === rec);
		});
		const picked = await ctx.ui.select(`Select ${role.toUpperCase()} provider`, labels);
		if (picked === undefined) return undefined;
		const idx = labels.indexOf(picked);
		if (idx < 0) return undefined;
		const pid = ordered[idx];
		if (!providers[pid].available) {
			ctx.ui.notify(
				`${providers[pid].label ?? pid} is not available: ${providers[pid].reason ?? "unknown"}`,
				"warning",
			);
			return undefined;
		}
		return pid;
	};

	const pickModel = async (role: "chat" | "encoder", providerId: string): Promise<string | undefined> => {
		const provider = providers[providerId];
		const models = provider.models ?? [];
		if (models.length === 0) {
			ctx.ui.notify(`No models reported for ${providerId}.`, "warning");
			return undefined;
		}
		const rec = recommended[role];
		const recModel = rec?.provider === providerId ? rec.model : undefined;
		const cur = role === "chat" ? currentChat : currentEncoder;
		const curModel = cur.provider === providerId ? cur.model : undefined;
		const ordered = [...models].sort((a, b) => {
			const wa = rankWeight(a === curModel, a === recModel);
			const wb = rankWeight(b === curModel, b === recModel);
			return wa - wb;
		});
		const labels = ordered.map((m) => labelWithMarkers(m, m === curModel, m === recModel));
		const picked = await ctx.ui.select(`${role.toUpperCase()} model on ${providerId}`, labels);
		if (picked === undefined) return undefined;
		const idx = labels.indexOf(picked);
		return idx >= 0 ? ordered[idx] : undefined;
	};

	const chatProvider = await pickProvider("chat");
	if (!chatProvider) return;
	const chatModel = await pickModel("chat", chatProvider);
	if (!chatModel) return;
	const encoderProvider = await pickProvider("encoder");
	if (!encoderProvider) return;
	const encoderModel = await pickModel("encoder", encoderProvider);
	if (!encoderModel) return;

	const result = await postSidecar<SidecarLLMSettingApplyResponse>("/llmsetting/apply", {
		chat: `${chatProvider}/${chatModel}`,
		encoder: `${encoderProvider}/${encoderModel}`,
	});
	if (!result?.ok) {
		ctx.ui.notify(`JARVIS model-setting apply failed: ${result?.error ?? "sidecar unavailable"}`, "error");
		return;
	}

	// pi-agent/models.json was just rewritten by the sidecar; refresh and swap
	// the live chat model so the user does not need to restart Pi.
	refreshPiRuntimeAuth(ctx);
	const next = ctx.modelRegistry.find(chatProvider, chatModel);
	if (!next) {
		ctx.ui.notify(
			`JARVIS models saved (chat=${result.chat}, encoder=${result.encoder}). Pi model registry did not contain ${chatProvider}/${chatModel} after refresh; restart Pi to apply.`,
			"warning",
		);
		return;
	}
	const swapped = await pi.setModel(next);
	if (!swapped) {
		ctx.ui.notify(
			`JARVIS models saved, but Pi refused to switch to ${chatProvider}/${chatModel} (missing API key?). Restart Pi to retry.`,
			"warning",
		);
		return;
	}
	ctx.ui.notify(
		`JARVIS chat -> ${chatProvider}/${chatModel} (live). Encoder=${encoderProvider}/${encoderModel}.`,
		"info",
	);
}

async function refreshProjectCache(): Promise<void> {
	const data = await postSidecar<SidecarProjectsResponse>("/projects", undefined, "GET");
	if (!data?.ok || !Array.isArray(data.projects)) {
		projectCache = [];
		projectCacheLoaded = false;
		return;
	}
	projectCache = data.projects.map(sidecarProjectToCached).filter((project): project is CachedProject => !!project);
	projectCacheLoaded = true;
}

function patchProjectCache(project: SidecarResolvedProject | SidecarSwitchResponse | undefined): void {
	const cached = sidecarProjectToCached(project);
	if (!cached) {
		const projectId = project?.project_id;
		if (projectId) projectCache = projectCache.filter((item) => item.project_id !== projectId);
		return;
	}
	const index = projectCache.findIndex((item) => item.project_id === cached.project_id);
	if (index === -1) {
		projectCache = [...projectCache, cached];
	} else {
		projectCache = [...projectCache.slice(0, index), cached, ...projectCache.slice(index + 1)];
	}
}

function sidecarProjectToCached(
	project: SidecarResolvedProject | SidecarSwitchResponse | undefined,
): CachedProject | null {
	if (!project?.project_id || !project.name || !project.path || !project.code_path) return null;
	return {
		project_id: project.project_id,
		name: project.name,
		slug: project.slug ?? project.name,
		path: project.path,
		code_path: project.code_path,
		code_path_normalized: normalizePathForCompare(project.code_path),
	};
}

export function resolveProjectByPathLocal(filePath: string): CachedProject | null {
	if (!filePath || projectCache.length === 0) return null;
	let abs: string;
	try {
		abs = path.resolve(filePath);
	} catch {
		return null;
	}
	const absNorm = normalizePathForCompare(abs);
	let best: CachedProject | null = null;
	let bestLen = -1;
	let tied = false;
	for (const proj of projectCache) {
		const code = proj.code_path_normalized;
		if (!code) continue;
		if (absNorm !== code && !absNorm.startsWith(`${code}/`)) continue;
		if (code.length > bestLen) {
			best = proj;
			bestLen = code.length;
			tied = false;
		} else if (code.length === bestLen) {
			tied = true;
		}
	}
	return tied ? null : best;
}

async function resolveProjectByPathSidecar(absPath: string): Promise<CachedProject | null> {
	const data = await postSidecar<SidecarResolveProjectByPathResponse>("/resolve_project_by_path", {
		path: absPath,
	});
	if (!data?.project?.path) return null;
	patchProjectCache(data.project);
	return sidecarProjectToCached(data.project);
}

async function resolveRegisteredProjectForPath(absPath: string): Promise<CachedProject | null> {
	return (
		resolveProjectByPathLocal(absPath) ?? (projectCacheLoaded ? null : await resolveProjectByPathSidecar(absPath))
	);
}

async function resolveActiveProjectPath(ctx: ExtensionContext, pi?: ExtensionAPI): Promise<string | undefined> {
	if (!isProjectRoute(currentRoute)) return undefined;
	if (activeProjectPath) return activeProjectPath;
	const response = await postSidecar<SidecarContextResponse>("/context", {
		cwd_hint: ctx.cwd,
		mode: modeForRoute(currentRoute),
		user_message: lastUserMessage,
		active_project_path: isProjectRoute(currentRoute) ? currentActiveProjectHint() : undefined,
		bench_conv_id: pi ? benchConvId(pi) : undefined,
		hints: {
			cwd: ctx.cwd,
		},
	});
	applyContextProjectState(response, modeForRoute(currentRoute));
	lastContextResponse = response;
	return activeProjectPath;
}

function currentActiveProjectHint(): string | undefined {
	return activeProjectPath;
}

function applyContextProjectState(response: SidecarContextResponse | undefined, _mode: SidecarContextMode): void {
	if (!response) return;
	activeProjectPath = response.active_project_path ?? activeProjectPath;
	activeCodePath = response.code_path ?? activeCodePath;
	activeProjectId = response.project_id ?? activeProjectId;
}

function clearActiveProjectState(): void {
	activeProjectPath = undefined;
	activeCodePath = undefined;
	activeProjectId = undefined;
}

function extractToolPath(input: unknown): string | undefined {
	if (!input || typeof input !== "object") return undefined;
	const args = input as { path?: unknown; file_path?: unknown };
	if (typeof args.path === "string" && args.path.trim()) return args.path;
	if (typeof args.file_path === "string" && args.file_path.trim()) return args.file_path;
	return undefined;
}

function extractToolPaths(input: unknown): string[] | undefined {
	if (!input || typeof input !== "object") return undefined;
	const items = (input as { items?: unknown }).items;
	if (!Array.isArray(items)) return undefined;
	const paths = items
		.map((item) => extractToolPath(item))
		.filter((item): item is string => typeof item === "string" && item.trim().length > 0);
	return paths.length ? paths : undefined;
}

function extractToolCommand(input: unknown): string | undefined {
	if (!input || typeof input !== "object") return undefined;
	const args = input as { command?: unknown; cmd?: unknown };
	if (typeof args.command === "string" && args.command.trim()) return args.command;
	if (typeof args.cmd === "string" && args.cmd.trim()) return args.cmd;
	return undefined;
}

function resolveToolPath(rawPath: string, ctx: ExtensionContext): string | undefined {
	try {
		return path.isAbsolute(rawPath) ? path.resolve(rawPath) : path.resolve(ctx.cwd, rawPath);
	} catch {
		return undefined;
	}
}

function resolveTurnMutationPath(rawPath: string, cwd?: string, ctx?: ExtensionContext): string | undefined {
	try {
		if (path.isAbsolute(rawPath)) return path.resolve(rawPath);
		const base = cwd || ctx?.cwd || activeCodePath || lastContextResponse?.code_path || process.cwd();
		return path.resolve(base, rawPath);
	} catch {
		return undefined;
	}
}

function normalizePathForCompare(value: string): string {
	return path.resolve(value).replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

function sameProjectPath(left: string | null | undefined, right: string | null | undefined): boolean {
	if (!left || !right) return false;
	return normalizePathForCompare(left) === normalizePathForCompare(right);
}

function pathIsInsideRoot(filePath: string, rootPath: string | null | undefined): boolean {
	if (!rootPath) return false;
	const fileNorm = normalizePathForCompare(filePath);
	const rootNorm = normalizePathForCompare(rootPath);
	return fileNorm === rootNorm || fileNorm.startsWith(`${rootNorm}/`);
}

function isJarvisMutationTool(toolName: unknown): boolean {
	const tool = String(toolName ?? "").toLowerCase();
	return tool === "edit" || tool === "write" || tool === "write_file" || tool === "apply_patch";
}

function pushUniqueBounded(items: string[], value: string | undefined, limit: number): void {
	const clean = (value ?? "").replace(/\r?\n/g, " ").trim();
	if (!clean || items.includes(clean)) return;
	items.push(clean);
	while (items.length > limit) items.shift();
}

function recordJarvisTurnToolOutcome(
	toolName: unknown,
	isError: unknown,
	output: string,
	metadata: JarvisToolMetadata,
	ctx?: ExtensionContext,
): void {
	const tool = String(toolName ?? "tool");
	if (tool.toLowerCase() === "update_jarvis_md") {
		turnJarvisMdUpdated = true;
	}
	pushUniqueBounded(turnExecutedCommands, metadata.command, 8);
	for (const line of splitJarvisLines(output)) {
		if (isJarvisVerificationLine(line)) pushUniqueBounded(turnVerificationLines, line, 6);
	}
	if (isError || !isJarvisMutationTool(toolName) || !metadata.sourcePath) return;
	const absPath = resolveTurnMutationPath(metadata.sourcePath, metadata.cwd, ctx);
	if (!absPath) return;
	const key = normalizePathForCompare(absPath);
	if (turnSuccessfulFileMutations.some((mutation) => normalizePathForCompare(mutation.path) === key)) return;
	turnSuccessfulFileMutations.push({ path: absPath, tool, command: metadata.command });
	while (turnSuccessfulFileMutations.length > 12) turnSuccessfulFileMutations.shift();
}

function resetJarvisTurnChoreographyState(): void {
	turnSuccessfulFileMutations = [];
	turnExecutedCommands = [];
	turnVerificationLines = [];
	turnJarvisMdUpdated = false;
}

function activeProjectMemoryPath(): string | undefined {
	return activeProjectPath ?? lastContextResponse?.active_project_path ?? undefined;
}

function activeProjectCodeRoot(): string | undefined {
	return activeCodePath ?? lastContextResponse?.code_path ?? activeProjectMemoryPath();
}

function formatPathForHarness(filePath: string, roots: Array<string | null | undefined>): string {
	for (const root of roots) {
		if (!root || !pathIsInsideRoot(filePath, root)) continue;
		const relative = path.relative(path.resolve(root), path.resolve(filePath));
		if (relative && !relative.startsWith("..")) return relative.replace(/\\/g, "/");
	}
	return filePath.replace(/\\/g, "/");
}

function formatHarnessList(items: string[], limit: number): string {
	const clean = items.map((item) => item.trim()).filter(Boolean);
	const kept = clean.slice(0, limit);
	const suffix = clean.length > kept.length ? ` (+${clean.length - kept.length})` : "";
	return kept.length ? `${kept.join(", ")}${suffix}` : "none";
}

function defaultWorkspaceRoot(): string {
	return (lastContextResponse?.default_project_root ?? "").trim() || "C:\\jarvis_workspace";
}

function pathIsInsideConfiguredWorkspace(filePath: string): boolean {
	const root = defaultWorkspaceRoot();
	return !!root && pathIsInsideRoot(filePath, root);
}

function workspaceDirectChildForFile(filePath: string): string | undefined {
	const root = defaultWorkspaceRoot();
	if (!root || !pathIsInsideRoot(filePath, root)) return undefined;
	let relative: string;
	try {
		relative = path.relative(path.resolve(root), path.resolve(filePath));
	} catch {
		return undefined;
	}
	const parts = relative.split(/[\\/]+/).filter(Boolean);
	if (parts.length < 2) return undefined;
	if (parts[0] === "." || parts[0] === "..") return undefined;
	return path.join(path.resolve(root), parts[0]);
}

async function patchJarvisMemoryFromObservedWork(pi: ExtensionAPI): Promise<void> {
	if (!isProjectRoute(currentRoute) || turnJarvisMdUpdated || turnSuccessfulFileMutations.length === 0) return;
	const projectPath = activeProjectMemoryPath();
	const codeRoot = activeProjectCodeRoot();
	if (!projectPath) return;
	const roots = [codeRoot, projectPath];
	const relevantMutations = turnSuccessfulFileMutations.filter((mutation) =>
		roots.some((root) => pathIsInsideRoot(mutation.path, root)),
	);
	if (relevantMutations.length === 0) return;
	const files = relevantMutations.map((mutation) => formatPathForHarness(mutation.path, roots));
	const date = new Date().toISOString().slice(0, 10);
	const fileList = formatHarnessList(files, 6);
	const commandList = formatHarnessList(turnExecutedCommands, 4);
	const verification = turnVerificationLines[0] ?? "none";
	const data = await postSidecar<{ ok?: boolean }>("/update_jarvis_md", {
		project_path: projectPath,
		updates: [
			{
				field: "NOW",
				value: `- ${date} Status: ${fileList} updated; verification: ${verification}`,
			},
			{
				field: "RAW",
				value: `- ${date} files: ${fileList}; commands: ${commandList}`,
			},
		],
	});
	if (data?.ok) {
		turnJarvisMdUpdated = true;
		sendJarvisChatNotice(pi, "✓ JARVIS.md 갱신 (하네스 보강)");
	}
}

async function registerWorkspaceFolderFromObservedWrite(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	for (const mutation of turnSuccessfulFileMutations) {
		const projectDir = workspaceDirectChildForFile(mutation.path);
		if (!projectDir) continue;
		const existing = resolveProjectByPathLocal(projectDir) ?? (await resolveProjectByPathSidecar(projectDir));
		if (existing) continue;
		const name = path.basename(projectDir);
		const registered = await postSidecar<SidecarSwitchResponse>("/register_project", {
			path: projectDir,
			name,
		});
		if (!registered?.ok || !registered.path) continue;
		patchProjectCache(registered);
		const switched = await postSidecar<SidecarSwitchResponse>("/switch_project", {
			slug_or_name: registered.slug ?? registered.name ?? name,
			code_path: registered.code_path ?? projectDir,
			auto_create: false,
		});
		const selected = switched?.ok && switched.path ? switched : registered;
		patchProjectCache(selected);
		activeProjectPath = selected.path ?? activeProjectPath;
		activeCodePath = selected.code_path ?? activeCodePath;
		activeProjectId = selected.project_id ?? activeProjectId;
		enterProjectWorkPreservingHeavy();
		try {
			ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down", selected.name));
			applyRouteThinkingLevel(currentRoute, ctx, pi);
		} catch {
			/* stale */
		}
		sendJarvisChatNotice(pi, "✓ 등록 → 스위칭 → JARVIS.md seed (하네스 보강)");
		return;
	}
}

function projectPathExistsForBackstop(projectPath: string): boolean {
	try {
		return fs.existsSync(projectPath);
	} catch {
		return true;
	}
}

async function unregisterMissingWorkspaceProjectsBackstop(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	const data = await postSidecar<SidecarProjectsResponse>("/projects", undefined, "GET");
	if (!data?.ok || !Array.isArray(data.projects)) return;
	for (const project of data.projects) {
		const targetPath = project.path;
		if (!targetPath || !pathIsInsideConfiguredWorkspace(targetPath)) continue;
		if (projectPathExistsForBackstop(targetPath)) continue;
		const removed = await postSidecar<SidecarUnregisterResponse>("/unregister_project", {
			project_id: project.project_id,
		});
		if (!removed?.removed) continue;
		const projectId = removed.project_id ?? project.project_id;
		projectCache = projectCache.filter((item) => item.project_id !== projectId);
		const activeMatches =
			activeProjectId === projectId ||
			sameProjectPath(activeProjectPath, targetPath) ||
			sameProjectPath(activeCodePath, project.code_path);
		if (activeMatches) {
			clearActiveProjectState();
			setEffectiveRoute("chat");
			try {
				ctx.ui.setStatus("jarvis", jlcLabel(sidecarHealthy ? "ok" : "down"));
			} catch {
				/* stale */
			}
		}
		const line = `✓ 등록해제(백스톱): ${projectId} — 폴더 삭제 관측`;
		appendSubturnSummaryLines([formatSubturnSummaryLine("backstop", line)]);
		appendSubturnCommit("tool", line);
		appendSubturnEvent("unregister_backstop", { project_id: projectId, path: targetPath });
		sendJarvisChatNotice(pi, line);
	}
}

function currentSafetyRoots(ctx: ExtensionContext): string[] {
	void ctx;
	return [
		activeCodePath,
		lastContextResponse?.code_path ?? undefined,
		activeProjectPath,
		lastContextResponse?.active_project_path ?? undefined,
	].filter((root): root is string => typeof root === "string" && root.trim().length > 0);
}

function pathIsInsideCurrentSafetyScope(filePath: string, ctx: ExtensionContext): boolean {
	return currentSafetyRoots(ctx).some((root) => pathIsInsideRoot(filePath, root));
}

function safetyKey(kind: string, value: string): string {
	return `${kind}:${value}`;
}

function externalMutationRoot(absPath: string): string {
	const dir = path.dirname(absPath);
	return dir && dir !== "." ? dir : absPath;
}

function externalMutationRootKey(root: string): string {
	return safetyKey("external-mutation-root", normalizePathForCompare(root));
}

function hasApprovedExternalMutationRoot(absPath: string, confirmedKeys: Set<string>): boolean {
	for (const key of confirmedKeys) {
		if (!key.startsWith("external-mutation-root:")) continue;
		const approvedRoot = key.slice("external-mutation-root:".length);
		const fileNorm = normalizePathForCompare(absPath);
		if (fileNorm === approvedRoot || fileNorm.startsWith(`${approvedRoot}/`)) return true;
	}
	return false;
}

async function confirmJarvisSafety(
	ctx: ExtensionContext,
	confirmedKeys: Set<string>,
	key: string,
	title: string,
	message: string,
	reason: string,
	approvedKeys: string[] = [key],
): Promise<ToolBlockResult | undefined> {
	if (confirmedKeys.has(key)) return undefined;
	let confirmed = false;
	try {
		const confirm = ctx.ui.confirm;
		confirmed =
			typeof confirm === "function" ? await confirm.call(ctx.ui, title, message, { timeout: 60_000 }) : false;
	} catch {
		confirmed = false;
	}
	if (confirmed) {
		confirmedKeys.add(key);
		for (const approvedKey of approvedKeys) {
			confirmedKeys.add(approvedKey);
		}
		return undefined;
	}
	try {
		ctx.ui.notify(reason, "warning");
	} catch {
		// Non-interactive modes should still block safely.
	}
	return { block: true, reason };
}

async function maybeConfirmExternalMutationToolCall(
	toolName: string,
	absPath: string,
	ctx: ExtensionContext,
	confirmedKeys: Set<string>,
): Promise<ToolBlockResult | undefined> {
	if (pathIsInsideCurrentSafetyScope(absPath, ctx)) return undefined;
	return confirmExternalMutationRoots(toolName, [absPath], ctx, confirmedKeys);
}

async function confirmExternalMutationRoots(
	toolName: string,
	absolutePaths: string[],
	ctx: ExtensionContext,
	confirmedKeys: Set<string>,
	detailLines: string[] = [],
): Promise<ToolBlockResult | undefined> {
	const roots = Array.from(
		new Set(
			absolutePaths
				.filter((target) => !pathIsInsideCurrentSafetyScope(target, ctx))
				.filter((target) => !hasApprovedExternalMutationRoot(target, confirmedKeys))
				.map(externalMutationRoot)
				.map(normalizePathForCompare),
		),
	);
	if (roots.length === 0) return undefined;
	const approvedKeys = roots.map((root) => externalMutationRootKey(root));
	const key = safetyKey("external-mutation-request", approvedKeys.join("|"));
	const title = "JARVIS external file change";
	const message = [
		`Tool: ${toolName}`,
		...detailLines,
		"Session approval scope:",
		...roots.map((root) => `- ${root}`),
		"",
		"This path is outside registered JARVIS projects.",
		"Allow file changes under this scope for this JARVIS session?",
	].join("\n");
	return confirmJarvisSafety(
		ctx,
		confirmedKeys,
		key,
		title,
		message,
		`Blocked ${toolName}: external file change was not confirmed (${roots.join(", ")}).`,
		approvedKeys,
	);
}

function describeDestructiveShellCommand(command: string): string | undefined {
	const normalized = command.replace(/[`^]/g, "").toLowerCase();
	const checks: Array<[RegExp, string]> = [
		[/\brm\s+-[^\n;&|]*[rf][^\n;&|]*[rf]?/, "recursive/force delete command"],
		[/\b(remove-item|rm|del|erase|rmdir|rd)\b[\s\S]*(^|\s)(-recurse|-r|\/s)(?=\s|$)/, "recursive delete command"],
		[/\bgit\s+reset\s+--hard\b/, "git reset --hard"],
		[/\bgit\s+clean\b[\s\S]*-[a-z]*[fdx][a-z]*/, "git clean destructive cleanup"],
		[/(^|[;&|]\s*)(format|diskpart)\b/, "disk or filesystem destructive command"],
		[/\bchmod\s+-r\s+777\b/, "recursive permission broadening"],
		[
			/\b(curl|wget|irm|iwr|invoke-webrequest)\b[\s\S]*\|[\s\S]*\b(sh|bash|iex|invoke-expression)\b/,
			"download-pipe-shell command",
		],
	];
	for (const [pattern, reason] of checks) {
		if (pattern.test(normalized)) return reason;
	}
	return undefined;
}

function commandLooksMutating(command: string): boolean {
	const normalized = command.replace(/[`^]/g, "").toLowerCase();
	return (
		/\b(set-content|add-content|out-file|new-item|move-item|copy-item|remove-item|rename-item)\b/.test(normalized) ||
		/\b(rm|del|erase|rmdir|rd|mv|cp|tee)\b/.test(normalized) ||
		/\bsed\s+-i\b/.test(normalized) ||
		/(^|[^>])>{1,2}\s*["']?[a-z]:[\\/]/i.test(command)
	);
}

function extractAbsoluteCommandPaths(command: string): string[] {
	const paths = new Set<string>();
	for (const match of command.matchAll(/[A-Za-z]:[\\/](?=($|[\s"'`<>|;&)]))/g)) {
		const candidate = match[0];
		if (candidate) paths.add(candidate);
	}
	for (const match of command.matchAll(/[A-Za-z]:[\\/][^"'`\s<>|;&)]+/g)) {
		const candidate = match[0];
		if (candidate) paths.add(candidate);
	}
	for (const match of command.matchAll(/(^|[\s"'`=])\/(?=($|[\s;&|)]))/g)) {
		const prefix = match[1] ?? "";
		const candidate = match[0].slice(prefix.length);
		if (candidate) paths.add(candidate);
	}
	for (const match of command.matchAll(/(^|[\s"'`=])\/(?!\/)[^"'`\s<>|;&)]+/g)) {
		const prefix = match[1] ?? "";
		const candidate = match[0].slice(prefix.length);
		if (candidate && !candidate.includes("://")) paths.add(candidate);
	}
	return Array.from(paths);
}

async function filterUnregisteredPaths(paths: string[]): Promise<string[]> {
	const unregistered: string[] = [];
	for (const target of paths) {
		const project = await resolveRegisteredProjectForPath(target);
		if (!project) unregistered.push(target);
	}
	return unregistered;
}

async function cwdIsRegisteredProject(ctx: ExtensionContext): Promise<boolean> {
	const cwd = resolveToolPath(".", ctx);
	if (!cwd) return false;
	return Boolean(await resolveRegisteredProjectForPath(cwd));
}

async function unregisteredMutationTargetsForCommand(
	absolutePaths: string[],
	ctx: ExtensionContext,
	options?: { cwdAsMutationScope?: boolean },
): Promise<string[]> {
	if (absolutePaths.length > 0) {
		return filterUnregisteredPaths(absolutePaths);
	}
	if (await cwdIsRegisteredProject(ctx)) return [];
	return [options?.cwdAsMutationScope ? path.join(ctx.cwd, "__jarvis_session_scope__") : ctx.cwd];
}

async function maybeConfirmRiskyBashToolCall(
	input: unknown,
	ctx: ExtensionContext,
	confirmedKeys: Set<string>,
): Promise<ToolBlockResult | undefined> {
	const command = extractToolCommand(input);
	if (!command) return undefined;
	const absolutePaths = extractAbsoluteCommandPaths(command)
		.map((candidate) => resolveToolPath(candidate, ctx))
		.filter((candidate): candidate is string => Boolean(candidate));
	const destructiveReason = describeDestructiveShellCommand(command);
	if (destructiveReason) {
		const unregisteredTargets = await unregisteredMutationTargetsForCommand(absolutePaths, ctx);
		// Blast-radius commands (rm -rf, git reset --hard, ...) confirm once even inside a
		// registered project — but only when a human can answer. Headless runs have no
		// approver, so registered targets keep running silently there (prior behavior);
		// unregistered targets still gate everywhere.
		let targets = unregisteredTargets;
		if (targets.length === 0 && typeof ctx.ui.confirm === "function") {
			targets = absolutePaths.length > 0 ? absolutePaths : [ctx.cwd];
		}
		if (targets.length === 0) return undefined;
		const normalizedTargets = targets.map(normalizePathForCompare).join("|");
		const key = safetyKey("bash-destructive", `${destructiveReason}:${normalizedTargets}:${command.slice(0, 500)}`);
		return confirmJarvisSafety(
			ctx,
			confirmedKeys,
			key,
			"JARVIS risky shell command",
			[
				`Command: ${command}`,
				"",
				`Risk: ${destructiveReason}`,
				"Target:",
				...targets.map((target) => `- ${target}`),
				"",
				"Allow this risky command for this JARVIS session?",
			].join("\n"),
			`Blocked bash: risky shell command was not confirmed (${destructiveReason}).`,
		);
	}
	if (!commandLooksMutating(command)) return undefined;
	const unregisteredTargets = await unregisteredMutationTargetsForCommand(absolutePaths, ctx, {
		cwdAsMutationScope: true,
	});
	if (unregisteredTargets.length === 0) return undefined;
	return confirmExternalMutationRoots("bash", unregisteredTargets, ctx, confirmedKeys, [`Command: ${command}`]);
}

function shouldRefreshContextForSelectedProject(projectPath: string | null | undefined): boolean {
	if (!projectPath) return false;
	if (lastInjectedContextMode !== "deepdive") return true;
	return !sameProjectPath(lastContextResponse?.active_project_path, projectPath);
}

async function maybeSwitchProjectFromUserMessage(userText: string, ctx: ExtensionContext): Promise<boolean> {
	const request = parseProjectSwitchRequest(userText);
	if (!request) return false;
	const response = await postSidecar<SidecarSwitchResponse>("/switch_project", {
		slug_or_name: request.slugOrName,
		code_path: request.codePath,
		auto_create: request.autoCreate,
	});
	if (response?.ok && response.path) {
		const needsContextRefresh = shouldRefreshContextForSelectedProject(response.path);
		patchProjectCache(response);
		activeProjectPath = response.path;
		activeCodePath = response.code_path ?? activeCodePath;
		activeProjectId = response.project_id ?? activeProjectId;
		pendingProjectCreate = undefined;
		enterProjectWorkPreservingHeavy();
		if (request.autoCreate) {
			transientSystemDirective = [
				"[Project creation complete]",
				`The project ${request.slugOrName} has been created and selected.`,
				"Briefly tell the user the default project root was used and continue with their request.",
			].join("\n");
		}
		ctx.ui.setStatus("jarvis", jlcLabel("ok", response.name));
		return needsContextRefresh;
	}
	const warnings = response?.warnings ?? [];
	const isUnknown = warnings.some((warning) => warning.includes("unknown project:"));
	if (!request.autoCreate && isUnknown) {
		pendingProjectCreate = { slugOrName: request.slugOrName, codePath: request.codePath };
		transientSystemDirective = [
			"[Pending project creation]",
			`The user requested project ${request.slugOrName}, but it is not registered.`,
			"Ask a short yes/no question about creating it now. Do not create it until the user confirms.",
			"If they confirm, the extension will call the sidecar with auto_create=True on the next turn.",
		].join("\n");
		ctx.ui.notify(`Unknown project: ${request.slugOrName}. Waiting for confirmation.`, "warning");
	}
	return false;
}

async function maybeHandlePendingProjectCreation(userText: string, ctx: ExtensionContext): Promise<boolean> {
	if (!pendingProjectCreate) return false;
	if (parseProjectSwitchCommand(userText)) return false;
	if (isAffirmative(userText)) {
		const pending = pendingProjectCreate;
		const response = await postSidecar<SidecarSwitchResponse>("/switch_project", {
			slug_or_name: pending.slugOrName,
			code_path: pending.codePath,
			auto_create: true,
		});
		if (response?.ok && response.path) {
			const needsContextRefresh = shouldRefreshContextForSelectedProject(response.path);
			patchProjectCache(response);
			activeProjectPath = response.path;
			activeCodePath = response.code_path ?? activeCodePath;
			activeProjectId = response.project_id ?? activeProjectId;
			enterProjectWorkPreservingHeavy();
			transientSystemDirective = [
				"[Project creation complete]",
				`The project ${pending.slugOrName} has been created and selected.`,
				"Briefly tell the user the default project root was used and continue with their request.",
			].join("\n");
			ctx.ui.setStatus("jarvis", jlcLabel("ok"));
			pendingProjectCreate = undefined;
			return needsContextRefresh;
		}
		pendingProjectCreate = undefined;
		return false;
	}
	if (isNegative(userText)) {
		const pending = pendingProjectCreate;
		pendingProjectCreate = undefined;
		transientSystemDirective = [
			"[Project creation cancelled]",
			`The user declined to create ${pending.slugOrName}.`,
			"Acknowledge the cancellation and ask what project to switch to instead.",
		].join("\n");
	}
	return false;
}

async function maybeHandleSetupFlow(userText: string, ctx: ExtensionContext): Promise<void> {
	const setupPath =
		parseSetupDefaultRootCommand(userText) ?? (setupRequired ? extractAbsolutePath(userText) : undefined);
	if (!setupPath) return;
	const response = await postSidecar<SidecarSetupResponse>("/setup", {
		default_project_root: setupPath,
	});
	if (response?.ok) {
		setupRequired = response.setup_required === true;
		transientSystemDirective = [
			"[Default project root configured]",
			`default_project_root is now ${response.default_project_root ?? setupPath}.`,
			"Tell the user the location was saved, then continue normally.",
		].join("\n");
		ctx.ui.notify(`JARVIS default project root set: ${response.default_project_root ?? setupPath}`, "info");
	}
}

function benchConvId(pi: ExtensionAPI): string | undefined {
	try {
		const value = pi.getFlag("bench-conv");
		return typeof value === "string" && value.trim() ? value.trim() : undefined;
	} catch {
		return undefined; // pi may be stale after session replacement
	}
}

function autoPromptEncodingConvId(pi: ExtensionAPI): string | undefined {
	const bench = benchConvId(pi);
	if (bench) return bench;
	const envConvId = process.env.JARVIS_CONV_ID?.trim() || process.env.JLC_UI_CONV_ID?.trim();
	return envConvId || undefined;
}

// P9 simplification (2026-05-19): the creation-intent pre-promotion that
// lived here was removed. Mode is now LLM-authoritative — the marker is
// the only deepdive trigger, slash overrides are the user's last word.

async function checkHealth(): Promise<boolean> {
	const response = await postSidecar<{ ok?: boolean }>("/health", undefined, "GET");
	return response?.ok === true;
}

function installInterruptInputCheckpointHook(ctx: ExtensionContext, pi: ExtensionAPI): void {
	clearInterruptInputCheckpointHook();
	try {
		interruptInputUnsubscribe = ctx.ui.onTerminalInput((data) => {
			if (data === "\x1b" && !ctx.isIdle()) {
				void saveInterruptCheckpoint(ctx, pi, lastAssistantPartialText);
			}
			return undefined;
		});
	} catch {
		interruptInputUnsubscribe = undefined;
	}
}

function clearInterruptInputCheckpointHook(): void {
	if (!interruptInputUnsubscribe) return;
	try {
		interruptInputUnsubscribe();
	} catch {
		/* stale UI */
	}
	interruptInputUnsubscribe = undefined;
}

async function saveInterruptCheckpoint(ctx: ExtensionContext, pi: ExtensionAPI, assistantText: string): Promise<void> {
	if (interruptCheckpointSavedThisTurn) return;
	interruptCheckpointSavedThisTurn = true;
	const checkpointEvents = [...toolEvents, ...checkpointToolEvents];
	const subturnLog = readSubturnLogForCheckpoint();
	if (!lastUserMessage.trim() && !assistantText.trim() && checkpointEvents.length === 0 && !subturnLog.trim()) return;

	refreshTurnCheckpointScope();
	const scope = turnCheckpointScope;
	if (!scope?.path) {
		sendJarvisChatNotice(pi, "인터럽트 감지: 현재 작업을 상태저장하려 했지만 저장 scope가 없습니다.");
		return;
	}

	const scopeLabel = scope.kind === "project" ? "프로젝트" : "채팅";
	sendJarvisChatNotice(pi, `인터럽트 감지: 현재 작업을 ${scopeLabel} 메모리에 상태저장합니다.`);
	try {
		ctx.ui.setStatus("jlc-work", "JLC: saving interrupted work");
	} catch {
		/* stale ctx */
	}
	let cwd: string | undefined;
	try {
		cwd = ctx.cwd;
	} catch {
		cwd = process.cwd();
	}
	const response = await postSidecar<SidecarInterruptCheckpointResponse>("/interrupt_checkpoint", {
		project_path: scope.path,
		user_message: lastUserMessage,
		assistant_message: assistantText,
		tool_events: checkpointEvents,
		mode: currentMode,
		cwd,
		reason: "escape_interrupt",
		subturn_log: subturnLog,
	});
	if (response?.ok) {
		const savedPath = response.path ? ` (${response.path})` : "";
		sendJarvisChatNotice(pi, `인터럽트 체크포인트를 ${scopeLabel} JARVIS.md에 저장했습니다.${savedPath}`);
	} else {
		sendJarvisChatNotice(pi, `인터럽트 체크포인트 저장 실패: ${response?.error ?? "JARVIS sidecar unavailable"}`);
	}
}

function resetSubturnLogState(): void {
	subturnLogInitializedForUserTurnKey = undefined;
	jarvisEvidenceByToolResultKey = new Map();
	subturnActiveToolMetadata = new Map();
	turnCompressedOutputsTotal = 0;
	turnCompressionSavedTotal = 0;
	turnReadCompressionEditTargetPaths = new Set();
	turnReadCompressionKeysByPath = new Map();
	subturnStartedAt = "";
	subturnMode = undefined;
	subturnProjectPath = "";
	subturnCwd = "";
	subturnUserMessage = "";
	lastSubturnAssistantUpdateLength = 0;
	subturnSummaryLines = [];
	subturnCommitLines = [];
	subturnCommitNextId = 1;
	resetProviderCallCeilingState();
	resetLockedResourceReportStopState();
	resetSubturnCarry();
	summarizedToolEventCount = 0;
	summarizedAssistantEndCount = 0;
}

function defaultInternalMemoryRoot(): string {
	return process.env.JARVIS_WORKSPACE?.trim() || path.join(os.homedir(), ".jarvis-code", "workspaceMemory");
}

function ensureChatMemoryRoot(): string {
	const root = path.join(defaultInternalMemoryRoot(), CHAT_MEMORY_DIR_NAME);
	try {
		fs.mkdirSync(root, { recursive: true });
		const jarvisPath = path.join(root, "JARVIS.md");
		if (!fs.existsSync(jarvisPath)) {
			fs.writeFileSync(
				jarvisPath,
				[
					"# JARVIS.md - chat",
					"",
					"## NOW",
					"Global chat memory for non-project conversations.",
					"",
					"## MAP",
					"",
					"## LAW",
					"",
					"## BAN",
					"",
					"## HABIT",
					"",
					"## WHY",
					"",
					"## OMM",
					"",
					"## RAW",
					"",
				].join("\n"),
				"utf8",
			);
		}
	} catch {
		/* ignore chat memory bootstrap failures */
	}
	return root;
}

function initializeSubturnLog(
	projectPath: string,
	userMessage: string,
	userTurnKey: string,
	cwd: string,
	mode: SidecarContextMode,
): void {
	const logPath = path.join(projectPath, SUBTURN_LOG_FILENAME);
	const compactEnabled = subturnCompactEnabled();
	if (subturnLogInitializedForUserTurnKey === userTurnKey && subturnProjectPath === projectPath) {
		if (!compactEnabled) removeSubturnLogFile(projectPath);
		return;
	}
	if (subturnLogInitializedForUserTurnKey !== userTurnKey) {
		// New user turn: evidence refs from the previous turn must not leak in.
		// (Mid-turn project switches keep the map — see R1-3 live finding.)
		jarvisEvidenceByToolResultKey = new Map();
		subturnActiveToolMetadata = new Map();
		turnCompressedOutputsTotal = 0;
		turnCompressionSavedTotal = 0;
		turnReadCompressionEditTargetPaths = new Set();
		turnReadCompressionKeysByPath = new Map();
		resetLockedResourceReportStopState();
	}
	try {
		subturnLogPath = compactEnabled ? logPath : undefined;
		subturnLogInitializedForUserTurnKey = userTurnKey;
		subturnStartedAt = new Date().toISOString();
		subturnMode = mode;
		subturnProjectPath = projectPath;
		subturnCwd = cwd;
		subturnUserMessage = userMessage;
		lastSubturnAssistantUpdateLength = 0;
		subturnSummaryLines = [];
		subturnCommitLines = [];
		subturnCommitNextId = 1;
		resetSubturnCarry();
		summarizedToolEventCount = 0;
		summarizedAssistantEndCount = 0;
		if (compactEnabled) {
			writeSubturnCompactState("active");
		} else {
			removeSubturnLogFile(projectPath);
		}
	} catch {
		/* ignore log failures */
	}
}

function removeSubturnLogFile(projectPath: string): void {
	try {
		const logPath = path.join(projectPath, SUBTURN_LOG_FILENAME);
		if (fs.existsSync(logPath)) fs.unlinkSync(logPath);
	} catch {
		/* ignore cleanup failures */
	}
}

function appendSubturnSummaryLines(lines: string[]): void {
	const cleanLines = lines.map((line) => line.trim()).filter(Boolean);
	if (cleanLines.length === 0) return;
	subturnSummaryLines.push(...cleanLines);
	const maxLines = 40;
	if (subturnSummaryLines.length > maxLines) {
		subturnSummaryLines = subturnSummaryLines.slice(-maxLines);
	}
	let text = subturnSummaryLines.join("\n");
	while (text.length > SUBTURN_SUMMARY_MAX_CHARS && subturnSummaryLines.length > 1) {
		subturnSummaryLines.shift();
		text = subturnSummaryLines.join("\n");
	}
}

function appendSubturnCommit(kind: string, text: string): void {
	const clean = oneLineForSummary(text, 220);
	if (!clean) return;
	const index = String(subturnCommitNextId++).padStart(3, "0");
	subturnCommitLines.push(`${index} ${kind}: ${clean}`);
	while (subturnCommitLines.length > SUBTURN_COMMIT_MAX_ITEMS) {
		subturnCommitLines.shift();
	}
	let joined = subturnCommitLines.join("\n");
	while (joined.length > SUBTURN_COMMIT_MAX_CHARS && subturnCommitLines.length > 1) {
		subturnCommitLines.shift();
		joined = subturnCommitLines.join("\n");
	}
}

function resetSubturnCarry(): void {
	subturnCarryByKey = new Map();
	subturnCarryOrder = [];
	subturnEvidenceByKey = new Map();
	subturnEvidenceOrder = [];
	subturnActiveToolCarryKeys = new Map();
	subturnActiveToolDescriptors = new Map();
	// jarvisEvidenceByToolResultKey / subturnActiveToolMetadata intentionally
	// survive: resetSubturnCarry runs on mid-turn project switches via
	// initializeSubturnLog. They reset on user-turn boundaries only.
}

function upsertSubturnCarry(key: string, line: string): void {
	const cleanKey = key.trim();
	const cleanLine = line.replace(/\r?\n/g, " ").trim();
	if (!cleanKey || !cleanLine) return;
	if (!subturnCarryByKey.has(cleanKey)) subturnCarryOrder.push(cleanKey);
	subturnCarryByKey.set(cleanKey, cleanLine);
	const maxItems = 16;
	while (subturnCarryOrder.length > maxItems) {
		const oldest = subturnCarryOrder.shift();
		if (oldest) subturnCarryByKey.delete(oldest);
	}
}

function subturnCarryLinesForPrompt(): string[] {
	return subturnCarryOrder
		.map((key) => subturnCarryByKey.get(key))
		.filter((line): line is string => !!line)
		.map((line) => `- ${line}`);
}

function subturnHistoryLinesForPrompt(): string[] {
	const maxHistory = 10;
	return subturnSummaryLines.slice(-maxHistory);
}

function upsertSubturnEvidence(key: string, label: string, text: string): void {
	const cleanKey = key.trim();
	const cleanLabel = label.replace(/\r?\n/g, " ").trim();
	const cleanText = truncateForCheckpoint(text.replace(/\r/g, "").trim(), 6000);
	if (!cleanKey || !cleanLabel || !cleanText) return;
	if (!subturnEvidenceByKey.has(cleanKey)) subturnEvidenceOrder.push(cleanKey);
	subturnEvidenceByKey.set(cleanKey, { label: cleanLabel, text: cleanText });
	const maxItems = 4;
	while (subturnEvidenceOrder.length > maxItems) {
		const oldest = subturnEvidenceOrder.shift();
		if (oldest) subturnEvidenceByKey.delete(oldest);
	}
}

function subturnEvidenceBlocksForPrompt(): string[] {
	const blocks: string[] = [];
	for (const key of subturnEvidenceOrder) {
		const evidence = subturnEvidenceByKey.get(key);
		if (!evidence) continue;
		blocks.push(`### ${evidence.label}`, "```text", evidence.text, "```");
	}
	return blocks;
}

function subturnToolEventKey(toolCallId: unknown, toolName: unknown): string {
	const id = typeof toolCallId === "string" ? toolCallId.trim() : "";
	if (id) return id;
	return `tool:${String(toolName ?? "tool")}`;
}

function buildSubturnCompactMarkdown(status: "active" | "completed" | "idle" = "active"): string {
	const carryLines = subturnCarryLinesForPrompt();
	const historyLines = subturnHistoryLinesForPrompt();
	const evidenceBlocks = subturnEvidenceBlocksForPrompt();
	const assistantLines = carryLines.filter((line) => /^-\s*assistant:/i.test(line));
	const mapLines = carryLines.filter((line) => /^-\s*(observed|modified|tool):/i.test(line));
	const verifyLines = carryLines.filter((line) => /^-\s*verify:/i.test(line));
	const ommLines = carryLines.filter((line) => /\b(error|failed|timeout|pending)\b/i.test(line));
	const sectionOrNone = (lines: string[]) => (lines.length > 0 ? lines : ["- none yet"]);
	return [
		"# JARVIS_SUBTURN.md",
		"",
		`Status: ${status}`,
		subturnStartedAt ? `Started at: ${subturnStartedAt}` : undefined,
		subturnMode ? `Mode: ${subturnMode}` : undefined,
		subturnProjectPath ? `Project: ${subturnProjectPath}` : undefined,
		subturnCwd ? `CWD: ${subturnCwd}` : undefined,
		"",
		"## NOW",
		`- User request: ${oneLineForSummary(subturnUserMessage || "(empty)", 500)}`,
		...sectionOrNone(assistantLines),
		"",
		"## MAP",
		...sectionOrNone(mapLines),
		"",
		"## LAW",
		"- Responses reasoning/function_call/function_call_output replay is omitted from provider payloads.",
		"- Use this compact same-turn state before rereading files or retrying tools.",
		"",
		"## BAN",
		"- Do not assume omitted same-turn tool results are absent.",
		"- Do not repeat completed reads/edits/checks unless the compact state is insufficient.",
		"",
		"## HABIT",
		"- Continue from the compact state; reread exact files only when exact current contents are needed.",
		"",
		"## WHY",
		"- Bounded section state prevents same-turn payload growth while preserving progress.",
		"",
		"## OMM",
		...sectionOrNone([...ommLines, ...verifyLines]),
		"",
		"## RAW",
		"- Recent bounded event history:",
		...sectionOrNone(historyLines),
		"",
		"Recent exact tool evidence:",
		...(evidenceBlocks.length > 0 ? evidenceBlocks : ["- none yet"]),
		"",
	]
		.filter((line): line is string => line !== undefined)
		.join("\n");
}

function buildSubturnObserveState(payload?: unknown): SubturnObserveState {
	const carryLines = subturnCarryLinesForPrompt().map(stripSubturnObserveBullet).filter(Boolean);
	const historyLines = subturnHistoryLinesForPrompt()
		.map((line) => oneLineForSummary(stripSubturnObserveBullet(line), 260))
		.filter(Boolean);
	const visibleCarryLines = carryLines.filter((line) => !isNonBlockingSubturnObserveErrorLine(line));
	const visibleHistoryLines = historyLines.filter((line) => !isNonBlockingSubturnObserveErrorLine(line));
	const observedLines = visibleCarryLines.filter((line) => /^observed:/i.test(line));
	const modifiedLines = visibleCarryLines.filter((line) => /^modified:/i.test(line));
	const verifyLines = visibleCarryLines.filter((line) => /^verify:/i.test(line));
	const toolLines = visibleCarryLines.filter((line) => /^tool:/i.test(line));
	const assistantLines = visibleCarryLines.filter((line) => /^assistant:/i.test(line));
	const inspectedFiles = uniqueSubturnObserveItems(
		observedLines.map((line) => extractSubturnObservePath(line) ?? removeSubturnObservePrefix(line)),
		8,
		220,
	);
	const modifiedFiles = uniqueSubturnObserveItems(
		modifiedLines.map((line) => extractSubturnObservePath(line) ?? removeSubturnObservePrefix(line)),
		8,
		220,
	);
	const verification = uniqueSubturnObserveItems(
		[
			...verifyLines.map(removeSubturnObservePrefix),
			...toolLines
				.filter((line) => /\b(test|check|build|pytest|vitest|tsc|tsgo|biome|doctor)\b/i.test(line))
				.map(removeSubturnObservePrefix),
		],
		8,
		260,
	);
	const unresolvedErrors = uniqueSubturnObserveItems(
		[...visibleCarryLines, ...visibleHistoryLines].filter(isBlockingSubturnObserveErrorLine),
		8,
		280,
	);
	const decisions = uniqueSubturnObserveItems(
		assistantLines.map((line) => removeSubturnObservePrefix(line)).filter(Boolean),
		5,
		260,
	);
	const evidenceRefs = buildSubturnObserveEvidenceRefs();
	const currentPhase = inferSubturnObservePhase({
		inspectedFiles,
		modifiedFiles,
		verification,
		unresolvedErrors,
	});
	const repeatedReadTarget =
		currentPhase === "implement" && modifiedFiles.length === 0
			? detectRepeatedSubturnReadTarget(subturnCommitLines)
			: undefined;
	return {
		status: "active",
		started_at: subturnStartedAt || undefined,
		goal: oneLineForSummary(
			subturnUserMessage || lastUserMessage || latestUserTextFromProviderPayload(payload) || "(empty)",
			500,
		),
		route: currentRoute,
		mode: subturnMode,
		project_path: subturnProjectPath || undefined,
		cwd: subturnCwd || undefined,
		current_phase: currentPhase,
		completed_steps: buildSubturnCompletedSteps({ inspectedFiles, modifiedFiles, verification, decisions }),
		pending_steps: buildSubturnPendingSteps(currentPhase, unresolvedErrors.length > 0, repeatedReadTarget),
		inspected_files: inspectedFiles,
		modified_files: modifiedFiles,
		verification,
		unresolved_errors: unresolvedErrors,
		decisions,
		evidence_refs: evidenceRefs,
		internal_commits: subturnCommitLines.filter((line) => !isNonBlockingSubturnObserveErrorLine(line)),
		recent_history: uniqueSubturnObserveItems(visibleHistoryLines.slice(-8), 8, 260),
	};
}

function latestUserTextFromProviderPayload(payload: unknown): string {
	if (!payload || typeof payload !== "object") return "";
	const source = payload as { messages?: unknown; input?: unknown };
	const messages = Array.isArray(source.messages) ? source.messages : Array.isArray(source.input) ? source.input : [];
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (!message || typeof message !== "object") continue;
		const role = (message as { role?: unknown }).role;
		if (role !== "user") continue;
		const content = (message as { content?: unknown }).content;
		const text = typeof content === "string" ? content : contentToLogText(content);
		if (text.trim()) return stripJarvisMemoryBlock(text).trim();
	}
	return "";
}

function renderSubturnObserveState(state: SubturnObserveState): string {
	const list = (items: string[]) => (items.length > 0 ? items.map((item) => `- ${item}`) : ["- none yet"]);
	const evidence = state.evidence_refs.length
		? state.evidence_refs.map((item) => `- ${item.label}: ${item.summary}`)
		: ["- none yet"];
	return [
		"# JARVIS_SUBTURN_STATE",
		"",
		`Status: ${state.status}`,
		state.started_at ? `Started at: ${state.started_at}` : undefined,
		`Route: ${state.route}`,
		state.mode ? `Mode: ${state.mode}` : undefined,
		state.project_path ? `Project: ${state.project_path}` : undefined,
		state.cwd ? `CWD: ${state.cwd}` : undefined,
		subturnHostPathGuideline(),
		"",
		"## Goal",
		`- ${state.goal}`,
		"",
		"## Current",
		`- Phase: ${state.current_phase}`,
		"",
		"## Carry Rules",
		"- This bounded state replaces older raw same-turn assistant/tool transcript items.",
		"- Do not assume omitted tool results are absent.",
		"- Reread exact files only when exact current contents are needed.",
		"",
		"## Completed Steps",
		...list(state.completed_steps),
		"",
		"## Pending Steps",
		...list(state.pending_steps),
		"",
		"## Files",
		"- Inspected:",
		...list(state.inspected_files),
		"- Modified:",
		...list(state.modified_files),
		"",
		"## Verification",
		...list(state.verification),
		"",
		"## Errors",
		...list(state.unresolved_errors),
		"",
		"## Decisions",
		...list(state.decisions),
		"",
		"## Evidence Refs",
		...evidence,
		"",
		"## Internal Commit Log",
		...list(state.internal_commits),
		"",
		"## Recent History",
		...list(state.recent_history),
		"",
	]
		.filter((line): line is string => line !== undefined)
		.join("\n");
}

function subturnHostPathGuideline(): string | undefined {
	if (process.platform !== "win32") return undefined;
	return "Path format: Windows host. Use drive-letter paths like C:\\... or relative paths; do not use /mnt/c, /c, or /cygdrive/c pseudo paths.";
}

function stripSubturnObserveBullet(line: string): string {
	return line.replace(/^-\s*/, "").trim();
}

function removeSubturnObservePrefix(line: string): string {
	return line.replace(/^(observed|modified|verify|tool|assistant):\s*/i, "").trim();
}

function isBlockingSubturnObserveErrorLine(line: string): boolean {
	if (!/\b(error|failed|failure|timeout|blocked|unavailable|exception|traceback)\b/i.test(line)) return false;
	if (isNonBlockingSubturnObserveErrorLine(line)) return false;
	return true;
}

function isNonBlockingSubturnObserveErrorLine(line: string): boolean {
	if (isPcCeilingReportStopLine(line)) return true;
	if (isLockedResourceReportStopLine(line)) return true;
	if (isUnregisterBackstopLine(line)) return true;
	if (/\bread error\b/i.test(line) && /\bOffset \d+ is beyond end of file\b/i.test(line)) return true;
	if (/\bgit\b/i.test(line) && /fatal:\s+not a git repository\b/i.test(line)) return true;
	return false;
}

function isPcCeilingReportStopLine(line: string): boolean {
	return line.includes(PC_CEILING_REPORT_STOP_MARKER);
}

function isLockedResourceReportStopLine(line: string): boolean {
	return line.includes(LOCKED_RESOURCE_REPORT_STOP_MARKER);
}

function isUnregisterBackstopLine(line: string): boolean {
	return line.includes("등록해제(백스톱)") && line.includes("폴더 삭제 관측");
}

function detectRepeatedSubturnReadTarget(lines: string[]): string | undefined {
	const counts = new Map<string, number>();
	for (const line of lines.slice(-12)) {
		if (!/\bobserved:\s+read ok\b/i.test(line)) continue;
		const target = extractSubturnObservePath(line) ?? extractSubturnReadArgsKey(line);
		if (!target) continue;
		const count = (counts.get(target) ?? 0) + 1;
		counts.set(target, count);
		if (count >= 4) return target;
	}
	return undefined;
}

function extractSubturnReadArgsKey(line: string): string | undefined {
	const argsMatch = /\bargs=(.+?)(?:\s+=>|$)/i.exec(line);
	if (!argsMatch?.[1]) return undefined;
	return oneLineForSummary(argsMatch[1], 140);
}

function uniqueSubturnObserveItems(items: string[], maxItems: number, maxChars: number): string[] {
	const seen = new Set<string>();
	const result: string[] = [];
	for (const item of items) {
		const clean = oneLineForSummary(String(item ?? ""), maxChars);
		if (!clean) continue;
		const key = clean.toLowerCase();
		if (seen.has(key)) continue;
		seen.add(key);
		result.push(clean);
		if (result.length >= maxItems) break;
	}
	return result;
}

function extractSubturnObservePath(line: string): string | undefined {
	const quoted = /["']path["']\s*:\s*["']([^"']+)["']/i.exec(line);
	if (quoted?.[1]) return oneLineForSummary(quoted[1], 220);
	const pathEquals = /\bpath=(.+?)(?:\s+=>|\s+\w+=|$)/i.exec(line);
	if (pathEquals?.[1]) return oneLineForSummary(pathEquals[1].replace(/[",;)\]]+$/g, ""), 220);
	const fileEquals = /\b(?:file|target)=([^\s]+)/i.exec(line);
	if (fileEquals?.[1]) return oneLineForSummary(fileEquals[1].replace(/[",;)\]]+$/g, ""), 220);
	return undefined;
}

function buildSubturnObserveEvidenceRefs(): SubturnObserveEvidenceRef[] {
	const refs: SubturnObserveEvidenceRef[] = [];
	for (const key of subturnEvidenceOrder) {
		const evidence = subturnEvidenceByKey.get(key);
		if (!evidence) continue;
		refs.push({
			key,
			label: oneLineForSummary(evidence.label, 160),
			summary: summarizeSubturnObserveEvidence(evidence.text),
		});
		if (refs.length >= 4) break;
	}
	return refs;
}

function summarizeSubturnObserveEvidence(text: string): string {
	const clean = text.replace(/\r/g, "").trim();
	if (!clean) return "";
	const firstLines = clean
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean)
		.slice(0, 3)
		.join(" | ");
	return oneLineForSummary(firstLines || clean, 280);
}

function inferSubturnObservePhase(state: {
	inspectedFiles: string[];
	modifiedFiles: string[];
	verification: string[];
	unresolvedErrors: string[];
}): SubturnObservePhase {
	if (state.unresolvedErrors.length > 0 && state.modifiedFiles.length > 0) return "verify";
	if (state.verification.length > 0 && state.modifiedFiles.length > 0 && state.unresolvedErrors.length === 0) {
		return "report";
	}
	if (state.modifiedFiles.length > 0) return "verify";
	if (state.inspectedFiles.length > 0) return "implement";
	if (subturnSummaryLines.length > 0 || subturnCarryOrder.length > 0) return "inspect";
	return "start";
}

function buildSubturnCompletedSteps(state: {
	inspectedFiles: string[];
	modifiedFiles: string[];
	verification: string[];
	decisions: string[];
}): string[] {
	const steps: string[] = [];
	if (state.inspectedFiles.length > 0) steps.push(`Inspected ${state.inspectedFiles.slice(0, 3).join(", ")}`);
	if (state.modifiedFiles.length > 0) steps.push(`Modified ${state.modifiedFiles.slice(0, 3).join(", ")}`);
	if (state.modifiedFiles.length > 0 && state.verification.length > 0) {
		steps.push(`Verification observed: ${state.verification[0]}`);
	}
	if (state.decisions.length > 0) steps.push(`Latest assistant state: ${state.decisions[0]}`);
	return uniqueSubturnObserveItems(steps, 6, 280);
}

function buildSubturnPendingSteps(
	phase: SubturnObservePhase,
	hasErrors: boolean,
	repeatedReadTarget?: string,
): string[] {
	if (hasErrors) return ["Resolve the unresolved error before reporting."];
	if (phase === "start") return ["Identify the relevant project context and first action."];
	if (phase === "inspect") return ["Finish inspecting the implementation context."];
	if (phase === "implement" && repeatedReadTarget) {
		return [
			`Stop rereading ${repeatedReadTarget}; use the retained evidence/recent tool output to edit, or ask if exact context is still missing.`,
		];
	}
	if (phase === "implement") return ["Implement the scoped change."];
	if (phase === "verify") return ["Run focused verification and inspect any failures."];
	return ["Summarize the result for the user."];
}

function writeSubturnCompactState(status: "active" | "completed" | "idle" = "active"): void {
	if (!subturnCompactEnabled()) return;
	if (!subturnLogPath) return;
	try {
		fs.writeFileSync(subturnLogPath, buildSubturnCompactMarkdown(status), "utf8");
	} catch {
		/* ignore log failures */
	}
}

function summarizeToolDescriptor(toolName: unknown, args: unknown): { key: string; text: string } {
	const tool = String(toolName ?? "tool");
	const pathArg = extractToolPath(args);
	const command = extractToolCommand(args);
	if (pathArg) {
		const shortPath = pathArg.length > 120 ? `...${pathArg.slice(-117)}` : pathArg;
		return { key: `${tool}:path:${shortPath}`, text: `path=${shortPath}` };
	}
	if (command) {
		return {
			key: `${tool}:cmd:${command.slice(0, 120)}`,
			text: `command=${oneLineForSummary(command, 180)}`,
		};
	}
	return {
		key: `${tool}:args:${JSON.stringify(args ?? {}).slice(0, 120)}`,
		text: `args=${oneLineForSummary(JSON.stringify(args ?? {}), 180)}`,
	};
}

function carryKindForTool(toolName: unknown, descriptor: string): string {
	const tool = String(toolName ?? "tool").toLowerCase();
	if (/node --check|npm test|npm run|pytest|vitest|tsgo|biome/i.test(descriptor)) return "verify";
	if (tool === "edit" || tool === "write" || tool === "write_file" || tool === "apply_patch") return "modified";
	if (tool === "read" || tool === "grep" || tool === "find" || tool === "ls") return "observed";
	if (tool === "bash") return "observed";
	return "tool";
}

function shouldKeepSubturnEvidence(toolName: unknown, output: string): boolean {
	if (!output.trim()) return false;
	const tool = String(toolName ?? "tool").toLowerCase();
	if (tool === "read" || tool === "grep" || tool === "find") return true;
	if (tool === "bash") return output.length <= 6000;
	return false;
}

function refreshSubturnRollingSummary(): void {
	const completedToolEvents = toolEvents.slice(0, Math.max(0, toolEvents.length - SUBTURN_RECENT_ASSISTANT_CYCLES));
	if (summarizedToolEventCount < completedToolEvents.length) {
		const nextEvents = completedToolEvents.slice(summarizedToolEventCount);
		const lines: string[] = [];
		for (const event of nextEvents) {
			for (const result of event.toolResults ?? []) {
				lines.push(
					formatSubturnSummaryLine(
						"tool_result",
						`${result.toolName ?? "tool"} ${result.isError ? "error" : "ok"} ${oneLineForSummary(
							result.text ?? "",
							220,
						)}`,
					),
				);
			}
		}
		appendSubturnSummaryLines(lines);
		summarizedToolEventCount = completedToolEvents.length;
	}

	const assistantEnds = subturnSummaryLines.filter((line) => line.includes(" assistant:")).length;
	if (assistantEnds > summarizedAssistantEndCount + SUBTURN_RECENT_ASSISTANT_CYCLES) {
		summarizedAssistantEndCount = Math.max(0, assistantEnds - SUBTURN_RECENT_ASSISTANT_CYCLES);
	}
}

function formatSubturnSummaryLine(kind: string, text: string): string {
	const index = String(subturnSummaryLines.length + 1).padStart(2, "0");
	return `- ${index} ${kind}: ${text}`;
}

function oneLineForSummary(text: string, maxChars: number): string {
	return truncateForCheckpoint(text.replace(/\r?\n/g, " "), maxChars);
}

function subturnEventLogManagedRoots(): string[] {
	return [
		repoRoot(),
		defaultInternalMemoryRoot(),
		process.env.JARVIS_PROJECT_ROOT,
		process.env.JARVIS_USER_CODE_ROOT,
		process.platform === "win32" ? "C:\\jarvis_workspace" : undefined,
	]
		.filter((root): root is string => typeof root === "string" && root.trim().length > 0)
		.map((root) => path.resolve(root));
}

function subturnEventLogTarget(): string | undefined {
	const raw = process.env.JARVIS_SUBTURN_EVENT_LOG;
	if (typeof raw !== "string" || !raw.trim()) return undefined;
	const target = path.resolve(raw.trim());
	if (subturnEventLogManagedRoots().some((root) => pathIsInsideRoot(target, root))) return undefined;
	return target;
}

export function writeSubturnEventLogLine(kind: string, data: Record<string, unknown>): void {
	const target = subturnEventLogTarget();
	if (!target) return;
	try {
		const line = JSON.stringify({
			ts: new Date().toISOString(),
			kind,
			provider_call: providerCallCountThisTurn,
			turn_key: subturnLogInitializedForUserTurnKey ?? "",
			...data,
		});
		fs.appendFileSync(target, `${line}\n`, "utf8");
	} catch {
		// Best-effort durable debug sink only.
	}
}

function resetProviderCallCeilingState(): void {
	subturnPcCeilingReportStopActive = false;
	subturnPcCeilingProviderCall = undefined;
}

function resetLockedResourceReportStopState(): void {
	subturnLockedResourceActionCounts = new Map<string, number>();
	subturnLockedResourceRecordedCallIds = new Map<string, { key: string; count: number; line: string }>();
	subturnLockedResourceReportStopActive = false;
	subturnLockedResourceReportStopRecord = undefined;
}

function applyLockedResourceReportStop(payload: unknown): unknown {
	if (!subturnLockedResourceReportStopActive) return payload;
	return forceProviderPayloadToReportStop(payload, buildLockedResourceReportStopText());
}

function maybeBlockLockedResourceToolCall(toolName: string, input: unknown): ToolBlockResult | undefined {
	if (!subturnLockedResourceReportStopActive) return undefined;
	const reason = buildLockedResourceReportStopText();
	recordLockedResourceDebugEvent("locked_resource_tool_block", {
		tool: toolName,
		descriptor: summarizeToolDescriptor(toolName, input).text,
		content: reason,
	});
	return { block: true, reason, terminate: true };
}

function buildLockedResourceReportStopText(): string {
	const record = subturnLockedResourceReportStopRecord;
	const reason = record?.line
		? oneLineForSummary(record.line, 260)
		: "the resource is still locked by another process";
	return [LOCKED_RESOURCE_REPORT_STOP_TEXT, `반복 실패: ${reason}`].join("\n");
}

function recordLockedResourceToolOutcome(args: {
	toolCallId?: unknown;
	toolName?: unknown;
	input?: unknown;
	isError?: boolean;
	outputText: string;
}): { key: string; count: number; line: string } | undefined {
	const key = lockedResourceActionKey(args.toolName, args.input, args.outputText);
	if (!args.isError) {
		clearLockedResourceActionRecord(key);
		return undefined;
	}
	const line = lockedResourceActionLine(args.toolName, args.input, args.outputText);
	if (!isLockedResourceMutationAttempt(args.toolName, args.input, line)) return undefined;
	if (!isLockedResourceActionErrorLine(line)) return undefined;
	const callId = typeof args.toolCallId === "string" && args.toolCallId.trim() ? args.toolCallId.trim() : undefined;
	if (callId) {
		const existing = subturnLockedResourceRecordedCallIds.get(callId);
		if (existing) return existing;
	}
	const count = (subturnLockedResourceActionCounts.get(key) ?? 0) + 1;
	subturnLockedResourceActionCounts.set(key, count);
	const record = { key, count, line };
	if (callId) subturnLockedResourceRecordedCallIds.set(callId, record);
	if (count > 1 && !subturnLockedResourceReportStopActive) {
		subturnLockedResourceReportStopActive = true;
		subturnLockedResourceReportStopRecord = record;
		recordLockedResourceDebugEvent("locked_resource_report_stop", {
			tool: args.toolName,
			tool_call_id: args.toolCallId,
			locked_resource_key: key,
			attempts: count,
			reason: line,
			content: buildLockedResourceReportStopText(),
		});
	}
	return record;
}

function clearLockedResourceActionRecord(key: string): void {
	if (!key) return;
	subturnLockedResourceActionCounts.delete(key);
	for (const [callId, record] of subturnLockedResourceRecordedCallIds.entries()) {
		if (record.key === key) subturnLockedResourceRecordedCallIds.delete(callId);
	}
	if (subturnLockedResourceReportStopRecord?.key === key) {
		subturnLockedResourceReportStopActive = false;
		subturnLockedResourceReportStopRecord = undefined;
	}
}

function recordLockedResourceDebugEvent(kind: string, data: Record<string, unknown>): void {
	appendSubturnEvent(kind, data);
	void postSidecar("/debug/subturn/observe", {
		source: "jlc",
		event: kind,
		user_turn_key: subturnLogInitializedForUserTurnKey,
		data,
	});
}

function lockedResourceActionLine(toolName: unknown, input: unknown, outputText: string): string {
	const command = extractToolCommand(input);
	const descriptor = extractToolDescriptor(input);
	const rawPath = extractToolPath(input);
	return [
		String(toolName ?? ""),
		command ? `command=${command}` : "",
		descriptor,
		rawPath ? `path=${rawPath}` : "",
		outputText,
	]
		.filter(Boolean)
		.join(" ");
}

function extractToolDescriptor(input: unknown): string {
	if (!input || typeof input !== "object") return "";
	const descriptor = (input as { descriptor?: unknown }).descriptor;
	return typeof descriptor === "string" ? descriptor : "";
}

function lockedResourceActionKey(toolName: unknown, input: unknown, outputText: string): string {
	const command = extractToolCommand(input);
	const rawPath = extractToolPath(input);
	const descriptor = extractToolDescriptor(input);
	const target = command ? commandTargetKey(command) : rawPath ? normalizePathForCompare(rawPath) : "";
	const source = target || descriptor || outputText;
	return `${String(toolName ?? "tool").toLowerCase()}:${oneLineForSummary(source, 220).toLowerCase()}`;
}

function commandTargetKey(command: string): string {
	const absolutePaths = extractAbsoluteCommandPaths(command);
	if (absolutePaths.length > 0) return normalizePathForCompare(absolutePaths[0] ?? command);
	return oneLineForSummary(command.replace(/\s+/g, " ").trim(), 220).toLowerCase();
}

function isLockedResourceMutationAttempt(toolName: unknown, input: unknown, line: string): boolean {
	const tool = String(toolName ?? "").toLowerCase();
	if (tool === "write" || tool === "write_file" || tool === "edit" || tool === "apply_patch" || tool === "create") {
		return true;
	}
	const command = extractToolCommand(input);
	if (tool === "bash" && command) {
		return /\b(?:rm|rmdir|del|erase|unlink|remove-item|move-item|mv|delete)\b/i.test(command);
	}
	return /\b(?:delete|remove|write|edit|move|unlink|rmdir)\b/i.test(line);
}

function isLockedResourceActionErrorLine(line: string): boolean {
	const clean = removeSubturnObservePrefix(line);
	const hardLock =
		/(?:being|is)\s+used by another process|cannot access the file|in use by another|locked by|\bEBUSY\b|resource busy|device or resource busy|text file busy|\bENOTEMPTY\b|directory (?:is )?not empty/i.test(
			clean,
		);
	if (hardLock) return true;
	const permissionLike = /access (?:is )?denied|permission denied|operation not permitted|\bE(?:PERM|ACCES)\b/i.test(
		clean,
	);
	if (!permissionLike) return false;
	return /\b(?:unlink|rmdir|rm|remove(?:-item)?|delete|del|erase|directory|folder|file|write|edit)\b/i.test(clean);
}

function applyProviderCallCeiling(payload: unknown, providerCall: number): unknown {
	const ceiling = getSubturnProviderCallCeiling();
	if (!Number.isFinite(ceiling)) return payload;
	if (!subturnPcCeilingReportStopActive && providerCall < ceiling) return payload;
	const state = buildSubturnObserveState(payload);
	if (!subturnPcCeilingReportStopActive) {
		subturnPcCeilingReportStopActive = true;
		subturnPcCeilingProviderCall = providerCall;
		recordProviderCallCeilingDebugEvent("pc_ceiling_report_stop", {
			provider_call: providerCall,
			ceiling,
			phase: state.current_phase,
			pending: state.pending_steps[0] ?? "",
			unresolved_errors: state.unresolved_errors,
		});
	}
	return forceProviderPayloadToReportStop(
		payload,
		buildProviderCallCeilingReportStopText(subturnPcCeilingProviderCall ?? providerCall, ceiling, state),
	);
}

function maybeBlockProviderCallCeilingToolCall(toolName: string, input: unknown): ToolBlockResult | undefined {
	if (!subturnPcCeilingReportStopActive) return undefined;
	const ceiling = getSubturnProviderCallCeiling();
	const providerCall = subturnPcCeilingProviderCall ?? providerCallCountThisTurn;
	const reason = buildProviderCallCeilingReportStopText(providerCall, ceiling);
	recordProviderCallCeilingDebugEvent("pc_ceiling_tool_block", {
		tool: toolName,
		provider_call: providerCall,
		ceiling,
		descriptor: summarizeToolDescriptor(toolName, input).text,
		content: reason,
	});
	return { block: true, reason, terminate: true };
}

function recordProviderCallCeilingDebugEvent(kind: string, data: Record<string, unknown>): void {
	appendSubturnEvent(kind, data);
	void postSidecar("/debug/subturn/observe", {
		source: "jlc",
		event: kind,
		user_turn_key: subturnLogInitializedForUserTurnKey,
		data,
	});
}

function buildProviderCallCeilingReportStopText(
	providerCall: number,
	ceiling: number,
	state?: SubturnObserveState,
): string {
	const completed = state?.completed_steps.slice(0, 3).join("; ") || state?.internal_commits.slice(0, 3).join("; ");
	const blocker =
		state?.unresolved_errors[0] ||
		state?.pending_steps[0] ||
		(state?.current_phase ? `phase=${state.current_phase}` : "no further legal progress identified");
	return [
		`${PC_CEILING_REPORT_STOP_MARKER}: INCOMPLETE - provider-call ceiling reached (${providerCall}/${ceiling}).`,
		`여기까지 한 것: ${completed || "recorded progress is in the JARVIS subturn state"}.`,
		`안 되는 것/남은 것: ${blocker}.`,
		"Report this honestly to the user and stop. Do not call tools, search for another route, or restart the task.",
	].join("\n");
}

function forceProviderPayloadToReportStop(payload: unknown, text: string): unknown {
	const stopMessage = { role: "developer", content: text };
	if (!payload || typeof payload !== "object") return { messages: [stopMessage], tools: [] };
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	if (messages) return { ...record, messages: [...messages, stopMessage], tools: [] };
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	if (input) return { ...record, input: [...input, stopMessage], tools: [] };
	const instructions = typeof record.instructions === "string" ? record.instructions : undefined;
	if (instructions !== undefined) return { ...record, instructions: `${instructions}\n\n${text}`, tools: [] };
	return { ...record, messages: [stopMessage], tools: [] };
}

function filterChatRouteOnlyTools(payload: unknown): unknown {
	if (currentRoute === "chat") return payload;
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	if (!tools?.length) return payload;
	const filtered = tools.filter((tool) => {
		const name = providerToolSchemaName(tool);
		return !name || !CHAT_ROUTE_ONLY_TOOL_NAMES.has(name);
	});
	return filtered.length === tools.length ? payload : { ...record, tools: filtered };
}

function appendSubturnEvent(kind: string, data: Record<string, unknown>): void {
	writeSubturnEventLogLine(kind, data);
	if (!subturnCompactEnabled()) return;
	if (!subturnLogPath) return;
	void kind;
	void data;
	writeSubturnCompactState("active");
}

function completeSubturnLog(status: "completed" | "idle", assistantText = ""): void {
	if (!subturnCompactEnabled()) return;
	if (!subturnLogPath) return;
	if (assistantText.trim()) {
		upsertSubturnCarry("assistant:last", `assistant: ${oneLineForSummary(assistantText, 260)}`);
	}
	appendSubturnSummaryLines([formatSubturnSummaryLine("turn_complete", status)]);
	try {
		writeSubturnCompactState(status);
		fs.unlinkSync(subturnLogPath);
		subturnLogPath = undefined;
	} catch {
		/* ignore log failures */
	}
}

function readSubturnLogForCheckpoint(): string {
	if (!subturnCompactEnabled()) return "";
	if (!subturnLogPath) return "";
	try {
		return fs.readFileSync(subturnLogPath, "utf8");
	} catch {
		return "";
	}
}

function contentToLogText(content: unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) {
		const text = contentToText(content);
		if (text) return text;
	}
	try {
		return JSON.stringify(content ?? "");
	} catch {
		return String(content ?? "");
	}
}

function summarizeSubturnText(
	text: string,
	headChars = SUBTURN_TOOL_OUTPUT_HEAD_CHARS,
	tailChars = SUBTURN_TOOL_OUTPUT_TAIL_CHARS,
): Record<string, unknown> {
	const clean = text.replace(/\r/g, "").trim();
	if (clean.length <= headChars + tailChars) {
		return { chars: clean.length, text: clean };
	}
	return {
		chars: clean.length,
		head: clean.slice(0, headChars).trim(),
		tail: clean.slice(Math.max(0, clean.length - tailChars)).trim(),
		truncated: true,
	};
}

function truncateForCheckpoint(text: string, maxChars: number): string {
	const clean = text.replace(/\r/g, "").trim();
	if (clean.length <= maxChars) return clean;
	return `${clean.slice(0, Math.max(0, maxChars - 15)).trim()}...[truncated]`;
}

export async function storeJarvisEvidence(entry: JarvisEvidenceStoreEntry): Promise<{ ref: string } | undefined> {
	try {
		const response = await postSidecar<{ ok?: boolean; ref?: string }>("/evidence/store", entry);
		if (!response || response.ok === false || typeof response.ref !== "string" || !response.ref.trim()) {
			return undefined;
		}
		return { ref: response.ref };
	} catch {
		return undefined;
	}
}

export function buildJarvisCompressedMarker(
	kind: string,
	ref: string,
	counts: JarvisCompressedMarkerCounts,
	meta: JarvisCompressedMarkerMeta = {},
): string {
	const countParts = Object.entries(counts)
		.filter(([, value]) => typeof value === "number" && Number.isFinite(value))
		.map(([key, value]) => `${key}=${Math.max(0, Math.round(value as number))}`);
	const lines = [
		`[jarvis-compressed kind=${markerField(kind)} ref=${markerField(ref)}${countParts.length ? ` ${countParts.join(" ")}` : ""}]`,
		`command: ${markerLineValue(meta.command)}`,
		`cwd: ${markerLineValue(meta.cwd)}`,
		`exit_code: ${meta.exit_code ?? ""}`,
		"[/jarvis-compressed]",
	];
	return lines.join("\n");
}

function markerField(value: string): string {
	return String(value ?? "")
		.replace(/\s+/g, "_")
		.replace(/[^\w.-]/g, "_");
}

function markerLineValue(value: unknown): string {
	return String(value ?? "")
		.replace(/\r?\n/g, " ")
		.trim();
}

function metadataForJarvisToolEvent(
	event: Record<string, unknown>,
	fallbackExitCode: number | null | undefined,
): JarvisToolMetadata {
	const args = event.args ?? event.input ?? {};
	const argsRecord = args && typeof args === "object" ? (args as Record<string, unknown>) : {};
	const result = event.result && typeof event.result === "object" ? (event.result as Record<string, unknown>) : {};
	const command = extractToolCommand(args);
	const sourcePath = extractToolPath(args);
	const sourcePaths = extractToolPaths(args);
	const cwd =
		(typeof event.cwd === "string" && event.cwd) ||
		(typeof argsRecord.cwd === "string" && argsRecord.cwd) ||
		(typeof argsRecord.working_directory === "string" && argsRecord.working_directory) ||
		undefined;
	const exitCodeValue =
		event.exitCode ?? event.exit_code ?? result.exitCode ?? result.exit_code ?? argsRecord.exit_code;
	const exitCode =
		typeof exitCodeValue === "number"
			? exitCodeValue
			: typeof exitCodeValue === "string" && /^-?\d+$/.test(exitCodeValue.trim())
				? Number.parseInt(exitCodeValue, 10)
				: fallbackExitCode;
	return {
		command,
		cwd,
		exitCode,
		sourcePath,
		sourcePaths,
	};
}

function mergeJarvisToolMetadata(
	startMetadata: JarvisToolMetadata | undefined,
	endMetadata: JarvisToolMetadata,
): JarvisToolMetadata {
	return {
		command: endMetadata.command ?? startMetadata?.command,
		cwd: endMetadata.cwd ?? startMetadata?.cwd,
		exitCode: endMetadata.exitCode ?? startMetadata?.exitCode,
		sourcePath: endMetadata.sourcePath ?? startMetadata?.sourcePath,
		sourcePaths: endMetadata.sourcePaths ?? startMetadata?.sourcePaths,
	};
}

export function detectJarvisToolOutputKind(
	toolName: unknown,
	command: string | undefined,
	output: string,
	sourcePath?: string,
	sourcePaths?: string[],
): JarvisToolOutputKind | undefined {
	if (!output.trim()) return undefined;
	if (JARVIS_COMPRESSED_MARKER_RE.test(output) || JARVIS_READ_SKELETON_MARKER_RE.test(output)) return undefined;
	const tool = String(toolName ?? "").toLowerCase();
	const commandText = String(command ?? "").trim();
	if (tool === "read") {
		return (sourcePath || sourcePaths?.length) && jarvisReadOutputMeetsCompressionThreshold(output)
			? "read_skeleton"
			: undefined;
	}
	if (isSingleFileCatCommand(commandText)) return undefined;

	const lines = splitJarvisLines(output);
	const nonEmpty = lines.filter((line) => line.trim());
	const bytes = Buffer.byteLength(output, "utf8");
	if (nonEmpty.length === 0) return undefined;

	const searchMatches = parseJarvisSearchMatches(lines);
	const searchRatio = searchMatches.length / Math.max(1, nonEmpty.length);
	if (
		(searchMatches.length >= 40 || bytes >= JARVIS_COMPRESS_MIN_BYTES) &&
		searchRatio >= 0.3 &&
		isSearchToolOrCommand(tool, commandText)
	) {
		return "search_rg";
	}

	const pathLikeLines = nonEmpty.filter(isJarvisPathLikeListingLine).length;
	const pathLikeRatio = pathLikeLines / Math.max(1, nonEmpty.length);
	if (
		(nonEmpty.length >= 80 || bytes >= JARVIS_COMPRESS_MIN_BYTES) &&
		pathLikeRatio >= 0.5 &&
		isListingToolOrCommand(tool, commandText)
	) {
		return "directory_listing";
	}

	const logSignals = nonEmpty.slice(0, 200).filter(isJarvisLogSignalLine).length;
	if (
		(nonEmpty.length >= 50 || bytes >= JARVIS_COMPRESS_MIN_BYTES) &&
		isLogToolOrCommand(tool, commandText) &&
		logSignals >= 1
	) {
		return "build_log";
	}

	return undefined;
}

function jarvisToolOutputMeetsCompressionGuard(kind: JarvisToolOutputKind, output: string): boolean {
	if (JARVIS_COMPRESSED_MARKER_RE.test(output) || JARVIS_READ_SKELETON_MARKER_RE.test(output)) return false;
	const lines = splitJarvisLines(output).filter((line) => line.trim());
	const bytes = Buffer.byteLength(output, "utf8");
	if (kind === "read_skeleton")
		return lines.length >= JARVIS_READ_COMPRESS_MIN_LINES || bytes >= JARVIS_READ_COMPRESS_MIN_BYTES;
	if (bytes >= JARVIS_COMPRESS_MIN_BYTES) return true;
	if (kind === "search_rg") return lines.length >= 40;
	if (kind === "directory_listing") return lines.length >= 80;
	if (kind === "build_log") return lines.length >= 50;
	return false;
}

function jarvisCompressProviderPayload(payload: unknown): JarvisCompressionOutcome {
	if (jarvisEvidenceByToolResultKey.size === 0 || !payload || typeof payload !== "object") {
		const skips = payload && typeof payload === "object" ? { no_evidence_map: 1 } : undefined;
		return { payload, compressed_tool_outputs: 0, compression_saved_tokens_est: 0, compression_skips: skips };
	}
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	let compressedCount = 0;
	let savedTokens = 0;
	const skips: Record<string, number> = {};
	const skip = (reason: string) => {
		skips[reason] = (skips[reason] ?? 0) + 1;
	};

	const compressMessages = (items: Array<Record<string, unknown>>): Array<Record<string, unknown>> => {
		let changed = false;
		const nextItems = items.map((message) => {
			const next = jarvisCompressProviderToolResultMessage(message, skip);
			if (next !== message) {
				changed = true;
				compressedCount += 1;
				savedTokens += Math.max(0, (next as { __jarvis_saved_tokens_est?: number }).__jarvis_saved_tokens_est ?? 0);
				delete (next as { __jarvis_saved_tokens_est?: number }).__jarvis_saved_tokens_est;
			}
			return next;
		});
		return changed ? nextItems : items;
	};

	const nextMessages = messages ? compressMessages(messages) : undefined;
	const nextInput = input ? compressMessages(input) : undefined;
	const changed = (messages && nextMessages !== messages) || (input && nextInput !== input);
	if (!changed) {
		return { payload, compressed_tool_outputs: 0, compression_saved_tokens_est: 0, compression_skips: skips };
	}
	return {
		payload: {
			...record,
			...(nextMessages ? { messages: nextMessages } : {}),
			...(nextInput ? { input: nextInput } : {}),
		},
		compressed_tool_outputs: compressedCount,
		compression_saved_tokens_est: Math.round(savedTokens),
		compression_skips: skips,
	};
}

function jarvisCompressProviderToolResultMessage(
	message: Record<string, unknown>,
	skip: (reason: string) => void = () => {},
): Record<string, unknown> {
	if (classifyProviderPayloadMessage(message) !== "recent_tool_output") return message;
	const textTarget = jarvisToolResultTextTarget(message);
	if (!textTarget) {
		skip("no_text_target");
		return message;
	}
	if (JARVIS_COMPRESSED_MARKER_RE.test(textTarget.text)) {
		skip("already_compressed");
		return message;
	}
	if (JARVIS_READ_SKELETON_MARKER_RE.test(textTarget.text)) {
		skip("already_compressed");
		return message;
	}
	const key = jarvisToolResultKeyFromMessage(message);
	if (!key) {
		skip("no_key");
		return message;
	}
	const evidence = jarvisEvidenceByToolResultKey.get(key);
	if (!evidence) {
		skip("no_evidence_for_key");
		return message;
	}
	if (sha256Hex24(textTarget.text) !== evidence.originalRef) {
		skip("hash_mismatch");
		return message;
	}
	const decision = jarvisCompressStoredToolOutput(evidence, textTarget.text);
	if (!decision.changed) {
		skip(decision.skipReason ?? "decision_unchanged");
		return message;
	}
	for (const [reason, count] of Object.entries(decision.skipReasons ?? {})) {
		for (let index = 0; index < count; index++) skip(reason);
	}
	const next = applyJarvisToolResultText(message, textTarget, decision.content) as Record<string, unknown> & {
		__jarvis_saved_tokens_est?: number;
	};
	next.__jarvis_saved_tokens_est = decision.savedTokensEst;
	return next;
}

function jarvisCompressStoredToolOutput(evidence: JarvisStoredToolEvidence, output: string): JarvisCompressionDecision {
	if (evidence.storeSkippedReason) {
		return unchangedJarvisCompressionDecision(output, evidence.storeSkippedReason);
	}
	if (!jarvisToolOutputMeetsCompressionGuard(evidence.kind, output)) {
		return unchangedJarvisCompressionDecision(
			output,
			evidence.kind === "read_skeleton" ? "read_below_threshold" : undefined,
		);
	}
	if (!evidence.ref) return unchangedJarvisCompressionDecision(output, "no_evidence_for_key");
	const actualKind = detectJarvisToolOutputKind(
		evidence.toolName,
		evidence.command,
		output,
		evidence.sourcePath,
		evidence.sourcePaths,
	);
	if (actualKind !== evidence.kind) return unchangedJarvisCompressionDecision(output);
	let compressed: string | undefined;
	if (evidence.kind === "search_rg") compressed = jarvisCompressSearchOutput(evidence, output);
	if (evidence.kind === "directory_listing") compressed = jarvisCompressListingOutput(evidence, output);
	if (evidence.kind === "build_log") compressed = jarvisCompressLogOutput(evidence, output);
	if (evidence.kind === "read_skeleton") {
		return jarvisCompressReadOutput(evidence, output);
	}
	if (!compressed) return unchangedJarvisCompressionDecision(output);
	const originalTokens = estimateTextTokenCount(output);
	const compressedTokens = estimateTextTokenCount(compressed);
	if (compressedTokens >= originalTokens) return unchangedJarvisCompressionDecision(output);
	return {
		changed: true,
		kind: evidence.kind,
		content: compressed,
		originalTokensEst: originalTokens,
		compressedTokensEst: compressedTokens,
		savedTokensEst: originalTokens - compressedTokens,
	};
}

function unchangedJarvisCompressionDecision(output: string, skipReason?: string): JarvisCompressionDecision {
	const tokens = estimateTextTokenCount(output);
	return {
		changed: false,
		content: output,
		originalTokensEst: tokens,
		compressedTokensEst: tokens,
		savedTokensEst: 0,
		skipReason,
	};
}

function jarvisReadOutputMeetsCompressionThreshold(output: string): boolean {
	if (JARVIS_READ_SKELETON_MARKER_RE.test(output)) return false;
	const lines = splitJarvisLines(output).filter((line) => line.trim());
	const bytes = Buffer.byteLength(output, "utf8");
	return lines.length >= JARVIS_READ_COMPRESS_MIN_LINES || bytes >= JARVIS_READ_COMPRESS_MIN_BYTES;
}

function isJarvisReadCompressionEditTargetTool(toolName: unknown): boolean {
	const tool = String(toolName ?? "").toLowerCase();
	return tool === "edit" || tool === "write" || tool === "write_file";
}

function jarvisReadSourcePathKey(
	rawPath: string | undefined,
	cwd?: string,
	ctx?: ExtensionContext,
): string | undefined {
	if (!rawPath?.trim()) return undefined;
	const resolved = resolveTurnMutationPath(rawPath, cwd, ctx);
	try {
		return normalizePathForCompare(resolved ?? rawPath);
	} catch {
		return rawPath.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
	}
}

function recordJarvisReadCompressionKey(pathKey: string, evidenceKey: string): void {
	const existing = turnReadCompressionKeysByPath.get(pathKey) ?? [];
	if (existing[existing.length - 1] !== evidenceKey) {
		turnReadCompressionKeysByPath.set(pathKey, [...existing.filter((key) => key !== evidenceKey), evidenceKey]);
	}
}

function isLatestJarvisReadCompressionKey(pathKey: string | undefined, evidence: JarvisStoredToolEvidence): boolean {
	if (!pathKey) return false;
	const keys = turnReadCompressionKeysByPath.get(pathKey) ?? [];
	if (keys.length < 2) return false;
	const latestKey = keys[keys.length - 1];
	for (const [key, item] of jarvisEvidenceByToolResultKey.entries()) {
		if (item !== evidence) continue;
		return key === latestKey;
	}
	return false;
}

function isJarvisCodeSkeletonPath(sourcePath: string | undefined): boolean {
	const ext = path.extname(String(sourcePath ?? "")).toLowerCase();
	return [".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py"].includes(ext);
}

function isJarvisMemoryReadPath(sourcePath: string | undefined): boolean {
	return path.basename(String(sourcePath ?? "")).toLowerCase() === "jarvis.md";
}

type JarvisBatchReadSection = {
	header: string;
	path: string;
	startLine: number;
	endLine: number;
	lines: string[];
};

function jarvisCompressReadOutput(evidence: JarvisStoredToolEvidence, output: string): JarvisCompressionDecision {
	if (!evidence.ref) return unchangedJarvisCompressionDecision(output, "no_evidence_for_key");
	if (evidence.sourcePaths?.length) return jarvisCompressBatchReadOutput(evidence, output);
	const sourcePath = evidence.sourcePath;
	const sourcePathKey = evidence.sourcePathKey;
	if (sourcePathKey && turnReadCompressionEditTargetPaths.has(sourcePathKey)) {
		return unchangedJarvisCompressionDecision(output, "read_edit_target");
	}
	if (isJarvisMemoryReadPath(sourcePath)) {
		return unchangedJarvisCompressionDecision(output, "read_jarvis_md");
	}
	if (isLatestJarvisReadCompressionKey(sourcePathKey, evidence)) {
		return unchangedJarvisCompressionDecision(output, "read_latest_raw");
	}
	if (!isJarvisCodeSkeletonPath(sourcePath)) {
		return unchangedJarvisCompressionDecision(output, "read_non_code");
	}
	const skeleton = buildJarvisReadSkeleton(evidence, output);
	if (!skeleton) return unchangedJarvisCompressionDecision(output, "read_low_gain");
	const originalTokens = estimateTextTokenCount(output);
	const compressedTokens = estimateTextTokenCount(skeleton);
	if (compressedTokens >= originalTokens * JARVIS_READ_SKELETON_MAX_RATIO) {
		return unchangedJarvisCompressionDecision(output, "read_low_gain");
	}
	return {
		changed: true,
		kind: "read_skeleton",
		content: skeleton,
		originalTokensEst: originalTokens,
		compressedTokensEst: compressedTokens,
		savedTokensEst: originalTokens - compressedTokens,
	};
}

function jarvisCompressBatchReadOutput(evidence: JarvisStoredToolEvidence, output: string): JarvisCompressionDecision {
	const sections = parseJarvisBatchReadSections(output);
	if (!sections?.length) return unchangedJarvisCompressionDecision(output, "read_batch_unparsed");
	const skipReasons: Record<string, number> = {};
	const addSkip = (reason: string) => {
		skipReasons[reason] = (skipReasons[reason] ?? 0) + 1;
	};
	let skeletonSections = 0;
	const body: string[] = [
		`[JARVIS read-skeleton] paths=${sections.length} lines=${splitJarvisLines(output).length} ref=${evidence.ref}`,
		`원문 회수: retrieve_output(ref="${evidence.ref}") - 이 파일을 수정하려면 반드시 원문을 회수해서 봐라.`,
	];
	for (let index = 0; index < sections.length; index++) {
		const section = sections[index];
		body.push(section.header);
		const sourcePathKey = jarvisBatchSectionPathKey(evidence, section.path);
		const rawBody = section.lines.join("\n");
		if (sourcePathKey && turnReadCompressionEditTargetPaths.has(sourcePathKey)) {
			addSkip("read_edit_target");
			body.push(rawBody);
		} else if (isJarvisMemoryReadPath(section.path)) {
			addSkip("read_jarvis_md");
			body.push(rawBody);
		} else if (!isJarvisCodeSkeletonPath(section.path)) {
			addSkip("read_non_code");
			body.push(rawBody);
		} else {
			const skeletonBody = buildJarvisReadSkeletonBody(section.path, section.lines, section.startLine);
			if (!skeletonBody) {
				addSkip("read_low_gain");
				body.push(rawBody);
			} else {
				skeletonSections += 1;
				body.push(...skeletonBody);
			}
		}
		if (index < sections.length - 1) body.push("");
	}
	if (skeletonSections === 0) {
		return unchangedJarvisCompressionDecision(output, Object.keys(skipReasons)[0] ?? "decision_unchanged");
	}
	const compressed = body.join("\n");
	const originalTokens = estimateTextTokenCount(output);
	const compressedTokens = estimateTextTokenCount(compressed);
	if (compressedTokens >= originalTokens * JARVIS_READ_SKELETON_MAX_RATIO) {
		return unchangedJarvisCompressionDecision(output, "read_low_gain");
	}
	return {
		changed: true,
		kind: "read_skeleton",
		content: compressed,
		originalTokensEst: originalTokens,
		compressedTokensEst: compressedTokens,
		savedTokensEst: originalTokens - compressedTokens,
		skipReasons,
	};
}

function parseJarvisBatchReadSections(output: string): JarvisBatchReadSection[] | undefined {
	const lines = splitJarvisLines(output);
	const headers: Array<{ index: number; header: string; path: string; startLine: number; endLine: number }> = [];
	for (let index = 0; index < lines.length; index++) {
		const parsed = parseJarvisBatchReadHeader(lines[index] ?? "");
		if (parsed) headers.push({ index, ...parsed });
	}
	if (headers.length === 0) return undefined;
	if (lines.slice(0, headers[0]?.index ?? 0).some((line) => line.trim())) return undefined;
	return headers.map((header, index) => {
		const nextHeader = headers[index + 1];
		const endIndex = nextHeader ? nextHeader.index : lines.length;
		const sectionLines = lines.slice(header.index + 1, endIndex);
		if (nextHeader && sectionLines[sectionLines.length - 1] === "") sectionLines.pop();
		return {
			header: header.header,
			path: header.path,
			startLine: header.startLine,
			endLine: header.endLine,
			lines: sectionLines,
		};
	});
}

function parseJarvisBatchReadHeader(
	line: string,
): { header: string; path: string; startLine: number; endLine: number } | undefined {
	const match = /^---\s+(.+):(\d+)-(\d+)\s+---$/.exec(line.trim());
	if (!match?.[1]) return undefined;
	const startLine = Number.parseInt(match[2] ?? "", 10);
	const endLine = Number.parseInt(match[3] ?? "", 10);
	if (!Number.isFinite(startLine) || !Number.isFinite(endLine) || startLine < 1 || endLine < startLine) {
		return undefined;
	}
	return { header: line, path: match[1], startLine, endLine };
}

function jarvisBatchSectionPathKey(evidence: JarvisStoredToolEvidence, sectionPath: string): string | undefined {
	const paths = evidence.sourcePaths ?? [];
	const keys = evidence.sourcePathKeys ?? [];
	const exactIndex = paths.indexOf(sectionPath);
	if (exactIndex >= 0 && keys[exactIndex]) return keys[exactIndex];
	const normalizedSection = normalizePathForCompareSafe(sectionPath, evidence.cwd);
	const normalizedIndex = paths.findIndex(
		(item) => normalizePathForCompareSafe(item, evidence.cwd) === normalizedSection,
	);
	if (normalizedIndex >= 0 && keys[normalizedIndex]) return keys[normalizedIndex];
	return jarvisReadSourcePathKey(sectionPath, evidence.cwd);
}

function normalizePathForCompareSafe(rawPath: string, cwd?: string): string {
	return jarvisReadSourcePathKey(rawPath, cwd) ?? rawPath.replace(/\\/g, "/").replace(/\/+$/, "").toLowerCase();
}

function buildJarvisReadSkeleton(evidence: JarvisStoredToolEvidence, output: string): string | undefined {
	const lines = splitJarvisLines(output);
	if (lines.length === 0) return undefined;
	const body = buildJarvisReadSkeletonBody(evidence.sourcePath, lines, 1);
	if (!body) return undefined;
	return [
		`[JARVIS read-skeleton] path=${markerLineValue(evidence.sourcePath)} lines=${lines.length} ref=${evidence.ref}`,
		`원문 회수: retrieve_output(ref="${evidence.ref}") - 이 파일을 수정하려면 반드시 원문을 회수해서 봐라.`,
		...body,
	].join("\n");
}

function buildJarvisReadSkeletonBody(
	sourcePath: string | undefined,
	lines: string[],
	baseLine: number,
): string[] | undefined {
	const selected = jarvisReadSkeletonSelectedLines(sourcePath, lines);
	if (selected.size === 0) return undefined;
	const maxLine = Math.max(baseLine, baseLine + lines.length - 1);
	const width = Math.max(4, String(maxLine).length);
	const body: string[] = [];
	let previous = baseLine - 1;
	for (const index of [...selected].sort((left, right) => left - right)) {
		const lineNo = baseLine + index;
		if (lineNo - previous > 1) {
			body.push(formatJarvisReadSkeletonOmitted(previous + 1, lineNo - 1, width));
		}
		body.push(`${String(lineNo).padStart(width)}| ${lines[index] ?? ""}`);
		previous = lineNo;
	}
	if (previous < maxLine) {
		body.push(formatJarvisReadSkeletonOmitted(previous + 1, maxLine, width));
	}
	return body;
}

function formatJarvisReadSkeletonOmitted(startLine: number, endLine: number, width: number): string {
	const count = Math.max(0, endLine - startLine + 1);
	return `${String(startLine).padStart(width)}..${endLine}| ...(${count} lines)`;
}

function jarvisReadSkeletonSelectedLines(sourcePath: string | undefined, lines: string[]): Set<number> {
	const ext = path.extname(String(sourcePath ?? "")).toLowerCase();
	return ext === ".py" ? jarvisPythonSkeletonSelectedLines(lines) : jarvisJsSkeletonSelectedLines(lines);
}

function jarvisJsSkeletonSelectedLines(lines: string[]): Set<number> {
	const selected = new Set<number>();
	let braceDepth = 0;
	let importContinuation = false;
	for (let index = 0; index < lines.length; index++) {
		const line = lines[index] ?? "";
		const trimmed = line.trim();
		const depthBefore = braceDepth;
		if (importContinuation) {
			selected.add(index);
			if (/[;)]\s*$/.test(trimmed) || /\bfrom\s+["'][^"']+["']\s*;?\s*$/.test(trimmed)) {
				importContinuation = false;
			}
		} else if (isJarvisJsImportOrRequireLine(line)) {
			selected.add(index);
			if (
				/^\s*import\b/.test(line) &&
				!/[;]\s*$/.test(trimmed) &&
				!/\bfrom\s+["'][^"']+["']\s*;?\s*$/.test(trimmed)
			) {
				importContinuation = true;
			}
		}
		if (isJarvisJsDeclarationLine(line, depthBefore)) selected.add(index);
		braceDepth = Math.max(0, braceDepth + jarvisBraceDelta(line));
	}
	return selected;
}

function isJarvisJsImportOrRequireLine(line: string): boolean {
	return /^\s*import\b/.test(line) || /^\s*(?:const|let|var)\s+.+\s*=\s*require\(/.test(line);
}

function isJarvisJsDeclarationLine(line: string, depthBefore: number): boolean {
	const trimmed = line.trim();
	if (!trimmed) return false;
	if (/^\s*@[\w.(-]/.test(line)) return true;
	if (/^\s*export\s+/.test(line)) return true;
	if (/^\s*(?:async\s+)?function\s+[$A-Z_a-z][\w$]*/.test(line)) return true;
	if (/^\s*(?:abstract\s+)?class\s+[$A-Z_a-z][\w$]*/.test(line)) return true;
	if (/^\s*interface\s+[$A-Z_a-z][\w$]*/.test(line)) return true;
	if (/^\s*type\s+[$A-Z_a-z][\w$]*/.test(line)) return true;
	if (/^\s*enum\s+[$A-Z_a-z][\w$]*/.test(line)) return true;
	if (depthBefore === 0 && /^\s*(?:const|let|var)\s+[$A-Z_a-z][\w$]*/.test(line)) return true;
	if (depthBefore <= 1 && isJarvisJsMethodLikeLine(trimmed)) return true;
	return false;
}

function isJarvisJsMethodLikeLine(trimmed: string): boolean {
	if (/^(?:if|for|while|switch|catch|return|function|const|let|var|else|do|try|finally)\b/.test(trimmed)) return false;
	return /^(?:(?:public|private|protected|static|async|override|readonly|get|set)\s+)*(?:constructor|[$#A-Z_a-z][\w$#]*)\s*(?:<[^>{}]+>)?\([^;{}]*\)\s*(?::\s*[^=({]+)?\s*(?:[{;]|=>)?\s*$/.test(
		trimmed,
	);
}

function jarvisBraceDelta(line: string): number {
	const withoutStrings = line.replace(/(["'`])(?:\\.|(?!\1).)*\1/g, "");
	return (withoutStrings.match(/{/g) ?? []).length - (withoutStrings.match(/}/g) ?? []).length;
}

function jarvisPythonSkeletonSelectedLines(lines: string[]): Set<number> {
	const selected = new Set<number>();
	let pendingDocstring = false;
	for (let index = 0; index < lines.length; index++) {
		const line = lines[index] ?? "";
		const trimmed = line.trim();
		const indent = line.length - line.trimStart().length;
		if (!trimmed) continue;
		if (index === 0 && /^['"]{3}/.test(trimmed)) selected.add(index);
		if (pendingDocstring) {
			if (/^[rubfRUBF]*['"]{3}/.test(trimmed)) selected.add(index);
			pendingDocstring = false;
		}
		if (/^\s*@[\w.]/.test(line)) {
			selected.add(index);
			continue;
		}
		if (/^\s*(?:async\s+def|def|class)\s+[$A-Z_a-z][\w$]*/.test(line)) {
			selected.add(index);
			pendingDocstring = true;
			continue;
		}
		if (indent === 0 && /^[$A-Z_a-z][\w$]*(?:\s*:\s*[^=]+)?\s*=/.test(line)) selected.add(index);
	}
	return selected;
}

function jarvisCompressSearchOutput(evidence: JarvisStoredToolEvidence, output: string): string | undefined {
	const lines = splitJarvisLines(output);
	const matches = parseJarvisSearchMatches(lines);
	if (matches.length === 0) return undefined;
	const fileOrder = new Map<string, number>();
	for (const match of matches) {
		if (!fileOrder.has(match.path)) fileOrder.set(match.path, match.index);
	}
	const files = [...new Set(matches.map((match) => match.path))];
	const selectedFiles = files
		.sort((left, right) => {
			const leftError = matches.some((match) => match.path === left && isJarvisImportantLine(match.raw));
			const rightError = matches.some((match) => match.path === right && isJarvisImportantLine(match.raw));
			if (leftError !== rightError) return leftError ? -1 : 1;
			return (fileOrder.get(left) ?? 0) - (fileOrder.get(right) ?? 0);
		})
		.slice(0, SEARCH_LIMITS.maxFiles);
	const selected = new Set<number>();
	const perFileDropped: string[] = [];
	for (const file of selectedFiles) {
		if (selected.size >= SEARCH_LIMITS.maxTotalMatches) break;
		const fileMatches = matches.filter((match) => match.path === file);
		const fileSelected = new Set<number>();
		const add = (match: (typeof fileMatches)[number] | undefined) => {
			if (!match) return;
			if (selected.size >= SEARCH_LIMITS.maxTotalMatches) return;
			if (fileSelected.size >= SEARCH_LIMITS.maxMatchesPerFile) return;
			fileSelected.add(match.index);
			selected.add(match.index);
		};
		add(fileMatches[0]);
		add(fileMatches[fileMatches.length - 1]);
		for (const match of fileMatches.filter((item) => isJarvisImportantLine(item.raw))) add(match);
		for (const match of fileMatches) add(match);
		const droppedInFile = fileMatches.length - fileSelected.size;
		if (droppedInFile > 0)
			perFileDropped.push(`... ${droppedInFile} more matches in ${file} omitted (ref=${evidence.ref})`);
	}
	const keptLines = matches
		.filter((match) => selected.has(match.index))
		.sort((left, right) => left.index - right.index);
	const body = [
		"",
		...keptLines.map((match) => match.raw),
		...perFileDropped,
		"",
		"summary:",
		`- files_total: ${files.length}`,
		`- files_kept: ${selectedFiles.length}`,
		`- matches_total: ${matches.length}`,
		`- matches_kept: ${keptLines.length}`,
		`- dropped: ${Math.max(0, matches.length - keptLines.length)}`,
	];
	return buildJarvisCompressedView(
		evidence,
		{
			original_lines: matches.length,
			kept_lines: keptLines.length,
			dropped_lines: Math.max(0, matches.length - keptLines.length),
		},
		body,
	);
}

function jarvisCompressListingOutput(evidence: JarvisStoredToolEvidence, output: string): string | undefined {
	const lines = splitJarvisLines(output).filter((line) => line.trim());
	const errors = lines.filter(isJarvisErrorLikeLine);
	const entries = lines.filter((line) => !isJarvisErrorLikeLine(line));
	if (entries.length === 0) return undefined;
	const selected = new Set<number>();
	for (let i = 0; i < Math.min(LISTING_LIMITS.headEntries, entries.length); i++) selected.add(i);
	for (let i = Math.max(0, entries.length - LISTING_LIMITS.tailEntries); i < entries.length; i++) selected.add(i);
	const keptEntries = entries.filter((_, index) => selected.has(index));
	const topDirs = jarvisTopBuckets(
		entries.map(jarvisListingTopDir).filter((item): item is string => !!item),
		LISTING_LIMITS.maxTopLevelDirs,
	);
	const extensions = jarvisTopBuckets(
		entries.map(jarvisListingExtension).filter((item): item is string => !!item),
		LISTING_LIMITS.maxExtensionBuckets,
	);
	const body = [
		"",
		"head:",
		...entries.slice(0, LISTING_LIMITS.headEntries),
		"",
		"tail:",
		...entries.slice(Math.max(0, entries.length - LISTING_LIMITS.tailEntries)),
		"",
		"top_dirs:",
		...topDirs.map(([name, count]) => `- ${name}: ${count}`),
		"",
		"extensions:",
		...extensions.map(([name, count]) => `- ${name}: ${count}`),
		"",
		"errors:",
		...(errors.length ? errors : ["- none"]),
		"",
		"summary:",
		`- entries_total: ${entries.length}`,
		`- entries_kept: ${keptEntries.length}`,
		`- dropped: ${Math.max(0, entries.length - keptEntries.length)}`,
	];
	return buildJarvisCompressedView(
		evidence,
		{
			original_entries: entries.length,
			kept_entries: keptEntries.length,
			dropped_entries: Math.max(0, entries.length - keptEntries.length),
		},
		body,
	);
}

function jarvisCompressLogOutput(evidence: JarvisStoredToolEvidence, output: string): string | undefined {
	const lines = splitJarvisLines(output);
	if (lines.length === 0) return undefined;
	const selected = new Set<number>();
	const add = (index: number | undefined) => {
		if (index === undefined || index < 0 || index >= lines.length) return;
		if (selected.size >= LOG_LIMITS.maxTotalLines && !isJarvisVerificationLine(lines[index])) return;
		selected.add(index);
	};
	const addContext = (index: number) => {
		for (let i = index - LOG_LIMITS.errorContextLines; i <= index + LOG_LIMITS.errorContextLines; i++) add(i);
	};
	const errorIndices = lines
		.map((line, index) => (isJarvisErrorLikeLine(line) ? index : -1))
		.filter((index) => index >= 0);
	const warningIndices = lines
		.map((line, index) => (isJarvisWarningLine(line) ? index : -1))
		.filter((index) => index >= 0);
	const verificationIndices = lines
		.map((line, index) => (isJarvisVerificationLine(line) ? index : -1))
		.filter((index) => index >= 0);
	for (const index of errorIndices.slice(0, LOG_LIMITS.maxErrors)) addContext(index);
	for (const index of verificationIndices) add(index);
	let stackCount = 0;
	for (let i = 0; i < lines.length && stackCount < LOG_LIMITS.maxStackLines; i++) {
		if (!isJarvisStackLine(lines[i])) continue;
		add(i);
		stackCount += 1;
	}
	for (const index of warningIndices.slice(0, LOG_LIMITS.maxWarnings)) addContext(index);
	for (let i = 0; i < Math.min(LOG_LIMITS.firstLines, lines.length); i++) add(i);
	for (let i = Math.max(0, lines.length - LOG_LIMITS.lastLines); i < lines.length; i++) add(i);
	for (let i = 0; i < lines.length; i++) {
		if (!isJarvisLogSummaryLine(lines[i])) continue;
		add(i);
	}
	const keptIndices = [...selected].sort((left, right) => left - right);
	const omittedCounts = jarvisOmittedLogCounts(lines, selected);
	const body = [
		"",
		"kept:",
		...keptIndices.map((index) => lines[index]),
		"",
		"omitted:",
		...Object.entries(omittedCounts).map(([kind, count]) => `- ${kind}: ${count}`),
	];
	return buildJarvisCompressedView(
		evidence,
		{
			original_lines: lines.length,
			kept_lines: keptIndices.length,
			dropped_lines: Math.max(0, lines.length - keptIndices.length),
		},
		body,
	);
}

function buildJarvisCompressedView(
	evidence: JarvisStoredToolEvidence,
	counts: JarvisCompressedMarkerCounts,
	bodyLines: string[],
): string {
	const marker = buildJarvisCompressedMarker(evidence.kind, evidence.ref ?? "", counts, {
		command: evidence.command,
		cwd: evidence.cwd,
		exit_code: evidence.exitCode,
	});
	const markerLines = marker.split("\n");
	markerLines.splice(Math.max(0, markerLines.length - 1), 0, ...bodyLines);
	return markerLines.join("\n");
}

function jarvisToolResultTextTarget(
	message: Record<string, unknown>,
):
	| { field: "output"; text: string }
	| { field: "content"; text: string }
	| { field: "content_text_block"; text: string; index: number }
	| { field: "content_content_block"; text: string; index: number }
	| undefined {
	if (typeof message.output === "string") return { field: "output", text: message.output };
	if (typeof message.content === "string") return { field: "content", text: message.content };
	if (Array.isArray(message.content)) {
		for (let index = 0; index < message.content.length; index++) {
			const block = message.content[index];
			if (!block || typeof block !== "object") continue;
			const record = block as Record<string, unknown>;
			if (typeof record.text === "string") return { field: "content_text_block", text: record.text, index };
			if (typeof record.content === "string") return { field: "content_content_block", text: record.content, index };
		}
	}
	return undefined;
}

function applyJarvisToolResultText(
	message: Record<string, unknown>,
	target: NonNullable<ReturnType<typeof jarvisToolResultTextTarget>>,
	text: string,
): Record<string, unknown> {
	if (target.field === "output") return { ...message, output: text };
	if (target.field === "content") return { ...message, content: text };
	const content = Array.isArray(message.content) ? message.content : [];
	const nextContent = content.map((block, index) => {
		if (index !== target.index || !block || typeof block !== "object") return block;
		const record = block as Record<string, unknown>;
		if (target.field === "content_text_block") return { ...record, text };
		return { ...record, content: text };
	});
	return { ...message, content: nextContent };
}

function jarvisToolResultKeyFromMessage(message: Record<string, unknown>): string | undefined {
	const callId = payloadCallId(message);
	if (callId) return normalizeJarvisToolCallId(callId);
	const name = message.name ?? message.tool_name;
	return typeof name === "string" && name.trim() ? `tool:${name}` : undefined;
}

// pi's internal toolCallId can carry a "|"-suffixed segment, while provider
// payload serializers send only the part before "|" (see
// openai-responses-shared.ts `msg.toolCallId.split("|")`). Evidence map keys
// must normalize the same way on both the store and lookup sides, or wire
// compression never matches (R1-3 live finding, 2026-06-07).
function normalizeJarvisToolCallId(toolCallId: string): string {
	const [head] = toolCallId.split("|");
	return (head ?? toolCallId).trim();
}

function jarvisEvidenceToolResultKey(toolCallId: unknown, toolName: unknown): string {
	const id = typeof toolCallId === "string" ? normalizeJarvisToolCallId(toolCallId) : "";
	if (id) return id;
	return `tool:${String(toolName ?? "tool")}`;
}

function splitJarvisLines(text: string): string[] {
	return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
}

function parseJarvisSearchMatches(lines: string[]): Array<{ index: number; raw: string; path: string }> {
	const matches: Array<{ index: number; raw: string; path: string }> = [];
	for (let index = 0; index < lines.length; index++) {
		const line = lines[index];
		const match = /^(.+):(\d+):(.*)$/.exec(line);
		if (!match?.[1]) continue;
		matches.push({ index, raw: line, path: match[1] });
	}
	return matches;
}

function isSearchToolOrCommand(tool: string, command: string): boolean {
	return /\b(rg|ripgrep|grep|select-string)\b/i.test(`${tool} ${command}`);
}

function isListingToolOrCommand(tool: string, command: string): boolean {
	return /\b(rg\s+--files|find|ls|dir|get-childitem)\b/i.test(`${tool} ${command}`);
}

function isLogToolOrCommand(tool: string, command: string): boolean {
	return /\b(npm|pnpm|yarn|pytest|vitest|tsc|tsgo|biome|webpack|vite|gradle|mvn|cargo|dotnet|node\s+--check)\b/i.test(
		`${tool} ${command}`,
	);
}

function isSingleFileCatCommand(command: string): boolean {
	const clean = command.trim();
	if (!clean) return false;
	return /^(?:cat|type|get-content)(?:\s+--?[A-Za-z0-9-]+)*\s+["']?[^|&;*?]+["']?\s*$/i.test(clean);
}

function isJarvisPathLikeListingLine(line: string): boolean {
	const clean = line.trim();
	if (!clean) return false;
	if (/\s/.test(clean) && !/[\\/]/.test(clean)) return false;
	return /(?:[\\/]|^\.[\\/]|^[A-Za-z]:[\\/]|[A-Za-z0-9_-]+\.[A-Za-z0-9]{1,8}$)/.test(clean);
}

function isJarvisImportantLine(line: string): boolean {
	return isJarvisErrorLikeLine(line) || isJarvisWarningLine(line) || isJarvisVerificationLine(line);
}

function isJarvisErrorLikeLine(line: string): boolean {
	return /\b(error|failed|failure|fatal|panic|exception|traceback|assertionerror|timeout)\b/i.test(line);
}

function isJarvisWarningLine(line: string): boolean {
	return /\b(warn|warning|deprecated)\b/i.test(line);
}

function isJarvisVerificationLine(line: string): boolean {
	return /\b(node\s+--check\b.*\bOK\b|\b\d+\s+passed\b|\btests?\s+passed\b|\bcheck\s+passed\b|\bbuild\s+passed\b|\bPASS\b)/i.test(
		line,
	);
}

function isJarvisStackLine(line: string): boolean {
	return /^\s+(?:at\s+|File\s+"|from\s+)/.test(line) || /\bTraceback \(most recent call last\):/.test(line);
}

function isJarvisLogSummaryLine(line: string): boolean {
	return (
		isJarvisVerificationLine(line) || /\b(tests?|failures?|passed|failed|duration|summary|exit code)\b/i.test(line)
	);
}

function isJarvisLogSignalLine(line: string): boolean {
	return isJarvisImportantLine(line) || isJarvisStackLine(line) || /\b(pass|fail|test|suite|exit code)\b/i.test(line);
}

function jarvisListingTopDir(line: string): string | undefined {
	const clean = line.trim().replace(/^[.][\\/]/, "");
	const parts = clean.split(/[\\/]+/).filter(Boolean);
	return parts.length > 1 ? parts[0] : undefined;
}

function jarvisListingExtension(line: string): string | undefined {
	const base =
		line
			.trim()
			.split(/[\\/]+/)
			.pop() ?? "";
	const match = /(\.[A-Za-z0-9]{1,12})$/.exec(base);
	return match?.[1]?.toLowerCase();
}

function jarvisTopBuckets(values: string[], max: number): Array<[string, number]> {
	const counts = new Map<string, number>();
	for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
	return [...counts.entries()]
		.sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
		.slice(0, max);
}

function jarvisOmittedLogCounts(lines: string[], selected: Set<number>): Record<string, number> {
	const counts = { error: 0, warning: 0, debug: 0, info: 0 };
	for (let index = 0; index < lines.length; index++) {
		if (selected.has(index)) continue;
		const line = lines[index];
		if (isJarvisErrorLikeLine(line)) counts.error += 1;
		else if (isJarvisWarningLine(line)) counts.warning += 1;
		else if (/\b(debug|trace)\b/i.test(line)) counts.debug += 1;
		else counts.info += 1;
	}
	return counts;
}

function limitRetrieveOutputContent(content: string): {
	text: string;
	truncated: boolean;
	linesReturned: number;
} {
	const lineChunks = splitLinesKeepEnd(content);
	let kept = lineChunks.slice(0, RETRIEVE_OUTPUT_MAX_LINES).join("");
	let truncated = lineChunks.length > RETRIEVE_OUTPUT_MAX_LINES;
	if (Buffer.byteLength(kept, "utf8") > RETRIEVE_OUTPUT_MAX_BYTES) {
		kept = truncateUtf8Text(kept, RETRIEVE_OUTPUT_MAX_BYTES);
		truncated = true;
	}
	const linesReturned = splitLinesKeepEnd(kept).length;
	if (!truncated) return { text: kept, truncated, linesReturned };
	const notice = `\n\n[retrieve_output truncated: showing at most ${RETRIEVE_OUTPUT_MAX_LINES} lines or ${RETRIEVE_OUTPUT_MAX_BYTES} bytes. Specify start_line/end_line and call retrieve_output again.]`;
	return { text: `${kept}${notice}`, truncated, linesReturned };
}

function splitLinesKeepEnd(text: string): string[] {
	if (!text) return [];
	const matches = text.match(/.*(?:\r\n|\n|\r|$)/g) ?? [];
	if (matches[matches.length - 1] === "") matches.pop();
	return matches;
}

function truncateUtf8Text(text: string, maxBytes: number): string {
	if (Buffer.byteLength(text, "utf8") <= maxBytes) return text;
	let end = text.length;
	while (end > 0 && Buffer.byteLength(text.slice(0, end), "utf8") > maxBytes) {
		end -= 1;
	}
	return text.slice(0, end);
}

async function postSidecar<T = unknown>(
	path: string,
	body?: unknown,
	method = "POST",
	timeoutMs = 45000,
): Promise<T | undefined> {
	for (const baseUrl of sidecarUrlCandidates()) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);
		try {
			const response = await fetch(`${baseUrl}${path}`, {
				method,
				headers: body === undefined ? undefined : { "content-type": "application/json" },
				body: body === undefined ? undefined : JSON.stringify(body),
				signal: controller.signal,
			});
			if (!response.ok) {
				const errorBody = await response.text().catch(() => "");
				return {
					ok: false,
					error: `JARVIS sidecar HTTP ${response.status}`,
					body: errorBody.slice(0, 1000),
				} as T;
			}
			return (await response.json()) as T;
		} catch {
			// Try the next advertised sidecar URL. The launcher writes the current
			// runtime URL, while env/default remain compatibility fallbacks.
		} finally {
			clearTimeout(timer);
		}
	}
	sidecarHealthy = false;
	return undefined;
}

function isOkSidecarResponse(data: unknown): boolean {
	return typeof data === "object" && data !== null && (data as { ok?: unknown }).ok !== false;
}

function loadAutoPromptState(flagValue: boolean | string | undefined): AutoPromptState | undefined {
	if (typeof flagValue !== "string" || !flagValue.trim()) return undefined;
	const promptsFile = path.resolve(flagValue);
	if (!fs.existsSync(promptsFile)) return undefined;

	const prompts = fs
		.readFileSync(promptsFile, "utf-8")
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter((line) => line.length > 0);
	if (prompts.length === 0) return undefined;

	const progressFile = path.join(path.dirname(promptsFile), ".auto_progress.json");
	let idx = 0;
	if (fs.existsSync(progressFile)) {
		try {
			const parsed = JSON.parse(fs.readFileSync(progressFile, "utf-8")) as { idx?: unknown };
			if (typeof parsed.idx === "number" && Number.isFinite(parsed.idx)) {
				idx = Math.max(0, Math.min(prompts.length, Math.floor(parsed.idx)));
			}
		} catch {
			idx = 0;
		}
	}

	return {
		promptsFile,
		progressFile,
		prompts,
		idx,
		total: prompts.length,
	};
}

function persistAutoPromptState(state: AutoPromptState): void {
	fs.mkdirSync(path.dirname(state.progressFile), { recursive: true });
	fs.writeFileSync(
		state.progressFile,
		JSON.stringify({ idx: state.idx, total: state.total, ts: new Date().toISOString() }),
		"utf-8",
	);
}

async function fireAutoPrompt(state: AutoPromptState, ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	if (state.idx >= state.total) {
		clearAutoPromptWatchdog();
		setAutoPromptStatus(ctx, undefined);
		try {
			ctx.shutdown();
		} catch {
			/* ctx may be stale */
		}
		return;
	}
	setAutoPromptStatus(ctx, state);
	console.error(`=== [${state.idx + 1}/${state.total}] auto-prompts start ===`);
	if (AUTO_PROMPT_DELAY_MS > 0) {
		console.error(`=== auto-prompts delay ${AUTO_PROMPT_DELAY_MS}ms ===`);
		await new Promise((resolve) => setTimeout(resolve, AUTO_PROMPT_DELAY_MS));
	}
	await Promise.resolve();
	persistAutoPromptState(state);
	clearAutoPromptWatchdog();
	try {
		pi.sendUserMessage(state.prompts[state.idx]);
	} catch (err) {
		console.error(`[jarvis:auto-prompts] sendUserMessage failed: ${String(err)}`);
		return;
	}
	armAutoPromptWatchdog(state, ctx, pi, state.idx, state.prompts[state.idx]);
}

async function handleAutoPromptTurn(state: AutoPromptState, ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	if (autoPromptWatchdog?.aborting) {
		return;
	}
	clearAutoPromptWatchdog();
	state.idx += 1;
	persistAutoPromptState(state);
	if (state.idx >= state.total) {
		setAutoPromptStatus(ctx, undefined);
		console.error(`=== ${state.total}/${state.total} DONE ===`);
		try {
			ctx.shutdown();
		} catch {
			/* ctx may be stale */
		}
		return;
	}
	setAutoPromptStatus(ctx, state);
	if (state.idx % 100 === 0) {
		console.error(`=== MILESTONE ${state.idx}/${state.total} ===`);
	} else {
		console.error(`=== [${state.idx + 1}/${state.total}] ===`);
	}

	const resetEvery = Number.parseInt(process.env.JARVIS_AUTO_RESET_EVERY ?? "1", 10);
	let sessionResetDone = false;
	if (resetEvery > 0 && state.idx > 0 && state.idx % resetEvery === 0 && "newSession" in ctx) {
		type NewSessionFn = (options?: {
			withSession?: (freshCtx: { sendUserMessage: (c: string) => Promise<void> }) => Promise<void>;
		}) => Promise<{ cancelled: boolean }>;
		const maybeNewSession = (ctx as ExtensionContext & { newSession?: NewSessionFn }).newSession;
		if (typeof maybeNewSession === "function") {
			try {
				const prompt = state.prompts[state.idx];
				const { cancelled } = await maybeNewSession({
					withSession: async (freshCtx) => {
						await freshCtx.sendUserMessage(prompt);
						persistAutoPromptState(state);
						sessionResetDone = true;
					},
				});
				if (cancelled) {
					console.error(`[jarvis:auto-prompts] session reset cancelled`);
				}
			} catch (error) {
				console.error(`[jarvis:auto-prompts] session reset failed: ${String(error)}`);
			}
		}
	}

	// If the session was reset and the next prompt was already sent via
	// withSession, skip the pi.sendUserMessage below (pi is now stale).
	if (sessionResetDone) return;
	await Promise.resolve();
	persistAutoPromptState(state);
	clearAutoPromptWatchdog();
	try {
		pi.sendUserMessage(state.prompts[state.idx]);
	} catch (err) {
		console.error(`[jarvis:auto-prompts] sendUserMessage failed: ${String(err)}`);
		return;
	}
	armAutoPromptWatchdog(state, ctx, pi, state.idx, state.prompts[state.idx]);
}

function clearAutoPromptWatchdog(): void {
	if (autoPromptWatchdog?.timer) {
		clearTimeout(autoPromptWatchdog.timer);
	}
	autoPromptWatchdog = undefined;
}

function armAutoPromptWatchdog(
	state: AutoPromptState,
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	promptIndex: number,
	prompt: string,
): void {
	if (AUTO_PROMPT_STALL_TIMEOUT_MS <= 0) return;
	clearAutoPromptWatchdog();
	autoPromptWatchdog = {
		promptIndex,
		prompt,
		retryCount: 0,
		aborting: false,
	};
	autoPromptWatchdog.timer = setTimeout(() => {
		void handleAutoPromptStall(state, ctx, pi);
	}, AUTO_PROMPT_STALL_TIMEOUT_MS);
}

async function handleAutoPromptStall(state: AutoPromptState, ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	const watchdog = autoPromptWatchdog;
	if (!watchdog) return;
	if (watchdog.promptIndex !== state.idx) return;
	if (watchdog.retryCount >= 1) {
		console.error(
			`=== auto-prompts stalled for ${AUTO_PROMPT_STALL_TIMEOUT_MS}ms; giving up on prompt ${watchdog.promptIndex + 1}/${state.total} ===`,
		);
		clearAutoPromptWatchdog();
		return;
	}

	watchdog.retryCount += 1;
	watchdog.aborting = true;
	console.error(
		`=== auto-prompts stalled for ${AUTO_PROMPT_STALL_TIMEOUT_MS}ms; aborting and retrying prompt ${watchdog.promptIndex + 1}/${state.total} ===`,
	);
	try {
		await ctx.abort();
	} catch {
		/* ctx may be stale */
	}
	try {
		pi.sendUserMessage(watchdog.prompt);
	} catch (err) {
		console.error(`[jarvis:auto-prompts] retry sendUserMessage failed: ${String(err)}`);
		clearAutoPromptWatchdog();
		return;
	}
	armAutoPromptWatchdog(state, ctx, pi, watchdog.promptIndex, watchdog.prompt);
}

export function __resetJarvisJlcForTests(): void {
	clearAutoPromptWatchdog();
	clearInterruptInputCheckpointHook();
	activeProjectPath = undefined;
	activeCodePath = undefined;
	activeProjectId = undefined;
	lastContextResponse = undefined;
	lastInjectedContextMode = undefined;
	lastUserMessage = "";
	transientSystemDirective = "";
	toolEvents = [];
	checkpointToolEvents = [];
	lastAssistantPartialText = "";
	lastAssistantObservedModeMarker = undefined;
	interruptCheckpointSavedThisTurn = false;
	turnCheckpointScope = undefined;
	sidecarHealthy = false;
	setEffectiveRoute("chat");
	deepdiveThinkingPreference = undefined;
	deepdiveThinkingPreferenceLoaded = false;
	coldStartNoticeShown = false;
	startupContextWarmupFinished = false;
	startupContextWarmupPromise = undefined;
	setupRequired = false;
	lastTurnPromptSnapshot = undefined;
	providerCallCountThisTurn = 0;
	displayedInputTokensThisTurn = 0;
	displayedCallInputTokens = 0;
	displayedOutputTokensThisTurn = 0;
	completedOutputTokensThisTurn = 0;
	resetSubturnLogState();
	resetJarvisTurnChoreographyState();
	projectCache = [];
	projectCacheLoaded = false;
	pendingProjectCreate = undefined;
}

export function latestUserText(messages: AgentMessage[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role === "user") return messageContentToText(message.content);
	}
	return "";
}

function latestUserTurnKey(messages: AgentMessage[], userText: string): string {
	let userCount = 0;
	let latestUserIndex = -1;
	let latestTimestamp = "";
	for (let index = 0; index < messages.length; index++) {
		const message = messages[index];
		if (message.role !== "user") continue;
		userCount++;
		latestUserIndex = index;
		const timestamp = (message as { timestamp?: unknown }).timestamp;
		latestTimestamp = typeof timestamp === "string" || typeof timestamp === "number" ? String(timestamp) : "";
	}
	return `${userCount}:${latestUserIndex}:${latestTimestamp}:${userText}`;
}

export function latestAssistantText(messages: AgentMessage[]): string {
	const message = latestAssistantMessage(messages);
	return message ? contentToText(message.content) : "";
}

export function latestAssistantMessage(messages: AgentMessage[]): AssistantMessage | undefined {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role === "assistant") return message as AssistantMessage;
	}
	return undefined;
}

export function stripLeadingModeMarkerText(text: string, allowPartial = false): string {
	const withoutCompleteMarker = text.replace(MODE_MARKER_RE, "");
	if (withoutCompleteMarker !== text || !allowPartial) return withoutCompleteMarker;

	const leadingWhitespace = text.match(/^\s*/)?.[0] ?? "";
	const body = text.slice(leadingWhitespace.length);
	const lowerBody = body.toLowerCase();
	const isPartialMarker = MODE_MARKER_PREFIXES.some((marker) => marker.toLowerCase().startsWith(lowerBody));
	return isPartialMarker ? "" : text;
}

function sanitizeAssistantText(text: string, allowPartial = false): string {
	let sanitized = stripLeadingModeMarkerText(text, allowPartial);
	sanitized = sanitized.replace(MODE_MARKER_ANY_RE, "");

	const lines = sanitized.split(/\r?\n/);
	const kept: string[] = [];
	let skippingInternalBlock = false;

	for (const line of lines) {
		const trimmed = line.trim();
		if (trimmed === "") {
			if (!skippingInternalBlock) {
				kept.push("");
			}
			skippingInternalBlock = false;
			continue;
		}

		if (INTERNAL_ASSISTANT_BLOCK_START_RE.test(line)) {
			skippingInternalBlock = true;
			continue;
		}

		if (INTERNAL_ASSISTANT_LINE_RE.test(line)) {
			continue;
		}

		if (skippingInternalBlock) {
			continue;
		}

		kept.push(line);
	}

	return kept
		.join("\n")
		.replace(/[ \t]+\n/g, "\n")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

function sanitizeAssistantMessage(message: AssistantMessage, allowPartial: boolean): AssistantMessage {
	if (!Array.isArray(message.content)) return message;
	let stripped = false;
	const content = message.content.map((part) => {
		if (stripped || part.type !== "text") return part;
		const nextText = sanitizeAssistantText(part.text, allowPartial);
		if (nextText === part.text) return part;
		stripped = true;
		return { ...part, text: nextText };
	});
	return stripped ? { ...message, content } : message;
}

function sanitizeAssistantMessageInPlace(message: AssistantMessage, allowPartial: boolean): boolean {
	const stripped = sanitizeAssistantMessage(message, allowPartial);
	if (stripped === message) return false;
	Object.assign(message, stripped);
	return true;
}

export function injectMemoryIntoLatestUser(messages: AgentMessage[], memory: string): AgentMessage[] {
	let injected = false;
	const next = [...messages];
	for (let i = next.length - 1; i >= 0; i--) {
		const message = next[i];
		if (message.role !== "user") continue;
		const existingText = stripJarvisMemoryBlock(messageContentToText(message.content));
		next[i] = {
			...message,
			content: [
				{
					type: "text",
					text: `<jarvis_memory>\n${memory}\n</jarvis_memory>\n\n${existingText}`,
				},
			],
		};
		injected = true;
		break;
	}
	return injected ? next : messages;
}

function stripJarvisMemoryBlock(text: string): string {
	return text.replace(/^<jarvis_memory>[\s\S]*?<\/jarvis_memory>\s*/i, "").trimStart();
}

function messageContentToText(content: AgentMessage extends { content: infer C } ? C : unknown): string {
	if (typeof content === "string") return content;
	if (Array.isArray(content)) return contentToText(content);
	return "";
}

function formatRoleStatus(role?: SidecarRoleStatus): string {
	if (!role) return "";
	if (role.provider && role.model) {
		return `${role.provider}/${role.model}`;
	}
	return role.display ?? role.configured ?? "";
}

function contentToText(content: unknown): string {
	if (!Array.isArray(content)) return "";
	return content
		.filter(
			(part): part is Record<string, unknown> =>
				typeof part === "object" &&
				part !== null &&
				["text", "input_text", "output_text"].includes(String((part as { type?: unknown }).type ?? "")),
		)
		.map((part) => String(part.text ?? ""))
		.join("\n");
}

function buildTurnLlmMeta(
	assistantMessage: AssistantMessage | undefined,
	ctx: ExtensionContext,
	contextResponse?: SidecarContextResponse,
	promptSnapshot?: TurnPromptSnapshot,
	turnToolEvents: ToolEventSummary[] = [],
	turnUsage?: TurnUsageSummary,
): Record<string, unknown> {
	const usage = turnUsage ?? assistantMessage?.usage;
	let contextUsage: ReturnType<ExtensionContext["getContextUsage"]> | undefined;
	let modelProvider: string | undefined;
	let modelId: string | undefined;
	try {
		contextUsage = ctx.getContextUsage();
		modelProvider = ctx.model?.provider;
		modelId = ctx.model?.id;
	} catch {
		/* ctx stale — use fallbacks from assistantMessage */
	}
	const cacheReport = buildJarvisCacheReport(usage as TurnUsageSummary | undefined, {
		provider: assistantMessage?.provider ?? modelProvider,
		api: assistantMessage?.api,
		model: assistantMessage?.responseModel ?? assistantMessage?.model ?? modelId,
	});
	const chatSeconds =
		lastTurnStartedAtMs !== undefined ? Math.max(0, (Date.now() - lastTurnStartedAtMs) / 1000) : undefined;
	const meta: Record<string, unknown> = {
		provider: assistantMessage?.provider ?? modelProvider,
		model: assistantMessage?.responseModel ?? assistantMessage?.model ?? modelId,
		response_model: assistantMessage?.responseModel,
		response_id: assistantMessage?.responseId,
		api: assistantMessage?.api,
		tokens_in: usage?.input,
		tokens_out: usage?.output,
		cache_read_tokens: usage?.cacheRead,
		cache_write_tokens: usage?.cacheWrite,
		total_tokens: usage?.totalTokens,
		cost_usd: usage?.cost?.total,
		chat_seconds: chatSeconds,
		context_tokens: contextUsage?.tokens ?? undefined,
		context_window: contextUsage?.contextWindow,
		context_percent: contextUsage?.percent,
	};
	const promptBreakdown = buildPromptBreakdown(promptSnapshot, turnToolEvents);
	if (Object.keys(promptBreakdown).length > 0) {
		meta.prompt_breakdown = promptBreakdown;
	}
	if (usage) {
		meta.cache_meter = cacheReport.cache_meter;
		meta.cache_hit_pct = cacheReport.cache_hit_pct;
		meta.usage = usage;
	}
	if (contextResponse) {
		meta.prompt_context = Object.fromEntries(
			Object.entries({
				context_tokens: contextResponse.context_tokens,
				jhb_tokens: contextResponse.jhb_tokens,
				project_tokens: contextResponse.project_tokens,
				recall_tokens: contextResponse.recall_tokens,
				memory_mode: contextResponse.memory_mode,
				active_project_path: contextResponse.active_project_path,
				project_id: contextResponse.project_id,
				project_name: contextResponse.project_name,
			}).filter(([, value]) => value !== undefined),
		);
	}
	return Object.fromEntries(Object.entries(meta).filter(([, value]) => value !== undefined));
}

function recordFooterMeterEntry(
	pi: ExtensionAPI,
	assistantMessage: AssistantMessage | undefined,
	contextResponse?: SidecarContextResponse,
	turnUsage?: TurnUsageSummary,
): void {
	const usage = turnUsage ?? assistantMessage?.usage;
	if (!usage) return;
	const appendEntry = (pi as { appendEntry?: <T>(type: string, entry: T) => void }).appendEntry;
	if (typeof appendEntry !== "function") return;
	const chatIn = Math.max(0, usage.input ?? 0);
	const chatOut = Math.max(0, usage.output ?? 0);
	const chatTotal = Math.max(
		0,
		usage.totalTokens ?? chatIn + chatOut + Math.max(0, usage.cacheRead ?? 0) + Math.max(0, usage.cacheWrite ?? 0),
	);
	appendEntry<FooterMeterEntry>(FOOTER_METER_ENTRY_TYPE, {
		chat_in: chatIn,
		chat_out: chatOut,
		chat_total: chatTotal,
		jhb_tokens: Math.max(0, contextResponse?.jhb_tokens ?? 0),
	});
	lastTurnStartedAtMs = undefined;
}

function recordFooterMeterReset(pi: ExtensionAPI): void {
	const appendEntry = (pi as { appendEntry?: <T>(type: string, entry: T) => void }).appendEntry;
	if (typeof appendEntry !== "function") return;
	appendEntry(FOOTER_METER_RESET_ENTRY_TYPE, { started_at: new Date().toISOString() });
}

function buildPromptBreakdown(
	promptSnapshot: TurnPromptSnapshot | undefined,
	turnToolEvents: ToolEventSummary[],
): Record<string, number> {
	if (!promptSnapshot) return {};
	const latestUserIndex = findLatestUserIndex(promptSnapshot.messages);
	const history = promptSnapshot.messages.reduce((sum, message, index) => {
		if (index === latestUserIndex) return sum;
		return sum + estimateMessageTokenCount(message);
	}, 0);
	const modePrompt = estimateTextTokenCount(promptSnapshot.modePromptText);
	const overlay = estimateTextTokenCount(promptSnapshot.overlayText);
	const existingPrompt = estimateTextTokenCount(promptSnapshot.existingPromptText);
	const systemPromptTotal = estimateTextTokenCount(promptSnapshot.systemPrompt);
	const promptCtx = estimateTextTokenCount(promptSnapshot.promptContextText);
	const user = estimateTextTokenCount(promptSnapshot.userText);
	const toolLoop = estimateToolEventTokens(turnToolEvents);
	return Object.fromEntries(
		Object.entries({
			system_total: systemPromptTotal,
			mode_prompt: modePrompt,
			overlay,
			existing_prompt: existingPrompt,
			history,
			user,
			prompt_ctx: promptCtx,
			tool_loop: toolLoop,
		}).filter(([, value]) => value > 0),
	);
}

function buildPromptContextText(memory: string | undefined): string {
	if (!memory?.trim()) return "";
	return `<jarvis_memory>\n${memory}\n</jarvis_memory>\n\n`;
}

function summarizeProviderPayload(payload: unknown): string {
	if (!payload || typeof payload !== "object") return `payload_type=${typeof payload}`;
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	const systemPrompt =
		typeof record.systemPrompt === "string"
			? record.systemPrompt
			: typeof record.system === "string"
				? record.system
				: typeof record.instructions === "string"
					? record.instructions
					: "";
	const toolSchemaTokens = tools ? estimateTextTokenCount(JSON.stringify(tools)) : 0;
	const toolSummary = summarizeToolSchema(tools);
	if (!messages && !input) {
		return `keys=${Object.keys(record).join(",") || "(none)"} system_prompt~=${estimateTextTokenCount(systemPrompt)} tools=${tools?.length ?? 0} tool_schema~=${toolSchemaTokens}${toolSummary ? ` tool_names=${toolSummary}` : ""}`;
	}
	const payloadMessages = messages ?? input ?? [];
	const roleCounts = new Map<string, number>();
	const roleTokens = new Map<string, number>();
	const typeCounts = new Map<string, number>();
	const typeTokens = new Map<string, number>();
	const fieldTokens = new Map<string, number>();
	const functionNamesByCallId = new Map<string, string>();
	const topItems: Array<{ label: string; tokens: number }> = [];
	let msgTokens = 0;
	for (const message of payloadMessages) {
		const role = String(message.role ?? "unknown");
		const type = String(message.type ?? "message");
		if (type === "function_call") {
			const callId = message.call_id;
			const name = message.name;
			if (typeof callId === "string" && typeof name === "string") {
				functionNamesByCallId.set(callId, name);
			}
		}
		const messageTokens = estimateUnknownMessageTokens(message);
		roleCounts.set(role, (roleCounts.get(role) ?? 0) + 1);
		roleTokens.set(role, (roleTokens.get(role) ?? 0) + messageTokens);
		typeCounts.set(type, (typeCounts.get(type) ?? 0) + 1);
		typeTokens.set(type, (typeTokens.get(type) ?? 0) + messageTokens);
		for (const [field, tokens] of estimateMessageFieldTokens(message)) {
			fieldTokens.set(field, (fieldTokens.get(field) ?? 0) + tokens);
		}
		if (messageTokens > 0) {
			topItems.push({ label: summarizePayloadItem(message, functionNamesByCallId), tokens: messageTokens });
		}
		msgTokens += messageTokens;
	}
	const roles = Array.from(roleCounts.entries())
		.map(([role, count]) => `${role}=${count}`)
		.join(" ");
	const roleTokenSummary = Array.from(roleTokens.entries())
		.map(([role, count]) => `${role}~=${count}`)
		.join(" ");
	const typeSummary = Array.from(typeCounts.entries())
		.map(([type, count]) => `${type}=${count}`)
		.join(" ");
	const typeTokenSummary = Array.from(typeTokens.entries())
		.map(([type, count]) => `${type}~=${count}`)
		.join(" ");
	const fieldTokenSummary = Array.from(fieldTokens.entries())
		.filter(([, count]) => count > 0)
		.map(([field, count]) => `${field}~=${count}`)
		.join(" ");
	const topItemSummary = topItems
		.sort((left, right) => right.tokens - left.tokens)
		.slice(0, 5)
		.map((item) => `${item.label}:${item.tokens}`)
		.join(",");
	const messageKey = messages ? "messages" : "input";
	return `${messageKey}=${payloadMessages.length} ${roles} role_tokens[${roleTokenSummary}] types[${typeSummary}] type_tokens[${typeTokenSummary}] fields[${fieldTokenSummary}] top[${topItemSummary}] message_tokens~=${msgTokens} system_prompt~=${estimateTextTokenCount(systemPrompt)} tools=${tools?.length ?? 0} tool_schema~=${toolSchemaTokens}${toolSummary ? ` tool_names=${toolSummary}` : ""} keys=${Object.keys(record).join(",")}`;
}

function providerPayloadTracePreview(payload: unknown): string {
	if (PAYLOAD_TRACE_HISTORY_ENABLED) return providerPayloadHistoryPreview(payload);
	return providerPayloadRawPreview(payload);
}

function providerPayloadHistoryPreview(payload: unknown): string {
	try {
		const maxChars = Number.isFinite(PAYLOAD_TRACE_HISTORY_MAX_CHARS)
			? Math.max(1000, PAYLOAD_TRACE_HISTORY_MAX_CHARS)
			: 30000;
		if (!payload || typeof payload !== "object") return truncateForCheckpoint(String(payload ?? ""), maxChars);
		const record = payload as Record<string, unknown>;
		const messages = Array.isArray(record.messages) ? record.messages : undefined;
		const input = Array.isArray(record.input) ? record.input : undefined;
		const history = messages ?? input ?? [];
		const source = messages ? "messages" : input ? "input" : "none";
		const lines = [
			`message_history source=${source} count=${history.length}`,
			`instructions_tokens~=${typeof record.instructions === "string" ? estimateTextTokenCount(record.instructions) : 0}`,
		];
		history.forEach((message, index) => {
			lines.push(formatPayloadHistoryMessage(message, index));
		});
		return truncateForCheckpoint(lines.join("\n\n"), maxChars);
	} catch (error) {
		return `message_history_preview_error=${error instanceof Error ? error.message : String(error)}`;
	}
}

function formatPayloadHistoryMessage(message: unknown, index: number): string {
	if (!message || typeof message !== "object") {
		return `#${index} primitive\n${truncateForCheckpoint(String(message ?? ""), 2000)}`;
	}
	const record = message as Record<string, unknown>;
	const type = typeof record.type === "string" ? record.type : "message";
	const role = typeof record.role === "string" ? record.role : "";
	const name = typeof record.name === "string" ? record.name : "";
	const callId = typeof record.call_id === "string" ? record.call_id : "";
	const id = typeof record.id === "string" ? record.id : "";
	const header = [
		`#${index}`,
		role ? `role=${role}` : "",
		`type=${type}`,
		name ? `name=${name}` : "",
		callId ? `call_id=${callId}` : "",
		id ? `id=${id}` : "",
	]
		.filter(Boolean)
		.join(" ");
	const body = payloadHistoryMessageBody(record);
	return body ? `${header}\n${body}` : header;
}

function payloadHistoryMessageBody(record: Record<string, unknown>): string {
	const fields: string[] = [];
	for (const key of ["content", "output", "arguments", "summary"]) {
		if (!(key in record)) continue;
		const text = payloadHistoryValueText(record[key]);
		if (text.trim()) fields.push(`${key}:\n${truncateForCheckpoint(text, 4000)}`);
	}
	if (fields.length > 0) return fields.join("\n");
	const compact = JSON.stringify(record);
	return compact ? `json:\n${truncateForCheckpoint(compact, 4000)}` : "";
}

function payloadHistoryValueText(value: unknown): string {
	if (typeof value === "string") return value;
	if (Array.isArray(value)) {
		const parts = value.map((item) => {
			if (typeof item === "string") return item;
			if (!item || typeof item !== "object") return String(item ?? "");
			const record = item as Record<string, unknown>;
			const blockType = typeof record.type === "string" ? record.type : "block";
			const text =
				(typeof record.text === "string" && record.text) ||
				(typeof record.input_text === "string" && record.input_text) ||
				(typeof record.output_text === "string" && record.output_text) ||
				(typeof record.content === "string" && record.content) ||
				"";
			if (text) return `[${blockType}] ${text}`;
			return `[${blockType}] ${JSON.stringify(record)}`;
		});
		return parts.join("\n");
	}
	if (value && typeof value === "object") return JSON.stringify(value);
	return String(value ?? "");
}

function providerPayloadRawPreview(payload: unknown): string {
	if (!PAYLOAD_TRACE_RAW_ENABLED) return "";
	try {
		const maxChars = Number.isFinite(PAYLOAD_TRACE_RAW_MAX_CHARS)
			? Math.max(1000, PAYLOAD_TRACE_RAW_MAX_CHARS)
			: 12000;
		return truncateForCheckpoint(JSON.stringify(payload, null, 2), maxChars);
	} catch {
		return String(payload ?? "");
	}
}

function extractPayloadTokens(payload: unknown): { message_tokens: number; tool_schema_tokens: number } {
	if (!payload || typeof payload !== "object") {
		return { message_tokens: 0, tool_schema_tokens: 0 };
	}
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? record.messages : [];
	const input = Array.isArray(record.input) ? record.input : [];
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	return {
		message_tokens: estimateTextTokenCount(JSON.stringify(messages.length > 0 ? messages : input)),
		tool_schema_tokens: tools ? estimateTextTokenCount(JSON.stringify(tools)) : 0,
	};
}

function measureJarvisPrefixProbe(payload: unknown): JarvisPrefixProbe {
	if (!payload || typeof payload !== "object") {
		return emptyJarvisPrefixProbe();
	}
	const record = payload as Record<string, unknown>;
	const payloadMessages = providerPayloadMessages(record);
	const stablePromptMessages = leadingStablePromptMessages(payloadMessages);
	const liveMessages = payloadMessages.slice(stablePromptMessages.length);
	const promptFields = providerPayloadPromptFields(record);
	const tools = Array.isArray(record.tools) ? record.tools : [];
	const stablePrefix = {
		prompt_fields: promptFields,
		prompt_messages: stablePromptMessages,
		tools: canonicalToolsForJarvisPrefixProbe(tools),
	};
	return {
		stable_prefix_hash: sha256Hex24(stableJson(stablePrefix)),
		stable_prefix_tokens_est:
			estimateTextTokenCount(stableJson(promptFields)) +
			estimateTextTokenCount(JSON.stringify(stablePromptMessages)) +
			(tools.length > 0 ? estimateTextTokenCount(JSON.stringify(tools)) : 0),
		live_tokens_est: estimateTextTokenCount(JSON.stringify(liveMessages)),
	};
}

function emptyJarvisPrefixProbe(): JarvisPrefixProbe {
	return {
		stable_prefix_hash: sha256Hex24("{}"),
		stable_prefix_tokens_est: 0,
		live_tokens_est: 0,
	};
}

function providerPayloadMessages(record: Record<string, unknown>): Array<Record<string, unknown>> {
	const messages = Array.isArray(record.messages) ? record.messages : Array.isArray(record.input) ? record.input : [];
	return messages.filter((message): message is Record<string, unknown> => !!message && typeof message === "object");
}

function leadingStablePromptMessages(messages: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
	const stable: Array<Record<string, unknown>> = [];
	for (const message of messages) {
		if (classifyProviderPayloadMessage(message) !== "prompt_messages") break;
		stable.push(message);
	}
	return stable;
}

function providerPayloadPromptFields(record: Record<string, unknown>): Record<string, string> {
	const fields: Record<string, string> = {};
	for (const key of ["systemPrompt", "system", "instructions"]) {
		const value = record[key];
		if (typeof value === "string" && value.trim()) {
			fields[key] = value;
		}
	}
	return fields;
}

function canonicalToolsForJarvisPrefixProbe(tools: unknown[]): unknown[] {
	return tools
		.map((tool, index) => ({ index, sortKey: jarvisToolSchemaSortKey(tool), tool }))
		.sort((left, right) => {
			const byKey = left.sortKey.localeCompare(right.sortKey);
			return byKey !== 0 ? byKey : left.index - right.index;
		})
		.map((entry) => entry.tool);
}

function jarvisToolSchemaSortKey(tool: unknown): string {
	if (!tool || typeof tool !== "object") return stableJson(tool);
	const record = tool as Record<string, unknown>;
	const fnRecord =
		record.function && typeof record.function === "object" ? (record.function as Record<string, unknown>) : {};
	const defRecord =
		record.definition && typeof record.definition === "object" ? (record.definition as Record<string, unknown>) : {};
	const name =
		(typeof record.name === "string" && record.name) ||
		(typeof fnRecord.name === "string" && fnRecord.name) ||
		(typeof defRecord.name === "string" && defRecord.name) ||
		"";
	return `${name}\n${stableJson(tool)}`;
}

function stableJson(value: unknown): string {
	return JSON.stringify(canonicalizeJarvisProbeValue(value));
}

function canonicalizeJarvisProbeValue(value: unknown): unknown {
	if (Array.isArray(value)) return value.map(canonicalizeJarvisProbeValue);
	if (!value || typeof value !== "object") return value;
	const record = value as Record<string, unknown>;
	const sorted: Record<string, unknown> = {};
	for (const key of Object.keys(record).sort()) {
		const field = record[key];
		if (field !== undefined) sorted[key] = canonicalizeJarvisProbeValue(field);
	}
	return sorted;
}

function sha256Hex24(value: string): string {
	return createHash("sha256").update(value, "utf8").digest("hex").slice(0, 24);
}

function buildJarvisCacheReport(
	usage: TurnUsageSummary | undefined,
	hints: { provider?: unknown; api?: unknown; model?: unknown },
): JarvisCacheReport {
	const providerSignal = jarvisCacheSignalForProvider(hints);
	const hasCacheUsageField = usageHasProviderCacheField(usage);
	const reportedSignal: JarvisCacheSignal =
		providerSignal === "actual" && hasCacheUsageField ? "actual" : "unreported";
	const cacheRead = Math.max(0, usage?.cacheRead ?? 0);
	const input = Math.max(0, usage?.input ?? 0);
	const denominator = input + cacheRead;
	return {
		cache_meter: reportedSignal,
		cache_hit_pct: reportedSignal === "actual" && denominator > 0 ? cacheRead / denominator : null,
		provider_cache_read_tokens: usage?.cacheRead,
		provider_cache_write_tokens: usage?.cacheWrite,
	};
}

function usageHasProviderCacheField(usage: TurnUsageSummary | undefined): boolean {
	if (!usage) return false;
	const record = usage as unknown as { cacheRead?: unknown; cacheWrite?: unknown };
	return typeof record.cacheRead === "number" || typeof record.cacheWrite === "number";
}

function jarvisCacheSignalForProvider(hints: {
	provider?: unknown;
	api?: unknown;
	model?: unknown;
}): JarvisCacheSignal {
	const parts = [hints.provider, hints.api, hints.model]
		.map((value) => (typeof value === "string" ? value.toLowerCase() : ""))
		.filter(Boolean);
	const text = parts.join(" ");
	if (!text) return "unreported";
	if (/\bollama(?:-cloud)?\b/.test(text) || /\bopenai-compatible\b/.test(text)) return "unreported";
	if (/\bopenai(?:-codex)?\b/.test(text) || /\bopenai-responses\b/.test(text)) return "actual";
	return "unreported";
}

function buildProviderPayloadBreakdown(payload: unknown): Record<string, unknown> {
	if (!payload || typeof payload !== "object") {
		return { tokens: {}, messages: {}, totals: { estimated_total: 0 } };
	}
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	const payloadMessages = messages ?? input ?? [];
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	const systemPrompt = providerPayloadSystemPrompt(record);
	const payloadMetrics = extractPayloadTokens(payload);
	const tokens: Record<string, number> = {};
	const messageCounts: Record<string, number> = {};
	const topItems: Array<{ bucket: string; label: string; tokens: number }> = [];
	const functionNamesByCallId = new Map<string, string>();
	let messageBucketTokens = 0;

	const addTokens = (bucket: string, value: number): void => {
		const rounded = Math.max(0, Math.round(value));
		if (rounded <= 0) return;
		tokens[bucket] = (tokens[bucket] ?? 0) + rounded;
	};
	const addMessage = (bucket: string): void => {
		messageCounts[bucket] = (messageCounts[bucket] ?? 0) + 1;
	};
	const addTop = (bucket: string, label: string, value: number): void => {
		const rounded = Math.max(0, Math.round(value));
		if (rounded <= 0) return;
		topItems.push({ bucket, label, tokens: rounded });
	};

	if (systemPrompt.trim()) addTokens("instructions", estimateTextTokenCount(systemPrompt));
	if (tools?.length) addTokens("tool_schema", payloadMetrics.tool_schema_tokens);

	for (const message of payloadMessages) {
		const type = String(message.type ?? "message");
		if (type === "function_call") {
			const callId = message.call_id;
			const name = message.name;
			if (typeof callId === "string" && typeof name === "string") {
				functionNamesByCallId.set(callId, name);
			}
		}
	}

	for (const message of payloadMessages) {
		const bucket = classifyProviderPayloadMessage(message);
		const role = String(message.role ?? "");
		const type = String(message.type ?? "message");
		const messageTokens = estimateUnknownMessageTokens(message);
		addMessage(bucket);
		addTop(bucket, summarizePayloadItem(message, functionNamesByCallId), messageTokens);

		if (bucket === "user") {
			const text = providerPayloadMessageContentText(message);
			const memory = extractJarvisMemoryText(text);
			const userText = stripJarvisMemoryBlock(text);
			const memoryTokens = estimateTextTokenCount(memory);
			const userTokens = estimateTextTokenCount(userText);
			addTokens("jlc_memory", memoryTokens);
			addTokens("user_request", userTokens);
			messageBucketTokens += memoryTokens + userTokens;
			continue;
		}

		if (role === "assistant" || type === "reasoning") {
			let assistantTokens = 0;
			for (const [field, value] of estimateMessageFieldTokens(message)) {
				if (field === "reasoning") {
					addTokens("responses_reasoning", value);
				} else if (field === "encrypted") {
					addTokens("responses_encrypted", value);
				} else {
					addTokens(bucket, value);
					assistantTokens += value;
				}
			}
			messageBucketTokens += Math.max(messageTokens, assistantTokens);
			continue;
		}

		addTokens(bucket, messageTokens);
		messageBucketTokens += messageTokens;
	}

	const messageJsonOverhead = Math.max(0, payloadMetrics.message_tokens - messageBucketTokens);
	addTokens("message_json_overhead", messageJsonOverhead);
	const estimatedTotal =
		payloadMetrics.message_tokens + payloadMetrics.tool_schema_tokens + estimateTextTokenCount(systemPrompt);
	return {
		tokens,
		messages: messageCounts,
		totals: {
			message_tokens: payloadMetrics.message_tokens,
			tool_schema_tokens: payloadMetrics.tool_schema_tokens,
			instructions_tokens: estimateTextTokenCount(systemPrompt),
			classified_message_tokens: messageBucketTokens,
			message_json_overhead: messageJsonOverhead,
			estimated_total: estimatedTotal,
		},
		top_items: topItems.sort((left, right) => right.tokens - left.tokens).slice(0, 8),
		tool_schema_top: summarizeToolSchema(tools).split(",").filter(Boolean).slice(0, 8),
	};
}

function providerPayloadSystemPrompt(record: Record<string, unknown>): string {
	return typeof record.systemPrompt === "string"
		? record.systemPrompt
		: typeof record.system === "string"
			? record.system
			: typeof record.instructions === "string"
				? record.instructions
				: "";
}

function classifyProviderPayloadMessage(message: Record<string, unknown>): string {
	const role = String(message.role ?? "");
	const type = String(message.type ?? "message");
	const content = providerPayloadMessageContentText(message);
	if ((role === "system" || role === "developer") && /^# JARVIS_SUBTURN_STATE\b/i.test(content.trim())) {
		return "state_carry";
	}
	if ((role === "system" || role === "developer") && /# JARVIS_SUBTURN\.md\b/i.test(content)) {
		return "subturn_compact_hint";
	}
	if (role === "system" || role === "developer") return "prompt_messages";
	if (role === "user") return "user";
	if (type === "function_call") return "recent_tool_call_args";
	if (type === "function_call_output" || role === "tool") return "recent_tool_output";
	if (role === "assistant") return "recent_assistant";
	if (type === "reasoning") return "responses_reasoning";
	return "other_messages";
}

function providerPayloadMessageContentText(message: Record<string, unknown>): string {
	const content = message.content;
	if (typeof content === "string") return content;
	if (Array.isArray(content)) return contentToText(content);
	return payloadHistoryValueText(content);
}

function extractJarvisMemoryText(text: string): string {
	const match = /^<jarvis_memory>\s*([\s\S]*?)<\/jarvis_memory>/i.exec(text);
	return match?.[1]?.trim() ?? "";
}

function providerPayloadMessageCount(payload: unknown): number {
	if (!payload || typeof payload !== "object") return 0;
	const record = payload as Record<string, unknown>;
	if (Array.isArray(record.messages)) return record.messages.length;
	if (Array.isArray(record.input)) return record.input.length;
	return 0;
}

function summarizeToolSchema(tools: unknown[] | undefined): string {
	if (!tools?.length) return "";
	return tools
		.map((tool) => {
			if (!tool || typeof tool !== "object") return "unknown";
			const record = tool as Record<string, unknown>;
			const name = providerToolSchemaName(tool) || `unknown<${Object.keys(record).slice(0, 4).join("|")}>`;
			return `${name}:${estimateTextTokenCount(JSON.stringify(tool))}`;
		})
		.join(",");
}

function providerToolSchemaName(tool: unknown): string | undefined {
	if (!tool || typeof tool !== "object") return undefined;
	const record = tool as Record<string, unknown>;
	const fnRecord =
		record.function && typeof record.function === "object" ? (record.function as Record<string, unknown>) : undefined;
	const defRecord =
		record.definition && typeof record.definition === "object"
			? (record.definition as Record<string, unknown>)
			: undefined;
	return (
		(typeof record.name === "string" && record.name) ||
		(fnRecord && typeof fnRecord.name === "string" && fnRecord.name) ||
		(defRecord && typeof defRecord.name === "string" && defRecord.name) ||
		undefined
	);
}

function estimateUnknownMessageTokens(message: Record<string, unknown>): number {
	const fieldTotal = estimateMessageFieldTokens(message).reduce((sum, [, tokens]) => sum + tokens, 0);
	if (fieldTotal > 0) return fieldTotal;
	return estimateTextTokenCount(JSON.stringify(message));
}

function estimateMessageFieldTokens(message: Record<string, unknown>): Array<[string, number]> {
	const fields: Array<[string, number]> = [];
	const content = message.content;
	if (typeof content === "string") {
		fields.push(["content", estimateTextTokenCount(content)]);
	} else if (Array.isArray(content)) {
		let text = "";
		for (const block of content) {
			if (!block || typeof block !== "object") continue;
			const rec = block as Record<string, unknown>;
			if (
				(rec.type === "text" || rec.type === "input_text" || rec.type === "output_text") &&
				typeof rec.text === "string"
			) {
				text += rec.text;
			}
		}
		fields.push(["content", estimateTextTokenCount(text)]);
	}
	const output = message.output;
	if (typeof output === "string") {
		fields.push(["output", estimateTextTokenCount(output)]);
	} else if (Array.isArray(output)) {
		fields.push(["output", estimateTextTokenCount(JSON.stringify(output))]);
	}
	const args = message.arguments;
	if (typeof args === "string") fields.push(["arguments", estimateTextTokenCount(args)]);
	const summary = message.summary;
	if (typeof summary === "string") {
		fields.push(["reasoning", estimateTextTokenCount(summary)]);
	} else if (Array.isArray(summary)) {
		fields.push(["reasoning", estimateTextTokenCount(JSON.stringify(summary))]);
	}
	const encrypted = message.encrypted_content;
	if (typeof encrypted === "string") fields.push(["encrypted", estimateTextTokenCount(encrypted)]);
	return fields;
}

function trimPayloadToCurrentJarvisTurn(payload: unknown, options?: { stateCarry?: boolean }): unknown {
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	if (!messages && !input) return payload;
	const allowStateCarry =
		(options?.stateCarry ?? subturnStateCarryEnabled()) && !hasResponsesPreviousResponseId(record);

	return {
		...record,
		...(messages ? { messages: trimMessageListToCurrentJarvisTurn(messages, false, allowStateCarry) } : {}),
		...(input ? { input: trimMessageListToCurrentJarvisTurn(input, true, allowStateCarry) } : {}),
	};
}

function trimMessageListToCurrentJarvisTurn(
	messages: Array<Record<string, unknown>>,
	includeSubturnHint: boolean,
	allowStateCarry = false,
): Array<Record<string, unknown>> {
	let currentTurnStart = -1;
	for (let i = messages.length - 1; i >= 0; i--) {
		if (String(messages[i].role ?? "") === "user" && messageContainsJarvisMemory(messages[i])) {
			currentTurnStart = i;
			break;
		}
	}
	if (currentTurnStart < 0) {
		for (let i = messages.length - 1; i >= 0; i--) {
			if (String(messages[i].role ?? "") === "user") {
				currentTurnStart = i;
				break;
			}
		}
	}
	const promptMessages = messages.filter((message, index) => {
		if (currentTurnStart >= 0 && index >= currentTurnStart) return false;
		const role = String(message.role ?? "");
		return role === "system" || role === "developer";
	});
	const promptMessageSet = new Set(promptMessages);
	const currentTurnSource =
		currentTurnStart >= 0
			? messages.slice(currentTurnStart)
			: messages.filter((message) => !promptMessageSet.has(message));
	const stateCarryMessage = allowStateCarry
		? subturnStateCarryMessage(promptMessages, currentTurnSource, includeSubturnHint)
		: undefined;
	const currentTurnMessages = stateCarryMessage
		? trimCurrentTurnMessagesForStateCarry(currentTurnSource)
		: trimCurrentTurnMessagesForSubturn(currentTurnSource);
	const contextMessage = !stateCarryMessage && includeSubturnHint ? subturnContextMessage(promptMessages) : undefined;
	return [
		...promptMessages,
		...(stateCarryMessage ? [stateCarryMessage] : contextMessage ? [contextMessage] : []),
		...currentTurnMessages,
	];
}

function trimCurrentTurnMessagesForSubturn(messages: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
	const maxMessages = getSubturnPayloadMessageLimit();
	if (!Number.isFinite(maxMessages) || messages.length <= maxMessages) return messages;

	const hasUserAnchor = messages.length > 0 && isPayloadUserMessage(messages[0]);
	const anchorMessages = hasUserAnchor ? [messages[0]] : [];
	const tailMessages = hasUserAnchor ? messages.slice(1) : messages;
	const tailLimit = Math.max(1, maxMessages - anchorMessages.length);
	const groups = groupSubturnPayloadMessages(tailMessages);
	const keptTail = keepRecentSubturnGroups(groups, tailLimit);
	return [...anchorMessages, ...keptTail];
}

function trimCurrentTurnMessagesForStateCarry(
	messages: Array<Record<string, unknown>>,
): Array<Record<string, unknown>> {
	const maxTailMessages = getSubturnStateCarryRecentMessageLimit();
	const hasUserAnchor = messages.length > 0 && isPayloadUserMessage(messages[0]);
	const anchorMessages = hasUserAnchor ? [stripJarvisMemoryFromUserMessageAfterFirstProviderCall(messages[0])] : [];
	const tailMessages = hasUserAnchor ? messages.slice(1) : messages;
	if (maxTailMessages <= 0 || tailMessages.length === 0) return anchorMessages;
	const groups = groupSubturnPayloadMessages(tailMessages);
	const keptTail = keepRecentSubturnGroups(groups, maxTailMessages);
	return [...anchorMessages, ...keptTail];
}

function stripJarvisMemoryFromUserMessageAfterFirstProviderCall(
	message: Record<string, unknown>,
): Record<string, unknown> {
	if (providerCallCountThisTurn < 1) return message;
	const content = message.content;
	if (typeof content === "string") {
		const stripped = stripJarvisMemoryBlock(content);
		return stripped === content ? message : { ...message, content: stripped };
	}
	if (!Array.isArray(content)) return message;
	let changed = false;
	const nextContent = content.map((block) => {
		if (!block || typeof block !== "object") return block;
		const record = block as Record<string, unknown>;
		const text = typeof record.text === "string" ? record.text : undefined;
		if (text === undefined) return block;
		const stripped = stripJarvisMemoryBlock(text);
		if (stripped === text) return block;
		changed = true;
		return { ...record, text: stripped };
	});
	return changed ? { ...message, content: nextContent } : message;
}

function keepRecentSubturnGroups(
	groups: Array<Array<Record<string, unknown>>>,
	maxMessages: number,
): Array<Record<string, unknown>> {
	if (groups.length === 0) return [];
	const keptGroups: Array<Array<Record<string, unknown>>> = [];
	let keptCount = 0;
	for (let index = groups.length - 1; index >= 0; index--) {
		const group = groups[index];
		if (!group || group.length === 0) continue;
		if (keptGroups.length > 0 && keptCount + group.length > maxMessages) break;
		keptGroups.unshift(group);
		keptCount += group.length;
		if (keptCount >= maxMessages) break;
	}
	return keptGroups.flat();
}

function groupSubturnPayloadMessages(messages: Array<Record<string, unknown>>): Array<Array<Record<string, unknown>>> {
	const groups: Array<Array<Record<string, unknown>>> = [];
	for (let index = 0; index < messages.length; ) {
		const message = messages[index];
		if (!message) {
			index++;
			continue;
		}
		if (isResponsesFunctionCall(message)) {
			const group: Array<Record<string, unknown>> = [];
			const pendingCallIds = new Set<string>();
			while (index < messages.length && isResponsesFunctionCall(messages[index])) {
				const call = messages[index];
				group.push(call);
				const callId = payloadCallId(call);
				if (callId) pendingCallIds.add(callId);
				index++;
			}
			while (index < messages.length && isResponsesFunctionCallOutput(messages[index])) {
				const output = messages[index];
				const callId = payloadCallId(output);
				if (callId && pendingCallIds.size > 0 && !pendingCallIds.has(callId)) break;
				group.push(output);
				if (callId) pendingCallIds.delete(callId);
				index++;
			}
			groups.push(group);
			continue;
		}
		if (isPayloadAssistantMessage(message)) {
			const group = [message];
			index++;
			while (index < messages.length && isPayloadToolResultMessage(messages[index])) {
				group.push(messages[index]);
				index++;
			}
			groups.push(group);
			continue;
		}
		groups.push([message]);
		index++;
	}
	return groups;
}

function isPayloadUserMessage(message: Record<string, unknown> | undefined): boolean {
	return String(message?.role ?? "") === "user" && !isPayloadToolResultMessage(message);
}

function isPayloadAssistantMessage(message: Record<string, unknown> | undefined): boolean {
	return String(message?.role ?? "") === "assistant";
}

function isPayloadToolResultMessage(message: Record<string, unknown> | undefined): boolean {
	if (!message) return false;
	if (String(message.role ?? "") === "tool") return true;
	if (String(message.type ?? "") === "function_call_output") return true;
	const content = message.content;
	return (
		Array.isArray(content) &&
		content.some((block) => {
			if (!block || typeof block !== "object") return false;
			const type = String((block as Record<string, unknown>).type ?? "");
			return type === "tool_result" || type === "function_call_output";
		})
	);
}

function isResponsesFunctionCall(message: Record<string, unknown> | undefined): boolean {
	return String(message?.type ?? "") === "function_call";
}

function isResponsesFunctionCallOutput(message: Record<string, unknown> | undefined): boolean {
	return String(message?.type ?? "") === "function_call_output";
}

function payloadCallId(message: Record<string, unknown> | undefined): string | undefined {
	const value = message?.call_id ?? message?.tool_call_id;
	return typeof value === "string" && value.trim() ? value : undefined;
}

function hasResponsesPreviousResponseId(record: Record<string, unknown>): boolean {
	const value = record.previous_response_id;
	return typeof value === "string" && value.trim().length > 0;
}

function subturnStateCarryMessage(
	existingPromptMessages: Array<Record<string, unknown>>,
	currentTurnSource: Array<Record<string, unknown>>,
	preferDeveloperRole = false,
): Record<string, unknown> | undefined {
	if (!subturnStateCarryEnabled()) return undefined;
	if (!hasSubturnStateForCarry()) return undefined;
	const text = renderSubturnObserveState(buildSubturnObserveState({ input: currentTurnSource }));
	const role =
		existingPromptMessages.some((message) => String(message.role ?? "") === "developer") || preferDeveloperRole
			? "developer"
			: "system";
	return { role, content: text };
}

function hasSubturnStateForCarry(): boolean {
	return (
		subturnCarryOrder.length > 0 ||
		subturnSummaryLines.length > 0 ||
		subturnEvidenceOrder.length > 0 ||
		(providerCallCountThisTurn > 0 && subturnUserMessage.trim().length > 0)
	);
}

function subturnContextMessage(
	existingPromptMessages: Array<Record<string, unknown>>,
): Record<string, unknown> | undefined {
	if (!subturnCompactEnabled()) return undefined;
	const carryLines = subturnCarryLinesForPrompt();
	const historyLines = subturnHistoryLinesForPrompt();
	if (existingPromptMessages.length === 0) return undefined;
	if (!subturnLogPath && carryLines.length === 0 && historyLines.length === 0) return undefined;
	const text = buildSubturnCompactMarkdown("active");
	const role = existingPromptMessages.some((message) => String(message.role ?? "") === "developer")
		? "developer"
		: "system";
	return { role, content: text };
}

function messageContainsJarvisMemory(message: Record<string, unknown>): boolean {
	const content = message.content;
	if (typeof content === "string") {
		return content.includes("<jarvis_memory>") || content.includes("[JARVIS Code Memory]");
	}
	if (!Array.isArray(content)) return false;
	return content.some((block) => {
		if (!block || typeof block !== "object") return false;
		const text = (block as Record<string, unknown>).text;
		return typeof text === "string" && (text.includes("<jarvis_memory>") || text.includes("[JARVIS Code Memory]"));
	});
}

function summarizePayloadItem(message: Record<string, unknown>, functionNamesByCallId: Map<string, string>): string {
	const type = String(message.type ?? "message");
	const role = String(message.role ?? "");
	if (type === "function_call_output") {
		const callId = typeof message.call_id === "string" ? message.call_id : "";
		const name = callId ? functionNamesByCallId.get(callId) : undefined;
		return name ? `${type}:${name}` : type;
	}
	if (type === "function_call") {
		const name = typeof message.name === "string" ? message.name : "";
		return name ? `${type}:${name}` : type;
	}
	if (role) return `${type}:${role}`;
	return type;
}

function ensureContextInProviderPayload(payload: unknown, memory: string | undefined): unknown {
	if (!memory?.trim() || !payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	if (messages) {
		const nextMessages = injectJarvisMemoryIntoPayloadMessages(messages, memory);
		return nextMessages === messages ? payload : { ...record, messages: nextMessages };
	}
	if (input) {
		const nextInput = injectJarvisMemoryIntoPayloadMessages(input, memory);
		return nextInput === input ? payload : { ...record, input: nextInput };
	}
	return payload;
}

function injectJarvisMemoryIntoPayloadMessages(
	messages: Array<Record<string, unknown>>,
	memory: string,
): Array<Record<string, unknown>> {
	for (let index = messages.length - 1; index >= 0; index--) {
		const message = messages[index];
		if (String(message.role ?? "") !== "user") continue;
		const existing = payloadHistoryValueText(message.content);
		if (/<jarvis_memory>/i.test(existing)) return messages;
		const next = [...messages];
		next[index] = {
			...message,
			content: prependJarvisMemoryToPayloadContent(message.content, memory),
		};
		return next;
	}
	return messages;
}

function prependJarvisMemoryToPayloadContent(content: unknown, memory: string): unknown {
	const prefix = `<jarvis_memory>\n${memory}\n</jarvis_memory>\n\n`;
	if (typeof content === "string") return `${prefix}${content}`;
	if (!Array.isArray(content)) {
		return [{ type: "input_text", text: prefix.trimEnd() }];
	}
	const next = [...content];
	for (let index = 0; index < next.length; index++) {
		const item = next[index];
		if (!item || typeof item !== "object") continue;
		const record = item as Record<string, unknown>;
		const textKey =
			typeof record.text === "string"
				? "text"
				: typeof record.input_text === "string"
					? "input_text"
					: typeof record.output_text === "string"
						? "output_text"
						: typeof record.content === "string"
							? "content"
							: "";
		if (!textKey) continue;
		next[index] = { ...record, [textKey]: `${prefix}${String(record[textKey] ?? "")}` };
		return next;
	}
	return [{ type: "input_text", text: prefix.trimEnd() }, ...next];
}

function estimateTextTokenCount(text: string): number {
	if (!text.trim()) return 0;
	return estimateTokens({
		role: "user",
		content: [{ type: "text", text }],
	} as AgentMessage);
}

function estimateMessageTokenCount(message: AgentMessage): number {
	try {
		return estimateTokens(message);
	} catch {
		return estimateTextTokenCount(messageContentToText((message as { content?: unknown }).content));
	}
}

function estimateToolEventTokens(events: ToolEventSummary[]): number {
	if (!events.length) return 0;
	let total = 0;
	for (const event of events) {
		const parts: string[] = [];
		if (typeof event.turnIndex === "number") {
			parts.push(`turn=${event.turnIndex}`);
		}
		for (const result of event.toolResults ?? []) {
			parts.push(
				[result.toolName ?? "tool", result.isError ? "error" : "ok", (result.text ?? "").slice(0, 500)]
					.filter(Boolean)
					.join(": "),
			);
		}
		total += estimateTextTokenCount(parts.join("\n"));
	}
	return total;
}

function findLatestUserIndex(messages: AgentMessage[]): number {
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role === "user") return i;
	}
	return -1;
}

// Reserved for future thinking-level promotion. Currently not invoked
// after the deepdive-force rewrite; kept here intentionally.
function _selectThinkingLevel(
	userText: string,
	assistantText: string,
	ctx?: ExtensionContext,
): SupportedThinkingLevel | undefined {
	const user = userText.trim().toLowerCase();
	if (isSlashCommand(user, "/deepdive")) return ctx ? selectDeepdiveThinkingLevel(ctx) : "xhigh";
	if (isSlashCommand(user, "/chat")) return "medium";
	if (assistantText.includes("[MODE:HEAVY_DEEPDIVE]")) return ctx ? selectDeepdiveThinkingLevel(ctx) : "xhigh";
	if (assistantText.includes("[MODE:DEEPDIVE]")) return ctx ? selectDeepdiveThinkingLevel(ctx) : "xhigh";
	if (assistantText.includes("[MODE:UNREGISTERED_CODING]")) return "medium";
	if (assistantText.includes("[MODE:CHAT]")) return "medium";
	return undefined;
}

function applyRouteThinkingLevel(
	route: EffectiveTurnRoute,
	ctx: ExtensionContext,
	pi: ExtensionAPI,
): SupportedThinkingLevel {
	const level = route === "heavy_deepdive" ? selectDeepdiveThinkingLevel(ctx) : "medium";
	try {
		suppressThinkingPreferenceSaveOnce(level);
		pi.setThinkingLevel(level);
		ctx.ui.setHiddenThinkingLabel(level === "off" ? "" : `Thinking ${level}`);
	} catch {
		/* pi may be stale */
	}
	return level;
}

function parseProjectSwitchCommand(text: string): string | undefined {
	return parseProjectSwitchRequest(text)?.slugOrName;
}

function isAmbiguousProjectSelection(trace: Record<string, unknown> | undefined): boolean {
	return trace?.source === "ambiguous_registry_match";
}

function formatAmbiguousProjectCandidates(trace: Record<string, unknown> | undefined): string {
	const candidates = trace?.candidates;
	if (!Array.isArray(candidates) || candidates.length === 0) return "";
	return candidates
		.map((candidate) => {
			if (!candidate || typeof candidate !== "object") return "";
			const raw = candidate as { name?: unknown; slug?: unknown; project_id?: unknown };
			const name = typeof raw.name === "string" && raw.name.trim() ? raw.name.trim() : undefined;
			const slug = typeof raw.slug === "string" && raw.slug.trim() ? raw.slug.trim() : undefined;
			const projectId =
				typeof raw.project_id === "string" && raw.project_id.trim() ? raw.project_id.trim() : undefined;
			return name || slug || projectId || "";
		})
		.filter((value) => value.length > 0)
		.join(", ");
}

function parseProjectSwitchRequest(text: string):
	| {
			slugOrName: string;
			autoCreate: boolean;
			codePath?: string;
	  }
	| undefined {
	const match = text.trim().match(/^\/project\s+(.+?)\s*$/i);
	if (!match) return undefined;
	let value = match[1].trim();
	if (!value) return undefined;

	let codePath: string | undefined;
	const codePathMatch = value.match(/\s+--code-path\s+(.+?)\s*$/i);
	if (codePathMatch) {
		codePath = unwrapQuoted(codePathMatch[1].trim());
		value = value.slice(0, codePathMatch.index).trim();
	}

	let autoCreate = false;
	if (/\s+--new\s*$/i.test(value)) {
		autoCreate = true;
		value = value.replace(/\s+--new\s*$/i, "").trim();
	}

	const slugOrName = unwrapQuoted(value).trim();
	if (!slugOrName) return undefined;
	return { slugOrName, autoCreate, codePath };
}

function stripProjectTargetFiller(value: string): string {
	return unwrapQuoted(value)
		.replace(/^\s*(?:그럼|그러면|좋아|그래|ㅇㅇ)\s+/i, "")
		.replace(/^\s*(?:새|신규|새로운)\s+/i, "")
		.replace(/\s*(?:다시|새로)\s*$/i, "")
		.replace(/\s*(?:을|를|으로|로|좀|하나|한\s*개)\s*$/gi, "")
		.replace(/\s+(?:file|files|code)\s*$/i, "")
		.replace(/\s+(?:파일|코드)\s*$/i, "")
		.replace(/\s+(?:project|game|app|site|repo|repository)\s*$/i, "")
		.replace(/\s+(?:프로젝트|게임|앱|사이트|웹앱)\s*$/i, "")
		.trim();
}

function isGenericProjectTarget(value: string): boolean {
	const normalized = canonicalProjectText(value);
	return (
		!normalized || /^(?:project|game|app|site|repo|repository|web|프로젝트|게임|앱|사이트|웹앱)$/.test(normalized)
	);
}

function slugCandidateFromProjectTarget(value: string): string | undefined {
	const cleaned = stripProjectTargetFiller(value);
	if (isGenericProjectTarget(cleaned)) return undefined;
	const canonical = canonicalProjectText(cleaned);
	const asciiSlug = canonical
		.replace(/\s+/g, "-")
		.replace(/[^a-z0-9_-]+/g, "-")
		.replace(/-+/g, "-")
		.replace(/^-|-$/g, "");
	if (asciiSlug) return asciiSlug.slice(0, 120);
	return cleaned.slice(0, 120);
}

function parseAssistantProjectCreationPrompt(text: string):
	| {
			slugOrName: string;
			codePath?: string;
	  }
	| undefined {
	if (/(등록\s*안|등록하지|등록\s*없이|등록없이|without\s+register|do\s+not\s+register)/i.test(text)) {
		return undefined;
	}
	if (!/(jarvis\s*프로젝트|자비스\s*프로젝트|JARVIS\s+project|register|등록|생성|만들)/i.test(text)) {
		return undefined;
	}
	const absPath = extractAbsolutePathsFromText(text)[0];
	if (absPath) {
		return {
			slugOrName: path.basename(absPath),
			codePath: absPath,
		};
	}
	const quoted = text.match(/["'“‘`]([^"'“”‘’`]{3,120})["'”’`]/);
	const quotedTarget = quoted?.[1] ? slugCandidateFromProjectTarget(quoted[1]) : undefined;
	if (quotedTarget) {
		return { slugOrName: quotedTarget };
	}
	const targetBeforeProject = text.match(
		/(?:^|[\s"'“‘`])([A-Za-z0-9][A-Za-z0-9_-]{2,80})(?:\s*(?:을|를|로|으로|as|to))?\s+(?:jarvis\s*)?(?:project|프로젝트)/i,
	);
	const target = targetBeforeProject?.[1] ? slugCandidateFromProjectTarget(targetBeforeProject[1]) : undefined;
	if (target && !/^jarvis$/i.test(target)) {
		return { slugOrName: target };
	}
	return undefined;
}

function parseSetupDefaultRootCommand(text: string): string | undefined {
	const match = text.trim().match(/^\/setup-default-root\s+(.+?)\s*$/i);
	if (!match) return undefined;
	const value = unwrapQuoted(match[1].trim());
	return value || undefined;
}

function unwrapQuoted(value: string): string {
	const quoted = value.match(/^["'“‘](.+?)["'”’]$/);
	return (quoted?.[1] ?? value).trim();
}

function extractAbsolutePath(text: string): string | undefined {
	const trimmed = text.trim();
	if (!trimmed) return undefined;
	if (/^[A-Za-z]:[\\/]/.test(trimmed)) return trimmed;
	if (trimmed.startsWith("/")) return trimmed;
	return undefined;
}

function isAffirmative(text: string): boolean {
	const trimmed = text.trim();
	return (
		/^(?:yes|y|yeah|yep|ok|okay|sure)(?:\b|[\s.!?]|$)/i.test(trimmed) ||
		/^(?:응|네|예|그래|좋아|ㅇㅇ)(?:\s|[.!?。！？]|$)/i.test(trimmed)
	);
}

function isNegative(text: string): boolean {
	const trimmed = text.trim();
	return (
		/^(?:no|n|nope|cancel)(?:\b|[\s.!?]|$)/i.test(trimmed) ||
		/^(?:아니|아니오|싫어|취소)(?:\s|[.!?。！？]|$)/i.test(trimmed)
	);
}
