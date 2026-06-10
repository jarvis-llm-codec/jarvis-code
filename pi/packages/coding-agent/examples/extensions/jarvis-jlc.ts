import { type ChildProcess, execFile, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { type Static, Type } from "typebox";
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

type SidecarSubagentDelegateResponse = {
	subagent?: string;
	summary?: string;
	iters?: number;
	halt_reason?: string;
	elapsed_sec?: number;
	in_tokens?: number;
	out_tokens?: number;
	think_tokens?: number;
	sub_id?: string;
	error?: string;
};

type SidecarSubagentStreamEvent = {
	event?: "reasoning" | "content" | "activity" | "step" | "result" | "error" | string;
	kind?: string;
	text?: string;
	line?: string;
	result?: SidecarSubagentDelegateResponse;
	error?: string;
	status_code?: number;
};

type SidecarSubagentProgressDetails = {
	streaming: true;
	subagent?: string;
	sub_id?: string;
	activity: string[];
	reasoning_tail?: string;
	content_tail?: string;
	error?: string;
};

type SidecarOrchestrateFinderOutcome = {
	dimension?: string;
	ran?: boolean;
	summary?: string;
	halt_reason?: string;
	in_tokens?: number;
	out_tokens?: number;
	elapsed_sec?: number;
	error?: string | null;
};

type SidecarOrchestrateResponse = {
	orchestration_id?: string;
	state?: string;
	summary?: string;
	finders_total?: number;
	finders_ran?: number;
	stop_reason?: string | null;
	finders?: SidecarOrchestrateFinderOutcome[];
	in_tokens?: number;
	out_tokens?: number;
	elapsed_sec?: number;
	event_log_path?: string;
	error?: string;
};

type SidecarOrchestrateStreamEvent = {
	event?: "activity" | "step" | "result" | "error" | string;
	kind?: string;
	text?: string;
	line?: string;
	result?: SidecarOrchestrateResponse;
	error?: string;
	status_code?: number;
};

type SidecarOrchestrateProgressDetails = {
	streaming: true;
	activity: string[];
	result?: SidecarOrchestrateResponse;
	error?: string;
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
	error?: string;
	body?: string;
	memory_mode?: "light" | "full";
	scheduled_encode?: boolean;
	raw_saved?: boolean;
	memory_write_disabled?: boolean;
	memory_write_reenabled?: boolean;
	memory_write_notice?: string;
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

type SidecarDirectiveKind = "directive" | "report";

type SidecarGanMeta = {
	gan_id?: string;
	round?: number;
	role?: "worker" | "destroyer" | string;
	issues_open?: number;
	status?: "open" | "agreed" | "escalated" | string;
};

type SidecarJobMeta = {
	job_id?: string;
	cycle?: number;
	role?: "orchestrator" | "worker" | string;
	phase?: "dispatch" | "review" | string;
	status?: "open" | "done" | "escalated" | string;
};

type SidecarDirectiveItem = {
	id?: string;
	ts?: string;
	kind?: SidecarDirectiveKind;
	from_window?: string;
	to_window?: string;
	body?: string;
	gan?: SidecarGanMeta;
	job?: SidecarJobMeta;
};

type SidecarDirectivePendingResponse = {
	ok?: boolean;
	error?: string;
	items?: SidecarDirectiveItem[];
	unchanged?: boolean;
	queue_mtime_ns?: number;
	queue_size?: number;
	cursor?: number;
	cursor_at_end?: boolean;
};

type SidecarDirectiveWindow = {
	pair8?: string;
	label?: string | null;
	pid?: number | null;
	alive?: boolean;
	current?: boolean;
	path?: string;
	created_at?: string | null;
	role?: string;
	status?: string;
	contract?: string;
	stage?: string;
	active_job_id?: string;
	active_job_cycle?: number;
	active_job_phase?: string;
	active_job_role?: string;
	job_cycle_cap?: number;
	counterpart_window?: string;
};

type SidecarDirectiveWindowsResponse = {
	ok?: boolean;
	error?: string;
	windows?: SidecarDirectiveWindow[];
};

type SidecarDirectiveSendResponse = {
	ok?: boolean;
	error?: string;
	item?: SidecarDirectiveItem;
	windows?: SidecarDirectiveWindow[];
};

type SidecarControlBridgeRequest = {
	id?: string;
	kind?: string;
	to_window?: string;
	payload?: unknown;
	created_at?: string;
	deadline_at?: string;
};

type SidecarControlBridgePendingResponse = {
	ok?: boolean;
	error?: string;
	requests?: SidecarControlBridgeRequest[];
};

type SidecarControlBridgeAnswerResponse = {
	ok?: boolean;
	error?: string;
	request_id?: string;
};

type SidecarJobHistoryResponse = {
	ok?: boolean;
	error?: string;
	job_id?: string;
	status?: "open" | "done" | "escalated" | string;
	cycle?: number;
	cycle_cap?: number;
	items?: SidecarDirectiveItem[];
};

type SidecarSpawnWindowResponse = {
	ok?: boolean;
	error?: string;
	body?: string;
	pair8?: string;
	window?: {
		pair8?: string;
		pair_id?: string;
		url?: string;
		port?: number;
		pid?: number;
		runtime_path?: string;
		label?: string | null;
	};
	directive?: SidecarDirectiveItem | null;
};

type SecondEyesProviderPhase = "plan_draft" | "review" | "implement";

type SidecarWindowLabelResponse = {
	ok?: boolean;
	error?: string;
	pair8?: string;
	old_label?: string | null;
	label?: string | null;
	runtime_path?: string;
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
	pair_id?: string | null;
	window_label?: string | null;
	memory_write_enabled?: boolean;
	memory_write_disabled_reason?: string | null;
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
	auth_kind?: string | null;
	roles?: string[];
	catalog_source?: "live" | "cache" | "static" | "disabled" | "unavailable" | string;
	cache_stale?: boolean;
	catalog_warning?: string | null;
};

type SidecarLLMSettingCatalogResponse = {
	ok?: boolean;
	providers?: Record<string, SidecarLLMSettingProvider>;
	recommended?: Record<string, { provider?: string; model?: string }>;
	current?: { chat?: string | null; subagent?: string | null; router?: string | null; encoder?: string | null };
};

type SidecarLLMSettingApplyResponse = {
	ok?: boolean;
	error?: string;
	hint?: string;
	chat?: string;
	subagent?: string;
	router?: string;
	encoder?: string;
	config_path?: string;
	models_json_path?: string;
	corrections?: string[];
	reload_warning?: string;
};

type SidecarCredentialTarget = {
	label?: string;
	env_name?: string;
	kind?: string;
	configured?: boolean;
	source?: "bundled" | "custom";
	custom?: boolean;
	base_url?: string;
	roles?: string[];
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

type SidecarCredentialCustomResponse = {
	ok?: boolean;
	error?: string;
	provider_id?: string;
	label?: string;
	env_name?: string;
	source?: "bundled" | "custom";
	duplicate?: boolean;
	validation?: SidecarCredentialSetResponse["validation"];
};

type SidecarCredentialRemoveResponse = {
	ok?: boolean;
	error?: string;
	provider_id?: string;
	env_name?: string | null;
	removed_key?: boolean;
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

type JarvisTodoStatus = "pending" | "in_progress" | "completed";

type JarvisTodoItem = {
	content: string;
	status: JarvisTodoStatus;
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
const MEMORY_WRITE_DISABLED_NOTICE =
	"[Notice] Memory writes are off — another JARVIS window owns the session (M5.5 releases it)";
const MEMORY_WRITE_ENABLED_NOTICE = "✓ memory write enabled — this window now owns the session";
const SUBTURN_RECENT_ASSISTANT_CYCLES = 1;
const SUBTURN_SUMMARY_MAX_CHARS = 2500;
const SUBTURN_COMMIT_MAX_ITEMS = 16;
const SUBTURN_LEDGER_MAX_ITEMS = 150;
const SUBTURN_COMMIT_MAX_CHARS = 3000;
const SUBTURN_TOOL_OUTPUT_HEAD_CHARS = 1500;
const SUBTURN_TOOL_OUTPUT_TAIL_CHARS = 1500;
const SUBTURN_ASSISTANT_SAMPLE_CHARS = 1000;
const DEFAULT_SUBTURN_PAYLOAD_MESSAGE_LIMIT = 100;
const DEFAULT_SUBTURN_STATE_CARRY_RECENT_MESSAGES = 8;
// The ceiling is a runaway backstop, not a budget: JLC halves per-subturn
// prompt cost, so chopping completion-bound runs saves little and costs the
// finish (25 and even 200 chunked live runs). 1000 moves the net out of the
// way of any sane run; JARVIS_PC_CEILING still overrides.
const DEFAULT_SUBTURN_PC_CEILING = 1000;
const PC_CEILING_REPORT_STOP_MARKER = "JARVIS_PC_CEILING_REPORT_STOP";
const LOCKED_RESOURCE_REPORT_STOP_MARKER = "JARVIS_LOCKED_RESOURCE_REPORT_STOP";
const LOCKED_RESOURCE_REPORT_STOP_TEXT = `${LOCKED_RESOURCE_REPORT_STOP_MARKER}: locked resource cannot be deleted - ask the user to close the owning process before trying again. Report only owner-detection attempts and stop retrying delete/write.`;
// Repeated-failure brake: a model re-firing the byte-identical failing shell
// command is degenerating, not working (live 2026-06-12: CSS literals fired as
// bash 6x in a row inside one turn). Lessons only advise after the fact; this
// is the hard brake — one developer warning, then a forced report-stop.
const REPEATED_FAILURE_REPORT_STOP_MARKER = "JARVIS_REPEATED_FAILURE_REPORT_STOP";
const REPEATED_FAILURE_WARN_THRESHOLD = 3;
const REPEATED_FAILURE_REPORT_STOP_THRESHOLD = 5;
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
const WORKER_TOOLS_NEEDED_MARKER = "[JLC:NEED_WORKER_TOOLS]";
const WORKER_TOOLS_RETRY_MARKER = "[JLC WORKER-TOOLS RETRY]";
const VERIFY_INCOMPLETE_FOLLOWUP_MARKER = "[JLC VERIFICATION-FLOOR FOLLOW-UP]";
const MAX_VERIFY_CONTINUATIONS = 2;
const VERIFICATION_COMMAND_PATTERNS: RegExp[] = [
	/(?:^|[;&|]\s*)(?:npm\s+(?:run\s+)?test|yarn\s+test|pnpm\s+test|pytest|jest|vitest|mocha|go\s+test|cargo\s+test|gradle\s+test|mvn\s+test)\b/i,
	/(?:^|[;&|]\s*)(?:npm\s+run\s+build|tsc|tsgo|biome|eslint|ruff|mypy|make|cargo\s+build|go\s+build|vite\s+build|webpack|npm\s+run\s+check)\b/i,
	/(?:^|[;&|]\s*)(?:npm\s+(?:start|run\s+dev)|node\s+\S+|python3?\s+\S+\.py|cargo\s+run|go\s+run|\.\/\S+)/i,
];

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

export function sidecarUrlCandidates(): string[] {
	const envUrl = process.env.JARVIS_SIDECAR_URL?.trim();
	const pairId = process.env.JARVIS_PAIR_ID?.trim();
	if (pairId) {
		const candidates = [envUrl || DEFAULT_SIDECAR_URL].filter((value): value is string => Boolean(value));
		return [...new Set(candidates.map((value) => value.replace(/\/+$/, "")))];
	}
	const candidates = [envUrl, readRuntimeSidecarUrl(), DEFAULT_SIDECAR_URL].filter((value): value is string =>
		Boolean(value),
	);
	return [...new Set(candidates.map((value) => value.replace(/\/+$/, "")))];
}

function jarvisOriginWindow(): string | undefined {
	const pairId = process.env.JARVIS_PAIR_ID?.trim();
	return pairId ? pairId.slice(0, 8) : undefined;
}

const MODE_MARKER_ANY_RE = /\[MODE:[^\]]+\]/gi;
const MODE_MARKER_PREFIXES = ["[MODE:CHAT]", "[MODE:UNREGISTERED_CODING]", "[MODE:DEEPDIVE]", "[MODE:HEAVY_DEEPDIVE]"];

type EffectiveTurnRoute = "chat" | "chat_control" | "unregistered_coding" | "deepdive" | "heavy_deepdive";
type SidecarContextMode = "chat" | "deepdive";
type JarvisTurnTerminalReason = "stop" | "tool_calls" | "empty" | "no_action" | "error" | "aborted";
type SidecarRouteTurnResponse = {
	ok?: boolean;
	error?: string;
	route?: EffectiveTurnRoute | string;
	confidence?: "high" | "medium" | "low" | string;
	target_project_hint?: string | null;
	project_slug?: string | null;
	code_path_hint?: string | null;
	create_project?: boolean;
	register_project?: boolean;
	critic_mode?: boolean;
	critic_heavy?: boolean;
	expected_action?: "none" | "ask_user" | "spawn_window" | "project_work" | "tool" | string;
	pending_project_decision?: "none" | "confirm" | "decline" | "unclear" | string;
	needs_clarification?: boolean;
	clarification?: string | null;
	reason?: string;
};
type PostTurnRecoveryKind = "none" | "worker_tools_followup" | "verify_incomplete";
type PostTurnRecoveryDecision = {
	kind: PostTurnRecoveryKind;
};
type PostTurnRecoveryInput = {
	workerToolsRetryEligible: boolean;
	modifiedFilePaths?: readonly string[];
	verificationRanThisTurn?: boolean;
	route?: EffectiveTurnRoute;
	provider?: string | undefined;
	verifyContinuationCount?: number;
	maxVerifyContinuations?: number;
};
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
	completion_ledger: string[];
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

const ASK_USER_CUSTOM_LABEL = "Custom answer...";
const ASK_USER_BACK_LABEL = "← Back (previous question)";
const ASK_USER_RECOMMENDED_SUFFIX = " (recommended)";
const ASK_USER_NO_UI_ERROR = "no interactive UI - proceed with recommended defaults";
const ASK_USER_DIRECTIVE_TURN_ERROR =
	"directive turns carry a settled plan; missing info goes back to the dispatcher via handback";
const DIRECTIVE_SPAWN_BLOCK_REASON = "delegated turns must not re-delegate; hand the need back to your dispatcher";
const SECOND_EYES_DIRECTIVE_MARKER = "[CRITIC_REVIEW]";
const SECOND_EYES_MAIN_MARKER = "[CRITIC_MAIN]";
const SECOND_EYES_HEAVY_MARKER = "[CRITIC_HEAVY]";
const SECOND_EYES_PLAN_READY_MARKER = "[CRITIC_PLAN_READY]";
const SECOND_EYES_REMINDER_MARKER = "JARVIS_CRITIC_REVIEW_REQUIRED";
const LEGACY_SECOND_EYES_DIRECTIVE_MARKER = "[SECOND_EYES_REVIEW]";
const LEGACY_SECOND_EYES_MAIN_MARKER = "[SECOND_EYES_MAIN]";
const LEGACY_SECOND_EYES_HEAVY_MARKER = "[SECOND_EYES_HEAVY]";
const LEGACY_SECOND_EYES_PLAN_READY_MARKER = "[SECOND_EYES_PLAN_READY]";
const LEGACY_SECOND_EYES_REMINDER_MARKER = "JARVIS_SECOND_EYES_REVIEW_REQUIRED";
const CRITIC_REVIEW_MARKERS = [SECOND_EYES_DIRECTIVE_MARKER, LEGACY_SECOND_EYES_DIRECTIVE_MARKER] as const;
const CRITIC_MAIN_MARKERS = [SECOND_EYES_MAIN_MARKER, LEGACY_SECOND_EYES_MAIN_MARKER] as const;
const CRITIC_HEAVY_MARKERS = [SECOND_EYES_HEAVY_MARKER, LEGACY_SECOND_EYES_HEAVY_MARKER] as const;
const CRITIC_PLAN_READY_MARKERS = [SECOND_EYES_PLAN_READY_MARKER, LEGACY_SECOND_EYES_PLAN_READY_MARKER] as const;
const WORKER_MODEL_RECOMMENDED_SPEC = "openai-codex/gpt-5.5";
const CRITIC_WORKER_MODEL_RECOMMENDED_SPEC = "anthropic-agent-sdk/claude-opus-4-8";
// M7.9 map-and-step orchestration.
const MAP_DIR_NAME = ".jarvis-map";
const MAP_FILE_NAME = "map.md";
const MAP_LEDGER_FILE_NAME = "ledger.jsonl";
const MAP_FEATURE_ACCEPTANCE_REQUIRED_ERROR =
	"every map feature needs at least one acceptance criterion; checkpoints verify against them verbatim";
const MAP_RUN_ALREADY_OPEN_ERROR =
	"a map run is already open for this window; use append:true to extend it or replace:true to abandon it";
const MAP_CHECKPOINT_EDIT_BLOCK_REASON =
	"map checkpoint turns verify and dispatch only; rejected work goes back via job_send with a reason — never patch it here";
const MAP_DISPATCH_TICKET_REQUIRED_ERROR =
	"map run active: checkpoint dispatches must carry feature_ids so the ticket includes the map's acceptance criteria";
const MAP_SYNTHESIS_BODY_PREFIX = "[MAP SYNTHESIS ";
const ASK_USER_PARAMS = Type.Object({
	questions: Type.Array(
		Type.Object({
			// Aliases (text/prompt/title/label) are normalized at runtime, so the
			// schema stays lean and only advertises the canonical keys.
			question: Type.String(),
			options: Type.Optional(Type.Array(Type.String(), { maxItems: 6 })),
			recommended: Type.Optional(Type.String()),
			allow_custom: Type.Optional(Type.Boolean()),
		}),
		{ minItems: 1, maxItems: 6 },
	),
});

type AskUserQuestionInput = {
	question?: unknown;
	text?: unknown;
	prompt?: unknown;
	title?: unknown;
	label?: unknown;
	options?: unknown;
	recommended?: unknown;
	allow_custom?: unknown;
};

type AskUserQuestion = {
	question: string;
	options: string[];
	recommended?: string;
	allowCustom: boolean;
};

type AskUserAnswer = {
	question: string;
	answer: string | null;
	was_recommended: boolean;
	dismissed?: boolean;
	was_custom?: boolean;
};

type AskUserResult = { ok: true; answers: AskUserAnswer[] } | { ok: false; error: string };

const ASK_USER_ALIAS_KEYS = ["question", "text", "prompt", "title", "label"] as const;

function looksLikeAskUserQuestion(value: unknown): boolean {
	return (
		typeof value === "object" &&
		value !== null &&
		ASK_USER_ALIAS_KEYS.some((key) => typeof (value as Record<string, unknown>)[key] === "string")
	);
}

// Runs as the tool's prepareArguments hook, BEFORE core schema validation.
// The alias normalization inside execute never ran on the live path: the
// agent loop validates against ASK_USER_PARAMS first and `question` is
// required, so an alias-only call (text/prompt/...) bounced off the
// validator before execute. This reshapes those spellings into the
// canonical schema; anything it cannot recognize passes through unchanged
// so the validator still produces its informative error.
function coerceAskUserParams(args: unknown): unknown {
	if (typeof args !== "object" || args === null) return args;
	const record = args as Record<string, unknown>;
	let rawQuestions: unknown[];
	if (Array.isArray(record.questions)) {
		rawQuestions = record.questions;
	} else if (typeof record.questions === "string") {
		rawQuestions = [{ question: record.questions }];
	} else if (looksLikeAskUserQuestion(record)) {
		rawQuestions = [record];
	} else {
		return args;
	}
	const questions = rawQuestions.map((entry) => {
		if (typeof entry === "string") return { question: entry };
		if (typeof entry !== "object" || entry === null) return entry;
		const raw = entry as Record<string, unknown>;
		if (typeof raw.question === "string" && raw.question.trim()) return raw;
		const alias = pickAskUserQuestionText(raw as AskUserQuestionInput);
		return alias ? { ...raw, question: alias } : raw;
	});
	return { ...record, questions };
}

function normalizeAskUserQuestions(input: unknown): { questions?: AskUserQuestion[]; error?: string } {
	const rawQuestions = Array.isArray((input as { questions?: unknown } | undefined)?.questions)
		? ((input as { questions: unknown[] }).questions as unknown[])
		: [];
	if (rawQuestions.length < 1 || rawQuestions.length > 6) {
		return { error: "ask_user requires 1-6 questions" };
	}
	const questions: AskUserQuestion[] = [];
	for (let index = 0; index < rawQuestions.length; index++) {
		const raw = rawQuestions[index] as AskUserQuestionInput | undefined;
		const question = pickAskUserQuestionText(raw);
		if (!question) return { error: `question ${index + 1} must include question text` };
		// Too many options is the only hard error. Fewer than two degrades to a
		// free-form text question instead of rejecting the whole dialog, so a
		// model that omits or under-fills options does not bounce off the schema.
		let options = uniqueCleanStrings(Array.isArray(raw?.options) ? raw.options : []);
		if (options.length > 6) {
			return { error: `question ${index + 1} allows at most 6 options` };
		}
		if (options.length < 2) options = [];
		const recommended =
			typeof raw?.recommended === "string" && raw.recommended.trim() ? raw.recommended.trim() : undefined;
		questions.push({
			question,
			options,
			recommended,
			allowCustom: raw?.allow_custom !== false,
		});
	}
	return { questions };
}

// Accept the canonical `question` key plus the keys models most often guess.
function pickAskUserQuestionText(raw: AskUserQuestionInput | undefined): string {
	for (const value of [raw?.question, raw?.text, raw?.prompt, raw?.title, raw?.label]) {
		if (typeof value === "string" && value.trim()) return value.trim();
	}
	return "";
}

function uniqueCleanStrings(values: unknown[]): string[] {
	const seen = new Set<string>();
	const result: string[] = [];
	for (const value of values) {
		if (typeof value !== "string") continue;
		const clean = value.trim();
		if (!clean || seen.has(clean)) continue;
		seen.add(clean);
		result.push(clean);
	}
	return result;
}

function askUserDefaultAnswer(question: AskUserQuestion, dismissed = true): AskUserAnswer {
	return {
		question: question.question,
		answer: question.recommended ?? null,
		was_recommended: Boolean(question.recommended),
		...(dismissed ? { dismissed: true } : {}),
	};
}

function askUserChoices(question: AskUserQuestion): Array<{ display: string; answer: string; recommended: boolean }> {
	const ordered = [...question.options];
	if (question.recommended) {
		const existingIndex = ordered.indexOf(question.recommended);
		if (existingIndex >= 0) {
			ordered.splice(existingIndex, 1);
			ordered.unshift(question.recommended);
		}
	}
	return ordered.map((answer) => ({
		answer,
		recommended: question.recommended === answer,
		display: question.recommended === answer ? `${answer}${ASK_USER_RECOMMENDED_SUFFIX}` : answer,
	}));
}

function askUserDialogTitle(question: AskUserQuestion, index: number, total: number): string {
	const prefix = `Plan dialogue ${index + 1}/${total}: ${question.question}`;
	if (index > 0) return prefix;
	return `${prefix}\nEsc accepts recommended defaults and finishes.`;
}

function askUserErrorResult(message: string): { content: { type: "text"; text: string }[]; details: AskUserResult } {
	const details: AskUserResult = { ok: false, error: message };
	return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
}

async function runAskUserDialog(
	params: unknown,
	signal: AbortSignal | undefined,
	ctx: ExtensionContext | undefined,
): Promise<AskUserResult> {
	const normalized = normalizeAskUserQuestions(coerceAskUserParams(params));
	if (normalized.error || !normalized.questions) {
		return { ok: false, error: normalized.error ?? "invalid ask_user questions" };
	}
	if (!ctx?.hasUI || typeof ctx.ui?.select !== "function" || typeof ctx.ui?.input !== "function") {
		return { ok: false, error: ASK_USER_NO_UI_ERROR };
	}

	// Answers are kept by index (not appended) so "← Back" can revisit an
	// earlier question and overwrite its answer on the forward re-walk.
	const answers: (AskUserAnswer | undefined)[] = new Array(normalized.questions.length).fill(undefined);
	try {
		let index = 0;
		while (index < normalized.questions.length) {
			const question = normalized.questions[index];

			// Free-form question (no options): collect a typed answer directly.
			// Esc/cancel falls back to recommended defaults like the select path.
			if (question.options.length === 0) {
				const typed = await ctx.ui.input(
					askUserDialogTitle(question, index, normalized.questions.length),
					"Type your answer (Esc for default)",
					{ signal },
				);
				if (typed === undefined) {
					for (let i = index; i < normalized.questions.length; i++) {
						if (!answers[i]) answers[i] = askUserDefaultAnswer(normalized.questions[i]);
					}
					break;
				}
				const answer = typeof typed === "string" ? typed.trim() : "";
				answers[index] = answer
					? { question: question.question, answer, was_recommended: false, was_custom: true }
					: askUserDefaultAnswer(question);
				index++;
				continue;
			}

			const choices = askUserChoices(question);
			const labels = choices.map((choice) => choice.display);
			if (question.allowCustom) labels.push(ASK_USER_CUSTOM_LABEL);
			// Offer "← Back" on every question after the first so a mis-pick
			// upstream is fixable without restarting the whole dialog.
			if (index > 0) labels.push(ASK_USER_BACK_LABEL);
			const selected = await ctx.ui.select(
				askUserDialogTitle(question, index, normalized.questions.length),
				labels,
				{
					signal,
				},
			);
			if (selected === undefined) {
				// Esc accepts recommended defaults for the current question and
				// every still-unanswered one, then finishes.
				for (let i = index; i < normalized.questions.length; i++) {
					if (!answers[i]) answers[i] = askUserDefaultAnswer(normalized.questions[i]);
				}
				break;
			}

			if (index > 0 && selected === ASK_USER_BACK_LABEL) {
				index--;
				continue;
			}

			if (question.allowCustom && selected === ASK_USER_CUSTOM_LABEL) {
				const custom = await ctx.ui.input(`Custom answer for: ${question.question}`, "Type your answer", {
					signal,
				});
				const answer = typeof custom === "string" ? custom.trim() : "";
				answers[index] = answer
					? { question: question.question, answer, was_recommended: false, was_custom: true }
					: askUserDefaultAnswer(question);
				index++;
				continue;
			}

			const choice = choices.find((item) => item.display === selected);
			answers[index] = {
				question: question.question,
				answer: choice?.answer ?? selected,
				was_recommended: choice?.recommended === true,
			};
			index++;
		}
	} catch {
		return { ok: false, error: ASK_USER_NO_UI_ERROR };
	}

	const resolved = normalized.questions.map((question, i) => answers[i] ?? askUserDefaultAnswer(question));
	return { ok: true, answers: resolved };
}

const PLAN_DIALOGUE_PROMPT = `
PLAN DIALOGUE: when this turn kicks off a NEW user-facing artifact, ALWAYS
call ask_user once before build/recon (and before delegation on heavy turns).
This is the plan step; run it even when the request seems clear, and pre-fill
recommended answers from what the user already specified rather than skipping.
Ask up to 6 self-composed questions (one per genuine fork, no filler): basics (scope/platform/storage/stack) plus
design direction. For any visual artifact ALWAYS include one design question
whose recommended option says you study current web design/UX trends for it
(alternatives: user's own reference, or a minimal default) so the user sees and
consents to the web-learning step. Skip ONLY on decide/just-make-it,
same-project follow-up, bug fix, or non-user-facing work. Incoming
directive/job/gan dispatch turns NEVER re-ask — the plan already happened in
the dispatcher's window; if a critical decision is genuinely missing, ask the
dispatcher via job_send/report handback instead. If ok:false, use recommended
defaults and continue. The design answer seeds the recon queries below; carry
answers into plan and NOW.
`.trim();

// Universal clarify-before-act directive. Appended to EVERY route's mode prompt
// (chat..heavy) by modePromptForRoute, so the model surfaces the choices it is
// about to make on the user's behalf instead of guessing silently. The artifact-
// build specialization (PLAN_DIALOGUE_PROMPT) still rides the coding routes on
// top of this. Calibrated by an information-value gate so it does not become a
// robot that asks about everything (Jun, 2026-06-23).
const CLARIFY_DIRECTIVE_PROMPT = `
[CLARIFY BEFORE YOU ACT]
Users routinely under-specify. On any non-trivial request you are silently about
to make choices for the user (scope, style, stack, blast radius, edge cases).
Surface those instead of guessing. Ask yourself: "What am I about to decide FOR
the user where I am genuinely uncertain AND they would plausibly have a
preference?" Those are the forks.

- A NEW user-facing artifact (app, game, page, UI, dashboard, tool) ALWAYS has
  real forks -- visual style, scope, features, stack. A FAMILIAR concept
  ("tetris", "todo app", "calculator") is NOT "clear": its design is still
  unchosen. Call ask_user FIRST with up to ~10 concrete options BEFORE the first
  file or design recon. Do NOT pick the style/scope/features yourself and
  proceed -- choosing for the user and saving a Design Brief without asking is
  the exact failure to avoid. (The plan-dialogue rules below detail the questions.)
- DESTRUCTIVE actions that are BULK or IRREVERSIBLE (delete/unregister/clear many
  items, "delete all", drop data, deploy, overwrite, spend) ALWAYS require an
  ask_user confirmation BEFORE executing -- EVEN when the user already named the
  scope (e.g. "delete ALL", "지워줘", "모두 지워"). Naming a scope is NOT the same
  as confirming the destruction. First inspect what exists, then ask with the
  exact blast radius and count ("This will unregister ALL 3 projects: tetris,
  todo, foo"), and present EACH distinct scope as its OWN selectable option --
  never a yes/no, which hides the middle scope and reads as ambiguous. For a
  registry deletion that means three options: (1) remove the registry entry
  only, keep the files; (2) remove the entry AND delete the workspace
  folder/files; (3) cancel. Act only after an explicit choice. Never delete
  first and report after.
- For other EXPENSIVE or IRREVERSIBLE guesses (broad refactor, risky change):
  call ask_user FIRST with the real forks -- include good options the user may
  not have thought of; that surfacing is the point.
- If the request is clear and low-risk AND is NOT a fresh artifact build (a
  bounded edit, a question, a small known change): DO it, then add a one-line
  assumption receipt of any forks you chose ("edited with vanilla JS -- say if
  you want different").
- If there is no real fork: just act.

Information-value gate: only raise a fork whose answer would CHANGE what you do.
If you would act the same way regardless of the answer, do not ask. Never
manufacture questions to seem thorough. Never re-ask on an incoming
directive/job/handback turn -- the plan already happened upstream.
`.trim();

// Mandatory ask_user gate for a NEW user-facing artifact turn (Jun, 2026-06-24).
// The anthropic-agent-sdk regime is an autonomous-completion loop that barrels to
// "finish the task" and skips the general CLARIFY directive; codex pauses only
// because pi owns its loop. This directive is set on a classifierNewProject turn
// and pushed to the FRONT of the system prompt (salient); the sidecar PreToolUse
// hook code-enforces it via the [JLC:NEW_ARTIFACT_ASK_USER_GATE] marker (file/shell
// tools denied until ask_user fires). Keep the marker token verbatim.
const NEW_ARTIFACT_ASK_USER_GATE_PROMPT = `
[MANDATORY FIRST ACTION] [JLC:NEW_ARTIFACT_ASK_USER_GATE]
This turn starts a NEW user-facing artifact. Your FIRST tool call MUST be
ask_user, surfacing the real design / scope / feature / stack forks -- a familiar
concept like "tetris" or "todo" is NOT pre-decided; its visual style, scope, and
features are unchosen. Pre-fill recommended options from anything the user already
specified, but still ask. Do NOT call write/edit/bash or create any file before
ask_user returns. Building before asking is a turn failure.
`.trim();

// Chat is the entry mode for every normal turn. Keep this prompt small: the
// chat model decides whether to answer directly or escalate into project work.
const CHAT_MODE_PROMPT = `
[CHAT ENTRY MODE]

First line must be exactly one marker:

  [MODE:CHAT]
  [MODE:UNREGISTERED_CODING]
  [MODE:DEEPDIVE]
  [MODE:HEAVY_DEEPDIVE]

Use [MODE:CHAT] for casual talk, recall, acknowledgments, and off-project
questions; answer in <=2 short sentences. For a bounded action you may use your
available tools directly (pi basics read/write/edit/bash/grep/find/ls, plus
registry management, recall, docs, web, and managed_process).

Worker tools: new worker = spawn_window; existing worker = list_windows then
job_send; do not duplicate. If needed worker tools are absent, emit exactly:
[MODE:CHAT]
${WORKER_TOOLS_NEEDED_MARKER}
No other text.

Route notes: Use [MODE:UNREGISTERED_CODING] for explicit external/unregistered
file/code work. Use [MODE:DEEPDIVE] for focused registered project work,
localized bug symptoms, setup, files, commands, or implementation. Use
[MODE:HEAVY_DEEPDIVE] only for broad/high-risk project-wide work. Default to
[MODE:CHAT] when ambiguous.
`.trim();

const CHAT_CONTROL_MODE_PROMPT = `
[CHAT CONTROL MODE]

First line must be exactly [MODE:CHAT].

This is still chat mode for memory/context, but the user asked for a bounded
control or action. Use the tools needed for the request, then answer briefly.

Available: ask_user, worker/window messaging or spawn, model/window settings,
web_search/web_fetch, project registry (register/switch/unregister_project),
update_jarvis_md, recall_turns/retrieve_output, docs_search/package_info,
managed_process, and pi's basic tools (read/write/edit/bash/grep/find/ls) for a
bounded file or command action the user asked for.

Keep it bounded: you MAY manage the project registry (register_project,
switch_project, unregister_project) and update JARVIS.md when the user asks for a
bounded change. Do not start a substantial REGISTERED project build or set up
maps/delegation here.

Registry cleanup and workspace deletion are bounded actions you perform HERE,
never a reason to bail or escalate. To remove registrations -- including a bulk
"delete everything / reset all" -- call unregister_project (once per entry, or
for every registration). To delete the workspace folder(s) the user confirmed
removing, use bash (rm the confirmed path). You HAVE unregister_project and bash
in this turn: never reply that the tool is "not exposed", that you "cannot delete
in chat_control mode", or that deletion needs a different/"delete-capable" mode.
Just call the tools and report what was actually removed.
`.trim();

const UNREGISTERED_CODING_MODE_PROMPT = `
[UNREGISTERED CODING MODE]

EVERY reply MUST start with exactly [MODE:UNREGISTERED_CODING] on the first
line before any text or tool call.

This turn is standalone work on explicit external/unregistered material: a
path, pasted code, one-off script, command, or file/folder outside a registered
JARVIS project. Read, analyze, edit, and run focused checks when the user asks.

Do not copy project-memory persistence rules into unregistered_coding.
Unregistered may do plan/design/recon as turn-local working context, but must
not call switch_project, register_project, unregister_project, or
update_jarvis_md. Do not persist the turn with project_path and do not register
external folders unless the user explicitly asks for JARVIS project
registration or confirms your registration question.

${PLAN_DIALOGUE_PROMPT}

UNREGISTERED DESIGN RECON: when this turn kicks off a new user-facing artifact
(app, game, landing page, dashboard, UI component) built in unregistered
material, recon before the first artifact file write unless the user says to
skip, the task is a bug fix/refactor/non-user-facing utility, or the user chose
their own reference over web research. You may read the first 2KB of
~/.jarvis-code/design-taste.md because it is global user taste, not project
memory. If web_search/web_fetch are available, use up to 4 searches and up to 2
fetches. Distill a <=12-line turn-local Design Brief and use it for this turn
only. Never save it to JARVIS.md.

Verification: after editing external files, run the cheapest relevant syntax,
type, test, build, smoke, or direct executable probe you can run safely. If no
check is practical, say exactly why and what risk remains.
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

                    Todo tracking:
                    - For multi-step coding work, use the todo tool to keep
                      your task checklist current. The tool replaces the full
                      list each call; keep exactly one item in_progress unless
                      the work is paused or complete.

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

                    ${PLAN_DIALOGUE_PROMPT}

                    PROJECT DESIGN RECON (clarify-driven, NOT a forced rule):
                    Design recon is OPT-IN and the user's answer is supreme.
                    NEVER web_search/web_fetch for recon, and skip recon
                    entirely, when the user opts out, says keep it
                    simple/minimal/basic ("간단한", "그냥", "no research",
                    "웹조사 안함", "리서치 없이"), or does not ask for design
                    polish. Also skip when "## Design Brief" already exists, for
                    bug fixes/refactors/existing-code edits, or internal
                    scripts/CLI utilities/non-user-facing work.
                    For [MODE:DEEPDIVE] and [MODE:HEAVY_DEEPDIVE], when this turn
                    kicks off a NEW user-facing artifact (new app, game, landing
                    page, dashboard, or UI component) with no existing Design
                    Brief, your clarify/ask_user should surface the choice --
                    "research current design/UX trends for a polished look, or
                    keep it simple?" -- then FOLLOW the user's answer. Do recon
                    ONLY when the user opts in or asks for a polished look.
                    When recon runs, use existing tools only: web_search for 1-2 visual trend
                    searches like "<artifact type> UI design trends <current
                    year>" and 1-2 UX convention searches like "<artifact type>
                    UX patterns best practices"; total search cap 4. Optionally
                    web_fetch 1-2 discovered references; fetch cap 2. If search
                    or fetch is unavailable or empty, do not block the build;
                    save a brief line saying "recon degraded: search unavailable"
                    and continue. If ~/.jarvis-code/design-taste.md
                    exists, read only its first 2KB, merge it into the brief,
                    prefer it over search on conflicts, and never create or edit
                    that file. Distill, never dump source text. Save exactly one
                    Design Brief with update_jarvis_md field DESIGN_BRIEF before
                    writing artifact files. Keep the entire section <= 12 lines:
                    - Palette: ...
                    - Typography: ...
                    - Layout: ...
                    - Motion: ...
                    - UX conventions: ...
                    - Avoid: ...
                    - Sources: 2-3 domain names only, no full URLs.
                    If starting a new project, register or switch the project
                    path first so update_jarvis_md can write active JARVIS.md,
                    then run recon and save the brief.
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

                    CANONICAL SECTIONS (8, fixed — do not invent others except
                    the special DESIGN_BRIEF section required by design recon):
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
                    Memory writes are bookkeeping: never narrate them in your
                    reply (no "memory updated:" line). The answer is the answer;
                    the sidecar tracks what changed.
Default CHAT when truly ambiguous, but NEVER skip the first-line marker.
Skipping the marker is a worse error than picking the wrong mode — the
marker is what the sidecar reads; the mode body only refines behavior.
`.trim();

const CHAT_ROUTE_PROMPT = `
[ROUTE:CHAT_ENTRY]

Start light. Answer ordinary chat directly. Escalate only when the request
actually needs unregistered coding, registered project deepdive, or heavy
project-wide work. Localized bug symptoms are deepdive, not heavy.
Do not update JARVIS.md unless escalated into registered project deepdive; do
not register external folders without explicit user registration intent/confirm.
`.trim();

const CHAT_CONTROL_ROUTE_PROMPT = `
[ROUTE:CHAT_CONTROL]

This turn is chat-visible control/web work. Keep the required first-line marker
as [MODE:CHAT]. Use the control/web tool pocket already exposed to this turn.
Do not perform file/project mutation, shell commands, project registration,
project switching, map/delegation setup, or project memory writes.
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

const HEAVY_DEEPDIVE_OVERLAY_PROMPT = `
[HEAVY DEEPDIVE OVERLAY]

Use this overlay only for broad/high-risk registered project work: current
implementation analysis before planning, full redesign/rework, broad structure
review, multi-file refactor, architecture/game-loop/state/input/rendering/
asset/build inspection, project-wide regression/performance work, or adding
systems such as sound effects/BGM. Do not use HEAVY for localized bug symptoms
unless the user also asks for broad redesign, project-wide analysis, or
cross-system refactoring.

Budget rule: every tool call must either gather multiple pieces of necessary
information, apply a complete coherent change, or verify the completed work.
Single-purpose exploratory tool calls are discouraged unless the task is
genuinely ambiguous. Think hard, parallelize observation, patch once, verify
once.

DELEGATION DISPATCH CONTRACT: delegation/map initiation is HEAVY-only. For
build delegation via spawn_window/job_send/send_directive, relay source text
only: quote the user's original request and raw plan Q/A; include completion/
handback format and project path/registration. Do NOT invent stack,
architecture, file layout, implementation order, or code sketches; constraints
from user words are valid goals and stay quoted. For design-led delegation,
relay raw design answers; the worker runs web_search/web_fetch, distills/saves
Design Brief, or inherits an existing "## Design Brief". Review/repair
delegation is opposite: give lenses/checklists; fixes should be
symptom+diagnosis unless you truly know the patch.

Delegated builds default to WHOLE delegation: dispatch the full build to ONE
worker that keeps cohesion end to end. The worker self-checks its build runs
before handing back; tell it so in the ticket. You then verify the handback
once — confirm it is real, do not re-run the worker's own checks. No double
verification. Only when a build is too large for one worker context should you
persist a feature map with map_create and dispatch features in slices. Surface
the delegation depth in the plan: [A] you write the core/skeleton directly and
delegate the rest, or [B] delegate the whole build.

Focused deepdive must not initiate delegation/map. Active worker/directive
turns may still use handback bus tools; do not confuse handback with new
delegation initiation.
`.trim();

// M7.9: checkpoint turns are ticket processing, not exploration — keep this
// far lighter than DEEPDIVE (R3-3 token diet).
const MAP_CHECKPOINT_MODE_PROMPT = `
[MODE — MANDATORY FIRST-LINE TOKEN]
EVERY reply MUST start with [MODE:DEEPDIVE] alone on the very first line,
before any text or tool call — the sidecar gates memory I/O on it.

[MAP CHECKPOINT TURN]
A worker handed back the feature(s) listed under MAP STATUS below. This is
ticket processing, not exploration, and never implementation.

1. VERIFY — compare the handback against each feature's acceptance criteria,
   verbatim. Run the SINGLE cheapest decisive check per criterion (syntax
   check, targeted grep, one test) — the full build/launch pass belongs to the
   synthesis turn. Reading code alone never justifies a pass (Iron Law), but
   every extra tool call re-sends the whole context: verify cheap, verify once.
2. VERDICT — call feature_verdict once per arriving feature. pass requires the
   evidence you ran; reject requires a concrete reason — the reason is your
   only intent channel to the worker, so say what is wrong and what right
   looks like.
3. NEXT STEP — conditioned on the map and the verified artifacts: re-dispatch
   rejected features (job_send with feature_ids; the rejection reason rides
   the ticket), then slice the next 1-N unchecked features and dispatch them
   (batch small ones). Follow any escalation line in MAP STATUS.

Hard rules: never edit/write files here; never re-ask the user — the plan is
settled; end the turn after dispatching. The next checkpoint arrives
automatically when a worker hands back.
`.trim();

const MAP_SYNTHESIS_PROMPT = `
[MODE — MANDATORY FIRST-LINE TOKEN]
EVERY reply MUST start with [MODE:DEEPDIVE] alone on the very first line,
before any text or tool call — the sidecar gates memory I/O on it.

[MAP SYNTHESIS TURN]
Every feature on the map passed its checkpoint. Run the final cross-feature
verification yourself now: build, run the tests, launch the artifact's happy
path. Then report to the user: one line per feature with its evidence, any
deviations from the map, and where the map/ledger artifacts live. Persist
durable notes via update_jarvis_md. Your final text IS the user report —
write it for the user, not for a dispatcher.
`.trim();

// Whole-delegation end gate. Reuses the synthesis machinery (prompt swap +
// visibility) for builds handed to ONE worker (no feature map). Decision
// (2026-06-14, Jun): the worker self-verifies its build before handback, so
// the main does NOT re-run a heavy behavioral verification — that is wasteful
// double-checking, and the real behavioral playtest is the user's job. The
// main does a near-free real-ness check (did the worker actually produce what
// it claims), hands the user a short playtest checklist of the signature
// behaviors to confirm, and closes. A heavy automated playtest (headless
// browser / vision) is parked for an ~unlimited-token future; see
// _internal/PLAYTEST_END_GATE_MECHANISM_COMPARISON_2026_06_13.md.
const WHOLE_DELEGATION_END_GATE_PROMPT = `
[MODE — MANDATORY FIRST-LINE TOKEN]
EVERY reply MUST start with [MODE:DEEPDIVE] alone on the very first line,
before any text or tool call — the sidecar gates memory I/O on it.

[WHOLE-DELEGATION END GATE]
A worker handed back a build you delegated whole and already self-checked it
runs. Do NOT re-run a heavy verification or re-play the artifact — that double
work is wasteful and the real behavioral playtest is the user's. Once:
1. REAL-NESS CHECK ONLY — one near-free check that the worker actually produced
   what it claims (the named files/artifact exist, changed this cycle), not a
   blind trust of the handback prose.
2. PLAYTEST CHECKLIST — report to the user the few signature behaviors they
   should confirm by running it themselves (e.g. for a Pang clone: "balloons
   bounce to a size-locked height; a hit splits one into two arcing
   children"), plus where the artifact lives and how to run it.
Then close the job (job_close); only if the real-ness check failed, send the
worker ONE concrete fix directive (job_send). Do not re-ask the user and do not
patch it yourself. Your final text IS the user report.
`.trim();

const SECOND_EYES_MODE_PROMPT = `
[MODE — MANDATORY FIRST-LINE TOKEN]
EVERY reply MUST start with [MODE:DEEPDIVE] alone on the very first line,
before any text or tool call — the sidecar gates memory I/O on it.

[CRITIC MODE]
You are an independent Critic Mode reviewer, not the implementer. The main
window owns planning decisions, implementation, fixes, and user reporting. Your
job is to critique the main window's own plan draft before implementation and
later review the implemented artifact for defects the implementer may have
missed.

Hard rules:
- Review-only. You may read/search files and run bounded verification commands
  (lint/test/build/static checks) to gather evidence, but do not modify source
  files, install/update dependencies, run mutation commands, update project
  memory, switch/register projects, create maps, or open more workers.
- Do not implement, patch, or tell the user you changed anything. Send findings
  back to the main window; the main window applies any fixes.
- Do not own trend recon, architecture selection, product planning, or feature
  design. If the directive lacks a main-window draft, hand back a request for
  that draft instead of inventing one.
- Use job_send for handback when a job header exists. Every handback to the main
  window MUST start with ${SECOND_EYES_MAIN_MARKER}.
- Do not expand scope or propose new features. Compare only against the user's
  request, the directive, visible artifacts, and the reported checks.
- Must-fix is reserved for real defects: broken behavior, unmet requirements,
  serious UX/accessibility problems, misleading output, or clear edge-case
  failure. Taste-only changes and speculative refactors are not Must-fix.

Plan critique lens, before implementation:
- Review the main draft only. Challenge missing requirements, risky
  architecture, test gaps, UX traps, and user choices that genuinely need
  ask_user.
- Keep it short and actionable. The main window may ask at most one more
  critique round before implementation.

Implementation review lenses:
- Requirements: missing, contradicted, or scope-drifted behavior.
- UX/product: confusing flows, broken responsive layout, inaccessible controls,
  text overlap/overflow, weak first-screen signal for visual artifacts.
- Code: fragile state, obvious bugs, import/link mistakes, unhandled edge cases,
  maintainability risks that can cause real defects.

For a plan handback, return these sections:
${SECOND_EYES_MAIN_MARKER}
## Plan Verdict
## Must Clarify
## Risks
## Corrections

For an implementation review handback, return these sections:
${SECOND_EYES_MAIN_MARKER}
## Must-fix
## Should-fix
## Backlog

Each finding should include evidence and a concise fix direction. When done,
hand the review back to the main window with the job/directive bus named in the
job header. That handback is the only permitted write-like action.
`.trim();

const SECOND_EYES_MAIN_MODE_PROMPT = `
[MODE — MANDATORY FIRST-LINE TOKEN]
EVERY reply MUST start with [MODE:DEEPDIVE] alone on the very first line,
before any text or tool call — the sidecar gates memory I/O on it.

[CRITIC MODE MAIN]
You are the main implementer/orchestrator. The Critic Mode worker is review-only:
it may inspect and run bounded verification, but never edits files, runs
mutation commands, or reports completion to the user. You own all
requirements intake, trend recon, architecture draft, implementation, fixes,
final checks, and final user report.

Protocol:
- This is project work, never chat. If no active project context is injected,
  first call switch_project or register_project from the directive's project
  name/path before doing analysis, implementation, or user reporting.
- Before the first worker handoff, gather user choices when needed, perform any
  required recon yourself, and write a concise main-window plan draft. The
  worker critiques that draft; it must not become the planner.
- If the worker returns plan critique, either send one more concise revised
  main draft to the same worker, call ask_user when a real user
  preference/choice controls quality, or proceed to implementation. Do not
  exceed two plan critique rounds.
- Implement in this main window. Do cheap mechanical checks yourself.
- After implementation, send the same worker a review request with job_send.
  The worker reviews only and returns Must-fix / Should-fix / Backlog.
- Apply confirmed Must-fix items yourself. Never ask the worker to implement,
  fix, edit, patch, write files, or run mutation commands.
- Allow at most two fix/review cycles. If unresolved after that, close/escalate
  and report the remaining disagreement/blocker to the user. If resolved, close
  done and report completion.
- Do not spawn additional workers or create maps for this Critic Mode job.
`.trim();

function secondEyesModePrompt(prompt: string): string {
	const marker = currentSecondEyesHeavy() ? "[MODE:HEAVY_DEEPDIVE]" : "[MODE:DEEPDIVE]";
	const withMarker = prompt.replace(
		"EVERY reply MUST start with [MODE:DEEPDIVE] alone",
		`EVERY reply MUST start with ${marker} alone`,
	);
	return currentSecondEyesHeavy() ? `${withMarker}\n\n${HEAVY_DEEPDIVE_OVERLAY_PROMPT}` : withMarker;
}

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
let interruptCheckpointSavedThisTurn = false;
let interruptInputUnsubscribe: (() => void) | undefined;
let turnCheckpointScope: CheckpointScope | undefined;
let sidecarHealthy = false;
let currentMode: "chat" | "deepdive" = "chat";
let currentRoute: EffectiveTurnRoute = "chat";
let currentTodoList: JarvisTodoItem[] = [];
let readBeforeEditRegistry = new Map<string, number>();
let deepdiveThinkingPreference: SupportedThinkingLevel | undefined;
let deepdiveThinkingPreferenceLoaded = false;
let subagentModelUserSet = false;
let subagentModelUserSetLoaded = false;
let suppressNextThinkingPreferenceSave: SupportedThinkingLevel | undefined;
let coldStartNoticeShown = false;
let startupContextWarmupFinished = false;
let startupContextWarmupPromise: Promise<void> | undefined;
let setupRequired = false;
let currentWindowLabel: string | undefined;
let lastTurnPromptSnapshot: TurnPromptSnapshot | undefined;
let trimBeforeTokensSum = 0;
let trimAfterTokensSum = 0;
let lastToolSchemaTokens = 0;
let lastProviderCallRoute: EffectiveTurnRoute | undefined;
let lastProviderToolsBeforeFilter: string[] = [];
let lastProviderToolsAfterFilter: string[] = [];
let lastProviderActionIntentMatch = false;
let lastProviderChatFilterApplied = false;
let lastProviderRoutePromotedByClassifier = false;
let lastRouteClassifierDecision: SidecarRouteTurnResponse | undefined;
let lastRouteClassifierActionIntent = false;
let expectedToolActivityThisTurn = false;
let routePromotedByClassifierThisTurn = false;
// Set on a classifierNewProject route decision; consumed (front-pushed + cleared)
// in the before-provider system-prompt assembly so the new-artifact ask_user gate
// is salient. Errs toward asking if it ever leaks (Jun: ask_user aggressively).
let pendingNewArtifactAskUserGate = false;
let workerToolsRetryInFlight = false;
let workerWindowContextInjectedThisTurn = false;
let lastTurnStartedAtMs: number | undefined;
let lastProviderStartedAtMs: number | undefined;
let providerCallCountThisTurn = 0;
let activeModelProviderThisTurn: string | undefined;
let agentTurnActive = false;
let pendingDirectiveAutoTurn: SidecarDirectiveItem | undefined;
let activeDirectiveTurn: SidecarDirectiveItem | undefined;
let directiveTurnBusReplySent = false;
let pendingDirectiveReports: SidecarDirectiveItem[] = [];
let directivePollTimer: ReturnType<typeof setInterval> | undefined;
let directiveSensorRunning = false;
let controlBridgePollTimer: ReturnType<typeof setInterval> | undefined;
let controlBridgeSensorRunning = false;
const directiveKnownQueueState = new Map<SidecarDirectiveKind, { mtimeNs: number; size: number }>();
// M7.9 map-and-step run state. The ledger.jsonl on disk is the source of
// truth; this mirror is rebuilt from it on session start (pointer file).
type MapZone = "feature" | "skeleton";
type MapFeatureStage = "normal" | "escalated" | "main_direct";
type MapFeatureStatus = "todo" | "dispatched" | "passed";
type MapFeatureState = {
	id: string;
	title: string;
	summary?: string;
	zone: MapZone;
	acceptance: string[];
	status: MapFeatureStatus;
	rejections: number;
	stage: MapFeatureStage;
	lastRejectReason?: string;
};
type MapRunPhase = "stepping" | "synthesis" | "complete";
type ActiveMapRun = {
	mapId: string;
	title: string;
	projectPath: string;
	features: Map<string, MapFeatureState>;
	jobFeatures: Map<string, string[]>;
	phase: MapRunPhase;
	ledgerSeq: number;
};
// Result shapes for map_create / feature_verdict. Promoted to module scope so the
// shared run* helpers, the registerTool execute fns (regime A), and the
// control-bridge branches (regime B, anthropic-agent-sdk) all return the same
// shape — one implementation, no producer/consumer fork.
type MapCreateResult = {
	ok: boolean;
	error?: string;
	map_id?: string;
	project_path?: string;
	features?: Array<{ id: string; title: string; status: MapFeatureStatus }>;
};
type FeatureVerdictResult = {
	ok: boolean;
	error?: string;
	feature_id?: string;
	verdict?: "pass" | "reject";
	rejections?: number;
	stage?: MapFeatureStage;
	map_phase?: MapRunPhase;
	next?: string;
};
let activeMapRun: ActiveMapRun | undefined;
let mapIdCounter = 0;
// Set only for the lifetime of a checkpoint turn (a job handback arriving for
// dispatched map features); cleared everywhere activeDirectiveTurn is cleared.
let activeMapCheckpointTurn: { jobId: string; featureIds: string[] } | undefined;
let activeMapSynthesisTurn = false;
// Set only for the lifetime of a whole-delegation end-gate turn: a worker
// handback (job phase "review", orchestrator side) for a build that was NOT
// sliced into a feature map. Reuses the synthesis machinery (prompt swap +
// visibility) without touching the directive/job bus; cleared everywhere
// activeDirectiveTurn is cleared.
let activeEndGateTurn = false;
// Set only for a worker turn whose directive body starts with the Critic review
// marker; cleared everywhere activeDirectiveTurn is
// cleared. This is an internal review-only worker mode, not a user route.
let activeSecondEyesReviewTurn = false;
// Set only for the main/orchestrator turn receiving a Critic Mode handback.
// This turn may edit files; it must not spawn more workers because the
// existing Critic Mode worker remains review-only.
let activeSecondEyesMainTurn = false;
let activeSecondEyesHeavyTurn = false;
let secondEyesRequestedThisTurn = false;
let secondEyesReviewSpawnedThisTurn = false;
let secondEyesReminderInjectedThisTurn = false;
let askUserIssuedThisProviderCall = false;
// Set by the last pass verdict; consumed exactly once in agent_end so a
// retried turn cannot double-post the synthesis self-directive.
let pendingMapSynthesisPost = false;
let lastUserActivityAtMs = 0;
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
let subturnLedgerLines: string[] = [];
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
let verificationRanThisTurn = false;
let verifyContinuationCount = 0;
let turnJarvisMdUpdated = false;
let turnReadCompressionEditTargetPaths = new Set<string>();
let turnReadCompressionKeysByPath = new Map<string, string[]>();
let subturnPcCeilingReportStopActive = false;
let subturnPcCeilingProviderCall: number | undefined;
let subturnLockedResourceActionCounts = new Map<string, number>();
let subturnLockedResourceRecordedCallIds = new Map<string, { key: string; count: number; line: string }>();
let subturnLockedResourceReportStopActive = false;
let subturnLockedResourceReportStopRecord: { key: string; count: number; line: string } | undefined;
let subturnFailingCommandCounts = new Map<string, { count: number; stamp: number }>();
let subturnRepairActivityStamp = 0;
let subturnFailingCommandRecordedCallIds = new Map<string, { key: string; count: number; line: string }>();
let subturnRepeatedFailureReportStopActive = false;
let subturnRepeatedFailureReportStopRecord: { key: string; count: number; line: string } | undefined;

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

type ManagedProcessRecord = {
	id: string;
	command: string;
	args: string[];
	cwd: string;
	pid: number;
	ownerPid: number;
	startedAt: string;
	// OS-level start identity for the PID (opaque token). Lets us detect PID
	// reuse before killing: if the live PID's token no longer matches the one we
	// captured at spawn, the OS recycled the number onto an unrelated process and
	// the record no longer owns it. Undefined when the platform reader failed.
	procStartToken?: string;
	// OS-level start identity of the OWNER (the JARVIS process that spawned this).
	// Stale cross-instance cleanup uses it to confirm a still-alive ownerPid is
	// really the same owner and not a recycled PID, otherwise a dead owner whose
	// PID was reused would make us skip cleanup forever and leak the orphan.
	ownerProcStartToken?: string;
	// Unix process-group id (group leader == pid for our detached children).
	// Recorded for diagnostics and the cross-platform process-management contract;
	// undefined on Windows where Job Object / tree-kill handle containment.
	pgid?: number;
	logPath?: string;
	healthUrl?: string;
	child?: ChildProcess;
};

type ManagedProcessStateRecord = Omit<ManagedProcessRecord, "child">;

const managedProcesses = new Map<string, ManagedProcessRecord>();
let managedProcessStaleCleanupDone = false;
// v2 adds procStartToken + pgid. Older v1 files simply lack the token, which the
// reuse guard treats conservatively (cannot prove identity -> do not auto-kill).
const MANAGED_PROCESS_STATE_VERSION = 2;

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

function encModelFooterLabel(spec: string | undefined): string {
	const trimmed = (spec ?? "").trim();
	const slash = trimmed.indexOf("/");
	const provider = slash > 0 ? trimmed.slice(0, slash) : "";
	const model = slash > 0 ? trimmed.slice(slash + 1) : trimmed;
	return model ? (provider ? `(${provider}) ${model}` : model) : "";
}

function setEncModelStatus(ctx: ExtensionContext, spec: string | undefined): void {
	const modelLabel = encModelFooterLabel(spec);
	try {
		ctx.ui.setStatus("jlc-enc-model", modelLabel ? `${ANSI_PINK}${modelLabel}${ANSI_RESET}` : undefined);
	} catch {
		// ignore
	}
}

function renderEncBadge(ctx: ExtensionContext, s: EncSummary): void {
	stopEncodingStatus();
	const tok = s.enc_out ?? 0;
	const sec = s.enc_seconds ?? 0;
	const tokStr = tok >= 1000 ? `${(tok / 1000).toFixed(1)}K` : String(tok);
	const label = `enc:${tokStr}t/${sec.toFixed(1)}s`;
	try {
		ctx.ui.setStatus("jlc-enc", `${ANSI_PINK}${label}${ANSI_RESET}`);
	} catch {
		// ignore
	}
	// Mirror the chat footer shape "(provider) model" on a second
	// status key so footer.ts can right-align it under the chat model line
	// while the meter label stays left-aligned. Encoder reasoning is always
	// disabled operationally, so do not render effort text here.
	setEncModelStatus(ctx, s.enc_model_spec);
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

// Per-turn routine progress (the memory read/analyze/update cycle and its
// "✓ updated" confirmations) repeats every turn and carries no signal —
// surfacing it makes the core memory loop look like repetitive waste to users.
// Hidden by default; set JARVIS_PROGRESS_NOTICES=1 to surface it. Errors and
// warnings are never gated by this.
function jarvisProgressVisible(): boolean {
	const raw = (process.env.JARVIS_PROGRESS_NOTICES ?? "").trim().toLowerCase();
	return raw === "1" || raw === "true" || raw === "on" || raw === "yes";
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
			// Warmup stays silent (Jun decision, 2026-06-07) — background work is enough; only failure/degraded states notify.
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
	// If the user types during warmup, quietly wait for it to finish, then continue.
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

function directiveWindowLabel(window: string | undefined): string {
	const value = String(window ?? "").trim();
	return value || "external";
}

function directiveGanBadge(item: SidecarDirectiveItem | undefined): string {
	const gan = item?.gan;
	if (!gan?.gan_id) return "";
	const round = typeof gan.round === "number" && Number.isFinite(gan.round) ? Math.floor(gan.round) : "?";
	const status = String(gan.status ?? "open").trim();
	return `[GAN ${gan.gan_id} r${round}/3${status && status !== "open" ? ` ${status}` : ""}] `;
}

function jobCycleCapFromEnv(): number {
	const raw = process.env.JARVIS_JOB_CYCLE_CAP?.trim();
	const parsed = raw ? Number(raw) : 3;
	return Number.isFinite(parsed) && parsed >= 1 ? Math.floor(parsed) : 3;
}

function directiveJobBadge(item: SidecarDirectiveItem | undefined): string {
	const job = item?.job;
	if (!job?.job_id) return "";
	const cycle = typeof job.cycle === "number" && Number.isFinite(job.cycle) ? Math.floor(job.cycle) : "?";
	const phase = String(job.phase ?? "").trim() || "job";
	const status = String(job.status ?? "open").trim();
	const cap = jobCycleCapFromEnv();
	return `[JOB c${cycle}/${cap} ${phase}${status && status !== "open" ? ` ${status}` : ""}] `;
}

function directiveSessionBadge(item: SidecarDirectiveItem | undefined): string {
	return directiveGanBadge(item) || directiveJobBadge(item);
}

function directiveGanHeader(item: SidecarDirectiveItem): string | undefined {
	const gan = item.gan;
	if (!gan?.gan_id) return undefined;
	const round = typeof gan.round === "number" && Number.isFinite(gan.round) ? Math.floor(gan.round) : "?";
	const role = String(gan.role ?? "participant").trim() || "participant";
	const issues =
		typeof gan.issues_open === "number" && Number.isFinite(gan.issues_open) ? Math.floor(gan.issues_open) : "?";
	const remaining = typeof round === "number" ? Math.max(0, 3 - round) : "?";
	return [
		`[GAN ${gan.gan_id} round ${round}/3, you are ${role}, open issues ${issues}, remaining sends ${remaining}]`,
		`Respond with gan_send(gan_id="${gan.gan_id}") for a verdict/rebuttal/acceptance round, or gan_close(gan_id="${gan.gan_id}") with agreed/escalated for a terminal report.`,
		"Always pass this exact gan_id — never start a new GAN with the same window; the server rejects a second open GAN between the same pair.",
	].join("\n");
}

function lastJobHandbackSummary(item: SidecarDirectiveItem, history: SidecarJobHistoryResponse | undefined): string {
	const phase = String(item.job?.phase ?? "").trim();
	if (phase === "review") return oneLineForSummary(String(item.body ?? ""), 160);
	const items = Array.isArray(history?.items) ? history.items : [];
	const currentIndex = item.id ? items.findIndex((candidate) => candidate.id === item.id) : -1;
	const searchItems = currentIndex >= 0 ? items.slice(0, currentIndex) : items;
	const handback = [...searchItems]
		.reverse()
		.find((candidate) => String(candidate.job?.phase ?? "").trim() === "review");
	return handback ? oneLineForSummary(String(handback.body ?? ""), 160) : "none recorded";
}

async function directiveJobHeader(item: SidecarDirectiveItem): Promise<string | undefined> {
	const job = item.job;
	if (!job?.job_id) return undefined;
	const jobId = String(job.job_id).trim();
	const [history, windows] = await Promise.all([
		postSidecar<SidecarJobHistoryResponse>(`/job/${encodeURIComponent(jobId)}`, undefined, "GET", 8000),
		postSidecar<SidecarDirectiveWindowsResponse>("/directives/windows", undefined, "GET", 8000),
	]);
	const cycle = typeof job.cycle === "number" && Number.isFinite(job.cycle) ? Math.floor(job.cycle) : "?";
	const cap =
		typeof history?.cycle_cap === "number" && Number.isFinite(history.cycle_cap)
			? Math.floor(history.cycle_cap)
			: jobCycleCapFromEnv();
	const role = String(job.role ?? "participant").trim() || "participant";
	const phase = String(job.phase ?? "job").trim() || "job";
	const counterpart = String(item.from_window ?? "").trim();
	const counterpartLabel = displayWindowFromList(windows?.windows, counterpart);
	const handback = lastJobHandbackSummary(item, history);
	const lines = [
		`[JOB ${jobId} cycle ${cycle}/${cap}, you are ${role}, counterpart: ${counterpartLabel}, phase: ${phase}]`,
		`Last handback: ${handback}.`,
		`Respond with job_send(job_id="${jobId}", to_window="${counterpart}") to hand back review/progress or dispatch the next cycle,`,
		`or job_close(job_id="${jobId}") with done/escalated.`,
	];
	if (secondEyesReviewContext(item)) {
		lines.push(
			`Critic Mode worker turn: review-only plan critique / artifact verification; start any job_send handback with ${SECOND_EYES_MAIN_MARKER}.`,
		);
	}
	if (secondEyesMainContext(item)) {
		lines.push(
			"Critic Mode main turn: you implement/fix here; the counterpart is review-only. Use the same job for follow-up review; do not spawn more workers.",
		);
	}
	if (phase === "review") {
		lines.push(
			"Review depth follows the user's original ask: when the user never asked for review/quality passes, verify completion lightly and close done — do not invent extra fix cycles.",
		);
	}
	return lines.join("\n");
}

function sanitizeWindowLabel(value: unknown): string | undefined {
	const text = String(value ?? "")
		.replace(/[\u0000-\u001f\u007f]/g, "")
		.trim();
	return text ? text.slice(0, 32) : undefined;
}

function displayWindowName(pair8?: string, label?: string | null): string {
	const clean = sanitizeWindowLabel(label);
	return clean ? `${clean} (${pair8 ?? "unknown"})` : (pair8 ?? "unknown");
}

function displayWindowFromList(windows: SidecarDirectiveWindow[] | undefined, pair8: string | undefined): string {
	if (!pair8) return "unknown";
	const found = windows?.find((item) => item.pair8 === pair8);
	return displayWindowName(pair8, found?.label);
}

function directivePendingPath(kind: SidecarDirectiveKind, limit: number, consume = true): string {
	const params = new URLSearchParams({
		kind,
		consume: consume ? "true" : "false",
		limit: String(Math.max(1, limit)),
	});
	const known = directiveKnownQueueState.get(kind);
	if (known) {
		params.set("known_mtime_ns", String(known.mtimeNs));
		params.set("known_size", String(known.size));
	}
	return `/directives/pending?${params.toString()}`;
}

function rememberDirectiveQueueState(
	kind: SidecarDirectiveKind,
	data: SidecarDirectivePendingResponse | undefined,
): void {
	if (typeof data?.queue_mtime_ns !== "number" || typeof data.queue_size !== "number") return;
	directiveKnownQueueState.set(kind, {
		mtimeNs: data.queue_mtime_ns,
		size: data.queue_size,
	});
}

async function fetchPendingDirectives(
	kind: SidecarDirectiveKind,
	limit: number,
	options?: { consume?: boolean },
): Promise<SidecarDirectiveItem[]> {
	const data = await postSidecar<SidecarDirectivePendingResponse>(
		directivePendingPath(kind, limit, options?.consume ?? true),
		undefined,
		"GET",
		8000,
	);
	rememberDirectiveQueueState(kind, data);
	if (!data?.ok || data.unchanged) return [];
	return Array.isArray(data.items) ? data.items : [];
}

async function collectDirectiveReports(pi: ExtensionAPI): Promise<void> {
	const reports = await fetchPendingDirectives("report", 20);
	if (reports.length === 0) return;
	pendingDirectiveReports.push(...reports);
	for (const report of reports) {
		sendJarvisChatNotice(
			pi,
			`[Report received · window ${directiveWindowLabel(report.from_window)}] ${directiveSessionBadge(report)}${oneLineForSummary(String(report.body ?? ""), 180)}`,
		);
	}
}

function injectPendingDirectiveReports(): void {
	if (pendingDirectiveReports.length === 0) return;
	const reports = pendingDirectiveReports.splice(0);
	const lines = ["[Directive reports received]"];
	for (const report of reports) {
		lines.push(
			`[Report received · window ${directiveWindowLabel(report.from_window)}] ${directiveSessionBadge(report)}${String(report.body ?? "").trim()}`,
		);
	}
	appendTransientSystemDirective(lines.join("\n"));
}

function pendingDirectiveMatchesText(userText: string): boolean {
	if (!pendingDirectiveAutoTurn) return false;
	return String(pendingDirectiveAutoTurn.body ?? "").trim() === userText.trim();
}

function activateDirectiveTurn(item: SidecarDirectiveItem): void {
	activeDirectiveTurn = item;
	directiveTurnBusReplySent = false;
	activeMapCheckpointTurn = mapCheckpointContextForDirective(activeDirectiveTurn);
	activeMapSynthesisTurn = mapSynthesisContextForDirective(activeDirectiveTurn);
	activeEndGateTurn =
		!activeMapCheckpointTurn && !activeMapSynthesisTurn && wholeDelegationEndGateContext(activeDirectiveTurn);
	activeSecondEyesReviewTurn = secondEyesReviewContext(activeDirectiveTurn);
	activeSecondEyesMainTurn = secondEyesMainContext(activeDirectiveTurn);
	activeSecondEyesHeavyTurn = secondEyesHeavyContext(activeDirectiveTurn);
}

function markDirectiveTurnIfMatching(userText: string): void {
	if (!pendingDirectiveMatchesText(userText)) return;
	if (pendingDirectiveAutoTurn) activateDirectiveTurn(pendingDirectiveAutoTurn);
	pendingDirectiveAutoTurn = undefined;
}

function markSecondEyesMarkerTurnIfPresent(userText: string): void {
	const body = userText.trimStart();
	if (startsWithCriticReviewMarker(body)) {
		activeSecondEyesReviewTurn = true;
		activeSecondEyesMainTurn = false;
		activeSecondEyesHeavyTurn = includesCriticHeavyMarker(body);
		secondEyesRequestedThisTurn = false;
		secondEyesReviewSpawnedThisTurn = true;
		secondEyesReminderInjectedThisTurn = false;
		return;
	}
	if (startsWithCriticMainMarker(body)) {
		activeSecondEyesMainTurn = true;
		activeSecondEyesReviewTurn = false;
		activeSecondEyesHeavyTurn = includesCriticHeavyMarker(body);
		secondEyesRequestedThisTurn = false;
		secondEyesReviewSpawnedThisTurn = true;
		secondEyesReminderInjectedThisTurn = false;
	}
}

// --- M7.9 map-and-step: persistent map + deterministic ledger ----------------
// map.md is the human-readable map (append-only; never carries status).
// ledger.jsonl is the single source of truth for progress/rejections/stages,
// written exclusively by extension code from tool execute paths — never by
// the LLM. A per-window pointer file enables restart recovery without scans.

function mapRunsPointerDir(): string {
	return process.env.JARVIS_MAP_RUNS_DIR?.trim() || path.join(os.homedir(), ".jarvis-code", "map-runs");
}

function mapRunsPointerPath(): string | undefined {
	const win = jarvisOriginWindow();
	return win ? path.join(mapRunsPointerDir(), `${win}.json`) : undefined;
}

function mapArtifactDir(projectPath: string): string {
	return path.join(projectPath, MAP_DIR_NAME);
}

function newMapId(): string {
	mapIdCounter += 1;
	const digest = createHash("sha1").update(`${Date.now()}:${process.pid}:${mapIdCounter}`).digest("hex");
	return `m_${digest.slice(0, 8)}`;
}

function mapLedgerAppend(run: ActiveMapRun, kind: string, payload: Record<string, unknown> = {}): void {
	const dir = mapArtifactDir(run.projectPath);
	fs.mkdirSync(dir, { recursive: true });
	run.ledgerSeq += 1;
	const entry = { ts: new Date().toISOString(), map_id: run.mapId, seq: run.ledgerSeq, kind, ...payload };
	fs.appendFileSync(path.join(dir, MAP_LEDGER_FILE_NAME), `${JSON.stringify(entry)}\n`, "utf8");
}

function writeMapRunPointer(run: ActiveMapRun, status: "open" | "complete"): void {
	const pointerPath = mapRunsPointerPath();
	if (!pointerPath) return;
	fs.mkdirSync(path.dirname(pointerPath), { recursive: true });
	fs.writeFileSync(
		pointerPath,
		`${JSON.stringify({ project_path: run.projectPath, map_id: run.mapId, status }, null, 2)}\n`,
		"utf8",
	);
}

function writeMapFileAtomic(projectPath: string, content: string): void {
	const dir = mapArtifactDir(projectPath);
	fs.mkdirSync(dir, { recursive: true });
	const target = path.join(dir, MAP_FILE_NAME);
	const tmp = `${target}.tmp`;
	fs.writeFileSync(tmp, content, "utf8");
	fs.renameSync(tmp, target);
}

function mapFeatureMarkdown(feature: MapFeatureState): string {
	const lines = [`### ${feature.id} — ${feature.title} [zone: ${feature.zone}]`];
	if (feature.summary?.trim()) lines.push(feature.summary.trim());
	lines.push("Acceptance:");
	for (const criterion of feature.acceptance) lines.push(`- ${criterion}`);
	return lines.join("\n");
}

function buildMapMarkdown(run: ActiveMapRun): string {
	const lines = [
		`# JARVIS Map — ${run.title}`,
		`map_id: ${run.mapId} | project: ${run.projectPath} | created: ${new Date().toISOString()}`,
		`Status lives in ${MAP_LEDGER_FILE_NAME} — this file is append-only.`,
		"",
		"## Features",
		"",
	];
	for (const feature of run.features.values()) {
		lines.push(mapFeatureMarkdown(feature), "");
	}
	return `${lines.join("\n").trimEnd()}\n`;
}

function appendMapMarkdown(run: ActiveMapRun, added: MapFeatureState[]): void {
	const target = path.join(mapArtifactDir(run.projectPath), MAP_FILE_NAME);
	const lines = ["", `## Amendment — ${new Date().toISOString()}`, ""];
	for (const feature of added) lines.push(mapFeatureMarkdown(feature), "");
	fs.appendFileSync(target, `${lines.join("\n").trimEnd()}\n`, "utf8");
}

function rebuildMapRunFromLedger(projectPath: string): ActiveMapRun | undefined {
	const ledgerPath = path.join(mapArtifactDir(projectPath), MAP_LEDGER_FILE_NAME);
	let raw: string;
	try {
		raw = fs.readFileSync(ledgerPath, "utf8");
	} catch {
		return undefined;
	}
	let run: ActiveMapRun | undefined;
	for (const line of raw.split("\n")) {
		const trimmed = line.trim();
		if (!trimmed) continue;
		let entry: Record<string, unknown>;
		try {
			entry = JSON.parse(trimmed) as Record<string, unknown>;
		} catch {
			continue;
		}
		const kind = String(entry.kind ?? "");
		if (kind === "map_created") {
			run = {
				mapId: String(entry.map_id ?? ""),
				title: String(entry.title ?? ""),
				projectPath,
				features: new Map(),
				jobFeatures: new Map(),
				phase: "stepping",
				ledgerSeq: Number(entry.seq ?? 0) || 0,
			};
			continue;
		}
		if (!run || String(entry.map_id ?? "") !== run.mapId) continue;
		run.ledgerSeq = Math.max(run.ledgerSeq, Number(entry.seq ?? 0) || 0);
		if (kind === "map_abandoned") {
			run = undefined;
		} else if (kind === "feature_added") {
			const id = String(entry.feature_id ?? "");
			if (!id) continue;
			run.features.set(id, {
				id,
				title: String(entry.title ?? ""),
				summary: typeof entry.summary === "string" && entry.summary ? entry.summary : undefined,
				zone: entry.zone === "skeleton" ? "skeleton" : "feature",
				acceptance: Array.isArray(entry.acceptance) ? entry.acceptance.map(String) : [],
				status: "todo",
				rejections: 0,
				stage: "normal",
			});
		} else if (kind === "acceptance_appended") {
			const feature = run.features.get(String(entry.feature_id ?? ""));
			if (feature && Array.isArray(entry.acceptance)) feature.acceptance.push(...entry.acceptance.map(String));
		} else if (kind === "dispatch") {
			const ids = Array.isArray(entry.feature_ids) ? entry.feature_ids.map(String) : [];
			const jobId = String(entry.job_id ?? "");
			if (jobId && ids.length) run.jobFeatures.set(jobId, ids);
			for (const id of ids) {
				const feature = run.features.get(id);
				if (feature && feature.status !== "passed") feature.status = "dispatched";
			}
		} else if (kind === "verdict") {
			const feature = run.features.get(String(entry.feature_id ?? ""));
			if (!feature) continue;
			if (typeof entry.rejections_after === "number") feature.rejections = entry.rejections_after;
			const stageAfter = String(entry.stage_after ?? "");
			if (stageAfter === "normal" || stageAfter === "escalated" || stageAfter === "main_direct") {
				feature.stage = stageAfter;
			}
			if (entry.verdict === "pass") feature.status = "passed";
			else if (typeof entry.reason === "string" && entry.reason) feature.lastRejectReason = entry.reason;
		} else if (kind === "escalate") {
			const feature = run.features.get(String(entry.feature_id ?? ""));
			const stage = String(entry.stage ?? "");
			if (feature && (stage === "escalated" || stage === "main_direct")) feature.stage = stage;
		} else if (kind === "map_complete" || kind === "synthesis_done") {
			run.phase = "complete";
		}
	}
	if (run && run.phase !== "complete" && run.features.size > 0) {
		const allPassed = [...run.features.values()].every((feature) => feature.status === "passed");
		if (allPassed) run.phase = "synthesis";
	}
	return run;
}

function restoreMapRunFromPointer(): void {
	if (activeMapRun) return;
	const pointerPath = mapRunsPointerPath();
	if (!pointerPath) return;
	try {
		const pointer = JSON.parse(fs.readFileSync(pointerPath, "utf8")) as {
			project_path?: string;
			map_id?: string;
			status?: string;
		};
		if (pointer?.status !== "open" || !pointer.project_path) return;
		const run = rebuildMapRunFromLedger(pointer.project_path);
		if (!run || run.mapId !== pointer.map_id || run.phase === "complete") return;
		activeMapRun = run;
	} catch {
		// Missing/corrupt pointer or ledger: boot without a map run rather than block.
	}
}

function mapTicketValidationError(featureIds: string[]): string | undefined {
	const run = activeMapRun;
	if (!run) return "no open map run; call map_create first or drop feature_ids";
	const unknown = featureIds.filter((id) => !run.features.has(id));
	if (unknown.length > 0) {
		return `unknown feature ids: ${unknown.join(", ")} (known: ${[...run.features.keys()].join(", ")})`;
	}
	return undefined;
}

function buildMapTicketBlock(featureIds: string[]): string {
	const run = activeMapRun;
	if (!run) return "";
	const lines: string[] = [];
	for (const id of featureIds) {
		const feature = run.features.get(id);
		if (!feature) continue;
		lines.push(`[MAP ${run.mapId} FEATURE ${feature.id} — ${feature.title}]`);
		if (feature.summary) lines.push(feature.summary);
		lines.push("Acceptance criteria (checked verbatim at the checkpoint):");
		for (const criterion of feature.acceptance) lines.push(`- ${criterion}`);
		// The rejection reason is the dispatcher's intent channel (M7.9 letter):
		// it rides every re-dispatch ticket so the worker sees why the last
		// attempt was rejected.
		if (feature.lastRejectReason) lines.push(`Previous rejection reason: ${feature.lastRejectReason}`);
		lines.push("");
	}
	return lines.join("\n").trimEnd();
}

function recordMapDispatch(jobId: string, featureIds: string[], toWindow: string, model?: string): void {
	const run = activeMapRun;
	if (!run || !jobId) return;
	run.jobFeatures.set(jobId, [...featureIds]);
	let stage: MapFeatureStage = "normal";
	for (const id of featureIds) {
		const feature = run.features.get(id);
		if (!feature) continue;
		if (feature.status !== "passed") feature.status = "dispatched";
		if (feature.stage !== "normal") stage = feature.stage;
	}
	mapLedgerAppend(run, "dispatch", { feature_ids: featureIds, job_id: jobId, to_window: toWindow, model, stage });
}

// Checkpoint detection keys off the directive item's job metadata only —
// never off body text (the body is worker-authored prose).
function mapCheckpointContextForDirective(
	item: SidecarDirectiveItem | undefined,
): { jobId: string; featureIds: string[] } | undefined {
	const run = activeMapRun;
	if (!run || run.phase !== "stepping" || !item) return undefined;
	const jobId = String(item.job?.job_id ?? "").trim();
	if (!jobId) return undefined;
	const featureIds = run.jobFeatures.get(jobId);
	if (!featureIds?.length) return undefined;
	return { jobId, featureIds: [...featureIds] };
}

// main_direct lifts the verify-only restrictions: the escalate ladder ends
// with the orchestrator implementing the feature itself in this window.
function mapCheckpointRestrictionsLifted(checkpoint: { featureIds: string[] }): boolean {
	const run = activeMapRun;
	if (!run) return false;
	return checkpoint.featureIds.some((id) => run.features.get(id)?.stage === "main_direct");
}

function mapCheckpointHasEscalatedFeature(checkpoint: { featureIds: string[] }): boolean {
	const run = activeMapRun;
	if (!run) return false;
	return checkpoint.featureIds.some((id) => run.features.get(id)?.stage === "escalated");
}

// Rejection cap per feature: rejections at the cap still re-dispatch normally;
// the rejection that exceeds it advances the ladder (normal -> escalated ->
// main_direct). Letter default: 2.
function mapRejectionCapFromEnv(): number {
	const raw = Number(process.env.JARVIS_MAP_REJECTION_CAP ?? "");
	return Number.isFinite(raw) && raw >= 1 ? Math.floor(raw) : 2;
}

function mapSynthesisContextForDirective(item: SidecarDirectiveItem | undefined): boolean {
	const run = activeMapRun;
	if (!run || run.phase !== "synthesis" || !item) return false;
	return String(item.body ?? "")
		.trim()
		.startsWith(`${MAP_SYNTHESIS_BODY_PREFIX}${run.mapId}]`);
}

// Whole-delegation end gate: the orchestrator is processing a worker handback
// (job phase "review", role "orchestrator") for a build that was NOT sliced
// into a feature map. Keys off job metadata only, mirroring checkpoint
// detection. Map-backed handbacks own feature_ids and go through the
// checkpoint/synthesis path instead, so they are excluded here.
function wholeDelegationEndGateContext(item: SidecarDirectiveItem | undefined): boolean {
	const job = item?.job;
	if (!job?.job_id) return false;
	if (String(job.phase ?? "").trim() !== "review") return false;
	if (String(job.role ?? "").trim() !== "orchestrator") return false;
	if (mapCheckpointContextForDirective(item)) return false;
	return true;
}

function secondEyesReviewContext(item: SidecarDirectiveItem | undefined): boolean {
	const body = String(item?.body ?? "").trimStart();
	return startsWithCriticReviewMarker(body);
}

function secondEyesMainContext(item: SidecarDirectiveItem | undefined): boolean {
	const body = String(item?.body ?? "").trimStart();
	return startsWithCriticMainMarker(body);
}

function secondEyesHeavyContext(item: SidecarDirectiveItem | undefined): boolean {
	const body = String(item?.body ?? "");
	return includesCriticHeavyMarker(body);
}

function routeDecisionCriticMode(decision: SidecarRouteTurnResponse | undefined): boolean {
	return decision?.critic_mode === true;
}

function routeDecisionCriticHeavy(decision: SidecarRouteTurnResponse | undefined): boolean {
	return routeDecisionCriticMode(decision) && decision?.critic_heavy === true;
}

function enterSecondEyesRoute(): void {
	if (
		activeSecondEyesHeavyTurn ||
		currentRoute === "heavy_deepdive" ||
		routeDecisionCriticHeavy(lastRouteClassifierDecision)
	) {
		enterHeavyProjectWork();
		activeSecondEyesHeavyTurn = true;
		return;
	}
	enterProjectWork();
}

function currentSecondEyesHeavy(): boolean {
	return activeSecondEyesHeavyTurn || currentRoute === "heavy_deepdive";
}

function startsWithAnyMarker(text: string, markers: readonly string[]): boolean {
	return markers.some((marker) => text.startsWith(marker));
}

function includesAnyMarker(text: string, markers: readonly string[]): boolean {
	return markers.some((marker) => text.includes(marker));
}

function startsWithCriticReviewMarker(text: string): boolean {
	return startsWithAnyMarker(text, CRITIC_REVIEW_MARKERS);
}

function startsWithCriticMainMarker(text: string): boolean {
	return startsWithAnyMarker(text, CRITIC_MAIN_MARKERS);
}

function includesCriticHeavyMarker(text: string): boolean {
	return includesAnyMarker(text, CRITIC_HEAVY_MARKERS);
}

function includesCriticPlanReadyMarker(text: string): boolean {
	return includesAnyMarker(text, CRITIC_PLAN_READY_MARKERS);
}

function normalizeCriticMarkers(text: string): string {
	return [
		[LEGACY_SECOND_EYES_DIRECTIVE_MARKER, SECOND_EYES_DIRECTIVE_MARKER],
		[LEGACY_SECOND_EYES_MAIN_MARKER, SECOND_EYES_MAIN_MARKER],
		[LEGACY_SECOND_EYES_HEAVY_MARKER, SECOND_EYES_HEAVY_MARKER],
		[LEGACY_SECOND_EYES_PLAN_READY_MARKER, SECOND_EYES_PLAN_READY_MARKER],
		[LEGACY_SECOND_EYES_REMINDER_MARKER, SECOND_EYES_REMINDER_MARKER],
	].reduce((current, [legacy, currentMarker]) => current.split(legacy).join(currentMarker), text);
}

function currentWorkerModelSpec(ctx?: ExtensionContext): string | undefined {
	const provider = String(ctx?.model?.provider ?? "").trim();
	const model = String(ctx?.model?.id ?? "").trim();
	if (provider && model) return `${provider}/${model}`;
	return model || provider || undefined;
}

function workerModelRecommendationLabel(spec: string): string {
	if (spec === WORKER_MODEL_RECOMMENDED_SPEC) return `GPT 5.5로 띄우기 (${spec})`;
	if (spec === CRITIC_WORKER_MODEL_RECOMMENDED_SPEC) return `Claude Opus 4.8로 띄우기 (${spec})`;
	return `추천 모델로 띄우기 (${spec})`;
}

const WORKER_MODEL_CATALOG_PICK_LABEL = "가용 모델 목록에서 선택";

function isExactWorkerModelInput(value: string): boolean {
	const text = value.trim();
	if (!text) return false;
	return Array.from(text).every((char) => {
		const code = char.codePointAt(0) ?? 0;
		return code >= 33 && code <= 126;
	});
}

function recommendedWorkerSpecForAmbiguousInput(value: string, fallback: string): string {
	const text = value.trim().toLowerCase();
	if (
		text.includes("anthropic/") ||
		text.includes("claude") ||
		text.includes("opus") ||
		text.includes("앤트로픽") ||
		text.includes("클로드") ||
		text.includes("오퍼스")
	) {
		return CRITIC_WORKER_MODEL_RECOMMENDED_SPEC;
	}
	return fallback;
}

function userExplicitlyNamedOpenRouter(value: string): boolean {
	const text = value.trim().toLowerCase();
	return text.includes("openrouter") || text.includes("오픈라우터");
}

function modelLooksLikeClaudeFamily(value: string): boolean {
	const text = value.trim().toLowerCase();
	return (
		text.includes("anthropic/") ||
		text.includes("claude") ||
		text.includes("opus") ||
		text.includes("sonnet") ||
		text.includes("haiku") ||
		text.includes("앤트로픽") ||
		text.includes("클로드") ||
		text.includes("오퍼스") ||
		text.includes("소넷") ||
		text.includes("하이쿠")
	);
}

function recommendedWorkerSpecWhenModelNeedsConfirmation(value: string, userText: string): string | undefined {
	const split = splitModelSpec(value);
	if (!split) return undefined;
	if (split.provider.toLowerCase() !== "openrouter") return undefined;
	if (userExplicitlyNamedOpenRouter(userText)) return undefined;
	if (!modelLooksLikeClaudeFamily(split.model)) return undefined;
	return CRITIC_WORKER_MODEL_RECOMMENDED_SPEC;
}

function workerModelRoleAllowed(provider: SidecarLLMSettingProvider): boolean {
	const roles = Array.isArray(provider.roles) ? provider.roles : [];
	return roles.length === 0 || roles.includes("chat") || roles.includes("subagent");
}

function workerModelProviderWeight(
	providerId: string,
	provider: SidecarLLMSettingProvider,
	recommendedProvider?: string,
	currentProvider?: string,
): number {
	if (providerId === recommendedProvider) return 0;
	if (providerId === currentProvider) return 1;
	if (provider.auth_kind === "agent-sdk" || provider.auth_kind === "oauth") return 2;
	if (!provider.auth_env) return 3;
	return 4;
}

function workerModelLabel(name: string, markers: string[]): string {
	return markers.length ? `${name}  (${markers.join(", ")})` : name;
}

async function pickWorkerModelFromCatalog(
	ctx: ExtensionContext,
	recommendedSpec: string,
	currentSpec?: string,
): Promise<string | undefined> {
	const catalog = await fetchLLMSettingCatalog(false);
	if (!catalog?.ok || !catalog.providers) {
		ctx.ui.notify?.(
			`JARVIS worker model list failed: ${(catalog as { error?: string } | undefined)?.error ?? "sidecar unavailable"}`,
			"warning",
		);
		return undefined;
	}
	const recommended = splitModelSpec(recommendedSpec);
	const current = splitModelSpec(currentSpec);
	const entries = Object.entries(catalog.providers)
		.filter(
			([, provider]) => provider.available && workerModelRoleAllowed(provider) && (provider.models?.length ?? 0) > 0,
		)
		.sort(([a, ap], [b, bp]) => {
			const wa = workerModelProviderWeight(a, ap, recommended?.provider, current?.provider);
			const wb = workerModelProviderWeight(b, bp, recommended?.provider, current?.provider);
			if (wa !== wb) return wa - wb;
			return a.localeCompare(b);
		});
	if (entries.length === 0) {
		ctx.ui.notify?.("No authenticated worker-capable models are available in the catalog.", "warning");
		return undefined;
	}
	const providerLabels = entries.map(([providerId, provider]) => {
		const markers: string[] = [];
		if (providerId === recommended?.provider) markers.push("recommended");
		if (providerId === current?.provider) markers.push("current");
		const name = `${provider.label ?? providerId} [${providerId}]${modelCatalogLabelSuffix(provider)}`;
		return workerModelLabel(name, markers);
	});
	const providerPicked = await ctx.ui.select("워커 모델 provider 선택", providerLabels, { signal: ctx.signal });
	if (providerPicked === undefined) return undefined;
	const providerIndex = providerLabels.indexOf(providerPicked);
	if (providerIndex < 0) return undefined;
	const [providerId, provider] = entries[providerIndex];
	const models = provider.models ?? [];
	const modelWeight = (model: string): number => {
		if (providerId === recommended?.provider && model === recommended.model) return 0;
		if (providerId === current?.provider && model === current.model) return 1;
		if (model.endsWith(":free")) return 2;
		return 3;
	};
	const orderedModels = [...models].sort((a, b) => {
		const wa = modelWeight(a);
		const wb = modelWeight(b);
		if (wa !== wb) return wa - wb;
		return 0;
	});
	const modelLabels = orderedModels.map((model) => {
		const markers: string[] = [];
		if (providerId === recommended?.provider && model === recommended.model) markers.push("recommended");
		if (providerId === current?.provider && model === current.model) markers.push("current");
		return workerModelLabel(model, markers);
	});
	const modelPicked = await ctx.ui.select(`워커 모델 선택: ${provider.label ?? providerId}`, modelLabels, {
		signal: ctx.signal,
	});
	if (modelPicked === undefined) return undefined;
	const modelIndex = modelLabels.indexOf(modelPicked);
	return modelIndex >= 0 ? `${providerId}/${orderedModels[modelIndex]}` : undefined;
}

async function chooseWorkerModelForSpawn(
	paramsModel: unknown,
	ctx?: ExtensionContext,
	options?: { recommendedSpec?: string },
): Promise<string | undefined> {
	const provided = typeof paramsModel === "string" ? paramsModel.trim() : "";
	const baseRecommendedSpec = options?.recommendedSpec ?? WORKER_MODEL_RECOMMENDED_SPEC;
	const confirmationRecommendedSpec = provided
		? recommendedWorkerSpecWhenModelNeedsConfirmation(provided, lastUserMessage)
		: undefined;
	if (provided && isExactWorkerModelInput(provided) && !confirmationRecommendedSpec) return provided;
	const recommendedSpec =
		confirmationRecommendedSpec ??
		recommendedWorkerSpecForAmbiguousInput(provided || lastUserMessage, baseRecommendedSpec);
	const currentSpec = currentWorkerModelSpec(ctx);
	if (!ctx?.hasUI || typeof ctx.ui?.select !== "function") {
		return confirmationRecommendedSpec ?? (provided || recommendedSpec);
	}

	const recommendedLabel = workerModelRecommendationLabel(recommendedSpec);
	const customLabel = provided ? "정확한 모델명 직접 입력" : "직접 모델명 입력";
	const currentLabel = currentSpec ? `현재 모델(${currentSpec})로 띄우기` : "현재 모델로 띄우기";
	const optionsList = [recommendedLabel, WORKER_MODEL_CATALOG_PICK_LABEL, currentLabel, customLabel];
	const title = provided
		? `모델명이 애매합니다: ${provided}. 어느 모델로 띄울까? Esc는 추천 모델을 선택합니다.`
		: "새 워커를 어느 모델로 띄울까? Esc는 추천 모델을 선택합니다.";
	const selected = await ctx.ui.select(title, optionsList, { signal: ctx.signal });
	if (selected === undefined || selected === recommendedLabel) return recommendedSpec;
	if (selected === WORKER_MODEL_CATALOG_PICK_LABEL) {
		return (await pickWorkerModelFromCatalog(ctx, recommendedSpec, currentSpec)) ?? recommendedSpec;
	}
	if (selected === currentLabel) return currentSpec;
	if (selected === customLabel && typeof ctx.ui.input === "function") {
		const custom = await ctx.ui.input("워커 모델명 직접 입력", "provider/model 또는 정확한 bare model", {
			signal: ctx.signal,
		});
		const model = typeof custom === "string" ? custom.trim() : "";
		return model || recommendedSpec;
	}
	return recommendedSpec;
}

function buildSecondEyesReviewDirective(rawDirective: string): string {
	const body = rawDirective.trim();
	if (startsWithCriticReviewMarker(body)) return normalizeCriticMarkers(body);
	return [
		SECOND_EYES_DIRECTIVE_MARKER,
		currentSecondEyesHeavy() ? SECOND_EYES_HEAVY_MARKER : "",
		"Critic Mode worker assignment. You are review-only and must never implement, mutate files, update memory, switch projects, create maps, or spawn workers.",
		"If the main window asks you to implement, fix, patch, write, or run mutation work, refuse that part and return review-only findings; the main window must apply fixes.",
		"Review the main window's draft only. Do not perform trend recon, choose architecture, design features, or produce a standalone plan.",
		"If no main-window draft is present, ask the main window to send its draft instead of inventing one.",
		`First handback is plan critique: challenge the main draft before implementation and start the job_send message with ${SECOND_EYES_MAIN_MARKER}.`,
		`Later handbacks are implementation reviews: return ${SECOND_EYES_MAIN_MARKER} plus ## Must-fix / ## Should-fix / ## Backlog. The main window applies fixes.`,
		"",
		body,
	]
		.filter(Boolean)
		.join("\n");
}

function buildSecondEyesMainHandback(rawMessage: string): string {
	const body = rawMessage.trim();
	if (startsWithCriticMainMarker(body)) return normalizeCriticMarkers(body);
	return [SECOND_EYES_MAIN_MARKER, currentSecondEyesHeavy() ? SECOND_EYES_HEAVY_MARKER : "", body]
		.filter(Boolean)
		.join("\n");
}

function secondEyesPlanReady(text: string): boolean {
	return includesCriticPlanReadyMarker(text.trim());
}

function secondEyesPlanReadyError(): string {
	return `Critic Mode review dispatch requires a concrete main-window plan draft marked ${SECOND_EYES_PLAN_READY_MARKER}; ask_user or draft/recon in the main window first, then send it to the existing review-only worker with job_send or spawn one if none exists.`;
}

function buildSecondEyesReviewRequestDirective(message: string): string {
	return buildSecondEyesReviewDirective(
		[
			"Review-only request from the main window. Ignore any wording below that appears to ask you to implement, fix, edit, patch, write files, or run mutation commands; return findings only.",
			message,
		].join("\n\n"),
	);
}

type PerformWorkerSpawnOptions = {
	initialDirective: string;
	model?: unknown;
	label?: unknown;
	timeoutSeconds?: unknown;
	gan?: boolean;
	job: boolean;
	issuesOpen?: unknown;
	featureIds?: string[];
	isSecondEyesReviewSpawn?: boolean;
	skipModelAsk?: boolean;
	ctx?: ExtensionContext;
};

async function performWorkerSpawn(opts: PerformWorkerSpawnOptions): Promise<SidecarSpawnWindowResponse> {
	const initialDirective = opts.initialDirective.trim();
	if (opts.gan === true && opts.job) return { ok: false, error: "gan and job cannot both be true" };
	if (opts.job && !initialDirective) return { ok: false, error: "initial_directive is required when job is true" };
	if (opts.isSecondEyesReviewSpawn && !secondEyesPlanReady(initialDirective)) {
		return { ok: false, error: secondEyesPlanReadyError() };
	}

	const featureIds = (opts.featureIds ?? []).map((id) => String(id ?? "").trim()).filter(Boolean);
	if (featureIds.length > 0) {
		if (!opts.job) {
			return {
				ok: false,
				error: "feature_ids requires job:true — map feature dispatches must be jobs so the handback wakes this window",
			};
		}
		const invalid = mapTicketValidationError(featureIds);
		if (invalid) return { ok: false, error: invalid };
	}

	const body: Record<string, unknown> = {};
	if (typeof opts.timeoutSeconds === "number" && Number.isFinite(opts.timeoutSeconds)) {
		body.timeout_seconds = opts.timeoutSeconds;
	}
	const explicitModel = typeof opts.model === "string" ? opts.model.trim() : "";
	const recommendedSpec = opts.isSecondEyesReviewSpawn
		? CRITIC_WORKER_MODEL_RECOMMENDED_SPEC
		: WORKER_MODEL_RECOMMENDED_SPEC;
	const workerModel = opts.skipModelAsk
		? explicitModel || recommendedSpec
		: await chooseWorkerModelForSpawn(opts.model, opts.ctx, { recommendedSpec });
	if (workerModel) {
		body.model = workerModel;
	}
	if (typeof opts.label === "string" && opts.label.trim()) {
		body.label = opts.label.trim();
	}

	const data = await postSpawnWithBootRetry(body);
	let details = isOkSidecarResponse(data) ? data : undefined;
	const pair8 = details?.pair8 ?? details?.window?.pair8;
	if (details && initialDirective && pair8) {
		const directiveBody: Record<string, unknown> = {
			kind: "directive",
			to_window: pair8,
			message:
				featureIds.length > 0 ? `${buildMapTicketBlock(featureIds)}\n\n${initialDirective}` : initialDirective,
		};
		if (opts.gan === true) {
			const issuesOpen =
				typeof opts.issuesOpen === "number" && Number.isFinite(opts.issuesOpen)
					? Math.max(0, Math.floor(opts.issuesOpen))
					: undefined;
			if (issuesOpen === undefined) {
				return { ...details, ok: false, error: "issues_open is required when gan is true" };
			}
			directiveBody.gan_target = "new";
			directiveBody.issues_open = issuesOpen;
		} else if (opts.job) {
			directiveBody.job_target = "new";
		}
		const sent = await postSidecar<SidecarDirectiveSendResponse>("/directives", directiveBody);
		const directive = isOkSidecarResponse(sent) ? (sent?.item ?? null) : null;
		if (directive && opts.isSecondEyesReviewSpawn) {
			secondEyesReviewSpawnedThisTurn = true;
		}
		if (featureIds.length > 0 && directive?.job?.job_id) {
			recordMapDispatch(
				String(directive.job.job_id),
				featureIds,
				String(pair8),
				typeof body.model === "string" ? body.model : undefined,
			);
		}
		details = {
			...details,
			directive,
		};
	}
	return details ?? data ?? { ok: false, error: SIDECAR_BOOTING_SPAWN_ERROR };
}

// The synthesis trigger rides the existing bus as a self-directive: the
// sidecar delivers it like any other directive (poll + idle + cooldown), so
// no new wake machinery exists for it.
async function postMapSynthesisDirective(): Promise<void> {
	const run = activeMapRun;
	const own = jarvisOriginWindow();
	if (!run || run.phase !== "synthesis" || !own) return;
	await postSidecar("/directives", {
		kind: "directive",
		to_window: own,
		message:
			`${MAP_SYNTHESIS_BODY_PREFIX}${run.mapId}] All map features passed verification. ` +
			"Run the final cross-feature synthesis (build/tests/launch) and report the result to the user.",
	});
}

function finalizeMapRunAfterSynthesis(): void {
	const run = activeMapRun;
	if (!run || run.phase === "complete") return;
	run.phase = "complete";
	try {
		mapLedgerAppend(run, "map_complete");
		mapLedgerAppend(run, "synthesis_done", {});
		writeMapRunPointer(run, "complete");
	} catch {
		// Ledger finalization is best-effort inside agent_end; the in-memory
		// completion below still retires the run for this window.
	}
	activeMapRun = undefined;
}

function mapEscalateModelHint(): string {
	const model = process.env.JARVIS_MAP_ESCALATE_MODEL?.trim();
	return model ? ` (model: ${model})` : " (pick a stronger model than the failed worker)";
}

// Shared map_create implementation. The single source of truth for mutating the
// in-process activeMapRun ledger; called by BOTH the regime-A registerTool
// execute (which wraps the details into tool content) AND the regime-B
// control-bridge branch (anthropic-agent-sdk delegates execution back to pi so
// pi — and only pi — owns activeMapRun). Returns the same `details` object both
// paths surface, so producer (adapter) and consumer (pi) cannot drift.
async function runMapCreate(params: {
	project_path?: unknown;
	title?: unknown;
	append?: unknown;
	replace?: unknown;
	features?: unknown;
}): Promise<MapCreateResult> {
	const fail = (error: string): MapCreateResult => ({ ok: false, error });
	const projectPath = String(params.project_path ?? "").trim();
	let stats: fs.Stats | undefined;
	try {
		stats = fs.statSync(projectPath);
	} catch {
		/* missing path */
	}
	if (!projectPath || !stats?.isDirectory()) return fail("project_path must be an existing directory");
	const rawFeatures = Array.isArray(params.features) ? params.features : [];
	if (rawFeatures.length === 0) return fail("features must not be empty");
	const cleaned: Array<{ title: string; summary?: string; zone: MapZone; acceptance: string[] }> = [];
	for (const raw of rawFeatures) {
		const title = String(raw?.title ?? "").trim();
		if (!title) return fail("every feature needs a title");
		const acceptance = (Array.isArray(raw?.acceptance) ? raw.acceptance : [])
			.map((criterion: unknown) => String(criterion ?? "").trim())
			.filter(Boolean);
		if (acceptance.length === 0) return fail(`${MAP_FEATURE_ACCEPTANCE_REQUIRED_ERROR} (feature: ${title})`);
		const summary = String(raw?.summary ?? "").trim();
		cleaned.push({
			title,
			summary: summary || undefined,
			zone: raw?.zone === "skeleton" ? "skeleton" : "feature",
			acceptance,
		});
	}
	const append = params.append === true;
	if (append) {
		if (!activeMapRun || activeMapRun.phase === "complete") {
			return fail("no open map run to append to; call without append to create one");
		}
		if (path.resolve(activeMapRun.projectPath) !== path.resolve(projectPath)) {
			return fail(`the open map run belongs to ${activeMapRun.projectPath}; finish or replace it first`);
		}
	} else if (activeMapRun && activeMapRun.phase !== "complete") {
		if (params.replace !== true) {
			return fail(`${MAP_RUN_ALREADY_OPEN_ERROR} (open: ${activeMapRun.mapId} @ ${activeMapRun.projectPath})`);
		}
		mapLedgerAppend(activeMapRun, "map_abandoned");
		activeMapRun = undefined;
	}
	const run: ActiveMapRun =
		append && activeMapRun
			? activeMapRun
			: {
					mapId: newMapId(),
					title: String(params.title ?? "").trim() || path.basename(projectPath),
					projectPath,
					features: new Map(),
					jobFeatures: new Map(),
					phase: "stepping",
					ledgerSeq: 0,
				};
	let nextIndex = 1;
	for (const id of run.features.keys()) {
		const n = Number(/^f(\d+)$/.exec(id)?.[1] ?? 0);
		if (n >= nextIndex) nextIndex = n + 1;
	}
	const added: MapFeatureState[] = [];
	for (const spec of cleaned) {
		const feature: MapFeatureState = {
			id: `f${nextIndex}`,
			...spec,
			status: "todo",
			rejections: 0,
			stage: "normal",
		};
		nextIndex += 1;
		run.features.set(feature.id, feature);
		added.push(feature);
	}
	if (!append) {
		activeMapRun = run;
		mapLedgerAppend(run, "map_created", {
			title: run.title,
			project_path: run.projectPath,
			feature_ids: added.map((feature) => feature.id),
		});
	}
	for (const feature of added) {
		mapLedgerAppend(run, "feature_added", {
			feature_id: feature.id,
			title: feature.title,
			summary: feature.summary,
			zone: feature.zone,
			acceptance: feature.acceptance,
		});
	}
	if (append) appendMapMarkdown(run, added);
	else writeMapFileAtomic(projectPath, buildMapMarkdown(run));
	writeMapRunPointer(run, "open");
	return {
		ok: true,
		map_id: run.mapId,
		project_path: run.projectPath,
		features: [...run.features.values()].map((feature) => ({
			id: feature.id,
			title: feature.title,
			status: feature.status,
		})),
	};
}

// Shared feature_verdict implementation — same single-source-of-truth contract
// as runMapCreate. Mutates the in-process activeMapRun feature/run state, runs
// the rejection-cap escalate ladder, appends the ledger, and (on full pass) arms
// pendingMapSynthesisPost. Returns details PLUS the human `next` step string the
// model needs; the registerTool execute renders `next` after the JSON, the
// control-bridge branch returns it inline.
async function runFeatureVerdict(params: {
	feature_id?: unknown;
	verdict?: unknown;
	reason?: unknown;
	evidence?: unknown;
}): Promise<FeatureVerdictResult> {
	const fail = (error: string): FeatureVerdictResult => ({ ok: false, error });
	const run = activeMapRun;
	if (!run || run.phase === "complete") return fail("no open map run; verdicts only exist inside one");
	const featureId = String(params.feature_id ?? "").trim();
	const feature = run.features.get(featureId);
	if (!feature) {
		return fail(`unknown feature id: ${featureId} (known: ${[...run.features.keys()].join(", ")})`);
	}
	if (feature.status === "passed") return fail(`${featureId} already passed; verdicts are irreversible`);
	const reason = String(params.reason ?? "").trim();
	const evidence = String(params.evidence ?? "").trim();
	let next: string;
	if (params.verdict === "pass") {
		if (!evidence) {
			return fail("pass requires evidence — cite the runnable check you executed (build/test/launch output)");
		}
		feature.status = "passed";
		mapLedgerAppend(run, "verdict", {
			feature_id: featureId,
			verdict: "pass",
			evidence,
			rejections_after: feature.rejections,
			stage_after: feature.stage,
		});
		const remaining = [...run.features.values()].filter((other) => other.status !== "passed");
		if (remaining.length === 0) {
			run.phase = "synthesis";
			pendingMapSynthesisPost = true;
			next = "All map features passed — the final synthesis turn fires automatically; finish this turn now.";
		} else {
			next = `Passed. Remaining: ${remaining
				.map((other) => `${other.id} ${other.title} (${other.status})`)
				.join(
					", ",
				)}. Dispatch the next unchecked feature(s) via job_send/spawn_window with feature_ids, then end the turn.`;
		}
	} else {
		if (!reason) {
			return fail("rejection without a reason is forbidden — the reason is the intent channel to the worker");
		}
		feature.rejections += 1;
		feature.lastRejectReason = reason;
		const cap = mapRejectionCapFromEnv();
		let advanced: MapFeatureStage | undefined;
		if (feature.stage === "normal" && feature.rejections > cap) {
			feature.stage = "escalated";
			advanced = "escalated";
		} else if (feature.stage === "escalated") {
			feature.stage = "main_direct";
			advanced = "main_direct";
		}
		mapLedgerAppend(run, "verdict", {
			feature_id: featureId,
			verdict: "reject",
			reason,
			rejections_after: feature.rejections,
			stage_after: feature.stage,
		});
		if (advanced) mapLedgerAppend(run, "escalate", { feature_id: featureId, stage: advanced });
		if (advanced === "escalated") {
			next = `Rejection cap exceeded (${feature.rejections}/${cap}) — ESCALATE: close this job with status escalated, then spawn/dispatch a stronger worker${mapEscalateModelHint()} with feature_ids ["${featureId}"]; the rejection reason rides the ticket.`;
		} else if (advanced === "main_direct") {
			next = `The escalated worker failed too — MAIN-DIRECT: implement ${featureId} yourself in this window now (edit tools are unlocked for it), then record a pass verdict with evidence.`;
		} else {
			next = `Rejected (${feature.rejections}/${cap}). Re-dispatch ${featureId} to the worker via job_send with feature_ids — the rejection reason rides the ticket automatically.`;
		}
	}
	return {
		ok: true,
		feature_id: featureId,
		verdict: params.verdict as "pass" | "reject",
		rejections: feature.rejections,
		stage: feature.stage,
		map_phase: run.phase,
		next,
	};
}

function buildMapStatusDigest(): string {
	const run = activeMapRun;
	if (!run) return "";
	const lines = [`## MAP STATUS — ${run.title} (${run.mapId})`];
	for (const feature of run.features.values()) {
		const mark = feature.status === "passed" ? "[x]" : feature.status === "dispatched" ? "[>]" : "[ ]";
		const stageNote = feature.stage === "normal" ? "" : ` stage:${feature.stage}`;
		lines.push(`${mark} ${feature.id} ${feature.title} (rejections ${feature.rejections}${stageNote})`);
	}
	const checkpoint = activeMapCheckpointTurn;
	if (checkpoint) {
		lines.push("", `Arriving handback: job ${checkpoint.jobId} covering ${checkpoint.featureIds.join(", ")}`);
		for (const id of checkpoint.featureIds) {
			const feature = run.features.get(id);
			if (!feature) continue;
			lines.push(`${feature.id} — ${feature.title}; acceptance:`);
			for (const criterion of feature.acceptance) lines.push(`- ${criterion}`);
			if (feature.lastRejectReason) lines.push(`last rejection: ${feature.lastRejectReason}`);
			if (feature.stage === "escalated") {
				lines.push(
					`${feature.id} is ESCALATED: if rejecting again, close this job (escalated) and re-dispatch to a stronger worker${mapEscalateModelHint()}.`,
				);
			} else if (feature.stage === "main_direct") {
				lines.push(`${feature.id} is MAIN-DIRECT: implement it yourself in this window; edit tools are unlocked.`);
			}
		}
	}
	return lines.join("\n");
}

function contextIsIdle(ctx: ExtensionContext): boolean {
	try {
		const probe = (ctx as { isIdle?: () => boolean }).isIdle;
		return typeof probe === "function" ? probe.call(ctx) !== false : true;
	} catch {
		return false;
	}
}

function contextHasPendingUserInput(ctx: ExtensionContext): boolean {
	try {
		const probe = (ctx as { hasPendingMessages?: () => boolean }).hasPendingMessages;
		return typeof probe === "function" ? probe.call(ctx) === true : false;
	} catch {
		return false;
	}
}

function directiveAutoTurnCooldownMs(): number {
	const raw = process.env.JARVIS_AUTO_TURN_COOLDOWN_S?.trim();
	if (raw === "0" || raw?.toLowerCase() === "off" || raw?.toLowerCase() === "false") return 0;
	const parsed = raw ? Number(raw) : 20;
	return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed * 1000) : 20_000;
}

function directiveAutoTurnCooldownRemainingMs(): number {
	const cooldown = directiveAutoTurnCooldownMs();
	if (cooldown <= 0 || lastUserActivityAtMs <= 0) return 0;
	return Math.max(0, lastUserActivityAtMs + cooldown - Date.now());
}

function setPendingJobStatus(ctx: ExtensionContext, text: string | undefined): void {
	try {
		ctx.ui.setStatus("jlc-job", text);
	} catch {
		// ignore
	}
}

async function holdDirectiveAutoTurnForUserActivity(ctx: ExtensionContext, reason: string): Promise<boolean> {
	const directives = await fetchPendingDirectives("directive", 1, { consume: false });
	const first = directives[0];
	if (first?.job?.job_id) {
		setPendingJobStatus(ctx, `[JOB handback 1 pending - ${reason}]`);
	} else {
		setPendingJobStatus(ctx, undefined);
	}
	return directives.length > 0;
}

async function sendDirectiveReport(item: SidecarDirectiveItem, assistantText: string): Promise<void> {
	const toWindow = String(item.from_window ?? "").trim();
	if (!toWindow || toWindow === "external") return;
	const body = oneLineForSummary(assistantText, 2000);
	if (!body.trim()) return;
	await postSidecar<SidecarDirectiveSendResponse>(
		"/directives",
		{
			kind: "report",
			to_window: toWindow,
			message: body,
		},
		"POST",
		10000,
	);
}

async function fireDirectiveTurn(item: SidecarDirectiveItem, pi: ExtensionAPI): Promise<void> {
	const body = String(item.body ?? "").trim();
	if (!body) return;
	// DEBUG (env-gated): persist every directive that actually triggers a turn, so a
	// live run can correlate (by ts/pid) which directive woke a window -- in
	// particular to identify the trigger of the residual "empty wake" turn. The
	// sidecar's in-memory subturn events vanish on shutdown; this file survives.
	if (process.env.JARVIS_AGENT_SDK_DEBUG) {
		try {
			fs.appendFileSync(
				"C:/jarvis-code_v1.01/sidecar/logs/pi_directive_turns.jsonl",
				`${JSON.stringify({
					ts: Date.now() / 1000,
					pid: process.pid,
					event: "fire_directive_turn",
					id: item.id,
					kind: item.kind,
					from_window: item.from_window,
					to_window: item.to_window,
					gan_id: item.gan?.gan_id,
					job_id: item.job?.job_id,
					body_preview: body.slice(0, 200),
				})}\n`,
				"utf8",
			);
		} catch {
			/* debug net must never break the turn */
		}
	}
	pendingDirectiveAutoTurn = item;
	const ganHeader = directiveGanHeader(item);
	if (ganHeader) {
		appendTransientSystemDirective(ganHeader);
	} else {
		const jobHeader = await directiveJobHeader(item);
		if (jobHeader) {
			appendTransientSystemDirective(jobHeader);
		} else {
			const sender = String(item.from_window ?? "").trim();
			if (sender && sender !== "external") {
				appendTransientSystemDirective(
					[
						`[Directive turn from window ${sender}]`,
						"Your final text is delivered back to the sender as a passive report; it does not trigger a turn there.",
						`If the sender must act on your reply (e.g., a review request or follow-up task), call the send_directive tool addressed to "${sender}" — writing the word directive in plain text sends nothing.`,
					].join("\n"),
				);
			}
		}
	}
	sendJarvisChatNotice(
		pi,
		`[Directive received · window ${directiveWindowLabel(item.from_window)}] ${directiveSessionBadge(item)}${oneLineForSummary(body, 180)}`,
	);
	try {
		pi.sendUserMessage(body);
	} catch (err) {
		pendingDirectiveAutoTurn = undefined;
		console.error(`[jarvis:directives] sendUserMessage failed: ${String(err)}`);
	}
}

async function checkDirectiveSensor(
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	options?: { autoPromptActive?: boolean },
): Promise<void> {
	if (directiveSensorRunning) return;
	if (!sidecarHealthy) return;
	directiveSensorRunning = true;
	try {
		await collectDirectiveReports(pi);
		if (agentTurnActive || pendingDirectiveAutoTurn || activeDirectiveTurn || options?.autoPromptActive) return;
		if (!contextIsIdle(ctx)) return;
		if (contextHasPendingUserInput(ctx)) {
			await holdDirectiveAutoTurnForUserActivity(ctx, "user input pending");
			return;
		}
		const cooldownRemainingMs = directiveAutoTurnCooldownRemainingMs();
		if (cooldownRemainingMs > 0) {
			await holdDirectiveAutoTurnForUserActivity(ctx, "user activity cooldown");
			return;
		}
		setPendingJobStatus(ctx, undefined);
		const directives = await fetchPendingDirectives("directive", 1);
		const directive = directives[0];
		if (directive) await fireDirectiveTurn(directive, pi);
	} finally {
		directiveSensorRunning = false;
	}
}

function directivePollIntervalMs(): number {
	const raw = process.env.JARVIS_DIRECTIVE_POLL_MS?.trim();
	if (raw === "0" || raw?.toLowerCase() === "off" || raw?.toLowerCase() === "false") return 0;
	const parsed = Number(raw);
	if (Number.isFinite(parsed) && parsed >= 1000) return Math.floor(parsed);
	return 7000;
}

function startDirectiveIdlePoll(
	ctx: ExtensionContext,
	pi: ExtensionAPI,
	autoPromptActive: () => boolean = () => false,
): void {
	if (directivePollTimer) return;
	const intervalMs = directivePollIntervalMs();
	if (intervalMs <= 0) return;
	directivePollTimer = setInterval(() => {
		void checkDirectiveSensor(ctx, pi, { autoPromptActive: autoPromptActive() });
	}, intervalMs);
	(directivePollTimer as { unref?: () => void }).unref?.();
}

function clearDirectiveIdlePoll(): void {
	if (!directivePollTimer) return;
	clearInterval(directivePollTimer);
	directivePollTimer = undefined;
}

function controlBridgePollIntervalMs(): number {
	const raw = process.env.JARVIS_CONTROL_BRIDGE_POLL_MS?.trim();
	if (raw === "0" || raw?.toLowerCase() === "off" || raw?.toLowerCase() === "false") return 0;
	const parsed = Number(raw);
	if (Number.isFinite(parsed) && parsed >= 250) return Math.floor(parsed);
	return 750;
}

async function fetchPendingControlBridgeRequests(limit = 1): Promise<SidecarControlBridgeRequest[]> {
	const params = new URLSearchParams({ limit: String(Math.max(1, limit)) });
	const data = await postSidecar<SidecarControlBridgePendingResponse>(
		`/control/pending?${params.toString()}`,
		undefined,
		"GET",
		5000,
	);
	if (!data?.ok || !Array.isArray(data.requests)) return [];
	return data.requests;
}

async function answerControlBridgeRequest(requestId: string, result: unknown): Promise<void> {
	const data = await postSidecar<SidecarControlBridgeAnswerResponse>(
		`/control/${encodeURIComponent(requestId)}/answer`,
		{ result },
		"POST",
		10000,
	);
	if (data?.ok === false) {
		console.error(`[jarvis:control-bridge] answer failed: ${data.error ?? "unknown error"}`);
	}
}

async function handleControlBridgeRequest(
	request: SidecarControlBridgeRequest,
	ctx: ExtensionContext,
	_pi: ExtensionAPI,
): Promise<void> {
	const requestId = typeof request.id === "string" ? request.id.trim() : "";
	if (!requestId) return;
	const kind = String(request.kind ?? "").trim();
	let result:
		| AskUserResult
		| SidecarSpawnWindowResponse
		| MapCreateResult
		| FeatureVerdictResult
		| { ok: false; error: string };
	if (kind === "ask_user") {
		result = await runAskUserDialog(request.payload, ctx.signal, ctx);
	} else if (kind === "spawn_window") {
		const payload =
			request.payload && typeof request.payload === "object" ? (request.payload as Record<string, unknown>) : {};
		const rawInitialDirective = typeof payload.initial_directive === "string" ? payload.initial_directive.trim() : "";
		const isSecondEyesReviewSpawn =
			secondEyesRequestedThisTurn && !secondEyesReviewSpawnedThisTurn && !activeDirectiveTurn;
		const initialDirective = isSecondEyesReviewSpawn
			? buildSecondEyesReviewDirective(rawInitialDirective)
			: rawInitialDirective;
		if (isSecondEyesReviewSpawn && !rawInitialDirective) {
			result = {
				ok: false,
				error: secondEyesPlanReadyError(),
			};
			await answerControlBridgeRequest(requestId, result);
			return;
		}
		const featureIds = Array.isArray(payload.feature_ids)
			? payload.feature_ids.map((id) => String(id ?? "").trim()).filter(Boolean)
			: [];
		// gan and job are mutually exclusive (see performWorkerSpawn). Mirror the
		// spawn_window tool's guard (regime A): when the caller requests a GAN round
		// the directive is delivered as gan round 1, so job must be forced off. The
		// bridge path (regime B / agent-sdk) previously omitted this and let a critic
		// spawn (isSecondEyesReviewSpawn -> job) collide with gan:true.
		const bridgeJob =
			payload.gan === true
				? false
				: payload.job === true || isSecondEyesReviewSpawn || (payload.job !== false && featureIds.length > 0);
		result = await performWorkerSpawn({
			initialDirective,
			model: payload.model,
			label: payload.label,
			timeoutSeconds: payload.timeout_seconds,
			gan: payload.gan === true,
			job: bridgeJob,
			issuesOpen: payload.issues_open,
			featureIds,
			isSecondEyesReviewSpawn,
			skipModelAsk: typeof payload.model === "string" && payload.model.trim().length > 0,
			ctx,
		});
	} else if (kind === "map_create") {
		// Regime B (anthropic-agent-sdk) delegates execution back to pi so pi — and
		// ONLY pi — mutates its in-process activeMapRun ledger. Same shared helper
		// the regime-A registerTool execute calls; no split-brain ledger.
		const payload =
			request.payload && typeof request.payload === "object" ? (request.payload as Record<string, unknown>) : {};
		result = await runMapCreate(payload);
	} else if (kind === "feature_verdict") {
		const payload =
			request.payload && typeof request.payload === "object" ? (request.payload as Record<string, unknown>) : {};
		result = await runFeatureVerdict(payload);
	} else {
		result = { ok: false, error: `unsupported control bridge request: ${kind || "(missing kind)"}` };
	}
	await answerControlBridgeRequest(requestId, result);
}

async function checkControlBridgeSensor(ctx: ExtensionContext, pi: ExtensionAPI): Promise<void> {
	if (controlBridgeSensorRunning) return;
	if (!sidecarHealthy) return;
	controlBridgeSensorRunning = true;
	try {
		const requests = await fetchPendingControlBridgeRequests(1);
		const request = requests[0];
		if (request) await handleControlBridgeRequest(request, ctx, pi);
	} catch (error) {
		console.error(`[jarvis:control-bridge] ${String(error)}`);
	} finally {
		controlBridgeSensorRunning = false;
	}
}

function startControlBridgePoll(ctx: ExtensionContext, pi: ExtensionAPI): void {
	if (controlBridgePollTimer) return;
	const intervalMs = controlBridgePollIntervalMs();
	if (intervalMs <= 0) return;
	controlBridgePollTimer = setInterval(() => {
		void checkControlBridgeSensor(ctx, pi);
	}, intervalMs);
	(controlBridgePollTimer as { unref?: () => void }).unref?.();
}

function clearControlBridgePoll(): void {
	if (!controlBridgePollTimer) return;
	clearInterval(controlBridgePollTimer);
	controlBridgePollTimer = undefined;
}

function reportEncoderFailure(pi: ExtensionAPI, error: string | null | undefined): void {
	const detail = (error ?? "").trim();
	if (!detail) return;
	sendJarvisChatNotice(pi, `JLC encoder error: ${detail}\nKeeping previous memory.`);
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
	const yolo = jarvisYoloMode() ? "YOLO · " : "";
	const prefix = `${yolo}${currentWindowLabel ? `${currentWindowLabel} · ` : ""}`;
	if (state === "checking") return `${ANSI_YELLOW}${prefix}JLC checking${ANSI_RESET}`;
	if (state === "down") return `${ANSI_RED}${prefix}JLC down${ANSI_RESET}`;
	if (state === "degraded") return `${ANSI_RED}${prefix}JLC degraded${ANSI_RESET}`;
	const color = isProjectRoute(currentRoute) ? ANSI_RED : ANSI_YELLOW;
	const label = `JLC:${routeStatusLabel(currentRoute)}`;
	const body = isProjectRoute(currentRoute) && projectName ? `${label}:${projectName}` : label;
	return `${color}${prefix}${body}${ANSI_RESET}`;
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

function readJarvisUiState(): Record<string, unknown> {
	try {
		const statePath = jarvisUiStatePath();
		if (!fs.existsSync(statePath)) return {};
		const parsed = JSON.parse(fs.readFileSync(statePath, "utf-8")) as unknown;
		return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
			? (parsed as Record<string, unknown>)
			: {};
	} catch {
		return {};
	}
}

function writeJarvisUiState(state: Record<string, unknown>): boolean {
	try {
		const statePath = jarvisUiStatePath();
		fs.mkdirSync(path.dirname(statePath), { recursive: true });
		fs.writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf-8");
		return true;
	} catch {
		return false;
	}
}

function loadDeepdiveThinkingPreference(): SupportedThinkingLevel | undefined {
	if (deepdiveThinkingPreferenceLoaded) return deepdiveThinkingPreference;
	deepdiveThinkingPreferenceLoaded = true;
	const parsed = readJarvisUiState();
	const parsedLevel = normalizeThinkingLevel(String(parsed.deepdiveThinkingLevel ?? ""));
	if (parsedLevel) {
		deepdiveThinkingPreference = parsedLevel;
	}
	return deepdiveThinkingPreference;
}

function saveDeepdiveThinkingPreference(level: SupportedThinkingLevel): void {
	deepdiveThinkingPreferenceLoaded = true;
	deepdiveThinkingPreference = level;
	const state = readJarvisUiState();
	state.deepdiveThinkingLevel = level;
	writeJarvisUiState(state);
}

function loadSubagentModelUserSet(): boolean {
	if (subagentModelUserSetLoaded) return subagentModelUserSet;
	subagentModelUserSetLoaded = true;
	const state = readJarvisUiState();
	subagentModelUserSet = state.subagentModelUserSet === true;
	return subagentModelUserSet;
}

function saveSubagentModelUserSet(value = true): void {
	subagentModelUserSetLoaded = true;
	subagentModelUserSet = value;
	const state = readJarvisUiState();
	state.subagentModelUserSet = value;
	writeJarvisUiState(state);
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
	if (route === "chat_control") return CHAT_CONTROL_ROUTE_PROMPT;
	if (route === "unregistered_coding") return UNREGISTERED_CODING_ROUTE_PROMPT;
	if (route === "heavy_deepdive") return HEAVY_DEEPDIVE_ROUTE_PROMPT;
	if (route === "deepdive") return DEEPDIVE_ROUTE_PROMPT;
	return CHAT_ROUTE_PROMPT;
}

function baseModePromptForRoute(route: EffectiveTurnRoute): string {
	if (route === "chat_control") return CHAT_CONTROL_MODE_PROMPT;
	if (route === "unregistered_coding") return UNREGISTERED_CODING_MODE_PROMPT;
	if (route === "heavy_deepdive") return `${DEEPDIVE_MODE_PROMPT}\n\n${HEAVY_DEEPDIVE_OVERLAY_PROMPT}`;
	if (route === "deepdive") return DEEPDIVE_MODE_PROMPT;
	return CHAT_MODE_PROMPT;
}

// Regime split (2026-06-26, Jun+JARVIS): CLARIFY_DIRECTIVE_PROMPT exists to make the
// agent-sdk regime (B) ask before barreling ahead — the SDK owns its own loop and
// ignores soft prose otherwise. The pi-native regime (A / OpenAI-completions) ALREADY
// owns the loop and follows the directive literally, so stacking it there over-clarified
// and tripled the subturn count (~13 -> ~34) on builds. Inject it ONLY for the
// sidecar-chat-proxy regime. Regime A keeps its build-dialogue via DEEPDIVE's
// PLAN_DIALOGUE_PROMPT. This function is the single chokepoint for the clarify directive:
// regime-specific prose lives behind the regime gate so an OpenAI-path change can never
// bleed into the agent-sdk path (and vice versa). Add future regime-B-only nudges here,
// gated — never ungated on a shared prompt.
export function modePromptForRoute(route: EffectiveTurnRoute, provider: string | undefined): string {
	const base = baseModePromptForRoute(route);
	if (isSidecarChatProxyProvider(provider)) {
		return `${base}\n\n${CLARIFY_DIRECTIVE_PROMPT}`;
	}
	return base;
}

// The directive-bus toolset (list_windows/send_directive/gan_send/gan_close/
// job_send/job_close/spawn_window) is deliberately NOT route-gated: workers execute most
// directives on coding routes and must be able to hand work back on the bus —
// stripping them there forced models into prose handbacks and hand-rolled
// shell launchers (2026-06-11 live runs).
const CHAT_ROUTE_ONLY_TOOL_NAMES = new Set([
	"docs_search",
	"package_info",
	"set_window_label",
	"set_chat_model",
	"set_subagent_model",
	"set_encoder_model",
]);
const CHAT_ROUTE_BASE_ALLOWED_TOOL_NAMES = new Set<string>(["ask_user"]);
// pi's native coding tools. Chat routes already carry pi's full base prompt (which
// frames the model as a tool-using coder), so stripping these from chat both wasted
// that framing and dead-ended "do X for me" requests with a false "no tool exposed"
// (Jun, 2026-06-22: "I said save tokens, not disable the function"). They ride every
// chat-family route by default; the lean diet (JARVIS_LEAN_CHAT_TOOLS=1) is opt-in
// for weak local models that mis-fire when handed many tools.
const PI_BASIC_TOOL_NAMES = new Set<string>(["read", "bash", "edit", "write", "grep", "find", "ls"]);
// Capability tools every chat-family route exposes by default. These are
// "absence kills normal work" abilities (project registry, JARVIS.md, recall,
// docs, web, background process). Only the heavy multi-window orchestration
// workflow (spawn_window/job_*/send_directive/gan_*/map_create/feature_verdict)
// stays scoped to deepdive+, since its absence does NOT kill normal coding — it
// is a workflow mode, not a base ability. Gating these by the fallible route
// classifier was the single largest source of "model says it has no tool" flow
// deaths (Jun, 2026-06-23: "90%+ of the breakage is missing tools"). The lean
// diet (JARVIS_LEAN_CHAT_TOOLS=1) still strips back for weak local models.
const CHAT_ROUTE_CAPABILITY_TOOL_NAMES = new Set<string>([
	"register_project",
	"switch_project",
	"unregister_project",
	"update_jarvis_md",
	"recall_turns",
	"delegate_subagent",
	"ultracode",
	"retrieve_output",
	"search_within",
	"docs_search",
	"package_info",
	"web_search",
	"web_fetch",
	"managed_process",
	"list_windows",
]);
const CHAT_ROUTE_SPAWN_ALLOWED_TOOL_NAMES = new Set<string>(["ask_user", "list_windows", "job_send", "spawn_window"]);
const CHAT_ROUTE_TOOL_ACTION_ALLOWED_TOOL_NAMES = new Set<string>([
	"ask_user",
	"list_windows",
	"job_send",
	"spawn_window",
	"set_window_label",
	"set_chat_model",
	"set_subagent_model",
	"set_encoder_model",
]);
const CHAT_CONTROL_ALLOWED_TOOL_NAMES = new Set<string>([
	"ask_user",
	"list_windows",
	"job_send",
	"spawn_window",
	"set_window_label",
	"set_chat_model",
	"set_subagent_model",
	"set_encoder_model",
	"web_search",
	"web_fetch",
]);
const HANDOFF_BUS_TOOL_NAMES = new Set(["send_directive", "gan_send", "gan_close", "job_send", "job_close"]);
// Turn-ending background handoffs: the model intentionally ends the turn empty
// because a background job will re-engage this window later (e.g. ultracode).
// Keep this scoped - gan_send/job_send ride directive handbacks or worker flows.
const BACKGROUND_HANDOFF_TOOL_NAMES = new Set(["ultracode"]);
const DELEGATION_INITIATE_TOOL_NAMES = new Set(["map_create", "spawn_window", ...HANDOFF_BUS_TOOL_NAMES]);
const SECOND_EYES_ALLOWED_TOOL_NAMES = new Set([
	"read",
	"ls",
	"grep",
	"find",
	"search_within",
	"recall_turns",
	"delegate_subagent",
	"ultracode",
	"retrieve_output",
	"bash",
	"web_search",
	"web_fetch",
	"docs_search",
	"package_info",
	"list_windows",
	...HANDOFF_BUS_TOOL_NAMES,
]);
const UNREGISTERED_PROJECT_MEMORY_TOOL_NAMES = new Set([
	"switch_project",
	"register_project",
	"unregister_project",
	"update_jarvis_md",
]);

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
			const raw = typeof record.name === "string" ? record.name : typeof fn?.name === "string" ? fn.name : "";
			const name = normalizeToolSchemaNameRaw(raw);
			if (name) names.push(name);
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
			const raw = typeof record.name === "string" ? record.name : typeof fn?.name === "string" ? fn.name : "";
			const name = normalizeToolSchemaNameRaw(raw);
			if (name) names.push(name);
		}
	}
	return [...new Set(names)];
}

function isSlashCommand(text: string, command: string): boolean {
	const normalized = text.trim().toLowerCase();
	return normalized === command || normalized.startsWith(`${command} `);
}

function escapeRegexLiteral(value: string): string {
	return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function isDegenerateProjectToken(value: string): boolean {
	const token = value.trim().toLowerCase();
	if (!token) return true;
	if (token === "app" || token === "web") return true;
	if (/^\d+$/.test(token)) return true;
	return token.length < 3;
}

function isMatchableProjectToken(value: string): boolean {
	return !isDegenerateProjectToken(value);
}

function projectTokenMatchesText(token: string, text: string): boolean {
	const normalized = token.trim().toLowerCase();
	if (!isMatchableProjectToken(normalized)) return false;
	const pattern = new RegExp(`(?<![\\w-])${escapeRegexLiteral(normalized)}(?![\\w-])`, "i");
	return pattern.test(text);
}

function projectMatchesText(project: CachedProject, text: string): boolean {
	const normalizedText = text.toLowerCase();
	if (projectTokenMatchesText(project.project_id, normalizedText)) return true;
	if (projectTokenMatchesText(project.slug, normalizedText)) return true;
	if (project.name !== project.slug && projectTokenMatchesText(project.name, normalizedText)) return true;
	return false;
}

async function findRegisteredProjectForRouteHint(hint: unknown): Promise<CachedProject | undefined> {
	const text = String(hint ?? "").trim();
	if (!text) return undefined;
	if (!projectCacheLoaded) {
		try {
			await refreshProjectCache();
		} catch {
			// Route hints are best-effort; /context can still resolve below.
		}
	}
	const pathMatches = projectCache.filter(
		(project) =>
			sameProjectPath(text, project.path) ||
			sameProjectPath(text, project.code_path) ||
			extractAbsolutePathsFromText(text).some(
				(candidate) => sameProjectPath(candidate, project.path) || sameProjectPath(candidate, project.code_path),
			),
	);
	if (pathMatches.length === 1) return pathMatches[0];
	const textMatches = projectCache.filter((project) => projectMatchesText(project, text));
	return textMatches.length === 1 ? textMatches[0] : undefined;
}

function normalizeRouteClassifierRoute(route: unknown): EffectiveTurnRoute | undefined {
	const value = String(route ?? "").trim();
	if (
		value === "chat" ||
		value === "chat_control" ||
		value === "unregistered_coding" ||
		value === "deepdive" ||
		value === "heavy_deepdive"
	) {
		return value;
	}
	return undefined;
}

// --- cluster-2 fail-safe: deterministic "clear build request" detector --------
// The /route_turn classifier is a WEAK chat model (reasoning=none). When it
// silently returns route=chat for an obvious "build/create/edit this" request
// (D4), or fails entirely (D2), the turn is tool-stripped to ask_user and ends
// with no recovery — the "테트리스 만들어줘 makes no files" swamp.
//
// This predicate is a CONSERVATIVE backstop, NOT a router. It only fires when a
// strong build/create/edit VERB co-occurs with a code/file/app TARGET, so plain
// chat and questions ("안녕?", "이거 왜 이래?") stay chat. It never *promotes*
// the route on its own for the chat case — it only ARMS the existing no-action
// recovery (FIX A) and provides a fail-safe coding route when the classifier is
// unavailable (FIX B). Keep it tight: a false positive here turns chat into a
// coding turn, so prefer missing edge cases over over-promotion.
const BUILD_INTENT_VERB_RE =
	/(만들어|만들자|만들게|만들어줘|만들어주|구현|개발|작성해|코딩|코드\s*짜|짜줘|짜봐|고쳐|수정해|리팩터|리팩토|빌드해|생성해|\b(build|create|make|implement|write|code|add|fix|refactor|generate|scaffold|set\s*up|develop)\b)/i;
const BUILD_INTENT_TARGET_RE =
	/(게임|앱|프로그램|스크립트|함수|클래스|모듈|컴포넌트|파일|코드|페이지|사이트|봇|api|서버|기능|테트리스|클론|툴|유틸|\b(game|app|website|web\s*site|component|function|class|module|script|file|code|page|server|bot|api|feature|tool|util|clone|cli|endpoint|test|tests)\b)/i;

export function detectClearBuildIntent(userText: string): boolean {
	const text = (userText ?? "").trim();
	if (!text) return false;
	// Ignore JLC-internal retry/marker turns — those carry their own intent state
	// and must not be re-promoted by surface text matching.
	if (isWorkerToolsRetryPrompt(text) || isSlashCommand(text.toLowerCase(), "/chat")) {
		return false;
	}
	return BUILD_INTENT_VERB_RE.test(text) && BUILD_INTENT_TARGET_RE.test(text);
}

function routeClassifierDecisionIndicatesAction(decision: SidecarRouteTurnResponse | undefined): boolean {
	const route = normalizeRouteClassifierRoute(decision?.route);
	if (!route) return false;
	if (route !== "chat") return true;
	return (
		decision?.create_project === true ||
		decision?.register_project === true ||
		Boolean(decision?.code_path_hint) ||
		(!!decision?.expected_action && decision.expected_action !== "none")
	);
}

function routeTelemetryUserTextHead(userText: string): string {
	return String(userText ?? "")
		.replace(/\s+/g, " ")
		.trim()
		.slice(0, 80);
}

function recordRouteDecisionTelemetry(args: {
	decision: SidecarRouteTurnResponse;
	clearBuildIntent: boolean;
	routeSource: string;
}): void {
	const needsClarification = args.decision.needs_clarification === true;
	const classifierRoute = normalizeRouteClassifierRoute(args.decision.route) ?? String(args.decision.route ?? "");
	const clarifyOverrodeBuildIntent = needsClarification && args.clearBuildIntent;
	const data = {
		classifier_route: classifierRoute,
		needs_clarification: needsClarification,
		clear_build_intent: args.clearBuildIntent,
		effective_route: currentRoute,
		clarify_overrode_build_intent: clarifyOverrodeBuildIntent,
		user_text_len: lastUserMessage.length,
		user_text_head: routeTelemetryUserTextHead(lastUserMessage),
		route_source: args.routeSource,
	};
	recordSubturnDebugEvent("route_decision", data);
	recordTurnTimelineEvent("route_decision", data);
	if (clarifyOverrodeBuildIntent) {
		recordSubturnDebugEvent("route_clarify_override", data);
		recordTurnTimelineEvent("route_clarify_override", data);
	}
}

function enterRouteFromClassifier(route: EffectiveTurnRoute): void {
	if (route === "chat_control") {
		setEffectiveRoute("chat_control");
	} else if (route === "unregistered_coding") {
		enterUnregisteredCoding();
	} else if (route === "heavy_deepdive") {
		enterHeavyProjectWork();
	} else if (route === "deepdive") {
		enterProjectWorkPreservingHeavy();
	}
}

function recentRouteMessages(messages: AgentMessage[]): Array<{ role: string; text: string }> {
	return messages
		.slice(-8)
		.map((message) => ({
			role: String((message as { role?: unknown }).role ?? ""),
			text: stripJarvisMemoryBlock(messageContentToText((message as { content?: unknown }).content)).slice(0, 1600),
		}))
		.filter((item) => item.role && item.text.trim());
}

function mandatoryPreRouteEnabled(): boolean {
	const value = String(process.env.JARVIS_ROUTE_PREFLIGHT ?? "")
		.trim()
		.toLowerCase();
	if (!value) return true;
	return !(value === "0" || value === "false" || value === "no" || value === "off");
}

async function applyRouteDecisionBeforeContext(decision: SidecarRouteTurnResponse | undefined): Promise<boolean> {
	lastRouteClassifierDecision = decision;
	const classifierIndicatesAction = routeClassifierDecisionIndicatesAction(decision);
	// FIX A (cluster-2 D4): a clear build request must never end in a silent
	// no-action turn. If the weak classifier returns chat with no action intent
	// for an obvious "build/create/edit X" turn, treat it as action-intent so the
	// existing no-action recovery (decidePostTurnRecovery / route-skill retry)
	// arms. This only flips a boolean (it does NOT change the first call's tools),
	// so the first chat subturn still gets the ask_user diet — but if the model
	// answers in prose without acting, recovery fires and the NEXT subturn runs
	// with coding tools.
	const buildIntent = detectClearBuildIntent(lastUserMessage);
	lastRouteClassifierActionIntent = classifierIndicatesAction || buildIntent;
	const route = normalizeRouteClassifierRoute(decision?.route);
	// FIX B (cluster-2 D2): classifier genuinely UNAVAILABLE — sidecar unreachable
	// (decision === undefined) or it returned an error (ok === false). A transient
	// router hiccup must not silently turn a build request into ask_user-only. When
	// the turn clearly asks to build, fail SAFE onto a coding route so file tools
	// survive. We deliberately scope this to true transport/error failures: a
	// SUCCESSFUL classifier response that returns chat (or omits a route) is a
	// real product decision (natural-language new-project requests start in
	// chat-entry and let the chat model respond first), so it is left as chat and
	// only FIX A's armed action-intent guards against a silent no-action ending.
	const classifierUnavailable = decision === undefined || decision.ok === false;
	if (classifierUnavailable) {
		if (buildIntent) {
			enterUnregisteredCoding();
			routePromotedByClassifierThisTurn = true;
			return true;
		}
		return false;
	}
	if (!route) return false;
	const classifierNewProject = decision.create_project === true || decision.register_project === true;
	if (decision.needs_clarification) {
		setEffectiveRoute("chat");
		if (decision.clarification) {
			appendTransientSystemDirective(
				["[Route clarification]", decision.clarification, "Ask this clarification before using coding tools."].join(
					"\n",
				),
			);
		}
		recordRouteDecisionTelemetry({
			decision,
			clearBuildIntent: buildIntent,
			routeSource: "classifier_clarification",
		});
		return false;
	}
	// Language-agnostic enforcement (2026-06-24): create_project/register_project is the
	// classifier's SEMANTIC, language-neutral "this is a NEW project" signal (the LLM sets
	// it for any language, and it distinguishes "등록해줘" from "등록 어떻게?"). A flagged
	// new project MUST enter the deepdive build route regardless of which route string the
	// classifier picked -- never a no-build route like chat_control. The weak classifier
	// sometimes mislabels the route (live: a clear new-project build landed on chat_control)
	// even while setting the flags correctly, so code guarantees the build route here. The
	// universal clarify directive still fires inside deepdive ("confirm before building").
	if (classifierNewProject) {
		// eafac008 forced the deepdive route here but never created the project, so
		// activeProjectPath stayed null while DEEPDIVE_ROUTE_PROMPT claims a registered
		// project with an active code path. That contradiction made the model defer the
		// build to a "next implementation turn" and ask a second "build now?". Create +
		// select the project now (mirroring the maybeHandlePendingProjectCreation confirm
		// path) so the deepdive prompt is truthful and the model builds in THIS turn after
		// its clarify. Registering the (empty) project folder is not "build", so the
		// universal clarify still gates file writes. If no slug is resolvable or creation
		// fails, still force deepdive (never regress to chat_control) and let the
		// [Project clarification] directive ask the user to register/name it this turn.
		const newProjectSlug = (decision.project_slug ?? decision.target_project_hint ?? "").trim();
		if (newProjectSlug) {
			const response = await postSidecar<SidecarSwitchResponse>("/switch_project", {
				slug_or_name: newProjectSlug,
				code_path: decision.code_path_hint ?? undefined,
				auto_create: true,
			});
			if (response?.ok && response.path) {
				patchProjectCache(response);
				activeProjectPath = response.path;
				activeCodePath = response.code_path ?? activeCodePath;
				activeProjectId = response.project_id ?? activeProjectId;
			}
		}
		enterProjectWorkPreservingHeavy();
		// Mandatory: front-load the new-artifact ask_user gate so the autonomous
		// SDK loop asks before building (the sidecar PreToolUse hook also enforces it).
		pendingNewArtifactAskUserGate = true;
		routePromotedByClassifierThisTurn = true;
		return true;
	}
	// A clear build-intent the weak classifier returned as chat OR chat_control enters
	// deepdive directly. chat_control is a misroute blind spot: it carries no file tools
	// and (without create/register flags) neither rescue path above fires, so a build
	// request dead-ends in an ask_user loop ("테트리스 만들어줘 makes no files"). Scope
	// stays tight via the conservative detectClearBuildIntent (build VERB + TARGET);
	// directive/worker turns are excluded upstream in shouldCallRouteClassifier.
	if ((route === "chat" || route === "chat_control") && buildIntent) {
		enterProjectWorkPreservingHeavy();
		routePromotedByClassifierThisTurn = true;
		return true;
	}
	if (route === "chat") return false;
	if (isProjectRoute(route) && !classifierNewProject) {
		const hintedProject = await findRegisteredProjectForRouteHint(
			decision.target_project_hint ?? decision.code_path_hint,
		);
		if (hintedProject) {
			activeProjectPath = hintedProject.path;
			activeCodePath = hintedProject.code_path;
			activeProjectId = hintedProject.project_id;
		}
	}
	enterRouteFromClassifier(route);
	routePromotedByClassifierThisTurn = true;
	return true;
}

export function shouldCallRouteClassifier(userText: string, explicitChat: boolean): boolean {
	if (!mandatoryPreRouteEnabled()) return false;
	if (explicitChat) return false;
	// Route is normally frozen once it leaves "chat" (no per-turn re-classification).
	// Exception: a clear build command must be able to escape a chat_control turn it
	// was misrouted into, so re-classify and let the build-intent escalation in
	// applyRouteDecisionBeforeContext promote it to deepdive. Directive/worker turns
	// are still excluded by the guards below.
	if (currentRoute !== "chat" && !(currentRoute === "chat_control" && detectClearBuildIntent(userText))) {
		return false;
	}
	if (activeDirectiveTurn || activeSecondEyesReviewTurn || activeSecondEyesMainTurn || secondEyesRequestedThisTurn) {
		return false;
	}
	if (isWorkerToolsRetryPrompt(userText)) return false;
	return true;
}

async function maybeClassifyRouteBeforeContext(
	userText: string,
	messages: AgentMessage[],
	cwdHint: string,
	pi: ExtensionAPI,
	explicitChat: boolean,
): Promise<boolean> {
	if (!shouldCallRouteClassifier(userText, explicitChat)) return false;
	const decision = await postSidecar<SidecarRouteTurnResponse>("/route_turn", {
		user_message: userText,
		cwd_hint: cwdHint,
		active_project_path: currentActiveProjectHint(),
		recent_messages: recentRouteMessages(messages),
		pending_project: pendingProjectCreate
			? { slug_or_name: pendingProjectCreate.slugOrName, code_path: pendingProjectCreate.codePath }
			: undefined,
		bench_conv_id: benchConvId(pi),
	});
	return applyRouteDecisionBeforeContext(decision);
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

// Chat baseline thinking = "medium" (2026-06-17, Jun): clean 3-step ladder —
// chat "medium" → deepdive "high" → heavy xhigh, each a deliberate step up.
// (medium was the verified-good chat level; reverted from the 2026-06-16 "low".)
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
		verifyContinuationCount = 0;
		safetyConfirmedKeys = new Set<string>();
		currentTodoList = [];
		clearReadBeforeEditRegistry();
		// Deferred tools: narrow the active set at runtime (action methods can't
		// run during extension loading). Re-diets on each fresh session; load_tool
		// promotions within a session survive (session_start fires once per session).
		if (deferredToolsEnabled()) {
			applyDeferredToolsDiet(pi);
		}
		// Synchronous on purpose: the map run must be restored before the first
		// directive poll can fire a checkpoint turn for it.
		restoreMapRunFromPointer();
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
		startDirectiveIdlePoll(ctx, pi, () => !!autoPromptState);
		startControlBridgePoll(ctx, pi);
		if (sidecarHealthy) {
			await refreshProjectCache();
			const sidecarStatus = await postSidecar<SidecarStatusResponse>("/status", undefined, "GET");
			currentWindowLabel = sanitizeWindowLabel(sidecarStatus?.window_label);
			const chatRole = sidecarStatus?.roles?.chat;
			if (shouldAutoSwapToSidecarChatProxy(ctx, chatRole)) {
				await ensureSidecarChatProxyLive(pi, ctx, { provider: chatRole.provider, model: chatRole.model });
			} else {
				setChatModelStatus(ctx, chatRole);
			}
			try {
				ctx.ui.setStatus("jarvis", sidecarHealthy ? jlcLabel("ok") : jlcLabel("down"));
			} catch {
				/* stale */
			}
			setupRequired = sidecarStatus?.setup_required === true;
			if (sidecarStatus?.memory_write_enabled === false) {
				try {
					ctx.ui.notify(memoryWriteDisabledNotice(sidecarStatus.memory_write_disabled_reason), "warning");
				} catch {
					/* stale */
				}
			}
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
		clearDirectiveIdlePoll();
		clearControlBridgePoll();
		await stopAllManagedProcesses();
	});

	pi.on("context", async (event, ctx) => {
		if (DEBUG_CONTEXT) console.error("[jlc:debug-context-handler] ENTER");
		const messages = event.messages;
		const rawUserText = stripJarvisMemoryBlock(latestUserText(messages));
		const userText = workerToolsRetryOriginalUserRequest(rawUserText) ?? rawUserText;
		if (!userText.trim()) return;
		const isPendingDirectiveUserTurn = pendingDirectiveMatchesText(rawUserText);
		const userTurnKey = latestUserTurnKey(messages, rawUserText);
		const isVerifyIncompleteFollowUp = isVerifyIncompletePrompt(rawUserText);
		const isNewUserTurn = !isVerifyIncompleteFollowUp && userTurnKey !== lastObservedUserTurnKey;
		if (isNewUserTurn) {
			if (!isPendingDirectiveUserTurn) lastUserActivityAtMs = Date.now();
			lastObservedUserTurnKey = userTurnKey;
			lastInjectedContextTurnKey = "";
			pendingProjectSwitchContextRefresh = false;
			agentTurnActive = true;
			providerCallCountThisTurn = 0;
			lastProviderCallRoute = undefined;
			lastProviderToolsBeforeFilter = [];
			lastProviderToolsAfterFilter = [];
			lastProviderActionIntentMatch = false;
			lastProviderChatFilterApplied = false;
			lastProviderRoutePromotedByClassifier = false;
			lastRouteClassifierDecision = undefined;
			lastRouteClassifierActionIntent = false;
			expectedToolActivityThisTurn = false;
			routePromotedByClassifierThisTurn = false;
			verifyContinuationCount = 0;
			if (isWorkerToolsRetryPrompt(rawUserText)) {
				workerToolsRetryInFlight = true;
				expectedToolActivityThisTurn = true;
			} else {
				workerToolsRetryInFlight = false;
			}
			workerWindowContextInjectedThisTurn = false;
			resetProviderCallCeilingState();
			lastTurnStartedAtMs = Date.now();
			lastProviderStartedAtMs = undefined;
			setEffectiveRoute("chat");
			checkpointToolEvents = [];
			lastAssistantPartialText = "";
			interruptCheckpointSavedThisTurn = false;
			turnCheckpointScope = undefined;
			resetSubturnLogState();
			resetJarvisTurnChoreographyState();
			secondEyesRequestedThisTurn = false;
			secondEyesReviewSpawnedThisTurn = false;
			secondEyesReminderInjectedThisTurn = false;
			askUserIssuedThisProviderCall = false;
			activeSecondEyesReviewTurn = false;
			activeSecondEyesMainTurn = false;
			activeSecondEyesHeavyTurn = false;
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
		markDirectiveTurnIfMatching(rawUserText);
		markSecondEyesMarkerTurnIfPresent(rawUserText);
		if (activeSecondEyesReviewTurn || activeSecondEyesMainTurn) {
			enterSecondEyesRoute();
		}
		if (isNewUserTurn) {
			try {
				await collectDirectiveReports(pi);
				injectPendingDirectiveReports();
			} catch {
				/* report injection is best-effort */
			}
		}
		const normalizedUser = userText.trim().toLowerCase();
		const explicitChat = isSlashCommand(normalizedUser, "/chat");
		const explicitDeepdive = isSlashCommand(normalizedUser, "/deepdive");
		const utteredDeepdiveLevel = explicitDeepdive ? parseDeepdiveReasoningUtterance(userText) : undefined;
		if (explicitChat && !secondEyesRequestedThisTurn && !activeSecondEyesReviewTurn && !activeSecondEyesMainTurn) {
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
		if (!explicitChat) {
			await maybeClassifyRouteBeforeContext(userText, messages, cwdHint, pi, explicitChat);
		}
		if (!isPendingDirectiveUserTurn && routeDecisionCriticMode(lastRouteClassifierDecision)) {
			secondEyesRequestedThisTurn = true;
			secondEyesReviewSpawnedThisTurn = false;
			secondEyesReminderInjectedThisTurn = false;
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
		if (secondEyesRequestedThisTurn || activeSecondEyesReviewTurn || activeSecondEyesMainTurn) {
			enterSecondEyesRoute();
		}
		const classifierRoute = normalizeRouteClassifierRoute(lastRouteClassifierDecision?.route);
		const classifierNewProject =
			lastRouteClassifierDecision?.create_project === true || lastRouteClassifierDecision?.register_project === true;
		if (
			isProjectRoute(currentRoute) &&
			!activeProjectPath &&
			(explicitDeepdive || (classifierRoute !== undefined && classifierRoute !== "chat"))
		) {
			appendTransientSystemDirective(
				classifierNewProject
					? [
							"[New project]",
							"This is a NEW project and it is not registered yet (auto-create did not yield an active project). Register it THIS turn (call switch_project with a clear project name; the default project root is used when no path is given), then build the files and verify in this same turn. Do not defer to a later turn and do not ask a second 'build now?' — the clarify was the confirmation.",
						].join("\n")
					: [
							"[Project clarification]",
							"Deepdive needs a registered workspace project for this action. If the target project is clear, call switch_project first; otherwise ask which project to use before editing, launching, or updating JARVIS.md.",
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
				sendJarvisChatNotice(
					pi,
					`JLC initialization error: ${degradation.slice("JLC context degraded:".length).trim()}`,
				);
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
		const contextForInjection = contextWithTodoForRoute(response.context, currentRoute) ?? response.context;
		lastContextResponse = { ...response, context: contextForInjection };
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
				ctx.ui.notify("Project is ambiguous. I will ask a clarification question first.", "warning");
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
		return {
			messages: injectMemoryIntoLatestUser(event.messages, contextForInjection, response.workspace_block),
		};
	});

	pi.on("before_agent_start", async (event, ctx) => {
		installInterruptInputCheckpointHook(ctx, pi);
		const messages = (event as { messages?: AgentMessage[] }).messages ?? [];
		const promptText = (event as { prompt?: string }).prompt ?? "";
		const rawPromptText = promptText.trim();
		const userText = rawPromptText ? effectiveUserTextFromInternalRetry(rawPromptText) : lastUserMessage;
		agentTurnActive = true;
		activeModelProviderThisTurn = String(ctx.model?.provider ?? "").trim() || undefined;
		markDirectiveTurnIfMatching(userText);
		markSecondEyesMarkerTurnIfPresent(userText);
		if (
			!activeDirectiveTurn &&
			!secondEyesRequestedThisTurn &&
			routeDecisionCriticMode(lastRouteClassifierDecision)
		) {
			secondEyesRequestedThisTurn = true;
			secondEyesReviewSpawnedThisTurn = false;
			secondEyesReminderInjectedThisTurn = false;
		}
		const normalizedUser = userText.trim().toLowerCase();
		const explicitChat = isSlashCommand(normalizedUser, "/chat");
		const explicitDeepdive = isSlashCommand(normalizedUser, "/deepdive");
		const utteredDeepdiveLevel = explicitDeepdive ? parseDeepdiveReasoningUtterance(userText) : undefined;
		const classifierRouteForPrompt = normalizeRouteClassifierRoute(lastRouteClassifierDecision?.route);
		if (explicitChat && !secondEyesRequestedThisTurn && !activeSecondEyesReviewTurn && !activeSecondEyesMainTurn) {
			clearActiveProjectState();
			setEffectiveRoute("chat");
		} else if (explicitDeepdive) {
			enterHeavyProjectWork(utteredDeepdiveLevel);
		} else if (
			classifierRouteForPrompt &&
			classifierRouteForPrompt !== "chat" &&
			lastRouteClassifierDecision?.needs_clarification !== true
		) {
			// A classifier-flagged new project keeps the forced deepdive build route even when
			// the weak classifier mislabeled the route string as a no-build route like
			// chat_control. applyRouteDecisionBeforeContext already forced deepdive (and
			// created the project), so re-deriving the raw route here would silently regress
			// the build to chat_control and dead-end it -- the /context already ran in deepdive.
			if (
				lastRouteClassifierDecision?.create_project === true ||
				lastRouteClassifierDecision?.register_project === true
			) {
				enterProjectWorkPreservingHeavy();
			} else {
				enterRouteFromClassifier(classifierRouteForPrompt);
			}
			routePromotedByClassifierThisTurn = true;
		}
		if (secondEyesRequestedThisTurn || activeSecondEyesReviewTurn || activeSecondEyesMainTurn) {
			enterSecondEyesRoute();
		}
		expectedToolActivityThisTurn =
			!explicitChat &&
			(lastRouteClassifierActionIntent ||
				activeDirectiveTurn !== undefined ||
				secondEyesRequestedThisTurn ||
				activeSecondEyesReviewTurn ||
				activeSecondEyesMainTurn);

		const activeProjectForPreflight = isProjectRoute(currentRoute)
			? (lastContextResponse?.active_project_path ?? currentActiveProjectHint())
			: undefined;
		refreshTurnCheckpointScope();
		const preflight = activeProjectForPreflight
			? "[P1] project memory is already injected and is saved automatically. Do not narrate the memory cycle — no 'reading JARVIS.md', 'analyzing the project', or 'updating JARVIS.md' lines. Just answer."
			: "";
		const secondEyesInstruction =
			secondEyesRequestedThisTurn && !activeDirectiveTurn
				? [
						"[CRITIC MODE REQUESTED]",
						"The user requested Critic Mode. This is project work in deepdive or heavy_deepdive, never chat mode.",
						"The main window owns user choice gathering, design/trend recon, architecture decisions, plan draft, implementation, fixes, and final user report.",
						"If user-facing choices are still unresolved, call ask_user first and stop; do not call any other tool in the same response.",
						"Once user choices and the main-window plan draft are settled, dispatch exactly one review-only plan critique before implementation.",
						`If the user named an existing live worker/window, use job_send to that worker. Only call spawn_window(job=true) when no worker target exists. The directive must include ${SECOND_EYES_PLAN_READY_MARKER}, the project path, user choices/Q&A, and the main draft. Do not ask the worker to perform recon, select architecture, invent the plan, implement, or mutate files.`,
						"Do not create files or modify code before this Critic Mode review dispatch.",
						currentSecondEyesHeavy()
							? "This Critic Mode job is heavy deepdive; preserve the heavy marker in the worker directive and use HEAVY_DEEPDIVE."
							: "This Critic Mode job is deepdive; keep both main and worker at least DEEPDIVE.",
						"The worker is review-only: it may inspect and run bounded verification, but it never implements or fixes.",
						"After the plan handback, ask_user only if a real user choice controls quality; otherwise the main window implements.",
						"After implementation, send the same worker a review request with job_send. Apply confirmed Must-fix items yourself.",
						"Allow at most two plan critique rounds and at most two fix/review cycles. If unresolved, close/escalate and report to the user.",
					].join("\n")
				: "";
		const overlay = transientSystemDirective.trim();
		transientSystemDirective = "";
		// New-artifact ask_user gate (consume-once): set on a classifierNewProject
		// route decision, front-loaded in `parts` below so the mandate is salient.
		// Regime split (2026-06-26): this is an agent-sdk-regime nudge — its enforcement
		// PreToolUse hook only runs in the sidecar/regime-B path, so in regime A it was a
		// dangling unenforced prompt that only added ask-first pressure (a subturn
		// multiplier). Consume the flag regardless (so it never leaks to a later turn), but
		// only surface the prompt text in the regime that actually enforces it.
		const newArtifactGate =
			pendingNewArtifactAskUserGate && isSidecarChatProxyProvider(activeModelProviderThisTurn)
				? NEW_ARTIFACT_ASK_USER_GATE_PROMPT
				: "";
		pendingNewArtifactAskUserGate = false;
		const existingPrompt = (event as { systemPrompt?: string }).systemPrompt ?? "";
		const modePrompt = activeSecondEyesReviewTurn
			? secondEyesModePrompt(SECOND_EYES_MODE_PROMPT)
			: activeSecondEyesMainTurn
				? secondEyesModePrompt(SECOND_EYES_MAIN_MODE_PROMPT)
				: activeMapCheckpointTurn
					? MAP_CHECKPOINT_MODE_PROMPT
					: activeMapSynthesisTurn
						? MAP_SYNTHESIS_PROMPT
						: activeEndGateTurn
							? WHOLE_DELEGATION_END_GATE_PROMPT
							: modePromptForRoute(currentRoute, activeModelProviderThisTurn);
		const routePrompt = routePromptForRoute(currentRoute);
		const mapDigest = activeMapCheckpointTurn || activeMapSynthesisTurn ? buildMapStatusDigest() : "";
		// CACHE: the system prompt is the head of the cacheable prefix (system →
		// tools → history). Keep it byte-stable across steady-state turns so the
		// active model's automatic prefix cache survives. Stable directives lead;
		// only the rare/sporadic deltas (mapDigest on MAP turns, one-shot overlay)
		// trail. The volatile workspace feed (live folder listing, which mutates on
		// every folder create/register) is NOT in system — it rides the
		// <jarvis_workspace> tail on the latest user message (see the context hook /
		// injectMemoryIntoLatestUser), so a new folder no longer busts
		// system+tools+history. Memory is unchanged; only position moves.
		const parts: string[] = [];
		if (existingPrompt.trim()) parts.push(existingPrompt);
		parts.push(LOCAL_LANGUAGE_PROMPT);
		if (newArtifactGate) parts.push(newArtifactGate);
		parts.push(modePrompt);
		parts.push(routePrompt);
		if (preflight) parts.push(preflight);
		if (secondEyesInstruction) parts.push(secondEyesInstruction);
		if (mapDigest) parts.push(mapDigest);
		if (overlay) parts.push(overlay);

		// M7.9 visibility: the mode swap is invisible in the transcript, so
		// announce checkpoint/synthesis turns explicitly (chat notice + footer).
		if (activeSecondEyesReviewTurn) {
			sendJarvisChatNotice(
				pi,
				"[Critic Mode] independent review turn — mutation tools locked, verification tools preserved",
			);
			setWorkStatus(ctx, "Critic Mode review");
		} else if (activeSecondEyesMainTurn) {
			sendJarvisChatNotice(pi, "[Critic Mode] main implementation/fix turn — worker stays review-only");
			setWorkStatus(ctx, "Critic Mode main");
		} else if (activeMapCheckpointTurn) {
			const ids = activeMapCheckpointTurn.featureIds.join(", ");
			sendJarvisChatNotice(
				pi,
				`[MAP checkpoint] ${ids} verification turn — verification/dispatch only, implementation tools locked, memory injection skipped`,
			);
			setWorkStatus(ctx, `MAP checkpoint: ${ids}`);
		} else if (activeMapSynthesisTurn) {
			sendJarvisChatNotice(
				pi,
				"[MAP synthesis] all features PASS — final integration verification + user report turn",
			);
			setWorkStatus(ctx, "MAP synthesis");
		} else if (activeEndGateTurn) {
			sendJarvisChatNotice(
				pi,
				"[End gate] whole-delegation handback — real-ness check + user playtest checklist + close",
			);
			setWorkStatus(ctx, "end gate");
		}

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
		const toolName =
			normalizeToolSchemaNameRaw(String(event.toolName ?? "")) || String(event.toolName ?? "").toLowerCase();
		const askUserSameResponseBlock = maybeBlockToolAfterAskUser(toolName);
		if (askUserSameResponseBlock) return askUserSameResponseBlock;
		const directiveSpawnBlock = maybeBlockDirectiveSpawnToolCall(toolName);
		if (directiveSpawnBlock) return directiveSpawnBlock;
		const secondEyesInitialSpawnBlock = maybeBlockSecondEyesInitialSpawnToolCall(toolName, event.input);
		if (secondEyesInitialSpawnBlock) return secondEyesInitialSpawnBlock;
		const existingWorkerSpawnBlock = maybeBlockExistingWorkerRequestSpawnToolCall(toolName);
		if (existingWorkerSpawnBlock) return existingWorkerSpawnBlock;
		const existingWorkerDirectiveBlock = maybeBlockExistingWorkerRequestPassiveDirectiveToolCall(toolName);
		if (existingWorkerDirectiveBlock) return existingWorkerDirectiveBlock;
		const mapCheckpointBlock = maybeBlockMapCheckpointToolCall(toolName);
		if (mapCheckpointBlock) return mapCheckpointBlock;
		const secondEyesBlock = maybeBlockSecondEyesToolCall(toolName);
		if (secondEyesBlock) return secondEyesBlock;
		const secondEyesBashBlock = maybeBlockSecondEyesReviewBashToolCall(toolName, event.input);
		if (secondEyesBashBlock) return secondEyesBashBlock;
		const secondEyesMainBlock = maybeBlockSecondEyesMainToolCall(toolName);
		if (secondEyesMainBlock) return secondEyesMainBlock;
		const lockedResourceBlock = maybeBlockLockedResourceToolCall(toolName, event.input);
		if (lockedResourceBlock) return lockedResourceBlock;
		const repeatedFailureBlock = maybeBlockRepeatedFailureToolCall(toolName, event.input);
		if (repeatedFailureBlock) return repeatedFailureBlock;
		const pcCeilingBlock = maybeBlockProviderCallCeilingToolCall(toolName, event.input);
		if (pcCeilingBlock) return pcCeilingBlock;
		const processKillBlock = await maybeBlockProcessKillToolCall(toolName, event.input, ctx, safetyConfirmedKeys);
		if (processKillBlock) return processKillBlock;
		const jarvisLauncherBlock = maybeBlockJarvisLauncherToolCall(toolName, event.input);
		if (jarvisLauncherBlock) return jarvisLauncherBlock;
		if (toolName === "bash") {
			return maybeConfirmRiskyBashToolCall(event.input, ctx, safetyConfirmedKeys);
		}
		const isRead = READ_TOOL_NAMES.has(toolName);
		const isCode = CODE_TOOL_NAMES.has(toolName);
		if (!isRead && !isCode) return undefined;
		const readBeforeEditBlock = maybeBlockEditBeforeRead(toolName, event.input, ctx);
		if (readBeforeEditBlock) return readBeforeEditBlock;

		const rawPath = extractToolPath(event.input);
		if (!rawPath) return undefined;
		const toolCwd = extractToolCwd(event.input);
		if (isProjectRoute(currentRoute)) {
			const shortPath = rawPath.length > 80 ? `...${rawPath.slice(-77)}` : rawPath;
			notifyWork(ctx, `JLC: running ${toolName} ${shortPath}`);
		}
		const absPath = resolveTurnMutationPath(rawPath, toolCwd, ctx);
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
		askUserIssuedThisProviderCall = false;
		const beforeMetrics = extractPayloadTokens(event.payload);
		trimBeforeTokensSum += beforeMetrics.message_tokens;
		lastToolSchemaTokens = beforeMetrics.tool_schema_tokens;
		const beforeTrim = DEBUG_CONTEXT ? summarizeProviderPayload(event.payload) : "";
		// Checkpoint turns skip the JHB/project context block: the ticket and the
		// MAP STATUS digest already carry the dispatcher's intent, and the block
		// would be re-sent on every verification tool call (the dominant cost of
		// a checkpoint). Synthesis turns keep it — the user report needs memory.
		const payloadWithContext = ensureContextInProviderPayload(
			event.payload,
			activeMapCheckpointTurn ? undefined : contextWithTodoForRoute(lastContextResponse?.context, currentRoute),
		);
		const legacyPayload = trimPayloadToCurrentJarvisTurn(payloadWithContext, { stateCarry: false });
		const nextPayload = trimPayloadToCurrentJarvisTurn(payloadWithContext);
		const legacyMetrics = extractPayloadTokens(legacyPayload);
		const uncompressedAfterMetrics = extractPayloadTokens(nextPayload);
		const nextProviderCall = providerCallCountThisTurn + 1;
		// Evidence stores fired at tool_execution_end may still be in flight
		// (pi does not await that handler before the next provider call).
		await awaitJarvisPendingEvidenceStores();
		await awaitPendingToolLessonObserves();
		const compressionOutcome = jarvisCompressProviderPayload(nextPayload);
		turnCompressedOutputsTotal += compressionOutcome.compressed_tool_outputs;
		turnCompressionSavedTotal += compressionOutcome.compression_saved_tokens_est;
		const reportStopPayload = applyRepeatedFailureReportStop(
			applyLockedResourceReportStop(compressionOutcome.payload),
		);
		const payloadWithWorkerContext = await applyWorkerWindowContext(reportStopPayload);
		const payloadBeforeRouteFilters = applyProviderCallCeiling(
			applyToolLessonHints(applySecondEyesReminder(payloadWithWorkerContext)),
			nextProviderCall,
		);
		const providerCallRoute = currentRoute;
		const toolsBeforeFilter = providerPayloadToolNames(payloadBeforeRouteFilters);
		const actionIntentMatch = lastRouteClassifierActionIntent || expectedToolActivityThisTurn;
		const chatFilterApplied = chatRouteToolDietWouldApply() && toolsBeforeFilter.length > 0;
		const filteredProviderPayload = filterSecondEyesMainTools(
			filterSecondEyesTools(filterMapCheckpointTools(filterChatRouteOnlyTools(payloadBeforeRouteFilters))),
		);
		const providerPayload = applySecondEyesProviderPhase(filteredProviderPayload);
		const afterMetrics = extractPayloadTokens(providerPayload);
		const toolsAfterFilter = providerPayloadToolNames(providerPayload);
		lastProviderCallRoute = providerCallRoute;
		lastProviderToolsBeforeFilter = toolsBeforeFilter;
		lastProviderToolsAfterFilter = toolsAfterFilter;
		lastProviderActionIntentMatch = actionIntentMatch;
		lastProviderChatFilterApplied = chatFilterApplied;
		lastProviderRoutePromotedByClassifier = routePromotedByClassifierThisTurn;
		lastToolSchemaTokens = afterMetrics.tool_schema_tokens;
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
			route: providerCallRoute,
			tools_before_filter: toolsBeforeFilter,
			tools_after_filter: toolsAfterFilter,
			action_intent_match: actionIntentMatch,
			chat_filter_applied: chatFilterApplied,
			route_promoted_by_classifier: routePromotedByClassifierThisTurn,
			second_eyes_phase: currentSecondEyesProviderPhase(),
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
		await checkDirectiveSensor(ctx, pi, { autoPromptActive: !!autoPromptState });
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
		if (isConsequentialSubturnAction(event.toolName, descriptor, event.isError)) {
			appendSubturnLedger(`${tool}${descriptor ? ` ${descriptor}` : ""} => ${status}`);
		}
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
		maybeObserveToolLesson(event.toolName, metadata.command, event.isError === true, output);
		recordRepeatedFailureToolOutcome({
			toolCallId: event.toolCallId,
			toolName: event.toolName,
			command: metadata.command,
			isError: event.isError === true,
			outputText: output,
		});
		if (!event.isError) recordReadBeforeEditFromMetadata(event.toolName, metadata, ctx);
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
		// Capture the regime-B tool-activity trailer from the RAW streamed text BEFORE
		// sanitizeAssistantMessageInPlace (below) strips the sentinel from the live
		// message in place. This is the EARLIEST in-place strip — message_end runs on
		// the same already-stripped object, so capturing only there is too late. agent_end
		// reads the live (stripped) message, so without this capture the reconstruct
		// finds nothing. Only overwrite when non-empty so intermediate chunks without the
		// sentinel cannot clobber a captured trailer.
		const sdkTrailerUpdate = parseJarvisSdkToolTrailerFromText(text);
		if (sdkTrailerUpdate.length > 0) lastSdkToolTrailerRecords = sdkTrailerUpdate;
		// The model's [MODE:X] marker no longer mutates the route -- the upfront
		// /route_turn classifier is authoritative. The marker is still emitted and
		// stripped from user-facing text below, but it carries no routing effect.
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
		// Capture the regime-B tool-activity trailer from the RAW text BEFORE the
		// in-place / replacement sanitize below strips the sentinel. agent_end reads
		// the live message AFTER it has been stripped, so without this capture the
		// reconstruct finds nothing. Only overwrite when non-empty so a later empty
		// message_end in the same turn cannot clobber a captured trailer.
		const sdkTrailer = parseJarvisSdkToolTrailerFromText(rawText);
		if (sdkTrailer.length > 0) lastSdkToolTrailerRecords = sdkTrailer;
		const toolNames = assistantToolNames(assistantMessage);
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
			agentTurnActive = false;
			return;
		}

		const assistantMessage = latestAssistantMessage(event.messages);
		const turnUsage = summarizeAssistantUsage(event.messages);
		finishTurnMeter(ctx, assistantMessage);
		const assistantTextRaw = assistantMessage ? contentToText(assistantMessage.content) : "";
		const assistantText = sanitizeAssistantText(assistantTextRaw);
		const agentEndRawUserText = stripJarvisMemoryBlock(latestUserText(event.messages));
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
			agentTurnActive = false;
			activeDirectiveTurn = undefined;
			activeMapCheckpointTurn = undefined;
			activeMapSynthesisTurn = false;
			activeEndGateTurn = false;
			activeSecondEyesReviewTurn = false;
			activeSecondEyesMainTurn = false;
			activeSecondEyesHeavyTurn = false;
			return;
		}
		if (DEBUG_CONTEXT && turnUsage) {
			console.error(
				`[jlc:debug-usage] input=${turnUsage.input ?? 0} output=${turnUsage.output ?? 0} total=${turnUsage.totalTokens ?? 0} cache_read=${turnUsage.cacheRead ?? 0} cache_write=${turnUsage.cacheWrite ?? 0}`,
			);
		}
		// O(N) regression trace — appends per-turn chat_in + breakdown to jsonl.
		// Read ~/.jarvis-code/chat_in_trace.jsonl after 5+ turns to see whether
		// chat_in grows linearly and which sub-field carries it.
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
				provider_call_route: lastProviderCallRoute,
				tools_before_filter: lastProviderToolsBeforeFilter,
				tools_after_filter: lastProviderToolsAfterFilter,
				action_intent_match: lastProviderActionIntentMatch,
				chat_filter_applied: lastProviderChatFilterApplied,
				route_promoted_by_classifier: lastProviderRoutePromotedByClassifier,
				expected_tool_activity: expectedToolActivityThisTurn,
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

		if (!lastUserMessage.trim()) {
			completeSubturnLog("idle", assistantText);
			toolEvents = [];
			turnCheckpointScope = undefined;
			resetJarvisTurnChoreographyState();
			resetToChatMode(ctx, pi);
			agentTurnActive = false;
			activeDirectiveTurn = undefined;
			activeMapCheckpointTurn = undefined;
			activeMapSynthesisTurn = false;
			activeEndGateTurn = false;
			activeSecondEyesReviewTurn = false;
			activeSecondEyesMainTurn = false;
			activeSecondEyesHeavyTurn = false;
			await checkDirectiveSensor(ctx, pi, { autoPromptActive: !!autoPromptState });
			return;
		}
		// Turn-loss guard: a real user turn whose run ends with empty assistant
		// text (degenerate tool-call loop, dead final call) must still reach
		// /turn — raw store is the lossless record, and a directive/job
		// counterpart needs more than silence (live 2026-06-12: a 19-call turn
		// vanished from every memory layer through the old idle skip).
		let assistantTextForTurn = assistantText;
		let terminalReason = inferAssistantTerminalReason(assistantMessage, assistantText);
		const harnessSecondEyesSpawned = await maybeHarnessSpawnSecondEyesReview({
			assistantText,
			terminalReason,
			ctx,
			pi,
		});
		if (!harnessSecondEyesSpawned && !assistantTextForTurn.trim()) {
			const guardStopReason = assistantMessage?.stopReason;
			const backgroundHandoffTool = turnInvokedBackgroundHandoffTool(event.messages);
			let emptyAssistantEvent = "turn_loss_guard";
			if (guardStopReason === "aborted") {
				// User-initiated abort: persist the turn but do not blame the model
				// and do not notify - the user pressed Esc themselves.
				assistantTextForTurn = buildEmptyAssistantTurnLossMarker("aborted");
			} else if (guardStopReason === "error") {
				const detail = oneLineForSummary(assistantMessage?.errorMessage || "provider error", 200);
				assistantTextForTurn = buildEmptyAssistantTurnLossMarker("provider_error", detail);
				sendJarvisChatNotice(pi, `Provider error ended the turn: ${detail}. Turn history was preserved.`);
			} else if (activeDirectiveTurn && directiveTurnBusReplySent) {
				const summary = directiveHandbackCompletionSummary();
				assistantTextForTurn = buildDirectiveHandbackCompletionMarker(summary);
				terminalReason = "stop";
				emptyAssistantEvent = "directive_handback_empty_final";
				sendJarvisChatNotice(pi, summary);
			} else if (backgroundHandoffTool) {
				assistantTextForTurn = buildBackgroundHandoffMarker(backgroundHandoffTool);
				terminalReason = "stop";
				emptyAssistantEvent = "background_handoff_empty_final";
				sendJarvisChatNotice(
					pi,
					`⏳ ${backgroundHandoffTool} is running in the background — I'll report the synthesized result when it completes.`,
				);
			} else {
				assistantTextForTurn = buildEmptyAssistantTurnLossMarker();
				sendJarvisChatNotice(
					pi,
					`The model ended the turn with an empty response (provider calls: ${providerCallCountThisTurn}). Turn history was preserved.`,
				);
			}
			recordSubturnDebugEvent(emptyAssistantEvent, {
				provider_calls: providerCallCountThisTurn,
				tool_events: toolEvents.length,
				stop_reason: guardStopReason ?? "",
				terminal_reason: terminalReason,
				content: assistantTextForTurn,
			});
		}
		const workerToolsRetryEligible = shouldQueueWorkerToolsRetry(event.messages, assistantTextForTurn);
		const recoveryDecision = decidePostTurnRecovery({
			workerToolsRetryEligible,
			modifiedFilePaths: turnSuccessfulFileMutations.map((mutation) => mutation.path),
			verificationRanThisTurn,
			route: currentRoute,
			provider: activeModelProviderThisTurn,
			verifyContinuationCount,
		});
		recordSubturnDebugEvent("post_turn_recovery_decision", {
			decision: recoveryDecision.kind,
			terminal_reason: terminalReason,
			provider_calls: providerCallCountThisTurn,
			worker_tools_retry_eligible: workerToolsRetryEligible,
			modified_file_count: turnSuccessfulFileMutations.length,
			verification_ran: verificationRanThisTurn,
			verify_continuation_count: verifyContinuationCount,
			current_route: currentRoute,
			provider: activeModelProviderThisTurn ?? "",
			harness_second_eyes_spawned: harnessSecondEyesSpawned,
		});
		recordTurnTimelineEvent("recovery_decision", {
			decision: recoveryDecision.kind,
			terminal_reason: terminalReason,
			tools_after_filter: lastProviderToolsAfterFilter,
			worker_tools_retry_eligible: workerToolsRetryEligible,
			modified_file_count: turnSuccessfulFileMutations.length,
			verification_ran: verificationRanThisTurn,
			verify_continuation_count: verifyContinuationCount,
			harness_second_eyes_spawned: harnessSecondEyesSpawned,
		});
		let workerToolsRetryQueuedThisAgentEnd = false;
		let verificationFollowUpQueuedThisAgentEnd = false;
		if (recoveryDecision.kind === "worker_tools_followup") {
			workerToolsRetryInFlight = true;
			const retryPrompt = buildWorkerToolsRetryPrompt(lastUserMessage, assistantTextForTurn);
			recordSubturnDebugEvent("worker_tools_retry_queued", {
				user_message: oneLineForSummary(lastUserMessage, 400),
				assistant_text: oneLineForSummary(assistantTextForTurn, 300),
				tools_after_filter: lastProviderToolsAfterFilter,
				provider_calls: providerCallCountThisTurn,
			});
			try {
				pi.sendUserMessage(retryPrompt, { deliverAs: "followUp" });
				workerToolsRetryQueuedThisAgentEnd = true;
				setWorkStatus(ctx, "JLC: worker tools enabled; queued follow-up.");
			} catch {
				workerToolsRetryInFlight = false;
				sendJarvisChatNotice(pi, "Worker-tool retry could not be queued; preserved the signal in turn history.");
			}
		} else if (recoveryDecision.kind === "verify_incomplete") {
			const verifyPrompt = buildVerificationIncompletePrompt(lastUserMessage, turnSuccessfulFileMutations);
			recordSubturnDebugEvent("verification_floor_followup_queued", {
				files: verificationGateMutationPaths(turnSuccessfulFileMutations),
				verify_continuation_count: verifyContinuationCount,
				provider_calls: providerCallCountThisTurn,
			});
			try {
				pi.sendUserMessage(verifyPrompt, { deliverAs: "followUp" });
				verifyContinuationCount += 1;
				verificationFollowUpQueuedThisAgentEnd = true;
				setWorkStatus(ctx, "JLC: verification floor queued.");
			} catch {
				sendJarvisChatNotice(
					pi,
					"Verification-floor follow-up could not be queued; preserved the signal in turn history.",
				);
			}
		}
		if (workerToolsRetryInFlight && isWorkerToolsRetryPrompt(agentEndRawUserText)) {
			workerToolsRetryInFlight = false;
		}
		recordSubturnDebugEvent("turn_terminal", {
			terminal_reason: terminalReason,
			stop_reason: assistantMessage?.stopReason ?? "",
			provider_calls: providerCallCountThisTurn,
			worker_tools_retry_in_flight: workerToolsRetryInFlight,
			verification_followup_queued: verificationFollowUpQueuedThisAgentEnd,
		});
		recordTurnTimelineEvent("turn_terminal", {
			terminal_reason: terminalReason,
			stop_reason: assistantMessage?.stopReason ?? "",
			worker_tools_retry_in_flight: workerToolsRetryInFlight,
			queued_follow_up: workerToolsRetryQueuedThisAgentEnd || verificationFollowUpQueuedThisAgentEnd,
			verification_followup_queued: verificationFollowUpQueuedThisAgentEnd,
		});

		// Regime-B memory sensor (item 1): the agent-sdk SDK executes tools
		// internally, so pi.on("tool_execution_end") never fired this turn and
		// turnSuccessfulFileMutations is empty. Reconstruct it (and toolEvents)
		// from the adapter's trailer through pi's SAME regime-A writer BEFORE the
		// post-turn consumers run, so JARVIS.md patch / workspace auto-register /
		// /turn all see the observed work. Gated to regime B (provider gate) AND
		// the empty-list guard so regime A is byte-for-byte untouched and a future
		// tool_execution_end firing in B cannot double-record.
		if (isSidecarChatProxyProvider(activeModelProviderThisTurn) && turnSuccessfulFileMutations.length === 0) {
			const reconstructed = reconstructJarvisTurnFromSdkTrailer(
				assistantMessage,
				event.messages,
				ctx,
				lastSdkToolTrailerRecords,
			);
			if (reconstructed > 0) {
				recordTurnTimelineEvent("sdk_tool_trailer_reconstructed", {
					records: reconstructed,
					mutations: turnSuccessfulFileMutations.length,
				});
			}
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
			assistant_message: assistantTextForTurn,
			tool_events: toolEvents,
			llm_meta: buildTurnLlmMeta(
				assistantMessage,
				ctx,
				lastContextResponse,
				lastTurnPromptSnapshot,
				toolEvents,
				turnUsage,
				terminalReason,
			),
			bench_conv_id: benchConvId(pi),
			origin: activeDirectiveTurn ? "monologue_directive" : "user",
			origin_window: activeDirectiveTurn?.from_window ?? jarvisOriginWindow(),
		});
		recordTurnTimelineEvent("turn_save_result", {
			ok: response ? response.ok !== false : false,
			warning: response?.warning ?? "",
			error: response?.error ?? "",
			memory_write_disabled: response?.memory_write_disabled ?? false,
			memory_write_reenabled: response?.memory_write_reenabled ?? false,
			terminal_reason: terminalReason,
		});
		recordFooterMeterEntry(pi, assistantMessage, lastContextResponse, turnUsage);
		if (!response || response.ok === false) {
			const reason = response?.error ?? response?.warning ?? "sidecar unavailable";
			sendJarvisChatNotice(pi, `JLC turn save failed — this turn was not written to long-term memory: ${reason}`);
		} else if (response.memory_write_disabled) {
			try {
				ctx.ui.notify(memoryWriteDisabledNotice(response.warning), "warning");
			} catch {
				/* stale */
			}
		} else if (response.memory_write_reenabled) {
			try {
				ctx.ui.notify(memoryWriteEnabledNotice(response.memory_write_notice), "info");
			} catch {
				/* stale */
			}
		} else if (response.warning) {
			sendJarvisChatNotice(pi, `JLC memory encoding notice: ${response.warning}`);
		}
		if (activeDirectiveTurn) {
			try {
				if (activeMapSynthesisTurn) {
					// The synthesis self-directive came from this window; echoing the
					// report back to ourselves would fire another auto turn. The final
					// assistant text above IS the user report.
					directiveTurnBusReplySent = true;
					finalizeMapRunAfterSynthesis();
				}
				if (!directiveTurnBusReplySent) {
					const ganId = activeDirectiveTurn.gan?.gan_id;
					if (ganId) {
						// The GAN protocol expects gan_send/gan_close; a turn that ends
						// without either would otherwise leave the counterpart in total
						// silence (the legacy auto-report is suppressed for GAN turns).
						await sendDirectiveReport(
							activeDirectiveTurn,
							`[GAN ${ganId} — counterpart ended its turn without gan_send/gan_close; protocol stalled, raw reply follows] ${assistantTextForTurn}`,
						);
					} else if (activeDirectiveTurn.job?.job_id) {
						const jobId = activeDirectiveTurn.job.job_id;
						const sender = String(activeDirectiveTurn.from_window ?? "").trim();
						const role = String(activeDirectiveTurn.job.role ?? "").trim();
						if (role === "worker" && sender && sender !== "external" && assistantTextForTurn.trim()) {
							// A job handback carries no self-reported fields, so the worker
							// side can be promoted to a real review directive: it wakes the
							// orchestrator and the loop survives model non-compliance.
							// Orchestrator silence stays a passive report (no auto dispatch),
							// so a stall ping-pong is structurally impossible.
							await postSidecar<SidecarDirectiveSendResponse>("/directives", {
								kind: "directive",
								to_window: sender,
								message: activeSecondEyesReviewTurn
									? buildSecondEyesMainHandback(
											`[JOB ${jobId} auto handback — worker ended its turn without job_send; raw reply follows] ${assistantTextForTurn}`,
										)
									: `[JOB ${jobId} auto handback — worker ended its turn without job_send; raw reply follows] ${assistantTextForTurn}`,
								job_target: jobId,
							});
						} else {
							await sendDirectiveReport(
								activeDirectiveTurn,
								`[JOB ${jobId} — counterpart ended its turn without job_send/job_close; protocol stalled, raw reply follows] ${assistantTextForTurn}`,
							);
						}
					} else {
						await sendDirectiveReport(activeDirectiveTurn, assistantTextForTurn);
					}
				}
			} catch {
				/* report delivery is best-effort */
			} finally {
				activeDirectiveTurn = undefined;
				activeMapCheckpointTurn = undefined;
				activeMapSynthesisTurn = false;
				activeEndGateTurn = false;
				activeSecondEyesReviewTurn = false;
				activeSecondEyesMainTurn = false;
				activeSecondEyesHeavyTurn = false;
				directiveTurnBusReplySent = false;
			}
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
		if (verificationFollowUpQueuedThisAgentEnd) {
			writeSubturnCompactState("active");
		} else {
			completeSubturnLog("completed", assistantTextForTurn);
		}
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
		interruptCheckpointSavedThisTurn = false;
		turnCheckpointScope = undefined;
		clearInterruptInputCheckpointHook();
		lastTurnPromptSnapshot = undefined;
		providerCallCountThisTurn = 0;
		resetProviderCallCeilingState();
		if (!verificationFollowUpQueuedThisAgentEnd) {
			resetJarvisTurnChoreographyState();
		}

		// Project work is a one-turn transaction. The next user turn starts as
		// chat unless it explicitly enters deepdive or resolves to a project.
		if (!verificationFollowUpQueuedThisAgentEnd) {
			resetToChatMode(ctx, pi, { updateFooter: !footerResetDeferred });
		}

		if (autoPromptState) {
			agentTurnActive = false;
			await handleAutoPromptTurn(autoPromptState, ctx, pi);
			return;
		}
		agentTurnActive = false;
		if (pendingMapSynthesisPost) {
			pendingMapSynthesisPost = false;
			await postMapSynthesisDirective();
		}
		await checkDirectiveSensor(ctx, pi, { autoPromptActive: false });
	});

	pi.registerTool({
		name: "todo",
		label: "Todo",
		description:
			"Replace the current task checklist for this turn. Use it to track multi-step deepdive work without persisting project memory.",
		promptSnippet: "todo: replace the full task checklist; keep one item in_progress while working.",
		promptGuidelines: [
			"Use todo for multi-step deepdive work that needs an explicit checklist.",
			"Each call replaces the entire list; include every remaining or completed item you still want visible.",
			"Keep exactly one item in_progress unless all work is pending or complete.",
		],
		parameters: Type.Object({
			items: Type.Array(
				Type.Object({
					content: Type.String({ description: "Short task item text." }),
					status: Type.Union([Type.Literal("pending"), Type.Literal("in_progress"), Type.Literal("completed")]),
				}),
				{ description: "The complete replacement todo list." },
			),
		}),
		async execute(_toolCallId, params) {
			currentTodoList = normalizeTodoItems((params as { items?: unknown }).items);
			const inProgressCount = currentTodoList.filter((item) => item.status === "in_progress").length;
			const details = {
				ok: true,
				items: currentTodoList,
				count: currentTodoList.length,
				in_progress_count: inProgressCount,
				semantics: "replace_all",
			};
			return {
				content: [
					{
						type: "text",
						text: `TODO updated (${currentTodoList.length} items)\n${renderTodoList(currentTodoList)}`,
					},
				],
				details,
			};
		},
	});

	pi.registerTool({
		name: "managed_process",
		label: "Managed Process",
		description:
			"Start, inspect, or stop a JARVIS-owned background process by tracked id. Use this for dev servers and long-running commands instead of shell backgrounding, start /B, nohup, or broad process kills.",
		promptSnippet: "managed_process: start/status/list/stop owned background processes; health_url optional.",
		parameters: Type.Object({
			action: Type.Union([
				Type.Literal("start"),
				Type.Literal("status"),
				Type.Literal("stop"),
				Type.Literal("list"),
			]),
			id: Type.Optional(Type.String({ description: "Stable id for this managed process" })),
			command: Type.Optional(Type.String({ description: "Executable to start when action=start" })),
			args: Type.Optional(Type.Array(Type.String(), { description: "argv args for command" })),
			cwd: Type.Optional(Type.String({ description: "Working directory; defaults to current JARVIS cwd" })),
			env: Type.Optional(Type.Record(Type.String(), Type.String())),
			log_path: Type.Optional(Type.String({ description: "Optional stdout/stderr log file" })),
			health_url: Type.Optional(Type.String({ description: "Optional URL to wait for after start" })),
			wait_seconds: Type.Optional(Type.Number({ description: "Seconds to wait for health_url; default 10" })),
		}),
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			await cleanupStaleManagedProcesses();
			const action = String(params.action ?? "")
				.trim()
				.toLowerCase();
			const id = sanitizeManagedProcessId(typeof params.id === "string" ? params.id : undefined);
			if (action === "list") {
				const processes = Array.from(managedProcesses.values()).map(managedProcessDetails);
				const details = { ok: true, processes };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			if (action === "status") {
				const record = managedProcesses.get(id);
				const details = record
					? { ok: true, ...managedProcessDetails(record) }
					: { ok: false, id, error: "managed process id not found" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			if (action === "stop") {
				const record = managedProcesses.get(id);
				const details = record
					? await stopManagedProcess(record)
					: { ok: false, id, error: "managed process id not found" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			if (action !== "start") {
				const details = { ok: false, error: "action must be start, status, stop, or list" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}

			const existing = managedProcesses.get(id);
			if (existing && pidAlive(existing.pid)) {
				const details = { ok: false, id, error: "managed process id is already running", pid: existing.pid };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			if (existing) {
				managedProcesses.delete(id);
				removeManagedProcessState(existing);
			}

			const command = typeof params.command === "string" ? params.command.trim() : "";
			if (!command) {
				const details = { ok: false, id, error: "command is required when action=start" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const cwdRaw =
				typeof params.cwd === "string" && params.cwd.trim() ? params.cwd.trim() : (ctx?.cwd ?? process.cwd());
			const cwd = path.resolve(cwdRaw);
			if (!fs.existsSync(cwd)) {
				const details = { ok: false, id, error: `cwd does not exist: ${cwd}` };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const args = Array.isArray(params.args) ? params.args.map((arg) => String(arg)) : [];
			const extraEnv =
				params.env && typeof params.env === "object" && !Array.isArray(params.env)
					? Object.fromEntries(Object.entries(params.env).map(([key, value]) => [key, String(value)]))
					: {};
			const logPath =
				typeof params.log_path === "string" && params.log_path.trim()
					? path.resolve(cwd, params.log_path.trim())
					: undefined;
			let stdoutFd: number | undefined;
			let stderrFd: number | undefined;
			try {
				if (logPath) {
					fs.mkdirSync(path.dirname(logPath), { recursive: true });
					stdoutFd = fs.openSync(logPath, "a");
					stderrFd = fs.openSync(logPath, "a");
				}
				let record: ManagedProcessRecord | undefined;
				let spawnError: Error | undefined;
				// Resolve the owner start token (once, memoized) BEFORE spawn so the
				// first state write includes it AND no await sits between spawn() and
				// the exit-listener registration below — otherwise a fast-exiting
				// child's "exit" could fire during the (Windows PowerShell) probe and
				// be missed, leaving a stale record and risking a reused-PID token.
				await ensureOwnStartToken();
				const spawnCommand =
					process.platform === "win32" ? buildWindowsManagedProcessCommand(command, args) : command;
				const spawnArgs = process.platform === "win32" ? [] : args;
				const child = spawn(spawnCommand, spawnArgs, {
					cwd,
					detached: process.platform !== "win32",
					env: { ...process.env, ...extraEnv },
					shell: process.platform === "win32",
					stdio: logPath ? (["ignore", stdoutFd ?? "ignore", stderrFd ?? "ignore"] as const) : "ignore",
					windowsHide: true,
				});
				child.on("error", (error) => {
					spawnError = error instanceof Error ? error : new Error(String(error));
					if (record) forgetManagedProcess(record);
				});
				const spawnFailure = await waitForManagedProcessSpawn(child, () => spawnError);
				if (spawnFailure) throw spawnFailure;
				if (!child.pid) throw new Error("process did not expose a pid");
				record = {
					id,
					command,
					args,
					cwd,
					pid: child.pid,
					ownerPid: process.pid,
					startedAt: new Date().toISOString(),
					// Detached Unix children are their own group leader (pgid == pid);
					// Windows containment is via Job Object / tree-kill, not pgid.
					pgid: process.platform !== "win32" ? child.pid : undefined,
					// Filled now if already cached, otherwise by the background capture.
					ownerProcStartToken: cachedOwnStartToken || undefined,
					logPath,
					healthUrl:
						typeof params.health_url === "string" && params.health_url.trim()
							? params.health_url.trim()
							: undefined,
					child,
				};
				managedProcesses.set(id, record);
				writeManagedProcessState(record);
				captureManagedProcessStartToken(record);
				child.once("exit", () => {
					if (record) forgetManagedProcess(record);
				});
				const exitPromise = waitForManagedProcessExit(child);
				child.unref();
				const waitSeconds =
					typeof params.wait_seconds === "number" && Number.isFinite(params.wait_seconds)
						? Math.max(0, params.wait_seconds)
						: 10;
				let health: { ok: boolean; error?: string } | undefined;
				if (record.healthUrl && !signal?.aborted) {
					const readiness = await Promise.race([
						waitForManagedProcessHealth(record.healthUrl, waitSeconds).then((result) => ({
							type: "health" as const,
							result,
						})),
						exitPromise.then((exit) => ({ type: "exit" as const, exit })),
					]);
					health =
						readiness.type === "health"
							? readiness.result
							: { ok: false, error: formatManagedProcessExit(readiness.exit) };
				} else {
					const quickExit = await Promise.race([exitPromise, sleepMs(250).then(() => undefined)]);
					if (quickExit) {
						forgetManagedProcess(record);
						const details = {
							ok: false,
							...managedProcessDetails(record),
							error: formatManagedProcessExit(quickExit),
						};
						return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
					}
				}
				const details = { ok: health ? health.ok : true, ...managedProcessDetails(record), health };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			} catch (error) {
				const details = {
					ok: false,
					id,
					error: error instanceof Error ? error.message : String(error),
				};
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			} finally {
				if (stdoutFd !== undefined) {
					try {
						fs.closeSync(stdoutFd);
					} catch {
						/* best-effort */
					}
				}
				if (stderrFd !== undefined) {
					try {
						fs.closeSync(stderrFd);
					} catch {
						/* best-effort */
					}
				}
			}
		},
		renderCall(args, theme, _context) {
			const action = typeof args.action === "string" ? args.action : "process";
			const id = typeof args.id === "string" ? ` ${args.id}` : "";
			return new Text(theme.fg("toolTitle", theme.bold(`managed ${action}`)) + theme.fg("muted", id), 0, 0);
		},
		renderResult(result, _options, theme, _context) {
			const details = result.details as { ok?: boolean; error?: string; id?: string; pid?: number } | undefined;
			if (!details?.ok) return new Text(theme.fg("error", details?.error ?? "managed process failed"), 0, 0);
			return new Text(theme.fg("success", `${details.id ?? "process"} ${details.pid ?? ""}`.trim()), 0, 0);
		},
	});

	pi.registerTool({
		name: "ask_user",
		label: "Ask User",
		description:
			"Ask 1-6 user questions in one dialog; ok:false when headless. " +
			"Each needs 'question' (or text/prompt alias); 'options' optional (2-6, else free-form).",
		promptSnippet: "ask_user: 1-6 question dialog; options optional; ok:false headless.",
		parameters: ASK_USER_PARAMS,
		prepareArguments: (args) => coerceAskUserParams(args) as Static<typeof ASK_USER_PARAMS>,
		async execute(_toolCallId, params, signal, _onUpdate, ctx) {
			if (activeDirectiveTurn) {
				return askUserErrorResult(ASK_USER_DIRECTIVE_TURN_ERROR);
			}
			const details = await runAskUserDialog(params, signal, ctx);
			return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
		},
		renderCall(args, theme, _context) {
			const count = Array.isArray(args.questions) ? args.questions.length : 0;
			return new Text(
				theme.fg("toolTitle", theme.bold("ask user")) + theme.fg("muted", ` ${count} question(s)`),
				0,
				0,
			);
		},
		renderResult(result, _options, theme, _context) {
			const details = result.details as AskUserResult | undefined;
			if (!details) return new Text(theme.fg("warning", "No answer"), 0, 0);
			if (details.ok === false) return new Text(theme.fg("warning", details.error), 0, 0);
			const summary = details.answers
				.map((answer, index) => {
					const value = answer.answer ?? "(no default)";
					const suffix = answer.was_custom ? " custom" : answer.dismissed ? " default" : "";
					return `${index + 1}. ${value}${suffix}`;
				})
				.join("\n");
			return new Text(theme.fg("success", summary), 0, 0);
		},
	});

	pi.registerTool({
		name: "spawn_window",
		label: "Spawn JARVIS Window",
		description:
			"Open a visible new JARVIS Code terminal window with a fresh pair address; optionally send it an initial directive. " +
			"TRIGGER — spawn ONLY when the user explicitly asks for the work to run in a new, separate, additional, or parallel window, or to use a worker/delegate, in any language. " +
			"If the user did not ask to delegate, do the build directly in THIS window — broad or heavy scope is not a reason to spawn on your own. " +
			"This tool is the ONLY supported way to open a JARVIS window: never launch jarvis.ps1 via bash/PowerShell or a launcher script yourself — " +
			"manual launches mangle the initial prompt and skip the directive-bus wiring. Long or quoted task text is safe here; pass it as initial_directive. " +
			"If the user refers to an existing window (by label, role, or as already open), call list_windows first and address that window " +
			"with job_send by default, or gan_send only for explicit GAN rounds, instead of spawning a duplicate. " +
			"Inside an incoming directive/job/gan turn, delegated turns must not re-delegate; hand needs back to the dispatcher. " +
			"For a pure idle spawn request, call this tool without initial_directive. " +
			"When the user did ask to delegate work, pass the task as initial_directive and set job=true so the handback wakes this window; " +
			"its [REPORT] arrives back here automatically when the work is done, but only while this window is idle — " +
			"after spawning, end your turn and wait; do not poll for the result. " +
			"Optionally pass model as provider/model, or as a bare model name — the sidecar routes it to a provider with usable auth (API key, OAuth, or Agent SDK login) " +
			"and reports the pick in model_routing; if several equally preferred providers offer it, the error lists them so you can ask the user which one. " +
			"If the user names a worker model, pass the exact model string in model; JARVIS does not parse natural-language model names from chat text, and asks once if model is omitted. " +
			"Omit label unless the user explicitly names the window: the server assigns short sequential names (worker1, worker2, ...) that the user addresses windows by. " +
			"Default is job delivery when initial_directive is present, so worker replies wake this window for follow-up. Set job=false only for an explicit passive one-way notice; map feature dispatches require job=true. " +
			"For user-facing build jobs, include or tell the worker to check the project's Design Brief before writing artifacts. " +
			"Use job=false for an explicit passive one-way notice where a single report is enough and the reply should not wake this window.",
		parameters: Type.Object({
			initial_directive: Type.Optional(
				Type.String({
					description:
						"Task for the new window; for build delegation quote raw user request/Q&A, goals, handback, and project path",
				}),
			),
			model: Type.Optional(
				Type.String({
					description: "Chat model: provider/model, or a bare model name to auto-route to a keyed provider",
				}),
			),
			label: Type.Optional(
				Type.String({ description: "Explicit window name; omit to get an auto-assigned worker1, worker2, ..." }),
			),
			timeout_seconds: Type.Optional(Type.Number({ description: "Seconds to wait for the new window address" })),
			gan: Type.Optional(Type.Boolean({ description: "Send initial_directive as GAN round 1 after spawn" })),
			job: Type.Optional(Type.Boolean({ description: "Send initial_directive as job cycle 1 after spawn" })),
			issues_open: Type.Optional(Type.Number({ description: "Open issue count for GAN round 1 when gan is true" })),
			feature_ids: Type.Optional(
				Type.Array(Type.String(), {
					description: "Open map feature ids this dispatch covers (requires job:true); criteria get injected",
				}),
			),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const rawInitialDirective =
				typeof params.initial_directive === "string" ? params.initial_directive.trim() : "";
			const isSecondEyesReviewSpawn =
				secondEyesRequestedThisTurn && !secondEyesReviewSpawnedThisTurn && !activeDirectiveTurn;
			const initialDirective = isSecondEyesReviewSpawn
				? buildSecondEyesReviewDirective(rawInitialDirective)
				: rawInitialDirective;
			if (isSecondEyesReviewSpawn && !rawInitialDirective) {
				const details: SidecarSpawnWindowResponse = {
					ok: false,
					error: secondEyesPlanReadyError(),
				};
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const featureIds = Array.isArray(params.feature_ids)
				? params.feature_ids.map((id) => String(id ?? "").trim()).filter(Boolean)
				: [];
			const useJob =
				params.gan === true
					? false
					: params.job === true ||
						isSecondEyesReviewSpawn ||
						(params.job !== false && (featureIds.length > 0 || initialDirective.length > 0));
			const finalDetails = await performWorkerSpawn({
				initialDirective,
				model: params.model,
				label: params.label,
				timeoutSeconds: params.timeout_seconds,
				gan: params.gan === true,
				job: useJob,
				issuesOpen: params.issues_open,
				featureIds,
				isSecondEyesReviewSpawn,
				ctx,
			});
			// The helper preserves the server's own error (e.g. the 504 booting
			// hint) so slow boots are not misdiagnosed as dead sidecars.
			const text = JSON.stringify(finalDetails, null, 2);
			return { content: [{ type: "text", text }], details: finalDetails };
		},
		renderCall(_args, theme, _context) {
			return new Text(theme.fg("toolTitle", theme.bold("spawn window")), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Spawning..."), 0, 0);
			const data = result.details as SidecarSpawnWindowResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Spawn failed"), 0, 0);
			const pair8 = data.pair8 ?? data.window?.pair8 ?? "?";
			const label = sanitizeWindowLabel(
				data.window?.label ?? (data.window as { old_label?: string } | undefined)?.old_label,
			);
			if (!expanded) return new Text(theme.fg("success", `window ${displayWindowName(pair8, label)}`), 0, 0);
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "set_chat_model",
		label: "Set Chat Model",
		description:
			"List or apply the active chat model. Call without arguments to fetch the provider catalog; " +
			"call with model as provider/model to save and live-swap this window's chat model.",
		parameters: Type.Object({
			model: Type.Optional(Type.String({ description: "Chat model as provider/model" })),
			force: Type.Optional(Type.Boolean({ description: "Skip catalog/key validation and save as-is" })),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const modelSpec = typeof params.model === "string" ? params.model.trim() : "";
			if (!modelSpec) {
				const catalog = await fetchLLMSettingCatalog(false);
				const details = isOkSidecarResponse(catalog) ? catalog : undefined;
				const text = JSON.stringify(
					details ?? catalog ?? { ok: false, error: "JARVIS sidecar unavailable" },
					null,
					2,
				);
				return { content: [{ type: "text", text }], details };
			}
			const split = splitModelSpec(modelSpec);
			if (!split) {
				const details = { ok: false, error: "model must be provider/model" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const applied = await applyLLMSetting(
				pi,
				ctx,
				{ chat: modelSpec, ...(params.force === true ? { force: true } : {}) },
				split,
			);
			const details = applied
				? { ...applied.result, live_swapped: applied.liveSwapped === true }
				: { ok: false, error: "JARVIS model-setting apply failed" };
			const text = JSON.stringify(details, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const model = String(args.model ?? "").trim();
			return new Text(
				theme.fg("toolTitle", theme.bold("chat model")) + (model ? ` ${theme.fg("accent", model)}` : ""),
				0,
				0,
			);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Applying..."), 0, 0);
			const data = result.details as (SidecarLLMSettingApplyResponse & { live_swapped?: boolean }) | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Model change failed"), 0, 0);
			if (!expanded) {
				const label = data.chat ? `chat ${data.chat}` : "catalog";
				if (data.reload_warning) return new Text(theme.fg("warning", `${label} (saved, reload failed)`), 0, 0);
				return new Text(theme.fg("success", label), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "set_subagent_model",
		label: "Set Subagent Model",
		description:
			"List or apply the active subagent model. Call without arguments to fetch the provider catalog; " +
			"call with model as provider/model to save the subagent model used by delegate_subagent.",
		parameters: Type.Object({
			model: Type.Optional(Type.String({ description: "Subagent model as provider/model" })),
			force: Type.Optional(Type.Boolean({ description: "Skip catalog/key validation and save as-is" })),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const modelSpec = typeof params.model === "string" ? params.model.trim() : "";
			if (!modelSpec) {
				const catalog = await fetchLLMSettingCatalog(false);
				const details = isOkSidecarResponse(catalog) ? catalog : undefined;
				const text = JSON.stringify(
					details ?? catalog ?? { ok: false, error: "JARVIS sidecar unavailable" },
					null,
					2,
				);
				return { content: [{ type: "text", text }], details };
			}
			if (!splitModelSpec(modelSpec)) {
				const details = { ok: false, error: "model must be provider/model" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const applied = await applyLLMSetting(pi, ctx, {
				subagent: modelSpec,
				...(params.force === true ? { force: true } : {}),
			});
			const details = applied ? applied.result : { ok: false, error: "JARVIS model-setting apply failed" };
			if (applied?.result.ok) {
				saveSubagentModelUserSet(true);
				try {
					ctx.ui.notify(`JARVIS subagent -> ${applied.result.subagent}.`, "info");
				} catch {
					/* stale */
				}
			}
			const text = JSON.stringify(details, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const model = String(args.model ?? "").trim();
			return new Text(
				theme.fg("toolTitle", theme.bold("subagent model")) + (model ? ` ${theme.fg("accent", model)}` : ""),
				0,
				0,
			);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Applying..."), 0, 0);
			const data = result.details as SidecarLLMSettingApplyResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Model change failed"), 0, 0);
			if (!expanded) {
				const label = data.subagent ? `subagent ${data.subagent}` : "catalog";
				if (data.reload_warning) return new Text(theme.fg("warning", `${label} (saved, reload failed)`), 0, 0);
				return new Text(theme.fg("success", label), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "set_encoder_model",
		label: "Set Encoder Model",
		description:
			"List or apply the active encoder model. Call without arguments to fetch the provider catalog; " +
			"call with model as provider/model to save and reload the sidecar encoder model.",
		parameters: Type.Object({
			model: Type.Optional(Type.String({ description: "Encoder model as provider/model" })),
			force: Type.Optional(Type.Boolean({ description: "Skip catalog/key validation and save as-is" })),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const modelSpec = typeof params.model === "string" ? params.model.trim() : "";
			if (!modelSpec) {
				const catalog = await fetchLLMSettingCatalog(false);
				const details = isOkSidecarResponse(catalog) ? catalog : undefined;
				const text = JSON.stringify(
					details ?? catalog ?? { ok: false, error: "JARVIS sidecar unavailable" },
					null,
					2,
				);
				return { content: [{ type: "text", text }], details };
			}
			if (!splitModelSpec(modelSpec)) {
				const details = { ok: false, error: "model must be provider/model" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const applied = await applyLLMSetting(pi, ctx, {
				encoder: modelSpec,
				...(params.force === true ? { force: true } : {}),
			});
			const details = applied ? applied.result : { ok: false, error: "JARVIS model-setting apply failed" };
			if (applied?.result.ok && !applied.result.reload_warning) {
				try {
					ctx.ui.notify(`JARVIS encoder -> ${applied.result.encoder}.`, "info");
				} catch {
					/* stale */
				}
			}
			const text = JSON.stringify(details, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const model = String(args.model ?? "").trim();
			return new Text(
				theme.fg("toolTitle", theme.bold("encoder model")) + (model ? ` ${theme.fg("accent", model)}` : ""),
				0,
				0,
			);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Applying..."), 0, 0);
			const data = result.details as SidecarLLMSettingApplyResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Model change failed"), 0, 0);
			if (!expanded) {
				const label = data.encoder ? `encoder ${data.encoder}` : "catalog";
				if (data.reload_warning) return new Text(theme.fg("warning", `${label} (saved, reload failed)`), 0, 0);
				return new Text(theme.fg("success", label), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "list_windows",
		label: "List JARVIS Windows",
		description: "List active JARVIS window addresses before job_send to an existing worker.",
		parameters: Type.Object({}),
		async execute() {
			const data = await postSidecar<SidecarDirectiveWindowsResponse>("/directives/windows", undefined, "GET");
			const details = isOkSidecarResponse(data) ? data : undefined;
			const text = JSON.stringify(details ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(_args, theme, _context) {
			return new Text(theme.fg("toolTitle", theme.bold("windows")), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Checking..."), 0, 0);
			const data = result.details as SidecarDirectiveWindowsResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", "Sidecar unavailable"), 0, 0);
			const windows = data.windows ?? [];
			if (!expanded) {
				const live = windows.filter((item) => item.alive).length;
				return new Text(theme.fg("success", `${live}/${windows.length} live`), 0, 0);
			}
			const lines = windows.map((item) => {
				const marker = item.current ? " current" : "";
				const live = item.alive ? "alive" : "stale";
				const name = displayWindowName(item.pair8, item.label);
				const role = item.role ? ` role=${item.role}` : "";
				const status = item.status ? ` status=${item.status}` : "";
				const contract = item.contract ? ` contract=${item.contract}` : "";
				const stage = item.stage ? ` stage=${item.stage}` : "";
				const job = item.active_job_id ? ` job=${item.active_job_id}` : "";
				return `${name} ${live}${marker}${role}${status}${contract}${stage}${job}${item.pid ? ` pid=${item.pid}` : ""}`;
			});
			return new Text(lines.join("\n") || "no windows", 0, 0);
		},
	});

	pi.registerTool({
		name: "send_directive",
		label: "Send Directive",
		description:
			"Low-level passive one-way message to another JARVIS window. Prefer job_send for worker tasks, reviews, handbacks, or anything that should wake this window for a follow-up cycle. " +
			"Use list_windows first if the target address or label is unknown. " +
			"Use this only for explicit passive notices or bootstrap/legacy wiring; after sending, finish and end your turn.",
		parameters: Type.Object({
			to_window: Type.String({ description: "Target JARVIS window pair8 address or unique label" }),
			message: Type.String({ description: "Directive or message body" }),
		}),
		async execute(_toolCallId, params) {
			const message = activeSecondEyesReviewTurn ? buildSecondEyesMainHandback(params.message) : params.message;
			const data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
				to_window: params.to_window,
				message,
				kind: "directive",
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (
				details?.ok &&
				activeDirectiveTurn &&
				details.item?.to_window &&
				details.item.to_window === String(activeDirectiveTurn.from_window ?? "").trim()
			) {
				// The model answered the sender with an auto-executing directive;
				// the passive turn-end auto-report would only duplicate it.
				directiveTurnBusReplySent = true;
			}
			let text = JSON.stringify(details ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			if (details?.ok) {
				text +=
					"\n\nDelivered. The reply will arrive automatically as a new turn once this window is idle — finish and end your turn now; do not poll or wait in-turn (it blocks the delivery).";
			}
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const target = String(args.to_window ?? "").trim() || "?";
			return new Text(theme.fg("toolTitle", theme.bold("directive ")) + theme.fg("accent", target), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Sending..."), 0, 0);
			const data = result.details as SidecarDirectiveSendResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Send failed"), 0, 0);
			const item = data.item;
			if (!expanded) return new Text(theme.fg("success", `sent ${item?.id ? item.id.slice(0, 8) : ""}`), 0, 0);
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "gan_send",
		label: "GAN Send",
		description:
			"Send a structured GAN consensus round to another JARVIS window. Use this only when the user explicitly asks to run GAN/consensus. " +
			"If gan_id is omitted, this starts round 1 and the server issues gan_id. If gan_id is provided, the server stamps the next round. " +
			"When replying inside an existing GAN (the turn header shows a gan_id), you MUST pass that gan_id; the server rejects a second open GAN between the same two windows. " +
			"Protocol: round 1 hands the work to the destroyer, round 2 is the destroyer's verdict (sets the issue baseline), round 3 is the rebuttal or acceptance; " +
			"maximum 3 send rounds; enumerate open issues and pass issues_open. From round 3 onward issues_open must strictly decrease. " +
			"Tie-breakers: P0 correctness/security/data-loss issues are won by the destroyer; " +
			"style/preferences are the worker's call; unresolved issues must be closed with gan_close status escalated. This tool does not spawn windows; use spawn_window separately. " +
			"The counterpart's round arrives automatically as a new turn once this window is idle — after sending, end your turn; do not poll while waiting.",
		parameters: Type.Object({
			to_window: Type.String({ description: "Target JARVIS window pair8 address or unique label" }),
			message: Type.String({ description: "Verdict, acceptance, or rebuttal body with enumerated open issues" }),
			issues_open: Type.Number({ description: "Number of open issues in this round" }),
			gan_id: Type.Optional(Type.String({ description: "Existing gan_id; omit to start a new GAN" })),
		}),
		async execute(_toolCallId, params) {
			const issuesOpen =
				typeof params.issues_open === "number" && Number.isFinite(params.issues_open)
					? Math.max(0, Math.floor(params.issues_open))
					: undefined;
			if (issuesOpen === undefined) {
				const details: SidecarDirectiveSendResponse = { ok: false, error: "issues_open must be a finite number" };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const ganId = typeof params.gan_id === "string" ? params.gan_id.trim() : "";
			let data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
				to_window: params.to_window,
				message: params.message,
				kind: "directive",
				gan_target: ganId || "new",
				issues_open: issuesOpen,
			});
			if (!ganId && data && data.ok === false) {
				// Same self-heal as job_send: the duplicate-open 409 names the open
				// session in a machine-generated message — continue it once.
				const detail = `${String(data.error ?? "")} ${String((data as { body?: string }).body ?? "")}`;
				const existing = detail.match(/open gan (g_[0-9a-f]{8}) already exists/)?.[1];
				if (existing) {
					data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
						to_window: params.to_window,
						message: params.message,
						kind: "directive",
						gan_target: existing,
						issues_open: issuesOpen,
					});
				}
			}
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (
				details?.ok &&
				activeDirectiveTurn &&
				details.item?.to_window &&
				details.item.to_window === String(activeDirectiveTurn.from_window ?? "").trim()
			) {
				directiveTurnBusReplySent = true;
			}
			let text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			if (details?.ok) {
				text +=
					"\n\nDelivered. The counterpart's GAN round will arrive automatically as a new turn once this window is idle — finish and end your turn now; do not poll while waiting.";
			}
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const target = String(args.to_window ?? "").trim() || "?";
			return new Text(theme.fg("toolTitle", theme.bold("gan send ")) + theme.fg("accent", target), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Sending GAN round..."), 0, 0);
			const data = result.details as SidecarDirectiveSendResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "GAN send failed"), 0, 0);
			const item = data.item;
			const gan = item?.gan;
			if (!expanded && gan?.gan_id) {
				const target = displayWindowFromList(data.windows, item?.to_window);
				return new Text(theme.fg("success", `[GAN r${gan.round ?? "?"}/3] -> ${target}`), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "gan_close",
		label: "GAN Close",
		description:
			"Close a GAN consensus session with a terminal report. status must be agreed or escalated. " +
			"Use agreed only when the remaining issues are resolved. Use escalated when issues remain after the tie-break rules or the round cap; " +
			"the summary must include remaining issues and both sides' arguments. After close, the server rejects any further append for that gan_id.",
		parameters: Type.Object({
			gan_id: Type.String({ description: "Existing server-issued gan_id" }),
			status: Type.Union([Type.Literal("agreed"), Type.Literal("escalated")]),
			summary: Type.String({ description: "Terminal consensus or escalation summary" }),
		}),
		async execute(_toolCallId, params) {
			const ganId = String(params.gan_id ?? "").trim();
			const status = String(params.status ?? "").trim();
			const summary = String(params.summary ?? "").trim();
			if (!ganId || !summary || (status !== "agreed" && status !== "escalated")) {
				const details: SidecarDirectiveSendResponse = {
					ok: false,
					error: "gan_id, status agreed|escalated, and summary are required",
				};
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
				kind: "report",
				message: summary,
				gan_target: ganId,
				gan_status: status,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (details?.ok) {
				appendSubturnLedger(`GAN ${ganId} ${status} => ${oneLineForSummary(summary, 120)}`);
				if (
					activeDirectiveTurn &&
					details.item?.to_window &&
					details.item.to_window === String(activeDirectiveTurn.from_window ?? "").trim()
				) {
					directiveTurnBusReplySent = true;
				}
			}
			const text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const ganId = String(args.gan_id ?? "").trim() || "?";
			return new Text(theme.fg("toolTitle", theme.bold("gan close ")) + theme.fg("accent", ganId), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Closing GAN..."), 0, 0);
			const data = result.details as SidecarDirectiveSendResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "GAN close failed"), 0, 0);
			const gan = data.item?.gan;
			if (!expanded && gan?.gan_id) {
				return new Text(theme.fg("success", `[GAN ${gan.status ?? "closed"} r${gan.round ?? "?"}/3]`), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "job_send",
		label: "Job Send",
		description:
			"Send a structured job/build-loop handoff to another JARVIS window. Jobs are the default for delegated build work (see spawn_window); also use this to continue cycles of an already-open job. " +
			"If job_id is omitted, this starts cycle 1 and the server issues job_id. If job_id is provided, the server stamps role, phase, and cycle from direction. " +
			"Protocol: orchestrator to worker is dispatch; worker to orchestrator is review/progress handback; maximum dispatch cycles defaults to 3. " +
			"In Critic Mode main turns, this tool is only for sending plan-review or artifact-review requests to the existing review-only worker; never ask that worker to implement, fix, edit, patch, write files, or run mutation commands. " +
			"For user-facing build dispatches, quote the user's original request and raw plan Q/A, mention project path and whether Design Brief exists or worker must run recon; do not prescribe stack/layout unless the user did. " +
			"When a provider-call ceiling is reached inside a job turn, summarize completed and remaining work and hand it back with this tool. " +
			"When a map run is open, dispatch map features with feature_ids so the ticket carries the acceptance criteria. " +
			"The counterpart's job turn arrives automatically once that window is idle; after sending, end your turn and do not poll.",
		parameters: Type.Object({
			to_window: Type.String({ description: "Target JARVIS window pair8 address or unique label" }),
			message: Type.String({
				description: "Dispatch, review request, progress handback, or next-cycle instruction",
			}),
			job_id: Type.Optional(
				Type.String({ description: "Existing job_id from the turn header; omit to start a new job" }),
			),
			feature_ids: Type.Optional(
				Type.Array(Type.String(), {
					description: "Open map feature ids this dispatch covers; their acceptance criteria get injected",
				}),
			),
		}),
		async execute(_toolCallId, params) {
			const jobId = typeof params.job_id === "string" ? params.job_id.trim() : "";
			const featureIds = Array.isArray(params.feature_ids)
				? params.feature_ids.map((id) => String(id ?? "").trim()).filter(Boolean)
				: [];
			if (
				activeMapCheckpointTurn &&
				featureIds.length === 0 &&
				!mapCheckpointRestrictionsLifted(activeMapCheckpointTurn)
			) {
				const details: SidecarDirectiveSendResponse = { ok: false, error: MAP_DISPATCH_TICKET_REQUIRED_ERROR };
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const initialCriticDispatch = secondEyesReviewSpawnRequired();
			let message = params.message;
			if (activeSecondEyesReviewTurn) {
				message = buildSecondEyesMainHandback(message);
			} else if (activeSecondEyesMainTurn || initialCriticDispatch) {
				if (initialCriticDispatch && !secondEyesPlanReady(message)) {
					const details: SidecarDirectiveSendResponse = { ok: false, error: secondEyesPlanReadyError() };
					return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
				}
				message = buildSecondEyesReviewRequestDirective(message);
			}
			if (featureIds.length > 0) {
				const invalid = mapTicketValidationError(featureIds);
				if (invalid) {
					const details: SidecarDirectiveSendResponse = { ok: false, error: invalid };
					return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
				}
				message = `${buildMapTicketBlock(featureIds)}\n\n${message}`;
			}
			let data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
				to_window: params.to_window,
				message,
				kind: "directive",
				job_target: jobId || "new",
			});
			if (!jobId && data && data.ok === false) {
				// Live failure mode: the model forgets to echo job_id, the duplicate-
				// open guard 409s naming the open session, and a weaker model retries
				// the same call until it burns its ceiling. The 409 detail is a
				// machine-generated protocol message, so self-heal once with the id.
				const detail = `${String(data.error ?? "")} ${String((data as { body?: string }).body ?? "")}`;
				const existing = detail.match(/open job (j_[0-9a-f]{8}) already exists/)?.[1];
				if (existing) {
					data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
						to_window: params.to_window,
						message,
						kind: "directive",
						job_target: existing,
					});
				}
			}
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (
				details?.ok &&
				activeDirectiveTurn &&
				details.item?.to_window &&
				details.item.to_window === String(activeDirectiveTurn.from_window ?? "").trim()
			) {
				directiveTurnBusReplySent = true;
			}
			if (details?.ok && featureIds.length > 0) {
				recordMapDispatch(
					String(details.item?.job?.job_id ?? ""),
					featureIds,
					String(details.item?.to_window ?? params.to_window ?? ""),
				);
			}
			if (details?.ok && initialCriticDispatch) {
				secondEyesReviewSpawnedThisTurn = true;
			}
			let text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			if (details?.ok) {
				text +=
					"\n\nDelivered. The counterpart's job turn will arrive automatically once that window is idle — finish and end your turn now; do not poll while waiting.";
			}
			if (details?.ok && featureIds.length === 0 && activeMapRun?.phase === "stepping") {
				text +=
					"\n\nNote: a map run is open — dispatches for map features should pass feature_ids so the ticket carries the acceptance criteria.";
			}
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const target = String(args.to_window ?? "").trim() || "?";
			return new Text(theme.fg("toolTitle", theme.bold("job send ")) + theme.fg("accent", target), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Sending job handoff..."), 0, 0);
			const data = result.details as SidecarDirectiveSendResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Job send failed"), 0, 0);
			const item = data.item;
			const job = item?.job;
			if (!expanded && job?.job_id) {
				const target = displayWindowFromList(data.windows, item?.to_window);
				return new Text(
					theme.fg(
						"success",
						`[JOB c${job.cycle ?? "?"}/${jobCycleCapFromEnv()} ${job.phase ?? "job"}] -> ${target}`,
					),
					0,
					0,
				);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "job_close",
		label: "Job Close",
		description:
			"Close a job/build-loop session with a terminal report. status must be done or escalated. " +
			"Use done when the delegated work is complete. Use escalated when the cycle cap or unresolved blocker remains; summary must include remaining work and reason. " +
			"Only the orchestrator (the window that dispatched cycle 1) terminally closes a job, after reviewing the worker's handback. " +
			"If a worker calls this, the server converts it into a review handback so the orchestrator still gets the final say — a worker should normally hand back with job_send, not close. " +
			"After a terminal close, the server rejects any further append for that job_id.",
		parameters: Type.Object({
			job_id: Type.String({ description: "Existing server-issued job_id" }),
			status: Type.Union([Type.Literal("done"), Type.Literal("escalated")]),
			summary: Type.String({ description: "Terminal completion or escalation summary" }),
		}),
		async execute(_toolCallId, params) {
			const jobId = String(params.job_id ?? "").trim();
			const status = String(params.status ?? "").trim();
			const summary = String(params.summary ?? "").trim();
			if (!jobId || !summary || (status !== "done" && status !== "escalated")) {
				const details: SidecarDirectiveSendResponse = {
					ok: false,
					error: "job_id, status done|escalated, and summary are required",
				};
				return { content: [{ type: "text", text: JSON.stringify(details, null, 2) }], details };
			}
			const data = await postSidecar<SidecarDirectiveSendResponse>("/directives", {
				kind: "report",
				message: summary,
				job_target: jobId,
				job_status: status,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (details?.ok) {
				appendSubturnLedger(`JOB ${jobId} ${status} => ${oneLineForSummary(summary, 120)}`);
				if (
					activeDirectiveTurn &&
					details.item?.to_window &&
					details.item.to_window === String(activeDirectiveTurn.from_window ?? "").trim()
				) {
					directiveTurnBusReplySent = true;
				}
			}
			const text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const jobId = String(args.job_id ?? "").trim() || "?";
			return new Text(theme.fg("toolTitle", theme.bold("job close ")) + theme.fg("accent", jobId), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Closing job..."), 0, 0);
			const data = result.details as SidecarDirectiveSendResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Job close failed"), 0, 0);
			const job = data.item?.job;
			if (!expanded && job?.job_id) {
				return new Text(
					theme.fg("success", `[JOB ${job.status ?? "closed"} c${job.cycle ?? "?"}/${jobCycleCapFromEnv()}]`),
					0,
					0,
				);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "map_create",
		label: "Map Create",
		description:
			"Persist the feature map of a delegated build that is too large for one worker context, before the first dispatch. " +
			"TRIGGER — use ONLY when a delegated build will not fit a single worker context; whole delegation to one worker is the default. When the map is warranted, record every feature with its acceptance criteria here, then dispatch features via job_send/spawn_window with feature_ids. " +
			"The map lives in <project>/.jarvis-map/ (map.md + ledger.jsonl) and survives restarts; progress is tracked automatically at checkpoints. " +
			"Use append:true to add features to the open map; replace:true abandons the open map and starts fresh.",
		parameters: Type.Object({
			project_path: Type.String({ description: "Absolute project root; artifacts live in <project>/.jarvis-map/" }),
			title: Type.Optional(Type.String({ description: "Map title; defaults to the project folder name" })),
			append: Type.Optional(Type.Boolean({ description: "Add features to the open map run" })),
			replace: Type.Optional(Type.Boolean({ description: "Abandon the open map run and start a new one" })),
			features: Type.Array(
				Type.Object({
					title: Type.String({ description: "Feature name in the user's vocabulary" }),
					summary: Type.Optional(Type.String({ description: "One-line scope note" })),
					zone: Type.Optional(
						Type.Union([Type.Literal("feature"), Type.Literal("skeleton")], {
							description: "skeleton = global scaffolding/refactor zone; the worker owns the method there",
						}),
					),
					acceptance: Type.Array(Type.String(), {
						description: "Checkable acceptance criteria; checkpoints verify these verbatim",
					}),
				}),
				{ minItems: 1 },
			),
		}),
		async execute(_toolCallId, params) {
			// Single source of truth lives in runMapCreate (module scope) so regime A
			// (this execute) and regime B (control-bridge map_create branch) cannot
			// drift — both mutate the same in-process activeMapRun ledger.
			const details = await runMapCreate(params);
			const text = details.ok
				? `${JSON.stringify(details, null, 2)}\n\n` +
					"Map saved. Dispatch features one at a time (batch small ones) via job_send or spawn_window with feature_ids — the ticket gets the acceptance criteria injected automatically."
				: JSON.stringify(details, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const count = Array.isArray(args.features) ? args.features.length : 0;
			return new Text(
				theme.fg("toolTitle", theme.bold("map create ")) +
					theme.fg("accent", `${count} feature${count !== 1 ? "s" : ""}`),
				0,
				0,
			);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Saving map..."), 0, 0);
			const data = result.details as
				| { ok?: boolean; error?: string; map_id?: string; features?: unknown[] }
				| undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "map_create failed"), 0, 0);
			if (!expanded) {
				return new Text(theme.fg("success", `map ${data.map_id} — ${data.features?.length ?? 0} features`), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "feature_verdict",
		label: "Feature Verdict",
		description:
			"Record the checkpoint verdict for one map feature. pass requires evidence (the runnable check you executed); " +
			"reject requires a concrete reason — it rides the next dispatch ticket as the intent channel to the worker. " +
			"The ledger, rejection cap, and escalate ladder advance automatically; the result tells you the exact next step.",
		parameters: Type.Object({
			feature_id: Type.String({ description: "Map feature id from MAP STATUS (e.g. f2)" }),
			verdict: Type.Union([Type.Literal("pass"), Type.Literal("reject")]),
			reason: Type.Optional(
				Type.String({ description: "Required on reject: what is wrong and what right looks like" }),
			),
			evidence: Type.Optional(Type.String({ description: "Required on pass: the runnable check you executed" })),
		}),
		async execute(_toolCallId, params) {
			// Single source of truth lives in runFeatureVerdict (module scope) so
			// regime A (this execute) and regime B (control-bridge feature_verdict
			// branch) cannot drift — both mutate the same in-process activeMapRun.
			const result = await runFeatureVerdict(params);
			// `next` is the human step string; strip it from the `details` surfaced to
			// renderResult so the JSON shape stays byte-identical to the prior version.
			const { next, ...details } = result;
			if (!details.ok) {
				return { content: [{ type: "text" as const, text: JSON.stringify(details, null, 2) }], details };
			}
			return {
				content: [{ type: "text", text: `${JSON.stringify(details, null, 2)}\n\n${next ?? ""}` }],
				details,
			};
		},
		renderCall(args, theme, _context) {
			const id = String(args.feature_id ?? "?");
			const verdict = String(args.verdict ?? "?");
			return new Text(theme.fg("toolTitle", theme.bold("verdict ")) + theme.fg("accent", `${id} ${verdict}`), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Recording verdict..."), 0, 0);
			const data = result.details as
				| {
						ok?: boolean;
						error?: string;
						feature_id?: string;
						verdict?: string;
						rejections?: number;
						stage?: string;
				  }
				| undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "feature_verdict failed"), 0, 0);
			if (!expanded) {
				const note = data.verdict === "pass" ? "pass" : `reject ${data.rejections} (${data.stage})`;
				return new Text(theme.fg("success", `${data.feature_id} ${note}`), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
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
		name: "delegate_subagent",
		label: "Delegate Subagent",
		description:
			"Delegate to a lightweight in-process subagent (presets: destroyer/deep_research/codebase_explore/multi_file_refactor; ad-hoc needs system_prompt). Returns only the subagent's final message.",
		promptSnippet:
			"delegate_subagent: run an in-process subagent such as destroyer for read-only code critique; returns final summary plus sub_id for resume.",
		parameters: Type.Object({
			name: Type.String({
				description: "Preset name such as destroyer, deep_research, codebase_explore, or multi_file_refactor.",
			}),
			task: Type.String({
				description: "Specific task for the subagent, including target files and desired output.",
			}),
			read_only: Type.Optional(Type.Boolean({ description: "Expose only read-only tools inside the subagent." })),
			system_prompt: Type.Optional(Type.String({ description: "Required for ad-hoc names that are not presets." })),
			sub_id: Type.Optional(Type.String({ description: "Existing subagent id to resume by caller request." })),
		}),
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			await ensureSubagentModelSelectedForDelegate(pi, ctx);
			const body = {
				name: params.name,
				task: params.task,
				read_only: params.read_only,
				system_prompt: params.system_prompt,
				sub_id: params.sub_id,
				project_root: activeCodePath ?? activeProjectPath ?? ctx?.cwd,
				bench_conv_id: benchConvId(pi),
			};
			let data: SidecarSubagentDelegateResponse | undefined;
			if (onUpdate) {
				const progress: SidecarSubagentProgressDetails = {
					streaming: true,
					subagent: typeof params.name === "string" ? params.name : "subagent",
					sub_id: typeof params.sub_id === "string" ? params.sub_id : undefined,
					activity: [],
				};
				const emitProgress = () => {
					(onUpdate as (update: { content: Array<{ type: "text"; text: string }>; details: unknown }) => void)({
						content: [{ type: "text", text: renderSubagentProgressText(progress) }],
						details: { ...progress, activity: [...progress.activity] },
					});
				};
				emitProgress();
				const streamed = await postSubagentDelegateStream(body, signal, (event) => {
					if (event.event === "reasoning" && event.text) {
						progress.reasoning_tail = appendSubagentTail(progress.reasoning_tail, event.text);
					} else if (event.event === "content" && event.text) {
						progress.content_tail = appendSubagentTail(progress.content_tail, event.text);
					} else if ((event.event === "activity" || event.event === "step") && event.line) {
						pushSubagentActivity(progress, event.line);
						const subId = event.line.match(/id=([A-Za-z0-9_-]+)/)?.[1];
						if (subId) progress.sub_id = subId;
					} else if (event.event === "error") {
						progress.error = event.error ?? "Subagent stream failed";
					}
					emitProgress();
				});
				if (streamed?.aborted) {
					data = { error: streamed.error ?? "Subagent cancelled" };
				} else if (streamed?.result) {
					data = streamed.result;
				} else if (streamed?.fallback) {
					data = streamed.fallback;
				} else if (streamed?.error) {
					progress.error = streamed.error;
					pushSubagentActivity(progress, "streaming interrupted — finishing without live view...");
					emitProgress();
				}
			}
			if (!data && !signal?.aborted) {
				data = await postSidecar<SidecarSubagentDelegateResponse>(
					"/subagent/delegate",
					body,
					"POST",
					SUBAGENT_DELEGATE_FETCH_TIMEOUT_MS,
				);
			}
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (!details) {
				const text = JSON.stringify(data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
				return { content: [{ type: "text", text }], details };
			}
			const summary = String(details.summary ?? "");
			const meta = {
				subagent: details.subagent,
				sub_id: details.sub_id,
				halt_reason: details.halt_reason,
				iters: details.iters,
				elapsed_sec: details.elapsed_sec,
				in_tokens: details.in_tokens,
				out_tokens: details.out_tokens,
				think_tokens: details.think_tokens,
			};
			const text = summary
				? `${summary}\n\n[subagent]\n${JSON.stringify(meta, null, 2)}`
				: JSON.stringify(details, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const name = String(args.name ?? "subagent");
			return new Text(theme.fg("toolTitle", theme.bold("subagent ")) + theme.fg("accent", name), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) {
				const progress = result.details as SidecarSubagentProgressDetails | undefined;
				if (progress?.streaming) {
					return new Text(
						theme.fg("warning", renderSubagentProgressText(progress, expanded ? 24 : SUBAGENT_STREAM_MAX_LINES)),
						0,
						0,
					);
				}
				return new Text(theme.fg("warning", "Running subagent..."), 0, 0);
			}
			const data = result.details as SidecarSubagentDelegateResponse | undefined;
			if (!data) return new Text(theme.fg("error", "Subagent unavailable"), 0, 0);
			if (data.error) return new Text(theme.fg("error", data.error), 0, 0);
			if (!expanded) {
				const label = data.subagent ?? "subagent";
				const subId = data.sub_id ? ` ${data.sub_id}` : "";
				return new Text(theme.fg("success", `${label}${subId}`), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
	});

	pi.registerTool({
		name: "ultracode",
		label: "Orchestrate",
		description:
			"Run a parallel multi-angle review: fan out N read-only finders across the given dimensions, adversarially verify, and synthesize a confirmed result. Use for thorough review/audit/investigation. Each dimension = one finder; keep dimensions focused (e.g. correctness, security, perf).",
		promptSnippet:
			"ultracode: run parallel read-only finders across focused dimensions, verify adversarially, and return a synthesized confirmed result.",
		parameters: Type.Object({
			task: Type.String({
				description: "Specific review, audit, or investigation task including the target files or scope.",
			}),
			dimensions: Type.Array(
				Type.String({ description: "Focused finder dimension such as correctness or security." }),
			),
			max_concurrency: Type.Optional(Type.Number({ description: "Maximum concurrent finder calls." })),
			max_wallclock_sec: Type.Optional(Type.Number({ description: "Soft wallclock budget in seconds." })),
			max_calls: Type.Optional(Type.Number({ description: "Soft provider-call budget." })),
		}),
		async execute(_toolCallId, params, signal, onUpdate, ctx) {
			const dimensions = (Array.isArray(params.dimensions) ? params.dimensions : [])
				.map((dimension) => String(dimension).trim())
				.filter((dimension) => dimension.length > 0);
			const body: Record<string, unknown> = {
				task: String(params.task ?? ""),
				dimensions,
				project_root: activeCodePath ?? activeProjectPath ?? ctx?.cwd,
			};
			const maxConcurrency = Number(params.max_concurrency);
			if (Number.isFinite(maxConcurrency)) body.max_concurrency = maxConcurrency;
			const maxWallclockSec = Number(params.max_wallclock_sec);
			if (Number.isFinite(maxWallclockSec)) body.max_wallclock_sec = maxWallclockSec;
			const maxCalls = Number(params.max_calls);
			if (Number.isFinite(maxCalls)) body.max_calls = maxCalls;

			let data: SidecarOrchestrateResponse | undefined;
			if (onUpdate) {
				const progress: SidecarOrchestrateProgressDetails = {
					streaming: true,
					activity: [],
				};
				const emitProgress = () => {
					(onUpdate as (update: { content: Array<{ type: "text"; text: string }>; details: unknown }) => void)({
						content: [{ type: "text", text: renderOrchestrateProgressText(progress) }],
						details: {
							...progress,
							activity: [...progress.activity],
						},
					});
				};
				emitProgress();
				const streamed = await postOrchestrateStream(body, signal, (event) => {
					if ((event.event === "activity" || event.event === "step") && (event.line || event.text)) {
						pushOrchestrateActivity(progress, String(event.line ?? event.text ?? ""));
					} else if (event.event === "result" && event.result) {
						progress.result = event.result;
					} else if (event.event === "error") {
						progress.error = event.error ?? "Orchestration stream failed";
					}
					emitProgress();
				});
				if (streamed?.aborted) {
					data = { error: streamed.error ?? "Orchestration cancelled" };
				} else if (streamed?.result) {
					data = streamed.result;
				} else if (streamed?.fallback) {
					data = streamed.fallback;
				} else if (streamed?.error) {
					progress.error = streamed.error;
					pushOrchestrateActivity(progress, "streaming interrupted - finishing without live view...");
					emitProgress();
				}
			}
			if (!data && !signal?.aborted) {
				data = await postSidecar<SidecarOrchestrateResponse>(
					"/orchestrate",
					body,
					"POST",
					ORCHESTRATE_FETCH_TIMEOUT_MS,
				);
			}
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (!details) {
				const text = JSON.stringify(data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
				return { content: [{ type: "text", text }], details };
			}
			const summary = String(details.summary ?? "");
			const meta = {
				orchestration_id: details.orchestration_id,
				state: details.state,
				finders: `${details.finders_ran ?? 0}/${details.finders_total ?? 0}`,
				stop_reason: details.stop_reason,
				elapsed_sec: details.elapsed_sec,
				in_tokens: details.in_tokens,
				out_tokens: details.out_tokens,
				event_log_path: details.event_log_path,
			};
			const text = summary
				? `${summary}\n\n[ultracode]\n${JSON.stringify(meta, null, 2)}`
				: JSON.stringify(details, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const dimensions = Array.isArray(args.dimensions) ? args.dimensions.length : 0;
			return new Text(
				theme.fg("toolTitle", theme.bold("ultracode ")) + theme.fg("accent", `${dimensions} dimensions`),
				0,
				0,
			);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) {
				const progress = result.details as SidecarOrchestrateProgressDetails | undefined;
				if (progress?.streaming) {
					return new Text(
						theme.fg(
							"warning",
							renderOrchestrateProgressText(progress, expanded ? 24 : SUBAGENT_STREAM_MAX_LINES),
						),
						0,
						0,
					);
				}
				return new Text(theme.fg("warning", "Running ultracode..."), 0, 0);
			}
			const data = result.details as SidecarOrchestrateResponse | undefined;
			if (!data) return new Text(theme.fg("error", "Orchestration unavailable"), 0, 0);
			if (data.error) return new Text(theme.fg("error", data.error), 0, 0);
			if (!expanded) {
				const state = data.state ?? "done";
				const counts = `${data.finders_ran ?? 0}/${data.finders_total ?? 0}`;
				return new Text(theme.fg("success", `ultracode ${state} ${counts}`), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
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
				const text =
					"ref not found — the original text is already available or has expired. If you know the file path, use read.";
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
			"Patch active JARVIS.md sections; batch 2+ updates; DESIGN_BRIEF stores design recon; refresh NOW/MAP/RAW after code; never edit whole file.",
		parameters: Type.Object({
			field: Type.Optional(Type.String({ description: "NOW|MAP|LAW|BAN|HABIT|WHY|OMM|RAW|DESIGN_BRIEF" })),
			value: Type.Optional(Type.String()),
			updates: Type.Optional(
				Type.Array(
					Type.Object({
						field: Type.String({ description: "NOW|MAP|LAW|BAN|HABIT|WHY|OMM|RAW|DESIGN_BRIEF" }),
						value: Type.String(),
					}),
				),
			),
		}),
		renderCall(_args, theme, _context) {
			return new Text(theme.fg("toolTitle", theme.bold("memory")), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("muted", "saving…"), 0, 0);
			const data = result.details as { ok?: boolean; error?: string; fields?: string[] } | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "memory update failed"), 0, 0);
			if (!expanded) {
				const fields = Array.isArray(data.fields) ? data.fields.join(", ") : "";
				return new Text(theme.fg("muted", fields ? `JARVIS.md · ${fields}` : "JARVIS.md saved"), 0, 0);
			}
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
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
			const routeAllowsRegistration =
				lastRouteClassifierDecision?.create_project === true ||
				lastRouteClassifierDecision?.register_project === true ||
				lastRouteClassifierDecision?.pending_project_decision === "confirm";
			const allowRegistration =
				isProjectRoute(currentRoute) ||
				routeAllowsRegistration ||
				(pendingProjectCreate !== undefined && lastRouteClassifierDecision?.pending_project_decision === "confirm");
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
		name: "set_window_label",
		label: "Set Window Label",
		description:
			"Rename this JARVIS window only when the user explicitly asks for a window name change. " +
			"Do not rename automatically when the subtask or topic changes.",
		parameters: Type.Object({
			label: Type.String({ description: "New short display label for this window" }),
		}),
		async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
			const data = await postSidecar<SidecarWindowLabelResponse>("/label", {
				label: params.label,
			});
			const details = isOkSidecarResponse(data) ? data : undefined;
			if (details?.ok) {
				currentWindowLabel = sanitizeWindowLabel(details.label);
				try {
					const oldName = sanitizeWindowLabel(details.old_label) ?? details.pair8 ?? "unnamed";
					const newName = currentWindowLabel ?? details.pair8 ?? "unnamed";
					ctx?.ui?.notify?.(`[${oldName}] → [${newName}]`, "info");
					ctx?.ui?.setStatus?.(
						"jarvis",
						jlcLabel(sidecarHealthy ? "ok" : "down", lastContextResponse?.project_name),
					);
				} catch {
					/* stale */
				}
			}
			const text = JSON.stringify(details ?? data ?? { ok: false, error: "JARVIS sidecar unavailable" }, null, 2);
			return { content: [{ type: "text", text }], details };
		},
		renderCall(args, theme, _context) {
			const label = sanitizeWindowLabel(args.label) ?? "?";
			return new Text(theme.fg("toolTitle", theme.bold("window label ")) + theme.fg("accent", label), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("warning", "Renaming..."), 0, 0);
			const data = result.details as SidecarWindowLabelResponse | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "Rename failed"), 0, 0);
			if (!expanded) return new Text(theme.fg("success", sanitizeWindowLabel(data.label) ?? "label cleared"), 0, 0);
			return new Text(JSON.stringify(data, null, 2), 0, 0);
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
		renderCall(_args, theme, _context) {
			return new Text(theme.fg("toolTitle", theme.bold("project")), 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme, _context) {
			if (isPartial) return new Text(theme.fg("muted", "switching…"), 0, 0);
			const data = result.details as { ok?: boolean; error?: string; name?: string } | undefined;
			if (!data?.ok) return new Text(theme.fg("error", data?.error ?? "switch failed"), 0, 0);
			if (!expanded) return new Text(theme.fg("muted", data.name ?? "switched"), 0, 0);
			return new Text(JSON.stringify(data, null, 2), 0, 0);
		},
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

	// =========================================================================
	// Deferred Tools — context diet (flag-gated, default OFF)
	// =========================================================================
	// When JARVIS_DEFERRED_TOOLS=1, narrow the active tool set to a core working
	// set and let the model discover/load the rest via tool_search + load_tool.
	// Flag OFF = byte-identical to today (no setActiveTools call, no extra tools).

	// All deferred-tools logic is gated on the flag. When OFF, zero code paths
	// execute and zero pi methods are called → byte-identical to today.
	// NOTE: only registration (registerTool) happens here at load time. The
	// active-set narrowing runs at runtime via applyDeferredToolsDiet, invoked
	// from the session_start handler — getAllTools/setActiveTools are action
	// methods that throw during extension loading.
	if (deferredToolsEnabled()) {
		pi.registerTool({
			name: "tool_search",
			label: "Tool Search",
			description:
				"List available deferred tools that are not currently loaded. " +
				"Returns tool names and short descriptions. Optionally filter by a keyword query. " +
				"Call load_tool to activate a tool so its full schema appears next turn.",
			promptSnippet: "tool_search: list deferred tools not yet loaded; optional keyword filter.",
			parameters: Type.Object({
				query: Type.Optional(
					Type.String({ description: "Optional keyword to filter tool names and descriptions." }),
				),
			}),
			async execute(_toolCallId, params) {
				const activeNames = new Set(pi.getActiveTools());
				const allTools = pi.getAllTools();
				const query = typeof params.query === "string" ? params.query.trim().toLowerCase() : "";
				const deferred: Array<{ name: string; description: string }> = [];
				for (const tool of allTools) {
					if (activeNames.has(tool.name)) continue;
					const desc = tool.description ?? "";
					const snippet = desc.split(/\.\s/)[0]?.trim() || desc;
					if (query && !tool.name.toLowerCase().includes(query) && !snippet.toLowerCase().includes(query)) {
						continue;
					}
					deferred.push({ name: tool.name, description: snippet });
				}
				const details = { ok: true, deferred_count: deferred.length, tools: deferred };
				return {
					content: [{ type: "text", text: JSON.stringify(details, null, 2) }],
					details,
				};
			},
		});

		pi.registerTool({
			name: "load_tool",
			label: "Load Tool",
			description:
				"Activate one or more deferred tools by name so their full schemas appear in the next turn. " +
				"Use tool_search first to discover available tool names.",
			promptSnippet: "load_tool: activate deferred tools by name; schemas appear next turn.",
			parameters: Type.Object({
				names: Type.Array(Type.String({ description: "Tool names to activate." }), {
					description: "Array of tool names to load into the active set.",
				}),
			}),
			async execute(_toolCallId, params) {
				const requested = Array.isArray(params.names)
					? params.names.map((n) => String(n).trim()).filter(Boolean)
					: [];
				if (requested.length === 0) {
					const details = {
						ok: false as const,
						error: "names array is empty",
						loaded: [] as string[],
						already_active: [] as string[],
						not_found: [] as string[],
						active_count: pi.getActiveTools().length,
					};
					return {
						content: [{ type: "text", text: JSON.stringify(details) }],
						details,
					};
				}
				const allToolNames = new Set(pi.getAllTools().map((t) => t.name));
				const currentActive = pi.getActiveTools();
				const currentSet = new Set(currentActive);
				const loaded: string[] = [];
				const notFound: string[] = [];
				const alreadyActive: string[] = [];
				for (const name of requested) {
					if (currentSet.has(name)) {
						alreadyActive.push(name);
					} else if (allToolNames.has(name)) {
						loaded.push(name);
					} else {
						notFound.push(name);
					}
				}
				if (loaded.length > 0) {
					pi.setActiveTools([...currentActive, ...loaded]);
				}
				const details = {
					ok: notFound.length === 0,
					error: notFound.length > 0 ? `tools not found: ${notFound.join(", ")}` : "",
					loaded,
					already_active: alreadyActive,
					not_found: notFound,
					active_count: pi.getActiveTools().length,
				};
				return {
					content: [{ type: "text", text: JSON.stringify(details, null, 2) }],
					details,
				};
			},
		});

		// Active-set narrowing runs at runtime (session_start → applyDeferredToolsDiet),
		// not here — getAllTools/setActiveTools can't be called during loading.
	}

	pi.registerCommand("jarvis-status", {
		description: "Show JARVIS sidecar and active project status.",
		handler: async (_args, ctx) => {
			const healthy = await checkHealth();
			sidecarHealthy = healthy;
			const status = healthy ? await postSidecar<SidecarStatusResponse>("/status", undefined, "GET") : undefined;
			currentWindowLabel = sanitizeWindowLabel(status?.window_label);
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
		description: "Pick JARVIS chat + subagent + encoder models (writes data/config.yaml).",
		handler: async (args, ctx) => {
			const forceRefresh = /\b(?:refresh|--refresh|-r)\b/i.test(args ?? "");
			await runModelSetting(pi, ctx, forceRefresh);
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
			if (synced.ok === false) {
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

function memoryWriteDisabledNotice(_detail?: string | null): string {
	return MEMORY_WRITE_DISABLED_NOTICE;
}

function memoryWriteEnabledNotice(detail?: string | null): string {
	return detail?.trim() || MEMORY_WRITE_ENABLED_NOTICE;
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
	while (true) {
		const catalog = await postSidecar<SidecarCredentialCatalogResponse>("/credentials/catalog", undefined, "GET");
		if (!catalog?.ok || !catalog.targets) {
			ctx.ui.notify(
				`JARVIS api-key: credential catalog failed (${catalog?.error ?? "sidecar unavailable"})`,
				"error",
			);
			return;
		}
		const entries = Object.entries(catalog.targets);
		const labels = entries.map(([id, target]) => apiKeyProviderLabel(id, target));
		const addLabel = "[+] Add custom provider…";
		const separator = "─────────────────────────────";
		const picked = await ctx.ui.select("API key setup — select a provider:", [...labels, separator, addLabel]);
		if (picked === undefined) return;
		if (picked === separator) continue;
		if (picked === addLabel) {
			await runApiKeyAddCustomProvider(ctx, entries);
			continue;
		}
		const idx = labels.indexOf(picked);
		if (idx < 0) continue;
		const [providerId, target] = entries[idx]!;
		const action = await pickApiKeyAction(ctx, target);
		if (action === undefined) return;
		if (action === "Change key") {
			await changeApiKeyForTarget(ctx, providerId, target);
			continue;
		}
		if (action === "Remove") {
			await removeCustomApiKeyProvider(ctx, providerId, target);
		}
	}
}

function apiKeyProviderLabel(id: string, target: SidecarCredentialTarget): string {
	const isCustom = target.custom === true || target.source === "custom";
	const marker = isCustom && target.configured ? "[*]" : target.configured ? "[v]" : "[ ]";
	const custom = isCustom ? "  (custom)" : "";
	const status = target.configured ? "key set" : "no key";
	return `${marker} ${target.label ?? id}${custom}   ${status}`;
}

async function pickApiKeyAction(ctx: ExtensionContext, target: SidecarCredentialTarget): Promise<string | undefined> {
	const isCustom = target.custom === true || target.source === "custom";
	const actions = isCustom ? ["Change key", "Remove"] : ["Change key"];
	return ctx.ui.select(`${target.label ?? "Provider"} API key`, actions);
}

async function runApiKeyAddCustomProvider(
	ctx: ExtensionContext,
	entries: Array<[string, SidecarCredentialTarget]>,
): Promise<void> {
	const baseUrl = (await ctx.ui.input("Custom provider base URL", "https://api.example.com/v1"))?.trim();
	if (baseUrl === undefined) return;
	if (!baseUrl) {
		ctx.ui.notify("Base URL was empty; nothing saved.", "warning");
		return;
	}
	const label = (await ctx.ui.input("Custom provider display name", "GLM / Zhipu"))?.trim();
	if (label === undefined) return;
	if (!label) {
		ctx.ui.notify("Provider label was empty; nothing saved.", "warning");
		return;
	}
	const duplicate = findApiKeyDuplicate(entries, label, baseUrl);
	if (duplicate) {
		const confirmed = await ctx.ui.confirm("Custom provider already exists", "already exists — change its key?");
		if (!confirmed) return;
		await changeApiKeyForTarget(ctx, duplicate[0], duplicate[1]);
		return;
	}
	const apiKey = await ctx.ui.input(`Enter ${label} API key`, "API key");
	if (apiKey === undefined) return;
	if (!apiKey.trim()) {
		ctx.ui.notify("API key was empty; nothing saved.", "warning");
		return;
	}
	const result = await postSidecar<SidecarCredentialCustomResponse>("/credentials/custom", {
		label,
		base_url: baseUrl,
		api_key: apiKey.trim(),
		validate: true,
	});
	if (!result?.ok) {
		ctx.ui.notify(`JARVIS custom provider save failed: ${result?.error ?? "sidecar unavailable"}`, "error");
		return;
	}
	refreshPiRuntimeAuth(ctx);
	notifyApiKeyValidation(ctx, result.label ?? label, result.validation);
	ctx.ui.notify("Pick models with /model-setting.", "info");
}

function findApiKeyDuplicate(
	entries: Array<[string, SidecarCredentialTarget]>,
	label: string,
	baseUrl: string,
): [string, SidecarCredentialTarget] | undefined {
	const providerId = providerIdFromLabel(label);
	const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
	return entries.find(([id, target]) => {
		if (id === providerId) return true;
		return Boolean(target.base_url && normalizeBaseUrl(target.base_url) === normalizedBaseUrl);
	});
}

function providerIdFromLabel(label: string): string {
	return label
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.replace(/-+/g, "-");
}

function normalizeBaseUrl(baseUrl: string): string {
	return baseUrl.trim().replace(/\/+$/, "");
}

async function changeApiKeyForTarget(
	ctx: ExtensionContext,
	providerId: string,
	target: SidecarCredentialTarget,
): Promise<void> {
	const envName = target.env_name;
	if (!envName) {
		ctx.ui.notify("Selected provider has no API-key environment variable.", "warning");
		return;
	}
	const label = target.label ?? providerId;
	const key = await ctx.ui.input(`Enter ${label} API key`, envName);
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
	if (!result) {
		ctx.ui.notify("JARVIS api-key: sidecar unavailable", "error");
		return;
	}
	refreshPiRuntimeAuth(ctx);
	notifyApiKeyValidation(ctx, label, result.validation, result.error);
}

async function removeCustomApiKeyProvider(
	ctx: ExtensionContext,
	providerId: string,
	target: SidecarCredentialTarget,
): Promise<void> {
	const confirmed = await ctx.ui.confirm(
		`Remove ${target.label ?? providerId}`,
		"Remove this custom provider and its saved API key?",
	);
	if (!confirmed) return;
	const result = await postSidecar<SidecarCredentialRemoveResponse>("/credentials/custom/remove", {
		provider_id: providerId,
		remove_key: true,
	});
	if (!result?.ok) {
		ctx.ui.notify(`JARVIS custom provider remove failed: ${result?.error ?? "sidecar unavailable"}`, "error");
		return;
	}
	refreshPiRuntimeAuth(ctx);
	ctx.ui.notify(`Removed ${target.label ?? providerId}. Pick models with /model-setting.`, "info");
}

function notifyApiKeyValidation(
	ctx: ExtensionContext,
	label: string,
	validation?: SidecarCredentialSetResponse["validation"],
	error?: string,
): void {
	if (validation?.ok) {
		if (validation.skipped || validation.warning) {
			const suffix = validation.warning ? ` — ${validation.warning}` : "";
			ctx.ui.notify(`${label}: saved${suffix}`, "info");
			return;
		}
		const suffix = typeof validation.models === "number" ? ` — ${validation.models} models` : "";
		ctx.ui.notify(`${label}: connected${suffix}`, "info");
		return;
	}
	const detail = validation?.error ?? error ?? "check URL or key";
	if (/could not reach \/models/i.test(detail)) {
		ctx.ui.notify(`${label}: saved, but could not reach /models — check URL or key`, "warning");
		return;
	}
	ctx.ui.notify(`${label}: saved, but validation failed: ${detail}`, "warning");
}

function modelCatalogLabelSuffix(provider: SidecarLLMSettingProvider): string {
	const source = provider.catalog_source;
	if (!source || source === "live" || source === "unavailable" || source === "disabled") return "";
	return "  (cached list)";
}

function splitModelSpec(value: string | undefined): { provider: string; model: string } | undefined {
	const text = String(value ?? "").trim();
	const slash = text.indexOf("/");
	if (slash <= 0 || slash >= text.length - 1) return undefined;
	const provider = text.slice(0, slash).trim();
	const model = text.slice(slash + 1).trim();
	return provider && model ? { provider, model } : undefined;
}

async function fetchLLMSettingCatalog(forceRefresh = false): Promise<SidecarLLMSettingCatalogResponse | undefined> {
	const catalogPath = forceRefresh ? "/llmsetting/catalog?refresh=1" : "/llmsetting/catalog";
	return postSidecar<SidecarLLMSettingCatalogResponse>(catalogPath, undefined, "GET");
}

type LLMSettingRole = "chat" | "subagent" | "encoder";
type LLMSettingPick = { provider: string; model: string };

function splitLLMSettingRoleSpec(value: string | null | undefined): { provider?: string; model?: string } {
	if (!value || !value.includes("/")) return {};
	const [provider, model] = value.split("/", 2);
	return { provider, model };
}

function llmSettingRankWeight(isCurrent: boolean, isRecommended: boolean): number {
	if (isCurrent) return 0;
	if (isRecommended) return 1;
	return 2;
}

function llmSettingLabelWithMarkers(name: string, isCurrent: boolean, isRecommended: boolean): string {
	const marker = isCurrent ? "*" : isRecommended ? "+" : " ";
	const tags: string[] = [];
	if (isCurrent) tags.push("current");
	if (isRecommended) tags.push("recommended");
	const suffix = tags.length ? `  (${tags.join(", ")})` : "";
	return `${marker} ${name}${suffix}`;
}

async function pickLLMSettingProvider(
	ctx: ExtensionContext,
	catalog: SidecarLLMSettingCatalogResponse,
	role: LLMSettingRole,
): Promise<string | undefined> {
	const providers = catalog.providers ?? {};
	const recommended = catalog.recommended ?? {};
	const current = catalog.current ?? {};
	const rec = recommended[role]?.provider;
	const curProvider = splitLLMSettingRoleSpec(current[role]).provider;
	const ordered = Object.keys(providers)
		.filter((pid) => {
			const allowed = providers[pid]?.roles;
			return !Array.isArray(allowed) || allowed.includes(role);
		})
		.sort((a, b) => {
			const wa = llmSettingRankWeight(a === curProvider, a === rec);
			const wb = llmSettingRankWeight(b === curProvider, b === rec);
			return wa - wb;
		});
	const labels = ordered.map((pid) => {
		const provider = providers[pid] ?? {};
		const name = `${provider.label ?? pid}${modelCatalogLabelSuffix(provider)}`;
		if (!provider.available) return `x ${name}  (${provider.reason ?? "unavailable"})`;
		return llmSettingLabelWithMarkers(name, pid === curProvider, pid === rec);
	});
	const picked = await ctx.ui.select(`Select ${role.toUpperCase()} provider`, labels);
	if (picked === undefined) return undefined;
	const idx = labels.indexOf(picked);
	if (idx < 0) return undefined;
	const pid = ordered[idx];
	const provider = providers[pid];
	if (!provider?.available) {
		ctx.ui.notify(`${provider?.label ?? pid} is not available: ${provider?.reason ?? "unknown"}`, "warning");
		return undefined;
	}
	return pid;
}

async function pickLLMSettingModel(
	ctx: ExtensionContext,
	catalog: SidecarLLMSettingCatalogResponse,
	role: LLMSettingRole,
	providerId: string,
): Promise<string | undefined> {
	const providers = catalog.providers ?? {};
	const provider = providers[providerId];
	if (!provider) return undefined;
	const models = provider?.models ?? [];
	if (models.length === 0) {
		ctx.ui.notify(`No models reported for ${providerId}.`, "warning");
		return undefined;
	}
	const recommended = catalog.recommended ?? {};
	const current = catalog.current ?? {};
	const rec = recommended[role];
	const recModel = rec?.provider === providerId ? rec.model : undefined;
	const cur = splitLLMSettingRoleSpec(current[role]);
	const curModel = cur.provider === providerId ? cur.model : undefined;
	const modelWeight = (model: string): number => {
		if (model === curModel) return 0;
		if (model === recModel) return 1;
		if (model.endsWith(":free")) return 2;
		return 3;
	};
	const ordered = [...models].sort((a, b) => modelWeight(a) - modelWeight(b));
	const labels = ordered.map((model) => llmSettingLabelWithMarkers(model, model === curModel, model === recModel));
	const picked = await ctx.ui.select(
		`${role.toUpperCase()} model on ${providerId}${modelCatalogLabelSuffix(provider)}`,
		labels,
	);
	if (picked === undefined) return undefined;
	const idx = labels.indexOf(picked);
	return idx >= 0 ? ordered[idx] : undefined;
}

async function pickLLMSettingModelSpec(
	ctx: ExtensionContext,
	catalog: SidecarLLMSettingCatalogResponse,
	role: LLMSettingRole,
): Promise<LLMSettingPick | undefined> {
	const provider = await pickLLMSettingProvider(ctx, catalog, role);
	if (!provider) return undefined;
	const model = await pickLLMSettingModel(ctx, catalog, role, provider);
	return model ? { provider, model } : undefined;
}

// Providers implemented inside the Python sidecar, exposed to Pi through the
// local OpenAI-compatible /v1/chat/completions proxy.
const SIDECAR_CHAT_PROXY_PROVIDERS = new Set<string>(["anthropic-agent-sdk"]);
const TRUE_SIDECAR_CHAT_PROVIDERS = new Set<string>();

// Footer honesty: only set this override for providers that truly execute chat
// outside Pi. The set is intentionally empty until that proxy exists; otherwise
// the footer can say "Claude" while provider calls still hit GLM/GPT.
function setChatModelStatus(
	ctx: ExtensionContext,
	chat: { provider?: string | null; model?: string | null } | undefined,
): void {
	const provider = chat?.provider ?? undefined;
	const model = chat?.model ?? undefined;
	const routed = provider !== undefined && TRUE_SIDECAR_CHAT_PROVIDERS.has(provider);
	try {
		ctx.ui.setStatus("jlc-chat-model", routed && model ? `(${provider}) ${model}` : undefined);
	} catch {
		/* footer stale on reload */
	}
}

function isSidecarChatProxyProvider(provider: string | undefined): boolean {
	return provider !== undefined && SIDECAR_CHAT_PROXY_PROVIDERS.has(provider);
}

function explicitWorkerModelSpec(): { provider: string; model: string } | undefined {
	if (process.env.JARVIS_SPAWNED !== "1") return undefined;
	const provider = process.env.JARVIS_DEFAULT_PROVIDER?.trim();
	const model = process.env.JARVIS_DEFAULT_MODEL?.trim();
	if (!provider || !model) return undefined;
	return { provider, model };
}

function sameModelSpec(
	left: { provider?: string | null; model?: string | null } | undefined,
	right: { provider?: string | null; model?: string | null } | undefined,
): boolean {
	return Boolean(left?.provider && left?.model && left.provider === right?.provider && left.model === right?.model);
}

function shouldAutoSwapToSidecarChatProxy(
	ctx: ExtensionContext,
	target: { provider?: string | null; model?: string | null } | undefined,
): target is { provider: string; model: string } {
	if (!target?.provider || !target.model) return false;
	if (!isSidecarChatProxyProvider(target.provider)) return false;
	const explicitWorkerModel = explicitWorkerModelSpec();
	if (!explicitWorkerModel) return true;
	if (sameModelSpec(explicitWorkerModel, target)) return true;
	return ctx.model?.provider === target.provider && ctx.model?.id === target.model;
}

function sidecarChatProxyBaseUrl(): string | undefined {
	const baseUrl = sidecarUrlCandidates()[0];
	return baseUrl ? `${baseUrl}/v1` : undefined;
}

function registerSidecarChatProxyProvider(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	target: { provider: string; model: string },
): boolean {
	if (!isSidecarChatProxyProvider(target.provider)) return true;
	const baseUrl = sidecarChatProxyBaseUrl();
	if (!baseUrl) {
		ctx.ui.notify(`JARVIS sidecar proxy unavailable for ${target.provider}/${target.model}.`, "error");
		return false;
	}
	const pairId = process.env.JARVIS_PAIR_ID?.trim();
	try {
		pi.registerProvider(target.provider, {
			name: "JARVIS Sidecar Chat",
			baseUrl,
			apiKey: "jarvis-local-sidecar-proxy",
			api: "openai-completions",
			authHeader: false,
			headers: pairId ? { "X-Jarvis-Pair": pairId } : undefined,
			models: [
				{
					id: target.model,
					name: target.model,
					reasoning: true,
					thinkingLevelMap: {
						off: null,
						minimal: "minimal",
						low: "low",
						medium: "medium",
						high: "high",
						xhigh: "xhigh",
					},
					input: ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: 200000,
					maxTokens: 32000,
					compat: {
						supportsStore: false,
						supportsDeveloperRole: true,
						supportsReasoningEffort: true,
						supportsUsageInStreaming: true,
						maxTokensField: "max_completion_tokens",
						thinkingFormat: "openai",
					},
				},
			],
		});
		return true;
	} catch (error) {
		ctx.ui.notify(
			`JARVIS sidecar proxy registration failed for ${target.provider}/${target.model}: ${
				error instanceof Error ? error.message : String(error)
			}`,
			"error",
		);
		return false;
	}
}

async function ensureSidecarChatProxyLive(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	target: { provider: string; model: string },
): Promise<boolean> {
	if (!registerSidecarChatProxyProvider(pi, ctx, target)) return false;
	if (ctx.model?.provider === target.provider && ctx.model?.id === target.model) {
		setChatModelStatus(ctx, undefined);
		return true;
	}
	refreshPiRuntimeAuth(ctx);
	const next = ctx.modelRegistry.find?.(target.provider, target.model);
	if (!next) {
		ctx.ui.notify(
			`JARVIS sidecar proxy registered, but ${target.provider}/${target.model} was not in the Pi model registry; restart Pi.`,
			"warning",
		);
		return false;
	}
	const swapped = await pi.setModel(next);
	if (!swapped) {
		ctx.ui.notify(
			`JARVIS sidecar proxy was registered, but Pi refused ${target.provider}/${target.model}.`,
			"warning",
		);
		return false;
	}
	setChatModelStatus(ctx, undefined);
	return true;
}

async function applyLLMSetting(
	pi: ExtensionAPI,
	ctx: ExtensionContext,
	body: { chat?: string; subagent?: string; router?: string; encoder?: string; force?: boolean },
	liveChat?: { provider: string; model: string },
): Promise<{ result: SidecarLLMSettingApplyResponse; liveSwapped?: boolean } | undefined> {
	const result = await postSidecar<SidecarLLMSettingApplyResponse>("/llmsetting/apply", body);
	if (!result?.ok) {
		const hint = result?.hint ? ` (${result.hint})` : "";
		ctx.ui.notify(`JARVIS model-setting apply failed: ${result?.error ?? "sidecar unavailable"}${hint}`, "error");
		return undefined;
	}
	for (const note of result.corrections ?? []) {
		ctx.ui.notify(`JARVIS model-setting: ${note}`, "info");
	}
	if (result.reload_warning) {
		ctx.ui.notify(`JARVIS model-setting: saved, but reload failed — ${result.reload_warning}`, "warning");
	}
	if (body.encoder && !result.reload_warning) {
		// The footer's enc-model line is otherwise only refreshed by the next
		// encode summary, so push the newly saved spec immediately.
		setEncModelStatus(ctx, result.encoder);
	}
	// Fuzzy validation may have corrected the requested spec; swap to what was
	// actually saved, not what the caller typed.
	const swapTarget = liveChat ? (splitModelSpec(result.chat) ?? liveChat) : undefined;
	if (!swapTarget) return { result };
	if (isSidecarChatProxyProvider(swapTarget.provider)) {
		const liveSwapped = await ensureSidecarChatProxyLive(pi, ctx, swapTarget);
		return { result, liveSwapped };
	}

	// Keep the footer's chat line honest on every swap: clear the override for
	// Pi-native models so the footer falls back to Pi's now-updated model line.
	// Do not show sidecar-internal roles here unless a real chat proxy exists.
	setChatModelStatus(ctx, swapTarget);

	refreshPiRuntimeAuth(ctx);
	const next = ctx.modelRegistry.find(swapTarget.provider, swapTarget.model);
	if (!next) {
		ctx.ui.notify(
			`JARVIS models saved (chat=${result.chat}, encoder=${result.encoder}), but ${swapTarget.provider}/${swapTarget.model} was not in the Pi model registry after refresh; saved but not swapped; restart Pi.`,
			"warning",
		);
		return { result, liveSwapped: false };
	}
	const swapped = await pi.setModel(next);
	if (!swapped) {
		ctx.ui.notify(
			`JARVIS models saved, but Pi refused ${swapTarget.provider}/${swapTarget.model}; saved but not swapped; restart Pi.`,
			"warning",
		);
		return { result, liveSwapped: false };
	}
	return { result, liveSwapped: true };
}

async function runModelSetting(pi: ExtensionAPI, ctx: ExtensionContext, forceRefresh = false): Promise<void> {
	const catalog = await fetchLLMSettingCatalog(forceRefresh);
	if (!catalog?.ok || !catalog.providers) {
		ctx.ui.notify(
			`JARVIS model-setting: sidecar catalog fetch failed (${(catalog as { error?: string } | undefined)?.error ?? "no response"})`,
			"error",
		);
		return;
	}
	const chatPick = await pickLLMSettingModelSpec(ctx, catalog, "chat");
	if (!chatPick) return;
	const subagentPick = await pickLLMSettingModelSpec(ctx, catalog, "subagent");
	if (!subagentPick) return;
	const encoderPick = await pickLLMSettingModelSpec(ctx, catalog, "encoder");
	if (!encoderPick) return;

	const applied = await applyLLMSetting(
		pi,
		ctx,
		{
			chat: `${chatPick.provider}/${chatPick.model}`,
			subagent: `${subagentPick.provider}/${subagentPick.model}`,
			encoder: `${encoderPick.provider}/${encoderPick.model}`,
		},
		{ provider: chatPick.provider, model: chatPick.model },
	);
	if (!applied) return;
	saveSubagentModelUserSet(true);
	setChatModelStatus(ctx, { provider: chatPick.provider, model: chatPick.model });
	if (applied.liveSwapped !== true) {
		return;
	}
	ctx.ui.notify(
		`JARVIS chat -> ${chatPick.provider}/${chatPick.model} (live). Subagent=${subagentPick.provider}/${subagentPick.model}. Encoder=${encoderPick.provider}/${encoderPick.model}.`,
		"info",
	);
}

async function ensureSubagentModelSelectedForDelegate(pi: ExtensionAPI, ctx: ExtensionContext): Promise<void> {
	if (loadSubagentModelUserSet()) return;
	if (activeDirectiveTurn) return;
	if (!ctx?.hasUI || typeof ctx.ui?.select !== "function") return;
	const catalog = await fetchLLMSettingCatalog(false);
	if (!catalog?.ok || !catalog.providers) {
		ctx.ui.notify(
			`JARVIS subagent model picker skipped: ${(catalog as { error?: string } | undefined)?.error ?? "sidecar catalog unavailable"}`,
			"warning",
		);
		return;
	}
	const pick = await pickLLMSettingModelSpec(ctx, catalog, "subagent");
	if (!pick) return;
	const applied = await applyLLMSetting(pi, ctx, {
		subagent: `${pick.provider}/${pick.model}`,
	});
	if (!applied?.result.ok) return;
	saveSubagentModelUserSet(true);
	ctx.ui.notify(`JARVIS subagent -> ${applied.result.subagent}.`, "info");
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

function extractToolCwd(input: unknown): string | undefined {
	if (!input || typeof input !== "object") return undefined;
	const args = input as { cwd?: unknown; working_directory?: unknown };
	if (typeof args.cwd === "string" && args.cwd.trim()) return args.cwd;
	if (typeof args.working_directory === "string" && args.working_directory.trim()) return args.working_directory;
	return undefined;
}

function managedProcessStateDir(): string {
	const configured = process.env.JARVIS_MANAGED_PROCESS_DIR;
	return configured?.trim() ? path.resolve(configured) : path.join(os.homedir(), ".jarvis-code", "managed-processes");
}

function sanitizeManagedProcessId(value: string | undefined): string {
	const raw = value?.trim() || `proc-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
	const safe = raw.replace(/[^A-Za-z0-9_.-]+/g, "-").replace(/^-+|-+$/g, "");
	return safe || `proc-${Date.now().toString(36)}`;
}

function managedProcessStatePath(record: Pick<ManagedProcessRecord, "id" | "ownerPid">): string {
	return path.join(managedProcessStateDir(), `${record.ownerPid}-${sanitizeManagedProcessId(record.id)}.json`);
}

function managedProcessStatePayload(record: ManagedProcessRecord): ManagedProcessStateRecord & { version: number } {
	return {
		version: MANAGED_PROCESS_STATE_VERSION,
		id: record.id,
		command: record.command,
		args: record.args,
		cwd: record.cwd,
		pid: record.pid,
		ownerPid: record.ownerPid,
		startedAt: record.startedAt,
		procStartToken: record.procStartToken,
		ownerProcStartToken: record.ownerProcStartToken,
		pgid: record.pgid,
		logPath: record.logPath,
		healthUrl: record.healthUrl,
	};
}

function writeManagedProcessState(record: ManagedProcessRecord): void {
	try {
		const dir = managedProcessStateDir();
		fs.mkdirSync(dir, { recursive: true });
		fs.writeFileSync(
			managedProcessStatePath(record),
			`${JSON.stringify(managedProcessStatePayload(record), null, 2)}\n`,
			{
				encoding: "utf-8",
				mode: 0o600,
			},
		);
	} catch {
		/* process tracking is still kept in memory */
	}
}

function removeManagedProcessState(record: Pick<ManagedProcessRecord, "id" | "ownerPid">): void {
	try {
		fs.rmSync(managedProcessStatePath(record), { force: true });
	} catch {
		/* best-effort cleanup */
	}
}

function readManagedProcessStateFile(filePath: string): ManagedProcessStateRecord | undefined {
	try {
		const parsed = JSON.parse(fs.readFileSync(filePath, "utf-8")) as Partial<ManagedProcessStateRecord> & {
			version?: unknown;
		};
		const id = typeof parsed.id === "string" ? sanitizeManagedProcessId(parsed.id) : "";
		const command = typeof parsed.command === "string" ? parsed.command : "";
		const cwd = typeof parsed.cwd === "string" ? parsed.cwd : "";
		const pid = typeof parsed.pid === "number" ? parsed.pid : Number(parsed.pid);
		const ownerPid = typeof parsed.ownerPid === "number" ? parsed.ownerPid : Number(parsed.ownerPid);
		if (!id || !command || !cwd || !Number.isInteger(pid) || !Number.isInteger(ownerPid)) return undefined;
		const pgidRaw = typeof parsed.pgid === "number" ? parsed.pgid : Number(parsed.pgid);
		return {
			id,
			command,
			args: Array.isArray(parsed.args) ? parsed.args.map((arg) => String(arg)) : [],
			cwd,
			pid,
			ownerPid,
			startedAt: typeof parsed.startedAt === "string" ? parsed.startedAt : "",
			procStartToken:
				typeof parsed.procStartToken === "string" && parsed.procStartToken ? parsed.procStartToken : undefined,
			ownerProcStartToken:
				typeof parsed.ownerProcStartToken === "string" && parsed.ownerProcStartToken
					? parsed.ownerProcStartToken
					: undefined,
			pgid: Number.isInteger(pgidRaw) && pgidRaw > 0 ? pgidRaw : undefined,
			logPath: typeof parsed.logPath === "string" ? parsed.logPath : undefined,
			healthUrl: typeof parsed.healthUrl === "string" ? parsed.healthUrl : undefined,
		};
	} catch {
		return undefined;
	}
}

function pidAlive(pid: number): boolean {
	if (!Number.isInteger(pid) || pid <= 0) return false;
	try {
		process.kill(pid, 0);
		return true;
	} catch (error) {
		return (error as NodeJS.ErrnoException)?.code === "EPERM";
	}
}

function sleepMs(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

// --- PID-reuse guard -------------------------------------------------------
// When a managed process exits, the OS is free to recycle its PID onto a totally
// unrelated process. A kill that trusts only the recorded PID number can then
// slaughter the wrong process. We capture an OS-level "start token" for each PID
// at spawn and re-check it before ever terminating, so a recycled PID is treated
// as not-ours. (Follow-up ticket T1 / docs cross-platform-porting "Process
// management".)

function execFileText(file: string, args: string[], timeoutMs = 4000): Promise<string | undefined> {
	return new Promise((resolve) => {
		try {
			execFile(file, args, { timeout: timeoutMs, windowsHide: true }, (error, stdout) => {
				resolve(error ? undefined : String(stdout));
			});
		} catch {
			resolve(undefined);
		}
	});
}

// null = read attempted and failed (don't retry); undefined = not yet tried.
let cachedLinuxBootId: string | null | undefined;

// The kernel boot id changes on every boot. It anchors the Linux start token to
// THIS boot so a token persisted before a reboot can never falsely match a
// recycled PID on a later boot (field 22 below is jiffies-since-boot, which
// resets each boot — early-boot daemons even get repeatable values).
function readLinuxBootId(): string | undefined {
	if (cachedLinuxBootId !== undefined) return cachedLinuxBootId ?? undefined;
	try {
		const raw = fs.readFileSync("/proc/sys/kernel/random/boot_id", "utf-8").trim();
		cachedLinuxBootId = raw || null;
	} catch {
		cachedLinuxBootId = null;
	}
	return cachedLinuxBootId ?? undefined;
}

// Pure: build the Linux start token from a /proc/<pid>/stat line and the current
// boot id. comm (field 2) is wrapped in parens and can itself contain spaces or
// ')', so parse from the final ')'. starttime is field 22; the text after the
// final ')' begins at field 3 (state), so starttime is index 19. starttime alone
// is jiffies-since-boot and is NOT reboot-safe, so the boot id is mandatory — a
// missing boot id yields no token (unprovable -> the caller stays conservative).
export function parseLinuxProcStartToken(stat: string, bootId: string): string | undefined {
	if (!bootId) return undefined;
	const close = stat.lastIndexOf(")");
	if (close < 0) return undefined;
	const rest = stat.slice(close + 1).trim();
	if (!rest) return undefined;
	const fields = rest.split(/\s+/);
	const starttime = fields[19];
	return starttime && /^\d+$/.test(starttime) ? `linux:${bootId}:${starttime}` : undefined;
}

export async function readProcessStartToken(pid: number): Promise<string | undefined> {
	if (!Number.isInteger(pid) || pid <= 0) return undefined;
	try {
		if (process.platform === "linux") {
			const bootId = readLinuxBootId();
			// Without a boot id we cannot anchor jiffies-since-boot to this boot, so
			// the token would not be reboot-safe — produce none and stay conservative.
			if (!bootId) return undefined;
			let stat = "";
			try {
				stat = fs.readFileSync(`/proc/${pid}/stat`, "utf-8");
			} catch {
				return undefined;
			}
			return parseLinuxProcStartToken(stat, bootId);
		}
		if (process.platform === "darwin") {
			const out = await execFileText("ps", ["-o", "lstart=", "-p", String(pid)]);
			const value = out?.replace(/\s+/g, " ").trim();
			return value ? `darwin:${value}` : undefined;
		}
		if (process.platform === "win32") {
			const out = await execFileText("powershell", [
				"-NoProfile",
				"-NonInteractive",
				"-Command",
				`$p = Get-Process -Id ${pid} -ErrorAction SilentlyContinue; if ($p) { $p.StartTime.Ticks }`,
			]);
			const value = out?.trim();
			return value && /^\d+$/.test(value) ? `win:${value}` : undefined;
		}
	} catch {
		return undefined;
	}
	return undefined;
}

export type ManagedPidIdentity = "owned" | "reused" | "unknown" | "dead";

// Pure decision: does the record still own the live PID?
//  - dead:    PID is gone — nothing to kill, the record can be dropped.
//  - owned:   proven same process (matching OS start token, or a live child
//             handle owned by this very process) — safe to terminate.
//  - reused:  both tokens known and DIFFERENT — the OS recycled the PID onto an
//             unrelated process; never kill it, drop our stale record.
//  - unknown: cannot prove identity (a token is missing, or only a coarse token
//             matched) — the safe direction is to NOT auto-kill / NOT auto-allow.
// coarseToken marks a low-resolution token (e.g. macOS `ps lstart`, second
// granularity): a MATCH is too weak to authorize a kill (a same-second PID reuse
// would collide), so it degrades to "unknown". A MISMATCH is still a safe "reused".
export function classifyManagedPidIdentity(input: {
	alive: boolean;
	childLiveOwned: boolean;
	recordToken?: string;
	liveToken?: string;
	coarseToken?: boolean;
}): ManagedPidIdentity {
	if (!input.alive) return "dead";
	if (input.childLiveOwned) return "owned";
	if (input.recordToken && input.liveToken) {
		if (input.recordToken !== input.liveToken) return "reused";
		return input.coarseToken ? "unknown" : "owned";
	}
	return "unknown";
}

// A child handle from THIS process that has not reported exit is proof the PID is
// still ours: Node fires "exit" before the OS can recycle the number, and the
// exit handler removes the record. Cross-instance records (loaded from a state
// file) have no child handle and must fall back to the OS start token.
function managedRecordChildLiveOwned(record: ManagedProcessRecord): boolean {
	const child = record.child;
	return (
		record.ownerPid === process.pid &&
		!!child &&
		child.exitCode === null &&
		child.signalCode === null &&
		!child.killed
	);
}

async function verifyManagedPidIdentity(record: ManagedProcessRecord): Promise<ManagedPidIdentity> {
	const alive = pidAlive(record.pid);
	if (!alive) return "dead";
	if (managedRecordChildLiveOwned(record)) return "owned";
	const liveToken = await readProcessStartToken(record.pid);
	return classifyManagedPidIdentity({
		alive,
		childLiveOwned: false,
		recordToken: record.procStartToken,
		liveToken,
		// macOS `ps lstart` is only second-resolution, so a token match is not
		// strong enough on its own to authorize killing a cross-instance PID.
		coarseToken: process.platform === "darwin",
	});
}

// The start token of THIS process (the owner). Stable for our lifetime, so read
// it once and memoize; "" records a failed/empty read so we don't keep retrying.
let cachedOwnStartToken: string | undefined;
let ownStartTokenPromise: Promise<string | undefined> | undefined;
function ensureOwnStartToken(): Promise<string | undefined> {
	if (cachedOwnStartToken !== undefined) return Promise.resolve(cachedOwnStartToken || undefined);
	if (!ownStartTokenPromise) {
		ownStartTokenPromise = readProcessStartToken(process.pid).then((token) => {
			cachedOwnStartToken = token ?? "";
			return token;
		});
	}
	return ownStartTokenPromise;
}

// Capture the OS start tokens in the background so the hot start path is never
// blocked by a (Windows) PowerShell probe; persist them once known, but only if
// this record is still the live one (a fast exit may have forgotten it already).
function captureManagedProcessStartToken(record: ManagedProcessRecord): void {
	void Promise.all([readProcessStartToken(record.pid), ensureOwnStartToken()]).then(([procToken, ownerToken]) => {
		if (managedProcesses.get(record.id) !== record) return;
		let changed = false;
		if (procToken && record.procStartToken !== procToken) {
			record.procStartToken = procToken;
			changed = true;
		}
		if (ownerToken && record.ownerProcStartToken !== ownerToken) {
			record.ownerProcStartToken = ownerToken;
			changed = true;
		}
		if (changed) writeManagedProcessState(record);
	});
}

function quoteWindowsShellToken(value: string): string {
	if (value.length === 0 || /[\s"&()<>^|]/.test(value)) return `"${value.replace(/"/g, '\\"')}"`;
	return value;
}

function buildWindowsManagedProcessCommand(command: string, args: string[]): string {
	return [command, ...args].map(quoteWindowsShellToken).join(" ");
}

function forgetManagedProcess(record: ManagedProcessRecord): void {
	if (managedProcesses.get(record.id)?.pid === record.pid) managedProcesses.delete(record.id);
	removeManagedProcessState(record);
}

type ManagedProcessExit = { code: number | null; signal: NodeJS.Signals | null };

function waitForManagedProcessSpawn(
	child: ChildProcess,
	getSpawnError: () => Error | undefined,
): Promise<Error | undefined> {
	const existingError = getSpawnError();
	if (existingError) return Promise.resolve(existingError);
	return new Promise((resolve) => {
		let settled = false;
		const finish = (error?: Error) => {
			if (settled) return;
			settled = true;
			child.off("spawn", onSpawn);
			child.off("error", onError);
			resolve(error);
		};
		const onSpawn = () => finish();
		const onError = (error: Error) => finish(error);
		child.once("spawn", onSpawn);
		child.once("error", onError);
	});
}

function waitForManagedProcessExit(child: ChildProcess): Promise<ManagedProcessExit> {
	return new Promise((resolve) => {
		child.once("exit", (code, signal) => resolve({ code, signal }));
	});
}

function formatManagedProcessExit(exit: ManagedProcessExit): string {
	const reason = exit.signal ? `signal ${exit.signal}` : `exit code ${exit.code ?? "unknown"}`;
	return `process exited before it became ready (${reason})`;
}

async function waitUntilPidExits(pid: number, timeoutMs: number): Promise<boolean> {
	const deadline = Date.now() + Math.max(0, timeoutMs);
	while (Date.now() <= deadline) {
		if (!pidAlive(pid)) return true;
		await sleepMs(100);
	}
	return !pidAlive(pid);
}

async function terminateManagedPid(pid: number, timeoutMs = 3000): Promise<boolean> {
	if (!pidAlive(pid)) return true;
	if (process.platform === "win32") {
		await new Promise<void>((resolve) => {
			const killer = spawn("taskkill", ["/F", "/T", "/PID", String(pid)], {
				stdio: "ignore",
				windowsHide: true,
			});
			killer.once("error", () => resolve());
			killer.once("close", () => resolve());
		});
		return waitUntilPidExits(pid, timeoutMs);
	}
	try {
		process.kill(-pid, "SIGTERM");
	} catch {
		try {
			process.kill(pid, "SIGTERM");
		} catch {
			/* already gone */
		}
	}
	if (await waitUntilPidExits(pid, Math.min(timeoutMs, 1500))) return true;
	try {
		process.kill(-pid, "SIGKILL");
	} catch {
		try {
			process.kill(pid, "SIGKILL");
		} catch {
			/* already gone */
		}
	}
	return waitUntilPidExits(pid, timeoutMs);
}

function terminateManagedPidBestEffort(pid: number): void {
	if (!pidAlive(pid)) return;
	if (process.platform === "win32") {
		try {
			const killer = spawn("taskkill", ["/F", "/T", "/PID", String(pid)], {
				stdio: "ignore",
				detached: true,
				windowsHide: true,
			});
			killer.once("error", () => undefined);
			killer.unref();
		} catch {
			/* best-effort */
		}
		return;
	}
	try {
		process.kill(-pid, "SIGKILL");
	} catch {
		try {
			process.kill(pid, "SIGKILL");
		} catch {
			/* best-effort */
		}
	}
}

// Is the record's owner JARVIS process genuinely still running? A live ownerPid
// is not enough — that PID number could have been recycled after the owner died,
// which would otherwise make us skip cleanup forever and leak the orphan. Only an
// unmatched owner token disproves it; a missing/unreadable token stays
// conservative (treat as alive) so we never reap an actually-live owner's procs.
async function ownerProcessStillAlive(record: ManagedProcessStateRecord): Promise<boolean> {
	if (!pidAlive(record.ownerPid)) return false;
	if (!record.ownerProcStartToken) return true;
	const liveOwnerToken = await readProcessStartToken(record.ownerPid);
	if (!liveOwnerToken) return true;
	return liveOwnerToken === record.ownerProcStartToken;
}

async function cleanupStaleManagedProcesses(): Promise<void> {
	if (managedProcessStaleCleanupDone) return;
	managedProcessStaleCleanupDone = true;
	let files: string[] = [];
	try {
		files = fs
			.readdirSync(managedProcessStateDir(), { withFileTypes: true })
			.filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
			.map((entry) => path.join(managedProcessStateDir(), entry.name));
	} catch {
		return;
	}
	for (const filePath of files) {
		const record = readManagedProcessStateFile(filePath);
		if (!record) {
			try {
				fs.rmSync(filePath, { force: true });
			} catch {
				/* best-effort */
			}
			continue;
		}
		if (record.ownerPid === process.pid) continue;
		if (await ownerProcessStillAlive(record)) continue;
		const identity = await verifyManagedPidIdentity(record);
		// SECURITY: a state file on disk is UNTRUSTED input. Any same-user process
		// (including the agent itself, which can write files) can forge a record
		// with a target PID and that PID's real, precomputed start token. So the
		// disk sweep must NEVER terminate a live PID — doing so would bypass the
		// shell kill guard and become an arbitrary-process-kill primitive. Disk
		// cleanup only reclaims OBSOLETE files; the sole paths allowed to kill are
		// this process's own in-memory child records (stopManagedProcess /
		// session shutdown), which carry an unforgeable live ChildProcess handle.
		if (identity === "dead" || identity === "reused") {
			// Our process is gone or the PID was recycled — the file is obsolete.
			try {
				fs.rmSync(filePath, { force: true });
			} catch {
				/* best-effort */
			}
		}
		// "owned"/"unknown" on a live PID: we will NOT kill on a disk record's
		// say-so. Leave the file; a genuine orphan is reclaimed once its PID dies
		// (-> "dead" -> removed). A forged file is inert since we never act on it.
	}
}

async function managedProcessOwnsPidVerified(pid: number): Promise<boolean> {
	for (const record of managedProcesses.values()) {
		if (record.pid !== pid) continue;
		return (await verifyManagedPidIdentity(record)) === "owned";
	}
	return false;
}

function managedProcessDetails(record: ManagedProcessRecord): Record<string, unknown> {
	return {
		id: record.id,
		pid: record.pid,
		alive: pidAlive(record.pid),
		command: record.command,
		args: record.args,
		cwd: record.cwd,
		started_at: record.startedAt,
		log_path: record.logPath,
		health_url: record.healthUrl,
	};
}

async function waitForManagedProcessHealth(url: string, waitSeconds: number): Promise<{ ok: boolean; error?: string }> {
	const deadline = Date.now() + Math.max(0, waitSeconds) * 1000;
	let lastError = "";
	while (Date.now() <= deadline) {
		try {
			const controller = new AbortController();
			const timer = setTimeout(() => controller.abort(), 1500);
			const response = await fetch(url, { signal: controller.signal });
			clearTimeout(timer);
			if (response.ok) return { ok: true };
			lastError = `HTTP ${response.status}`;
		} catch (error) {
			lastError = error instanceof Error ? error.message : String(error);
		}
		await sleepMs(250);
	}
	return { ok: false, error: lastError || "health URL did not become ready" };
}

async function stopManagedProcess(record: ManagedProcessRecord): Promise<Record<string, unknown>> {
	const identity = await verifyManagedPidIdentity(record);
	// Never terminate a PID the OS has recycled onto an unrelated process, nor one
	// whose identity we cannot prove. Drop the bookkeeping either way.
	if (identity === "reused" || identity === "unknown") {
		managedProcesses.delete(record.id);
		removeManagedProcessState(record);
		return {
			ok: true,
			id: record.id,
			pid: record.pid,
			stopped: false,
			skipped: identity === "reused" ? "pid_reused" : "identity_unverified",
		};
	}
	const stopped = await terminateManagedPid(record.pid);
	managedProcesses.delete(record.id);
	removeManagedProcessState(record);
	return { ok: stopped, id: record.id, pid: record.pid, stopped };
}

async function stopAllManagedProcesses(): Promise<void> {
	const records = Array.from(managedProcesses.values());
	await Promise.all(records.map((record) => stopManagedProcess(record).catch(() => undefined)));
}

function stopAllManagedProcessesBestEffort(): void {
	for (const record of managedProcesses.values()) {
		terminateManagedPidBestEffort(record.pid);
		removeManagedProcessState(record);
	}
	managedProcesses.clear();
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

const TODO_CONTEXT_START = "<jarvis_todo>";
const TODO_CONTEXT_END = "</jarvis_todo>";

function clearReadBeforeEditRegistry(): void {
	readBeforeEditRegistry = new Map<string, number>();
}

function maybeStatPath(absPath: string): { exists: boolean; mtimeMs?: number; uncertain?: boolean } {
	try {
		const stat = fs.statSync(absPath);
		return { exists: true, mtimeMs: stat.mtimeMs };
	} catch (error) {
		const code = (error as NodeJS.ErrnoException)?.code;
		if (code === "ENOENT" || code === "ENOTDIR") return { exists: false };
		return { exists: true, uncertain: true };
	}
}

function recordReadBeforeEditPath(rawPath: string | undefined, cwd: string | undefined, ctx?: ExtensionContext): void {
	if (!rawPath) return;
	const absPath = resolveTurnMutationPath(rawPath, cwd, ctx);
	if (!absPath) return;
	const stat = maybeStatPath(absPath);
	if (!stat.exists || stat.uncertain || typeof stat.mtimeMs !== "number" || !Number.isFinite(stat.mtimeMs)) return;
	readBeforeEditRegistry.set(normalizePathForCompare(absPath), stat.mtimeMs);
}

function recordReadBeforeEditFromMetadata(
	toolName: unknown,
	metadata: JarvisToolMetadata,
	ctx?: ExtensionContext,
): void {
	const tool = String(toolName ?? "").toLowerCase();
	if (tool !== "read" && tool !== "edit" && tool !== "write" && tool !== "write_file") return;
	recordReadBeforeEditPath(metadata.sourcePath, metadata.cwd, ctx);
	for (const sourcePath of metadata.sourcePaths ?? []) {
		recordReadBeforeEditPath(sourcePath, metadata.cwd, ctx);
	}
}

function maybeBlockEditBeforeRead(
	toolName: string,
	input: unknown,
	ctx: ExtensionContext,
): { block: true; reason: string } | undefined {
	if (!isProjectRoute(currentRoute)) return undefined;
	if (toolName !== "edit" && toolName !== "write" && toolName !== "write_file") return undefined;
	const rawPath = extractToolPath(input);
	if (!rawPath) return undefined;
	const absPath = resolveTurnMutationPath(rawPath, extractToolCwd(input), ctx);
	if (!absPath) return undefined;
	const stat = maybeStatPath(absPath);
	if ((toolName === "write" || toolName === "write_file") && !stat.exists) return undefined;
	if (stat.uncertain) return undefined;
	const key = normalizePathForCompare(absPath);
	const readMtimeMs = readBeforeEditRegistry.get(key);
	if (readMtimeMs === undefined) {
		// T2a observability: blocks are rare (cooperative models read first), so emit
		// unconditionally — they should surface in the live sidecar debug stream the moment
		// the gate actually fires, even without the durable file sink pre-enabled.
		recordSubturnDebugEvent("read_before_edit_block", {
			tool: toolName,
			path: absPath,
			reason_kind: "unread",
			effective_route: currentRoute,
		});
		return {
			block: true,
			reason: `Read ${absPath} before editing it in this project route.`,
		};
	}
	if (typeof stat.mtimeMs === "number" && Number.isFinite(stat.mtimeMs) && stat.mtimeMs > readMtimeMs) {
		recordSubturnDebugEvent("read_before_edit_block", {
			tool: toolName,
			path: absPath,
			reason_kind: "stale_mtime",
			effective_route: currentRoute,
			read_mtime: readMtimeMs,
			disk_mtime: stat.mtimeMs,
		});
		return {
			block: true,
			reason: `Read ${absPath} again before editing it; the file changed on disk after the last read.`,
		};
	}
	// T2a denominator: allow events fire once per edit (high frequency), so gate them behind
	// the durable debug sink being active. With logging off there is nobody to read the rate,
	// and emitting a per-edit sidecar POST would be wasted work. With JARVIS_SUBTURN_EVENT_LOG
	// set, both block and allow flow to the JSONL so block-rate (blocks / blocks+allows) is measurable.
	if (subturnEventLogTarget()) {
		recordSubturnDebugEvent("read_before_edit_allow", {
			tool: toolName,
			path: absPath,
			effective_route: currentRoute,
		});
	}
	return undefined;
}

function isJarvisTodoStatus(value: unknown): value is JarvisTodoStatus {
	return value === "pending" || value === "in_progress" || value === "completed";
}

function normalizeTodoItems(input: unknown): JarvisTodoItem[] {
	const rawItems = Array.isArray(input) ? input : [];
	const items: JarvisTodoItem[] = [];
	for (const raw of rawItems) {
		if (!raw || typeof raw !== "object") continue;
		const record = raw as { content?: unknown; status?: unknown };
		const content = typeof record.content === "string" ? record.content.trim() : "";
		if (!content) continue;
		items.push({
			content,
			status: isJarvisTodoStatus(record.status) ? record.status : "pending",
		});
		if (items.length >= 50) break;
	}
	return items;
}

function todoStatusMarker(status: JarvisTodoStatus): string {
	if (status === "completed") return "[x]";
	if (status === "in_progress") return "[>]";
	return "[ ]";
}

function renderTodoList(items: JarvisTodoItem[]): string {
	if (!items.length) return "- (empty)";
	return items.map((item) => `- ${todoStatusMarker(item.status)} ${item.content}`).join("\n");
}

function stripTodoContextBlock(context: string): string {
	return context
		.replace(/(?:\r?\n){0,2}<jarvis_todo>[\s\S]*?<\/jarvis_todo>(?:\r?\n){0,2}/g, "\n\n")
		.replace(/\n{3,}/g, "\n\n")
		.trim();
}

function todoContextBlock(): string {
	if (!currentTodoList.length) return "";
	return `${TODO_CONTEXT_START}\n## TODO\n${renderTodoList(currentTodoList)}\n${TODO_CONTEXT_END}`;
}

function contextWithTodoForRoute(
	context: string | undefined,
	route: EffectiveTurnRoute = currentRoute,
): string | undefined {
	if (context === undefined) return undefined;
	const base = stripTodoContextBlock(context);
	if (!isProjectRoute(route) || !currentTodoList.length) return base;
	return `${base}${base ? "\n\n" : ""}${todoContextBlock()}`;
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
	if (shellLikeToolName(tool) && metadata.command && looksLikeVerification(metadata.command)) {
		verificationRanThisTurn = true;
	}
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
	verificationRanThisTurn = false;
	turnJarvisMdUpdated = false;
	lastSdkToolTrailerRecords = [];
}

// Regime-B order-of-operations fix: pi strips the [[JARVIS_TOOL_ACTIVITY]]
// sentinel from the assistant message IN PLACE during streaming
// (sanitizeAssistantMessageInPlace in message_update, sanitizeAssistantMessage in
// message_end) BEFORE agent_end's reconstruct can read it. So agent_end's live
// extract finds nothing and JARVIS.md stays blank. We capture the trailer records
// from the RAW message_end text (before any sanitize) here, and agent_end falls
// back to these when the live extract is empty. Reset per-turn so no cross-turn
// leak. Regime A never emits the sentinel, so this stays empty there.
let lastSdkToolTrailerRecords: JarvisSdkToolActivityRecord[] = [];

// Regime-B tool-activity trailer (item 1 memory sensor).
//
// In the anthropic-agent-sdk regime the SDK runs its own tool loop and never
// returns tool_calls to pi, so pi.on("tool_execution_end") never fires and
// turnSuccessfulFileMutations stays empty — silently no-opping every post-turn
// observed-work consumer (JARVIS.md patch, workspace auto-register, /turn
// tool_events). The adapter instead emits a regime-neutral structured signal
// (`jarvis_tool_activity`) describing each executed tool. We reconstruct
// turnSuccessfulFileMutations + toolEvents from that signal and feed pi's SAME
// regime-A pipeline (recordJarvisTurnToolOutcome), so there is a single source
// of truth and no server-side orchestration duplication.
type JarvisSdkToolActivityRecord = {
	tool: string;
	tool_use_id?: string;
	abs_path?: string | null;
	mutation_kind?: string;
	success?: boolean;
	command?: string | null;
	result_preview?: string;
};

// The producer attaches the trailer as a top-level `jarvis_tool_activity` array
// on its final OpenAI chunk. pi-ai's openai-completions provider only harvests
// delta.content/reasoning onto the materialized AssistantMessage and drops
// unknown top-level chunk keys, and AssistantMessage carries no metadata slot —
// so the adapter ALSO emits the trailer as a sentinel line embedded in the final
// assistant text. We parse that sentinel here. (If a future provider carry
// surfaces the array on message metadata/extra, extend extractJarvisSdkToolTrailerRecords.)
const JARVIS_SDK_TOOL_TRAILER_MARKER = "[[JARVIS_TOOL_ACTIVITY]]";
const JARVIS_SDK_TOOL_TRAILER_RE = /^[ \t]*\[\[JARVIS_TOOL_ACTIVITY\]\]\s*(\{[\s\S]*?\})\s*$/m;

function coerceJarvisSdkToolActivityRecords(value: unknown): JarvisSdkToolActivityRecord[] {
	if (!Array.isArray(value)) return [];
	const records: JarvisSdkToolActivityRecord[] = [];
	for (const entry of value) {
		if (!entry || typeof entry !== "object") continue;
		const record = entry as Record<string, unknown>;
		const tool = typeof record.tool === "string" ? record.tool : "";
		if (!tool) continue;
		records.push({
			tool,
			tool_use_id: typeof record.tool_use_id === "string" ? record.tool_use_id : undefined,
			abs_path: typeof record.abs_path === "string" ? record.abs_path : null,
			mutation_kind: typeof record.mutation_kind === "string" ? record.mutation_kind : undefined,
			success: record.success !== false,
			command: typeof record.command === "string" ? record.command : null,
			result_preview: typeof record.result_preview === "string" ? record.result_preview : "",
		});
	}
	return records;
}

function extractJarvisSdkToolTrailerRecords(
	assistantMessage: AssistantMessage | undefined,
	messages?: AgentMessage[],
	preCapturedRecords?: JarvisSdkToolActivityRecord[],
): JarvisSdkToolActivityRecord[] {
	// Preferred: a future openai-completions carry that surfaces the array on the
	// message itself (metadata/extra). Tolerated read-only here so the consumer is
	// transport-agnostic and a later producer move needs no consumer change.
	const carry = (assistantMessage as { jarvis_tool_activity?: unknown } | undefined)?.jarvis_tool_activity;
	const carried = coerceJarvisSdkToolActivityRecords(carry);
	if (carried.length) return carried;
	// Fallback (the live transport): a sentinel line in the assistant text.
	const text = assistantMessage ? contentToText(assistantMessage.content) : "";
	const fromText = parseJarvisSdkToolTrailerFromText(text);
	if (fromText.length) return fromText;
	if (messages) {
		for (let i = messages.length - 1; i >= 0; i--) {
			const message = messages[i];
			if (message.role !== "assistant") continue;
			const records = parseJarvisSdkToolTrailerFromText(contentToText((message as AssistantMessage).content));
			if (records.length) return records;
		}
	}
	// Last resort: the trailer captured from the RAW message_end text BEFORE pi's
	// in-place sanitize stripped the sentinel from the live message above. This is
	// the live regime-B path (pi always strips the sentinel before agent_end).
	if (preCapturedRecords?.length) return preCapturedRecords;
	return [];
}

function parseJarvisSdkToolTrailerFromText(text: string): JarvisSdkToolActivityRecord[] {
	if (!text || !text.includes(JARVIS_SDK_TOOL_TRAILER_MARKER)) return [];
	const match = text.match(JARVIS_SDK_TOOL_TRAILER_RE);
	if (!match) return [];
	let parsed: unknown;
	try {
		parsed = JSON.parse(match[1]);
	} catch {
		return [];
	}
	const container = parsed as { jarvis_tool_activity?: unknown } | null;
	return coerceJarvisSdkToolActivityRecords(container?.jarvis_tool_activity);
}

// Strip the trailer sentinel so it never leaks into displayed/persisted text.
function stripJarvisSdkToolTrailer(text: string): string {
	if (!text || !text.includes(JARVIS_SDK_TOOL_TRAILER_MARKER)) return text;
	return text
		.replace(JARVIS_SDK_TOOL_TRAILER_RE, "")
		.replace(/\n{3,}/g, "\n\n")
		.trimEnd();
}

// Regime B ONLY: reconstruct turnSuccessfulFileMutations + toolEvents from the
// adapter trailer using pi's EXISTING regime-A writer so the shape is identical.
// Returns the number of trailer records consumed (0 = nothing reconstructed).
function reconstructJarvisTurnFromSdkTrailer(
	assistantMessage: AssistantMessage | undefined,
	messages: AgentMessage[] | undefined,
	ctx?: ExtensionContext,
	preCapturedRecords?: JarvisSdkToolActivityRecord[],
): number {
	const trailer = extractJarvisSdkToolTrailerRecords(assistantMessage, messages, preCapturedRecords);
	if (!trailer.length) return 0;
	for (const record of trailer) {
		recordJarvisTurnToolOutcome(
			record.tool, // already mapped to {write,edit,write_file,apply_patch,bash} on the producer side
			!record.success, // isError
			record.result_preview ?? "", // output (scanned for verification lines)
			{ command: record.command ?? undefined, sourcePath: record.abs_path ?? undefined },
			ctx,
		);
	}
	// Mirror the regime-A turn_end push (5302-5309) so /turn (tool_events) and the
	// subturn/checkpoint consumers see the activity. Gate identically (caller is
	// regime-B only) to avoid double-push vs the regime-A path.
	toolEvents.push({
		turnIndex: toolEvents.length,
		toolResults: trailer.map((record) => ({
			toolName: record.tool,
			isError: !record.success,
			text: (record.result_preview ?? "").slice(0, 2000),
		})),
	});
	return trailer.length;
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
		if (jarvisProgressVisible()) sendJarvisChatNotice(pi, "✓ JARVIS.md updated (harness backfill)");
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
		sendJarvisChatNotice(pi, "✓ registered → switched → JARVIS.md seed (harness backfill)");
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
		const line = `✓ unregister (backstop): ${projectId} — folder deletion observed`;
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

function jarvisYoloMode(): boolean {
	return process.env.JARVIS_YOLO === "1";
}

function shellLikeToolName(toolName: string): boolean {
	const tool = String(toolName ?? "").toLowerCase();
	return tool === "bash" || tool === "shell" || tool === "powershell" || tool === "pwsh";
}

function normalizeShellCommandForSafety(command: string): string {
	return command.replace(/[`^]/g, "").replace(/\r\n/g, "\n").toLowerCase();
}

function describeBroadProcessKillCommand(command: string): string | undefined {
	const normalized = normalizeShellCommandForSafety(command);
	const checks: Array<[RegExp, string]> = [
		[/\btaskkill(?:\.exe)?\b[\s\S]*(?:\/{1,2}|-)?im\b/, "taskkill image-name process kill"],
		[
			/\bwmic(?:\.exe)?\b[\s\S]*\bprocess\b[\s\S]*\bwhere\b[\s\S]*\bname\s*=[\s\S]*\bdelete\b/,
			"wmic process-name delete",
		],
		[/\b(?:stop-process|spps)\b[\s\S]*-(?:name|processname)\b/, "PowerShell Stop-Process by name"],
		[
			/\b(?:get-process|gps)\b(?![^\n|]*-(?:id|pid)\b)[\s\S]*\|[\s\S]*\b(?:stop-process|spps|kill)\b/,
			"PowerShell pipeline process-name kill",
		],
		[/\bpkill\b/, "pkill process-name kill"],
		[/\bkillall\b/, "killall process-name kill"],
		[/\bkill\b[\s\S]*\$\([^)]*\b(?:pgrep|pidof)\b[^)]*\)/, "kill via pgrep/pidof"],
		[/\b(?:pgrep|pidof)\b[\s\S]*\|[\s\S]*(?:xargs\s+)?\bkill\b/, "pgrep/pidof pipeline kill"],
		[
			/\b(?:eval|invoke-expression|iex)\b[\s\S]*(?:taskkill|stop-process|pkill|killall|pgrep|pidof|wmic\b[\s\S]*\bprocess\b)/,
			"indirect process kill through eval",
		],
		[/\b(?:powershell|pwsh)(?:\.exe)?\b[\s\S]*-(?:enc|encodedcommand)\b/, "encoded PowerShell command"],
		[
			/\b(?:bash|sh|zsh|cmd(?:\.exe)?|powershell(?:\.exe)?|pwsh(?:\.exe)?)\b[\s\S]*(?:kill|taskkill|stop-process|pkill|killall|pgrep|pidof|wmic\b[\s\S]*\bprocess\b)/,
			"indirect shell process kill",
		],
	];
	for (const [pattern, reason] of checks) {
		if (pattern.test(normalized)) return reason;
	}
	return undefined;
}

function addPidKillTargets(rawTargets: string | undefined, pids: Set<number>): void {
	for (const match of rawTargets?.matchAll(/\d+/g) ?? []) {
		const pid = Number(match[0]);
		if (Number.isInteger(pid) && pid > 0) pids.add(pid);
	}
}

function extractPidKillTargets(command: string): number[] {
	const normalized = normalizeShellCommandForSafety(command);
	const pids = new Set<number>();
	if (/\btaskkill(?:\.exe)?\b/.test(normalized)) {
		for (const match of normalized.matchAll(/(?:\/{1,2}|-)?pid\s+([0-9,\s]+)/g)) {
			addPidKillTargets(match[1], pids);
		}
	}
	for (const match of normalized.matchAll(/\b(?:stop-process|spps|kill)\b(?:[^\n;&|]*?)-(?:id|pid)\s+([0-9,\s]+)/g)) {
		addPidKillTargets(match[1], pids);
	}
	for (const match of normalized.matchAll(/(?:^|[;&|]\s*)kill(?:\s+-[a-z0-9]+)*((?:\s+\d+)+)/g)) {
		addPidKillTargets(match[1], pids);
	}
	return Array.from(pids);
}

function hardBlockProcessKill(reason: string): ToolBlockResult {
	return {
		block: true,
		reason: `Blocked process kill: ${reason}. Use managed_process stop for JARVIS-owned background processes; image/name-based broad kills are never allowed.`,
	};
}

async function maybeBlockProcessKillToolCall(
	toolName: string,
	input: unknown,
	ctx: ExtensionContext,
	confirmedKeys: Set<string>,
): Promise<ToolBlockResult | undefined> {
	if (!shellLikeToolName(toolName)) return undefined;
	const command = extractToolCommand(input);
	if (!command) return undefined;
	const broadReason = describeBroadProcessKillCommand(command);
	if (broadReason) return hardBlockProcessKill(broadReason);
	const pids = extractPidKillTargets(command);
	if (pids.length === 0) return undefined;
	// Verified ownership: a PID only auto-passes if it is still provably one of
	// our managed processes (matching OS start token / live child handle), so a
	// recycled PID number can never auto-allow a kill of an unrelated process.
	const ownership = await Promise.all(pids.map((pid) => managedProcessOwnsPidVerified(pid)));
	const unsafePids = pids.filter((_, index) => !ownership[index]);
	if (unsafePids.length === 0) return undefined;
	if (jarvisYoloMode()) {
		return hardBlockProcessKill(`unowned PID kill in YOLO mode (${unsafePids.join(", ")})`);
	}
	const key = safetyKey("process-kill-pid", `${unsafePids.join(",")}:${command.slice(0, 500)}`);
	return confirmJarvisSafety(
		ctx,
		confirmedKeys,
		key,
		"JARVIS process kill",
		[
			`Command: ${command}`,
			"",
			`PID target(s): ${unsafePids.join(", ")}`,
			"",
			"Only JARVIS managed_process-owned PIDs are automatically allowed.",
			"Allow this PID kill for this JARVIS session?",
		].join("\n"),
		`Blocked process kill: unowned PID kill was not confirmed (${unsafePids.join(", ")}).`,
	);
}

async function maybeConfirmExternalMutationToolCall(
	toolName: string,
	absPath: string,
	ctx: ExtensionContext,
	confirmedKeys: Set<string>,
): Promise<ToolBlockResult | undefined> {
	if (jarvisYoloMode()) return undefined;
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
	if (jarvisYoloMode()) return undefined;
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
	if (jarvisYoloMode()) return undefined;
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
	const decision = lastRouteClassifierDecision?.pending_project_decision;
	if (decision === "confirm") {
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
	if (decision === "decline") {
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
		sendJarvisChatNotice(pi, "Interrupt detected: tried to save current work state, but no save scope is available.");
		return;
	}

	const scopeLabel = scope.kind === "project" ? "project" : "chat";
	sendJarvisChatNotice(pi, `Interrupt detected: saving current work to ${scopeLabel} memory.`);
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
		sendJarvisChatNotice(pi, `Interrupt checkpoint saved to ${scopeLabel} JARVIS.md.${savedPath}`);
	} else {
		sendJarvisChatNotice(pi, `Interrupt checkpoint save failed: ${response?.error ?? "JARVIS sidecar unavailable"}`);
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
	subturnLedgerLines = [];
	subturnCommitNextId = 1;
	resetProviderCallCeilingState();
	resetLockedResourceReportStopState();
	resetRepeatedFailureReportStopState();
	resetToolLessonTurnState();
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
		resetRepeatedFailureReportStopState();
		resetToolLessonTurnState();
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
		subturnLedgerLines = [];
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

const CONSEQUENTIAL_SUBTURN_TOOLS = new Set<string>([
	"write",
	"edit",
	"str_replace",
	"create",
	"apply_patch",
	"register_project",
	"switch_project",
	"unregister_project",
	"delete_project",
	"remove_project",
]);
const DESTRUCTIVE_BASH_RE =
	/\b(rm|rmdir|del|erase|move|mv|mkdir|md|ni|new-item|remove-item|ri|set-content|out-file|add-content|copy-item)\b|workspace_registry|active_project\.json/i;

function isConsequentialSubturnAction(toolName: unknown, descriptor: string, isError: boolean): boolean {
	if (isError) return false;
	const tool = String(toolName ?? "").toLowerCase();
	if (CONSEQUENTIAL_SUBTURN_TOOLS.has(tool)) return true;
	if (tool === "bash" || tool === "shell" || tool === "powershell") {
		return DESTRUCTIVE_BASH_RE.test(descriptor ?? "");
	}
	return false;
}

function appendSubturnLedger(text: string): void {
	const clean = oneLineForSummary(text, 160);
	if (!clean) return;
	subturnLedgerLines.push(clean);
	if (subturnLedgerLines.length > SUBTURN_LEDGER_MAX_ITEMS) {
		subturnLedgerLines.splice(0, subturnLedgerLines.length - SUBTURN_LEDGER_MAX_ITEMS);
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
		completion_ledger: subturnLedgerLines.slice(),
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
		"## Completed Actions (durable ledger - irreversible, already done; do not repeat)",
		...list(state.completion_ledger),
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
	return line.includes("unregister (backstop)") && line.includes("folder deletion observed");
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

function turnTimelineLogTarget(): string | undefined {
	const raw = process.env.JARVIS_TURN_TIMELINE_LOG;
	if (typeof raw === "string") {
		const trimmed = raw.trim();
		if (!trimmed || trimmed === "0" || trimmed.toLowerCase() === "false") return undefined;
		return path.resolve(trimmed);
	}
	return path.join(os.homedir(), ".jarvis-code", "turn_timeline.jsonl");
}

export function writeTurnTimelineEventLine(event: string, data: Record<string, unknown>): void {
	const target = turnTimelineLogTarget();
	if (!target) return;
	try {
		fs.mkdirSync(path.dirname(target), { recursive: true });
		const line = JSON.stringify({
			ts: new Date().toISOString(),
			event,
			...data,
		});
		fs.appendFileSync(target, `${line}\n`, "utf8");
	} catch {
		// Best-effort local timeline; never let telemetry break a user turn.
	}
}

function recordTurnTimelineEvent(event: string, data: Record<string, unknown>): void {
	writeTurnTimelineEventLine(event, {
		turn_key: subturnLogInitializedForUserTurnKey ?? "",
		route: currentRoute,
		mode: currentMode,
		provider_calls: providerCallCountThisTurn,
		origin_window: jarvisOriginWindow() ?? "",
		model_provider: activeModelProviderThisTurn ?? "",
		...data,
	});
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

function resetRepeatedFailureReportStopState(): void {
	subturnFailingCommandCounts = new Map<string, { count: number; stamp: number }>();
	subturnRepairActivityStamp = 0;
	subturnFailingCommandRecordedCallIds = new Map<string, { key: string; count: number; line: string }>();
	subturnRepeatedFailureReportStopActive = false;
	subturnRepeatedFailureReportStopRecord = undefined;
}

// --- repeated-failure brake -----------------------------------------------
// Counts byte-identical failing shell commands within a turn, but only while
// NO repair activity happens in between: any successful non-read-only tool
// call (edit/write/another shell command) advances a stamp that resets the
// counter to 1, so the canonical red -> edit -> rerun loop never accumulates
// no matter how many reruns it takes. Only a verbatim rerun with nothing
// attempted in between (the degenerate loop this brake exists for) counts up.
// Success of the same command clears its key entirely. One developer warning
// at WARN_THRESHOLD; at STOP_THRESHOLD the turn is forced to a report-stop
// and further tool calls are blocked — same machinery as the locked-resource
// brake.
const REPEATED_FAILURE_NON_REPAIR_TOOLS = new Set([
	"read",
	"grep",
	"find",
	"glob",
	"ls",
	"list",
	"recall_turns",
	"search_within",
	"ask_user",
]);
function repeatedFailureCommandKey(toolName: unknown, command: unknown): string | undefined {
	const tool = String(toolName ?? "").toLowerCase();
	if (!TOOL_LESSON_SHELL_TOOLS.has(tool)) return undefined;
	const commandText = typeof command === "string" ? command.replace(/\s+/g, " ").trim() : "";
	if (!commandText) return undefined;
	return `${tool}|${commandText.slice(0, 400)}`;
}

function recordRepeatedFailureToolOutcome(args: {
	toolCallId?: unknown;
	toolName?: unknown;
	command?: unknown;
	isError: boolean;
	outputText: string;
}): void {
	if (!args.isError && !REPEATED_FAILURE_NON_REPAIR_TOOLS.has(String(args.toolName ?? "").toLowerCase())) {
		subturnRepairActivityStamp += 1;
	}
	const key = repeatedFailureCommandKey(args.toolName, args.command);
	if (!key) return;
	if (!args.isError) {
		subturnFailingCommandCounts.delete(key);
		for (const [callId, record] of subturnFailingCommandRecordedCallIds.entries()) {
			if (record.key === key) subturnFailingCommandRecordedCallIds.delete(callId);
		}
		return;
	}
	const callId = typeof args.toolCallId === "string" && args.toolCallId.trim() ? args.toolCallId.trim() : undefined;
	if (callId && subturnFailingCommandRecordedCallIds.has(callId)) return;
	const previous = subturnFailingCommandCounts.get(key);
	const count = previous && previous.stamp === subturnRepairActivityStamp ? previous.count + 1 : 1;
	subturnFailingCommandCounts.set(key, { count, stamp: subturnRepairActivityStamp });
	const line = oneLineForSummary(`${String(args.command ?? "")} => ${args.outputText}`, 260);
	const record = { key, count, line };
	if (callId) subturnFailingCommandRecordedCallIds.set(callId, record);
	if (count === REPEATED_FAILURE_WARN_THRESHOLD && !subturnRepeatedFailureReportStopActive) {
		pendingToolLessonHints.push(buildRepeatedFailureWarningText(record));
		recordSubturnDebugEvent("repeated_failure_warning", {
			tool: args.toolName,
			tool_call_id: args.toolCallId,
			failing_command_key: key,
			attempts: count,
			reason: line,
		});
	}
	if (count >= REPEATED_FAILURE_REPORT_STOP_THRESHOLD && !subturnRepeatedFailureReportStopActive) {
		subturnRepeatedFailureReportStopActive = true;
		subturnRepeatedFailureReportStopRecord = record;
		recordSubturnDebugEvent("repeated_failure_report_stop", {
			tool: args.toolName,
			tool_call_id: args.toolCallId,
			failing_command_key: key,
			attempts: count,
			reason: line,
			content: buildRepeatedFailureReportStopText(),
		});
	}
}

function buildRepeatedFailureWarningText(record: { count: number; line: string }): string {
	return [
		`REPEATED FAILURE: the exact same command has failed ${record.count} times in this turn.`,
		"Do NOT retry it verbatim — fix the command, switch approach, or stop and report what blocks you.",
		`Repeated failure command: ${oneLineForSummary(record.line, 200)}`,
	].join("\n");
}

function buildRepeatedFailureReportStopText(): string {
	const record = subturnRepeatedFailureReportStopRecord;
	const reason = record?.line ? oneLineForSummary(record.line, 260) : "the same command kept failing";
	return [
		`${REPEATED_FAILURE_REPORT_STOP_MARKER}: same command failed ${record?.count ?? REPEATED_FAILURE_REPORT_STOP_THRESHOLD} times - stop retrying and report progress so far plus the cause of failure.`,
		`Repeated failure: ${reason}`,
	].join("\n");
}

function applyRepeatedFailureReportStop(payload: unknown): unknown {
	if (!subturnRepeatedFailureReportStopActive) return payload;
	return forceProviderPayloadToReportStop(payload, buildRepeatedFailureReportStopText());
}

function maybeBlockRepeatedFailureToolCall(toolName: string, input: unknown): ToolBlockResult | undefined {
	if (!subturnRepeatedFailureReportStopActive) return undefined;
	const reason = buildRepeatedFailureReportStopText();
	recordSubturnDebugEvent("repeated_failure_tool_block", {
		tool: toolName,
		descriptor: summarizeToolDescriptor(toolName, input).text,
		content: reason,
	});
	return { block: true, reason, terminate: true };
}

function buildEmptyAssistantTurnLossMarker(reason?: "aborted" | "provider_error", detail?: string): string {
	if (reason === "aborted") {
		return (
			`[JARVIS turn-loss guard] The user aborted this turn after ` +
			`${providerCallCountThisTurn} provider call(s). Tool activity is preserved in tool_events; ` +
			`treat this turn as incomplete.`
		);
	}
	if (reason === "provider_error") {
		return (
			`[JARVIS turn-loss guard] This turn ended on a provider error after ` +
			`${providerCallCountThisTurn} provider call(s): ${detail ?? "unknown error"}. ` +
			`Tool activity is preserved in tool_events; treat this turn as incomplete.`
		);
	}
	const failure = subturnRepeatedFailureReportStopRecord ?? subturnLockedResourceReportStopRecord;
	const failureNote = failure ? ` Last repeated failure: ${oneLineForSummary(failure.line, 200)}` : "";
	return (
		`[JARVIS turn-loss guard] The model ended this turn with empty assistant text after ` +
		`${providerCallCountThisTurn} provider call(s). Tool activity is preserved in tool_events; ` +
		`treat this turn as incomplete.${failureNote}`
	);
}

function lastSuccessfulHandoffToolName(): string | undefined {
	for (let eventIndex = toolEvents.length - 1; eventIndex >= 0; eventIndex--) {
		const results = toolEvents[eventIndex]?.toolResults ?? [];
		for (let resultIndex = results.length - 1; resultIndex >= 0; resultIndex--) {
			const result = results[resultIndex];
			const name = String(result.toolName ?? "").trim();
			if (!name || result.isError) continue;
			if (HANDOFF_BUS_TOOL_NAMES.has(name)) return name;
		}
	}
	return undefined;
}

// Regime-B bridged control tools surface MCP-prefixed names such as
// "mcp__jarvis_control__ultracode"; regime-A pi-native tools are bare.
export function stripMcpToolPrefix(name: string): string {
	return name.startsWith("mcp__") ? (name.split("__").pop() ?? name) : name;
}

// Regime-B agent-sdk renders tool calls as "[agent tool: <name>]" markers inside
// assistant text/thinking blocks. Regime-A renders native "toolCall" content
// blocks. toolEvents only captures the last sub-tool in some regime-B turns, so
// scan the full message history.
const AGENT_SDK_TOOL_MARKER_RE = /\[agent tool:\s*([^\]]+?)\s*\]/gi;

export function turnInvokedBackgroundHandoffTool(messages: AgentMessage[]): string | undefined {
	for (const message of messages) {
		if (String((message as { role?: unknown }).role ?? "") !== "assistant") continue;
		const content = (message as { content?: unknown }).content;
		if (!Array.isArray(content)) continue;
		for (const part of content) {
			if (!part || typeof part !== "object") continue;
			const record = part as Record<string, unknown>;
			const type = String(record.type ?? "");
			if (type === "toolCall") {
				const rawName = String(
					(record as { toolName?: unknown }).toolName ?? (record as { name?: unknown }).name ?? "",
				).trim();
				const name = stripMcpToolPrefix(rawName);
				if (name && BACKGROUND_HANDOFF_TOOL_NAMES.has(name)) return name;
				continue;
			}
			if (!["thinking", "reasoning", "summary", "text", "output_text"].includes(type)) continue;
			for (const key of ["thinking", "reasoning", "summary", "text", "content"]) {
				const value = record[key];
				if (typeof value !== "string") continue;
				AGENT_SDK_TOOL_MARKER_RE.lastIndex = 0;
				for (
					let match = AGENT_SDK_TOOL_MARKER_RE.exec(value);
					match;
					match = AGENT_SDK_TOOL_MARKER_RE.exec(value)
				) {
					const name = stripMcpToolPrefix((match[1] ?? "").trim());
					if (name && BACKGROUND_HANDOFF_TOOL_NAMES.has(name)) return name;
				}
			}
		}
	}
	return undefined;
}

function buildBackgroundHandoffMarker(tool: string): string {
	return (
		`[JARVIS background handoff] ${tool} is running in the background; ` +
		"this window will be re-engaged with the synthesized result when it completes. " +
		"No final assistant prose was needed; tool activity is preserved in tool_events."
	);
}

function directiveHandbackCompletionSummary(): string {
	const tool = lastSuccessfulHandoffToolName() ?? "directive bus";
	const target = activeDirectiveTurn?.from_window
		? directiveWindowLabel(String(activeDirectiveTurn.from_window))
		: "main window";
	const job = activeDirectiveTurn?.job;
	if (job?.job_id) {
		const cycle = job.cycle ? ` c${job.cycle}` : "";
		const phase = job.phase ? ` ${job.phase}` : "";
		const subject = activeSecondEyesReviewTurn ? "Critic Mode review handback" : "Job handback";
		return `${subject} sent to ${target} via ${tool} (${job.job_id}${cycle}${phase}).`;
	}
	const gan = activeDirectiveTurn?.gan;
	if (gan?.gan_id) {
		const round = gan.round ? ` r${gan.round}` : "";
		return `GAN handback sent to ${target} via ${tool} (${gan.gan_id}${round}).`;
	}
	return `Directive handback sent to ${target} via ${tool}.`;
}

function buildDirectiveHandbackCompletionMarker(summary: string): string {
	return (
		`[JARVIS handback complete] ${summary} ` +
		"No final assistant prose was needed; tool activity is preserved in tool_events."
	);
}

function isWorkerToolsRetryPrompt(text: string): boolean {
	return text.trim().startsWith(WORKER_TOOLS_RETRY_MARKER);
}

function isVerifyIncompletePrompt(text: string): boolean {
	return text.trim().startsWith(VERIFY_INCOMPLETE_FOLLOWUP_MARKER);
}

function assistantSignaledWorkerToolsNeeded(text: string): boolean {
	return text.split(/\r?\n/).some((line) => line.trim() === WORKER_TOOLS_NEEDED_MARKER);
}

function originalUserRequestLineFromRetryPrompt(text: string): string | undefined {
	for (const line of text.split(/\r?\n/)) {
		const trimmed = line.trim();
		const prefix = "Original user request:";
		if (trimmed.startsWith(prefix)) {
			const value = trimmed.slice(prefix.length).trim();
			if (value) return value;
		}
	}
	return undefined;
}

function workerToolsRetryOriginalUserRequest(text: string): string | undefined {
	if (!isWorkerToolsRetryPrompt(text)) return undefined;
	return originalUserRequestLineFromRetryPrompt(text);
}

function verifyIncompleteOriginalUserRequest(text: string): string | undefined {
	if (!isVerifyIncompletePrompt(text)) return undefined;
	return originalUserRequestLineFromRetryPrompt(text);
}

function effectiveUserTextFromInternalRetry(text: string): string {
	return workerToolsRetryOriginalUserRequest(text) ?? verifyIncompleteOriginalUserRequest(text) ?? text;
}

// ask_user is a clarification dialog, not an external action: a turn whose only
// tool result is ask_user has not built or changed anything, so the no-action guard
// must stay free to fire. Every other tool (write/edit/bash/read/...) is real work
// that disarms it.
const NO_ACTION_NEUTRAL_TOOL_NAMES = new Set<string>(["ask_user"]);

function turnHadActionToolResult(): boolean {
	return toolEvents.some((event) =>
		(event.toolResults ?? []).some(
			(result) => result.toolName && !NO_ACTION_NEUTRAL_TOOL_NAMES.has(String(result.toolName)),
		),
	);
}

export function turnHasToolActivity(messages: AgentMessage[], assistantText = ""): boolean {
	// A non-clarification tool RESULT anywhere in this user-turn counts as real
	// action — even when it ran on an earlier subturn and the turn closed with a
	// text-only "done" summary (multi-step build: write html -> css -> js -> text).
	// The earlier strict check (lastToolActivityProviderCall === providerCallCount)
	// only credited a tool on the FINAL provider call, so such a real build read as
	// no-action and the no-action guard misfired a spurious retry (live incident
	// 2026-06-22: counter app built real files yet a retry fired).
	if (turnHadActionToolResult()) return true;
	if (assistantText && AGENT_SDK_TOOL_ACTIVITY_RE.test(assistantText)) return true;
	if (providerCallCountThisTurn > 1) return false;
	if (toolEvents.some((event) => (event.toolResults?.length ?? 0) > 0)) return true;
	for (const message of messages) {
		const role = String((message as { role?: unknown }).role ?? "");
		if (role === "tool" || role === "toolResult") return true;
		if (role === "assistant" && assistantMessageHasAgentSdkToolActivity(message)) return true;
		if (role !== "assistant") continue;
		const content = (message as { content?: unknown }).content;
		if (!Array.isArray(content)) continue;
		if (
			content.some(
				(block) => block && typeof block === "object" && (block as { type?: unknown }).type === "toolCall",
			)
		) {
			return true;
		}
	}
	return false;
}

const AGENT_SDK_TOOL_ACTIVITY_RE =
	/^\s*\[(?:read|write|edit|shell|web search|web fetch|glob|grep|list|todo|agent task|ask user|job send|job close|agent tool:[^\]]+)(?:\]|:)/im;

function assistantMessageHasAgentSdkToolActivity(message: AgentMessage): boolean {
	const content = (message as { content?: unknown }).content;
	if (!Array.isArray(content)) return false;
	for (const part of content) {
		if (!part || typeof part !== "object") continue;
		const record = part as Record<string, unknown>;
		const type = String(record.type ?? "");
		if (!["thinking", "reasoning", "summary", "text", "output_text"].includes(type)) continue;
		for (const key of ["thinking", "reasoning", "summary", "text", "content"]) {
			const value = record[key];
			if (typeof value === "string" && AGENT_SDK_TOOL_ACTIVITY_RE.test(value)) return true;
		}
	}
	return false;
}

export function decidePostTurnRecovery(input: PostTurnRecoveryInput): PostTurnRecoveryDecision {
	if (input.workerToolsRetryEligible) return { kind: "worker_tools_followup" };
	const modifiedFileCount = input.modifiedFilePaths?.length ?? 0;
	if (modifiedFileCount === 0) return { kind: "none" };
	if (input.verificationRanThisTurn === true) return { kind: "none" };
	if (!input.route || !isProjectRoute(input.route)) return { kind: "none" };
	if (isSidecarChatProxyProvider(input.provider)) return { kind: "none" };
	const maxContinuations = Math.max(0, input.maxVerifyContinuations ?? MAX_VERIFY_CONTINUATIONS);
	const continuationCount = Math.max(0, input.verifyContinuationCount ?? 0);
	if (continuationCount >= maxContinuations) return { kind: "none" };
	return { kind: "verify_incomplete" };
}

function inferAssistantTerminalReason(
	assistantMessage: AssistantMessage | undefined,
	assistantText: string,
): JarvisTurnTerminalReason {
	const stopReason = String(assistantMessage?.stopReason ?? "");
	if (stopReason === "error") return "error";
	if (stopReason === "aborted") return "aborted";
	if (stopReason === "tool_calls") return "tool_calls";
	if (assistantMessage && assistantToolNames(assistantMessage).length > 0) return "tool_calls";
	if (!assistantText.trim()) return "empty";
	return "stop";
}

function workerToolsWereAvailableInLastProviderCall(): boolean {
	const tools = new Set(lastProviderToolsAfterFilter);
	// Only the ACTION tools count: spawn_window (new worker) or job_send (existing
	// worker). list_windows is read-only/informational and now rides every chat
	// route as a capability tool, so it must NOT mark worker tools as "available" —
	// doing so would suppress the worker-tools retry on plain chat routes.
	return tools.has("spawn_window") || tools.has("job_send");
}

function shouldQueueWorkerToolsRetry(messages: AgentMessage[], assistantText: string): boolean {
	if (!lastUserMessage.trim()) return false;
	if (providerCallCountThisTurn <= 0) return false;
	if (workerToolsRetryInFlight || isWorkerToolsRetryPrompt(lastUserMessage)) return false;
	if (!assistantSignaledWorkerToolsNeeded(assistantText)) return false;
	if (turnHasToolActivity(messages)) return false;
	if (workerToolsWereAvailableInLastProviderCall()) return false;
	return true;
}

function buildWorkerToolsRetryPrompt(userText: string, assistantText: string): string {
	void assistantText;
	return [
		WORKER_TOOLS_RETRY_MARKER,
		`The previous assistant signaled ${WORKER_TOOLS_NEEDED_MARKER}.`,
		"Worker/window tools are now enabled for this follow-up.",
		"Start with [MODE:CHAT], then call the appropriate tool now:",
		"- new worker/agent window: call spawn_window",
		"- existing worker/window: call list_windows if needed, then job_send",
		"Do not repeat the signal. Do not answer with another promise.",
		`Original user request: ${oneLineForSummary(userText, 400)}`,
	].join("\n");
}

export function looksLikeVerification(command: string): boolean {
	const clean = command.replace(/\r\n/g, "\n").trim();
	if (!clean) return false;
	return VERIFICATION_COMMAND_PATTERNS.some((pattern) => pattern.test(clean));
}

function verificationGateMutationPaths(mutations: JarvisTurnFileMutation[]): string[] {
	const roots = [
		activeCodePath,
		activeProjectPath,
		lastContextResponse?.code_path,
		lastContextResponse?.active_project_path,
	];
	const paths = mutations.map((mutation) => formatPathForHarness(mutation.path, roots));
	return [...new Set(paths)];
}

function buildVerificationIncompletePrompt(userText: string, mutations: JarvisTurnFileMutation[]): string {
	const files = formatHarnessList(verificationGateMutationPaths(mutations), 8);
	return [
		VERIFY_INCOMPLETE_FOLLOWUP_MARKER,
		`[Verification floor] You modified files this turn (${files}) but no verification command (build / test / run) was observed.`,
		"A coding turn is NOT complete until verification has actually run.",
		"Run the build/tests, or run the program to confirm your change works, then report the result.",
		"If verification genuinely does not apply here, state explicitly why - then you may finish.",
		`Original user request: ${oneLineForSummary(userText, 400)}`,
	].join("\n");
}

// --- tool lessons: machine-global memory of repeated tool failures -------
// Nothing rides the prompt up front; the sidecar answers a failure observe
// with a short advisory hint only when the same command shape has already
// failed before, and that hint lands as one developer line right after the
// failed tool result. Success right after a failure in the same turn is
// paired server-side as the lesson's working alternative.

const TOOL_LESSON_SHELL_TOOLS = new Set(["bash", "shell", "powershell", "pwsh"]);
let pendingToolLessonHints: string[] = [];
let pendingToolLessonObserves: Array<Promise<void>> = [];
let toolLessonTurnSeq = 0;

function resetToolLessonTurnState(): void {
	toolLessonTurnSeq += 1;
	pendingToolLessonHints = [];
	pendingToolLessonObserves = [];
}

function maybeObserveToolLesson(toolName: unknown, command: unknown, isError: boolean, outputText: string): void {
	const tool = String(toolName ?? "").toLowerCase();
	if (!TOOL_LESSON_SHELL_TOOLS.has(tool)) return;
	const commandText = typeof command === "string" ? command.trim() : "";
	if (!commandText) return;
	const observe = (async () => {
		try {
			const data = await postSidecar<{ ok?: boolean; hint?: string }>("/tool_lesson/observe", {
				tool,
				command: commandText,
				is_error: isError,
				output_head: outputText ? outputText.slice(0, 600) : "",
				turn_id: String(toolLessonTurnSeq),
			});
			const hint = typeof data?.hint === "string" ? data.hint.trim() : "";
			if (hint) pendingToolLessonHints.push(hint);
		} catch {
			/* lessons are advisory; sidecar hiccups must never break tool flow */
		}
	})();
	pendingToolLessonObserves.push(observe);
}

async function awaitPendingToolLessonObserves(): Promise<void> {
	if (!pendingToolLessonObserves.length) return;
	const waits = pendingToolLessonObserves.splice(0, pendingToolLessonObserves.length);
	await Promise.allSettled(waits);
}

function applyToolLessonHints(payload: unknown): unknown {
	if (!pendingToolLessonHints.length) return payload;
	const text = pendingToolLessonHints.splice(0, pendingToolLessonHints.length).join("\n");
	const hintMessage = { role: "developer", content: text };
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	if (messages) return { ...record, messages: [...messages, hintMessage] };
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	if (input) return { ...record, input: [...input, hintMessage] };
	return payload;
}

function maybeBlockDirectiveSpawnToolCall(toolName: string): ToolBlockResult | undefined {
	if (toolName !== "spawn_window" || !activeDirectiveTurn) return undefined;
	// M7.9 narrow exception to the M7.8 re-delegation block: a checkpoint turn
	// whose arriving feature is ESCALATED must be able to spawn the stronger
	// worker. Everything else stays blocked exactly as before.
	if (activeMapCheckpointTurn && mapCheckpointHasEscalatedFeature(activeMapCheckpointTurn)) return undefined;
	return { block: true, reason: DIRECTIVE_SPAWN_BLOCK_REASON };
}

function maybeBlockSecondEyesInitialSpawnToolCall(toolName: string, input: unknown): ToolBlockResult | undefined {
	if (toolName !== "spawn_window" || !secondEyesReviewSpawnRequired()) return undefined;
	const record = input && typeof input === "object" ? (input as Record<string, unknown>) : {};
	const rawInitialDirective = typeof record.initial_directive === "string" ? record.initial_directive.trim() : "";
	if (!rawInitialDirective || !secondEyesPlanReady(rawInitialDirective)) {
		return { block: true, reason: secondEyesPlanReadyError() };
	}
	return undefined;
}

function maybeBlockExistingWorkerRequestSpawnToolCall(toolName: string): ToolBlockResult | undefined {
	if (toolName !== "spawn_window") return undefined;
	if (secondEyesReviewSpawnRequired()) return undefined;
	const action = String(lastRouteClassifierDecision?.expected_action ?? "")
		.trim()
		.toLowerCase();
	if (currentRoute !== "chat_control" || action !== "tool") return undefined;
	return {
		block: true,
		reason:
			"Existing worker/window request: call list_windows, then job_send to the named existing worker/window. Do not spawn a duplicate. If the target is ambiguous, call ask_user.",
	};
}

function maybeBlockExistingWorkerRequestPassiveDirectiveToolCall(toolName: string): ToolBlockResult | undefined {
	if (toolName !== "send_directive") return undefined;
	const action = String(lastRouteClassifierDecision?.expected_action ?? "")
		.trim()
		.toLowerCase();
	if (currentRoute !== "chat_control" || action !== "tool") return undefined;
	return {
		block: true,
		reason:
			"Existing worker/window request: use job_send so the target can hand back and wake this window. send_directive is passive/legacy only.",
	};
}

// M7.9 checkpoint turns: verify + dispatch only. Implementation tools are
// hard-blocked unless the escalate ladder reached main_direct for an arriving
// feature (then the orchestrator implements it here itself).
const MAP_CHECKPOINT_TOOL_BLOCK_NAMES = new Set(["edit", "write", "write_file"]);
const SECOND_EYES_MAIN_BLOCKED_TOOL_NAMES = new Set(["spawn_window", "map_create", "gan_send", "gan_close"]);

function maybeBlockMapCheckpointToolCall(toolName: string): ToolBlockResult | undefined {
	const checkpoint = activeMapCheckpointTurn;
	if (!checkpoint || !MAP_CHECKPOINT_TOOL_BLOCK_NAMES.has(toolName)) return undefined;
	if (mapCheckpointRestrictionsLifted(checkpoint)) return undefined;
	return { block: true, reason: MAP_CHECKPOINT_EDIT_BLOCK_REASON };
}

function maybeBlockSecondEyesToolCall(toolName: string): ToolBlockResult | undefined {
	if (!activeSecondEyesReviewTurn || SECOND_EYES_ALLOWED_TOOL_NAMES.has(toolName)) return undefined;
	return {
		block: true,
		reason:
			"Critic Mode is review-only mode: inspect files and run bounded verification only; do not mutate files, update memory, spawn workers, or create maps; hand findings back with job_send/send_directive.",
	};
}

function maybeBlockSecondEyesReviewBashToolCall(toolName: string, input: unknown): ToolBlockResult | undefined {
	if (!activeSecondEyesReviewTurn || !TOOL_LESSON_SHELL_TOOLS.has(toolName)) return undefined;
	const command = extractToolCommand(input);
	if (!command) return undefined;
	const normalized = command.replace(/[`^]/g, "").toLowerCase();
	const mutatingPackageManager =
		/\b(?:npm|pnpm|yarn)\s+(?:install|i|add|remove|rm|update|upgrade|audit\s+fix|dedupe|link|unlink)\b/.test(
			normalized,
		);
	const mutatingGit = /\bgit\s+(?:commit|push|reset|clean|checkout|switch|merge|rebase|apply|am|stash|restore)\b/.test(
		normalized,
	);
	if (
		mutatingPackageManager ||
		mutatingGit ||
		describeDestructiveShellCommand(command) ||
		commandLooksMutating(command)
	) {
		return {
			block: true,
			reason:
				"Critic Mode reviewer may run read-only diagnostics/tests, but must not run install/update, git mutation, delete/move/copy, redirect-write, or other mutating shell commands.",
		};
	}
	return undefined;
}

function maybeBlockSecondEyesMainToolCall(toolName: string): ToolBlockResult | undefined {
	if (!activeSecondEyesMainTurn || !SECOND_EYES_MAIN_BLOCKED_TOOL_NAMES.has(toolName)) return undefined;
	return {
		block: true,
		reason:
			"Critic Mode main uses the existing review worker only: implement or fix here, then use job_send/job_close; do not spawn extra workers, create maps, or start GAN rounds.",
	};
}

function maybeBlockToolAfterAskUser(toolName: string): ToolBlockResult | undefined {
	if (toolName === "ask_user") {
		askUserIssuedThisProviderCall = true;
		return undefined;
	}
	if (!askUserIssuedThisProviderCall) return undefined;
	return {
		block: true,
		reason:
			"ask_user choices are pending: finish ask_user first, then continue with tools on the next model response using those answers. Do not run read/search/spawn/write or other tools in the same response.",
	};
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

function commandStartsWithReadOnlyJarvisScriptAccess(command: string): boolean {
	const normalized = command.replace(/[`^]/g, "").trim().toLowerCase();
	return /^(?:get-content|gc|cat|type|rg|grep|select-string|findstr|ls|dir)\b[\s\S]*\bjarvis\.ps1\b/.test(normalized);
}

function maybeBlockJarvisLauncherToolCall(toolName: string, input: unknown): ToolBlockResult | undefined {
	const tool = String(toolName ?? "").toLowerCase();
	if (tool !== "bash" && tool !== "shell" && tool !== "powershell" && tool !== "pwsh") return undefined;
	const command = extractToolCommand(input);
	if (!command || !/\bjarvis\.ps1\b/i.test(command)) return undefined;
	if (commandStartsWithReadOnlyJarvisScriptAccess(command)) return undefined;
	const normalized = command.replace(/[`^]/g, "").toLowerCase();
	const looksExecutable =
		/(^|[;&|]\s*)(&\s*)?(?:["']?(?:[a-z]:[\\/]|\.{1,2}[\\/])[^"';&|\r\n]*["']?\s*)?jarvis\.ps1\b/i.test(normalized) ||
		/\b(?:powershell(?:\.exe)?|pwsh(?:\.exe)?|start-process)\b[\s\S]*\bjarvis\.ps1\b/i.test(normalized) ||
		/\b-file\s+["']?[^"';&|\r\n]*jarvis\.ps1\b/i.test(normalized);
	if (!looksExecutable) return undefined;
	return {
		block: true,
		reason:
			"Blocked shell launch of jarvis.ps1; use spawn_window instead so JARVIS prompt and directive-bus wiring stay intact.",
	};
}

function buildLockedResourceReportStopText(): string {
	const record = subturnLockedResourceReportStopRecord;
	const reason = record?.line
		? oneLineForSummary(record.line, 260)
		: "the resource is still locked by another process";
	return [LOCKED_RESOURCE_REPORT_STOP_TEXT, `Repeated failure: ${reason}`].join("\n");
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

function recordSubturnDebugEvent(kind: string, data: Record<string, unknown>): void {
	appendSubturnEvent(kind, data);
	void postSidecar("/debug/subturn/observe", {
		source: "jlc",
		event: kind,
		user_turn_key: subturnLogInitializedForUserTurnKey,
		data,
	});
}

function recordLockedResourceDebugEvent(kind: string, data: Record<string, unknown>): void {
	recordSubturnDebugEvent(kind, data);
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

function activeJobTurnStopHint(): string | undefined {
	const jobId = activeDirectiveTurn?.job?.job_id;
	const counterpart = String(activeDirectiveTurn?.from_window ?? "").trim();
	if (!jobId || !counterpart || counterpart === "external") return undefined;
	return `This is a job turn: summarize completed/remaining work and hand it back with job_send(job_id="${jobId}", to_window="${counterpart}") before stopping.`;
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
	const jobHint = activeJobTurnStopHint();
	return [
		`${PC_CEILING_REPORT_STOP_MARKER}: INCOMPLETE - provider-call ceiling reached (${providerCall}/${ceiling}).`,
		`Completed so far: ${completed || "recorded progress is in the JARVIS subturn state"}.`,
		`Blocked/remaining: ${blocker}.`,
		...(jobHint ? [jobHint] : []),
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

function appendDeveloperInstruction(payload: unknown, text: string): unknown {
	const message = { role: "developer", content: text };
	if (!payload || typeof payload !== "object") return { messages: [message] };
	const record = payload as Record<string, unknown>;
	const messages = Array.isArray(record.messages) ? (record.messages as Array<Record<string, unknown>>) : undefined;
	if (messages) return { ...record, messages: [...messages, message] };
	const input = Array.isArray(record.input) ? (record.input as Array<Record<string, unknown>>) : undefined;
	if (input) return { ...record, input: [...input, message] };
	const instructions = typeof record.instructions === "string" ? record.instructions : undefined;
	if (instructions !== undefined) return { ...record, instructions: `${instructions}\n\n${text}` };
	return { ...record, messages: [message] };
}

function workerWindowContextWanted(): boolean {
	if (workerToolsRetryInFlight) return true;
	if (currentRoute === "chat_control") return true;
	const action = String(lastRouteClassifierDecision?.expected_action ?? "")
		.trim()
		.toLowerCase();
	return action === "spawn_window" || action === "tool";
}

function workerWindowContextLine(window: SidecarDirectiveWindow): string {
	const name = displayWindowName(window.pair8, window.label);
	const live = window.alive === false ? "stale" : "alive";
	const fields = [
		`${name} ${live}`,
		window.role ? `role=${window.role}` : "",
		window.status ? `status=${window.status}` : "",
		window.contract ? `contract=${window.contract}` : "",
		window.stage ? `stage=${window.stage}` : "",
		window.active_job_id ? `job=${window.active_job_id}` : "",
		window.counterpart_window ? `counterpart=${window.counterpart_window}` : "",
	].filter(Boolean);
	return `- ${fields.join(" ")}`;
}

function buildWorkerWindowContextText(windows: SidecarDirectiveWindow[]): string | undefined {
	const relevant = windows
		.filter((window) => !window.current)
		.sort((a, b) => {
			const aLive = a.alive === false ? 1 : 0;
			const bLive = b.alive === false ? 1 : 0;
			return aLive - bLive || String(a.label ?? a.pair8 ?? "").localeCompare(String(b.label ?? b.pair8 ?? ""));
		})
		.slice(0, 12);
	if (relevant.length === 0) {
		return [
			"[JARVIS WORKER WINDOW SNAPSHOT]",
			"No existing worker windows are listed. For a worker request, spawn_window is appropriate after model selection if needed.",
		].join("\n");
	}
	return [
		"[JARVIS WORKER WINDOW SNAPSHOT]",
		...relevant.map(workerWindowContextLine),
		"",
		"Worker routing rules:",
		"- If the user names one of these existing workers/windows, use job_send to that label/pair8; do not spawn a duplicate.",
		"- If no worker target is named and several live workers exist, ask_user which worker or whether to spawn a new one.",
		"- If no live worker exists, spawn_window is appropriate after model selection if needed.",
		"- A worker with contract=critic or role=critic is review-only. Never ask it to implement, fix, edit, patch, write files, or run mutation work; use a builder worker/main window or ask_user.",
	].join("\n");
}

async function applyWorkerWindowContext(payload: unknown): Promise<unknown> {
	if (workerWindowContextInjectedThisTurn || !workerWindowContextWanted()) return payload;
	workerWindowContextInjectedThisTurn = true;
	try {
		const data = await postSidecar<SidecarDirectiveWindowsResponse>("/directives/windows", undefined, "GET", 3000);
		const windows = data && isOkSidecarResponse(data) && Array.isArray(data.windows) ? data.windows : [];
		const text = buildWorkerWindowContextText(windows);
		return text ? appendDeveloperInstruction(payload, text) : payload;
	} catch {
		return payload;
	}
}

function buildSecondEyesReminderText(repeated: boolean, agentSdkProxy: boolean): string {
	const handoff = agentSdkProxy
		? [
				`Once choices and the main-window plan draft are settled, send a concise plan-critique directive containing ${SECOND_EYES_PLAN_READY_MARKER} before implementation or any final user report: use job_send for a named existing worker, or spawn_window with job=true only when no worker target exists.`,
				"In the Agent SDK provider path, spawn_window is bridged through the JARVIS control bridge; the owning Pi harness performs the actual window side effect.",
			]
		: [
				`Once choices and the main-window plan draft are settled, send a concise plan-critique directive containing ${SECOND_EYES_PLAN_READY_MARKER} before implementation or any final user report: use job_send for a named existing worker, or spawn_window with job=true only when no worker target exists.`,
			];
	return [
		`${SECOND_EYES_REMINDER_MARKER}: the user requested Critic Mode, but the review-only worker has not been spawned yet.`,
		repeated ? "This is a repeated reminder because the initial Critic Mode worker is still missing." : "",
		"If user-facing choices are unresolved, call ask_user first and stop; do not call any other tool in the same response.",
		...handoff,
		`The directive must include ${SECOND_EYES_PLAN_READY_MARKER}, the project path, user choices/Q&A if any, and the main-window plan draft. Without that marker, do not spawn the worker.`,
		"The worker reviews the main draft only; do not ask it to perform recon, choose architecture, or invent the plan.",
		"The worker is review-only. It may inspect and run bounded verification; later it reviews the implemented artifact. The main window applies all fixes.",
	]
		.filter(Boolean)
		.join("\n");
}

function applySecondEyesReminder(payload: unknown): unknown {
	if (!secondEyesReviewSpawnRequired()) return payload;
	const repeated = secondEyesReminderInjectedThisTurn;
	secondEyesReminderInjectedThisTurn = true;
	return appendDeveloperInstruction(
		payload,
		buildSecondEyesReminderText(repeated, isSidecarChatProxyProvider(activeModelProviderThisTurn)),
	);
}

function currentSecondEyesProviderPhase(): SecondEyesProviderPhase | undefined {
	if (secondEyesReviewSpawnRequired()) return "plan_draft";
	if (activeSecondEyesReviewTurn) return "review";
	if (activeSecondEyesMainTurn) return "implement";
	return undefined;
}

function applySecondEyesProviderPhase(payload: unknown): unknown {
	const phase = currentSecondEyesProviderPhase();
	if (!phase || !isSidecarChatProxyProvider(activeModelProviderThisTurn)) return payload;
	if (!payload || typeof payload !== "object") return payload;
	return { ...(payload as Record<string, unknown>), jarvis_critic_phase: phase, jarvis_second_eyes_phase: phase };
}

function secondEyesReviewSpawnRequired(): boolean {
	return (
		secondEyesRequestedThisTurn &&
		!activeDirectiveTurn &&
		!activeSecondEyesReviewTurn &&
		!activeSecondEyesMainTurn &&
		!secondEyesReviewSpawnedThisTurn
	);
}

async function maybeHarnessSpawnSecondEyesReview(args: {
	assistantText: string;
	terminalReason: JarvisTurnTerminalReason;
	ctx: ExtensionContext;
	pi: ExtensionAPI;
}): Promise<boolean> {
	if (!secondEyesReviewSpawnRequired()) return false;
	if (args.terminalReason !== "stop" && args.terminalReason !== "empty") return false;
	const planText = args.assistantText.trim();
	if (!planText) return false;
	if (!secondEyesPlanReady(planText)) {
		const event = {
			ok: false,
			error: "missing_plan_ready_marker",
			terminal_reason: args.terminalReason,
			plan_chars: planText.length,
		};
		recordSubturnDebugEvent("harness_second_eyes_spawn_skipped", event);
		appendSubturnEvent("harness_second_eyes_spawn_skipped", event);
		return false;
	}
	const workerModel = await chooseWorkerModelForSpawn(undefined, args.ctx, {
		recommendedSpec: CRITIC_WORKER_MODEL_RECOMMENDED_SPEC,
	});
	const details = await performWorkerSpawn({
		initialDirective: buildSecondEyesReviewDirective(planText),
		model: workerModel,
		job: true,
		isSecondEyesReviewSpawn: true,
		skipModelAsk: true,
		ctx: args.ctx,
	});
	const ok = details.ok === true && details.directive !== undefined && details.directive !== null;
	const event = {
		ok,
		error: details.error,
		pair8: details.pair8 ?? details.window?.pair8,
		directive_id: details.directive?.id,
		job_id: details.directive?.job?.job_id,
		model: workerModel,
		terminal_reason: args.terminalReason,
		plan_chars: planText.length,
	};
	recordSubturnDebugEvent("harness_second_eyes_spawn", event);
	appendSubturnEvent("harness_second_eyes_spawn", event);
	if (ok) {
		sendJarvisChatNotice(args.pi, "[Critic Mode] review worker spawned from the main plan draft.");
		try {
			setWorkStatus(args.ctx, "Critic Mode plan review");
		} catch {
			/* stale */
		}
		return true;
	}
	if (details.error) {
		sendJarvisChatNotice(args.pi, `Critic Mode review worker spawn failed: ${details.error}`);
	}
	return false;
}

function providerPayloadToolNames(payload: unknown): string[] {
	if (!payload || typeof payload !== "object") return [];
	const tools = Array.isArray((payload as Record<string, unknown>).tools)
		? ((payload as Record<string, unknown>).tools as unknown[])
		: [];
	return tools.map((tool) => providerToolSchemaName(tool)).filter((name): name is string => !!name);
}

function chatRouteToolDietWouldApply(): boolean {
	return (
		(currentRoute === "chat" || currentRoute === "chat_control") &&
		!activeMapCheckpointTurn &&
		!activeMapSynthesisTurn &&
		!activeEndGateTurn &&
		!activeSecondEyesReviewTurn &&
		!activeSecondEyesMainTurn
	);
}

function leanChatToolsEnabled(): boolean {
	const value = String(process.env.JARVIS_LEAN_CHAT_TOOLS ?? "")
		.trim()
		.toLowerCase();
	return value === "1" || value === "true" || value === "yes" || value === "on";
}

export function deferredToolsEnabled(): boolean {
	const value = String(process.env.JARVIS_DEFERRED_TOOLS ?? "")
		.trim()
		.toLowerCase();
	return value === "1" || value === "true" || value === "yes" || value === "on";
}

/**
 * JLC tools that must always be active (never deferred). These are either
 * high-frequency coding tools (pi builtins) or structurally required for
 * the interactive flow (ask_user modal, todo tracking, subagent orchestration).
 * tool_search and load_tool are added dynamically when deferred mode is ON.
 */
export const DEFERRED_TOOLS_ALWAYS_ACTIVE = new Set<string>([
	// pi builtins — every coding turn
	"read",
	"bash",
	"edit",
	"write",
	"grep",
	"find",
	"ls",
	// jlc core — structurally required
	"todo",
	"ask_user",
	"delegate_subagent",
	"ultracode",
	"recall_turns",
	// diet tools themselves (added when deferred mode ON)
	"tool_search",
	"load_tool",
]);

/**
 * Narrow the active tool set to the always-active core. Deferred tools remain
 * registered (full schema in the registry) and can be promoted via load_tool.
 * MUST run at runtime (session_start), never during extension loading —
 * getAllTools/setActiveTools are action methods that throw if the runtime is
 * not yet bound ("Extension runtime not initialized. Action methods cannot be
 * called during extension loading.").
 */
export function applyDeferredToolsDiet(pi: ExtensionAPI): void {
	const narrowed = pi
		.getAllTools()
		.map((t) => t.name)
		.filter((name) => DEFERRED_TOOLS_ALWAYS_ACTIVE.has(name));
	pi.setActiveTools(narrowed);
}

// The control/clarification pocket each chat-family route exposes for its JLC
// orchestration tools (ask_user, worker/window, model settings, web). pi's basic
// coding tools are layered on top of this by chatRouteAllowedToolNames.
function chatRouteControlAllowedToolNames(): ReadonlySet<string> {
	if (currentRoute === "chat_control") return CHAT_CONTROL_ALLOWED_TOOL_NAMES;
	if (workerToolsRetryInFlight) return CHAT_ROUTE_TOOL_ACTION_ALLOWED_TOOL_NAMES;
	const action = String(lastRouteClassifierDecision?.expected_action ?? "none")
		.trim()
		.toLowerCase();
	if (action === "spawn_window") return CHAT_ROUTE_SPAWN_ALLOWED_TOOL_NAMES;
	if (action === "tool") return CHAT_ROUTE_TOOL_ACTION_ALLOWED_TOOL_NAMES;
	return CHAT_ROUTE_BASE_ALLOWED_TOOL_NAMES;
}

function chatRouteAllowedToolNames(): ReadonlySet<string> {
	const control = chatRouteControlAllowedToolNames();
	// Default: pi's native coding tools AND the base capability set stay available
	// in chat so the model can actually act on "do X for me" — including registry
	// management, recall, docs, web, and bounded process work. Only the opt-in lean
	// diet (JARVIS_LEAN_CHAT_TOOLS=1) strips back to the control/clarification
	// pocket for weak local models. Heavy multi-window orchestration is NOT here;
	// it stays scoped to deepdive+ via the normal-route blocklist.
	if (leanChatToolsEnabled()) return control;
	return new Set<string>([...control, ...PI_BASIC_TOOL_NAMES, ...CHAT_ROUTE_CAPABILITY_TOOL_NAMES]);
}

function filterChatRouteOnlyTools(payload: unknown): unknown {
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	if (!tools?.length) return payload;
	if (secondEyesReviewSpawnRequired()) {
		const filtered = tools.filter((tool) => {
			const name = providerToolSchemaName(tool);
			return (
				name === "ask_user" ||
				name === "list_windows" ||
				name === "job_send" ||
				name === "spawn_window" ||
				name === "delegate_subagent" ||
				name === "ultracode"
			);
		});
		return filtered.length === tools.length ? payload : { ...record, tools: filtered };
	}
	if (chatRouteToolDietWouldApply()) {
		const allowedToolNames = chatRouteAllowedToolNames();
		const filtered = tools.filter((tool) => {
			const name = providerToolSchemaName(tool);
			return !!name && allowedToolNames.has(name);
		});
		return filtered.length === tools.length ? payload : { ...record, tools: filtered };
	}
	const normalRouteToolDiet =
		!activeMapCheckpointTurn &&
		!activeMapSynthesisTurn &&
		!activeEndGateTurn &&
		!activeSecondEyesReviewTurn &&
		!activeSecondEyesMainTurn;
	const filtered = tools.filter((tool) => {
		const name = providerToolSchemaName(tool);
		if (!name) return true;
		if (CHAT_ROUTE_ONLY_TOOL_NAMES.has(name)) return false;
		if (normalRouteToolDiet && currentRoute === "unregistered_coding") {
			if (UNREGISTERED_PROJECT_MEMORY_TOOL_NAMES.has(name)) return false;
			if (!activeDirectiveTurn && DELEGATION_INITIATE_TOOL_NAMES.has(name)) return false;
		}
		if (
			normalRouteToolDiet &&
			currentRoute === "deepdive" &&
			!activeDirectiveTurn &&
			DELEGATION_INITIATE_TOOL_NAMES.has(name)
		) {
			return false;
		}
		return true;
	});
	return filtered.length === tools.length ? payload : { ...record, tools: filtered };
}

// M7.9 checkpoint tool allowlist: evidence gathering + verdict + dispatch.
// Allowlist (not blocklist) so future tools default to hidden in checkpoint
// turns — the verify-only invariant survives tool growth.
const MAP_CHECKPOINT_ALLOWED_TOOL_NAMES = new Set([
	// verification evidence
	"read",
	"ls",
	"grep",
	"find",
	"bash",
	"managed_process",
	// verdict + map bookkeeping (feature_verdict lands with the ladder slice)
	"feature_verdict",
	"map_create",
	// dispatch / bus
	"job_send",
	"job_close",
	"send_directive",
	"spawn_window",
	"list_windows",
	// recall + retrieval + per-feature memory persistence
	"recall_turns",
	"retrieve_output",
	"update_jarvis_md",
]);

function filterMapCheckpointTools(payload: unknown): unknown {
	const checkpoint = activeMapCheckpointTurn;
	if (!checkpoint || mapCheckpointRestrictionsLifted(checkpoint)) return payload;
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	if (!tools?.length) return payload;
	const filtered = tools.filter((tool) => {
		const name = providerToolSchemaName(tool);
		return !name || MAP_CHECKPOINT_ALLOWED_TOOL_NAMES.has(name);
	});
	return filtered.length === tools.length ? payload : { ...record, tools: filtered };
}

function filterSecondEyesTools(payload: unknown): unknown {
	if (!activeSecondEyesReviewTurn) return payload;
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	if (!tools?.length) return payload;
	const filtered = tools.filter((tool) => {
		const name = providerToolSchemaName(tool);
		return !name || SECOND_EYES_ALLOWED_TOOL_NAMES.has(name);
	});
	return filtered.length === tools.length ? payload : { ...record, tools: filtered };
}

function filterSecondEyesMainTools(payload: unknown): unknown {
	if (!activeSecondEyesMainTurn) return payload;
	if (!payload || typeof payload !== "object") return payload;
	const record = payload as Record<string, unknown>;
	const tools = Array.isArray(record.tools) ? record.tools : undefined;
	if (!tools?.length) return payload;
	const filtered = tools.filter((tool) => {
		const name = providerToolSchemaName(tool);
		return !name || !SECOND_EYES_MAIN_BLOCKED_TOOL_NAMES.has(name);
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
		`Original text retrieval: retrieve_output(ref="${evidence.ref}") - before editing this file, you must retrieve and inspect the original text.`,
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
		`Original text retrieval: retrieve_output(ref="${evidence.ref}") - before editing this file, you must retrieve and inspect the original text.`,
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

// The /spawn fetch must outlive the sidecar's runtime-file poll (150s default)
// so a slow child boot is reported by the server's honest 504 detail instead
// of a generic abort here.
const SPAWN_FETCH_TIMEOUT_MS = 180_000;
const SUBAGENT_DELEGATE_FETCH_TIMEOUT_MS = 360_000;
const SUBAGENT_DELEGATE_IDLE_TIMEOUT_MS = 120_000;
const ORCHESTRATE_FETCH_TIMEOUT_MS = SUBAGENT_DELEGATE_FETCH_TIMEOUT_MS;
const ORCHESTRATE_IDLE_TIMEOUT_MS = SUBAGENT_DELEGATE_IDLE_TIMEOUT_MS;
const SUBAGENT_STREAM_MAX_LINES = 12;
const SUBAGENT_STREAM_MAX_TEXT_CHARS = 3000;

const SIDECAR_BOOTING_SPAWN_ERROR =
	"JARVIS sidecar is not accepting connections yet — right after a window boots, heavy imports can keep the port unbound for 1-2 minutes. " +
	"Wait a moment and call spawn_window again; do not launch jarvis.ps1 manually and do not assume the spawn failed permanently.";

function spawnBootRetryEnvMs(name: string, fallback: number, min: number): number {
	const raw = process.env[name]?.trim();
	if (!raw) return fallback;
	const value = Number(raw);
	return Number.isFinite(value) && value >= min ? value : fallback;
}

// postSidecar returns undefined only when no sidecar accepted the connection
// at all. Right after a window boots, its sidecar imports torch/
// sentence-transformers for 1-2 minutes before the port binds (live evidence
// 2026-06-11: ~75-100s) — retry through that window instead of misreporting
// "unavailable". Server-side errors (HTTP 4xx/5xx) come back as objects and
// are NOT retried.
async function postSpawnWithBootRetry(body: Record<string, unknown>): Promise<SidecarSpawnWindowResponse | undefined> {
	const budgetMs = spawnBootRetryEnvMs("JARVIS_SPAWN_BOOT_RETRY_MS", 90_000, 0);
	const intervalMs = spawnBootRetryEnvMs("JARVIS_SPAWN_BOOT_RETRY_INTERVAL_MS", 5_000, 1);
	const deadline = Date.now() + budgetMs;
	for (;;) {
		const data = await postSidecar<SidecarSpawnWindowResponse>("/spawn", body, "POST", SPAWN_FETCH_TIMEOUT_MS);
		if (data !== undefined) return data;
		if (Date.now() >= deadline) return undefined;
		await new Promise((resolve) => setTimeout(resolve, intervalMs));
	}
}

function appendSubagentTail(existing: string | undefined, chunk: string): string {
	const combined = `${existing ?? ""}${chunk}`;
	return combined.length > SUBAGENT_STREAM_MAX_TEXT_CHARS
		? combined.slice(combined.length - SUBAGENT_STREAM_MAX_TEXT_CHARS)
		: combined;
}

function pushSubagentActivity(details: SidecarSubagentProgressDetails, line: string): void {
	const trimmed = line.trim();
	if (!trimmed) return;
	details.activity.push(trimmed);
	while (details.activity.length > SUBAGENT_STREAM_MAX_LINES) details.activity.shift();
}

function renderSubagentProgressText(
	details: SidecarSubagentProgressDetails,
	maxActivityLines = SUBAGENT_STREAM_MAX_LINES,
): string {
	const lines: string[] = [];
	const label = details.subagent ?? "subagent";
	const suffix = details.sub_id ? ` ${details.sub_id}` : "";
	lines.push(`Running ${label}${suffix}...`);
	if (details.reasoning_tail) {
		const text = details.reasoning_tail.trim();
		if (text) lines.push("", "[reasoning]", text);
	}
	if (details.content_tail) {
		const text = details.content_tail.trim();
		if (text) lines.push("", "[content]", text);
	}
	const activity = details.activity.slice(-Math.max(1, maxActivityLines));
	if (activity.length > 0) {
		lines.push("", "[activity]", ...activity);
	}
	if (details.error) {
		lines.push("", `[error] ${details.error}`);
	}
	return lines.join("\n");
}

function pushOrchestrateActivity(details: SidecarOrchestrateProgressDetails, line: string): void {
	const trimmed = line.trim();
	if (!trimmed) return;
	details.activity.push(trimmed);
	while (details.activity.length > SUBAGENT_STREAM_MAX_LINES) details.activity.shift();
}

function renderOrchestrateProgressText(
	details: SidecarOrchestrateProgressDetails,
	maxActivityLines = SUBAGENT_STREAM_MAX_LINES,
): string {
	const lines: string[] = ["Running ultracode..."];
	const activity = details.activity.slice(-Math.max(1, maxActivityLines));
	if (activity.length > 0) {
		lines.push("", "[activity]", ...activity);
	}
	if (details.result) {
		const state = details.result.state ?? "done";
		const ran = details.result.finders_ran ?? 0;
		const total = details.result.finders_total ?? 0;
		lines.push("", `[result] ${state} ${ran}/${total}`);
	}
	if (details.error) {
		lines.push("", `[error] ${details.error}`);
	}
	return lines.join("\n");
}

function parseSseDataBlock(block: string): string | undefined {
	const dataLines = block
		.split("\n")
		.filter((line) => line.startsWith("data:"))
		.map((line) => line.slice(5).replace(/^ /, ""));
	if (dataLines.length === 0) return undefined;
	return dataLines.join("\n");
}

async function postSubagentDelegateStream(
	body: Record<string, unknown>,
	signal: AbortSignal | undefined,
	onEvent: (event: SidecarSubagentStreamEvent) => void,
): Promise<
	| {
			result?: SidecarSubagentDelegateResponse;
			fallback?: SidecarSubagentDelegateResponse;
			error?: string;
			aborted?: boolean;
	  }
	| undefined
> {
	for (const baseUrl of sidecarUrlCandidates()) {
		const controller = new AbortController();
		const abortFromCaller = () => controller.abort();
		let sawStreamEvent = false;
		let idleTimedOut = false;
		let idleTimer: ReturnType<typeof setTimeout> | undefined;
		const armIdleTimer = () => {
			if (idleTimer) clearTimeout(idleTimer);
			idleTimer = setTimeout(() => {
				idleTimedOut = true;
				controller.abort();
			}, SUBAGENT_DELEGATE_IDLE_TIMEOUT_MS);
		};
		const clearIdleTimer = () => {
			if (idleTimer) {
				clearTimeout(idleTimer);
				idleTimer = undefined;
			}
		};
		if (signal?.aborted) {
			return { aborted: true, error: "Subagent cancelled" };
		}
		signal?.addEventListener("abort", abortFromCaller, { once: true });
		try {
			armIdleTimer();
			const pairId = process.env.JARVIS_PAIR_ID?.trim();
			const headers: Record<string, string> = {
				accept: "text/event-stream",
				"content-type": "application/json",
			};
			if (pairId) headers["X-Jarvis-Pair"] = pairId;
			const response = await fetch(`${baseUrl}/subagent/delegate`, {
				method: "POST",
				headers,
				body: JSON.stringify(body),
				signal: controller.signal,
			});
			if (!response.ok) {
				const errorBody = await response.text().catch(() => "");
				return {
					error: `JARVIS sidecar HTTP ${response.status}${errorBody ? `: ${errorBody.slice(0, 300)}` : ""}`,
				};
			}
			const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
			if (!contentType.includes("text/event-stream") || !response.body) {
				const parsed = await response.json().catch(() => null);
				return parsed === null
					? { error: "JARVIS sidecar returned malformed JSON" }
					: { fallback: parsed as SidecarSubagentDelegateResponse };
			}
			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			let result: SidecarSubagentDelegateResponse | undefined;
			let streamError: string | undefined;
			let done = false;

			while (!done) {
				const read = await reader.read();
				if (read.done) break;
				armIdleTimer();
				buffer += decoder.decode(read.value, { stream: true });
				buffer = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
				let splitAt = buffer.indexOf("\n\n");
				while (splitAt >= 0) {
					const block = buffer.slice(0, splitAt);
					buffer = buffer.slice(splitAt + 2);
					const data = parseSseDataBlock(block);
					if (data === "[DONE]") {
						done = true;
						break;
					}
					if (data) {
						const event = JSON.parse(data) as SidecarSubagentStreamEvent;
						sawStreamEvent = true;
						armIdleTimer();
						onEvent(event);
						if (event.event === "result" && event.result) result = event.result;
						if (event.event === "error") streamError = event.error ?? "Subagent stream failed";
					}
					splitAt = buffer.indexOf("\n\n");
				}
			}
			const tail = parseSseDataBlock(buffer.trim());
			if (tail && tail !== "[DONE]") {
				const event = JSON.parse(tail) as SidecarSubagentStreamEvent;
				sawStreamEvent = true;
				armIdleTimer();
				onEvent(event);
				if (event.event === "result" && event.result) result = event.result;
				if (event.event === "error") streamError = event.error ?? "Subagent stream failed";
			}
			if (result) return { result };
			if (streamError) return { error: streamError };
			return { error: "Subagent stream ended without a result" };
		} catch {
			if (signal?.aborted) return { aborted: true, error: "Subagent cancelled" };
			if (idleTimedOut) return { error: "Subagent stream idle timeout" };
			if (sawStreamEvent) return { error: "Subagent stream interrupted" };
			// Try the next advertised sidecar URL.
		} finally {
			clearIdleTimer();
			signal?.removeEventListener("abort", abortFromCaller);
		}
	}
	sidecarHealthy = false;
	return undefined;
}

async function postOrchestrateStream(
	body: Record<string, unknown>,
	signal: AbortSignal | undefined,
	onEvent: (event: SidecarOrchestrateStreamEvent) => void,
): Promise<
	| {
			result?: SidecarOrchestrateResponse;
			fallback?: SidecarOrchestrateResponse;
			error?: string;
			aborted?: boolean;
	  }
	| undefined
> {
	for (const baseUrl of sidecarUrlCandidates()) {
		const controller = new AbortController();
		const abortFromCaller = () => controller.abort();
		let sawStreamEvent = false;
		let idleTimedOut = false;
		let idleTimer: ReturnType<typeof setTimeout> | undefined;
		const armIdleTimer = () => {
			if (idleTimer) clearTimeout(idleTimer);
			idleTimer = setTimeout(() => {
				idleTimedOut = true;
				controller.abort();
			}, ORCHESTRATE_IDLE_TIMEOUT_MS);
		};
		const clearIdleTimer = () => {
			if (idleTimer) {
				clearTimeout(idleTimer);
				idleTimer = undefined;
			}
		};
		if (signal?.aborted) {
			return { aborted: true, error: "Orchestration cancelled" };
		}
		signal?.addEventListener("abort", abortFromCaller, { once: true });
		try {
			armIdleTimer();
			const pairId = process.env.JARVIS_PAIR_ID?.trim();
			const headers: Record<string, string> = {
				accept: "text/event-stream",
				"content-type": "application/json",
			};
			if (pairId) headers["X-Jarvis-Pair"] = pairId;
			const response = await fetch(`${baseUrl}/orchestrate`, {
				method: "POST",
				headers,
				body: JSON.stringify(body),
				signal: controller.signal,
			});
			if (!response.ok) {
				const errorBody = await response.text().catch(() => "");
				return {
					error: `JARVIS sidecar HTTP ${response.status}${errorBody ? `: ${errorBody.slice(0, 300)}` : ""}`,
				};
			}
			const contentType = response.headers.get("content-type")?.toLowerCase() ?? "";
			if (!contentType.includes("text/event-stream") || !response.body) {
				const parsed = await response.json().catch(() => null);
				return parsed === null
					? { error: "JARVIS sidecar returned malformed JSON" }
					: { fallback: parsed as SidecarOrchestrateResponse };
			}
			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			let result: SidecarOrchestrateResponse | undefined;
			let streamError: string | undefined;
			let done = false;

			while (!done) {
				const read = await reader.read();
				if (read.done) break;
				armIdleTimer();
				buffer += decoder.decode(read.value, { stream: true });
				buffer = buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
				let splitAt = buffer.indexOf("\n\n");
				while (splitAt >= 0) {
					const block = buffer.slice(0, splitAt);
					buffer = buffer.slice(splitAt + 2);
					const data = parseSseDataBlock(block);
					if (data === "[DONE]") {
						done = true;
						break;
					}
					if (data) {
						const event = JSON.parse(data) as SidecarOrchestrateStreamEvent;
						sawStreamEvent = true;
						armIdleTimer();
						onEvent(event);
						if (event.event === "result" && event.result) result = event.result;
						if (event.event === "error") streamError = event.error ?? "Orchestration stream failed";
					}
					splitAt = buffer.indexOf("\n\n");
				}
			}
			const tail = parseSseDataBlock(buffer.trim());
			if (tail && tail !== "[DONE]") {
				const event = JSON.parse(tail) as SidecarOrchestrateStreamEvent;
				sawStreamEvent = true;
				armIdleTimer();
				onEvent(event);
				if (event.event === "result" && event.result) result = event.result;
				if (event.event === "error") streamError = event.error ?? "Orchestration stream failed";
			}
			if (result) return { result };
			if (streamError) return { error: streamError };
			return { error: "Orchestration stream ended without a result" };
		} catch {
			if (signal?.aborted) return { aborted: true, error: "Orchestration cancelled" };
			if (idleTimedOut) return { error: "Orchestration stream idle timeout" };
			if (sawStreamEvent) return { error: "Orchestration stream interrupted" };
			// Try the next advertised sidecar URL.
		} finally {
			clearIdleTimer();
			signal?.removeEventListener("abort", abortFromCaller);
		}
	}
	sidecarHealthy = false;
	return undefined;
}

export async function postSidecar<T = unknown>(
	path: string,
	body?: unknown,
	method = "POST",
	timeoutMs = 45000,
): Promise<T | undefined> {
	for (const baseUrl of sidecarUrlCandidates()) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), timeoutMs);
		try {
			const pairId = process.env.JARVIS_PAIR_ID?.trim();
			const headers: Record<string, string> = {};
			if (body !== undefined) headers["content-type"] = "application/json";
			if (pairId) headers["X-Jarvis-Pair"] = pairId;
			const response = await fetch(`${baseUrl}${path}`, {
				method,
				headers: Object.keys(headers).length > 0 ? headers : undefined,
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
			const parsed = await response.json().catch(() => null);
			if (parsed === null) {
				return {
					ok: false,
					error: "JARVIS sidecar returned malformed JSON",
				} as T;
			}
			return parsed as T;
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

export function __jarvisMapRunSnapshotForTests():
	| {
			mapId: string;
			projectPath: string;
			phase: MapRunPhase;
			features: MapFeatureState[];
			jobFeatures: Record<string, string[]>;
			checkpoint?: { jobId: string; featureIds: string[] };
	  }
	| undefined {
	if (!activeMapRun) return undefined;
	return {
		mapId: activeMapRun.mapId,
		projectPath: activeMapRun.projectPath,
		phase: activeMapRun.phase,
		features: [...activeMapRun.features.values()].map((feature) => ({
			...feature,
			acceptance: [...feature.acceptance],
		})),
		jobFeatures: Object.fromEntries([...activeMapRun.jobFeatures.entries()].map(([jobId, ids]) => [jobId, [...ids]])),
		checkpoint: activeMapCheckpointTurn
			? { jobId: activeMapCheckpointTurn.jobId, featureIds: [...activeMapCheckpointTurn.featureIds] }
			: undefined,
	};
}

export function __setTurnToolActivityStateForTests(state: {
	providerCallCountThisTurn?: number;
	toolResultNames?: string[];
	activeModelProvider?: string;
}): void {
	if (state.providerCallCountThisTurn !== undefined) {
		providerCallCountThisTurn = state.providerCallCountThisTurn;
	}
	if (state.activeModelProvider !== undefined) {
		activeModelProviderThisTurn = state.activeModelProvider;
	}
	if (state.toolResultNames) {
		toolEvents.push({
			turnIndex: 0,
			toolResults: state.toolResultNames.map((toolName) => ({ toolName, isError: false, text: "" })),
		});
	}
}

export function __setProjectRouteStateForTests(state: {
	route?: EffectiveTurnRoute;
	activeProjectPath?: string;
	activeCodePath?: string;
	lastUserMessage?: string;
	defaultProjectRoot?: string;
}): void {
	if (state.route !== undefined) setEffectiveRoute(state.route);
	if (state.activeProjectPath !== undefined) activeProjectPath = state.activeProjectPath;
	if (state.activeCodePath !== undefined) activeCodePath = state.activeCodePath;
	if (state.lastUserMessage !== undefined) lastUserMessage = state.lastUserMessage;
	if (state.defaultProjectRoot !== undefined) {
		lastContextResponse = {
			...(lastContextResponse ?? {}),
			default_project_root: state.defaultProjectRoot,
		} as typeof lastContextResponse;
	}
}

export function __getTurnReconstructionSnapshotForTests(): {
	mutations: JarvisTurnFileMutation[];
	toolEvents: ToolEventSummary[];
} {
	return {
		mutations: turnSuccessfulFileMutations.map((mutation) => ({ ...mutation })),
		toolEvents: toolEvents.map((event) => ({
			turnIndex: event.turnIndex,
			toolResults: event.toolResults?.map((result) => ({ ...result })),
		})),
	};
}

export function __resetJarvisJlcForTests(): void {
	clearAutoPromptWatchdog();
	clearInterruptInputCheckpointHook();
	clearDirectiveIdlePoll();
	clearControlBridgePoll();
	stopAllManagedProcessesBestEffort();
	managedProcessStaleCleanupDone = false;
	activeProjectPath = undefined;
	activeCodePath = undefined;
	activeProjectId = undefined;
	lastContextResponse = undefined;
	lastInjectedContextMode = undefined;
	lastUserMessage = "";
	transientSystemDirective = "";
	pendingNewArtifactAskUserGate = false;
	toolEvents = [];
	checkpointToolEvents = [];
	lastAssistantPartialText = "";
	interruptCheckpointSavedThisTurn = false;
	turnCheckpointScope = undefined;
	sidecarHealthy = false;
	setEffectiveRoute("chat");
	currentTodoList = [];
	clearReadBeforeEditRegistry();
	deepdiveThinkingPreference = undefined;
	deepdiveThinkingPreferenceLoaded = false;
	subagentModelUserSet = false;
	subagentModelUserSetLoaded = false;
	coldStartNoticeShown = false;
	startupContextWarmupFinished = false;
	startupContextWarmupPromise = undefined;
	setupRequired = false;
	currentWindowLabel = undefined;
	lastTurnPromptSnapshot = undefined;
	lastProviderCallRoute = undefined;
	lastProviderToolsBeforeFilter = [];
	lastProviderToolsAfterFilter = [];
	lastProviderActionIntentMatch = false;
	lastProviderChatFilterApplied = false;
	lastProviderRoutePromotedByClassifier = false;
	lastRouteClassifierDecision = undefined;
	lastRouteClassifierActionIntent = false;
	expectedToolActivityThisTurn = false;
	routePromotedByClassifierThisTurn = false;
	workerToolsRetryInFlight = false;
	verifyContinuationCount = 0;
	workerWindowContextInjectedThisTurn = false;
	providerCallCountThisTurn = 0;
	activeModelProviderThisTurn = undefined;
	agentTurnActive = false;
	pendingDirectiveAutoTurn = undefined;
	activeDirectiveTurn = undefined;
	directiveTurnBusReplySent = false;
	pendingDirectiveReports = [];
	directiveSensorRunning = false;
	controlBridgeSensorRunning = false;
	directiveKnownQueueState.clear();
	activeMapRun = undefined;
	activeMapCheckpointTurn = undefined;
	activeMapSynthesisTurn = false;
	activeEndGateTurn = false;
	activeSecondEyesReviewTurn = false;
	activeSecondEyesMainTurn = false;
	activeSecondEyesHeavyTurn = false;
	secondEyesRequestedThisTurn = false;
	secondEyesReviewSpawnedThisTurn = false;
	secondEyesReminderInjectedThisTurn = false;
	askUserIssuedThisProviderCall = false;
	pendingMapSynthesisPost = false;
	lastUserActivityAtMs = 0;
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

// Cross-boundary test hook (item 2): drive the control-bridge CONSUMER half the
// anthropic-agent-sdk adapter routes map_create/feature_verdict through. The
// answer is posted via answerControlBridgeRequest -> postSidecar(/control/{id}/
// answer), which the test fetch mock captures, proving the kind string the
// adapter (producer) emits actually reaches a real pi branch (NOT the
// "unsupported control bridge request" else) and mutates pi's own activeMapRun.
export async function __handleControlBridgeRequestForTests(request: SidecarControlBridgeRequest): Promise<void> {
	const ctx = { signal: new AbortController().signal } as unknown as ExtensionContext;
	const pi = {} as unknown as ExtensionAPI;
	await handleControlBridgeRequest(request, ctx, pi);
}

// Seed an in-process map run so a feature_verdict consumer test has a feature to
// flip (mirrors the activeMapRun a real map_create would have established).
export function __seedActiveMapRunForTests(seed: {
	mapId?: string;
	projectPath?: string;
	phase?: MapRunPhase;
	features: Array<{ id: string; title?: string; acceptance?: string[]; status?: MapFeatureStatus }>;
}): void {
	const features = new Map<string, MapFeatureState>();
	for (const f of seed.features) {
		features.set(f.id, {
			id: f.id,
			title: f.title ?? f.id,
			acceptance: f.acceptance ? [...f.acceptance] : ["x"],
			zone: "feature",
			status: f.status ?? "dispatched",
			rejections: 0,
			stage: "normal",
		});
	}
	activeMapRun = {
		mapId: seed.mapId ?? "m_seed",
		title: "seed",
		projectPath: seed.projectPath ?? process.cwd(),
		features,
		jobFeatures: new Map(),
		phase: seed.phase ?? "stepping",
		ledgerSeq: 0,
	};
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

	// The model is mandated to lead with a [MODE:*] marker, so the real answer
	// begins at the first marker. Anything before it is leaked internal preamble
	// (e.g. a "reasoning:" line the model added on its own). Drop it by POSITION
	// — not by matching words — so this works in every chat language, not just
	// the ones we happened to enumerate in a regex.
	const firstMarker = sanitized.match(MODE_MARKER_ANY_RE)?.[0];
	if (firstMarker) {
		const markerIndex = sanitized.indexOf(firstMarker);
		if (markerIndex > 0) sanitized = sanitized.slice(markerIndex);
	}

	// The marker(s) are routing signals the sidecar reads, never user content.
	sanitized = sanitized.replace(MODE_MARKER_ANY_RE, "");

	// The agent-sdk tool-activity trailer (regime-B memory sensor) rides a sentinel
	// line in the assistant text; the consumer parses it off the RAW message, so by
	// the time text is sanitized for display/persistence it must be stripped.
	sanitized = stripJarvisSdkToolTrailer(sanitized);

	return sanitized
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

export function injectMemoryIntoLatestUser(
	messages: AgentMessage[],
	memory: string,
	workspace?: string,
): AgentMessage[] {
	let injected = false;
	const next = [...messages];
	// CACHE: the volatile workspace feed rides the latest user message (a "live"
	// tail message, never part of the cacheable prefix) instead of the system
	// prompt, so a folder change no longer invalidates system+tools+history.
	const workspaceText = (workspace ?? "").trim();
	const workspaceBlock = workspaceText ? `\n\n<jarvis_workspace>\n${workspaceText}\n</jarvis_workspace>` : "";
	for (let i = next.length - 1; i >= 0; i--) {
		const message = next[i];
		if (message.role !== "user") continue;
		const existingText = stripJarvisMemoryBlock(messageContentToText(message.content));
		next[i] = {
			...message,
			content: [
				{
					type: "text",
					text: `<jarvis_memory>\n${memory}\n</jarvis_memory>${workspaceBlock}\n\n${existingText}`,
				},
			],
		};
		injected = true;
		break;
	}
	return injected ? next : messages;
}

function stripJarvisMemoryBlock(text: string): string {
	return text
		.replace(/^<jarvis_memory>[\s\S]*?<\/jarvis_memory>\s*/i, "")
		.replace(/^<jarvis_workspace>[\s\S]*?<\/jarvis_workspace>\s*/i, "")
		.trimStart();
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
	terminalReason?: JarvisTurnTerminalReason,
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
		terminal_reason: terminalReason,
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

const TOOL_NAME_CANONICAL_ALIASES: Record<string, string> = {
	askuserquestion: "ask_user",
	askuser: "ask_user",
	jobsend: "job_send",
	listwindows: "list_windows",
	senddirective: "send_directive",
	jobclose: "job_close",
	spawnwindow: "spawn_window",
	switchproject: "switch_project",
	registerproject: "register_project",
	unregisterproject: "unregister_project",
	setchatmodel: "set_chat_model",
	setsubagentmodel: "set_subagent_model",
	setencodermodel: "set_encoder_model",
	generateimage: "generate_image",
	editimage: "edit_image",
	recallturns: "recall_turns",
	mapcreate: "map_create",
	updatejarvismd: "update_jarvis_md",
	managedprocess: "managed_process",
	featureverdict: "feature_verdict",
	gansend: "gan_send",
	ganclose: "gan_close",
};

function normalizeToolSchemaNameRaw(raw: string): string {
	const value = String(raw || "").trim();
	if (!value) return "";
	const lastSegment = value.includes("__") ? (value.split("__").at(-1) ?? value) : value;
	const snake = lastSegment.replace(/([a-z0-9])([A-Z])/g, "$1_$2");
	const sanitized = snake
		.replace(/[^A-Za-z0-9_]/g, "_")
		.replace(/_+/g, "_")
		.replace(/^_|_$/g, "")
		.toLowerCase();
	if (!sanitized) return "";
	if (Object.hasOwn(TOOL_NAME_CANONICAL_ALIASES, sanitized)) {
		return TOOL_NAME_CANONICAL_ALIASES[sanitized];
	}
	const collapsed = sanitized.replaceAll("_", "");
	return Object.hasOwn(TOOL_NAME_CANONICAL_ALIASES, collapsed) ? TOOL_NAME_CANONICAL_ALIASES[collapsed] : sanitized;
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
	const raw =
		(typeof record.name === "string" && record.name) ||
		(fnRecord && typeof fnRecord.name === "string" && fnRecord.name) ||
		(defRecord && typeof defRecord.name === "string" && defRecord.name) ||
		undefined;
	return raw ? normalizeToolSchemaNameRaw(raw) : undefined;
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
		subturnLedgerLines.length > 0 ||
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
	const level = route === "heavy_deepdive" ? selectDeepdiveThinkingLevel(ctx) : "high";
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
