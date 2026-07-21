import { type Component, truncateToWidth, visibleWidth } from "@earendil-works/pi-tui";
import type { AgentSession } from "../../../core/agent-session.js";
import { estimateTokens } from "../../../core/compaction/compaction.js";
import type { ReadonlyFooterDataProvider } from "../../../core/footer-data-provider.js";
import { theme } from "../theme/theme.js";

const JLC_METER_ENTRY_TYPE = "jarvis-jlc-meter";
const JLC_METER_RESET_ENTRY_TYPE = "jarvis-jlc-meter-reset";
const JLC_CHAT_THINKING_STATUS_KEY = "jlc-chat-thinking";
const JLC_SUBAGENT_THINKING_STATUS_KEY = "jlc-subagent-thinking";
const JLC_SUBAGENT_MODEL_STATUS_KEY = "jlc-subagent-model";

type JlcMeterEntry = {
	chat_in?: number;
	chat_out?: number;
	chat_total?: number;
	jhb_tokens?: number;
};

/**
 * Sanitize text for display in a single-line status.
 * Removes newlines, tabs, carriage returns, and other control characters.
 */
function sanitizeStatusText(text: string): string {
	// Replace newlines, tabs, carriage returns with space, then collapse multiple spaces
	return text
		.replace(/[\r\n\t]/g, " ")
		.replace(/ +/g, " ")
		.trim();
}

/**
 * Format token counts (similar to web-ui)
 */
function formatTokens(count: number): string {
	if (count < 1000) return count.toString();
	if (count < 10000) return `${(count / 1000).toFixed(1)}k`;
	if (count < 1000000) return `${Math.round(count / 1000)}k`;
	if (count < 10000000) return `${(count / 1000000).toFixed(1)}M`;
	return `${Math.round(count / 1000000)}M`;
}

function numberOrZero(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : 0;
}

function getCustomType(entry: unknown): string | undefined {
	if (!entry || typeof entry !== "object") return undefined;
	const record = entry as Record<string, unknown>;
	return record.type === "custom" && typeof record.customType === "string" ? record.customType : undefined;
}

function getJlcMeterData(entry: unknown): JlcMeterEntry | undefined {
	if (getCustomType(entry) !== JLC_METER_ENTRY_TYPE) return undefined;
	const data = (entry as Record<string, unknown>).data;
	return data && typeof data === "object" ? (data as JlcMeterEntry) : undefined;
}

function getThinkingLevelStatus(
	value: string,
): "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultra" | undefined {
	return value === "off" ||
		value === "minimal" ||
		value === "low" ||
		value === "medium" ||
		value === "high" ||
		value === "xhigh" ||
		value === "max" ||
		value === "ultra"
		? value
		: undefined;
}

/**
 * Map a reasoning effort level to a footer color function. The mapping is by
 * effort tier (cost), not by which source reported the level:
 *   minimal/low -> green (success), medium -> deep green (thinkingMedium),
 *   high -> yellow (warning),
 *   xhigh/max/ultra -> red (error). off/undefined stays uncolored (dim).
 */
function thinkingLevelColor(
	level: "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultra" | undefined,
): ((text: string) => string) | undefined {
	switch (level) {
		case "minimal":
		case "low":
			return (text: string) => theme.fg("success", text);
		case "medium":
			return (text: string) => theme.fg("thinkingMedium", text);
		case "high":
			return (text: string) => theme.fg("warning", text);
		case "xhigh":
		case "max":
		case "ultra":
			return (text: string) => theme.fg("error", text);
		default:
			return undefined;
	}
}

function calculateJlcMeterTotals(entries: unknown[]): { found: boolean; jlcTotal: number; rawProjectedTotal: number } {
	let startIndex = -1;
	for (let i = entries.length - 1; i >= 0; i--) {
		if (getCustomType(entries[i]) === JLC_METER_RESET_ENTRY_TYPE) {
			startIndex = i;
			break;
		}
	}

	let found = startIndex >= 0;
	let jlcTotal = 0;
	let rawProjectedTotal = 0;
	let rawPrefix = 0;
	for (const entry of entries.slice(startIndex + 1)) {
		const meter = getJlcMeterData(entry);
		if (!meter) continue;
		const turnTotal = numberOrZero(meter.chat_total) || numberOrZero(meter.chat_in) + numberOrZero(meter.chat_out);
		if (turnTotal <= 0) continue;
		found = true;
		jlcTotal += turnTotal;
		rawPrefix += turnTotal;
		rawProjectedTotal += rawPrefix;
	}

	return { found, jlcTotal, rawProjectedTotal };
}

/**
 * Footer component that shows pwd, token stats, and context usage.
 * Computes token/context stats from session, gets git branch and extension statuses from provider.
 */
export class FooterComponent implements Component {
	constructor(
		private session: AgentSession,
		private footerData: ReadonlyFooterDataProvider,
	) {}

	setSession(session: AgentSession): void {
		this.session = session;
	}

	/**
	 * No-op: git branch caching now handled by provider.
	 * Kept for compatibility with existing call sites in interactive-mode.
	 */
	invalidate(): void {
		// No-op: git branch is cached/invalidated by provider
	}

	/**
	 * Clean up resources.
	 * Git watcher cleanup now handled by provider.
	 */
	dispose(): void {
		// Git watcher cleanup handled by provider
	}

	render(width: number): string[] {
		const state = this.session.state;

		// Prefer JLC meter entries for the current app run. They are reset by
		// jarvis-jlc on session_start, so restarting starts the footer at 0/0
		// even when the underlying session file still contains older entries.
		// Raw projection uses the intended prefix model:
		// turn1 = t1, turn2 = t1+t2, turn3 = t1+t2+t3.
		// If no JLC meter has been installed, fall back to message-derived stats.
		let jlcTotal = 0;
		let rawProjectedTotal = 0;
		let completedTranscriptTokens = 0;
		let pendingTurnTranscriptTokens = 0;

		const entries = this.session.sessionManager.getEntries();
		const jlcMeterTotals = calculateJlcMeterTotals(entries);

		for (const entry of entries) {
			if (entry.type !== "message") continue;

			if (entry.message.role === "assistant") {
				const u = entry.message.usage;
				const turnTotal = u.totalTokens ?? u.input + u.output + u.cacheRead + u.cacheWrite;
				jlcTotal += turnTotal;
				rawProjectedTotal += completedTranscriptTokens + turnTotal;
				pendingTurnTranscriptTokens += estimateTokens(entry.message);
				completedTranscriptTokens += pendingTurnTranscriptTokens;
				pendingTurnTranscriptTokens = 0;
				continue;
			}

			pendingTurnTranscriptTokens += estimateTokens(entry.message);
		}

		if (jlcMeterTotals.found) {
			jlcTotal = jlcMeterTotals.jlcTotal;
			rawProjectedTotal = jlcMeterTotals.rawProjectedTotal;
		}

		// Build stats line. No $-cost figure here: provider-list prices are
		// unreliable for JARVIS-managed models and read as billed spend.
		const statsParts = [];

		// JLC vs legacy cumulative token comparison.
		const ratio = jlcTotal > 0 && rawProjectedTotal > jlcTotal ? Math.round(rawProjectedTotal / jlcTotal) : 0;
		const ratioStr = ratio >= 2 ? ` · ${ratio}:1` : "";
		statsParts.push(`jlc ${formatTokens(jlcTotal)} vs legacy ${formatTokens(rawProjectedTotal)}${ratioStr}`);

		let statsLeft = statsParts.join(" ");
		const extensionStatuses = this.footerData.getExtensionStatuses();

		// Add model name on the right side. JARVIS sidecar proxies are registered
		// as Pi providers, so normal/proxied turns should show state.model. The
		// jlc-chat-model override is only for a future runtime that bypasses Pi's
		// provider registry entirely.
		const jlcChatModel = sanitizeStatusText(extensionStatuses.get("jlc-chat-model") ?? "");
		const modelName = jlcChatModel || state.model?.id || "no-model";
		const jlcThinkingLevel = getThinkingLevelStatus(
			sanitizeStatusText(extensionStatuses.get(JLC_CHAT_THINKING_STATUS_KEY) ?? ""),
		);
		const thinkingLevel = jlcThinkingLevel ?? state.thinkingLevel;
		const subagentModel = sanitizeStatusText(extensionStatuses.get(JLC_SUBAGENT_MODEL_STATUS_KEY) ?? "");
		const subagentThinkingLevel = getThinkingLevelStatus(
			sanitizeStatusText(extensionStatuses.get(JLC_SUBAGENT_THINKING_STATUS_KEY) ?? ""),
		);
		// Subagent model/effort renders on its own right-aligned line below the chat
		// model line (see subLine near the return), not inline after the chat model.
		const rightSideBody = `${modelName} ${thinkingLevel}`;
		// Color by the resolved effort level, not by which source set it.
		// Medium uses a deeper green than generic success so it does not read yellow.
		const rightSideColor = thinkingLevelColor(thinkingLevel);

		let statsLeftWidth = visibleWidth(statsLeft);

		// If statsLeft is too wide, truncate it
		if (statsLeftWidth > width) {
			statsLeft = truncateToWidth(statsLeft, width, "...");
			statsLeftWidth = visibleWidth(statsLeft);
		}

		// Calculate available space for padding (minimum 2 spaces between stats and model)
		const minPadding = 2;

		const rightSideWithoutProvider = rightSideBody;

		// Prepend the provider in parentheses if there are multiple providers and there's enough room
		let rightSide = rightSideWithoutProvider;
		// jlcChatModel already carries its own "(provider) model" label.
		if (!jlcChatModel && this.footerData.getAvailableProviderCount() > 1 && state.model) {
			rightSide = `(${state.model.provider}) ${rightSideWithoutProvider}`;
			if (statsLeftWidth + minPadding + visibleWidth(rightSide) > width) {
				// Too wide, fall back
				rightSide = rightSideWithoutProvider;
			}
		}

		const rightSideWidth = visibleWidth(rightSide);
		const totalNeeded = statsLeftWidth + minPadding + rightSideWidth;

		let statsLine: string;
		if (totalNeeded <= width) {
			// Both fit - add padding to right-align model
			const padding = " ".repeat(width - statsLeftWidth - rightSideWidth);
			statsLine = statsLeft + padding + rightSide;
		} else {
			// Need to truncate right side
			const availableForRight = width - statsLeftWidth - minPadding;
			if (availableForRight > 0) {
				const truncatedRight = truncateToWidth(rightSide, availableForRight, "");
				const truncatedRightWidth = visibleWidth(truncatedRight);
				const padding = " ".repeat(Math.max(0, width - statsLeftWidth - truncatedRightWidth));
				statsLine = statsLeft + padding + truncatedRight;
			} else {
				// Not enough space for right side at all
				statsLine = statsLeft;
			}
		}

		// Apply dim to each part separately. statsLeft may contain color codes (for context %)
		// that end with a reset, which would clear an outer dim wrapper. So we dim the parts
		// before and after the colored section independently.
		const dimStatsLeft = theme.fg("dim", statsLeft);
		const remainder = statsLine.slice(statsLeft.length);
		const rightSideStart = remainder.length - rightSide.length;
		const padding = rightSideStart > 0 ? remainder.slice(0, rightSideStart) : "";
		const trailingRightSide = rightSideStart > 0 ? remainder.slice(rightSideStart) : remainder;
		const dimPadding = theme.fg("dim", padding);
		const styledRightSide = rightSideColor ? rightSideColor(trailingRightSide) : theme.fg("dim", trailingRightSide);

		const workStatus = sanitizeStatusText(extensionStatuses.get("jlc-work") ?? "");
		const encStatus = sanitizeStatusText(extensionStatuses.get("jlc-enc") ?? "");
		const encModelStatus = sanitizeStatusText(extensionStatuses.get("jlc-enc-model") ?? "");
		const statusLine =
			extensionStatuses.size > 0
				? Array.from(extensionStatuses.entries())
						.filter(
							([key]) =>
								key !== "jlc-work" &&
								key !== "jlc-enc" &&
								key !== "jlc-enc-model" &&
								key !== "jlc-chat-model" &&
								key !== JLC_CHAT_THINKING_STATUS_KEY &&
								key !== JLC_SUBAGENT_THINKING_STATUS_KEY &&
								key !== JLC_SUBAGENT_MODEL_STATUS_KEY,
						)
						.sort(([a], [b]) => a.localeCompare(b))
						.map(([, text]) => sanitizeStatusText(text))
						.filter((text) => text.length > 0)
						.join(" · ")
				: "";
		const fixedStatusLine = theme.fg("dim", truncateToWidth(statusLine, width, "..."));
		const thirdLine =
			encStatus || encModelStatus || workStatus
				? (() => {
						// Left = encStatus (`enc:Nt/Ns`); right = encModelStatus
						// (`(provider) model effort`), aligned directly under the
						// chat model line on row 1. workStatus appended after enc on
						// the left when present.
						const leftRaw = workStatus ? `${encStatus}  ${workStatus}` : encStatus;
						const right = truncateToWidth(encModelStatus, width, "...");
						const rightWidth = visibleWidth(right);
						const left = truncateToWidth(leftRaw, Math.max(0, width - rightWidth - 2), "");
						const leftWidth = visibleWidth(left);
						const padding = " ".repeat(Math.max(2, width - leftWidth - rightWidth));
						return theme.fg("dim", left + padding + right);
					})()
				: undefined;

		// Subagent model + effort on its own right-aligned line, directly under the
		// chat model line. Colored by the subagent effort so Alt+2 cycling is visible.
		const subLine =
			subagentModel && subagentThinkingLevel
				? (() => {
						const text = `sub:${subagentModel} ${subagentThinkingLevel}`;
						const truncated = truncateToWidth(text, width, "");
						const pad = " ".repeat(Math.max(0, width - visibleWidth(truncated)));
						const color = thinkingLevelColor(subagentThinkingLevel);
						return pad + (color ? color(truncated) : theme.fg("dim", truncated));
					})()
				: undefined;

		return [
			dimStatsLeft + dimPadding + styledRightSide,
			...(subLine ? [subLine] : []),
			fixedStatusLine,
			...(thirdLine ? [thirdLine] : []),
		];
	}
}
