import type { AssistantMessage } from "@earendil-works/pi-ai";
import { Container, Markdown, type MarkdownTheme, Spacer, Text } from "@earendil-works/pi-tui";
import { getMarkdownTheme, theme } from "../theme/theme.js";

const OSC133_ZONE_START = "\x1b]133;A\x07";
const OSC133_ZONE_END = "\x1b]133;B\x07";
const OSC133_ZONE_FINAL = "\x1b]133;C\x07";

function getFoldedThinkingLabel(thinking: string): string {
	const firstLine = thinking
		.split(/\r?\n/)
		.map((line) => line.trim())
		.find((line) => line.length > 0 && !isThinkingPlaceholder(line));
	return firstLine ? `${firstLine}...` : "...";
}

function isThinkingPlaceholder(line: string): boolean {
	return /^\[?thinking(?:\.\.\.)?\]?$/i.test(line) || /^thinking\s+(off|minimal|low|medium|high|xhigh)$/i.test(line);
}

const AGENT_SDK_TOOL_ACTIVITY_LINE_RE =
	/^\[(?:read|write|edit|shell|web search|web fetch|glob|grep|list|todo|agent task|ask user|job send|job close|done|failed(?::[^\]]+)?|agent tool:[^\]]+)(?:\]|:)/i;

// Tool labels whose following lines are a result body we keep in the collapsed
// view (sidecar Path A: the body rides reasoning_content). A '[done...]' /
// '[failed...]' label is followed by the bounded tool/file body; everything up to
// the next activity label is that body.
const AGENT_SDK_TOOL_BODY_OPENER_RE = /^\[(?:done|failed)\b/i;

function visibleAgentSdkToolActivityLines(thinking: string): string[] {
	const out: string[] = [];
	let inBody = false;
	for (const raw of thinking.split(/\r?\n/)) {
		const trimmed = raw.trim();
		if (AGENT_SDK_TOOL_ACTIVITY_LINE_RE.test(trimmed)) {
			out.push(trimmed);
			// A result-bearing label opens a body section; the next label closes it.
			inBody = AGENT_SDK_TOOL_BODY_OPENER_RE.test(trimmed);
			continue;
		}
		// Keep the tool/file body lines too (not just the label), so e.g. a file
		// read streams its content in the collapsed view. Preserve leading indent
		// (source code), trim only trailing whitespace. The sidecar already bounds
		// the body (size + line count), so this cannot flood unbounded.
		if (inBody) {
			out.push(raw.replace(/\s+$/, ""));
		}
	}
	return out;
}

/**
 * Component that renders a complete assistant message
 */
export class AssistantMessageComponent extends Container {
	private contentContainer: Container;
	private hideThinkingBlock: boolean;
	private markdownTheme: MarkdownTheme;
	private finalized: boolean;
	private showFinalizedThinking: boolean;
	private cachedThinkingContent: AssistantMessage["content"] = [];
	private lastMessage?: AssistantMessage;
	private hasToolCalls = false;

	constructor(
		message?: AssistantMessage,
		hideThinkingBlock = false,
		markdownTheme: MarkdownTheme = getMarkdownTheme(),
		_hiddenThinkingLabel = "...",
		finalized = true,
		showFinalizedThinking = false,
	) {
		super();

		this.hideThinkingBlock = hideThinkingBlock;
		this.markdownTheme = markdownTheme;
		this.finalized = finalized;
		this.showFinalizedThinking = showFinalizedThinking;

		// Container for text/thinking content
		this.contentContainer = new Container();
		this.addChild(this.contentContainer);

		if (message) {
			this.updateContent(message);
		}
	}

	override invalidate(): void {
		super.invalidate();
		if (this.lastMessage) {
			this.updateContent(this.lastMessage);
		}
	}

	setHideThinkingBlock(hide: boolean): void {
		this.hideThinkingBlock = hide;
		if (this.lastMessage) {
			this.updateContent(this.lastMessage);
		}
	}

	setHiddenThinkingLabel(_label: string): void {
		if (this.lastMessage) {
			this.updateContent(this.lastMessage);
		}
	}

	setFinalized(finalized: boolean): void {
		this.finalized = finalized;
		if (this.lastMessage) {
			this.updateContent(this.lastMessage);
		}
	}

	setShowFinalizedThinking(show: boolean): void {
		this.showFinalizedThinking = show;
		if (this.lastMessage) {
			this.updateContent(this.lastMessage);
		}
	}

	override render(width: number): string[] {
		const lines = super.render(width);
		if (this.hasToolCalls || lines.length === 0) {
			return lines;
		}

		lines[0] = OSC133_ZONE_START + lines[0];
		lines[lines.length - 1] = OSC133_ZONE_END + OSC133_ZONE_FINAL + lines[lines.length - 1];
		return lines;
	}

	updateContent(message: AssistantMessage): void {
		this.lastMessage = message;

		// Clear content container
		this.contentContainer.clear();

		const incomingThinking = message.content.filter((c) => c.type === "thinking" && c.thinking.trim());
		if (incomingThinking.length > 0) {
			this.cachedThinkingContent = incomingThinking;
		}
		const renderContent =
			incomingThinking.length === 0 && this.cachedThinkingContent.length > 0
				? [...this.cachedThinkingContent, ...message.content]
				: message.content;

		const hasVisibleContent = renderContent.some(
			(c) => (c.type === "text" && c.text.trim()) || (c.type === "thinking" && c.thinking.trim()),
		);

		if (hasVisibleContent) {
			this.contentContainer.addChild(new Spacer(1));
		}

		// Render content in order
		for (let i = 0; i < renderContent.length; i++) {
			const content = renderContent[i];
			if (content.type === "text" && content.text.trim()) {
				// Assistant text messages with no background - trim the text
				// Set paddingY=0 to avoid extra spacing before tool executions
				this.contentContainer.addChild(new Markdown(content.text.trim(), 1, 0, this.markdownTheme));
			} else if (content.type === "thinking" && content.thinking.trim()) {
				// Add spacing only when another visible assistant content block follows.
				// This avoids a superfluous blank line before separately-rendered tool execution blocks.
				const hasVisibleContentAfter = renderContent
					.slice(i + 1)
					.some((c) => (c.type === "text" && c.text.trim()) || (c.type === "thinking" && c.thinking.trim()));

				if (this.hideThinkingBlock || (this.finalized && !this.showFinalizedThinking)) {
					const activityLines = visibleAgentSdkToolActivityLines(content.thinking);
					if (activityLines.length > 0) {
						for (const line of activityLines) {
							this.contentContainer.addChild(new Text(theme.italic(theme.fg("thinkingText", line)), 1, 0));
						}
					} else {
						// Show static thinking label when hidden
						this.contentContainer.addChild(
							new Text(theme.italic(theme.fg("thinkingText", getFoldedThinkingLabel(content.thinking))), 1, 0),
						);
					}
					if (hasVisibleContentAfter) {
						this.contentContainer.addChild(new Spacer(1));
					}
				} else {
					// Thinking traces in thinkingText color, italic
					this.contentContainer.addChild(
						new Markdown(content.thinking.trim(), 1, 0, this.markdownTheme, {
							color: (text: string) => theme.fg("thinkingText", text),
							italic: true,
						}),
					);
					if (hasVisibleContentAfter) {
						this.contentContainer.addChild(new Spacer(1));
					}
				}
			}
		}

		// Check if aborted - show after partial content
		// But only if there are no tool calls (tool execution components will show the error)
		const hasToolCalls = message.content.some((c) => c.type === "toolCall");
		this.hasToolCalls = hasToolCalls;
		if (!hasToolCalls) {
			if (message.stopReason === "aborted") {
				const abortMessage =
					message.errorMessage && message.errorMessage !== "Request was aborted"
						? message.errorMessage
						: "Operation aborted";
				if (hasVisibleContent) {
					this.contentContainer.addChild(new Spacer(1));
				} else {
					this.contentContainer.addChild(new Spacer(1));
				}
				this.contentContainer.addChild(new Text(theme.fg("error", abortMessage), 1, 0));
			} else if (message.stopReason === "error") {
				const errorMsg = message.errorMessage || "Unknown error";
				this.contentContainer.addChild(new Spacer(1));
				this.contentContainer.addChild(new Text(theme.fg("error", `Error: ${errorMsg}`), 1, 0));
			}
		}
	}
}
