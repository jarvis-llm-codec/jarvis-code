import type { AssistantMessage, ThinkingContent } from "../types.js";
import type { AssistantMessageEventStream } from "../utils/event-stream.js";

export const REASONING_PLACEHOLDER_TEXT = "[thinking...]";

export function emitReasoningPlaceholder(
	output: AssistantMessage,
	stream: AssistantMessageEventStream,
): ThinkingContent {
	const existing = getReasoningPlaceholder(output);
	if (existing) return existing;

	const block: ThinkingContent = { type: "thinking", thinking: REASONING_PLACEHOLDER_TEXT };
	output.content.push(block);
	const contentIndex = output.content.length - 1;
	stream.push({ type: "thinking_start", contentIndex, partial: output });
	stream.push({
		type: "thinking_delta",
		contentIndex,
		delta: block.thinking,
		partial: output,
	});
	return block;
}

export function getReasoningPlaceholder(output: AssistantMessage): ThinkingContent | undefined {
	const block = output.content.length === 1 ? output.content[0] : undefined;
	return block?.type === "thinking" && isReasoningPlaceholder(block) ? block : undefined;
}

export function clearReasoningPlaceholder(block: ThinkingContent): void {
	if (isReasoningPlaceholder(block)) {
		block.thinking = "";
	}
}

function isReasoningPlaceholder(block: ThinkingContent): boolean {
	return block.thinking === REASONING_PLACEHOLDER_TEXT && !block.thinkingSignature;
}
