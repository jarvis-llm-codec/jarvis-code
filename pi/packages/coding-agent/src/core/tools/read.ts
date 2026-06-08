import { basename, dirname, isAbsolute, relative, resolve as resolvePath, sep } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { Api, ImageContent, Model, TextContent } from "@earendil-works/pi-ai";
import { Text } from "@earendil-works/pi-tui";
import { constants } from "fs";
import { access as fsAccess, readFile as fsReadFile, stat as fsStat } from "fs/promises";
import { type Static, Type } from "typebox";
import { getReadmePath } from "../../config.js";
import { keyHint, keyText } from "../../modes/interactive/components/keybinding-hints.js";
import { getLanguageFromPath, highlightCode, type Theme } from "../../modes/interactive/theme/theme.js";
import { formatDimensionNote, resizeImage } from "../../utils/image-resize.js";
import { detectSupportedImageMimeTypeFromFile } from "../../utils/mime.js";
import { formatPathRelativeToCwdOrAbsolute } from "../../utils/paths.js";
import type { ToolDefinition, ToolRenderResultOptions } from "../extensions/types.js";
import { resolveReadPath } from "./path-utils.js";
import { getTextOutput, invalidArgText, replaceTabs, shortenPath, str } from "./render-utils.js";
import { wrapToolDefinition } from "./tool-definition-wrapper.js";
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, formatSize, type TruncationResult, truncateHead } from "./truncate.js";

const readItemSchema = Type.Object({
	path: Type.String({ description: "Path to the file to read (relative or absolute)" }),
	offset: Type.Optional(Type.Number({ description: "Line number to start reading from (1-indexed)" })),
	limit: Type.Optional(Type.Number({ description: "Maximum number of lines to read" })),
});

const readSchema = Type.Object({
	path: Type.Optional(Type.String({ description: "Path to the file to read (relative or absolute)" })),
	offset: Type.Optional(Type.Number({ description: "Line number to start reading from (1-indexed)" })),
	limit: Type.Optional(Type.Number({ description: "Maximum number of lines to read" })),
	items: Type.Optional(
		Type.Array(readItemSchema, {
			description: "Batch of files or file ranges to read in one call. Prefer this for multi-file inspection.",
		}),
	),
});

export type ReadToolInput = Static<typeof readSchema>;

export interface ReadToolDetails {
	truncation?: TruncationResult;
	items?: Array<{ path: string; startLine?: number; endLine?: number; truncation?: TruncationResult }>;
}

interface CompactReadClassification {
	kind: "docs" | "resource" | "skill";
	label: string;
}

const COMPACT_RESOURCE_FILE_NAMES = new Set(["JARVIS.md", "JARVIS.MD", "CLAUDE.md", "CLAUDE.MD"]);

/**
 * Pluggable operations for the read tool.
 * Override these to delegate file reading to remote systems (for example SSH).
 */
export interface ReadOperations {
	/** Read file contents as a Buffer */
	readFile: (absolutePath: string) => Promise<Buffer>;
	/** Check if file is readable (throw if not) */
	access: (absolutePath: string) => Promise<void>;
	/** Read file metadata. Used to allow rereads after a file changes. */
	stat?: (absolutePath: string) => Promise<{ size: number; mtimeMs: number }>;
	/** Detect image MIME type, return null or undefined for non-images */
	detectImageMimeType?: (absolutePath: string) => Promise<string | null | undefined>;
}

const defaultReadOperations: ReadOperations = {
	readFile: (path) => fsReadFile(path),
	access: (path) => fsAccess(path, constants.R_OK),
	stat: (path) => fsStat(path),
	detectImageMimeType: detectSupportedImageMimeTypeFromFile,
};

export interface ReadToolOptions {
	/** Whether to auto-resize images to 2000x2000 max. Default: true */
	autoResizeImages?: boolean;
	/** Custom operations for file reading. Default: local filesystem */
	operations?: ReadOperations;
}

type ReadRenderArgs = {
	path?: string;
	file_path?: string;
	offset?: number;
	limit?: number;
	items?: Array<{ path?: string; offset?: number; limit?: number }>;
};

interface ReadCoverageRange {
	startLine: number;
	endLine: number;
	signature: string;
}

interface SingleReadResult {
	content: (TextContent | ImageContent)[];
	details?: ReadToolDetails;
	displayPath: string;
	startLine?: number;
	endLine?: number;
}

function splitReadPathLineSuffix(
	path: string,
	offset?: number,
	limit?: number,
): { path: string; offset?: number; limit?: number } {
	const match = path.match(/^(.*):(\d+)(?:-(\d+))?$/);
	if (!match || !match[1]) return { path, offset, limit };
	const start = Number.parseInt(match[2] ?? "", 10);
	const end = match[3] !== undefined ? Number.parseInt(match[3], 10) : undefined;
	if (!Number.isFinite(start) || start < 1) return { path, offset, limit };
	const nextOffset = offset ?? start;
	let nextLimit = limit;
	if (nextLimit === undefined && end !== undefined && Number.isFinite(end) && end >= start) {
		nextLimit = end - start + 1;
	}
	return { path: match[1], offset: nextOffset, limit: nextLimit };
}

function formatReadLineRange(args: ReadRenderArgs | undefined, theme: Theme): string {
	if (args?.offset === undefined && args?.limit === undefined) return "";
	const startLine = args.offset ?? 1;
	const endLine = args.limit !== undefined ? startLine + args.limit - 1 : "";
	return theme.fg("warning", `:${startLine}${endLine ? `-${endLine}` : ""}`);
}

function formatReadCall(args: ReadRenderArgs | undefined, theme: Theme): string {
	if (args?.items && args.items.length > 0) {
		return `${theme.fg("toolTitle", theme.bold("read"))} ${theme.fg("accent", `${args.items.length} items`)}`;
	}
	const rawPath = str(args?.file_path ?? args?.path);
	const path = rawPath !== null ? shortenPath(rawPath) : null;
	const invalidArg = invalidArgText(theme);
	const pathDisplay = path === null ? invalidArg : path ? theme.fg("accent", path) : theme.fg("toolOutput", "...");
	return `${theme.fg("toolTitle", theme.bold("read"))} ${pathDisplay}${formatReadLineRange(args, theme)}`;
}

function formatBatchHeader(result: SingleReadResult): string {
	const range =
		result.startLine !== undefined && result.endLine !== undefined ? `:${result.startLine}-${result.endLine}` : "";
	return `--- ${result.displayPath}${range} ---`;
}

function fileSignature(size: number | undefined, mtimeMs: number | undefined, buffer: Buffer): string {
	if (size !== undefined && mtimeMs !== undefined) return `${size}:${mtimeMs}`;
	return `buffer:${buffer.length}`;
}

function findCoveredRange(
	ranges: ReadCoverageRange[] | undefined,
	startLine: number,
	endLine: number,
	signature: string,
): ReadCoverageRange | undefined {
	return ranges?.find(
		(range) => range.signature === signature && range.startLine <= startLine && range.endLine >= endLine,
	);
}

function rememberCoverage(
	coverage: Map<string, ReadCoverageRange[]>,
	absolutePath: string,
	startLine: number,
	endLine: number,
	signature: string,
): void {
	const ranges = coverage.get(absolutePath) ?? [];
	ranges.push({ startLine, endLine, signature });
	coverage.set(absolutePath, ranges);
}

function trimTrailingEmptyLines(lines: string[]): string[] {
	let end = lines.length;
	while (end > 0 && lines[end - 1] === "") {
		end--;
	}
	return lines.slice(0, end);
}

function getNonVisionImageNote(model: Model<Api> | undefined): string | undefined {
	if (!model || model.input.includes("image")) {
		return undefined;
	}
	return "[Current model does not support images. The image will be omitted from this request.]";
}

function toPosixPath(filePath: string): string {
	return filePath.split(sep).join("/");
}

function getPiDocsClassification(absolutePath: string): CompactReadClassification | undefined {
	const packageRoot = dirname(getReadmePath());
	const relativePath = relative(resolvePath(packageRoot), resolvePath(absolutePath));
	if (
		relativePath === "" ||
		relativePath === ".." ||
		relativePath.startsWith(`..${sep}`) ||
		isAbsolute(relativePath)
	) {
		return undefined;
	}

	const label = toPosixPath(relativePath);
	if (label === "README.md" || label.startsWith("docs/") || label.startsWith("examples/")) {
		return { kind: "docs", label };
	}
	return undefined;
}

function getCompactReadClassification(
	args: ReadRenderArgs | undefined,
	cwd: string,
): CompactReadClassification | undefined {
	const rawPath = str(args?.file_path ?? args?.path);
	if (!rawPath) return undefined;

	const absolutePath = resolveReadPath(rawPath, cwd);
	const fileName = basename(absolutePath);
	if (fileName === "SKILL.md") {
		return { kind: "skill", label: basename(dirname(absolutePath)) || fileName };
	}

	const docsClassification = getPiDocsClassification(absolutePath);
	if (docsClassification) return docsClassification;

	if (COMPACT_RESOURCE_FILE_NAMES.has(fileName)) {
		return { kind: "resource", label: formatPathRelativeToCwdOrAbsolute(absolutePath, cwd) };
	}

	return undefined;
}

function formatCompactReadCall(
	classification: CompactReadClassification,
	args: ReadRenderArgs | undefined,
	theme: Theme,
): string {
	const expandHint = theme.fg("dim", ` (${keyText("app.tools.expand")} to expand)`);
	if (classification.kind === "skill") {
		return (
			theme.fg("customMessageLabel", `\x1b[1m[skill]\x1b[22m `) +
			theme.fg("customMessageText", classification.label) +
			formatReadLineRange(args, theme) +
			expandHint
		);
	}

	return (
		theme.fg("toolTitle", theme.bold(`read ${classification.kind}`)) +
		" " +
		theme.fg("accent", classification.label) +
		formatReadLineRange(args, theme) +
		expandHint
	);
}

function formatReadResult(
	args: ReadRenderArgs | undefined,
	result: { content: (TextContent | ImageContent)[]; details?: ReadToolDetails },
	options: ToolRenderResultOptions,
	theme: Theme,
	showImages: boolean,
	cwd: string,
	isError: boolean,
): string {
	if (!options.expanded && !isError && getCompactReadClassification(args, cwd)) {
		return "";
	}

	const rawPath = str(args?.file_path ?? args?.path);
	const output = getTextOutput(result, showImages);
	const lang = rawPath ? getLanguageFromPath(rawPath) : undefined;
	const renderedLines = lang ? highlightCode(replaceTabs(output), lang) : output.split("\n");
	const lines = trimTrailingEmptyLines(renderedLines);
	const maxLines = options.expanded ? lines.length : 10;
	const displayLines = lines.slice(0, maxLines);
	const remaining = lines.length - maxLines;
	let text = `\n${displayLines.map((line) => (lang ? replaceTabs(line) : theme.fg("toolOutput", replaceTabs(line)))).join("\n")}`;
	if (remaining > 0) {
		text += `${theme.fg("muted", `\n... (${remaining} more lines,`)} ${keyHint("app.tools.expand", "to expand")})`;
	}

	const truncation = result.details?.truncation;
	if (truncation?.truncated) {
		if (truncation.firstLineExceedsLimit) {
			text += `\n${theme.fg("warning", `[First line exceeds ${formatSize(truncation.maxBytes ?? DEFAULT_MAX_BYTES)} limit]`)}`;
		} else if (truncation.truncatedBy === "lines") {
			text += `\n${theme.fg("warning", `[Truncated: showing ${truncation.outputLines} of ${truncation.totalLines} lines (${truncation.maxLines ?? DEFAULT_MAX_LINES} line limit)]`)}`;
		} else {
			text += `\n${theme.fg("warning", `[Truncated: ${truncation.outputLines} lines shown (${formatSize(truncation.maxBytes ?? DEFAULT_MAX_BYTES)} limit)]`)}`;
		}
	}
	return text;
}

export function createReadToolDefinition(
	cwd: string,
	options?: ReadToolOptions,
): ToolDefinition<typeof readSchema, ReadToolDetails | undefined> {
	const autoResizeImages = options?.autoResizeImages ?? true;
	const ops = options?.operations ?? defaultReadOperations;
	const coverage = new Map<string, ReadCoverageRange[]>();

	async function readOne(
		{ path, offset, limit }: { path: string; offset?: number; limit?: number },
		signal?: AbortSignal,
		ctx?: { model?: Model<Api> },
	): Promise<SingleReadResult> {
		if (signal?.aborted) throw new Error("Operation aborted");
		const normalized = splitReadPathLineSuffix(path, offset, limit);
		const readPath = normalized.path;
		const readOffset = normalized.offset;
		const readLimit = normalized.limit;
		const absolutePath = resolveReadPath(readPath, cwd);

		await ops.access(absolutePath);
		if (signal?.aborted) throw new Error("Operation aborted");
		const stat = ops.stat ? await ops.stat(absolutePath) : undefined;
		const mimeType = ops.detectImageMimeType ? await ops.detectImageMimeType(absolutePath) : undefined;
		let content: (TextContent | ImageContent)[];
		let details: ReadToolDetails | undefined;
		const nonVisionImageNote = getNonVisionImageNote(ctx?.model);
		if (mimeType) {
			const buffer = await ops.readFile(absolutePath);
			const base64 = buffer.toString("base64");
			if (autoResizeImages) {
				const resized = await resizeImage({ type: "image", data: base64, mimeType });
				if (!resized) {
					let textNote = `Read image file [${mimeType}]\n[Image omitted: could not be resized below the inline image size limit.]`;
					if (nonVisionImageNote) textNote += `\n${nonVisionImageNote}`;
					content = [{ type: "text", text: textNote }];
				} else {
					const dimensionNote = formatDimensionNote(resized);
					let textNote = `Read image file [${resized.mimeType}]`;
					if (dimensionNote) textNote += `\n${dimensionNote}`;
					if (nonVisionImageNote) textNote += `\n${nonVisionImageNote}`;
					content = [
						{ type: "text", text: textNote },
						{ type: "image", data: resized.data, mimeType: resized.mimeType },
					];
				}
			} else {
				let textNote = `Read image file [${mimeType}]`;
				if (nonVisionImageNote) textNote += `\n${nonVisionImageNote}`;
				content = [
					{ type: "text", text: textNote },
					{ type: "image", data: base64, mimeType },
				];
			}
			return { content, details, displayPath: readPath };
		}

		const buffer = await ops.readFile(absolutePath);
		const signature = fileSignature(stat?.size, stat?.mtimeMs, buffer);
		const textContent = buffer.toString("utf-8");
		const allLines = textContent.split("\n");
		const totalFileLines = allLines.length;
		const startLine = readOffset ? Math.max(0, readOffset - 1) : 0;
		const startLineDisplay = startLine + 1;
		if (startLine >= allLines.length) {
			throw new Error(`Offset ${readOffset} is beyond end of file (${allLines.length} lines total)`);
		}

		let selectedContent: string;
		let userLimitedLines: number | undefined;
		if (readLimit !== undefined) {
			const endLine = Math.min(startLine + readLimit, allLines.length);
			selectedContent = allLines.slice(startLine, endLine).join("\n");
			userLimitedLines = endLine - startLine;
		} else {
			selectedContent = allLines.slice(startLine).join("\n");
		}

		const truncation = truncateHead(selectedContent);
		let outputText: string;
		let endLineDisplay: number;
		if (truncation.firstLineExceedsLimit) {
			const firstLineSize = formatSize(Buffer.byteLength(allLines[startLine] ?? "", "utf-8"));
			outputText = `[Line ${startLineDisplay} is ${firstLineSize}, exceeds ${formatSize(DEFAULT_MAX_BYTES)} limit. Use bash: sed -n '${startLineDisplay}p' ${readPath} | head -c ${DEFAULT_MAX_BYTES}]`;
			details = { truncation };
			endLineDisplay = startLineDisplay;
		} else if (truncation.truncated) {
			endLineDisplay = startLineDisplay + truncation.outputLines - 1;
			const nextOffset = endLineDisplay + 1;
			outputText = truncation.content;
			if (truncation.truncatedBy === "lines") {
				outputText += `\n\n[Showing lines ${startLineDisplay}-${endLineDisplay} of ${totalFileLines}. Use offset=${nextOffset} to continue.]`;
			} else {
				outputText += `\n\n[Showing lines ${startLineDisplay}-${endLineDisplay} of ${totalFileLines} (${formatSize(DEFAULT_MAX_BYTES)} limit). Use offset=${nextOffset} to continue.]`;
			}
			details = { truncation };
		} else if (userLimitedLines !== undefined && startLine + userLimitedLines < allLines.length) {
			const remaining = allLines.length - (startLine + userLimitedLines);
			const nextOffset = startLine + userLimitedLines + 1;
			endLineDisplay = startLine + userLimitedLines;
			outputText = `${truncation.content}\n\n[${remaining} more lines in file. Use offset=${nextOffset} to continue.]`;
		} else {
			endLineDisplay = startLine + (userLimitedLines ?? allLines.length - startLine);
			outputText = truncation.content;
		}

		const covered = findCoveredRange(coverage.get(absolutePath), startLineDisplay, endLineDisplay, signature);
		if (covered) {
			outputText = `[Read notice: ${readPath}:${startLineDisplay}-${endLineDisplay} was already read from unchanged file content in lines ${covered.startLine}-${covered.endLine}. Reuse the prior result unless you need a different range after an edit.]\n\n${outputText}`;
		}
		rememberCoverage(coverage, absolutePath, startLineDisplay, endLineDisplay, signature);

		return {
			content: [{ type: "text", text: outputText }],
			details,
			displayPath: readPath,
			startLine: startLineDisplay,
			endLine: endLineDisplay,
		};
	}

	return {
		name: "read",
		label: "read",
		description: `Read the contents of one file or a batch of file ranges. Supports text files and images (jpg, png, gif, webp). Images are sent as attachments. For text files, output is truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). Use items for multi-file inspection. Use offset/limit for large files. When you need the full file, continue with offset until complete.`,
		promptSnippet: "Read file contents",
		promptGuidelines: [
			"Use read to examine files instead of cat or sed.",
			"For multi-file inspection, call read once with items=[...] instead of separate read calls.",
			"Do not reread the same unchanged file range; use offset only to continue into unread ranges.",
		],
		parameters: readSchema,
		async execute(_toolCallId, input: ReadToolInput, signal?: AbortSignal, _onUpdate?, ctx?) {
			if (input.items && input.items.length > 0) {
				const batchContent: (TextContent | ImageContent)[] = [];
				const batchText: string[] = [];
				const itemDetails: NonNullable<ReadToolDetails["items"]> = [];
				for (const item of input.items) {
					const result = await readOne(item, signal, ctx);
					const text = getTextOutput({ content: result.content }, true);
					batchText.push(`${formatBatchHeader(result)}\n${text}`);
					for (const block of result.content) {
						if (block.type === "image") batchContent.push(block);
					}
					itemDetails.push({
						path: result.displayPath,
						startLine: result.startLine,
						endLine: result.endLine,
						truncation: result.details?.truncation,
					});
				}
				return {
					content: [{ type: "text", text: batchText.join("\n\n") }, ...batchContent],
					details: { items: itemDetails },
				};
			}
			if (!input.path) throw new Error("path required unless items is provided");
			const result = await readOne({ path: input.path, offset: input.offset, limit: input.limit }, signal, ctx);
			return { content: result.content, details: result.details };
		},
		renderCall(args, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			const classification = !context.expanded ? getCompactReadClassification(args, context.cwd) : undefined;
			text.setText(
				classification ? formatCompactReadCall(classification, args, theme) : formatReadCall(args, theme),
			);
			return text;
		},
		renderResult(result, options, theme, context) {
			const text = (context.lastComponent as Text | undefined) ?? new Text("", 0, 0);
			text.setText(
				formatReadResult(context.args, result, options, theme, context.showImages, context.cwd, context.isError),
			);
			return text;
		},
	};
}

export function createReadTool(cwd: string, options?: ReadToolOptions): AgentTool<typeof readSchema> {
	return wrapToolDefinition(createReadToolDefinition(cwd, options));
}
