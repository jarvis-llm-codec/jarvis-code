import { APP_NAME } from "../config.js";
import type { SourceInfo } from "./source-info.js";

export type SlashCommandSource = "extension" | "prompt" | "skill";

export interface SlashCommandInfo {
	name: string;
	description?: string;
	source: SlashCommandSource;
	sourceInfo: SourceInfo;
}

export interface BuiltinSlashCommand {
	name: string;
	description: string;
}

export interface JarvisBlockedBuiltinSlashCommand {
	name: string;
	message: string;
}

export const JARVIS_HIDDEN_ALLOWED_BUILTIN_SLASH_COMMANDS: ReadonlyArray<string> = [
	"export",
	"name",
	"session",
	"changelog",
	"new",
	"resume",
];

export const JARVIS_BLOCKED_BUILTIN_SLASH_COMMANDS: ReadonlyArray<JarvisBlockedBuiltinSlashCommand> = [
	{
		name: "model",
		message:
			"Use /model-setting to select JARVIS models. The Pi /model command is disabled so model roles stay in sync.",
	},
	{
		name: "login",
		message:
			"Use /api-key or /gpt-login in JARVIS Code. The Pi /login command is disabled so auth stays in the JARVIS provider config.",
	},
	{
		name: "logout",
		message:
			"Use /gpt-logout for GPT OAuth, or remove saved API keys through JARVIS settings. The Pi /logout command is disabled.",
	},
	{
		name: "scoped-models",
		message: "Model scope is managed by JARVIS /model-setting. The Pi /scoped-models command is disabled.",
	},
	{
		name: "share",
		message: "Pi session sharing is disabled in JARVIS Code.",
	},
	{
		name: "fork",
		message: "Pi session branching is disabled in JARVIS Code because it does not branch JLC memory.",
	},
	{
		name: "clone",
		message: "Pi session cloning is disabled in JARVIS Code because it does not clone JLC memory.",
	},
	{
		name: "tree",
		message: "Pi session tree navigation is disabled in JARVIS Code because it can desync JLC memory.",
	},
	{
		name: "import",
		message: "Pi session import is disabled in JARVIS Code because imported JSONL does not rebuild JLC memory.",
	},
	{
		name: "compact",
		message: "Context compaction is disabled in JARVIS Code.",
	},
];

export const BUILTIN_SLASH_COMMANDS: ReadonlyArray<BuiltinSlashCommand> = [
	{ name: "settings", description: "Open settings menu" },
	{ name: "copy", description: "Copy last agent message to clipboard" },
	{ name: "hotkeys", description: "Show all keyboard shortcuts" },
	{ name: "reload", description: "Reload keybindings, extensions, skills, prompts, and themes" },
	{ name: "quit", description: `Quit ${APP_NAME}` },
];

export function matchesBuiltinSlashCommand(text: string, commandName: string): boolean {
	const trimmed = text.trim();
	const prefix = `/${commandName}`;
	return trimmed === prefix || trimmed.startsWith(`${prefix} `);
}

export function getJarvisBlockedBuiltinSlashCommand(text: string): JarvisBlockedBuiltinSlashCommand | undefined {
	return JARVIS_BLOCKED_BUILTIN_SLASH_COMMANDS.find((command) => matchesBuiltinSlashCommand(text, command.name));
}
