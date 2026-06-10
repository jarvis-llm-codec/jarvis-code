/**
 * jarvis-image.ts — image generation/editing tools ( generate_image · edit_image )
 *
 * Gives JARVIS Code one more brush. It adds image tools beside the coding
 * tools (read/write/edit/bash), so images can be created during a conversation
 * and saved into the workspace.
 * The core stays untouched — this is a pure extension like jarvis-face.
 *
 * provider = NVIDIA NIM (build.nvidia.com). Reuses the existing nvapi key.
 * text→image works immediately with FLUX.1-dev (no 429s, best quality).
 * Game sprite "sheets" draw multiple frames together in one image, so
 * frame-to-frame consistency comes naturally. Reference edits such as
 * "add a new pose to an existing character" use the FLUX.1 Kontext
 * asset-upload flow.
 *
 * Design principles:
 *  - Keep payloads small: save generated files to disk and never put base64 in
 *    the model context. Return only saved paths plus metadata (bytes/model/seed).
 *  - Prefer environment variables for keys, with credentials.yaml as a fallback
 *    (matching the sidecar). No keys are hardcoded in code or the repo.
 *  - Make failures readable: non-200 responses return the status code plus a
 *    short body summary gracefully.
 *
 * Environment variables (key lookup order):
 *  NVIDIA_API_KEY → NIM_API_KEY → NIM_API_KEY_1 → NIM_API_KEY_2,
 *  then the same names in the env block of credentials.yaml.
 *  JLC_IMAGE_MODEL      (optional) — override the default text→image model
 *  JLC_IMAGE_EDIT_MODEL (optional) — override the default edit model
 *  JLC_IMAGE_DIR        (optional) — save folder when output_path is omitted (default ~/.jarvis-code/jlc-images)
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { defineTool, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const GENAI_BASE = "https://ai.api.nvidia.com/v1/genai";
const ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets";
const VERSION_FILE = "jarvis_version.json";
const FALLBACK_JARVIS_VERSION = "1.01.0";
// flux.1-dev = full (non-distilled) model. It gives the cleanest quality (A/B decision, 2026-06-10).
// For richer or faster output, choose flux.2-klein-4b / flux.1-schnell with the model parameter.
const DEFAULT_MODEL = "black-forest-labs/flux.1-dev";
const DEFAULT_EDIT_MODEL = "black-forest-labs/flux.1-kontext-dev";
const MISSING_NVIDIA_KEY_MESSAGE =
	"Image generation is off. Add your NVIDIA key: /api-key → NVIDIA NIM (free tier: build.nvidia.com)";

// Width/height grid allowed by FLUX.1-dev — other values return 422, so snap to the nearest value.
const ALLOWED_DIMS = [768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344];

const KEY_ENV_NAMES = ["NVIDIA_API_KEY", "NIM_API_KEY", "NIM_API_KEY_1", "NIM_API_KEY_2"];
let cachedJarvisUserAgent: string | undefined;

interface NvArtifact {
	base64: string;
	finishReason?: string;
	seed?: number;
}
interface NvResponse {
	artifacts?: NvArtifact[];
	detail?: unknown;
	error?: { message?: string; code?: number };
}

function findRepoRoot(start: string): string | undefined {
	let current = path.resolve(start);
	for (;;) {
		if (fs.existsSync(path.join(current, VERSION_FILE))) return current;
		const parent = path.dirname(current);
		if (parent === current) return undefined;
		current = parent;
	}
}

function readJarvisVersion(): string {
	const starts = [
		process.env.JARVIS_CODE_ROOT,
		process.cwd(),
		__dirname,
		path.resolve(__dirname, "../../../.."),
		path.resolve(__dirname, "../../../../.."),
	].filter((candidate): candidate is string => Boolean(candidate?.trim()));
	for (const start of starts) {
		const root = findRepoRoot(start);
		if (!root) continue;
		try {
			const raw = JSON.parse(fs.readFileSync(path.join(root, VERSION_FILE), "utf-8")) as { version?: unknown };
			const version = typeof raw.version === "string" ? raw.version.trim() : "";
			if (/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(version)) return version;
		} catch {
			/* next candidate */
		}
	}
	return FALLBACK_JARVIS_VERSION;
}

function jarvisUserAgent(): string {
	if (!cachedJarvisUserAgent) {
		cachedJarvisUserAgent = `jarvis-code/${readJarvisVersion()} (pi-agent)`;
	}
	return cachedJarvisUserAgent;
}

function withJarvisUserAgent(headers: Record<string, string>): Record<string, string> {
	const next = { ...headers };
	const key = Object.keys(next).find((name) => name.toLowerCase() === "user-agent");
	if (!key) {
		next["User-Agent"] = jarvisUserAgent();
		return next;
	}
	if (!next[key].includes(jarvisUserAgent())) {
		next[key] = `${next[key]} ${jarvisUserAgent()}`.trim();
	}
	return next;
}

/** Finds an nvapi key from environment variables, then credentials.yaml (env block). */
function resolveApiKey(): string | undefined {
	for (const name of KEY_ENV_NAMES) {
		const v = process.env[name]?.trim();
		if (v) return v;
	}
	for (const name of KEY_ENV_NAMES) {
		const v = readKeyFromCredentials(name);
		if (v) return v;
	}
	return undefined;
}

/**
 * Reads keys from the env block of credentials.yaml. It checks the same places
 * as the sidecar's credentials_path(): the sibling of JARVIS_CODE_CONFIG, then
 * common fallback candidates. Uses a simple regex instead of a YAML dependency;
 * the env block has the fixed `  KEY: value` format.
 */
function readKeyFromCredentials(key: string): string | undefined {
	const candidates: string[] = [];
	const cfg = process.env.JARVIS_CODE_CONFIG?.trim();
	if (cfg) candidates.push(path.join(path.dirname(cfg), "credentials.yaml"));
	const home = process.env.USERPROFILE || process.env.HOME;
	if (home) candidates.push(path.join(home, ".jarvis-code", "credentials.yaml"));
	candidates.push(path.resolve(process.cwd(), "data", "credentials.yaml"));
	const re = new RegExp(`^\\s*${key}\\s*:\\s*["']?([^"'#\\r\\n]+?)["']?\\s*$`, "m");
	for (const file of candidates) {
		try {
			const text = fs.readFileSync(file, "utf-8");
			const m = re.exec(text);
			if (m?.[1]) return m[1].trim();
		} catch {
			/* next candidate */
		}
	}
	return undefined;
}

/**
 * Each model has its own allowed steps/cfg_scale range. Distilled models
 * (klein/schnell) use only a few steps, and klein fixes cfg_scale at 1.
 * Invalid values return 422, so defaults and user values are clamped per model.
 */
function resolveSampling(model: string, steps?: number, cfg?: number): { steps: number; cfg_scale: number } {
	const m = model.toLowerCase();
	if (m.includes("klein")) {
		return { steps: Math.min(Math.max(Math.round(steps ?? 4), 1), 4), cfg_scale: 1 }; // klein: steps≤4, cfg=1 fixed
	}
	if (m.includes("schnell")) {
		return { steps: Math.min(Math.max(Math.round(steps ?? 4), 1), 8), cfg_scale: cfg ?? 0 }; // schnell: few steps, cfg 0
	}
	return { steps: steps ?? 40, cfg_scale: cfg ?? 3.5 }; // full models such as flux.1-dev
}

function snapDim(n: number | undefined, fallback: number): number {
	if (!n || !Number.isFinite(n)) return fallback;
	let best = ALLOWED_DIMS[0];
	for (const d of ALLOWED_DIMS) if (Math.abs(d - n) < Math.abs(best - n)) best = d;
	return best;
}

/** Maps actual format to extension from base64 magic bytes. NVIDIA FLUX usually returns JPEG. */
function extForBase64(b64: string): string {
	const head = Buffer.from(b64.slice(0, 16), "base64");
	if (head[0] === 0xff && head[1] === 0xd8) return ".jpg";
	if (head[0] === 0x89 && head[1] === 0x50) return ".png";
	if (head[0] === 0x52 && head[1] === 0x49) return ".webp"; // RIFF
	return ".jpg";
}

function mimeForPath(p: string): string {
	const ext = path.extname(p).toLowerCase();
	if (ext === ".png") return "image/png";
	if (ext === ".webp") return "image/webp";
	return "image/jpeg";
}

function slugify(s: string): string {
	return (
		s
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, "-")
			.replace(/^-+|-+$/g, "")
			.slice(0, 40) || "image"
	);
}

function resolveCwd(ctx: ExtensionContext): string {
	// cwd can throw in a stale runtime — same defense as jarvis-jlc.
	try {
		return ctx.cwd;
	} catch {
		return process.cwd();
	}
}

/** Saves artifacts[].base64 to disk and returns paths relative to cwd. */
function saveArtifacts(
	arts: NvArtifact[],
	cwd: string,
	prompt: string,
	outputPath: string | undefined,
): { abs: string[]; rel: string[]; bytes: number; seeds: (number | undefined)[] } {
	// Default to one global location (~/.jarvis-code/jlc-images), following dev-tool
	// conventions (~/.ollama, ~/.aws) by grouping under a hidden home directory.
	// Casual "draw this" requests do not scatter files into the active cwd or
	// pollute personal Pictures folders that may sync to cloud storage. For project
	// assets, the agent should pass output_path (e.g. ./assets/x.png) or copy files;
	// that is agent work, not tool work. JLC_IMAGE_DIR can pin a different location.
	const baseDir = process.env.JLC_IMAGE_DIR || path.join(os.homedir(), ".jarvis-code", "jlc-images");
	const ts = Date.now();
	const abs: string[] = [];
	const seeds: (number | undefined)[] = [];
	for (let i = 0; i < arts.length; i++) {
		const a = arts[i];
		const ext = extForBase64(a.base64);
		let outAbs: string;
		if (outputPath) {
			const given = path.isAbsolute(outputPath) ? outputPath : path.resolve(cwd, outputPath);
			if (arts.length === 1) outAbs = given;
			else {
				const e = path.extname(given) || ext;
				outAbs = `${given.slice(0, given.length - path.extname(given).length)}-${i + 1}${e}`;
			}
		} else {
			outAbs = path.join(baseDir, `${slugify(prompt)}-${ts}${arts.length > 1 ? `-${i + 1}` : ""}${ext}`);
		}
		fs.mkdirSync(path.dirname(outAbs), { recursive: true });
		fs.writeFileSync(outAbs, Buffer.from(a.base64, "base64"));
		abs.push(outAbs);
		seeds.push(a.seed);
	}
	// Show relative paths inside cwd, absolute paths outside it (global output, etc.).
	const rel = abs.map((p) => {
		const r = path.relative(cwd, p);
		return r && !r.startsWith("..") ? r : p;
	});
	const bytes = abs.reduce((n, p) => n + fs.statSync(p).size, 0);
	return { abs, rel, bytes, seeds };
}

function errorResult(text: string, error: string, extra?: Record<string, unknown>) {
	return {
		content: [{ type: "text" as const, text }],
		details: { ok: false, error, ...extra },
	};
}

async function readBodyText(res: Response): Promise<string> {
	try {
		return await res.text();
	} catch {
		return "";
	}
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve, reject) => {
		if (signal?.aborted) return reject(new Error("aborted"));
		const t = setTimeout(resolve, ms);
		signal?.addEventListener("abort", () => {
			clearTimeout(t);
			reject(new Error("aborted"));
		});
	});
}

interface GenAIOutcome {
	ok: boolean;
	status: number;
	json: NvResponse | null;
	bodyText: string;
	networkError?: string;
}

/**
 * GenAI POST with light retry. The hosted endpoints occasionally throw a
 * transient 5xx / hang on a cold call (witnessed live), so a couple of retries
 * keep a user's first request from failing on a blip. 4xx is returned as-is.
 */
async function postGenAI(
	model: string,
	apiKey: string,
	body: unknown,
	signal: AbortSignal | undefined,
	extraHeaders: Record<string, string> = {},
): Promise<GenAIOutcome> {
	const headers = withJarvisUserAgent({
		Authorization: `Bearer ${apiKey}`,
		"Content-Type": "application/json",
		Accept: "application/json",
		...extraHeaders,
	});
	const payload = JSON.stringify(body);
	let last: GenAIOutcome = { ok: false, status: 0, json: null, bodyText: "", networkError: "no attempt" };
	for (let attempt = 0; attempt < 3; attempt++) {
		try {
			const res = await fetch(`${GENAI_BASE}/${model}`, { method: "POST", headers, body: payload, signal });
			if (res.ok)
				return {
					ok: true,
					status: res.status,
					json: (await res.json().catch(() => null)) as NvResponse,
					bodyText: "",
				};
			const bodyText = (await readBodyText(res)).slice(0, 300);
			last = { ok: false, status: res.status, json: null, bodyText };
			if (res.status >= 500 && attempt < 2) {
				await sleep(1500 * (attempt + 1), signal);
				continue;
			}
			return last;
		} catch (e) {
			last = { ok: false, status: 0, json: null, bodyText: "", networkError: (e as Error).message };
			if (attempt < 2 && !signal?.aborted) {
				await sleep(1500 * (attempt + 1), signal);
				continue;
			}
			return last;
		}
	}
	return last;
}

/** NVIDIA NVCF asset upload: create asset → PUT to presigned URL. Returns assetId. */
async function uploadAsset(apiKey: string, bytes: Buffer, mime: string, signal?: AbortSignal): Promise<string> {
	const createRes = await fetch(ASSETS_URL, {
		method: "POST",
		headers: withJarvisUserAgent({
			Authorization: `Bearer ${apiKey}`,
			"Content-Type": "application/json",
			accept: "application/json",
		}),
		body: JSON.stringify({ contentType: mime, description: "jarvis-image reference" }),
		signal,
	});
	if (!createRes.ok) throw new Error(`asset create failed (${createRes.status}): ${await readBodyText(createRes)}`);
	const { assetId, uploadUrl } = (await createRes.json()) as { assetId: string; uploadUrl: string };
	const putRes = await fetch(uploadUrl, {
		method: "PUT",
		headers: withJarvisUserAgent({
			"Content-Type": mime,
			"x-amz-meta-nvcf-asset-description": "jarvis-image reference",
		}),
		body: bytes,
		signal,
	});
	if (!putRes.ok) throw new Error(`asset upload failed (${putRes.status})`);
	return assetId;
}

export const generateImageTool = defineTool({
	name: "generate_image",
	label: "Generate image",
	description:
		"Generate an image from a text prompt using NVIDIA NIM (FLUX), saving file(s) to the workspace. " +
		"For game sprite work, ask for a full sprite sheet in ONE prompt (e.g. '4 walk-cycle frames in a row, " +
		"identical character in every frame') — frames drawn together stay consistent. Returns the saved path(s); " +
		"image bytes are written to disk, not into the conversation. To restyle/repose an EXISTING image, use edit_image.",
	promptSnippet: "generate_image: text→image via NVIDIA FLUX (sprite sheets = ask for all frames in one prompt)",
	promptGuidelines: [
		"By default images save to a global ~/.jarvis-code/jlc-images folder. When generating an asset for the current project, pass output_path (e.g. ./assets/sprite.png) so it lands in the project directly.",
		"For multi-frame sprites, request the whole sheet in a single prompt; the model keeps the character consistent within one image.",
		"width/height must be one of 768,832,896,960,1024,1088,1152,1216,1280,1344 — other values are snapped to the nearest allowed size.",
	],
	parameters: Type.Object({
		prompt: Type.String({ description: "What to generate. Be specific about style, layout, background." }),
		output_path: Type.Optional(
			Type.String({
				description: "Where to save (relative to cwd ok). Default: ~/.jarvis-code/jlc-images/<slug>-<ts>.<ext>",
			}),
		),
		width: Type.Optional(Type.Number({ description: "Image width (snapped to allowed grid). Default 1024." })),
		height: Type.Optional(Type.Number({ description: "Image height (snapped to allowed grid). Default 1024." })),
		steps: Type.Optional(
			Type.Number({ description: "Sampling steps (auto-clamped per model: klein/schnell ≤4, flux.1-dev ~40)." }),
		),
		cfg_scale: Type.Optional(
			Type.Number({ description: "Prompt adherence (klein forces 1, schnell 0, flux.1-dev ~3.5)." }),
		),
		seed: Type.Optional(Type.Number({ description: "Seed for reproducibility. Default 0 (random)." })),
		model: Type.Optional(
			Type.String({
				description: `Override model id. Default ${DEFAULT_MODEL}. Options: flux.2-klein-4b (fast/rich), flux.1-dev (clean), flux.1-schnell (fastest).`,
			}),
		),
	}),

	async execute(_toolCallId, params, signal, onUpdate, ctx: ExtensionContext) {
		const apiKey = resolveApiKey();
		if (!apiKey) {
			return errorResult(MISSING_NVIDIA_KEY_MESSAGE, "missing_api_key");
		}
		const cwd = resolveCwd(ctx);
		const model = (params.model || process.env.JLC_IMAGE_MODEL || DEFAULT_MODEL).trim();
		const width = snapDim(params.width, 1024);
		const height = snapDim(params.height, 1024);
		const { steps, cfg_scale } = resolveSampling(model, params.steps, params.cfg_scale);

		onUpdate?.({ content: [{ type: "text", text: `generating with ${model} (${width}×${height})…` }] } as never);
		const r = await postGenAI(
			model,
			apiKey,
			{
				prompt: params.prompt,
				width,
				height,
				steps,
				cfg_scale,
				seed: params.seed ?? 0,
			},
			signal,
		);
		if (r.networkError) return errorResult(`Network error calling NVIDIA: ${r.networkError}`, "network", { model });
		if (!r.ok) {
			return errorResult(`NVIDIA image request failed (${r.status}): ${r.bodyText}`, "request_failed", {
				model,
				status: r.status,
			});
		}
		const json = r.json ?? {};
		const arts = json.artifacts?.filter((a) => a.base64) ?? [];
		if (arts.length === 0) {
			return errorResult(
				`Model returned no image. ${JSON.stringify(json.detail ?? json.error ?? {})}`.slice(0, 300),
				"no_image",
				{
					model,
				},
			);
		}

		const { rel, bytes, seeds } = saveArtifacts(arts, cwd, params.prompt, params.output_path);
		return {
			content: [
				{
					type: "text" as const,
					text:
						`Generated ${rel.length} image(s) with ${model} (${width}×${height}):\n` +
						rel.map((p, i) => `  - ${p}${seeds[i] != null ? ` (seed ${seeds[i]})` : ""}`).join("\n") +
						`\n(${(bytes / 1024).toFixed(1)} KB total, saved to disk)`,
				},
			],
			details: { ok: true, model, paths: rel, bytes, seeds },
		};
	},
});

export const editImageTool = defineTool({
	name: "edit_image",
	label: "Edit image",
	description:
		"Edit/restyle/repose an EXISTING image while keeping the same subject, using NVIDIA FLUX Kontext " +
		"(in-context editing). Pass `image` = path to the source image and a prompt describing the change " +
		"(e.g. 'same character, jumping pose'). Use this to add new sprite poses that match a base character. " +
		"Saves the result to disk and returns its path.",
	promptSnippet: "edit_image: reference-consistent edit of an existing image via NVIDIA FLUX Kontext",
	parameters: Type.Object({
		prompt: Type.String({ description: "How to change the image (the subject is preserved)." }),
		image: Type.String({ description: "Path to the source image (relative to cwd ok)." }),
		output_path: Type.Optional(
			Type.String({ description: "Where to save. Default: ~/.jarvis-code/jlc-images/<slug>-<ts>.<ext>" }),
		),
		steps: Type.Optional(Type.Number({ description: "Sampling steps. Default 30." })),
		cfg_scale: Type.Optional(Type.Number({ description: "Edit strength (>1). Default 3.5." })),
		seed: Type.Optional(Type.Number({ description: "Seed. Default 0." })),
		model: Type.Optional(Type.String({ description: `Override model id. Default ${DEFAULT_EDIT_MODEL}.` })),
	}),

	async execute(_toolCallId, params, signal, onUpdate, ctx: ExtensionContext) {
		const apiKey = resolveApiKey();
		if (!apiKey) {
			return errorResult(MISSING_NVIDIA_KEY_MESSAGE, "missing_api_key");
		}
		const cwd = resolveCwd(ctx);
		const model = (params.model || process.env.JLC_IMAGE_EDIT_MODEL || DEFAULT_EDIT_MODEL).trim();

		const srcAbs = path.isAbsolute(params.image) ? params.image : path.resolve(cwd, params.image);
		let srcBytes: Buffer;
		try {
			srcBytes = fs.readFileSync(srcAbs);
		} catch (e) {
			return errorResult(`Source image not readable: ${params.image} (${(e as Error).message})`, "bad_source");
		}
		const mime = mimeForPath(srcAbs);

		let assetId: string;
		try {
			onUpdate?.({ content: [{ type: "text", text: "uploading reference image…" }] } as never);
			assetId = await uploadAsset(apiKey, srcBytes, mime, signal);
		} catch (e) {
			return errorResult(`Reference upload failed: ${(e as Error).message}`, "asset_upload", { model });
		}

		onUpdate?.({ content: [{ type: "text", text: `editing with ${model}…` }] } as never);
		const r = await postGenAI(
			model,
			apiKey,
			{
				prompt: params.prompt,
				// NVIDIA asset reference format: data:<mime>;example_id,<assetId>
				image: `data:${mime};example_id,${assetId}`,
				cfg_scale: params.cfg_scale ?? 3.5,
				steps: params.steps ?? 30,
				seed: params.seed ?? 0,
			},
			signal,
			{ "NVCF-INPUT-ASSET-REFERENCES": assetId },
		);
		if (r.networkError) return errorResult(`Network error calling NVIDIA: ${r.networkError}`, "network", { model });
		if (!r.ok) {
			return errorResult(
				`NVIDIA edit request failed (${r.status}): ${r.bodyText}` +
					(r.status >= 500
						? " — the FLUX Kontext edit endpoint may be gated on the free tier; sprite sheets via generate_image are the reliable path."
						: ""),
				"request_failed",
				{ model, status: r.status },
			);
		}
		const json = r.json ?? {};
		const arts = json.artifacts?.filter((a) => a.base64) ?? [];
		if (arts.length === 0) {
			return errorResult(
				`Edit returned no image. ${JSON.stringify(json.detail ?? json.error ?? {})}`.slice(0, 300),
				"no_image",
				{
					model,
				},
			);
		}

		const { rel, bytes } = saveArtifacts(arts, cwd, params.prompt, params.output_path);
		return {
			content: [
				{
					type: "text" as const,
					text:
						`Edited image saved with ${model}:\n` +
						rel.map((p) => `  - ${p}`).join("\n") +
						`\n(${(bytes / 1024).toFixed(1)} KB)`,
				},
			],
			details: { ok: true, model, paths: rel, bytes },
		};
	},
});

export default function jarvisImage(pi: ExtensionAPI) {
	pi.registerTool(generateImageTool);
	pi.registerTool(editImageTool);
}
