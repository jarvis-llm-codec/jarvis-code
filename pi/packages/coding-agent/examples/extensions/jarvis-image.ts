/**
 * jarvis-image.ts — 이미지 생성/편집 도구 ( generate_image · edit_image )
 *
 * 자비스코드의 손에 붓을 하나 더 쥐여준다. 코딩 도구(read/write/edit/bash)
 * 옆에 이미지 도구를 얹어, 대화 중에 이미지를 만들고 워크스페이스에 저장한다.
 * 코어는 건드리지 않는다 — jarvis-face와 같은 순수 확장.
 *
 * provider = NVIDIA NIM (build.nvidia.com). Jun이 이미 쓰는 nvapi 키를 그대로
 * 재사용한다. text→image는 FLUX.1-dev로 즉시 되고(429 없음, 품질 최상),
 * 게임 스프라이트 "시트"는 한 장 안에 여러 프레임을 함께 그려 프레임 간
 * 일관성이 저절로 잡힌다. "기존 캐릭터에 새 포즈 추가" 같은 레퍼런스 편집은
 * FLUX.1 Kontext의 asset-upload 플로우로 간다.
 *
 * 설계 원칙:
 *  - 크기로 막는다: 생성물을 디스크에 저장하고, 모델 컨텍스트엔 base64를 절대
 *    안 넣는다. 반환값은 저장 경로 + 메타데이터(바이트/모델/시드)뿐.
 *  - 키는 환경변수 우선, credentials.yaml 폴백 (사이드카와 대칭). 코드/레포에
 *    키 하드코딩 0.
 *  - 실패는 사람이 읽을 수 있게: 비-200은 상태코드 + 본문 요약으로 graceful.
 *
 * 환경변수 (키 탐색 순서):
 *  NVIDIA_API_KEY → NIM_API_KEY → NIM_API_KEY_1 → NIM_API_KEY_2,
 *  없으면 credentials.yaml의 env 블록에서 같은 이름들을 찾는다.
 *  JLC_IMAGE_MODEL      (선택) — text→image 기본 모델 덮어쓰기
 *  JLC_IMAGE_EDIT_MODEL (선택) — 편집 기본 모델 덮어쓰기
 *  JLC_IMAGE_DIR        (선택) — output_path 미지정 시 저장 폴더 (기본 ~/.jarvis-code/jlc-images)
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { defineTool, type ExtensionAPI, type ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const GENAI_BASE = "https://ai.api.nvidia.com/v1/genai";
const ASSETS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/assets";
// flux.1-dev = 풀(비증류) 모델. 정돈된 품질이 가장 낫다 (Jun A/B 결정, 2026-06-10).
// 화려/고속이 필요하면 model 파라미터로 flux.2-klein-4b / flux.1-schnell 선택.
const DEFAULT_MODEL = "black-forest-labs/flux.1-dev";
const DEFAULT_EDIT_MODEL = "black-forest-labs/flux.1-kontext-dev";

// FLUX.1-dev가 허용하는 width/height 격자 — 그 밖의 값은 422. 가장 가까운 값으로 스냅.
const ALLOWED_DIMS = [768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344];

const KEY_ENV_NAMES = ["NVIDIA_API_KEY", "NIM_API_KEY", "NIM_API_KEY_1", "NIM_API_KEY_2"];

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

/** 환경변수 → credentials.yaml(env 블록) 순으로 nvapi 키를 찾는다. */
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
 * credentials.yaml의 env 블록에서 키를 읽는다. 사이드카의 credentials_path()와
 * 같은 자리를 본다: JARVIS_CODE_CONFIG의 sibling, 없으면 흔한 후보들.
 * yaml 의존성 없이 단순 정규식 — env 블록은 `  KEY: value` 고정 포맷이다.
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
			/* 다음 후보 */
		}
	}
	return undefined;
}

/**
 * 모델마다 허용 steps/cfg_scale 범위가 다르다 — 증류 모델(klein/schnell)은 소수
 * 스텝 전용이고 klein은 cfg_scale이 1로 고정이다. 안 맞으면 422. 모델에 맞게
 * 기본값을 주고 사용자 값도 유효범위로 클램프한다.
 */
function resolveSampling(model: string, steps?: number, cfg?: number): { steps: number; cfg_scale: number } {
	const m = model.toLowerCase();
	if (m.includes("klein")) {
		return { steps: Math.min(Math.max(Math.round(steps ?? 4), 1), 4), cfg_scale: 1 }; // klein: steps≤4, cfg=1 고정
	}
	if (m.includes("schnell")) {
		return { steps: Math.min(Math.max(Math.round(steps ?? 4), 1), 8), cfg_scale: cfg ?? 0 }; // schnell: 소수 스텝, cfg 0
	}
	return { steps: steps ?? 40, cfg_scale: cfg ?? 3.5 }; // flux.1-dev 등 풀 모델
}

function snapDim(n: number | undefined, fallback: number): number {
	if (!n || !Number.isFinite(n)) return fallback;
	let best = ALLOWED_DIMS[0];
	for (const d of ALLOWED_DIMS) if (Math.abs(d - n) < Math.abs(best - n)) best = d;
	return best;
}

/** base64 매직바이트로 실제 포맷 → 확장자. NVIDIA FLUX는 보통 JPEG를 돌려준다. */
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
	// cwd는 stale 런타임에서 throw할 수 있다 — jarvis-jlc와 같은 방어.
	try {
		return ctx.cwd;
	} catch {
		return process.cwd();
	}
}

/** artifacts[].base64 들을 디스크에 저장하고 cwd 상대경로 목록을 돌려준다. */
function saveArtifacts(
	arts: NvArtifact[],
	cwd: string,
	prompt: string,
	outputPath: string | undefined,
): { abs: string[]; rel: string[]; bytes: number; seeds: (number | undefined)[] } {
	// 기본은 글로벌 한 곳(~/.jarvis-code/jlc-images) — dev 툴 관행(~/.ollama, ~/.aws)대로
	// 숨김 홈 dir에 그룹화. "그냥 그려줘"가 켠 위치에 안 흩어지고, 개인 Pictures(클라우드
	// 동기화)도 안 오염시킨다. 프로젝트 자산은 에이전트가 output_path로(예 ./assets/x.png)
	// 주거나 복사 — 그건 툴이 아니라 에이전트 일. JLC_IMAGE_DIR로 다른 곳 고정 가능.
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
				outAbs = given.slice(0, given.length - path.extname(given).length) + `-${i + 1}` + e;
			}
		} else {
			outAbs = path.join(baseDir, `${slugify(prompt)}-${ts}${arts.length > 1 ? `-${i + 1}` : ""}${ext}`);
		}
		fs.mkdirSync(path.dirname(outAbs), { recursive: true });
		fs.writeFileSync(outAbs, Buffer.from(a.base64, "base64"));
		abs.push(outAbs);
		seeds.push(a.seed);
	}
	// cwd 안이면 상대경로, 밖(글로벌 등)이면 절대경로로 보여준다.
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
	const headers = {
		Authorization: `Bearer ${apiKey}`,
		"Content-Type": "application/json",
		Accept: "application/json",
		...extraHeaders,
	};
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

/** NVIDIA NVCF 자산 업로드: 자산 생성 → presigned URL에 PUT. assetId 반환. */
async function uploadAsset(apiKey: string, bytes: Buffer, mime: string, signal?: AbortSignal): Promise<string> {
	const createRes = await fetch(ASSETS_URL, {
		method: "POST",
		headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json", accept: "application/json" },
		body: JSON.stringify({ contentType: mime, description: "jarvis-image reference" }),
		signal,
	});
	if (!createRes.ok) throw new Error(`asset create failed (${createRes.status}): ${await readBodyText(createRes)}`);
	const { assetId, uploadUrl } = (await createRes.json()) as { assetId: string; uploadUrl: string };
	const putRes = await fetch(uploadUrl, {
		method: "PUT",
		headers: { "Content-Type": mime, "x-amz-meta-nvcf-asset-description": "jarvis-image reference" },
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
			return errorResult(
				"No NVIDIA API key found. Set NVIDIA_API_KEY (or NIM_API_KEY) in the environment, " +
					"or add it under the `env:` block of your jarvis-code credentials.yaml.",
				"missing_api_key",
			);
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
			return errorResult(
				"No NVIDIA API key found. Set NVIDIA_API_KEY (or NIM_API_KEY) in the environment, " +
					"or add it under the `env:` block of your jarvis-code credentials.yaml.",
				"missing_api_key",
			);
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
				// NVIDIA asset 참조 포맷: data:<mime>;example_id,<assetId>
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
