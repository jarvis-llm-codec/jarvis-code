/**
 * jarvis-face.ts — JARVIS face ( ○ ~ ○ )
 *
 * Replaces the "Working..." spinner with a living face.
 * The left circle is Jun, the right is JARVIS — same size, standing side by side.
 * The middle ~ is both the signal moving between them and the mouth.
 *
 * The face is not a tool: it is pure UI driven only by harness events.
 * 0 model calls, 0 tokens, 0 turns. In headless mode (runner/rpc),
 * setWorkingMessage, setWorkingIndicator, and setWidget are all no-ops, so
 * autonomous runs (auto-prompts) are unaffected.
 *
 * Every expression is derived from real events (no phantom expressions):
 *
 *   idle      ○ ~ ○   waiting for input — lives above the editor, blinks irregularly
 *   thinking  ○ . ○   thinking_delta stream
 *   writing   ○ ~ ○   text_delta — the mouth moves while speaking (~ o ~ -)
 *   tool      ● - ●   tool_execution_start → end, filled eyes = focus
 *   oops      ○ _ ○   tool_execution_end isError — recoverable, next event settles it
 *   done      ^ ~ ^   turn_end on a final report turn with no tools
 *   startled  ! o !   user abort (aborted) — next idle briefly startles, then settles
 *   fatal     x _ x   assistant stopReason error (provider death/run-fatal only)
 *
 * Animation runs inside the pi TUI Loader frames/intervalMs contract (no own
 * stdout/timers). The "random aliveness" of blinking is baked into frame
 * sequences as irregular gaps.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

interface FaceGlyphs {
	open: string;
	blink: string;
	focus: string;
	happy: string;
	err: string;
}

// Prefer Unicode, with ASCII fallback for limited terminals — still the same face.
const UNICODE_GLYPHS: FaceGlyphs = { open: "◯", blink: "-", focus: "⬤", happy: "^", err: "x" };
const ASCII_GLYPHS: FaceGlyphs = { open: "o", blink: "-", focus: "O", happy: "^", err: "x" };

const FRAME_INTERVAL_MS = 130;
// ×130ms ≈ 2.2s / 3.1s / 3.6s — regular timing feels mechanical; uneven timing feels alive.
const BLINK_GAPS = [17, 24, 28];

// Claude Code-style shimmer — a light passes over the letters one position at a time.
// The text itself is theme coral (orange), with a brighter light moving over it (Jun-specified).
const SHIMMER_BASE = "\x1b[38;2;217;119;87m"; // theme coral #D97757 — natural text color
const SHIMMER_MID = "\x1b[38;2;240;169;143m"; // light edge — bright coral
const SHIMMER_PEAK = "\x1b[38;2;255;231;220m"; // light center — warm white
const RESET_FG = "\x1b[39m";
const SHIMMER_TAIL = 4; // spare ticks for the light to fully leave the word before re-entering

export function supportsUnicodeFace(env: NodeJS.ProcessEnv = process.env): boolean {
	if (env.JLC_FACE_ASCII) return false; // forced fallback switch
	if (process.platform === "win32") {
		return Boolean(
			env.WT_SESSION || // Windows Terminal
				env.TERM_PROGRAM === "vscode" ||
				/UTF-?8/i.test(env.LANG ?? ""),
		);
	}
	return /UTF-?8/i.test(`${env.LC_ALL ?? ""}${env.LC_CTYPE ?? ""}${env.LANG ?? ""}`);
}

function face(eye: string, mouth: string): string {
	return `${eye} ${mouth} ${eye}`;
}

/** Keeps eyes open, then blinks for one frame at irregular moments. */
export function buildBlinkFrames(g: FaceGlyphs, mouth: string): string[] {
	const frames: string[] = [];
	for (const gap of BLINK_GAPS) {
		for (let i = 0; i < gap; i++) frames.push(face(g.open, mouth));
		frames.push(face(g.blink, mouth));
	}
	return frames;
}

/** Thinking: eyes blink and the mouth occasionally mutters (. → _ → , ). Keeps long thinking cute. */
export function buildThinkFrames(g: FaceGlyphs): string[] {
	const blinkAt = new Set([17, 42, 64]); // eye blinks (irregular)
	const mouthMoves = new Map<number, string>([
		// Mouth muttering — rare, brief, in two places.
		[24, "_"],
		[25, "_"],
		[26, ","],
		[54, ","],
		[55, "_"],
	]);
	const frames: string[] = [];
	for (let i = 0; i < 72; i++) {
		const eye = blinkAt.has(i) ? g.blink : g.open;
		frames.push(face(eye, mouthMoves.get(i) ?? "."));
	}
	return frames;
}

/** Speaking: the mouth moves ~ o ~ -, with irregular blinks mixed in. */
export function buildTalkFrames(g: FaceGlyphs): string[] {
	const mouths = ["~", "o", "~", "-"];
	const blinkAt = new Set([13, 37, 58]);
	const frames: string[] = [];
	for (let i = 0; i < 72; i++) {
		const eye = blinkAt.has(i) ? g.blink : g.open;
		frames.push(face(eye, mouths[i % mouths.length] ?? "~"));
	}
	return frames;
}

const IDLE_WIDGET_KEY = "jarvis-face-idle";
// The warmup (boot) face was withdrawn after the 2026-06-07 live run because hidden background warmup was enough (Jun decision).
// Warmup stays silent — only failure/degraded notices (jlc) speak.

/**
 * A living face while waiting for input — it lives as a widget above the editor.
 * (The Working slot does not exist while idle, so the surface is different.)
 * Truly random blinks (2.4~5.4s): only two renders per blink, so idle cost ≈ 0.
 */
export function createIdleFaceComponent(
	tui: { requestRender(): void },
	glyphs: FaceGlyphs,
	opts: { startled?: boolean } = {},
): { render(width: number): string[]; invalidate(): void; dispose(): void } {
	let mood: "open" | "blink" | "smile" | "startled" = opts.startled ? "startled" : "open";
	let blinkTimer: NodeJS.Timeout | undefined;
	let recoverTimer: NodeJS.Timeout | undefined;
	let startledTimer: NodeJS.Timeout | undefined;
	const scheduleBlink = () => {
		// Regular timing feels mechanical; uneven timing feels alive.
		blinkTimer = setTimeout(
			() => {
				// Occasionally smiles while idle (Jun request) — rarer than blinks and held longer.
				const smiling = Math.random() < 0.18;
				mood = smiling ? "smile" : "blink";
				tui.requestRender();
				recoverTimer = setTimeout(
					() => {
						mood = "open";
						tui.requestRender();
						scheduleBlink();
					},
					smiling ? 900 : 110,
				);
			},
			2400 + Math.random() * 3000,
		);
	};
	if (opts.startled) {
		// The user stopped it — briefly show an "oh!" startled face, then return to normal idle.
		startledTimer = setTimeout(() => {
			mood = "open";
			tui.requestRender();
			scheduleBlink();
		}, 1300);
	} else {
		scheduleBlink();
	}
	return {
		render() {
			if (mood === "startled") return [` ${face("!", "o")}`];
			const eye = mood === "smile" ? glyphs.happy : mood === "blink" ? glyphs.blink : glyphs.open;
			return [` ${face(eye, "~")}`];
		},
		invalidate() {
			// Rendering is cheap, so there is no cache and nothing to invalidate.
		},
		dispose() {
			if (blinkTimer) clearTimeout(blinkTimer);
			if (recoverTimer) clearTimeout(recoverTimer);
			if (startledTimer) clearTimeout(startledTimer);
		},
	};
}

type FaceState = "think" | "gen" | "tool" | "oops" | "done" | "fatal";

interface FaceCfg {
	faces: string[];
	label: string;
	/** Work states run a light over the label (Claude Code-style shimmer). */
	shimmer: boolean;
}

export function buildFaceStates(g: FaceGlyphs): Record<FaceState, FaceCfg> {
	return {
		think: { faces: buildThinkFrames(g), label: "thinking", shimmer: true },
		gen: { faces: buildTalkFrames(g), label: "writing", shimmer: true },
		tool: { faces: [face(g.focus, "-")], label: "working", shimmer: true },
		oops: { faces: [face(g.open, "_")], label: "oops, recovering", shimmer: false },
		done: { faces: [face(g.happy, "~")], label: "done", shimmer: false },
		fatal: { faces: [face(g.err, "_")], label: "failed", shimmer: false },
	};
}

/** One label frame — the letter at pos is the light center; neighbors are coral, rest are gray. */
export function buildShimmerLabel(label: string, pos: number): string {
	let out = "";
	for (let j = 0; j < label.length; j++) {
		const distance = Math.abs(j - pos);
		const color = distance === 0 ? SHIMMER_PEAK : distance === 1 ? SHIMMER_MID : SHIMMER_BASE;
		out += color + label[j];
	}
	return out + RESET_FG;
}

/**
 * Put the label directly into each frame. The message is static, but frames
 * change every tick, so the face and the light crossing the text animate as
 * one motion. Make span a multiple of the shimmer cycle so the light does not
 * jump at the loop seam (blink-pattern seams are already irregular and hidden).
 */
export function buildLabeledFrames(faces: string[], label: string, shimmer: boolean): string[] {
	if (!shimmer) return faces.map((f) => `${f}  ${SHIMMER_BASE}${label}${RESET_FG}`);
	const cycle = label.length + SHIMMER_TAIL;
	const span = Math.ceil(Math.max(faces.length, cycle) / cycle) * cycle;
	const frames: string[] = [];
	for (let i = 0; i < span; i++) {
		frames.push(`${faces[i % faces.length]}  ${buildShimmerLabel(label, i % cycle)}`);
	}
	return frames;
}

export default function jarvisFace(pi: ExtensionAPI) {
	const glyphs = supportsUnicodeFace() ? UNICODE_GLYPHS : ASCII_GLYPHS;
	const states = buildFaceStates(glyphs);

	let current: { state: FaceState; label?: string } | null = null;

	const setFace = (ctx: ExtensionContext, state: FaceState, label?: string) => {
		// Ignore repeated calls for the same state, so token-level text_delta events do not restart animation.
		if (current && current.state === state && current.label === label) return;
		current = { state, label };
		const cfg = states[state];
		const text = label ? `${cfg.label} · ${label}` : cfg.label;
		ctx.ui.setWorkingIndicator({
			frames: buildLabeledFrames(cfg.faces, text, cfg.shimmer),
			intervalMs: FRAME_INTERVAL_MS,
		});
		// The label lives inside frames; keep the message slot empty to avoid duplicate display.
		ctx.ui.setWorkingMessage("");
	};

	// Abort is not failure — startle the next idle face once with "oh!" (! o !).
	let pendingIdleReaction: "startled" | null = null;

	const showIdleFace = (ctx: ExtensionContext) => {
		const startled = pendingIdleReaction === "startled";
		pendingIdleReaction = null;
		ctx.ui.setWidget(IDLE_WIDGET_KEY, (tui) => createIdleFaceComponent(tui, glyphs, { startled }), {
			placement: "aboveEditor",
		});
	};

	const hideIdleFace = (ctx: ExtensionContext) => {
		ctx.ui.setWidget(IDLE_WIDGET_KEY, undefined);
	};

	pi.on("session_start", async (_event, ctx) => {
		showIdleFace(ctx);
	});

	pi.on("agent_start", async (_event, ctx) => {
		// While working, the idle face leaves and the Working-slot face takes over.
		hideIdleFace(ctx);
		// Reapply on every run, so it recovers if a session reset restores the default spinner.
		current = null;
		setFace(ctx, "think");
	});

	pi.on("message_update", async (event, ctx) => {
		const kind = event.assistantMessageEvent?.type;
		if (kind === "thinking_delta") setFace(ctx, "think");
		else if (kind === "text_delta") setFace(ctx, "gen");
	});

	pi.on("tool_execution_start", async (event, ctx) => {
		setFace(ctx, "tool", event.toolName);
	});

	pi.on("tool_execution_end", async (event, ctx) => {
		if (event.isError) {
			// Recoverable error — this board can self-correct, so it is not x_x. Briefly grimace; the next event settles it.
			setFace(ctx, "oops", event.toolName);
		} else {
			setFace(ctx, "think");
		}
	});

	pi.on("turn_end", async (event, ctx) => {
		// A turn with no tools is a final report. Do not show done during intermediate tool turns.
		if ((event.toolResults?.length ?? 0) === 0) setFace(ctx, "done");
	});

	pi.on("message_end", async (event, ctx) => {
		const message = event.message as { role?: string; stopReason?: string };
		if (message.role !== "assistant") return;
		// Only true death (provider failure) gets x_x. User abort startles the next idle face.
		if (message.stopReason === "error") setFace(ctx, "fatal");
		else if (message.stopReason === "aborted") pendingIdleReaction = "startled";
	});

	pi.on("agent_end", async (_event, ctx) => {
		current = null;
		// Report complete → return to the waiting-for-input face.
		showIdleFace(ctx);
	});
}
