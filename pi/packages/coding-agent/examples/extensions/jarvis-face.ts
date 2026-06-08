/**
 * jarvis-face.ts — 자비스 얼굴 ( ○ ~ ○ )
 *
 * "Working..." 스피너 자리를 살아있는 얼굴로 대체한다.
 * 왼쪽 동그라미가 Jun, 오른쪽이 자비스 — 같은 크기, 나란히 선 둘.
 * 가운데 ~ 는 둘 사이를 오가는 신호이자 입.
 *
 * 얼굴은 도구가 아니다: 하네스 이벤트만 보고 그리는 순수 UI.
 * 모델 호출 0, 토큰 0, 턴 0. 헤드리스(runner/rpc)에서는 setWorkingMessage,
 * setWorkingIndicator, setWidget 모두 no-op이라 자율런(auto-prompts)에 영향이 없다.
 *
 * 표정은 전부 실재 이벤트에서 역산 (유령 표정 금지):
 *
 *   대기   ○ ~ ○   입력 대기 — 에디터 위 위젯으로 상주, 불규칙하게 깜빡
 *   생각   ○ . ○   thinking_delta 스트림
 *   생성   ○ ~ ○   text_delta — 말할 때 입이 움직인다 (~ o ~ -)
 *   도구   ● - ●   tool_execution_start → end, 눈 채움 = 집중
 *   삐끗   ○ _ ○   tool_execution_end isError — 회복 가능, 다음 이벤트가 수습
 *   완료   ^ ~ ^   도구 없는 최종 보고 턴의 turn_end
 *   놀람   ! o !   사용자 중단(aborted) — 다음 idle이 잠깐 놀랐다 풀린다
 *   실패   x _ x   assistant stopReason error (provider 사망/run-fatal만)
 *
 * 애니메이션은 pi TUI Loader의 frames/intervalMs 계약 안에서 돈다
 * (자체 stdout/타이머 없음). 깜빡임의 "랜덤한 살아있음"은 불규칙 간격을
 * 프레임 시퀀스에 미리 박아 만든다.
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

interface FaceGlyphs {
	open: string;
	blink: string;
	focus: string;
	happy: string;
	err: string;
}

// 유니코드 우선, 가난한 터미널은 ASCII 폴백 — 그래도 같은 얼굴이다.
const UNICODE_GLYPHS: FaceGlyphs = { open: "◯", blink: "-", focus: "⬤", happy: "^", err: "x" };
const ASCII_GLYPHS: FaceGlyphs = { open: "o", blink: "-", focus: "O", happy: "^", err: "x" };

const FRAME_INTERVAL_MS = 130;
// ×130ms ≈ 2.2s / 3.1s / 3.6s — 일정하면 기계 같고, 들쭉날쭉해야 살아있다.
const BLINK_GAPS = [17, 24, 28];

// 클로드코드풍 shimmer — 글자 위로 빛이 한 칸씩 지나간다.
// 글자 자체가 테마 코랄(오렌지), 그 위로 더 밝은 빛이 지나간다 (Jun 지정).
const SHIMMER_BASE = "\x1b[38;2;217;119;87m"; // 테마 코랄 #D97757 — 글자 본색
const SHIMMER_MID = "\x1b[38;2;240;169;143m"; // 빛의 가장자리 — 밝은 코랄
const SHIMMER_PEAK = "\x1b[38;2;255;231;220m"; // 빛의 중심 — 따뜻한 흰빛
const RESET_FG = "\x1b[39m";
const SHIMMER_TAIL = 4; // 빛이 단어 밖으로 완전히 빠져나간 뒤 다시 들어오는 여유 틱

export function supportsUnicodeFace(env: NodeJS.ProcessEnv = process.env): boolean {
	if (env.JLC_FACE_ASCII) return false; // 강제 폴백 스위치
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

/** 눈을 뜬 채로 있다가 불규칙한 순간에 한 프레임 깜빡인다. */
export function buildBlinkFrames(g: FaceGlyphs, mouth: string): string[] {
	const frames: string[] = [];
	for (const gap of BLINK_GAPS) {
		for (let i = 0; i < gap; i++) frames.push(face(g.open, mouth));
		frames.push(face(g.blink, mouth));
	}
	return frames;
}

/** 생각할 때: 눈은 깜빡이고, 입이 가끔 오물거린다 (. → _ → , ). 긴 thinking이 귀엽게. */
export function buildThinkFrames(g: FaceGlyphs): string[] {
	const blinkAt = new Set([17, 42, 64]); // 눈 깜빡임 (불규칙)
	const mouthMoves = new Map<number, string>([
		// 입 오물거림 — 드물게, 짧게, 두 군데서
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

/** 말할 때: 입이 ~ o ~ - 로 움직이고, 깜빡임도 불규칙하게 섞인다. */
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
// 웜업(boot) 얼굴은 2026-06-07 라이브에서 미표시 + 백그라운드 웜업이 충분해 철회 (Jun 결정).
// 웜업은 침묵 — 실패/degraded 알림(jlc)만 소리 낸다.

/**
 * 입력 대기 중에도 살아있는 얼굴 — 에디터 위 위젯으로 상주한다.
 * (Working 슬롯은 idle엔 존재하지 않으므로 표면이 다르다.)
 * 진짜 랜덤 깜빡임(2.4~5.4s): 렌더는 깜빡임당 2번뿐이라 idle 비용 ≈ 0.
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
		// 일정하면 기계 같고, 들쭉날쭉해야 살아있다.
		blinkTimer = setTimeout(
			() => {
				// 대기 중에 가끔 씩 웃는다 (Jun 요청) — 깜빡임보다 드물고, 오래 머문다.
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
		// 사용자가 멈췄다 — "앗!" 하고 잠깐 놀란 얼굴, 그 다음 평소 대기로.
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
			// 렌더가 싸서 캐시 없음 — 무효화할 것도 없다.
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
	/** 작업형 상태는 라벨 위로 빛이 흐른다 (클로드코드풍 shimmer) */
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

/** 라벨 한 프레임 — pos 위치의 글자가 빛의 중심, 양옆은 코랄, 나머지는 회색. */
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
 * 라벨을 프레임 안에 직접 넣는다 — message는 정적이지만 프레임은 틱마다 갈리므로,
 * 얼굴과 글자 위를 지나는 빛이 한 호흡으로 애니메이션된다.
 * span은 shimmer 사이클의 배수로 맞춰 루프 이음새에서 빛이 안 튄다
 * (깜빡임 패턴의 이음새는 원래 불규칙이라 안 보인다).
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
		// 같은 상태 반복 호출 무시 — text_delta가 토큰마다 와도 애니메이션이 안 끊긴다.
		if (current && current.state === state && current.label === label) return;
		current = { state, label };
		const cfg = states[state];
		const text = label ? `${cfg.label} · ${label}` : cfg.label;
		ctx.ui.setWorkingIndicator({
			frames: buildLabeledFrames(cfg.faces, text, cfg.shimmer),
			intervalMs: FRAME_INTERVAL_MS,
		});
		// 라벨은 프레임 안에 산다 — message 슬롯은 비워서 이중 표기 방지.
		ctx.ui.setWorkingMessage("");
	};

	// abort은 실패가 아니다 — 다음 idle 얼굴을 "앗!"(! o !)로 한 번 놀래킨다.
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
		// 일하는 동안 idle 얼굴은 들어가고 Working 슬롯의 얼굴이 이어받는다.
		hideIdleFace(ctx);
		// 런마다 무조건 재적용 — 세션 리셋이 indicator를 기본 스피너로 되돌려도 자가 복구.
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
			// 회복 가능한 에러 — 우리 판은 자가수정하니까 x_x 아님. 잠깐 찡그리고 다음 이벤트가 수습.
			setFace(ctx, "oops", event.toolName);
		} else {
			setFace(ctx, "think");
		}
	});

	pi.on("turn_end", async (event, ctx) => {
		// 도구 없는 턴 = 최종 보고. 중간 도구 턴에서 완료 표정 금지.
		if ((event.toolResults?.length ?? 0) === 0) setFace(ctx, "done");
	});

	pi.on("message_end", async (event, ctx) => {
		const message = event.message as { role?: string; stopReason?: string };
		if (message.role !== "assistant") return;
		// 진짜 죽음(provider 사망)만 x_x. 사용자 중단은 다음 idle을 놀래킨다.
		if (message.stopReason === "error") setFace(ctx, "fatal");
		else if (message.stopReason === "aborted") pendingIdleReaction = "startled";
	});

	pi.on("agent_end", async (_event, ctx) => {
		current = null;
		// 보고 끝 → 다시 입력 대기 얼굴로.
		showIdleFace(ctx);
	});
}
