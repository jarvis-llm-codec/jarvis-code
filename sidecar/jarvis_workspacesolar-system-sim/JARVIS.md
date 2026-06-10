---
project: jarvis_workspacesolar-system-sim
updated: 2026-06-19T08:42:55+00:00
---
# JARVIS.md — jarvis_workspacesolar-system-sim

## NOW — Current Active Task
<!-- JARVIS_INTERRUPT_CHECKPOINT_BEGIN -->
### Interrupted Turn Checkpoint

Status: interrupted by user ESC. Current work state was saved automatically.
Saved at: 2026-06-19T08:42:55+00:00
Reason: escape_interrupt
Mode: deepdive
CWD: C:\jarvis-code_v1.01\pi

User request:
> 같이 짜보자.

Assistant partial work/state:
> 같이 짜자. 근데 먼저 맞추고 가야 할 게 있어.
>
> 지금 `solar-system-sim`은 이미 구현 완료 상태야(28/28 테스트 통과, 빌드 OK, http://127.0.0.1:5175/ 구동). 그래서 "같이 짜보자"가 어느 쪽인지 헷갈려:

Tool work observed:
- ask_user: ok - started args={"questions":[{"options":["기존 solar-system-sim 개선/기능추가 (별명 라벨, 위성 궤도선, UI 다듬기 등)","새 시뮬레이터 프로젝트 (예: 생태계/유체/도시/파동/DNA/경제/날씨 중 택1)","solar-system-sim을 처음부터 다시 짜기 (리디자인)","다른 아이디어 — 직접 말해줄게"],"question":"무엇을 같이 짜볼까?","recommended":"기존 solar-system...

Resume guidance:
- Treat this as the latest interrupted working state.
- Continue from here unless the user gives a newer direction.
<!-- JARVIS_INTERRUPT_CHECKPOINT_END -->

### Solar System Simulator — IMPLEMENTED (pending cycle-3 review handback)

Status: COMPLETE (local). Build verified, dev server live at http://127.0.0.1:5175/, tests 28/28 pass, tsc --noEmit clean, vite build succeeds.
- Stack: Vite 8.0.16 + Babylon.js 9.13 + TypeScript 5.9 + Vitest 4.1.9.
- Path: C:\jarvis_workspace\solar-system-sim
- Implemented all 7 Must-fix: Vite base './', radius/orbit scale coupling (sun cap + Jupiter ratio cap), Kepler elliptical orbits (Newton-solved), custom ring mesh+radial UV, NASA PD texture sources+LICENSE+procedural fallback (no network startup), moon log-distance scaling, dynamic camera lowerRadiusLimit on focus.
- Implemented Should-fix: ambient/hemi fill light, time scale slider + fwd/pause/rev, drift-guard accumulator state, aria-live info panel + keyboard focus rings, unit tests isolated (node env), procedural starfield skybox, state validate/coerce, log-scale disclosed in info panel.
- Open: hand back to worker2 for cycle-3 Second-Eyes review (job j_33377aac).
- Next: dispatch cycle-3 review to worker2; address any Must-fix returned.

## MAP — Project Map and Symbol Index
## Entry & Build
- src/main.ts — bootstrap: Engine + createWorld + createUI + render loop + picking + state persist.
- index.html — #renderCanvas, loads /src/main.ts.
- vite.config.ts — base './', port 5175, host 127.0.0.1.
- vitest.config.ts — node env, tests/**/*.test.ts.

## Source modules
- src/data/bodies.ts — SUN, PLANETS, MOONS, ALL_BODIES, bodyById, BodyData.
- src/sim/scales.ts — ORBIT_SCALE, RADIUS_SCALE, clampSunDisplayRadius, planetDisplayRadius, moonDisplaySemiMajorAxis, orbitDisplayDistance, SUN_JUPITER_RADIUS_RATIO_CAP.
- src/sim/kepler.ts — solveKepler, trueAnomalyFromE, computeOrbit, orbitEllipsePoints, meanMotion, OrbitParams.
- src/state/store.ts — SimState, DEFAULT_STATE, loadState, saveState, validateState, advanceSim, simDate, formatSimDate.
- src/render/scene.ts — createWorld (WorldHandles), builds bodies/rings/trails/labels/camera/lights/skybox.
- src/render/rings.ts — buildRingMesh (custom annulus + radial UV), alignRingToEquator.
- src/render/skybox.ts — createSkybox (procedural starfield sphere).
- src/render/textures.ts — textureUrl, probeTexture (HEAD), proceduralTextureDataUrl, TEXTURE_SOURCES.
- src/ui/overlay.ts — createUI (UIHandles): top status bar, control dock, info panel, body selector.

## Tests & commands
- tests/orbital.test.ts — 28 tests (kepler, scaling, state, data sanity).
- npm test / npm run build / npm run dev / npm run preview.

## LAW — Learned Agent Warnings
LAW-001: Editing bodies.ts -> keep eccentricity in [0,1) and orbitalPeriodDays>0; verify npm test 'body data sanity'.
LAW-002: Touching vite.config.ts -> base must stay './' (subpath deploy); verify npm run build then grep base './'.
LAW-003: Editing scales.ts -> Sun display radius must stay <= 0.18*inner orbit and Jupiter <= sun/RATIO_CAP; verify npm test 'scaling coupling'.
LAW-004: scene.ts focus() -> must set camera.lowerRadiusLimit from focused body display radius; verify grep 'lowerRadiusLimit' in scene.ts.
LAW-005: textures.ts -> startup must NOT require network; procedural fallback path must render when probeTexture=false; verify build offline (no textures dir present).

## BAN — Forbidden Actions
BAN-001: Never bundle copyright-restricted textures in public/textures; because public-domain-only policy (see public/LICENSE.txt); verify no non-NASA/USGS files committed.
BAN-002: Never sum per-frame dt into a rebased float for daysSinceEpoch; because floating-point drift; verify advanceSim uses single accumulator in src/state/store.ts.
BAN-003: Never change vitest version range to ^4.2.0 or higher without checking npm registry; because vitest 4.2.0 does not exist (latest 4.1.9); verify 'npm view vitest version' before bumping.
BAN-004: Never use start with unquoted Windows paths in bash; because backslash swallowing; verify use powershell Start-Process or cmd start "" with quoted path.

## HABIT — User and Project Preferences
- Format: `HABIT-001: When <situation>, prefer <style/workflow>`.
- Use for user/project preferences that affect future choices.

## WHY — Why History Yells (Decision Rationale)
- Record decision rationale only: `Decision -> Why -> Tradeoff`.
- Do not duplicate changelog, NOW, or RAW evidence.

## OMM — Oh My Mistake (Failure Retrospectives)
OMM entries are operational mistake-prevention rules, not apologies.
Use this exact shape:
### OMM-001: Short title
- Trigger: When this rule must be recalled.
- Mistake: What failed before, concretely.
- Rule: What must/never happen next time.
- Required action: What to inspect or change before proceeding.
- Verify: Command, test, log, or observable check.

## RAW — Raw Evidence Pointers
- 2026-06-19: User '계속 이어나가봐' -> resumed solar-system-sim implementation.
- Files written: src/data/bodies.ts, src/sim/scales.ts, src/sim/kepler.ts, src/state/store.ts, src/render/scene.ts, src/render/rings.ts, src/render/skybox.ts, src/render/textures.ts, src/ui/overlay.ts, src/main.ts, tests/orbital.test.ts, public/LICENSE.txt, public/textures/README.md, README.md.
- package.json: vitest range fixed ^4.2.0->^4.1.0 (4.2.0 nonexistent); npm install OK (47 pkgs).
- Commands run: npm test -> 28/28 pass (after 2 assertion fixes: kepler wrap equivariance, ellipse endpoint float tol). npm run build -> tsc --noEmit clean, vite build success (1.38s). npm run dev -> http://127.0.0.1:5175 health OK.
- Verification: curl / returns index.html with /src/main.ts; curl /src/main.ts returns transformed module.
- Turn: t308 (resumed from interrupted t307).

## Design Brief
Palette: deep space black #01030a starfield; Sun emissive amber #ffb347/#ff8c42; Astro-console cyan accent #56b4e9; panel graphite #0d1117/#161b22; text #e6edf3; planet hues from albedo (Earth blue, Mars rust, Jupiter banded tan).
Typography: system UI sans (Segoe/Inter) for labels, monospace for data readouts (date, distance, AU) — NASA-console clarity.
Layout: fullscreen 3D canvas; floating glassy left control dock (time speed + play/pause + toggles); right info panel on body select; slim top status bar (sim date + speed).
Motion: easeOut camera fly-to on focus; orbit-trail fade-in; subtle self-rotation; reduced-motion respected.
UX conventions: Astro-UXDS explicit labels + optical hierarchy; click-to-focus; hover highlight; keyboard-accessible controls; aria-live info panel; log-scale documented in info panel.
Avoid: garish neon, flat cartoon spheres, paid assets, network-required startup, jarring camera snaps, low-contrast text.
Sources: Astro UXDS (astrouxds.com), NASA Solar System (solarsystem.nasa.gov), GitHub 3D solar-system-simulation refs.
