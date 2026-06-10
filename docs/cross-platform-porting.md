# Cross-Platform Porting Audit

Date: 2026-06-14

Scope: Phase 0 audit, M1 single-window Unix launcher draft, and M2 design notes.
This is intentionally separate from the B: crash hardening work. Windows
`jarvis.ps1` remains the production Windows entrypoint.

## Phase 0 Audit

| Area | Evidence | Current coupling | Unix replacement | Risk / unknown |
| --- | --- | --- | --- | --- |
| Visible worker spawn | `sidecar/jarvis_sidecar/spawn.py:120-171` | Builds `wt.exe` plus PowerShell plus `jarvis.ps1`; uses Windows creation flags. | Add an OS launcher layer: macOS `open -na Terminal` or `osascript`, Linux `gnome-terminal` / `konsole` / `xterm`, and `tmux` fallback. Keep runtime-file wait semantics. | Terminal quoting and env propagation differ by terminal app; desktopless Linux needs tmux/headless mode. |
| Pair/runtime labels | `sidecar/jarvis_sidecar/window_labels.py:36-57`, `sidecar/jarvis_sidecar/spawn.py:174-201` | Runtime files are portable JSON, but current spawner only launches Windows terminals. | Reuse the same `sidecar-runtime-<pair8>.json` files for Unix windows. | Stale runtime cleanup depends on reliable PID liveness. |
| Windows wrapper process containment | `jarvis.ps1:59-164` | Win32 Job Object P/Invoke with kill-on-close and breakaway support. | Use `subprocess.Popen(..., start_new_session=True)` or `setsid`; terminate tracked process groups with `killpg`. | Unix process group semantics do not exactly match Job Object sibling survival. |
| Credentials env lift | `jarvis.ps1:241-259` | PowerShell parses `credentials.yaml` `env:` block into process env for Pi. | Implemented in Python launcher at `scripts/jarvis-launcher.py:244-265`. | Parser is intentionally shallow, matching current wrapper behavior rather than full YAML. |
| Wrapper CLI normalization | `jarvis.ps1:548-599` | PowerShell consumes `--recent-turns`, `--auto-prompts`, `--window-label`, `--sidecar-window`, `--yolo`. | Implemented in Python launcher at `scripts/jarvis-launcher.py:268-318`. | `--sidecar-window` is recorded but M1 does not open a separate sidecar terminal. |
| Sidecar env and runtime | `jarvis.ps1:613-635`, `jarvis.ps1:667-685` | Pair-specific runtime file, wrapper PID, sidecar URL, history, and Pi agent env are set by PowerShell. | Implemented in Python launcher at `scripts/jarvis-launcher.py:321-361` and runtime write at `scripts/jarvis-launcher.py:529-548`. | Python launcher lacks the Windows watchdog restart loop by design in M1. |
| Port checks | `jarvis.ps1:652-656`, `jarvis.ps1:960-978` | Uses `Get-NetTCPConnection`. | Python launcher uses socket connect probing at `scripts/jarvis-launcher.py:417-438`. For ownership-level diagnostics use `lsof -iTCP`, `ss -lptn`, or `netstat` depending on OS. | Socket probing finds free ports but does not prove owner identity. |
| Visible sidecar / restart | `jarvis.ps1:981-1015`, `jarvis.ps1:1290-1340` | Uses `Get-NetTCPConnection`, `Get-CimInstance`, named mutex, and `Start-Process -WindowStyle`. | M1 omits visible sidecar. M2 should use lock files for port allocation and terminal launcher strategies for visible windows. | Unix desktop availability and terminal permissions are unknown. |
| Watchdog | `jarvis.ps1:687-863`, `sidecar/jarvis_sidecar/wrapper_watch.py:19-70` | PowerShell `Start-Job` restarts sidecar; sidecar also watches wrapper PID. | M1 keeps sidecar wrapper-watch through `JARVIS_WRAPPER_PID`; M2 can add a Python watchdog process. | Restart loop must not race shutdown cleanup. |
| Process enumeration / kill | `jarvis.ps1:1083-1180`, `jarvis.ps1:1223-1267` | Uses CIM process tree and `taskkill.exe /T /F /PID`. | M2 should terminate only tracked process groups/PIDs; use `killpg`, then `SIGKILL` fallback. | Avoid broad name kills; PID reuse validation is still a follow-up. |
| Pi runner | `pi/pi-test.ps1:83-90` | Windows-only `tsx.cmd` wrapper. | Python launcher directly selects `tsx` vs `tsx.cmd` at `scripts/jarvis-launcher.py:92-101` and runs `cli.ts` at `scripts/jarvis-launcher.py:604-605`. | M1 assumes `npm ci` or installer has populated `pi/node_modules`. |
| JARVIS extensions | `jarvis.ps1:1353-1385` | Windows wrapper loads JLC, face, and image extensions. | Python launcher now loads all three at `scripts/jarvis-launcher.py:558-567`. | Extension behavior still needs real macOS/Linux runtime testing. |
| Release surface | `jarvis.sh:1-15`, `scripts/build-release.py:19-31`, `scripts/build-release.py:41-49` | Unix shim already delegates to `scripts/jarvis-launcher.py` and is included in release builds. | Keep `jarvis.sh` thin; move logic to Python launcher for cross-platform parity. | Installer must ensure executable bit in release artifacts on Unix. |
| Sidecar config roots | `sidecar/jarvis_sidecar/config.py:282-301`, `sidecar/jarvis_sidecar/config.py:540-550` | Config and memory roots use env overrides and `Path.home()`; Windows default code root uses drive anchor. | Already portable: Unix default code root becomes `~/jarvis_workspace`. | Existing user docs still emphasize `C:\jarvis_workspace`. |
| File locking | `sidecar/jarvis_sidecar/file_locks.py:41-49`, `sidecar/jarvis_sidecar/file_locks.py:73-80` | Explicit `msvcrt` / `fcntl` branches. | Already portable. | Network filesystems may have weaker lock semantics. |
| Raw store / directives | `sidecar/jarvis_sidecar/raw_store.py:24-38`, `sidecar/jarvis_sidecar/raw_store.py:247-250`, `sidecar/jarvis_sidecar/directives.py:33-39`, `sidecar/jarvis_sidecar/directives.py:682-745` | File-based JSONL plus cursor files; uses portable paths and locks. | No porting required. | Cursor migration path still contains `_windows` naming at `sidecar/jarvis_sidecar/directives.py:69-71`; semantic only. |
| JHB/window listing | `sidecar/jarvis_sidecar/directives.py:890-925`, `sidecar/jarvis_sidecar/pairing.py:84-109` | Uses PID liveness with `os.kill` on Unix and Win32 on Windows. | Already portable. | Window list directory name `_windows` is historical but not OS-bound. |
| Provider fallback | `sidecar/jlc_agentic/providers/__init__.py:31-52` | Fallback includes `C:/jarvis-code/config.yaml`. | Prefer env or `~/.jarvis-code/config.yaml`; remove or demote the `C:/` fallback in a later cleanup. | Hidden Windows default could confuse Unix installs if no env/config exists. |
| Register project guard | `sidecar/jlc_agentic/agentic/tools/register_project.py:18-37` | Denylist explicitly covers Windows and POSIX sensitive roots. | Already cross-platform. | macOS app/project locations under `/Applications` are rejected by design. |
| Pi shell backend | `pi/packages/coding-agent/src/utils/shell.ts:62-105`, `pi/packages/coding-agent/src/core/tools/bash.ts:74-80` | Windows prefers Git Bash; Unix uses `/bin/bash`, PATH bash, then `sh`; Unix children are detached for group kills. | Already has OS branches. | MSYS path conversion is Windows-only cosmetic risk; managed process helper is preferred for background processes. |

## M1 Single-Window Launcher

M1 status: implemented as a Python launcher behind the existing `jarvis.sh`
shim, and validated on Linux (WSL2 Ubuntu, kernel 6.6, x86_64, Node 20 via nvm)
with a fresh native-filesystem clone. macOS has not yet had a real runtime pass.

Implemented:

- `jarvis.sh` remains a thin POSIX shim to Python (`jarvis.sh:1-15`).
- `scripts/jarvis-launcher.py` chooses `.venv/bin/python` and `.bin/tsx` on
  Unix while retaining Windows branches (`scripts/jarvis-launcher.py:92-101`).
- The launcher bootstraps the sidecar venv and requirements
  (`scripts/jarvis-launcher.py:452-466`).
- It lifts `credentials.yaml` `env:` values into the Pi process env
  (`scripts/jarvis-launcher.py:244-265`).
- It normalizes `--window-label`, `--recent-turns`, `--auto-prompts`,
  `--sidecar-window`, and `--yolo` (`scripts/jarvis-launcher.py:268-318`).
- It writes pair-specific runtime metadata and `JARVIS_WRAPPER_PID`
  (`scripts/jarvis-launcher.py:321-361`, `scripts/jarvis-launcher.py:529-548`).
- It starts one hidden sidecar process, polls health, and then runs Pi in the
  same terminal (`scripts/jarvis-launcher.py:508-555`,
  `scripts/jarvis-launcher.py:573-605`).
- It loads JLC, face, and image extensions (`scripts/jarvis-launcher.py:558-567`).

Deliberately not in M1:

- No Unix multi-window spawn implementation.
- No visible sidecar terminal for `--sidecar-window`.
- No Python watchdog restart process beyond the sidecar's existing wrapper PID
  watch.
- No OS-specific owner lookup for busy ports.

Static validation expected on Windows:

```powershell
python -m py_compile scripts\jarvis-launcher.py
bash -n jarvis.sh
$env:JARVIS_WRAPPER_DRY_RUN='1'; python scripts\jarvis-launcher.py --yolo --window-label worker1 --recent-turns 0 --offline --print probe
```

Do not execute `jarvis.sh` through WSL against the Windows checkout as a dry-run
substitute; WSL Python treats `sidecar/.venv` as a Unix venv path.

Runtime prerequisites (discovered during the first real Linux pass):

- Node.js >= 20 is required. On Node 18, `undici` throws
  `ReferenceError: File is not defined` when tsx loads `cli.ts` (the `File`
  global landed in Node 20). `pi/package.json` already declares
  `engines.node >= 20`; treat it as a hard launch prerequisite, not just a warning.
- A working `python3 -m venv` with `ensurepip`. On Debian/Ubuntu this means the
  `python3-venv` (e.g. `python3.12-venv`) package; without it `venv` produces an
  interpreter with no `pip` and the sidecar requirements never install. The
  no-sudo workaround is `python3 -m venv --without-pip` followed by `get-pip.py`.

Real Unix validation - status:

- Linux (WSL2 Ubuntu, Node 20 via nvm, native ext4 clone): PASS for the M1
  single-window path. Verified: `npm ci` populates `node_modules/.bin/tsx`;
  `./jarvis.sh --help` drives tsx -> `cli.ts` -> Pi help; `JARVIS_WRAPPER_DRY_RUN=1`
  dry-run emits the expected JSON (arg normalization, three extension paths,
  config-derived default provider, forward args); `load_credentials_env` lifts a
  `credentials.yaml` env value; `python -m jarvis_sidecar` serves `/health` =
  `{"ok": true, "service": "jarvis-jlc-sidecar", ...}`. The sidecar imports
  cleanly without `torch`/`sentence-transformers` (those stay lazy in the
  retriever path), so a minimal sidecar venv is enough to come up healthy.
- Linux desktop terminal inventory and visible-window spawn: NOT validated (WSL
  is headless); this is M2 scope.
- macOS: NOT yet validated (no macOS runtime available). Expected to pass the
  same M1 path because the exercised code is POSIX-only, but this stays unproven
  until a real macOS run.
- Pending on any platform (left for an interactive operator turn): one real Pi
  turn against a live model, and confirming `data/sidecar-runtime-<pair8>.json`
  is written on start and removed on wrapper exit. The credentials env lift was
  confirmed at unit level; an end-to-end custom-provider turn is still pending.

## M2 Terminal Spawning — Implemented

Status: implemented in `sidecar/jarvis_sidecar/spawn.py` behind the unchanged
`spawn_window()` / `launch_visible_jarvis_window()` API. Windows path is
unchanged; Unix and macOS backends are new.

Runtime validation:

- tmux headless backend: **live PASS on real Linux** (WSL2 Ubuntu, 2026-06-14).
  Real `detect_unix_terminal()` selected tmux; real `build_unix_launch_argv()`
  produced `tmux new-session -d -s jarvis-<label> env PATH=... HOME=... <cmd>`;
  `Popen(start_new_session=True)` launched it (rc 0) and the command executed
  inside the detached session (marker file written). The env-forwarding fix was
  separately verified against an intentionally **stale** tmux server: the
  forwarded PATH/HOME overrode the server's stale values while a planted secret
  did not reach the command line. This is the only path exercised end-to-end with
  the actual code.
- Linux desktop terminals (gnome-terminal/konsole/xterm) and macOS: **not yet
  runtime-validated** — unit-covered only (no GUI terminal installed in the WSL
  test box; no macOS host). Next interactive operator turn (WSLg): `sudo apt
  install xterm` (or gnome-terminal) in WSL Ubuntu + an active Node 20 so the
  worker window renders on the Windows desktop and boots a real jarvis.

Review fixes (GAN pass, 2026-06-14) folded in:

- **BLOCKER** `scripts/jarvis-launcher.py` `sidecar_healthy()` now validates
  `pair_id` (mirroring `jarvis.ps1` `Test-Sidecar`). Without it a spawned worker
  inherited the parent port, saw the parent sidecar as healthy, skipped starting
  its own, and never wrote a runtime file — so the spawn waiter always timed out.
- **HIGH** tmux env staleness: a pre-existing tmux server hands a new session its
  own stale environment. Forwarded via an `env KEY=VALUE` command prefix
  (`TMUX_FORWARD_ENV_KEYS`, PATH/HOME/locale only — tmux's own `-e` proved
  unreliable for PATH). Secrets are excluded so they never land on the
  world-readable tmux command line; the worker loads credentials from file.
- **MED** display gate: `detect_unix_terminal()` skips GUI terminals when no
  `DISPLAY`/`WAYLAND_DISPLAY` is set, so a headless host falls back to tmux
  instead of launching a GUI terminal that fails and stalls the 150s spawn wait.

- `launch_visible_jarvis_window()` dispatches on `_current_platform()`
  (`windows` / `macos` / `unix`); the platform is a monkeypatchable seam so the
  Unix/macOS backends are testable on a Windows dev host.
- `jarvis_launch_target()` returns `jarvis.ps1` on Windows and `jarvis.sh` on
  every Unix host. These are never crossed — `jarvis.ps1` under a Unix shell is
  exactly the `0x80070002` a Linux worker spawn hit before this dispatcher.
- Linux terminal preference order (`detect_unix_terminal`, first on PATH wins):
  `gnome-terminal --` → `konsole -e` → `xterm -e` → `tmux new-session -d`
  (headless fallback, session named `jarvis-<worker-label>` to avoid collisions).
- macOS uses `osascript ... do script` (`build_macos_launch_argv`); each token is
  POSIX-quoted and embedded in the AppleScript string literal. UNVERIFIED.
- Provider/model/window-label ride as explicit `jarvis.sh` CLI flags
  (`jarvis_worker_command`), not only env: `jarvis-launcher.py` forwards
  `--provider`/`--model` to Pi and consumes `--window-label`, and a reused tmux
  server would otherwise carry stale env.
- Process containment: `start_new_session=True` (setsid) replaces the Windows
  `CREATE_BREAKAWAY_FROM_JOB`, so the launched terminal/tmux server leads its own
  session and process group and a sibling worker survives a parent-group kill.
- `child_spawn_env()` env-cut and the runtime-file wait (`wait_for_spawned_runtime`)
  are shared by all backends; success is still inferred only from a new runtime
  file, never from terminal process exit.

Still pending in M2 (not yet implemented):

Process management:

Implemented for managed helper processes (`jarvis-jlc.ts`, 2026-06-15):

- Each managed process now records an OS-level start token (`procStartToken`)
  plus its `pgid` and its **owner's** start token (`ownerProcStartToken`),
  alongside the existing pid, command, cwd, owner PID, and start time. Token
  sources by platform:
  - Linux: **boot-id-anchored** `/proc/<pid>/stat` starttime —
    `linux:<boot_id>:<starttime>`. starttime alone is jiffies-since-boot and would
    falsely match a recycled PID across a reboot, so the kernel boot id anchors it
    to this boot; without a readable boot id no token is produced (stays `unknown`).
  - Windows: `Get-Process StartTime.Ticks` — absolute, 100 ns resolution.
  - macOS: `ps -o lstart` — absolute but only **second** resolution, so it is
    flagged *coarse*: a token *match* is too weak to authorize a kill (a same-second
    PID reuse could collide) and degrades to `unknown`; a *mismatch* is still a safe
    `reused`. (A high-resolution macOS start-time source is future work; until then
    macOS cross-instance orphan auto-cleanup is intentionally conservative.)
- **PID-reuse validation before every kill.** `verifyManagedPidIdentity`
  re-reads the live PID's start token and compares it to the recorded one before
  terminating. A recycled PID classifies as `reused` (never killed) and an
  unprovable / coarse-only identity as `unknown` (also not auto-killed). The same
  verified check gates the shell kill guard (`managedProcessOwnsPidVerified`) and
  `stopManagedProcess`.
- **SECURITY — disk state never authorizes a kill.** A state file on disk is
  untrusted input: any same-user process (the agent itself can write files) could
  forge a record with a target PID and that PID's real, precomputed start token
  (`/proc/<pid>/stat` and the boot id are world-readable), making the cross-instance
  sweep terminate an arbitrary process and bypass the shell kill guard entirely.
  Because same-user forgery cannot be prevented cryptographically (the attacker
  shares the user's read access), `cleanupStaleManagedProcesses` therefore NEVER
  terminates a live PID — it only reclaims obsolete files (`dead`/`reused`). The
  only paths permitted to kill are this process's own in-memory child records
  (`stopManagedProcess` / session shutdown), which carry an unforgeable live
  `ChildProcess` handle. Trade-off: a genuine orphan from a crashed instance is no
  longer auto-terminated (its file is reclaimed once its PID dies); the user can
  stop it via OS tools. Security wins over convenience here.
- **Owner identity, not just owner liveness.** The stale sweep no longer skips a
  record merely because `ownerPid` is alive — a dead owner's PID could itself be
  recycled, which would leak the orphan's file forever. `ownerProcessStillAlive`
  validates the live owner PID against the recorded `ownerProcStartToken` (captured
  before the first state write) and only skips when the genuine owner is still
  running. Safe direction throughout: if reuse cannot be disproven, do not kill; if
  owner identity cannot be disproven, do not reap its records.
- Broad `kill-all` is already structurally absent: shutdown only ever iterates
  tracked records (`stopAllManagedProcesses`), and image/name-based kills are
  hard-blocked by the shell guard. Detached Unix children lead their own
  session/group (`process.kill(-pid)` terminates the group, SIGTERM then SIGKILL).

Still pending (sidecar/launcher, not the managed-helper path):

- Apply the same `pgid` + start-token record to the sidecar and the launcher's
  `terminate_started_sidecar` so a manual sidecar cleanup also validates identity.
- Shutdown order for the sidecar: stop watchdog/sentinel, SIGTERM process group,
  wait, SIGKILL process group, remove runtime file.

Non-goal (decided 2026-06-15): isolating workers under a **separate OS user /
sandbox** (follow-up ticket T3's strong form). A worker exists to code inside the
user's workspace — a separate user would strip its access to the user's files and
credentials and break delegated coding (e.g. the Tetris worker run). The
achievable isolation — each worker leading its own session/process group so it
cannot take down siblings via a group kill — is already in place
(`start_new_session=True` / `CREATE_BREAKAWAY_FROM_JOB`). Combined with the
hard-blocked broad-kill guard and per-PID identity validation, that is the
intended containment model for a companion tool.

Port and watchdog:

- Replace Windows named mutex with a cross-process lock file in `data/`.
- Use socket bind/connect probing for free port selection.
- If ownership diagnostics are needed, use `lsof -nP -iTCP:<port> -sTCP:LISTEN`
  or `ss -lptn` as optional diagnostics only; launcher correctness should not
  depend on either being installed.
- Implement watchdog as a small Python helper or sidecar-owned thread/process,
  not terminal-specific shell code.

## B Drive Hardening Follow-Ups

Recorded only; intentionally not fixed in this pass:

| Ticket | Why it matters | Future fix |
| --- | --- | --- |
| ~~PID reuse in `managedProcessOwnsPid`~~ (RESOLVED 2026-06-15) | A stale record could return true if the OS reused the PID, killing the wrong process. | Done: OS start token captured at spawn and re-validated before every kill (`verifyManagedPidIdentity`); reused/unprovable PIDs are never auto-killed. |
| ~~Multi-PID `taskkill` parsing~~ (RESOLVED earlier) | `taskkill /PID a /PID b` could under-detect later PID targets. | Done: `extractPidKillTargets` uses `matchAll` over every PID flag occurrence. |
| Indirect execution and sandboxing | String guards cannot catch kills hidden inside scripts or binaries. | Process-group isolation is in place; the **separate-OS-user** form is an intentional non-goal for the companion model (see Process management above) — workers need workspace access. Revisit only if a sandboxed-worker mode is ever wanted. |
