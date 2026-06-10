<div align="center">

<img src="docs/jarvis-mark.svg" alt="JARVIS CODE" width="150" />

# JARVIS CODE

**Your coding companion that never forgets.**

[![License](https://img.shields.io/badge/License-Apache_2.0-0088ff)](LICENSE)
[![Built on](https://img.shields.io/badge/Built_on-Pi_MIT-ff8800)](NOTICE.md)
[![Platform](https://img.shields.io/badge/Platform-Win_macOS_Linux-5b5b66)](#-install)
[![Web](https://img.shields.io/badge/web-jlc--codec.org-0088ff)](https://jlc-codec.org)

</div>

---

**JARVIS CODE** is an independent terminal coding agent with **durable, long-term project memory**. It remembers your codebase, your decisions, and your past conversations across sessions — so you never re-explain yourself to a blank context again.

> The mark **◯ ~ ◯** is the two of you — **you** on the left, **your agent** on the right, and `~` the signal between. Two, standing side by side.

## 🤔 Why JARVIS CODE?

You think you're raising an AI — but every session, you meet a stranger. Most agents drag a finite conversation until it's compacted or cleared. JARVIS CODE **carries memory instead of dragging the transcript**:

- **Right where you left off** — context survives shutdown, restart, and even model swaps. No re-explaining.
- **No `/compact`, no `/clear`** — one continuous session, proven over a public **10,000-turn** run.
- **Linear cost, not O(n²)** — it never drags a giant prefix, so cost-per-turn stays flat as the work grows.
- **Zero handoff · switching · onboarding** — move across projects and machines, and grasp an unfamiliar codebase right away.
- **`JARVIS.md` per project** — self-improving memory that gets smarter about your codebase the more you use it.

Built on the open **pi-agent** harness (MIT, by Mario Zechner) with the **JLC** memory system grafted in and tuned — proven memory on a proven agent.

Provider setup, supported tiers, and custom provider examples are in [Providers](docs/providers.md).

## 🔄 Every agent is stateless — JARVIS CODE is built for it

**Did you know?** An LLM remembers nothing between turns — every agent is stateless underneath. Most hide it by **replaying the entire conversation on every turn**, fighting their own nature until the context fills and collapses into `/compact` and `/clear`.

JARVIS CODE does the opposite. It is **designed for statelessness** — the context resets every turn, and memory is carried *outside* the window by the JLC codec.

```text
  OTHER AGENTS                   JARVIS CODE
  ────────────                   ───────────
  context  ▁▃▅▇█▉  piles up      context  ▁▁▁▁▁▁   reset every turn
                   ↓             memory   ▁▂▃▅▆▇   carried outside ↑
  → compact · clear · collapse   → linear · stable · never forgets
```

Working *with* the grain instead of against it, it never slows as it grows — and never forgets.

## ✨ What's inside

- 🧠 **JLC memory** — a bounded, self-organizing memory injected into every model turn; full history kept locally for recall
- 🖥️ **Terminal-native engine** — a fast TUI coding agent (TypeScript, under `pi/`)
- 🔌 **Python sidecar** — a FastAPI service for memory, project routing, and raw recall
- 📁 **Per-project `JARVIS.md`** — project memory that lives with the repo, like `CLAUDE.md`
- 🎨 Bundled skills and the signature orange-blue terminal theme

The user-facing command is `jarvis`. (`pi/` is the internal engine folder, kept for fork compatibility.)

## 🚀 Install

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.sh | sh
```

On Windows, missing prerequisites (Node.js, Python, Git, MSVC redistributable) are installed via `winget` when available. The installer also preloads the local `BAAI/bge-m3` embedding model (~2.3 GB) unless `JARVIS_CODE_NO_MODEL_PRELOAD=1` is set.

> After install, **open a new terminal** so the `jarvis` command is on your PATH.

Manual / advanced install: [README-INSTALL.md](README-INSTALL.md).

## ▶️ First run

JARVIS CODE needs one LLM credential before it opens.

**Sign in with GPT** (ChatGPT OAuth):

```bash
jarvis gpt-login
jarvis
```

**Or use an API key:**

```bash
jarvis api-key
jarvis model-setting
jarvis
```

Diagnostics anytime:

```bash
jarvis doctor
```

Running from a source checkout: `.\jarvis.ps1` (Windows) or `./jarvis.sh` (macOS/Linux).

## 🧩 How it works

Long session history is stored locally for recall, while the live engine keeps only a bounded recent-turn window. The model's context is assembled by JLC — it carries the **memory**, not the whole transcript.

- [Architecture](docs/architecture.md)
- [Memory & Projects](docs/memory-and-projects.md)
- [Providers](docs/providers.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🗑️ Uninstall

**Windows:**

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/uninstall.ps1 | iex
```

**macOS / Linux:**

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/uninstall.sh | sh
```

Uninstall removes the install directory and command shim. User data under `~/.jarvis-code` and the model cache are kept unless explicit removal options are set.

## 📄 License & attribution

JARVIS CODE is licensed under [Apache-2.0](LICENSE) (© 2026 Jun).

The internal engine under `pi/` is derived from **Pi by Mario Zechner (Earendil Works)**, distributed under the MIT license. Original notices are preserved in [NOTICE.md](NOTICE.md) and [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES).

<div align="center">

**◯ ~ ◯** &nbsp;·&nbsp; [jlc-codec.org](https://jlc-codec.org)

</div>
