<div align="center">

<img src="docs/jarvis-mark.svg" alt="JARVIS CODE" width="150" />

# JARVIS CODE

**Your coding companion that doesn't lose the thread.**

[![License](https://img.shields.io/badge/License-Apache_2.0-0088ff)](LICENSE)
[![Built on](https://img.shields.io/badge/Built_on-Pi_MIT-ff8800)](NOTICE.md)
[![Platform](https://img.shields.io/badge/Platform-Windows-5b5b66)](#-install)
[![Web](https://img.shields.io/badge/web-jlc--codec.org-0088ff)](https://jlc-codec.org)

</div>

---

**JARVIS CODE** is an independent terminal coding agent with **durable, long-term project memory**. It carries your codebase, your decisions, and the thread of past sessions forward — so you never re-explain yourself to a blank context again.

> The mark **◯ ~ ◯** is the two of you — **you** on the left, **your agent** on the right, and `~` the signal between. Two, standing side by side.

## 🤔 Why JARVIS CODE?

You think you're raising an AI — but every session, you meet a stranger. Most agents drag a finite conversation until it's compacted or cleared. JARVIS CODE **carries memory instead of dragging the transcript**:

- **Right where you left off** — context survives shutdown, restart, and even model swaps. No re-explaining.
- **No `/compact`, no `/clear`** — one continuous session, proven over a public **[10,000-turn run](https://jlc-codec.org/evidence)**.
- **Linear cost, not O(n²)** — it never drags a giant prefix, so cost-per-turn stays flat as the work grows. ([the honest numbers, incl. where it *doesn't* win →](https://jlc-codec.org/docs/#cost-model))
- **Zero handoff · switching · onboarding** — move across projects and machines, and grasp an unfamiliar codebase right away.
- **`JARVIS.md` per project** — self-improving memory that gets smarter about your codebase the more you use it.

Built on the open **pi-agent** harness (MIT, by Mario Zechner) with the **JLC** memory system grafted in and tuned — proven memory on a proven agent.

## 🔄 Every agent is stateless — JARVIS CODE is built for it

**Did you know?** An LLM remembers nothing between turns — every agent is stateless underneath. Most hide it by **replaying the entire conversation on every turn**, fighting their own nature until the context fills and collapses into `/compact` and `/clear`.

JARVIS CODE does the opposite. It is **designed for statelessness** — the context resets every turn, and memory is carried *outside* the window by the JLC codec. It forgets the noise on purpose and keeps the thread.

```text
  OTHER AGENTS                   JARVIS CODE
  ────────────                   ───────────
  context  ▁▃▅▇█▉  piles up      context  ▁▁▁▁▁▁   reset every turn
                   ↓             memory   ▁▂▃▅▆▇   carried outside ↑
  → compact · clear · collapse   → linear · stable · keeps the thread
```

Working *with* the grain instead of against it, it never slows as it grows — and never loses the thread.

## ✨ What's inside

- 🧠 **JLC memory** — a bounded, self-organizing memory injected into every model turn; full history kept locally for recall
- 🪟 **Multi-window orchestration** — spawn worker windows, delegate builds as reviewed jobs, or let two windows argue to consensus; all windows share one memory
- 🧭 **Plan dialogue + design recon** — a vague "build me X" pops a quick dialog of choices, then distills current design trends into a per-project brief before the first file is written
- 🦙 **Local-first endpoints** — keyless presets for Ollama, LM Studio, and llama.cpp, right next to cloud providers in `/model-setting`
- ✋ **Tool lessons** — a failed command and its fix are remembered and offered next time the same failure appears, at zero always-on token cost
- 📁 **Per-project `JARVIS.md`** — project memory that lives with the repo, like `CLAUDE.md`
- 🎨 Bundled skills and the signature orange-blue terminal theme

The user-facing command is `jarvis`. (`pi/` is the internal engine folder, kept for fork compatibility.)

## 📊 Benchmarks — the memory layer costs nothing

Memory is JLC's job; these runs answer a different question: **does carrying it slow the model down?** July 2026, chat = GPT-5.5 (subscription route), encoder = gpt-5.4-mini, consumer laptop (i5-8500, 8 GB RAM). The agent never scores itself.

| Benchmark | Score | Scoring |
|---|---|---|
| Aider Polyglot (Python + Go + JS subsets, 122 tasks) | **122/122 (100%)** | clean pytest / go test / npm test owned by the runner, single attempt |
| HumanEval, base tests | **98.2%** pass@1 (161/164) | official `evalplus` harness |
| HumanEval+ (80× extra tests) | **93.9%** pass@1 (154/164) | official `evalplus` harness |

Run integrity: 0 timeouts · 0 modified test files (hash-checked) · 0 benchmark-data lookups. Runner code lives in [`bench/`](bench/).

**Honest scope** — runs are agentic (one attempt; the agent may write its *own* scratch tests, never the benchmark's). HumanEval and Exercism are in every frontier model's training data, so these scores demonstrate *scaffold-neutrality, not model intelligence*. The Polyglot subset is the easiest slice of aider's 225-task six-language set — not comparable to full-set scores. Comparison table, roadmap, and caveats: [jlc-codec.org/benchmarks](https://jlc-codec.org/benchmarks/). SWE-bench is next on the campaign.

## 🚀 Install

> **Windows is first-class today. macOS / Linux are coming soon.**

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.ps1 | iex
```

Missing prerequisites (Node.js, Python, Git, MSVC redistributable) are installed via `winget` when available. The installer also preloads the local `BAAI/bge-m3` embedding model (~2.3 GB) unless `JARVIS_CODE_NO_MODEL_PRELOAD=1` is set.

> After install, **open a new terminal** so the `jarvis` command is on your PATH.

Manual / advanced install: [README-INSTALL.md](README-INSTALL.md).

## 💾 What gets installed & how big

Everything installs under `%LOCALAPPDATA%\JARVIS-Code`. Rough footprint on a fresh machine:

| Component | Size |
|---|---|
| App — engine + sidecar + skills/theme | ~16 MB |
| Node dependencies | ~560 MB |
| Python sidecar (incl. PyTorch, CPU) | ~1.3 GB |
| `bge-m3` embedding model — **powers recall (required)** | ~2.3 GB download · ~4.3 GB on disk |
| Prerequisites — Node 20+, Python 3.10+, Git, MSVC — *only if missing* | ~0.5 GB |

**Total ≈ 6 GB** (plus prerequisites if you don't already have them). Recall runs on a **BM25 + bge-m3** hybrid — keyword search plus semantic embeddings — so the model is core, not optional; `JARVIS_CODE_NO_MODEL_PRELOAD=1` only **defers** its download to first use. Your memory data under `~/.jarvis-code` starts small and grows slowly with use.

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

Diagnostics anytime: `jarvis doctor`

## 🧩 How it works

Long session history is stored locally for recall, while the live engine keeps only a bounded recent-turn window. The model's context is assembled by JLC — it carries the **memory**, not the whole transcript.

- 📖 **Full docs & the honest cost model** — [jlc-codec.org/docs](https://jlc-codec.org/docs)
- 🔬 **Proof: the public 10,000-turn run, raw** — [jlc-codec.org/evidence](https://jlc-codec.org/evidence)
- 🧱 In-repo notes — [Architecture](docs/architecture.md) · [Memory & Projects](docs/memory-and-projects.md) · [Providers](docs/providers.md) · [Troubleshooting](docs/troubleshooting.md)

## 🗑️ Uninstall

**Windows** (PowerShell):

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/uninstall.ps1 | iex
```

Uninstall removes the install directory and command shim. User data under `~/.jarvis-code` and the model cache are kept unless explicit removal options are set.

## 📄 License & attribution

JARVIS CODE is licensed under [Apache-2.0](LICENSE) (© 2026 Jun).

The internal engine under `pi/` is derived from **Pi by Mario Zechner (Earendil Works)**, distributed under the MIT license. Original notices are preserved in [NOTICE.md](NOTICE.md) and [THIRD_PARTY_LICENSES/](THIRD_PARTY_LICENSES).

<div align="center">

**◯ ~ ◯** &nbsp;·&nbsp; [jlc-codec.org](https://jlc-codec.org)

</div>
