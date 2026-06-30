# Core concepts

The vocabulary JARVIS CODE uses, defined once. If a term in another doc is unclear, it's probably here.

---

## Two kinds of memory

JARVIS CODE keeps memory in two places, and they do different jobs.

### JHB — the carried conversation memory
**JHB** (the *JARVIS Handbook*) is the running memory of the **current conversation**. It's a small Markdown document — capped near **~2,000 tokens** — that the [encoder](#the-four-roles) rewrites after every turn. The chat model never sees the raw transcript; it sees the JHB. That's what lets the conversation run forever without the context growing. The JHB stays roughly the same size whether you're on turn 10 or turn 10,000.

### JARVIS.md — the per-project memory
**JARVIS.md** is a file kept **per project** (per repository). It's where the agent records what it has learned about *your codebase* so it's still there next session. It has named sections:

| Section | Holds |
|---|---|
| **NOW** | Current work state |
| **MAP** | File & symbol map |
| **LAW** | Rules that keep recurring |
| **BAN** | What not to do |
| **OMM** | A mistake → next run's guardrail (it learns from its own errors) |
| **RAW** | Pointers to verifiable evidence |

**OMM** is the compounding one: a mistake made once becomes a guardrail, so the same mistake doesn't repeat — without you having to flag it.

---

## Modes

- **Chat mode** — everyday conversation and quick questions. Plain reasoning, fast.
- **Deep-dive mode** — coding and multi-step work. Reasoning is pushed to the limit to pull out the model's best, with no fear of burning context (memory lives outside the window).

The [router](#the-four-roles) decides which one a message needs.

---

## The four roles

JARVIS CODE isn't "an AI" — it's four model roles you assign in [`/model-setting`](providers.md):

| Role | Job |
|---|---|
| **chat** | The model you talk to — coding and conversation (smart, expensive) |
| **encoder** | Compresses each turn into the JHB, every turn (cheap, fast) |
| **router** | Classifies intent (chat vs deep-dive vs build); mirrors the encoder |
| **subagent** | Runs tool-calling subtasks, and the ultracode fan-out workers; defaults to chat |

Why the chat/encoder split lowers total cost: **[cost-model.md](cost-model.md)**.

---

## Other terms

- **Sidecar** — the local background process that *is* the JLC engine. The terminal front-end talks to it; it does the encoding, memory, and routing. It runs on your machine.
- **Pair** — one paired session: the terminal UI front-end joined to its sidecar. (You normally never think about this; it matters when running multiple windows.)
- **Ultracode** — fan-out mode: for a big task, the agent spins up **N parallel subagent workers** on fresh contexts, then verifies and synthesizes their results. The same "split the context to beat the long-context curse" idea as JLC, applied across *space* (many small contexts) instead of *time* (one session split across turns).
- **JLC** — *JARVIS LLM Codec*, the core that carries memory outside the context window. The thing the whole project is built around.

---

**Next:** [cost-model.md](cost-model.md) · [architecture.md](architecture.md) · [memory-and-projects.md](memory-and-projects.md)
