# Core concepts

The vocabulary JARVIS CODE uses, defined once. If a term in another doc is unclear, it's probably here.

---

## Two kinds of memory

JARVIS CODE keeps memory in two places, and they do different jobs.

### JHB — the carried conversation memory
**JHB** (the *JARVIS Hippocampus Buffer*) is the running memory of the **current conversation**. It's a small Markdown document — capped near **~2,000 tokens** — that the [encoder](#the-four-roles) rewrites after every turn. The chat model never sees the raw transcript; it sees the JHB. That's what lets the conversation run forever without the context growing. The JHB stays roughly the same size whether you're on turn 10 or turn 10,000. The mechanism — carrying a *regenerated* memory instead of replaying the transcript — is the subject of the JLC paper, *[Forgetting Is All You Need](https://doi.org/10.5281/zenodo.20266424)*.

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

## Raw recall — handing over the diary, a page at a time

The JHB is the *notes*. But notes can blur a detail you suddenly need 4,000 turns later. So JLC also keeps the full **raw transcript** on your disk — the *diary* — and hands the model the relevant pages when you reach back for them.

Not the whole diary. Pasting the entire history back every turn is the brute-force move every other agent makes — and it's exactly what makes cost grow with the conversation. Instead, when your message reaches into the past — *"remember when…", "what did we decide about X?"* — JLC reconstructs the right pages on the fly:

1. **BM25** narrows the raw turns to a candidate pool by keyword,
2. **bge-m3 cosine similarity** reranks them by meaning,
3. the **top 5** — plus the most recent raw turn, and small bonuses for recency and past corrections — are handed to the model.

So the model runs on the *notes* by default, and gets *reconstructed pages of the diary* only when you actually reach for the past. (The model can also pull specific turns on demand via a recall tool.) The JHB carries the understanding; raw recall backs it with the exact words when they matter.

Most agents never built this layer — they never had to, because they just paste the whole history in. This is the part that took the work.

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
- **Ultracode** — fan-out mode: for a big task, the agent spins up **N parallel subagent workers** on fresh contexts, then verifies and synthesizes their results. The same "split the context to beat the long-context curse" idea as JLC, applied across *space* (many small contexts) instead of *time* (one session split across turns). Note the trade: unlike JLC, ultracode **spends** tokens rather than saving them — N fresh contexts cost more, in exchange for higher quality on hard tasks. It's a quality lever, not a cost one.
- **JLC** — *JARVIS LLM Codec*, the core that carries memory outside the context window. The thing the whole project is built around. Want the depth? The mechanism is laid out across three Zenodo papers: **[v1](https://doi.org/10.5281/zenodo.19681402)** (the original thesis), **[v2](https://doi.org/10.5281/zenodo.19776258)** (the 1,000-turn proof), **[v3](https://doi.org/10.5281/zenodo.20266424)** (the current write-up).

---

**Next:** [cost-model.md](cost-model.md) · [architecture.md](architecture.md) · [memory-and-projects.md](memory-and-projects.md)
