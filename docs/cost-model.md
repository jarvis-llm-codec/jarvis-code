# Cost model — why two models, and why the encoder is cheap

The most unusual thing about JARVIS CODE is that it runs **two models on every turn**:

- a **chat** model — expensive, smart, does the actual coding and conversation
- an **encoder** — cheap, fast, compresses the turn into memory

The common reflex is *"two models = more cost."* For anything longer than a short chat, the opposite is true. This page explains the mechanism and gives the honest numbers — including where it does **not** win.

> 🤖 **Don't trust the prose — audit the data.** Every number below is backed by raw logs at **[jlc-codec.org/evidence](https://jlc-codec.org/evidence)** (public 1,000- and 10,000-turn runs, with SHA-256 hashes). Paste a log into your AI and ask it to check the math.

---

## The problem with one model

An LLM is **stateless between turns** — it remembers nothing on its own. A normal coding agent hides this by **replaying the entire conversation every turn**. So a 25-token question at turn 8,000 doesn't cost 25 tokens; it costs 25 *plus the 7,999 turns before it.*

Cost grows with the **turn count**, not the question — that's **O(n²)**. Eventually the window fills and the agent has to `/compact` (compress and lose detail) or `/clear` (throw memory away).

People underestimate this badly. "8,000 short questions, how much can that be?" Naive math: `8,000 × 25 ≈ 200k tokens`. Reality, measured on our 10,000-turn run at the crash point (turn 8,757): a full-replay agent would have fed the model **~252 billion tokens**. (See `/evidence` → *"Short chats are cheap, they said."*)

---

## What JARVIS CODE does instead

The chat model **never receives the conversation history.** Every turn it gets only:

```
[ system directives  ~500 tokens ]
[ JHB (carried memory)  ~2,000 tokens ]   ← the compressed session
[ the current user message ]
```

That input is **flat (O(1)) in the length of the session** — bounded by the JHB cap, not by the turn number. The **encoder** is what makes this possible: after each turn it compresses what just happened into the JHB (a small Markdown document), so the next turn inherits the memory without inheriting the transcript.

**And the encoder's own input is bounded too** — this is the part that's easy to miss. The encoder never reads the conversation history either. Each turn it sees only *the turn that just happened* plus *the current JHB*, and folds them into the next JHB. So its call is O(1) in session length, exactly like the chat call — both capped by the JHB, neither growing with the turn count. That's the line between this and transcript replay: nothing in the loop ever re-reads the history. The memory is **rewritten forward**, not retrieved from the past.

See [concepts.md](concepts.md) for what the JHB is, and [architecture.md](architecture.md) for how the pieces fit.

---

## Why the encoder is — and should be — a *cheap* model

This is the part people push back on, so here it is plainly:

1. **The encoder's job needs no genius.** It does bounded compression and pattern-matching against a ~2,000-token document. Its reasoning is deliberately capped (effectively off — a budget of ~200 tokens). It is not solving your problem; it is taking notes.
2. **It runs on every single turn.** A frontier model in this seat would burn your rate limit and your budget doing clerical work, every turn, forever.
3. **So you spend intelligence where it matters.** The expensive model is reserved for the coding and the conversation. The mechanical memory work goes to a small, fast model.

Recommended encoder: a fast **~14–24B-class** model — the cheapest capable tier on your plan (see [Which encoder to use](#which-encoder-to-use) below). You set it in [`/model-setting`](providers.md) — see [Configure it](#configure-it) below.

---

## Which encoder to use

Rule of thumb: **use the fastest, cheapest model your existing plan already gives you.** You don't need a second provider or a separate budget for the encoder — whatever lightweight tier you already pay for is the right seat for it. Match it to the subscription you have:

| You already have | chat + subagent | encoder |
|---|---|---|
| **OpenAI / GPT** | GPT-5.5 | GPT-5.4 mini |
| **Claude (Anthropic)** | Opus 4.8 | Haiku |
| **Ollama Cloud** | a frontier model (e.g. GLM-5.2) | Devstral Small 24B |

**One caveat on size.** The encoder compresses, it doesn't reason — but it still has to hold the memory faithfully, and that has a floor. Across a lot of runs, **8B is not enough** — memory quality gets shaky. Aim for **~14B and up** (≈14–24B is the sweet spot). On a subscription the lowest tier is usually fine; but where a provider *also* offers very small models (e.g. Ollama Cloud), don't bottom out — pick a **14B+** model that can still exercise some judgment.

## The honest numbers

**The honest comparison isn't "cheaper" — it's "cheaper at the same memory."** Every other agent's cost knob (`/compact`, `/clear`, a sliding window) is also a *forgetting* knob: it gets cheaper by dropping context. JLC is the one architecture where you cap the cost without proportionally capping what you remember — the forgetting is curated by the encoder, not blind truncation. The numbers below are against a **naive full-replay baseline**; a real agent mitigates with cache and compaction, but only by losing the continuity JLC keeps.

All measured, all recomputable from the public artifacts at [jlc-codec.org/evidence](https://jlc-codec.org/evidence).

| What | Number | Source |
|---|---|---|
| Cumulative prefill, 1,000-turn run | **653K** tokens (JLC) vs **86.6M** (full-replay baseline) = **~99.2% fewer** | `1k-run/` + Zenodo transcript (recompute with tiktoken) |
| Carried memory size | JHB stays ~**2,000 tokens**, flat across the whole run (does not grow with turns) | `meta.json`, `meter.paperlog` |
| Compute gap at turn 8,757 (10k run) | chat **58M** tokens vs **251,950M** full-replay = **~4,346 : 1** | `/evidence` HUD + `meter.paperlog` |
| Real-world cost, full 10k run (chat **and** encoder) | **~41% of one week** on a ~$20/month Ollama Cloud Pro plan | `/evidence` provider usage meter |
| Prompt-cache hit rate (JLC static head) | ~**90%** of input served from cache | `chat_in_trace` |

---

## Where it does **not** win — read this

Honesty is the point; an overclaim that your own AI can disprove is worse than no claim.

- **Short sessions are roughly break-even.** For a ~10-turn chat with a warm prompt cache, a standard agent can be **~15% cheaper** than JLC (you're paying for an encoder call you didn't need yet). The crossover is around **15–20 turns** (a modeled estimate, not a controlled A/B). JLC's advantage is for work that **runs long**.
- **The big ratios are raw compute, not dollars.** "4,346:1" and "~99%" are *tokens fed to the model*. A legacy agent gets prompt-cache discounts that shrink the **money** gap. The real-money figure is the provider meter: ~41% of a week for the whole 10k run.
- **The encoder is not free — just cheap.** You add one small model call per turn.
- **JLC's O(n) is for conversation turns.** It does not shrink the context an agentic tool-loop accumulates *inside* a single turn.
- For the full economic picture (cache math, long vs short), this is deliberately conservative — a floor, not a ceiling.

> **Rule of thumb.** Under ~15 turns with a warm cache, just use a normal chat — JLC won't save you money there. Past **~15–20 turns** it pulls ahead, and the longer the session runs the wider the gap. Reach for it on work that **runs long** — big refactors, multi-day features, anything you'd otherwise have to `/compact` or `/clear`.

---

## Configure it

Model roles live in `~/.jarvis-code/config.yaml` (the single active config — see [providers.md](providers.md)):

```yaml
roles:
  chat:     openai/gpt-5.5          # smart + expensive — coding & conversation
  subagent: openai/gpt-5.5          # tool-running subtasks (defaults to chat)
  encoder:  openai/gpt-5.4-mini     # cheap + fast — memory compression, every turn
  router:   openai/gpt-5.4-mini     # lightweight intent classifier (mirrors encoder)
```

Or set them interactively inside the agent with **`/model-setting`** (chat / subagent / encoder; router follows the encoder automatically). Any provider with an OpenAI-compatible endpoint works — see [providers.md](providers.md).

---

**Next:** [concepts.md](concepts.md) (what the JHB and the roles actually are) · [overview.md](overview.md) (the big picture) · [the evidence](https://jlc-codec.org/evidence) (verify all of the above).
