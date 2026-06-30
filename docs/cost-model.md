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

See [concepts.md](concepts.md) for what the JHB is, and [architecture.md](architecture.md) for how the pieces fit.

---

## Why the encoder is — and should be — a *cheap* model

This is the part people push back on, so here it is plainly:

1. **The encoder's job needs no genius.** It does bounded compression and pattern-matching against a ~2,000-token document. Its reasoning is deliberately capped (effectively off — a budget of ~200 tokens). It is not solving your problem; it is taking notes.
2. **It runs on every single turn.** A frontier model in this seat would burn your rate limit and your budget doing clerical work, every turn, forever.
3. **So you spend intelligence where it matters.** The expensive model is reserved for the coding and the conversation. The mechanical memory work goes to a small, fast model.

Recommended encoder: a ~24B-class model such as `ollama-cloud/devstral-small-2:24b`. You set it in [`/model-setting`](providers.md) — see [Configure it](#configure-it) below.

---

## The honest numbers

All measured, all recomputable from the public artifacts at [jlc-codec.org/evidence](https://jlc-codec.org/evidence).

| What | Number | Source |
|---|---|---|
| Cumulative prefill, 1,000-turn run | **653K** tokens (JLC) vs **86.6M** (full-replay baseline) = **99.25% fewer** | `1k-run/` + Zenodo transcript (recompute with tiktoken) |
| Carried memory size | JHB stays ~**2,000 tokens**, flat across the whole run (does not grow with turns) | `meta.json`, `meter.paperlog` |
| Compute gap at turn 8,757 (10k run) | chat **58M** tokens vs **251,950M** full-replay = **~4,346 : 1** | `/evidence` HUD + `meter.paperlog` |
| Real-world cost, full 10k run (chat **and** encoder) | **~41% of one week** on a ~$20/month Ollama Cloud Pro plan | `/evidence` provider usage meter |
| Prompt-cache hit rate (JLC static head) | ~**90%** of input served from cache | `chat_in_trace` |

---

## Where it does **not** win — read this

Honesty is the point; an overclaim that your own AI can disprove is worse than no claim.

- **Short sessions are roughly break-even.** For a ~10-turn chat with a warm prompt cache, a standard agent can be **~15% cheaper** than JLC (you're paying for an encoder call you didn't need yet). The crossover is around **15–20 turns** (a modeled estimate, not a controlled A/B). JLC's advantage is for work that **runs long**.
- **The big ratios are raw compute, not dollars.** "4,346:1" and "99.25%" are *tokens fed to the model*. A legacy agent gets prompt-cache discounts that shrink the **money** gap. The real-money figure is the provider meter: ~41% of a week for the whole 10k run.
- **The encoder is not free — just cheap.** You add one small model call per turn.
- **JLC's O(n) is for conversation turns.** It does not shrink the context an agentic tool-loop accumulates *inside* a single turn.
- For the full economic picture (cache math, long vs short), this is deliberately conservative — a floor, not a ceiling.

---

## Configure it

Model roles live in `~/.jarvis-code/config.yaml` (the single active config — see [providers.md](providers.md)):

```yaml
roles:
  chat:     anthropic-agent-sdk/claude-opus-4-8   # smart + expensive — coding & conversation
  subagent: anthropic-agent-sdk/claude-opus-4-8   # tool-running subtasks (defaults to chat)
  encoder:  ollama-cloud/devstral-small-2:24b     # cheap + fast — memory compression, every turn
  router:   ollama-cloud/devstral-small-2:24b     # lightweight intent classifier (mirrors encoder)
```

Or set them interactively inside the agent with **`/model-setting`** (chat / subagent / encoder; router follows the encoder automatically). Any provider with an OpenAI-compatible endpoint works — see [providers.md](providers.md).

---

**Next:** [concepts.md](concepts.md) (what the JHB and the roles actually are) · [overview.md](overview.md) (the big picture) · [the evidence](https://jlc-codec.org/evidence) (verify all of the above).
