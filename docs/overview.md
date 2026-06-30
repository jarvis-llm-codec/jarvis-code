# Overview — what JARVIS CODE is, and why it's different

**JARVIS CODE is a terminal coding agent that doesn't forget.** It holds one continuous conversation — across turns, restarts, and model swaps — without ever running `/compact` or `/clear`. It does this by carrying memory **outside** the context window instead of replaying a transcript that grows every turn.

It's open source (Apache-2.0), built on the [pi-agent](https://github.com/earendil-works/pi) harness (MIT) with the JLC — **JARVIS LLM Codec** — embedded.

> 🤖 **Skeptical? Have your AI check us.** This whole site is designed to survive an audit. Paste these docs (or the raw logs at **[jlc-codec.org/evidence](https://jlc-codec.org/evidence)**) into your AI and ask: *"Do these cost claims hold up? Should I use this?"* Every load-bearing number is recomputable from a public 10,000-turn run.

---

## Why it's different

| | The usual coding agent | JARVIS CODE |
|---|---|---|
| **Memory** | Replays the whole conversation every turn | Carries a small compressed memory (JHB) outside the window |
| **Long sessions** | Fills up → `/compact` or `/clear` → loses context | No compact, no clear — one unbroken session |
| **Cost curve** | Grows with turn count — O(n²) | Flat per turn — O(n) |
| **Restart / model swap** | Starts over | Picks up exactly where it left off |
| **New codebase** | Re-explain it each session | Understands it from persistent project memory |

The mechanism, with honest numbers (including where it *doesn't* win): **[cost-model.md](cost-model.md)**.

---

## The four model roles

Instead of "an AI," JARVIS CODE uses four named roles — each a model you choose in [`/model-setting`](providers.md):

| Role | Job | Typical model |
|---|---|---|
| **chat** | The model you talk to — coding and conversation | smart / expensive |
| **encoder** | Compresses each turn into the carried memory (JHB), every turn | cheap / fast (e.g. a 24B model) |
| **router** | Lightweight intent classifier (deep-dive vs chat vs build) | mirrors the encoder |
| **subagent** | Runs tool-calling subtasks and ultracode fan-out workers | defaults to chat |

The chat/encoder split is the heart of the cost story — see **[cost-model.md](cost-model.md)** for why running a *second, cheaper* model lowers total cost on long work.

---

## The thesis, in one line

> Don't grow the context. Carry the memory.

An LLM is stateless and attention is O(n²) in length — the longer the chat, the more it costs and the more it degrades, until something has to be thrown away. JLC accepts the forgetting and writes the memory down outside the window. The cost curve bends from quadratic to linear, and the conversation stops having a ceiling.

Paper: *"Forgetting Is All You Need"* — [Zenodo DOI](https://doi.org/10.5281/zenodo.20266424).

---

## Proof, not promises

- A public **10,000-turn** run on a consumer mini-PC — raw logs, integrity hashes, and a live recording at **[jlc-codec.org/evidence](https://jlc-codec.org/evidence)**.
- Across 925 adversarial fabrication traps it stayed honest on **923** (and we publish the 2 it missed, by turn number).
- The entire run — chat *and* encoder — fit in **~41% of one week** on a ~$20/month plan.

---

## Where to go next

1. **[quickstart.md](quickstart.md)** — install and run your first session (5 minutes)
2. **[concepts.md](concepts.md)** — the vocabulary: deep-dive vs chat, JARVIS.md, JHB, pair, sidecar, ultracode
3. **[cost-model.md](cost-model.md)** — why two models, and why the encoder is cheap
4. **[providers.md](providers.md)** — pick your models and providers
5. **[faq.md](faq.md)** — the questions skeptics actually ask
6. **[architecture.md](architecture.md)** — how the engine, codec, and extension fit together

Reference: [install.md](install.md) · [memory-and-projects.md](memory-and-projects.md) · [troubleshooting.md](troubleshooting.md)
