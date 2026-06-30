# FAQ

Straight answers, including the unflattering ones. The full numbers and raw logs are at [jlc-codec.org/evidence](https://jlc-codec.org/evidence).

---

### Two models? Isn't that more expensive, not less?

For a short chat, slightly — yes. For work that runs long, no, and the gap widens fast. The expensive **chat** model only ever sees a small, fixed-size context; the cheap **encoder** does the per-turn memory work that a normal agent pays the *expensive* model to redo every turn (by replaying the whole transcript). Full mechanism and numbers: **[cost-model.md](cost-model.md)**.

### So is it actually cheaper?

**Not always — and we won't pretend otherwise.** Short sessions (~10 turns) with a warm cache: a standard agent can be ~15% cheaper. The crossover is around 15–20 turns; past that JLC pulls ahead, reaching ~2.5–3× on long/mixed sessions (modeled estimate). The dramatic ratios you'll see (e.g. ~4,346:1) are *raw compute*, not dollars — cache discounts shrink the money gap. JLC is for work that **runs long**.

### Can I use a free or local model?

Yes. Any provider with an OpenAI-compatible endpoint works, including local models via Ollama. A common setup is a frontier model for **chat** and a small local/cloud model for **encoder**. See [providers.md](providers.md).

### Do I need an API key?

Depends on your provider. A Claude subscription works without a separate API key; other providers use a key. You set this up on first launch — see [quickstart.md](quickstart.md).

### Is my code or conversation sent anywhere I don't control?

Your memory (JHB and JARVIS.md) lives **on your machine**. Model calls go to whatever provider you configure — same as any agent. The sidecar that does the encoding runs locally ([concepts.md](concepts.md)).

### What encoder model should I use?

A **~14–24B class model is the sweet spot** — e.g. `ollama-cloud/devstral-small-2:24b`. The encoder does bounded compression, not reasoning, so a small fast model is *correct*, not a compromise. But don't go too small: **8B isn't enough** (memory quality gets shaky) — aim for ~14B and up. See [cost-model.md](cost-model.md).

### Does it work offline?

The agent and its memory are local, but model inference needs whatever provider you point it at. With a local model server (e.g. Ollama running locally), you can run fully offline.

### Is the 10,000-turn run real?

Yes, and we publish the receipts: raw per-turn logs, the final ~2KB memory, integrity hashes, and an unedited recording — at [jlc-codec.org/evidence](https://jlc-codec.org/evidence). It crashed once at ~8,700 turns (a host-runtime memory limit, **not** JLC), recovered from its on-disk memory, and finished. We left the crash in the footage on purpose.

### How is this different from Claude Code / Cursor / Aider?

Those are excellent — until the conversation grows. They replay (and eventually compact or clear) the transcript, so cost grows and context gets dropped. JARVIS CODE carries memory outside the window, so one session can run indefinitely and resume after restarts or model swaps. It's the difference between hiding statelessness and being built for it ([overview.md](overview.md)).

### What's the catch?

It's young, Windows is the first-class install today (macOS/Linux installer is coming), and short one-off chats won't save you money. It shines on long-running, multi-session coding work — which is most real work.

---

**More:** [overview.md](overview.md) · [cost-model.md](cost-model.md) · [troubleshooting.md](troubleshooting.md) · [the evidence](https://jlc-codec.org/evidence)
