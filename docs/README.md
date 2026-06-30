# JARVIS CODE — Documentation

A terminal coding agent that doesn't lose the thread. One continuous conversation across turns, restarts, and model swaps — no `/compact`, no `/clear`. Open source, Apache-2.0.

> 🤖 **Evaluating it with an AI?** Paste these docs into your assistant and ask whether the cost claims hold up. The raw proof — public 10,000-turn run, logs, hashes — is at **[jlc-codec.org/evidence](https://jlc-codec.org/evidence)**.

---

## Read in this order

**Start here**
1. [overview.md](overview.md) — what it is and why it's different (start here)
2. [quickstart.md](quickstart.md) — install and run your first session
3. [concepts.md](concepts.md) — the vocabulary: chat vs deep-dive, JARVIS.md, JHB, encoder, pair, sidecar, ultracode

**Understand it**
4. [cost-model.md](cost-model.md) — ⭐ why two models, and why the encoder is cheap (the honest numbers)
5. [architecture.md](architecture.md) — engine + codec + extension, and what goes in the model's context
6. [memory-and-projects.md](memory-and-projects.md) — project memory, deep-dive modes, runtime knobs

**Reference**
7. [providers.md](providers.md) — choose models per role; custom providers (YAML / UI)
8. [install.md](install.md) — full install reference, requirements, uninstall
9. [troubleshooting.md](troubleshooting.md) — common failures and fixes
10. [faq.md](faq.md) — the questions skeptics actually ask

**Proof**
- [jlc-codec.org/evidence](https://jlc-codec.org/evidence) — raw 1k/10k run artifacts, integrity hashes, live recording
- Paper: *"Forgetting Is All You Need"* — [Zenodo](https://doi.org/10.5281/zenodo.20266424)

---

## For maintainers / contributors (internal)

These are engineering notes, not user docs:

- [cross-platform-porting.md](cross-platform-porting.md) — porting audit and per-subsystem coupling
- [release-packaging.md](release-packaging.md) — how a release is built and published

---

*Built on [pi-agent](https://github.com/earendil-works/pi) (MIT) by Mario Zechner · JLC core embedded & tuned.*
