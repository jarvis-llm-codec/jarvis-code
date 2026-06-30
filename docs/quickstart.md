# Quickstart — your first session in 5 minutes

This gets you from zero to a working coding session. For the full install reference (requirements, uninstall, folder locations) see [install.md](install.md); to understand what you're looking at, see [concepts.md](concepts.md).

---

## 1. Install

**Windows** (supported today) — in PowerShell:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

**macOS · Linux** — installer is being finalized (**coming soon**). Until then, install from source — see [install.md](install.md).

### What gets installed & how big

Everything lands under `%LOCALAPPDATA%\JARVIS-Code`. Rough sizes on a fresh machine:

| Component | Size |
|---|---|
| App + Node dependencies | ~580 MB |
| Python sidecar (incl. PyTorch, CPU) | ~1.3 GB |
| `bge-m3` embedding model — *optional* | ~2.3 GB download · ~4.3 GB on disk |
| Prerequisites (Node 20+, Python 3.10+, Git, MSVC) — *only if missing* | ~0.5 GB |

**≈ 2 GB without the model, ≈ 6 GB with it** (plus prerequisites if you don't already have them). The model powers local recall and is optional — skip it with `JARVIS_CODE_NO_MODEL_PRELOAD=1`. Memory data under `~/.jarvis-code` starts small and grows slowly.

---

## 2. Launch

```
jarvis
```

On first launch it walks you through:

1. **Provider login** — connect a model provider (e.g. a Claude subscription, or an Ollama Cloud key). No setup files to hand-edit.
2. **Model roles** — pick the model for each role. The two that matter:
   - **chat** — the smart, expensive model that does the coding
   - **encoder** — a cheap, fast model that handles memory (runs every turn)

   A good starting split: a frontier model for **chat**, a ~24B model (e.g. `ollama-cloud/devstral-small-2:24b`) for **encoder**. Why this split saves money: [cost-model.md](cost-model.md). To change it later: `/model-setting`.

Subsequent launches skip all of this — it remembers.

---

## 3. Open a project

Run `jarvis` from inside a code repository. The agent keeps a per-project memory file (**JARVIS.md**) so it learns your codebase as you work — and remembers it next time. Switching projects needs no re-setup or re-explaining ([concepts.md](concepts.md)).

---

## 4. Work — and keep going

Just talk to it: ask questions, describe changes, let it build. Two things you'll notice that other agents don't do:

- **It never stops to `/compact` or `/clear`.** The conversation keeps going — proven across 10,000 turns ([evidence](https://jlc-codec.org/evidence)).
- **Close it and come back tomorrow** — it resumes exactly where you left off, including *why* it abandoned an approach.

Light reasoning for everyday chat; for coding it pushes reasoning to the limit — without fear of burning context, because memory lives outside the window.

---

## Next steps

- [concepts.md](concepts.md) — what JARVIS.md, the JHB, "pair", and the sidecar actually are
- [cost-model.md](cost-model.md) — why the two-model setup is cheaper on long work
- [providers.md](providers.md) — connect other providers / local models
- [troubleshooting.md](troubleshooting.md) — if something doesn't start
