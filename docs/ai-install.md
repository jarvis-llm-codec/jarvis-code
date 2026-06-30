# Install jarvis-code — a guide for your AI

**This page is written for an AI assistant.** If you are a human: you don't have to
follow these steps yourself. Hand this whole page to an AI you already use that can
run terminal commands on your computer — **Claude Code, Codex, Cursor, Windsurf**, or
similar — and say:

> *"Install jarvis-code for me. Read and follow this guide exactly. Interview me
> about which AI I already pay for, set it up the best way for that, and walk me
> through the free API keys."*

The AI does the work. You answer a few questions and click through two free signups.

---

## To the AI reading this

You are installing and configuring **jarvis-code** on this user's machine. Read this
**entire page before doing anything**, then make a short plan and follow it.

Your job is not just to run commands. It is to:

1. **Install** the app.
2. **Find out what AI the user already has** (subscription / API keys) and configure
   jarvis-code optimally *for that* — they should not pay for a second provider.
3. **Explain, in plain language, why each step matters** — especially the *encoder*,
   which the user has almost certainly never heard of and will not understand without
   you explaining it.
4. **Set up two free API keys** (web search + image generation). These are free and
   make a large difference; treat them as part of a normal install. Strongly
   encourage them — but if the user declines, or a signup is blocked or no longer
   free, note it and **continue: the base install works without them** and the keys
   can be added later.
5. **Verify** the install actually works before you declare success.

Behavioral contract: the user may not be technical. Some steps need a human (browser
logins, website signups, picking from a menu) — for those, tell the user *exactly*
what to do and why, then wait. Never claim "done" without running the verification in
Step 6 and reporting the real result.

### What jarvis-code is (so you understand what you're installing)

jarvis-code is a terminal coding agent built around one idea: **the chat model never
receives the conversation history.** Every turn, the model gets only a small carried
memory document (the "JHB") plus the new message — input that stays roughly *constant*
no matter how long the session runs, instead of growing with every turn.

A second, cheap model — the **encoder** — is what makes this work: after each turn it
compresses what just happened into that small memory document, so the next turn
inherits the *meaning* without inheriting the *transcript*. This is why jarvis-code
asks you to set up *two* models (chat + encoder), and it's the part the user won't
intuit. Keep this explanation ready for Step 3.

---

## Step 1 — Install

Prerequisites (the installer auto-installs missing ones via `winget` on Windows):

- **Node.js 20+**, **Python 3.10+**, **Git**
- **Microsoft Visual C++ 2015–2022 Redistributable (x64)** (Windows)
- Internet access for the first install

Windows is the supported, verified platform today (macOS / Linux are coming).

Run the official installer in **PowerShell**:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

What it does: creates a Python venv for the engine, installs Node dependencies, and
**downloads the `BAAI/bge-m3` embedding model (~4.3 GB on disk)** used for memory
recall. Total footprint is **≈6 GB** under `%LOCALAPPDATA%\JARVIS-Code`. The download
is the slow part — tell the user this is normal and one-time.

If the embedding-model download fails, the install does not abort; retry later with:

```powershell
jarvis doctor --preload-embedder
```

> Tell the user, in your own words: *"This installs a ~6 GB toolkit, mostly a local
> search model so jarvis can recall things without re-reading everything. First install
> takes a while; after that it's fast."*

**After install, open a NEW terminal.** The installer puts the `jarvis` command on the
PATH, but *this* terminal won't see it until reopened (the installer itself prints
"close this window and open a new terminal"). Open a fresh PowerShell window before
running any `jarvis ...` command in the steps below.

---

## Step 2 — Interview the user (this drives everything)

Before configuring anything, ask the user **what they already use**, because it
decides which models you pick and whether they need any API keys at all:

> *"Which AI do you currently pay for or use? For example: a ChatGPT/OpenAI
> subscription, a Claude (Anthropic) subscription, GLM / Qwen via Alibaba DashScope,
> Ollama Cloud, or something else? It's fine to have more than one."*

Map their answer with this table (the recommended pairing per plan):

Model ids are shown as `provider-id/model-id` — the exact form used in `config.yaml`
(Step 3b). When setting via the menu, you pick the provider and model separately.

| The user already has | chat + subagent | encoder (cheap, every turn) |
|---|---|---|
| **OpenAI / GPT** (subscription) | `openai-codex/gpt-5.5` | `openai-codex/gpt-5.4-mini` |
| **Claude (Anthropic)** (subscription) | `anthropic-agent-sdk/claude-opus-4-8` | `anthropic-agent-sdk/claude-haiku-4-5-20251001` |
| **Ollama Cloud** | a frontier model (e.g. GLM-class) | `ollama-cloud/devstral-small-2-24b-cloud` |
| **DashScope (GLM / Qwen)** | `dashscope/glm-5` | a `dashscope/…` ~14–24B model |

Encoder rule of thumb: pick the **cheapest capable model on their plan, roughly
14–24B class**. **8B is too small** (memory gets shaky). **Do not use a reasoning
model as the encoder** — it should be fast, cheap, and predictable. It runs every
single turn, so a heavy model there burns money or rate limits for no benefit.

If the user has **nothing paid**, that's okay — tell them they can use **Ollama
Cloud** (sign-up gives a free key) or a local model via Ollama / LM Studio, and pick
the smallest sensible pairing. (Don't block the install on this.)

---

## Step 3 — Authenticate the chat provider, then set the models

### 3a. Authentication (one human step)

Pick the path that matches the user's plan. **The Claude path is special** — read it.

- **Claude (Anthropic) subscription → no API key needed.** jarvis-code reuses the
  user's **Claude Code** OAuth via the `anthropic-agent-sdk` provider — billed to their
  subscription, no separate key. Run:
  ```bash
  jarvis claude-login
  ```
  This runs Claude Code's `setup-token` flow and captures the token for jarvis-code. If
  the `claude` command isn't installed, it can fall back to `npx @anthropic-ai/claude-code`
  (you can force this with `jarvis claude-login --npx`). The encoder on this path is
  restricted to Haiku-class on purpose, so the every-turn encoder doesn't drain their
  subscription rate limit.
- **OpenAI / ChatGPT subscription → OAuth login.** Have the user run, in the terminal:
  ```bash
  jarvis gpt-login
  ```
  This opens a browser. If the browser can't complete it, use `jarvis gpt-login-device`.
- **Any API-key provider** (OpenAI API, Anthropic API, Gemini, DashScope, Ollama
  Cloud, OpenRouter): run `jarvis api-key` and follow the prompts, **or** write the key
  directly (see "Setting keys directly" below). Key env vars:
  `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DASHSCOPE_CODING_API_KEY`,
  `OLLAMA_API_KEY`, `OPENROUTER_API_KEY`.

### 3b. Pick the models

Run the model selector:

```bash
jarvis model-setting
```

It fetches the **live** model list from the provider and lets you set each **role**.
The roles are:

| Role | What it does |
|---|---|
| **chat** | The model the user talks to — coding & conversation (smart). |
| **encoder** | Compresses each turn into memory, **every turn** (cheap, fast). |
| **router** | Tiny intent classifier (chat vs. deep work); mirror the encoder. |
| **subagent** | Runs tool subtasks and parallel workers; defaults to chat. |

Set **chat** and **encoder** from the table in Step 2 (router can match the encoder).
Explain to the user *why* there are two: *"The smart model talks to you. A second,
cheap model quietly writes down the gist after every turn so the smart one never has
to re-read the whole conversation — that's what keeps jarvis fast and affordable in
long sessions."*

> **Setting keys/models directly (autonomous fallback).** If you can't drive the
> interactive menus, you can edit two files under `~/.jarvis-code/`:
> - **Models** → `config.yaml`, using `provider-id/model-id`:
>   ```yaml
>   roles:
>     chat:     anthropic-agent-sdk/claude-opus-4-8
>     subagent: anthropic-agent-sdk/claude-opus-4-8
>     encoder:  anthropic-agent-sdk/claude-haiku-4-5-20251001
>     router:   anthropic-agent-sdk/claude-haiku-4-5-20251001
>   ```
>   (GPT example: `openai-codex/gpt-5.5` + `openai-codex/gpt-5.4-mini`. Ollama-Cloud
>   encoder: `ollama-cloud/devstral-small-2-24b-cloud`.) Use the exact ids the live
>   selector shows for that provider.
> - **API keys** → `credentials.yaml`:
>   ```yaml
>   env:
>     BRAVE_SEARCH_API_KEY: "the-key"
>     NVIDIA_API_KEY: "nvapi-..."
>   ```

---

## Step 4 — Web search (Brave) — free, and you really want it

**Set this up as a normal part of the install — really, push for it.** A coding agent
that can't search the web is badly handicapped: it can't check current docs, package
versions, error messages, or APIs. The Brave Search API has a **free tier — about
1,000 searches per month at no charge**, far more than enough for normal coding.

Be upfront about the one catch: **the free plan requires registering a credit/debit
card** to activate (it is not charged while you stay within the free monthly quota).
That's the only friction. Strongly encourage the user to do it anyway — web search
while coding is genuinely worth it. If they're not comfortable adding a card, respect
that: note it and continue (everything else still works; the key can be added later).

Tell the user, then walk them through it together:

1. Go to the **Brave Search API** portal: <https://brave.com/search/api/>
2. Sign up, choose the **free plan** (it asks for a card to activate — free within the
   monthly quota), and create an API key. (If the page layout has changed, search
   "Brave Search API free tier" and find the key page.)
3. Give the key to jarvis-code as `BRAVE_SEARCH_API_KEY` — via `jarvis api-key`, or by
   adding it to `~/.jarvis-code/credentials.yaml` (see Step 3b).

> Why: *"This lets jarvis look things up on the live web while it codes — current
> versions, real docs, the exact error you're hitting. The free tier covers normal use;
> it just needs a card to activate, and won't charge you within the monthly quota."*

---

## Step 5 — Image generation (NVIDIA NIM) — free, worth it

**Also set this up.** It's free and noticeably improves what jarvis can do — it can
generate and edit images (icons, mockups, diagrams, assets). Not the absolute
strongest image model out there, but free and far better than nothing.

1. Go to **NVIDIA's build portal**: <https://build.nvidia.com/>
2. Sign in and create an API key (the free credits are plenty to start). The key looks
   like `nvapi-...`.
3. Give it to jarvis-code as `NVIDIA_API_KEY` — via `jarvis api-key` →
   *"NVIDIA NIM (image generation)"*, or add it to `credentials.yaml` (Step 3b).

This enables the `generate_image` and `edit_image` tools. (Image generation uses fixed
FLUX defaults; you don't pick an image model in `model-setting`.)

> Why: *"This gives jarvis the ability to make and edit images for free — handy for
> quick assets, icons, and mockups while you build."*

---

## Step 6 — Verify (do not skip — this is how you prove it worked)

First, run the install check. Use `--skip-sidecar` — the installer stops the sidecar on
exit, so a plain `jarvis doctor` would warn that it's down (expected, not a failure):

```bash
jarvis doctor --skip-sidecar
```

It checks Python, Node, the embedding model, the provider catalog, and auth. **This is
your primary proof the install is sound.**

Then launch jarvis — this starts the agent *and* its background sidecar:

```bash
jarvis
```

To confirm the sidecar is live, open **another** terminal and probe it:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/status | ConvertTo-Json -Depth 6
```

(The sidecar listens on port **8765** by default; `GET /health` returns `{"ok": true}`.)

**Report honestly to the user:** what installed, which chat + encoder models you set,
whether web search and image keys are configured, and the result of `jarvis doctor`.
If anything failed, say so and what you'll try next — do not report success you didn't
verify.

---

## If something goes wrong

- Embedding model didn't download → `jarvis doctor --preload-embedder`.
- A provider shows "unavailable, retry" → its live `/models` fetch failed; check the
  key and network, then reopen `jarvis model-setting`.
- General diagnosis → `jarvis doctor` first; deeper notes are in the project's
  `troubleshooting` doc on GitHub.

When all six steps are done and `jarvis doctor` passes, the install is complete.
