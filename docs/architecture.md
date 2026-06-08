# Architecture

JARVIS Code has three main parts.

## Terminal Agent Engine

The internal engine lives under `pi/`. This folder name is kept for
compatibility with the forked codebase. It is not the public product name.

The engine provides:

- terminal UI
- file tools
- shell execution
- model provider plumbing
- extension hooks
- session persistence

## JLC Sidecar

The Python sidecar lives under `sidecar/`. It provides:

- project registry
- project switch and creation support
- raw turn storage
- JHB/JLC memory retrieval
- web fetch, official-doc search, package metadata, and browser smoke-check helpers
- provider role configuration
- `/context`, `/turn`, and related endpoints

External-information tools are split by job:

- `web_search`: search the public web through Brave Search
- `web_fetch`: read a specific public URL after search or when the user gives a link
- `docs_search`: search official documentation domains and optionally fetch top results
- `package_info`: read npm, PyPI, or GitHub release metadata in structured form
- `browser_check`: smoke-check a public or localhost URL; uses a local headless
  Chrome/Edge/Chromium when available and falls back to HTTP checks otherwise

The launcher starts the sidecar before starting the terminal agent.

## JARVIS Extension

The JARVIS extension lives at:

```text
pi/packages/coding-agent/examples/extensions/jarvis-jlc.ts
```

It connects the terminal agent to the sidecar. It injects JLC memory into the
latest user turn and trims provider payloads so old chat prefixes are not sent
as raw replay.

## Context Policy

JARVIS Code keeps durable history locally for recall, but the live model payload
is assembled from:

- system/developer prompt
- the current user turn
- JLC memory context
- current-turn tool calls and tool results

The live agent runtime keeps only a bounded recent user-turn fallback. Durable
long-term memory belongs to the sidecar and raw store, not the model prefix.
