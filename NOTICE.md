# Notices

JARVIS Code is Copyright 2026 Jun, licensed under the Apache License,
Version 2.0 (see `LICENSE`).

JARVIS Code is an independent fork derived from Pi by Earendil Works.

JARVIS Code is no longer a Pi distribution and does not receive automatic Pi
upstream updates. It keeps some internal folder names, package names, and import
paths from the forked codebase for compatibility.

## Pi Attribution

The internal agent engine under `pi/` is derived from the Pi Agent Harness Mono
Repo and related packages by Earendil Works and contributors.

Original project references:

- https://pi.dev
- https://github.com/earendil-works/pi
- https://github.com/earendil-works/pi-mono

The original Pi code is Copyright (c) 2025 Mario Zechner, distributed under the
MIT license. A release package
must preserve the original Pi license, either at `pi/LICENSE` when the full
engine source is included, or under `THIRD_PARTY_LICENSES/PI-LICENSE` when the
engine docs are stripped from the public release surface.

## JARVIS Code Product Boundary

User-facing product name: `JARVIS Code`

User-facing command: `jarvis`

Internal fork folder: `pi/`

The `pi/` folder is an internal engine implementation detail. User-facing
documentation, installer messages, release notes, and update flows should refer
to JARVIS Code, not Pi.
