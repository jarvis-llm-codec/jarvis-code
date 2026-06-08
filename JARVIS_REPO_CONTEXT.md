# JARVIS Repo Context

This is product-owned JARVIS Code context. Release builds place it at the
installed JARVIS root so JARVIS can reason about its own layout even when the
current working directory is inside the bundled `pi` engine.

Identity:
- JARVIS Code is a standalone product forked from Pi Agent. Do not assume
  upstream Pi update behavior or Pi project semantics.
- The installed JARVIS root contains `jarvis.ps1` / `jarvis.sh`, `pi`,
  `sidecar`, `pi-agent`, `data`, and `jarvis-resources`.

Important paths:
- JARVIS install root: the parent directory of `pi-agent`, or the launcher root.
- Pi engine: `<install-root>/pi`.
- Sidecar: `<install-root>/sidecar`.
- Local agent state: `<install-root>/pi-agent`.
- Internal JARVIS data root: `~/.jarvis-code`.
- Internal memory-project root: `~/.jarvis-code/workspaceMemory`.
- Active project registry: `~/.jarvis-code/workspaceMemory/workspace_registry.json`.
- Default user code root: `C:\jarvis_workspace` on Windows, `~/jarvis_workspace`
  on Unix-like systems unless configured otherwise.

Rules:
- Do not use `~/.jarvis-code/registry.json` as the active project registry.
  That is legacy agentic registry state; current project routing uses
  `workspace_registry.json`.
- Keep internal JARVIS state and user code separate.
- Do not create user projects under `~/.jarvis-code` or inside the installed
  JARVIS root.
- Treat the install root, `pi`, `sidecar`, `pi-agent`, and `~/.jarvis-code` as
  protected/internal paths.
- Project-internal read/edit/write/bash is allowed for registered projects.
- Reading outside registered projects is allowed, but editing/deleting/writing
  outside registered projects requires explicit user confirmation for the
  session.
- Durable project memory uses `JARVIS.md`; old split memory files such as
  `jarvis/NOW.md` and `jarvis/MAP.md` are retired.
- Do not update this file via memory tools. It is read-only repo/product
  context.
- When updating multiple JARVIS.md memory sections, prefer one batched
  `update_jarvis_md` call with `updates=[{field,value}, ...]`.

Useful checks:
- Sidecar status:
  `Invoke-RestMethod http://127.0.0.1:8765/status | ConvertTo-Json -Depth 6`
- Registered projects:
  `Invoke-RestMethod http://127.0.0.1:8765/projects | ConvertTo-Json -Depth 8`
- Doctor:
  `jarvis doctor`
- If memory looks degraded, check `/status` for `agent_loaded`,
  `last_agent_error`, `registry_path`, and `registry_project_count`.
