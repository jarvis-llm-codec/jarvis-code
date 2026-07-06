# JARVIS Code Install

## Requirements

- Node.js 20 or newer, auto-installed on Windows when `winget` is available
- Python 3.10 or newer, auto-installed on Windows when `winget` is available
- Git, auto-installed on Windows when `winget` is available
- Microsoft Visual C++ 2015-2022 Redistributable (x64), auto-installed on
  Windows when `winget` is available
- PowerShell on Windows
- `curl`, `tar`, and a POSIX shell on macOS/Linux

## One-Line Install

Windows:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.ps1 | iex
```

Windows with curl:

```powershell
iex ((curl.exe -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.ps1) -join "`n")
```

macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.sh | sh
```

The installer downloads the release source, installs Node dependencies in
`pi/`, creates `sidecar/.venv`, installs Python dependencies, and registers a
`jarvis` command. It also preloads the local `BAAI/bge-m3` embedding model on
first install; this can download about 2.3 GB. If the preload fails, install
continues and `jarvis doctor --preload-embedder` can be used to inspect or retry
the model load.

On Windows, missing Node.js, Python, Git, or Microsoft Visual C++ Redistributable
(x64) is installed through `winget` when available. The VC++ runtime is required
by the local torch/sentence-transformers stack. Set
`JARVIS_CODE_NO_PREREQ_INSTALL=1` or pass `-NoPrereqInstall` to skip automatic
prerequisite installation.

Set `JARVIS_CODE_NO_MODEL_PRELOAD=1` or pass `-NoModelPreload` on Windows to
skip the bge-m3 preload and let the first JARVIS memory warmup download it.
Set `JARVIS_CODE_REQUIRE_MODEL_PRELOAD=1` or pass `-RequireModelPreload` on
Windows to make preload failure abort installation.

## Configurable Install

Windows:

```powershell
$env:JARVIS_CODE_REPO = "jarvis-llm-codec/jlc"
irm https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.ps1 | iex
```

Windows with curl:

```powershell
$env:JARVIS_CODE_REPO = "jarvis-llm-codec/jlc"
iex ((curl.exe -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.ps1) -join "`n")
```

macOS/Linux:

```bash
JARVIS_CODE_REPO=jarvis-llm-codec/jlc \
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/install.sh | sh
```

Install location defaults:

- Windows: `%LOCALAPPDATA%\JARVIS-Code`
- macOS/Linux: `$HOME/.local/share/jarvis-code`

Override with `JARVIS_CODE_INSTALL_DIR`.

## Manual Source Install

From an extracted release folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

or:

```bash
./install.sh
```

## First Run

JARVIS Code checks for an LLM credential before opening the terminal UI. If no
GPT OAuth or API-key credential is configured, `jarvis` stops and prints setup
commands instead of entering a broken first-run flow.

For GPT OAuth:

```bash
jarvis gpt-login
jarvis
```

If the browser callback is blocked:

```bash
jarvis gpt-login-device
jarvis
```

For API-key providers:

```bash
jarvis api-key
jarvis model-setting
jarvis
```

Normal start after setup:

```bash
jarvis
```

Run diagnostics:

```bash
jarvis doctor
```

To force-check and download the embedding model:

```bash
jarvis doctor --preload-embedder
```

## Uninstall

Windows:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/uninstall.ps1 | iex
```

macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/uninstall.sh | sh
```

Default uninstall removes the install directory and `jarvis` command shim, then
keeps user data and the Hugging Face model cache.

To also remove JARVIS user data on Windows:

```powershell
$env:JARVIS_CODE_REMOVE_USER_DATA = "1"
irm https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/uninstall.ps1 | iex
```

On macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jlc/main/uninstall.sh | sh -s -- --remove-user-data
```

To force a fresh `BAAI/bge-m3` model download on the next install, also set
`JARVIS_CODE_REMOVE_MODEL_CACHE=1` on Windows or pass `--remove-model-cache` on
macOS/Linux.

## What the Installer Creates

- `pi/node_modules/`
- `sidecar/.venv/`
- Hugging Face cache files for `BAAI/bge-m3`
- `data/`
- `pi-agent/`
- copied default skills and themes under `pi-agent/`
- a `jarvis` command shim in the user bin directory

These are local runtime/install artifacts and are not part of the published
source.
