# Install

JARVIS Code supports Windows, macOS, and Linux through platform-specific
installers and a shared Python launcher.

## Windows

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

Or with curl on Windows:

```powershell
iex ((curl.exe -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1) -join "`n")
```

## macOS/Linux

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.sh | sh
```

## Requirements

- Node.js 20 or newer
- Python 3.10 or newer
- Git
- Microsoft Visual C++ 2015-2022 Redistributable (x64) on Windows
- Internet access for first install

On Windows, the installer attempts to install missing Node.js, Python, Git, and
Microsoft Visual C++ Redistributable (x64) automatically with `winget`. The VC++
runtime is required by torch for local embedding support. On macOS/Linux, the
installer attempts to install Git with a supported package manager when it is
missing.

The installer creates a Python virtual environment for the sidecar and installs
Node dependencies for the internal agent engine. It also preloads the local
`BAAI/bge-m3` embedding model; first install can download about 2.3 GB. Preload
failure does not abort install by default; run `jarvis doctor --preload-embedder`
to retry and inspect the error.

Skip model preload when needed:

```powershell
$env:JARVIS_CODE_NO_MODEL_PRELOAD = "1"
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/install.ps1 | iex
```

Run diagnostics after install:

```bash
jarvis doctor
```

## First Run Auth

`jarvis` checks for an LLM credential before opening the terminal UI. If neither
GPT OAuth nor an API-key credential is configured, it stops and prints setup
commands.

GPT OAuth:

```bash
jarvis gpt-login
```

If browser login cannot complete:

```bash
jarvis gpt-login-device
```

API-key providers:

```bash
jarvis api-key
jarvis model-setting
```

Then start:

```bash
jarvis
```

## Uninstall

Windows:

```powershell
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/uninstall.ps1 | iex
```

macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/uninstall.sh | sh
```

Default uninstall removes the install directory and command shim. It does not
remove `~/.jarvis-code`, `C:\jarvis_workspace`, the Hugging Face model cache, or
system prerequisites such as Node.js, Python, Git, and VC++ Redistributable.

Optional full local JARVIS data removal:

```powershell
$env:JARVIS_CODE_REMOVE_USER_DATA = "1"
irm https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/uninstall.ps1 | iex
```

or:

```bash
curl -fsSL https://raw.githubusercontent.com/jarvis-llm-codec/jarvis-code/main/uninstall.sh | sh -s -- --remove-user-data
```

## Runtime Locations

Default install locations:

- Windows: `%LOCALAPPDATA%\JARVIS-Code`
- macOS/Linux: `$HOME/.local/share/jarvis-code`

Runtime state is stored under the install directory unless overridden by
environment variables:

- `data/`: sidecar runtime state and logs
- `pi-agent/`: local agent auth, settings, sessions, and copied default resources
- `sidecar/.venv/`: Python virtual environment
- `pi/node_modules/`: Node dependencies
- Hugging Face cache: local `BAAI/bge-m3` embedding model files

Do not copy these folders into a public release.
