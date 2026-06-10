# Troubleshooting

## `jarvis` Command Not Found

Restart the terminal after installation. If it still fails, add the user bin
directory printed by the installer to `PATH`.

Default command shim locations:

- Windows: `%LOCALAPPDATA%\JARVIS-Code\bin`
- macOS/Linux: `$HOME/.local/bin`

## Node Dependencies Missing

Run the installer again, or run manually:

```bash
cd pi
npm ci
```

If `npm ci` fails because the lock file is stale, use:

```bash
npm install
```

## Sidecar Does Not Start

Check Python and recreate the sidecar environment:

```bash
python -m venv sidecar/.venv
sidecar/.venv/bin/python -m pip install -r sidecar/requirements.txt
```

On Windows:

```powershell
python -m venv sidecar\.venv
sidecar\.venv\Scripts\python.exe -m pip install -r sidecar\requirements.txt
```

## Torch or bge-m3 Fails on Windows

If `jarvis doctor` reports `torch\lib\c10.dll` or `WinError 126`, install the
Microsoft Visual C++ 2015-2022 Redistributable (x64):

```powershell
winget install --id Microsoft.VCRedist.2015+.x64 --exact
```

Then run:

```powershell
jarvis doctor --preload-embedder
```

## Memory Looks Degraded

Check the sidecar:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/status | ConvertTo-Json -Depth 6
```

or:

```bash
curl http://127.0.0.1:8765/status
```

If `jarvis doctor` reports `agent_loaded=False` right after an install or
update, an older sidecar process may still be serving port `8765`. Run `jarvis`
once to let the wrapper restart the sidecar from the current install.

## Provider Missing Or Unavailable In Model Setting

Run:

```bash
jarvis doctor
```

If the provider is missing, check the catalog line. Without a user overlay it
prints `user overlay: none (~/.jarvis-code/llm_catalog.user.yaml)`. With an
overlay it prints the bundled provider count plus the number of valid user
providers. Invalid overlay entries are skipped with a warning; each custom
provider needs `base_url` and a known `api_format`.

If the provider is visible but unavailable, set the provider's `auth_env`
environment variable to `YOUR_KEY_HERE` in the terminal that launches JARVIS,
then run `jarvis model-setting` again. For providers without a working
`/models` endpoint, add `models_static` in
`~/.jarvis-code/llm_catalog.user.yaml`.

See [Providers](providers.md) for overlay examples.

## Custom Provider Saved But Fetch Failed

`/api-key` saves a custom provider before it tests `/models`. If it prints
`saved, but could not reach /models`, the overlay entry and key were still
written. Start the local server, fix the base URL or key with `/api-key`, or add
`models_static` manually for providers that do not expose a compatible
`/models` endpoint.

## Remove Local Runtime State

Only do this when you intentionally want a fresh local install:

- `data/`
- `pi-agent/`
- `sidecar/.venv/`
- `pi/node_modules/`

Do not remove project source code under your own workspace.

For normal product removal, use `uninstall.ps1` or `uninstall.sh`. The default
uninstall keeps user data and model cache; pass the documented removal options
only when you intentionally want a fresh JARVIS state.
