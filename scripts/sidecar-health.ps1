$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:PYTHONPATH = Join-Path $Root "sidecar"
Push-Location (Join-Path $Root "sidecar")
try {
    python -m jarvis_sidecar
} finally {
    Pop-Location
}

