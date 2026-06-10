# Thin wrapper that runs scripts/llmsetting.py via the sidecar venv (or system python).
# All real logic lives in the Python script.

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent
$UserProfileDir = [Environment]::GetFolderPath("UserProfile")
if (-not $UserProfileDir) { $UserProfileDir = $HOME }
$DefaultConfigPath = Join-Path (Join-Path $UserProfileDir ".jarvis-code") "config.yaml"
$ConfigPath = if ($env:JARVIS_CODE_CONFIG) { $env:JARVIS_CODE_CONFIG } else { $DefaultConfigPath }
$env:JARVIS_CODE_CONFIG = $ConfigPath
$venvPython = Join-Path $repoRoot "sidecar\.venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $py = $venvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $py = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $py = "py"
} else {
    Write-Error "Python not found. Run sidecar bootstrap first (cd sidecar; python -m pip install -r requirements.txt)."
    exit 2
}

$script = Join-Path $PSScriptRoot "llmsetting.py"
& $py $script @args
exit $LASTEXITCODE
