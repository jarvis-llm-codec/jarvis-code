param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$PiRoot = Join-Path $Root "pi"
$PiRunnerPath = Join-Path $PiRoot "pi-test.ps1"
$SidecarRoot = Join-Path $Root "sidecar"
$SidecarVenvDir = Join-Path $SidecarRoot ".venv"
$SidecarVenvPython = Join-Path $SidecarVenvDir "Scripts\python.exe"
$SidecarVenvSitePackages = Join-Path $SidecarVenvDir "Lib\site-packages"
$SidecarRequirements = Join-Path $SidecarRoot "requirements.txt"
$DoctorScript = Join-Path $Root "scripts\jarvis-doctor.py"
$PiAgentDir = Join-Path $Root "pi-agent"
$DefaultResourcesDir = Join-Path $Root "jarvis-resources"
$ExtensionPath = Join-Path $PiRoot "packages\coding-agent\examples\extensions\jarvis-jlc.ts"
$FaceExtensionPath = Join-Path $PiRoot "packages\coding-agent\examples\extensions\jarvis-face.ts"
$ImageExtensionPath = Join-Path $PiRoot "packages\coding-agent\examples\extensions\jarvis-image.ts"
$DataDir = Join-Path $Root "data"
$SidecarRuntimePath = Join-Path $DataDir "sidecar-runtime.json"
$WrapperLogPath = Join-Path $DataDir "jarvis-wrapper.log"
$SidecarWatchdogPath = Join-Path $DataDir "sidecar-watchdog-$PID.run"
$AuthScript = Join-Path $Root "scripts\jarvis-auth.py"
$DryRun = $env:JARVIS_WRAPPER_DRY_RUN -eq "1"
$SkipSidecar = $DryRun -or ($RemainingArgs -contains "--help") -or ($RemainingArgs -contains "-h") -or ($RemainingArgs -contains "--version") -or ($RemainingArgs -contains "-v")
$EnableExtensionDiscovery = $env:JARVIS_ENABLE_EXTENSION_DISCOVERY -eq "1"
$HasProviderArg = $RemainingArgs -contains "--provider"
$HasModelArg = $RemainingArgs -contains "--model"
$DefaultProviderArgs = @()
$DefaultProvider = $env:JARVIS_DEFAULT_PROVIDER
$DefaultModel = $env:JARVIS_DEFAULT_MODEL
$ConfigPath = if ($env:JARVIS_CODE_CONFIG) { $env:JARVIS_CODE_CONFIG } else { Join-Path $Root "data\config.yaml" }
# The sidecar and in-app /model-setting must resolve the same roles file
# regardless of whether chat provider/model were supplied as CLI or
# environment defaults.
$env:JARVIS_CODE_CONFIG = $ConfigPath
if (-not $env:JARVIS_CODE_CODING_AGENT_DIR) {
    $env:JARVIS_CODE_CODING_AGENT_DIR = $PiAgentDir
}
if (-not $env:PI_CODING_AGENT_DIR) {
    $env:PI_CODING_AGENT_DIR = $env:JARVIS_CODE_CODING_AGENT_DIR
}
$JarvisAuthCommands = @(
    "gpt-login",
    "gpt-login-device",
    "gpt-auth-status",
    "gpt-logout",
    "api-key",
    "model-setting",
    "auth-status"
)
$script:JarvisAuthLastExitCode = 0

function Invoke-JarvisDoctor {
    param([string[]] $DoctorArgs)
    $python = $SidecarVenvPython
    if (-not (Test-Path $python)) {
        $found = Get-Command python -ErrorAction SilentlyContinue
        if (-not $found) {
            throw "Python is required to run JARVIS doctor, and sidecar venv was not found at $SidecarVenvPython"
        }
        $python = $found.Source
    }
    if (-not (Test-Path $DoctorScript)) {
        throw "JARVIS doctor script not found at $DoctorScript"
    }
    $oldPythonPath = $env:PYTHONPATH
    try {
        $pythonPathParts = @($SidecarRoot)
        if ($oldPythonPath) {
            $pythonPathParts += $oldPythonPath
        }
        $env:PYTHONPATH = [string]::Join(";", $pythonPathParts)
        & $python $DoctorScript @DoctorArgs
        exit $LASTEXITCODE
    } finally {
        if ($null -eq $oldPythonPath) {
            Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $oldPythonPath
        }
    }
}

function Invoke-JarvisAuth {
    param([string[]] $AuthArgs)
    $python = $SidecarVenvPython
    if (-not (Test-Path $python)) {
        $found = Get-Command python -ErrorAction SilentlyContinue
        if (-not $found) {
            $found = Get-Command py -ErrorAction SilentlyContinue
        }
        if (-not $found) {
            throw "Python is required to run JARVIS auth setup, and sidecar venv was not found at $SidecarVenvPython"
        }
        $python = $found.Source
    }
    if (-not (Test-Path $AuthScript)) {
        throw "JARVIS auth script not found at $AuthScript"
    }
    if ($AuthArgs.Count -gt 0 -and $AuthArgs[0] -eq "auth-status") {
        $rest = @()
        if ($AuthArgs.Count -gt 1) {
            $rest = $AuthArgs[1..($AuthArgs.Count - 1)]
        }
        $AuthArgs = @("gpt-auth-status") + $rest
    }
    $oldPythonPath = $env:PYTHONPATH
    try {
        $pythonPathParts = @($SidecarRoot)
        if ($oldPythonPath) {
            $pythonPathParts += $oldPythonPath
        }
        $env:PYTHONPATH = [string]::Join(";", $pythonPathParts)
        & $python $AuthScript @AuthArgs
        $script:JarvisAuthLastExitCode = $LASTEXITCODE
    } finally {
        if ($null -eq $oldPythonPath) {
            Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $oldPythonPath
        }
    }
}

function Write-Utf8NoBomFile {
    param(
        [string] $Path,
        [string] $Content
    )
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Set-JarvisDefaultSettings {
    $settingsPath = Join-Path $PiAgentDir "settings.json"
    $settings = $null
    $hadBom = $false
    if (Test-Path $settingsPath) {
        try {
            $bytes = [System.IO.File]::ReadAllBytes($settingsPath)
            $hadBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
            $content = [System.Text.Encoding]::UTF8.GetString($bytes).TrimStart([char]0xFEFF)
            $settings = $content | ConvertFrom-Json
        } catch {
            return
        }
    }
    if (-not $settings) {
        $settings = [pscustomobject]@{}
    }

    $themeProperty = $settings.PSObject.Properties["theme"]
    $hasTheme = $themeProperty -and -not [string]::IsNullOrWhiteSpace([string]$themeProperty.Value)
    if ($hasTheme -and -not $hadBom) {
        return
    }

    if (-not $hasTheme) {
        if ($themeProperty) {
            $settings.theme = "orange-blue"
        } else {
            $settings | Add-Member -NotePropertyName "theme" -NotePropertyValue "orange-blue"
        }
    }
    Write-Utf8NoBomFile -Path $settingsPath -Content (($settings | ConvertTo-Json -Depth 20) + "`n")
}

function Initialize-JarvisDefaultResources {
    if (-not (Test-Path $PiAgentDir)) {
        New-Item -ItemType Directory -Path $PiAgentDir -Force | Out-Null
    }

    $sourceSkills = Join-Path $DefaultResourcesDir "skills"
    if (Test-Path $sourceSkills) {
        $targetSkills = Join-Path $PiAgentDir "skills"
        New-Item -ItemType Directory -Path $targetSkills -Force | Out-Null
        foreach ($skillDir in Get-ChildItem -LiteralPath $sourceSkills -Directory -ErrorAction SilentlyContinue) {
            Copy-Item -LiteralPath $skillDir.FullName -Destination $targetSkills -Recurse -Force
        }
    }

    $sourceThemes = Join-Path $DefaultResourcesDir "themes"
    if (Test-Path $sourceThemes) {
        $targetThemes = Join-Path $PiAgentDir "themes"
        New-Item -ItemType Directory -Path $targetThemes -Force | Out-Null
        foreach ($themeFile in Get-ChildItem -LiteralPath $sourceThemes -File -Filter "*.json" -ErrorAction SilentlyContinue) {
            Copy-Item -LiteralPath $themeFile.FullName -Destination (Join-Path $targetThemes $themeFile.Name) -Force
        }
    }

    Set-JarvisDefaultSettings
}

Initialize-JarvisDefaultResources

if ($RemainingArgs.Count -gt 0 -and $RemainingArgs[0] -eq "doctor") {
    $DoctorArgs = @()
    if ($RemainingArgs.Count -gt 1) {
        $DoctorArgs = $RemainingArgs[1..($RemainingArgs.Count - 1)]
    }
    Invoke-JarvisDoctor -DoctorArgs $DoctorArgs
}

if ($RemainingArgs.Count -gt 0 -and $JarvisAuthCommands -contains $RemainingArgs[0]) {
    $AuthArgs = @($RemainingArgs[0])
    if ($RemainingArgs.Count -gt 1) {
        $AuthArgs += $RemainingArgs[1..($RemainingArgs.Count - 1)]
    }
    Invoke-JarvisAuth -AuthArgs $AuthArgs
    exit $script:JarvisAuthLastExitCode
}

function Get-RolesChatFromConfig {
    # Reads roles.chat from data/config.yaml as `provider/model`.
    # Returns hashtable @{ provider; model } on success, $null otherwise.
    # Uses a regex parser (not full YAML) because we only need one stable line
    # and we don't want to depend on PyYAML / sidecar bootstrap just to launch.
    param([string] $ConfigPath)
    if (-not $ConfigPath -or -not (Test-Path $ConfigPath)) { return $null }
    try {
        $content = Get-Content -Raw -LiteralPath $ConfigPath
    } catch {
        return $null
    }
    $rolesMatch = [regex]::Match($content, '(?ms)^roles:\s*\r?\n((?:[ \t]+[^\r\n]+\r?\n?)+)')
    if (-not $rolesMatch.Success) { return $null }
    $chatMatch = [regex]::Match($rolesMatch.Groups[1].Value, '(?m)^[ \t]+chat:[ \t]*(\S+)')
    if (-not $chatMatch.Success) { return $null }
    $chat = $chatMatch.Groups[1].Value.Trim('"', "'")
    if ($chat -notmatch '^([^/\s]+)/(.+)$') { return $null }
    return @{ provider = $Matches[1]; model = $Matches[2].Trim() }
}

function Test-BuiltInProvider {
    param([string] $Provider)
    $builtIns = @(
        "amazon-bedrock",
        "anthropic",
        "openai",
        "azure-openai-responses",
        "openai-codex",
        "deepseek",
        "google",
        "google-vertex",
        "github-copilot",
        "openrouter",
        "vercel-ai-gateway",
        "xai",
        "groq",
        "cerebras",
        "zai",
        "mistral",
        "minimax",
        "minimax-cn",
        "moonshotai",
        "moonshotai-cn",
        "huggingface",
        "fireworks",
        "together",
        "opencode",
        "opencode-go",
        "kimi-coding",
        "cloudflare-workers-ai",
        "cloudflare-ai-gateway",
        "xiaomi",
        "xiaomi-token-plan-cn",
        "xiaomi-token-plan-ams",
        "xiaomi-token-plan-sgp"
    )
    return $builtIns -contains $Provider
}

function Test-RegisteredPiModel {
    param(
        [string] $Provider,
        [string] $Model
    )
    $modelsPath = Join-Path $PiAgentDir "models.json"
    if (-not (Test-Path $modelsPath)) { return $false }
    try {
        $raw = Get-Content -Raw -LiteralPath $modelsPath | ConvertFrom-Json
    } catch {
        return $false
    }
    $providers = $raw.providers
    if (-not $providers) { return $false }
    $providerProp = $providers.PSObject.Properties[$Provider]
    if (-not $providerProp) { return $false }
    $providerBlock = $providerProp.Value
    if (-not $providerBlock.models) { return $true }
    foreach ($entry in @($providerBlock.models)) {
        if ($entry.id -eq $Model) { return $true }
    }
    return $false
}

function Test-LaunchableConfigModel {
    param(
        [string] $Provider,
        [string] $Model
    )
    if (-not $Provider -or -not $Model) { return $false }
    if (Test-BuiltInProvider -Provider $Provider) { return $true }
    return Test-RegisteredPiModel -Provider $Provider -Model $Model
}

# Priority: explicit CLI flags > JARVIS_DEFAULT_* env > launchable
# data/config.yaml roles.chat. Do not infer a provider from API-key env vars on
# first run: without a Pi-registered model, custom providers fail before the TUI
# can open /login or /model-setting.
if (-not $HasProviderArg -and -not $HasModelArg -and (-not $DefaultProvider -or -not $DefaultModel)) {
    $cfgRoles = Get-RolesChatFromConfig -ConfigPath $ConfigPath
    if ($cfgRoles -and (Test-LaunchableConfigModel -Provider $cfgRoles.provider -Model $cfgRoles.model)) {
        if (-not $DefaultProvider) { $DefaultProvider = $cfgRoles.provider }
        if (-not $DefaultModel) { $DefaultModel = $cfgRoles.model }
    }
}
if (-not $HasProviderArg -and -not $HasModelArg) {
    if ($DefaultProvider -and $DefaultModel) {
        $DefaultProviderArgs = @("--provider", $DefaultProvider, "--model", $DefaultModel)
    }
}
$NormalizedRemainingArgs = @()
$RecentTurnsOverride = $null
$ShowSidecarWindow = $false
for ($i = 0; $i -lt $RemainingArgs.Count; $i++) {
    $arg = $RemainingArgs[$i]
    if ($arg -eq "--auto-prompts" -and ($i + 1) -lt $RemainingArgs.Count) {
        $next = $RemainingArgs[$i + 1]
        if ($next -and -not [System.IO.Path]::IsPathRooted($next)) {
            $next = [System.IO.Path]::GetFullPath((Join-Path $Root $next))
        }
        $NormalizedRemainingArgs += @($arg, $next)
        $i++
        continue
    }
    if ($arg -like "--auto-prompts=*") {
        $value = $arg.Substring("--auto-prompts=".Length)
        if ($value -and -not [System.IO.Path]::IsPathRooted($value)) {
            $value = [System.IO.Path]::GetFullPath((Join-Path $Root $value))
        }
        $NormalizedRemainingArgs += "--auto-prompts=$value"
        continue
    }
    if ($arg -eq "--recent-turns" -and ($i + 1) -lt $RemainingArgs.Count) {
        $RecentTurnsOverride = $RemainingArgs[$i + 1]
        $i++
        continue
    }
    if ($arg -like "--recent-turns=*") {
        $RecentTurnsOverride = $arg.Substring("--recent-turns=".Length)
        continue
    }
    if ($arg -eq "--sidecar-window") {
        $ShowSidecarWindow = $true
        continue
    }
    $NormalizedRemainingArgs += $arg
}
if ($null -ne $RecentTurnsOverride) {
    $parsed = 0
    if (-not [int]::TryParse($RecentTurnsOverride, [ref]$parsed) -or $parsed -lt 0) {
        throw "--recent-turns expects a non-negative integer (got '$RecentTurnsOverride')"
    }
    $env:JARVIS_RECENT_TURNS = "$parsed"
} else {
    # Normal interactive default: include only the immediately previous raw
    # turn so short replies can resolve against the previous assistant question.
    # Broader prior context comes from automatic raw recall.
    # Bench/JHB-only runs should pass --recent-turns 0 explicitly.
    $env:JARVIS_RECENT_TURNS = "1"
}
$Port = if ($env:JARVIS_SIDECAR_PORT) { $env:JARVIS_SIDECAR_PORT } else { "8765" }
$env:JARVIS_SIDECAR_URL = "http://127.0.0.1:$Port"
$env:JARVIS_SIDECAR_RUNTIME = $SidecarRuntimePath
$env:JARVIS_WRAPPER_LOG = $WrapperLogPath
$env:JARVIS_DISABLE_COMPACTION = "1"
$env:JARVIS_DISABLE_AUTO_COMPACTION = "1"
$env:JARVIS_RUNTIME_HISTORY_TURNS = "100"
$env:JARVIS_SUBTURN_COMPACT = if ($env:JARVIS_SUBTURN_COMPACT) { $env:JARVIS_SUBTURN_COMPACT } else { "0" }
$env:JARVIS_DISABLE_PI_AGENT_UPDATE = "1"
$env:PI_SKIP_VERSION_CHECK = "1"
$env:PI_SKIP_PACKAGE_UPDATE_CHECK = "1"
# The sidecar owns the default for JARVIS_WORKSPACE / JARVIS_RAW_STORE /
# JARVIS_CODE_CONFIG (user-home under ~/.jarvis-code by default). External
# env overrides still win because the sidecar checks them before its default.
if (-not $env:JARVIS_CODE_CODING_AGENT_DIR) {
    $env:JARVIS_CODE_CODING_AGENT_DIR = $PiAgentDir
}
if (-not $env:PI_CODING_AGENT_DIR) {
    $env:PI_CODING_AGENT_DIR = $env:JARVIS_CODE_CODING_AGENT_DIR
}

function Test-Sidecar {
    try {
        $response = Invoke-RestMethod -Uri "$env:JARVIS_SIDECAR_URL/health" -TimeoutSec 2
        return $response.ok -eq $true -and $response.service -eq "jarvis-jlc-sidecar"
    } catch {
        return $false
    }
}

function Test-PortListening {
    param([int] $Port)

    $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    return $connections.Count -gt 0
}

function Set-SidecarPort {
    param([int] $Port)

    $script:Port = "$Port"
    $env:JARVIS_SIDECAR_PORT = "$Port"
    $env:JARVIS_SIDECAR_URL = "http://127.0.0.1:$Port"
}

function Write-SidecarRuntime {
    param([int] $ProcessId)

    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    }
    [ordered]@{
        url = $env:JARVIS_SIDECAR_URL
        port = [int]$Port
        pid = $ProcessId
        started_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $SidecarRuntimePath -Encoding UTF8
}

function Start-SidecarWatchdog {
    param(
        [string] $PythonExe,
        [string] $Port,
        [string] $WindowStyle
    )

    if (-not (Test-Path $DataDir)) {
        New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
    }
    Set-Content -LiteralPath $SidecarWatchdogPath -Value "$PID" -Encoding UTF8

    $watchdogScript = {
        param(
            [string] $PythonExe,
            [string] $SidecarRoot,
            [string] $SidecarVenvDir,
            [string] $SidecarVenvSitePackages,
            [string] $DataDir,
            [string] $RuntimePath,
            [string] $LogPath,
            [string] $SentinelPath,
            [string] $Port,
            [string] $WindowStyle,
            [string] $ConfigPath,
            [string] $PiAgentDir
        )

        function Write-WatchdogLog {
            param([string] $Message)
            try {
                $timestamp = (Get-Date).ToUniversalTime().ToString("o")
                Add-Content -LiteralPath $LogPath -Value "[$timestamp] sidecar-watchdog $Message" -Encoding UTF8
            } catch {
            }
        }

        function Test-WatchdogSidecar {
            try {
                $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
                return $response.ok -eq $true -and $response.service -eq "jarvis-jlc-sidecar"
            } catch {
                return $false
            }
        }

        function Test-PortOwnedByNonJarvis {
            $connections = @(Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue)
            foreach ($conn in $connections) {
                try {
                    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)"
                    if (-not $proc -or $proc.CommandLine -notlike "*jarvis_sidecar*") {
                        return $true
                    }
                } catch {
                    return $true
                }
            }
            return $false
        }

        $env:JARVIS_SIDECAR_PORT = $Port
        $env:JARVIS_SIDECAR_URL = "http://127.0.0.1:$Port"
        $env:JARVIS_SIDECAR_RUNTIME = $RuntimePath
        $env:JARVIS_CODE_CONFIG = $ConfigPath
        $env:JARVIS_CODE_CODING_AGENT_DIR = $PiAgentDir
        $env:PI_CODING_AGENT_DIR = $PiAgentDir
        $pythonPathItems = @($SidecarRoot)
        if (Test-Path $SidecarVenvSitePackages) {
            $pythonPathItems += $SidecarVenvSitePackages
        }
        if ($env:PYTHONPATH) {
            $pythonPathItems += $env:PYTHONPATH
        }
        $env:PYTHONPATH = ($pythonPathItems -join ";")
        if (Test-Path $SidecarVenvDir) {
            $env:VIRTUAL_ENV = $SidecarVenvDir
            $env:PATH = "$(Join-Path $SidecarVenvDir 'Scripts');$env:PATH"
        }

        Write-WatchdogLog "start port=$Port"
        $attempt = 0
        while (Test-Path $SentinelPath) {
            if (Test-WatchdogSidecar) {
                Start-Sleep -Seconds 2
                continue
            }
            if (Test-PortOwnedByNonJarvis) {
                Write-WatchdogLog "port $Port is occupied by a non-JARVIS process; cannot restart"
                Start-Sleep -Seconds 2
                continue
            }

            $attempt++
            try {
                Write-WatchdogLog "restart attempt=$attempt windowStyle=$WindowStyle"
                $proc = Start-Process -FilePath $PythonExe `
                    -ArgumentList @("-m", "jarvis_sidecar") `
                    -WorkingDirectory $SidecarRoot `
                    -WindowStyle $WindowStyle `
                    -PassThru
                [ordered]@{
                    url = $env:JARVIS_SIDECAR_URL
                    port = [int]$Port
                    pid = $proc.Id
                    started_at = (Get-Date).ToUniversalTime().ToString("o")
                    restarted_by = "wrapper-watchdog"
                } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $RuntimePath -Encoding UTF8
            } catch {
                Write-WatchdogLog "restart failed error=$($_.Exception.Message)"
            }
            Start-Sleep -Seconds 2
        }
        Write-WatchdogLog "stop"
    }

    return Start-Job -ScriptBlock $watchdogScript -ArgumentList @(
        $PythonExe,
        $SidecarRoot,
        $SidecarVenvDir,
        $SidecarVenvSitePackages,
        $DataDir,
        $SidecarRuntimePath,
        $WrapperLogPath,
        $SidecarWatchdogPath,
        $Port,
        $WindowStyle,
        $env:JARVIS_CODE_CONFIG,
        $PiAgentDir
    )
}

function Clear-SidecarRuntime {
    if (Test-Path $SidecarRuntimePath) {
        Remove-Item -LiteralPath $SidecarRuntimePath -Force -ErrorAction SilentlyContinue
    }
}

function Write-WrapperLog {
    param([string] $Message)

    try {
        if (-not (Test-Path $DataDir)) {
            New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
        }
        $timestamp = (Get-Date).ToUniversalTime().ToString("o")
        Add-Content -LiteralPath $WrapperLogPath -Value "[$timestamp] $Message" -Encoding UTF8
    } catch {
        # Best-effort diagnostic logging only.
    }
}

function Select-SidecarPort {
    param([int] $PreferredPort)

    if (Test-Sidecar) { return $PreferredPort }
    if (-not (Test-PortListening -Port $PreferredPort)) { return $PreferredPort }

    for ($candidate = $PreferredPort + 1; $candidate -le ($PreferredPort + 20); $candidate++) {
        if (-not (Test-PortListening -Port $candidate)) {
            Write-Warning "Port $PreferredPort is already in use by a non-JARVIS process; using sidecar port $candidate instead."
            return $candidate
        }
    }

    throw "No free JARVIS sidecar port found in range $PreferredPort-$($PreferredPort + 20)."
}

function Stop-ExistingSidecarForVisibleWindow {
    param([string] $Port)

    $connections = @(Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue)
    if ($connections.Count -eq 0) { return }

    $candidateIds = New-Object System.Collections.Generic.HashSet[int]
    foreach ($conn in $connections) {
        [void]$candidateIds.Add([int]$conn.OwningProcess)
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)"
            if ($proc -and $proc.ParentProcessId) {
                [void]$candidateIds.Add([int]$proc.ParentProcessId)
            }
        } catch {
        }
    }

    $sidecarIds = @()
    foreach ($processId in $candidateIds) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId"
            if ($proc -and $proc.CommandLine -like "*jarvis_sidecar*") {
                $sidecarIds += [int]$processId
            }
        } catch {
        }
    }
    if ($sidecarIds.Count -eq 0) {
        Write-Warning "--sidecar-window requested, but port $Port is owned by a non-JARVIS process; leaving it alone."
        return
    }

    Write-Host "[jarvis] --sidecar-window requested; restarting existing hidden sidecar on port $Port..."
    foreach ($processId in ($sidecarIds | Sort-Object -Descending -Unique)) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
            Write-Warning "Failed to stop sidecar process $processId`: $($_.Exception.Message)"
        }
    }

    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 250
        if (-not (Test-Sidecar)) { return }
    }
}

function Get-HostPython {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Initialize-SidecarVenv {
    # Bootstraps sidecar/.venv on first run so users never lean on a polluted
    # global Python (the SWE-bench / chatterbox-tts / numpy ABI minefield).
    # Returns the venv python path on success, $null on failure (caller falls
    # back to PATH python and emits a warning).
    param(
        [string] $HostPython
    )

    if (-not $HostPython) {
        Write-Warning "JARVIS sidecar venv missing and 'python' is not on PATH; cannot bootstrap."
        return $null
    }

    Write-Host "[jarvis] First-time setup: creating sidecar venv at $SidecarVenvDir"
    & $HostPython -m venv $SidecarVenvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $SidecarVenvPython)) {
        Write-Warning "Failed to create sidecar venv (python -m venv exit=$LASTEXITCODE)."
        return $null
    }

    Write-Host "[jarvis] Installing sidecar requirements (this can take a minute)..."
    & $SidecarVenvPython -m pip install --disable-pip-version-check --quiet --upgrade pip "setuptools<82" wheel
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "pip upgrade in sidecar venv failed (exit=$LASTEXITCODE); requirements install will still be attempted."
    }
    & $SidecarVenvPython -m pip install --disable-pip-version-check -r $SidecarRequirements
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed to install sidecar requirements into venv (exit=$LASTEXITCODE). Delete '$SidecarVenvDir' and re-run to retry."
        return $null
    }
    Write-Host "[jarvis] Sidecar venv ready."
    return $SidecarVenvPython
}

function Resolve-SidecarPython {
    if (Test-Path $SidecarVenvPython) {
        return $SidecarVenvPython
    }
    $hostPython = Get-HostPython
    $bootstrapped = Initialize-SidecarVenv -HostPython $hostPython
    if ($bootstrapped) {
        return $bootstrapped
    }
    return $hostPython
}

function Get-CurrentProcessLineageIds {
    $ids = New-Object System.Collections.Generic.HashSet[int]
    $processId = [int]$PID
    while ($processId -gt 0) {
        if (-not $ids.Add($processId)) { break }
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction Stop
        } catch {
            break
        }
        if (-not $proc.ParentProcessId -or $proc.ParentProcessId -eq $processId) { break }
        $processId = [int]$proc.ParentProcessId
    }
    return $ids
}

function Test-JarvisCodeProcess {
    param(
        $Process,
        [bool] $IncludePi,
        [bool] $IncludeSidecar
    )

    if (-not $Process.CommandLine) { return $false }
    $cmd = $Process.CommandLine
    $normalizedCmd = $cmd.Replace("\", "/")
    $normalizedRoot = $Root.Replace("\", "/")
    $normalizedPiRoot = $PiRoot.Replace("\", "/")
    $normalizedJarvisScript = (Join-Path $Root "jarvis.ps1").Replace("\", "/")
    $normalizedFaceExtension = $FaceExtensionPath.Replace("\", "/")
    $normalizedJlcExtension = $ExtensionPath.Replace("\", "/")

    if ($IncludeSidecar -and $normalizedCmd.IndexOf("jarvis_sidecar", [StringComparison]::OrdinalIgnoreCase) -ge 0) {
        return $true
    }

    if (-not $IncludePi) { return $false }
    if ($normalizedCmd.IndexOf($normalizedRoot, [StringComparison]::OrdinalIgnoreCase) -lt 0) { return $false }

    if ($normalizedCmd.IndexOf($normalizedJarvisScript, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
        return $true
    }
    if ($normalizedCmd.IndexOf($normalizedPiRoot, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        return $false
    }

    return (
        $normalizedCmd.IndexOf("pi-test.ps1", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        $normalizedCmd.IndexOf("packages/coding-agent/src/cli.ts", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        $normalizedCmd.IndexOf("packages/coding-agent/dist/cli.js", [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        $normalizedCmd.IndexOf($normalizedJlcExtension, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
        $normalizedCmd.IndexOf($normalizedFaceExtension, [StringComparison]::OrdinalIgnoreCase) -ge 0
    )
}

function Stop-JarvisCodeProcesses {
    param(
        [switch] $IncludePi,
        [switch] $IncludeSidecar,
        [string] $Reason = "cleanup"
    )

    $protectedIds = Get-CurrentProcessLineageIds
    $candidateIds = New-Object System.Collections.Generic.HashSet[int]
    $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    foreach ($proc in $allProcesses) {
        $processId = [int]$proc.ProcessId
        if ($protectedIds.Contains($processId)) { continue }
        if (Test-JarvisCodeProcess -Process $proc -IncludePi $IncludePi.IsPresent -IncludeSidecar $IncludeSidecar.IsPresent) {
            [void]$candidateIds.Add($processId)
        }
    }
    if ($candidateIds.Count -eq 0) { return }

    $rootIds = @()
    foreach ($processId in $candidateIds) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction Stop
        } catch {
            continue
        }
        if (-not $candidateIds.Contains([int]$proc.ParentProcessId)) {
            $rootIds += $processId
        }
    }

    foreach ($processId in ($rootIds | Sort-Object -Descending -Unique)) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction Stop
            $cmd = if ($proc.CommandLine) { $proc.CommandLine } else { $proc.Name }
            $shortCmd = $cmd.Substring(0, [Math]::Min(100, $cmd.Length))
            $message = "stopping existing JARVIS Code process tree pid=$processId reason=$Reason cmd=$shortCmd..."
            if ($Reason -eq "shutdown") {
                Write-WrapperLog $message
            } else {
                Write-Host "[jarvis] $message"
            }
            & taskkill.exe /T /F /PID $processId 2>&1 | Out-Null
        } catch {
            Write-Warning "Failed to stop JARVIS Code process $processId`: $($_.Exception.Message)"
        }
    }
    # Brief drain so the next health check or port bind does not race a dying socket.
    Start-Sleep -Milliseconds 700
}

function Stop-AllJarvisSidecars {
    param([string] $Reason = "sidecar cleanup")

    Stop-JarvisCodeProcesses -IncludeSidecar -Reason $Reason
}

$StartedSidecarProcess = $null
$SidecarWatchdogJob = $null
$KeepStartedSidecar = $false
$PushedPiLocation = $false
if ((-not $SkipSidecar) -and $env:JARVIS_AUTH_PREFLIGHT -ne "0") {
    Invoke-JarvisAuth -AuthArgs @("preflight")
    $authExitCode = $script:JarvisAuthLastExitCode
    if ($authExitCode -ne 0) {
        exit $authExitCode
    }
}
try {
    Write-WrapperLog "wrapper start pid=$PID args=$($RemainingArgs -join ' ')"
    if (-not $SkipSidecar) {
        Clear-SidecarRuntime
        Stop-JarvisCodeProcesses -IncludePi -IncludeSidecar -Reason "startup singleton"
        $selectedPort = Select-SidecarPort -PreferredPort ([int]$Port)
        Set-SidecarPort -Port $selectedPort
        $pythonExe = Resolve-SidecarPython
        if (-not $pythonExe) {
            throw "python was not found on PATH and sidecar venv could not be created; cannot start JARVIS sidecar"
        }

        $previousPythonPath = $env:PYTHONPATH
        $previousVirtualEnv = $env:VIRTUAL_ENV
        $previousPath = $env:PATH
        try {
            $pythonPathItems = @($SidecarRoot)
            if (Test-Path $SidecarVenvSitePackages) {
                $pythonPathItems += $SidecarVenvSitePackages
            }
            if ($previousPythonPath) {
                $pythonPathItems += $previousPythonPath
            }
            $env:PYTHONPATH = ($pythonPathItems -join ";")
            if (Test-Path $SidecarVenvDir) {
                $env:VIRTUAL_ENV = $SidecarVenvDir
                $env:PATH = "$(Join-Path $SidecarVenvDir 'Scripts');$previousPath"
            }
            $windowStyle = if ($ShowSidecarWindow) { "Normal" } else { "Hidden" }
            $StartedSidecarProcess = Start-Process -FilePath $pythonExe `
                -ArgumentList @("-m", "jarvis_sidecar") `
                -WorkingDirectory $SidecarRoot `
                -WindowStyle $windowStyle `
                -PassThru
            Write-WrapperLog "sidecar started pid=$($StartedSidecarProcess.Id) url=$env:JARVIS_SIDECAR_URL python=$pythonExe"
        } finally {
            $env:PYTHONPATH = $previousPythonPath
            $env:VIRTUAL_ENV = $previousVirtualEnv
            $env:PATH = $previousPath
        }

        $deadline = (Get-Date).AddSeconds(20)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Milliseconds 500
            if (Test-Sidecar) { break }
        }
        if (Test-Sidecar) {
            Write-SidecarRuntime -ProcessId $StartedSidecarProcess.Id
            $SidecarWatchdogJob = Start-SidecarWatchdog -PythonExe $pythonExe -Port $Port -WindowStyle $windowStyle
        }
    }

    if ((-not $SkipSidecar) -and -not (Test-Sidecar)) {
        Write-Warning "JARVIS sidecar did not become healthy; Pi will continue with degraded memory."
    }

    Push-Location $PiRoot
    $PushedPiLocation = $true
    $forwardArgs = @()
    if (-not $EnableExtensionDiscovery) {
        $forwardArgs += "--no-extensions"
    }
    $forwardArgs += @("--extension", $ExtensionPath)
    $forwardArgs += @("--extension", $FaceExtensionPath)
    $forwardArgs += @("--extension", $ImageExtensionPath)
    $forwardArgs += $DefaultProviderArgs
    $forwardArgs += $NormalizedRemainingArgs
    if ($DryRun) {
        [ordered]@{
            pi_root = $PiRoot
            sidecar_url = $env:JARVIS_SIDECAR_URL
            config_path = $env:JARVIS_CODE_CONFIG
            pi_agent_dir = $env:JARVIS_CODE_CODING_AGENT_DIR
            skip_sidecar = $SkipSidecar
            enable_extension_discovery = $EnableExtensionDiscovery
            extension_path = $ExtensionPath
            face_extension_path = $FaceExtensionPath
            image_extension_path = $ImageExtensionPath
            default_provider_args = $DefaultProviderArgs
            forward_args = $forwardArgs
        } | ConvertTo-Json -Depth 4
    } else {
        if (-not (Test-Path -LiteralPath $PiRunnerPath)) {
            throw "Missing JARVIS engine launcher at $PiRunnerPath. Re-run the JARVIS Code installer to repair the installation."
        }
        & $PiRunnerPath @forwardArgs
        Write-WrapperLog "pi-test returned exitCode=$LASTEXITCODE"
    }
} finally {
    if ($PushedPiLocation) {
        Pop-Location
    }
    if ($SidecarWatchdogJob) {
        Remove-Item -LiteralPath $SidecarWatchdogPath -Force -ErrorAction SilentlyContinue
        try {
            Wait-Job -Job $SidecarWatchdogJob -Timeout 5 | Out-Null
            Receive-Job -Job $SidecarWatchdogJob -ErrorAction SilentlyContinue | Out-Null
        } catch {
        }
        Remove-Job -Job $SidecarWatchdogJob -Force -ErrorAction SilentlyContinue
    }
    if ((-not $DryRun) -and (-not $SkipSidecar) -and (-not $KeepStartedSidecar)) {
        try {
            # taskkill /T tree-kills leftover sidecar/Pi workers so the next
            # JARVIS window starts as the only live conversation.
            Stop-JarvisCodeProcesses -IncludePi -IncludeSidecar -Reason "shutdown"
            Write-WrapperLog "shutdown cleanup completed"
        } catch {
            $sidecarPid = if ($StartedSidecarProcess) { $StartedSidecarProcess.Id } else { "none" }
            Write-WrapperLog "shutdown cleanup failed sidecarPid=$sidecarPid error=$($_.Exception.Message)"
            # Best-effort cleanup only. Do not mask the Pi exit reason.
        }
        Clear-SidecarRuntime
    }
    Write-WrapperLog "wrapper end"
}
