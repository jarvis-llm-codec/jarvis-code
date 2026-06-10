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
$WrapperLogPath = Join-Path $DataDir "jarvis-wrapper.log"

function Get-JarvisPairRuntimePrefix {
    param([string] $PairId)
    $prefix = ($PairId -replace '[^A-Za-z0-9]', '')
    if ($prefix.Length -lt 8) {
        return ("{0:x8}" -f [Math]::Abs($PID))
    }
    return $prefix.Substring(0, 8)
}

function Test-JarvisProcessAlive {
    param([int] $ProcessId)
    if ($ProcessId -le 0) { return $false }
    try {
        $proc = Get-Process -Id $ProcessId -ErrorAction Stop
        return $null -ne $proc
    } catch {
        return $false
    }
}

function Write-EarlyWrapperLog {
    param([string] $Message)
    try {
        if (-not (Test-Path $DataDir)) {
            New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
        }
        $timestamp = (Get-Date).ToUniversalTime().ToString("o")
        Add-Content -LiteralPath $WrapperLogPath -Value "[$timestamp] $Message" -Encoding UTF8
    } catch {
    }
}

function Format-JarvisExitCode {
    param([int] $ExitCode)
    $unsigned = [BitConverter]::ToUInt32([BitConverter]::GetBytes([int]$ExitCode), 0)
    return "$ExitCode (0x$($unsigned.ToString('X8')))"
}

$script:JarvisProcessJobHandle = [IntPtr]::Zero

function Initialize-JarvisProcessJobObject {
    if ($env:JARVIS_ENABLE_PROCESS_JOB -ne "1") {
        Write-EarlyWrapperLog "process job disabled by default (set JARVIS_ENABLE_PROCESS_JOB=1 to enable)"
        return
    }
    if ($env:JARVIS_DISABLE_PROCESS_JOB -eq "1") { return }
    if ($env:JARVIS_IN_PROCESS_JOB -eq "1") {
        Write-EarlyWrapperLog "process job skipped (inherited JARVIS_IN_PROCESS_JOB=1)"
        return
    }
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { return }

    try {
        if (-not ("JarvisProcessJobObjectV2.Native" -as [type])) {
            Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace JarvisProcessJobObjectV2 {
    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION {
        public Int64 PerProcessUserTimeLimit;
        public Int64 PerJobUserTimeLimit;
        public UInt32 LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public UInt32 ActiveProcessLimit;
        public Int64 Affinity;
        public UInt32 PriorityClass;
        public UInt32 SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS {
        public UInt64 ReadOperationCount;
        public UInt64 WriteOperationCount;
        public UInt64 OtherOperationCount;
        public UInt64 ReadTransferCount;
        public UInt64 WriteTransferCount;
        public UInt64 OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    public static class Native {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool SetInformationJobObject(
            IntPtr hJob,
            int jobObjectInfoClass,
            ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION lpJobObjectInfo,
            int cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool IsProcessInJob(IntPtr ProcessHandle, IntPtr JobHandle, out bool Result);

        [DllImport("kernel32.dll", SetLastError = true)]
        public static extern bool CloseHandle(IntPtr hObject);
    }
}
"@
        }

        $currentProcess = [Diagnostics.Process]::GetCurrentProcess()
        $inExistingJob = $false
        $probeOk = [JarvisProcessJobObjectV2.Native]::IsProcessInJob(
            $currentProcess.Handle,
            [IntPtr]::Zero,
            [ref]$inExistingJob
        )
        if ($probeOk -and $inExistingJob) {
            $env:JARVIS_IN_PROCESS_JOB = "1"
            Write-EarlyWrapperLog "process job skipped (current process already belongs to a job)"
            return
        }
        if (-not $probeOk) {
            $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            Write-EarlyWrapperLog "process job membership probe failed win32=$err; continuing with local job attempt"
        }

        $jobHandle = [JarvisProcessJobObjectV2.Native]::CreateJobObject([IntPtr]::Zero, $null)
        if ($jobHandle -eq [IntPtr]::Zero) {
            Write-EarlyWrapperLog "process job create failed"
            return
        }

        $info = New-Object JarvisProcessJobObjectV2.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        # KILL_ON_JOB_CLOSE cleans the wrapper itself on terminal close.
        # BREAKAWAY_OK/SILENT_BREAKAWAY_OK let Pi shell tools and spawned
        # JARVIS windows use their own process supervision without colliding
        # with this outer job object.
        $info.BasicLimitInformation.LimitFlags = 0x2000 -bor 0x0800 -bor 0x1000
        $infoLength = [Runtime.InteropServices.Marshal]::SizeOf([type][JarvisProcessJobObjectV2.JOBOBJECT_EXTENDED_LIMIT_INFORMATION])
        $setOk = [JarvisProcessJobObjectV2.Native]::SetInformationJobObject($jobHandle, 9, [ref]$info, $infoLength)
        if (-not $setOk) {
            $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            [void][JarvisProcessJobObjectV2.Native]::CloseHandle($jobHandle)
            Write-EarlyWrapperLog "process job set limits failed win32=$err"
            return
        }

        $assignOk = [JarvisProcessJobObjectV2.Native]::AssignProcessToJobObject($jobHandle, $currentProcess.Handle)
        if (-not $assignOk) {
            $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            [void][JarvisProcessJobObjectV2.Native]::CloseHandle($jobHandle)
            Write-EarlyWrapperLog "process job assign failed win32=$err"
            return
        }

        $script:JarvisProcessJobHandle = $jobHandle
        $env:JARVIS_IN_PROCESS_JOB = "1"
        Write-EarlyWrapperLog "process job initialized kill_on_close=1 breakaway_ok=1"
    } catch {
        Write-EarlyWrapperLog "process job initialization failed error=$($_.Exception.Message)"
    }
}

Initialize-JarvisProcessJobObject

function Normalize-JarvisWindowLabel {
    param([string] $Label)
    $value = ([string]$Label) -replace '[\x00-\x1F\x7F]', ''
    $value = $value.Trim()
    if (-not $value) { return $null }
    if ($value.Length -gt 32) {
        $value = $value.Substring(0, 32)
    }
    return $value
}

function Read-JarvisRuntimeLabel {
    param([string] $RuntimePath)
    if (-not $RuntimePath -or -not (Test-Path $RuntimePath)) { return $null }
    try {
        $runtime = Get-Content -LiteralPath $RuntimePath -Raw | ConvertFrom-Json
        return Normalize-JarvisWindowLabel -Label ([string]$runtime.label)
    } catch {
        return $null
    }
}

function Initialize-JarvisPairId {
    if (-not $env:JARVIS_PAIR_ID) {
        $env:JARVIS_PAIR_ID = [guid]::NewGuid().ToString()
        return
    }

    # Watchdog restarts the Python sidecar directly with the same pair. This
    # guard runs only when the wrapper itself is launched and prevents a child
    # terminal from inheriting another live window's pair/runtime binding.
    $inheritedPair = [string]$env:JARVIS_PAIR_ID
    $runtimePrefix = Get-JarvisPairRuntimePrefix -PairId $inheritedPair
    $runtimePath = Join-Path $DataDir "sidecar-runtime-$runtimePrefix.json"
    if (-not (Test-Path $runtimePath)) { return }

    try {
        $runtime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
        $runtimePair = [string]$runtime.pair_id
        $runtimePid = [int]$runtime.pid
        if ($runtimePair -eq $inheritedPair -and $runtimePid -gt 0 -and $runtimePid -ne [int]$PID -and (Test-JarvisProcessAlive -ProcessId $runtimePid)) {
            $env:JARVIS_PAIR_ID = [guid]::NewGuid().ToString()
            Write-EarlyWrapperLog "inherited pair $runtimePrefix owned by live pid $runtimePid; regenerated"
        }
    } catch {
        Write-EarlyWrapperLog "failed to inspect inherited pair runtime path=$runtimePath error=$($_.Exception.Message)"
    }
}

Initialize-JarvisPairId
if (-not $env:JARVIS_PAIR_ID) {
    $env:JARVIS_PAIR_ID = [guid]::NewGuid().ToString()
}
$PairId = $env:JARVIS_PAIR_ID
$PairRuntimePrefix = Get-JarvisPairRuntimePrefix -PairId $PairId
$SidecarRuntimePath = Join-Path $DataDir "sidecar-runtime-$PairRuntimePrefix.json"
$SidecarWatchdogPath = Join-Path $DataDir "sidecar-watchdog-$PID.run"
$WindowLabel = Normalize-JarvisWindowLabel -Label $env:JARVIS_WINDOW_LABEL
if (-not $WindowLabel) {
    $WindowLabel = Read-JarvisRuntimeLabel -RuntimePath $SidecarRuntimePath
}
$AuthScript = Join-Path $Root "scripts\jarvis-auth.py"
$DryRun = $env:JARVIS_WRAPPER_DRY_RUN -eq "1"
$SkipSidecar = $DryRun -or ($RemainingArgs -contains "--help") -or ($RemainingArgs -contains "-h") -or ($RemainingArgs -contains "--version") -or ($RemainingArgs -contains "-v")
$EnableExtensionDiscovery = $env:JARVIS_ENABLE_EXTENSION_DISCOVERY -eq "1"
$HasProviderArg = ($RemainingArgs -contains "--provider") -or (@($RemainingArgs | Where-Object { $_ -like "--provider=*" }).Count -gt 0)
$HasModelArg = ($RemainingArgs -contains "--model") -or (@($RemainingArgs | Where-Object { $_ -like "--model=*" }).Count -gt 0)
# Extract the explicit CLI chat provider/model values (used to detect a
# sidecar-routed spawn model below). Spawned workers receive these flags.
$CliProvider = $null
$CliModel = $null
for ($ci = 0; $ci -lt $RemainingArgs.Count; $ci++) {
    if ($RemainingArgs[$ci] -eq "--provider" -and ($ci + 1) -lt $RemainingArgs.Count) { $CliProvider = $RemainingArgs[$ci + 1] }
    elseif ($RemainingArgs[$ci] -eq "--model" -and ($ci + 1) -lt $RemainingArgs.Count) { $CliModel = $RemainingArgs[$ci + 1] }
    elseif ($RemainingArgs[$ci] -like "--provider=*") { $CliProvider = $RemainingArgs[$ci].Substring("--provider=".Length) }
    elseif ($RemainingArgs[$ci] -like "--model=*") { $CliModel = $RemainingArgs[$ci].Substring("--model=".Length) }
}
if (-not $CliProvider -and $CliModel -and $CliModel.Contains("/")) {
    $splitModel = $CliModel.Split("/", 2)
    if ($splitModel.Count -eq 2 -and $splitModel[0] -and $splitModel[1]) {
        $CliProvider = $splitModel[0]
        $CliModel = $splitModel[1]
        $HasProviderArg = $true
    }
}
$StripCliChatModel = $false
$DefaultProviderArgs = @()
$DefaultProvider = $env:JARVIS_DEFAULT_PROVIDER
$DefaultModel = $env:JARVIS_DEFAULT_MODEL
$UserProfileDir = [Environment]::GetFolderPath("UserProfile")
if (-not $UserProfileDir) { $UserProfileDir = $HOME }
$DefaultConfigPath = Join-Path (Join-Path $UserProfileDir ".jarvis-code") "config.yaml"
$ConfigPath = if ($env:JARVIS_CODE_CONFIG) { $env:JARVIS_CODE_CONFIG } else { $DefaultConfigPath }
# The sidecar and in-app /model-setting must resolve the same roles file
# regardless of whether chat provider/model were supplied as CLI or
# environment defaults.
$env:JARVIS_CODE_CONFIG = $ConfigPath
# Keys saved via /api-key land in credentials.yaml (env: section) next to the
# active config. The sidecar loads that file itself, but pi resolves provider
# apiKey entries from this process env only — lift them here so ollama-cloud
# and custom providers authenticate. Vars already set in the console win.
$CredentialsPath = Join-Path (Split-Path $ConfigPath -Parent) "credentials.yaml"
if (Test-Path $CredentialsPath) {
    $inEnvBlock = $false
    foreach ($credLine in [IO.File]::ReadAllLines($CredentialsPath)) {
        if ($credLine -match '^env:\s*$') { $inEnvBlock = $true; continue }
        if (-not $inEnvBlock) { continue }
        if ($credLine -match '^\S') { $inEnvBlock = $false; continue }
        if ($credLine -match '^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(\S.*)$') {
            $credName = $Matches[1]
            $credValue = $Matches[2].Trim().Trim('"').Trim("'")
            if ($credValue -and -not (Get-Item "env:$credName" -ErrorAction SilentlyContinue)) {
                Set-Item "env:$credName" $credValue
            }
        }
    }
}
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
    "claude-login",
    "anthropic-login",
    "claude-auth-status",
    "claude-logout",
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
    # Reads roles.chat from the active config.yaml as `provider/model`.
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

function Get-RolesEncoderFromConfig {
    # Reads roles.encoder from the active config.yaml as `provider/model`. Used as the
    # Pi nominal provider when roles.chat is sidecar-routed (e.g. Agent SDK) and
    # therefore not Pi-launchable. Returns @{ provider; model } or $null.
    param([string] $ConfigPath)
    if (-not $ConfigPath -or -not (Test-Path $ConfigPath)) { return $null }
    try {
        $content = Get-Content -Raw -LiteralPath $ConfigPath
    } catch {
        return $null
    }
    $rolesMatch = [regex]::Match($content, '(?ms)^roles:\s*\r?\n((?:[ \t]+[^\r\n]+\r?\n?)+)')
    if (-not $rolesMatch.Success) { return $null }
    $encMatch = [regex]::Match($rolesMatch.Groups[1].Value, '(?m)^[ \t]+encoder:[ \t]*(\S+)')
    if (-not $encMatch.Success) { return $null }
    $enc = $encMatch.Groups[1].Value.Trim('"', "'")
    if ($enc -notmatch '^([^/\s]+)/(.+)$') { return $null }
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

function Test-SidecarRoutedProvider {
    param([string] $Provider)
    return $Provider -eq "anthropic-agent-sdk"
}

function Resolve-SidecarRoutedChatLaunchArgs {
    param(
        [string] $Provider,
        [string] $Model
    )
    if (-not (Test-SidecarRoutedProvider -Provider $Provider)) {
        throw "Provider $Provider/$Model is not Pi-launchable."
    }
    $env:JARVIS_CHAT_MODEL_OVERRIDE = "$Provider/$Model"
    $encRoles = Get-RolesEncoderFromConfig -ConfigPath $ConfigPath
    if ($encRoles -and (Test-LaunchableConfigModel -Provider $encRoles.provider -Model $encRoles.model)) {
        return @("--provider", $encRoles.provider, "--model", $encRoles.model)
    }
    # Both chat and encoder are sidecar-routed (e.g. anthropic-agent-sdk/*). The
    # sidecar drives all LLM work; pi only needs a window-init shell that resolves
    # without a key. Launch pi on the encoder's sidecar provider (registered in
    # pi-agent/models.json as a keyless openai-completions shim) instead of throwing.
    if ($encRoles -and (Test-SidecarRoutedProvider -Provider $encRoles.provider)) {
        return @("--provider", $encRoles.provider, "--model", $encRoles.model)
    }
    throw "Provider $Provider/$Model is sidecar-routed or not Pi-launchable; roles.encoder must be a Pi-launchable provider/model before opening the window."
}

# Priority: explicit CLI flags > JARVIS_DEFAULT_* env > launchable
# active config.yaml roles.chat. Do not infer a provider from API-key env vars on
# first run: without a Pi-registered model, custom providers fail before the TUI
# can open /login or /model-setting.
# jarvis.ps1 authoritatively owns JARVIS_CHAT_MODEL_OVERRIDE per launch: the
# sidecar-routed (Resolve-SidecarRoutedChatLaunchArgs) and CLI-launchable
# branches below re-set it whenever THIS launch needs it. Clear any value
# inherited from the launching shell up front so a stale Agent SDK override
# (e.g. left over from an Opus regime test) cannot hijack a Pi-native main
# window's chat role — sidecar/footer showing Opus while config.yaml says
# glm-5.2. Worker spawns re-derive it from their own CLI args. (Jun, 2026-06-26)
Remove-Item Env:JARVIS_CHAT_MODEL_OVERRIDE -ErrorAction SilentlyContinue
if (-not $HasProviderArg -and -not $HasModelArg -and (-not $DefaultProvider -or -not $DefaultModel)) {
    $cfgRoles = Get-RolesChatFromConfig -ConfigPath $ConfigPath
    if ($cfgRoles -and (Test-LaunchableConfigModel -Provider $cfgRoles.provider -Model $cfgRoles.model)) {
        if (-not $DefaultProvider) { $DefaultProvider = $cfgRoles.provider }
        if (-not $DefaultModel) { $DefaultModel = $cfgRoles.model }
    } elseif ($cfgRoles -and (Test-SidecarRoutedProvider -Provider $cfgRoles.provider)) {
        # roles.chat isn't Pi-launchable (e.g. the sidecar-routed Agent SDK
        # provider). The JLC sidecar drives chat regardless of Pi's nominal
        # provider, so run Pi on the encoder role's provider, which IS
        # Pi-launchable. If no launchable encoder is configured, fail before
        # opening a window so the requested chat model cannot silently fall back.
        $DefaultProviderArgs = Resolve-SidecarRoutedChatLaunchArgs -Provider $cfgRoles.provider -Model $cfgRoles.model
    }
}
# A spawned worker's sidecar inherits the parent's config roles.chat. When the main
# window is sidecar-routed (e.g. the Agent SDK), that inherited chat provider would
# hijack the worker even though Pi was handed a different --provider/--model — so a
# "gpt-5.5 worker" spawned from an Anthropic main came up as Claude. Pin the worker's
# chat to the requested model via JARVIS_CHAT_MODEL_OVERRIDE so the sidecar reports
# and drives THIS model. This is needed for BOTH a Pi-native model (gpt-5.5 must beat
# the inherited Agent SDK) and a sidecar-routed one. (Jun, 2026-06-16)
if ($HasProviderArg -and $HasModelArg -and $CliProvider -and $CliModel) {
    if (-not (Test-LaunchableConfigModel -Provider $CliProvider -Model $CliModel)) {
        # Sidecar-routed (e.g. Agent SDK) isn't Pi-launchable: run Pi on the encoder
        # provider and strip the CLI flags so Pi never sees the non-launchable model.
        $StripCliChatModel = $true
        $DefaultProviderArgs = Resolve-SidecarRoutedChatLaunchArgs -Provider $CliProvider -Model $CliModel
    } else {
        $env:JARVIS_CHAT_MODEL_OVERRIDE = "$CliProvider/$CliModel"
    }
}
if (-not $HasProviderArg -and -not $HasModelArg) {
    if ($DefaultProvider -and $DefaultModel -and (Test-LaunchableConfigModel -Provider $DefaultProvider -Model $DefaultModel)) {
        $DefaultProviderArgs = @("--provider", $DefaultProvider, "--model", $DefaultModel)
    } elseif ($DefaultProvider -and $DefaultModel) {
        $DefaultProviderArgs = Resolve-SidecarRoutedChatLaunchArgs -Provider $DefaultProvider -Model $DefaultModel
    } elseif ($DefaultProvider -or $DefaultModel) {
        throw "Both JARVIS_DEFAULT_PROVIDER and JARVIS_DEFAULT_MODEL are required when either is set."
    }
}
$NormalizedRemainingArgs = @()
$RecentTurnsOverride = $null
$ShowSidecarWindow = $false
for ($i = 0; $i -lt $RemainingArgs.Count; $i++) {
    $arg = $RemainingArgs[$i]
    if ($StripCliChatModel -and ($arg -eq "--provider" -or $arg -eq "--model") -and ($i + 1) -lt $RemainingArgs.Count) {
        # Sidecar-routed chat model: don't forward the non-launchable provider/model
        # to Pi (the sidecar drives chat via JARVIS_CHAT_MODEL_OVERRIDE).
        $i++
        continue
    }
    if ($arg -eq "--window-label" -and ($i + 1) -lt $RemainingArgs.Count) {
        $WindowLabel = Normalize-JarvisWindowLabel -Label $RemainingArgs[$i + 1]
        $i++
        continue
    }
    if ($arg -like "--window-label=*") {
        $WindowLabel = Normalize-JarvisWindowLabel -Label $arg.Substring("--window-label=".Length)
        continue
    }
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
    if ($arg -eq "--yolo") {
        # Skip all interactive safety confirmations (Claude Code's
        # skip-permissions equivalent). Inherited by spawned workers.
        $env:JARVIS_YOLO = "1"
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
# The sidecar self-exits when this pid dies; X-closing the console window
# never runs the wrapper's finally block, so cleanup must live in the child.
$env:JARVIS_WRAPPER_PID = "$PID"
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
        if (-not ($response.ok -eq $true -and $response.service -eq "jarvis-jlc-sidecar")) {
            return $false
        }
        if ($env:JARVIS_PAIR_ID) {
            return [string]$response.pair_id -eq [string]$env:JARVIS_PAIR_ID
        }
        return $true
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
    $runtimePayload = [ordered]@{
        url = $env:JARVIS_SIDECAR_URL
        port = [int]$Port
        pid = $ProcessId
        pair_id = $env:JARVIS_PAIR_ID
        started_at = (Get-Date).ToUniversalTime().ToString("o")
    }
    if ($WindowLabel) {
        $runtimePayload["label"] = $WindowLabel
    }
    $runtimeJson = ($runtimePayload | ConvertTo-Json -Depth 3) + "`n"
    Write-Utf8NoBomFile -Path $SidecarRuntimePath -Content $runtimeJson
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
            [string] $PiAgentDir,
            [string] $PairId
        )

        function Write-Utf8NoBomFile {
            param(
                [string] $Path,
                [string] $Content
            )
            $encoding = New-Object System.Text.UTF8Encoding($false)
            [System.IO.File]::WriteAllText($Path, $Content, $encoding)
        }

        function Normalize-JarvisWindowLabel {
            param([string] $Label)
            $value = ([string]$Label) -replace '[\x00-\x1F\x7F]', ''
            $value = $value.Trim()
            if (-not $value) { return $null }
            if ($value.Length -gt 32) {
                $value = $value.Substring(0, 32)
            }
            return $value
        }

        function Read-JarvisRuntimeLabel {
            param([string] $RuntimePath)
            if (-not $RuntimePath -or -not (Test-Path $RuntimePath)) { return $null }
            try {
                $runtime = Get-Content -LiteralPath $RuntimePath -Raw | ConvertFrom-Json
                return Normalize-JarvisWindowLabel -Label ([string]$runtime.label)
            } catch {
                return $null
            }
        }

        function Write-WatchdogLog {
            param([string] $Message)
            try {
                $timestamp = (Get-Date).ToUniversalTime().ToString("o")
                Add-Content -LiteralPath $LogPath -Value "[$timestamp] sidecar-watchdog $Message" -Encoding UTF8
            } catch {
            }
        }

        function Test-WatchdogSidecar {
            # The sidecar serves /health on the same async loop that runs bge-m3
            # embedding and encoder turns; under that load a single 2s probe can
            # time out on a perfectly live sidecar. Retry once before declaring it
            # dead so a transient busy spell does not trigger a (futile) respawn.
            foreach ($probe in 1..2) {
                try {
                    $response = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 4
                    if (-not ($response.ok -eq $true -and $response.service -eq "jarvis-jlc-sidecar")) {
                        return $false
                    }
                    if ($PairId) {
                        return [string]$response.pair_id -eq [string]$PairId
                    }
                    return $true
                } catch {
                    if ($probe -ge 2) { return $false }
                    Start-Sleep -Milliseconds 500
                }
            }
            return $false
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

        function Test-PortHeldByJarvisSidecar {
            # True when a jarvis_sidecar process is already LISTENing on the port.
            # Distinct from Test-PortOwnedByNonJarvis (which only flags foreign
            # owners): here we detect that OUR sidecar is present, so a respawn
            # would be futile — the new process hits the port guard in
            # jarvis_sidecar.__main__ and self-exits immediately.
            $connections = @(Get-NetTCPConnection -LocalPort ([int]$Port) -State Listen -ErrorAction SilentlyContinue)
            foreach ($conn in $connections) {
                try {
                    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)"
                    if ($proc -and $proc.CommandLine -like "*jarvis_sidecar*") {
                        return $true
                    }
                } catch {
                }
            }
            return $false
        }

        $env:JARVIS_SIDECAR_PORT = $Port
        $env:JARVIS_SIDECAR_URL = "http://127.0.0.1:$Port"
        $env:JARVIS_SIDECAR_RUNTIME = $RuntimePath
        $env:JARVIS_PAIR_ID = $PairId
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
            if (Test-PortHeldByJarvisSidecar) {
                # Health probe failed but a jarvis_sidecar still owns the port: it
                # is alive-but-busy (blocking on embedding / a turn), not dead.
                # Respawning cannot recover it — the new process self-exits on the
                # port guard, and with --sidecar-window that self-exit flashes a
                # visible console (live incident 2026-06-22). Back off and re-probe;
                # the busy spell clears and the next probe passes. A genuinely
                # wedged-but-listening sidecar needs a kill-first restart, which is
                # deliberately out of scope here (blind respawn never recovered it).
                Write-WatchdogLog "health probe failed but jarvis_sidecar still holds port $Port (busy, not dead); skipping futile respawn"
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
                $runtimeLabel = Read-JarvisRuntimeLabel -RuntimePath $RuntimePath
                $runtimePayload = [ordered]@{
                    url = $env:JARVIS_SIDECAR_URL
                    port = [int]$Port
                    pid = $proc.Id
                    pair_id = $PairId
                    started_at = (Get-Date).ToUniversalTime().ToString("o")
                    restarted_by = "wrapper-watchdog"
                }
                if ($runtimeLabel) {
                    $runtimePayload["label"] = $runtimeLabel
                }
                $runtimeJson = ($runtimePayload | ConvertTo-Json -Depth 3) + "`n"
                Write-Utf8NoBomFile -Path $RuntimePath -Content $runtimeJson
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
        $PiAgentDir,
        $env:JARVIS_PAIR_ID
    )
}

function Clear-SidecarRuntime {
    if (-not (Test-Path $SidecarRuntimePath)) { return }
    try {
        $runtime = Get-Content -LiteralPath $SidecarRuntimePath -Raw | ConvertFrom-Json
        $runtimePid = [int]$runtime.pid
        if ($runtimePid -le 0 -or [string]$runtime.pair_id -ne [string]$env:JARVIS_PAIR_ID) {
            Write-WrapperLog "skip runtime cleanup path=$SidecarRuntimePath reason=pair-or-pid-mismatch"
            return
        }
        if (Test-JarvisProcessAlive -ProcessId $runtimePid) {
            Write-WrapperLog "skip runtime cleanup path=$SidecarRuntimePath reason=pid-alive pid=$runtimePid"
            return
        }
        Remove-Item -LiteralPath $SidecarRuntimePath -Force -ErrorAction Stop
        Write-WrapperLog "runtime cleanup removed path=$SidecarRuntimePath"
    } catch {
        Write-WrapperLog "runtime cleanup failed path=$SidecarRuntimePath error=$($_.Exception.Message)"
    }
}

function Get-JhbStorageRoot {
    $fallback = Join-Path ([Environment]::GetFolderPath("UserProfile")) ".jarvis-code\conversation"
    $python = $SidecarVenvPython
    if (-not (Test-Path $python)) {
        $python = Get-HostPython
    }
    if (-not $python) { return $fallback }
    $oldPythonPath = $env:PYTHONPATH
    try {
        $pythonPathItems = @($SidecarRoot)
        if (Test-Path $SidecarVenvSitePackages) {
            $pythonPathItems += $SidecarVenvSitePackages
        }
        if ($oldPythonPath) {
            $pythonPathItems += $oldPythonPath
        }
        $env:PYTHONPATH = ($pythonPathItems -join ";")
        $code = "from pathlib import Path; from jlc_agentic.config import load_config; print(Path(load_config().jhb.storage_path).expanduser())"
        $value = (& $python -c $code 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
            return [string]$value
        }
    } catch {
    } finally {
        if ($null -eq $oldPythonPath) {
            Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $oldPythonPath
        }
    }
    return $fallback
}

function Clear-CurrentPairJhbRoot {
    $storageRoot = Get-JhbStorageRoot
    if ([string]::IsNullOrWhiteSpace($storageRoot)) { return }
    $windowsRoot = Join-Path $storageRoot "_windows"
    $target = Join-Path $windowsRoot "jhb-$PairRuntimePrefix"
    try {
        if (-not (Test-Path -LiteralPath $target)) { return }
        $resolvedTarget = [System.IO.Path]::GetFullPath($target)
        $resolvedGuard = [System.IO.Path]::GetFullPath($windowsRoot)
        $separatorChars = [char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
        $guardPrefix = $resolvedGuard.TrimEnd($separatorChars) + [System.IO.Path]::DirectorySeparatorChar
        if ($resolvedTarget -eq $resolvedGuard -or -not $resolvedTarget.StartsWith($guardPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            Write-WrapperLog "skip JHB cleanup path=$target reason=outside-guard"
            return
        }
        if ((Split-Path -Leaf $resolvedTarget) -ne "jhb-$PairRuntimePrefix") {
            Write-WrapperLog "skip JHB cleanup path=$target reason=name-mismatch"
            return
        }
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction Stop
        Write-WrapperLog "JHB cleanup removed path=$resolvedTarget"
    } catch {
        Write-WrapperLog "JHB cleanup failed path=$target error=$($_.Exception.Message)"
    }
}

function Clear-CurrentPairWorkerSlot {
    # Wipe-on-close hygiene for a SPAWNED worker only: delete the bounded
    # _windows/worker<N> (or overflow-*) slot this worker owned. That slot now
    # holds the worker's PRIVATE conversation raw-store + retriever store + JHB
    # (full isolation), so removing it leaves no remnant. The sidecar's
    # wipe-on-start is the primary freshness guarantee; this is disk hygiene.
    #
    # SAFETY: gated hard on JARVIS_SPAWNED so the MAIN window (never spawned)
    # can never reach this. The main's durable home is `conversation/` itself,
    # which is NOT under `_windows`, so even the per-candidate guards below
    # (must be under _windows + name worker*/overflow-* + owner.pair8 match)
    # would refuse it. Three independent guards, plus the env gate.
    if ([string]::IsNullOrWhiteSpace($env:JARVIS_SPAWNED) -or $env:JARVIS_SPAWNED -eq "0") { return }
    $storageRoot = Get-JhbStorageRoot
    if ([string]::IsNullOrWhiteSpace($storageRoot)) { return }
    $windowsRoot = Join-Path $storageRoot "_windows"
    if (-not (Test-Path -LiteralPath $windowsRoot)) { return }
    $resolvedGuard = [System.IO.Path]::GetFullPath($windowsRoot)
    $separatorChars = [char[]]@([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    $guardPrefix = $resolvedGuard.TrimEnd($separatorChars) + [System.IO.Path]::DirectorySeparatorChar
    try {
        $slots = Get-ChildItem -LiteralPath $windowsRoot -Directory -ErrorAction Stop
    } catch {
        return
    }
    foreach ($slot in $slots) {
        $name = $slot.Name
        if (-not ($name.StartsWith("worker") -or $name.StartsWith("overflow-"))) { continue }
        $ownerFile = Join-Path $slot.FullName "owner.json"
        if (-not (Test-Path -LiteralPath $ownerFile)) { continue }
        try {
            $owner = Get-Content -LiteralPath $ownerFile -Raw -ErrorAction Stop | ConvertFrom-Json
        } catch {
            continue
        }
        if ($owner.pair8 -ne $PairRuntimePrefix) { continue }
        $resolvedTarget = [System.IO.Path]::GetFullPath($slot.FullName)
        if ($resolvedTarget -eq $resolvedGuard -or -not $resolvedTarget.StartsWith($guardPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            Write-WrapperLog "skip worker slot cleanup path=$($slot.FullName) reason=outside-guard"
            continue
        }
        try {
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction Stop
            Write-WrapperLog "worker slot cleanup removed path=$resolvedTarget pair8=$PairRuntimePrefix"
        } catch {
            Write-WrapperLog "worker slot cleanup failed path=$resolvedTarget error=$($_.Exception.Message)"
        }
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

    if (-not (Test-PortListening -Port $PreferredPort)) { return $PreferredPort }

    for ($candidate = $PreferredPort + 1; $candidate -le ($PreferredPort + 20); $candidate++) {
        if (-not (Test-PortListening -Port $candidate)) {
            Write-Warning "Port $PreferredPort is already in use; using sidecar port $candidate instead."
            return $candidate
        }
    }

    for ($candidate = $PreferredPort + 21; $candidate -le ($PreferredPort + 50); $candidate++) {
        if (-not (Test-PortListening -Port $candidate)) {
            Write-Warning "Ports $PreferredPort-$($PreferredPort + 20) are in use; using sidecar port $candidate instead."
            return $candidate
        }
    }

    throw "No free JARVIS sidecar port found in range $PreferredPort-$($PreferredPort + 50)."
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

function Get-DescendantProcessIds {
    param([int[]] $RootProcessIds)

    $childrenByParent = @{}
    foreach ($proc in @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)) {
        $parentId = [int]$proc.ParentProcessId
        if (-not $childrenByParent.ContainsKey($parentId)) {
            $childrenByParent[$parentId] = @()
        }
        $childrenByParent[$parentId] += [int]$proc.ProcessId
    }

    $result = New-Object System.Collections.Generic.HashSet[int]
    $queue = New-Object System.Collections.Generic.Queue[int]
    foreach ($rootId in $RootProcessIds) {
        $queue.Enqueue([int]$rootId)
    }
    while ($queue.Count -gt 0) {
        $parent = $queue.Dequeue()
        foreach ($child in @($childrenByParent[$parent])) {
            if ($result.Add([int]$child)) {
                $queue.Enqueue([int]$child)
            }
        }
    }
    return $result
}

function Stop-CurrentPairProcesses {
    param($SidecarProcess)

    $sidecarRootIds = New-Object System.Collections.Generic.HashSet[int]
    if ($SidecarProcess) {
        [void]$sidecarRootIds.Add([int]$SidecarProcess.Id)
    }
    if (Test-Path $SidecarRuntimePath) {
        try {
            $runtime = Get-Content -LiteralPath $SidecarRuntimePath -Raw | ConvertFrom-Json
            if ($runtime.pid -and [string]$runtime.pair_id -eq [string]$env:JARVIS_PAIR_ID) {
                [void]$sidecarRootIds.Add([int]$runtime.pid)
            }
        } catch {
            Write-WrapperLog "failed to read sidecar runtime for shutdown cleanup path=$SidecarRuntimePath error=$($_.Exception.Message)"
        }
    }

    foreach ($sidecarRootId in ($sidecarRootIds | Sort-Object -Descending -Unique)) {
        try {
            & taskkill.exe /T /F /PID $sidecarRootId 2>&1 | Out-Null
        } catch {
            Write-WrapperLog "failed to stop sidecar tree pid=$sidecarRootId error=$($_.Exception.Message)"
        }
    }

    $descendants = Get-DescendantProcessIds -RootProcessIds @([int]$PID)
    $candidateIds = New-Object System.Collections.Generic.HashSet[int]
    foreach ($processId in $descendants) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction Stop
            if (Test-JarvisCodeProcess -Process $proc -IncludePi $true -IncludeSidecar $true) {
                [void]$candidateIds.Add([int]$processId)
            }
        } catch {
        }
    }
    foreach ($processId in ($candidateIds | Sort-Object -Descending -Unique)) {
        try {
            & taskkill.exe /T /F /PID $processId 2>&1 | Out-Null
        } catch {
            Write-WrapperLog "failed to stop pi descendant pid=$processId error=$($_.Exception.Message)"
        }
    }
    Start-Sleep -Milliseconds 300
}

if ($RemainingArgs.Count -gt 0 -and $RemainingArgs[0] -eq "kill-all") {
    Stop-JarvisCodeProcesses -IncludePi -IncludeSidecar -Reason "manual kill-all"
    exit 0
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
        $portAllocationMutex = New-Object System.Threading.Mutex($false, "JARVIS_CODE_SIDECAR_PORT_ALLOC")
        $portAllocationLockHeld = $false
        try {
            $portAllocationLockHeld = $portAllocationMutex.WaitOne([TimeSpan]::FromSeconds(45))
            if (-not $portAllocationLockHeld) {
                throw "timed out waiting for sidecar port allocation lock"
            }
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
        } finally {
            if ($portAllocationLockHeld) {
                $portAllocationMutex.ReleaseMutex()
            }
            $portAllocationMutex.Dispose()
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
            pair_id = $env:JARVIS_PAIR_ID
            window_label = $WindowLabel
            sidecar_runtime = $env:JARVIS_SIDECAR_RUNTIME
            config_path = $env:JARVIS_CODE_CONFIG
            pi_agent_dir = $env:JARVIS_CODE_CODING_AGENT_DIR
            skip_sidecar = $SkipSidecar
            chat_model_override = $env:JARVIS_CHAT_MODEL_OVERRIDE
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
        $piExitCode = [int]$LASTEXITCODE
        $formattedExitCode = Format-JarvisExitCode -ExitCode $piExitCode
        Write-WrapperLog "pi-test returned exitCode=$formattedExitCode"
        if ($piExitCode -ne 0) {
            Write-Host "[jarvis-error] Pi exited with code $formattedExitCode" -ForegroundColor Red
            Write-Host "[jarvis-error] Wrapper log: $WrapperLogPath" -ForegroundColor DarkGray
        }
    }
} finally {
    if ($PushedPiLocation) {
        Pop-Location
    }
    # Watchdog teardown must come first: it restarts a dead sidecar every 2s
    # while the sentinel exists, so killing the sidecar before removing the
    # sentinel races a zombie respawn during the descendant scan.
    if ($SidecarWatchdogJob) {
        Remove-Item -LiteralPath $SidecarWatchdogPath -Force -ErrorAction SilentlyContinue
        Write-WrapperLog "watchdog sentinel removed path=$SidecarWatchdogPath"
        try {
            Wait-Job -Job $SidecarWatchdogJob -Timeout 5 | Out-Null
            Receive-Job -Job $SidecarWatchdogJob -ErrorAction SilentlyContinue | Out-Null
        } catch {
        }
        Remove-Job -Job $SidecarWatchdogJob -Force -ErrorAction SilentlyContinue
        Write-WrapperLog "watchdog job removed"
    }
    if ((-not $DryRun) -and (-not $SkipSidecar) -and (-not $KeepStartedSidecar)) {
        try {
            Stop-CurrentPairProcesses -SidecarProcess $StartedSidecarProcess
            Write-WrapperLog "sidecar shutdown cleanup completed"
        } catch {
            $sidecarPid = if ($StartedSidecarProcess) { $StartedSidecarProcess.Id } else { "none" }
            Write-WrapperLog "sidecar shutdown cleanup failed sidecarPid=$sidecarPid error=$($_.Exception.Message)"
            # Best-effort cleanup only. Do not mask the Pi exit reason.
        }
        Clear-SidecarRuntime
        Clear-CurrentPairJhbRoot
        Clear-CurrentPairWorkerSlot
    }
    # The pair id is per-launch wiring, not console state. Leaving it in the
    # console env makes the next launch in the same console reuse the pair
    # (set-if-absent), which replayed old directives live on 2026-06-11.
    Remove-Item Env:\JARVIS_PAIR_ID -ErrorAction SilentlyContinue
    # Same residue problem: the job flag outlives the job object in this
    # console, so the next launch here would silently skip job creation.
    Remove-Item Env:\JARVIS_IN_PROCESS_JOB -ErrorAction SilentlyContinue
    Write-WrapperLog "pair env cleared on exit"
    Write-WrapperLog "wrapper end"
}
