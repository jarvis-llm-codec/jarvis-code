param(
    [string] $InstallDir = $(if ($env:JARVIS_CODE_INSTALL_DIR) { $env:JARVIS_CODE_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "JARVIS-Code" }),
    [string] $Repo = $(if ($env:JARVIS_CODE_REPO) { $env:JARVIS_CODE_REPO } else { "jarvis-llm-codec/jarvis-code" }),
    [string] $Branch = $(if ($env:JARVIS_CODE_BRANCH) { $env:JARVIS_CODE_BRANCH } else { "main" }),
    [string] $ArchiveUrl = $(if ($env:JARVIS_CODE_ARCHIVE_URL) { $env:JARVIS_CODE_ARCHIVE_URL } else { "" }),
    [switch] $NoPathUpdate,
    [switch] $NoPrereqInstall,
    [switch] $NoModelPreload,
    [switch] $RequireModelPreload
)

$ErrorActionPreference = "Stop"

# Allow this process to load .ps1 helpers (such as npm.ps1) even when the system
# execution policy is Restricted. Process scope only and reverts when the shell
# closes; managed/GPO machines that block this are covered by npm.cmd preference.
try {
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force -ErrorAction Stop
} catch {
}

function Write-Step {
    param([string] $Message)
    Write-Host "[jarvis-install] $Message"
}

function Test-LocalPackage {
    param([AllowEmptyString()][string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return $false
    }
    return (Test-Path (Join-Path $Path "jarvis.ps1")) -and
        (Test-Path (Join-Path $Path "sidecar")) -and
        (Test-Path (Join-Path $Path "pi"))
}

function Get-ScriptDirectory {
    if ($PSScriptRoot) {
        return $PSScriptRoot
    }
    if ($MyInvocation.MyCommand.Path) {
        return Split-Path -Parent $MyInvocation.MyCommand.Path
    }
    return ""
}

function New-PythonCommand {
    param(
        [string] $Command,
        [string[]] $BaseArgs = @()
    )
    return [pscustomobject]@{
        Command = $Command
        Args = $BaseArgs
    }
}

function Test-AutoPrereqInstall {
    return (-not $NoPrereqInstall) -and ($env:JARVIS_CODE_NO_PREREQ_INSTALL -ne "1")
}

function Test-ModelPreload {
    return (-not $NoModelPreload) -and ($env:JARVIS_CODE_NO_MODEL_PRELOAD -ne "1")
}

function Test-RequireModelPreload {
    return $RequireModelPreload -or ($env:JARVIS_CODE_REQUIRE_MODEL_PRELOAD -eq "1")
}

function Update-ProcessPath {
    $parts = @()
    foreach ($scope in @("Machine", "User")) {
        $value = [Environment]::GetEnvironmentVariable("Path", $scope)
        if ($value) {
            $parts += $value.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)
        }
    }
    $parts += $env:Path.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)

    $seen = @{}
    $deduped = @()
    foreach ($part in $parts) {
        $trimmed = $part.Trim()
        if (-not $trimmed) { continue }
        $key = $trimmed.ToLowerInvariant()
        if ($seen.ContainsKey($key)) { continue }
        $seen[$key] = $true
        $deduped += $trimmed
    }
    $env:Path = $deduped -join ";"
}

function Install-WingetPackage {
    param(
        [string] $PackageId,
        [string] $DisplayName
    )

    if (-not (Test-AutoPrereqInstall)) {
        throw "$DisplayName is required. Install it manually or rerun without -NoPrereqInstall."
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$DisplayName is required, and winget was not found. Install $DisplayName manually, then rerun the installer."
    }

    Write-Step "$DisplayName not found; installing with winget"
    $wingetArgs = @(
        "install",
        "--id", $PackageId,
        "--exact",
        "--source", "winget",
        "--silent",
        "--accept-package-agreements",
        "--accept-source-agreements"
    )
    $process = Start-Process -FilePath $winget.Source -ArgumentList $wingetArgs -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "winget failed to install $DisplayName with exit code $($process.ExitCode). Install it manually, then rerun the installer."
    }
    Update-ProcessPath
}

function Find-NodeCommand {
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) { return $node.Source }
    return $null
}

function Test-NodeVersion {
    param([AllowEmptyString()][string] $Node)
    if (-not $Node) { return $false }
    try {
        $versionText = & $Node --version
        if ($versionText -notmatch '^v?(\d+)\.') {
            return $false
        }
        return [int]$Matches[1] -ge 20
    } catch {
        return $false
    }
}

function Get-NodeCommand {
    $node = Find-NodeCommand
    if (Test-NodeVersion $node) { return $node }

    Install-WingetPackage -PackageId "OpenJS.NodeJS.LTS" -DisplayName "Node.js 20 or newer"
    $node = Find-NodeCommand
    if (Test-NodeVersion $node) { return $node }

    throw "Node.js 20 or newer is required. Install it from https://nodejs.org/ or run: winget install OpenJS.NodeJS.LTS"
}

function Test-PythonVersion {
    param(
        [string] $Command,
        [string[]] $BaseArgs = @()
    )
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command @BaseArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
}

function Find-PythonCommand {
    $candidates = @()

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $candidates += New-PythonCommand -Command $py.Source -BaseArgs @("-3")
        foreach ($minor in @("14", "13", "12", "11", "10")) {
            $candidates += New-PythonCommand -Command $py.Source -BaseArgs @("-3.$minor")
        }
    }

    foreach ($name in @("python", "python3")) {
        $python = Get-Command $name -ErrorAction SilentlyContinue
        if ($python) {
            $candidates += New-PythonCommand -Command $python.Source
        }
    }

    foreach ($candidate in $candidates) {
        if (Test-PythonVersion -Command $candidate.Command -BaseArgs $candidate.Args) {
            return $candidate
        }
    }

    return $null
}

function Get-PythonCommand {
    $python = Find-PythonCommand
    if ($python) { return $python }

    Install-WingetPackage -PackageId "Python.Python.3.12" -DisplayName "Python 3.10 or newer"
    $python = Find-PythonCommand
    if ($python) { return $python }

    throw "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ or run: winget install Python.Python.3.12"
}

function Invoke-Python {
    param(
        [pscustomobject] $Python,
        [string[]] $CommandArgs
    )
    $command = $Python.Command
    $allArgs = @()
    if ($Python.Args) {
        $allArgs += $Python.Args
    }
    $allArgs += $CommandArgs
    & $command @allArgs
}

function Assert-NodeVersion {
    param([string] $Node)
    $versionText = & $Node --version
    if ($versionText -notmatch '^v?(\d+)\.') {
        throw "Could not parse Node.js version: $versionText"
    }
    if ([int]$Matches[1] -lt 20) {
        throw "Node.js 20 or newer is required; found $versionText"
    }
    return $versionText
}

function Get-NpmCommand {
    # Prefer npm.cmd over npm.ps1: PowerShell resolves bare "npm" to the
    # ExternalScript (npm.ps1) first, which fails under a Restricted policy.
    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if ($npmCmd) { return $npmCmd.Source }
    $npm = Get-Command npm -ErrorAction SilentlyContinue
    if (-not $npm) { throw "npm is required." }
    return $npm.Source
}

function Find-GitCommand {
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($git) { return $git.Source }
    $candidates = @(
        (Join-Path $env:ProgramFiles "Git\cmd\git.exe"),
        (Join-Path $env:ProgramFiles "Git\bin\git.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Git\cmd\git.exe")
    )
    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    if ($programFilesX86) {
        $candidates += Join-Path $programFilesX86 "Git\cmd\git.exe"
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            $gitDir = Split-Path -Parent $candidate
            $pathParts = @($env:Path.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries))
            if ($pathParts -notcontains $gitDir) {
                $env:Path = "$gitDir;$env:Path"
            }
            return $candidate
        }
    }
    return $null
}

function Get-GitCommand {
    $git = Find-GitCommand
    if ($git) { return $git }

    Install-WingetPackage -PackageId "Git.Git" -DisplayName "Git"
    $git = Find-GitCommand
    if ($git) { return $git }

    throw "Git is required. Install it from https://git-scm.com/ or run: winget install Git.Git"
}

function Test-VcRedistX64 {
    $paths = @(
        "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64",
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\X64"
    )
    foreach ($path in $paths) {
        try {
            $item = Get-ItemProperty -LiteralPath $path -ErrorAction Stop
            if ($item.Installed -eq 1 -or $item.Version) {
                return $true
            }
        } catch {
            continue
        }
    }
    return $false
}

function Install-VcRedistX64 {
    if (Test-VcRedistX64) {
        Write-Step "using Microsoft Visual C++ Redistributable (x64)"
        return
    }

    Install-WingetPackage -PackageId "Microsoft.VCRedist.2015+.x64" -DisplayName "Microsoft Visual C++ 2015-2022 Redistributable (x64)"
    if (Test-VcRedistX64) {
        Write-Step "using Microsoft Visual C++ Redistributable (x64)"
        return
    }

    throw "Microsoft Visual C++ 2015-2022 Redistributable (x64) is required. Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe"
}

function Stop-JarvisSidecarOnPort {
    param([int] $Port = 8765)

    try {
        $connections = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
    } catch {
        return
    }
    if ($connections.Count -eq 0) {
        return
    }

    $candidateIds = New-Object System.Collections.Generic.HashSet[int]
    foreach ($conn in $connections) {
        [void]$candidateIds.Add([int]$conn.OwningProcess)
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($conn.OwningProcess)" -ErrorAction Stop
            if ($proc.ParentProcessId) {
                [void]$candidateIds.Add([int]$proc.ParentProcessId)
            }
        } catch {
        }
    }

    $sidecarIds = @()
    foreach ($processId in $candidateIds) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction Stop
            if ($proc -and $proc.CommandLine -like "*jarvis_sidecar*") {
                $sidecarIds += [int]$processId
            }
        } catch {
        }
    }
    if ($sidecarIds.Count -eq 0) {
        return
    }

    Write-Step "stopping existing JARVIS sidecar on port $Port so the next run uses the updated install"
    foreach ($processId in ($sidecarIds | Sort-Object -Descending -Unique)) {
        try {
            & taskkill.exe /T /F /PID $processId 2>&1 | Out-Null
        } catch {
            Write-Warning "Failed to stop JARVIS sidecar process $processId`: $($_.Exception.Message)"
        }
    }
    Start-Sleep -Milliseconds 700
}

function Copy-Package {
    param(
        [string] $Source,
        [string] $Destination
    )
    if (-not (Test-Path $Destination)) {
        New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    }

    $resolvedSource = (Resolve-Path $Source).Path.TrimEnd("\")
    $resolvedDestination = (Resolve-Path $Destination).Path.TrimEnd("\")
    if ($resolvedSource -eq $resolvedDestination) {
        return
    }

    $excludeDirs = @(
        ".git",
        "_internal",
        "data",
        "pi-agent",
        "node_modules",
        ".venv",
        ".pytest_cache",
        "__pycache__"
    )
    $excludeFiles = @("*.pyc", "*.pyo", ".env", "*.log")

    $items = Get-ChildItem -LiteralPath $Source -Force
    foreach ($item in $items) {
        if ($item.PSIsContainer -and $excludeDirs -contains $item.Name) { continue }
        if (-not $item.PSIsContainer) {
            $skip = $false
            foreach ($pattern in $excludeFiles) {
                if ($item.Name -like $pattern) { $skip = $true; break }
            }
            if ($skip) { continue }
        }
        Copy-Item -LiteralPath $item.FullName -Destination $Destination -Recurse -Force
    }
}

function Download-Package {
    param([string] $Destination)

    if (-not $ArchiveUrl) {
        if (-not $Repo) {
            throw "Set JARVIS_CODE_REPO=jarvis-llm-codec/jarvis-code or JARVIS_CODE_ARCHIVE_URL before running the remote installer."
        }
        $script:ArchiveUrl = "https://github.com/$Repo/archive/refs/heads/$Branch.zip"
    }

    $tmpRoot = Join-Path $env:TEMP ("jarvis-install-" + [guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null
    $zipPath = Join-Path $tmpRoot "jarvis-code.zip"
    $extractPath = Join-Path $tmpRoot "extract"

    Write-Step "downloading $ArchiveUrl"
    Invoke-WebRequest -Uri $ArchiveUrl -OutFile $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractPath -Force
    $source = Get-ChildItem -LiteralPath $extractPath -Directory | Select-Object -First 1
    if (-not $source) {
        throw "Archive did not contain a source folder."
    }
    Copy-Package -Source $source.FullName -Destination $Destination
}

function Install-NodeDependencies {
    param([string] $Root)
    $npm = Get-NpmCommand
    $piRoot = Join-Path $Root "pi"
    if (-not (Test-Path $piRoot)) { throw "Missing pi engine folder at $piRoot" }
    Push-Location $piRoot
    try {
        $oldHusky = $env:HUSKY
        $env:HUSKY = "0"
        if (Test-Path "package-lock.json") {
            Write-Step "installing Node dependencies with npm ci"
            & $npm ci --include=dev
        } else {
            Write-Step "installing Node dependencies with npm install"
            & $npm install --include=dev
        }
        if ($LASTEXITCODE -ne 0) {
            throw "npm dependency install failed with exit code $LASTEXITCODE"
        }
    } finally {
        if ($null -eq $oldHusky) {
            Remove-Item Env:\HUSKY -ErrorAction SilentlyContinue
        } else {
            $env:HUSKY = $oldHusky
        }
        Pop-Location
    }
}

function Install-SidecarVenv {
    param([string] $Root)
    $python = Get-PythonCommand
    $sidecarRoot = Join-Path $Root "sidecar"
    $venvDir = Join-Path $sidecarRoot ".venv"
    $venvPython = Join-Path $venvDir "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Step "creating sidecar venv"
        Invoke-Python -Python $python -CommandArgs @("-m", "venv", $venvDir)
        if ($LASTEXITCODE -ne 0) { throw "python -m venv failed with exit code $LASTEXITCODE" }
    }
    Write-Step "installing sidecar Python dependencies"
    & $venvPython -m pip install --disable-pip-version-check --quiet --upgrade pip "setuptools<82" wheel
    if ($LASTEXITCODE -ne 0) { throw "pip bootstrap failed with exit code $LASTEXITCODE" }
    & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $sidecarRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }
}

function Install-EmbedderModel {
    param([string] $Root)
    if (-not (Test-ModelPreload)) {
        Write-Step "skipping bge-m3 preload (disabled)"
        return
    }
    $venvPython = Join-Path $Root "sidecar\.venv\Scripts\python.exe"
    $doctor = Join-Path $Root "scripts\jarvis-doctor.py"
    if (-not (Test-Path $venvPython)) {
        throw "Cannot preload bge-m3 because sidecar venv Python was not found at $venvPython"
    }
    if (-not (Test-Path $doctor)) {
        throw "Cannot preload bge-m3 because JARVIS doctor was not found at $doctor"
    }
    Write-Step "preloading bge-m3 embedding model (first install may download about 2.3 GB)"
    & $venvPython $doctor --preload-embedder --require-embedder --skip-sidecar
    if ($LASTEXITCODE -ne 0) {
        $message = "bge-m3 preload failed with exit code $LASTEXITCODE. Install will continue; run 'jarvis doctor --preload-embedder' after install to see details."
        if (Test-RequireModelPreload) {
            throw "$message Rerun without -RequireModelPreload to allow degraded install."
        }
        Write-Warning $message
    }
}

function Install-JarvisCommand {
    param([string] $Root)
    $binDir = Join-Path $Root "bin"
    New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    $cmdPath = Join-Path $binDir "jarvis.cmd"
    $launcher = Join-Path $Root "jarvis.ps1"
    "@echo off`r`npowershell -NoProfile -ExecutionPolicy Bypass -File `"$launcher`" %*`r`n" |
        Set-Content -LiteralPath $cmdPath -Encoding ASCII

    if ($NoPathUpdate) {
        Write-Step "created command shim at $cmdPath"
        return
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) { $userPath = "" }
    $parts = $userPath.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)
    if ($parts -notcontains $binDir) {
        [Environment]::SetEnvironmentVariable("Path", ($userPath.TrimEnd(";") + ";$binDir").TrimStart(";"), "User")
        Write-Step "added $binDir to the user PATH; restart the terminal if jarvis is not found"
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
    param([string] $Root)
    $piAgentDir = Join-Path $Root "pi-agent"
    $settingsPath = Join-Path $piAgentDir "settings.json"
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

function Install-JarvisDefaultResources {
    param([string] $Root)
    $resourcesDir = Join-Path $Root "jarvis-resources"
    $piAgentDir = Join-Path $Root "pi-agent"

    $sourceSkills = Join-Path $resourcesDir "skills"
    if (Test-Path $sourceSkills) {
        $targetSkills = Join-Path $piAgentDir "skills"
        New-Item -ItemType Directory -Path $targetSkills -Force | Out-Null
        foreach ($skillDir in Get-ChildItem -LiteralPath $sourceSkills -Directory -ErrorAction SilentlyContinue) {
            Copy-Item -LiteralPath $skillDir.FullName -Destination $targetSkills -Recurse -Force
        }
    }

    $sourceThemes = Join-Path $resourcesDir "themes"
    if (Test-Path $sourceThemes) {
        $targetThemes = Join-Path $piAgentDir "themes"
        New-Item -ItemType Directory -Path $targetThemes -Force | Out-Null
        foreach ($themeFile in Get-ChildItem -LiteralPath $sourceThemes -File -Filter "*.json" -ErrorAction SilentlyContinue) {
            Copy-Item -LiteralPath $themeFile.FullName -Destination (Join-Path $targetThemes $themeFile.Name) -Force
        }
    }

    Set-JarvisDefaultSettings -Root $Root
}

$node = Get-NodeCommand
$nodeVersion = Assert-NodeVersion $node
Write-Step "using Node $nodeVersion"

$git = Get-GitCommand
$gitVersion = & $git --version
Write-Step "using $gitVersion"

Install-VcRedistX64

$scriptDir = Get-ScriptDirectory
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

if (Test-LocalPackage $scriptDir) {
    Write-Step "installing from local package $scriptDir"
    Copy-Package -Source $scriptDir -Destination $InstallDir
} else {
    Download-Package -Destination $InstallDir
}

New-Item -ItemType Directory -Path (Join-Path $InstallDir "data") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $InstallDir "pi-agent") -Force | Out-Null
Install-JarvisDefaultResources -Root $InstallDir

Install-NodeDependencies -Root $InstallDir
Install-SidecarVenv -Root $InstallDir
Install-JarvisCommand -Root $InstallDir
Install-EmbedderModel -Root $InstallDir
$sidecarPort = if ($env:JARVIS_SIDECAR_PORT) { [int]$env:JARVIS_SIDECAR_PORT } else { 8765 }
Stop-JarvisSidecarOnPort -Port $sidecarPort

Write-Step "installed JARVIS Code at $InstallDir"
Write-Host ""
Write-Host "  ============================================================"
Write-Host "   JARVIS Code is installed."
Write-Host ""
Write-Host "   The 'jarvis' command was added to your PATH, but THIS"
Write-Host "   terminal started before that, so it can't see it yet."
Write-Host ""
Write-Host "   Next steps:"
Write-Host "     1) Close this window and open a NEW terminal."
Write-Host "     2) Sign in once:   jarvis gpt-login"
Write-Host "     3) Start JARVIS:   jarvis"
Write-Host ""
Write-Host "   Diagnostics anytime:   jarvis doctor"
Write-Host "  ============================================================"
Write-Host ""
