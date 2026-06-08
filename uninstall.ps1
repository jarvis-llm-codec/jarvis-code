param(
    [string] $InstallDir = $(if ($env:JARVIS_CODE_INSTALL_DIR) { $env:JARVIS_CODE_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "JARVIS-Code" }),
    [string] $UserDataDir = $(if ($env:JARVIS_CODE_USER_DATA_DIR) { $env:JARVIS_CODE_USER_DATA_DIR } else { Join-Path ([Environment]::GetFolderPath("UserProfile")) ".jarvis-code" }),
    [switch] $RemoveUserData,
    [switch] $RemoveModelCache,
    [switch] $KeepPath
)

$ErrorActionPreference = "Stop"

$RemoveUserData = [bool]$RemoveUserData -or $env:JARVIS_CODE_REMOVE_USER_DATA -eq "1"
$RemoveModelCache = [bool]$RemoveModelCache -or $env:JARVIS_CODE_REMOVE_MODEL_CACHE -eq "1"
$KeepPath = [bool]$KeepPath -or $env:JARVIS_CODE_KEEP_PATH -eq "1"

function Write-Step {
    param([string] $Message)
    Write-Host "[jarvis-uninstall] $Message"
}

function Get-NormalizedPath {
    param([AllowEmptyString()][string] $Path)
    if ([string]::IsNullOrWhiteSpace($Path)) {
        return ""
    }
    try {
        $resolved = Resolve-Path -LiteralPath $Path -ErrorAction Stop
        return [System.IO.Path]::GetFullPath($resolved.Path).TrimEnd([char[]]@("\", "/"))
    } catch {
        return [System.IO.Path]::GetFullPath($Path).TrimEnd([char[]]@("\", "/"))
    }
}

function Test-SamePath {
    param(
        [string] $Left,
        [string] $Right
    )
    $leftPath = Get-NormalizedPath $Left
    $rightPath = Get-NormalizedPath $Right
    if (-not $leftPath -or -not $rightPath) {
        return $false
    }
    return [string]::Equals($leftPath, $rightPath, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-SafeDeletePath {
    param(
        [string] $Path,
        [string] $Purpose
    )
    $resolved = Get-NormalizedPath $Path
    if (-not $resolved) {
        throw "Refusing to delete empty $Purpose path."
    }

    $root = [System.IO.Path]::GetPathRoot($resolved)
    if ($root -and (Test-SamePath $resolved $root)) {
        throw "Refusing to delete drive root for $Purpose`: $resolved"
    }

    $protected = @(
        [Environment]::GetFolderPath("UserProfile"),
        $env:LOCALAPPDATA,
        $env:APPDATA,
        $env:ProgramFiles,
        [Environment]::GetEnvironmentVariable("ProgramFiles(x86)"),
        [System.IO.Path]::GetTempPath()
    )
    foreach ($protectedPath in $protected) {
        if ($protectedPath -and (Test-SamePath $resolved $protectedPath)) {
            throw "Refusing to delete protected $Purpose path: $resolved"
        }
    }
    return $resolved
}

function Test-JarvisInstallDir {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $hasCoreFiles = (Test-Path -LiteralPath (Join-Path $Path "jarvis.ps1")) -and
        (Test-Path -LiteralPath (Join-Path $Path "sidecar")) -and
        (Test-Path -LiteralPath (Join-Path $Path "pi"))
    if ($hasCoreFiles) {
        return $true
    }
    $leaf = Split-Path -Leaf (Get-NormalizedPath $Path)
    if ($leaf -in @("JARVIS-Code", "jarvis-code")) {
        return (Test-Path -LiteralPath (Join-Path $Path "install.ps1")) -or
            (Test-Path -LiteralPath (Join-Path $Path "install.sh")) -or
            (Test-Path -LiteralPath (Join-Path $Path "bin\jarvis.cmd"))
    }
    return $false
}

function Assert-JarvisInstallDir {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    if (-not (Test-JarvisInstallDir $Path)) {
        throw "Refusing to delete '$Path' because it does not look like a JARVIS Code install directory."
    }
}

function Test-JarvisUserDataDir {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $leaf = Split-Path -Leaf (Get-NormalizedPath $Path)
    if ($leaf -eq ".jarvis-code") {
        return $true
    }
    foreach ($name in @("config.yaml", "providers.yaml", "auth.json", "workspaceMemory", "conversation", "raw-store")) {
        if (Test-Path -LiteralPath (Join-Path $Path $name)) {
            return $true
        }
    }
    return $false
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

    Write-Step "stopping existing JARVIS sidecar on port $Port"
    foreach ($processId in ($sidecarIds | Sort-Object -Descending -Unique)) {
        try {
            & taskkill.exe /T /F /PID $processId *> $null
        } catch {
            Write-Warning "Failed to stop JARVIS sidecar process $processId`: $($_.Exception.Message)"
        }
    }
    Start-Sleep -Milliseconds 700
}

function Remove-JarvisPathEntry {
    param([string] $BinDir)
    if ($KeepPath) {
        Write-Step "keeping user PATH unchanged"
        return
    }

    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath) {
        return
    }

    $target = Get-NormalizedPath $BinDir
    $kept = @()
    $removed = $false
    foreach ($part in $userPath.Split(";", [System.StringSplitOptions]::RemoveEmptyEntries)) {
        $trimmed = $part.Trim()
        if (-not $trimmed) { continue }
        if ($target -and (Test-SamePath $trimmed $target)) {
            $removed = $true
            continue
        }
        $kept += $trimmed
    }

    if ($removed) {
        [Environment]::SetEnvironmentVariable("Path", ($kept -join ";"), "User")
        Write-Step "removed $target from the user PATH; restart terminals to refresh PATH"
    }
}

function Get-BgeM3CacheDir {
    if ($env:HF_HUB_CACHE) {
        return (Join-Path $env:HF_HUB_CACHE "models--BAAI--bge-m3")
    }
    if ($env:HF_HOME) {
        return (Join-Path (Join-Path $env:HF_HOME "hub") "models--BAAI--bge-m3")
    }
    return (Join-Path (Join-Path ([Environment]::GetFolderPath("UserProfile")) ".cache\huggingface\hub") "models--BAAI--bge-m3")
}

$sidecarPort = if ($env:JARVIS_SIDECAR_PORT) { [int]$env:JARVIS_SIDECAR_PORT } else { 8765 }
Stop-JarvisSidecarOnPort -Port $sidecarPort

$installPath = Assert-SafeDeletePath -Path $InstallDir -Purpose "JARVIS Code install"
Assert-JarvisInstallDir -Path $installPath

$binDir = Join-Path $installPath "bin"
Remove-JarvisPathEntry -BinDir $binDir

if (Test-Path -LiteralPath $installPath) {
    Remove-Item -LiteralPath $installPath -Recurse -Force
    Write-Step "removed install directory $installPath"
} else {
    Write-Step "install directory not found: $installPath"
}

if ($RemoveUserData) {
    $userDataPath = Assert-SafeDeletePath -Path $UserDataDir -Purpose "JARVIS Code user data"
    if (Test-Path -LiteralPath $userDataPath) {
        if (-not (Test-JarvisUserDataDir $userDataPath)) {
            throw "Refusing to delete '$userDataPath' because it does not look like JARVIS Code user data."
        }
        Remove-Item -LiteralPath $userDataPath -Recurse -Force
        Write-Step "removed user data $userDataPath"
    } else {
        Write-Step "user data directory not found: $userDataPath"
    }
} else {
    Write-Step "kept user data at $UserDataDir"
}

if ($RemoveModelCache) {
    $modelCache = Assert-SafeDeletePath -Path (Get-BgeM3CacheDir) -Purpose "bge-m3 model cache"
    if ((Split-Path -Leaf $modelCache) -ne "models--BAAI--bge-m3") {
        throw "Refusing to delete unexpected model cache path: $modelCache"
    }
    if (Test-Path -LiteralPath $modelCache) {
        Remove-Item -LiteralPath $modelCache -Recurse -Force
        Write-Step "removed bge-m3 model cache $modelCache"
    } else {
        Write-Step "bge-m3 model cache not found: $modelCache"
    }
} else {
    Write-Step "kept Hugging Face model cache"
}

Write-Step "uninstalled JARVIS Code"
Write-Step "system prerequisites such as Node.js, Python, Git, and VC++ Redistributable were not removed"
