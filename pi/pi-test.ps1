$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$noEnv = $false
$forwardArgs = New-Object System.Collections.Generic.List[string]

function Write-PiTestLog {
	param([string] $Message)

	if (-not $env:JARVIS_WRAPPER_LOG) { return }
	try {
		$dir = Split-Path -Parent $env:JARVIS_WRAPPER_LOG
		if ($dir -and -not (Test-Path $dir)) {
			New-Item -ItemType Directory -Path $dir -Force | Out-Null
		}
		$timestamp = (Get-Date).ToUniversalTime().ToString("o")
		Add-Content -LiteralPath $env:JARVIS_WRAPPER_LOG -Value "[$timestamp] $Message" -Encoding UTF8
	} catch {
		# Best-effort diagnostic logging only.
	}
}

foreach ($arg in $args) {
	if ($arg -eq "--no-env") {
		$noEnv = $true
	} else {
		$forwardArgs.Add($arg)
	}
}

$env:JARVIS_DISABLE_COMPACTION = "1"
$env:JARVIS_DISABLE_AUTO_COMPACTION = "1"
$env:JARVIS_RUNTIME_HISTORY_TURNS = "100"
$env:JARVIS_DISABLE_PI_AGENT_UPDATE = "1"
$env:PI_SKIP_VERSION_CHECK = "1"
$env:PI_SKIP_PACKAGE_UPDATE_CHECK = "1"

if ($noEnv) {
	$envVarsToUnset = @(
		"ANTHROPIC_API_KEY",
		"ANTHROPIC_OAUTH_TOKEN",
		"OPENAI_API_KEY",
		"GEMINI_API_KEY",
		"GROQ_API_KEY",
		"CEREBRAS_API_KEY",
		"XAI_API_KEY",
		"OPENROUTER_API_KEY",
		"ZAI_API_KEY",
		"MISTRAL_API_KEY",
		"MINIMAX_API_KEY",
		"MINIMAX_CN_API_KEY",
		"AI_GATEWAY_API_KEY",
		"OPENCODE_API_KEY",
		"COPILOT_GITHUB_TOKEN",
		"GH_TOKEN",
		"GITHUB_TOKEN",
		"GOOGLE_APPLICATION_CREDENTIALS",
		"GOOGLE_CLOUD_PROJECT",
		"GCLOUD_PROJECT",
		"GOOGLE_CLOUD_LOCATION",
		"AWS_PROFILE",
		"AWS_ACCESS_KEY_ID",
		"AWS_SECRET_ACCESS_KEY",
		"AWS_SESSION_TOKEN",
		"AWS_REGION",
		"AWS_DEFAULT_REGION",
		"AWS_BEARER_TOKEN_BEDROCK",
		"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
		"AWS_CONTAINER_CREDENTIALS_FULL_URI",
		"AWS_WEB_IDENTITY_TOKEN_FILE",
		"AZURE_OPENAI_API_KEY",
		"AZURE_OPENAI_BASE_URL",
		"AZURE_OPENAI_RESOURCE_NAME"
	)

	foreach ($name in $envVarsToUnset) {
		Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
	}

	Write-Host "Running without API keys..."
}

$tsxBin = Join-Path $scriptDir "node_modules/.bin/tsx.cmd"
if (-not (Test-Path -LiteralPath $tsxBin)) {
	throw "tsx not found at $tsxBin. Run npm install from the repo root first."
}

$cliPath = Join-Path $scriptDir "packages/coding-agent/src/cli.ts"
Write-PiTestLog "pi-test start args=$($forwardArgs -join ' ')"
& $tsxBin $cliPath @forwardArgs
$exitCode = $LASTEXITCODE
Write-PiTestLog "pi-test child exitCode=$exitCode"
if ($exitCode -ne 0) {
	exit $exitCode
}
