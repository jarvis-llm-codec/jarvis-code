#!/usr/bin/env python3
"""Cross-platform JARVIS Code launcher.

The Windows PowerShell wrapper remains the most complete Windows entrypoint.
This launcher provides the shared path for macOS/Linux and the future common
runtime: set JARVIS environment, start the Python sidecar, and run the internal
agent engine with the JARVIS extensions loaded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
PI_ROOT = ROOT / "pi"
SIDECAR_ROOT = ROOT / "sidecar"
SIDECAR_VENV = SIDECAR_ROOT / ".venv"
DATA_DIR = ROOT / "data"
PI_AGENT_DIR = ROOT / "pi-agent"
DEFAULT_RESOURCES_DIR = ROOT / "jarvis-resources"
EXTENSION_PATH = PI_ROOT / "packages" / "coding-agent" / "examples" / "extensions" / "jarvis-jlc.ts"
FACE_EXTENSION_PATH = PI_ROOT / "packages" / "coding-agent" / "examples" / "extensions" / "jarvis-face.ts"
DOCTOR_SCRIPT = ROOT / "scripts" / "jarvis-doctor.py"
AUTH_SCRIPT = ROOT / "scripts" / "jarvis-auth.py"

AUTH_COMMANDS = {
    "gpt-login",
    "gpt-login-device",
    "gpt-auth-status",
    "gpt-logout",
    "api-key",
    "model-setting",
    "auth-status",
}

BUILT_IN_MODEL_PROVIDERS = {
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
    "xiaomi-token-plan-sgp",
}


def is_windows() -> bool:
    return os.name == "nt"


def venv_python() -> Path:
    if is_windows():
        return SIDECAR_VENV / "Scripts" / "python.exe"
    return SIDECAR_VENV / "bin" / "python"


def tsx_path() -> Path:
    if is_windows():
        return PI_ROOT / "node_modules" / ".bin" / "tsx.cmd"
    return PI_ROOT / "node_modules" / ".bin" / "tsx"


def truthy(value: str | None) -> bool:
    return value is not None and value.lower() in {"1", "true", "yes", "on"}


def has_arg(args: list[str], *names: str) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in args)


def skip_sidecar(args: list[str]) -> bool:
    return truthy(os.environ.get("JARVIS_WRAPPER_DRY_RUN")) or has_arg(args, "--help", "-h", "--version", "-v")


def read_chat_role_from_config(config_path: Path) -> tuple[str, str] | None:
    if not config_path.exists():
        return None
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    roles_match = re.search(r"(?ms)^roles:\s*\n((?:[ \t]+[^\n]+\n?)+)", content)
    if not roles_match:
        return None
    chat_match = re.search(r"(?m)^[ \t]+chat:[ \t]*(\S+)", roles_match.group(1))
    if not chat_match:
        return None
    chat = chat_match.group(1).strip("\"'")
    if "/" not in chat:
        return None
    provider, model = chat.split("/", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        return None
    return provider, model


def model_registered_in_pi_models(provider: str, model: str) -> bool:
    path = PI_AGENT_DIR / "models.json"
    if not path.exists():
        return False
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        return False
    provider_block = providers.get(provider)
    if not isinstance(provider_block, dict):
        return False
    models = provider_block.get("models")
    if models is None:
        return True
    if not isinstance(models, list):
        return False
    return any(isinstance(entry, dict) and entry.get("id") == model for entry in models)


def launchable_config_model(provider: str, model: str) -> bool:
    if provider in BUILT_IN_MODEL_PROVIDERS:
        return True
    return model_registered_in_pi_models(provider, model)


def default_provider_args(args: list[str], config_path: Path) -> list[str]:
    if has_arg(args, "--provider") or has_arg(args, "--model"):
        return []

    provider = os.environ.get("JARVIS_DEFAULT_PROVIDER")
    model = os.environ.get("JARVIS_DEFAULT_MODEL")
    if not provider or not model:
        role = read_chat_role_from_config(config_path)
        if role and launchable_config_model(role[0], role[1]):
            provider = provider or role[0]
            model = model or role[1]

    if not provider or not model:
        return []
    return ["--provider", provider, "--model", model]


def normalize_args(args: list[str]) -> tuple[list[str], str | None]:
    normalized: list[str] = []
    recent_turns: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--recent-turns" and i + 1 < len(args):
            recent_turns = args[i + 1]
            i += 2
            continue
        if arg.startswith("--recent-turns="):
            recent_turns = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--auto-prompts" and i + 1 < len(args):
            value = args[i + 1]
            if value and not os.path.isabs(value):
                value = str((ROOT / value).resolve())
            normalized.extend([arg, value])
            i += 2
            continue
        if arg.startswith("--auto-prompts="):
            value = arg.split("=", 1)[1]
            if value and not os.path.isabs(value):
                value = str((ROOT / value).resolve())
            normalized.append(f"--auto-prompts={value}")
            i += 1
            continue
        normalized.append(arg)
        i += 1
    return normalized, recent_turns


def configure_env(args: list[str]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PI_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap_default_resources()

    config_path = Path(os.environ.get("JARVIS_CODE_CONFIG", str(DATA_DIR / "config.yaml")))
    port = os.environ.get("JARVIS_SIDECAR_PORT", "8765")

    os.environ["JARVIS_CODE_CONFIG"] = str(config_path)
    os.environ["JARVIS_SIDECAR_PORT"] = port
    os.environ["JARVIS_SIDECAR_URL"] = f"http://127.0.0.1:{port}"
    os.environ["JARVIS_SIDECAR_RUNTIME"] = str(DATA_DIR / "sidecar-runtime.json")
    os.environ["JARVIS_WRAPPER_LOG"] = str(DATA_DIR / "jarvis-wrapper.log")
    os.environ["JARVIS_DISABLE_COMPACTION"] = "1"
    os.environ["JARVIS_DISABLE_AUTO_COMPACTION"] = "1"
    os.environ["JARVIS_RUNTIME_HISTORY_TURNS"] = "100"
    os.environ.setdefault("JARVIS_SUBTURN_COMPACT", "0")
    os.environ["JARVIS_DISABLE_PI_AGENT_UPDATE"] = "1"
    os.environ["PI_SKIP_VERSION_CHECK"] = "1"
    os.environ["PI_SKIP_PACKAGE_UPDATE_CHECK"] = "1"
    os.environ.setdefault("JARVIS_CODE_CODING_AGENT_DIR", str(PI_AGENT_DIR))
    os.environ.setdefault("PI_CODING_AGENT_DIR", os.environ["JARVIS_CODE_CODING_AGENT_DIR"])

    normalized, recent_turns = normalize_args(args)
    if recent_turns is not None:
        try:
            parsed = int(recent_turns)
        except ValueError as exc:
            raise SystemExit(f"--recent-turns expects a non-negative integer, got {recent_turns!r}") from exc
        if parsed < 0:
            raise SystemExit(f"--recent-turns expects a non-negative integer, got {recent_turns!r}")
        os.environ["JARVIS_RECENT_TURNS"] = str(parsed)
    else:
        os.environ["JARVIS_RECENT_TURNS"] = "1"

    args[:] = normalized
    return config_path


def ensure_default_settings() -> None:
    settings_path = PI_AGENT_DIR / "settings.json"
    settings: dict[str, object]
    if settings_path.exists():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        settings = raw
    else:
        settings = {}

    if isinstance(settings.get("theme"), str) and settings["theme"].strip():
        return

    settings["theme"] = "orange-blue"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def copy_default_resource_dir(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, target)


def bootstrap_default_resources() -> None:
    copy_default_resource_dir(DEFAULT_RESOURCES_DIR / "skills", PI_AGENT_DIR / "skills")
    copy_default_resource_dir(DEFAULT_RESOURCES_DIR / "themes", PI_AGENT_DIR / "themes")
    ensure_default_settings()


def health_url() -> str:
    return f"{os.environ['JARVIS_SIDECAR_URL']}/health"


def sidecar_healthy() -> bool:
    try:
        with urllib.request.urlopen(health_url(), timeout=2) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("ok") is True and data.get("service") == "jarvis-jlc-sidecar"
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def choose_port() -> None:
    if sidecar_healthy():
        return
    preferred = int(os.environ.get("JARVIS_SIDECAR_PORT", "8765"))
    if port_is_free(preferred):
        return
    for candidate in range(preferred + 1, preferred + 21):
        if port_is_free(candidate):
            os.environ["JARVIS_SIDECAR_PORT"] = str(candidate)
            os.environ["JARVIS_SIDECAR_URL"] = f"http://127.0.0.1:{candidate}"
            print(f"[jarvis] port {preferred} is busy; using sidecar port {candidate}", file=sys.stderr)
            return
    raise SystemExit(f"No free JARVIS sidecar port found in range {preferred}-{preferred + 20}.")


def find_host_python() -> str:
    for name in ("python3", "python"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Python 3.10 or newer is required.")


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_sidecar_venv() -> Path:
    py = venv_python()
    if py.exists():
        return py

    host_python = find_host_python()
    print(f"[jarvis] creating sidecar venv at {SIDECAR_VENV}")
    run_checked([host_python, "-m", "venv", str(SIDECAR_VENV)])
    if not py.exists():
        raise SystemExit(f"failed to create sidecar venv at {SIDECAR_VENV}")

    print("[jarvis] installing sidecar requirements")
    run_checked([str(py), "-m", "pip", "install", "--disable-pip-version-check", "--quiet", "--upgrade", "pip", "setuptools<82", "wheel"])
    run_checked([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(SIDECAR_ROOT / "requirements.txt")])
    return py


def sidecar_env() -> dict[str, str]:
    env = os.environ.copy()
    python_path = [str(SIDECAR_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        python_path.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def run_doctor(args: list[str]) -> int:
    py = venv_python()
    python = str(py) if py.exists() else sys.executable
    if not DOCTOR_SCRIPT.exists():
        print(f"JARVIS doctor script not found at {DOCTOR_SCRIPT}", file=sys.stderr)
        return 1
    env = sidecar_env()
    return subprocess.call([python, str(DOCTOR_SCRIPT), *args], cwd=str(ROOT), env=env)


def run_auth_command(args: list[str]) -> int:
    if not AUTH_SCRIPT.exists():
        print(f"JARVIS auth script not found at {AUTH_SCRIPT}", file=sys.stderr)
        return 1
    py = ensure_sidecar_venv()
    command_args = ["gpt-auth-status" if args and args[0] == "auth-status" else args[0], *args[1:]]
    return subprocess.call([str(py), str(AUTH_SCRIPT), *command_args], cwd=str(ROOT), env=sidecar_env())


def run_auth_preflight() -> int:
    if os.environ.get("JARVIS_AUTH_PREFLIGHT") == "0":
        return 0
    if not AUTH_SCRIPT.exists():
        print(f"JARVIS auth script not found at {AUTH_SCRIPT}", file=sys.stderr)
        return 1
    py = ensure_sidecar_venv()
    return subprocess.call([str(py), str(AUTH_SCRIPT), "preflight"], cwd=str(ROOT), env=sidecar_env())


def start_sidecar(args: list[str]) -> subprocess.Popen[str] | None:
    if skip_sidecar(args):
        return None
    choose_port()
    if sidecar_healthy():
        return None

    py = ensure_sidecar_venv()
    proc = subprocess.Popen(
        [str(py), "-m", "jarvis_sidecar"],
        cwd=str(SIDECAR_ROOT),
        env=sidecar_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    deadline = time.time() + 20
    while time.time() < deadline:
        if sidecar_healthy():
            runtime_path = Path(os.environ["JARVIS_SIDECAR_RUNTIME"])
            runtime_path.write_text(
                json.dumps(
                    {
                        "url": os.environ["JARVIS_SIDECAR_URL"],
                        "port": int(os.environ["JARVIS_SIDECAR_PORT"]),
                        "pid": proc.pid,
                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    print("[jarvis] warning: sidecar did not become healthy; continuing with degraded memory", file=sys.stderr)
    return proc


def build_forward_args(args: list[str], config_path: Path) -> list[str]:
    forward: list[str] = []
    if not truthy(os.environ.get("JARVIS_ENABLE_EXTENSION_DISCOVERY")):
        forward.append("--no-extensions")
    if EXTENSION_PATH.exists():
        forward.extend(["--extension", str(EXTENSION_PATH)])
    if FACE_EXTENSION_PATH.exists():
        forward.extend(["--extension", str(FACE_EXTENSION_PATH)])
    forward.extend(default_provider_args(args, config_path))
    forward.extend(args)
    return forward


def run_agent(args: list[str], config_path: Path) -> int:
    tsx = tsx_path()
    if not tsx.exists():
        print(f"tsx not found at {tsx}", file=sys.stderr)
        print("Run the installer again, or run `npm ci` from the pi directory.", file=sys.stderr)
        return 1

    forward = build_forward_args(args, config_path)
    if truthy(os.environ.get("JARVIS_WRAPPER_DRY_RUN")):
        print(
            json.dumps(
                {
                    "pi_root": str(PI_ROOT),
                    "sidecar_url": os.environ.get("JARVIS_SIDECAR_URL"),
                    "config_path": str(config_path),
                    "pi_agent_dir": os.environ.get("JARVIS_CODE_CODING_AGENT_DIR"),
                    "forward_args": forward,
                },
                indent=2,
            )
        )
        return 0

    cli = PI_ROOT / "packages" / "coding-agent" / "src" / "cli.ts"
    return subprocess.call([str(tsx), str(cli), *forward], cwd=str(PI_ROOT), env=os.environ.copy())


def terminate_started_sidecar(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if truthy(os.environ.get("JARVIS_KEEP_SIDECAR")):
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    if args and args[0] == "doctor":
        configure_env([])
        return run_doctor(args[1:])
    if args and args[0] in AUTH_COMMANDS:
        configure_env([])
        return run_auth_command(args)
    config_path = configure_env(args)
    sidecar_proc: subprocess.Popen[str] | None = None
    try:
        if not skip_sidecar(args):
            auth_code = run_auth_preflight()
            if auth_code != 0:
                return auth_code
        sidecar_proc = start_sidecar(args)
        return run_agent(args, config_path)
    finally:
        terminate_started_sidecar(sidecar_proc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
