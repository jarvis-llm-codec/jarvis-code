#!/usr/bin/env python3
"""Cross-platform JARVIS Code launcher.

The Windows PowerShell wrapper remains the most complete Windows entrypoint.
This launcher provides the shared path for macOS/Linux and the future common
runtime: set JARVIS environment, start the Python sidecar, and run the internal
agent engine with the JARVIS extensions loaded.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
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
IMAGE_EXTENSION_PATH = PI_ROOT / "packages" / "coding-agent" / "examples" / "extensions" / "jarvis-image.ts"
DOCTOR_SCRIPT = ROOT / "scripts" / "jarvis-doctor.py"
AUTH_SCRIPT = ROOT / "scripts" / "jarvis-auth.py"
MAX_WINDOW_LABEL_CHARS = 32
AUTO_SIDECAR_WINDOW_RUNS = 3
FIRST_RUN_COUNT_PATH = DATA_DIR / "first_run_count.json"
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
PYTORCH_CUDA_INSTALL_NOTE = "~2.7 GB, several minutes"
SIDECAR_REQUIREMENTS_INSTALL_NOTE = "~1.3 GB, first run only, takes minutes"

AUTH_COMMANDS = {
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
SIDECAR_ROUTED_MODEL_PROVIDERS = {"anthropic-agent-sdk"}


def is_windows() -> bool:
    return os.name == "nt"


def ensure_supported_python() -> None:
    if sys.version_info < (3, 11):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise SystemExit(f"JARVIS Code requires Python 3.11 or newer; found {version} at {sys.executable}")


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


def cpu_only_requested() -> bool:
    return truthy(os.environ.get("JARVIS_CODE_CPU_ONLY"))


def configure_bundled_node_path() -> None:
    node_bin = ROOT / "node" / "bin"
    if node_bin.is_dir():
        os.environ["PATH"] = f"{node_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def detect_nvidia_gpu() -> str | None:
    """Return an NVIDIA GPU name when nvidia-smi exists and runs successfully."""
    if cpu_only_requested():
        return None
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            name = line.strip()
            if name:
                return name
        return "NVIDIA GPU"
    try:
        probe = subprocess.run(
            [nvidia_smi],
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return "NVIDIA GPU" if probe.returncode == 0 else None


def sanitize_window_label(value: str | None) -> str | None:
    text = "".join(ch for ch in str(value or "") if ord(ch) >= 32 and ord(ch) != 127).strip()
    return text[:MAX_WINDOW_LABEL_CHARS] if text else None


def pair_runtime_prefix(pair_id: str | None) -> str:
    prefix = re.sub(r"[^A-Za-z0-9]", "", str(pair_id or ""))
    if len(prefix) >= 8:
        return prefix[:8]
    return f"{abs(os.getpid()):08x}"[-8:]


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if is_windows():
        process_query_limited_information = 0x1000
        still_active = 259
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def initialize_pair_id() -> str:
    pair_id = os.environ.get("JARVIS_PAIR_ID")
    if not pair_id:
        pair_id = str(uuid.uuid4())
        os.environ["JARVIS_PAIR_ID"] = pair_id
        return pair_id

    runtime_path = DATA_DIR / f"sidecar-runtime-{pair_runtime_prefix(pair_id)}.json"
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        runtime_pair = str(runtime.get("pair_id") or "")
        runtime_pid = int(runtime.get("pid") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return pair_id
    if runtime_pair == pair_id and runtime_pid > 0 and runtime_pid != os.getpid() and pid_alive(runtime_pid):
        pair_id = str(uuid.uuid4())
        os.environ["JARVIS_PAIR_ID"] = pair_id
    return pair_id


def has_arg(args: list[str], *names: str) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in args)


def skip_sidecar(args: list[str]) -> bool:
    return truthy(os.environ.get("JARVIS_WRAPPER_DRY_RUN")) or has_arg(args, "--help", "-h", "--version", "-v")


def read_role_from_config(config_path: Path, role_name: str) -> tuple[str, str] | None:
    if not config_path.exists():
        return None
    try:
        content = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    roles_match = re.search(r"(?ms)^roles:\s*\n((?:[ \t]+[^\n]+\n?)+)", content)
    if not roles_match:
        return None
    chat_match = re.search(rf"(?m)^[ \t]+{re.escape(role_name)}:[ \t]*(\S+)", roles_match.group(1))
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


def read_chat_role_from_config(config_path: Path) -> tuple[str, str] | None:
    return read_role_from_config(config_path, "chat")


def read_encoder_role_from_config(config_path: Path) -> tuple[str, str] | None:
    return read_role_from_config(config_path, "encoder")


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
    if provider in SIDECAR_ROUTED_MODEL_PROVIDERS:
        return False
    if provider in BUILT_IN_MODEL_PROVIDERS:
        return True
    return model_registered_in_pi_models(provider, model)


def sidecar_routed_provider(provider: str) -> bool:
    return provider in SIDECAR_ROUTED_MODEL_PROVIDERS


def sidecar_routed_launch_args(provider: str, model: str, config_path: Path) -> list[str]:
    os.environ["JARVIS_CHAT_MODEL_OVERRIDE"] = f"{provider}/{model}"
    encoder = read_encoder_role_from_config(config_path)
    if encoder and launchable_config_model(encoder[0], encoder[1]):
        return ["--provider", encoder[0], "--model", encoder[1]]
    # Both chat and encoder are sidecar-routed (e.g. anthropic-agent-sdk/*). The
    # sidecar drives all LLM work; Pi only needs a window-init shell that
    # resolves without a key, so launch Pi on the encoder's sidecar provider
    # (registered in pi-agent/models.json as a keyless openai-completions shim)
    # instead of throwing. Parity port of jarvis.ps1
    # Resolve-SidecarRoutedChatLaunchArgs — without this branch a chat+encoder
    # pair both on the Claude subscription launched fine on Windows but crashed
    # the macOS/Linux launcher at window open. (2026-07-10, live user report)
    if encoder and sidecar_routed_provider(encoder[0]):
        return ["--provider", encoder[0], "--model", encoder[1]]
    raise RuntimeError(
        f"Provider {provider}/{model} is sidecar-routed or not Pi-launchable; "
        "roles.encoder must be a Pi-launchable provider/model before opening the window."
    )


def extract_provider_model_args(args: list[str]) -> tuple[str | None, str | None]:
    provider: str | None = None
    model: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
            continue
        if arg.startswith("--provider="):
            provider = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
            continue
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            i += 1
            continue
        i += 1
    if not provider and model and "/" in model:
        provider, model = model.split("/", 1)
    return (provider.strip() if provider else None, model.strip() if model else None)


def strip_provider_model_args(args: list[str]) -> list[str]:
    stripped: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"--provider", "--model"} and i + 1 < len(args):
            i += 2
            continue
        if arg.startswith("--provider=") or arg.startswith("--model="):
            i += 1
            continue
        stripped.append(arg)
        i += 1
    return stripped


def resolve_provider_args(args: list[str], config_path: Path) -> tuple[list[str], list[str]]:
    cli_provider, cli_model = extract_provider_model_args(args)
    if cli_provider and cli_model:
        if launchable_config_model(cli_provider, cli_model):
            os.environ["JARVIS_CHAT_MODEL_OVERRIDE"] = f"{cli_provider}/{cli_model}"
            return [], args
        if not sidecar_routed_provider(cli_provider):
            raise RuntimeError(f"Provider {cli_provider}/{cli_model} is not Pi-launchable.")
        return sidecar_routed_launch_args(cli_provider, cli_model, config_path), strip_provider_model_args(args)
    if cli_provider or cli_model:
        return [], args

    provider = os.environ.get("JARVIS_DEFAULT_PROVIDER")
    model = os.environ.get("JARVIS_DEFAULT_MODEL")
    if provider and model:
        if launchable_config_model(provider, model):
            return ["--provider", provider, "--model", model], args
        if not sidecar_routed_provider(provider):
            raise RuntimeError(f"Provider {provider}/{model} is not Pi-launchable.")
        return sidecar_routed_launch_args(provider, model, config_path), args
    if provider or model:
        raise RuntimeError("Both JARVIS_DEFAULT_PROVIDER and JARVIS_DEFAULT_MODEL are required when either is set.")

    role = read_chat_role_from_config(config_path)
    if role and launchable_config_model(role[0], role[1]):
        provider, model = role
    elif role and sidecar_routed_provider(role[0]):
        return sidecar_routed_launch_args(role[0], role[1], config_path), args
    elif role:
        return [], args
    if not provider or not model:
        return [], args
    return ["--provider", provider, "--model", model], args


def load_credentials_env(config_path: Path) -> None:
    credentials_path = config_path.with_name("credentials.yaml")
    try:
        lines = credentials_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    in_env_block = False
    for line in lines:
        if re.match(r"^env:\s*$", line):
            in_env_block = True
            continue
        if not in_env_block:
            continue
        if re.match(r"^\S", line):
            break
        match = re.match(r"^\s+([A-Za-z_][A-Za-z0-9_]*):\s*(\S.*)$", line)
        if not match:
            continue
        name = match.group(1)
        value = match.group(2).strip().strip("\"'")
        if value and name not in os.environ:
            os.environ[name] = value


def normalize_args(args: list[str]) -> tuple[list[str], str | None]:
    normalized: list[str] = []
    recent_turns: str | None = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--window-label" and i + 1 < len(args):
            label = sanitize_window_label(args[i + 1])
            if label:
                os.environ["JARVIS_WINDOW_LABEL"] = label
            i += 2
            continue
        if arg.startswith("--window-label="):
            label = sanitize_window_label(arg.split("=", 1)[1])
            if label:
                os.environ["JARVIS_WINDOW_LABEL"] = label
            i += 1
            continue
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
        if arg == "--sidecar-window":
            os.environ["JARVIS_SHOW_SIDECAR_WINDOW"] = "1"
            i += 1
            continue
        if arg == "--yolo":
            os.environ["JARVIS_YOLO"] = "1"
            i += 1
            continue
        normalized.append(arg)
        i += 1
    return normalized, recent_turns


def configure_env(args: list[str]) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PI_AGENT_DIR.mkdir(parents=True, exist_ok=True)
    bootstrap_default_resources()
    configure_bundled_node_path()

    config_path = Path(os.environ.get("JARVIS_CODE_CONFIG", str(Path.home() / ".jarvis-code" / "config.yaml")))
    port = os.environ.get("JARVIS_SIDECAR_PORT", "8765")
    pair_id = initialize_pair_id()
    pair_prefix = pair_runtime_prefix(pair_id)

    os.environ["JARVIS_CODE_CONFIG"] = str(config_path)
    os.environ["JARVIS_SIDECAR_PORT"] = port
    os.environ["JARVIS_SIDECAR_URL"] = f"http://127.0.0.1:{port}"
    os.environ["JARVIS_SIDECAR_RUNTIME"] = str(DATA_DIR / f"sidecar-runtime-{pair_prefix}.json")
    os.environ["JARVIS_WRAPPER_PID"] = str(os.getpid())
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
    load_credentials_env(config_path)

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
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    if not (data.get("ok") is True and data.get("service") == "jarvis-jlc-sidecar"):
        return False
    # Mirror jarvis.ps1 Test-Sidecar: a healthy sidecar serving a DIFFERENT pair
    # is the parent's, not ours. Without this gate a spawned worker inherits the
    # parent's port, sees the parent sidecar as healthy, and skips starting its
    # own — so no runtime file for the worker's pair ever appears and the spawn
    # waiter times out. Rejecting the mismatch lets choose_port() pick a free
    # sibling port and bring up the worker's own sidecar.
    expected_pair = os.environ.get("JARVIS_PAIR_ID")
    if expected_pair:
        return str(data.get("pair_id") or "") == str(expected_pair)
    return True


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
    # The running interpreter already passed ensure_supported_python(), while
    # whichever python3 happens to be first on PATH may be older than 3.11 and
    # would build a broken sidecar venv.
    return sys.executable


def run_checked(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def install_sidecar_requirements(py: Path) -> None:
    pip_tmp = SIDECAR_ROOT / ".piptmp"
    shutil.rmtree(pip_tmp, ignore_errors=True)
    pip_tmp.mkdir(parents=True, exist_ok=True)
    pip_env = os.environ.copy()
    pip_env["TMPDIR"] = str(pip_tmp)
    print(f"[jarvis] installing sidecar requirements ({SIDECAR_REQUIREMENTS_INSTALL_NOTE})")
    try:
        run_checked(
            [
                str(py),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "--upgrade",
                "pip",
                "setuptools<82",
                "wheel",
            ],
            env=pip_env,
        )
        gpu_name = detect_nvidia_gpu()
        if gpu_name:
            print(
                f"[jarvis] NVIDIA GPU detected ({gpu_name}) - installing CUDA PyTorch "
                f"({PYTORCH_CUDA_INSTALL_NOTE})",
            )
            run_checked(
                [
                    str(py),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "--index-url",
                    PYTORCH_CUDA_INDEX_URL,
                    "torch",
                ],
                env=pip_env,
            )
        elif cpu_only_requested():
            print("[jarvis] JARVIS_CODE_CPU_ONLY=1 - using CPU PyTorch packages")
        run_checked(
            [str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(SIDECAR_ROOT / "requirements.txt")],
            env=pip_env,
        )
    finally:
        shutil.rmtree(pip_tmp, ignore_errors=True)


def ensure_sidecar_venv() -> Path:
    py = venv_python()
    if py.exists():
        return py

    host_python = find_host_python()
    print(f"[jarvis] creating sidecar venv at {SIDECAR_VENV}")
    run_checked([host_python, "-m", "venv", str(SIDECAR_VENV)])
    if not py.exists():
        raise SystemExit(f"failed to create sidecar venv at {SIDECAR_VENV}")

    install_sidecar_requirements(py)
    return py


def _read_first_run_count() -> int:
    try:
        data = json.loads(FIRST_RUN_COUNT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    try:
        value = int(data.get("count", 0))
    except (AttributeError, TypeError, ValueError):
        return 0
    return max(0, value)


def _write_first_run_count(count: int) -> None:
    payload = {
        "count": count,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        FIRST_RUN_COUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIRST_RUN_COUNT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def apply_first_run_sidecar_window(args: list[str]) -> None:
    if skip_sidecar(args):
        return
    previous_count = _read_first_run_count()
    _write_first_run_count(previous_count + 1)
    if previous_count < AUTO_SIDECAR_WINDOW_RUNS:
        os.environ["JARVIS_SHOW_SIDECAR_WINDOW"] = "1"
        print(
            f"[jarvis] showing sidecar window for first-run visibility "
            f"({previous_count + 1}/{AUTO_SIDECAR_WINDOW_RUNS})",
        )


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
    show_sidecar_window = truthy(os.environ.get("JARVIS_SHOW_SIDECAR_WINDOW"))
    popen_kwargs: dict[str, object] = {
        "cwd": str(SIDECAR_ROOT),
        "env": sidecar_env(),
        "stdin": subprocess.DEVNULL,
        "text": True,
    }
    if show_sidecar_window and is_windows():
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    else:
        popen_kwargs["stdout"] = subprocess.DEVNULL
        popen_kwargs["stderr"] = subprocess.DEVNULL
    if not is_windows():
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen([str(py), "-m", "jarvis_sidecar"], **popen_kwargs)

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
                        "pair_id": os.environ.get("JARVIS_PAIR_ID"),
                        **(
                            {"label": os.environ["JARVIS_WINDOW_LABEL"]}
                            if os.environ.get("JARVIS_WINDOW_LABEL")
                            else {}
                        ),
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
    if IMAGE_EXTENSION_PATH.exists():
        forward.extend(["--extension", str(IMAGE_EXTENSION_PATH)])
    provider_args, remaining_args = resolve_provider_args(args, config_path)
    forward.extend(provider_args)
    forward.extend(remaining_args)
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
                    "pair_id": os.environ.get("JARVIS_PAIR_ID"),
                    "window_label": os.environ.get("JARVIS_WINDOW_LABEL"),
                    "sidecar_runtime": os.environ.get("JARVIS_SIDECAR_RUNTIME"),
                    "config_path": str(config_path),
                    "pi_agent_dir": os.environ.get("JARVIS_CODE_CODING_AGENT_DIR"),
                    "skip_sidecar": skip_sidecar(args),
                    "chat_model_override": os.environ.get("JARVIS_CHAT_MODEL_OVERRIDE"),
                    "enable_extension_discovery": truthy(os.environ.get("JARVIS_ENABLE_EXTENSION_DISCOVERY")),
                    "extension_path": str(EXTENSION_PATH),
                    "face_extension_path": str(FACE_EXTENSION_PATH),
                    "image_extension_path": str(IMAGE_EXTENSION_PATH),
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
        remove_started_sidecar_runtime(proc)
        return
    terminate_process_tree(proc, force=False)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        terminate_process_tree(proc, force=True)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        remove_started_sidecar_runtime(proc)


def terminate_process_tree(proc: subprocess.Popen[str], *, force: bool) -> None:
    if is_windows():
        if force:
            proc.kill()
        else:
            proc.terminate()
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        return
    except OSError:
        if force:
            proc.kill()
        else:
            proc.terminate()


def remove_started_sidecar_runtime(proc: subprocess.Popen[str]) -> None:
    runtime = os.environ.get("JARVIS_SIDECAR_RUNTIME")
    if not runtime:
        return
    path = Path(runtime)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    except (OSError, json.JSONDecodeError):
        return
    try:
        runtime_pid = int(payload.get("pid") or 0)
    except (TypeError, ValueError):
        return
    if runtime_pid != proc.pid:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def main(argv: Iterable[str]) -> int:
    ensure_supported_python()
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
            apply_first_run_sidecar_window(args)
        sidecar_proc = start_sidecar(args)
        return run_agent(args, config_path)
    finally:
        terminate_started_sidecar(sidecar_proc)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
