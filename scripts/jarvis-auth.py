#!/usr/bin/env python3
"""Terminal auth helper for JARVIS Code.

This intentionally runs before the TUI. If no usable LLM credential/model
configuration exists, the launcher stops and tells the user which terminal
command to run.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
SIDECAR_ROOT = ROOT / "sidecar"
PI_AGENT_DIR_DEFAULT = ROOT / "pi-agent"
OPENAI_CODEX_PROVIDER = "openai-codex"
OPENAI_CODEX_DEFAULT_MODEL = "gpt-5.5"
OPENAI_CODEX_ENCODER_MODEL = "gpt-5.4-mini"
ANTHROPIC_AGENT_SDK_PROVIDER = "anthropic-agent-sdk"
ANTHROPIC_AGENT_SDK_CHAT_MODEL = "claude-opus-4-8"
ANTHROPIC_AGENT_SDK_ENCODER_MODEL = "claude-haiku-4-5-20251001"
ANTHROPIC_AGENT_SDK_PI_PROVIDER_ENTRY: dict[str, Any] = {
    "name": "Anthropic Agent SDK (sidecar-routed, window-init shell only)",
    "baseUrl": "http://localhost:8765/v1",
    "api": "openai-completions",
    "apiKey": "JARVIS_LOCAL_KEYLESS",
    "authHeader": False,
    "headers": {"User-Agent": "jarvis-code/1.01.0 (pi-agent)"},
    "models": [
        {"id": ANTHROPIC_AGENT_SDK_CHAT_MODEL, "name": "Claude Opus 4.8 (Agent SDK)"},
        {"id": ANTHROPIC_AGENT_SDK_ENCODER_MODEL, "name": "Claude Haiku 4.5 (Agent SDK)"},
    ],
}
CLAUDE_CODE_OAUTH_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
NO_CREDENTIAL_EXIT = 42
MODEL_SETTING_EXIT = 43
ADD_CUSTOM_PROVIDER = "__add_custom_provider__"

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

if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))


@dataclass
class AuthState:
    oauth_configured: bool
    claude_agent_sdk_configured: bool
    api_key_envs: list[str]
    roles: dict[str, str | None]
    chat_usable: bool
    encoder_usable: bool
    reason: str

    @property
    def has_any_llm_credential(self) -> bool:
        return self.oauth_configured or self.claude_agent_sdk_configured or bool(self.api_key_envs)

    @property
    def ready(self) -> bool:
        return self.chat_usable and self.encoder_usable


def pi_agent_dir() -> Path:
    override = os.environ.get("JARVIS_CODE_CODING_AGENT_DIR") or os.environ.get("PI_CODING_AGENT_DIR")
    return Path(override) if override else PI_AGENT_DIR_DEFAULT


def pi_auth_path() -> Path:
    return pi_agent_dir() / "auth.json"


def sidecar_venv_python() -> Path:
    if os.name == "nt":
        return SIDECAR_ROOT / ".venv" / "Scripts" / "python.exe"
    return SIDECAR_ROOT / ".venv" / "bin" / "python"


def python_auth_path() -> Path:
    override = os.environ.get("JARVIS_CODE_OAUTH_AUTH_PATH")
    return Path(override) if override else Path.home() / ".jarvis-code" / "auth.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8").lstrip("\ufeff"))
    except (OSError, json.JSONDecodeError):
        return None


def has_python_chatgpt_auth(path: Path | None = None) -> bool:
    path = path or python_auth_path()
    raw = _read_json(path)
    if not isinstance(raw, dict):
        return False
    return all(raw.get(key) for key in ("access_token", "refresh_token", "expires_at_unix"))


def sync_python_chatgpt_auth_to_pi_auth(
    *,
    source_python_auth_path: Path | None = None,
    target_pi_auth_path: Path | None = None,
) -> bool:
    source_path = source_python_auth_path or python_auth_path()
    raw = _read_json(source_path)
    if not isinstance(raw, dict):
        return False
    if not all(raw.get(key) for key in ("access_token", "refresh_token", "expires_at_unix")):
        return False

    target = target_pi_auth_path or pi_auth_path()
    existing = _read_json(target)
    if not isinstance(existing, dict):
        existing = {}
    existing[OPENAI_CODEX_PROVIDER] = {
        "type": "oauth",
        "access": raw["access_token"],
        "refresh": raw["refresh_token"],
        "expires": int(raw["expires_at_unix"]) * 1000,
    }
    if raw.get("account_id"):
        existing[OPENAI_CODEX_PROVIDER]["accountId"] = raw["account_id"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def remove_pi_openai_codex_auth(target_pi_auth_path: Path | None = None) -> None:
    target = target_pi_auth_path or pi_auth_path()
    existing = _read_json(target)
    if not isinstance(existing, dict):
        return
    if OPENAI_CODEX_PROVIDER not in existing:
        return
    existing.pop(OPENAI_CODEX_PROVIDER, None)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_catalog() -> dict[str, Any]:
    from jarvis_sidecar.llm_setting import load_catalog

    return load_catalog()


def _load_credentials_into_env() -> None:
    from jarvis_sidecar.config import ensure_sidecar_config, load_credentials_into_env

    ensure_sidecar_config()
    load_credentials_into_env()


def configured_api_key_envs(catalog: dict[str, Any] | None = None) -> list[str]:
    from jarvis_sidecar.llm_setting import provider_supports_model_setting

    _load_credentials_into_env()
    cat = catalog or _load_catalog()
    envs: list[str] = []
    for cfg in cat.get("providers", {}).values():
        if not isinstance(cfg, dict) or not provider_supports_model_setting(cfg):
            continue
        env_name = cfg.get("auth_env")
        if isinstance(env_name, str) and env_name.strip() and os.environ.get(env_name, "").strip():
            envs.append(env_name)
    return sorted(set(envs))


def current_roles() -> dict[str, str | None]:
    from jarvis_sidecar.llm_setting import current_roles as llm_current_roles

    return llm_current_roles()


def split_role(value: str | None) -> tuple[str, str] | None:
    if not value or "/" not in value:
        return None
    provider, model = value.split("/", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        return None
    return provider, model


def model_registered_in_pi_models(provider: str, model: str) -> bool:
    path = pi_agent_dir() / "models.json"
    raw = _read_json(path)
    if not isinstance(raw, dict):
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


def launchable_for_pi(provider: str, model: str) -> bool:
    return provider in BUILT_IN_MODEL_PROVIDERS or model_registered_in_pi_models(provider, model)


def roles_reference_provider(roles: dict[str, str | None], provider: str) -> bool:
    for value in roles.values():
        role = split_role(value)
        if role and role[0] == provider:
            return True
    return False


def _agent_sdk_auth_available() -> bool:
    """Claude Agent SDK auth: a headless CLAUDE_CODE_OAUTH_TOKEN, or an interactive
    Claude Code login stored at ~/.claude/.credentials.json. 2026-06-20: expired
    access-token-only files are not usable by headless workers."""
    from jlc_agentic.claude_auth import agent_sdk_auth_available

    return agent_sdk_auth_available()


def _inspect_agent_sdk_auth():
    from jlc_agentic.claude_auth import inspect_agent_sdk_auth

    return inspect_agent_sdk_auth()


def role_usable(role_value: str | None, *, catalog: dict[str, Any], oauth_configured: bool) -> tuple[bool, str]:
    role = split_role(role_value)
    if role is None:
        return False, "role is not configured"
    provider, model = role
    if provider == OPENAI_CODEX_PROVIDER:
        if not oauth_configured:
            return False, "OpenAI Codex OAuth is not configured"
        if not launchable_for_pi(provider, model):
            return False, f"{provider}/{model} is not launchable by Pi"
        return True, "ok"

    provider_cfg = catalog.get("providers", {}).get(provider)
    if isinstance(provider_cfg, dict) and provider_cfg.get("auth_kind") == "agent-sdk":
        # Sidecar-routed (Claude Agent SDK): the JLC sidecar drives chat, so this
        # is NOT a Pi-registered model. Gated on the Claude Code CLI login, not on
        # pi-agent/models.json. (2026-06-15)
        if not _agent_sdk_auth_available():
            return False, "Claude Code OAuth token required (run `jarvis claude-login`)"
        return True, "ok"
    env_name = provider_cfg.get("auth_env") if isinstance(provider_cfg, dict) else None
    if isinstance(env_name, str) and env_name.strip() and not os.environ.get(env_name, "").strip():
        return False, f"{env_name} is not configured"
    if not launchable_for_pi(provider, model):
        return False, f"{provider}/{model} is not registered in pi-agent/models.json"
    return True, "ok"


def inspect_auth_state(*, source_python_auth_path: Path | None = None) -> AuthState:
    _load_credentials_into_env()
    catalog = _load_catalog()
    python_auth = source_python_auth_path or python_auth_path()
    oauth_configured = has_python_chatgpt_auth(python_auth)
    if oauth_configured:
        sync_python_chatgpt_auth_to_pi_auth(source_python_auth_path=python_auth)
    claude_agent_sdk_configured = _agent_sdk_auth_available()
    api_envs = configured_api_key_envs(catalog)
    roles = current_roles()
    chat_ok, chat_reason = role_usable(roles.get("chat"), catalog=catalog, oauth_configured=oauth_configured)
    encoder_ok, encoder_reason = role_usable(roles.get("encoder"), catalog=catalog, oauth_configured=oauth_configured)
    reason = "ok" if chat_ok and encoder_ok else f"chat: {chat_reason}; encoder: {encoder_reason}"
    return AuthState(
        oauth_configured=oauth_configured,
        claude_agent_sdk_configured=claude_agent_sdk_configured,
        api_key_envs=api_envs,
        roles=roles,
        chat_usable=chat_ok,
        encoder_usable=encoder_ok,
        reason=reason,
    )


def apply_openai_codex_defaults() -> dict[str, str]:
    from jarvis_sidecar.llm_setting import apply_picks

    return apply_picks(
        (OPENAI_CODEX_PROVIDER, OPENAI_CODEX_DEFAULT_MODEL),
        (OPENAI_CODEX_PROVIDER, OPENAI_CODEX_ENCODER_MODEL),
        catalog=_load_catalog(),
    )


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _merge_anthropic_agent_sdk_pi_provider(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        merged: dict[str, Any] = {}
    else:
        merged = _json_clone(raw)
    providers = merged.get("providers")
    if not isinstance(providers, dict):
        providers = {}
        merged["providers"] = providers
    providers.setdefault(ANTHROPIC_AGENT_SDK_PROVIDER, _json_clone(ANTHROPIC_AGENT_SDK_PI_PROVIDER_ENTRY))
    return merged


def _has_anthropic_agent_sdk_pi_provider(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    providers = raw.get("providers")
    return isinstance(providers, dict) and ANTHROPIC_AGENT_SDK_PROVIDER in providers


def write_anthropic_agent_sdk_pi_provider(existing_raw: Any = None) -> Path:
    path = pi_agent_dir() / "models.json"
    raw = existing_raw if existing_raw is not None else _read_json(path)
    merged = _merge_anthropic_agent_sdk_pi_provider(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def anthropic_agent_sdk_pi_provider_registered() -> bool:
    return _has_anthropic_agent_sdk_pi_provider(_read_json(pi_agent_dir() / "models.json"))


def repair_missing_anthropic_agent_sdk_pi_provider(roles: dict[str, str | None]) -> bool:
    if not roles_reference_provider(roles, ANTHROPIC_AGENT_SDK_PROVIDER):
        return False
    if anthropic_agent_sdk_pi_provider_registered():
        return False
    path = write_anthropic_agent_sdk_pi_provider()
    print(f"[jarvis-auth] registered {ANTHROPIC_AGENT_SDK_PROVIDER} provider in {path} for existing Claude roles.")
    return True


def claude_agent_sdk_importable_in_sidecar_venv(py: Path | None = None) -> bool:
    python = py or sidecar_venv_python()
    if not python.exists():
        return False
    try:
        completed = subprocess.run(
            [str(python), "-c", "import claude_agent_sdk"],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _manual_claude_agent_sdk_install_command(py: Path) -> str:
    return f'"{py}" -m pip install claude-agent-sdk'


def ensure_claude_agent_sdk_installed() -> bool:
    if not _env_flag_enabled("JARVIS_CLAUDE_AGENT_SDK_AUTO_INSTALL", True):
        return False
    py = sidecar_venv_python()
    if claude_agent_sdk_importable_in_sidecar_venv(py):
        return True
    manual = _manual_claude_agent_sdk_install_command(py)
    if not py.exists():
        print(
            f"[jarvis-auth] sidecar venv Python not found at {py}; install later with: {manual}",
            file=sys.stderr,
        )
        return False

    print("[jarvis-auth] installing claude-agent-sdk into the sidecar venv (one-time, ~250MB)...")
    try:
        completed = subprocess.run(
            [str(py), "-m", "pip", "install", "claude-agent-sdk"],
            cwd=str(ROOT),
            check=False,
        )
    except FileNotFoundError:
        print(f"[jarvis-auth] failed to start sidecar venv Python. Run manually: {manual}", file=sys.stderr)
        return False
    if completed.returncode != 0:
        print(
            f"[jarvis-auth] claude-agent-sdk install failed with exit code {completed.returncode}. "
            f"Run manually: {manual}",
            file=sys.stderr,
        )
        return False
    if not claude_agent_sdk_importable_in_sidecar_venv(py):
        print(f"[jarvis-auth] claude-agent-sdk still is not importable. Run manually: {manual}", file=sys.stderr)
        return False
    return True


def apply_anthropic_agent_sdk_defaults() -> dict[str, str]:
    from jarvis_sidecar.llm_setting import apply_picks

    models_path = pi_agent_dir() / "models.json"
    existing_models_json = _read_json(models_path)
    try:
        existing_models_text = models_path.read_text(encoding="utf-8")
    except OSError:
        existing_models_text = None
    paths = apply_picks(
        (ANTHROPIC_AGENT_SDK_PROVIDER, ANTHROPIC_AGENT_SDK_CHAT_MODEL),
        (ANTHROPIC_AGENT_SDK_PROVIDER, ANTHROPIC_AGENT_SDK_ENCODER_MODEL),
        subagent=(ANTHROPIC_AGENT_SDK_PROVIDER, ANTHROPIC_AGENT_SDK_CHAT_MODEL),
        router=(ANTHROPIC_AGENT_SDK_PROVIDER, ANTHROPIC_AGENT_SDK_ENCODER_MODEL),
        catalog=_load_catalog(),
    )
    if _has_anthropic_agent_sdk_pi_provider(existing_models_json) and existing_models_text is not None:
        models_path.write_text(existing_models_text, encoding="utf-8")
    else:
        models_path = write_anthropic_agent_sdk_pi_provider(existing_models_json)
    paths["models_json_path"] = str(models_path)
    ensure_claude_agent_sdk_installed()
    return paths


def print_anthropic_agent_sdk_defaults_configured() -> None:
    print(
        f"[jarvis-auth] configured chat={ANTHROPIC_AGENT_SDK_PROVIDER}/{ANTHROPIC_AGENT_SDK_CHAT_MODEL}, "
        f"encoder={ANTHROPIC_AGENT_SDK_PROVIDER}/{ANTHROPIC_AGENT_SDK_ENCODER_MODEL} "
        "from Claude subscription credentials."
    )


def apply_anthropic_agent_sdk_defaults_if_seed() -> bool:
    state = inspect_auth_state()
    if not state.claude_agent_sdk_configured or not roles_are_untouched_seed(state.roles):
        return False
    apply_anthropic_agent_sdk_defaults()
    print_anthropic_agent_sdk_defaults_configured()
    return True


def print_no_credentials_guidance() -> None:
    print("JARVIS Code needs an LLM credential before starting.")
    print()
    print("Choose one setup path:")
    print()
    print("  GPT OAuth subscription:")
    print("    jarvis gpt-login")
    print("    # if browser callback fails:")
    print("    jarvis gpt-login-device")
    print()
    print("  Claude OAuth subscription:")
    print("    jarvis claude-login")
    print()
    print("  API key:")
    print("    jarvis api-key")
    print("    jarvis model-setting")
    print()
    print("Then run:")
    print("    jarvis")
    print()
    print("Diagnostics still work:")
    print("    jarvis doctor")


def print_model_setting_guidance(state: AuthState) -> None:
    print("JARVIS Code found credentials, but chat/encoder model setup is not ready.")
    print(f"Current chat role:    {state.roles.get('chat') or '<missing>'}")
    print(f"Current encoder role: {state.roles.get('encoder') or '<missing>'}")
    print(f"Reason: {state.reason}")
    print()
    print("Run this in the terminal, then start JARVIS again:")
    print("    jarvis model-setting")


def roles_are_untouched_seed(roles: dict[str, str | None]) -> bool:
    """True while chat/encoder are missing or still the ensure_sidecar_config()
    seed defaults — i.e. the user has never picked models."""
    from jarvis_sidecar.config import DEFAULT_ROLE_CONFIG

    defaults = DEFAULT_ROLE_CONFIG.get("roles", {})
    return all(
        split_role(roles.get(name)) is None or roles.get(name) == defaults.get(name)
        for name in ("chat", "encoder")
    )


def cmd_preflight(_args: argparse.Namespace) -> int:
    if os.environ.get("JARVIS_AUTH_PREFLIGHT") == "0":
        return 0

    state = inspect_auth_state()
    if repair_missing_anthropic_agent_sdk_pi_provider(state.roles):
        state = inspect_auth_state()
    if state.ready:
        if state.claude_agent_sdk_configured and roles_reference_provider(state.roles, ANTHROPIC_AGENT_SDK_PROVIDER):
            ensure_claude_agent_sdk_installed()
        return 0

    # Auto-configure OpenAI Codex defaults ONLY on first run, i.e. while the
    # roles are still the untouched ensure_sidecar_config() seed values. When
    # the user has picked roles and they are merely unusable in this console
    # (e.g. an API-key env var is missing), overwriting them silently destroys
    # the explicit chat/encoder picks and rewrites models.json
    # (2026-06-07 models.json wipe incident) — guide instead, never overwrite.
    if state.oauth_configured and roles_are_untouched_seed(state.roles):
        apply_openai_codex_defaults()
        state = inspect_auth_state()
        if state.ready:
            print(
                f"[jarvis-auth] configured chat={OPENAI_CODEX_PROVIDER}/{OPENAI_CODEX_DEFAULT_MODEL}, "
                f"encoder={OPENAI_CODEX_PROVIDER}/{OPENAI_CODEX_ENCODER_MODEL} from saved GPT OAuth."
            )
            return 0
    elif state.claude_agent_sdk_configured and roles_are_untouched_seed(state.roles):
        apply_anthropic_agent_sdk_defaults()
        state = inspect_auth_state()
        if state.ready:
            print_anthropic_agent_sdk_defaults_configured()
            return 0

    if not state.has_any_llm_credential:
        print_no_credentials_guidance()
        return NO_CREDENTIAL_EXIT

    print_model_setting_guidance(state)
    return MODEL_SETTING_EXIT


def run_login_cli(args: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SIDECAR_ROOT), *(part for part in [env.get("PYTHONPATH")] if part)]
    )
    return subprocess.call([sys.executable, "-m", "jlc_agentic.cli.login", *args], cwd=str(ROOT), env=env)


def cmd_gpt_login(args: argparse.Namespace) -> int:
    login_args = ["login"]
    if args.device_code:
        login_args.append("--device-code")
    login_args.append("chatgpt")
    code = run_login_cli(login_args)
    if code != 0:
        return code
    if not sync_python_chatgpt_auth_to_pi_auth():
        print("GPT OAuth login succeeded, but syncing Pi auth failed.", file=sys.stderr)
        return 1
    apply_openai_codex_defaults()
    print(
        f"JARVIS default chat set to {OPENAI_CODEX_PROVIDER}/{OPENAI_CODEX_DEFAULT_MODEL}, "
        f"encoder set to {OPENAI_CODEX_PROVIDER}/{OPENAI_CODEX_ENCODER_MODEL}."
    )
    return 0


def cmd_gpt_status(_args: argparse.Namespace) -> int:
    code = run_login_cli(["status"])
    if has_python_chatgpt_auth():
        sync_python_chatgpt_auth_to_pi_auth()
    return code


def cmd_gpt_logout(_args: argparse.Namespace) -> int:
    code = run_login_cli(["logout", "chatgpt"])
    remove_pi_openai_codex_auth()
    return code


def bundled_claude_cli() -> str | None:
    """Return the Claude Code CLI bundled by claude-agent-sdk, when present."""
    exe = "claude.exe" if os.name == "nt" else "claude"
    try:
        import claude_agent_sdk  # type: ignore[import-not-found]  # noqa: PLC0415

        bundled = Path(claude_agent_sdk.__file__).resolve().parent / "_bundled" / exe
        if bundled.is_file():
            return str(bundled)
    except Exception:
        pass

    venv_root = ROOT / "sidecar" / ".venv"
    candidates = [
        venv_root / "Lib" / "site-packages" / "claude_agent_sdk" / "_bundled" / exe,
        *sorted(venv_root.glob("lib/python*/site-packages/claude_agent_sdk/_bundled/" + exe)),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def claude_setup_token_command(*, prefer_npx: bool = False) -> list[str] | None:
    if prefer_npx:
        npx = shutil.which("npx")
        if npx:
            return [npx, "-y", "@anthropic-ai/claude-code", "setup-token"]

    claude = shutil.which("claude")
    if claude:
        return [claude, "setup-token"]
    bundled = bundled_claude_cli()
    if bundled:
        return [bundled, "setup-token"]
    if not prefer_npx:
        npx = shutil.which("npx")
        if npx:
            return [npx, "-y", "@anthropic-ai/claude-code", "setup-token"]
    return None


def save_claude_oauth_token(token: str) -> Path:
    from jarvis_sidecar.config import save_credential_env

    path = save_credential_env(CLAUDE_CODE_OAUTH_ENV, token)
    sync_claude_oauth_token_to_user_env(token)
    return path


def _env_flag_enabled(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def sync_claude_oauth_token_to_user_env(token: str) -> bool:
    if not _env_flag_enabled("JARVIS_CLAUDE_LOGIN_SYNC_USER_ENV", True):
        return False
    return set_user_environment_variable(CLAUDE_CODE_OAUTH_ENV, token)


def remove_claude_oauth_token_from_user_env() -> bool:
    if not _env_flag_enabled("JARVIS_CLAUDE_LOGIN_SYNC_USER_ENV", True):
        return False
    return remove_user_environment_variable(CLAUDE_CODE_OAUTH_ENV)


def set_user_environment_variable(name: str, value: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg  # type: ignore[import-not-found]  # noqa: PLC0415

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        _broadcast_windows_environment_change()
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort persistence
        print(f"Warning: could not persist {name} to the Windows user environment: {exc}", file=sys.stderr)
        return False


def remove_user_environment_variable(name: str) -> bool:
    if os.name != "nt":
        return False
    try:
        import winreg  # type: ignore[import-not-found]  # noqa: PLC0415

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            "Environment",
            0,
            winreg.KEY_SET_VALUE,
        ) as key:
            try:
                winreg.DeleteValue(key, name)
            except FileNotFoundError:
                pass
        _broadcast_windows_environment_change()
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort cleanup
        print(f"Warning: could not remove {name} from the Windows user environment: {exc}", file=sys.stderr)
        return False


def _broadcast_windows_environment_change() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes  # noqa: PLC0415

        hwnd_broadcast = 0xFFFF
        wm_settingchange = 0x001A
        smto_abortifhung = 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(
            hwnd_broadcast,
            wm_settingchange,
            0,
            "Environment",
            smto_abortifhung,
            5000,
            None,
        )
    except Exception:
        pass


def extract_claude_oauth_token(text: str) -> str | None:
    match = re.search(r"\bsk-ant-oat[0-9A-Za-z_-]+\b", text)
    return match.group(0) if match else None


def redact_claude_oauth_tokens(text: str) -> str:
    return re.sub(r"\bsk-ant-oat[0-9A-Za-z_-]+\b", "<CLAUDE_CODE_OAUTH_TOKEN>", text)


def run_claude_setup_token_command(command: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
    except FileNotFoundError as exc:
        print(f"Claude setup-token runner was not found: {exc.filename}", file=sys.stderr)
        return 127, ""

    combined = (completed.stdout or "") + (completed.stderr or "")
    sanitized_stdout = redact_claude_oauth_tokens(completed.stdout or "")
    sanitized_stderr = redact_claude_oauth_tokens(completed.stderr or "")
    if sanitized_stdout:
        print(sanitized_stdout, end="" if sanitized_stdout.endswith("\n") else "\n")
    if sanitized_stderr:
        print(sanitized_stderr, end="" if sanitized_stderr.endswith("\n") else "\n", file=sys.stderr)
    return int(completed.returncode), combined


def prompt_and_save_claude_token() -> bool:
    token = getpass.getpass("Paste Claude Code OAuth token (hidden, leave empty to cancel): ").strip()
    if not token:
        return False
    path = save_claude_oauth_token(token)
    print(f"Saved {CLAUDE_CODE_OAUTH_ENV} to {path}.")
    print("Synced it to the user environment for Claude Code CLI where supported.")
    return True


def cmd_claude_login(args: argparse.Namespace) -> int:
    _load_credentials_into_env()
    if args.token:
        path = save_claude_oauth_token(args.token)
        print(f"Saved {CLAUDE_CODE_OAUTH_ENV} to {path}.")
        print("Synced it to the user environment for Claude Code CLI where supported.")
        if not apply_anthropic_agent_sdk_defaults_if_seed():
            print("Next: restart JARVIS, then pick or spawn an anthropic-agent-sdk model.")
        return 0

    status = _inspect_agent_sdk_auth()
    if status.available and not args.refresh:
        print(f"Claude Agent SDK OAuth already available ({status.source}: {status.reason}).")
        if not apply_anthropic_agent_sdk_defaults_if_seed():
            print("Use --refresh to run Claude setup-token anyway.")
        return 0

    if not args.no_setup_token:
        command = claude_setup_token_command(prefer_npx=args.npx)
        if command is None:
            print("Claude Code setup-token runner was not found.", file=sys.stderr)
            print(
                "Claude Code CLI not found. Reinstall jarvis-code, install Claude Code, "
                "or install Node.js for npx.",
                file=sys.stderr,
            )
            return 2
        print("Starting Claude Code setup-token...")
        print("JARVIS will capture and save the token automatically; token text is redacted from this output.")
        code, output = run_claude_setup_token_command(command)
        if code != 0:
            return code
        token = extract_claude_oauth_token(output)
        if token:
            path = save_claude_oauth_token(token)
            print(f"Saved {CLAUDE_CODE_OAUTH_ENV} to {path}.")
            print("Synced it to the user environment for Claude Code CLI where supported.")
            print("Claude Agent SDK OAuth is ready.")
            if not apply_anthropic_agent_sdk_defaults_if_seed():
                print("Next: restart JARVIS, then pick or spawn an anthropic-agent-sdk model.")
            return 0
        _load_credentials_into_env()
        status = _inspect_agent_sdk_auth()
        if status.available:
            print(f"Claude Agent SDK OAuth is ready ({status.source}: {status.reason}).")
            apply_anthropic_agent_sdk_defaults_if_seed()
            return 0

    print("JARVIS still does not see a usable Claude Agent SDK OAuth token.")
    print("Reason:", _inspect_agent_sdk_auth().reason)
    print("Automatic token capture did not find a token in setup-token output.")
    if prompt_and_save_claude_token():
        status = _inspect_agent_sdk_auth()
        if status.available:
            print("Claude Agent SDK OAuth is ready.")
            if not apply_anthropic_agent_sdk_defaults_if_seed():
                print("Next: restart JARVIS, then pick or spawn an anthropic-agent-sdk model.")
            return 0
    print("Claude OAuth token was not saved.", file=sys.stderr)
    return 2


def cmd_claude_status(_args: argparse.Namespace) -> int:
    _load_credentials_into_env()
    status = _inspect_agent_sdk_auth()
    print("Claude Agent SDK OAuth:")
    print(f"  available: {status.available}")
    print(f"  source: {status.source}")
    print(f"  reason: {status.reason}")
    if status.credentials_path:
        print(f"  credentials: {status.credentials_path}")
    if status.has_refresh_token is not None:
        print(f"  refresh token: {'yes' if status.has_refresh_token else 'no'}")
    return 0 if status.available else 1


def cmd_claude_logout(_args: argparse.Namespace) -> int:
    from jarvis_sidecar.config import remove_credential_env

    path = remove_credential_env(CLAUDE_CODE_OAUTH_ENV)
    remove_claude_oauth_token_from_user_env()
    print(f"Removed saved {CLAUDE_CODE_OAUTH_ENV} from {path}.")
    print(f"Removed {CLAUDE_CODE_OAUTH_ENV} from the user environment where supported.")
    print("Claude Code CLI credentials under ~/.claude were left untouched.")
    return 0


def api_key_targets(catalog: dict[str, Any] | None = None) -> list[tuple[str, dict[str, Any]]]:
    cat = catalog or _load_catalog()
    from jarvis_sidecar.llm_setting import load_repo_providers

    repo_provider_ids = set(load_repo_providers())
    targets: list[tuple[str, dict[str, Any]]] = []
    for provider_id, cfg in cat.get("providers", {}).items():
        if cfg.get("enabled") is False or cfg.get("auth_kind") == "oauth":
            continue
        env_name = cfg.get("auth_env")
        if not isinstance(env_name, str) or not env_name.strip():
            continue
        entry = dict(cfg)
        source = "bundled" if provider_id in repo_provider_ids else "custom"
        entry["source"] = source
        entry["custom"] = source == "custom"
        targets.append((provider_id, entry))
    return targets


def select_api_key_target(provider: str | None) -> tuple[str, dict[str, Any]] | None:
    targets = api_key_targets()
    if provider:
        provider = provider.strip()
        for provider_id, cfg in targets:
            env_name = str(cfg.get("auth_env", ""))
            if provider in {provider_id, env_name}:
                return provider_id, cfg
        print(f"Unknown API-key provider/env: {provider}", file=sys.stderr)
        return None

    print("API key setup — select a provider:")
    print()
    for idx, (provider_id, cfg) in enumerate(targets, start=1):
        label = cfg.get("label", provider_id)
        print(f"  {idx}. {api_key_target_label(provider_id, cfg)}")
    add_idx = len(targets) + 1
    print("  " + "─" * 29)
    print(f"  {add_idx}. [+] Add custom provider...")
    print()
    raw = input("Choose provider number: ").strip()
    try:
        choice = int(raw)
    except ValueError:
        print("Invalid provider number.", file=sys.stderr)
        return None
    if choice == add_idx:
        return ADD_CUSTOM_PROVIDER, {}
    if choice < 1 or choice > len(targets):
        print("Provider number out of range.", file=sys.stderr)
        return None
    return targets[choice - 1]


def api_key_target_label(provider_id: str, cfg: dict[str, Any]) -> str:
    is_custom = bool(cfg.get("custom") or cfg.get("source") == "custom")
    configured = bool(os.environ.get(str(cfg.get("auth_env", "")), "").strip())
    marker = "[*]" if is_custom and configured else "[v]" if configured else "[ ]"
    custom = "  (custom)" if is_custom else ""
    status = "key set" if configured else "no key"
    return f"{marker} {cfg.get('label', provider_id)}{custom}   {status}"


def cmd_api_key(args: argparse.Namespace) -> int:
    _load_credentials_into_env()
    target = select_api_key_target(args.provider)
    if target is None:
        return 2
    provider_id, cfg = target
    if provider_id == ADD_CUSTOM_PROVIDER:
        return cmd_api_key_add_custom(args)
    if cfg.get("custom") and args.provider is None:
        action = input("Choose action: [1] Change key  [2] Remove: ").strip()
        if action == "2":
            return cmd_api_key_remove_custom(provider_id, cfg)
        if action not in {"", "1"}:
            print("Invalid action.", file=sys.stderr)
            return 2
    return save_api_key_for_target(provider_id, cfg, args.value)


def save_api_key_for_target(provider_id: str, cfg: dict[str, Any], value_arg: str | None) -> int:
    env_name = str(cfg["auth_env"])
    label = str(cfg.get("label", provider_id))
    value = value_arg or getpass.getpass(f"Enter {label} API key ({env_name}): ").strip()
    if not value:
        print("API key was empty; nothing saved.", file=sys.stderr)
        return 2

    from jarvis_sidecar.config import save_credential_env
    from jarvis_sidecar.llm_setting import fetch_models, provider_supports_model_setting

    path = save_credential_env(env_name, value)
    if not provider_supports_model_setting(cfg):
        print(f"Saved {env_name} to {path}. Live model validation skipped for this provider.")
    elif models := fetch_models(provider_id, cfg):
        print(f"Saved {env_name} to {path}. Validation ok ({len(models)} models).")
    else:
        print(f"Saved {env_name} to {path}. Live model validation did not return models.")
    print()
    print("Next:")
    print("    jarvis model-setting")
    return 0


def cmd_api_key_add_custom(args: argparse.Namespace) -> int:
    from jarvis_sidecar.llm_setting import (
        custom_provider_auth_env,
        custom_provider_id_from_label,
        find_provider_duplicate,
        upsert_user_provider,
    )

    base_url = input("Custom provider base URL: ").strip().rstrip("/")
    if not base_url:
        print("Base URL was empty; nothing saved.", file=sys.stderr)
        return 2
    label = input("Custom provider display name: ").strip()
    if not label:
        print("Provider label was empty; nothing saved.", file=sys.stderr)
        return 2
    provider_id = custom_provider_id_from_label(label)
    duplicate = find_provider_duplicate(provider_id, base_url)
    if duplicate is not None:
        existing_id, existing_cfg = duplicate
        answer = input("already exists — change its key? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            return 0
        return save_api_key_for_target(existing_id, existing_cfg, args.value)

    env_name = custom_provider_auth_env(provider_id)
    cfg = {
        "label": label,
        "auth_env": env_name,
        "base_url": base_url,
        "api_format": "openai-completions",
        "models_endpoint": "/models",
    }
    value = args.value or getpass.getpass(f"Enter {label} API key: ").strip()
    if not value:
        print("API key was empty; nothing saved.", file=sys.stderr)
        return 2
    try:
        upsert_user_provider(provider_id, cfg)
    except (OSError, ValueError) as exc:
        print(f"Custom provider save failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return save_api_key_for_target(provider_id, cfg, value)


def cmd_api_key_remove_custom(provider_id: str, cfg: dict[str, Any]) -> int:
    answer = input(f"Remove {cfg.get('label', provider_id)} and its saved API key? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        return 0
    from jarvis_sidecar.config import remove_credential_env
    from jarvis_sidecar.llm_setting import remove_user_provider

    try:
        _path, removed = remove_user_provider(provider_id)
        env_name = removed.get("auth_env") if isinstance(removed, dict) else cfg.get("auth_env")
        if isinstance(env_name, str) and env_name.strip():
            remove_credential_env(env_name)
    except (OSError, ValueError) as exc:
        print(f"Custom provider remove failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"Removed {cfg.get('label', provider_id)}. Pick models with /model-setting.")
    return 0


def cmd_model_setting(args: argparse.Namespace) -> int:
    _load_credentials_into_env()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SIDECAR_ROOT), *(part for part in [env.get("PYTHONPATH")] if part)]
    )
    return subprocess.call([sys.executable, str(ROOT / "scripts" / "llmsetting.py"), *args.args], cwd=str(ROOT), env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis auth")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.set_defaults(func=cmd_preflight)

    gpt_login = sub.add_parser("gpt-login")
    gpt_login.add_argument("--device-code", action="store_true")
    gpt_login.set_defaults(func=cmd_gpt_login)

    gpt_login_device = sub.add_parser("gpt-login-device")
    gpt_login_device.set_defaults(func=lambda args: cmd_gpt_login(argparse.Namespace(device_code=True)))

    gpt_status = sub.add_parser("gpt-auth-status")
    gpt_status.set_defaults(func=cmd_gpt_status)

    gpt_logout = sub.add_parser("gpt-logout")
    gpt_logout.set_defaults(func=cmd_gpt_logout)

    claude_login = sub.add_parser("claude-login")
    claude_login.add_argument("--token", help=f"save an existing {CLAUDE_CODE_OAUTH_ENV} without running setup-token")
    claude_login.add_argument("--refresh", action="store_true", help="run setup-token even if Claude auth already looks usable")
    claude_login.add_argument("--no-setup-token", action="store_true", help="skip running setup-token and prompt for a token")
    claude_login.add_argument("--npx", action="store_true", help="prefer npx @anthropic-ai/claude-code over an installed claude command")
    claude_login.set_defaults(func=cmd_claude_login)

    anthropic_login = sub.add_parser("anthropic-login")
    anthropic_login.add_argument("--token", help=f"save an existing {CLAUDE_CODE_OAUTH_ENV} without running setup-token")
    anthropic_login.add_argument("--refresh", action="store_true", help="run setup-token even if Claude auth already looks usable")
    anthropic_login.add_argument("--no-setup-token", action="store_true", help="skip running setup-token and prompt for a token")
    anthropic_login.add_argument("--npx", action="store_true", help="prefer npx @anthropic-ai/claude-code over an installed claude command")
    anthropic_login.set_defaults(func=cmd_claude_login)

    claude_status = sub.add_parser("claude-auth-status")
    claude_status.set_defaults(func=cmd_claude_status)

    claude_logout = sub.add_parser("claude-logout")
    claude_logout.set_defaults(func=cmd_claude_logout)

    api_key = sub.add_parser("api-key")
    api_key.add_argument("provider", nargs="?", help="provider id or env name, e.g. openai or OPENAI_API_KEY")
    api_key.add_argument("--value", help="API key value. If omitted, a hidden prompt is used.")
    api_key.set_defaults(func=cmd_api_key)

    model_setting = sub.add_parser("model-setting")
    model_setting.add_argument("args", nargs=argparse.REMAINDER)
    model_setting.set_defaults(func=cmd_model_setting)

    status = sub.add_parser("status")
    status.set_defaults(func=cmd_preflight)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
