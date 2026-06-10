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
    api_key_envs: list[str]
    roles: dict[str, str | None]
    chat_usable: bool
    encoder_usable: bool
    reason: str

    @property
    def has_any_llm_credential(self) -> bool:
        return self.oauth_configured or bool(self.api_key_envs)

    @property
    def ready(self) -> bool:
        return self.chat_usable and self.encoder_usable


def pi_agent_dir() -> Path:
    override = os.environ.get("JARVIS_CODE_CODING_AGENT_DIR") or os.environ.get("PI_CODING_AGENT_DIR")
    return Path(override) if override else PI_AGENT_DIR_DEFAULT


def pi_auth_path() -> Path:
    return pi_agent_dir() / "auth.json"


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
    env_name = provider_cfg.get("auth_env") if isinstance(provider_cfg, dict) else None
    if isinstance(env_name, str) and env_name.strip() and not os.environ.get(env_name, "").strip():
        return False, f"{env_name} is not configured"
    if not launchable_for_pi(provider, model):
        return False, f"{provider}/{model} is not registered in pi-agent/models.json"
    return True, "ok"


def inspect_auth_state(*, source_python_auth_path: Path | None = None) -> AuthState:
    catalog = _load_catalog()
    python_auth = source_python_auth_path or python_auth_path()
    oauth_configured = has_python_chatgpt_auth(python_auth)
    if oauth_configured:
        sync_python_chatgpt_auth_to_pi_auth(source_python_auth_path=python_auth)
    api_envs = configured_api_key_envs(catalog)
    roles = current_roles()
    chat_ok, chat_reason = role_usable(roles.get("chat"), catalog=catalog, oauth_configured=oauth_configured)
    encoder_ok, encoder_reason = role_usable(roles.get("encoder"), catalog=catalog, oauth_configured=oauth_configured)
    reason = "ok" if chat_ok and encoder_ok else f"chat: {chat_reason}; encoder: {encoder_reason}"
    return AuthState(
        oauth_configured=oauth_configured,
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
    if state.ready:
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
