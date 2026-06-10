from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

SUPPORTED_OAUTH_PROVIDERS = frozenset({"chatgpt"})
DEFAULT_OAUTH_TOKEN_PATH = "~/.jarvis-code/auth.json"


class ValidationError(ValueError):
    pass


DEFAULT_TEMPLATE = """# ~/.jarvis-code/providers.yaml
defaults:
  primary: claude-sonnet-4-6
  fallback: [gpt-5.4-mini, deepseek-chat]

providers:
  anthropic:
    api_keys: ["${ANTHROPIC_API_KEY}", "${ANTHROPIC_API_KEY_2}"]
    models:
      claude-sonnet-4-6:
        litellm_id: "anthropic/claude-sonnet-4-6"
        cost_in_per_1m: 3.0
        cost_out_per_1m: 15.0
        tier: quality

  openai:
    api_keys: ["${OPENAI_API_KEY}"]
    models:
      gpt-5.4-mini:
        litellm_id: "openai/gpt-5.4-mini"
        cost_in_per_1m: 0.25
        cost_out_per_1m: 2.0
        tier: cheap
      gpt-5.4:
        litellm_id: "openai/gpt-5.4"
        cost_in_per_1m: 1.25
        cost_out_per_1m: 10.0
        tier: quality

  deepseek:
    api_keys: ["${DEEPSEEK_API_KEY}"]
    models:
      deepseek-chat:
        litellm_id: "deepseek/deepseek-chat"
        cost_in_per_1m: 0.14
        cost_out_per_1m: 0.28
        tier: cheap

  openrouter:
    api_keys: ["${OPENROUTER_API_KEY}"]
    models:
      kimi-k2.5:
        litellm_id: "openrouter/moonshotai/kimi-k2.5"
        cost_in_per_1m: 0.6
        cost_out_per_1m: 2.5
        tier: balanced
      glm-5:
        litellm_id: "openrouter/zai-org/glm-5"
        cost_in_per_1m: 0.5
        cost_out_per_1m: 1.5
        tier: balanced

routing:
  rules:
    - when: { tier: cheap }
      use: [deepseek-chat, gpt-5.4-mini]
    - when: { tier: quality }
      use: [claude-sonnet-4-6, gpt-5.4]
"""


def load_provider_config(path: Path | str) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValidationError("providers.yaml must contain a mapping")

    config = _substitute_env(raw)
    _validate_and_normalize(config)
    return config


def save_default_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(DEFAULT_TEMPLATE, encoding="utf-8")


def _substitute_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _substitute_env(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_substitute_env(child) for child in value]
    if isinstance(value, str):
        return _ENV_RE.sub(_replace_env, value)
    return value


def _replace_env(match: re.Match[str]) -> str:
    name = match.group(1)
    if name not in os.environ:
        log.warning("Environment variable %s is not set; using empty string", name)
        return ""
    return os.environ[name]


def _validate_and_normalize(config: dict[str, Any]) -> None:
    defaults = config.get("defaults")
    if not isinstance(defaults, dict) or not defaults.get("primary"):
        raise ValidationError("defaults.primary is required")

    providers = config.get("providers")
    if not isinstance(providers, dict):
        raise ValidationError("providers must be a mapping")

    for provider_name, provider in providers.items():
        if not isinstance(provider, dict):
            raise ValidationError(f"providers.{provider_name} must be a mapping")

        # Adapter-based providers (e.g. `adapter: anthropic-agent-sdk`) are resolved
        # directly by providers.get_llm and never flow through the litellm/OAuth
        # router, so they carry no litellm_id / api_keys / oauth_provider. Exempt
        # them from the router's per-model validation. (2026-06-15)
        if provider.get("adapter"):
            models = provider.get("models")
            if models is not None and not isinstance(models, dict):
                raise ValidationError(f"providers.{provider_name}.models must be a mapping")
            continue

        oauth_provider = provider.get("oauth_provider")
        api_keys = provider.get("api_keys")
        if oauth_provider is not None:
            if api_keys is not None:
                raise ValidationError(
                    f"providers.{provider_name}: oauth_provider and api_keys are mutually exclusive"
                )
            if not isinstance(oauth_provider, str) or oauth_provider not in SUPPORTED_OAUTH_PROVIDERS:
                raise ValidationError(
                    f"providers.{provider_name}.oauth_provider={oauth_provider!r} not supported. "
                    f"Supported: {sorted(SUPPORTED_OAUTH_PROVIDERS)}"
                )
            if not provider.get("api_base"):
                raise ValidationError(
                    f"providers.{provider_name}: oauth_provider requires api_base"
                )
            provider.setdefault("oauth_token_path", DEFAULT_OAUTH_TOKEN_PATH)

        models = provider.get("models")
        if not isinstance(models, dict):
            raise ValidationError(f"providers.{provider_name}.models must be a mapping")
        if oauth_provider is not None:
            for alias, model in models.items():
                if isinstance(model, dict) and model.get("api_base"):
                    raise ValidationError(
                        f"providers.{provider_name}.models.{alias}: api_base override is "
                        f"not allowed under oauth_provider (token is bound to provider api_base)"
                    )
        for alias, model in models.items():
            if not isinstance(model, dict):
                raise ValidationError(f"providers.{provider_name}.models.{alias} must be a mapping")
            if not model.get("litellm_id"):
                raise ValidationError(
                    f"providers.{provider_name}.models.{alias}.litellm_id is required"
                )
            model.setdefault("cost_in_per_1m", 0.0)
            model.setdefault("cost_out_per_1m", 0.0)
