from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from jlc_agentic.key_pool import KeyPool
from jlc_agentic.provider_config import load_provider_config, save_default_template
from jlc_agentic.provider_router import ProviderRouter
from jlc_agentic.providers import load_config as _load_role_config

DEFAULT_PROVIDERS_CONFIG = Path("~/.jarvis-code/providers.yaml").expanduser()


def print_roles_summary(config_path: str | None = None) -> None:
    """Print the resolved chat/subagent/encoder role mapping to stderr so it
    appears alongside aider's startup banner. Helps Jun verify at a glance
    which provider/model each role maps to without grep-ing config.yaml."""
    try:
        cfg = _load_role_config(config_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[jarvis] role config load failed: {exc}", file=sys.stderr)
        return
    roles = cfg.get("roles") or {}
    if not roles:
        return

    def _fmt(entry: Any) -> str:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            provider = entry.get("provider", "?")
            model = entry.get("model", "?")
            return f"{provider}/{model}"
        return str(entry)

    print("[jarvis] roles:", file=sys.stderr)
    for role in ("chat", "subagent", "encoder"):
        entry = roles.get(role)
        if entry is None:
            print(f"[jarvis]   {role:<8} : (unset)", file=sys.stderr)
        else:
            print(f"[jarvis]   {role:<8} : {_fmt(entry)}", file=sys.stderr)


def init_provider_router(args: Any) -> ProviderRouter | None:
    path = Path(
        getattr(args, "providers_config", None) or DEFAULT_PROVIDERS_CONFIG
    ).expanduser()
    if not path.exists():
        save_default_template(path)
        print(
            "[jarvis] providers.yaml template created at "
            f"{path}. Fill in API keys to enable ProviderRouter.",
            file=sys.stderr,
        )
        return None

    try:
        config = load_provider_config(path)
        key_pool = KeyPool(_provider_keys(config))
        return ProviderRouter(config, key_pool)
    except Exception as exc:
        print(f"[jarvis] provider router init failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _provider_keys(config: dict[str, Any]) -> dict[str, list[str]]:
    providers = config.get("providers") or {}
    keys_by_provider: dict[str, list[str]] = {}
    for provider_name, provider in providers.items():
        raw_keys = provider.get("api_keys") or []
        keys_by_provider[provider_name] = [
            str(key)
            for key in raw_keys
            if key is not None and str(key) != ""
        ]
    return keys_by_provider
