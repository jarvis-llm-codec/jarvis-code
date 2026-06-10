"""Shared core for the JARVIS Code LLM selector.

Both scripts/llmsetting.py (standalone arrow-key UI) and the sidecar's
/llmsetting/* endpoints (Pi /model-setting slash command) import from here so
the catalog, fetch behavior, and write paths stay consistent.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "scripts" / "llm_catalog.yaml"
FETCH_TIMEOUT_S = 5
KNOWN_API_FORMATS = {
    "openai-completions",
    "anthropic",
    "google-generative-ai",
    "openai-codex-responses",
}


def _config_path() -> Path:
    from .config import config_path

    return config_path()


def user_catalog_path() -> Path:
    return _config_path().with_name("llm_catalog.user.yaml")


def _models_json_path() -> Path:
    override = os.environ.get("JARVIS_CODE_CODING_AGENT_DIR") or os.environ.get("PI_CODING_AGENT_DIR")
    base = Path(override) if override else REPO_ROOT / "pi-agent"
    return base / "models.json"


def load_catalog() -> dict[str, Any]:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(catalog, dict):
        catalog = {}
    catalog.setdefault("providers", {})
    if not isinstance(catalog["providers"], dict):
        catalog["providers"] = {}

    overlay_path = user_catalog_path()
    user_providers = _load_valid_user_providers(catalog["providers"], overlay_path)
    if user_providers:
        providers = dict(catalog["providers"])
        providers.update(user_providers)
        catalog["providers"] = providers
    return catalog


def catalog_overlay_summary() -> dict[str, Any]:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    repo_providers = catalog.get("providers") if isinstance(catalog, dict) else {}
    if not isinstance(repo_providers, dict):
        repo_providers = {}
    overlay_path = user_catalog_path()
    return {
        "repo_count": len(repo_providers),
        "user_count": len(_load_valid_user_providers(repo_providers, overlay_path, warn=False)),
        "user_path": str(overlay_path.expanduser()),
        "user_exists": overlay_path.exists(),
    }


def _load_valid_user_providers(
    repo_providers: dict[str, Any],
    overlay_path: Path,
    *,
    warn: bool = True,
) -> dict[str, Any]:
    if not overlay_path.exists():
        return {}
    try:
        raw = yaml.safe_load(overlay_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        if warn:
            _catalog_warning(f"failed to read {overlay_path}: {exc}; using bundled catalog only")
        return {}
    if not isinstance(raw, dict):
        if warn:
            _catalog_warning(f"{overlay_path} must be a mapping with a providers section; ignoring overlay")
        return {}
    user_providers = raw.get("providers")
    if not isinstance(user_providers, dict):
        if warn:
            _catalog_warning(f"{overlay_path} has no providers mapping; ignoring overlay")
        return {}

    merged: dict[str, Any] = {}
    for provider_id, user_cfg in user_providers.items():
        if not isinstance(provider_id, str) or not isinstance(user_cfg, dict):
            if warn:
                _catalog_warning(f"skipping invalid user catalog provider {provider_id!r}")
            continue
        base_cfg = repo_providers.get(provider_id)
        if not isinstance(base_cfg, dict):
            base_cfg = {}
        cfg = _deep_merge(base_cfg, user_cfg)
        base_url = str(cfg.get("base_url") or "").strip()
        api_format = str(cfg.get("api_format") or "").strip()
        if not base_url or not api_format:
            if warn:
                _catalog_warning(f"skipping user catalog provider {provider_id!r}: base_url and api_format are required")
            continue
        if api_format not in KNOWN_API_FORMATS:
            if warn:
                _catalog_warning(f"skipping user catalog provider {provider_id!r}: unknown api_format {api_format!r}")
            continue
        merged[provider_id] = cfg
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _catalog_warning(message: str) -> None:
    print(f"[jarvis-llm-catalog] {message}", file=sys.stderr)


def fetch_models(provider_id: str, cfg: dict[str, Any]) -> list[str] | None:
    """Return list of model IDs, or None if the provider is unreachable.

    Tries `models_endpoint` first when an API key is set; falls back to
    `models_static`. Returns None only when there is no key AND no static
    list (i.e. the user cannot reach the provider at all).
    """
    if cfg.get("auth_kind") == "oauth":
        return _fetch_oauth_models(provider_id, cfg)

    if not cfg.get("auth_env"):
        return list(cfg.get("models_static") or []) or None
    api_key = os.environ.get(cfg["auth_env"], "").strip()
    if not api_key:
        return None

    if cfg.get("models_endpoint"):
        url = cfg["base_url"].rstrip("/") + cfg["models_endpoint"]
        headers = {"Authorization": f"Bearer {api_key}"}
        if cfg.get("api_format") == "google-generative-ai":
            url = f"{url}?key={api_key}"
            headers = {}
        elif cfg.get("api_format") == "anthropic":
            headers = {
                "x-api-key": api_key,
                "anthropic-version": cfg.get("anthropic_version", "2023-06-01"),
            }
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and isinstance(data.get("data"), list):
                return [m["id"] for m in data["data"] if isinstance(m, dict) and "id" in m]
            if isinstance(data, dict) and isinstance(data.get("models"), list):
                out: list[str] = []
                for m in data["models"]:
                    name = m.get("name", "") if isinstance(m, dict) else ""
                    if name:
                        out.append(name.split("/")[-1])
                return out
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass

    static = cfg.get("models_static")
    if static:
        return list(static)
    return None


def _fetch_oauth_models(provider_id: str, cfg: dict[str, Any]) -> list[str] | None:
    if provider_id != "openai-codex":
        return None

    try:
        from jlc_agentic.codex_responses_adapter import build_codex_user_agent
        from jlc_agentic.openai_oauth import TokenManager

        try:
            from jlc_agentic import __version__ as version
        except Exception:  # noqa: BLE001
            version = "0.0.0"

        mgr = TokenManager()
        access_token = mgr.get_access_token()
        account_id = mgr.get_account_id() or ""
    except Exception:  # noqa: BLE001
        return None

    endpoint = cfg.get("models_endpoint") or "/models"
    url = _with_query_param(
        cfg["base_url"].rstrip("/") + endpoint,
        "client_version",
        version,
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "ChatGPT-Account-Id": account_id,
            "originator": "jarvis-code",
            "User-Agent": build_codex_user_agent(version),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError):
        return None

    models = _extract_model_ids(data)
    return models or None


def _extract_model_ids(data: Any) -> list[str]:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [
            str(m["id"])
            for m in data["data"]
            if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]
        ]
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        out: list[str] = []
        for m in data["models"]:
            if isinstance(m, str) and m:
                out.append(m)
                continue
            name = ""
            if isinstance(m, dict):
                name = m.get("id") or m.get("slug") or m.get("name") or ""
            if name:
                out.append(str(name).split("/")[-1])
        return out
    if isinstance(data, list):
        return [m for m in data if isinstance(m, str) and m]
    return []


def _with_query_param(url: str, key: str, value: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = [(k, v) for k, v in query if k != key]
    query.append((key, value))
    return urllib.parse.urlunsplit(
        parts._replace(query=urllib.parse.urlencode(query))
    )


def fetch_all(catalog: dict[str, Any]) -> dict[str, list[str] | None]:
    """Fetch model lists for every provider in the catalog. Disabled providers
    are reported as None so callers can treat them uniformly with missing-key
    providers."""
    out: dict[str, list[str] | None] = {}
    for pid, cfg in catalog["providers"].items():
        if cfg.get("enabled") is False:
            out[pid] = None
            continue
        out[pid] = fetch_models(pid, cfg)
    return out


def current_roles() -> dict[str, str | None]:
    """Read roles from data/config.yaml. Missing values come back as None."""
    path = _config_path()
    if not path.exists():
        return {"chat": None, "subagent": None, "encoder": None}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"chat": None, "subagent": None, "encoder": None}
    roles = raw.get("roles") or {}
    return {
        "chat": roles.get("chat"),
        "subagent": roles.get("subagent"),
        "encoder": roles.get("encoder"),
    }


def apply_picks(
    chat: tuple[str, str],
    encoder: tuple[str, str],
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write chat + encoder picks to data/config.yaml AND register the matching
    provider/model entries in pi-agent/models.json. Returns the resolved file
    paths so callers can surface them to the user."""
    cat = catalog or load_catalog()
    cfg_path = _write_config(chat, encoder)
    providers_path = _write_providers_yaml(cat, [chat, encoder], primary=chat)
    models_path = _write_models_json(cat, [chat, encoder])
    return {
        "config_path": str(cfg_path),
        "providers_path": str(providers_path),
        "models_json_path": str(models_path),
    }


def _write_config(chat: tuple[str, str], encoder: tuple[str, str]) -> Path:
    path = _config_path()
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raw = {}
    roles = raw.setdefault("roles", {})
    roles["chat"] = f"{chat[0]}/{chat[1]}"
    roles["subagent"] = f"{chat[0]}/{chat[1]}"
    roles["encoder"] = f"{encoder[0]}/{encoder[1]}"
    raw.pop("encoder", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _providers_path() -> Path:
    return _config_path().with_name("providers.yaml")


def _alias_for(provider_id: str, model_id: str) -> str:
    if provider_id == "openai-codex":
        return f"openai-codex-{model_id}"
    return model_id


def _litellm_id(provider_id: str, model_id: str, cfg: dict[str, Any]) -> str:
    api_format = cfg.get("api_format", "openai-completions")
    if provider_id == "openai-codex":
        return f"openai/{model_id}"
    if api_format == "anthropic":
        return f"anthropic/{model_id}"
    if api_format == "google-generative-ai":
        return f"gemini/{model_id}"
    return f"openai/{model_id}"


def _tier_for_role(provider_id: str, model_id: str, primary: tuple[str, str]) -> str:
    if (provider_id, model_id) == primary:
        return "quality"
    return "cheap"


def _write_providers_yaml(
    catalog: dict[str, Any],
    picks: list[tuple[str, str]],
    *,
    primary: tuple[str, str],
) -> Path:
    """Write only the selected providers/models.

    /model-setting is authoritative for active JARVIS roles. Keeping unselected
    API-key providers in providers.yaml lets ProviderRouter fall back into stale
    paths, so this file is intentionally pruned to the chosen chat/encoder
    entries.
    """
    path = _providers_path()
    providers: dict[str, Any] = {}
    seen: set[tuple[str, str]] = set()
    for provider_id, model_id in picks:
        if (provider_id, model_id) in seen:
            continue
        seen.add((provider_id, model_id))
        cat_cfg = catalog["providers"][provider_id]
        provider = providers.setdefault(provider_id, {"models": {}})
        if cat_cfg.get("auth_kind") == "oauth":
            provider["oauth_provider"] = "chatgpt"
            provider["oauth_token_path"] = "~/.jarvis-code/auth.json"
            provider["api_base"] = cat_cfg["base_url"]
        else:
            auth_env = cat_cfg.get("auth_env")
            provider["api_keys"] = [f"${{{auth_env}}}"] if auth_env else []
            if cat_cfg.get("base_url"):
                provider["api_base"] = cat_cfg["base_url"]

        provider["models"][_alias_for(provider_id, model_id)] = {
            "litellm_id": _litellm_id(provider_id, model_id, cat_cfg),
            "cost_in_per_1m": 0.0,
            "cost_out_per_1m": 0.0,
            "tier": _tier_for_role(provider_id, model_id, primary),
        }

    raw = {
        "managed_by": "jarvis-model-setting",
        "defaults": {
            "primary": _alias_for(primary[0], primary[1]),
            "fallback": [],
        },
        "providers": providers,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _write_models_json(catalog: dict[str, Any], picks: list[tuple[str, str]]) -> Path:
    path = _models_json_path()
    # Merge into the existing registry instead of rebuilding from scratch.
    # Rebuilding silently destroyed every curated provider/model entry that
    # was not part of the current picks (2026-06-07 models.json wipe incident).
    raw: dict[str, Any] = {"providers": {}}
    if path.exists():
        try:
            existing_raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            existing_raw = None
        if isinstance(existing_raw, dict) and isinstance(existing_raw.get("providers"), dict):
            raw = existing_raw
    providers = raw["providers"]

    seen: set[tuple[str, str]] = set()
    for provider_id, model_id in picks:
        if (provider_id, model_id) in seen:
            continue
        seen.add((provider_id, model_id))
        if provider_id == "openai-codex":
            providers.pop(provider_id, None)
            continue
        cat_cfg = catalog["providers"][provider_id]
        provider_block = providers.setdefault(provider_id, {})
        provider_block["name"] = cat_cfg.get("label", provider_id)
        provider_block["baseUrl"] = cat_cfg["base_url"]
        provider_block["api"] = cat_cfg.get("api_format", "openai-completions")
        if cat_cfg.get("auth_kind") == "oauth":
            provider_block["apiKey"] = None
            provider_block["authHeader"] = False
        else:
            provider_block["apiKey"] = cat_cfg.get("auth_env")
            provider_block["authHeader"] = True
        models_list = provider_block.setdefault("models", [])
        existing = next((m for m in models_list if m.get("id") == model_id), None)
        reasoning_meta = _reasoning_metadata(provider_id, model_id)
        if existing is not None:
            if reasoning_meta is not None:
                existing["reasoning"] = True
                existing["thinkingLevelMap"] = reasoning_meta
            continue
        if existing is None:
            entry: dict[str, Any] = {
                "id": model_id,
                "name": f"{cat_cfg.get('label', provider_id)} {model_id}",
                "reasoning": reasoning_meta is not None,
                "input": ["text"],
                "contextWindow": 131072,
                "maxTokens": 16384,
                "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            }
            if reasoning_meta is not None:
                entry["thinkingLevelMap"] = reasoning_meta
            models_list.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _reasoning_metadata(provider_id: str, model_id: str) -> dict[str, str] | None:
    """Return default Pi thinking metadata for selector-registered models.

    Provider /models endpoints usually expose only IDs, not reliable reasoning
    capability metadata. Register fetched models as reasoning-capable so
    deepdive can request the maximum effort; providers that do not support it
    should be handled by runtime warnings/errors instead of hiding the option.
    """
    _ = (provider_id, model_id)
    return {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
    }
