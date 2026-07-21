"""Shared core for the JARVIS Code LLM selector.

Both scripts/llmsetting.py (standalone arrow-key UI) and the sidecar's
/llmsetting/* endpoints (Pi /model-setting slash command) import from here so
the catalog, fetch behavior, and write paths stay consistent.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from jlc_agentic.user_agent import (
    JARVIS_CODE_VERSION,
    with_jarvis_user_agent,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "scripts" / "llm_catalog.yaml"
FETCH_TIMEOUT_S = 3
MODEL_CATALOG_CACHE_TTL_S = 3600
# Serializes the read-modify-write of the shared model-catalog cache file so the
# parallel fetch_all_detailed() fan-out cannot lose a provider's update. (2026-06-15)
_CACHE_FILE_LOCK = threading.Lock()
KNOWN_API_FORMATS = {
    "openai-completions",
    "anthropic",
    "google-generative-ai",
    "openai-codex-responses",
}
MODEL_SETTING_ROLES = {"chat", "subagent", "router", "encoder", "llm"}
BUILT_IN_PI_PROVIDERS = frozenset({
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
})
KEYLESS_LOCAL_PRESET_PROVIDERS = frozenset({"ollama", "lmstudio", "llamacpp"})
# Pi's current models.json loader/client path requires a non-empty apiKey for
# non-built-in custom providers. This is an internal placeholder, not a user
# credential; authHeader stays false so the registry adds no extra bearer header.
PI_KEYLESS_LOCAL_API_KEY = "JARVIS_LOCAL_KEYLESS"
AUTH_SIBLING_REDIRECT_PROVIDERS = frozenset({"anthropic", "openai"})


@dataclass(frozen=True)
class ModelCatalogResult:
    models: list[str] | None
    source: str
    cache_stale: bool = False
    warning: str | None = None


def _config_path() -> Path:
    from .config import config_path

    return config_path()


def user_catalog_path() -> Path:
    return _config_path().with_name("llm_catalog.user.yaml")


def custom_provider_id_from_label(label: str) -> str:
    provider_id = re.sub(r"[^a-z0-9]+", "-", label.strip().lower()).strip("-")
    provider_id = re.sub(r"-+", "-", provider_id)
    if not provider_id:
        raise ValueError("label must contain at least one letter or digit")
    return provider_id


def custom_provider_auth_env(provider_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", provider_id).strip("_").upper()
    if not token:
        raise ValueError("provider_id must contain at least one letter or digit")
    return f"{token}_API_KEY"


def provider_roles(cfg: dict[str, Any]) -> set[str]:
    raw = cfg.get("roles")
    if raw is None:
        return set(MODEL_SETTING_ROLES)
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return set(MODEL_SETTING_ROLES)
    return {str(value).strip().lower() for value in values if str(value).strip()}


def provider_supports_model_setting(cfg: dict[str, Any]) -> bool:
    return bool(provider_roles(cfg) & MODEL_SETTING_ROLES)


def _models_json_path() -> Path:
    override = os.environ.get("JARVIS_CODE_CODING_AGENT_DIR") or os.environ.get("PI_CODING_AGENT_DIR")
    base = Path(override) if override else REPO_ROOT / "pi-agent"
    return base / "models.json"


def _canonical_model_id_for_provider(provider_id: str, model_id: str) -> str:
    model = str(model_id or "").strip()
    if provider_id == "openai-codex":
        prefix = "openai-codex-"
        if model.startswith(prefix):
            return model[len(prefix):]
    if provider_id in {"anthropic", "anthropic-agent-sdk"}:
        return _canonical_claude_model_id(model)
    return model


def _canonical_claude_model_id(model_id: str) -> str:
    model = str(model_id or "").strip()
    if not model.startswith("claude-"):
        return model
    return re.sub(r"(?<=\d)\.(?=\d)", "-", model)


def split_model_spec(value: str | None) -> tuple[str, str] | None:
    text = str(value or "").strip()
    if "/" not in text:
        return None
    provider, model = text.split("/", 1)
    provider = provider.strip()
    model = _canonical_model_id_for_provider(provider, model)
    if not provider or not model:
        return None
    return provider, model


def _read_models_json() -> dict[str, Any]:
    path = _models_json_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def registered_pi_model(provider_id: str, model_id: str) -> bool:
    raw = _read_models_json()
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        return False
    provider_block = providers.get(provider_id)
    if not isinstance(provider_block, dict):
        return False
    models = provider_block.get("models")
    if models is None:
        return True
    if not isinstance(models, list):
        return False
    for entry in models:
        if isinstance(entry, dict) and str(entry.get("id") or "").strip() == model_id:
            return True
        if isinstance(entry, str) and entry.strip() == model_id:
            return True
    return False


def launchable_model(provider_id: str, model_id: str) -> bool:
    if not provider_id or not model_id:
        return False
    if provider_id in BUILT_IN_PI_PROVIDERS:
        return True
    return registered_pi_model(provider_id, model_id)


def _agent_sdk_auth_available() -> bool:
    """True if the Claude Agent SDK can authenticate with the subscription:
    a headless CLAUDE_CODE_OAUTH_TOKEN, or an interactive Claude Code login whose
    credentials the CLI stored at ~/.claude/.credentials.json. 2026-06-20:
    reject expired access-token-only files so model-setting does not accept a
    worker model that will fail later with authentication_failed."""
    from jlc_agentic.claude_auth import agent_sdk_auth_available  # noqa: PLC0415

    return agent_sdk_auth_available()


def _provider_auth_usable(cfg: dict[str, Any]) -> bool:
    if cfg.get("auth_kind") == "oauth":
        return True
    if cfg.get("auth_kind") == "agent-sdk":
        return _agent_sdk_auth_available()
    auth_env = str(cfg.get("auth_env") or "").strip()
    if not auth_env:
        return True  # keyless/local providers (e.g. local ollama)
    return bool(os.environ.get(auth_env, "").strip())


def _auth_unavailable_error(provider_id: str, cfg: dict[str, Any], *, suffix: str = "") -> ValueError:
    if cfg.get("auth_kind") == "agent-sdk":
        return ValueError(
            f"provider {provider_id!r} needs the Claude Code CLI + a login "
            "(run `claude` and log in, or `claude setup-token`) — it uses no API key"
        )
    auth_env = str(cfg.get("auth_env") or "").strip()
    return ValueError(f"provider {provider_id!r} has no API key (set {auth_env} first){suffix}")


def _is_keyless_provider_cfg(cfg: dict[str, Any]) -> bool:
    if cfg.get("auth_kind") == "oauth":
        return False
    return not str(cfg.get("auth_env") or "").strip()


def _is_keyless_local_preset_provider(provider_id: str, cfg: dict[str, Any]) -> bool:
    return provider_id in KEYLESS_LOCAL_PRESET_PROVIDERS and _is_keyless_provider_cfg(cfg)


def _resolve_catalog_model(requested: str, models: list[str]) -> str | None:
    canonical = _canonical_claude_model_id(requested)
    for candidate in (requested, canonical):
        if candidate in models:
            return candidate
    return _fuzzy_unique_match(canonical, models) or _fuzzy_unique_match(requested, models)


def _register_keyless_local_preset_model(
    catalog: dict[str, Any],
    provider_id: str,
    cfg: dict[str, Any],
    requested_model: str,
) -> tuple[str | None, str | None]:
    if not _is_keyless_local_preset_provider(provider_id, cfg):
        return None, None
    result = fetch_model_catalog(provider_id, cfg, force_refresh=True, allow_fallback=False)
    resolved = _resolve_catalog_model(requested_model, result.models or [])
    if not resolved:
        return None, result.warning or result.source
    _write_models_json(catalog, [(provider_id, resolved)])
    return resolved, None


def validate_launchable_model_spec(value: str | None) -> tuple[str, str, str | None]:
    """Resolve a spawn model spec to a launchable, authenticated (provider, model).

    Accepts 'provider/model' or a bare model name. A bare model routes to the
    unique catalog provider that both lists it and has usable auth, so
    "glm-5.1" lands on the keyed ollama-cloud instead of erroring — and an
    explicit keyless provider is rejected here rather than booting a window
    that cannot chat. Ambiguity raises with the candidate list so the caller
    can ask the user. Returns (provider, model, routing_note).
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("model must be a 'provider/model' or model name")
    cat = load_catalog()
    cat_providers = {
        pid: cfg
        for pid, cfg in cat.get("providers", {}).items()
        if isinstance(pid, str) and isinstance(cfg, dict) and cfg.get("enabled") is not False
    }

    split = split_model_spec(text)
    # Only treat "a/b" as provider/model when "a" is a real provider. Otherwise
    # the slash belongs to the model id itself (e.g. OpenRouter
    # "nex-agi/nex-n2-pro:free"); fall through to bare routing on the full
    # string so it resolves against each provider's live /models list.
    if split is not None and split[0] not in cat_providers and split[0] not in BUILT_IN_PI_PROVIDERS:
        split = None
    if split is not None:
        provider_id, model_id = split
        routing_note: str | None = None
        cfg = cat_providers.get(provider_id)
        if cfg is not None and _is_keyless_local_preset_provider(provider_id, cfg):
            registered_model, unavailable_reason = _register_keyless_local_preset_model(cat, provider_id, cfg, model_id)
            if registered_model is not None:
                return provider_id, registered_model, None
            raise ValueError(
                f"local provider {provider_id!r} is unavailable for model {model_id!r}; "
                f"start the local server and retry ({unavailable_reason or 'could not reach /models'}) "
                "or pick it once via /model-setting"
            )
        if cfg is not None and not _provider_auth_usable(cfg):
            # Redirect a key-less explicit pick to the preferred usable sibling
            # serving the same model (anthropic -> anthropic-agent-sdk) before
            # failing — mirrors validate_model_pick so spawn and chat agree.
            sibling = _usable_sibling_for_model(cat_providers, provider_id, model_id)
            if sibling is not None:
                original_provider = provider_id
                original_model = model_id
                provider_id, model_id = sibling
                routing_note = (
                    f"routed {original_provider}/{original_model} to {provider_id}/{model_id} "
                    "because the requested provider has no usable auth"
                )
                cfg = cat_providers.get(provider_id)
            else:
                raise _auth_unavailable_error(provider_id, cfg, suffix="; the spawned window could not chat")
        if cfg is not None and _is_sidecar_routed_provider_cfg(cfg):
            result = fetch_model_catalog(provider_id, cfg, allow_fallback=True)
            resolved = _resolve_catalog_model(model_id, result.models or [])
            if not resolved:
                raise ValueError(f"model {model_id!r} not in {provider_id!r} catalog")
            return provider_id, resolved, routing_note
        if not launchable_model(provider_id, model_id):
            # Not in models.json yet — confirm against the provider's LIVE
            # catalog and auto-register (mirrors the bare path), so an explicit
            # provider/model that shows in /model-setting can spawn without a
            # separate registration step. A model absent from the live catalog
            # is rejected rather than persisted as a phantom entry.
            live = fetch_model_catalog(provider_id, cfg, force_refresh=True, allow_fallback=False) if cfg is not None else None
            live_resolved = _resolve_catalog_model(model_id, live.models or []) if live is not None else None
            if not live_resolved:
                raise ValueError(
                    f"model {provider_id}/{model_id} is not registered and not in "
                    f"{provider_id}'s live catalog; register it via /model-setting"
                )
            _write_models_json(cat, [(provider_id, live_resolved)])
            model_id = live_resolved
        return provider_id, model_id, routing_note

    # Registration is deferred until a single match wins: writing models.json
    # per candidate would leave stray registry entries on the ambiguous path.
    matches: list[tuple[str, str, bool]] = []
    for pid in sorted(cat_providers):
        cfg = cat_providers[pid]
        if not _provider_auth_usable(cfg):
            continue
        if _is_sidecar_routed_provider_cfg(cfg):
            # Sidecar-routed worker models are valid spawn targets: jarvis.ps1
            # strips the non-Pi-launchable CLI provider/model, runs Pi on the
            # encoder model, and pins chat via JARVIS_CHAT_MODEL_OVERRIDE.
            result = fetch_model_catalog(pid, cfg, allow_fallback=True)
            resolved = _resolve_catalog_model(text, result.models or [])
            if resolved:
                matches.append((pid, resolved, False))
            continue
        require_live = _is_keyless_local_preset_provider(pid, cfg)
        result = fetch_model_catalog(pid, cfg, force_refresh=require_live, allow_fallback=not require_live)
        resolved = _resolve_catalog_model(text, result.models or [])
        is_launchable = launchable_model(pid, resolved) if resolved else False
        live_confirmed = require_live
        if not is_launchable and not require_live:
            # Keyed cloud provider: the cached catalog either missed a freshly
            # added model or listed one not yet in models.json. A single live
            # refresh both discovers and confirms it, so a model that shows in
            # /model-setting can route and auto-register without ever persisting
            # a stale entry.
            live = fetch_model_catalog(pid, cfg, force_refresh=True, allow_fallback=False)
            live_resolved = _resolve_catalog_model(text, live.models or [])
            if live_resolved:
                resolved = live_resolved
                live_confirmed = True
        if resolved and (is_launchable or live_confirmed):
            matches.append((pid, resolved, not is_launchable))
    if matches:
        # Prefer the subscription/OAuth seat over a metered key when several
        # providers serve the model (bare 'gpt-5.5' -> codex, 'opus' -> agent-sdk),
        # collapsing to a single winner unless two tie at the cheapest rank.
        winner = _prefer_ranked_provider(
            [(p, m) for p, m, _ in matches],
            rank_of=lambda pid: _provider_cost_rank(cat_providers[pid]),
            ambiguity_msg=lambda top: (
                f"model {text!r} is available from multiple equally-preferred providers: "
                f"{', '.join(f'{p}/{m}' for p, m in top)}; pass provider/model (ask the user which one)"
            ),
        )
        if winner is not None:
            pid, resolved = winner
            needs_register = next(nr for p, m, nr in matches if (p, m) == (pid, resolved))
            if needs_register:
                _write_models_json(cat, [(pid, resolved)])
            return pid, resolved, f"routed {text!r} to {pid}/{resolved}"
    raise ValueError(
        f"model {text!r} not found in any authenticated provider catalog or reachable keyless local provider; "
        "pass provider/model explicitly or register it first"
    )


def _normalize_pick_key(value: str) -> str:
    key = re.sub(r"[\s_]+", "-", str(value or "").strip().lower())
    return re.sub(r"-+", "-", key)


def _fuzzy_unique_match(requested: str, candidates: list[str]) -> str | None:
    wanted = _normalize_pick_key(requested)
    matched = [c for c in candidates if _normalize_pick_key(c) == wanted]
    return matched[0] if len(matched) == 1 else None


def _is_sidecar_routed_provider_cfg(cfg: dict[str, Any]) -> bool:
    """Providers whose chat turn is executed by the JLC sidecar (the Agent SDK),
    not by Pi. They are NOT Pi-launchable, yet they are valid spawn targets: the
    spawned worker runs Pi on the encoder provider and the sidecar drives chat
    with this model (jarvis.ps1 + JARVIS_CHAT_MODEL_OVERRIDE). (Jun, 2026-06-16)
    """
    return isinstance(cfg, dict) and cfg.get("auth_kind") == "agent-sdk"


def _provider_cost_rank(cfg: dict[str, Any]) -> int:
    """Lower = preferred when several authenticated providers serve the same model.

    A subscription/OAuth seat (free under the plan: openai-codex, anthropic-agent-sdk)
    beats a keyless local server, which beats a metered API key. So a bare
    'gpt-5.5' lands on the Codex OAuth seat and 'claude-opus-4-8' on the Agent-SDK
    subscription unless the user names the API-key provider explicitly. The catalog's
    ``auth_kind`` field is the oauth-vs-api list this policy reads. (Jun, 2026-06-16)
    """
    if cfg.get("auth_kind") in ("oauth", "agent-sdk"):
        return 0  # subscription / OAuth — free under the plan, prefer it
    if not str(cfg.get("auth_env") or "").strip():
        return 1  # keyless local server
    return 2  # metered API key


def _prefer_ranked_provider(
    candidates: list[tuple[str, str]],
    *,
    rank_of: Callable[[str], int],
    ambiguity_msg: Callable[[list[tuple[str, str]]], str],
) -> tuple[str, str] | None:
    """Collapse same-model provider candidates to the single cheapest (subscription
    over api). Returns the winner, ``None`` when empty, or raises with the caller's
    message when two providers tie at the cheapest rank (a real fork to surface)."""
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda c: rank_of(c[0]))
    best = rank_of(ranked[0][0])
    top = [c for c in ranked if rank_of(c[0]) == best]
    if len(top) == 1:
        return top[0]
    raise ValueError(ambiguity_msg(top))


def _usable_sibling_for_model(
    providers: dict[str, Any], requested_provider: str, model_id: str
) -> tuple[str, str] | None:
    """When ``requested_provider`` cannot authenticate, find the preferred enabled
    sibling provider that has usable auth AND serves ``model_id``.

    This is what lets a subscription-only user who asks for
    ``anthropic/claude-opus-4-8`` (no ANTHROPIC_API_KEY) transparently land on the
    ``anthropic-agent-sdk`` sibling that serves the same model via the logged-in
    Claude Code CLI. When several usable siblings serve it, the subscription/OAuth
    seat wins over a metered key (see ``_provider_cost_rank``). Returns the chosen
    ``(provider, resolved_model)`` or ``None`` when there is no usable sibling;
    raises only on a genuine tie at the cheapest rank.
    """
    if requested_provider not in AUTH_SIBLING_REDIRECT_PROVIDERS:
        return None
    candidates: list[tuple[str, str]] = []
    for pid, cfg in providers.items():
        if pid == requested_provider or not isinstance(cfg, dict):
            continue
        if cfg.get("enabled") is False or not _provider_auth_usable(cfg):
            continue
        result = fetch_model_catalog(pid, cfg)
        resolved = _resolve_catalog_model(model_id, result.models or [])
        if resolved:
            candidates.append((pid, resolved))
    return _prefer_ranked_provider(
        candidates,
        rank_of=lambda pid: _provider_cost_rank(providers[pid]),
        ambiguity_msg=lambda top: (
            f"{requested_provider!r} can't authenticate and {model_id!r} is served by "
            f"multiple equally-preferred providers: {', '.join(f'{p}/{m}' for p, m in top)}; "
            "pass provider/model explicitly"
        ),
    )


def validate_model_pick(
    provider_id: str,
    model_id: str,
    *,
    catalog: dict[str, Any] | None = None,
) -> tuple[str, str, str | None]:
    """Validate a provider/model pick against the catalog before it is saved.

    Returns the (possibly fuzzy-corrected) provider/model plus a human-readable
    note describing any correction or unverifiable state. Raises ValueError when
    the pick cannot be resolved — config.yaml must never receive a spec that is
    known-broken at save time (typos reload-fail, keyless providers 401 on the
    first turn).
    """
    cat = catalog or load_catalog()
    providers = {
        pid: cfg
        for pid, cfg in cat.get("providers", {}).items()
        if isinstance(pid, str) and isinstance(cfg, dict) and provider_supports_model_setting(cfg)
    }
    notes: list[str] = []

    resolved_provider = provider_id
    if resolved_provider not in providers:
        corrected = _fuzzy_unique_match(resolved_provider, list(providers))
        if corrected is None:
            raise ValueError(f"unknown provider {provider_id!r}; pick one of: {', '.join(sorted(providers))}")
        notes.append(f"provider {provider_id!r} corrected to {corrected!r}")
        resolved_provider = corrected
    model_id = _canonical_model_id_for_provider(resolved_provider, model_id)
    cfg = providers[resolved_provider]

    if cfg.get("enabled") is False:
        raise ValueError(f"provider {resolved_provider!r} is disabled in the catalog: {cfg.get('note') or 'disabled'}")

    if not _provider_auth_usable(cfg):
        # The named provider can't authenticate. Before failing, see if a sibling
        # serves the same model with usable auth — a subscription-only user who asks
        # for `anthropic/claude-opus-4-8` (no key) is transparently routed to the
        # `anthropic-agent-sdk` subscription twin; likewise `openai/...` -> codex
        # OAuth. Only when there is no usable sibling do we surface the original,
        # provider-appropriate auth error.
        sibling = _usable_sibling_for_model(providers, resolved_provider, model_id)
        if sibling is not None:
            sib_provider, sib_model = sibling
            notes.append(
                f"provider {resolved_provider!r} has no usable auth; routed to "
                f"{sib_provider!r} (same model, subscription/login already available)"
            )
            resolved_provider, model_id = sib_provider, sib_model
            cfg = providers[resolved_provider]
        elif cfg.get("auth_kind") == "agent-sdk":
            # Gated on the Claude Code CLI being present + logged in (or a setup-token),
            # not an API key — that login is what bills the subscription's Agent SDK path.
            raise _auth_unavailable_error(resolved_provider, cfg)
        else:
            raise _auth_unavailable_error(resolved_provider, cfg)

    result = fetch_model_catalog(resolved_provider, cfg)
    models = result.models
    resolved_model = model_id
    if models is None:
        notes.append(f"model list for {resolved_provider!r} unavailable ({result.warning or result.source}); saved unverified")
    elif resolved_model not in models:
        corrected = _fuzzy_unique_match(resolved_model, models)
        if corrected is None:
            wanted = _normalize_pick_key(resolved_model)
            near = [m for m in models if wanted in _normalize_pick_key(m)][:5]
            hint = f"; close matches: {', '.join(near)}" if near else ""
            raise ValueError(f"model {model_id!r} not in {resolved_provider!r} catalog{hint}")
        notes.append(f"model {model_id!r} corrected to {corrected!r}")
        resolved_model = corrected

    return resolved_provider, resolved_model, "; ".join(notes) or None


def model_catalog_cache_path() -> Path:
    override = os.environ.get("JARVIS_MODEL_CATALOG_CACHE")
    if override:
        return Path(override).expanduser()
    return Path("~/.jarvis-code/model-catalog-cache.json").expanduser()


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


def load_repo_providers() -> dict[str, Any]:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    providers = catalog.get("providers") if isinstance(catalog, dict) else {}
    return providers if isinstance(providers, dict) else {}


def load_raw_user_catalog() -> dict[str, Any]:
    path = user_catalog_path()
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("user catalog must be a mapping")
    return raw


def load_raw_user_providers() -> dict[str, Any]:
    raw = load_raw_user_catalog()
    providers = raw.get("providers")
    if providers is None:
        return {}
    if not isinstance(providers, dict):
        raise ValueError("user catalog providers must be a mapping")
    return providers


def provider_source(provider_id: str) -> str:
    return "bundled" if provider_id in load_repo_providers() else "custom"


def find_provider_duplicate(provider_id: str, base_url: str) -> tuple[str, dict[str, Any]] | None:
    normalized_base = _normalize_base_url(base_url)
    catalog = load_catalog()
    providers = catalog.get("providers", {})
    if not isinstance(providers, dict):
        return None
    for existing_id, cfg in providers.items():
        if not isinstance(existing_id, str) or not isinstance(cfg, dict):
            continue
        if existing_id == provider_id:
            return existing_id, cfg
        existing_base = str(cfg.get("base_url") or "")
        if existing_base and _normalize_base_url(existing_base) == normalized_base:
            return existing_id, cfg
    return None


def upsert_user_provider(provider_id: str, cfg: dict[str, Any]) -> Path:
    if provider_id in load_repo_providers():
        raise ValueError(f"cannot write bundled provider {provider_id!r} to user catalog")
    auth_env_value = cfg.get("auth_env", custom_provider_auth_env(provider_id))
    auth_env = str(auth_env_value or "").strip()
    required = {
        "label": str(cfg.get("label") or provider_id).strip(),
        "base_url": str(cfg.get("base_url") or "").strip(),
        "api_format": str(cfg.get("api_format") or "openai-completions").strip(),
        "models_endpoint": str(cfg.get("models_endpoint") or "/models").strip(),
    }
    if auth_env:
        required["auth_env"] = auth_env
    if not required["label"]:
        raise ValueError("label is required")
    _validate_base_url(required["base_url"])
    if required["api_format"] not in KNOWN_API_FORMATS:
        raise ValueError(f"unknown api_format {required['api_format']!r}")

    path = user_catalog_path()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    content = _upsert_provider_block(text, provider_id, required)
    _write_user_catalog_text(path, content)
    return path


def remove_user_provider(provider_id: str) -> tuple[Path, dict[str, Any] | None]:
    if provider_id in load_repo_providers():
        raise ValueError(f"cannot remove bundled provider {provider_id!r}")
    path = user_catalog_path()
    if not path.exists():
        return path, None
    raw_providers = load_raw_user_providers()
    removed = raw_providers.get(provider_id)
    text = path.read_text(encoding="utf-8")
    content, did_remove = _remove_provider_block(text, provider_id)
    if did_remove:
        _write_user_catalog_text(path, content)
    return path, removed if isinstance(removed, dict) else None


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
    if user_providers is None:
        return {}
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


def _normalize_base_url(base_url: str) -> str:
    return base_url.strip().rstrip("/")


def _validate_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an http or https URL")


def _write_user_catalog_text(path: Path, content: str) -> None:
    from .config import _atomic_write_text

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, content)


def _provider_block(provider_id: str, cfg: dict[str, Any]) -> str:
    dumped = yaml.safe_dump({provider_id: cfg}, sort_keys=False, allow_unicode=True)
    return "".join(f"  {line}" if line.strip() else line for line in dumped.splitlines(keepends=True))


def _find_providers_section(lines: list[str]) -> tuple[int | None, int]:
    start: int | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if re.match(r"^providers:\s*(?:#.*)?$", line.rstrip("\r\n")):
            start = idx
            break
    if start is None:
        return None, end
    for idx in range(start + 1, len(lines)):
        raw = lines[idx].rstrip("\r\n")
        if raw and not raw.startswith((" ", "\t")) and not raw.lstrip().startswith("#"):
            end = idx
            break
    return start, end


def _find_provider_block(lines: list[str], section_start: int, section_end: int, provider_id: str) -> tuple[int, int] | None:
    key_re = re.compile(rf"^  {re.escape(provider_id)}:\s*(?:#.*)?$")
    next_key_re = re.compile(r"^  [A-Za-z0-9_-]+:\s*(?:#.*)?$")
    start: int | None = None
    for idx in range(section_start + 1, section_end):
        raw = lines[idx].rstrip("\r\n")
        if start is None:
            if key_re.match(raw):
                start = idx
            continue
        if next_key_re.match(raw):
            return start, idx
    if start is not None:
        return start, section_end
    return None


def _upsert_provider_block(text: str, provider_id: str, cfg: dict[str, Any]) -> str:
    lines = text.splitlines(keepends=True)
    block = _provider_block(provider_id, cfg)
    if lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += "\n"
    section_start, section_end = _find_providers_section(lines)
    if section_start is None:
        prefix = "".join(lines)
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        return f"{prefix}providers:\n{block}"
    existing = _find_provider_block(lines, section_start, section_end, provider_id)
    if existing is not None:
        start, end = existing
        return "".join(lines[:start]) + block + "".join(lines[end:])
    insert_at = section_end
    before = "".join(lines[:insert_at])
    after = "".join(lines[insert_at:])
    if before and not before.endswith("\n"):
        before += "\n"
    return before + block + after


def _remove_provider_block(text: str, provider_id: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    section_start, section_end = _find_providers_section(lines)
    if section_start is None:
        return text, False
    existing = _find_provider_block(lines, section_start, section_end, provider_id)
    if existing is None:
        return text, False
    start, end = existing
    return "".join(lines[:start]) + "".join(lines[end:]), True


def fetch_models(provider_id: str, cfg: dict[str, Any], *, force_refresh: bool = False) -> list[str] | None:
    """Return list of model IDs, or None if no live/cache/static list exists."""
    return fetch_model_catalog(provider_id, cfg, force_refresh=force_refresh).models


def fetch_model_catalog(
    provider_id: str,
    cfg: dict[str, Any],
    *,
    force_refresh: bool = False,
    allow_fallback: bool = True,
) -> ModelCatalogResult:
    """Resolve the display model list for one provider.

    Live provider lists are display-only. Curated metadata remains in
    pi-agent/models.json and is merged when a model is selected.
    """
    if cfg.get("enabled") is False:
        return ModelCatalogResult(None, "disabled", warning=str(cfg.get("note") or "disabled in catalog"))

    static_models = _static_model_ids(provider_id, cfg)
    if cfg.get("auth_kind") == "oauth":
        # The ChatGPT subscription backend serves a model list for OAuth
        # sessions (verified live 2026-06-12). Live first; the curated
        # models_static list is only the offline/endpoint-drift fallback.
        cached = _read_model_cache(provider_id, cfg)
        if cached is not None and cached["fresh"] and not force_refresh:
            return ModelCatalogResult(list(cached["models"]), "cache")
        live_models = _fetch_oauth_models(provider_id, cfg)
        if live_models:
            _write_model_cache(provider_id, cfg, live_models)
            return ModelCatalogResult(live_models, "live")
        if not allow_fallback:
            return ModelCatalogResult(None, "unavailable", warning="could not reach /models")
        if cached is not None:
            return ModelCatalogResult(list(cached["models"]), "cache", cache_stale=True, warning="using cached list")
        if static_models:
            return ModelCatalogResult(static_models, "static")
        return ModelCatalogResult(None, "unavailable", warning="not logged in")

    if cfg.get("auth_kind") == "agent-sdk":
        # Claude subscription backend: list models with the logged-in Claude
        # Code OAuth token (no ANTHROPIC_API_KEY). Anthropic's /v1/models accepts
        # the Bearer token plus the anthropic-beta oauth header (verified live
        # 2026-07-01). Only attempt the live fetch when the catalog wires an
        # endpoint (mirrors the keyed path below); without one — or when the
        # fetch fails / the user is logged out — fall back to models_static.
        if cfg.get("models_endpoint"):
            cached = _read_model_cache(provider_id, cfg)
            if cached is not None and cached["fresh"] and not force_refresh:
                return ModelCatalogResult(list(cached["models"]), "cache")
            live_models = _fetch_agent_sdk_models(cfg)
            if live_models:
                _write_model_cache(provider_id, cfg, live_models)
                return ModelCatalogResult(live_models, "live")
            if not allow_fallback:
                return ModelCatalogResult(None, "unavailable", warning="could not reach /models")
            if cached is not None:
                return ModelCatalogResult(list(cached["models"]), "cache", cache_stale=True, warning="using cached list")
        if static_models:
            return ModelCatalogResult(static_models, "static")
        return ModelCatalogResult(None, "unavailable", warning="not logged in to Claude")

    auth_env = str(cfg.get("auth_env") or "").strip()
    api_key = os.environ.get(auth_env, "").strip() if auth_env else ""
    if auth_env and not api_key:
        if static_models and allow_fallback:
            return ModelCatalogResult(static_models, "static")
        return ModelCatalogResult(None, "unavailable", warning=f"no API key (set {auth_env})")

    endpoint = cfg.get("models_endpoint")
    if not endpoint:
        if static_models and allow_fallback:
            return ModelCatalogResult(static_models, "static")
        return ModelCatalogResult(None, "unavailable", warning="no models endpoint")

    cached = _read_model_cache(provider_id, cfg)
    if cached is not None and cached["fresh"] and not force_refresh:
        return ModelCatalogResult(list(cached["models"]), "cache")

    try:
        live_models = _fetch_live_models(provider_id, cfg, api_key)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        live_models = None
    if live_models:
        _write_model_cache(provider_id, cfg, live_models)
        return ModelCatalogResult(live_models, "live")

    if not allow_fallback:
        return ModelCatalogResult(None, "unavailable", warning="could not reach /models")

    if cached is not None:
        return ModelCatalogResult(list(cached["models"]), "cache", cache_stale=True, warning="using cached list")
    if static_models:
        return ModelCatalogResult(static_models, "static", warning="using cached list")
    return ModelCatalogResult(None, "unavailable", warning="could not reach /models")


def _fetch_live_models(provider_id: str, cfg: dict[str, Any], api_key: str) -> list[str] | None:
    _ = provider_id
    endpoint = str(cfg.get("models_endpoint") or "")
    url = cfg["base_url"].rstrip("/") + endpoint
    headers = with_jarvis_user_agent({"Authorization": f"Bearer {api_key}"} if api_key else None)
    if cfg.get("api_format") == "google-generative-ai":
        url = _with_query_param(url, "key", api_key)
        headers = with_jarvis_user_agent()
    elif cfg.get("api_format") == "anthropic":
        headers = with_jarvis_user_agent(
            {
                "x-api-key": api_key,
                "anthropic-version": cfg.get("anthropic_version", "2023-06-01"),
            }
        )
        return _fetch_anthropic_models(url, headers)

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    models = _extract_model_ids(data)
    return models or None


def _fetch_anthropic_models(url: str, headers: dict[str, str]) -> list[str] | None:
    out: list[str] = []
    seen_pages = 0
    next_url = url
    while next_url and seen_pages < 20:
        seen_pages += 1
        req = urllib.request.Request(next_url, headers=headers)
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        out.extend(_extract_model_ids(data))
        if not isinstance(data, dict) or not data.get("has_more"):
            break
        last_id = data.get("last_id")
        if not isinstance(last_id, str) or not last_id:
            break
        next_url = _with_query_param(url, "after_id", last_id)
    return _dedupe_models(out) or None


def _fetch_agent_sdk_models(cfg: dict[str, Any]) -> list[str] | None:
    """List Claude models with the subscription OAuth token (no API key).

    Anthropic's /v1/models honours the Claude Code OAuth bearer token when the
    ``anthropic-beta: oauth-*`` header is present, so a subscription-only user
    gets the live catalogue without ANTHROPIC_API_KEY.
    """
    try:
        from jlc_agentic.claude_auth import get_agent_sdk_access_token

        token = get_agent_sdk_access_token()
    except Exception:  # noqa: BLE001
        token = None
    if not token:
        return None

    base_url = str(cfg.get("base_url") or "https://api.anthropic.com/v1").rstrip("/")
    endpoint = str(cfg.get("models_endpoint") or "/models")
    url = base_url + endpoint
    headers = with_jarvis_user_agent(
        {
            "Authorization": f"Bearer {token}",
            "anthropic-version": cfg.get("anthropic_version", "2023-06-01"),
            "anthropic-beta": cfg.get("anthropic_oauth_beta", "oauth-2025-04-20"),
        }
    )
    try:
        return _fetch_anthropic_models(url, headers)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None


def _static_model_ids(provider_id: str, cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    static = cfg.get("models_static")
    if isinstance(static, list):
        out.extend(str(model).strip() for model in static if str(model).strip())

    try:
        raw = json.loads(_models_json_path().read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    providers = raw.get("providers") if isinstance(raw, dict) else {}
    provider_block = providers.get(provider_id) if isinstance(providers, dict) else None
    models = provider_block.get("models") if isinstance(provider_block, dict) else None
    if isinstance(models, list):
        for entry in models:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"].strip():
                out.append(entry["id"].strip())
            elif isinstance(entry, str) and entry.strip():
                out.append(entry.strip())
    return _dedupe_models(out)


def _fetch_oauth_models(provider_id: str, cfg: dict[str, Any]) -> list[str] | None:
    if provider_id != "openai-codex":
        return None

    try:
        from jlc_agentic.openai_oauth import TokenManager

        mgr = TokenManager()
        access_token = mgr.get_access_token()
        account_id = mgr.get_account_id() or ""
    except Exception:  # noqa: BLE001
        return None

    endpoint = cfg.get("models_endpoint") or "/models"
    url = _with_query_param(
        cfg["base_url"].rstrip("/") + endpoint,
        "client_version",
        JARVIS_CODE_VERSION,
    )
    req = urllib.request.Request(
        url,
        headers=with_jarvis_user_agent(
            {
                "Authorization": f"Bearer {access_token}",
                "ChatGPT-Account-Id": account_id,
                "originator": "jarvis-code",
            }
        ),
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
        return _dedupe_models([
            str(m["id"])
            for m in data["data"]
            if isinstance(m, dict) and isinstance(m.get("id"), str) and m["id"]
        ])
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
        return _dedupe_models(out)
    if isinstance(data, list):
        return _dedupe_models([m for m in data if isinstance(m, str) and m])
    return []


def _dedupe_models(models: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for model in models:
        value = str(model or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _cache_key(cfg: dict[str, Any]) -> str:
    return "|".join(
        str(cfg.get(key) or "").strip()
        for key in ("base_url", "models_endpoint", "api_format", "auth_env", "auth_kind")
    )


def _read_cache_file() -> dict[str, Any]:
    path = model_catalog_cache_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"version": 1, "providers": {}}
    if not isinstance(raw, dict):
        return {"version": 1, "providers": {}}
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        raw["providers"] = {}
    return raw


def _read_model_cache(provider_id: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    raw = _read_cache_file()
    providers = raw.get("providers")
    entry = providers.get(provider_id) if isinstance(providers, dict) else None
    if not isinstance(entry, dict) or entry.get("cache_key") != _cache_key(cfg):
        return None
    models = entry.get("models")
    fetched_at = entry.get("fetched_at")
    if not isinstance(models, list) or not isinstance(fetched_at, (int, float)):
        return None
    clean = _dedupe_models([str(model) for model in models])
    if not clean:
        return None
    return {
        "models": clean,
        "fresh": (time.time() - float(fetched_at)) < MODEL_CATALOG_CACHE_TTL_S,
        "fetched_at": fetched_at,
    }


def _write_model_cache(provider_id: str, cfg: dict[str, Any], models: list[str]) -> None:
    from .config import _atomic_write_text

    # Lock the full read-modify-write: fetch_all_detailed() fans out across threads
    # and they all touch this one shared file — without the lock, concurrent writes
    # drop each other's provider entries (last-writer-wins). (2026-06-15)
    with _CACHE_FILE_LOCK:
        path = model_catalog_cache_path()
        raw = _read_cache_file()
        providers = raw.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            raw["providers"] = providers
        providers[provider_id] = {
            "cache_key": _cache_key(cfg),
            "models": _dedupe_models(models),
            "fetched_at": time.time(),
            "fetched_at_iso": datetime.now(UTC).replace(microsecond=0).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(path, json.dumps(raw, indent=2, ensure_ascii=False) + "\n")


def clear_model_catalog_cache() -> None:
    try:
        model_catalog_cache_path().unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _with_query_param(url: str, key: str, value: str) -> str:
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query = [(k, v) for k, v in query if k != key]
    query.append((key, value))
    return urllib.parse.urlunsplit(
        parts._replace(query=urllib.parse.urlencode(query))
    )


def fetch_all(catalog: dict[str, Any], *, force_refresh: bool = False) -> dict[str, list[str] | None]:
    """Fetch model lists for every provider in the catalog. Disabled providers
    are reported as None so callers can treat them uniformly with missing-key
    providers."""
    return {pid: result.models for pid, result in fetch_all_detailed(catalog, force_refresh=force_refresh).items()}


def fetch_all_detailed(catalog: dict[str, Any], *, force_refresh: bool = False) -> dict[str, ModelCatalogResult]:
    """Resolve detailed model catalog status for every provider.

    Each provider's model list is an independent network fetch (up to
    FETCH_TIMEOUT_S each). Running them sequentially made /model-setting wait the
    SUM of every provider's latency (10 providers × 3s ≈ 30s on a cold catalog).
    Fan out across a thread pool so the wait is the SLOWEST single fetch instead.
    The shared cache file is guarded by _CACHE_FILE_LOCK. (2026-06-15)
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(catalog["providers"].items())
    out: dict[str, ModelCatalogResult] = {}

    def _resolve(item: tuple[str, dict[str, Any]]) -> tuple[str, ModelCatalogResult]:
        pid, cfg = item
        if cfg.get("enabled") is False:
            return pid, ModelCatalogResult(None, "disabled", warning=str(cfg.get("note") or "disabled in catalog"))
        return pid, fetch_model_catalog(pid, cfg, force_refresh=force_refresh)

    if not items:
        return out
    with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
        for pid, result in pool.map(_resolve, items):
            out[pid] = result
    return out


def current_roles() -> dict[str, str | None]:
    """Read roles from data/config.yaml. Missing values come back as None."""
    path = _config_path()
    if not path.exists():
        return {"chat": None, "subagent": None, "router": None, "encoder": None}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {"chat": None, "subagent": None, "router": None, "encoder": None}
    roles = raw.get("roles") or {}
    return {
        "chat": roles.get("chat"),
        "subagent": roles.get("subagent"),
        "router": roles.get("router"),
        "encoder": roles.get("encoder"),
    }


def _catalog_providers_for_apply(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    providers = catalog.get("providers")
    if not isinstance(providers, dict):
        return {}
    return {
        pid: cfg
        for pid, cfg in providers.items()
        if isinstance(pid, str) and isinstance(cfg, dict) and provider_supports_model_setting(cfg)
    }


def _resolve_role_pick(
    catalog: dict[str, Any],
    role: str,
    spec: str | tuple[str, str] | None,
    *,
    validate_model: bool,
) -> tuple[str, str]:
    if isinstance(spec, tuple):
        provider_id, model_id = spec
        provider_id = str(provider_id or "").strip()
        model_id = _canonical_model_id_for_provider(provider_id, str(model_id or ""))
        if not provider_id or not model_id:
            raise ValueError(f"{role} role is missing or invalid")
    else:
        split = split_model_spec(spec)
        if split is None:
            raise ValueError(f"{role} role is missing or invalid")
        provider_id, model_id = split

    providers = _catalog_providers_for_apply(catalog)
    cfg = providers.get(provider_id)
    if cfg is None:
        raise ValueError(f"{role} role references unknown provider {provider_id!r}")
    if cfg.get("enabled") is False:
        raise ValueError(f"{role} role references disabled provider {provider_id!r}")

    if validate_model:
        result = fetch_model_catalog(provider_id, cfg)
        models = result.models
        if models is not None:
            resolved_model = _resolve_catalog_model(model_id, models)
            if resolved_model is None:
                raise ValueError(f"{role} role model {model_id!r} is not in {provider_id!r} catalog")
            model_id = resolved_model
    return provider_id, model_id


def _referenced_role_picks(catalog: dict[str, Any], roles: dict[str, str | None]) -> list[tuple[str, str]]:
    picks: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for role in ("chat", "subagent", "router", "encoder"):
        spec = roles.get(role)
        if not spec:
            continue
        pick = _resolve_role_pick(catalog, role, spec, validate_model=False)
        if pick in seen:
            continue
        seen.add(pick)
        picks.append(pick)
    return picks


def apply_picks(
    chat: tuple[str, str],
    encoder: tuple[str, str],
    *,
    subagent: tuple[str, str] | None = None,
    router: tuple[str, str] | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write chat + subagent + router + encoder picks to data/config.yaml AND
    register the matching provider/model entries in pi-agent/models.json.
    Returns the resolved file paths so callers can surface them to the user."""
    cat = catalog or load_catalog()
    chat = _resolve_role_pick(cat, "chat", chat, validate_model=False)
    subagent = _resolve_role_pick(cat, "subagent", subagent or chat, validate_model=False)
    router = _resolve_role_pick(cat, "router", router or chat, validate_model=False)
    encoder = _resolve_role_pick(cat, "encoder", encoder, validate_model=False)
    cfg_path = _write_config(chat, subagent, router, encoder)
    role_picks = _referenced_role_picks(cat, current_roles())
    providers_path = _write_providers_yaml(cat, role_picks, primary=chat)
    models_path = _write_models_json(cat, role_picks)
    return {
        "config_path": str(cfg_path),
        "providers_path": str(providers_path),
        "models_json_path": str(models_path),
    }


def _resolve_current_router_pick(
    catalog: dict[str, Any],
    roles: dict[str, str | None],
    *,
    fallback: tuple[str, str],
) -> tuple[str, str]:
    try:
        return _resolve_role_pick(
            catalog,
            "current router",
            roles.get("router") or roles.get("chat"),
            validate_model=True,
        )
    except ValueError:
        return fallback


def _resolve_current_subagent_pick(
    catalog: dict[str, Any],
    roles: dict[str, str | None],
    *,
    fallback: tuple[str, str],
) -> tuple[str, str]:
    try:
        return _resolve_role_pick(
            catalog,
            "current subagent",
            roles.get("subagent") or roles.get("chat"),
            validate_model=True,
        )
    except ValueError:
        return fallback


def apply_partial_picks(
    *,
    chat: tuple[str, str] | None = None,
    subagent: tuple[str, str] | None = None,
    router: tuple[str, str] | None = None,
    encoder: tuple[str, str] | None = None,
    catalog: dict[str, Any] | None = None,
) -> dict[str, str]:
    if chat is None and subagent is None and router is None and encoder is None:
        raise ValueError("chat, subagent, router, or encoder must be provided")
    cat = catalog or load_catalog()
    roles = current_roles()
    resolved_chat = (
        _resolve_role_pick(cat, "chat", chat, validate_model=False)
        if chat is not None
        else _resolve_role_pick(cat, "current chat", roles.get("chat"), validate_model=True)
    )
    resolved_subagent = (
        _resolve_role_pick(cat, "subagent", subagent, validate_model=False)
        if subagent is not None
        else _resolve_current_subagent_pick(cat, roles, fallback=resolved_chat)
    )
    resolved_router = (
        _resolve_role_pick(cat, "router", router, validate_model=False)
        if router is not None
        else _resolve_current_router_pick(cat, roles, fallback=resolved_chat)
    )
    resolved_encoder = (
        _resolve_role_pick(cat, "encoder", encoder, validate_model=False)
        if encoder is not None
        else _resolve_role_pick(cat, "current encoder", roles.get("encoder"), validate_model=True)
    )
    return apply_picks(
        resolved_chat,
        resolved_encoder,
        subagent=resolved_subagent,
        router=resolved_router,
        catalog=cat,
    )


def _write_config(
    chat: tuple[str, str],
    subagent: tuple[str, str],
    router: tuple[str, str],
    encoder: tuple[str, str],
) -> Path:
    path = _config_path()
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raw = {}
    roles = raw.setdefault("roles", {})
    roles["chat"] = f"{chat[0]}/{chat[1]}"
    roles["subagent"] = f"{subagent[0]}/{subagent[1]}"
    roles["router"] = f"{router[0]}/{router[1]}"
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
    """Write only the role-referenced providers/models.

    /model-setting is authoritative for active JARVIS roles. Keeping providers
    that no role references lets ProviderRouter fall back into stale paths, so
    this file is pruned to the final chat/subagent/router/encoder references.
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
        elif cat_cfg.get("auth_kind") == "agent-sdk":
            # Adapter-routed (Claude Agent SDK): no api_keys / api_base / litellm_id.
            # get_llm resolves it directly and bypasses the router. (2026-06-15)
            provider["adapter"] = "anthropic-agent-sdk"
        else:
            auth_env = cat_cfg.get("auth_env")
            provider["api_keys"] = [f"${{{auth_env}}}"] if auth_env else []
            if cat_cfg.get("base_url"):
                provider["api_base"] = cat_cfg["base_url"]

        if cat_cfg.get("auth_kind") == "agent-sdk":
            provider["models"][_alias_for(provider_id, model_id)] = {
                "tier": _tier_for_role(provider_id, model_id, primary),
            }
        else:
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


def _merge_models_json_user_agent(provider_block: dict[str, Any]) -> None:
    raw_headers = provider_block.get("headers")
    provider_block["headers"] = with_jarvis_user_agent(raw_headers if isinstance(raw_headers, dict) else None)
    models = provider_block.get("models")
    if not isinstance(models, list):
        return
    for model in models:
        if not isinstance(model, dict):
            continue
        model_headers = model.get("headers")
        if isinstance(model_headers, dict):
            model["headers"] = with_jarvis_user_agent(model_headers)


def _ensure_models_json_user_agents(providers: dict[str, Any]) -> None:
    for provider_block in providers.values():
        if isinstance(provider_block, dict):
            _merge_models_json_user_agent(provider_block)


def _upsert_models_json_entry(
    provider_block: dict[str, Any],
    cat_cfg: dict[str, Any],
    provider_id: str,
    model_id: str,
    *,
    ctx_window: int,
    max_tokens: int,
) -> None:
    models_list = provider_block.setdefault("models", [])
    existing = next((m for m in models_list if isinstance(m, dict) and m.get("id") == model_id), None)
    reasoning_meta = _reasoning_metadata(provider_id, model_id)
    if existing is not None:
        # Refresh managed reasoning metadata on every apply. The registry is
        # regenerated by this generator (managed_by: jarvis-model-setting), so
        # the canonical map from _reasoning_metadata must win — otherwise a
        # thinkingLevelMap written by an older build (e.g. before "max" was
        # added for the GPT-5.6 family) survives forever and new levels never
        # surface on re-apply.
        if reasoning_meta is not None:
            existing["reasoning"] = True
            existing["thinkingLevelMap"] = reasoning_meta
        return
    entry: dict[str, Any] = {
        "id": model_id,
        "name": f"{cat_cfg.get('label', provider_id)} {model_id}",
        "reasoning": reasoning_meta is not None,
        "input": ["text"],
        "contextWindow": ctx_window,
        "maxTokens": max_tokens,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }
    if reasoning_meta is not None:
        entry["thinkingLevelMap"] = reasoning_meta
    models_list.append(entry)


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
    _ensure_models_json_user_agents(providers)

    seen: set[tuple[str, str]] = set()
    for provider_id, model_id in picks:
        if (provider_id, model_id) in seen:
            continue
        seen.add((provider_id, model_id))
        if catalog["providers"].get(provider_id, {}).get("auth_kind") == "agent-sdk":
            # Adapter-routed: the JLC chat turn flows through the sidecar (get_llm),
            # not pi's models.json client. Writing no pi entry makes the launcher
            # omit --provider so pi uses its default; chat still routes here. Skip.
            continue
        if provider_id == "openai-codex":
            cat_cfg = catalog["providers"][provider_id]
            provider_block = providers.setdefault(provider_id, {})
            provider_block.setdefault("name", cat_cfg.get("label", provider_id))
            _merge_models_json_user_agent(provider_block)
            # OAuth/subscription provider: api/baseUrl come from pi's built-in
            # openai-codex provider. Custom GPT-5.6 variants still need an entry
            # so pi can resolve their full effort ladder from the registry.
            _upsert_models_json_entry(
                provider_block,
                cat_cfg,
                provider_id,
                model_id,
                ctx_window=272000,
                max_tokens=128000,
            )
            continue
        cat_cfg = catalog["providers"][provider_id]
        provider_block = providers.setdefault(provider_id, {})
        provider_block.setdefault("name", cat_cfg.get("label", provider_id))
        provider_block.setdefault("baseUrl", _pi_base_url(cat_cfg))
        provider_block.setdefault("api", _pi_api_format(cat_cfg))
        _merge_models_json_user_agent(provider_block)
        if cat_cfg.get("auth_kind") == "oauth":
            provider_block.setdefault("apiKey", None)
            provider_block.setdefault("authHeader", False)
        elif _is_keyless_provider_cfg(cat_cfg):
            if not provider_block.get("apiKey"):
                provider_block["apiKey"] = PI_KEYLESS_LOCAL_API_KEY
            provider_block["authHeader"] = False
        else:
            provider_block.setdefault("apiKey", cat_cfg.get("auth_env"))
            provider_block.setdefault("authHeader", True)
        _upsert_models_json_entry(
            provider_block,
            cat_cfg,
            provider_id,
            model_id,
            ctx_window=131072,
            max_tokens=16384,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _pi_api_format(cfg: dict[str, Any]) -> str:
    api_format = cfg.get("api_format", "openai-completions")
    if api_format == "anthropic":
        return "anthropic-messages"
    return str(api_format)


def _pi_base_url(cfg: dict[str, Any]) -> str:
    base_url = str(cfg.get("base_url") or "")
    if cfg.get("api_format") == "anthropic" and base_url.endswith("/v1"):
        return base_url[:-3]
    return base_url


def _is_gpt56_family(model_id: str | None) -> bool:
    mid = str(model_id or "").strip().lower().split("/", 1)[-1]
    return mid == "gpt-5.6" or mid.startswith("gpt-5.6-")


def _reasoning_metadata(provider_id: str, model_id: str) -> dict[str, str] | None:
    """Return default Pi thinking metadata for selector-registered models.

    Provider /models endpoints usually expose only IDs, not reliable reasoning
    capability metadata. Register fetched models as reasoning-capable so the
    effort selector can expose their standard ladder. Ultra is an explicit
    GPT-5.6-family capability and must stay hidden for every other model.
    """
    _ = provider_id
    metadata = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
    }
    if _is_gpt56_family(model_id):
        metadata["xhigh"] = "xhigh"
        metadata["max"] = "max"
        metadata["ultra"] = "ultra"
    return metadata
