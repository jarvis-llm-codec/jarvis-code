from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import yaml

from .base import ProviderAdapter
from .openai_compatible import OpenAICompatibleAdapter
from jlc_agentic.router_llm_adapter import LLMRouterAdapter

_log = logging.getLogger(__name__)

DEFAULT_DASHSCOPE_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"
DEFAULT_DASHSCOPE_MODEL_CHAT = "qwen3.6-plus"
DEFAULT_DASHSCOPE_MODEL_ENCODER = "glm-5"

_VALID_ROLES = {"chat", "subagent", "encoder", "router"}
_CACHE: dict[tuple[str, str | None, str | None], ProviderAdapter] = {}
_CACHE_LOCK = Lock()


def _read_yaml(p: Path) -> dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load jarvis-code config.yaml and merge providers.yaml siblings.

    Resolution order for config.yaml: explicit ``path`` arg →
    ``JARVIS_CODE_CONFIG`` env → ``~/.jarvis-code/config.yaml`` →
    ``C:/jarvis-code/config.yaml``. If the resolved file has no
    ``providers`` key, ``providers.yaml`` from the same directory is read
    and shallow-merged so ``_build_legacy`` finds provider entries (v1.01
    splits the two files; v1.x kept them combined).
    """
    if path:
        p = Path(path)
        if not p.exists():
            return {}
        cfg = _read_yaml(p)
    else:
        env_path = os.environ.get("JARVIS_CODE_CONFIG")
        candidates = [
            Path(env_path) if env_path else None,
            Path.home() / ".jarvis-code" / "config.yaml",
            Path("C:/jarvis-code/config.yaml"),
        ]
        p = None
        cfg = {}
        for cand in candidates:
            if cand is not None and cand.exists():
                p = cand
                cfg = _read_yaml(cand)
                break
        if p is None:
            return {}

    if "providers" not in cfg or not cfg.get("providers"):
        providers_yaml = p.parent / "providers.yaml"
        if providers_yaml.exists():
            prov_cfg = _read_yaml(providers_yaml)
            for key in ("providers", "defaults"):
                if key in prov_cfg and key not in cfg:
                    cfg[key] = prov_cfg[key]
    return cfg


def _get_provider_router() -> Any | None:
    """Return the registered ProviderRouter or None.

    Pi-era JARVIS registers the router through the sidecar-local singleton.
    Do not import the legacy Aider hook here; optional imports made router
    registration order harder to reason about and broke tests when Aider was
    not installed.
    """
    return _get_sidecar_provider_router()


def _get_sidecar_provider_router() -> Any | None:
    try:
        from jarvis_sidecar.provider_router_holder import get_provider_router
    except Exception:  # noqa: BLE001
        return None
    try:
        return get_provider_router()
    except Exception:  # noqa: BLE001
        return None


def _resolve_role_alias(role: str, cfg: dict[str, Any], router: Any) -> str:
    """Map a role to a providers.yaml alias by reading config.yaml `roles`.

    Supported `roles.{role}` shapes (for the user-facing config.yaml):
      - "provider/model" string  — preferred (W2.6, "company/model" UX)
      - {provider, model} dict   — legacy (pre-W2.6); still accepted
      - "alias" bare string      — alias key as-is, looked up in providers.yaml

    Raises KeyError if the role is missing AND no defaults.primary is set,
    or if the (provider, model) pair has no matching alias under the router.
    """
    role_cfg = (cfg.get("roles") or {}).get(role)
    if role_cfg is None:
        primary = (router.config.get("defaults") or {}).get("primary")
        if primary:
            return primary
        raise KeyError(
            f"config.yaml has no roles.{role} and providers.yaml has no defaults.primary"
        )

    if isinstance(role_cfg, dict):
        provider_name = role_cfg.get("provider")
        model_name = role_cfg.get("model")
        if not provider_name or not model_name:
            raise ValueError(f"roles.{role} dict must include both 'provider' and 'model'")
    elif isinstance(role_cfg, str):
        if "/" in role_cfg:
            provider_name, model_name = role_cfg.split("/", 1)
            if not provider_name or not model_name:
                raise ValueError(
                    f"roles.{role}={role_cfg!r}: expected 'provider/model'"
                )
        else:
            # Bare alias (e.g. legacy 'gpt-5.4'); accept iff the router knows it.
            if role_cfg in router._models:
                return role_cfg
            raise KeyError(
                f"roles.{role}={role_cfg!r}: alias not found in providers.yaml"
            )
    else:
        raise TypeError(f"roles.{role}: expected str or dict, got {type(role_cfg).__name__}")

    alias = router.find_alias(provider_name, model_name)
    if alias is None:
        raise KeyError(
            f"roles.{role}={provider_name}/{model_name}: "
            f"no matching alias under provider {provider_name!r} in providers.yaml"
        )
    return alias


def _expand_env_ref(value: Any) -> tuple[str | None, str | None]:
    """Parse a providers.yaml api_keys entry.

    Accepts the v1.01 shape ``"${OLLAMA_API_KEY}"`` (env-var indirection) and
    a bare literal key. Returns ``(literal, env_name)``; exactly one of the
    two is non-None.
    """
    if not isinstance(value, str):
        return None, None
    if value.startswith("${") and value.endswith("}"):
        return None, value[2:-1]
    return value, None


def _build_legacy(role: str, cfg: dict[str, Any]) -> ProviderAdapter:
    """v1.01: ProviderRouter + litellm are dead. Resolve roles directly to an
    OpenAI-compatible HTTP adapter (same path the chat side uses from pi).

    Shapes accepted under ``roles.{role}``:
      - ``"provider/model"`` (W2.6+; preferred)
      - ``{provider, model}``  dict (older config compat)

    Provider entry under ``providers.{provider}`` must supply ``api_base``
    (preferred) or ``base_url``, plus either ``api_keys: ["${ENV}", ...]`` /
    ``api_key`` / ``api_key_env``. Models may carry ``litellm_id`` like
    ``"openai/devstral-small-2:24b"`` — the ``openai/`` prefix is stripped.
    """
    providers = cfg.get("providers") or {}
    roles = cfg.get("roles") or {}
    role_cfg = roles.get(role)
    if role_cfg is None:
        raise KeyError(f"config.yaml has no roles.{role}")

    if isinstance(role_cfg, str):
        if "/" not in role_cfg:
            raise ValueError(
                f"roles.{role}={role_cfg!r}: expected 'provider/model'"
            )
        provider_name, model = role_cfg.split("/", 1)
    elif isinstance(role_cfg, dict):
        provider_name = role_cfg.get("provider")
        model = role_cfg.get("model")
        if not provider_name or not model:
            raise ValueError(
                f"roles.{role} dict must include both 'provider' and 'model'"
            )
    else:
        raise TypeError(f"roles.{role}: expected str or dict")

    provider_cfg = providers.get(provider_name)
    if not provider_cfg:
        raise KeyError(
            f"roles.{role}: provider {provider_name!r} not found in providers.yaml"
        )

    base_url = (
        provider_cfg.get("api_base")
        or provider_cfg.get("base_url")
    )
    if not base_url:
        raise ValueError(
            f"provider {provider_name!r} missing api_base / base_url"
        )

    api_key = provider_cfg.get("api_key")
    api_key_env = provider_cfg.get("api_key_env")
    if not api_key and not api_key_env:
        for entry in provider_cfg.get("api_keys") or []:
            literal, env_name = _expand_env_ref(entry)
            if env_name:
                api_key_env = env_name
                break
            if literal:
                api_key = literal
                break

    models = provider_cfg.get("models") or {}
    model_cfg = models.get(model)
    resolved_model = model
    if isinstance(model_cfg, dict):
        litellm_id = model_cfg.get("litellm_id") or model
        resolved_model = (
            litellm_id.split("/", 1)[1] if "/" in litellm_id else litellm_id
        )

    return OpenAICompatibleAdapter(
        base_url=base_url,
        model=resolved_model,
        api_key=api_key,
        api_key_env=api_key_env,
    )


def _provider_default_model(
    provider_name: str,
    provider_cfg: dict[str, Any],
    cfg: dict[str, Any],
) -> str:
    configured = provider_cfg.get("default_model") or provider_cfg.get("model")
    if configured:
        return str(configured)
    models = provider_cfg.get("models") or {}
    primary = (cfg.get("defaults") or {}).get("primary")
    if primary and primary in models:
        return str(primary)
    if len(models) == 1:
        return str(next(iter(models)))
    raise ValueError(
        f"provider {provider_name!r} needs an explicit model in 'provider/model'"
    )


def build_adapter_for_spec(
    spec: str,
    *,
    config_path: str | None = None,
) -> ProviderAdapter:
    """Build a provider adapter for ``provider/model`` without a configured role."""
    clean = (spec or "").strip()
    if not clean:
        raise ValueError(
            "model spec must be a non-empty provider or provider/model string"
        )
    cfg = load_config(config_path)
    providers = cfg.get("providers") or {}
    if "/" in clean:
        provider_name, model = clean.split("/", 1)
        if not provider_name or not model:
            raise ValueError(f"model spec {spec!r}: expected 'provider/model'")
    else:
        provider_name = clean
        provider_cfg = providers.get(provider_name)
        if not isinstance(provider_cfg, dict):
            raise KeyError(f"provider {provider_name!r} not found in providers.yaml")
        model = _provider_default_model(provider_name, provider_cfg, cfg)
    role = "__subagent_model_spec__"
    spec_cfg = dict(cfg)
    spec_cfg["roles"] = {**(cfg.get("roles") or {}), role: f"{provider_name}/{model}"}
    agent_sdk = _role_agent_sdk_model(role, spec_cfg)
    if agent_sdk is not None:
        from .anthropic_agent_sdk import AnthropicAgentSDKAdapter

        _provider_name, model_name, provider_cfg = agent_sdk
        return AnthropicAgentSDKAdapter(model=model_name, config=provider_cfg)
    return _build_legacy(role, spec_cfg)


def _role_uses_oauth(role: str, cfg: dict[str, Any]) -> bool:
    role_cfg = (cfg.get("roles") or {}).get(role)
    if isinstance(role_cfg, str) and "/" in role_cfg:
        provider_name = role_cfg.split("/", 1)[0]
    elif isinstance(role_cfg, dict):
        provider_name = role_cfg.get("provider")
    else:
        return False
    provider_cfg = (cfg.get("providers") or {}).get(provider_name)
    return isinstance(provider_cfg, dict) and bool(provider_cfg.get("oauth_provider"))


def _role_agent_sdk_model(
    role: str, cfg: dict[str, Any]
) -> tuple[str, str, dict[str, Any]] | None:
    """If ``roles.{role}`` points at a provider whose ``adapter`` is the Agent SDK
    delegating adapter, return ``(provider_name, model_name, provider_cfg)``.

    This provider bypasses the ProviderRouter / LiteLLM entirely: the Claude
    subscription credit pool is only drawn on when usage flows *through the Agent
    SDK*, and a raw API call would be gray-area + uncredited. (2026-06-15)
    """
    role_cfg = (cfg.get("roles") or {}).get(role)
    if isinstance(role_cfg, str) and "/" in role_cfg:
        provider_name, model_name = role_cfg.split("/", 1)
    elif isinstance(role_cfg, dict):
        provider_name = role_cfg.get("provider")
        model_name = role_cfg.get("model")
    else:
        return None
    if not provider_name or not model_name:
        return None
    provider_cfg = (cfg.get("providers") or {}).get(provider_name)
    if isinstance(provider_cfg, dict) and provider_cfg.get("adapter") == "anthropic-agent-sdk":
        return provider_name, model_name, provider_cfg
    return None


def _chat_model_override_cache_token(role: str) -> str | None:
    if role != "chat":
        return None
    return os.environ.get("JARVIS_CHAT_MODEL_OVERRIDE", "").strip() or None


def _role_spec_fingerprint(role: str, cfg: dict[str, Any]) -> str:
    """Fingerprint the *resolved* spec for ``role`` so the adapter cache detects a
    direct config.yaml swap (model / provider / adapter) even when ProviderRouter
    registration state is unchanged.

    Folded into the cache key alongside ``_chat_model_override_cache_token`` so a
    user editing config.yaml directly (without going through /llmsetting/apply,
    which clear_cache()s) no longer gets a stale adapter — including the wrong-TYPE
    case where roles.chat swaps between a legacy provider and the
    anthropic-agent-sdk delegating provider. (2026-06-22 audit fix.)

    Only the bits that change the constructed adapter are hashed: the role's
    provider/model and that provider's adapter marker + connection fields. The
    override-isolation cache token stays a separate key element, so this addition
    does not weaken JARVIS_CHAT_MODEL_OVERRIDE re-keying.
    """
    role_cfg = (cfg.get("roles") or {}).get(role)
    if isinstance(role_cfg, str):
        provider_name = role_cfg.split("/", 1)[0] if "/" in role_cfg else role_cfg
    elif isinstance(role_cfg, dict):
        provider_name = role_cfg.get("provider")
    else:
        provider_name = None
    provider_cfg = (cfg.get("providers") or {}).get(provider_name) if provider_name else None
    relevant: dict[str, Any] = {"role_cfg": role_cfg}
    if isinstance(provider_cfg, dict):
        relevant["provider"] = {
            k: provider_cfg.get(k)
            for k in ("adapter", "oauth_provider", "api_base", "base_url", "api_key_env", "models")
        }
    payload = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _spawned_worker_override_is_strict() -> bool:
    return os.environ.get("JARVIS_SPAWNED") == "1"


def _chat_model_override_spec() -> str | None:
    spec = os.environ.get("JARVIS_CHAT_MODEL_OVERRIDE", "").strip()
    if not spec:
        return None
    if "/" not in spec:
        message = f"malformed JARVIS_CHAT_MODEL_OVERRIDE={spec!r}; expected provider/model"
        if _spawned_worker_override_is_strict():
            raise RuntimeError(message)
        _log.warning(message)
        return None
    return spec


def _apply_chat_model_override(cfg: dict[str, Any]) -> dict[str, Any]:
    """Honor a per-process chat-model override for spawned workers.

    A worker launched with a sidecar-routed chat model (e.g. the Agent SDK) can't
    run that model through Pi, so jarvis.ps1 passes it as ``JARVIS_CHAT_MODEL_OVERRIDE``
    and runs Pi on the encoder provider instead. When the override names a provider
    this sidecar actually has configured, rewrite ``roles.chat`` to it (a per-call
    copy — never mutating the shared config) so the chat turn lands on the requested
    model. The Claude Agent SDK worker path may be launched from a config whose
    providers.yaml does not already contain ``anthropic-agent-sdk``; in that case
    inject a per-process adapter block so the override still drives chat. Other
    unconfigured providers are ignored only for non-worker compatibility; spawned
    workers fail hard so a requested model never silently falls back to persisted
    ``roles.chat``. (Jun, 2026-06-16: workers on the Claude subscription backend.)
    """
    spec = _chat_model_override_spec()
    if not spec:
        return cfg
    provider_name = spec.split("/", 1)[0].strip()
    providers = cfg.get("providers") or {}
    if not provider_name:
        return cfg
    if provider_name not in providers:
        if provider_name == "anthropic-agent-sdk":
            new_cfg = dict(cfg)
            new_cfg["providers"] = {**providers, provider_name: {"adapter": "anthropic-agent-sdk"}}
            new_cfg["roles"] = {**(cfg.get("roles") or {}), "chat": spec}
            logging.getLogger(__name__).info(
                "JARVIS_CHAT_MODEL_OVERRIDE=%r injected %r provider for spawned worker",
                spec,
                provider_name,
            )
            return new_cfg
        if provider_name:
            message = (
                f"JARVIS_CHAT_MODEL_OVERRIDE={spec!r} names provider {provider_name!r} "
                "not in this sidecar's providers config"
            )
            if _spawned_worker_override_is_strict():
                raise RuntimeError(message)
            logging.getLogger(__name__).warning(
                "%s; using config roles.chat instead",
                message,
            )
        return cfg
    new_cfg = dict(cfg)
    new_cfg["roles"] = {**(cfg.get("roles") or {}), "chat": spec}
    return new_cfg


def get_llm(role: str, *, config_path: str | None = None) -> ProviderAdapter:
    """Return the cached LLM adapter for ``role`` (chat / subagent / encoder / router).

    Routing rules:
      1. If a ProviderRouter is registered by the sidecar, resolve the role through config.yaml
         `roles.{role} = "provider/model"` → providers.yaml alias →
         LLMRouterAdapter. This is the preferred path because OAuth, KeyPool,
         tier fallback, and llm_meta capture all flow through the router.
      2. If no router is registered (offline / unit tests that don't boot
         the sidecar), fall back to the legacy path: roles.{role} = {provider,
         model} + providers[name] OpenAI-compatible adapter.

    On any config or shape error we raise RuntimeError instead of silently
    falling back to a default. (2026-05-04: silent dashscope fallback caused
    1000_kimi_run1 to ship to dashscope instead of ollama_cloud and burn the
    1200/h quota — fail loud is the rule now.)
    """
    if role not in _VALID_ROLES:
        raise ValueError(f"unknown role: {role}")

    # Load + resolve config BEFORE the cache lookup so a direct config.yaml swap
    # (model/provider/adapter) re-keys the cache. The same resolved cfg is reused
    # for adapter construction below — no double load. (2026-06-22 audit fix.)
    cfg = load_config(config_path)
    if role == "chat":
        cfg = _apply_chat_model_override(cfg)
    elif role == "router":
        # The router role is the lightweight routing classifier. Its model mirrors
        # the `encoder` role (the canonical fast/cheap role) so the user configures
        # only ONE fast model — routing automatically follows whatever encoder they
        # run, never a model pinned by name that they may lack credentials for. The
        # standalone roles.router config is honored only as a fallback when no encoder
        # is configured. (Jun, 2026-06-24: "don't pin a model, follow the encoder.")
        _roles = cfg.get("roles") or {}
        _encoder_spec = _roles.get("encoder")
        if _encoder_spec and _roles.get("router") != _encoder_spec:
            cfg = dict(cfg)
            cfg["roles"] = {**_roles, "router": _encoder_spec}
    cache_key = (
        role,
        config_path,
        _chat_model_override_cache_token(role),
        _role_spec_fingerprint(role, cfg),
    )
    current_router = _get_provider_router()
    cached = _CACHE.get(cache_key)
    if cached is not None and _cache_entry_is_valid(cached, current_router):
        return cached

    with _CACHE_LOCK:
        current_router = _get_provider_router()
        cached = _CACHE.get(cache_key)
        if cached is not None and _cache_entry_is_valid(cached, current_router):
            return cached
        if cached is not None:
            _CACHE.pop(cache_key, None)

        try:
            agent_sdk = _role_agent_sdk_model(role, cfg)
            if agent_sdk is not None:
                # Claude subscription credit path: delegate the turn to the Agent
                # SDK. Bypasses ProviderRouter/LiteLLM (raw API would not draw on
                # the Agent SDK credit pool). Lazy import keeps the package optional.
                from .anthropic_agent_sdk import AnthropicAgentSDKAdapter

                _provider_name, model_name, provider_cfg = agent_sdk
                llm = AnthropicAgentSDKAdapter(model=model_name, config=provider_cfg)
            elif _role_uses_oauth(role, cfg):
                # OAuth needs the router's token manager and Codex Responses
                # adapter. The direct HTTP adapter accepts API keys only.
                router = current_router
                if router is None:
                    raise RuntimeError(
                        f"roles.{role} uses OAuth, but ProviderRouter is not initialized"
                    )
                alias = _resolve_role_alias(role, cfg, router)
                # Encoder and router are both fast, reasoning-off classification
                # calls: disable reasoning and cap the codex stream/call timeouts so
                # neither blocks the turn. (router mirrors the encoder model above.)
                extra_kwargs = (
                    {
                        "reasoning_effort": "none",
                        "codex_stream_timeout_sec": 25.0,
                        "codex_call_timeout_sec": 90.0,
                    }
                    if role in ("encoder", "router")
                    else None
                )
                llm = LLMRouterAdapter(router, alias, extra_kwargs=extra_kwargs)
            else:
                # Keep API-key providers on the direct streaming path so
                # reasoning_content is preserved without LiteLLM rewriting.
                llm = _build_legacy(role, cfg)
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(
                f"provider config error for role={role} ({exc}). "
                "Refusing silent fallback. Fix config or env, then retry."
            ) from exc

        _CACHE[cache_key] = llm
        return llm


def _cache_entry_is_valid(cached: ProviderAdapter, current_router: Any) -> bool:
    """Cache entry is invalidated when the ProviderRouter registration state
    flipped between cache time and call time.

    If get_llm fires BEFORE sidecar router registration we cache the legacy adapter;
    once the router lands, that legacy entry is stale (would route through
    OpenAICompatibleAdapter forever, never reaching OAuth / KeyPool). Detect
    by comparing what we cached against the router state at call time:
      - cached LLMRouterAdapter + no current router  → stale (router torn down)
      - cached legacy adapter + current router exists → stale (router showed up)
    Either flip drops the cache entry on the next call so the new path wins.
    """
    cached_is_router = isinstance(cached, LLMRouterAdapter)
    if cached_is_router:
        return current_router is not None and cached.router is current_router  # type: ignore[attr-defined]
    return current_router is None


def clear_cache() -> None:
    """Drop all cached adapters. Mainly for tests; safe to call at runtime."""
    with _CACHE_LOCK:
        _CACHE.clear()
