from __future__ import annotations

import json as _json
import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from jlc_agentic.key_pool import AllKeysDisabledError, KeyPool
from jlc_agentic.openai_oauth import TokenManager
from jlc_agentic.user_agent import with_jarvis_user_agent


class OAuthTokenError(RuntimeError):
    """OAuth token retrieval/refresh failed for a provider."""

log = logging.getLogger(__name__)


_OAUTH_DUMMY_API_KEY = "sk-jarvis-code-oauth-dummy"
_FALLBACK_RED = "\033[91m"
_FALLBACK_RESET = "\033[0m"
_LITELLM: Any | None = None
_DEFAULT_CODEX_STREAM_TIMEOUT_SEC = 120.0

# OpenAI Responses reasoning.effort support is model-specific. Older models
# top out at "high" while GPT-5.6 accepts xhigh/max/ultra. "none" is the encoder's
# non-reasoning sentinel and passes through unchanged.
_OPENAI_EFFORT_CLAMP = {"xhigh": "high", "ultra": "high", "max": "high"}


def _is_gpt56_family(model_id: str | None) -> bool:
    mid = str(model_id or "").strip().lower().split("/", 1)[-1]
    return mid == "gpt-5.6" or mid.startswith("gpt-5.6-")


def _openai_model_supports_ultra(model: str | None) -> bool:
    return _is_gpt56_family(model)


def _codex_reasoning_effort(
    kwargs: dict[str, Any], model: str | None = None
) -> str:
    """Effective reasoning effort for a Codex (OpenAI Responses) call.

    Precedence: explicit ``reasoning_effort`` kwarg > the user-selected effort
    parked on ``turn_context`` (set by ChatTurn.run, read on this same worker
    thread) > "high" (the historical default). Clamped to the Responses enum's
    ceiling so heavier JLC efforts degrade gracefully. GPT-5.6 keeps its
    explicit xhigh/ultra values instead of taking the legacy clamp.
    """
    raw = kwargs.get("reasoning_effort")
    if not raw:
        from jlc_agentic.providers import turn_context  # noqa: PLC0415

        raw = turn_context.get().get("reasoning_effort")
    eff = str(raw or "high").strip().lower()
    if eff == "off":
        eff = "none"
    if _openai_model_supports_ultra(model) and eff in {"xhigh", "max", "ultra"}:
        return eff
    return _OPENAI_EFFORT_CLAMP.get(eff, eff)


class _LiteLLMProxy:
    def completion(self, **kwargs: Any) -> Any:
        return _get_litellm().completion(**kwargs)

    def responses(self, **kwargs: Any) -> Any:
        return _get_litellm().responses(**kwargs)


litellm = _LiteLLMProxy()


def _get_litellm() -> Any:
    global _LITELLM
    if _LITELLM is None:
        try:
            import litellm as imported_litellm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "litellm package not importable. Install sidecar requirements."
            ) from exc
        _LITELLM = imported_litellm
    return _LITELLM


@dataclass
class ResolvedModel:
    alias: str
    provider: str
    litellm_id: str
    api_key: str | None
    api_base: str | None
    cost_in_per_1m: float
    cost_out_per_1m: float
    extra_headers: dict[str, str] = field(default_factory=dict)
    is_oauth: bool = False
    oauth_provider: str | None = None


class ProviderRouter:
    def __init__(self, config: dict[str, Any], key_pool: KeyPool) -> None:
        self.config = config
        self.key_pool = key_pool
        self._models = self._index_models(config)
        self._oauth_token_managers: dict[Path, TokenManager] = {}
        # session_id is fixed per-router-instance (conversation), not per-call
        # so multi-turn requests share session continuity on the codex backend.
        self._oauth_session_id = str(uuid.uuid4())
        self.last_stream_meta: dict[str, Any] | None = None

    def resolve(self, alias: str) -> ResolvedModel:
        resolved_alias = self._resolve_alias(alias)
        resolved = self._build_resolved(resolved_alias, block=True)
        if resolved is None:
            raise ValueError(f"No API key available for alias: {alias}")
        return resolved

    def call(self, alias: str, messages: list[Any], **kwargs: Any) -> dict[str, Any]:
        attempts = self._call_aliases(alias)[:3]
        last_error: BaseException | None = None
        skipped = 0

        for fallback_attempts, candidate in enumerate(attempts):
            try:
                resolved = self._build_resolved(
                    candidate,
                    block=fallback_attempts == 0,
                )
            except AllKeysDisabledError as exc:
                last_error = exc
                log.warning(
                    "Skipping alias=%s because all provider keys are disabled",
                    candidate,
                )
                continue
            except OAuthTokenError as exc:
                last_error = exc
                log.warning(
                    "Skipping OAuth alias=%s: %s",
                    candidate,
                    exc,
                )
                continue
            if resolved is None:
                skipped += 1
                log.warning("Skipping alias=%s because no API key is currently available", candidate)
                continue
            started = time.perf_counter()
            try:
                if resolved.is_oauth and resolved.oauth_provider == "chatgpt":
                    from jlc_agentic.codex_responses_adapter import (
                        chat_messages_to_responses_input,
                        chat_tools_to_responses_tools,
                        responses_to_chat_completion,
                        split_instructions_and_input,
                    )
                    bare_model = resolved.litellm_id.split("/", 1)[-1]
                    instructions, non_system = split_instructions_and_input(messages)
                    response_tools = chat_tools_to_responses_tools(kwargs.get("tools"))
                    reasoning_effort = _codex_reasoning_effort(kwargs, bare_model)
                    response_body = {
                        "input": chat_messages_to_responses_input(non_system),
                        "instructions": instructions,
                        "model": bare_model,
                        "reasoning": {"effort": reasoning_effort},
                        "store": False,
                        "stream": False,
                    }
                    if reasoning_effort != "none":
                        response_body["reasoning"]["summary"] = "detailed"
                        response_body["include"] = ["reasoning.encrypted_content"]
                    if response_tools:
                        response_body["tools"] = response_tools
                    timeout_sec = _coerce_timeout_sec(
                        kwargs.get("codex_call_timeout_sec"),
                        default=_DEFAULT_CODEX_STREAM_TIMEOUT_SEC,
                    )
                    response_obj = self._post_codex_responses(
                        resolved, response_body, timeout_sec=timeout_sec
                    )
                    response = responses_to_chat_completion(response_obj, bare_model)
                    _warn_if_model_fallback(
                        requested_model=bare_model,
                        response_model=_response_model(response),
                        alias=resolved.alias,
                        provider=resolved.provider,
                    )
                else:
                    if resolved.is_oauth:
                        # Defense-in-depth: only chatgpt OAuth has a wired
                        # adapter. Other oauth_provider values must not leak
                        # the dummy api_key into a real keyed call path.
                        raise OAuthTokenError(
                            f"Unsupported oauth_provider={resolved.oauth_provider!r}; "
                            "no adapter registered."
                        )
                    call_kwargs = {
                        "model": resolved.litellm_id,
                        "messages": messages,
                        "api_key": resolved.api_key,
                        "api_base": resolved.api_base,
                        **kwargs,
                    }
                    if resolved.extra_headers:
                        call_kwargs["extra_headers"] = resolved.extra_headers
                    response = litellm.completion(**call_kwargs)
                    _warn_if_model_fallback(
                        requested_model=resolved.litellm_id,
                        response_model=_response_model(response),
                        alias=resolved.alias,
                        provider=resolved.provider,
                    )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status_code = self._status_code(exc)
                if status_code is None and isinstance(exc, (TimeoutError, ConnectionError)):
                    status_code = 503
                if (
                    status_code is not None
                    and resolved.api_key is not None
                    and not resolved.is_oauth
                ):
                    self.key_pool.report_failure(resolved.provider, resolved.api_key, status_code)
                log.warning(
                    "LiteLLM call failed for alias=%s provider=%s status=%s",
                    resolved.alias,
                    resolved.provider,
                    status_code,
                )
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)
            if resolved.api_key is not None and not resolved.is_oauth:
                self.key_pool.report_success(resolved.provider, resolved.api_key)
            return {
                "response": response,
                "llm_meta": self._llm_meta(resolved, response, latency_ms, fallback_attempts),
            }

        if last_error is not None:
            message = str(last_error)
        elif skipped:
            message = "all fallback candidates skipped because no API key was available"
        else:
            message = "no fallback candidates configured"
        raise RuntimeError(f"ProviderRouter call failed after {len(attempts)} attempts: {message}")

    def stream_call(
        self,
        alias: str,
        messages: list[Any],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Generator yielding chat-completion-shaped dict chunks.

        After full iteration, ``self.last_stream_meta`` carries provider
        info. Pre-first-chunk failures fall back through ``_call_aliases``;
        once any chunk has been yielded, mid-stream failures propagate
        (we cannot safely retry on a different provider after partial
        output has reached the UI).
        """
        self.last_stream_meta = None
        attempts = self._call_aliases(alias)[:3]
        last_error: BaseException | None = None
        skipped = 0

        for fallback_attempts, candidate in enumerate(attempts):
            try:
                resolved = self._build_resolved(
                    candidate,
                    block=fallback_attempts == 0,
                )
            except AllKeysDisabledError as exc:
                last_error = exc
                log.warning(
                    "Skipping alias=%s because all provider keys are disabled",
                    candidate,
                )
                continue
            except OAuthTokenError as exc:
                last_error = exc
                log.warning("Skipping OAuth alias=%s: %s", candidate, exc)
                continue
            if resolved is None:
                skipped += 1
                log.warning(
                    "Skipping alias=%s because no API key is currently available",
                    candidate,
                )
                continue

            started = time.perf_counter()
            yielded_any = False
            stream_usage: dict[str, Any] | None = None
            try:
                for chunk in self._stream_attempt(resolved, messages, **kwargs):
                    yielded_any = True
                    if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
                        stream_usage = chunk["usage"]
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status_code = self._status_code(exc)
                if status_code is None and isinstance(
                    exc, (TimeoutError, ConnectionError)
                ):
                    status_code = 503
                if (
                    status_code is not None
                    and resolved.api_key is not None
                    and not resolved.is_oauth
                ):
                    self.key_pool.report_failure(
                        resolved.provider, resolved.api_key, status_code
                    )
                log_fn = log.info if yielded_any and kwargs.get("allow_partial_stream") else log.warning
                log_fn(
                    "Streaming provider call failed for alias=%s provider=%s status=%s yielded_any=%s",
                    resolved.alias,
                    resolved.provider,
                    status_code,
                    yielded_any,
                )
                if yielded_any:
                    raise
                continue

            latency_ms = int((time.perf_counter() - started) * 1000)
            if resolved.api_key is not None and not resolved.is_oauth:
                self.key_pool.report_success(
                    resolved.provider, resolved.api_key
                )
            meta_response = {"usage": stream_usage} if stream_usage is not None else None
            self.last_stream_meta = self._llm_meta(
                resolved, meta_response, latency_ms, fallback_attempts
            )
            return

        if last_error is not None:
            raise last_error
        if skipped:
            raise RuntimeError(
                "all fallback candidates skipped because no API key was available"
            )
        raise RuntimeError("no fallback candidates configured")

    def _stream_attempt(
        self,
        resolved: ResolvedModel,
        messages: list[Any],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Yield dict chunks from a single provider attempt."""
        if resolved.is_oauth and resolved.oauth_provider == "chatgpt":
            from jlc_agentic.codex_responses_adapter import (
                chat_messages_to_responses_input,
                chat_tools_to_responses_tools,
                iter_responses_stream_chunks,
                split_instructions_and_input,
            )
            bare_model = resolved.litellm_id.split("/", 1)[-1]
            instructions, non_system = split_instructions_and_input(messages)
            response_tools = chat_tools_to_responses_tools(kwargs.get("tools"))
            reasoning_effort = _codex_reasoning_effort(kwargs, bare_model)
            response_body = {
                "input": chat_messages_to_responses_input(non_system),
                "instructions": instructions,
                "model": bare_model,
                "reasoning": {"effort": reasoning_effort},
                "store": False,
                "stream": True,
            }
            if reasoning_effort != "none":
                response_body["reasoning"]["summary"] = "detailed"
                response_body["include"] = ["reasoning.encrypted_content"]
            if response_tools:
                response_body["tools"] = response_tools
            timeout_sec = _coerce_timeout_sec(
                kwargs.get("codex_stream_timeout_sec"),
                default=_DEFAULT_CODEX_STREAM_TIMEOUT_SEC,
            )
            stream = self._stream_codex_responses(resolved, response_body, timeout_sec=timeout_sec)
            fallback_checked = False
            no_model_logged = False
            for chunk in iter_responses_stream_chunks(stream, bare_model):
                response_model = _response_model(chunk)
                if response_model:
                    if not fallback_checked:
                        _warn_if_model_fallback(
                            requested_model=bare_model,
                            response_model=response_model,
                            alias=resolved.alias,
                            provider=resolved.provider,
                        )
                        fallback_checked = True
                elif not no_model_logged:
                    _warn_missing_model_field()
                    no_model_logged = True
                yield chunk
            return

        if resolved.is_oauth:
            raise OAuthTokenError(
                f"Unsupported oauth_provider={resolved.oauth_provider!r}; "
                "no streaming adapter registered."
            )

        # Direct httpx SSE streaming for openai-compat providers. We skip
        # litellm here because its pydantic ModelResponseStream schema
        # silently drops `reasoning_content` from the delta, breaking the
        # BLUE reasoning channel that AgenticLoop already expects (and that
        # the encoder's LLMClient httpx path never lost). Mirrors the
        # encoder pattern in jlc_agentic/llm_client.py:160-211.
        yield from self._stream_openai_compat(resolved, messages, **kwargs)

    def _stream_codex_responses(
        self,
        resolved: ResolvedModel,
        body: dict[str, Any],
        *,
        timeout_sec: float = _DEFAULT_CODEX_STREAM_TIMEOUT_SEC,
    ) -> Iterator[dict[str, Any]]:
        """Stream ChatGPT OAuth Responses API events without LiteLLM."""
        if not resolved.api_base:
            raise RuntimeError(
                f"provider {resolved.provider!r} has no api_base; cannot stream"
            )
        url = f"{resolved.api_base.rstrip('/')}/responses"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        headers.update(resolved.extra_headers)
        headers = with_jarvis_user_agent(headers)
        with httpx.stream(
            "POST", url, headers=headers, json=body, timeout=timeout_sec
        ) as resp:
            if resp.status_code >= 400:
                detail = resp.read()[:500]
                raise httpx.HTTPStatusError(
                    f"{resolved.provider} HTTP {resp.status_code}: {detail!r}",
                    request=resp.request,
                    response=resp,
                )
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield _json.loads(data)
                except _json.JSONDecodeError:
                    continue

    def _post_codex_responses(
        self,
        resolved: ResolvedModel,
        body: dict[str, Any],
        *,
        timeout_sec: float = _DEFAULT_CODEX_STREAM_TIMEOUT_SEC,
    ) -> dict[str, Any]:
        """Call ChatGPT OAuth Responses API without SSE streaming."""
        if not resolved.api_base:
            raise RuntimeError(
                f"provider {resolved.provider!r} has no api_base; cannot call"
            )
        url = f"{resolved.api_base.rstrip('/')}/responses"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(resolved.extra_headers)
        headers = with_jarvis_user_agent(headers)
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout_sec)
        if resp.status_code >= 400:
            detail = resp.content[:500]
            raise httpx.HTTPStatusError(
                f"{resolved.provider} HTTP {resp.status_code}: {detail!r}",
                request=resp.request,
                response=resp,
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"{resolved.provider} returned non-JSON Responses payload"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"{resolved.provider} returned unexpected Responses payload"
            )
        return data

    def _stream_openai_compat(
        self,
        resolved: ResolvedModel,
        messages: list[Any],
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Sync SSE stream from an OpenAI-compatible /chat/completions endpoint."""
        if not resolved.api_base:
            raise RuntimeError(
                f"provider {resolved.provider!r} has no api_base; cannot stream"
            )
        url = f"{resolved.api_base.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if resolved.api_key:
            headers["Authorization"] = f"Bearer {resolved.api_key}"
        if resolved.extra_headers:
            headers.update(resolved.extra_headers)
        headers = with_jarvis_user_agent(headers)
        # Strip the litellm `openai/` prefix so the wire model name matches
        # what the upstream provider expects (e.g. 'kimi-k2.5:cloud').
        bare_model = resolved.litellm_id.split("/", 1)[-1]
        body: dict[str, Any] = {
            "model": bare_model,
            "messages": messages,
            "stream": True,
        }
        for key in (
            "tools",
            "parallel_tool_calls",
            "tool_choice",
            "temperature",
            "max_tokens",
            "top_p",
            "reasoning_effort",
        ):
            if key in kwargs:
                body[key] = kwargs[key]
        extra_body = kwargs.get("extra_body")
        if isinstance(extra_body, dict):
            body.update(extra_body)
        with httpx.stream(
            "POST", url, headers=headers, json=body, timeout=120.0
        ) as resp:
            if resp.status_code >= 400:
                detail = resp.read()[:500]
                raise httpx.HTTPStatusError(
                    f"{resolved.provider} HTTP {resp.status_code}: {detail!r}",
                    request=resp.request,
                    response=resp,
                )
            fallback_checked = False
            no_model_logged = False
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = _json.loads(data)
                except _json.JSONDecodeError:
                    continue
                response_model = _response_model(obj)
                if response_model:
                    if not fallback_checked:
                        _warn_if_model_fallback(
                            requested_model=bare_model,
                            response_model=response_model,
                            alias=resolved.alias,
                            provider=resolved.provider,
                        )
                        fallback_checked = True
                elif not no_model_logged:
                    _warn_missing_model_field()
                    no_model_logged = True
                yield obj

    def _resolve_alias(self, alias: str) -> str:
        if alias.startswith("tier:"):
            tier = alias.split(":", 1)[1]
            for rule in self.config.get("routing", {}).get("rules", []):
                when = rule.get("when", {})
                if when.get("tier") == tier:
                    uses = rule.get("use") or []
                    if uses:
                        return uses[0]
            raise KeyError(alias)

        if alias not in self._models:
            raise KeyError(alias)
        return alias

    def _call_aliases(self, alias: str) -> list[str]:
        chain = self._routing_aliases(alias)
        for fallback in self.config.get("defaults", {}).get("fallback", []) or []:
            try:
                resolved_fallback = self._resolve_alias(fallback)
            except KeyError:
                log.warning("Skipping invalid fallback alias: %s", fallback)
                continue
            fallback = resolved_fallback
            if fallback not in chain:
                chain.append(fallback)
        return chain

    def _routing_aliases(self, alias: str) -> list[str]:
        if not alias.startswith("tier:"):
            return [self._resolve_alias(alias)]

        tier = alias.split(":", 1)[1]
        for rule in self.config.get("routing", {}).get("rules", []):
            when = rule.get("when", {})
            if when.get("tier") == tier:
                aliases = []
                for candidate in rule.get("use") or []:
                    try:
                        aliases.append(self._resolve_alias(candidate))
                    except KeyError:
                        log.warning("Skipping invalid routing alias: %s", candidate)
                if aliases:
                    return aliases
        raise KeyError(alias)

    def _build_resolved(self, alias: str, block: bool) -> ResolvedModel | None:
        provider_name, provider, model = self._models[alias]

        if provider.get("oauth_provider"):
            return self._build_resolved_oauth(alias, provider_name, provider, model)

        api_keys = provider.get("api_keys")
        if api_keys == []:
            if not block:
                return None
            raise ValueError(f"Provider {provider_name} has no API keys configured")

        api_key: str | None = None
        if api_keys is not None:
            usable_keys = [key for key in api_keys if key]
            if not usable_keys:
                if not block:
                    return None
                raise ValueError(f"Provider {provider_name} has no API keys configured")
            api_key = self.key_pool.take(provider_name, block=block)
            if api_key is None:
                return None

        extra_headers = {}
        extra_headers.update(provider.get("extra_headers") or {})
        extra_headers.update(model.get("extra_headers") or {})

        return ResolvedModel(
            alias=alias,
            provider=provider_name,
            litellm_id=model["litellm_id"],
            api_key=api_key,
            api_base=model.get("api_base") or provider.get("api_base"),
            cost_in_per_1m=float(model.get("cost_in_per_1m", 0.0)),
            cost_out_per_1m=float(model.get("cost_out_per_1m", 0.0)),
            extra_headers=with_jarvis_user_agent(extra_headers),
        )

    def _build_resolved_oauth(
        self,
        alias: str,
        provider_name: str,
        provider: dict[str, Any],
        model: dict[str, Any],
    ) -> ResolvedModel:
        token_path = Path(
            provider.get("oauth_token_path") or "~/.jarvis-code/auth.json"
        ).expanduser()
        mgr = self._token_manager(token_path)
        try:
            access_token = mgr.get_access_token()
        except RuntimeError as exc:
            raise OAuthTokenError(str(exc)) from exc
        account_id = mgr.get_account_id() or ""

        extra_headers: dict[str, Any] = {}
        extra_headers.update(provider.get("extra_headers") or {})
        extra_headers.update(model.get("extra_headers") or {})
        extra_headers["Authorization"] = f"Bearer {access_token}"
        extra_headers["ChatGPT-Account-Id"] = account_id
        extra_headers["originator"] = "jarvis-code"
        extra_headers["session_id"] = self._oauth_session_id

        return ResolvedModel(
            alias=alias,
            provider=provider_name,
            litellm_id=model["litellm_id"],
            api_key=_OAUTH_DUMMY_API_KEY,
            api_base=provider.get("api_base"),
            cost_in_per_1m=float(model.get("cost_in_per_1m", 0.0)),
            cost_out_per_1m=float(model.get("cost_out_per_1m", 0.0)),
            extra_headers=with_jarvis_user_agent(extra_headers),
            is_oauth=True,
            oauth_provider=provider.get("oauth_provider"),
        )

    def _token_manager(self, path: Path) -> TokenManager:
        cached = self._oauth_token_managers.get(path)
        if cached is None:
            cached = TokenManager(path)
            self._oauth_token_managers[path] = cached
        return cached

    def _llm_meta(
        self,
        resolved: ResolvedModel,
        response: Any,
        latency_ms: int,
        fallback_attempts: int,
    ) -> dict[str, Any]:
        usage = self._usage(response)
        tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cache_read_tokens = self._usage_nested_int(
            usage,
            ("cache_read_tokens",),
            ("cached_tokens",),
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
        )
        cache_write_tokens = self._usage_nested_int(
            usage,
            ("cache_write_tokens",),
            ("cache_creation_input_tokens",),
            ("prompt_tokens_details", "cache_creation_tokens"),
            ("input_tokens_details", "cache_creation_tokens"),
        )
        reasoning_tokens = self._usage_nested_int(
            usage,
            ("reasoning_tokens",),
            ("completion_tokens_details", "reasoning_tokens"),
            ("output_tokens_details", "reasoning_tokens"),
        )
        total_tokens = self._usage_nested_int(usage, ("total_tokens",), ("total",))
        if total_tokens is None:
            total_tokens = tokens_in + tokens_out + (cache_read_tokens or 0) + (cache_write_tokens or 0)
        if tokens_in or tokens_out or total_tokens:
            if cache_read_tokens is None:
                cache_read_tokens = 0
            if cache_write_tokens is None:
                cache_write_tokens = 0
        cost_usd = (
            tokens_in * resolved.cost_in_per_1m / 1_000_000
            + tokens_out * resolved.cost_out_per_1m / 1_000_000
        )
        return {
            "alias": resolved.alias,
            "provider": resolved.provider,
            "litellm_id": resolved.litellm_id,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "usage": usage,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "key_idx": self._key_index(resolved),
            "fallback_attempts": fallback_attempts,
        }

    def _key_index(self, resolved: ResolvedModel) -> int | None:
        if resolved.is_oauth or resolved.api_key is None:
            return None
        provider = self.config.get("providers", {}).get(resolved.provider, {})
        keys = provider.get("api_keys") or []
        try:
            return keys.index(resolved.api_key)
        except ValueError:
            return None

    def find_alias(self, provider_name: str, model_name: str) -> str | None:
        """Look up the providers.yaml model alias for a (provider, model_name) pair.

        Used by the role-routing layer (config.yaml roles: 'provider/model')
        to map a user-friendly 'company/model' string into the alias key the
        router uses internally. Match order:
          1. exact alias key match under the provider
          2. litellm_id suffix match (e.g. 'openai/gpt-5.4' → 'gpt-5.4')
        Returns None when no model under the named provider matches.
        """
        for alias, (pname, _provider, model) in self._models.items():
            if pname != provider_name:
                continue
            if alias == model_name:
                return alias
            litellm_id = str(model.get("litellm_id") or "")
            suffix = litellm_id.split("/", 1)[1] if "/" in litellm_id else litellm_id
            if suffix == model_name:
                return alias
        return None

    @staticmethod
    def _index_models(config: dict[str, Any]) -> dict[str, tuple[str, dict[str, Any], dict[str, Any]]]:
        indexed: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
        for provider_name, provider in (config.get("providers") or {}).items():
            for alias, model in (provider.get("models") or {}).items():
                if alias in indexed:
                    existing_provider = indexed[alias][0]
                    raise ValueError(
                        f"Duplicate model alias '{alias}' "
                        f"in providers '{existing_provider}' and '{provider_name}'"
                    )
                indexed[alias] = (provider_name, provider, model)
        return indexed

    @staticmethod
    def _usage(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            usage = response.get("usage") or {}
            return usage if isinstance(usage, dict) else {}
        usage = getattr(response, "usage", {})
        if isinstance(usage, dict):
            return usage
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }

    @staticmethod
    def _usage_nested_int(usage: dict[str, Any], *paths: tuple[str, ...]) -> int | None:
        for path in paths:
            current: Any = usage
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is None:
                continue
            try:
                return max(0, int(current))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        for attr in ("status_code", "status"):
            value = getattr(exc, attr, None)
            if isinstance(value, int):
                return value
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
        return value if isinstance(value, int) else None


def _response_model(response: Any) -> str | None:
    if isinstance(response, dict):
        value = response.get("model")
    else:
        value = getattr(response, "model", None)
    return value if isinstance(value, str) and value else None


def _normalize_model_id(model: str) -> str:
    return model.split("/", 1)[-1].strip().lower()


def _model_matches_request(requested_model: str, response_model: str) -> bool:
    requested = _normalize_model_id(requested_model)
    actual = _normalize_model_id(response_model)
    return actual == requested or actual.startswith(requested)


def _warn_if_model_fallback(
    *,
    requested_model: str,
    response_model: str | None,
    alias: str,
    provider: str,
) -> None:
    if response_model is None:
        _warn_missing_model_field()
        return
    if _model_matches_request(requested_model, response_model):
        return
    try:
        sys.stderr.write(
            f"{_FALLBACK_RED}[fallback] requested={requested_model} actual={response_model} "
            f"alias={alias} provider={provider}{_FALLBACK_RESET}\n"
        )
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _warn_missing_model_field() -> None:
    if os.environ.get("JLC_MODEL_CHECK_VERBOSE", "0") != "1":
        return
    try:
        sys.stderr.write("[model-check] response has no model field; fallback check skipped\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def _coerce_timeout_sec(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _litellm_chunk_to_dict(chunk: Any) -> dict[str, Any]:
    """Normalize a litellm streaming chunk to a plain dict.

    AgenticLoop reads chunk["choices"][0]["delta"].{content,reasoning,
    tool_calls} via dict semantics, so we coerce ModelResponseStream
    pydantic objects into dicts up front. Already-dict chunks pass
    through unchanged.
    """
    if isinstance(chunk, dict):
        return chunk
    dump = getattr(chunk, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=False)
        except Exception:  # noqa: BLE001
            pass
    choices = getattr(chunk, "choices", None) or []
    out_choices: list[dict[str, Any]] = []
    for ch in choices:
        delta = getattr(ch, "delta", None)
        delta_dict: dict[str, Any] = {}
        if delta is not None:
            for key in ("content", "reasoning", "reasoning_content", "tool_calls"):
                value = getattr(delta, key, None)
                if value is not None:
                    delta_dict[key] = value
        out_choices.append(
            {
                "delta": delta_dict,
                "finish_reason": getattr(ch, "finish_reason", None),
            }
        )
    return {"choices": out_choices}
