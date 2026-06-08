"""OpenAI-compatible async client with optional injected completion client."""
from __future__ import annotations

import asyncio
import re
import sys
from typing import Any

import httpx

from .config import ProviderConfig


class LLMClient:
    """Provider-ordered LLM client with retry and fallback."""

    def __init__(
        self,
        providers: list[ProviderConfig],
        completion_client: Any | None = None,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        self.providers = providers
        self._completion_client = completion_client
        # Optional payload kwargs merged into every chat completion call (e.g.
        # `{"reasoning_effort": "none"}` for the encoder so we don't pay for
        # thinking tokens on a structured-compression task). Unknown keys are
        # dropped server-side by OpenAI-compatible endpoints.
        self._extra_payload = dict(extra_payload) if extra_payload else {}
        # AsyncClient is bound to the event loop that created it. JLC runs the
        # encoder from a thread fallback (asyncio.run inside a daemon thread)
        # which produces a fresh loop per call — reusing a client across loops
        # raises "Event loop is closed". Cache one client per loop instead.
        self._client_by_loop: "dict[int, httpx.AsyncClient]" = {}

    def _get_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        key = id(loop)
        client = self._client_by_loop.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0),
            )
            self._client_by_loop[key] = client
        return client

    async def close(self) -> None:
        for client in list(self._client_by_loop.values()):
            try:
                await client.aclose()
            except Exception:
                pass
        self._client_by_loop.clear()

    async def chat(self, system: str, user: str, max_tokens: int = 2048, on_chunk: Any = None) -> str:
        """Run a chat completion. If `on_chunk` callable is given, the request
        is made in streaming mode and each text delta is forwarded to it as
        soon as it arrives (also accumulated and returned in full at the end).
        """
        if self._completion_client is not None:
            content = await self._chat_via_injected_client(system, user, max_tokens)
            return self._strip_markdown_fences(content)

        if not self.providers:
            raise RuntimeError("No LLM providers configured")

        errors: list[str] = []
        for provider in self.providers:
            if not provider.api_key and not self._is_local_provider(provider.base_url):
                message = f"missing env {provider.api_key_env}"
                errors.append(f"{provider.name}: {message}")
                print(f"[jlc:llm] provider skipped: {provider.name}: {message}", file=sys.stderr)
                continue
            try:
                content = await self._call_with_retry(provider, system, user, max_tokens, on_chunk)
                return self._strip_markdown_fences(content)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider.name}: {exc}")
                print(f"[jlc:llm] provider failed: {provider.name}: {exc}", file=sys.stderr)

        raise RuntimeError(f"All providers failed: {'; '.join(errors)}")

    async def _chat_via_injected_client(self, system: str, user: str, max_tokens: int) -> str:
        client = self._completion_client
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }

        if hasattr(client, "acompletion"):
            res = await client.acompletion(**payload)
        elif hasattr(client, "completion"):
            maybe = client.completion(**payload)
            if asyncio.iscoroutine(maybe):
                res = await maybe
            else:
                res = maybe
        elif callable(client):
            maybe = client(**payload)
            if asyncio.iscoroutine(maybe):
                res = await maybe
            else:
                res = maybe
        else:
            raise RuntimeError("Injected completion client must be callable or provide completion/acompletion")

        if isinstance(res, dict):
            return self._extract_content(res)
        if hasattr(res, "model_dump"):
            return self._extract_content(res.model_dump())
        raise RuntimeError("Injected completion client returned unsupported response type")

    async def _call_with_retry(self, provider: ProviderConfig, system: str, user: str, max_tokens: int, on_chunk: Any = None) -> str:
        backoff = 30.0
        max_backoff = 300.0
        max_attempts = 8
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            try:
                return await self._post_chat_completion(provider, system, user, max_tokens, on_chunk)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    ra = exc.response.headers.get("retry-after", "")
                    if ra.strip().isdigit():
                        backoff = max(backoff, float(ra.strip()))
                print(f"[jlc:llm] {provider.name} attempt {attempt}/{max_attempts} HTTP {status}, retry in {backoff:.0f}s", file=sys.stderr)
            except (httpx.HTTPError, asyncio.TimeoutError, ValueError) as exc:
                print(f"[jlc:llm] {provider.name} attempt {attempt}/{max_attempts} {type(exc).__name__}, retry in {backoff:.0f}s", file=sys.stderr)

            if attempt >= max_attempts:
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

        raise RuntimeError(f"{provider.name} exhausted {max_attempts} attempts")

    async def _post_chat_completion(self, provider: ProviderConfig, system: str, user: str, max_tokens: int, on_chunk: Any = None) -> str:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"

        payload: dict[str, Any] = {
            "model": provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": min(max_tokens, provider.max_tokens),
        }
        if self._extra_payload:
            payload.update(self._extra_payload)

        client = self._get_client()
        if on_chunk is None:
            response = await client.post(f"{provider.base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            if response.status_code >= 400:
                print(f"[jlc:llm] {provider.name} {response.status_code}: {response.text[:500]}", file=sys.stderr)
            response.raise_for_status()
            return self._extract_content(response.json())

        # Streaming path — OpenAI-compatible SSE. Forward each delta to
        # on_chunk while accumulating the full content for the return value.
        import json as _json
        payload["stream"] = True
        parts: list[str] = []
        async with client.stream(
            "POST",
            f"{provider.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                print(f"[jlc:llm] {provider.name} {response.status_code}: {body[:500]!r}", file=sys.stderr)
                response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = _json.loads(data)
                except _json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # Some servers split reasoning into a separate field — surface
                # both so the user can watch the encoder's <think> live.
                think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                content = delta.get("content") or ""
                if think:
                    # Pass kind="reasoning" so the caller can colorize the
                    # encoder's <think> separately from its answer text.
                    try:
                        on_chunk(think, "reasoning")
                    except TypeError:
                        try:
                            on_chunk(think)
                        except Exception:
                            pass
                    except Exception:
                        pass
                if content:
                    parts.append(str(content))
                    try:
                        on_chunk(content, "content")
                    except TypeError:
                        try:
                            on_chunk(content)
                        except Exception:
                            pass
                    except Exception:
                        pass
        return "".join(parts).strip()

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("response missing choices")

        first = choices[0]
        message = first.get("message", {}) if isinstance(first, dict) else {}
        content = message.get("content")
        # Tool-call responses can have content=None — caller decides what to
        # do with empty output instead of trapping the LLM in a retry loop.
        if content is None:
            return ""
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            text = "".join(parts).strip()
            if text:
                return text
        raise ValueError("response content missing")

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        stripped = text.strip()
        m = re.match(r"^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$", stripped)
        if m:
            return m.group(1).strip()
        return stripped

    @staticmethod
    def _is_local_provider(base_url: str) -> bool:
        lowered = base_url.lower()
        return lowered.startswith("http://localhost") or lowered.startswith("http://127.0.0.1")
