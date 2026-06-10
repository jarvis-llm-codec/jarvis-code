from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from jlc_agentic.user_agent import with_jarvis_user_agent

_log = logging.getLogger(__name__)


class _SilenceWatcher:
    """Background watchdog that fires on_silence(gap) when chunk-gap exceeds threshold."""

    def __init__(
        self,
        threshold_sec: float,
        on_silence: Callable[[float], str | None] | None,
        poll_interval_sec: float = 0.5,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        self.threshold_sec = threshold_sec
        self.on_silence = on_silence
        self.poll_interval_sec = poll_interval_sec
        self.on_abort = on_abort
        self._last_tick = time.monotonic()
        self._lock = threading.Lock()
        self._fired_for_gap = False
        self._stop = threading.Event()
        self._abort = False
        self._thread: threading.Thread | None = None

    def tick(self) -> None:
        with self._lock:
            self._last_tick = time.monotonic()
            self._fired_for_gap = False

    @property
    def aborted(self) -> bool:
        return self._abort

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.poll_interval_sec)
            if self._stop.is_set():
                return
            with self._lock:
                gap = time.monotonic() - self._last_tick
                already_fired = self._fired_for_gap
            if gap >= self.threshold_sec and not already_fired and self.on_silence is not None:
                with self._lock:
                    self._fired_for_gap = True
                try:
                    decision = self.on_silence(gap)
                except Exception:  # noqa: BLE001
                    _log.exception("silence callback raised; treating as no-op")
                    decision = None
                if decision == "abort":
                    self._abort = True
                    if self.on_abort is not None:
                        try:
                            self.on_abort()
                        except Exception:  # noqa: BLE001
                            pass
                    return

    def start(self) -> None:
        if self.on_silence is None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class OpenAICompatibleAdapter:
    """OpenAI-compatible streaming chat completion client."""

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
        on_token: Any | None = None,
        **kwargs: Any,
    ) -> str:
        """v1.01 encoder/subagent-facing async chat. Concatenates streamed
        content chunks into the full assistant string. When `on_token` is
        supplied each chunk is also forwarded so the UI can render live.

        Reasoning chunks (provider-specific `reasoning`/`reasoning_content`
        delta fields used by GLM/devstral on Ollama Cloud) are forwarded to
        `on_token(text, "reasoning")` but never folded into the returned
        content — encoder.py parses content-only output.
        """
        import asyncio
        if on_token is None:
            on_token = kwargs.pop("on_chunk", None)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        call_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in {"temperature", "top_p"}
        }
        # This async entry point is used for JLC encoding. Encoding is a
        # structured patch task and must not spend tokens on model reasoning.
        call_kwargs["reasoning_effort"] = "none"
        call_kwargs["max_tokens"] = max_tokens

        adapter = self

        def _run() -> str:
            chunks: list[str] = []
            for evt in adapter.stream_chat_completions(messages, stream=True, **call_kwargs):
                choices = evt.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning and on_token is not None:
                    try:
                        on_token(reasoning, "reasoning")
                    except TypeError:
                        on_token(reasoning)
                    except Exception:  # noqa: BLE001
                        pass
                piece = delta.get("content")
                if piece:
                    chunks.append(piece)
                    if on_token is not None:
                        try:
                            on_token(piece, "content")
                        except TypeError:
                            on_token(piece)
                        except Exception:  # noqa: BLE001
                            pass
            return "".join(chunks)

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        """Symmetry with LLMRouterAdapter.close — urllib is stateless so
        this is a no-op, but slim.JarvisAgentic.close() awaits it."""
        return None

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_sec: float = 600.0,
        silence_threshold_sec: float | None = None,
        on_silence: Callable[[float], str | None] | None = None,
    ) -> None:
        key = api_key
        if not key and api_key_env:
            key = os.environ.get(api_key_env)
        if not key:
            key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("API key not provided (explicit, api_key_env, or OPENAI_API_KEY)")

        self.api_key = key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec
        self.silence_threshold_sec = silence_threshold_sec
        self.on_silence = on_silence

    def stream_chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        parallel_tool_calls: bool = True,
        stream: bool = True,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        if not stream:
            raise ValueError("OpenAICompatibleAdapter currently supports stream=True only")

        body: dict[str, Any] = {
            "model": kwargs.get("model") or self.model,
            "messages": messages,
            "stream": True,
        }
        if kwargs.get("reasoning_effort"):
            sys.stderr.write(
                f"[openai_compat] reasoning_effort={kwargs.get('reasoning_effort')}\n"
            )
            sys.stderr.flush()
        if tools:
            body["tools"] = tools
            body["parallel_tool_calls"] = parallel_tool_calls
        for key, value in kwargs.items():
            if key in {"temperature", "top_p", "max_tokens", "reasoning_effort"}:
                body[key] = value

        req = Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=with_jarvis_user_agent(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                }
            ),
            method="POST",
        )

        # 2026-05-04: 1000/10000-turn bench 안정성 — transient 에러는 30초마다
        # retry (max 30분 cap), 4xx client 에러는 즉시 raise (request 자체가
        # invalid라 retry해도 풀리지 않음 → loop layer가 turn-level error로
        # fallback, L4가 client에서 다음 prompt로 진행). Retry 대상: 408 timeout,
        # 429 rate-limit (Retry-After 존중), 5xx 서버 에러, URLError (network/
        # timeout/connection reset/Cloudflare proxied 520-525). MAX_RETRIES cap은
        # 인프라 다운 시 무한 hang 방지 — cap 도달 시 RuntimeError raise하면 L4가
        # 그 turn만 error 마킹하고 진행.
        from urllib.error import HTTPError as _HTTPError
        MAX_RETRIES = 60  # 30s * 60 = 30분 cap
        _attempt = 0
        resp = None

        def _close_error_response(exc: BaseException) -> None:
            close = getattr(exc, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

        while True:
            _attempt += 1
            if _attempt > MAX_RETRIES:
                raise RuntimeError(
                    f"max retries ({MAX_RETRIES}) exceeded — provider unreachable"
                )
            try:
                resp = urlopen(req, timeout=self.timeout_sec)
                break
            except _HTTPError as exc:
                if exc.code == 429:
                    # 429: provider may specify Retry-After header (seconds).
                    retry_after = exc.headers.get("Retry-After") if hasattr(exc, "headers") else None
                    try:
                        wait_s = int(retry_after) if retry_after else 30
                    except (TypeError, ValueError):
                        wait_s = 30
                    sys.stderr.write(
                        f"[openai_compat] HTTP 429 on attempt {_attempt}/{MAX_RETRIES}, "
                        f"sleeping {wait_s}s (Retry-After={retry_after})...\n"
                    )
                    sys.stderr.flush()
                    _close_error_response(exc)
                    time.sleep(wait_s)
                    continue
                if exc.code in (408, 500, 502, 503, 504):
                    # 408 timeout / 5xx server errors — retryable.
                    sys.stderr.write(
                        f"[openai_compat] HTTP {exc.code} on attempt {_attempt}/{MAX_RETRIES}, "
                        f"sleeping 30s before retry...\n"
                    )
                    sys.stderr.flush()
                    _close_error_response(exc)
                    time.sleep(30)
                    continue
                # Other 4xx client errors — request invalid, retry pointless.
                raise RuntimeError(f"request failed: {exc}") from exc
            except URLError as exc:
                sys.stderr.write(
                    f"[openai_compat] URLError ({exc}) on attempt {_attempt}/{MAX_RETRIES}, "
                    f"sleeping 30s before retry...\n"
                )
                sys.stderr.flush()
                time.sleep(30)
                continue

        def _close_socket() -> None:
            try:
                resp.close()
            except Exception:  # noqa: BLE001
                pass

        watcher: _SilenceWatcher | None = None
        if self.silence_threshold_sec and self.on_silence is not None:
            poll = max(0.02, min(self.silence_threshold_sec / 4.0, 1.0))
            watcher = _SilenceWatcher(
                threshold_sec=self.silence_threshold_sec,
                on_silence=self.on_silence,
                on_abort=_close_socket,
                poll_interval_sec=poll,
            )
            watcher.start()

        try:
            with resp:
                for raw in resp:
                    if watcher is not None:
                        watcher.tick()
                        if watcher.aborted:
                            _close_socket()
                            return
                    line = raw.decode("utf-8", errors="replace")
                    if not line.strip():
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    yield chunk
        finally:
            if watcher is not None:
                watcher.stop()
