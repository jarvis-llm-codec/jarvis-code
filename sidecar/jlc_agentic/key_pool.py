from __future__ import annotations

import logging
import threading
import time
from typing import TypedDict

log = logging.getLogger(__name__)


class _KeyState(TypedDict):
    disabled: bool
    cooldown_until: float


class AllKeysDisabledError(RuntimeError):
    pass


class KeyPool:
    """Provider-scoped API key rotation with simple cooldown handling."""

    def __init__(self, providers: dict[str, list[str]]) -> None:
        self._providers = {provider: list(keys) for provider, keys in providers.items()}
        self._state: dict[tuple[str, str], _KeyState] = {}
        self._next_index: dict[str, int] = {provider: 0 for provider in self._providers}
        self._lock = threading.Lock()
        for provider, keys in self._providers.items():
            for key in keys:
                self._state[(provider, key)] = {"disabled": False, "cooldown_until": 0.0}

    def take(self, provider: str, block: bool = True) -> str | None:
        """Return the next available key.

        When block is false, cooldown pressure returns None instead of sleeping.
        """
        while True:
            sleep_for = 0.0
            with self._lock:
                key = self._take_now(provider)
                if key is not None:
                    return key
                if not block:
                    return None
                sleep_for = self._sleep_until_next_key(provider)

            if sleep_for > 0:
                time.sleep(sleep_for)

    def report_failure(self, provider: str, key: str, status_code: int) -> None:
        with self._lock:
            if (provider, key) not in self._state:
                log.warning("Ignoring failure for unknown API key on provider %s", provider)
                return
            state = self._state_for(provider, key)
            now = time.time()
            if status_code == 429:
                state["cooldown_until"] = now + 60.0
            elif status_code == 401:
                state["disabled"] = True
                log.warning("Disabling API key for provider %s after 401 response", provider)
            elif 500 <= status_code <= 599:
                state["cooldown_until"] = now + 5.0

    def report_success(self, provider: str, key: str) -> None:
        self._state_for(provider, key)

    def _take_now(self, provider: str) -> str | None:
        keys = self._providers.get(provider)
        if not keys:
            raise ValueError(f"No API keys configured for provider: {provider}")

        now = time.time()
        start = self._next_index.get(provider, 0) % len(keys)
        for offset in range(len(keys)):
            idx = (start + offset) % len(keys)
            key = keys[idx]
            state = self._state_for(provider, key)
            if state["disabled"]:
                continue
            if state["cooldown_until"] > now:
                continue
            self._next_index[provider] = (idx + 1) % len(keys)
            return key
        return None

    def _sleep_until_next_key(self, provider: str) -> float:
        keys = self._providers.get(provider)
        if not keys:
            raise ValueError(f"No API keys configured for provider: {provider}")

        cooldowns = [
            self._state_for(provider, key)["cooldown_until"]
            for key in keys
            if not self._state_for(provider, key)["disabled"]
        ]
        if not cooldowns:
            raise AllKeysDisabledError(
                f"All API keys are disabled for provider: {provider}"
            )

        next_ready = min(cooldowns)
        return max(0.0, next_ready - time.time())

    def _state_for(self, provider: str, key: str) -> _KeyState:
        return self._state.setdefault(
            (provider, key),
            {"disabled": False, "cooldown_until": 0.0},
        )
