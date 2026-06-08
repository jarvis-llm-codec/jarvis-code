"""DashScope backward-compatible alias to OpenAI-compatible adapter."""
from __future__ import annotations

import os
from typing import Callable

from jlc_agentic.providers.openai_compatible import OpenAICompatibleAdapter, _SilenceWatcher, urlopen

DEFAULT_BASE_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"
DEFAULT_MODEL = "qwen3.6-plus"


class DashScopeLLMClient(OpenAICompatibleAdapter):
    """Backward-compatible wrapper with DashScope defaults."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout_sec: float = 600.0,
        silence_threshold_sec: float | None = None,
        on_silence: Callable[[float], str | None] | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_CODING_API_KEY"),
            timeout_sec=timeout_sec,
            silence_threshold_sec=silence_threshold_sec,
            on_silence=on_silence,
        )

