from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProviderAdapter(Protocol):
    def stream_chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        parallel_tool_calls: bool = True,
        stream: bool = True,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        ...
