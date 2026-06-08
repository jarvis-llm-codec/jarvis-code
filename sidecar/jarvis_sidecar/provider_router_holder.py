from __future__ import annotations

import threading
from typing import Any

_provider_router: Any | None = None
_lock = threading.Lock()


def set_provider_router(router: Any | None) -> None:
    global _provider_router
    with _lock:
        _provider_router = router


def get_provider_router() -> Any | None:
    with _lock:
        return _provider_router
