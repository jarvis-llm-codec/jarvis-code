"""Public package surface for JARVIS Code's JLC engine.

Keep this module lightweight. Python executes package ``__init__`` before
``python -m jlc_agentic.cli.login``; importing the full JLC runtime here makes
OAuth login depend on embedder/torch/provider imports that the login command
does not need.
"""
from __future__ import annotations

import threading
from typing import Any

__version__ = "0.0.0"

_slim_singleton: Any | None = None
_slim_lock = threading.Lock()


def get_slim() -> Any:
    """Return the process-wide JarvisAgentic singleton."""
    global _slim_singleton
    if _slim_singleton is None:
        with _slim_lock:
            if _slim_singleton is None:
                from .slim import JarvisAgentic

                _slim_singleton = JarvisAgentic()
    return _slim_singleton


def __getattr__(name: str) -> Any:
    if name == "JarvisAgentic":
        from .slim import JarvisAgentic

        return JarvisAgentic
    if name in {"Registry", "ProjectEntry"}:
        from .registry import ProjectEntry, Registry

        return {"Registry": Registry, "ProjectEntry": ProjectEntry}[name]
    if name in {"ScanReport", "initialize_jarvis_md", "scan_project"}:
        from .registration import ScanReport, initialize_jarvis_md, scan_project

        return {
            "ScanReport": ScanReport,
            "initialize_jarvis_md": initialize_jarvis_md,
            "scan_project": scan_project,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "JarvisAgentic",
    "Registry",
    "ProjectEntry",
    "ScanReport",
    "scan_project",
    "initialize_jarvis_md",
    "get_slim",
]
