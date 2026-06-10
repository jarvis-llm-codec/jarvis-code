from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping

_FALLBACK_VERSION = "1.01.0"
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VERSION_FILE = _REPO_ROOT / "jarvis_version.json"


@lru_cache(maxsize=1)
def jarvis_code_version() -> str:
    try:
        raw = json.loads(_VERSION_FILE.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return _FALLBACK_VERSION
    version = str(raw.get("version") or "").strip() if isinstance(raw, dict) else ""
    return version if _VERSION_RE.match(version) else _FALLBACK_VERSION


def jarvis_code_user_agent(version: str | None = None) -> str:
    value = (version or jarvis_code_version()).strip()
    if not _VERSION_RE.match(value):
        value = _FALLBACK_VERSION
    return f"jarvis-code/{value} (pi-agent)"


JARVIS_CODE_VERSION = jarvis_code_version()
JARVIS_CODE_USER_AGENT = jarvis_code_user_agent(JARVIS_CODE_VERSION)


def with_jarvis_user_agent(headers: Mapping[str, object] | None = None) -> dict[str, str]:
    out = {str(key): str(value) for key, value in (headers or {}).items()}
    existing_key = next((key for key in out if key.lower() == "user-agent"), None)
    if existing_key is None:
        out["User-Agent"] = JARVIS_CODE_USER_AGENT
        return out
    existing = out[existing_key].strip()
    if JARVIS_CODE_USER_AGENT not in existing:
        out[existing_key] = f"{existing} {JARVIS_CODE_USER_AGENT}".strip()
    return out
