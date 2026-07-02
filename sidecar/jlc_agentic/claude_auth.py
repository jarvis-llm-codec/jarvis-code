from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_EXPIRY_SKEW_SECONDS = 60


@dataclass(frozen=True)
class ClaudeAgentSDKAuthStatus:
    available: bool
    source: str
    reason: str
    credentials_path: str | None = None
    expires_at_ms: int | None = None
    has_refresh_token: bool | None = None


def agent_sdk_auth_available() -> bool:
    return inspect_agent_sdk_auth().available


def get_agent_sdk_access_token() -> str | None:
    """Return the current Claude subscription OAuth access token, or None.

    Prefers the explicit headless env token, then the logged-in Claude Code CLI
    credentials (``~/.claude/.credentials.json`` -> ``claudeAiOauth.accessToken``).
    The Claude CLI keeps this token refreshed on disk, so reading it fresh per
    call is enough; this helper does NOT refresh and never logs the token. A
    stale token simply yields a failed request that the caller falls back from.
    """
    env = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    if env:
        return env

    path = Path.home() / ".claude" / ".credentials.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    oauth = raw.get("claudeAiOauth") if isinstance(raw, dict) else None
    if not isinstance(oauth, dict):
        return None

    token = str(oauth.get("accessToken") or oauth.get("access_token") or "").strip()
    return token or None


def inspect_agent_sdk_auth(*, now_ms: int | None = None) -> ClaudeAgentSDKAuthStatus:
    """Inspect Claude Agent SDK subscription auth without exposing secrets.

    The headless token path is explicit and wins. The CLI credentials path must
    contain either a refresh token or a still-valid access token; a stale
    ``~/.claude/.credentials.json`` with no refresh token makes Claude Code fail
    later with ``authentication_failed``, so do not count it as usable.
    """
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip():
        return ClaudeAgentSDKAuthStatus(
            available=True,
            source="env",
            reason="CLAUDE_CODE_OAUTH_TOKEN is set",
        )

    path = Path.home() / ".claude" / ".credentials.json"
    if not path.exists():
        return ClaudeAgentSDKAuthStatus(
            available=False,
            source="none",
            reason="Claude Code credentials file was not found",
            credentials_path=str(path),
        )

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ClaudeAgentSDKAuthStatus(
            available=False,
            source="cli_credentials",
            reason=f"Claude Code credentials file is unreadable: {type(exc).__name__}",
            credentials_path=str(path),
        )

    oauth = raw.get("claudeAiOauth") if isinstance(raw, dict) else None
    if not isinstance(oauth, dict):
        return ClaudeAgentSDKAuthStatus(
            available=False,
            source="cli_credentials",
            reason="Claude Code credentials file has no claudeAiOauth block",
            credentials_path=str(path),
        )

    access_token = str(oauth.get("accessToken") or oauth.get("access_token") or "").strip()
    refresh_token = str(oauth.get("refreshToken") or oauth.get("refresh_token") or "").strip()
    expires_at_ms = _coerce_epoch_ms(
        oauth.get("expiresAt")
        or oauth.get("expires_at")
        or oauth.get("expires_at_ms")
        or oauth.get("expires_at_unix")
    )
    has_refresh_token = bool(refresh_token)

    if has_refresh_token:
        return ClaudeAgentSDKAuthStatus(
            available=True,
            source="cli_credentials",
            reason="Claude Code credentials include a refresh token",
            credentials_path=str(path),
            expires_at_ms=expires_at_ms,
            has_refresh_token=True,
        )

    if access_token and expires_at_ms is None:
        return ClaudeAgentSDKAuthStatus(
            available=True,
            source="cli_credentials",
            reason="Claude Code credentials include an access token with unknown expiry",
            credentials_path=str(path),
            expires_at_ms=None,
            has_refresh_token=False,
        )

    now = now_ms if now_ms is not None else int(time.time() * 1000)
    if access_token and expires_at_ms and expires_at_ms > now + (_EXPIRY_SKEW_SECONDS * 1000):
        return ClaudeAgentSDKAuthStatus(
            available=True,
            source="cli_credentials",
            reason="Claude Code access token is still valid",
            credentials_path=str(path),
            expires_at_ms=expires_at_ms,
            has_refresh_token=False,
        )

    if access_token:
        return ClaudeAgentSDKAuthStatus(
            available=False,
            source="cli_credentials",
            reason="Claude Code access token is expired and no refresh token is stored",
            credentials_path=str(path),
            expires_at_ms=expires_at_ms,
            has_refresh_token=False,
        )

    return ClaudeAgentSDKAuthStatus(
        available=False,
        source="cli_credentials",
        reason="Claude Code credentials contain no access token",
        credentials_path=str(path),
        expires_at_ms=expires_at_ms,
        has_refresh_token=False,
    )


def _coerce_epoch_ms(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    # seconds since epoch; current millisecond epochs are already 13 digits.
    if parsed < 100_000_000_000:
        return parsed * 1000
    return parsed
