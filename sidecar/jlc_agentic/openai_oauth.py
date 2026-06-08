"""ChatGPT Plus/Pro OAuth -- PKCE + Device Code flows.

Ports the opencode (codex.ts) auth pattern to Python stdlib.
References:
  - opencode codex.ts: C:/JJUN_DEV/_tmp_opencode_repo/packages/opencode/src/plugin/codex.ts
  - Gemini analysis: C:/JJUN_DEV/_tmp_opencode_oauth_analysis.md

No third-party HTTP libs -- uses urllib.request + http.server.
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import socketserver
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# === Constants ===
OAUTH_ISSUER = "https://auth.openai.com"
OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # opencode shared ID (codex.ts:12)
OAUTH_SCOPE = "openid profile email offline_access"
OAUTH_REDIRECT_URI_BROWSER = "http://localhost:1455/auth/callback"
OAUTH_REDIRECT_URI_HEADLESS = "https://auth.openai.com/deviceauth/callback"
DEFAULT_AUTH_PATH = Path.home() / ".jarvis-code" / "auth.json"
TOKEN_REFRESH_LEADTIME_SECS = 300
DEVICE_POLL_SAFETY_MARGIN_MS = 3000
LOCAL_CALLBACK_PORTS = list(range(1455, 1466))


# === PKCE ===
@dataclass
class PKCEChallenge:
    verifier: str
    challenge: str
    method: str = "S256"


def generate_pkce() -> PKCEChallenge:
    """RFC 7636 PKCE challenge generator. verifier=43chars, S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return PKCEChallenge(verifier=verifier, challenge=challenge, method="S256")


def generate_state() -> str:
    """CSRF state token -- 32 bytes URL-safe."""
    return base64.urlsafe_b64encode(secrets.token_bytes(24)).decode("ascii").rstrip("=")


# === Auth URL ===
def build_redirect_uri(port: int = 1455) -> str:
    """Build the localhost callback URI for the given bound port.

    Phase B caller flow:
        server = LocalCallbackServer(state)
        actual_port = server.start()  # may fall back from 1455
        redirect_uri = build_redirect_uri(actual_port)
        url = build_authorization_url(pkce, state, redirect_uri=redirect_uri)
    The same redirect_uri must also be passed to exchange_code_for_token().
    """
    return f"http://localhost:{port}/auth/callback"


def build_authorization_url(
    pkce: PKCEChallenge,
    state: str,
    redirect_uri: str = OAUTH_REDIRECT_URI_BROWSER,
) -> str:
    """OAuth authorization URL builder."""
    params = {
        "response_type": "code",
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": OAUTH_SCOPE,
        "state": state,
        "code_challenge": pkce.challenge,
        "code_challenge_method": pkce.method,
    }
    return f"{OAUTH_ISSUER}/oauth/authorize?{urllib.parse.urlencode(params)}"


# === Local Callback Server ===
class CallbackResult:
    """Container for the callback handler to write into."""

    def __init__(self) -> None:
        self.code: str | None = None
        self.state_received: str | None = None
        self.error: str | None = None
        self.done = threading.Event()


def _make_callback_handler(result: CallbackResult, expected_state: str):
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/auth/callback":
                self.send_response(404)
                self.end_headers()
                return
            qs = urllib.parse.parse_qs(parsed.query)
            code = (qs.get("code") or [None])[0]
            state = (qs.get("state") or [None])[0]
            err = (qs.get("error") or [None])[0]
            result.state_received = state
            if err:
                result.error = err
            elif state is None or not secrets.compare_digest(state, expected_state):
                result.error = "state_mismatch"
            elif not code:
                result.error = "missing_code"
            else:
                result.code = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            msg = "Login successful -- you can close this tab." if result.code else f"Login failed: {result.error}"
            self.wfile.write(f"<html><body><h2>{msg}</h2></body></html>".encode("utf-8"))
            result.done.set()

    return _Handler


class LocalCallbackServer:
    """Single-shot localhost listener for OAuth redirect."""

    def __init__(self, expected_state: str, ports: list[int] = LOCAL_CALLBACK_PORTS) -> None:
        self.expected_state = expected_state
        self.ports = ports
        self.result = CallbackResult()
        self._server: socketserver.TCPServer | None = None
        self._thread: threading.Thread | None = None
        self._port: int | None = None

    def start(self) -> int:
        """Bind to first free port in self.ports. Returns the bound port."""
        last_err: Exception | None = None
        for port in self.ports:
            try:
                handler = _make_callback_handler(self.result, self.expected_state)
                self._server = socketserver.TCPServer(("localhost", port), handler)
                self._port = port
                self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
                self._thread.start()
                return port
            except OSError as exc:
                last_err = exc
                continue
        raise RuntimeError(f"No free port in {self.ports}: {last_err!r}")

    def wait(self, timeout_secs: int = 120) -> CallbackResult:
        ok = self.result.done.wait(timeout=timeout_secs)
        if not ok:
            self.result.error = "timeout"
        return self.result

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


# === Token Endpoint ===
def exchange_code_for_token(
    code: str,
    code_verifier: str,
    redirect_uri: str = OAUTH_REDIRECT_URI_BROWSER,
) -> dict[str, Any]:
    """POST {issuer}/oauth/token with grant_type=authorization_code."""
    body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": OAUTH_CLIENT_ID,
            "code_verifier": code_verifier,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OAUTH_ISSUER}/oauth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """POST {issuer}/oauth/token with grant_type=refresh_token."""
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": OAUTH_CLIENT_ID,
            "scope": OAUTH_SCOPE,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OAUTH_ISSUER}/oauth/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# === Device Code Flow ===
@dataclass
class DeviceCodeChallenge:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    expires_in: int
    interval: int


def start_device_code_flow() -> DeviceCodeChallenge:
    """POST {issuer}/api/accounts/deviceauth/usercode."""
    body = urllib.parse.urlencode(
        {
            "client_id": OAUTH_CLIENT_ID,
            "scope": OAUTH_SCOPE,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OAUTH_ISSUER}/api/accounts/deviceauth/usercode",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return DeviceCodeChallenge(
        device_code=data["device_code"],
        user_code=data["user_code"],
        verification_uri=data["verification_uri"],
        verification_uri_complete=data.get("verification_uri_complete"),
        expires_in=int(data.get("expires_in", 600)),
        interval=int(data.get("interval", 5)),
    )


def poll_device_code(device_code: str, interval_secs: int, timeout_secs: int = 600) -> dict[str, Any]:
    """Poll {issuer}/api/accounts/deviceauth/token until success / expiry."""
    deadline = time.time() + timeout_secs
    interval = interval_secs + (DEVICE_POLL_SAFETY_MARGIN_MS / 1000.0)
    while time.time() < deadline:
        body = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": OAUTH_CLIENT_ID,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{OAUTH_ISSUER}/api/accounts/deviceauth/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                err_body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                err_body = {}
            err_kind = err_body.get("error", "")
            if err_kind in ("authorization_pending", "slow_down"):
                time.sleep(interval)
                if err_kind == "slow_down":
                    interval += 5.0
                continue
            raise RuntimeError(f"Device code polling failed: {err_kind} ({err_body})") from exc
    raise TimeoutError("Device code authorization timed out")


# === Token Manager ===
@dataclass
class StoredAuth:
    access_token: str
    refresh_token: str
    expires_at_unix: int
    issued_at_unix: int
    account_id: str | None = None


class TokenManager:
    """Persists OAuth tokens to ~/.jarvis-code/auth.json with 0600 perms (best-effort on Windows).

    Auto-refreshes when access_token is within TOKEN_REFRESH_LEADTIME_SECS of expiry.
    All public methods are guarded by an RLock so concurrent callers (e.g. parallel
    ChatTurns hitting the router) cannot fire overlapping refreshes — the
    refresh_token is single-use and OAuth servers reject the second attempt.
    """

    def __init__(self, path: Path | str = DEFAULT_AUTH_PATH) -> None:
        self.path = Path(path).expanduser()
        self._cache: StoredAuth | None = None
        self._lock = threading.RLock()

    def save(self, auth: StoredAuth) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(asdict(auth), indent=2).encode("utf-8")
            fd, tmp_name = tempfile.mkstemp(
                prefix=".auth_",
                suffix=".json.tmp",
                dir=str(self.path.parent),
            )
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
                try:
                    os.chmod(tmp_path, 0o600)
                except OSError:
                    pass
                os.replace(tmp_path, self.path)
            except Exception:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise
            self._cache = auth

    def load(self) -> StoredAuth | None:
        with self._lock:
            if self._cache is not None:
                return self._cache
            if not self.path.exists():
                return None
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._cache = StoredAuth(**data)
            return self._cache

    def clear(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()
            self._cache = None

    def get_access_token(self) -> str:
        with self._lock:
            auth = self.load()
            if auth is None:
                raise RuntimeError(
                    f"No saved auth at {self.path}. Run `jarvis-code login chatgpt` first."
                )
            if auth.expires_at_unix - TOKEN_REFRESH_LEADTIME_SECS > time.time():
                return auth.access_token
            refreshed = refresh_access_token(auth.refresh_token)
            now = int(time.time())
            new_auth = StoredAuth(
                access_token=refreshed["access_token"],
                refresh_token=refreshed.get("refresh_token", auth.refresh_token),
                expires_at_unix=now + int(refreshed.get("expires_in", 3600)),
                issued_at_unix=now,
                account_id=auth.account_id,
            )
            self.save(new_auth)
            return new_auth.access_token

    def get_account_id(self) -> str | None:
        with self._lock:
            auth = self.load()
            return auth.account_id if auth else None


def extract_account_id_from_id_token(id_token: str) -> str | None:
    """JWT body decode without signature verification for the ChatGPT account claim."""
    try:
        parts = id_token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))
        auth_claim = payload.get("https://api.openai.com/auth") or {}
        if isinstance(auth_claim, dict):
            acct = auth_claim.get("chatgpt_account_id")
            if acct:
                return str(acct)
        return None
    except Exception:
        return None
