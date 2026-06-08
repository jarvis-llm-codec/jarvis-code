"""Login CLI for ChatGPT OAuth."""
from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import webbrowser
from typing import Sequence

from jlc_agentic.openai_oauth import (
    CallbackResult,
    DeviceCodeChallenge,
    LocalCallbackServer,
    StoredAuth,
    TokenManager,
    build_authorization_url,
    build_redirect_uri,
    exchange_code_for_token,
    extract_account_id_from_id_token,
    generate_pkce,
    generate_state,
    poll_device_code,
    start_device_code_flow,
)

SUPPORTED_PROVIDER = "chatgpt"


class _InvalidTokenResponse(ValueError):
    """OAuth server returned a response missing required token fields."""


def _provider_ok(provider: str) -> bool:
    if provider == SUPPORTED_PROVIDER:
        return True
    print(f"❌ Unknown provider: {provider}. Supported: {SUPPORTED_PROVIDER}")
    return False


def _store_tokens(tokens: dict, token_manager: TokenManager) -> StoredAuth:
    missing = [k for k in ("access_token", "refresh_token") if k not in tokens]
    if missing:
        raise _InvalidTokenResponse(
            f"Token response missing required fields: {missing}. "
            f"Keys present: {sorted(tokens)}"
        )
    account_id = extract_account_id_from_id_token(tokens.get("id_token", ""))
    now = int(time.time())
    auth = StoredAuth(
        access_token=tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_at_unix=now + int(tokens.get("expires_in", 3600)),
        issued_at_unix=now,
        account_id=account_id,
    )
    token_manager.save(auth)
    return auth


def _print_login_success(auth: StoredAuth) -> None:
    mins = (auth.expires_at_unix - int(time.time())) // 60
    print(f"✅ Logged in (account_id={auth.account_id or '<unknown>'}, expires in {mins} minutes)")


def _callback_error_exit(result: CallbackResult) -> int:
    if result.error == "timeout":
        print("❌ Timeout — no callback within 120s")
        return 2
    if result.error == "state_mismatch":
        print("❌ State mismatch (possible CSRF)")
        return 3
    if result.error == "missing_code":
        print("❌ No authorization code in callback")
        return 4
    if result.error:
        print(f"❌ Authorization failed: {result.error}")
        return 5
    print("❌ Empty callback")
    return 6


def cmd_login_browser(provider: str, *, token_manager: TokenManager | None = None) -> int:
    if not _provider_ok(provider):
        return 1

    mgr = token_manager or TokenManager()
    pkce = generate_pkce()
    state = generate_state()
    server = LocalCallbackServer(state)

    try:
        port = server.start()
        redirect_uri = build_redirect_uri(port)
        url = build_authorization_url(pkce, state, redirect_uri=redirect_uri)
        print("🌐 Opening browser…")
        webbrowser.open(url)
        print(f"   Or visit: {url}")

        result = server.wait(timeout_secs=120)
    finally:
        server.stop()

    if result.error or not result.code:
        return _callback_error_exit(result)

    try:
        tokens = exchange_code_for_token(result.code, pkce.verifier, redirect_uri=redirect_uri)
    except urllib.error.HTTPError as exc:
        print(f"❌ Token exchange failed: HTTP {exc.code} {exc.reason}")
        print("   The authorization code is single-use — please run login again.")
        return 5
    except urllib.error.URLError as exc:
        print(f"❌ Token exchange failed: {exc.reason}")
        print("   Check your network connection and run login again.")
        return 5

    try:
        auth = _store_tokens(tokens, mgr)
    except _InvalidTokenResponse as exc:
        print(f"❌ {exc}")
        print("   Please run login again.")
        return 5

    _print_login_success(auth)
    return 0


def cmd_login_device_code(provider: str, *, token_manager: TokenManager | None = None) -> int:
    if not _provider_ok(provider):
        return 1

    mgr = token_manager or TokenManager()
    ch: DeviceCodeChallenge = start_device_code_flow()
    print(f"📱 Visit:  {ch.verification_uri}")
    print(f"   Code:   {ch.user_code}")
    print(f"   Polling every {ch.interval}s (expires in {ch.expires_in}s)…")

    try:
        tokens = poll_device_code(ch.device_code, ch.interval, timeout_secs=ch.expires_in)
    except TimeoutError:
        print("❌ Device code expired — please run login again")
        return 7
    except RuntimeError as exc:
        print(f"❌ Device code authorization failed: {exc}")
        return 7

    try:
        auth = _store_tokens(tokens, mgr)
    except _InvalidTokenResponse as exc:
        print(f"❌ {exc}")
        print("   Please run login again.")
        return 7

    _print_login_success(auth)
    return 0


def cmd_logout(provider: str, *, token_manager: TokenManager | None = None) -> int:
    if provider != SUPPORTED_PROVIDER:
        print(f"❌ Unknown provider: {provider}")
        return 8

    mgr = token_manager or TokenManager()
    mgr.clear()
    print("✅ Logged out")
    return 0


def cmd_status(*, token_manager: TokenManager | None = None) -> int:
    mgr = token_manager or TokenManager()
    auth = mgr.load()
    if auth is None:
        print("⚪ Not logged in")
        return 0

    remaining = auth.expires_at_unix - int(time.time())
    if remaining > 0:
        print(
            f"🟢 Logged in (account_id={auth.account_id or '<unknown>'}, "
            f"expires in {remaining // 60}m {remaining % 60}s)"
        )
    else:
        print(f"🟡 Token expired {-remaining}s ago — will refresh on next call")
    return 0


def _normalize_argv(argv: Sequence[str] | None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == SUPPORTED_PROVIDER:
        return ["login", *args]
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jarvis-code-login")
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login")
    login_parser.add_argument("provider")
    login_parser.add_argument("--device-code", action="store_true")

    logout_parser = subparsers.add_parser("logout")
    logout_parser.add_argument("provider")

    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_argv(argv))

    if args.command == "login":
        if args.device_code:
            return cmd_login_device_code(args.provider)
        return cmd_login_browser(args.provider)
    if args.command == "logout":
        return cmd_logout(args.provider)
    if args.command == "status":
        return cmd_status()

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
