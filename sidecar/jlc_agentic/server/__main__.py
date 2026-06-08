from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

from .app import run


def _init_provider_router() -> None:
    try:
        from jlc_agentic.bootstrap import init_provider_router, print_roles_summary
        from jlc_agentic.providers import clear_cache
        from jarvis_sidecar.provider_router_holder import set_provider_router

        router = init_provider_router(SimpleNamespace(providers_config=None))
        set_provider_router(router)
        clear_cache()
        print_roles_summary()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[jarvis-ui] provider router init skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7150)
    args = parser.parse_args()
    _init_provider_router()
    run(host=args.host, port=args.port)
