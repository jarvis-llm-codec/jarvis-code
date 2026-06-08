"""Web search tool — Brave Search API."""
from __future__ import annotations

import json
import os
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web via Brave Search API. "
            "Requires BRAVE_SEARCH_API_KEY env var."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def handler(query: str, top_k: int = 5) -> dict:
    """Run web search via Brave Search API."""
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not key:
        return {
            "ok": False,
            "error": "BRAVE_SEARCH_API_KEY env var is not set.",
        }

    params = urlencode({"q": query, "count": max(1, min(int(top_k), 20))})
    req = Request(
        f"{BRAVE_ENDPOINT}?{params}",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": key,
        },
    )
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            with urlopen(req, timeout=10) as resp:  # noqa: S310
                payload = resp.read().decode("utf-8", errors="replace")
            data = json.loads(payload)
            web = (data.get("web") or {}).get("results") or []
            results = [
                {
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "snippet": r.get("description"),
                }
                for r in web[:top_k]
            ]
            return {"ok": True, "provider": "brave", "results": results}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == 1:
                time.sleep(0.5)
                continue
    return {"ok": False, "error": f"Brave Search failed after 2 attempts: {last_exc}"}
