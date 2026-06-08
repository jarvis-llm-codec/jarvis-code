"""JRE search tool stub."""
from __future__ import annotations

SCHEMA = {
    "type": "function",
    "function": {
        "name": "jre_search",
        "description": "Search JRE memory (stub for Phase 1).",
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


def handler(query: str, top_k: int = 5) -> dict:
    """Stub response until JRE wiring is added."""
    return {"ok": False, "query": query, "top_k": top_k, "error": "JRE not yet wired"}

