"""Recall turns tool."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from jlc_agentic.retriever import JLCRetriever
try:
    from jarvis_sidecar.raw_store import extract_turn_numbers
except Exception:  # pragma: no cover - sidecar package unavailable in standalone mode
    extract_turn_numbers = None
try:
    from jarvis_sidecar.raw_store import extract_local_dates
except Exception:  # pragma: no cover - sidecar package unavailable in standalone mode
    extract_local_dates = None

SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_turns",
        "description": (
            "Search the FULL prior conversation transcript (raw turn-level memory, "
            "not the compressed JHB summary). Use this BEFORE asking the user a "
            "clarifying question or saying you do not know whenever the user asks "
            "for a specific prior conversation fact and JHB/JARVIS.md do not "
            "clearly contain the answer. This includes names, family/people, "
            "places, dates, preferences, decisions, numbers, previous errors, "
            "exact wording, code snippets, project details, or cues like "
            "\"the thing we discussed\", \"remember\", \"예전에\", \"그때\", "
            "\"했었지?\". "
            "Create 2-4 independent focused query variants up front and pass them "
            "in one call. Returns ranked turn fragments separated by query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "top_k": {"type": "integer", "default": 5},
                                },
                                "required": ["query"],
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "minItems": 1,
                    "description": (
                        "Focused query variants, e.g. "
                        "[\"work blocker before dinner\", "
                        "{\"query\":\"auth edge case production tests\", \"top_k\":5}]"
                    ),
                },
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["queries"],
            "additionalProperties": False,
        },
    },
}


def _normalize_queries(
    queries: list[str | dict[str, Any]] | str | None,
    default_top_k: int,
) -> list[dict[str, Any]]:
    if queries is None:
        return []
    raw_items: list[Any] = [queries] if isinstance(queries, str) else list(queries)
    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, str):
            q = item.strip()
            tk = default_top_k
        elif isinstance(item, dict):
            q = str(item.get("query") or "").strip()
            try:
                tk = int(item.get("top_k", default_top_k))
            except (TypeError, ValueError):
                tk = default_top_k
        else:
            continue
        if q:
            normalized.append({"query": q, "top_k": max(1, tk)})
    return normalized


async def _run_queries(
    retriever: JLCRetriever,
    *,
    conv_id: str,
    normalized: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    async def _one(item: dict[str, Any]) -> dict[str, Any]:
        if callable(extract_turn_numbers):
            turn_numbers = set(extract_turn_numbers(item["query"], max_turns=item["top_k"]))
            if turn_numbers:
                fragments = retriever.load_turns_by_number(turn_numbers, session_id=conv_id)
                fragments.sort(key=lambda frag: int(frag.get("turn") or 0))
                return {
                    "query": item["query"],
                    "top_k": item["top_k"],
                    "result": {
                        "confidence": "HIGH" if fragments else "LOW",
                        "fragments": fragments,
                        "source": "turn_number",
                    },
                }
        if callable(extract_local_dates):
            local_dates = extract_local_dates(item["query"], max_dates=item["top_k"])
            if local_dates:
                fragments = retriever.load_turns_by_local_dates(local_dates, session_id=conv_id, limit=item["top_k"])
                return {
                    "query": item["query"],
                    "top_k": item["top_k"],
                    "result": {
                        "confidence": "HIGH" if fragments else "LOW",
                        "fragments": fragments,
                        "source": "date",
                    },
                }
        result = await retriever.hybrid_search(
            conv_id=conv_id,
            query=item["query"],
            top_k=item["top_k"],
        )
        return {
            "query": item["query"],
            "top_k": item["top_k"],
            "result": result,
        }

    return await asyncio.gather(*(_one(item) for item in normalized))


def handler(
    queries: list[str | dict[str, Any]] | str | None = None,
    top_k: int = 5,
    query: str | None = None,
    conv_id: str = "conversation",
    storage_root: str | None = None,
    retriever: JLCRetriever | None = None,
) -> dict:
    """Run batched retriever hybrid_search and return query-separated fragments.

    conv_id / storage_root / retriever are not exposed to the LLM (see
    SCHEMA) -- they are injected by schema.get_dispatcher via closure so the
    tool always reads from the same JHB store the host writes to and reuses the
    host singleton retriever.

    W2.9.24: the public tool schema exposes only recall_turns(queries=...).
    query= remains accepted for direct legacy callers but is not registered as
    a model-visible tool.
    """
    if retriever is None:
        if storage_root is None:
            from jlc_agentic.config import load_config
            storage_root = load_config().jhb.storage_path
        retriever = JLCRetriever(storage_root=Path(storage_root).expanduser())
    effective_queries = queries if queries is not None else query
    normalized = _normalize_queries(effective_queries, top_k)
    if not normalized:
        return {"ok": False, "results": [], "error": "queries must contain at least one query"}
    results = asyncio.run(_run_queries(retriever, conv_id=conv_id, normalized=normalized))
    return {"ok": True, "results": results, "query_count": len(results)}
