"""Recall turns tool."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from jlc_agentic import recall_budget, recall_snippets, recall_trace
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
            query=item["query"],
            top_k=item["top_k"],
            session_id=conv_id,
        )
        return {
            "query": item["query"],
            "top_k": item["top_k"],
            "result": result,
        }

    return await asyncio.gather(*(_one(item) for item in normalized))


def _cap_result_fragments(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    remaining = recall_budget.evidence_token_budget()
    per_fragment = recall_budget.fragment_token_budget(remaining)
    out: list[dict[str, Any]] = []
    caps: list[dict[str, Any]] = []
    snippets: list[dict[str, Any]] = []
    for result_item in results:
        item = dict(result_item)
        result = item.get("result")
        if not isinstance(result, dict):
            out.append(item)
            continue
        fragments = [frag for frag in (result.get("fragments") or []) if isinstance(frag, dict)]
        snippet_fragments, snippet_meta = recall_snippets.snippet_fragments(
            fragments,
            query=str(item.get("query") or ""),
        )
        capped_fragments, cap_meta = recall_budget.cap_fragments(
            snippet_fragments,
            total_budget_tokens=remaining,
            per_fragment_tokens=per_fragment,
        )
        cap_meta = dict(cap_meta)
        cap_meta["snippet"] = snippet_meta
        remaining = max(0, remaining - int(cap_meta.get("served_tokens") or 0))
        capped_result = dict(result)
        capped_result["fragments"] = capped_fragments
        capped_result["served_policy"] = (
            cap_meta["served_policy"]
            if cap_meta["served_policy"] == "capped"
            else snippet_meta["served_policy"]
        )
        capped_result["cap"] = cap_meta
        capped_result["snippet"] = snippet_meta
        item["result"] = capped_result
        caps.append(cap_meta)
        snippets.append(snippet_meta)
        out.append(item)
    original_tokens = sum(int(cap.get("original_tokens") or 0) for cap in caps)
    served_tokens = sum(int(cap.get("served_tokens") or 0) for cap in caps)
    truncated = any(bool(cap.get("truncated")) for cap in caps)
    snipped = any(bool(meta.get("truncated")) for meta in snippets)
    return out, {
        "served_policy": "capped" if truncated else "snipped" if snipped else "full",
        "budget_tokens": recall_budget.evidence_token_budget(),
        "per_fragment_budget_tokens": per_fragment,
        "original_tokens": original_tokens,
        "served_tokens": served_tokens,
        "truncated_tokens": max(0, original_tokens - served_tokens),
        "truncated": truncated,
        "snippet": {
            "served_policy": "snipped" if snipped else "full",
            "original_chars": sum(int(meta.get("original_chars") or 0) for meta in snippets),
            "served_chars": sum(int(meta.get("served_chars") or 0) for meta in snippets),
            "truncated_chars": sum(
                max(0, int(meta.get("original_chars") or 0) - int(meta.get("served_chars") or 0))
                for meta in snippets
            ),
            "truncated": snipped,
            "results": snippets,
        },
        "results": caps,
    }


def handler(
    queries: list[str | dict[str, Any]] | str | None = None,
    top_k: int = 5,
    query: str | None = None,
    conv_id: str = "conversation",
    storage_root: str | None = None,
    retriever: JLCRetriever | None = None,
    trace_surface: str = "recall_tool",
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
    results, cap_meta = _cap_result_fragments(results)
    candidates: list[dict[str, Any]] = []
    served_turns: list[Any] = []
    for result_item in results:
        result = result_item.get("result") if isinstance(result_item, dict) else None
        if not isinstance(result, dict):
            continue
        for rank, frag in enumerate(result.get("fragments") or [], start=1):
            if not isinstance(frag, dict):
                continue
            served_turns.append(frag.get("turn"))
            candidates.append({
                "query": result_item.get("query"),
                "rank": rank,
                "turn": frag.get("turn"),
                "score": frag.get("score", 0.0),
                "source": result.get("source", "hybrid"),
                "user_chars": len(str(frag.get("user", ""))),
                "assistant_chars": len(str(frag.get("assistant", ""))),
            })
    recall_trace.emit(
        "recall_trace",
        surface=trace_surface,
        conv_id=conv_id,
        query_count=len(results),
        queries=[
            {
                "query_sha256": recall_trace.sha256_text(item["query"]),
                "query_preview": recall_trace.preview_text(item["query"]),
                "top_k": item["top_k"],
            }
            for item in normalized
        ],
        candidates=candidates,
        served_turns=served_turns,
        served_chars=sum(
            len(str(frag.get("user", ""))) + len(str(frag.get("assistant", "")))
            for result_item in results
            for frag in ((result_item.get("result") or {}).get("fragments") or [])
            if isinstance(frag, dict)
        ),
        served_policy=cap_meta["served_policy"],
        cap=cap_meta,
    )
    return {
        "ok": True,
        "results": results,
        "query_count": len(results),
        "served_policy": cap_meta["served_policy"],
        "cap": cap_meta,
    }
