"""Bounded evidence windows for recalled turn fragments."""
from __future__ import annotations

import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any

DEFAULT_WINDOW_CHARS = 1200
DEFAULT_FRAGMENT_CHARS = 2500
MIN_WINDOW_CHARS = 800
MAX_WINDOW_CHARS = 1500
MAX_ANCHORS = 80

_BACKTICK_RE = re.compile(r"`([^`\r\n]{1,120})`")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_]*(?:[-_][A-Za-z0-9_]+)+")
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")
_PARA_RE = re.compile(r"\n\s*\n+")

_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "before",
    "could",
    "from",
    "have",
    "into",
    "latest",
    "please",
    "project",
    "reply",
    "should",
    "specifically",
    "that",
    "their",
    "there",
    "these",
    "thing",
    "this",
    "turn",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def snippets_enabled() -> bool:
    return str(os.environ.get("JARVIS_RECALL_SNIPPETS", "1")).strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def snippet_window_chars() -> int:
    configured = _env_int("JARVIS_RECALL_SNIPPET_WINDOW_CHARS", DEFAULT_WINDOW_CHARS)
    return min(MAX_WINDOW_CHARS, max(MIN_WINDOW_CHARS, configured))


def snippet_fragment_chars() -> int:
    configured = _env_int("JARVIS_RECALL_SNIPPET_MAX_CHARS", DEFAULT_FRAGMENT_CHARS)
    return max(MIN_WINDOW_CHARS, configured)


def _nfc(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text or ""))


def _casefold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _normalize_with_map(text: str) -> tuple[str, list[int], str]:
    source = _nfc(text)
    normalized: list[str] = []
    index_map: list[int] = []
    for idx, ch in enumerate(source):
        folded = unicodedata.normalize("NFC", ch).casefold()
        if not folded:
            continue
        normalized.append(folded)
        index_map.extend([idx] * len(folded))
    return "".join(normalized), index_map, source


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _add_anchor(out: list[str], seen: set[str], value: str) -> None:
    anchor = _nfc(value).strip()
    if not anchor:
        return
    key = _casefold(anchor)
    if len(key) < 2 or key in seen:
        return
    if key in _STOPWORDS:
        return
    seen.add(key)
    out.append(anchor)


def _query_anchors(query: str) -> list[str]:
    raw = _nfc(query)
    anchors: list[str] = []
    seen: set[str] = set()

    for match in _BACKTICK_RE.finditer(raw):
        _add_anchor(anchors, seen, match.group(1))
    for match in _IDENTIFIER_RE.finditer(raw):
        _add_anchor(anchors, seen, match.group(0))

    for match in _CJK_RE.finditer(raw):
        seq = match.group(0)
        if len(seq) >= 2:
            _add_anchor(anchors, seen, seq[:40])
        max_n = min(6, len(seq))
        for n in range(max_n, 1, -1):
            for start in range(0, max(0, len(seq) - n + 1)):
                _add_anchor(anchors, seen, seq[start:start + n])
                if len(anchors) >= MAX_ANCHORS:
                    break
            if len(anchors) >= MAX_ANCHORS:
                break

    words = [_nfc(match.group(0)) for match in _WORD_RE.finditer(raw)]
    meaningful: list[str] = []
    for word in words:
        key = _casefold(word)
        if _has_cjk(word):
            continue
        if len(key) >= 4 and key not in _STOPWORDS:
            meaningful.append(word)
            _add_anchor(anchors, seen, word)
        elif _IDENTIFIER_RE.fullmatch(word):
            meaningful.append(word)
            _add_anchor(anchors, seen, word)
    for size in (3, 2):
        for idx in range(0, max(0, len(meaningful) - size + 1)):
            _add_anchor(anchors, seen, " ".join(meaningful[idx:idx + size]))
            if len(anchors) >= MAX_ANCHORS:
                break
        if len(anchors) >= MAX_ANCHORS:
            break

    anchors.sort(key=lambda item: (len(_casefold(item)), item), reverse=True)
    return anchors[:MAX_ANCHORS]


def _tokenize(text: str) -> list[str]:
    return [_casefold(token) for token in _WORD_RE.findall(text or "")]


def _window_bounds(text_len: int, start: int, end: int, window_chars: int) -> tuple[int, int]:
    if text_len <= window_chars:
        return 0, text_len
    center = max(start, min(end, (start + end) // 2))
    left = max(0, center - window_chars // 2)
    right = min(text_len, left + window_chars)
    left = max(0, right - window_chars)
    return left, right


def _merge_bounds(bounds: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not bounds:
        return []
    bounds = sorted(bounds)
    merged = [bounds[0]]
    for start, end in bounds[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 80:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _trim_bounds_to_budget(bounds: list[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    used = 0
    for start, end in bounds:
        size = max(0, end - start)
        if size <= 0:
            continue
        if used + size > max_chars:
            remaining = max(0, max_chars - used)
            if remaining >= MIN_WINDOW_CHARS // 2:
                kept.append((start, start + remaining))
            break
        kept.append((start, end))
        used += size
    return kept


def _render_bounds(text: str, bounds: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    text_len = len(text)
    for start, end in bounds:
        part = text[start:end].strip()
        if not part:
            continue
        if start > 0:
            part = "..." + part
        if end < text_len:
            part = part + "..."
        parts.append(part)
    return "\n...\n".join(parts).strip()


def _paragraph_units(text: str, *, window_chars: int) -> list[tuple[int, int, str]]:
    units: list[tuple[int, int, str]] = []
    pos = 0
    for part in _PARA_RE.split(text):
        start = text.find(part, pos)
        if start < 0:
            start = pos
        end = start + len(part)
        pos = end
        clean = part.strip()
        if not clean:
            continue
        if len(clean) <= window_chars:
            units.append((start, end, clean))
            continue
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(end, chunk_start + window_chars)
            units.append((chunk_start, chunk_end, text[chunk_start:chunk_end]))
            if chunk_end >= end:
                break
            chunk_start = max(chunk_start + 1, chunk_end - 120)
    return units


def _bm25_unit_bounds(query: str, text: str, *, window_chars: int) -> tuple[int, int] | None:
    units = _paragraph_units(text, window_chars=window_chars)
    if not units:
        return None
    query_tokens = [token for token in _tokenize(query) if token not in _STOPWORDS]
    if not query_tokens:
        return None
    doc_tokens = [_tokenize(unit_text) for _start, _end, unit_text in units]
    document_count = len(doc_tokens)
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    avg_len = max(1.0, sum(len(tokens) for tokens in doc_tokens) / max(1, document_count))
    best: tuple[float, int, int] | None = None
    for idx, tokens in enumerate(doc_tokens):
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0.0
        length = len(tokens)
        for token in query_tokens:
            freq = counts.get(token, 0)
            if freq <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            denom = freq + 1.2 * (1 - 0.75 + 0.75 * length / avg_len)
            score += idf * (freq * 2.2 / denom)
        if best is None or score > best[0]:
            start, end, _unit_text = units[idx]
            best = (score, start, end)
    if best is None or best[0] <= 0:
        return None
    return best[1], best[2]


def _head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    head = max_chars // 2
    tail = max_chars - head
    return f"{text[:head].strip()}...\n...\n{text[-tail:].strip()}"


def evidence_snippet(query: str, text: str) -> tuple[str, dict[str, Any]]:
    source_text = _nfc(text)
    window_chars = snippet_window_chars()
    max_chars = snippet_fragment_chars()
    if not snippets_enabled() or not source_text.strip() or len(source_text) <= max_chars:
        return source_text, {
            "policy": "full",
            "method": "none",
            "original_chars": len(source_text),
            "served_chars": len(source_text),
            "truncated": False,
        }

    normalized_text, index_map, mapped_text = _normalize_with_map(source_text)
    matches: list[tuple[int, int, str]] = []
    for anchor in _query_anchors(query):
        normalized_anchor = _casefold(anchor)
        if not normalized_anchor:
            continue
        search_from = 0
        while True:
            found = normalized_text.find(normalized_anchor, search_from)
            if found < 0:
                break
            if found < len(index_map):
                start = index_map[found]
                end_index = min(len(index_map) - 1, found + len(normalized_anchor) - 1)
                end = index_map[end_index] + 1
                matches.append((start, end, anchor))
            search_from = found + max(1, len(normalized_anchor))
            if len(matches) >= MAX_ANCHORS:
                break
        if len(matches) >= MAX_ANCHORS:
            break

    bounds: list[tuple[int, int]] = []
    method = "literal"
    anchors_used: list[str] = []
    if matches:
        matches.sort(key=lambda item: (-(item[1] - item[0]), item[0]))
        for start, end, anchor in matches:
            candidate = _window_bounds(len(mapped_text), start, end, window_chars)
            if any(not (candidate[1] < existing[0] or candidate[0] > existing[1]) for existing in bounds):
                continue
            bounds.append(candidate)
            anchors_used.append(anchor)
            if sum(max(0, right - left) for left, right in bounds) >= max_chars:
                break
    else:
        bm25_bounds = _bm25_unit_bounds(query, mapped_text, window_chars=window_chars)
        if bm25_bounds is not None:
            method = "bm25_paragraph"
            start, end = bm25_bounds
            center_start, center_end = _window_bounds(
                len(mapped_text),
                start,
                end,
                min(max_chars, max(window_chars, end - start)),
            )
            bounds.append((center_start, center_end))
        else:
            method = "head_tail"

    if bounds:
        bounds = _trim_bounds_to_budget(_merge_bounds(bounds), max_chars)
        snippet = _render_bounds(mapped_text, bounds)
    else:
        snippet = _head_tail(mapped_text, max_chars)

    if not snippet:
        snippet = _head_tail(mapped_text, max_chars)
        method = "head_tail"
    return snippet, {
        "policy": "snipped" if snippet != mapped_text else "full",
        "method": method,
        "anchors": anchors_used[:5],
        "original_chars": len(mapped_text),
        "served_chars": len(snippet),
        "truncated": len(snippet) < len(mapped_text),
    }


def snippet_fragments(
    fragments: list[dict[str, Any]],
    *,
    query: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not snippets_enabled() or not query.strip():
        copied = [dict(fragment) for fragment in fragments if isinstance(fragment, dict)]
        return copied, {
            "served_policy": "full",
            "enabled": snippets_enabled(),
            "query_present": bool(query.strip()),
            "fragments": [],
        }

    out: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        user = _nfc(fragment.get("user", ""))
        assistant = _nfc(fragment.get("assistant", ""))
        combined = f"Q: {user}\nA: {assistant}"
        snippet, meta = evidence_snippet(query, combined)
        served = dict(fragment)
        served["snippet_policy"] = meta["policy"]
        served["snippet_method"] = meta["method"]
        if meta.get("truncated"):
            served["user"] = snippet
            served["assistant"] = ""
            served["snippet"] = snippet
            served["snipped"] = True
            served["original_chars_est"] = meta.get("original_chars", 0)
            served["served_chars_est"] = meta.get("served_chars", 0)
        out.append(served)
        frag_meta = dict(meta)
        frag_meta["turn"] = fragment.get("turn")
        metas.append(frag_meta)

    original_chars = sum(int(meta.get("original_chars") or 0) for meta in metas)
    served_chars = sum(int(meta.get("served_chars") or 0) for meta in metas)
    truncated = any(bool(meta.get("truncated")) for meta in metas)
    return out, {
        "served_policy": "snipped" if truncated else "full",
        "enabled": True,
        "window_chars": snippet_window_chars(),
        "fragment_max_chars": snippet_fragment_chars(),
        "original_chars": original_chars,
        "served_chars": served_chars,
        "truncated_chars": max(0, original_chars - served_chars),
        "truncated": truncated,
        "fragments": metas,
    }
