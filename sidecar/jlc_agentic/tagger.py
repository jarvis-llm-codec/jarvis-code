"""JLC Tag Extractor — regex-based, zero LLM calls."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TagPattern:
    name: str
    pattern: str
    _compiled: re.Pattern | None = field(default=None, repr=False, compare=False)

    @property
    def regex(self) -> re.Pattern:
        if self._compiled is None:
            self._compiled = re.compile(self.pattern)
        return self._compiled


# Built-in patterns
_BUILTIN_PATTERNS: list[TagPattern] = [
    TagPattern("file", r'(?:[\w./\\-]+\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml|toml|sh|bat|css|html|sql|env|cfg|ini))'),
    TagPattern("port", r'(?:localhost|127\.0\.0\.1):(\d{4,5})'),
    TagPattern("url", r'https?://[^\s<>"\')\]]+'),
    TagPattern("error", r'(?:HTTP\s*[45]\d{2}|exit\s*code\s*\d+|Error:|Exception:|Traceback)'),
    TagPattern("hashtag", r'#[A-Za-z][\w-]{1,30}'),
    TagPattern("mention", r'@[A-Za-z][\w-]{1,30}'),
]


class JLCTagger:
    """Extract tags from conversation turns using regex patterns."""

    def __init__(
        self,
        custom_patterns: list[dict[str, str]] | None = None,
        max_tags_per_turn: int = 20,
    ) -> None:
        self._patterns = list(_BUILTIN_PATTERNS)
        for cp in custom_patterns or []:
            name = cp.get("name", "custom")
            pattern = cp.get("pattern", "")
            if pattern:
                self._patterns.append(TagPattern(name, pattern))
        self._max_tags = max_tags_per_turn

    def extract(self, user_msg: str, assistant_msg: str) -> list[str]:
        """Extract tags from user + assistant messages. Deduplicated, sorted, capped."""
        text = f"{user_msg or ''}\n{assistant_msg or ''}"
        tags: set[str] = set()

        for tp in self._patterns:
            for match in tp.regex.finditer(text):
                if tp.name == "port":
                    # Capture group 1 = port number
                    port = match.group(1) if match.lastindex else match.group()
                    tags.add(f"port:{port}")
                elif tp.name == "hashtag":
                    tags.add(match.group().lower())
                elif tp.name == "mention":
                    tags.add(match.group().lower())
                else:
                    raw = match.group().lower()
                    # Prefix with category for non-hashtag/mention
                    tags.add(f"{tp.name}:{raw}")

        result = sorted(tags)
        return result[: self._max_tags]

    def update_index(
        self, tags_data: dict[str, Any], turn: int, tags: list[str],
    ) -> dict[str, Any]:
        """Incrementally update tags index with new turn's tags."""
        tag_index = tags_data.get("tags", {})

        for tag in tags:
            entry = tag_index.get(tag)
            if entry is None:
                tag_index[tag] = {
                    "count": 1,
                    "first_seen": turn,
                    "last_seen": turn,
                    "turns": [turn],
                }
            else:
                entry["count"] += 1
                entry["last_seen"] = turn
                if turn not in entry["turns"]:
                    entry["turns"].append(turn)

        tags_data["tags"] = tag_index
        return tags_data
