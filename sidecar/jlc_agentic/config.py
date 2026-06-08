"""Slim JLC configuration loader."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ProviderConfig:
    name: str
    model: str
    base_url: str
    api_key_env: str = ""
    max_tokens: int = 2048

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


def _default_providers() -> list[ProviderConfig]:
    """Deprecated W2.9.8: encoder routing now comes from roles.encoder only."""
    return []


@dataclass
class EncoderConfig:
    providers: list[ProviderConfig] = field(default_factory=_default_providers)


@dataclass
class JHBConfig:
    max_sections: int = 20
    target_tokens: int = 2000
    storage_path: str = "~/.jarvis-code/conversation"


@dataclass
class ConversationTailConfig:
    enabled: bool = True
    count: int = 5
    max_tokens_per_turn: int = 220
    encoder_role: str = "encoder"


@dataclass
class TaggerConfig:
    custom_patterns: list[dict[str, str]] = field(default_factory=list)
    max_tags_per_turn: int = 20


@dataclass
class GraphConfig:
    batch_interval: int = 5
    max_nodes: int = 500
    max_edges: int = 2000
    prune_stale_turns: int = 200


@dataclass
class EmbedderConfig:
    model_name: str = "BAAI/bge-m3"
    cache_dir: str = "~/.cache/huggingface"
    device: str = "cpu"


@dataclass
class JLCConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    jhb: JHBConfig = field(default_factory=JHBConfig)
    conversation_tail: ConversationTailConfig = field(default_factory=ConversationTailConfig)
    tagger: TaggerConfig = field(default_factory=TaggerConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)


def load_config(path: str | Path | None = None) -> JLCConfig:
    if path is None:
        path = Path("~/.jarvis-code/config.yaml").expanduser()
    else:
        path = Path(path).expanduser()
    if not path.exists():
        return JLCConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # W2.9.8: legacy encoder.providers is intentionally ignored. Encoder LLM
    # routing is resolved through providers.get_llm("encoder") -> roles.encoder.
    encoder = EncoderConfig()

    jhb = JHBConfig(**raw.get("jhb", {}))
    tail_raw = raw.get("conversation_tail", {})
    if tail_raw is False:
        conversation_tail = ConversationTailConfig(enabled=False)
    elif isinstance(tail_raw, dict):
        conversation_tail = ConversationTailConfig(**tail_raw)
    else:
        conversation_tail = ConversationTailConfig()

    tagger = TaggerConfig(**raw.get("tagger", {}))
    graph = GraphConfig(**raw.get("graph", {}))
    embedder = EmbedderConfig(**raw.get("embedder", {}))

    return JLCConfig(
        encoder=encoder,
        jhb=jhb,
        conversation_tail=conversation_tail,
        tagger=tagger,
        graph=graph,
        embedder=embedder,
    )
