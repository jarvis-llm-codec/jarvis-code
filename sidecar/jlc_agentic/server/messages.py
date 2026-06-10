"""Web UI WebSocket protocol types."""
from __future__ import annotations

from typing import Literal, TypedDict

Priority = Literal["P0", "P1", "P2", "P3"]


class Block(TypedDict):
    id: str
    title: str
    priority: Priority
    bullets: list[str]
    token: int


class MixerState(TypedDict):
    english_only: bool
    custom_note_enabled: bool
    custom_note: str
    english_only_token: int
    custom_note_token: int
    message_token: int
    total_token: int


class Inbound(TypedDict, total=False):
    type: str
    text: str
    turn_id: str
    key: str
    enabled: bool
    # JARVIS route for this turn (chat | unregistered_coding | deepdive |
    # heavy_deepdive), as decided by the /route_turn judge. Carries through to
    # the chat provider's reasoning effort. Absent/empty = no override.
    route: str


class Outbound(TypedDict, total=False):
    type: str
    seq: int
    text: str
    level: Literal["warn", "error"]
    turn_id: str
    jhb_blocks: list[Block]
    mixer: MixerState
    turn: int
    jhb_token: int
    block_id: str
    title: str
    priority: Priority
    block: Block
    total_token: int
