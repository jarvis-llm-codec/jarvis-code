"""In-memory UI state plus persistent mixer/JHB loading."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jlc_agentic.encoder import JLCEncoder

from .messages import Block, MixerState, Priority

MIXER_PATH = Path.home() / ".jarvis-code" / "mixer.json"


def _count_tokens(text: str) -> int:
    return JLCEncoder.count_tokens(JLCEncoder.__new__(JLCEncoder), text)


def _default_mixer() -> MixerState:
    return {
        "english_only": True,
        "custom_note_enabled": False,
        "custom_note": "",
        "english_only_token": 24,
        "custom_note_token": 0,
        "message_token": 0,
        "total_token": 24,
    }


def load_mixer() -> MixerState:
    state = _default_mixer()
    if MIXER_PATH.exists():
        try:
            raw = json.loads(MIXER_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                state.update({
                    "english_only": bool(raw.get("english_only", state["english_only"])),
                    "custom_note_enabled": bool(raw.get("custom_note_enabled", raw.get("custom_note", "") != "")),
                    "custom_note": str(raw.get("custom_note", "")),
                })
        except Exception:
            pass
    return refresh_mixer_tokens(state)


def save_mixer(state: MixerState) -> None:
    MIXER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MIXER_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_mixer_tokens(state: MixerState, message_text: str = "", jhb_token: int = 0) -> MixerState:
    state["english_only_token"] = 24 if state.get("english_only") else 0
    note = state.get("custom_note", "") if state.get("custom_note_enabled") else ""
    state["custom_note_token"] = _count_tokens(note) if note else 0
    state["message_token"] = _count_tokens(message_text) if message_text else 0
    state["total_token"] = state["english_only_token"] + state["custom_note_token"] + state["message_token"] + jhb_token
    return state


def parse_jhb_blocks(markdown: str) -> list[Block]:
    blocks: list[Block] = []
    parts = re.split(r"(?m)^##\s+", markdown or "")
    for index, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        heading = lines[0].strip()
        match = re.search(r"\[(P[0-3])\]\s*$", heading)
        priority: Priority = "P2"
        if match:
            priority = match.group(1)  # type: ignore[assignment]
            heading = heading[: match.start()].strip()
        bullets = []
        for line in lines[1:]:
            item = line.strip()
            if item.startswith(("- ", "* ")):
                bullets.append(item[2:].strip())
            elif item:
                bullets.append(item)
        body = "\n".join(lines[1:])
        blocks.append({
            "id": f"jhb-{index}-{uuid.uuid5(uuid.NAMESPACE_URL, heading).hex[:8]}",
            "title": heading or f"JHB Block {index}",
            "priority": priority,
            "bullets": bullets[:12] or ([body.strip()] if body.strip() else []),
            "token": _count_tokens(part),
        })
    return blocks


@dataclass
class ConnectionState:
    ws_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)


class UIState:
    def __init__(self) -> None:
        self.connections: dict[str, ConnectionState] = {}
        self.mixer = load_mixer()
        self.turn = 1

    def connect(self) -> ConnectionState:
        ws_id = uuid.uuid4().hex
        state = ConnectionState(ws_id=ws_id)
        self.connections[ws_id] = state
        return state

    def disconnect(self, ws_id: str) -> None:
        self.connections.pop(ws_id, None)

    def next_turn(self) -> int:
        self.turn += 1
        return self.turn


ui_state = UIState()
