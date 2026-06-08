"""FastAPI app and Uvicorn entrypoint for the local Web UI sidecar."""
from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ws import websocket_endpoint

app = FastAPI(title="Jarvis Code UI Sidecar")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.websocket("/ws")(websocket_endpoint)


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


def run(host: str = "127.0.0.1", port: int = 7150) -> None:
    uvicorn.run(app, host=host, port=port, log_level="info")


_started = False
_lock = threading.Lock()


def start_sidecar_once(host: str = "127.0.0.1", port: int = 7150) -> None:
    global _started
    with _lock:
        if _started:
            return
        thread = threading.Thread(target=run, kwargs={"host": host, "port": port}, daemon=True)
        thread.start()
        _started = True
