from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.chess_core.config import GameConfig
from backend.game.match import MatchEngine

app = FastAPI(title="Chess AlphaZero Arena", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_match_lock = asyncio.Lock()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {
        "models": [
            {
                "id": "mlp",
                "name": "MLP + MCTS",
                "description": "Dense AlphaZero policy/value network with PUCT search",
            },
            {
                "id": "transformer",
                "name": "Transformer + MCTS",
                "description": "Square-token Transformer policy/value network with PUCT search",
            },
        ]
    }


@app.websocket("/ws/match")
async def match_socket(ws: WebSocket) -> None:
    await ws.accept()
    engine = MatchEngine(GameConfig())

    async def send(event: dict[str, Any]) -> None:
        await ws.send_json(event)

    try:
        while True:
            payload = await ws.receive_json()
            action = payload.get("action")
            if action == "start":
                if _match_lock.locked():
                    await send({"type": "error", "message": "A match is already running"})
                    continue
                async with _match_lock:
                    await engine.play(
                        on_event=send,
                        white_model=payload.get("white_model", "mlp"),
                        black_model=payload.get("black_model", "transformer"),
                        simulations=int(payload.get("simulations", 64)),
                    )
            elif action == "cancel":
                engine.cancel()
                await send({"type": "info", "message": "Cancel requested"})
            elif action == "ping":
                await send({"type": "pong"})
            else:
                await send({"type": "error", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        engine.cancel()
