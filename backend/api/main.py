from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.analytics.store import append_match, clear_matches, load_matches
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


class MatchMoveIn(BaseModel):
    ply: int
    side: str
    model: str
    uci: str
    san: str
    rootValue: float = 0.0
    topVisits: int = 0
    totalTopVisits: int = 1


class MatchIn(BaseModel):
    id: str | None = None
    playedAt: str | None = None
    white: str
    black: str
    simulations: int = 64
    result: str
    winner: str
    termination: str
    moves: list[MatchMoveIn] = Field(default_factory=list)


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
        ],
        "rules": {
            "pairing": "White = MLP, Black = Transformer",
            "mode": "One game per Start · result is win/loss/draw · then both self-train",
        },
    }


@app.get("/api/analytics/matches")
def get_matches() -> dict[str, Any]:
    matches = load_matches()
    return {"matches": matches, "count": len(matches)}


@app.post("/api/analytics/matches")
def post_match(match: MatchIn) -> dict[str, Any]:
    payload = match.model_dump()
    if not payload.get("id"):
        from backend.analytics.store import build_match_record

        payload = build_match_record(
            white=payload["white"],
            black=payload["black"],
            simulations=payload["simulations"],
            result=payload["result"],
            winner=payload["winner"],
            termination=payload["termination"],
            history=[
                {
                    "ply": m["ply"],
                    "side": m["side"],
                    "model": m["model"],
                    "uci": m["uci"],
                    "san": m["san"],
                    "mcts": {
                        "root_value": m["rootValue"],
                        "top_moves": [
                            {"visits": m["topVisits"]},
                        ],
                    },
                }
                for m in payload["moves"]
            ],
        )
    saved = append_match(payload)
    return {"ok": True, "match": saved}


@app.delete("/api/analytics/matches")
def delete_matches() -> dict[str, Any]:
    clear_matches()
    return {"ok": True, "count": 0}


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
                        white_model="mlp",
                        black_model="transformer",
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
