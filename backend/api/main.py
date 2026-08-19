from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.analytics.selfplay_store import clear_selfplay, load_selfplay
from backend.analytics.store import append_match, clear_matches, load_matches
from backend.chess_core.config import GameConfig, MCTSConfig
from backend.game.human_match import HumanMatchEngine
from backend.game.match import MatchEngine
from backend.training.self_play import train_self_play

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
            "mode": "User sets train count N · duel → train · repeat N times",
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


@app.get("/api/analytics/selfplay")
def get_selfplay() -> dict[str, Any]:
    games = load_selfplay()
    return {"games": games, "count": len(games)}


@app.delete("/api/analytics/selfplay")
def delete_selfplay() -> dict[str, Any]:
    clear_selfplay()
    return {"ok": True, "count": 0}


@app.websocket("/ws/selfplay")
async def selfplay_socket(ws: WebSocket) -> None:
    await ws.accept()
    cancel_event = threading.Event()
    play_task: asyncio.Task[Any] | None = None
    loop = asyncio.get_running_loop()

    async def send(event: dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:
            pass

    def emit(event: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(send(event), loop)

    async def run_play(model: str, games: int, simulations: int) -> None:
        async with _match_lock:
            await asyncio.to_thread(
                train_self_play,
                model,
                games=games,
                simulations=simulations,
                on_event=emit,
                cancel_event=cancel_event,
            )

    try:
        while True:
            payload = await ws.receive_json()
            action = payload.get("action")
            if action == "start":
                if (play_task is not None and not play_task.done()) or _match_lock.locked():
                    await send({"type": "error", "message": "A match or self-play run is already running"})
                    continue
                model = str(payload.get("model") or "transformer").lower()
                if model not in ("mlp", "transformer"):
                    model = "transformer"
                try:
                    games = max(1, min(int(payload.get("games") or 4), 100))
                except (TypeError, ValueError):
                    games = 4
                try:
                    raw_sims = payload.get("simulations")
                    simulations = (
                        MCTSConfig().simulations
                        if raw_sims in (None, "", 0, "0")
                        else max(8, min(int(raw_sims), 1024))
                    )
                except (TypeError, ValueError):
                    simulations = MCTSConfig().simulations
                cancel_event.clear()
                play_task = asyncio.create_task(run_play(model, games, simulations))

                def _on_done(task: asyncio.Task[Any]) -> None:
                    try:
                        exc = task.exception()
                    except asyncio.CancelledError:
                        return
                    if exc is not None:
                        asyncio.create_task(send({"type": "error", "message": f"Self-play failed: {exc}"}))

                play_task.add_done_callback(_on_done)
            elif action == "cancel":
                cancel_event.set()
                await send({"type": "info", "message": "Cancel requested — finishes the current game first"})
            elif action == "ping":
                await send({"type": "pong"})
            else:
                await send({"type": "error", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        cancel_event.set()
        if play_task is not None and not play_task.done():
            play_task.cancel()


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
                try:
                    train_count = int(payload.get("train_count") or payload.get("max_games") or 1)
                except (TypeError, ValueError):
                    train_count = 1
                train_count = max(1, min(train_count, 100))
                async with _match_lock:
                    await engine.play(
                        on_event=send,
                        white_model="mlp",
                        black_model="transformer",
                        train_count=train_count,
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


@app.websocket("/ws/human")
async def human_socket(ws: WebSocket) -> None:
    await ws.accept()
    engine = HumanMatchEngine(GameConfig())
    play_task: asyncio.Task[Any] | None = None

    async def send(event: dict[str, Any]) -> None:
        try:
            await ws.send_json(event)
        except Exception:
            # Client disconnected mid-game
            pass

    async def run_play(human_color: str, model: str) -> None:
        async with _match_lock:
            await engine.play(
                on_event=send,
                human_color=human_color,
                model=model,
            )

    try:
        while True:
            payload = await ws.receive_json()
            action = payload.get("action")
            if action == "start":
                if (play_task is not None and not play_task.done()) or _match_lock.locked():
                    await send({"type": "error", "message": "A match is already running"})
                    continue
                human_color = str(payload.get("human_color") or "white")
                model = str(payload.get("model") or "transformer")
                # Must not block this receive loop — human moves arrive on the same socket.
                play_task = asyncio.create_task(run_play(human_color, model))

                def _on_done(task: asyncio.Task[Any]) -> None:
                    try:
                        exc = task.exception()
                    except asyncio.CancelledError:
                        return
                    if exc is not None:
                        asyncio.create_task(
                            send({"type": "error", "message": f"Match failed: {exc}"})
                        )

                play_task.add_done_callback(_on_done)
            elif action == "move":
                uci = str(payload.get("uci") or "")
                result = engine.submit_human_move(uci)
                if not result.get("ok"):
                    await send({"type": "error", "message": result.get("message", "Move failed")})
            elif action == "cancel":
                engine.cancel()
                if play_task is not None and not play_task.done():
                    play_task.cancel()
                await send({"type": "info", "message": "Cancel requested"})
            elif action == "ping":
                await send({"type": "pong"})
            else:
                await send({"type": "error", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        engine.cancel()
        if play_task is not None and not play_task.done():
            play_task.cancel()
