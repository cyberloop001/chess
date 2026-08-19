"""Self-play training history for the Analytics dashboard."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HISTORY_PATH = DATA_DIR / "selfplay_history.json"
MAX_GAMES = 500
_lock = threading.Lock()


def _ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def load_selfplay() -> list[dict[str, Any]]:
    with _lock:
        _ensure_store()
        try:
            raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except (json.JSONDecodeError, OSError):
            return []


def save_selfplay(games: list[dict[str, Any]]) -> None:
    with _lock:
        _ensure_store()
        HISTORY_PATH.write_text(json.dumps(games[:MAX_GAMES], indent=2), encoding="utf-8")


def clear_selfplay() -> None:
    save_selfplay([])


def append_selfplay(game: dict[str, Any]) -> dict[str, Any]:
    games = load_selfplay()
    games.insert(0, game)
    save_selfplay(games)
    return game


def new_run_id() -> str:
    return uuid.uuid4().hex[:10]


def build_selfplay_record(
    *,
    run_id: str,
    model: str,
    game: int,
    games_in_run: int,
    simulations: int,
    plies: int,
    result: str,
    termination: str,
    policy_loss: float,
    value_loss: float,
    total_loss: float,
    samples: int,
    replay_size: int,
    saved_to: str | None,
) -> dict[str, Any]:
    return {
        "id": f"{int(datetime.now(tz=timezone.utc).timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        "runId": run_id,
        "playedAt": datetime.now(tz=timezone.utc).isoformat(),
        "model": model,
        "game": game,
        "gamesInRun": games_in_run,
        "simulations": simulations,
        "plies": plies,
        "result": result,
        "termination": termination,
        "policyLoss": round(float(policy_loss), 5),
        "valueLoss": round(float(value_loss), 5),
        "totalLoss": round(float(total_loss), 5),
        "samples": int(samples),
        "replaySize": int(replay_size),
        "savedTo": saved_to,
    }
