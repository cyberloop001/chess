"""Persistent analytics / match history on the server."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HISTORY_PATH = DATA_DIR / "match_history.json"
MAX_MATCHES = 200
_lock = threading.Lock()


def _ensure_store() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        HISTORY_PATH.write_text("[]", encoding="utf-8")


def load_matches() -> list[dict[str, Any]]:
    with _lock:
        _ensure_store()
        try:
            raw = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
        except (json.JSONDecodeError, OSError):
            return []


def save_matches(matches: list[dict[str, Any]]) -> None:
    with _lock:
        _ensure_store()
        HISTORY_PATH.write_text(
            json.dumps(matches[:MAX_MATCHES], indent=2),
            encoding="utf-8",
        )


def clear_matches() -> None:
    save_matches([])


def append_match(match: dict[str, Any]) -> dict[str, Any]:
    matches = load_matches()
    matches.insert(0, match)
    save_matches(matches)
    return match


def build_match_record(
    *,
    white: str,
    black: str,
    simulations: int,
    result: str,
    winner: str,
    termination: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    moves = []
    for entry in history:
        mcts = entry.get("mcts") or {}
        top = mcts.get("top_moves") or []
        visits = [int(t.get("visits", 0)) for t in top]
        top_visits = visits[0] if visits else 0
        total_top = sum(visits) or 1
        moves.append(
            {
                "ply": entry.get("ply"),
                "side": entry.get("side"),
                "model": entry.get("model"),
                "uci": entry.get("uci"),
                "san": entry.get("san"),
                "rootValue": float(mcts.get("root_value", 0.0)),
                "topVisits": top_visits,
                "totalTopVisits": total_top,
            }
        )

    return {
        "id": f"{int(datetime.now(tz=timezone.utc).timestamp() * 1000)}-{uuid.uuid4().hex[:6]}",
        "playedAt": datetime.now(tz=timezone.utc).isoformat(),
        "white": white,
        "black": black,
        "simulations": simulations,
        "result": result,
        "winner": winner,
        "termination": termination,
        "moves": moves,
    }
