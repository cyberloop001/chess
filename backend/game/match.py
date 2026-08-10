from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import chess
import torch

from backend.chess_core.config import GameConfig, MCTSConfig, ModelConfig
from backend.mcts import MCTS
from backend.models import build_model
from backend.models.base import AlphaZeroNet

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"


class MatchEngine:
    """Runs an AlphaZero-style match between MLP+MCTS and Transformer+MCTS."""

    def __init__(self, config: GameConfig | None = None, device: str | None = None):
        self.config = config or GameConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_cfg = ModelConfig()
        self._networks: dict[str, AlphaZeroNet] = {}
        self._cancelled = False

    def _load_network(self, name: str) -> AlphaZeroNet:
        key = name.lower()
        if key in self._networks:
            return self._networks[key]
        net = build_model(key, config=self.model_cfg).to(self.device)
        weight_path = WEIGHTS_DIR / f"{key}.pt"
        if weight_path.exists():
            state = torch.load(weight_path, map_location=self.device)
            net.load_state_dict(state)
        net.eval()
        self._networks[key] = net
        return net

    def cancel(self) -> None:
        self._cancelled = True

    async def play(
        self,
        on_event: EventCallback | None = None,
        white_model: str | None = None,
        black_model: str | None = None,
        simulations: int | None = None,
    ) -> dict[str, Any]:
        self._cancelled = False
        white_name = (white_model or self.config.white_model).lower()
        black_name = (black_model or self.config.black_model).lower()
        mcts_cfg = MCTSConfig(
            simulations=simulations or self.config.mcts.simulations,
            c_puct=self.config.mcts.c_puct,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            dirichlet_epsilon=0.0,  # deterministic play for matches
            temperature=0.0,
        )

        white_net = self._load_network(white_name)
        black_net = self._load_network(black_name)
        agents = {
            chess.WHITE: (white_name, MCTS(white_net, mcts_cfg)),
            chess.BLACK: (black_name, MCTS(black_net, mcts_cfg)),
        }

        board = chess.Board()
        history: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            if on_event is None:
                return
            result = on_event(event)
            if asyncio.iscoroutine(result):
                await result

        await emit(
            {
                "type": "match_start",
                "fen": board.fen(),
                "white": white_name,
                "black": black_name,
                "simulations": mcts_cfg.simulations,
            }
        )

        while not board.is_game_over() and len(history) < self.config.max_moves:
            if self._cancelled:
                await emit({"type": "match_cancelled", "fen": board.fen()})
                return {"result": "cancelled", "history": history, "fen": board.fen()}

            name, agent = agents[board.turn]
            await emit(
                {
                    "type": "thinking",
                    "side": "white" if board.turn == chess.WHITE else "black",
                    "model": name,
                    "fen": board.fen(),
                    "ply": len(history),
                }
            )

            # Run MCTS off the event loop
            move, _, stats = await asyncio.to_thread(agent.search, board)
            san = board.san(move)
            board.push(move)
            entry = {
                "ply": len(history) + 1,
                "side": "white" if board.turn == chess.BLACK else "black",
                "model": name,
                "uci": move.uci(),
                "san": san,
                "fen": board.fen(),
                "mcts": asdict(stats),
            }
            history.append(entry)
            await emit({"type": "move", **entry})

            if self.config.move_delay_ms > 0:
                await asyncio.sleep(self.config.move_delay_ms / 1000.0)

        result = board.result(claim_draw=True)
        outcome = {
            "type": "match_end",
            "result": result,
            "fen": board.fen(),
            "winner": _winner_label(result, white_name, black_name),
            "history": history,
            "termination": board.outcome(claim_draw=True).termination.name
            if board.outcome(claim_draw=True)
            else "unknown",
        }
        await emit(outcome)
        return outcome


def _winner_label(result: str, white: str, black: str) -> str:
    if result == "1-0":
        return white
    if result == "0-1":
        return black
    return "draw"
