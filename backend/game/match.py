from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import chess
import numpy as np
import torch

from backend.chess_core.config import GameConfig, MCTSConfig, ModelConfig
from backend.chess_core.encoding import board_to_tensor
from backend.mcts import MCTS
from backend.models import build_model
from backend.models.base import AlphaZeroNet
from backend.training.trainer import train_from_samples

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"

# Competition always uses these two architectures
MLP_ID = "mlp"
TRANSFORMER_ID = "transformer"


@dataclass
class SeriesConfig:
    simulations: int = 64
    move_delay_ms: int = 400
    max_games: int = 32
    train_epochs: int = 2
    train_batch_size: int = 16
    train_lr: float = 1e-3
    swap_colors_each_game: bool = False
    # Keep playing while result is draw; stop when one model wins
    play_until_decisive: bool = True


class MatchEngine:
    """Runs MLP vs Transformer with rematch-until-win and post-game training."""

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
            try:
                state = torch.load(weight_path, map_location=self.device, weights_only=True)
                net.load_state_dict(state)
            except (RuntimeError, ValueError, TypeError) as exc:
                # Incompatible checkpoint (e.g. different hidden sizes) — start fresh
                print(f"Skipping incompatible weights at {weight_path}: {exc}")
                try:
                    weight_path.unlink(missing_ok=True)
                except OSError:
                    pass
        net.eval()
        self._networks[key] = net
        return net

    def cancel(self) -> None:
        self._cancelled = True

    async def play_series(
        self,
        on_event: EventCallback | None = None,
        *,
        white_model: str | None = None,
        black_model: str | None = None,
        simulations: int | None = None,
        play_until_win: bool = True,
        max_games: int = 32,
    ) -> dict[str, Any]:
        """Play MLP vs Transformer; rematch on draws; train after each game."""
        self._cancelled = False
        series = SeriesConfig(
            simulations=self.config.mcts.simulations,
            move_delay_ms=self.config.move_delay_ms,
            max_games=max_games,
            play_until_decisive=play_until_win,
        )
        if simulations is not None:
            series.simulations = simulations

        # Fix sides to the two competing models (ignore same-model selections)
        white_name, black_name = _resolve_sides(white_model, black_model)
        mlp_net = self._load_network(MLP_ID)
        transformer_net = self._load_network(TRANSFORMER_ID)
        nets = {MLP_ID: mlp_net, TRANSFORMER_ID: transformer_net}

        async def emit(event: dict[str, Any]) -> None:
            if on_event is None:
                return
            result = on_event(event)
            if asyncio.iscoroutine(result):
                await result

        await emit(
            {
                "type": "series_start",
                "white": white_name,
                "black": black_name,
                "simulations": series.simulations,
                "play_until_win": series.play_until_decisive,
                "max_games": series.max_games,
            }
        )

        games: list[dict[str, Any]] = []
        series_winner: str | None = None

        for game_idx in range(series.max_games):
            if self._cancelled:
                await emit({"type": "match_cancelled", "fen": chess.Board().fen()})
                break

            if series.swap_colors_each_game and game_idx > 0:
                white_name, black_name = black_name, white_name

            outcome = await self._play_one(
                on_event=emit,
                white_name=white_name,
                black_name=black_name,
                nets=nets,
                series=series,
                game_index=game_idx + 1,
            )
            if outcome.get("result") == "cancelled":
                break

            games.append(outcome)

            # Self-train both models from this game's positions
            train_info = await asyncio.to_thread(
                self._train_from_outcome,
                nets,
                outcome,
                series,
            )
            await emit({"type": "training_complete", "game": game_idx + 1, **train_info})

            winner = outcome.get("winner")
            if winner in (MLP_ID, TRANSFORMER_ID):
                series_winner = winner
                if series.play_until_decisive:
                    break
            # Draw → continue another game

        final = {
            "type": "series_end",
            "games_count": len(games),
            "series_winner": series_winner or "none",
            "games": [
                {
                    "result": g.get("result"),
                    "winner": g.get("winner"),
                    "white": g.get("white"),
                    "black": g.get("black"),
                }
                for g in games
            ],
        }
        await emit(final)
        return final

    async def play(
        self,
        on_event: EventCallback | None = None,
        white_model: str | None = None,
        black_model: str | None = None,
        simulations: int | None = None,
    ) -> dict[str, Any]:
        """Backward-compatible single entry → series until one model wins."""
        return await self.play_series(
            on_event=on_event,
            white_model=white_model,
            black_model=black_model,
            simulations=simulations,
            play_until_win=True,
            max_games=32,
        )

    async def _play_one(
        self,
        *,
        on_event: EventCallback | None,
        white_name: str,
        black_name: str,
        nets: dict[str, AlphaZeroNet],
        series: SeriesConfig,
        game_index: int,
    ) -> dict[str, Any]:
        mcts_cfg = MCTSConfig(
            simulations=series.simulations,
            c_puct=self.config.mcts.c_puct,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            dirichlet_epsilon=0.15,
            temperature=0.0,
        )
        agents = {
            chess.WHITE: (white_name, MCTS(nets[white_name], mcts_cfg)),
            chess.BLACK: (black_name, MCTS(nets[black_name], mcts_cfg)),
        }

        board = chess.Board()
        history: list[dict[str, Any]] = []
        train_steps: list[dict[str, Any]] = []

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
                "game_index": game_index,
            }
        )

        while not board.is_game_over() and len(history) < self.config.max_moves:
            if self._cancelled:
                await emit({"type": "match_cancelled", "fen": board.fen()})
                return {"result": "cancelled", "history": history, "fen": board.fen()}

            name, agent = agents[board.turn]
            side = "white" if board.turn == chess.WHITE else "black"
            planes = board_to_tensor(board)

            await emit(
                {
                    "type": "thinking",
                    "side": side,
                    "model": name,
                    "fen": board.fen(),
                    "ply": len(history),
                    "game_index": game_index,
                }
            )

            move, policy, stats = await asyncio.to_thread(agent.search, board)
            san = board.san(move)
            board.push(move)

            train_steps.append(
                {
                    "model": name,
                    "side": side,
                    "planes": planes,
                    "policy": policy.astype(np.float32),
                }
            )

            entry = {
                "ply": len(history) + 1,
                "side": side,
                "model": name,
                "uci": move.uci(),
                "san": san,
                "fen": board.fen(),
                "mcts": asdict(stats),
                "game_index": game_index,
            }
            history.append(entry)
            await emit({"type": "move", **entry})

            if series.move_delay_ms > 0:
                await asyncio.sleep(series.move_delay_ms / 1000.0)

        result = board.result(claim_draw=True)
        # Attach value targets from the side that moved
        z_white = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
        for step in train_steps:
            step["z"] = z_white if step["side"] == "white" else -z_white

        outcome = {
            "type": "match_end",
            "result": result,
            "fen": board.fen(),
            "white": white_name,
            "black": black_name,
            "winner": _winner_label(result, white_name, black_name),
            "history": history,
            "train_steps": train_steps,
            "game_index": game_index,
            "termination": board.outcome(claim_draw=True).termination.name
            if board.outcome(claim_draw=True)
            else "unknown",
        }
        await emit({k: v for k, v in outcome.items() if k != "train_steps"})
        return outcome

    def _train_from_outcome(
        self,
        nets: dict[str, AlphaZeroNet],
        outcome: dict[str, Any],
        series: SeriesConfig,
    ) -> dict[str, Any]:
        steps: list[dict[str, Any]] = outcome.get("train_steps") or []
        reports = []
        for model_id in (MLP_ID, TRANSFORMER_ID):
            samples = [s for s in steps if s["model"] == model_id]
            report = train_from_samples(
                nets[model_id],
                samples,
                model_name=model_id,
                epochs=series.train_epochs,
                batch_size=series.train_batch_size,
                lr=series.train_lr,
                device=self.device,
                save=True,
            )
            reports.append(asdict(report))
        return {"models": reports}


def _resolve_sides(white_model: str | None, black_model: str | None) -> tuple[str, str]:
    """Always White=MLP, Black=Transformer."""
    del white_model, black_model
    return MLP_ID, TRANSFORMER_ID


def _winner_label(result: str, white: str, black: str) -> str:
    if result == "1-0":
        return white
    if result == "0-1":
        return black
    return "draw"
