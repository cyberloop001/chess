"""Human vs model match with post-game training."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import chess
import numpy as np
import torch

from backend.analytics.store import append_match, build_match_record
from backend.chess_core.config import GameConfig, MCTSConfig, ModelConfig
from backend.chess_core.encoding import ACTION_SIZE, board_to_tensor, encode_move
from backend.mcts import MCTS
from backend.models import build_model
from backend.models.base import AlphaZeroNet
from backend.training.replay import get_replay_buffer
from backend.training.trainer import train_from_samples

EventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


class HumanMatchEngine:
    """Interactive human vs MLP/Transformer game, then train the model."""

    def __init__(self, config: GameConfig | None = None, device: str | None = None):
        self.config = config or GameConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_cfg = ModelConfig()
        self._networks: dict[str, AlphaZeroNet] = {}
        self._cancelled = False
        self._active = False
        self._waiting_human = False
        self._human_move_future: asyncio.Future[str] | None = None
        self._board: chess.Board | None = None

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
        fut = self._human_move_future
        if fut is not None and not fut.done():
            fut.cancel()

    def submit_human_move(self, uci: str) -> dict[str, Any]:
        if not self._active or not self._waiting_human or self._board is None:
            return {"ok": False, "message": "Not waiting for a human move"}
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return {"ok": False, "message": f"Invalid UCI: {uci}"}
        if move not in self._board.legal_moves:
            if len(uci) == 4:
                promo = chess.Move.from_uci(uci + "q")
                if promo in self._board.legal_moves:
                    move = promo
                    uci = promo.uci()
                else:
                    return {"ok": False, "message": "Illegal move"}
            else:
                return {"ok": False, "message": "Illegal move"}
        fut = self._human_move_future
        if fut is None or fut.done():
            return {"ok": False, "message": "No pending move request"}
        fut.set_result(uci)
        return {"ok": True, "uci": uci}

    async def play(
        self,
        on_event: EventCallback,
        *,
        human_color: str = "white",
        model: str = "transformer",
        simulations: int | None = None,
    ) -> dict[str, Any]:
        self._cancelled = False
        self._active = True
        self._waiting_human = False
        self._human_move_future = None

        model_id = model.lower().strip()
        if model_id not in ("mlp", "transformer"):
            model_id = "transformer"
        human_is_white = human_color.lower().startswith("w")
        human_color_const = chess.WHITE if human_is_white else chess.BLACK
        white_name = "human" if human_is_white else model_id
        black_name = model_id if human_is_white else "human"

        sims = simulations if simulations is not None else self.config.mcts.simulations
        mcts_cfg = MCTSConfig(
            simulations=max(int(sims), 1),
            c_puct=self.config.mcts.c_puct,
            dirichlet_alpha=self.config.mcts.dirichlet_alpha,
            dirichlet_epsilon=0.25,
            temperature=0.0,
        )
        net = self._load_network(model_id)
        agent = MCTS(net, mcts_cfg)
        board = chess.Board()
        self._board = board
        history: list[dict[str, Any]] = []
        train_steps: list[dict[str, Any]] = []

        async def emit(event: dict[str, Any]) -> None:
            result = on_event(event)
            if asyncio.iscoroutine(result):
                await result

        await emit(
            {
                "type": "match_start",
                "mode": "human",
                "fen": board.fen(),
                "white": white_name,
                "black": black_name,
                "human_color": "white" if human_is_white else "black",
                "model": model_id,
                "simulations": mcts_cfg.simulations,
            }
        )

        try:
            while not board.is_game_over(claim_draw=True):
                if self._cancelled:
                    await emit({"type": "match_cancelled", "fen": board.fen()})
                    return {"result": "cancelled", "fen": board.fen()}

                side = "white" if board.turn == chess.WHITE else "black"
                is_human = board.turn == human_color_const
                planes = board_to_tensor(board)

                if is_human:
                    legal = [m.uci() for m in board.legal_moves]
                    await emit(
                        {
                            "type": "your_turn",
                            "side": side,
                            "fen": board.fen(),
                            "ply": len(history),
                            "legal_moves": legal,
                        }
                    )
                    try:
                        uci = await self._await_human_move()
                    except asyncio.CancelledError:
                        await emit({"type": "match_cancelled", "fen": board.fen()})
                        return {"result": "cancelled", "fen": board.fen()}

                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        await emit({"type": "error", "message": f"Illegal move: {uci}"})
                        continue

                    san = board.san(move)
                    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                    policy[encode_move(board, move)] = 1.0
                    board.push(move)

                    entry = {
                        "ply": len(history) + 1,
                        "side": side,
                        "model": "human",
                        "uci": move.uci(),
                        "san": san,
                        "fen": board.fen(),
                    }
                    train_steps.append(
                        {
                            "model": model_id,
                            "side": side,
                            "planes": planes,
                            "policy": policy,
                        }
                    )
                    history.append(entry)
                    await emit({"type": "move", **entry})
                else:
                    await emit(
                        {
                            "type": "thinking",
                            "side": side,
                            "model": model_id,
                            "fen": board.fen(),
                            "ply": len(history),
                        }
                    )
                    move, policy, stats = await asyncio.to_thread(agent.search, board)
                    san = board.san(move)
                    board.push(move)
                    entry = {
                        "ply": len(history) + 1,
                        "side": side,
                        "model": model_id,
                        "uci": move.uci(),
                        "san": san,
                        "fen": board.fen(),
                        "mcts": asdict(stats),
                    }
                    train_steps.append(
                        {
                            "model": model_id,
                            "side": side,
                            "planes": planes,
                            "policy": policy.astype(np.float32),
                        }
                    )
                    history.append(entry)
                    await emit({"type": "move", **entry})
                    if self.config.move_delay_ms > 0:
                        await asyncio.sleep(self.config.move_delay_ms / 1000.0)

            result = board.result(claim_draw=True)
            outcome = board.outcome(claim_draw=True)
            termination = outcome.termination.name if outcome else "unknown"
            winner = _winner_label(result, white_name, black_name)
            z_white = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
            for step in train_steps:
                step["z"] = z_white if step["side"] == "white" else -z_white

            end_event = {
                "type": "match_end",
                "mode": "human",
                "result": result,
                "fen": board.fen(),
                "white": white_name,
                "black": black_name,
                "winner": winner,
                "termination": termination,
                "model": model_id,
                "human_color": "white" if human_is_white else "black",
            }
            await emit(end_event)

            try:
                record = build_match_record(
                    white=white_name,
                    black=black_name,
                    simulations=mcts_cfg.simulations,
                    result=result,
                    winner=winner,
                    termination=termination,
                    history=history,
                )
                append_match(record)
                await emit({"type": "analytics_saved", "match_id": record["id"]})
            except Exception as exc:  # noqa: BLE001
                await emit({"type": "info", "message": f"Analytics save failed: {exc}"})

            await emit({"type": "training_start", "model": model_id})
            try:
                train_info = await asyncio.to_thread(
                    self._train_model,
                    net,
                    model_id,
                    train_steps,
                )
                await emit({"type": "training_complete", "model": model_id, **train_info})
            except Exception as exc:  # noqa: BLE001
                await emit(
                    {
                        "type": "error",
                        "message": f"Training failed: {exc}",
                    }
                )
                train_info = {"models": []}

            # Do not spread end_event here — it contains type="match_end" and would
            # overwrite series_end, leaving the UI stuck on "training…".
            await emit(
                {
                    "type": "series_end",
                    "game_count": 1,
                    "train_count": 1,
                    "series_winner": winner,
                    "result": result,
                    "termination": termination,
                    "white": white_name,
                    "black": black_name,
                    "winner": winner,
                    "fen": board.fen(),
                    "mode": "human",
                    "model": model_id,
                    "human_color": "white" if human_is_white else "black",
                }
            )
            return {**end_event, "train": train_info}
        finally:
            self._active = False
            self._waiting_human = False
            self._board = None
            self._human_move_future = None

    async def _await_human_move(self) -> str:
        loop = asyncio.get_running_loop()
        self._human_move_future = loop.create_future()
        self._waiting_human = True
        try:
            return await self._human_move_future
        finally:
            self._waiting_human = False
            self._human_move_future = None

    def _train_model(
        self,
        net: AlphaZeroNet,
        model_id: str,
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        train_cfg = self.config.train
        buffer = get_replay_buffer()
        train_steps = buffer.prepare_training_batch(
            steps,
            max_total=train_cfg.replay_batch_max,
        )
        report = train_from_samples(
            net,
            train_steps,
            model_name=model_id,
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            lr=train_cfg.lr,
            device=self.device,
            save=True,
        )
        info = asdict(report)
        info["replay_size"] = len(buffer)
        info["game_samples"] = len(steps)
        return {"models": [info], "replay_size": len(buffer)}


def _winner_label(result: str, white: str, black: str) -> str:
    if result == "1-0":
        return white
    if result == "0-1":
        return black
    return "draw"
