"""AlphaZero-style self-play for MLP or Transformer.

Games run until a real terminal (mate / claimed draw), a high ply ceiling
treated as a draw, or a lopsided MCTS value held long enough to resign.
The same rules apply to both networks.
"""

from __future__ import annotations

import argparse
import threading
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chess
import torch

from backend.analytics.selfplay_store import append_selfplay, build_selfplay_record, new_run_id
from backend.chess_core.config import MCTSConfig, ModelConfig, TrainConfig
from backend.chess_core.encoding import board_to_tensor
from backend.mcts import MCTS
from backend.models import build_model
from backend.models.base import AlphaZeroNet
from backend.training.replay import get_replay_buffer
from backend.training.trainer import WEIGHTS_DIR, train_from_samples

WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
DRAW = "1/2-1/2"


def _load_network(model_name: str, device: torch.device) -> AlphaZeroNet:
    net = build_model(model_name, config=ModelConfig()).to(device)
    weight_path = WEIGHTS_DIR / f"{model_name}.pt"
    if weight_path.exists():
        try:
            state = torch.load(weight_path, map_location=device, weights_only=True)
            net.load_state_dict(state)
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"Skipping incompatible weights at {weight_path}: {exc}")
    net.eval()
    return net


def _assign_values(trajectory: list[dict[str, Any]], z: float) -> None:
    for i, step in enumerate(trajectory):
        step["z"] = z if i % 2 == 0 else -z


def self_play_game(
    model_name: str,
    simulations: int | None = None,
    *,
    max_plies: int | None = None,
    resign_threshold: float | None = None,
    resign_plies: int | None = None,
    net: AlphaZeroNet | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    train_cfg = TrainConfig()
    simulations = MCTSConfig().simulations if simulations is None else simulations
    max_plies = train_cfg.self_play_max_plies if max_plies is None else max_plies
    resign_threshold = (
        train_cfg.self_play_resign_threshold if resign_threshold is None else resign_threshold
    )
    resign_plies = train_cfg.self_play_resign_plies if resign_plies is None else resign_plies

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if net is None:
        net = _load_network(model_name, device)

    mcts = MCTS(
        net,
        MCTSConfig(simulations=simulations, dirichlet_epsilon=0.25, temperature=1.0),
    )
    board = chess.Board()
    trajectory: list[dict[str, Any]] = []
    streak = 0
    streak_sign = 0
    result = DRAW
    termination = "unknown"

    while True:
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            result = board.result(claim_draw=True)
            termination = outcome.termination.name.lower() if outcome else "game_over"
            break
        if len(trajectory) >= max_plies:
            result = DRAW
            termination = "ply_limit"
            break
        if streak >= resign_plies and streak_sign != 0:
            result = WHITE_WIN if streak_sign > 0 else BLACK_WIN
            termination = "resign"
            break

        move, policy, stats = mcts.search(board)
        white_value = stats.root_value if board.turn == chess.WHITE else -stats.root_value
        sign = 0
        if white_value >= resign_threshold:
            sign = 1
        elif white_value <= -resign_threshold:
            sign = -1
        if sign != 0 and sign == streak_sign:
            streak += 1
        else:
            streak_sign = sign
            streak = 1 if sign else 0

        trajectory.append(
            {
                "fen": board.fen(),
                "planes": board_to_tensor(board),
                "policy": policy,
                "move": move.uci(),
                "model": model_name,
            }
        )
        board.push(move)

    z = {WHITE_WIN: 1.0, BLACK_WIN: -1.0}.get(result, 0.0)
    _assign_values(trajectory, z)
    return {
        "trajectory": trajectory,
        "result": result,
        "termination": termination,
        "plies": len(trajectory),
        "model": model_name,
    }


def train_self_play(
    model_name: str,
    *,
    games: int = 1,
    simulations: int | None = None,
    max_plies: int | None = None,
    save_traj: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    train_cfg = TrainConfig()
    sims = MCTSConfig().simulations if simulations is None else simulations
    total = max(1, games)
    run_id = new_run_id()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = _load_network(model_name, device)
    buffer = get_replay_buffer(model_name)
    reports: list[dict[str, Any]] = []
    last_game: dict[str, Any] | None = None

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    emit(
        {
            "type": "selfplay_start",
            "run_id": run_id,
            "model": model_name,
            "games": total,
            "simulations": sims,
        }
    )

    cancelled = False
    for game_idx in range(total):
        if cancel_event is not None and cancel_event.is_set():
            cancelled = True
            break
        emit(
            {
                "type": "selfplay_thinking",
                "run_id": run_id,
                "model": model_name,
                "game": game_idx + 1,
                "games": total,
            }
        )
        game = self_play_game(
            model_name,
            sims,
            max_plies=max_plies,
            net=net,
            device=device,
        )
        last_game = game
        steps = game["trajectory"]
        train_steps = buffer.prepare_training_batch(steps, max_total=train_cfg.replay_batch_max)
        report = train_from_samples(
            net,
            train_steps,
            model_name=model_name,
            epochs=train_cfg.epochs,
            batch_size=train_cfg.batch_size,
            lr=train_cfg.lr,
            device=device,
            save=True,
        )
        record = build_selfplay_record(
            run_id=run_id,
            model=model_name,
            game=game_idx + 1,
            games_in_run=total,
            simulations=sims,
            plies=game["plies"],
            result=game["result"],
            termination=game["termination"],
            policy_loss=report.policy_loss,
            value_loss=report.value_loss,
            total_loss=report.total_loss,
            samples=report.samples,
            replay_size=len(buffer),
            saved_to=report.saved_to,
        )
        append_selfplay(record)
        info = asdict(report)
        info.update(
            {
                "game": game_idx + 1,
                "plies": game["plies"],
                "result": game["result"],
                "termination": game["termination"],
                "replay_size": len(buffer),
            }
        )
        reports.append(info)
        print(
            f"{model_name} game {game_idx + 1}/{total}: {game['plies']} plies "
            f"{game['result']} ({game['termination']})  loss={report.total_loss}"
        )
        emit({"type": "selfplay_game", **record, "games": total})

    if save_traj and last_game is not None:
        save_traj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(last_game["trajectory"], save_traj)
        print(f"Saved last trajectory to {save_traj}")

    emit(
        {
            "type": "selfplay_complete" if not cancelled else "selfplay_cancelled",
            "run_id": run_id,
            "model": model_name,
            "game_count": len(reports),
            "games": total,
        }
    )
    return {"model": model_name, "run_id": run_id, "games": reports, "cancelled": cancelled}


def main() -> None:
    train_cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Self-play train MLP or Transformer")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--simulations", type=int, default=MCTSConfig().simulations)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=train_cfg.self_play_max_plies)
    parser.add_argument("--save", type=Path, default=None, help="Optional path for last trajectory")
    parser.add_argument("--no-train", action="store_true", help="Generate a game only, do not update weights")
    args = parser.parse_args()

    if args.no_train:
        game = self_play_game(args.model, args.simulations, max_plies=args.max_plies)
        print(
            f"Self-play finished: {game['plies']} plies  {game['result']} ({game['termination']})"
        )
        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save(game["trajectory"], args.save)
            print(f"Saved to {args.save}")
        return

    train_self_play(
        args.model,
        games=args.games,
        simulations=args.simulations,
        max_plies=args.max_plies,
        save_traj=args.save,
    )


if __name__ == "__main__":
    main()
