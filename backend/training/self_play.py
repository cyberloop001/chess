"""Minimal self-play stub for future AlphaZero training loops."""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import torch

from backend.chess_core.config import MCTSConfig, ModelConfig
from backend.chess_core.encoding import board_to_tensor
from backend.mcts import MCTS
from backend.models import build_model


def self_play_game(model_name: str, simulations: int = 32) -> list[dict]:
    net = build_model(model_name, config=ModelConfig())
    net.eval()
    mcts = MCTS(net, MCTSConfig(simulations=simulations, dirichlet_epsilon=0.25, temperature=1.0))
    board = chess.Board()
    trajectory: list[dict] = []

    while not board.is_game_over() and board.fullmove_number < 80:
        move, policy, _ = mcts.search(board)
        trajectory.append(
            {
                "fen": board.fen(),
                "planes": board_to_tensor(board),
                "policy": policy,
                "move": move.uci(),
            }
        )
        board.push(move)

    result = board.result(claim_draw=True)
    z = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
    # Flip value target by side that moved
    for i, step in enumerate(trajectory):
        step["z"] = z if i % 2 == 0 else -z
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AlphaZero-style self-play game")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    traj = self_play_game(args.model, args.simulations)
    print(f"Self-play finished: {len(traj)} plies")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(traj, args.save)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()
