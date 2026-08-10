from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

from backend.chess_core.config import MCTSConfig
from backend.chess_core.encoding import ACTION_SIZE, board_to_tensor, encode_move, policy_mask
from backend.models.base import AlphaZeroNet


@dataclass
class MCTSStats:
    simulations: int
    root_value: float
    top_moves: list[dict]


class Node:
    __slots__ = ("parent", "prior", "move", "children", "visit_count", "value_sum", "is_expanded")

    def __init__(self, parent: Node | None, prior: float, move: chess.Move | None = None):
        self.parent = parent
        self.prior = prior
        self.move = move
        self.children: dict[int, Node] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def u(self, c_puct: float, parent_visits: int) -> float:
        return c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

    def best_child(self, c_puct: float) -> Node:
        best_score = -1e9
        best_node = None
        for child in self.children.values():
            score = child.q + child.u(c_puct, self.visit_count)
            if score > best_score:
                best_score = score
                best_node = child
        assert best_node is not None
        return best_node


class MCTS:
    """PUCT Monte Carlo Tree Search driven by a policy/value network."""

    def __init__(self, network: AlphaZeroNet, config: MCTSConfig | None = None):
        self.network = network
        self.config = config or MCTSConfig()

    def search(self, board: chess.Board) -> tuple[chess.Move, np.ndarray, MCTSStats]:
        root = Node(parent=None, prior=1.0)
        self._expand(root, board)

        # Dirichlet noise at root for exploration (training-style; mild at play time)
        if root.children and self.config.dirichlet_epsilon > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(actions))
            eps = self.config.dirichlet_epsilon
            for a, n in zip(actions, noise):
                child = root.children[a]
                child.prior = (1 - eps) * child.prior + eps * float(n)

        for _ in range(self.config.simulations):
            node = root
            scratch = board.copy(stack=False)
            # Selection
            while node.is_expanded and node.children:
                node = node.best_child(self.config.c_puct)
                assert node.move is not None
                scratch.push(node.move)

            # Expansion + evaluation
            if scratch.is_game_over():
                value = self._terminal_value(scratch)
            else:
                value = self._expand(node, scratch)

            # Backup (negate at each step — value from side-to-move)
            self._backup(node, value)

        visit_counts = np.zeros(ACTION_SIZE, dtype=np.float64)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        move, action = self._select_action(board, root, visit_counts)
        policy = visit_counts / max(visit_counts.sum(), 1.0)

        ranked = sorted(root.children.items(), key=lambda kv: kv[1].visit_count, reverse=True)[:5]
        top_moves = []
        for act, child in ranked:
            assert child.move is not None
            top_moves.append(
                {
                    "uci": child.move.uci(),
                    "visits": child.visit_count,
                    "prior": round(child.prior, 4),
                    "q": round(child.q, 4),
                }
            )

        stats = MCTSStats(
            simulations=self.config.simulations,
            root_value=round(root.q, 4),
            top_moves=top_moves,
        )
        return move, policy, stats

    def _expand(self, node: Node, board: chess.Board) -> float:
        planes = board_to_tensor(board)
        priors, value = self.network.infer(planes)
        mask = policy_mask(board)
        priors = priors * mask
        total = priors.sum()
        if total <= 0:
            # Uniform over legal moves if network assigns zero mass
            legal = list(board.legal_moves)
            if not legal:
                node.is_expanded = True
                return value
            p = 1.0 / len(legal)
            for move in legal:
                action = encode_move(board, move)
                node.children[action] = Node(parent=node, prior=p, move=move)
            node.is_expanded = True
            return value

        priors = priors / total
        for move in board.legal_moves:
            action = encode_move(board, move)
            node.children[action] = Node(parent=node, prior=float(priors[action]), move=move)
        node.is_expanded = True
        return value

    def _backup(self, node: Node, value: float) -> None:
        # value is from the side that just expanded / evaluated (side to move at that node)
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent  # type: ignore[assignment]

    def _terminal_value(self, board: chess.Board) -> float:
        # From side-to-move perspective: if checkmate, side to move lost
        if board.is_checkmate():
            return -1.0
        return 0.0

    def _select_action(
        self, board: chess.Board, root: Node, visit_counts: np.ndarray
    ) -> tuple[chess.Move, int]:
        temp = self.config.temperature
        if temp <= 1e-6:
            action = int(np.argmax(visit_counts))
        else:
            probs = visit_counts ** (1.0 / temp)
            probs = probs / probs.sum()
            action = int(np.random.choice(len(probs), p=probs))
        move = root.children[action].move
        assert move is not None
        return move, action
