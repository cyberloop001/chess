from __future__ import annotations

import torch
import torch.nn as nn

from backend.chess_core.config import ModelConfig
from backend.chess_core.encoding import ACTION_SIZE
from backend.models.base import AlphaZeroNet


class MLPNet(AlphaZeroNet):
    """Dense AlphaZero-style policy + value network."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        cfg = config or ModelConfig()
        in_dim = cfg.board_planes * 8 * 8
        layers: list[nn.Module] = []
        prev = in_dim
        for h in cfg.mlp_hidden:
            layers.extend([nn.Linear(prev, h), nn.ReLU(inplace=True)])
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, ACTION_SIZE)
        self.value_head = nn.Sequential(
            nn.Linear(prev, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, 8, 8)
        h = self.trunk(x.flatten(1))
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy_logits, value.squeeze(-1)
