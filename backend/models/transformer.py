from __future__ import annotations

import torch
import torch.nn as nn

from backend.chess_core.config import ModelConfig
from backend.models.base import AlphaZeroNet


class TransformerNet(AlphaZeroNet):
    """Square-token Transformer with AlphaZero policy + value heads."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        cfg = config or ModelConfig()
        self.d_model = cfg.d_model
        self.square_embed = nn.Linear(cfg.board_planes, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, cfg.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

        policy_planes = 73
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, policy_planes),
        )
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 1),
        )
        self._planes = policy_planes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, 8, 8) -> tokens (B, 64, C)
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        h_tok = self.square_embed(tokens) + self.pos_embed
        h_tok = self.encoder(h_tok)
        h_tok = self.norm(h_tok)

        # Per-square policy planes -> flatten to ACTION_SIZE
        policy_logits = self.policy_head(h_tok).reshape(b, 64 * self._planes)
        # Value from mean-pooled representation
        pooled = h_tok.mean(dim=1)
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)
        return policy_logits, value
