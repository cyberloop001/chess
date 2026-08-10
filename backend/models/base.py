from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.chess_core.encoding import ACTION_SIZE


class AlphaZeroNet(nn.Module, ABC):
    """Shared AlphaZero interface: (policy logits, value in [-1, 1])."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    @torch.no_grad()
    def infer(self, planes: np.ndarray) -> tuple[np.ndarray, float]:
        self.eval()
        device = next(self.parameters()).device
        t = torch.from_numpy(planes).unsqueeze(0).to(device)
        logits, value = self.forward(t)
        policy = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return policy, float(value.item())
