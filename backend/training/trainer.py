"""AlphaZero-style training from a finished game trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from backend.models.base import AlphaZeroNet

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"


@dataclass
class TrainResult:
    model: str
    steps: int
    policy_loss: float
    value_loss: float
    total_loss: float
    samples: int
    saved_to: str | None


def train_from_samples(
    net: AlphaZeroNet,
    samples: list[dict[str, Any]],
    *,
    model_name: str,
    epochs: int = 4,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: torch.device | None = None,
    save: bool = True,
) -> TrainResult:
    """Update network with policy CE + value MSE on game samples."""
    if not samples:
        return TrainResult(model_name, 0, 0.0, 0.0, 0.0, 0, None)

    device = device or next(net.parameters()).device
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    planes = np.stack([s["planes"] for s in samples]).astype(np.float32)
    policies = np.stack([s["policy"] for s in samples]).astype(np.float32)
    values = np.asarray([s["z"] for s in samples], dtype=np.float32)

    n = len(samples)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    steps = 0

    for _ in range(max(1, epochs)):
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            x = torch.from_numpy(planes[idx]).to(device)
            pi = torch.from_numpy(policies[idx]).to(device)
            z = torch.from_numpy(values[idx]).to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, pred_v = net(x)
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(pi * log_probs).sum(dim=-1).mean()
            value_loss = F.mse_loss(pred_v, z)
            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            steps += 1

    net.eval()
    saved_to = None
    if save:
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        path = WEIGHTS_DIR / f"{model_name}.pt"
        torch.save(net.state_dict(), path)
        saved_to = str(path)

    p_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
    v_loss = float(np.mean(value_losses)) if value_losses else 0.0
    return TrainResult(
        model=model_name,
        steps=steps,
        policy_loss=round(p_loss, 5),
        value_loss=round(v_loss, 5),
        total_loss=round(p_loss + v_loss, 5),
        samples=n,
        saved_to=saved_to,
    )
