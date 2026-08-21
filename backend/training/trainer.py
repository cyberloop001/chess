"""AlphaZero-style training from a finished game trajectory."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
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
    train_trace: list[dict[str, float]] = field(default_factory=list)
    layer_deltas: list[dict[str, Any]] = field(default_factory=list)
    weight_hist: dict[str, Any] | None = None


def _snapshot_params(net: AlphaZeroNet) -> dict[str, torch.Tensor]:
    return {name: p.detach().float().cpu().clone() for name, p in net.named_parameters() if p.requires_grad}


def _short_name(name: str) -> str:
    parts = name.replace(".weight", ".w").replace(".bias", ".b").split(".")
    if len(parts) <= 3:
        return name
    return "…".join(parts[-3:])


def _layer_deltas(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Relative |Δw| / (|w|+ε) per tensor, plus a histogram over all weight updates."""
    rows: list[dict[str, Any]] = []
    all_rel: list[np.ndarray] = []
    for name, w0 in before.items():
        w1 = after.get(name)
        if w1 is None or w0.numel() == 0:
            continue
        delta = (w1 - w0).abs()
        base = w0.abs()
        rel = (delta / (base + 1e-8)).numpy().reshape(-1)
        mean_abs = float(delta.mean().item())
        rms = float(torch.sqrt((delta * delta).mean()).item())
        rel_mean = float(rel.mean())
        rows.append(
            {
                "name": name,
                "short": _short_name(name),
                "numel": int(w0.numel()),
                "mean_abs_delta": round(mean_abs, 8),
                "rms_delta": round(rms, 8),
                "rel_mean": round(rel_mean, 8),
            }
        )
        # Cap contribution so huge tensors don't dominate histogram sampling cost.
        if rel.size > 50_000:
            idx = np.random.choice(rel.size, 50_000, replace=False)
            all_rel.append(rel[idx])
        else:
            all_rel.append(rel)

    rows.sort(key=lambda r: r["rel_mean"], reverse=True)
    hist: dict[str, Any] | None = None
    if all_rel:
        flat = np.concatenate(all_rel)
        # Log-scale-friendly bins for relative updates.
        edges = np.geomspace(1e-8, max(float(flat.max()), 1e-6), 25)
        counts, bin_edges = np.histogram(flat, bins=edges)
        hist = {
            "counts": [int(c) for c in counts.tolist()],
            "edges": [float(e) for e in bin_edges.tolist()],
            "mean_rel": round(float(flat.mean()), 8),
            "median_rel": round(float(np.median(flat)), 8),
            "sampled": int(flat.size),
        }
    return rows, hist or {"counts": [], "edges": [], "mean_rel": 0.0, "median_rel": 0.0, "sampled": 0}


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
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> TrainResult:
    """Update network with policy CE + value MSE on game samples."""
    if not samples:
        return TrainResult(model_name, 0, 0.0, 0.0, 0.0, 0, None)

    device = device or next(net.parameters()).device
    before = _snapshot_params(net)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    planes = np.stack([s["planes"] for s in samples]).astype(np.float32)
    policies = np.stack([s["policy"] for s in samples]).astype(np.float32)
    values = np.asarray([s["z"] for s in samples], dtype=np.float32)

    n = len(samples)
    policy_losses: list[float] = []
    value_losses: list[float] = []
    train_trace: list[dict[str, float]] = []
    steps = 0
    total_epochs = max(1, epochs)

    for epoch in range(total_epochs):
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
            grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
            optimizer.step()

            p_item = float(policy_loss.item())
            v_item = float(value_loss.item())
            t_item = p_item + v_item
            policy_losses.append(p_item)
            value_losses.append(v_item)
            steps += 1
            point = {
                "step": float(steps),
                "epoch": float(epoch + 1),
                "policy_loss": round(p_item, 6),
                "value_loss": round(v_item, 6),
                "total_loss": round(t_item, 6),
                "grad_norm": round(grad_norm, 6),
            }
            train_trace.append(point)
            if on_step is not None:
                on_step({"type": "train_step", **point, "samples": n, "epochs": total_epochs})

    net.eval()
    after = _snapshot_params(net)
    layer_deltas, weight_hist = _layer_deltas(before, after)

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
        train_trace=train_trace,
        layer_deltas=layer_deltas,
        weight_hist=weight_hist,
    )
