"""Persistent replay buffer for AlphaZero-style training."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_PATH = DATA_DIR / "replay_buffer.npz"
_lock = threading.Lock()


@dataclass
class ReplayBuffer:
    """Stores recent (planes, policy, z) samples across games."""

    max_samples: int = 40_000
    path: Path = field(default_factory=lambda: DEFAULT_PATH)

    def __post_init__(self) -> None:
        self._planes: list[np.ndarray] = []
        self._policies: list[np.ndarray] = []
        self._values: list[float] = []
        self.load()

    def __len__(self) -> int:
        return len(self._values)

    def add_steps(self, steps: list[dict[str, Any]]) -> int:
        """Append game steps and trim to max_samples. Returns new buffer size."""
        with _lock:
            for step in steps:
                if "planes" not in step or "policy" not in step or "z" not in step:
                    continue
                self._planes.append(np.asarray(step["planes"], dtype=np.float32))
                self._policies.append(np.asarray(step["policy"], dtype=np.float32))
                self._values.append(float(step["z"]))
            overflow = len(self._values) - self.max_samples
            if overflow > 0:
                self._planes = self._planes[overflow:]
                self._policies = self._policies[overflow:]
                self._values = self._values[overflow:]
            self._save_unlocked()
            return len(self._values)

    def prepare_training_batch(
        self,
        new_steps: list[dict[str, Any]],
        *,
        max_total: int = 2048,
    ) -> list[dict[str, Any]]:
        """Always keep the latest game; fill remaining slots from older replay."""
        clean_new = [
            {
                "planes": np.asarray(s["planes"], dtype=np.float32),
                "policy": np.asarray(s["policy"], dtype=np.float32),
                "z": float(s["z"]),
            }
            for s in new_steps
            if "planes" in s and "policy" in s and "z" in s
        ]
        added = self.add_steps(clean_new)
        del added

        if not clean_new:
            return self.sample(min(max_total, len(self)))

        if len(clean_new) >= max_total:
            return clean_new[-max_total:]

        need = max_total - len(clean_new)
        with _lock:
            older_n = max(0, len(self._values) - len(clean_new))
            if older_n <= 0:
                return clean_new
            take = min(need, older_n)
            idx = np.random.choice(older_n, size=take, replace=False)
            older = [
                {
                    "planes": self._planes[i],
                    "policy": self._policies[i],
                    "z": self._values[i],
                }
                for i in idx
            ]
        return clean_new + older

    def sample(self, n: int) -> list[dict[str, Any]]:
        with _lock:
            total = len(self._values)
            if total == 0 or n <= 0:
                return []
            take = min(n, total)
            idx = np.random.choice(total, size=take, replace=False)
            return [
                {
                    "planes": self._planes[i],
                    "policy": self._policies[i],
                    "z": self._values[i],
                }
                for i in idx
            ]

    def load(self) -> None:
        with _lock:
            if not self.path.exists():
                return
            try:
                data = np.load(self.path, allow_pickle=False)
                planes = data["planes"]
                policies = data["policies"]
                values = data["values"]
                self._planes = [planes[i] for i in range(len(values))]
                self._policies = [policies[i] for i in range(len(values))]
                self._values = [float(values[i]) for i in range(len(values))]
                overflow = len(self._values) - self.max_samples
                if overflow > 0:
                    self._planes = self._planes[overflow:]
                    self._policies = self._policies[overflow:]
                    self._values = self._values[overflow:]
            except (OSError, ValueError, KeyError) as exc:
                print(f"Could not load replay buffer: {exc}")
                self._planes, self._policies, self._values = [], [], []

    def _save_unlocked(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not self._values:
            return
        try:
            np.savez_compressed(
                self.path,
                planes=np.stack(self._planes).astype(np.float32),
                policies=np.stack(self._policies).astype(np.float32),
                values=np.asarray(self._values, dtype=np.float32),
            )
        except OSError as exc:
            print(f"Could not save replay buffer: {exc}")


# Shared process-wide buffer (Arena + Human both contribute)
_shared: ReplayBuffer | None = None


def get_replay_buffer() -> ReplayBuffer:
    global _shared
    if _shared is None:
        _shared = ReplayBuffer()
    return _shared
