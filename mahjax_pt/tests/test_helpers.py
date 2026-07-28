"""Shared helpers for mahjax_pt tests.

Utilities for:
  - Constructing synthetic inputs (batch data, dummy networks)
  - Timing function execution
  - Asserting tensor closeness
"""

import time
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Callable


# ═══════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════

def timeit(fn: Callable[[], None], warmup: int = 3, repeat: int = 20) -> Dict[str, float]:
    """Measure execution time of a callable.

    Args:
        fn: callable to measure (called with no arguments)
        warmup: number of warmup iterations (not measured)
        repeat: number of measured iterations

    Returns:
        {"min_ms": ..., "mean_ms": ..., "max_ms": ...}
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)  # ms

    return {
        "min_ms": float(np.min(times)),
        "mean_ms": float(np.mean(times)),
        "max_ms": float(np.max(times)),
        "n": repeat,
    }


def timeit_with_arg(fn: Callable, *args, warmup: int = 3, repeat: int = 20, **kwargs) -> Dict[str, float]:
    """Measure execution time of a callable with arguments.

    Returns: {"min_ms", "mean_ms", "max_ms", "n"}
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "min_ms": float(np.min(times)),
        "mean_ms": float(np.mean(times)),
        "max_ms": float(np.max(times)),
        "n": repeat,
    }


# ═══════════════════════════════════════════════════════════════════════
# Synthetic data generators
# ═══════════════════════════════════════════════════════════════════════

def make_random_hand_34(seed: int = 0) -> torch.Tensor:
    """Generate a random valid 34-dim hand tensor (14 tiles)."""
    rng = np.random.RandomState(seed)
    counts = np.zeros(34, dtype=np.int8)
    # Place 14 tiles randomly, avoiding >4 per type
    placed = 0
    while placed < 14:
        t = rng.randint(0, 34)
        if counts[t] < 4:
            counts[t] += 1
            placed += 1
    return torch.from_numpy(counts)


def make_random_hand_37(seed: int = 0) -> torch.Tensor:
    """Generate a random 37-dim hand tensor (14 tiles, red-fives possible)."""
    rng = np.random.RandomState(seed)
    counts = np.zeros(37, dtype=np.int8)
    placed = 0
    while placed < 14:
        t = rng.randint(0, 37)
        # For five types (4, 13, 22): max 3 normal + 1 red
        if t in (4, 13, 22):
            max_copies = 3 if t in (4, 13, 22) else 4
            if t == 34:
                max_copies = 1  # red 5m
            elif t == 35:
                max_copies = 1  # red 5p
            elif t == 36:
                max_copies = 1  # red 5s
            if counts[t] < max_copies:
                counts[t] += 1
                placed += 1
        elif t < 34:
            if counts[t] < 4:
                counts[t] += 1
                placed += 1
    return torch.from_numpy(counts)


def make_random_batch_hands_37(batch_size: int, seed: int = 0) -> torch.Tensor:
    """Generate B random 37-dim hand tensors."""
    rng = np.random.RandomState(seed)
    hands = np.zeros((batch_size, 37), dtype=np.int8)
    for b in range(batch_size):
        placed = 0
        while placed < 14:
            t = rng.randint(0, 37)
            if t in (34, 35, 36):
                if hands[b, t] < 1:
                    hands[b, t] += 1
                    placed += 1
            elif t < 34:
                if hands[b, t] < 4:
                    hands[b, t] += 1
                    placed += 1
    return torch.from_numpy(hands)


def make_gae_inputs(T: int, B: int, P: int = 4, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Generate synthetic GAE inputs with episode boundaries.

    Returns dict with: rewards, values, dones, current_players
    """
    rng = np.random.RandomState(seed)
    rewards = rng.randn(T, B, P).astype(np.float32) * 0.1
    values = rng.randn(T, B).astype(np.float32) * 0.5
    dones = np.zeros((T, B), dtype=bool)
    # Add sparse episode boundaries
    for step in range(40, T, 40):
        dones[step, :] = True
    current_players = np.zeros((T, B), dtype=np.int32)
    for b in range(B):
        current_players[:, b] = (np.arange(T) + rng.randint(0, P)) % P

    return {
        "rewards": torch.from_numpy(rewards),
        "values": torch.from_numpy(values),
        "dones": torch.from_numpy(dones),
        "current_players": torch.from_numpy(current_players),
    }


def make_ppo_batch(batch_size: int = 64, num_actions: int = 87,
                   feature_dim: int = 32, seed: int = 0) -> Dict[str, torch.Tensor]:
    """Generate synthetic PPO batch data (obs, actions, advantages etc.).

    Returns dict that can be passed directly to ppo_update-style functions.
    """
    rng = np.random.RandomState(seed)
    obs = torch.randn(batch_size, feature_dim)
    actions = torch.from_numpy(rng.randint(0, num_actions, size=batch_size).astype(np.int64))
    old_log_probs = torch.from_numpy(rng.randn(batch_size).astype(np.float32) * 0.2)
    old_values = torch.from_numpy(rng.randn(batch_size).astype(np.float32) * 0.5)
    current_players = torch.from_numpy(rng.randint(0, 4, size=batch_size).astype(np.int64))

    advantages = torch.zeros(batch_size, 4)
    targets = torch.zeros(batch_size, 4)
    valid_mask = torch.zeros(batch_size, 4, dtype=torch.bool)
    for b in range(batch_size):
        p = int(current_players[b])
        advantages[b, p] = float(rng.randn() * 0.5)
        targets[b, p] = float(old_values[b] + advantages[b, p])
        valid_mask[b, p] = True

    action_mask = torch.from_numpy(
        rng.rand(batch_size, num_actions).astype(np.float32) > 0.7)

    return {
        "obs": obs,
        "actions": actions,
        "old_log_probs": old_log_probs,
        "old_values": old_values,
        "current_players": current_players,
        "advantages": advantages,
        "targets": targets,
        "valid_mask": valid_mask,
        "action_mask": action_mask,
    }


# ═══════════════════════════════════════════════════════════════════════
# Tiny MLP for PPO precision/performance tests
# ═══════════════════════════════════════════════════════════════════════

class TinyACNet(torch.nn.Module):
    """Minimal Actor-Critic network for fast testing.

    Architecture: Linear→ReLU→Linear→ReLU→{actor, critic}
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64,
                 num_actions: int = 87):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.actor = torch.nn.Linear(hidden_dim, num_actions)
        self.critic = torch.nn.Linear(hidden_dim, 1)

        # Orthogonal init matching JAX conventions
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.critic.weight)
        for m in [self.fc1, self.fc2, self.actor, self.critic]:
            torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.actor(h), self.critic(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════
# Assertion helpers
# ═══════════════════════════════════════════════════════════════════════

def assert_tensors_close(actual: torch.Tensor, expected: torch.Tensor,
                         atol: float = 1e-5, rtol: float = 1e-5,
                         name: str = "tensor"):
    """Assert two tensors are element-wise close."""
    actual_np = actual.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)

    diff = np.abs(actual_np.astype(np.float64) - expected_np.astype(np.float64))
    max_diff = float(diff.max())

    if max_diff > atol:
        idx = np.unravel_index(np.argmax(diff), actual_np.shape)
        raise AssertionError(
            f"{name}: max_diff={max_diff:.2e} > atol={atol}. "
            f"At {idx}: actual={actual_np[idx]}, expected={expected_np[idx]}")


def assert_tensors_equal(actual: torch.Tensor, expected, name: str = "tensor"):
    """Assert two tensors are element-wise exactly equal."""
    actual_np = actual.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)

    if not np.array_equal(actual_np, expected_np):
        diff_mask = actual_np != expected_np
        n_diff = int(diff_mask.sum())
        idx = np.unravel_index(np.argmax(diff_mask.astype(int)), actual_np.shape)
        raise AssertionError(
            f"{name}: {n_diff} values differ. "
            f"First at {idx}: actual={actual_np[idx]}, expected={expected_np[idx]}")
