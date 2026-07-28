"""Performance tests: Memory usage benchmarks (CPU only).

Measures actual memory footprint of BatchState and PPOBuffer at various sizes.
"""

import pytest
import torch
import sys

from conftest import check_perf, cpu_device
from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel

REGRESSION_THRESHOLD = 0.30


def _estimate_tensor_bytes(t: torch.Tensor) -> int:
    """Estimate memory usage of a tensor in bytes."""
    return t.numel() * t.element_size()


def _estimate_state_bytes(state) -> int:
    """Estimate memory usage of a state object by walking its tensors."""
    total = 0
    visited = set()

    def _walk(obj):
        nonlocal total
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, torch.Tensor):
            total += _estimate_tensor_bytes(obj)
        elif hasattr(obj, '__dict__'):
            for v in obj.__dict__.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)

    _walk(state)
    return total


# ═══════════════════════════════════════════════════════════════════════
# PB19: BatchState memory
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [16, 64, 256])
def test_perf_batch_state_memory(B: int, cpu_device):
    """PB19: BatchState memory usage at B={B}."""
    if B > 256:
        pytest.skip("Skipping large B on CPU")

    env = RedMahjongParallel(round_mode="single")
    bs = env.init_batch(num_envs=B, device=cpu_device)

    total_bytes = _estimate_state_bytes(bs)
    per_env_bytes = total_bytes / B
    total_mb = total_bytes / (1024 * 1024)

    # Store as baseline
    failure = check_perf(f"batch_state_memory/B{B}_total_mb", total_mb,
                         REGRESSION_THRESHOLD)
    if failure:
        pytest.fail(failure)


# ═══════════════════════════════════════════════════════════════════════
# PB20: PPOBuffer memory
# ═══════════════════════════════════════════════════════════════════════

def test_perf_ppo_buffer_memory(cpu_device):
    """PB20: PPOBuffer memory at T=256,B=128 with full observation dict."""
    from mahjax_pt.examples.ppo_with_reg import PPOBuffer

    T, B = 256, 128
    num_actions, num_players = 87, 4

    buffer = PPOBuffer(T, B, num_actions, num_players, device=cpu_device)

    # Simulate one store to allocate obs buffers
    env = RedMahjongParallel(round_mode="single")
    bs = env.init_batch(num_envs=B, device=cpu_device)
    obs = env.observe_batch(bs)

    buffer.store(0, obs,
                 torch.zeros(B, dtype=torch.long),
                 torch.zeros(B),
                 torch.zeros(B),
                 torch.zeros(B, num_players),
                 torch.zeros(B, dtype=torch.bool),
                 torch.zeros(B, dtype=torch.long),
                 torch.zeros(B, num_actions))

    # Measure total memory in buffer
    total_bytes = 0
    # Fixed fields
    for name in ['actions', 'log_probs', 'values', 'rewards', 'dones',
                 'current_players', 'masks']:
        t = getattr(buffer, name)
        total_bytes += _estimate_tensor_bytes(t)

    # Observation fields
    for k, v in buffer._obs.items():
        total_bytes += _estimate_tensor_bytes(v)

    total_mb = total_bytes / (1024 * 1024)
    per_step_mb = total_mb / T

    failure = check_perf("ppo_buffer_T256_B128_total_mb", total_mb,
                         REGRESSION_THRESHOLD)
    if failure:
        pytest.fail(failure)

    # Also check per-step overhead is reasonable
    failure = check_perf("ppo_buffer_per_step_mb", per_step_mb,
                         REGRESSION_THRESHOLD)
    if failure:
        pytest.fail(failure)
