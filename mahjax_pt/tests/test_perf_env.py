"""Performance tests: Env throughput benchmarks (CPU only).

Measures serial and parallel env step/init throughput at various batch sizes.
"""

import pytest
import torch

from conftest import check_perf, cpu_device
from test_helpers import timeit_with_arg
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial
from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel

REGRESSION_THRESHOLD = 0.30


def _check_and_report(name: str, result: dict):
    mean_ms = result["mean_ms"]
    failure = check_perf(name, mean_ms, REGRESSION_THRESHOLD)
    if failure:
        pytest.fail(failure)


# ═══════════════════════════════════════════════════════════════════════
# PB11: Serial env throughput
# ═══════════════════════════════════════════════════════════════════════

def test_perf_serial_throughput(cpu_device):
    """PB11: RedMahjongSerial steps/s with random policy, 1000 steps."""
    env = RedMahjongSerial(round_mode="single")
    state = env.init(key=0)
    mask = state.legal_action_mask

    # Pre-compute actions (discard-only for consistency)
    actions = []
    s = state
    for _ in range(1000):
        legal = s.legal_action_mask.nonzero(as_tuple=False).flatten()
        if len(legal) > 0:
            actions.append(int(legal[0].item()))
            s = env.step(s, int(legal[0].item()))
        else:
            break
        if s.terminated:
            s = env.init(key=len(actions))
    n_actions = len(actions)

    def bench():
        s = env.init(key=0)
        for a in actions[:100]:
            s = env.step(s, a)
            if s.terminated:
                break

    result = timeit_with_arg(bench, warmup=2, repeat=10)
    steps_per_second = 100 / (result["mean_ms"] / 1000)
    result["steps_per_second"] = steps_per_second
    _check_and_report("env_serial_throughput", result)


# ═══════════════════════════════════════════════════════════════════════
# PB12: Parallel env throughput scaling
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [1, 4, 16, 64])
def test_perf_parallel_throughput(B: int, cpu_device):
    """PB12: RedMahjongParallel env-steps/s at various batch sizes."""
    if B > 64:
        pytest.skip("Skipping large batch on CPU")

    env = RedMahjongParallel(round_mode="single")
    bs = env.init_batch(num_envs=B, device=cpu_device)

    # Deterministic first legal action for each env
    actions_per_env = []
    for i in range(B):
        from mahjax_pt.red_mahjong.batch_state import unstack_state
        s = unstack_state(bs, i)
        legal = s.legal_action_mask.nonzero(as_tuple=False).flatten()
        actions_per_env.append(int(legal[0].item()) if len(legal) > 0 else 0)
    actions_tensor = torch.tensor(actions_per_env, dtype=torch.int32, device=cpu_device)

    def bench():
        env.step_batch(bs, actions_tensor)

    result = timeit_with_arg(bench, warmup=3, repeat=15)
    env_steps_per_second = B / (result["mean_ms"] / 1000)
    result["env_steps_per_second"] = env_steps_per_second
    _check_and_report(f"env_parallel_throughput/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB13: init_batch latency
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [16, 64, 256])
def test_perf_init_batch(B: int, cpu_device):
    """PB13: init_batch latency at various batch sizes."""
    if B > 256:
        pytest.skip("Skipping large batch on CPU")

    env = RedMahjongParallel(round_mode="single")

    def bench():
        env.init_batch(num_envs=B, device=cpu_device)

    result = timeit_with_arg(bench, warmup=1, repeat=5)
    _check_and_report(f"env_init_batch/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB14: step_batch latency (discard-only)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [16, 64, 256])
def test_perf_step_batch(B: int, cpu_device):
    """PB14: step_batch latency at various batch sizes (discard-only)."""
    if B > 256:
        pytest.skip("Skipping large batch on CPU")

    env = RedMahjongParallel(round_mode="single")
    bs = env.init_batch(num_envs=B, device=cpu_device)

    # Use first legal discard for each env
    actions = []
    for i in range(B):
        from mahjax_pt.red_mahjong.batch_state import unstack_state
        s = unstack_state(bs, i)
        discards = [a for a in range(37) if s.legal_action_mask[a].item()]
        actions.append(discards[0] if discards else 0)
    actions_tensor = torch.tensor(actions, dtype=torch.int32, device=cpu_device)

    def bench():
        env.step_batch(bs, actions_tensor)

    result = timeit_with_arg(bench, warmup=3, repeat=20)
    _check_and_report(f"env_step_batch/B{B}", result)
