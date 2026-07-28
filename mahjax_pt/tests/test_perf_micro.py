"""Performance tests: Single-function latency micro-benchmarks (CPU only).

Each test measures a specific function's execution time on synthetic inputs.
Baselines are recorded on first run; subsequent runs detect regressions > 30%.
"""

import pytest
import torch

from conftest import check_perf, cpu_device
from test_helpers import (
    timeit_with_arg,
    make_random_hand_34,
    make_random_hand_37,
    make_random_batch_hands_37,
)
from mahjax_pt.red_mahjong.hand import Hand
from mahjax_pt.red_mahjong.shanten import Shanten
from mahjax_pt.red_mahjong.tile import Tile

REGRESSION_THRESHOLD = 0.30


def _check_and_report(name: str, result: dict):
    """Check performance regression and report."""
    mean_ms = result["mean_ms"]
    failure = check_perf(name, mean_ms, REGRESSION_THRESHOLD)
    if failure:
        pytest.fail(failure)


# ═══════════════════════════════════════════════════════════════════════
# PB01: Hand.can_tsumo (single) × 1000
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", range(10))
def test_perf_can_tsumo_single(seed: int, cpu_device):
    """PB01: Single-hand can_tsumo latency (avg over 100 calls in one timing block)."""
    hands = [make_random_hand_37(seed + i) for i in range(100)]

    def bench():
        for h in hands:
            Hand.can_tsumo(h)

    result = timeit_with_arg(bench)
    _check_and_report(f"can_tsumo_single/s{seed}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB02: Hand.can_tsumo_batch scaling
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [1, 16, 64, 256, 1024])
def test_perf_can_tsumo_batch(B: int, cpu_device):
    """PB02: Batch can_tsumo throughput at various batch sizes."""
    if B > 256:
        pytest.skip("Skipping large batch on CPU" if B > 256 else "")

    hands = make_random_batch_hands_37(B)

    def bench():
        Hand.can_tsumo_batch(hands)

    result = timeit_with_arg(bench, warmup=3, repeat=10)
    _check_and_report(f"can_tsumo_batch/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB03: Hand.can_riichi_batch
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [16, 64, 256])
def test_perf_can_riichi_batch(B: int, cpu_device):
    """PB03: Batch can_riichi throughput."""
    hands = make_random_batch_hands_37(B)

    def bench():
        Hand.can_riichi_batch(hands)

    result = timeit_with_arg(bench, warmup=2, repeat=8)
    _check_and_report(f"can_riichi_batch/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB04: Shanten.number (single) × 500
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", range(5))
def test_perf_shanten_single(seed: int, cpu_device):
    """PB04: Single-hand shanten number latency."""
    hands = [make_random_hand_34(seed * 10 + i) for i in range(500)]

    def bench():
        for h in hands:
            Shanten.number(h)

    result = timeit_with_arg(bench)
    _check_and_report(f"shanten_single/s{seed}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB05: Shanten.number_batch scaling
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [64, 256, 1024])
def test_perf_shanten_batch(B: int, cpu_device):
    """PB05: Batch shanten throughput at various batch sizes."""
    if B > 256:
        pytest.skip("Skipping large batch on CPU")

    # Convert 37-dim hands to 34-dim for shanten
    hands_37 = make_random_batch_hands_37(B)
    hands_34 = Hand.to_34_batch(hands_37)

    def bench():
        Shanten.number_batch(hands_34)

    result = timeit_with_arg(bench, warmup=2, repeat=8)
    _check_and_report(f"shanten_batch/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB06: Yaku.judge (single)
# ═══════════════════════════════════════════════════════════════════════

def test_perf_yaku_judge(cpu_device):
    """PB06: Single-hand yaku judge latency (avg over 100 hands)."""
    from mahjax_pt.red_mahjong.state import default_state
    from mahjax_pt.red_mahjong.yaku import Yaku

    hands = [make_random_hand_34(i) for i in range(100)]
    base_state = default_state()

    def bench():
        for h in hands:
            Yaku.judge(h, torch.tensor(False), torch.tensor(0, dtype=torch.int8), base_state)

    result = timeit_with_arg(bench, warmup=1, repeat=5)
    _check_and_report("yaku_judge_single", result)


# ═══════════════════════════════════════════════════════════════════════
# PB07: _observe_dict (single) × 500
# ═══════════════════════════════════════════════════════════════════════

def test_perf_observe_single(cpu_device):
    """PB07: Single-state observe_dict latency."""
    from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial
    env = RedMahjongSerial(round_mode="single")
    states = [env.init(key=i) for i in range(500)]

    def bench():
        for s in states:
            env.observe(s)

    result = timeit_with_arg(bench, warmup=1, repeat=5)
    _check_and_report("observe_dict_single", result)


# ═══════════════════════════════════════════════════════════════════════
# PB08: _observe_dict_batch scaling
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [16, 64, 256])
def test_perf_observe_batch(B: int, cpu_device):
    """PB08: Batch observe throughput at various batch sizes."""
    from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel
    env = RedMahjongParallel(round_mode="single")
    bs = env.init_batch(num_envs=B, device=cpu_device)

    def bench():
        env.observe_batch(bs)

    result = timeit_with_arg(bench, warmup=2, repeat=8)
    _check_and_report(f"observe_dict_batch/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB09: Hand.chi_batch
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("B", [64, 256])
def test_perf_chi_batch(B: int, cpu_device):
    """PB09: Batch chi hand mutation throughput."""
    from mahjax_pt.red_mahjong.action import Action

    hands = make_random_batch_hands_37(B)
    targets = torch.randint(0, 27, (B,), dtype=torch.int32)  # only suit tiles
    actions = torch.full((B,), Action.CHI_L, dtype=torch.int32)

    def bench():
        Hand.chi_batch(hands, targets, actions)

    result = timeit_with_arg(bench, warmup=2, repeat=8)
    _check_and_report(f"chi_batch/B{B}", result)


# ═══════════════════════════════════════════════════════════════════════
# PB10: Hand.can_tsumo_batch (large B=4096)
# ═══════════════════════════════════════════════════════════════════════

def test_perf_can_tsumo_large(cpu_device):
    """PB10: Large batch can_tsumo stress test."""
    B = 4096
    hands = make_random_batch_hands_37(B)

    def bench():
        Hand.can_tsumo_batch(hands)

    result = timeit_with_arg(bench, warmup=2, repeat=5)
    _check_and_report("can_tsumo_batch/B4096", result)
