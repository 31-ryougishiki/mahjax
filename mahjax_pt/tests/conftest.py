"""Shared fixtures and utilities for mahjax_pt tests.

Golden data replay (precision tests):
    Golden .npz files are pre-recorded from the JAX reference env.
    Tests load them and replay against the PT implementation.
    If golden data is missing, tests skip gracefully.

Performance tests:
    Baseline values are stored in baselines/cpu_perf.json.
    First run records baselines; subsequent runs compare against them.
"""

import os
import json
import time
import pytest
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable

# ── Paths ──────────────────────────────────────────────────────────────
_TEST_DIR = Path(__file__).resolve().parent
_GOLDEN_DIR = _TEST_DIR / "golden"
_BASELINE_DIR = _TEST_DIR / "baselines"
_BASELINE_FILE = _BASELINE_DIR / "cpu_perf.json"


# ═══════════════════════════════════════════════════════════════════════
# Golden data helpers
# ═══════════════════════════════════════════════════════════════════════

def has_golden(pattern: str = "env_playout_s0.npz") -> bool:
    """Check if any golden data exists matching the pattern."""
    return any(_GOLDEN_DIR.glob(pattern))


def load_golden(name: str) -> Dict[str, Any]:
    """Load a golden .npz file and return as a dict of numpy arrays.

    Args:
        name: filename without extension, e.g. "env_init_s0"

    Returns:
        Dict of {key: np.ndarray} loaded from .npz
    """
    path = _GOLDEN_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Golden file not found: {path}")
    return dict(np.load(path, allow_pickle=False))


def load_golden_playout(seed: int) -> List[Dict[str, Any]]:
    """Load a full playout golden file for a given seed.

    Returns a list of per-step state dicts.
    The file format is:
        steps: int
        0/field1, 0/field2, ...  (init state)
        1/field1, 1/field2, ...  (after step 1)
        ...
    """
    data = load_golden(f"env_playout_s{seed}")
    num_steps = int(data["steps"])
    steps = []
    for i in range(num_steps):
        prefix = f"{i}/"
        step = {}
        for key in data:
            if key.startswith(prefix):
                field = key[len(prefix):]
                step[field] = data[key]
        steps.append(step)
    return steps


# ═══════════════════════════════════════════════════════════════════════
# Field comparison map for EnvState
# ═══════════════════════════════════════════════════════════════════════

# Each entry: (golden_key, pt_accessor_fn, tolerance)
# tolerance: 'exact' for integer/bool tensors, float for max absolute diff
StateCheck = Tuple[str, Callable[[Any], Any], str]

def _make_checks() -> List[StateCheck]:
    """Build the list of field comparisons between golden dict and PT EnvState."""
    return [
        # ── Top-level env fields ──
        ("current_player",
         lambda s: s.current_player, "exact"),
        ("terminated",
         lambda s: s.terminated, "exact"),
        ("legal_action_mask",
         lambda s: s.legal_action_mask, "exact"),
        ("rewards",
         lambda s: s.rewards, "close"),  # float32 settlement
        ("step_count",
         lambda s: s.step_count, "exact"),

        # ── PlayerStateArrays ──
        ("players.hand",
         lambda s: s.players.hand, "exact"),
        ("players.hand_with_red",
         lambda s: s.players.hand_with_red, "exact"),
        ("players.melds",
         lambda s: s.players.melds, "exact"),
        ("players.meld_counts",
         lambda s: s.players.meld_counts, "exact"),
        ("players.discard_counts",
         lambda s: s.players.discard_counts, "exact"),
        ("players.river",
         lambda s: s.players.river, "exact"),
        ("players.riichi",
         lambda s: s.players.riichi, "exact"),
        ("players.riichi_declared",
         lambda s: s.players.riichi_declared, "exact"),
        ("players.has_won",
         lambda s: s.players.has_won, "exact"),
        ("players.n_kan",
         lambda s: s.players.n_kan, "exact"),
        ("players.has_yaku",
         lambda s: s.players.has_yaku, "exact"),
        ("players.is_hand_concealed",
         lambda s: s.players.is_hand_concealed, "exact"),
        ("players.furiten_by_discard",
         lambda s: s.players.furiten_by_discard, "exact"),
        ("players.furiten_by_pass",
         lambda s: s.players.furiten_by_pass, "exact"),
        ("players.ippatsu",
         lambda s: s.players.ippatsu, "exact"),
        ("players.fan",
         lambda s: s.players.fan, "exact"),
        ("players.fu",
         lambda s: s.players.fu, "exact"),

        # ── RoundState ──
        ("round_state.round",
         lambda s: s.round_state.round, "exact"),
        ("round_state.dealer",
         lambda s: s.round_state.dealer, "exact"),
        ("round_state.next_deck_ix",
         lambda s: s.round_state.next_deck_ix, "exact"),
        ("round_state.last_deck_ix",
         lambda s: s.round_state.last_deck_ix, "exact"),
        ("round_state.last_draw",
         lambda s: s.round_state.last_draw, "exact"),
        ("round_state.last_player",
         lambda s: s.round_state.last_player, "exact"),
        ("round_state.target",
         lambda s: s.round_state.target, "exact"),
        ("round_state.draw_next",
         lambda s: s.round_state.draw_next, "exact"),
        ("round_state.is_haitei",
         lambda s: s.round_state.is_haitei, "exact"),
        ("round_state.is_abortive_draw_normal",
         lambda s: s.round_state.is_abortive_draw_normal, "exact"),
        ("round_state.terminated_round",
         lambda s: s.round_state.terminated_round, "exact"),
        ("round_state.score",
         lambda s: s.round_state.score, "exact"),
        ("round_state.deck",
         lambda s: s.round_state.deck, "exact"),
        ("round_state.dora_indicators",
         lambda s: s.round_state.dora_indicators, "exact"),
        ("round_state.n_kan_doras",
         lambda s: s.round_state.n_kan_doras, "exact"),
        ("round_state.honba",
         lambda s: s.round_state.honba, "exact"),
        ("round_state.kyotaku",
         lambda s: s.round_state.kyotaku, "exact"),
    ]


CHECKS = _make_checks()


def _pt_val(v):
    """Extract value from a PT tensor or scalar for comparison."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return np.asarray(v)


def assert_state_equal(golden: Dict[str, Any], pt_state,
                       checks: List[StateCheck] = None,
                       atol: float = 1e-5) -> List[str]:
    """Compare a PT EnvState against golden dict. Returns list of failure messages.

    Args:
        golden: dict loaded from .npz, keys match CHECKS
        pt_state: PT EnvState instance
        checks: list of (key, accessor, tolerance) — defaults to CHECKS
        atol: absolute tolerance for 'close' checks

    Returns:
        List of failure descriptions (empty = all pass)
    """
    if checks is None:
        checks = CHECKS

    failures = []
    for key, accessor, tol in checks:
        if key not in golden:
            continue

        expected = golden[key]
        actual = _pt_val(accessor(pt_state))

        if tol == "exact":
            if expected.shape != actual.shape:
                failures.append(f"{key}: shape mismatch {expected.shape} vs {actual.shape}")
            elif not np.array_equal(expected, actual):
                # Find first mismatch
                diff_mask = expected != actual
                n_diff = int(diff_mask.sum())
                idx = np.unravel_index(np.argmax(diff_mask), expected.shape)
                failures.append(
                    f"{key}: {n_diff} values differ. "
                    f"First at {idx}: golden={expected[idx]}, pt={actual[idx]}")
        elif tol == "close":
            diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            max_diff = float(diff.max())
            if max_diff > atol:
                idx = np.unravel_index(np.argmax(diff), expected.shape)
                failures.append(
                    f"{key}: max_diff={max_diff:.2e} > {atol}. "
                    f"At {idx}: golden={expected[idx]}, pt={actual[idx]}")
    return failures


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def golden_dir():
    """Path to golden data directory."""
    if not _GOLDEN_DIR.exists():
        _GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return _GOLDEN_DIR


@pytest.fixture(scope="session")
def baseline_dir():
    """Path to performance baseline directory."""
    if not _BASELINE_DIR.exists():
        _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    return _BASELINE_DIR


@pytest.fixture(scope="session")
def cpu_device():
    """CPU device for performance tests."""
    return torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════
# Performance baseline helpers
# ═══════════════════════════════════════════════════════════════════════

def load_baselines() -> Dict[str, float]:
    """Load performance baselines from JSON file."""
    if _BASELINE_FILE.exists():
        with open(_BASELINE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_baselines(data: Dict[str, float]):
    """Save performance baselines to JSON file."""
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def check_perf(name: str, value_ms: float, threshold: float = 0.30) -> Optional[str]:
    """Check if a performance measurement has regressed vs baseline.

    Args:
        name: test name (used as key in baseline file)
        value_ms: current measured value in milliseconds
        threshold: allowed regression fraction (0.30 = 30%)

    Returns:
        None if ok, or a failure message string
    """
    baselines = load_baselines()

    if name not in baselines:
        # First run: record baseline
        baselines[name] = value_ms
        save_baselines(baselines)
        return None  # no regression check on first run

    baseline = baselines[name]
    if baseline <= 0:
        return None

    # Update baseline (keep it fresh)
    baselines[name] = value_ms
    save_baselines(baselines)

    ratio = value_ms / baseline
    if ratio > 1.0 + threshold:
        return (f"REGRESSION: {name}: {value_ms:.1f}ms > {baseline:.1f}ms "
                f"(+{(ratio - 1) * 100:.0f}%)")

    return None
