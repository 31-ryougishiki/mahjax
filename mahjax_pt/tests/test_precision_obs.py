"""Precision tests: Observation parity via Golden data replay.

Compares PT observation output against JAX golden observation data.

Prerequisites:
    Golden .npz files in tests/golden/ recorded by scripts/record_golden_obs.py
"""

import pytest
import torch
import numpy as np

from conftest import has_golden, load_golden
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial
from test_precision_env import _overlay_state


def _get_pt_obs(state, observe_type: str = "dict"):
    """Run PT observe on a state and return dict of numpy arrays."""
    env = RedMahjongSerial(round_mode="single", observe_type=observe_type)
    obs = env.observe(state)
    result = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, np.ndarray):
            result[k] = v
        else:
            result[k] = np.asarray(v)
    return result


@pytest.mark.parametrize("seed", list(range(50)))
def test_observe_dict_precision(seed: int):
    """PA11: PT observe_dict matches JAX golden for 50 seeds."""
    if not has_golden(f"obs_s{seed}.npz"):
        pytest.skip(f"Golden file obs_s{seed}.npz not found")

    golden = load_golden(f"obs_s{seed}")
    env = RedMahjongSerial(round_mode="single", observe_type="dict")
    state = env.init(key=seed)

    # Overlay state from golden (which may have multiple steps of history)
    _overlay_state(state, golden)

    pt_obs = _get_pt_obs(state)
    failures = []

    for key in golden:
        if key.startswith("players.") or key.startswith("round_state.") or key in (
            "current_player", "terminated", "legal_action_mask", "rewards", "step_count"
        ):
            continue  # These are state fields, not obs fields

        if key not in pt_obs:
            failures.append(f"Key '{key}' missing from PT observe output")
            continue

        expected = golden[key]
        actual = pt_obs[key]

        if expected.shape != actual.shape:
            failures.append(f"{key}: shape mismatch {expected.shape} vs {actual.shape}")
        elif expected.dtype == bool or expected.dtype.kind in ('i', 'u'):
            if not np.array_equal(expected, actual):
                n_diff = int((expected != actual).sum())
                failures.append(f"{key}: {n_diff} int values differ")
        else:
            diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
            max_diff = float(diff.max())
            if max_diff > 1e-5:
                failures.append(f"{key}: float max_diff={max_diff:.2e}")

    if failures:
        pytest.fail(f"Seed {seed} observation mismatch:\n" + "\n".join(failures))
