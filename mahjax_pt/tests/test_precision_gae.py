"""Precision tests: GAE computation via Golden data replay.

Compares PT compute_gae_vectorized output against JAX golden GAE output.
Same numpy inputs → both frameworks compute → compare advantages/targets/valid_mask.

Prerequisites:
    Golden .npz files in tests/golden/ recorded by scripts/record_golden_gae.py
"""

import pytest
import torch
import numpy as np

from conftest import has_golden, load_golden


def pt_compute_gae_vectorized(rewards, values, dones, current_players,
                               gamma=1.0, gae_lambda=0.95):
    """Mirror of ppo_with_reg.py:compute_gae_vectorized."""
    T, B, P = rewards.shape
    device = rewards.device

    advantages = torch.zeros(T, B, P, device=device)
    targets = torch.zeros(T, B, P, device=device)
    valid_mask = torch.zeros(T, B, P, dtype=torch.bool, device=device)

    gae_acc = torch.zeros(B, P, device=device)
    reward_accum = torch.zeros(B, P, device=device)
    next_value = torch.zeros(B, P, device=device)
    has_next_value = torch.zeros(B, P, dtype=torch.bool, device=device)
    next_valid = torch.zeros(B, P, dtype=torch.bool, device=device)

    b_idx = torch.arange(B, device=device)

    for t in reversed(range(T)):
        cp = current_players[t]
        done = dones[t]

        gae_acc[done] = 0.0
        reward_accum[done] = 0.0
        has_next_value[done] = False
        next_value[done] = 0.0

        reward_accum = reward_accum + rewards[t]
        player_reward = reward_accum[b_idx, cp].clone()
        reward_accum[b_idx, cp] = 0.0

        not_done = (~done).float()
        td_error = player_reward + gamma * next_value[b_idx, cp] * not_done - values[t]
        new_gae = td_error + gamma * gae_lambda * gae_acc[b_idx, cp] * not_done
        gae_acc[b_idx, cp] = new_gae

        is_valid = has_next_value[b_idx, cp] | done | next_valid[b_idx, cp]

        advantages[t, b_idx, cp] = torch.where(
            is_valid, new_gae, torch.zeros_like(new_gae))
        targets[t, b_idx, cp] = torch.where(
            is_valid, new_gae + values[t], values[t])
        valid_mask[t, b_idx, cp] = is_valid

        next_value[b_idx, cp] = values[t]
        has_next_value[b_idx, cp] = True
        next_valid[done] = True
        next_valid[b_idx, cp] = is_valid | done

    return advantages, targets, valid_mask


def _run_gae_test(golden_name: str, tol: float = 1e-6):
    """Load golden GAE inputs/outputs, run PT GAE, compare."""
    golden = load_golden(golden_name)

    # Extract inputs
    rewards = torch.from_numpy(golden["rewards"])
    values = torch.from_numpy(golden["values"])
    dones = torch.from_numpy(golden["dones"])
    current_players = torch.from_numpy(golden["current_players"].astype(np.int64))

    # Run PT GAE
    pt_adv, pt_tgt, pt_vm = pt_compute_gae_vectorized(
        rewards, values, dones, current_players)

    # Compare
    failures = []

    for name, pt_val, golden_key in [
        ("advantages", pt_adv, "advantages"),
        ("targets", pt_tgt, "targets"),
        ("valid_mask", pt_vm, "valid_mask"),
    ]:
        expected = golden[golden_key]
        actual = pt_val.numpy()

        if expected.shape != actual.shape:
            failures.append(f"{name}: shape {expected.shape} vs {actual.shape}")
            continue

        diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        if max_diff > tol:
            idx = np.unravel_index(np.argmax(diff), expected.shape)
            failures.append(
                f"{name}: max_diff={max_diff:.2e} (mean={mean_diff:.2e}) > tol={tol}. "
                f"At {idx}: golden={expected[idx]:.8f}, pt={actual[idx]:.8f}")

    return failures


# ═══════════════════════════════════════════════════════════════════════
# PA12: Small-scale GAE
# ═══════════════════════════════════════════════════════════════════════

def test_gae_small():
    """PA12: GAE T=8,B=4,P=4 against golden."""
    if not has_golden("gae_T8_B4.npz"):
        pytest.skip("Golden file gae_T8_B4.npz not found")

    failures = _run_gae_test("gae_T8_B4", tol=1e-6)
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA13: Large-scale GAE
# ═══════════════════════════════════════════════════════════════════════

def test_gae_large():
    """PA13: GAE T=256,B=128,P=4 against golden."""
    if not has_golden("gae_T256_B128.npz"):
        pytest.skip("Golden file gae_T256_B128.npz not found")

    failures = _run_gae_test("gae_T256_B128", tol=1e-6)
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA14: GAE with episode boundaries
# ═══════════════════════════════════════════════════════════════════════

def test_gae_with_boundaries():
    """PA14: GAE T=256,B=64,P=4 with dones every ~40 steps against golden."""
    if not has_golden("gae_T256_B64_boundaries.npz"):
        pytest.skip("Golden file gae_T256_B64_boundaries.npz not found")

    failures = _run_gae_test("gae_T256_B64_boundaries", tol=1e-6)
    if failures:
        pytest.fail("\n".join(failures))
