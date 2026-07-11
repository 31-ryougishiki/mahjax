#!/usr/bin/env python3
"""L2: GAE Computation Parity — JAX calculate_gae vs PyTorch compute_gae_vectorized.

Verifies that the two GAE implementations produce identical (advantages, targets, valid_mask)
given the same (rewards, values, dones, current_players) inputs.

Uses real red_mahjong dimensions: T=256 steps, B=8 envs, P=4 players.
Tests: normal cases, episode boundaries, edge cases (all-zero, extreme values).
"""

import sys
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import torch

SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# JAX reference GAE (exact replica of examples/ppo_with_reg.py:calculate_gae)
# ═══════════════════════════════════════════════════════════════════════════

GAMMA = 1.0
GAE_LAMBDA = 0.95
NUM_PLAYERS = 4


def jax_single_env_gae(rewards, values, dones, current_players):
    """GAE for one environment. All inputs: (T, 4) or (T,) or (T,...)."""
    def scan_fn(carry, inputs):
        gae, next_value, reward_accum, has_next_value, prev_done, next_valid = carry
        player, reward, value, done = inputs

        # Reset accumulators on episode boundary
        gae = jnp.where(done, 0, gae)
        reward_accum = jnp.where(done, 0, reward_accum)
        has_next_value = jnp.where(done, False, has_next_value)
        next_value = jnp.where(done, 0, next_value)

        reward_accum = reward_accum + reward
        player_reward = reward_accum[player]
        reward_accum = reward_accum.at[player].set(0.0)

        td_error = player_reward + GAMMA * next_value[player] - value
        new_gae = td_error + GAMMA * GAE_LAMBDA * gae[player]
        gae = gae.at[player].set(new_gae)

        is_valid = has_next_value[player] | done | next_valid[player]
        advantage = jnp.where(is_valid, new_gae, 0.0)
        target = jnp.where(is_valid, advantage + value, value)

        new_carry = (
            gae, next_value.at[player].set(value), reward_accum,
            has_next_value.at[player].set(True), done,
            next_valid.at[player].set(is_valid) | done,
        )
        output = (
            jnp.zeros(NUM_PLAYERS).at[player].set(advantage),
            jnp.zeros(NUM_PLAYERS).at[player].set(target),
            jnp.zeros(NUM_PLAYERS, dtype=bool).at[player].set(is_valid),
        )
        return new_carry, output

    init = (
        jnp.zeros(NUM_PLAYERS),           # gae acc
        jnp.zeros(NUM_PLAYERS),           # next_value
        jnp.zeros(NUM_PLAYERS),           # reward_accum
        jnp.zeros(NUM_PLAYERS, dtype=bool),  # has_next_value
        False,                             # prev_done (unused)
        jnp.zeros(NUM_PLAYERS, dtype=bool),  # next_valid
    )

    # Pack inputs for scan
    inputs = (current_players, rewards, values, dones)

    _, (adv, targets, valid_mask) = lax.scan(scan_fn, init, inputs, reverse=True)
    return adv, targets, valid_mask  # (T, 4), (T, 4), (T, 4)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch GAE (exact replica of ppo_with_reg.py:compute_gae_vectorized)
# ═══════════════════════════════════════════════════════════════════════════

def pt_compute_gae_vectorized(rewards, values, dones, current_players,
                              gamma=1.0, gae_lambda=0.95):
    """Vectorized GAE — no per-environment Python loop over B."""
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
        # NOTE: next_valid is NOT reset on episode boundaries (matches JAX)

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
        # JAX: next_valid.at[player].set(is_valid) | done
        # When done=True, bool | array broadcasts True to ALL players
        next_valid[done] = True
        next_valid[b_idx, cp] = is_valid | done

    return advantages, targets, valid_mask


# ═══════════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════════

def run_gae_test(name, rewards, values, dones, current_players, tol=1e-6):
    """Run GAE on same inputs and compare."""
    T, B, P = rewards.shape

    # JAX: per-env
    jax_advs = []
    jax_tgts = []
    jax_vmasks = []
    for b in range(B):
        r = jnp.asarray(rewards[:, b, :])
        v = jnp.asarray(values[:, b])
        d = jnp.asarray(dones[:, b])
        cp = jnp.asarray(current_players[:, b].astype(np.int32))
        adv, tgt, vm = jax_single_env_gae(r, v, d, cp)
        jax_advs.append(np.array(adv))
        jax_tgts.append(np.array(tgt))
        jax_vmasks.append(np.array(vm))

    jax_adv = np.stack(jax_advs, axis=1)      # (T, B, P)
    jax_tgt = np.stack(jax_tgts, axis=1)
    jax_vm = np.stack(jax_vmasks, axis=1)

    # PT: vectorized
    pt_r = torch.from_numpy(rewards)
    pt_v = torch.from_numpy(values)
    pt_d = torch.from_numpy(dones)
    pt_cp = torch.from_numpy(current_players.astype(np.int64))

    pt_adv, pt_tgt, pt_vm = pt_compute_gae_vectorized(pt_r, pt_v, pt_d, pt_cp)

    pt_adv_np = pt_adv.numpy()
    pt_tgt_np = pt_tgt.numpy()
    pt_vm_np = pt_vm.numpy()

    # Compare
    adv_diff = np.abs(jax_adv - pt_adv_np).max()
    tgt_diff = np.abs(jax_tgt - pt_tgt_np).max()
    vm_diff = (jax_vm != pt_vm_np).sum()

    max_diff = max(adv_diff, tgt_diff)
    ok = max_diff < tol and vm_diff == 0

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         adv_max_diff={adv_diff:.2e}  tgt_max_diff={tgt_diff:.2e}  "
          f"vm_mismatches={vm_diff}")

    if not ok:
        # Show first mismatch location
        if adv_diff >= tol:
            idx = np.unravel_index(np.abs(jax_adv - pt_adv_np).argmax(),
                                   jax_adv.shape)
            print(f"         First adv mismatch at {idx}: "
                  f"JAX={jax_adv[idx]:.6f}  PT={pt_adv_np[idx]:.6f}")
        if vm_diff > 0:
            mismatch_idx = np.where(jax_vm != pt_vm_np)
            print(f"         First vm mismatch at "
                  f"({mismatch_idx[0][0]},{mismatch_idx[1][0]},{mismatch_idx[2][0]})")

    return ok


def main():
    print(f"\n{'='*60}")
    print("L2: GAE Computation Parity")
    print(f"{'='*60}\n")

    all_ok = True

    # ── Test 1: Normal case ──
    T, B, P = 256, 8, 4
    np.random.seed(SEED)
    rewards = np.random.randn(T, B, P).astype(np.float32) * 0.1
    # Scale rewards to emulate normalized rewards (/MAX_REWARD)
    rewards = rewards * (1.0 / 320.0)
    values = np.random.randn(T, B).astype(np.float32) * 0.5
    # Generate reasonable current_players (0-3) cycling
    current_players = np.zeros((T, B), dtype=np.int32)
    for b in range(B):
        current_players[:, b] = np.random.randint(0, P, size=T).astype(np.int32)
    # Sparse dones (episode boundaries every ~50 steps)
    dones = np.zeros((T, B), dtype=bool)
    dones[49, :] = True   # episode boundary at t=49
    dones[99, :] = True
    dones[149, :] = True
    dones[199, :] = True

    all_ok &= run_gae_test("Normal (T=256, B=8)", rewards, values, dones,
                           current_players)

    # ── Test 2: Small, deterministic ──
    T2, B2 = 4, 2
    rewards2 = np.array([
        [[1.0, 0, 0, 0], [0, 2.0, 0, 0]],
        [[0, 0, 3.0, 0], [4.0, 0, 0, 0]],
        [[0, 5.0, 0, 0], [0, 0, 0, 6.0]],
        [[7.0, 0, 0, 0], [0, 0, 8.0, 0]],
    ], dtype=np.float32) / 320.0
    values2 = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ], dtype=np.float32)
    current_players2 = np.array([
        [0, 1],
        [2, 0],
        [1, 3],
        [0, 2],
    ], dtype=np.int32)
    dones2 = np.zeros((T2, B2), dtype=bool)

    all_ok &= run_gae_test("Small deterministic (T=4, B=2)", rewards2, values2,
                           dones2, current_players2)

    # ── Test 3: With episode boundaries ──
    dones3 = np.zeros((T2, B2), dtype=bool)
    dones3[1, :] = True  # reset at t=1

    all_ok &= run_gae_test("With episode boundaries", rewards2, values2,
                           dones3, current_players2)

    # ── Test 4: All-zero rewards ──
    rewards4 = np.zeros((T2, B2, P), dtype=np.float32)
    all_ok &= run_gae_test("All-zero rewards", rewards4, values2,
                           dones2, current_players2)

    # ── Test 5: Extreme values ──
    np.random.seed(123)
    rewards5 = np.random.randn(T, B, P).astype(np.float32) * 2.0
    values5 = np.random.randn(T, B).astype(np.float32) * 10.0
    current_players5 = np.zeros((T, B), dtype=np.int32)
    for b in range(B):
        current_players5[:, b] = (np.arange(T) % 4).astype(np.int32)
    dones5 = np.zeros((T, B), dtype=bool)
    dones5[63, ::2] = True
    dones5[127, 1::2] = True
    dones5[191, :] = True

    all_ok &= run_gae_test("Extreme values + mixed boundaries", rewards5,
                           values5, dones5, current_players5, tol=1e-5)

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"L2 Result: {'[PASS] ALL IDENTICAL' if all_ok else '[FAIL] See diffs above'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
