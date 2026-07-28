#!/usr/bin/env python3
"""Record JAX GAE golden data for mahjax_pt precision tests.

Generates .npz files with inputs AND JAX-computed outputs of GAE.

Usage:
    python scripts/record_golden_gae.py
    python scripts/record_golden_gae.py --output ./mahjax_pt/tests/golden

Requires: JAX
"""

import os, sys, argparse
import numpy as np

_JAX = None


def _init_jax():
    global _JAX
    if _JAX is not None:
        return
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_disable_jit', True)
    jax.config.update('jax_platform_name', 'cpu')
    _JAX = jax


def jax_gae(rewards, values, dones, current_players, gamma=1.0, gae_lambda=0.95):
    """JAX GAE implementation using lax.scan."""
    _init_jax()
    import jax.numpy as jnp
    from jax import lax

    P = rewards.shape[-1]

    def scan_fn(carry, inputs):
        gae, next_value, reward_accum, has_next_value, prev_done, next_valid = carry
        player, reward, value, done = inputs

        gae = jnp.where(done, 0, gae)
        reward_accum = jnp.where(done, 0, reward_accum)
        has_next_value = jnp.where(done, False, has_next_value)
        next_value = jnp.where(done, 0, next_value)

        reward_accum = reward_accum + reward
        player_reward = reward_accum[player]
        reward_accum = reward_accum.at[player].set(0.0)

        td_error = player_reward + gamma * next_value[player] - value
        new_gae = td_error + gamma * gae_lambda * gae[player]
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
            jnp.zeros(P).at[player].set(advantage),
            jnp.zeros(P).at[player].set(target),
            jnp.zeros(P, dtype=bool).at[player].set(is_valid),
        )
        return new_carry, output

    init = (
        jnp.zeros(P),
        jnp.zeros(P),
        jnp.zeros(P),
        jnp.zeros(P, dtype=bool),
        False,
        jnp.zeros(P, dtype=bool),
    )
    inputs = (current_players, rewards, values, dones)
    _, (adv, targets, valid_mask) = lax.scan(scan_fn, init, inputs, reverse=True)
    return adv, targets, valid_mask


def make_gae_inputs(T, B, P, seed=0, with_boundaries=True):
    """Generate synthetic GAE inputs."""
    rng = np.random.RandomState(seed)
    rewards = rng.randn(T, B, P).astype(np.float32) * 0.1
    values = rng.randn(T, B).astype(np.float32) * 0.5
    dones = np.zeros((T, B), dtype=bool)
    if with_boundaries:
        for step in range(40, T, 40):
            dones[step, :] = True
    current_players = np.zeros((T, B), dtype=np.int32)
    for b in range(B):
        current_players[:, b] = (np.arange(T) + rng.randint(0, P)) % P
    return rewards, values, dones, current_players


def record_gae(output_dir: str):
    """Record GAE golden data for 3 test configurations."""
    _init_jax()
    import jax.numpy as jnp

    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ("gae_T8_B4", 8, 4, 4, 42, True),
        ("gae_T256_B128", 256, 128, 4, 42, True),
        ("gae_T256_B64_boundaries", 256, 64, 4, 99, True),
    ]

    for name, T, B, P, seed, boundaries in configs:
        path = os.path.join(output_dir, f"{name}.npz")
        print(f"  Recording {name}...")

        rewards, values, dones, cps = make_gae_inputs(
            T, B, P, seed, boundaries)

        # JAX computation
        jr = jnp.asarray(rewards)
        jv = jnp.asarray(values)
        jd = jnp.asarray(dones)
        jc = jnp.asarray(cps.astype(np.int32))

        # GAE per-env (JAX operates on (T, P) per env)
        all_adv = np.zeros((T, B, P), dtype=np.float32)
        all_tgt = np.zeros((T, B, P), dtype=np.float32)
        all_vm = np.zeros((T, B, P), dtype=bool)

        for b in range(B):
            adv, tgt, vm = jax_gae(
                jr[:, b, :], jv[:, b], jd[:, b], jc[:, b])
            all_adv[:, b, :] = np.array(adv)
            all_tgt[:, b, :] = np.array(tgt)
            all_vm[:, b, :] = np.array(vm)

        golden = {
            "rewards": rewards,
            "values": values,
            "dones": dones,
            "current_players": cps,
            "advantages": all_adv,
            "targets": all_tgt,
            "valid_mask": all_vm,
        }
        np.savez_compressed(path, **golden)
        print(f"    T={T}, B={B}, P={P} → {path}")


def main():
    parser = argparse.ArgumentParser(description="Record JAX GAE golden data")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "mahjax_pt", "tests", "golden")
    args.output = os.path.abspath(args.output)

    print(f"Recording GAE golden data to: {args.output}\n")
    record_gae(args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
