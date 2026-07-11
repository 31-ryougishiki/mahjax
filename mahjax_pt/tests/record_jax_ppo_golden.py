#!/usr/bin/env python3
"""Record JAX PPO training golden data for 30 updates (fast vmap version).

Uses jax.vmap + lax.scan for efficient rollout (matching ppo_with_reg.py),
then records all intermediate data for PT replay comparison.
"""

import os, sys, pickle, time
from typing import NamedTuple
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax
import distrax

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset

# ═══════════════════════════════════════════════════════════════════════════
SEED = 42; NUM_ENVS = 2; NUM_STEPS = 8; NUM_UPDATES = 30
NUM_PLAYERS = 4; NUM_ACTIONS = 87
GAMMA = 1.0; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
ENT_COEF = 0.01; VF_COEF = 0.5; LR = 3e-4
MAX_REWARD = 320.0; NEG = -1e9; HIDDEN_DIM = 64
# Obs dim: hand(14) + last_draw(1) + action_history(600) + shanten(1) + furiten(1)
# + scores(4) + round(1) + honba(1) + kyotaku(1) + prevalent_wind(1)
# + seat_wind(1) + dora_indicators(5) = 631
OBS_DIM = 631

BASE_ENV = mahjax.make("red_mahjong", round_mode="single", observe_type="dict")
step_fn = auto_reset(BASE_ENV.step, BASE_ENV.init)
BATCH_SIZE = NUM_STEPS * NUM_ENVS

# ═══════════════════════════════════════════════════════════════════════════
# Observation flattener
# ═══════════════════════════════════════════════════════════════════════════

def flatten_obs(obs):
    """Flatten single (unbatched) observation dict into (OBS_DIM,) float32."""
    parts = [
        obs["hand"].astype(jnp.float32).reshape(-1),                # (14,)
        obs["last_draw"].astype(jnp.float32).reshape(-1),           # (1,)
        obs["action_history"].astype(jnp.float32).reshape(-1),      # (600,)
        obs["shanten_count"].astype(jnp.float32).reshape(-1),       # (1,)
        obs["furiten"].astype(jnp.float32).reshape(-1),             # (1,)
        obs["scores"].astype(jnp.float32).reshape(-1),              # (4,)
        obs["round"].astype(jnp.float32).reshape(-1),               # (1,)
        obs["honba"].astype(jnp.float32).reshape(-1),               # (1,)
        obs["kyotaku"].astype(jnp.float32).reshape(-1),             # (1,)
        obs["prevalent_wind"].astype(jnp.float32).reshape(-1),      # (1,)
        obs["seat_wind"].astype(jnp.float32).reshape(-1),           # (1,)
        obs["dora_indicators"].astype(jnp.float32).reshape(-1),     # (5,)
    ]
    return jnp.concatenate(parts, axis=-1)  # (631,)


# ═══════════════════════════════════════════════════════════════════════════
# MLP Network
# ═══════════════════════════════════════════════════════════════════════════

class JaxObsMLP:
    def __init__(self, rng):
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
        self.W1 = jax.random.orthogonal(k1, OBS_DIM, m=HIDDEN_DIM)
        self.b1 = jnp.zeros(HIDDEN_DIM)
        self.W2 = jax.random.orthogonal(k2, HIDDEN_DIM, m=HIDDEN_DIM)
        self.b2 = jnp.zeros(HIDDEN_DIM)
        self.W3 = jax.random.orthogonal(k3, HIDDEN_DIM, m=NUM_ACTIONS) * 0.01
        self.b3 = jnp.zeros(NUM_ACTIONS)
        self.W4 = jax.random.orthogonal(k4, OBS_DIM, m=HIDDEN_DIM)
        self.b4 = jnp.zeros(HIDDEN_DIM)
        self.W5 = jax.random.orthogonal(k5, HIDDEN_DIM, m=HIDDEN_DIM)
        self.b5 = jnp.zeros(HIDDEN_DIM)
        self.W6 = jax.random.orthogonal(k6, HIDDEN_DIM, m=1)
        self.b6 = jnp.zeros(1)

    def __call__(self, x):
        h = jnp.tanh(x @ self.W1 + self.b1)
        h = jnp.tanh(h @ self.W2 + self.b2)
        logits = h @ self.W3 + self.b3
        h2 = jnp.tanh(x @ self.W4 + self.b4)
        h2 = jnp.tanh(h2 @ self.W5 + self.b5)
        value = (h2 @ self.W6 + self.b6).squeeze(-1)
        return logits, value

    def params_list(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                self.W4, self.b4, self.W5, self.b5, self.W6, self.b6]


# ═══════════════════════════════════════════════════════════════════════════
# Transition namedtuple (matching ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

class Transition(NamedTuple):
    is_new_episode: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation_flat: jnp.ndarray
    action_mask: jnp.ndarray
    current_player: jnp.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# Rollout function (vmap + scan, matching ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

def make_collect_rollout(params_list):
    """Create a rollout function that uses vmap + scan."""
    def step_fn_scan(carry, _):
        state, rng = carry
        rng, action_key, env_key = jax.random.split(rng, 3)

        observation = BASE_ENV.observe(state)
        obs_flat = flatten_obs(observation)
        action_mask = state.legal_action_mask.astype(jnp.bool_)
        current_player = jnp.asarray(state.current_player, dtype=jnp.int32)
        is_new_episode = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)

        # Forward pass
        mlp = JaxObsMLP.__new__(JaxObsMLP)
        (mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3,
         mlp.W4, mlp.b4, mlp.W5, mlp.b5, mlp.W6, mlp.b6) = params_list
        logits, value = mlp(obs_flat)
        logits = jnp.where(action_mask, logits, NEG)
        dist = distrax.Categorical(logits=logits)
        action, log_prob = dist.sample_and_log_prob(seed=action_key)

        next_state = step_fn(state, action, env_key)
        reward = jnp.asarray(next_state.rewards, dtype=jnp.float32) / MAX_REWARD

        transition = Transition(
            is_new_episode=is_new_episode, action=action, value=value,
            reward=reward, log_prob=log_prob, observation_flat=obs_flat,
            action_mask=action_mask, current_player=current_player,
        )
        return (next_state, rng), transition

    def collect_rollout(env_state, key):
        batched_scan = jax.vmap(
            lambda c, x: lax.scan(step_fn_scan, c, None, length=NUM_STEPS))
        keys = jax.random.split(key, NUM_ENVS)
        (env_state, _), transitions = batched_scan((env_state, keys), None)
        return env_state, transitions
    return collect_rollout


# ═══════════════════════════════════════════════════════════════════════════
# GAE + PPO helpers
# ═══════════════════════════════════════════════════════════════════════════

def jax_masked_mean(x, mask):
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(
        mask.astype(jnp.float32).sum(), 1.0)


def calculate_gae_single(transitions):
    """Single-env GAE (matching ppo_with_reg.py:calculate_gae)."""
    def scan_fn(carry, t: Transition):
        gae, next_value, reward_accum, has_next_value, is_new_episode, next_valid_mask = carry
        player, reward, value, done = t.current_player, t.reward, t.value, t.is_new_episode
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

        is_valid = has_next_value[player] | done | next_valid_mask[player]
        advantage = jnp.where(is_valid, new_gae, 0.0)
        target = jnp.where(is_valid, advantage + value, value)

        new_carry = (
            gae, next_value.at[player].set(value), reward_accum,
            has_next_value.at[player].set(True), done,
            next_valid_mask.at[player].set(is_valid) | done
        )
        output = (
            jnp.zeros(NUM_PLAYERS).at[player].set(advantage),
            jnp.zeros(NUM_PLAYERS).at[player].set(target),
            jnp.zeros(NUM_PLAYERS, dtype=bool).at[player].set(is_valid)
        )
        return new_carry, output

    init = (jnp.zeros(NUM_PLAYERS), jnp.zeros(NUM_PLAYERS), jnp.zeros(NUM_PLAYERS),
            jnp.zeros(NUM_PLAYERS, dtype=bool), False, jnp.zeros(NUM_PLAYERS, dtype=bool))
    _, (adv, targets, valid_mask) = lax.scan(scan_fn, init, transitions, reverse=True)
    return adv, targets, valid_mask


def jax_ppo_loss_fn(params_list, x, actions, old_log_probs, advantages, targets,
                    valid_mask, action_mask, old_values, current_players):
    mlp = JaxObsMLP.__new__(JaxObsMLP)
    (mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3,
     mlp.W4, mlp.b4, mlp.W5, mlp.b5, mlp.W6, mlp.b6) = params_list
    logits, values = mlp(x)
    logits = jnp.where(action_mask, logits, NEG)
    dist = distrax.Categorical(logits=logits)
    log_ratio = dist.log_prob(actions) - old_log_probs
    ratio = jnp.exp(log_ratio)[..., None]

    adv = jnp.take_along_axis(advantages, current_players[..., None], axis=1)
    mask = jnp.take_along_axis(valid_mask.astype(jnp.float32),
                               current_players[..., None], axis=1)

    ppo_loss = -jax_masked_mean(
        jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv), mask)
    entropy = jax_masked_mean(dist.entropy()[..., None], mask)

    ov = old_values
    value_clipped = ov[..., None] + jnp.clip(
        values[..., None] - ov[..., None], -CLIP_EPS, CLIP_EPS)
    tgt = jnp.take_along_axis(targets, current_players[..., None], axis=1)
    loss_critic = (0.5 * VF_COEF *
                   jax_masked_mean(jnp.maximum((values[..., None] - tgt) ** 2,
                                               (value_clipped - tgt) ** 2), mask))

    total_loss = ppo_loss - ENT_COEF * entropy + loss_critic
    return total_loss


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("JAX PPO Golden Data Recording (vmap version)")
    print(f"Config: B={NUM_ENVS}, T={NUM_STEPS}, {NUM_UPDATES} updates")
    print("=" * 70)

    rng = jax.random.PRNGKey(SEED)
    rng, net_key, env_key = jax.random.split(rng, 3)

    # Init network
    mlp = JaxObsMLP(net_key)
    params = mlp.params_list()

    # Init optimizer
    opt = optax.adamw(learning_rate=LR, eps=1e-5)
    opt_state = opt.init(params)

    # Init environments (batched)
    env_keys = jax.random.split(env_key, NUM_ENVS)
    env_state = jax.vmap(BASE_ENV.init)(env_keys)

    # Create rollout function
    collect_rollout_fn = make_collect_rollout(params)

    # Storage
    golden = {
        "config": {"num_envs": NUM_ENVS, "num_steps": NUM_STEPS,
                   "num_updates": NUM_UPDATES, "num_actions": NUM_ACTIONS,
                   "num_players": NUM_PLAYERS, "obs_dim": OBS_DIM,
                   "hidden_dim": HIDDEN_DIM, "seed": SEED},
        "init_params": [np.array(p) for p in params],
        "updates": [],
    }

    for update_idx in range(NUM_UPDATES):
        print(f"Update {update_idx + 1}/{NUM_UPDATES} ...", end=" ", flush=True)
        t0 = time.time()

        params_before = [np.array(p) for p in params]

        # ── Rollout ──────────────────────────────────────────────────
        # Update the rollout function with current params
        collect_rollout_fn = make_collect_rollout(params)
        rng, rollout_key = jax.random.split(rng)
        env_state, transitions = collect_rollout_fn(env_state, rollout_key)

        # Convert transitions to numpy, transpose (B,T,...) → (T,B,...)
        rollout = {
            "obs_flat": np.array(transitions.observation_flat).transpose(1,0,2),
            "actions": np.array(transitions.action).transpose(1,0),
            "log_probs": np.array(transitions.log_prob).transpose(1,0),
            "values": np.array(transitions.value).transpose(1,0),
            "rewards": np.array(transitions.reward).transpose(1,0,2),
            "dones": np.array(transitions.is_new_episode).transpose(1,0),
            "cps": np.array(transitions.current_player).transpose(1,0),
            "masks": np.array(transitions.action_mask).transpose(1,0,2),
        }

        # ── GAE ──────────────────────────────────────────────────────
        # JAX vmap over B envs produces (B,T,P); transpose to (T,B,P)
        adv_raw, tgt_raw, vm_raw = jax.vmap(calculate_gae_single)(transitions)

        vmf = vm_raw.astype(jnp.float32)
        adv_mean = jax_masked_mean(adv_raw, vmf)
        adv_var = jax_masked_mean((adv_raw - adv_mean) ** 2, vmf)
        adv_norm = (adv_raw - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)

        gae_data = {
            "advantages_raw": np.array(adv_raw).transpose(1,0,2),
            "targets_raw": np.array(tgt_raw).transpose(1,0,2),
            "valid_mask": np.array(vm_raw).transpose(1,0,2),
            "adv_mean": float(adv_mean),
            "adv_var": float(adv_var),
            "advantages_norm": np.array(adv_norm).transpose(1,0,2),
        }

        # ── PPO Update ───────────────────────────────────────────────
        # Flatten (T,B,...) → (T*B,...)
        BATCH_SIZE = NUM_STEPS * NUM_ENVS

        def flat(x):
            return x.reshape(BATCH_SIZE, *x.shape[2:])

        obs_f = flat(jnp.asarray(rollout["obs_flat"]))
        acts_f = flat(jnp.asarray(rollout["actions"]))
        logp_f = flat(jnp.asarray(rollout["log_probs"]))
        vals_f = flat(jnp.asarray(rollout["values"]))
        adv_f = flat(jnp.asarray(gae_data["advantages_norm"]))
        tgt_f = flat(jnp.asarray(gae_data["targets_raw"]))
        vm_f = flat(jnp.asarray(gae_data["valid_mask"]))
        am_f = flat(jnp.asarray(rollout["masks"]))
        cp_f = flat(jnp.asarray(rollout["cps"]))

        update_epochs = 2
        minibatch_data = []

        for epoch in range(update_epochs):
            rng, perm_key = jax.random.split(rng)
            perm = jax.random.permutation(perm_key, BATCH_SIZE)

            jx_mb = obs_f[perm]
            ja_mb = acts_f[perm]
            jlp_mb = logp_f[perm]
            jv_mb = vals_f[perm]
            jadv_mb = adv_f[perm]
            jtgt_mb = tgt_f[perm]
            jvm_mb = vm_f[perm]
            jam_mb = am_f[perm]
            jcp_mb = cp_f[perm]

            grads = jax.grad(jax_ppo_loss_fn)(
                params, jx_mb, ja_mb, jlp_mb, jadv_mb, jtgt_mb,
                jvm_mb, jam_mb, jv_mb, jcp_mb)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            minibatch_data.append({
                "epoch": epoch,
                "perm": np.array(perm),
                "grads": [np.array(g) for g in grads],
            })

        params_after = [np.array(p) for p in params]

        golden["updates"].append({
            "update_idx": update_idx,
            "params_before": params_before,
            "params_after": params_after,
            "rollout": rollout,
            "gae": gae_data,
            "flattened": {
                "obs_flat": np.array(obs_f),
                "actions": np.array(acts_f),
                "log_probs": np.array(logp_f),
                "values": np.array(vals_f),
                "advantages_norm": np.array(adv_f),
                "targets": np.array(tgt_f),
                "valid_mask": np.array(vm_f),
                "action_mask": np.array(am_f),
                "current_players": np.array(cp_f),
            },
            "minibatches": minibatch_data,
        })

        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s")
        sys.stdout.flush()

    # ── Save ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "golden_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ppo_30updates.pkl")

    print(f"\nSaving to {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump(golden, f)

    file_size = os.path.getsize(out_path) / 1024 / 1024
    print(f"Done. File size: {file_size:.1f} MB")
    return 0


if __name__ == "__main__":
    main()
