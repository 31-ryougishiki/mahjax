#!/usr/bin/env python3
"""Record JAX ACNet PPO golden data in float64 for PT replay verification.

Uses the FULL ACNet (transformer-based) with real mahjong environment rollout.
All computations in float64 to eliminate ULP-level precision differences.

Output: golden_data/acnet_ppo_f64.pkl
"""

import os, sys, pickle, time
import numpy as np
import jax, jax.numpy as jnp
import optax, distrax

# Keep float32 for env compatibility; replay script uses float64 for comparison

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'examples'))
import mahjax
from mahjax.wrappers.auto_reset_wrapper import auto_reset
from networks.red_network import ACNet

# ═══════════════════════════════════════════════════════════════════════════
SEED = 42; NUM_ENVS = 2; NUM_STEPS = 8; NUM_UPDATES = 30
NUM_PLAYERS = 4; NUM_ACTIONS = 87
GAMMA = 1.0; GAE_LAMBDA = 0.95; CLIP_EPS = 0.2
ENT_COEF = 0.01; VF_COEF = 0.5; LR = 3e-4; MAX_REWARD = 320.0; NEG = -1e9


def flat(tree):
    """Flatten JAX pytree to ordered list. Uses sorted keys for deterministic
    ordering that is IDENTICAL between params and grads (jax.grad() may return
    a FrozenDict with different insertion order than the original params).
    """
    r = []
    if isinstance(tree, dict):
        for k in sorted(tree.keys()):
            r.extend(flat(tree[k]))
    elif isinstance(tree, (jnp.ndarray, np.ndarray)):
        r.append(tree)
    return r

# ═══════════════════════════════════════════════════════════════════════════
# PPO helpers
# ═══════════════════════════════════════════════════════════════════════════

def jax_masked_mean(x, mask):
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(mask.astype(jnp.float32).sum(), 1.0)


def jax_gae_single(cps, rewards, values, dones):
    """Single-env GAE using Python for-loop (recording only, not for training)."""
    T = len(cps)
    gae_acc = jnp.zeros(4)
    reward_accum = jnp.zeros(4)
    next_value = jnp.zeros(4)
    has_next_value = jnp.zeros(4, dtype=bool)
    next_valid = jnp.zeros(4, dtype=bool)

    advantages = jnp.zeros((T, 4))
    targets = jnp.zeros((T, 4))
    valid_mask = jnp.zeros((T, 4), dtype=bool)

    for t in range(T - 1, -1, -1):
        cp = int(cps[t]); done = bool(dones[t])
        gae_acc = jnp.where(done, 0, gae_acc)
        reward_accum = jnp.where(done, 0, reward_accum)
        has_next_value = jnp.where(done, False, has_next_value)
        next_value = jnp.where(done, 0, next_value)

        reward_accum = reward_accum + jnp.array(rewards[t])
        player_reward = reward_accum[cp]
        reward_accum = reward_accum.at[cp].set(0.0)

        not_done = 0.0 if done else 1.0
        td = player_reward + GAMMA * next_value[cp] * not_done - float(values[t])
        new_gae = td + GAMMA * GAE_LAMBDA * gae_acc[cp] * not_done
        gae_acc = gae_acc.at[cp].set(new_gae)

        is_valid = has_next_value[cp] | done | next_valid[cp]

        advantages = advantages.at[t, cp].set(
            jnp.where(is_valid, new_gae, 0.0))
        targets = targets.at[t, cp].set(
            jnp.where(is_valid, new_gae + values[t], values[t]))
        valid_mask = valid_mask.at[t, cp].set(is_valid)

        next_value = next_value.at[cp].set(values[t])
        has_next_value = has_next_value.at[cp].set(True)
        next_valid = next_valid.at[cp].set(is_valid | done)
        next_valid = jnp.where(done, True, next_valid)

    return advantages, targets, valid_mask


def jax_ppo_loss(params, network, obs, actions, old_log_probs, advantages,
                 targets, valid_mask, action_mask, old_values, current_players):
    logits, values = network.apply(params, obs)
    logits = jnp.where(action_mask, logits, NEG)
    dist = distrax.Categorical(logits=logits)
    log_ratio = dist.log_prob(actions) - old_log_probs
    ratio = jnp.exp(log_ratio)[..., None]
    adv = jnp.take_along_axis(advantages, current_players[..., None], axis=1)
    mask = jnp.take_along_axis(valid_mask.astype(jnp.float32), current_players[..., None], axis=1)
    ppo_loss = -jax_masked_mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv), mask)
    entropy = jax_masked_mean(dist.entropy()[..., None], mask)
    vc = old_values[..., None] + jnp.clip(values[..., None] - old_values[..., None], -CLIP_EPS, CLIP_EPS)
    tgt = jnp.take_along_axis(targets, current_players[..., None], axis=1)
    loss_critic = 0.5 * VF_COEF * jax_masked_mean(jnp.maximum((values[..., None] - tgt) ** 2, (vc - tgt) ** 2), mask)
    approx_kl = jax_masked_mean((ratio - 1.0) - log_ratio[..., None], mask)
    clip_frac = jax_masked_mean((jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32), mask)
    explained_var = jnp.maximum(1.0 - jax_masked_mean((tgt - values[..., None]) ** 2, mask) / (jax_masked_mean((tgt - jax_masked_mean(tgt, mask)) ** 2, mask) + 1e-8), 0.0)
    total_loss = ppo_loss - ENT_COEF * entropy + loss_critic
    return total_loss, {"total_loss": total_loss, "actor_loss": ppo_loss, "critic_loss": loss_critic, "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac, "explained_var": explained_var}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("JAX ACNet PPO Golden Data Recording (float64)")
    print(f"Config: B={NUM_ENVS}, T={NUM_STEPS}, {NUM_UPDATES} updates")
    print("=" * 70)

    BASE_ENV = mahjax.make("red_mahjong", round_mode="single", observe_type="dict")
    step_fn = auto_reset(BASE_ENV.step, BASE_ENV.init)

    rng = jax.random.PRNGKey(SEED)
    rng, net_key, env_key = jax.random.split(rng, 3)
    network = ACNet()
    dummy_obs = BASE_ENV.observe(BASE_ENV.init(jax.random.PRNGKey(0)))
    params = network.init(net_key, dummy_obs)
    print(f"  ACNet params: {len(flat(params))}")

    opt = optax.adamw(learning_rate=LR, eps=1e-5, weight_decay=0.0)
    opt_state = opt.init(params)

    # Init envs as Python list (avoids vmap on auto_reset)
    env_keys = jax.random.split(env_key, NUM_ENVS)
    env_states = [BASE_ENV.init(k) for k in env_keys]

    from collections import namedtuple
    Transition = namedtuple('Transition', ['current_player', 'reward', 'value', 'is_new_episode'])

    golden = {"config": {"num_envs": NUM_ENVS, "num_steps": NUM_STEPS, "num_updates": NUM_UPDATES, "num_actions": NUM_ACTIONS, "num_players": NUM_PLAYERS, "seed": SEED, "precision": "float64", "network": "ACNet"}, "init_params": [np.array(p) for p in flat(params)], "updates": []}

    for update_idx in range(NUM_UPDATES):
        print(f"Update {update_idx + 1}/{NUM_UPDATES} ...", end=" ", flush=True)
        t0 = time.time()
        params_before = [np.array(p) for p in flat(params)]

        # ── Rollout ──
        obs_TB, acts_TB, logp_TB, vals_TB, rews_TB, dones_TB, cps_TB, masks_TB = [], [], [], [], [], [], [], []
        for t in range(NUM_STEPS):
            rng, akey, ekey = jax.random.split(rng, 3)
            obs_per = [BASE_ENV.observe(s) for s in env_states]
            obs_bat = jax.tree.map(lambda *xs: jnp.stack(xs), *obs_per)
            obs_TB.append(obs_bat)
            cps_TB.append(np.array([s.current_player for s in env_states], np.int32))
            dones_TB.append(np.array([s.terminated or s.truncated for s in env_states], bool))
            masks_TB.append(np.array([np.array(s.legal_action_mask) for s in env_states]))

            # ACNet adds batch dims internally; call per-env, then stack
            logits_per, vals_per = [], []
            for o in obs_per:
                l, v = network.apply(params, o)
                logits_per.append(np.array(l).squeeze(0))   # (1,87) → (87,)
                vals_per.append(np.array(v).squeeze())       # (1,) → ()
            logits_bat = jnp.array(logits_per)  # (B, 87)
            vals_TB.append(np.array(vals_per))

            def _sample(_logits, _mask, _key):
                _logits = jnp.where(_mask, _logits, NEG)
                d = distrax.Categorical(logits=_logits)
                a = d.sample(seed=_key)
                return a, d.log_prob(a)

            akeys = jax.random.split(akey, NUM_ENVS)
            acts, logp = jax.vmap(_sample)(logits_bat, jnp.array(masks_TB[-1]), akeys)
            acts_TB.append(np.array(acts)); logp_TB.append(np.array(logp))

            ekeys = jax.random.split(ekey, NUM_ENVS)
            for b in range(NUM_ENVS):
                env_states[b] = step_fn(env_states[b], int(acts_TB[-1][b]), ekeys[b])
            rews_TB.append(np.array([np.array(s.rewards, np.float64) for s in env_states]) / MAX_REWARD)

        rollout = {"actions": np.stack(acts_TB), "log_probs": np.stack(logp_TB), "values": np.stack(vals_TB), "rewards": np.stack(rews_TB), "dones": np.stack(dones_TB), "cps": np.stack(cps_TB), "masks": np.stack(masks_TB), "obs": {k: np.stack([np.array(o[k]) for o in obs_TB]) for k in obs_TB[0]}}

        # ── GAE ──
        all_adv, all_tgt, all_vm = [], [], []
        for b in range(NUM_ENVS):
            a, t, v = jax_gae_single(rollout["cps"][:, b], rollout["rewards"][:, b], rollout["values"][:, b], rollout["dones"][:, b])
            all_adv.append(np.array(a)); all_tgt.append(np.array(t)); all_vm.append(np.array(v))

        adv_raw = np.stack(all_adv, 1); tgt_raw = np.stack(all_tgt, 1); vm_raw = np.stack(all_vm, 1)
        vmf = vm_raw.astype(np.float64)
        adv_mean = float((adv_raw * vmf).sum() / max(vmf.sum(), 1.0))
        adv_var = float(((adv_raw - adv_mean)**2 * vmf).sum() / max(vmf.sum(), 1.0))
        adv_norm = (adv_raw - adv_mean) / (np.sqrt(adv_var) + 1e-8)
        gae_data = {"advantages_raw": adv_raw, "targets_raw": tgt_raw, "valid_mask": vm_raw, "adv_mean": adv_mean, "adv_var": adv_var, "advantages_norm": adv_norm}

        # ── Flatten ──
        BS = NUM_STEPS * NUM_ENVS
        def fl(arr): return arr.reshape(BS, *arr.shape[2:])

        obs_f = {k: fl(np.array(v)) for k, v in rollout["obs"].items()}
        acts_f = fl(np.array(rollout["actions"])); logp_f = fl(np.array(rollout["log_probs"]))
        vals_f = fl(np.array(rollout["values"])); adv_f = fl(np.array(gae_data["advantages_norm"]))
        tgt_f = fl(np.array(gae_data["targets_raw"])); vm_f = fl(np.array(gae_data["valid_mask"]))
        am_f = fl(np.array(rollout["masks"])); cp_f = fl(np.array(rollout["cps"]))

        # ── PPO Update ──
        mb_data = []
        for epoch in range(2):
            rng, pkey = jax.random.split(rng)
            perm = jax.random.permutation(pkey, BS)

            def pm(arr): return arr[perm]
            mb_obs = {k: pm(jnp.array(v)) for k, v in obs_f.items()}
            mb_act = pm(jnp.array(acts_f)); mb_logp = pm(jnp.array(logp_f))
            mb_val = pm(jnp.array(vals_f)); mb_adv = pm(jnp.array(adv_f))
            mb_tgt = pm(jnp.array(tgt_f)); mb_vm = pm(jnp.array(vm_f))
            mb_am = pm(jnp.array(am_f)); mb_cp = pm(jnp.array(cp_f))

            loss_fn = lambda p: jax_ppo_loss(p, network, mb_obs, mb_act, mb_logp, mb_adv, mb_tgt, mb_vm, mb_am, mb_val, mb_cp)[0]
            _, metrics = jax_ppo_loss(params, network, mb_obs, mb_act, mb_logp, mb_adv, mb_tgt, mb_vm, mb_am, mb_val, mb_cp)
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            mb_data.append({"epoch": epoch, "perm": np.array(perm), "grads": [np.array(g) for g in flat(grads)], "loss": float(metrics["total_loss"]), "metrics": {k: float(v) for k, v in metrics.items()}})

        golden["updates"].append({"update_idx": update_idx, "params_before": params_before, "params_after": [np.array(p) for p in flat(params)], "rollout": rollout, "gae": gae_data, "flattened": {"obs": obs_f, "actions": acts_f, "log_probs": logp_f, "values": vals_f, "advantages_norm": adv_f, "targets": tgt_f, "valid_mask": vm_f, "action_mask": am_f, "current_players": cp_f}, "minibatches": mb_data})

        print(f"{time.time() - t0:.1f}s")

    out_dir = os.path.join(os.path.dirname(__file__), "golden_data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "acnet_ppo_30updates_f64.pkl")
    print(f"\nSaving to {out_path} ...")
    with open(out_path, "wb") as f: pickle.dump(golden, f)
    print(f"Done. {os.path.getsize(out_path)/1024/1024:.1f} MB, {len(golden['updates'])} updates")
    return 0


if __name__ == "__main__":
    main()
