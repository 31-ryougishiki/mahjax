#!/usr/bin/env python3
"""L4 Extended: Full ACNet PPO Loss Parity — JAX vs PyTorch.

Verifies the FULL ACNet PPO loss values match between JAX and PT.
Uses the proven weight transfer from test_ppo_weight_transfer.py.

Note: Gradient comparison requires consistent dict ordering that JAX's
jax.grad/optax reorder (alphabetically). The simplified MLP L4 test
(test_ppo_update_parity.py) already proves gradient + optimizer step parity.
"""

import sys, os
import numpy as np
import jax, jax.numpy as jnp
import torch

_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'examples')
sys.path.insert(0, _EXAMPLES_DIR)

from test_ppo_weight_transfer import JAC, flat, build_jax_to_pt_map, make_obs
import distrax

SEED = 42; BATCH = 32; NA = 87
CLIP_EPS = 0.2; ENT_COEF = 0.01; VF_COEF = 0.5
NEG = -1e9


def jax_masked_mean(x, mask):
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(mask.astype(jnp.float32).sum(), 1.0)


def jax_ppo_loss(network, params, obs, actions, old_log_probs, advantages,
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
    value_clipped = old_values[..., None] + jnp.clip(values[..., None] - old_values[..., None], -CLIP_EPS, CLIP_EPS)
    tgt = jnp.take_along_axis(targets, current_players[..., None], axis=1)
    loss_critic = 0.5 * VF_COEF * jax_masked_mean(jnp.maximum((values[..., None] - tgt) ** 2, (value_clipped - tgt) ** 2), mask)
    approx_kl = jax_masked_mean((ratio - 1.0) - log_ratio[..., None], mask)
    clip_frac = jax_masked_mean((jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32), mask)
    explained_var = jnp.maximum(1.0 - jax_masked_mean((tgt - values[..., None]) ** 2, mask) / (jax_masked_mean((tgt - jax_masked_mean(tgt, mask)) ** 2, mask) + 1e-8), 0.0)
    total_loss = ppo_loss - ENT_COEF * entropy + loss_critic
    return total_loss, {"total_loss": total_loss, "actor_loss": ppo_loss, "critic_loss": loss_critic, "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac, "explained_var": explained_var}


def pt_masked_mean(x, mask):
    return (x * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def pt_ppo_step(network, obs, actions, old_log_probs, advantages, targets,
                valid_mask, action_mask, old_values, current_players):
    logits, values_new = network(obs)
    logits = torch.where(action_mask, logits, torch.full_like(logits, NEG))
    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(actions)
    entropy = dist.entropy()
    log_ratio = logp_new - old_log_probs
    ratio = torch.exp(log_ratio).unsqueeze(-1)
    adv = advantages.gather(1, current_players.unsqueeze(-1))
    vmask = valid_mask.float().gather(1, current_players.unsqueeze(-1))
    clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    ppo_loss = -pt_masked_mean(torch.min(ratio * adv, clip_adv), vmask)
    vt = values_new.unsqueeze(-1)
    val_clipped = old_values.unsqueeze(-1) + torch.clamp(vt - old_values.unsqueeze(-1), -CLIP_EPS, CLIP_EPS)
    tgt = targets.gather(1, current_players.unsqueeze(-1))
    loss_critic = 0.5 * VF_COEF * pt_masked_mean(torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)
    approx_kl = pt_masked_mean((ratio - 1.0) - log_ratio.unsqueeze(-1), vmask)
    clip_frac = pt_masked_mean((torch.abs(ratio - 1.0) > CLIP_EPS).float(), vmask)
    explained_var = torch.clamp(1.0 - pt_masked_mean((tgt - vt) ** 2, vmask) / (pt_masked_mean((tgt - pt_masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8), min=0.0)
    total_loss = (ppo_loss - ENT_COEF * pt_masked_mean(entropy.unsqueeze(-1), vmask) + loss_critic)
    return total_loss, {"total_loss": total_loss, "actor_loss": ppo_loss, "critic_loss": loss_critic, "entropy": pt_masked_mean(entropy.unsqueeze(-1), vmask), "approx_kl": approx_kl, "clip_frac": clip_frac, "explained_var": explained_var}


def make_ppo_batch(batch_size=BATCH, seed=SEED):
    np.random.seed(seed)
    obs_np = make_obs(batch_size)
    actions = np.random.randint(0, NA, size=batch_size).astype(np.int32)
    old_log_probs = np.random.randn(batch_size).astype(np.float32) * 0.2
    old_values = np.random.randn(batch_size).astype(np.float32) * 0.5
    current_players = np.random.randint(0, 4, size=batch_size).astype(np.int32)
    advantages = np.zeros((batch_size, 4), dtype=np.float32)
    targets = np.zeros((batch_size, 4), dtype=np.float32)
    valid_mask = np.zeros((batch_size, 4), dtype=bool)
    for b in range(batch_size):
        p = current_players[b]
        advantages[b, p] = np.float32(np.random.randn() * 0.5)
        targets[b, p] = old_values[b] + advantages[b, p]
        valid_mask[b, p] = True
    action_mask = np.random.rand(batch_size, NA).astype(np.float32) > 0.7
    return obs_np, actions, old_log_probs, old_values, current_players, advantages, targets, valid_mask, action_mask


def main():
    print(f"\n{'='*60}")
    print("L4 Extended: Full ACNet PPO Loss Parity")
    print(f"{'='*60}\n")

    # Init
    print("Initializing JAX ACNet...")
    jax_net = JAC()
    dummy_obs = jax.tree.map(lambda x: jnp.asarray(x), make_obs(2))
    jax_params = jax_net.init(jax.random.PRNGKey(SEED), dummy_obs)
    print(f"  JAX params: {len(flat(jax_params))}")

    from mahjax_pt.examples.networks.red_network import ACNet as PTACNet
    pt_net = PTACNet()
    pt_params = list(pt_net.parameters())
    print(f"  PT params: {len(pt_params)}")

    # Transfer weights
    print("Transferring weights...")
    jax_to_pt = build_jax_to_pt_map()
    jax_flat_init = flat(jax_params)
    with torch.no_grad():
        for jax_idx, mapping in jax_to_pt.items():
            if mapping is None: continue
            pt_idx, mode = mapping
            jv = np.array(jax_flat_init[jax_idx])
            pp = pt_params[pt_idx]; ps = tuple(pp.shape)
            if mode == 'direct': pp.data.copy_(torch.from_numpy(jv))
            elif mode == 'transpose': pp.data.copy_(torch.from_numpy(jv.T))
            elif mode == 'reshape_3d': pp.data.copy_(torch.from_numpy(jv.reshape(ps).T))
    print("  [PASS]")

    # Verify forward
    print("Verifying forward pass...")
    obs_np, actions, old_log_probs, old_values, current_players, advantages, targets, valid_mask, action_mask = make_ppo_batch()
    jax_obs = jax.tree.map(lambda x: jnp.asarray(x), obs_np)
    pt_obs = {}
    for k, v in obs_np.items():
        if k in ['hand', 'action_history', 'dora_indicators']: pt_obs[k] = torch.from_numpy(v).long()
        elif k in ['furiten']: pt_obs[k] = torch.from_numpy(v).bool()
        else: pt_obs[k] = torch.from_numpy(v).int()

    jl, jv = jax_net.apply(jax_params, jax_obs)
    pt_net.eval()
    with torch.no_grad(): pl, pv = pt_net(pt_obs)
    ld = float(np.abs(np.array(jl) - pl.numpy()).max())
    vd = float(np.abs(np.array(jv) - pv.numpy()).max())
    print(f"  logit_diff={ld:.2e}  value_diff={vd:.2e}")
    assert ld < 1e-6 and vd < 1e-6
    print("  [PASS]\n")

    # PPO Loss
    print("Computing PPO loss...")
    jax_loss, jax_metrics = jax_ppo_loss(jax_net, jax_params, jax_obs, jnp.asarray(actions), jnp.asarray(old_log_probs), jnp.asarray(advantages), jnp.asarray(targets), jnp.asarray(valid_mask), jnp.asarray(action_mask), jnp.asarray(old_values), jnp.asarray(current_players))
    pt_net.train()
    pt_loss, pt_metrics = pt_ppo_step(pt_net, pt_obs, torch.from_numpy(actions).long(), torch.from_numpy(old_log_probs), torch.from_numpy(advantages), torch.from_numpy(targets), torch.from_numpy(valid_mask), torch.from_numpy(action_mask), torch.from_numpy(old_values), torch.from_numpy(current_players).long())

    print(f"\n{'─'*60}")
    print("Loss Comparison (Full ACNet)")
    print(f"{'─'*60}")
    print(f"  {'Metric':<20} {'JAX':>12} {'PT':>12} {'Diff':>12} {'Status':>8}")

    all_ok = True
    for key in ["total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac", "explained_var"]:
        jm = float(jax_metrics[key]); pm = float(pt_metrics[key].item())
        diff = abs(jm - pm); rel = diff / max(abs(jm), 1e-8)
        ok = rel < 1e-4
        if not ok: all_ok = False
        print(f"  {key:<18} {jm:12.8f} {pm:12.8f} {diff:12.2e} {('PASS' if ok else 'FAIL'):>8}")

    print(f"\n{'='*60}")
    print(f"L4 Extended: {'[PASS] FULL ACNet PPO LOSS IDENTICAL' if all_ok else '[FAIL]'}")
    print(f"(Gradient/optimizer: verified by test_ppo_update_parity.py)")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
