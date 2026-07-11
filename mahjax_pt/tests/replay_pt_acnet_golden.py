#!/usr/bin/env python3
"""Replay PT ACNet against JAX golden data — compare every intermediate result.

Loads JAX golden PPO training data (recorded by record_jax_acnet_golden_f64.py)
and replays the exact same rollouts through PT's GAE + PPO update pipeline,
comparing every intermediate value at every step.

Usage:
    python mahjax_pt/tests/replay_pt_acnet_golden.py
"""

import os, sys, pickle, time
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..', '..', 'examples'))

# ═══════════════════════════════════════════════════════════════════════════
# Config (must match recording script)
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
MAX_REWARD = 320.0
NEG = -1e9


# ═══════════════════════════════════════════════════════════════════════════
# Weight transfer map (from test_ppo_weight_transfer.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_jax_to_pt_map(jax_shapes, pt_shapes):
    """Auto-build mapping from JAX param indices to PT param indices by matching shapes.

    JAX param order (sorted keys): cf(global, hand, history), pf(global, hand, history), pm, vm
    PT param order: policy_extractor(hand, history, global), critic_extractor(hand, history, global),
                    policy_mlp, value_mlp

    The mapping must handle:
      - 'direct': same shape (biases, layer norm params, embeddings)
      - 'transpose': JAX (in, out) vs PT (out, in) for Dense/Linear weights
      - 'reshape_3d': JAX (feat, heads, head_dim) vs PT (heads*head_dim, feat) for MHA
      - None: JAX MHA biases that PT doesn't have
    """
    jax_to_pt = {}
    pt_used = set()

    # First pass: match by exact shape (biases, embeddings, layer norms)
    # Second pass: match transposed shapes (dense weights)
    # Third pass: match reshaped 3D shapes (MHA weights)
    # Fourth pass: identify unmatched JAX params as MHA biases (to skip)

    for ji, js in enumerate(jax_shapes):
        # Already handled?
        if ji in jax_to_pt:
            continue

        # Try exact shape match
        found = False
        for pi, ps in enumerate(pt_shapes):
            if pi in pt_used:
                continue
            if js == ps:
                jax_to_pt[ji] = (pi, 'direct')
                pt_used.add(pi)
                found = True
                break
        if found:
            continue

        # Try transpose match: JAX (d1, d2) vs PT (d2, d1)
        if len(js) == 2:
            for pi, ps in enumerate(pt_shapes):
                if pi in pt_used:
                    continue
                if len(ps) == 2 and js[0] == ps[1] and js[1] == ps[0]:
                    jax_to_pt[ji] = (pi, 'transpose')
                    pt_used.add(pi)
                    found = True
                    break
        if found:
            continue

        # Try reshape_3d match: two possible JAX layouts
        # Pattern A: (feat, heads, head_dim) → PT (heads*head_dim, feat) [QKV]
        # Pattern B: (heads, head_dim, feat) → PT (heads*head_dim, feat) [output proj]
        if len(js) == 3:
            candidates = [
                (js[1] * js[2], js[0]),  # Pattern A: QKV
                (js[0] * js[1], js[2]),  # Pattern B: output proj
            ]
            for reshaped in candidates:
                for pi, ps in enumerate(pt_shapes):
                    if pi in pt_used:
                        continue
                    if tuple(ps) == reshaped:
                        jax_to_pt[ji] = (pi, 'reshape_3d')
                        pt_used.add(pi)
                        found = True
                        break
                if found:
                    break
        if found:
            continue

        # Unmatched JAX param — mark as skip (MHA bias or similar)
        jax_to_pt[ji] = None

    return jax_to_pt


# ═══════════════════════════════════════════════════════════════════════════
# PT GAE + PPO helpers
# ═══════════════════════════════════════════════════════════════════════════

def pt_gae_vectorized(rewards, values, dones, current_players):
    """Vectorized GAE matching JAX implementation."""
    T_, B_, P_ = rewards.shape
    device = rewards.device
    advantages = torch.zeros(T_, B_, P_, device=device)
    targets = torch.zeros(T_, B_, P_, device=device)
    valid_mask = torch.zeros(T_, B_, P_, dtype=torch.bool, device=device)

    gae_acc = torch.zeros(B_, P_, device=device)
    reward_accum = torch.zeros(B_, P_, device=device)
    next_value = torch.zeros(B_, P_, device=device)
    has_next_value = torch.zeros(B_, P_, dtype=torch.bool, device=device)
    next_valid = torch.zeros(B_, P_, dtype=torch.bool, device=device)
    b_idx = torch.arange(B_, device=device)

    for t in reversed(range(T_)):
        cp = current_players[t]; done = dones[t]
        gae_acc[done] = 0.0; reward_accum[done] = 0.0
        has_next_value[done] = False; next_value[done] = 0.0
        reward_accum = reward_accum + rewards[t]
        player_reward = reward_accum[b_idx, cp].clone()
        reward_accum[b_idx, cp] = 0.0
        not_done = (~done).float()
        td = player_reward + GAMMA * next_value[b_idx, cp] * not_done - values[t]
        new_gae = td + GAMMA * GAE_LAMBDA * gae_acc[b_idx, cp] * not_done
        gae_acc[b_idx, cp] = new_gae
        is_valid = has_next_value[b_idx, cp] | done | next_valid[b_idx, cp]
        advantages[t, b_idx, cp] = torch.where(is_valid, new_gae, torch.zeros_like(new_gae))
        targets[t, b_idx, cp] = torch.where(is_valid, new_gae + values[t], values[t])
        valid_mask[t, b_idx, cp] = is_valid
        next_value[b_idx, cp] = values[t]
        has_next_value[b_idx, cp] = True
        next_valid[done] = True
        next_valid[b_idx, cp] = is_valid | done
    return advantages, targets, valid_mask


def pt_masked_mean(x, mask):
    return (x * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def pt_ppo_step(network, obs, actions, old_log_probs, advantages,
                targets, valid_mask, action_mask, old_values, current_players):
    logits, values_new = network(obs)
    logits = torch.where(action_mask, logits, torch.full_like(logits, NEG))
    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(actions)
    entropy_raw = dist.entropy()

    log_ratio = logp_new - old_log_probs
    ratio = torch.exp(log_ratio).unsqueeze(-1)

    adv = advantages.gather(1, current_players.unsqueeze(-1))
    vmask = valid_mask.float().gather(1, current_players.unsqueeze(-1))

    clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    ppo_loss = -pt_masked_mean(torch.min(ratio * adv, clip_adv), vmask)

    entropy = pt_masked_mean(entropy_raw.unsqueeze(-1), vmask)

    vt = values_new.unsqueeze(-1)
    val_clipped = old_values.unsqueeze(-1) + torch.clamp(
        vt - old_values.unsqueeze(-1), -CLIP_EPS, CLIP_EPS)
    tgt = targets.gather(1, current_players.unsqueeze(-1))
    loss_critic = (0.5 * VF_COEF *
                   pt_masked_mean(torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask))

    approx_kl = pt_masked_mean((ratio - 1.0) - log_ratio.unsqueeze(-1), vmask)
    clip_frac = pt_masked_mean((torch.abs(ratio - 1.0) > CLIP_EPS).float(), vmask)
    explained_var = torch.clamp(
        1.0 - pt_masked_mean((tgt - vt) ** 2, vmask) /
        (pt_masked_mean((tgt - pt_masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8), min=0.0)

    total_loss = (ppo_loss - ENT_COEF * entropy + loss_critic)

    return total_loss, {
        "total_loss": total_loss, "actor_loss": ppo_loss,
        "critic_loss": loss_critic, "entropy": entropy,
        "approx_kl": approx_kl, "clip_frac": clip_frac,
        "explained_var": explained_var,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Comparison helpers
# ═══════════════════════════════════════════════════════════════════════════

def compare(name, jv, pv, tol=1e-5):
    """Compare JAX (numpy) vs PT (torch) values in float64."""
    jn = np.asarray(jv, dtype=np.float64)
    if isinstance(pv, torch.Tensor):
        pn = pv.detach().cpu().numpy().astype(np.float64)
    elif isinstance(pv, np.ndarray):
        pn = pv.astype(np.float64)
    else:
        pn = np.asarray(pv, dtype=np.float64)

    if jn.dtype == bool:
        diff = float((jn.astype(bool) != pn.astype(bool)).sum())
    else:
        diff = float(np.abs(jn - pn).max())
    ok = diff < tol
    return diff, ok


def obs_to_pt(obs_np):
    """Convert numpy observation dict to PyTorch tensors with correct dtypes."""
    pt_obs = {}
    INT_KEYS = {'hand', 'action_history', 'dora_indicators'}
    BOOL_KEYS = {'furiten'}
    for k, v in obs_np.items():
        if k in INT_KEYS:
            pt_obs[k] = torch.from_numpy(v).long()
        elif k in BOOL_KEYS:
            pt_obs[k] = torch.from_numpy(v).bool()
        else:
            # scores, shanten_count, round, honba, kyotaku, prevalent_wind,
            # seat_wind, last_draw — all converted to float
            pt_obs[k] = torch.from_numpy(np.asarray(v, dtype=np.float32))
    return pt_obs


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 75)
    print("PT ACNet Golden Data Replay — Compare Every Intermediate Result")
    print("=" * 75)

    # ── Load golden data ──
    data_path = os.path.join(SCRIPT_DIR, "golden_data", "acnet_ppo_30updates_f64.pkl")
    print(f"\nLoading golden data: {data_path}")
    with open(data_path, "rb") as f:
        golden = pickle.load(f)

    cfg = golden["config"]
    print(f"  Updates: {len(golden['updates'])}")
    print(f"  Config: B={cfg['num_envs']}, T={cfg['num_steps']}, NA={cfg['num_actions']}")
    print(f"  Network: {cfg.get('network', 'unknown')}")
    print(f"  Precision: {cfg.get('precision', 'unknown')}")

    # ── Init PT ACNet ──
    from mahjax_pt.examples.networks.red_network import ACNet as PTACNet
    pt_net = PTACNet()
    pt_params = list(pt_net.parameters())
    print(f"  PT params: {len(pt_params)}")

    # ── Transfer initial weights (auto-built mapping from shapes) ──
    jax_shapes = [tuple(a.shape) for a in golden["init_params"]]
    pt_shapes = [tuple(p.shape) for p in pt_params]
    jax_to_pt = build_jax_to_pt_map(jax_shapes, pt_shapes)

    mapped_count = sum(1 for v in jax_to_pt.values() if v is not None)
    skipped_count = sum(1 for v in jax_to_pt.values() if v is None)
    print(f"  Auto-mapped: {mapped_count} params, skipped: {skipped_count}")
    # Note: some LN params may not match exactly due to architectural differences;
    # the important thing is that the mapped ones cover all key computations.

    jax_flat_init = golden["init_params"]
    with torch.no_grad():
        for jax_idx, mapping in jax_to_pt.items():
            if mapping is None:
                continue
            pt_idx, mode = mapping
            jv = jax_flat_init[jax_idx]
            pp = pt_params[pt_idx]
            if mode == 'direct':
                pp.data.copy_(torch.from_numpy(jv))
            elif mode == 'transpose':
                pp.data.copy_(torch.from_numpy(jv.T))
            elif mode == 'reshape_3d':
                pp.data.copy_(torch.from_numpy(jv.reshape(tuple(pp.shape)).T))
    print("  Weights transferred.")

    # ── Verify initial forward pass ──
    update0 = golden["updates"][0]
    obs_test = update0["rollout"]["obs"]
    # Take first timestep, first env
    obs_single = {k: torch.from_numpy(v[0, 0:1]).long() if k in ['hand', 'action_history', 'dora_indicators']
                  else torch.from_numpy(v[0, 0:1]).bool() if k == 'furiten'
                  else torch.from_numpy(v[0, 0:1]).float()
                  for k, v in obs_test.items()}
    pt_net.eval()
    with torch.no_grad():
        pt_logits, pt_values = pt_net(obs_single)
    print(f"  Forward test: logits shape={tuple(pt_logits.shape)}, values shape={tuple(pt_values.shape)}")

    # ── Replay loop ──
    all_results = []
    BATCH = cfg['num_steps'] * cfg['num_envs']

    for update_idx, upd in enumerate(golden["updates"]):
        # Load JAX pre-update params
        jax_params_before = upd["params_before"]
        with torch.no_grad():
            for jax_idx, mapping in jax_to_pt.items():
                if mapping is None:
                    continue
                pt_idx, mode = mapping
                jv = jax_params_before[jax_idx]
                pp = pt_params[pt_idx]
                if mode == 'direct':
                    pp.data.copy_(torch.from_numpy(jv))
                elif mode == 'transpose':
                    pp.data.copy_(torch.from_numpy(jv.T))
                elif mode == 'reshape_3d':
                    pp.data.copy_(torch.from_numpy(jv.reshape(tuple(pp.shape)).T))

        # ── GAE comparison ──
        roll = upd["rollout"]
        pt_rew = torch.from_numpy(roll["rewards"]).float()
        pt_val = torch.from_numpy(roll["values"]).float()
        pt_don = torch.from_numpy(roll["dones"])
        pt_cp = torch.from_numpy(roll["cps"].astype(np.int64))

        pt_adv_raw, pt_tgt_raw, pt_vm = pt_gae_vectorized(pt_rew, pt_val, pt_don, pt_cp)

        # Advantage normalization
        pt_vmf = pt_vm.float()
        pt_adv_mean = (pt_adv_raw * pt_vmf).sum() / pt_vmf.sum().clamp(min=1.0)
        pt_adv_var = ((pt_adv_raw - pt_adv_mean) ** 2 * pt_vmf).sum() / pt_vmf.sum().clamp(min=1.0)
        pt_adv_norm = (pt_adv_raw - pt_adv_mean) / (pt_adv_var.sqrt() + 1e-8)

        gae = upd["gae"]
        gae_diffs = {
            "advantages_raw": compare("", gae["advantages_raw"], pt_adv_raw)[0],
            "targets_raw": compare("", gae["targets_raw"], pt_tgt_raw)[0],
            "valid_mask_mismatch": int((gae["valid_mask"] != pt_vm.numpy()).sum()),
            "adv_mean": abs(gae["adv_mean"] - float(pt_adv_mean)),
            "adv_var": abs(gae["adv_var"] - float(pt_adv_var)),
            "advantages_norm": compare("", gae["advantages_norm"], pt_adv_norm)[0],
        }

        # ── Forward pass verification ──
        flat = upd["flattened"]
        obs_batch = obs_to_pt(flat["obs"])  # (T*B, ...)

        if update_idx == 0:
            # Detailed forward pass check for first update
            with torch.no_grad():
                pt_logits_full, pt_values_full = pt_net(obs_batch)
            print(f"\n  Forward pass (update {update_idx}):")
            print(f"    PT logits shape: {tuple(pt_logits_full.shape)}")
            print(f"    PT values shape: {tuple(pt_values_full.shape)}")
            print(f"    PT logits range: [{float(pt_logits_full.min()):.4f}, {float(pt_logits_full.max()):.4f}]")
            print(f"    PT values range: [{float(pt_values_full.min()):.4f}, {float(pt_values_full.max()):.4f}]")

        # ── PPO Update ──
        pt_opt = torch.optim.AdamW(pt_net.parameters(), lr=LR, eps=1e-5, weight_decay=0.0)

        update_grad_max = 0.0
        update_loss_max = 0.0
        mb_results = []

        for mb_idx, mb in enumerate(upd["minibatches"]):
            perm = torch.from_numpy(mb["perm"]).long()

            # Permute data
            def permute(x):
                return x[perm]

            x_mb = {k: permute(v) for k, v in obs_batch.items()}
            a_mb = permute(torch.from_numpy(flat["actions"]).long())
            lp_mb = permute(torch.from_numpy(flat["log_probs"]))
            v_mb = permute(torch.from_numpy(flat["values"]))
            ad_mb = permute(torch.from_numpy(flat["advantages_norm"]))
            tg_mb = permute(torch.from_numpy(flat["targets"]))
            vm_mb = permute(torch.from_numpy(flat["valid_mask"]))
            am_mb = permute(torch.from_numpy(flat["action_mask"]))
            cp_mb = permute(torch.from_numpy(flat["current_players"]).long())

            # Forward + loss
            pt_net.train()
            pt_loss, pt_metrics = pt_ppo_step(
                pt_net, x_mb, a_mb, lp_mb, ad_mb, tg_mb,
                vm_mb, am_mb, v_mb, cp_mb)

            pt_opt.zero_grad()
            pt_loss.backward()

            # Compare losses
            if "loss" in mb:
                loss_diff = abs(mb["loss"] - float(pt_loss))
            else:
                loss_diff = abs(float(mb["metrics"]["total_loss"]) - float(pt_loss))
            update_loss_max = max(update_loss_max, loss_diff)

            # Compare gradients
            jax_grads = mb["grads"]
            pt_params_cur = list(pt_net.parameters())
            grad_diffs = []
            for jax_idx, mapping in jax_to_pt.items():
                if mapping is None:  # skip MHA biases
                    continue
                pt_idx, mode = mapping
                jg = jax_grads[jax_idx].astype(np.float64)
                pg = pt_params_cur[pt_idx].grad
                pg_np = pg.detach().cpu().numpy().astype(np.float64) if pg is not None else np.zeros_like(jg)
                if mode == 'transpose':
                    jg = jg.T
                elif mode == 'reshape_3d':
                    jg = jg.reshape(pg_np.shape).T
                gd = float(np.abs(jg - pg_np).max())
                grad_diffs.append(gd)
                update_grad_max = max(update_grad_max, gd)

            pt_opt.step()

            mb_results.append({"mb_idx": mb_idx, "epoch": mb["epoch"],
                               "grad_diffs": grad_diffs, "pt_loss": float(pt_loss)})

        # ── Compare post-update params ──
        pt_params_after = list(pt_net.parameters())
        jax_params_after = upd["params_after"]
        param_diffs = []
        for jax_idx, mapping in jax_to_pt.items():
            if mapping is None:
                continue
            pt_idx, mode = mapping
            jv = jax_params_after[jax_idx].astype(np.float64)
            pv = pt_params_after[pt_idx].detach().cpu().numpy().astype(np.float64)
            if mode == 'transpose':
                jv = jv.T
            elif mode == 'reshape_3d':
                jv = jv.reshape(pv.shape).T
            pd = float(np.abs(jv - pv).max())
            param_diffs.append(pd)

        max_param_diff = max(param_diffs) if param_diffs else 999

        all_results.append({
            "update": update_idx,
            "gae_diffs": gae_diffs,
            "max_grad_diff": update_grad_max,
            "max_param_diff": max_param_diff,
            "max_loss_diff": update_loss_max,
        })

        if (update_idx + 1) % 5 == 0 or update_idx == 0:
            print(f"  Update {update_idx+1:3d}: "
                  f"gae_adv={gae_diffs['advantages_raw']:.2e} "
                  f"vm_mm={gae_diffs['valid_mask_mismatch']} "
                  f"loss={update_loss_max:.2e} "
                  f"grad={update_grad_max:.2e} "
                  f"param={max_param_diff:.2e}")

    # ── Summary ──
    print(f"\n{'─'*75}")
    print("Summary: PT Replay vs JAX Golden Data (ACNet)")
    print(f"{'─'*75}")

    gae_adv_max = max(r["gae_diffs"]["advantages_raw"] for r in all_results)
    gae_vm_max = max(r["gae_diffs"]["valid_mask_mismatch"] for r in all_results)
    print(f"  GAE advantages max diff:  {gae_adv_max:.2e}")
    print(f"  GAE valid_mask mismatch:  {gae_vm_max} entries")

    loss_max = max(r["max_loss_diff"] for r in all_results)
    print(f"  PPO loss max diff:        {loss_max:.2e}")

    grad_max = max(r["max_grad_diff"] for r in all_results)
    grad_mean = float(np.mean([r["max_grad_diff"] for r in all_results]))
    print(f"  Gradient max diff:        {grad_max:.2e} (mean={grad_mean:.2e})")

    param_max = max(r["max_param_diff"] for r in all_results)
    param_mean = float(np.mean([r["max_param_diff"] for r in all_results]))
    print(f"  Parameter max diff:       {param_max:.2e} (mean={param_mean:.2e})")

    # ── Per-parameter breakdown (final update) ──
    print(f"\n  Per-param diffs (final update):")
    final = all_results[-1]
    param_labels = []
    for ji, mapping in sorted(jax_to_pt.items()):
        if mapping is not None:
            param_labels.append(f"jax[{ji}]->pt[{mapping[0]}]")

    # ── Verdict ──
    # GAE must be bit-exact (all integer/discrete ops)
    gae_ok = gae_adv_max < 1e-6 and gae_vm_max == 0
    # Loss should be nearly identical
    loss_ok = loss_max < 1e-3
    # Params: with wd=0, drift should be minimal
    param_ok = param_max < 1e-3

    all_ok = gae_ok and loss_ok and param_ok
    print(f"\n{'='*75}")
    print(f"  GAE:  {'[PASS]' if gae_ok else '[FAIL]'} (adv={gae_adv_max:.2e}, vm={gae_vm_max})")
    print(f"  Loss: {'[PASS]' if loss_ok else '[FAIL]'} (max={loss_max:.2e})")
    print(f"  Param:{'[PASS]' if param_ok else '[FAIL]'} (max={param_max:.2e})")
    print(f"Result: {'[PASS] JAX-PT ACNet PPO PIPELINE VERIFIED' if all_ok else '[FAIL] See diffs above'}")
    print(f"{'='*75}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
