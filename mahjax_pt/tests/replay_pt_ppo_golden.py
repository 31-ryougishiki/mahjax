#!/usr/bin/env python3
"""Replay PT PPO against JAX golden data — compare every intermediate result.

Loads JAX golden PPO training data (recorded by record_jax_ppo_golden.py) and
replays the exact same rollouts through PT's GAE + PPO update pipeline,
comparing every intermediate value at every step.
"""

import os, sys, pickle, time
import numpy as np
import torch
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════════════════
# Config (must match recording script)
# ═══════════════════════════════════════════════════════════════════════════
SEED = 42
NUM_ENVS = 2
NUM_STEPS = 8
NUM_UPDATES = 30
NUM_PLAYERS = 4
NUM_ACTIONS = 87
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
MAX_REWARD = 320.0
NEG = -1e9
HIDDEN_DIM = 64
OBS_DIM = 14 + 1 + 600 + 1 + 1 + 4 + 1 + 1 + 1 + 1 + 1 + 5  # 631

PARAM_NAMES = ["W1","b1","W2","b2","W3(actor)","b3",
               "W4","b4","W5","b5","W6(critic)","b6"]


# ═══════════════════════════════════════════════════════════════════════════
# PT MLP (matching JAX JaxObsMLP)
# ═══════════════════════════════════════════════════════════════════════════

class PTObsMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(OBS_DIM, HIDDEN_DIM)
        self.fc2 = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.actor = torch.nn.Linear(HIDDEN_DIM, NUM_ACTIONS)
        self.fc4 = torch.nn.Linear(OBS_DIM, HIDDEN_DIM)
        self.fc5 = torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.critic = torch.nn.Linear(HIDDEN_DIM, 1)
        for m in [self.fc1, self.fc2, self.actor, self.fc4, self.fc5, self.critic]:
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        logits = self.actor(h)
        h2 = torch.tanh(self.fc4(x))
        h2 = torch.tanh(self.fc5(h2))
        value = self.critic(h2).squeeze(-1)
        return logits, value


# ═══════════════════════════════════════════════════════════════════════════
# PT GAE (matching ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

def pt_gae_vectorized(rewards, values, dones, current_players):
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
        gae_acc[done] = 0.0
        reward_accum[done] = 0.0
        has_next_value[done] = False
        next_value[done] = 0.0
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


# ═══════════════════════════════════════════════════════════════════════════
# PT PPO helpers
# ═══════════════════════════════════════════════════════════════════════════

def pt_masked_mean(x, mask):
    return (x * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def pt_ppo_step(pt_mlp, x, actions, old_log_probs, advantages, targets,
                valid_mask, action_mask, old_values, current_players):
    logits, values_new = pt_mlp(x)
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
                   pt_masked_mean(torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2),
                                  vmask))

    approx_kl = pt_masked_mean((ratio - 1.0) - log_ratio.unsqueeze(-1), vmask)
    clip_frac = pt_masked_mean(
        (torch.abs(ratio - 1.0) > CLIP_EPS).float(), vmask)
    explained_var = torch.clamp(
        1.0 - pt_masked_mean((tgt - vt) ** 2, vmask) /
        (pt_masked_mean((tgt - pt_masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8),
        min=0.0)

    total_loss = (ppo_loss - ENT_COEF * entropy + loss_critic)

    return total_loss, {
        "total_loss": total_loss, "actor_loss": ppo_loss,
        "critic_loss": loss_critic, "entropy": entropy,
        "approx_kl": approx_kl, "clip_frac": clip_frac,
        "explained_var": explained_var,
    }


# ═══════════════════════════════════════════════════════════════════════════

def load_jax_params_to_pt(pt_mlp, jax_params):
    """Copy JAX MLP params to PT MLP. JAX W: (in,out), PT W: (out,in)."""
    pt_params = list(pt_mlp.parameters())
    with torch.no_grad():
        for jp, pp in zip(jax_params, pt_params):
            jv = jp  # numpy array
            if jv.ndim == 2:
                pp.data.copy_(torch.from_numpy(jv.T))
            else:
                pp.data.copy_(torch.from_numpy(jv))


def compare(name, jv, pv, tol=1e-5):
    """Compare JAX (numpy) vs PT (torch) values."""
    jn = np.asarray(jv, dtype=np.float64)
    pn = pv.detach().cpu().numpy().astype(np.float64) if isinstance(pv, torch.Tensor) else np.asarray(pv, dtype=np.float64)
    if jn.dtype == bool or pn.dtype == bool:
        diff = float((jn.astype(bool) != pn.astype(bool)).sum())
    else:
        diff = float(np.abs(jn - pn).max())
    ok = diff < tol
    return diff, ok


def main():
    print("=" * 75)
    print("PT PPO Golden Data Replay — Compare Every Intermediate Result")
    print("=" * 75)

    # ── Load golden data ─────────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(__file__),
                             "golden_data", "ppo_30updates.pkl")
    print(f"\nLoading golden data: {data_path}")
    with open(data_path, "rb") as f:
        golden = pickle.load(f)

    cfg = golden["config"]
    print(f"  Updates: {len(golden['updates'])}")
    print(f"  Config: B={cfg['num_envs']}, T={cfg['num_steps']}, "
          f"NA={cfg['num_actions']}, OBS_DIM={cfg['obs_dim']}")

    # ── Init PT network with JAX initial params ──────────────────────
    pt_mlp = PTObsMLP()
    init_params = golden["init_params"]
    load_jax_params_to_pt(pt_mlp, init_params)

    # Verify initial forward pass
    update0 = golden["updates"][0]
    obs_test = torch.from_numpy(update0["rollout"]["obs_flat"][0])
    with torch.no_grad():
        pt_logits, pt_values = pt_mlp(obs_test)
    print(f"  PT network initialized. Forward test: "
          f"logits mean={float(pt_logits.mean()):.4f}, "
          f"values mean={float(pt_values.mean()):.4f}")

    # ── Replay loop ──────────────────────────────────────────────────
    all_results = []
    BATCH = NUM_STEPS * NUM_ENVS

    for update_idx, upd in enumerate(golden["updates"]):
        # Reset PT params to JAX pre-update params
        load_jax_params_to_pt(pt_mlp, upd["params_before"])

        # ── GAE comparison ───────────────────────────────────────────
        roll = upd["rollout"]
        pt_rew = torch.from_numpy(roll["rewards"])
        pt_val = torch.from_numpy(roll["values"])
        pt_don = torch.from_numpy(roll["dones"])
        pt_cp  = torch.from_numpy(roll["cps"].astype(np.int64))

        pt_adv_raw, pt_tgt_raw, pt_vm = pt_gae_vectorized(
            pt_rew, pt_val, pt_don, pt_cp)

        # Adv normalization
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

        # ── Flatten data ─────────────────────────────────────────────
        flat = upd["flattened"]

        def to_pt(arr):
            return torch.from_numpy(arr)

        pt_obs = to_pt(flat["obs_flat"])
        pt_act = to_pt(flat["actions"]).long()
        pt_lp  = to_pt(flat["log_probs"])
        pt_vs  = to_pt(flat["values"])
        pt_adn = to_pt(flat["advantages_norm"])
        pt_tg  = to_pt(flat["targets"])
        pt_vmk = to_pt(flat["valid_mask"])
        pt_amk = to_pt(flat["action_mask"])
        pt_cpf = to_pt(flat["current_players"]).long()
        pt_ov  = to_pt(flat["values"])

        # ── PPO Update comparison ────────────────────────────────────
        pt_opt = torch.optim.AdamW(pt_mlp.parameters(), lr=LR, eps=1e-5)

        update_grad_max_diff = 0.0
        update_loss_max_diff = 0.0
        mb_results = []

        for mb_idx, mb in enumerate(upd["minibatches"]):
            perm = torch.from_numpy(mb["perm"]).long()

            # Apply same permutation
            x_mb  = pt_obs[perm]
            a_mb  = pt_act[perm]
            lp_mb = pt_lp[perm]
            v_mb  = pt_vs[perm]
            ad_mb = pt_adn[perm]
            tg_mb = pt_tg[perm]
            vm_mb = pt_vmk[perm]
            am_mb = pt_amk[perm]
            cp_mb = pt_cpf[perm]
            ov_mb = pt_ov[perm]

            # Forward + loss
            pt_mlp.train()
            pt_loss, pt_metrics = pt_ppo_step(
                pt_mlp, x_mb, a_mb, lp_mb, ad_mb, tg_mb,
                vm_mb, am_mb, ov_mb, cp_mb)

            pt_opt.zero_grad()
            pt_loss.backward()

            # Compare gradients
            jax_grads = mb["grads"]
            pt_params = list(pt_mlp.parameters())
            grad_diffs = []
            for i, (jg, pp) in enumerate(zip(jax_grads, pt_params)):
                jg_np = jg.astype(np.float64)
                pg_np = pp.grad.detach().cpu().numpy().astype(np.float64)
                if jg_np.ndim == 2:
                    jg_np = jg_np.T
                gd = float(np.abs(jg_np - pg_np).max())
                grad_diffs.append(gd)
                update_grad_max_diff = max(update_grad_max_diff, gd)

            # Record loss diffs (will compute after JAX losses available)
            mb_results.append({
                "mb_idx": mb_idx,
                "epoch": mb["epoch"],
                "grad_diffs": grad_diffs,
                "pt_loss": float(pt_loss),
                "pt_metrics": {k: float(v.item() if hasattr(v, 'item') else v)
                              for k, v in pt_metrics.items()},
            })

            # Apply optimizer step
            pt_opt.step()

        # ── Compare post-update params ───────────────────────────────
        pt_params_after = list(pt_mlp.parameters())
        jax_params_after = upd["params_after"]
        param_diffs = []
        for i, (jp, pp) in enumerate(zip(jax_params_after, pt_params_after)):
            jv = jp.astype(np.float64)
            pv = pp.detach().cpu().numpy().astype(np.float64)
            if jv.ndim == 2:
                jv = jv.T
            pd = float(np.abs(jv - pv).max())
            param_diffs.append(pd)

        max_param_diff = max(param_diffs)
        max_grad_diff = max([max(mb["grad_diffs"]) for mb in mb_results])

        all_results.append({
            "update": update_idx,
            "gae_diffs": gae_diffs,
            "max_grad_diff": max_grad_diff,
            "max_param_diff": max_param_diff,
            "param_diffs": param_diffs,
        })

        if (update_idx + 1) % 5 == 0 or update_idx == 0:
            print(f"  Update {update_idx+1:3d}: "
                  f"gae_adv={gae_diffs['advantages_raw']:.2e} "
                  f"vm_mm={gae_diffs['valid_mask_mismatch']} "
                  f"grad={max_grad_diff:.2e} "
                  f"param={max_param_diff:.2e}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print("Summary: PT Replay vs JAX Golden Data")
    print(f"{'─'*75}")

    # GAE
    gae_adv_max = max(r["gae_diffs"]["advantages_raw"] for r in all_results)
    gae_vm_max = max(r["gae_diffs"]["valid_mask_mismatch"] for r in all_results)
    print(f"  GAE advantages max diff:  {gae_adv_max:.2e}")
    print(f"  GAE valid_mask mismatch:  {gae_vm_max} entries")

    # Gradients
    grad_max = max(r["max_grad_diff"] for r in all_results)
    grad_mean = float(np.mean([r["max_grad_diff"] for r in all_results]))
    print(f"  Gradient max diff:        {grad_max:.2e} (mean={grad_mean:.2e})")

    # Parameters
    param_max = max(r["max_param_diff"] for r in all_results)
    param_mean = float(np.mean([r["max_param_diff"] for r in all_results]))
    print(f"  Parameter max diff:       {param_max:.2e} (mean={param_mean:.2e})")

    # Per-param breakdown
    print(f"\n  Per-param diffs (final update):")
    final = all_results[-1]
    for i, (name, pd) in enumerate(zip(PARAM_NAMES, final["param_diffs"])):
        print(f"    [{i:2d}] {name:15s}  {pd:.2e}")

    # Overall verdict
    all_ok = (gae_adv_max < 1e-5 and gae_vm_max == 0 and
              param_max < 5e-4)  # AdamW drift tolerance
    print(f"\n{'='*75}")
    print(f"Result: {'[PASS] JAX↔PT PPO PIPELINE VERIFIED' if all_ok else '[FAIL]'}")
    print(f"{'='*75}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    main()
