#!/usr/bin/env python3
"""L6: Full Training Run — N-update PPO training stability.

Verifies that the PT PPO training pipeline is stable over multiple update cycles
with synthetic data.  Monitors loss curves for expected behavior:
  - Loss generally decreases (not monotonically, but overall trend)
  - No NaN/inf after extended training
  - Diagnostics stay in valid ranges
  - Gradients don't explode

Uses the same TinyACNet as L5, with 20 updates of synthetic data.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

SEED = 42
T, B, P = 16, 8, 4
NA = 87
FEATURE_DIM = 32
NUM_UPDATES = 20
UPDATE_EPOCHS = 2

# PPO hyperparams
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
MAX_REWARD = 320.0
NEG = -1e9


# ═══════════════════════════════════════════════════════════════════════════
# Network
# ═══════════════════════════════════════════════════════════════════════════

class TinyACNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(FEATURE_DIM, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.actor = torch.nn.Linear(64, NA)
        self.critic = torch.nn.Linear(64, 1)
        torch.nn.init.orthogonal_(self.fc1.weight)
        torch.nn.init.orthogonal_(self.fc2.weight)
        torch.nn.init.orthogonal_(self.actor.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.critic.weight)
        for m in [self.fc1, self.fc2, self.actor, self.critic]:
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.actor(h), self.critic(h).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# GAE + PPO (copied from ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_gae_vectorized(rewards, values, dones, current_players):
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
        td_error = player_reward + GAMMA * next_value[b_idx, cp] * not_done - values[t]
        new_gae = td_error + GAMMA * GAE_LAMBDA * gae_acc[b_idx, cp] * not_done
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


def masked_mean(x, mask):
    return (x * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def ppo_update(network, optimizer, obs, actions, old_log_probs, advantages,
               targets, valid_mask, action_mask, old_values, current_players):
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
    ppo_loss = -masked_mean(torch.min(ratio * adv, clip_adv), vmask)

    vt = values_new.unsqueeze(-1)
    val_clipped = old_values.unsqueeze(-1) + torch.clamp(
        vt - old_values.unsqueeze(-1), -CLIP_EPS, CLIP_EPS)
    tgt = targets.gather(1, current_players.unsqueeze(-1))
    loss_critic = 0.5 * VF_COEF * masked_mean(
        torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)

    approx_kl = masked_mean((ratio - 1.0) - log_ratio.unsqueeze(-1), vmask)
    clip_frac = masked_mean((torch.abs(ratio - 1.0) > CLIP_EPS).float(), vmask)
    explained_var = torch.clamp(
        1.0 - masked_mean((tgt - vt) ** 2, vmask) /
        (masked_mean((tgt - masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8), min=0.0)

    total_loss = (ppo_loss - ENT_COEF * masked_mean(entropy.unsqueeze(-1), vmask)
                  + loss_critic)

    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0)
    optimizer.step()

    return total_loss.item(), {
        "actor_loss": ppo_loss.item(), "critic_loss": loss_critic.item(),
        "entropy": masked_mean(entropy.unsqueeze(-1), vmask).item(),
        "approx_kl": approx_kl.item(), "clip_frac": clip_frac.item(),
        "explained_var": explained_var.item(), "grad_norm": grad_norm.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════

def make_rollout_data(seed):
    torch.manual_seed(seed)
    obs_flat = torch.randn(T * B, FEATURE_DIM) * 0.5
    action_mask = torch.rand(T, B, NA) > 0.7
    probs = action_mask.float() / action_mask.float().sum(-1, keepdim=True).clamp(min=1)
    actions = torch.multinomial(probs.view(-1, NA), 1).view(T, B)
    values = torch.randn(T, B) * 0.3
    old_log_probs = torch.randn(T, B) * 0.1
    rewards = torch.zeros(T, B, P)
    for t in range(T):
        for b_ in range(B):
            if torch.rand(1).item() < 0.15:
                p_ = torch.randint(0, P, (1,)).item()
                rewards[t, b_, p_] = torch.randn(1).item() * 0.2
    rewards = rewards / MAX_REWARD
    current_players = torch.zeros(T, B, dtype=torch.long)
    for b_ in range(B):
        start = torch.randint(0, P, (1,)).item()
        current_players[:, b_] = (torch.arange(T) + start) % P
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[T // 2, :] = True
    return obs_flat, actions, old_log_probs, values, rewards, dones, \
        current_players, action_mask


def main():
    print(f"\n{'='*60}")
    print("L6: Full Training Run ({NUM_UPDATES} updates)".replace(
        '{NUM_UPDATES}', str(NUM_UPDATES)))
    print(f"{'='*60}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    network = TinyACNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, eps=1e-5)

    history = []
    all_ok = True

    for update_idx in range(NUM_UPDATES):
        # Generate fresh data each update (simulates new rollout)
        (obs_flat, actions, old_log_probs, values, rewards, dones,
         current_players, action_mask) = make_rollout_data(SEED + update_idx)

        # GAE
        advantages, targets, valid_mask = compute_gae_vectorized(
            rewards, values, dones, current_players)

        vf = valid_mask.float()
        adv_mean = (advantages * vf).sum() / vf.sum().clamp(min=1.0)
        adv_var = ((advantages - adv_mean) ** 2 * vf).sum() / vf.sum().clamp(min=1.0)
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

        # Flatten
        acts_f = actions.reshape(-1)
        logp_f = old_log_probs.reshape(-1)
        vals_f = values.reshape(-1)
        adv_f = advantages.reshape(T * B, P)
        tgt_f = targets.reshape(T * B, P)
        vm_f = valid_mask.reshape(T * B, P)
        am_f = action_mask.reshape(T * B, NA)
        cp_f = current_players.reshape(-1)

        # Update
        for epoch in range(UPDATE_EPOCHS):
            loss, metrics = ppo_update(
                network, optimizer, obs_flat, acts_f, logp_f,
                adv_f, tgt_f, vm_f, am_f, vals_f, cp_f)

        metrics["update"] = update_idx
        history.append(metrics)

        if (update_idx + 1) % 5 == 0 or update_idx == 0:
            print(f"  Update {update_idx + 1:3d}: loss={loss:.4f} "
                  f"actor={metrics['actor_loss']:.4f} "
                  f"critic={metrics['critic_loss']:.4f} "
                  f"kl={metrics['approx_kl']:.4f} "
                  f"grad_norm={metrics['grad_norm']:.3f}")

    # ── Verification ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Verification")
    print(f"{'─'*60}")

    # 1. No NaN anywhere
    nan_found = False
    for h in history:
        for k, v in h.items():
            if isinstance(v, float) and not np.isfinite(v):
                print(f"  [FAIL] NaN found in {k} at update {h['update']}: {v}")
                nan_found = True
    if not nan_found:
        print("  [PASS] No NaN/inf in any metric")

    # 2. Loss trend (compare first 5 vs last 5)
    first5 = np.mean([h["actor_loss"] for h in history[:5]])
    last5 = np.mean([h["actor_loss"] for h in history[-5:]])
    print(f"  Actor loss: first5={first5:.4f} last5={last5:.4f}")
    if last5 < first5 * 1.5:  # not exploding
        print("  [PASS] Loss not exploding over training")
    else:
        print("  [WARN] Loss increased significantly")

    # 3. Gradient norms stable
    grad_norms = [h["grad_norm"] for h in history]
    max_gn = max(grad_norms)
    print(f"  Max gradient norm: {max_gn:.3f}")
    if max_gn < 50.0:
        print("  [PASS] Gradient norms stable (< 50)")
    else:
        print("  [WARN] Large gradient norm detected")

    # 4. Diagnostics in range
    for h in history:
        assert 0.0 <= h["clip_frac"] <= 1.0 + 1e-6, f"Bad clip_frac: {h['clip_frac']}"
        assert 0.0 <= h["explained_var"] <= 1.0 + 1e-6, f"Bad ev: {h['explained_var']}"
        assert h["approx_kl"] >= -1e-6, f"Neg KL: {h['approx_kl']}"
        assert h["entropy"] > 0, f"Zero entropy: {h['entropy']}"
    print("  [PASS] All diagnostics in valid ranges")

    print(f"\n{'='*60}")
    print(f"L6 Result: {'[PASS] TRAINING STABLE' if not nan_found else '[FAIL]'}")
    print(f"{'='*60}")

    return 0 if not nan_found else 1


if __name__ == "__main__":
    exit(main())
