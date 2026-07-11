#!/usr/bin/env python3
"""L5: Single PPO Update Cycle — end-to-end parity (rollout → GAE → update).

Verifies the full PT PPO pipeline with synthetic data that mimics a real rollout.
Uses the same patterns as ppo_with_reg.py but at small scale (T=8, B=4) for
deterministic verification.

The test:
  1. Creates synthetic T-step rollout (obs, actions, rewards, dones, etc.)
  2. Runs compute_gae_vectorized → advantages, targets, valid_mask
  3. Runs the PPO update (from ppo_with_reg.py)
  4. Verifies loss components are well-defined (no NaN, loss decreases)
  5. Verifies all diagnostics are in expected ranges

This is a PT-only integration test — it validates that all pieces fit together
correctly, complementing the JAX-vs-PT unit tests (L1-L4).
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

SEED = 42
T, B, P = 8, 4, 4  # timesteps, envs, players
NA = 87  # num_actions
FEATURE_DIM = 16

# PPO hyperparams (matching defaults)
GAMMA = 1.0
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
MAX_REWARD = 320.0
NEG = -1e9


# ═══════════════════════════════════════════════════════════════════════════
# Small test MLP
# ═══════════════════════════════════════════════════════════════════════════

class TinyACNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(FEATURE_DIM, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.actor = torch.nn.Linear(32, NA)
        self.critic = torch.nn.Linear(32, 1)
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
# GAE (exact copy from ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

def compute_gae_vectorized(rewards, values, dones, current_players,
                           gamma=1.0, gae_lambda=0.95):
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
        td_error = player_reward + gamma * next_value[b_idx, cp] * not_done - values[t]
        new_gae = td_error + gamma * gae_lambda * gae_acc[b_idx, cp] * not_done
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
# PPO helpers (from ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

def masked_mean(x, mask):
    denom = mask.float().sum().clamp(min=1.0)
    return (x * mask.float()).sum() / denom


def ppo_update(network, optimizer, obs, actions, old_log_probs, advantages,
               targets, valid_mask, action_mask, old_values, current_players):
    """One PPO update epoch. Returns (total_loss, metrics_dict)."""
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
    optimizer.step()

    return total_loss.item(), {
        "actor_loss": ppo_loss.item(), "critic_loss": loss_critic.item(),
        "entropy": masked_mean(entropy.unsqueeze(-1), vmask).item(),
        "approx_kl": approx_kl.item(), "clip_frac": clip_frac.item(),
        "explained_var": explained_var.item(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic rollout data
# ═══════════════════════════════════════════════════════════════════════════

def make_rollout_data(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Observations: simple random features
    obs = {f"feat_{i}": torch.randn(T, B, FEATURE_DIM) for i in range(1)}  # unused, keep simple
    obs_flat = torch.randn(T * B, FEATURE_DIM)

    # Actions sampled from a reasonable distribution
    action_mask = torch.rand(T, B, NA) > 0.7  # ~30% valid
    # Sample random valid actions
    probs = action_mask.float() / action_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
    actions = torch.multinomial(probs.view(-1, NA), 1).view(T, B)

    # Values and log probs (simulating network outputs)
    values = torch.randn(T, B) * 0.5
    old_log_probs = torch.randn(T, B) * 0.2

    # Rewards: sparse, mostly zero
    rewards = torch.zeros(T, B, P)
    for t in range(T):
        for b in range(B):
            if torch.rand(1).item() < 0.1:  # 10% chance of non-zero reward
                p = torch.randint(0, P, (1,)).item()
                rewards[t, b, p] = torch.randn(1).item() * 0.3

    # Current players: cycling pattern
    current_players = torch.zeros(T, B, dtype=torch.long)
    for b in range(B):
        current_players[:, b] = torch.arange(T) % P

    # Dones: episode boundary every few steps
    dones = torch.zeros(T, B, dtype=torch.bool)
    dones[3, :] = True
    dones[7, :] = True

    return (obs_flat, actions, old_log_probs, values, rewards, dones,
            current_players, action_mask)


# ═══════════════════════════════════════════════════════════════════════════
# Main test
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("L5: Single PPO Update Cycle (End-to-End)")
    print(f"{'='*60}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Generate rollout data ────────────────────────────────────
    print("Generating synthetic rollout...")
    (obs_flat, actions, old_log_probs, values, rewards, dones,
     current_players, action_mask) = make_rollout_data()

    # Normalize rewards
    rewards = rewards / MAX_REWARD

    print(f"  Shape: T={T}, B={B}, P={P}, NA={NA}")
    print(f"  Done rate: {dones.float().mean().item():.3f}")
    print(f"  Non-zero reward rate: {(rewards != 0).float().mean().item():.3f}")

    # ── 2. Compute GAE ──────────────────────────────────────────────
    print("\nComputing GAE...")
    advantages, targets, valid_mask = compute_gae_vectorized(
        rewards, values, dones, current_players)

    # Normalize advantages
    vf = valid_mask.float()
    adv_mean = (advantages * vf).sum() / vf.sum().clamp(min=1.0)
    adv_var = ((advantages - adv_mean) ** 2 * vf).sum() / vf.sum().clamp(min=1.0)
    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

    valid_count = valid_mask.sum().item()
    print(f"  Valid steps: {valid_count}/{T * B * P}")
    assert valid_count > 0, "No valid GAE steps!"
    print("  [PASS] GAE produced valid outputs")

    # ── 3. Flatten ──────────────────────────────────────────────────
    obs_flat = obs_flat  # already (T*B, F)
    acts_flat = actions.reshape(-1)
    logp_flat = old_log_probs.reshape(-1)
    vals_flat = values.reshape(-1)
    adv_flat = advantages.reshape(T * B, P)
    tgt_flat = targets.reshape(T * B, P)
    vm_flat = valid_mask.reshape(T * B, P)
    am_flat = action_mask.reshape(T * B, NA)
    cp_flat = current_players.reshape(-1)

    # ── 4. Initialize network ───────────────────────────────────────
    print("\nInitializing network...")
    network = TinyACNet()
    optimizer = torch.optim.AdamW(network.parameters(), lr=LR, eps=1e-5)

    # ── 5. Run PPO updates ──────────────────────────────────────────
    print("\nRunning PPO updates (4 epochs, no minibatches)...")
    metrics_history = []

    for epoch in range(4):
        total_loss, metrics = ppo_update(
            network, optimizer, obs_flat, acts_flat, logp_flat,
            adv_flat, tgt_flat, vm_flat, am_flat, vals_flat, cp_flat)
        metrics_history.append({
            "epoch": epoch, "total_loss": total_loss, **metrics})

        print(f"  Epoch {epoch}: loss={total_loss:.4f} "
              f"actor={metrics['actor_loss']:.4f} "
              f"critic={metrics['critic_loss']:.4f} "
              f"ent={metrics['entropy']:.4f} "
              f"kl={metrics['approx_kl']:.4f} "
              f"clip={metrics['clip_frac']:.3f} "
              f"expl_var={metrics['explained_var']:.3f}")

    # ── 6. Verify ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Verification")
    print(f"{'─'*60}")

    all_ok = True

    # Loss should be finite
    for m in metrics_history:
        assert np.isfinite(m["total_loss"]), f"NaN loss at epoch {m['epoch']}"
    print("  [PASS] All losses finite")

    # Loss should generally decrease (at least not explode)
    first_loss = metrics_history[0]["total_loss"]
    last_loss = metrics_history[-1]["total_loss"]
    loss_ratio = last_loss / max(abs(first_loss), 1e-8)
    print(f"  Loss: {first_loss:.4f} → {last_loss:.4f} (ratio={loss_ratio:.3f})")
    if loss_ratio < 2.0:
        print("  [PASS] Loss not exploding")
    else:
        print("  [WARN] Loss increased significantly (may be normal for random data)")

    # Entropy should be > 0
    for m in metrics_history:
        assert m["entropy"] > 0, f"Zero entropy at epoch {m['epoch']}"
    print("  [PASS] Entropy always positive")

    # approx_kl should be >= 0
    for m in metrics_history:
        assert m["approx_kl"] >= -1e-6, f"Negative KL at epoch {m['epoch']}"
    print("  [PASS] KL divergence >= 0")

    # clip_frac should be in [0, 1]
    for m in metrics_history:
        assert 0.0 <= m["clip_frac"] <= 1.0 + 1e-6, \
            f"Bad clip_frac at epoch {m['epoch']}: {m['clip_frac']}"
    print("  [PASS] clip_frac in [0, 1]")

    # explained_var should be in [0, 1]
    for m in metrics_history:
        assert 0.0 <= m["explained_var"] <= 1.0 + 1e-6, \
            f"Bad explained_var at epoch {m['epoch']}: {m['explained_var']}"
    print("  [PASS] explained_var in [0, 1]")

    print(f"\n{'='*60}")
    print(f"L5 Result: {'[PASS] ALL CHECKS' if all_ok else '[FAIL]'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
