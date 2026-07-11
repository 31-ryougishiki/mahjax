#!/usr/bin/env python3
"""Plot 30-step PPO loss curves: JAX vs PT comparison.

Loads golden data, computes PPO loss at each update's first minibatch
for both JAX and PT, and plots the loss curves side by side.
"""

import pickle, os, sys
import numpy as np
import jax, jax.numpy as jnp
import torch
import distrax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Config (matching recording)
# ═══════════════════════════════════════════════════════════════════════════
CLIP_EPS = 0.2; ENT_COEF = 0.01; VF_COEF = 0.5; LR = 3e-4
NEG = -1e9; HIDDEN_DIM = 64
OBS_DIM = 631; NUM_ACTIONS = 87

# ═══════════════════════════════════════════════════════════════════════════
# JAX MLP (matching recording)
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
# PT MLP
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
# Loss helpers
# ═══════════════════════════════════════════════════════════════════════════

def jax_masked_mean(x, mask):
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(
        mask.astype(jnp.float32).sum(), 1.0)

def pt_masked_mean(x, mask):
    return (x * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

def jax_ppo_loss(mlp, x, actions, old_log_probs, advantages, targets,
                 valid_mask, action_mask, old_values, current_players):
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
    ov_ = old_values
    value_clipped = ov_[..., None] + jnp.clip(
        values[..., None] - ov_[..., None], -CLIP_EPS, CLIP_EPS)
    tgt = jnp.take_along_axis(targets, current_players[..., None], axis=1)
    loss_critic = (0.5 * VF_COEF *
                   jax_masked_mean(jnp.maximum((values[..., None] - tgt) ** 2,
                                               (value_clipped - tgt) ** 2), mask))
    approx_kl = jax_masked_mean((ratio - 1.0) - log_ratio[..., None], mask)
    clip_frac = jax_masked_mean(
        (jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32), mask)
    explained_var = jnp.maximum(
        1.0 - jax_masked_mean((tgt - values[..., None]) ** 2, mask) /
        (jax_masked_mean((tgt - jax_masked_mean(tgt, mask)) ** 2, mask) + 1e-8), 0.0)
    total = ppo_loss - ENT_COEF * entropy + loss_critic
    return total, {"actor": ppo_loss, "critic": loss_critic, "entropy": entropy,
                   "approx_kl": approx_kl, "clip_frac": clip_frac,
                   "explained_var": explained_var}


def pt_ppo_loss(mlp, x, actions, old_log_probs, advantages, targets,
                valid_mask, action_mask, old_values, current_players):
    logits, values_new = mlp(x)
    logits = torch.where(action_mask, logits, torch.full_like(logits, NEG))
    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(actions)
    ent = dist.entropy()
    log_ratio = logp_new - old_log_probs
    ratio = torch.exp(log_ratio).unsqueeze(-1)
    adv = advantages.gather(1, current_players.unsqueeze(-1))
    vmask = valid_mask.float().gather(1, current_players.unsqueeze(-1))
    clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    ppo_loss = -pt_masked_mean(torch.min(ratio * adv, clip_adv), vmask)
    entropy = pt_masked_mean(ent.unsqueeze(-1), vmask)
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
    total = ppo_loss - ENT_COEF * entropy + loss_critic
    return total, {"actor": ppo_loss, "critic": loss_critic, "entropy": entropy,
                   "approx_kl": approx_kl, "clip_frac": clip_frac,
                   "explained_var": explained_var}


def load_params_to_jax(mlp, params_np):
    (mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3,
     mlp.W4, mlp.b4, mlp.W5, mlp.b5, mlp.W6, mlp.b6) = [jnp.asarray(p) for p in params_np]

def load_params_to_pt(mlp, params_np):
    pt_list = list(mlp.parameters())
    with torch.no_grad():
        for jp, pp in zip(params_np, pt_list):
            if jp.ndim == 2: pp.data.copy_(torch.from_numpy(jp.T))
            else: pp.data.copy_(torch.from_numpy(jp))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    data_path = os.path.join(os.path.dirname(__file__),
                             "golden_data", "ppo_30updates.pkl")
    with open(data_path, "rb") as f:
        golden = pickle.load(f)

    # Init networks
    rng = jax.random.PRNGKey(golden["config"]["seed"])
    jax_mlp = JaxObsMLP(rng)
    pt_mlp = PTObsMLP()

    # Collect losses
    jax_losses = []; pt_losses = []
    jax_mets = []; pt_mets = []
    param_diffs = []

    for update_idx, upd in enumerate(golden["updates"]):
        # Set JAX params (pre-update)
        load_params_to_jax(jax_mlp, upd["params_before"])
        # Set PT params (pre-update, from JAX)
        load_params_to_pt(pt_mlp, upd["params_before"])

        # Get first minibatch data
        flat = upd["flattened"]
        mb = upd["minibatches"][0]
        perm = mb["perm"]

        # JAX forward + loss
        jx_mb = jnp.asarray(flat["obs_flat"][perm])
        ja_mb = jnp.asarray(flat["actions"][perm])
        jlp_mb = jnp.asarray(flat["log_probs"][perm])
        jv_mb = jnp.asarray(flat["values"][perm])
        jadv_mb = jnp.asarray(flat["advantages_norm"][perm])
        jtgt_mb = jnp.asarray(flat["targets"][perm])
        jvm_mb = jnp.asarray(flat["valid_mask"][perm])
        jam_mb = jnp.asarray(flat["action_mask"][perm])
        jcp_mb = jnp.asarray(flat["current_players"][perm])

        j_loss, j_met = jax_ppo_loss(
            jax_mlp, jx_mb, ja_mb, jlp_mb, jadv_mb, jtgt_mb,
            jvm_mb, jam_mb, jv_mb, jcp_mb)

        # PT forward + loss
        pt_mlp.eval()
        with torch.no_grad():
            px_mb = torch.from_numpy(flat["obs_flat"][perm])
            pa_mb = torch.from_numpy(flat["actions"][perm]).long()
            plp_mb = torch.from_numpy(flat["log_probs"][perm])
            pv_mb = torch.from_numpy(flat["values"][perm])
            padv_mb = torch.from_numpy(flat["advantages_norm"][perm])
            ptgt_mb = torch.from_numpy(flat["targets"][perm])
            pvm_mb = torch.from_numpy(flat["valid_mask"][perm])
            pam_mb = torch.from_numpy(flat["action_mask"][perm])
            pcp_mb = torch.from_numpy(flat["current_players"][perm]).long()

            p_loss, p_met = pt_ppo_loss(
                pt_mlp, px_mb, pa_mb, plp_mb, padv_mb, ptgt_mb,
                pvm_mb, pam_mb, pv_mb, pcp_mb)

        jax_losses.append(float(j_loss))
        pt_losses.append(float(p_loss))
        jax_mets.append({k: float(v) for k, v in j_met.items()})
        pt_mets.append({k: float(v.item() if hasattr(v, 'item') else v) for k, v in p_met.items()})

        # Param diff
        jp_after = upd["params_after"]
        pt_list = list(pt_mlp.parameters())
        # Load PT with JAX post-update params to compare
        load_params_to_pt(pt_mlp, jp_after)
        # Actually, we want JAX_post - PT_post diff. Let's use the replay results.
        # For now, compute from the replay script's perspective
        # Actually just skip this, we already have the diff from earlier

    # ── Print raw numbers ──
    print(f"{'Update':>6s}  {'JAX loss':>10s}  {'PT loss':>10s}  {'Diff':>10s}")
    for i in range(len(jax_losses)):
        diff = jax_losses[i] - pt_losses[i]
        print(f"{i+1:6d}  {jax_losses[i]:10.6f}  {pt_losses[i]:10.6f}  {diff:10.2e}")

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    updates = np.arange(1, len(jax_losses) + 1)

    # Row 1: main losses
    ax = axes[0, 0]
    ax.plot(updates, jax_losses, 'b-o', label='JAX', markersize=4)
    ax.plot(updates, pt_losses, 'r--s', label='PT', markersize=4)
    ax.set_title('Total PPO Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # Loss diff
    ax = axes[0, 1]
    loss_diffs = [abs(j - p) for j, p in zip(jax_losses, pt_losses)]
    ax.semilogy(updates, loss_diffs, 'k-o', markersize=4)
    ax.set_title('|JAX loss - PT loss|', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update'); ax.set_ylabel('Absolute diff')
    ax.grid(True, alpha=0.3)

    # Actor + Critic
    ax = axes[0, 2]
    ax.plot(updates, [m['actor'] for m in jax_mets], 'b-o', label='JAX actor', markersize=3)
    ax.plot(updates, [m['actor'] for m in pt_mets], 'r--s', label='PT actor', markersize=3)
    ax.plot(updates, [m['critic'] for m in jax_mets], 'b-^', label='JAX critic', markersize=3)
    ax.plot(updates, [m['critic'] for m in pt_mets], 'r--v', label='PT critic', markersize=3)
    ax.set_title('Actor & Critic Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 2: diagnostics
    ax = axes[1, 0]
    ax.plot(updates, [m['entropy'] for m in jax_mets], 'b-o', label='JAX', markersize=3)
    ax.plot(updates, [m['entropy'] for m in pt_mets], 'r--s', label='PT', markersize=3)
    ax.set_title('Entropy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(updates, [m['approx_kl'] for m in jax_mets], 'b-o', label='JAX', markersize=3)
    ax.plot(updates, [m['approx_kl'] for m in pt_mets], 'r--s', label='PT', markersize=3)
    ax.set_title('Approx KL Divergence', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(updates, [m['clip_frac'] for m in jax_mets], 'b-o', label='JAX', markersize=3)
    ax.plot(updates, [m['clip_frac'] for m in pt_mets], 'r--s', label='PT', markersize=3)
    ax.set_title('Clip Fraction', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update'); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle('PPO Training Loss Curves: JAX vs PyTorch (30 updates, real mahjong env, B=2, T=8)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "ppo_loss_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close()

    return 0


if __name__ == "__main__":
    main()
