#!/usr/bin/env python3
"""L4: PPO Update Parity — JAX vs PyTorch loss, gradient, and parameter update.

The core missing verification layer.  Proves that identical batch data + identical
network outputs produce identical PPO loss values and gradients.

Strategy:
  1. Use a simple shared MLP (not full ACNet) initialized with identical weights
     in both frameworks to guarantee forward-pass equivalence.
  2. Feed the same batch data through both → identical logits & values.
  3. Run the full PPO loss computation (actor, critic, entropy, diagnostics).
  4. Compare: losses (<1e-4), metrics (<1e-4), gradients (<1e-4 per-layer).

This isolates the PPO math from the complex ACNet weight-transfer problem
(which is verified separately in L3 / test_full_ppo_parity.py).
"""

import sys
import numpy as np
import jax, jax.numpy as jnp
import torch, torch.nn.functional as F

SEED = 42
BATCH = 64
NUM_PLAYERS = 4
NUM_ACTIONS = 87
FEATURE_DIM = 32  # Use a small feature dim for the shared MLP

# PPO hyperparams
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
NEG = -1e9


# ═══════════════════════════════════════════════════════════════════════════
# Shared MLP — identical architecture in JAX and PT
# ═══════════════════════════════════════════════════════════════════════════

class JaxMLP:
    """Simple JAX MLP with explicit params (no Flax)."""
    def __init__(self, rng, in_dim, hidden_dim, out_dim):
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
        # jax.random.orthogonal(key, n, shape=(), m=None) → (n, m) with optional shape
        # JAX Linear: W is (in_dim, out_dim); PT Linear: W is (out_dim, in_dim)
        self.W1 = jax.random.orthogonal(k1, in_dim, m=hidden_dim)
        self.b1 = jnp.zeros(hidden_dim)
        self.W2 = jax.random.orthogonal(k2, hidden_dim, m=hidden_dim)
        self.b2 = jnp.zeros(hidden_dim)
        self.W3 = jax.random.orthogonal(k3, hidden_dim, m=out_dim) * 0.01
        self.b3 = jnp.zeros(out_dim)
        self.W4 = jax.random.orthogonal(k4, in_dim, m=hidden_dim)
        self.b4 = jnp.zeros(hidden_dim)
        self.W5 = jax.random.orthogonal(k5, hidden_dim, m=hidden_dim)
        self.b5 = jnp.zeros(hidden_dim)
        self.W6 = jax.random.orthogonal(k6, hidden_dim, m=1)
        self.b6 = jnp.zeros(1)

    def __call__(self, x):
        # Policy head
        h = jnp.tanh(x @ self.W1 + self.b1)
        h = jnp.tanh(h @ self.W2 + self.b2)
        logits = h @ self.W3 + self.b3
        # Value head
        h2 = jnp.tanh(x @ self.W4 + self.b4)
        h2 = jnp.tanh(h2 @ self.W5 + self.b5)
        value = (h2 @ self.W6 + self.b6).squeeze(-1)
        return logits, value

    def params_list(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3,
                self.W4, self.b4, self.W5, self.b5, self.W6, self.b6]


class PTMLP(torch.nn.Module):
    """PyTorch MLP with identical architecture."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.actor = torch.nn.Linear(hidden_dim, out_dim)
        self.fc4 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.critic = torch.nn.Linear(hidden_dim, 1)
        # Init all to zero bias
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
# JAX PPO helpers
# ═══════════════════════════════════════════════════════════════════════════

def jax_masked_mean(x, mask):
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(
        mask.astype(jnp.float32).sum(), 1.0)


def jax_ppo_loss_fn(jax_mlp, x, actions, old_log_probs, advantages,
                    targets, valid_mask, action_mask, old_values, current_players):
    """PPO loss matching examples/ppo_with_reg.py."""
    import distrax
    logits, values = jax_mlp(x)
    logits = jnp.where(action_mask, logits, NEG)

    dist = distrax.Categorical(logits=logits)
    log_ratio = dist.log_prob(actions) - old_log_probs
    ratio = jnp.exp(log_ratio)[..., None]

    adv = jnp.take_along_axis(advantages, current_players[..., None], axis=1)
    mask = jnp.take_along_axis(valid_mask.astype(jnp.float32),
                               current_players[..., None], axis=1)

    ppo_loss = -jax_masked_mean(
        jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv),
        mask)
    entropy = jax_masked_mean(dist.entropy()[..., None], mask)

    value_clipped = old_values[..., None] + jnp.clip(
        values[..., None] - old_values[..., None], -CLIP_EPS, CLIP_EPS)
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

    total_loss = ppo_loss - ENT_COEF * entropy + loss_critic
    return total_loss, {
        "total_loss": total_loss, "actor_loss": ppo_loss,
        "critic_loss": loss_critic, "entropy": entropy,
        "approx_kl": approx_kl, "clip_frac": clip_frac,
        "explained_var": explained_var,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch PPO helpers
# ═══════════════════════════════════════════════════════════════════════════

def pt_masked_mean(x, mask):
    denom = mask.float().sum().clamp(min=1.0)
    return (x * mask.float()).sum() / denom


def pt_ppo_step(pt_mlp, x, actions, old_log_probs, advantages, targets,
                valid_mask, action_mask, old_values, current_players):
    """PPO step matching ppo_with_reg.py."""
    logits, values_new = pt_mlp(x)
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

    total_loss = (ppo_loss - ENT_COEF * pt_masked_mean(entropy.unsqueeze(-1), vmask)
                  + loss_critic)

    metrics = {
        "total_loss": total_loss, "actor_loss": ppo_loss,
        "critic_loss": loss_critic,
        "entropy": pt_masked_mean(entropy.unsqueeze(-1), vmask),
        "approx_kl": approx_kl, "clip_frac": clip_frac,
        "explained_var": explained_var,
    }
    return total_loss, metrics


# ═══════════════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════════════

def make_batch_data(batch_size=BATCH, seed=SEED):
    np.random.seed(seed)
    x = np.random.randn(batch_size, FEATURE_DIM).astype(np.float32) * 0.5

    actions = np.random.randint(0, NUM_ACTIONS, size=batch_size).astype(np.int32)
    old_log_probs = np.random.randn(batch_size).astype(np.float32) * 0.2
    old_values = np.random.randn(batch_size).astype(np.float32) * 0.5

    current_players = np.random.randint(0, NUM_PLAYERS, size=batch_size).astype(np.int32)
    advantages = np.zeros((batch_size, NUM_PLAYERS), dtype=np.float32)
    targets = np.zeros((batch_size, NUM_PLAYERS), dtype=np.float32)
    valid_mask = np.zeros((batch_size, NUM_PLAYERS), dtype=bool)
    for b in range(batch_size):
        p = current_players[b]
        advantages[b, p] = np.float32(np.random.randn() * 0.5)
        targets[b, p] = old_values[b] + advantages[b, p]
        valid_mask[b, p] = True

    action_mask = np.random.rand(batch_size, NUM_ACTIONS).astype(np.float32) > 0.7
    return x, actions, old_log_probs, old_values, current_players, \
        advantages, targets, valid_mask, action_mask


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("L4: PPO Update Parity (Loss + Gradient + Parameters)")
    print(f"{'='*60}\n")

    # ── 1. Generate data ────────────────────────────────────────────
    print("Generating batch data...")
    (x, actions, old_log_probs, old_values, current_players,
     advantages, targets, valid_mask, action_mask) = make_batch_data()

    # ── 2. Initialize MLPs ──────────────────────────────────────────
    print("Initializing MLPs with identical random weights...")
    rng = jax.random.PRNGKey(SEED)
    jax_mlp = JaxMLP(rng, FEATURE_DIM, 64, NUM_ACTIONS)

    pt_mlp = PTMLP(FEATURE_DIM, 64, NUM_ACTIONS)
    # Copy JAX weights → PT (JAX W: [in,out], PT Linear.weight: [out,in])
    jax_params = jax_mlp.params_list()
    pt_params = list(pt_mlp.parameters())
    with torch.no_grad():
        for i, (jp, pp) in enumerate(zip(jax_params, pt_params)):
            jv = np.array(jp)
            if jv.ndim == 2:
                pp.data.copy_(torch.from_numpy(jv.T))  # transpose
            else:
                pp.data.copy_(torch.from_numpy(jv))

    # ── 3. Verify forward pass ──────────────────────────────────────
    jx = jnp.asarray(x)
    px = torch.from_numpy(x)
    jl, jv = jax_mlp(jx)
    pt_mlp.eval()
    with torch.no_grad():
        pl, pv = pt_mlp(px)

    ldiff = float(np.abs(np.array(jl) - pl.numpy()).max())
    vdiff = float(np.abs(np.array(jv) - pv.numpy()).max())
    print(f"  Forward: logit_diff={ldiff:.2e}  value_diff={vdiff:.2e}")
    assert ldiff < 1e-5, f"Forward logit mismatch: {ldiff}"
    assert vdiff < 1e-5, f"Forward value mismatch: {vdiff}"
    print("  [PASS] Forward pass IDENTICAL\n")

    # ── 4. JAX PPO loss + gradients ─────────────────────────────────
    print("Computing JAX PPO loss + gradients...")
    jax_a = jnp.asarray(actions)
    jax_olp = jnp.asarray(old_log_probs)
    jax_ov = jnp.asarray(old_values)
    jax_cp = jnp.asarray(current_players)
    jax_adv = jnp.asarray(advantages)
    jax_tgt = jnp.asarray(targets)
    jax_vm = jnp.asarray(valid_mask)
    jax_am = jnp.asarray(action_mask)

    jax_loss, jax_metrics = jax_ppo_loss_fn(
        jax_mlp, jx, jax_a, jax_olp, jax_adv,
        jax_tgt, jax_vm, jax_am, jax_ov, jax_cp)

    # JAX gradient: compute gradient w.r.t. all 12 params
    def jax_total_loss(params_list):
        mlp = JaxMLP.__new__(JaxMLP)
        mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3, \
            mlp.W4, mlp.b4, mlp.W5, mlp.b5, mlp.W6, mlp.b6 = params_list
        loss, _ = jax_ppo_loss_fn(
            mlp, jx, jax_a, jax_olp, jax_adv,
            jax_tgt, jax_vm, jax_am, jax_ov, jax_cp)
        return loss

    jax_grads = jax.grad(jax_total_loss)(jax_params)

    # ── 5. PT PPO loss + gradients ──────────────────────────────────
    print("Computing PyTorch PPO loss + gradients...")
    pt_mlp.train()
    pt_a = torch.from_numpy(actions).long()
    pt_olp = torch.from_numpy(old_log_probs)
    pt_ov = torch.from_numpy(old_values)
    pt_cp = torch.from_numpy(current_players).long()
    pt_adv = torch.from_numpy(advantages)
    pt_tgt = torch.from_numpy(targets)
    pt_vm = torch.from_numpy(valid_mask)
    pt_am = torch.from_numpy(action_mask)

    pt_loss, pt_metrics = pt_ppo_step(
        pt_mlp, px, pt_a, pt_olp, pt_adv, pt_tgt,
        pt_vm, pt_am, pt_ov, pt_cp)

    pt_mlp.zero_grad()
    pt_loss.backward()

    # ── 6. Compare metrics ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Metric Comparison")
    print(f"{'─'*60}")
    print(f"  {'Metric':<20} {'JAX':>12} {'PT':>12} {'Diff':>12} {'Status':>8}")

    all_ok = True
    metric_keys = ["total_loss", "actor_loss", "critic_loss", "entropy",
                   "approx_kl", "clip_frac", "explained_var"]

    for key in metric_keys:
        jv = float(jax_metrics[key])
        pv = float(pt_metrics[key].item() if hasattr(pt_metrics[key], 'item')
                   else pt_metrics[key])
        diff = abs(jv - pv)
        rel = diff / max(abs(jv), 1e-8)
        ok = rel < 1e-4
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  {key:<18} {jv:12.8f} {pv:12.8f} {diff:12.2e} {status:>8}")

    # ── 7. Compare gradients ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Gradient Comparison")
    print(f"{'─'*60}")

    pt_params_list = list(pt_mlp.parameters())
    grad_diffs = []
    for i, (jg, pp) in enumerate(zip(jax_grads, pt_params_list)):
        jg_np = np.array(jg)
        pg = pp.grad.detach().numpy() if pp.grad is not None else np.zeros_like(jg_np)
        if jg_np.ndim == 2:
            jg_np = jg_np.T  # transpose to match PT layout
        d = np.abs(jg_np - pg).max()
        grad_diffs.append(float(d))

    max_gd = max(grad_diffs)
    mean_gd = float(np.mean(grad_diffs))
    grad_ok = max_gd < 1e-4
    if not grad_ok:
        all_ok = False

    print(f"  Compared {len(grad_diffs)} parameters")
    print(f"  Max gradient diff:  {max_gd:.2e}")
    print(f"  Mean gradient diff: {mean_gd:.2e}")
    for i, d in enumerate(grad_diffs):
        label = ["W1","b1","W2","b2","W3(actor)","b3",
                 "W4","b4","W5","b5","W6(critic)","b6"][i]
        p = pt_params_list[i]
        ok = "PASS" if d < 1e-4 else "FAIL"
        print(f"    [{i}] {label:12s} shape={str(tuple(p.shape)):20s}  diff={d:.2e}  {ok}")

    print(f"  {'[PASS]' if grad_ok else '[FAIL]'} Gradient check")

    # ── 8. Compare optimizer step ────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Optimizer Step Comparison (JAX optax.adamw vs PT AdamW)")
    print(f"{'─'*60}")

    import optax
    # JAX: package params into a single array for optax
    jax_opt = optax.adamw(learning_rate=3e-4, eps=1e-5)
    jax_opt_state = jax_opt.init(jax_params)
    jax_updates, _ = jax_opt.update(jax_grads, jax_opt_state, jax_params)
    jax_new_params = optax.apply_updates(jax_params, jax_updates)

    # PT: re-run to ensure fresh grads
    pt_loss2, _ = pt_ppo_step(
        pt_mlp, px, pt_a, pt_olp, pt_adv, pt_tgt,
        pt_vm, pt_am, pt_ov, pt_cp)
    pt_opt = torch.optim.AdamW(pt_mlp.parameters(), lr=3e-4, eps=1e-5)
    pt_mlp.zero_grad()
    pt_loss2.backward()
    pt_opt.step()

    param_diffs = []
    pt_new_list = list(pt_mlp.parameters())
    for i, (jnp_p, pp) in enumerate(zip(jax_new_params, pt_new_list)):
        jv = np.array(jnp_p)
        pv = pp.detach().numpy()
        if jv.ndim == 2:
            jv = jv.T
        d = np.abs(jv - pv).max()
        param_diffs.append(float(d))

    max_pd = max(param_diffs) if param_diffs else 999.0
    mean_pd = float(np.mean(param_diffs)) if param_diffs else 999.0
    param_ok = max_pd < 1e-4
    if not param_ok:
        all_ok = False

    print(f"  Max parameter diff after 1 step:  {max_pd:.2e}")
    print(f"  Mean parameter diff after 1 step: {mean_pd:.2e}")
    for i, d in enumerate(param_diffs):
        label = ["W1","b1","W2","b2","W3(actor)","b3",
                 "W4","b4","W5","b5","W6(critic)","b6"][i]
        p = pt_new_list[i]
        ok = "PASS" if d < 1e-4 else "FAIL"
        print(f"    [{i}] {label:12s} shape={str(tuple(p.shape)):20s}  diff={d:.2e}  {ok}")

    print(f"  {'[PASS]' if param_ok else '[FAIL]'} Parameter update check")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"L4 Result: {'[PASS] ALL IDENTICAL' if all_ok else '[FAIL] See diffs above'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
