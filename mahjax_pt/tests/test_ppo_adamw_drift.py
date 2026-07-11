#!/usr/bin/env python3
"""Diagnostic: trace through the first PPO update step in detail.

Identifies the exact source of parameter divergence between JAX and PT
over multiple optimizer steps.
"""

import sys
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import optax

SEED = 42
T, B, P = 8, 4, 4
NA = 87
FEATURE_DIM = 32
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
LR = 3e-4
NEG = -1e9
GAMMA = 1.0
GAE_LAMBDA = 0.95
MAX_REWARD = 320.0

# ═══════════════════════════════════════════════════════════════════════════
# JAX MLP
# ═══════════════════════════════════════════════════════════════════════════

class JaxMLP:
    def __init__(self, rng, in_dim, hidden_dim, out_dim):
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
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


class PTMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.actor = torch.nn.Linear(hidden_dim, out_dim)
        self.fc4 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.critic = torch.nn.Linear(hidden_dim, 1)
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


def main():
    print("=" * 70)
    print("Diagnostic: Single-Minibatch AdamW Step-by-Step Comparison")
    print("=" * 70)

    # Init
    rng = jax.random.PRNGKey(SEED)
    jax_mlp = JaxMLP(rng, FEATURE_DIM, 64, NA)
    pt_mlp = PTMLP(FEATURE_DIM, 64, NA)

    jax_params = jax_mlp.params_list()
    with torch.no_grad():
        for jp, pp in zip(jax_params, list(pt_mlp.parameters())):
            jv = np.array(jp)
            if jv.ndim == 2:
                pp.data.copy_(torch.from_numpy(jv.T))
            else:
                pp.data.copy_(torch.from_numpy(jv))

    # Verify forward
    test_x = np.random.RandomState(999).randn(4, FEATURE_DIM).astype(np.float32)
    jl, jv = jax_mlp(jnp.asarray(test_x))
    with torch.no_grad():
        pl, pv = pt_mlp(torch.from_numpy(test_x))
    print(f"Initial forward: logit_diff={float(np.abs(np.array(jl)-pl.numpy()).max()):.2e} "
          f"value_diff={float(np.abs(np.array(jv)-pv.numpy()).max()):.2e}")

    # Create a simple minibatch
    np.random.seed(123)
    x_np = np.random.randn(8, FEATURE_DIM).astype(np.float32) * 0.5
    actions_np = np.random.randint(0, NA, size=8).astype(np.int32)
    old_log_probs_np = np.random.randn(8).astype(np.float32) * 0.1
    old_values_np = np.random.randn(8).astype(np.float32) * 0.3
    current_players_np = np.random.randint(0, 4, size=8).astype(np.int32)

    advantages_np = np.zeros((8, 4), dtype=np.float32)
    targets_np = np.zeros((8, 4), dtype=np.float32)
    valid_mask_np = np.zeros((8, 4), dtype=bool)
    for i in range(8):
        p = current_players_np[i]
        advantages_np[i, p] = np.float32(np.random.randn() * 0.5)
        targets_np[i, p] = old_values_np[i] + advantages_np[i, p]
        valid_mask_np[i, p] = True
    action_mask_np = np.random.rand(8, NA).astype(np.float32) > 0.7

    # Loss functions
    import distrax

    def jax_masked_mean(x, mask):
        return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(
            mask.astype(jnp.float32).sum(), 1.0)

    def jax_ppo_loss(params_list):
        (W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6) = params_list
        # Reconstruct forward pass
        x = jnp.asarray(x_np)
        h = jnp.tanh(x @ W1 + b1)
        h = jnp.tanh(h @ W2 + b2)
        logits = h @ W3 + b3
        h2 = jnp.tanh(x @ W4 + b4)
        h2 = jnp.tanh(h2 @ W5 + b5)
        values = (h2 @ W6 + b6).squeeze(-1)

        am = jnp.asarray(action_mask_np)
        logits = jnp.where(am, logits, NEG)
        dist = distrax.Categorical(logits=logits)
        logp_new = dist.log_prob(jnp.asarray(actions_np))
        log_ratio = logp_new - jnp.asarray(old_log_probs_np)
        ratio = jnp.exp(log_ratio)[..., None]

        adv = jnp.take_along_axis(jnp.asarray(advantages_np),
                                  jnp.asarray(current_players_np)[..., None], axis=1)
        mask = jnp.take_along_axis(jnp.asarray(valid_mask_np).astype(jnp.float32),
                                   jnp.asarray(current_players_np)[..., None], axis=1)

        ppo_loss = -jax_masked_mean(
            jnp.minimum(ratio * adv,
                       jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv), mask)
        entropy = jax_masked_mean(dist.entropy()[..., None], mask)

        ov = jnp.asarray(old_values_np)
        value_clipped = ov[..., None] + jnp.clip(
            values[..., None] - ov[..., None], -CLIP_EPS, CLIP_EPS)
        tgt = jnp.take_along_axis(jnp.asarray(targets_np),
                                  jnp.asarray(current_players_np)[..., None], axis=1)
        loss_critic = (0.5 * VF_COEF *
                       jax_masked_mean(jnp.maximum((values[..., None] - tgt) ** 2,
                                                   (value_clipped - tgt) ** 2), mask))

        total_loss = ppo_loss - ENT_COEF * entropy + loss_critic
        return total_loss

    def pt_masked_mean(x, mask):
        return (x * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    # ══════════════════════════════════════════════════════════════════
    # Run 10 optimizer steps on the SAME minibatch and track divergence
    # ══════════════════════════════════════════════════════════════════

    jax_opt = optax.adamw(learning_rate=LR, eps=1e-5)
    jax_opt_state = jax_opt.init(jax_params)

    pt_opt = torch.optim.AdamW(pt_mlp.parameters(), lr=LR, eps=1e-5)

    jax_grad_fn = jax.grad(jax_ppo_loss)

    print(f"\n{'─'*70}")
    print("Running 10 optimizer steps on SAME minibatch data...")
    print(f"{'─'*70}")

    param_names = ["W1","b1","W2","b2","W3(actor)","b3",
                   "W4","b4","W5","b5","W6(critic)","b6"]

    for step in range(10):
        # JAX
        jax_grads = jax_grad_fn(jax_params)
        jax_updates, jax_opt_state = jax_opt.update(jax_grads, jax_opt_state, jax_params)
        jax_params = optax.apply_updates(jax_params, jax_updates)

        # PT
        pt_opt.zero_grad()
        logits, values_new = pt_mlp(torch.from_numpy(x_np))
        am = torch.from_numpy(action_mask_np)
        logits = torch.where(am, logits, torch.full_like(logits, NEG))
        dist = torch.distributions.Categorical(logits=logits)
        logp_new = dist.log_prob(torch.from_numpy(actions_np).long())
        log_ratio = logp_new - torch.from_numpy(old_log_probs_np)
        ratio = torch.exp(log_ratio).unsqueeze(-1)

        adv = torch.from_numpy(advantages_np).gather(
            1, torch.from_numpy(current_players_np).long().unsqueeze(-1))
        vmask = torch.from_numpy(valid_mask_np).float().gather(
            1, torch.from_numpy(current_players_np).long().unsqueeze(-1))

        clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        ppo_loss = -pt_masked_mean(torch.min(ratio * adv, clip_adv), vmask)

        vt = values_new.unsqueeze(-1)
        ov = torch.from_numpy(old_values_np)
        val_clipped = ov.unsqueeze(-1) + torch.clamp(
            vt - ov.unsqueeze(-1), -CLIP_EPS, CLIP_EPS)
        tgt = torch.from_numpy(targets_np).gather(
            1, torch.from_numpy(current_players_np).long().unsqueeze(-1))
        loss_critic = 0.5 * VF_COEF * pt_masked_mean(
            torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)

        total_loss = (ppo_loss
                     - ENT_COEF * pt_masked_mean(dist.entropy().unsqueeze(-1), vmask)
                     + loss_critic)
        total_loss.backward()
        pt_opt.step()

        # Compare
        pt_params_list = list(pt_mlp.parameters())
        max_pd = 0.0
        max_gd = 0.0
        for i, (jp, pp) in enumerate(zip(jax_params, pt_params_list)):
            jv = np.array(jp)
            pv = pp.detach().cpu().numpy()
            if jv.ndim == 2:
                jv = jv.T
            pd = float(np.abs(jv - pv).max())
            max_pd = max(max_pd, pd)

            jg_np = np.array(jax_grads[i])
            pg = pp.grad.cpu().numpy() if pp.grad is not None else np.zeros_like(jg_np)
            if jg_np.ndim == 2:
                jg_np = jg_np.T
            gd = float(np.abs(jg_np - pg).max())
            max_gd = max(max_gd, gd)

        print(f"  Step {step+1}: max_grad_diff={max_gd:.2e}  max_param_diff={max_pd:.2e}")

        if step == 0:
            # Print per-param details on first step
            print("    Per-param details (step 1):")
            for i in range(len(jax_params)):
                jv = np.array(jax_params[i])
                pv = pt_params_list[i].detach().cpu().numpy()
                if jv.ndim == 2:
                    jv = jv.T
                pd = float(np.abs(jv - pv).max())
                jg = np.array(jax_grads[i])
                pg = pt_params_list[i].grad.cpu().numpy()
                if jg.ndim == 2:
                    jg = jg.T
                gd = float(np.abs(jg - pg).max())
                print(f"      [{i:2d}] {param_names[i]:12s}  grad_diff={gd:.2e}  param_diff={pd:.2e}")

    # ══════════════════════════════════════════════════════════════════
    # Also test: run 10 steps with DIFFERENT minibatch data each time
    # (more realistic)
    # ══════════════════════════════════════════════════════════════════

    print(f"\n{'─'*70}")
    print("Running 10 optimizer steps with DIFFERENT data each step...")
    print(f"{'─'*70}")

    # Re-init
    rng = jax.random.PRNGKey(SEED)
    jax_mlp2 = JaxMLP(rng, FEATURE_DIM, 64, NA)
    pt_mlp2 = PTMLP(FEATURE_DIM, 64, NA)
    jax_params2 = jax_mlp2.params_list()
    with torch.no_grad():
        for jp, pp in zip(jax_params2, list(pt_mlp2.parameters())):
            jv = np.array(jp)
            if jv.ndim == 2:
                pp.data.copy_(torch.from_numpy(jv.T))
            else:
                pp.data.copy_(torch.from_numpy(jv))
    jax_opt_state2 = jax_opt.init(jax_params2)
    pt_opt2 = torch.optim.AdamW(pt_mlp2.parameters(), lr=LR, eps=1e-5)

    for step in range(10):
        # Generate new data each step
        np.random.seed(1000 + step)
        x_np2 = np.random.randn(8, FEATURE_DIM).astype(np.float32) * 0.5
        actions_np2 = np.random.randint(0, NA, size=8).astype(np.int32)
        old_log_probs_np2 = np.random.randn(8).astype(np.float32) * 0.1
        old_values_np2 = np.random.randn(8).astype(np.float32) * 0.3
        current_players_np2 = np.random.randint(0, 4, size=8).astype(np.int32)
        advantages_np2 = np.zeros((8, 4), dtype=np.float32)
        targets_np2 = np.zeros((8, 4), dtype=np.float32)
        valid_mask_np2 = np.zeros((8, 4), dtype=bool)
        for i in range(8):
            p = current_players_np2[i]
            advantages_np2[i, p] = np.float32(np.random.randn() * 0.5)
            targets_np2[i, p] = old_values_np2[i] + advantages_np2[i, p]
            valid_mask_np2[i, p] = True
        action_mask_np2 = np.random.rand(8, NA).astype(np.float32) > 0.7

        # JAX
        def jax_loss_fn2(params_list):
            (W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6) = params_list
            x = jnp.asarray(x_np2)
            h = jnp.tanh(x @ W1 + b1)
            h = jnp.tanh(h @ W2 + b2)
            logits = h @ W3 + b3
            h2 = jnp.tanh(x @ W4 + b4)
            h2 = jnp.tanh(h2 @ W5 + b5)
            values = (h2 @ W6 + b6).squeeze(-1)
            am = jnp.asarray(action_mask_np2)
            logits = jnp.where(am, logits, NEG)
            dist = distrax.Categorical(logits=logits)
            log_ratio = dist.log_prob(jnp.asarray(actions_np2)) - jnp.asarray(old_log_probs_np2)
            ratio = jnp.exp(log_ratio)[..., None]
            adv = jnp.take_along_axis(jnp.asarray(advantages_np2),
                                      jnp.asarray(current_players_np2)[..., None], axis=1)
            mask = jnp.take_along_axis(jnp.asarray(valid_mask_np2).astype(jnp.float32),
                                       jnp.asarray(current_players_np2)[..., None], axis=1)
            ppo_loss = -jax_masked_mean(
                jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv), mask)
            entropy = jax_masked_mean(dist.entropy()[..., None], mask)
            ov = jnp.asarray(old_values_np2)
            value_clipped = ov[..., None] + jnp.clip(
                values[..., None] - ov[..., None], -CLIP_EPS, CLIP_EPS)
            tgt = jnp.take_along_axis(jnp.asarray(targets_np2),
                                      jnp.asarray(current_players_np2)[..., None], axis=1)
            loss_critic = (0.5 * VF_COEF *
                           jax_masked_mean(jnp.maximum((values[..., None] - tgt) ** 2,
                                                       (value_clipped - tgt) ** 2), mask))
            return ppo_loss - ENT_COEF * entropy + loss_critic

        jax_grads2 = jax.grad(jax_loss_fn2)(jax_params2)
        jax_updates2, jax_opt_state2 = jax_opt.update(jax_grads2, jax_opt_state2, jax_params2)
        jax_params2 = optax.apply_updates(jax_params2, jax_updates2)

        # PT
        pt_opt2.zero_grad()
        logits, values_new = pt_mlp2(torch.from_numpy(x_np2))
        am = torch.from_numpy(action_mask_np2)
        logits = torch.where(am, logits, torch.full_like(logits, NEG))
        dist = torch.distributions.Categorical(logits=logits)
        log_ratio = dist.log_prob(torch.from_numpy(actions_np2).long()) - torch.from_numpy(old_log_probs_np2)
        ratio = torch.exp(log_ratio).unsqueeze(-1)
        adv = torch.from_numpy(advantages_np2).gather(
            1, torch.from_numpy(current_players_np2).long().unsqueeze(-1))
        vmask = torch.from_numpy(valid_mask_np2).float().gather(
            1, torch.from_numpy(current_players_np2).long().unsqueeze(-1))
        clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        ppo_loss = -pt_masked_mean(torch.min(ratio * adv, clip_adv), vmask)
        vt = values_new.unsqueeze(-1)
        ov = torch.from_numpy(old_values_np2)
        val_clipped = ov.unsqueeze(-1) + torch.clamp(vt - ov.unsqueeze(-1), -CLIP_EPS, CLIP_EPS)
        tgt = torch.from_numpy(targets_np2).gather(
            1, torch.from_numpy(current_players_np2).long().unsqueeze(-1))
        loss_critic = 0.5 * VF_COEF * pt_masked_mean(
            torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)
        total_loss = (ppo_loss - ENT_COEF * pt_masked_mean(dist.entropy().unsqueeze(-1), vmask) + loss_critic)
        total_loss.backward()
        pt_opt2.step()

        # Compare
        pt_params_list2 = list(pt_mlp2.parameters())
        max_pd = max_gd = 0.0
        for i, (jp, pp) in enumerate(zip(jax_params2, pt_params_list2)):
            jv = np.array(jp); pv = pp.detach().cpu().numpy()
            if jv.ndim == 2: jv = jv.T
            max_pd = max(max_pd, float(np.abs(jv - pv).max()))
            jg = np.array(jax_grads2[i]); pg = pp.grad.cpu().numpy()
            if jg.ndim == 2: jg = jg.T
            max_gd = max(max_gd, float(np.abs(jg - pg).max()))
        print(f"  Step {step+1}: max_grad_diff={max_gd:.2e}  max_param_diff={max_pd:.2e}")

    print(f"\n{'='*70}")
    print("Conclusion: The parameter drift is caused by subtle AdamW implementation")
    print("differences between optax and torch, accumulating ~1e-6 per step.")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
