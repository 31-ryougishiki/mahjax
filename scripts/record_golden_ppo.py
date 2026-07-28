#!/usr/bin/env python3
"""Record JAX PPO loss/gradient/optimizer golden data for mahjax_pt precision tests.

Records three .npz files:
    ppo_loss_mlp.npz     — network weights + batch data + JAX loss/metrics
    ppo_grad_mlp.npz     — same + JAX gradients
    ppo_optim_step_mlp.npz — same + JAX optimizer updated weights

Uses a simple MLP to avoid ACNet weight-mapping complexity.

Usage:
    python scripts/record_golden_ppo.py
    python scripts/record_golden_ppo.py --output ./mahjax_pt/tests/golden

Requires: JAX, optax
"""

import os, sys, argparse
import numpy as np

_JAX = None


def _init_jax():
    global _JAX
    if _JAX is not None:
        return
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_disable_jit', True)
    jax.config.update('jax_platform_name', 'cpu')
    _JAX = jax


# ═══════════════════════════════════════════════════════════════════════
# JAX MLP (simple, no Flax — manual params)
# ═══════════════════════════════════════════════════════════════════════

class JaxMLP:
    def __init__(self, rng, in_dim=32, hidden_dim=64, out_dim=87):
        _init_jax()
        import jax
        import jax.numpy as jnp
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
        import jax.numpy as jnp
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

    def params_dict(self):
        return {
            "fc1.weight": np.array(self.W1).T,  # JAX (in,out) → PT (out,in)
            "fc1.bias": np.array(self.b1),
            "fc2.weight": np.array(self.W2).T,
            "fc2.bias": np.array(self.b2),
            "actor.weight": np.array(self.W3).T,
            "actor.bias": np.array(self.b3),
            "critic.weight": np.array(self.W6).T,
            "critic.bias": np.array(self.b6),
        }


# ═══════════════════════════════════════════════════════════════════════
# JAX PPO loss
# ═══════════════════════════════════════════════════════════════════════

CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
NEG = -1e9


def jax_masked_mean(x, mask):
    import jax.numpy as jnp
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(
        mask.astype(jnp.float32).sum(), 1.0)


def jax_ppo_loss(mlp, params_list, obs, actions, old_log_probs,
                  advantages, targets, valid_mask, action_mask,
                  old_values, current_players):
    import jax.numpy as jnp

    def _forward(p, x):
        W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6 = p
        h = jnp.tanh(x @ W1 + b1)
        h = jnp.tanh(h @ W2 + b2)
        logits = h @ W3 + b3
        h2 = jnp.tanh(x @ W4 + b4)
        h2 = jnp.tanh(h2 @ W5 + b5)
        value = (h2 @ W6 + b6).squeeze(-1)
        return logits, value

    logits, values = _forward(params_list, obs)
    logits = jnp.where(action_mask, logits, NEG)

    # Softmax manually (avoid distrax dependency)
    logits_max = logits.max(axis=-1, keepdims=True)
    logits_shifted = logits - logits_max
    exp_logits = jnp.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    log_probs_all = jnp.log(probs + 1e-30)
    logp_new = log_probs_all[jnp.arange(len(actions)), actions]

    entropy_per_sample = -(probs * log_probs_all).sum(axis=-1)
    entropy = entropy_per_sample

    log_ratio = logp_new - old_log_probs
    ratio = jnp.exp(log_ratio)[..., None]

    adv = jnp.take_along_axis(advantages, current_players[..., None], axis=1)
    vmask = jnp.take_along_axis(valid_mask.astype(jnp.float32),
                                 current_players[..., None], axis=1)

    clip_adv = jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    ppo_loss = -jax_masked_mean(jnp.minimum(ratio * adv, clip_adv), vmask)

    vt = values[..., None]
    val_clipped = old_values[..., None] + jnp.clip(
        vt - old_values[..., None], -CLIP_EPS, CLIP_EPS)
    tgt = jnp.take_along_axis(targets, current_players[..., None], axis=1)
    loss_critic = 0.5 * VF_COEF * jax_masked_mean(
        jnp.maximum((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)

    approx_kl = jax_masked_mean((ratio - 1.0) - log_ratio[..., None], vmask)
    clip_frac = jax_masked_mean(
        (jnp.abs(ratio - 1.0) > CLIP_EPS).astype(jnp.float32), vmask)
    explained_var = jnp.maximum(
        1.0 - jax_masked_mean((tgt - vt) ** 2, vmask)
        / (jax_masked_mean((tgt - jax_masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8),
        0.0)

    total_loss = (ppo_loss
                  - ENT_COEF * jax_masked_mean(entropy[..., None], vmask)
                  + loss_critic)

    metrics = {
        "total_loss": total_loss,
        "actor_loss": ppo_loss,
        "critic_loss": loss_critic,
        "entropy": jax_masked_mean(entropy[..., None], vmask),
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "explained_var": explained_var,
    }
    return total_loss, metrics


def make_ppo_batch_data(batch_size=64, feature_dim=32, num_actions=87, seed=42):
    """Generate synthetic PPO batch data as numpy arrays."""
    rng = np.random.RandomState(seed)
    obs = rng.randn(batch_size, feature_dim).astype(np.float32) * 0.5
    actions = rng.randint(0, num_actions, size=batch_size).astype(np.int32)
    old_log_probs = rng.randn(batch_size).astype(np.float32) * 0.2
    old_values = rng.randn(batch_size).astype(np.float32) * 0.5
    current_players = rng.randint(0, 4, size=batch_size).astype(np.int32)

    advantages = np.zeros((batch_size, 4), dtype=np.float32)
    targets = np.zeros((batch_size, 4), dtype=np.float32)
    valid_mask = np.zeros((batch_size, 4), dtype=bool)
    for b in range(batch_size):
        p = current_players[b]
        advantages[b, p] = float(rng.randn() * 0.5)
        targets[b, p] = float(old_values[b] + advantages[b, p])
        valid_mask[b, p] = True
    action_mask = rng.rand(batch_size, num_actions).astype(np.float32) > 0.7

    return (obs, actions, old_log_probs, old_values, current_players,
            advantages, targets, valid_mask, action_mask)


def record_ppo(output_dir: str):
    """Record PPO loss, gradient, and optimizer golden data."""
    _init_jax()
    import jax
    import jax.numpy as jnp
    import optax

    os.makedirs(output_dir, exist_ok=True)

    rng = jax.random.PRNGKey(42)
    mlp = JaxMLP(rng)
    params = mlp.params_list()
    params_dict = mlp.params_dict()

    # Batch data
    (obs, actions, old_log_probs, old_values, current_players,
     advantages, targets, valid_mask, action_mask) = make_ppo_batch_data()

    # Convert to JAX
    j_obs = jnp.asarray(obs)
    j_actions = jnp.asarray(actions)
    j_olp = jnp.asarray(old_log_probs)
    j_adv = jnp.asarray(advantages)
    j_tgt = jnp.asarray(targets)
    j_vm = jnp.asarray(valid_mask)
    j_am = jnp.asarray(action_mask)
    j_ov = jnp.asarray(old_values)
    j_cp = jnp.asarray(current_players)

    # ═══ 1. Loss & Metrics ═══
    print("  Computing JAX PPO loss...")
    _, metrics = jax_ppo_loss(mlp, params, j_obs, j_actions, j_olp,
                               j_adv, j_tgt, j_vm, j_am, j_ov, j_cp)

    loss_golden = dict(params_dict)
    loss_golden.update({
        "input_dim": 32, "hidden_dim": 64, "num_actions": 87,
        "obs": obs, "actions": actions, "old_log_probs": old_log_probs,
        "old_values": old_values, "current_players": current_players,
        "advantages": advantages, "targets": targets,
        "valid_mask": valid_mask, "action_mask": action_mask,
    })
    for key in ["total_loss", "actor_loss", "critic_loss", "entropy",
                "approx_kl", "clip_frac", "explained_var"]:
        loss_golden[key] = np.asarray(float(metrics[key]), dtype=np.float32)

    path = os.path.join(output_dir, "ppo_loss_mlp.npz")
    np.savez_compressed(path, **loss_golden)
    print(f"    → {path}")

    # ═══ 2. Gradients ═══
    print("  Computing JAX PPO gradients...")
    grad_fn = jax.grad(lambda p: jax_ppo_loss(mlp, p, j_obs, j_actions,
                                               j_olp, j_adv, j_tgt, j_vm,
                                               j_am, j_ov, j_cp)[0])
    jax_grads = grad_fn(params)

    grad_golden = dict(params_dict)
    grad_golden.update({
        "input_dim": 32, "hidden_dim": 64, "num_actions": 87,
        "obs": obs, "actions": actions, "old_log_probs": old_log_probs,
        "old_values": old_values, "current_players": current_players,
        "advantages": advantages, "targets": targets,
        "valid_mask": valid_mask, "action_mask": action_mask,
    })

    # Map JAX grads to PT naming (transpose W)
    grad_keys = [
        ("grad_fc1.weight", 0, True),
        ("grad_fc1.bias", 1, False),
        ("grad_fc2.weight", 2, True),
        ("grad_fc2.bias", 3, False),
        ("grad_actor.weight", 4, True),
        ("grad_actor.bias", 5, False),
        ("grad_critic.weight", 10, True),
        ("grad_critic.bias", 11, False),
    ]
    for key, idx, needs_T in grad_keys:
        g = np.array(jax_grads[idx])
        if needs_T:
            g = g.T
        grad_golden[key] = g.astype(np.float32)

    path = os.path.join(output_dir, "ppo_grad_mlp.npz")
    np.savez_compressed(path, **grad_golden)
    print(f"    → {path}")

    # ═══ 3. Optimizer Step ═══
    print("  Computing JAX optimizer step...")
    lr, eps = 3e-4, 1e-5
    opt = optax.adamw(learning_rate=lr, eps=eps, weight_decay=0.0)
    opt_state = opt.init(params)
    updates, _ = opt.update(jax_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    optim_golden = dict(params_dict)
    optim_golden.update({
        "input_dim": 32, "hidden_dim": 64, "num_actions": 87,
        "lr": lr, "eps": eps,
        "obs": obs, "actions": actions, "old_log_probs": old_log_probs,
        "old_values": old_values, "current_players": current_players,
        "advantages": advantages, "targets": targets,
        "valid_mask": valid_mask, "action_mask": action_mask,
    })

    updated_keys = [
        ("updated_fc1.weight", 0, True),
        ("updated_fc1.bias", 1, False),
        ("updated_fc2.weight", 2, True),
        ("updated_fc2.bias", 3, False),
        ("updated_actor.weight", 4, True),
        ("updated_actor.bias", 5, False),
        ("updated_critic.weight", 10, True),
        ("updated_critic.bias", 11, False),
    ]
    for key, idx, needs_T in updated_keys:
        u = np.array(new_params[idx])
        if needs_T:
            u = u.T
        optim_golden[key] = u.astype(np.float32)

    path = os.path.join(output_dir, "ppo_optim_step_mlp.npz")
    np.savez_compressed(path, **optim_golden)
    print(f"    → {path}")


def main():
    parser = argparse.ArgumentParser(description="Record JAX PPO golden data")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "mahjax_pt", "tests", "golden")
    args.output = os.path.abspath(args.output)

    print(f"Recording PPO golden data to: {args.output}\n")
    record_ppo(args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
