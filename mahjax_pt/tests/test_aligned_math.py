#!/usr/bin/env python3
"""Eliminate ALL framework-level float32 differences.

Strategy: run forward/backward through numpy (consistent everywhere),
use JAX/PT only for AdamW optimizer. This isolates the optimizer as
the ONLY possible source of divergence.
"""

import numpy as np
import jax, jax.numpy as jnp, optax, torch

LR=3e-4; EPS=1e-5; B1=0.9; B2=0.999
CLIP_EPS=0.2; ENT_COEF=0.01; VF_COEF=0.5; NEG=-1e9

# ═══════════════════════════════════════════════════════════════════════════
# All computation done in numpy (consistent between frameworks)
# ═══════════════════════════════════════════════════════════════════════════

def numpy_mlp_forward(params, x):
    """params = [W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6]
    All shapes: W=(in,out), b=(out,), x=(B,in)"""
    W1,b1,W2,b2,W3,b3,W4,b4,W5,b5,W6,b6 = params
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    logits = h2 @ W3 + b3
    hv1 = np.tanh(x @ W4 + b4)
    hv2 = np.tanh(hv1 @ W5 + b5)
    values = (hv2 @ W6 + b6).squeeze(-1)
    return logits, values

def numpy_ppo_loss(params, x, actions, old_log_probs, advantages, targets,
                   valid_mask, action_mask, old_values, current_players):
    logits, values = numpy_mlp_forward(params, x)
    # Mask
    logits = np.where(action_mask, logits, np.float32(NEG))
    # Softmax
    logits_max = logits.max(axis=-1, keepdims=True)
    logits_shifted = logits - logits_max
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    # log_prob
    B = len(actions)
    log_probs_new = np.log(probs[np.arange(B), actions] + 1e-30)
    # entropy
    log_probs_all = np.log(probs + 1e-30)
    entropy_per_sample = -(probs * log_probs_all).sum(axis=-1)

    log_ratio = log_probs_new - old_log_probs
    ratio = np.exp(log_ratio)[:, None]  # (B, 1)

    # gather to current player
    adv_1 = advantages[np.arange(B), current_players][:, None]  # (B, 1)
    mask_1 = valid_mask.astype(np.float32)[np.arange(B), current_players][:, None]

    def masked_mean(v, m):
        return (v * m).sum() / max(m.sum(), 1.0)

    # PPO loss
    clip_adv = np.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_1
    obj = np.minimum(ratio * adv_1, clip_adv)
    ppo_loss = -masked_mean(obj, mask_1)

    # Entropy
    entropy = masked_mean(entropy_per_sample[:, None], mask_1)

    # Critic loss
    vt = values[:, None]  # (B, 1)
    ov = old_values[:, None]
    val_clipped = ov + np.clip(vt - ov, -CLIP_EPS, CLIP_EPS)
    tgt_1 = targets[np.arange(B), current_players][:, None]
    critic_raw = np.maximum((vt - tgt_1)**2, (val_clipped - tgt_1)**2)
    critic_loss = 0.5 * VF_COEF * masked_mean(critic_raw, mask_1)

    loss = ppo_loss - ENT_COEF * entropy + critic_loss
    return loss

# Numerical gradient (central differences, slow but numpy-consistent)
def numpy_grad(params, loss_fn, *args, eps=1e-4):
    """Compute gradients via central-difference. Returns list of grad arrays."""
    grads = []
    for p in params:
        flat = p.ravel()
        g_flat = np.zeros_like(flat)
        for i in range(len(flat)):
            orig = flat[i]
            flat[i] = orig + eps
            loss_plus = loss_fn(params, *args)
            flat[i] = orig - eps
            loss_minus = loss_fn(params, *args)
            flat[i] = orig
            g_flat[i] = (loss_plus - loss_minus) / (2 * eps)
        grads.append(g_flat.reshape(p.shape))
    return grads

# ═══════════════════════════════════════════════════════════════════════════
# Main: compare JAX optax vs PT manual AdamW using numpy-computed grads
# ═══════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    FEATURE_DIM=32; HIDDEN_DIM=16; NUM_ACTIONS=87; BATCH=8; NUM_PLAYERS=4

    # Generate random params (numpy)
    def randW(i, o):
        return np.random.randn(i, o).astype(np.float32) * 0.1
    np_params = [
        randW(FEATURE_DIM, HIDDEN_DIM), np.zeros(HIDDEN_DIM, dtype=np.float32),
        randW(HIDDEN_DIM, HIDDEN_DIM), np.zeros(HIDDEN_DIM, dtype=np.float32),
        randW(HIDDEN_DIM, NUM_ACTIONS)*0.01, np.zeros(NUM_ACTIONS, dtype=np.float32),
        randW(FEATURE_DIM, HIDDEN_DIM), np.zeros(HIDDEN_DIM, dtype=np.float32),
        randW(HIDDEN_DIM, HIDDEN_DIM), np.zeros(HIDDEN_DIM, dtype=np.float32),
        randW(HIDDEN_DIM, 1), np.zeros(1, dtype=np.float32),
    ]

    # Generate batch data
    x_np = np.random.randn(BATCH, FEATURE_DIM).astype(np.float32) * 0.5
    a_np = np.random.randint(0, NUM_ACTIONS, size=BATCH).astype(np.int32)
    olp_np = np.random.randn(BATCH).astype(np.float32) * 0.1
    ov_np = np.random.randn(BATCH).astype(np.float32) * 0.3
    cp_np = np.random.randint(0, NUM_PLAYERS, size=BATCH).astype(np.int32)
    adv_np = np.zeros((BATCH, NUM_PLAYERS), dtype=np.float32)
    tgt_np = np.zeros((BATCH, NUM_PLAYERS), dtype=np.float32)
    vm_np = np.zeros((BATCH, NUM_PLAYERS), dtype=bool)
    for i in range(BATCH):
        p_ = cp_np[i]; adv_np[i,p_] = np.float32(np.random.randn()*0.5)
        tgt_np[i,p_] = ov_np[i]+adv_np[i,p_]; vm_np[i,p_] = True
    am_np = np.random.rand(BATCH, NUM_ACTIONS).astype(np.float32) > 0.7

    print("=== Comparing 1 AdamW step with IDENTICAL numpy grads ===\n")

    # Compute gradient using numpy (same for both frameworks)
    grads_np = numpy_grad(np_params, numpy_ppo_loss, x_np, a_np, olp_np,
                          adv_np, tgt_np, vm_np, am_np, ov_np, cp_np, eps=1e-4)

    # ═══ JAX optax step ═══
    jax_params = [jnp.array(p) for p in np_params]
    jax_grads = [jnp.array(g) for g in grads_np]
    opt_j = optax.adamw(learning_rate=LR, eps=EPS)
    st_j = opt_j.init(jax_params)
    up_j, st_j = opt_j.update(jax_grads, st_j, jax_params)
    jax_new = optax.apply_updates(jax_params, up_j)

    # ═══ PT manual optax step ═══
    pt_params = [torch.from_numpy(p).clone() for p in np_params]
    pt_grads = [torch.from_numpy(g) for g in grads_np]
    b1_t = torch.tensor(B1, dtype=torch.float32)
    b2_t = torch.tensor(B2, dtype=torch.float32)
    count_t = torch.tensor(1, dtype=torch.int32)
    bc1 = 1.0 - b1_t ** count_t
    bc2 = 1.0 - b2_t ** count_t
    mu = [torch.zeros_like(p) for p in pt_params]
    nu = [torch.zeros_like(p) for p in pt_params]
    with torch.no_grad():
        for p, g, m, n in zip(pt_params, pt_grads, mu, nu):
            mu_new = (1 - b1_t) * g + b1_t * m
            nu_new = (1 - b2_t) * g * g + b2_t * n
            mu_hat = mu_new / bc1; nu_hat = nu_new / bc2
            p.sub_(LR * mu_hat / (nu_hat.sqrt() + EPS))
            m.copy_(mu_new); n.copy_(nu_new)

    # ═══ PT native AdamW step ═══
    pp_native = [torch.nn.Parameter(torch.from_numpy(p).clone()) for p in np_params]
    opt_n = torch.optim.AdamW(pp_native, lr=LR, eps=EPS)
    opt_n.zero_grad()
    for p, g in zip(pp_native, pt_grads):
        p.grad = g.clone()
    opt_n.step()

    # Compare
    names = ['W1','b1','W2','b2','W3','b3','W4','b4','W5','b5','W6','b6']
    print(f"{'Param':12s} {'JAX-native':>14s} {'JAX-manual':>14s}")
    print('-'*45)
    for i, name in enumerate(names):
        jv = np.array(jax_new[i]); nm = pt_params[i].numpy(); nn = pp_native[i].detach().numpy()
        if jv.ndim == 2: jv=jv.T; nm=nm.T; nn=nn.T
        jm = float(np.abs(jv.astype(np.float64) - nm.astype(np.float64)).max())
        jn = float(np.abs(jv.astype(np.float64) - nn.astype(np.float64)).max())
        print(f'{name:12s} {jn:14.2e} {jm:14.2e}')

    # Summary
    jv_all = np.concatenate([np.array(jax_new[i]).ravel() for i in range(len(jax_new))])
    pm_all = np.concatenate([pt_params[i].numpy().ravel() for i in range(len(pt_params))])
    pn_all = np.concatenate([pp_native[i].detach().numpy().ravel() for i in range(len(pp_native))])

    jm_diff = float(np.abs(jv_all.astype(np.float64) - pm_all.astype(np.float64)).max())
    jn_diff = float(np.abs(jv_all.astype(np.float64) - pn_all.astype(np.float64)).max())

    print(f"\n{'─'*45}")
    print(f"JAX vs PT manual (optax formula): {jm_diff:.2e}")
    print(f"JAX vs PT native AdamW:           {jn_diff:.2e}")
    print(f"\nGDZ: gradients are IDENTICAL (from numpy),")
    print(f"so the above diffs come PURELY from the optimizer implementation.")

    return 0

if __name__ == "__main__":
    main()
