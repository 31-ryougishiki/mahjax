#!/usr/bin/env python3
"""L1: PPO Math Primitives — JAX vs PyTorch numerical equivalence.

Verifies that the fundamental math operations used in the PPO pipeline
produce identical results in both frameworks.

Tests:
  1. masked_mean          — average over masked entries
  2. Categorical log_prob — log probability of sampled actions
  3. entropy              — distribution entropy
  4. PPO clip             — clipped advantage ratio
  5. Advantage norm       — (adv - mean) / (std + eps)
  6. explained_var        — value function explained variance
  7. approx_kl            — Taylor KL approximation: (r-1) - log(r)

All use real red_mahjong action-space dimensions (87 actions, 4 players).
"""

import numpy as np
import jax, jax.numpy as jnp
import torch, torch.nn.functional as F

SEED = 42
B, NA, NP = 32, 87, 4  # batch, num_actions, num_players
EPS = 1e-8
CLIP_EPS = 0.2


# ═══════════════════════════════════════════════════════════════════════════
# Helper: shared random data
# ═══════════════════════════════════════════════════════════════════════════

def make_test_data():
    np.random.seed(SEED)
    return {
        "logits": np.random.randn(B, NA).astype(np.float32) * 2.0,
        "actions": np.random.randint(0, NA, size=B).astype(np.int32),
        "old_log_probs": np.random.randn(B).astype(np.float32) * 0.1,
        "advantages": np.random.randn(B, NP).astype(np.float32) * 0.5,
        "targets": np.random.randn(B, NP).astype(np.float32) * 0.5 + 0.1,
        "values": np.random.randn(B).astype(np.float32) * 0.3,
        "valid_mask": np.random.rand(B, NP).astype(np.float32) > 0.3,
        "action_mask": np.random.rand(B, NA).astype(np.float32) > 0.5,
    }


NEG = -1e9


# ═══════════════════════════════════════════════════════════════════════════
# JAX implementations
# ═══════════════════════════════════════════════════════════════════════════

def jax_masked_mean(x, mask):
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(
        mask.astype(jnp.float32).sum(), 1.0)


def jax_categorical_log_prob(logits, actions):
    log_probs = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(-1)


def jax_entropy(logits, mask=None):
    probs = jax.nn.softmax(logits)
    ent = -(probs * jax.nn.log_softmax(logits)).sum(axis=-1)
    if mask is not None:
        return jax_masked_mean(ent, mask.any(axis=-1))
    return ent.mean()


def jax_ppo_clip_loss(ratio, adv, eps=CLIP_EPS):
    clip_adv = jnp.clip(ratio, 1 - eps, 1 + eps) * adv
    return -jnp.minimum(ratio * adv, clip_adv)


def jax_adv_normalize(advantages, valid_mask):
    mf = valid_mask.astype(jnp.float32)
    mean = jax_masked_mean(advantages, mf)
    var = jax_masked_mean((advantages - mean) ** 2, mf)
    return (advantages - mean) / (jnp.sqrt(var) + EPS)


def jax_explained_var(targets, values, valid_mask):
    mf = valid_mask.astype(jnp.float32)
    mse = jax_masked_mean((targets - values) ** 2, mf)
    var = jax_masked_mean((targets - jax_masked_mean(targets, mf)) ** 2, mf)
    return jnp.maximum(1.0 - mse / (var + EPS), 0.0)


def jax_approx_kl(ratio, log_ratio, valid_mask):
    """Taylor approximation: KL ≈ (r-1) - log(r)"""
    return jax_masked_mean((ratio - 1.0) - log_ratio, valid_mask.astype(jnp.float32))


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch implementations (mirroring ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════════

def pt_masked_mean(x, mask):
    denom = mask.float().sum().clamp(min=1.0)
    return (x * mask.float()).sum() / denom


def pt_categorical_log_prob(logits, actions):
    return F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(-1)


def pt_entropy(logits, mask=None):
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    if mask is not None:
        return pt_masked_mean(ent, mask.any(dim=-1))
    return ent.mean()


def pt_ppo_clip_loss(ratio, adv, eps=CLIP_EPS):
    clip_adv = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(ratio * adv, clip_adv)


def pt_adv_normalize(advantages, valid_mask):
    mf = valid_mask.float()
    adv_sum = (advantages * mf).sum()
    adv_count = mf.sum().clamp(min=1.0)
    mean = adv_sum / adv_count
    var = ((advantages - mean) ** 2 * mf).sum() / adv_count
    return (advantages - mean) / (var.sqrt() + EPS)


def pt_explained_var(targets, values, valid_mask):
    mf = valid_mask.float()
    mse = pt_masked_mean((targets - values) ** 2, mf)
    var = pt_masked_mean(
        (targets - pt_masked_mean(targets, mf)) ** 2, mf)
    return torch.clamp(1.0 - mse / (var + EPS), min=0.0)


def pt_approx_kl(ratio, log_ratio, valid_mask):
    """Taylor approximation: KL ≈ (r-1) - log(r)"""
    return pt_masked_mean((ratio - 1.0) - log_ratio, valid_mask.float())


# ═══════════════════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════════════════

def run_test(name, jax_fn, pt_fn, jax_args, pt_args, tol=1e-5):
    """Run a single parity check."""
    jax_val = jax_fn(*jax_args)
    if isinstance(jax_val, jnp.ndarray):
        jax_val = np.array(jax_val)
    else:
        jax_val = float(jax_val)

    pt_val = pt_fn(*pt_args)
    if isinstance(pt_val, torch.Tensor):
        pt_val = pt_val.detach().numpy()
    else:
        pt_val = float(pt_val)

    diff = np.abs(jax_val - pt_val)
    if isinstance(diff, np.ndarray):
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
    else:
        max_diff = mean_diff = float(diff)

    ok = max_diff < tol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
    return ok


def main():
    data = make_test_data()

    # Convert data
    jl = jnp.asarray(data["logits"])
    ja = jnp.asarray(data["actions"])
    jo = jnp.asarray(data["old_log_probs"])
    jadv = jnp.asarray(data["advantages"])
    jtgt = jnp.asarray(data["targets"])
    jval = jnp.asarray(data["values"])
    jvm = jnp.asarray(data["valid_mask"])
    jam = jnp.asarray(data["action_mask"])

    pl = torch.from_numpy(data["logits"])
    pa = torch.from_numpy(data["actions"]).long()
    po = torch.from_numpy(data["old_log_probs"])
    padv = torch.from_numpy(data["advantages"])
    ptgt = torch.from_numpy(data["targets"])
    pval = torch.from_numpy(data["values"])
    pvm = torch.from_numpy(data["valid_mask"])
    pam = torch.from_numpy(data["action_mask"])

    tests = []
    results = []

    # ── 1. masked_mean ──
    jm = jax_masked_mean(jadv, jvm)
    pm = pt_masked_mean(padv, pvm)
    tests.append(("masked_mean", float(jm), float(pm)))

    # ── 2. Categorical log_prob ──
    jlp = jax_categorical_log_prob(jl, ja)
    plp = pt_categorical_log_prob(pl, pa)
    tests.append(("log_prob", jlp, plp))

    # ── 3. Entropy ──
    je = jax_entropy(jl, jam)
    pe = pt_entropy(pl, pam)
    tests.append(("entropy", float(je), float(pe)))

    # ── 4. PPO clip loss ──
    # Compute ratio from log_probs
    jlp2 = jax_categorical_log_prob(jl, ja)
    plp2 = pt_categorical_log_prob(pl, pa)
    jratio = jnp.exp(jlp2 - jo).reshape(-1, 1)  # (B, 1)
    pratio = torch.exp(plp2 - po).unsqueeze(-1)   # (B, 1)

    # Use a single-column advantage for this test
    jadv1 = jadv[:, :1]  # (B, 1)
    padv1 = padv[:, :1]  # (B, 1)

    jclip = jax_ppo_clip_loss(jratio, jadv1)
    pclip = pt_ppo_clip_loss(pratio, padv1)
    tests.append(("ppo_clip_loss", jclip, pclip))

    # ── 5. Advantage normalization ──
    jadv_norm = jax_adv_normalize(jadv, jvm)
    padv_norm = pt_adv_normalize(padv, pvm)
    tests.append(("adv_normalize", jadv_norm, padv_norm))

    # ── 6. explained_var ──
    # Flatten (B, NP) → (B*NP,) for scalar comparison
    jtgt_f = jtgt.reshape(-1)
    ptgt_f = ptgt.reshape(-1)
    jvm_f = jvm.reshape(-1)
    pvm_f = pvm.reshape(-1)
    # Use same values broadcast
    jval_exp = jnp.repeat(jval[:, None], NP, axis=1).reshape(-1)
    pval_exp = pval.unsqueeze(-1).expand(-1, NP).reshape(-1)

    jev = jax_explained_var(jtgt_f, jval_exp, jvm_f)
    pev = pt_explained_var(ptgt_f, pval_exp, pvm_f)
    tests.append(("explained_var", float(jev), float(pev)))

    # ── 7. approx_kl ──
    jlr = jnp.log(jratio)
    plr = torch.log(pratio)
    jak = jax_approx_kl(jratio, jlr, jnp.ones_like(jratio))
    pak = pt_approx_kl(pratio, plr, torch.ones_like(pratio))
    tests.append(("approx_kl", float(jak), float(pak)))

    # ── Print results ──
    print(f"\n{'='*60}")
    print("L1: PPO Math Primitives Parity")
    print(f"{'='*60}\n")

    all_ok = True
    for name, jv, pv in tests:
        if isinstance(jv, (np.ndarray, jnp.ndarray)):
            jv_np = np.array(jv)
            pv_np = pv.detach().numpy() if isinstance(pv, torch.Tensor) else np.array(pv)
            max_diff = float(np.abs(jv_np - pv_np).max())
            mean_diff = float(np.abs(jv_np - pv_np).mean())
        else:
            max_diff = mean_diff = float(abs(jv - pv))

        ok = max_diff < 1e-5
        if not ok:
            all_ok = False
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:20s}  max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}")
        if not ok:
            print(f"         JAX={float(np.mean(np.array(jv))):.8f}  PT={float(np.mean(np.array(pv))):.8f}")

    print(f"\n{'='*60}")
    print(f"L1 Result: {'[PASS] ALL IDENTICAL' if all_ok else '[FAIL] See diffs above'}")
    print(f"{'='*60}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
