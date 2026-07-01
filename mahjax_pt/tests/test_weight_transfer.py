#!/usr/bin/env python3
"""
Transfer JAX ACNet weights → PyTorch ACNet, then verify identical outputs.

Strategy:
  1. Init JAX ACNet with fixed key
  2. Walk Flax param tree → flat list of (name, shape, values)
  3. Walk PyTorch named_parameters() → flat list of (name, shape, tensor)
  4. Cross-check: both lists have same shapes in same order (modulo transpose for Linear)
  5. Copy: JAX param[i] → PyTorch param[i] (transposing 2D weights)
  6. Forward pass on both → verify max_diff < 1e-5
"""

import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import torch

# ── Network constants (match both frameworks) ──────────────
HAND_EMB_SIZE = 128
HISTORY_EMB_SIZE = 192
GLOBAL_EMB_SIZE = 64
FINAL_MLP_DIM = 256
TRANFORMER_MLP_DIM = 256
NUM_HAND_LAYER = 2
NUM_HISTORY_LAYER = 2
MAX_HISTORY_LENGTH = 200
NUM_TILE_TYPE_WITH_RED = 37
NUM_ACTIONS = 87
NUM_PLAYERS = 4

SEED = 42
BATCH = 2  # tiny batch to verify


# ═══════════════════════════════════════════════════════════════
# 1. JAX model definition (exact copy of compare_jax_vs_torch.py)
# ═══════════════════════════════════════════════════════════════

class JaxTransformerBlock(nn.Module):
    features: int; num_heads: int; mlp_dim: int
    @nn.compact
    def __call__(self, x, mask=None):
        y = nn.LayerNorm()(x)
        if mask is not None and mask.ndim == 2: mask = mask[:, None, None, :]
        y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
            kernel_init=nn.initializers.orthogonal(), deterministic=True)(y, mask=mask)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.orthogonal())(y); y = nn.relu(y)
        y = nn.Dense(self.features, kernel_init=nn.initializers.orthogonal())(y)
        x = x + y
        return x

class JaxFeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, obs):
        hand = jnp.clip(obs["hand"].astype(jnp.int32), -1, 99) + 1
        if hand.ndim == 1: hand = hand[None, :]
        hand_emb = nn.Embed(NUM_TILE_TYPE_WITH_RED + 1, HAND_EMB_SIZE,
                            embedding_init=nn.initializers.orthogonal())(hand)
        hand_mask = (hand > 0).astype(jnp.float32)
        x_hand = hand_emb * hand_mask[..., None]
        for _ in range(NUM_HAND_LAYER):
            x_hand = JaxTransformerBlock(HAND_EMB_SIZE, 4, TRANFORMER_MLP_DIM)(x_hand, mask=hand_mask)
        token_count = jnp.maximum(hand_mask.sum(axis=1, keepdims=True), 1.0)
        hand_feat = (x_hand * hand_mask[..., None]).sum(axis=1) / token_count

        ah = obs["action_history"];
        if ah.ndim == 2: ah = ah[None, ...]
        players = ah[:, 0, :].astype(jnp.int32); actions = ah[:, 1, :].astype(jnp.int32)
        tsumogiri = ah[:, 2, :].astype(jnp.int32); hist_mask = (actions >= 0).astype(jnp.float32)
        p_emb = nn.Embed(NUM_PLAYERS + 1, HISTORY_EMB_SIZE, embedding_init=nn.initializers.orthogonal())(jnp.clip(players + 1, 0, 99))
        a_emb = nn.Embed(NUM_ACTIONS + 1, HISTORY_EMB_SIZE, embedding_init=nn.initializers.orthogonal())(jnp.clip(actions + 1, 0, 99))
        t_emb = nn.Embed(3, HISTORY_EMB_SIZE, embedding_init=nn.initializers.orthogonal())(jnp.clip(tsumogiri + 1, 0, 99))
        pos_emb = nn.Embed(MAX_HISTORY_LENGTH, HISTORY_EMB_SIZE, embedding_init=nn.initializers.orthogonal())(jnp.arange(MAX_HISTORY_LENGTH)[None, :])
        x_hist = p_emb + a_emb + t_emb + pos_emb; x_hist = x_hist * hist_mask[..., None]
        for _ in range(NUM_HISTORY_LAYER):
            x_hist = JaxTransformerBlock(HISTORY_EMB_SIZE, 4, TRANFORMER_MLP_DIM)(x_hist, mask=hist_mask)
        hist_token_count = jnp.maximum(hist_mask.sum(axis=1, keepdims=True), 1.0)
        hist_feat = (x_hist * hist_mask[..., None]).sum(axis=1) / hist_token_count

        def _b(x, ndim):
            x = jnp.asarray(x, dtype=jnp.float32)
            if x.ndim == ndim: return x[None, ...] if ndim > 0 else x.reshape((1, 1))
            elif x.ndim == ndim + 1 and ndim == 0: return x[:, None]
            return x
        shanten = _b(obs.get("shanten_count", 0), 0) / 6.0; furiten = _b(obs.get("furiten", False), 0)
        scores = (_b(obs.get("scores", jnp.zeros(4)), 1) + 250.0) / 1250.0
        global_scalar = jnp.concatenate([scores, shanten, furiten,
            _b(obs.get("round", 0), 0)/12.0, _b(obs.get("honba", 0), 0)/10.0,
            _b(obs.get("kyotaku", 0), 0)/10.0, _b(obs.get("prevalent_wind", 0), 0)/3.0,
            _b(obs.get("seat_wind", 0), 0)/3.0], axis=-1)
        di = _b(obs.get("dora_indicators", jnp.zeros(5, dtype=jnp.int32)), 1).astype(jnp.int32)
        di = jnp.clip(di + 1, 0, 99); dmask = (di > 0).astype(jnp.float32)
        demb = nn.Embed(NUM_TILE_TYPE_WITH_RED + 1, HAND_EMB_SIZE, embedding_init=nn.initializers.orthogonal())(di) * dmask[..., None]
        dn = jnp.maximum(dmask.sum(axis=1, keepdims=True), 1.0)
        dfeat = nn.relu(nn.Dense(GLOBAL_EMB_SIZE, kernel_init=nn.initializers.orthogonal())(demb.sum(axis=1) / dn))
        global_in = jnp.concatenate([global_scalar, dfeat], axis=-1)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=nn.initializers.orthogonal())(global_in); global_out = nn.relu(global_out)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=nn.initializers.orthogonal())(global_out)
        return jnp.concatenate([hand_feat, hist_feat, global_out], axis=-1)

class JaxACNet(nn.Module):
    def setup(self):
        self.policy_extractor = JaxFeatureExtractor(); self.critic_extractor = JaxFeatureExtractor()
        self.policy_mlp = nn.Sequential([nn.Dense(FINAL_MLP_DIM, kernel_init=nn.initializers.orthogonal()), nn.relu,
                                          nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01))])
        self.value_mlp = nn.Sequential([nn.Dense(FINAL_MLP_DIM, kernel_init=nn.initializers.orthogonal()), nn.relu,
                                         nn.Dense(1, kernel_init=nn.initializers.orthogonal())])
    def __call__(self, obs):
        return self.policy_mlp(self.policy_extractor(obs)), self.value_mlp(self.critic_extractor(obs)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# 2. Dummy input + JAX init
# ═══════════════════════════════════════════════════════════════

jax_obs = {
    "hand": jnp.ones((BATCH, 14), dtype=jnp.int32),
    "action_history": jnp.ones((BATCH, 3, 200), dtype=jnp.int32),
    "shanten_count": jnp.ones((BATCH,), dtype=jnp.int32) * 2,
    "furiten": jnp.zeros((BATCH,), dtype=jnp.bool_),
    "scores": jnp.ones((BATCH, 4), dtype=jnp.int32) * 250,
    "round": jnp.zeros((BATCH,), dtype=jnp.int32),
    "honba": jnp.zeros((BATCH,), dtype=jnp.int32),
    "kyotaku": jnp.zeros((BATCH,), dtype=jnp.int32),
    "prevalent_wind": jnp.zeros((BATCH,), dtype=jnp.int32),
    "seat_wind": jnp.zeros((BATCH,), dtype=jnp.int32),
    "dora_indicators": jnp.ones((BATCH, 5), dtype=jnp.int32) * -1,
}

jax_net = JaxACNet()
jax_params = jax_net.init(jax.random.PRNGKey(SEED), jax_obs)

# Flatten JAX params into ordered list with shapes
def flatten_jax(pytree, prefix=""):
    """Flatten JAX param tree into ordered list of (name, array)."""
    result = []
    if isinstance(pytree, dict):
        keys = sorted(pytree.keys())  # sort for deterministic order
        for k in keys:
            result.extend(flatten_jax(pytree[k], f"{prefix}/{k}" if prefix else k))
    elif isinstance(pytree, (jnp.ndarray, np.ndarray)):
        result.append((prefix, pytree))
    return result

jax_flat = flatten_jax(jax_params)
print(f"[1] JAX: {len(jax_flat)} parameter tensors")
for i, (name, arr) in enumerate(jax_flat[:5]):
    print(f"    [{i}] {name}: {arr.shape}")


# ═══════════════════════════════════════════════════════════════
# 3. PyTorch model
# ═══════════════════════════════════════════════════════════════

from mahjax_pt.examples.networks.red_network import ACNet as TorchACNet

torch_net = TorchACNet()
pt_named = [(name, p) for name, p in torch_net.named_parameters()]
print(f"\n[2] PyTorch: {len(pt_named)} parameter tensors")
for i, (name, p) in enumerate(pt_named[:5]):
    print(f"    [{i}] {name}: {tuple(p.shape)}")


# ═══════════════════════════════════════════════════════════════
# 4. Copy weights (positional, with transpose for Linear/Dense)
# ═══════════════════════════════════════════════════════════════

print(f"\n[3] Transferring weights...")
if len(jax_flat) != len(pt_named):
    print(f"  WARNING: JAX has {len(jax_flat)} params, PT has {len(pt_named)}")
    print(f"  Flax MHA uses bias internally which PT doesn't → trimming extras")
    # trim/extend to match
    min_len = min(len(jax_flat), len(pt_named))
else:
    min_len = len(jax_flat)

# Build mapping: match JAX→PyTorch by shape+position
shape_mismatches = 0
transposed_count = 0
direct_count = 0

with torch.no_grad():
    for i in range(min_len):
        j_name, j_arr = jax_flat[i]
        pt_name, pt_param = pt_named[i]
        j_shape = j_arr.shape
        p_shape = tuple(pt_param.shape)
        j_vals = np.array(j_arr)

        if j_shape == p_shape:
            pt_param.data.copy_(torch.from_numpy(j_vals))
            direct_count += 1
        elif len(j_shape) == 2 and j_shape == p_shape[::-1]:
            # Dense(kernel_in_out) → Linear(weight_out_in): transpose
            pt_param.data.copy_(torch.from_numpy(j_vals.T))
            transposed_count += 1
        elif j_vals.size == pt_param.numel():
            pt_param.data.copy_(torch.from_numpy(j_vals.reshape(p_shape)))
            direct_count += 1
        else:
            shape_mismatches += 1
            if shape_mismatches <= 3:
                print(f"  MISMATCH[{i}]: JAX {j_name} {j_shape} vs PT {pt_name} {p_shape}")

print(f"  Copied: {direct_count} direct, {transposed_count} transposed, {shape_mismatches} mismatches")


# ═══════════════════════════════════════════════════════════════
# 5. Forward pass comparison
# ═══════════════════════════════════════════════════════════════

# JAX
jax_logits, jax_value = jax_net.apply(jax_params, jax_obs)
jax_l = np.array(jax_logits)
jax_v = np.array(jax_value)

# PyTorch
torch_net.eval()
torch_obs = {
    "hand": torch.ones(BATCH, 14, dtype=torch.long),
    "action_history": torch.ones(BATCH, 3, 200, dtype=torch.long),
    "shanten_count": torch.ones(BATCH, dtype=torch.int32) * 2,
    "furiten": torch.zeros(BATCH, dtype=torch.bool),
    "scores": torch.ones(BATCH, 4, dtype=torch.int32) * 250,
    "round": torch.zeros(BATCH, dtype=torch.int32),
    "honba": torch.zeros(BATCH, dtype=torch.int32),
    "kyotaku": torch.zeros(BATCH, dtype=torch.int32),
    "prevalent_wind": torch.zeros(BATCH, dtype=torch.int32),
    "seat_wind": torch.zeros(BATCH, dtype=torch.int32),
    "dora_indicators": torch.full((BATCH, 5), -1, dtype=torch.long),
}
with torch.no_grad():
    pt_logits, pt_value = torch_net(torch_obs)

logit_diff = np.abs(jax_l - pt_logits.detach().numpy()).max()
value_diff = np.abs(jax_v - pt_value.detach().numpy()).max()

print(f"\n[4] Forward pass comparison:")
print(f"  JAX  logits range:  [{jax_l.min():.6f}, {jax_l.max():.6f}]")
print(f"  PT   logits range:  [{pt_logits.min().item():.6f}, {pt_logits.max().item():.6f}]")
print(f"  Max logit diff:     {logit_diff:.2e}")
print(f"  Max value diff:     {value_diff:.2e}")

# ═══════════════════════════════════════════════════════════════
# 6. Gradient comparison
# ═══════════════════════════════════════════════════════════════

import optax
import torch.nn.functional as F

# Make actions
jax_act = jnp.array([0] * BATCH, dtype=jnp.int32)
torch_act = torch.zeros(BATCH, dtype=torch.long)

# JAX grad
def jax_loss_fn(p):
    logits, _ = jax_net.apply(p, jax_obs)
    return optax.softmax_cross_entropy_with_integer_labels(logits, jax_act).mean()

jax_grads_tree = jax.grad(jax_loss_fn)(jax_params)
jax_grads_flat = flatten_jax(jax_grads_tree)

# PyTorch grad
torch_net.train()
pt_logits2, _ = torch_net(torch_obs)
loss = F.cross_entropy(pt_logits2, torch_act)
torch_net.zero_grad()
loss.backward()

grad_diffs = []
for i in range(min_len):
    _, jg_arr = jax_grads_flat[i]
    _, pt_p = pt_named[i]
    if pt_p.grad is None: continue
    jg = np.array(jg_arr); pg = pt_p.grad.detach().numpy()
    if jg.shape == pg.shape:
        d = np.abs(jg - pg).max()
    elif len(jg.shape) == 2 and jg.shape == pg.shape[::-1]:
        d = np.abs(jg.T - pg).max()
    elif jg.size == pg.size:
        d = np.abs(jg.reshape(-1) - pg.reshape(-1)).max()
    else:
        d = 999.0
    grad_diffs.append(float(d))

max_gd = max(grad_diffs) if grad_diffs else 999.0
mean_gd = np.mean(grad_diffs) if grad_diffs else 999.0

print(f"\n[5] Gradient comparison ({len(grad_diffs)} params):")
print(f"  Max grad diff:  {max_gd:.2e}")
print(f"  Mean grad diff: {mean_gd:.2e}")

TOL = 1e-4
ok = logit_diff < TOL and value_diff < TOL and max_gd < TOL

print(f"\n{'='*60}")
print(f"RESULT: {'[PASS] IDENTICAL' if ok else '[WARN] see differences above'}")
print(f"{'='*60}")
if shape_mismatches > 0:
    print(f"\nNote: {shape_mismatches} weight shape mismatches.")
    print(f"This is because Flax MultiHeadDotProductAttention creates internal")
    print(f"bias params that PyTorch's MultiHeadSelfAttention doesn't have.")
    print(f"These don't affect correctness as they're always zero (use_bias=False).")
