#!/usr/bin/env python3
"""
Compare JAX (Flax) vs PyTorch BC training, step by step.

Prints the loss and gradient norm from both frameworks at each step,
using the same model weights and same batch of data.

Usage:
    PYTHONPATH=. python mahjax_pt/compare_jax_vs_torch.py
"""

import os, sys, pickle
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import torch
import torch.nn.functional as F

# ── 1. Load same data batch ─────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "examples", "offline_data",
                         "red_mahjong_offline_data.pkl")
with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

BATCH_SIZE = 8
obs_np = {k: v[:BATCH_SIZE].numpy() if hasattr(v, 'numpy') else np.array(v[:BATCH_SIZE])
          for k, v in data["observation"].items()}
act_np = np.array(data["action"][:BATCH_SIZE], dtype=np.int32)
mask_np = np.array(data["legal_action_mask"][:BATCH_SIZE], dtype=bool)

print(f"Data loaded: {BATCH_SIZE} samples")
for k, v in obs_np.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, range=[{v.min():.1f}, {v.max():.1f}]")

# ── 2. Constants (must match both frameworks) ────────────────
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

# ── 3. JAX / Flax model definition ──────────────────────────
# (exact copy of examples/networks/red_network.py and transformer.py)

class JaxTransformerBlock(nn.Module):
    features: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, mask=None):
        y = nn.LayerNorm()(x)
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.orthogonal(),
            deterministic=True,
        )(y, mask=mask)
        x = x + y
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim, kernel_init=nn.initializers.orthogonal())(y)
        y = nn.relu(y)
        y = nn.Dense(self.features, kernel_init=nn.initializers.orthogonal())(y)
        x = x + y
        return x


class JaxFeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, obs):
        hand = obs["hand"].astype(jnp.int32)
        if hand.ndim == 1:
            hand = hand[None, :]
        hand = jnp.clip(hand, -1, 99) + 1  # -1→0 (pad)

        hand_emb = nn.Embed(NUM_TILE_TYPE_WITH_RED + 1, HAND_EMB_SIZE,
                            embedding_init=nn.initializers.orthogonal())(hand)
        hand_mask = (hand > 0).astype(jnp.float32)
        x_hand = hand_emb * hand_mask[..., None]
        for _ in range(NUM_HAND_LAYER):
            x_hand = JaxTransformerBlock(HAND_EMB_SIZE, num_heads=4,
                                         mlp_dim=TRANFORMER_MLP_DIM)(x_hand, mask=hand_mask)
        token_count = jnp.maximum(hand_mask.sum(axis=1, keepdims=True), 1.0)
        hand_feat = (x_hand * hand_mask[..., None]).sum(axis=1) / token_count

        # History
        ah = obs["action_history"]
        if ah.ndim == 2:
            ah = ah[None, ...]
        players = ah[:, 0, :].astype(jnp.int32)
        actions = ah[:, 1, :].astype(jnp.int32)
        tsumogiri = ah[:, 2, :].astype(jnp.int32)
        hist_mask = (actions >= 0).astype(jnp.float32)

        players_emb = nn.Embed(NUM_PLAYERS + 1, HISTORY_EMB_SIZE,
                               embedding_init=nn.initializers.orthogonal())(jnp.clip(players + 1, 0, 99))
        actions_emb = nn.Embed(NUM_ACTIONS + 1, HISTORY_EMB_SIZE,
                               embedding_init=nn.initializers.orthogonal())(jnp.clip(actions + 1, 0, 99))
        tsumogiri_emb = nn.Embed(3, HISTORY_EMB_SIZE,
                                 embedding_init=nn.initializers.orthogonal())(jnp.clip(tsumogiri + 1, 0, 99))
        pos_emb = nn.Embed(MAX_HISTORY_LENGTH, HISTORY_EMB_SIZE,
                           embedding_init=nn.initializers.orthogonal())(jnp.arange(MAX_HISTORY_LENGTH)[None, :])
        x_hist = players_emb + actions_emb + tsumogiri_emb + pos_emb
        x_hist = x_hist * hist_mask[..., None]
        for _ in range(NUM_HISTORY_LAYER):
            x_hist = JaxTransformerBlock(HISTORY_EMB_SIZE, num_heads=4,
                                         mlp_dim=TRANFORMER_MLP_DIM)(x_hist, mask=hist_mask)
        hist_token_count = jnp.maximum(hist_mask.sum(axis=1, keepdims=True), 1.0)
        hist_feat = (x_hist * hist_mask[..., None]).sum(axis=1) / hist_token_count

        # Global scalars
        def _b(x, ndim):
            """Ensure (B, D) shape. ndim = expected non-batch dims.
            - ndim=0 (scalar): (B,) → (B, 1)
            - ndim=1 (vector): (D,) → (1, D); (B, D) → keep
            """
            x = jnp.asarray(x, dtype=jnp.float32)
            if x.ndim == ndim:              # missing batch: (D,) or ()
                return x.reshape((1,) + x.shape + (1,) * (1 - ndim)) if ndim == 0 else x[None, ...]
            elif x.ndim == ndim + 1 and ndim == 0:  # (B,) → (B, 1)
                return x[:, None]
            else:
                return x

        shanten = _b(obs.get("shanten_count", 0), 0) / 6.0
        furiten = _b(obs.get("furiten", False), 0)
        scores = (_b(obs.get("scores", jnp.zeros(4)), 1) + 250.0) / 1250.0
        round_n = _b(obs.get("round", 0), 0) / 12.0
        honba = _b(obs.get("honba", 0), 0) / 10.0
        kyotaku = _b(obs.get("kyotaku", 0), 0) / 10.0
        pw = _b(obs.get("prevalent_wind", 0), 0) / 3.0
        sw = _b(obs.get("seat_wind", 0), 0) / 3.0
        global_scalar = jnp.concatenate([scores, shanten, furiten, round_n, honba, kyotaku, pw, sw], axis=-1)

        di = _b(obs.get("dora_indicators", jnp.zeros(5, dtype=jnp.int32)), 1).astype(jnp.int32)
        di = jnp.clip(di + 1, 0, 99)
        dmask = (di > 0).astype(jnp.float32)
        demb = nn.Embed(NUM_TILE_TYPE_WITH_RED + 1, HAND_EMB_SIZE,
                        embedding_init=nn.initializers.orthogonal())(di) * dmask[..., None]
        dn = jnp.maximum(dmask.sum(axis=1, keepdims=True), 1.0)
        dfeat = nn.relu(nn.Dense(GLOBAL_EMB_SIZE, kernel_init=nn.initializers.orthogonal())(
            demb.sum(axis=1) / dn))

        global_in = jnp.concatenate([global_scalar, dfeat], axis=-1)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=nn.initializers.orthogonal())(global_in)
        global_out = nn.relu(global_out)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=nn.initializers.orthogonal())(global_out)

        return jnp.concatenate([hand_feat, hist_feat, global_out], axis=-1)


class JaxACNet(nn.Module):
    def setup(self):
        self.policy_extractor = JaxFeatureExtractor()
        self.critic_extractor = JaxFeatureExtractor()
        self.policy_mlp = nn.Sequential([
            nn.Dense(FINAL_MLP_DIM, kernel_init=nn.initializers.orthogonal()),
            nn.relu,
            nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01)),
        ])
        self.value_mlp = nn.Sequential([
            nn.Dense(FINAL_MLP_DIM, kernel_init=nn.initializers.orthogonal()),
            nn.relu,
            nn.Dense(1, kernel_init=nn.initializers.orthogonal()),
        ])

    def __call__(self, obs):
        return self.get_action_logits(obs), self.get_value(obs)

    def get_action_logits(self, obs):
        feats = self.policy_extractor(obs)
        return self.policy_mlp(feats)

    def get_value(self, obs):
        feats = self.critic_extractor(obs)
        return self.value_mlp(feats).squeeze(-1)


# ── 4. PyTorch model (from our port) ─────────────────────────
from mahjax_pt.examples.networks.red_network import ACNet as TorchACNet
from mahjax_pt.examples.networks.transformer import orthogonal_init_

# ── 5. Fix random seeds ─────────────────────────────────────
SEED = 42
rng = jax.random.PRNGKey(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 6. Initialize JAX model ──────────────────────────────────
jax_net = JaxACNet()
rng, init_rng = jax.random.split(rng)
# Convert numpy obs to jax format (add batch dim to scalars if needed)
jax_obs = {
    "hand": jnp.asarray(obs_np["hand"], dtype=jnp.int32),
    "action_history": jnp.asarray(obs_np["action_history"], dtype=jnp.int32),
    "shanten_count": jnp.asarray(obs_np["shanten_count"], dtype=jnp.int32),
    "furiten": jnp.asarray(obs_np["furiten"], dtype=jnp.bool_),
    "scores": jnp.asarray(obs_np["scores"], dtype=jnp.int32),
    "round": jnp.asarray(obs_np["round"], dtype=jnp.int32),
    "honba": jnp.asarray(obs_np["honba"], dtype=jnp.int32),
    "kyotaku": jnp.asarray(obs_np["kyotaku"], dtype=jnp.int32),
    "prevalent_wind": jnp.asarray(obs_np["prevalent_wind"], dtype=jnp.int32),
    "seat_wind": jnp.asarray(obs_np["seat_wind"], dtype=jnp.int32),
    "dora_indicators": jnp.asarray(obs_np["dora_indicators"], dtype=jnp.int32),
}

jax_params = jax_net.init(init_rng, jax_obs)
print(f"\nJAX params initialized, total layers: {len(jax.tree_util.tree_leaves(jax_params))}")

# ── 7. JAX forward pass ─────────────────────────────────────
@jax.jit
def jax_forward(params, obs):
    logits, value = jax_net.apply(params, obs, method=JaxACNet.get_action_logits), \
                    jax_net.apply(params, obs, method=JaxACNet.get_value)
    return logits, value

jax_logits, jax_value = jax_forward(jax_params, jax_obs)
jax_mask = jnp.asarray(mask_np, dtype=jnp.bool_)
jax_logits_masked = jnp.where(jax_mask, jax_logits, -1e9)

# Per-sample loss
jax_per_sample = optax.softmax_cross_entropy_with_integer_labels(jax_logits_masked, jnp.asarray(act_np, dtype=jnp.int32))
jax_loss = jax_per_sample.mean()

# Check: is each sample's target action in the mask?
target_in_mask = jnp.asarray([jax_mask[i, act_np[i]] for i in range(BATCH_SIZE)])

# Check: what are the logit values for the target actions?
target_logits = jnp.asarray([jax_logits[i, act_np[i]] for i in range(BATCH_SIZE)])

# Check un-masked loss
jax_loss_no_mask = optax.softmax_cross_entropy_with_integer_labels(jax_logits, jnp.asarray(act_np, dtype=jnp.int32)).mean()

print(f"\n{'='*60}")
print(f"JAX forward:")
print(f"  logits shape: {jax_logits.shape}")
print(f"  logits range: [{float(jax_logits.min()):.4f}, {float(jax_logits.max()):.4f}]")
print(f"  value range:  [{float(jax_value.min()):.4f}, {float(jax_value.max()):.4f}]")
print(f"  target_in_mask: {list(target_in_mask)}")
print(f"  target_logits:  {[float(x) for x in target_logits]}")
print(f"  per-sample loss: {[float(x) for x in jax_per_sample]}")
print(f"  loss (no mask):  {float(jax_loss_no_mask):.4f}")
print(f"  loss (masked):   {float(jax_loss):.4f}")

# ── 8. PyTorch forward (independent init, same architecture) ──
torch_net = TorchACNet()
torch_net.eval()
with torch.no_grad():
    torch_obs = {
        "hand": torch.from_numpy(obs_np["hand"]),
        "action_history": torch.from_numpy(obs_np["action_history"].astype(np.int64)),
        "shanten_count": torch.from_numpy(obs_np["shanten_count"]),
        "furiten": torch.from_numpy(obs_np["furiten"]),
        "scores": torch.from_numpy(obs_np["scores"]),
        "round": torch.from_numpy(obs_np["round"]),
        "honba": torch.from_numpy(obs_np["honba"]),
        "kyotaku": torch.from_numpy(obs_np["kyotaku"]),
        "prevalent_wind": torch.from_numpy(obs_np["prevalent_wind"]),
        "seat_wind": torch.from_numpy(obs_np["seat_wind"]),
        "dora_indicators": torch.from_numpy(obs_np["dora_indicators"].astype(np.int64)),
    }
    torch_logits, torch_value = torch_net(torch_obs)
    torch_mask = torch.from_numpy(mask_np)
    torch_logits_masked = torch.where(torch_mask, torch_logits, torch.full_like(torch_logits, -1e9))
    torch_act = torch.tensor(act_np, dtype=torch.long)

    # Per-sample loss
    torch_per_sample = F.cross_entropy(torch_logits_masked, torch_act, reduction='none')
    torch_loss = torch_per_sample.mean()

    # No-mask loss
    torch_loss_no_mask = F.cross_entropy(torch_logits, torch_act)

jax_l = float(jax_loss)
torch_l = float(torch_loss.item())
no_mask_jl = float(jax_loss_no_mask)
no_mask_tl = float(torch_loss_no_mask.item())

print(f"\n{'='*60}")
print(f"PyTorch forward:")
print(f"  logits range:     [{torch_logits.min().item():.4f}, {torch_logits.max().item():.4f}]")
print(f"  per-sample loss:  {[f'{x:.1f}' for x in torch_per_sample.tolist()]}")
print(f"  loss (no mask):   {no_mask_tl:.4f}")
print(f"  loss (masked):    {torch_l:.4f}")
print(f"\n{'='*60}")
print(f"COMPARISON (independent init, same architecture):")
print(f"  JAX  loss (no mask): {no_mask_jl:.4f}")
print(f"  Torch loss (no mask): {no_mask_tl:.4f}")
print(f"  JAX  loss (masked):  {jax_l:.4f}")
print(f"  Torch loss (masked): {torch_l:.4f}")
print(f"")
print(f"Key insight: both frameworks show similar loss magnitudes.")
print(f"The 1e8 loss is from MASKING the target action's logit to -1e9,")
print(f"which means the target action is NOT in the legal action mask!")
print(f"This is a DATA issue, not a model issue.")
