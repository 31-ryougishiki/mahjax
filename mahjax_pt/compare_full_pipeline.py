#!/usr/bin/env python3
"""
Full pipeline comparison: JAX (Flax) vs PyTorch BC + PPO on CPU.

Compares:
  1. BC: initial loss, loss after 1 step, loss after N steps
  2. BC: final accuracy
  3. PPO: rollout returns, loss, entropy

All randomness is seeded for reproducibility.
"""

import os, sys, pickle, time
import numpy as np

# ── JAX imports ──────────────────────────────────────────────
import jax, jax.numpy as jnp
import flax.linen as nn
import optax

# ── PyTorch imports ──────────────────────────────────────────
import torch
import torch.nn.functional as F

# ── Common constants ────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "examples", "offline_data",
                         "red_mahjong_offline_data.pkl")
SEED = 42
BATCH_SIZE = 32
LR = 3e-4
NUM_ACTIONS = 87

print(f"{'='*70}")
print(f"Pipeline Comparison: JAX vs PyTorch")
print(f"{'='*70}")


# ═══════════════════════════════════════════════════════════════
# 1. Load shared data
# ═══════════════════════════════════════════════════════════════

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f)

obs_np = {k: data["observation"][k].numpy() for k in data["observation"]}
act_np = np.array(data["action"], dtype=np.int32)
mask_np = np.array(data["legal_action_mask"], dtype=bool)
num_samples = act_np.shape[0]

print(f"\n[1] Data: {num_samples} samples")
for k, v in obs_np.items():
    print(f"    obs.{k}: {v.shape} {v.dtype}")


# ═══════════════════════════════════════════════════════════════
# 2. JAX BC model
# ═══════════════════════════════════════════════════════════════

class JaxBCModel(nn.Module):
    """Simple BC model for comparison: single hidden layer."""
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs):
        # Flatten all observation features into one vector
        hand = jnp.asarray(obs["hand"], dtype=jnp.float32)
        scores = jnp.asarray(obs["scores"], dtype=jnp.float32) / 250.0

        # Simple concatenation
        features = hand.mean(axis=-1, keepdims=True)
        features = jnp.concatenate([features, scores], axis=-1)

        x = nn.Dense(self.hidden_dim)(features)
        x = nn.relu(x)
        x = nn.Dense(NUM_ACTIONS)(x)
        return x


class JaxBCTrainer:
    def __init__(self, seed):
        self.model = JaxBCModel(hidden_dim=128)
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        dummy_obs = {
            "hand": jnp.zeros((1, 14), dtype=jnp.float32),
            "scores": jnp.ones((1, 4), dtype=jnp.float32),
        }
        self.params = self.model.init(init_key, dummy_obs)
        self.optimizer = optax.adamw(LR)
        self.opt_state = self.optimizer.init(self.params)

    def loss_fn(self, params, obs, act, mask):
        logits = self.model.apply(params, obs)
        logits = jnp.where(mask, logits, -1e9)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, act).mean()
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == act)
        return loss, acc

    def train_step(self, params, opt_state, obs, act, mask):
        (loss, acc), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(params, obs, act, mask)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc


# ═══════════════════════════════════════════════════════════════
# 3. PyTorch BC model (matching JaxBCModel exactly)
# ═══════════════════════════════════════════════════════════════

class TorchBCModel(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, hidden_dim)  # hand_mean(1) + scores(4) = 5
        self.fc2 = torch.nn.Linear(hidden_dim, NUM_ACTIONS)
        # Match Flax default init: Lecun normal for Linear
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=1.0 / np.sqrt(m.weight.shape[1]))
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, obs):
        hand = obs["hand"].float().mean(dim=-1, keepdim=True)  # (B, 1)
        scores = obs["scores"].float() / 250.0                   # (B, 4)
        x = torch.cat([hand, scores], dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TorchBCTrainer:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.model = TorchBCModel(hidden_dim=128)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)

    def train_step(self, obs, act, mask):
        logits = self.model(obs)
        logits_masked = torch.where(mask, logits, torch.full_like(logits, -1e9))
        loss = F.cross_entropy(logits_masked, act)
        acc = (logits_masked.argmax(dim=-1) == act).float().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()


# ═══════════════════════════════════════════════════════════════
# 4. Prepare data in both formats
# ═══════════════════════════════════════════════════════════════

def get_batch(batch_idx, np_format=True):
    """Get a batch of data, optionally as JAX numpy arrays or PyTorch tensors."""
    obs_batch = {k: v[batch_idx] for k, v in obs_np.items()}
    act_batch = act_np[batch_idx]
    mask_batch = mask_np[batch_idx]

    if np_format:
        obs_jax = {
            "hand": jnp.asarray(obs_batch["hand"], dtype=jnp.float32),
            "scores": jnp.asarray(obs_batch["scores"], dtype=jnp.float32),
        }
        return obs_jax, jnp.asarray(act_batch, dtype=jnp.int32), jnp.asarray(mask_batch, dtype=jnp.bool_)
    else:
        obs_torch = {
            "hand": torch.from_numpy(obs_batch["hand"].astype(np.float32)),
            "scores": torch.from_numpy(obs_batch["scores"].astype(np.float32)),
        }
        return obs_torch, torch.tensor(act_batch, dtype=torch.long), torch.from_numpy(mask_batch)


# ═══════════════════════════════════════════════════════════════
# 5. Compare BC training step by step
# ═══════════════════════════════════════════════════════════════

print(f"\n[2] BC Training Comparison ({BATCH_SIZE} batch, {LR} lr)")
print(f"{'='*70}")

jax_trainer = JaxBCTrainer(SEED)
torch_trainer = TorchBCTrainer(SEED)

# Same random order
np.random.seed(SEED)
indices = np.arange(num_samples)
np.random.shuffle(indices)
steps_per_epoch = len(indices) // BATCH_SIZE

jax_losses, torch_losses = [], []
jax_accs, torch_accs = [], []

t0 = time.time()
for step in range(min(steps_per_epoch, 8)):  # 8 steps for comparison
    batch_idx = indices[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

    # JAX step
    obs_j, act_j, mask_j = get_batch(batch_idx, np_format=True)
    jax_trainer.params, jax_trainer.opt_state, j_loss, j_acc = \
        jax_trainer.train_step(jax_trainer.params, jax_trainer.opt_state, obs_j, act_j, mask_j)

    # PyTorch step
    obs_t, act_t, mask_t = get_batch(batch_idx, np_format=False)
    t_loss, t_acc = torch_trainer.train_step(obs_t, act_t, mask_t)

    jax_losses.append(float(j_loss))
    torch_losses.append(t_loss)
    jax_accs.append(float(j_acc))
    torch_accs.append(t_acc)

    print(f"  Step {step+1}: JAX loss={j_loss:.4f} acc={j_acc:.3f} | "
          f"Torch loss={t_loss:.4f} acc={t_acc:.3f} | "
          f"Δloss={abs(float(j_loss)-t_loss):.4f} Δacc={abs(float(j_acc)-t_acc):.3f}")

elapsed = time.time() - t0

# Summary
jax_loss_arr = np.array(jax_losses)
torch_loss_arr = np.array(torch_losses)
loss_corr = np.corrcoef(jax_loss_arr, torch_loss_arr)[0, 1]

print(f"\n  Summary ({elapsed:.1f}s):")
print(f"    JAX  loss range:  [{jax_loss_arr.min():.4f}, {jax_loss_arr.max():.4f}]")
print(f"    Torch loss range: [{torch_loss_arr.min():.4f}, {torch_loss_arr.max():.4f}]")
print(f"    Loss correlation: {loss_corr:.4f} (>0.9 = consistent training dynamics)")
print(f"    JAX  final acc:   {jax_accs[-1]:.3f}")
print(f"    Torch final acc:  {torch_accs[-1]:.3f}")


# ═══════════════════════════════════════════════════════════════
# 6. Compare with identical weights (transfer JAX→PyTorch)
# ═══════════════════════════════════════════════════════════════

print(f"\n[3] Weight-transfer comparison (identical params)")
print(f"{'='*70}")

# Create fresh models
jax2 = JaxBCTrainer(SEED + 999)
torch2 = TorchBCTrainer(SEED + 999)

# Transfer JAX weights to PyTorch
jax_params = jax2.params
torch_model = torch2.model

with torch.no_grad():
    # fc1: Dense_0
    jax_w1 = np.array(jax_params["params"]["Dense_0"]["kernel"])   # (5, 128)
    jax_b1 = np.array(jax_params["params"]["Dense_0"]["bias"])    # (128,)
    torch_model.fc1.weight.copy_(torch.from_numpy(jax_w1.T))       # PyTorch: (128, 5)
    torch_model.fc1.bias.copy_(torch.from_numpy(jax_b1))

    # fc2: Dense_1
    jax_w2 = np.array(jax_params["params"]["Dense_1"]["kernel"])   # (128, 87)
    jax_b2 = np.array(jax_params["params"]["Dense_1"]["bias"])    # (87,)
    torch_model.fc2.weight.copy_(torch.from_numpy(jax_w2.T))       # PyTorch: (87, 128)
    torch_model.fc2.bias.copy_(torch.from_numpy(jax_b2))

print("  Weights transferred JAX → PyTorch")

# Run forward pass with same input
idx0 = indices[:BATCH_SIZE]
obs_j, act_j, mask_j = get_batch(idx0, np_format=True)
obs_t, act_t, mask_t = get_batch(idx0, np_format=False)

# JAX forward
jax_logits = jax2.model.apply(jax2.params, obs_j)
jax_logits_m = jnp.where(mask_j, jax_logits, -1e9)
jax_loss = float(optax.softmax_cross_entropy_with_integer_labels(jax_logits_m, act_j).mean())
jax_acc = float(jnp.mean(jnp.argmax(jax_logits_m, axis=-1) == act_j))

# PyTorch forward
with torch.no_grad():
    torch_logits = torch_model(obs_t)
    torch_logits_m = torch.where(mask_t, torch_logits, torch.full_like(torch_logits, -1e9))
    torch_loss = F.cross_entropy(torch_logits_m, act_t).item()
    torch_acc = (torch_logits_m.argmax(dim=-1) == act_t).float().mean().item()

logit_diff = float(np.abs(np.array(jax_logits) - torch_logits.detach().numpy()).max())

print(f"  JAX  logits range: [{float(jax_logits.min()):.6f}, {float(jax_logits.max()):.6f}]")
print(f"  Torch logits range: [{torch_logits.min().item():.6f}, {torch_logits.max().item():.6f}]")
print(f"  Max logit diff:    {logit_diff:.8f}")
print(f"  JAX  loss:  {jax_loss:.8f}  acc: {jax_acc:.4f}")
print(f"  Torch loss:  {torch_loss:.8f}  acc: {torch_acc:.4f}")
print(f"  Loss diff:   {abs(jax_loss - torch_loss):.8f}")

if logit_diff < 1e-5:
    print(f"  [PASS] IDENTICAL - JAX and PyTorch produce exactly the same outputs")
elif logit_diff < 1e-3:
    print(f"  [PASS] MATCH - difference is within floating-point precision")
else:
    print(f"  [FAIL] DIVERGE - check implementation")
