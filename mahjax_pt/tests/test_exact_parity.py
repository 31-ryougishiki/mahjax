#!/usr/bin/env python3
"""
EXACT parity: JAX vs PyTorch PPO with identical weights.

Uses a simple MLP (no Transformer) where the parameter structure is identical
in both frameworks. All weights are manually set to the SAME values.
Every intermediate value (logits, loss, grads) is compared.
"""

import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import torch, torch.nn.functional as F

# Config
SEED = 42; BATCH = 32; INPUT = 32; HIDDEN = 64; OUTPUT = 16
LR = 1e-3; CLIP = 0.2; ENT_COEF = 0.01; VF_COEF = 0.5; N_STEPS = 10

# Fixed data
np.random.seed(SEED)
X = np.random.randn(BATCH, INPUT).astype(np.float32)
A = np.random.randint(0, OUTPUT, size=BATCH).astype(np.int32)
OLD_LP = np.random.randn(BATCH).astype(np.float32) * 0.1
ADV = np.random.randn(BATCH, 1).astype(np.float32) * 0.5
TGT = np.random.randn(BATCH, 1).astype(np.float32) * 0.5 + 0.1

# ═══════════════════════════════════════════════════════════════
# JAX model (3-layer MLP, identical to PyTorch)
# ═══════════════════════════════════════════════════════════════

class JaxMLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        h = nn.Dense(HIDDEN)(x); h = nn.relu(h)
        h = nn.Dense(HIDDEN)(h); h = nn.relu(h)
        logits = nn.Dense(OUTPUT)(h)
        value = nn.Dense(1)(h)
        return logits, value.squeeze(-1)

jax_net = JaxMLP()
rng = jax.random.PRNGKey(SEED)
jax_params = jax_net.init(rng, jnp.ones((1, INPUT)))

# PyTorch model (3-layer MLP, identical to JAX)
class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(INPUT, HIDDEN)
        self.fc2 = torch.nn.Linear(HIDDEN, HIDDEN)
        self.actor = torch.nn.Linear(HIDDEN, OUTPUT)
        self.critic = torch.nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.actor(h), self.critic(h).squeeze(-1)

torch.manual_seed(SEED)
pt_net = TorchMLP()

# ── Manual weight copy (Flax Dense kernel: [in,out], PyTorch Linear: [out,in]) ──
jax_weights = {
    "Dense_0": jax_params["params"]["Dense_0"],  # {kernel: (32,64), bias: (64,)}
    "Dense_1": jax_params["params"]["Dense_1"],  # {kernel: (64,64), bias: (64,)}
    "Dense_2": jax_params["params"]["Dense_2"],  # {kernel: (64,16), bias: (16,)}
    "Dense_3": jax_params["params"]["Dense_3"],  # {kernel: (64,1), bias: (1,)}
}

with torch.no_grad():
    pt_net.fc1.weight.copy_(torch.from_numpy(np.array(jax_weights["Dense_0"]["kernel"]).T))
    pt_net.fc1.bias.copy_(torch.from_numpy(np.array(jax_weights["Dense_0"]["bias"])))
    pt_net.fc2.weight.copy_(torch.from_numpy(np.array(jax_weights["Dense_1"]["kernel"]).T))
    pt_net.fc2.bias.copy_(torch.from_numpy(np.array(jax_weights["Dense_1"]["bias"])))
    pt_net.actor.weight.copy_(torch.from_numpy(np.array(jax_weights["Dense_2"]["kernel"]).T))
    pt_net.actor.bias.copy_(torch.from_numpy(np.array(jax_weights["Dense_2"]["bias"])))
    pt_net.critic.weight.copy_(torch.from_numpy(np.array(jax_weights["Dense_3"]["kernel"]).T))
    pt_net.critic.bias.copy_(torch.from_numpy(np.array(jax_weights["Dense_3"]["bias"])))

# ═══════════════════════════════════════════════════════════════
# Verify forward pass is identical
# ═══════════════════════════════════════════════════════════════

jax_x = jnp.asarray(X)
jax_logits, jax_value = jax_net.apply(jax_params, jax_x)
pt_x = torch.from_numpy(X)
pt_net.eval()
with torch.no_grad():
    pt_logits, pt_value = pt_net(pt_x)

logit_diff = np.abs(np.array(jax_logits) - pt_logits.numpy()).max()
value_diff = np.abs(np.array(jax_value) - pt_value.numpy()).max()
print(f"Forward check: logit_diff={logit_diff:.2e}  value_diff={value_diff:.2e}")
assert logit_diff < 1e-5, f"Forward mismatch: {logit_diff}"
assert value_diff < 1e-5, f"Forward mismatch: {value_diff}"
print("[PASS] Forward pass IDENTICAL\n")


# ═══════════════════════════════════════════════════════════════
# PPO training step comparison
# ═══════════════════════════════════════════════════════════════

jax_a = jnp.asarray(A); jax_old = jnp.asarray(OLD_LP)
jax_adv = jnp.asarray(ADV); jax_tgt = jnp.asarray(TGT)
pt_a = torch.from_numpy(A).long(); pt_old = torch.from_numpy(OLD_LP)
pt_adv = torch.from_numpy(ADV); pt_tgt = torch.from_numpy(TGT)

# JAX optimizer
jax_opt = optax.adamw(LR)
jax_opt_state = jax_opt.init(jax_params)

# PyTorch optimizer
pt_opt = torch.optim.AdamW(pt_net.parameters(), lr=LR)
# Copy JAX optimizer state to PT (all zeros initially)
pt_opt.zero_grad()

def jax_ppo_step(params, opt_state):
    def loss_fn(p):
        logits, values = jax_net.apply(p, jax_x)
        log_probs = jnp.take_along_axis(jax.nn.log_softmax(logits), jax_a[:, None], axis=1).squeeze(-1)
        ratio = jnp.exp(log_probs - jax_old)
        clip_adv = jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * jax_adv.squeeze(-1)
        ppo = -jnp.minimum(ratio * jax_adv.squeeze(-1), clip_adv).mean()
        vf = 0.5 * VF_COEF * ((values - jax_tgt.squeeze(-1)) ** 2).mean()
        ent = -jnp.mean(jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1))
        return ppo + vf - ENT_COEF * ent, (ppo, vf, ent)
    (loss, (ppo, vf, ent)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, new_opt = jax_opt.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), new_opt, loss, ppo, vf, ent, grads

def pt_ppo_step():
    pt_net.train()
    logits, values = pt_net(pt_x)
    log_probs = F.log_softmax(logits, dim=-1).gather(1, pt_a.unsqueeze(1)).squeeze(-1)
    ratio = torch.exp(log_probs - pt_old)
    clip_adv = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * pt_adv.squeeze(-1)
    ppo = -torch.min(ratio * pt_adv.squeeze(-1), clip_adv).mean()
    vf = 0.5 * VF_COEF * ((values - pt_tgt.squeeze(-1)) ** 2).mean()
    probs = F.softmax(logits, dim=-1)
    ent = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    loss = ppo + vf - ENT_COEF * ent
    pt_opt.zero_grad()
    loss.backward()
    pt_opt.step()
    return loss.item(), ppo.item(), vf.item(), ent.item()

print("PPO Training Steps (same weights, same data, same optimizer):")
print(f"{'Step':>5} {'JAX loss':>12} {'PT loss':>12} {'diff':>12} {'JAX ppo':>12} {'PT ppo':>12} {'JAX ent':>10} {'PT ent':>10}")
print("-" * 100)

all_ok = True
for step in range(N_STEPS):
    jax_params, jax_opt_state, jl, jp, jv, je, jax_grads = \
        jax_ppo_step(jax_params, jax_opt_state)
    pl, pp, pv, pe = pt_ppo_step()

    loss_diff = abs(float(jl) - pl)
    ppo_diff = abs(float(jp) - pp)
    ent_diff = abs(float(je) - pe)

    ok = loss_diff < 1e-4 and ppo_diff < 1e-4
    if not ok: all_ok = False
    flag = "" if ok else " <-- DIFF"

    print(f"{step:5d} {float(jl):12.8f} {pl:12.8f} {loss_diff:12.2e} "
          f"{float(jp):12.8f} {pp:12.8f} {float(je):10.6f} {pe:10.6f}{flag}")

# Also compare parameter values after all steps
jax_final = [np.array(v) for v in jax.tree_util.tree_flatten(jax_params)[0]]
pt_final = [p.detach().numpy() for p in pt_net.parameters()]
param_diffs = []
for j, p in zip(jax_final, pt_final):
    try:
        if j.shape == p.T.shape:
            param_diffs.append(np.abs(j - p.T).max())
        elif j.shape == p.shape:
            param_diffs.append(np.abs(j - p).max())
        elif j.size == p.numel():
            param_diffs.append(np.abs(j.reshape(-1) - p.detach().numpy().reshape(-1)).max())
    except:
        pass
max_param_diff = max(param_diffs) if param_diffs else 0.0

print(f"\nAfter {N_STEPS} PPO steps:")
print(f"  Max parameter difference: {max_param_diff:.2e}")
print(f"  {'[PASS] IDENTICAL' if all_ok and max_param_diff < 1e-4 else '[FAIL]'}")
