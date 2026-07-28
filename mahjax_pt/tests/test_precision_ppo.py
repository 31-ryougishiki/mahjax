"""Precision tests: PPO loss, gradient, and optimizer step via Golden data replay.

Uses a simple shared MLP (no Transformer/ACNet) to avoid weight-mapping complexity.
Same weights + same batch data → compare loss/metrics/gradients/updated_params.

Prerequisites:
    Golden .npz files in tests/golden/ recorded by scripts/record_golden_ppo.py
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from conftest import has_golden, load_golden
from test_helpers import TinyACNet


# ═══════════════════════════════════════════════════════════════════════
# PT PPO loss computation (mirrors ppo_with_reg.py)
# ═══════════════════════════════════════════════════════════════════════

CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
NEG = -1e9


def masked_mean(x, mask):
    denom = mask.float().sum().clamp(min=1.0)
    return (x * mask.float()).sum() / denom


def compute_ppo_loss_and_metrics(network, obs, actions, old_log_probs,
                                  advantages, targets, valid_mask, action_mask,
                                  old_values, current_players):
    """Compute PPO loss and all diagnostics (no backward, no optimizer step)."""
    logits, values_new = network(obs)
    logits = torch.where(action_mask, logits, torch.full_like(logits, NEG))
    dist = torch.distributions.Categorical(logits=logits)
    logp_new = dist.log_prob(actions)
    entropy = dist.entropy()

    log_ratio = logp_new - old_log_probs
    ratio = torch.exp(log_ratio).unsqueeze(-1)

    adv = advantages.gather(1, current_players.unsqueeze(-1))
    vmask = valid_mask.float().gather(1, current_players.unsqueeze(-1))

    clip_adv = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
    ppo_loss = -masked_mean(torch.min(ratio * adv, clip_adv), vmask)

    vt = values_new.unsqueeze(-1)
    val_clipped = old_values.unsqueeze(-1) + torch.clamp(
        vt - old_values.unsqueeze(-1), -CLIP_EPS, CLIP_EPS)
    tgt = targets.gather(1, current_players.unsqueeze(-1))
    loss_critic = 0.5 * VF_COEF * masked_mean(
        torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)

    approx_kl = masked_mean((ratio - 1.0) - log_ratio.unsqueeze(-1), vmask)
    clip_frac = masked_mean(
        (torch.abs(ratio - 1.0) > CLIP_EPS).float(), vmask)
    explained_var = torch.clamp(
        1.0 - masked_mean((tgt - vt) ** 2, vmask)
        / (masked_mean((tgt - masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8),
        min=0.0)

    total_loss = (ppo_loss
                  - ENT_COEF * masked_mean(entropy.unsqueeze(-1), vmask)
                  + loss_critic)

    return total_loss, {
        "total_loss": total_loss.item(),
        "actor_loss": ppo_loss.item(),
        "critic_loss": loss_critic.item(),
        "entropy": masked_mean(entropy.unsqueeze(-1), vmask).item(),
        "approx_kl": approx_kl.item(),
        "clip_frac": clip_frac.item(),
        "explained_var": explained_var.item(),
    }


def _load_network_from_golden(golden, network: TinyACNet):
    """Load network weights from golden dict into a TinyACNet."""
    # Golden stores weights as flat arrays with shape info
    param_map = {
        "fc1.weight": network.fc1.weight,
        "fc1.bias": network.fc1.bias,
        "fc2.weight": network.fc2.weight,
        "fc2.bias": network.fc2.bias,
        "actor.weight": network.actor.weight,
        "actor.bias": network.actor.bias,
        "critic.weight": network.critic.weight,
        "critic.bias": network.critic.bias,
    }
    with torch.no_grad():
        for name, param in param_map.items():
            if name in golden:
                param.data.copy_(torch.from_numpy(golden[name]))


def _get_network_weights(network: TinyACNet) -> dict:
    """Export network weights as numpy dict (for comparison with golden)."""
    return {
        "fc1.weight": network.fc1.weight.data.numpy().copy(),
        "fc1.bias": network.fc1.bias.data.numpy().copy(),
        "fc2.weight": network.fc2.weight.data.numpy().copy(),
        "fc2.bias": network.fc2.bias.data.numpy().copy(),
        "actor.weight": network.actor.weight.data.numpy().copy(),
        "actor.bias": network.actor.bias.data.numpy().copy(),
        "critic.weight": network.critic.weight.data.numpy().copy(),
        "critic.bias": network.critic.bias.data.numpy().copy(),
    }


# ═══════════════════════════════════════════════════════════════════════
# PA15: PPO loss & metrics precision
# ═══════════════════════════════════════════════════════════════════════

def test_ppo_loss_precision():
    """PA15: PPO loss and 7 metrics against golden (same weights, same batch)."""
    if not has_golden("ppo_loss_mlp.npz"):
        pytest.skip("Golden file ppo_loss_mlp.npz not found")

    golden = load_golden("ppo_loss_mlp")

    # Build network and load golden weights
    input_dim = int(golden.get("input_dim", 32))
    hidden_dim = int(golden.get("hidden_dim", 64))
    num_actions = int(golden.get("num_actions", 87))

    network = TinyACNet(input_dim=input_dim, hidden_dim=hidden_dim,
                        num_actions=num_actions)
    _load_network_from_golden(golden, network)

    # Load batch data from golden
    obs = torch.from_numpy(golden["obs"])
    actions = torch.from_numpy(golden["actions"].astype(np.int64))
    old_log_probs = torch.from_numpy(golden["old_log_probs"])
    advantages = torch.from_numpy(golden["advantages"])
    targets = torch.from_numpy(golden["targets"])
    valid_mask = torch.from_numpy(golden["valid_mask"])
    action_mask = torch.from_numpy(golden["action_mask"])
    old_values = torch.from_numpy(golden["old_values"])
    current_players = torch.from_numpy(golden["current_players"].astype(np.int64))

    # Compute PT loss
    _, metrics = compute_ppo_loss_and_metrics(
        network, obs, actions, old_log_probs, advantages, targets,
        valid_mask, action_mask, old_values, current_players)

    # Compare each metric
    metric_keys = ["total_loss", "actor_loss", "critic_loss", "entropy",
                   "approx_kl", "clip_frac", "explained_var"]
    failures = []
    for key in metric_keys:
        expected = float(golden[key])
        actual = metrics[key]
        diff = abs(expected - actual)
        rel = diff / max(abs(expected), 1e-8)
        if rel > 1e-4:
            failures.append(
                f"{key}: golden={expected:.8f} pt={actual:.8f} "
                f"diff={diff:.2e} rel={rel:.2e}")

    if failures:
        pytest.fail("PPO loss precision failures:\n" + "\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA16: PPO gradient precision
# ═══════════════════════════════════════════════════════════════════════

def test_ppo_gradient_precision():
    """PA16: PPO gradients against golden (same weights, same batch)."""
    if not has_golden("ppo_grad_mlp.npz"):
        pytest.skip("Golden file ppo_grad_mlp.npz not found")

    golden = load_golden("ppo_grad_mlp")

    input_dim = int(golden.get("input_dim", 32))
    hidden_dim = int(golden.get("hidden_dim", 64))
    num_actions = int(golden.get("num_actions", 87))

    network = TinyACNet(input_dim=input_dim, hidden_dim=hidden_dim,
                        num_actions=num_actions)
    _load_network_from_golden(golden, network)

    obs = torch.from_numpy(golden["obs"])
    actions = torch.from_numpy(golden["actions"].astype(np.int64))
    old_log_probs = torch.from_numpy(golden["old_log_probs"])
    advantages = torch.from_numpy(golden["advantages"])
    targets = torch.from_numpy(golden["targets"])
    valid_mask = torch.from_numpy(golden["valid_mask"])
    action_mask = torch.from_numpy(golden["action_mask"])
    old_values = torch.from_numpy(golden["old_values"])
    current_players = torch.from_numpy(golden["current_players"].astype(np.int64))

    # Compute loss and backward
    total_loss, _ = compute_ppo_loss_and_metrics(
        network, obs, actions, old_log_probs, advantages, targets,
        valid_mask, action_mask, old_values, current_players)
    total_loss.backward()

    # Compare each param's gradient
    grad_map = {
        "fc1.weight": network.fc1.weight,
        "fc1.bias": network.fc1.bias,
        "fc2.weight": network.fc2.weight,
        "fc2.bias": network.fc2.bias,
        "actor.weight": network.actor.weight,
        "actor.bias": network.actor.bias,
        "critic.weight": network.critic.weight,
        "critic.bias": network.critic.bias,
    }

    failures = []
    for name, param in grad_map.items():
        golden_key = f"grad_{name}"
        if golden_key not in golden:
            continue
        expected_grad = golden[golden_key]

        if param.grad is None:
            failures.append(f"{name}: no gradient")
            continue

        actual_grad = param.grad.numpy()
        diff = np.abs(expected_grad.astype(np.float64) - actual_grad.astype(np.float64))
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())

        if max_diff > 1e-4:
            failures.append(
                f"{name}: max_grad_diff={max_diff:.2e} mean={mean_diff:.2e}")

    if failures:
        pytest.fail("PPO gradient precision failures:\n" + "\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA17: Optimizer step precision
# ═══════════════════════════════════════════════════════════════════════

def test_ppo_optimizer_step_precision():
    """PA17: 1-step AdamW optimizer update against golden.

    Uses OptaxAlignedAdamW for exact JAX optax parity.
    """
    if not has_golden("ppo_optim_step_mlp.npz"):
        pytest.skip("Golden file ppo_optim_step_mlp.npz not found")

    golden = load_golden("ppo_optim_step_mlp")

    input_dim = int(golden.get("input_dim", 32))
    hidden_dim = int(golden.get("hidden_dim", 64))
    num_actions = int(golden.get("num_actions", 87))

    network = TinyACNet(input_dim=input_dim, hidden_dim=hidden_dim,
                        num_actions=num_actions)
    _load_network_from_golden(golden, network)

    # Load batch data
    obs = torch.from_numpy(golden["obs"])
    actions = torch.from_numpy(golden["actions"].astype(np.int64))
    old_log_probs = torch.from_numpy(golden["old_log_probs"])
    advantages = torch.from_numpy(golden["advantages"])
    targets = torch.from_numpy(golden["targets"])
    valid_mask = torch.from_numpy(golden["valid_mask"])
    action_mask = torch.from_numpy(golden["action_mask"])
    old_values = torch.from_numpy(golden["old_values"])
    current_players = torch.from_numpy(golden["current_players"].astype(np.int64))

    lr = float(golden.get("lr", 3e-4))
    eps = float(golden.get("eps", 1e-5))

    # Use OptaxAlignedAdamW for exact parity
    try:
        from mahjax_pt.red_mahjong.alignment import OptaxAlignedAdamW
        optimizer = OptaxAlignedAdamW(network.parameters(), lr=lr, eps=eps,
                                       weight_decay=0.0)
    except ImportError:
        # Fallback: use torch AdamW and note that tiny differences are expected
        optimizer = torch.optim.AdamW(network.parameters(), lr=lr, eps=eps,
                                       weight_decay=0.0)

    # Forward + backward
    total_loss, _ = compute_ppo_loss_and_metrics(
        network, obs, actions, old_log_probs, advantages, targets,
        valid_mask, action_mask, old_values, current_players)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Compare updated weights
    failures = []
    weight_map = {
        "updated_fc1.weight": network.fc1.weight,
        "updated_fc1.bias": network.fc1.bias,
        "updated_fc2.weight": network.fc2.weight,
        "updated_fc2.bias": network.fc2.bias,
        "updated_actor.weight": network.actor.weight,
        "updated_actor.bias": network.actor.bias,
        "updated_critic.weight": network.critic.weight,
        "updated_critic.bias": network.critic.bias,
    }

    for name, param in weight_map.items():
        if name not in golden:
            continue
        expected = golden[name]
        actual = param.data.numpy()
        diff = np.abs(expected.astype(np.float64) - actual.astype(np.float64))
        max_diff = float(diff.max())

        # Tighter tolerance for optim step: 1e-6 with aligned optimizer
        tol = 1e-6 if "OptaxAligned" in type(optimizer).__name__ else 1e-5
        if max_diff > tol:
            failures.append(
                f"{name}: max_diff={max_diff:.2e} > tol={tol}")

    if failures:
        pytest.fail("PPO optimizer step precision failures:\n" + "\n".join(failures))
