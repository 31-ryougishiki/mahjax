"""Performance tests: PPO pipeline component benchmarks (CPU only).

Measures GAE, PPO update, and full mini-round latency.
"""

import pytest
import torch

from conftest import check_perf, cpu_device
from test_helpers import (
    timeit_with_arg,
    make_gae_inputs,
    make_ppo_batch,
    TinyACNet,
)
from test_precision_gae import pt_compute_gae_vectorized
from test_precision_ppo import compute_ppo_loss_and_metrics

REGRESSION_THRESHOLD = 0.30


def _check_and_report(name: str, result: dict):
    mean_ms = result["mean_ms"]
    failure = check_perf(name, mean_ms, REGRESSION_THRESHOLD)
    if failure:
        pytest.fail(failure)


# ═══════════════════════════════════════════════════════════════════════
# PB15: GAE throughput
# ═══════════════════════════════════════════════════════════════════════

def test_perf_gae(cpu_device):
    """PB15: compute_gae_vectorized T=256,B=128,P=4 latency."""
    T, B, P = 256, 128, 4
    inputs = make_gae_inputs(T, B, P)

    def bench():
        pt_compute_gae_vectorized(
            inputs["rewards"], inputs["values"],
            inputs["dones"], inputs["current_players"])

    result = timeit_with_arg(bench, warmup=2, repeat=10)
    _check_and_report("gae_T256_B128", result)


# ═══════════════════════════════════════════════════════════════════════
# PB16: PPO update (small network)
# ═══════════════════════════════════════════════════════════════════════

def test_perf_ppo_update_small(cpu_device):
    """PB16: Single PPO update step (small MLP 32→64→87, batch=4096)."""
    batch = make_ppo_batch(batch_size=4096, feature_dim=32, num_actions=87)
    network = TinyACNet(input_dim=32, hidden_dim=64, num_actions=87)

    def bench():
        total_loss, _ = compute_ppo_loss_and_metrics(
            network, batch["obs"], batch["actions"], batch["old_log_probs"],
            batch["advantages"], batch["targets"], batch["valid_mask"],
            batch["action_mask"], batch["old_values"], batch["current_players"])
        total_loss.backward()

    result = timeit_with_arg(bench, warmup=2, repeat=8)
    _check_and_report("ppo_update_small_mlp", result)


# ═══════════════════════════════════════════════════════════════════════
# PB17: PPO update (ACNet)
# ═══════════════════════════════════════════════════════════════════════

def test_perf_ppo_update_acnet(cpu_device):
    """PB17: Single PPO update step (full ACNet, batch=128)."""
    try:
        from mahjax_pt.examples.networks.red_network import ACNet
        from mahjax_pt.red_mahjong.observation import _observe_dict
        from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial
    except ImportError as e:
        pytest.skip(f"ACNet import failed: {e}")

    batch_size = 128
    env = RedMahjongSerial(round_mode="single", observe_type="dict")
    network = ACNet()

    # Collect B observations from serial env
    obs_list = []
    for i in range(batch_size):
        state = env.init(key=i)
        obs_list.append(env.observe(state))

    # Stack into batch dict
    obs_batch = {}
    for key in obs_list[0].keys():
        vals = [o[key] for o in obs_list]
        if isinstance(vals[0], torch.Tensor):
            obs_batch[key] = torch.stack(vals)
        else:
            obs_batch[key] = torch.tensor(vals)

    batch = make_ppo_batch(batch_size=batch_size, feature_dim=None)

    def bench():
        logits, values = network(obs_batch)
        # Simple cross-entropy-style loss (approximates PPO loss cost)
        loss = torch.nn.functional.cross_entropy(
            logits, batch["actions"])
        loss.backward()

    result = timeit_with_arg(bench, warmup=2, repeat=5)
    _check_and_report("ppo_update_acnet_B128", result)


# ═══════════════════════════════════════════════════════════════════════
# PB18: Full PPO mini-round (rollout + GAE + update)
# ═══════════════════════════════════════════════════════════════════════

def test_perf_ppo_mini_round(cpu_device):
    """PB18: Full PPO round: B=16,T=32, small MLP, 1 update epoch."""
    from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel

    T, B, P = 32, 16, 4
    env = RedMahjongParallel(round_mode="single")
    network = TinyACNet(input_dim=32, hidden_dim=64, num_actions=87)

    def bench():
        bs = env.init_batch(num_envs=B, device=cpu_device)
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        dones_list = []
        cps_list = []

        # Rollout
        for t in range(T):
            is_new = bs.terminated | bs.truncated
            bs = env.reinit_terminated_batch(bs)
            obs = env.observe_batch(bs)

            # Use a tiny network forward
            if "shanten_count" in obs:
                feat = obs["shanten_count"].float().unsqueeze(-1)
            else:
                feat = torch.randn(B, 32)
            logits, values = network(feat)
            mask = bs.legal_action_mask
            logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

            cp = bs.current_player.clone()
            bs = env.step_batch(bs, actions)

            actions_list.append(actions)
            log_probs_list.append(log_probs)
            values_list.append(values)
            rewards_list.append(bs.rewards.clone() / 320.0)
            dones_list.append(is_new)
            cps_list.append(cp)

        # GAE
        rewards_t = torch.stack(rewards_list)
        values_t = torch.stack(values_list)
        dones_t = torch.stack(dones_list)
        cps_t = torch.stack(cps_list)

        pt_compute_gae_vectorized(rewards_t, values_t, dones_t, cps_t)

    result = timeit_with_arg(bench, warmup=1, repeat=3)
    _check_and_report("ppo_mini_round_B16_T32", result)
