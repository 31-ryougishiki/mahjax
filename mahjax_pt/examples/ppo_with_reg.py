#!/usr/bin/env python3
"""PPO + Magnet regularization trainer (PyTorch port).

Ported from examples/ppo_with_reg.py.
"""

import os
import sys
import time
import pickle
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("ppo")

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.auto_reset_wrapper import auto_reset
from mahjax_pt.red_mahjong.players import rule_based_player
from mahjax_pt.red_mahjong.action import Action
from mahjax_pt.examples.common import (
    default_bc_params_path,
    default_rl_params_path,
    get_network_cls,
)

MAX_REWARD = 320.0
NEG = -1e9


def masked_mean(x, mask):
    """Compute mean over entries where mask is True."""
    denom = mask.float().sum().clamp(min=1.0)
    return (x * mask.float()).sum() / denom


def compute_gae(rewards, values, dones, current_players, gamma=1.0, gae_lambda=0.95):
    """Compute GAE with per-player reward accumulators (turn-based MARL).

    rewards:  (T, B, 4) — 4-player reward vector
    values:   (T, B)    — value estimate for the acting player
    dones:    (T, B)    — episode boundary flags
    current_players: (T, B) — which player acted at each step
    """
    T, B, P = rewards.shape
    advantages = torch.zeros(T, B, P)
    targets = torch.zeros(T, B, P)
    valid_mask = torch.zeros(T, B, P, dtype=torch.bool)

    for b in range(B):
        gae = torch.zeros(P)
        next_value = torch.zeros(P)
        reward_accum = torch.zeros(P)
        has_next_value = torch.zeros(P, dtype=torch.bool)
        next_valid = torch.zeros(P, dtype=torch.bool)

        for t in reversed(range(T)):
            cp = int(current_players[t, b])
            d = dones[t, b]

            if d:
                gae.zero_()
                reward_accum.zero_()
                has_next_value.zero_()
                next_value.zero_()

            reward_accum += rewards[t, b]
            player_reward = reward_accum[cp].clone()
            reward_accum[cp] = 0.0

            td_error = player_reward + gamma * next_value[cp] * (1 - float(d)) - values[t, b]
            new_gae = td_error + gamma * gae_lambda * gae[cp] * (1 - float(d))
            gae[cp] = new_gae

            is_valid = has_next_value[cp] or d or next_valid[cp]
            advantages[t, b, cp] = new_gae if is_valid else 0.0
            targets[t, b, cp] = (advantages[t, b, cp] + values[t, b]) if is_valid else values[t, b]
            valid_mask[t, b, cp] = is_valid

            next_value[cp] = values[t, b]
            has_next_value[cp] = True
            next_valid[cp] = is_valid or d

    return advantages, targets, valid_mask


class PPOBuffer:
    def __init__(self, num_steps, num_envs):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.current_players = []
        self.action_masks = []

    def store(self, obs, action, log_prob, value, reward, done, cp, mask):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.current_players.append(cp)
        self.action_masks.append(mask)

    def get_batch(self):
        """Return stacked tensors from buffer.

        observations: list[T] of list[B] of dict
        actions:     list[T] of list[B] of int
        log_probs:   list[T] of list[B] of float
        values:      list[T] of list[B] of float
        rewards:     list[T] of list[B] of (4,) tensor
        dones:       list[T] of list[B] of bool
        current_players: list[T] of list[B] of int or tensor
        action_masks: list[T] of list[B] of (87,) tensor
        """
        T, B = self.num_steps, self.num_envs

        # Observations → {key: (T, B, ...)}
        keys = list(self.observations[0][0].keys())
        stacked_obs = {}
        for key in keys:
            vals = []
            for t in range(T):
                step_vals = []
                for b in range(B):
                    v = self.observations[t][b][key]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    step_vals.append(v)
                vals.append(torch.stack(step_vals))
            stacked_obs[key] = torch.stack(vals)

        def _stack_scalars(nested_list, dtype=torch.float32):
            """Convert list[T][B] of scalars → (T, B) tensor."""
            arr = torch.zeros(T, B, dtype=dtype)
            for t in range(T):
                for b in range(B):
                    arr[t, b] = float(nested_list[t][b])
            return arr

        def _stack_tensors(nested_list):
            """Convert list[T][B] of tensors → (T, B, ...) tensor."""
            return torch.stack([torch.stack(row) for row in nested_list])

        actions = _stack_scalars(self.actions, dtype=torch.long)
        log_probs = _stack_scalars(self.log_probs, dtype=torch.float32)
        values = _stack_scalars(self.values, dtype=torch.float32)
        rewards = _stack_tensors(self.rewards)
        dones = _stack_scalars(self.dones, dtype=torch.bool)
        current_players = _stack_scalars(self.current_players, dtype=torch.long)
        masks = _stack_tensors(self.action_masks)

        return stacked_obs, actions, log_probs, values, rewards, dones, current_players, masks


def train_ppo(
    env_name="red_mahjong",
    round_mode="single",
    seed=0,
    num_envs=4,
    num_steps=256,
    total_timesteps=100_000,
    gamma=1.0,
    gae_lambda=0.95,
    lr=3e-4,
    ent_coef=0.01,
    clip_eps=0.2,
    vf_coef=0.5,
    update_epochs=4,
    minibatch_size=256,
    mag_coef=0.2,
    pretrained_model_path=None,
    save_model=True,
    eval_interval=10,
    eval_num_envs=100,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "npu":
        import torch_npu

    if pretrained_model_path is None:
        pretrained_model_path = default_bc_params_path(env_name)

    # Environment
    env = make_env(env_name, round_mode=round_mode, observe_type="dict")
    step_fn = auto_reset(env.step, env.init)
    NUM_PLAYERS = env.num_players
    NUM_ACTIONS = env.num_actions
    BATCH_SIZE = num_envs * num_steps
    logger.info(f"Device: {device}, env={env_name}, round={round_mode}, seed={seed}")
    logger.info(f"Config: num_envs={num_envs}, num_steps={num_steps}, "
                f"total_timesteps={total_timesteps}, lr={lr}, batch={BATCH_SIZE}")

    # Network
    net_cls = get_network_cls(env_name)
    network = net_cls().to(device)
    torch.manual_seed(seed)

    # Load BC params for both baseline and magnet
    if os.path.exists(pretrained_model_path):
        logger.info(f"Loading BC params: {pretrained_model_path}")
        network.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        baseline_net = net_cls().to(device)
        baseline_net.load_state_dict(network.state_dict())
        magnet_net = net_cls().to(device)
        magnet_net.load_state_dict(network.state_dict())
        for p in baseline_net.parameters():
            p.requires_grad = False
        for p in magnet_net.parameters():
            p.requires_grad = False
    else:
        logger.warning(f"BC params not found at {pretrained_model_path}, training from scratch")
        baseline_net = None
        magnet_net = None

    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, eps=1e-5, weight_decay=0.0)

    # Env init
    logger.info(f"Initializing {num_envs} env(s) with seed={seed}")
    gen = torch.Generator().manual_seed(seed)
    states = []
    for i in range(num_envs):
        g = torch.Generator().manual_seed(seed + i + 1000)
        s = env.init(g)
        states.append(s)
        logger.debug(f"  Env {i}: cp={s.current_player}, n_legal={s.legal_action_mask.sum().item()}")

    num_updates = total_timesteps // BATCH_SIZE
    logger.info(f"Training: {num_updates} updates, batch={BATCH_SIZE}, minibatch={minibatch_size}")
    buffer = PPOBuffer(num_steps, num_envs)

    for update_idx in range(num_updates):
        logger.info(f"Update {update_idx+1}/{num_updates}: collecting rollout...")
        t_start = time.time()
        buffer.reset()

        # ── 1. Collect Rollout ──
        network.eval()
        # Timing accumulators reset every 16 steps
        block_t0 = time.time()
        t_obs, t_net, t_step = 0.0, 0.0, 0.0
        for t in range(num_steps):
            obs_batch = []; actions_b = []; log_probs_b = []; values_b = []
            rewards_b = []; dones_b = []; cps_b = []; masks_b = []

            # ── Batch network forward: stack all env observations ──
            obs_list = []; masks_list = []; cps_list = []; dones_list = []
            for i in range(num_envs):
                s = states[i]
                obs_list.append(env.observe(s))
                masks_list.append(s.legal_action_mask)
                cps_list.append(s.current_player)
                dones_list.append(s.terminated or s.truncated)

            _t0 = time.time()
            # Stack into batched tensors
            obs_stack = {}
            for k in obs_list[0].keys():
                vals = [o[k] for o in obs_list]
                tensors = [v if isinstance(v, torch.Tensor) else torch.tensor(v) for v in vals]
                obs_stack[k] = torch.stack(tensors).to(device)  # (B, ...)

            mask_stack = torch.stack(masks_list).to(device)  # (B, 87)

            with torch.no_grad():
                logits_all, values_all = network(obs_stack)  # (B, 87), (B,)
                logits_all = torch.where(mask_stack, logits_all,
                                         torch.full_like(logits_all, NEG))
                dist = torch.distributions.Categorical(logits=logits_all)
                actions_t = dist.sample()  # (B,)
                log_probs_t = dist.log_prob(actions_t)  # (B,)

            # Extract per-env results
            actions_list = [int(actions_t[i].item()) for i in range(num_envs)]
            log_probs_list = [float(log_probs_t[i].item()) for i in range(num_envs)]
            values_list = [float(values_all[i].item()) for i in range(num_envs)]
            t_net += time.time() - _t0

            # ── Env step: use batch for discard-only steps, serial (with auto_reset) otherwise ──
            _t0 = time.time()
            all_discard = all(a < Action.TSUMOGIRI + 1 for a in actions_list)
            do_profile = (t % 16 == 0)  # profile every 16 steps
            if all_discard and hasattr(env, 'step_batch'):
                # All discards → batch step (no auto_reset needed, no game-ending actions)
                states = env.step_batch(states, actions_list, profile=do_profile)
            else:
                # Mixed actions or game-ending actions → serial with auto_reset
                for i in range(num_envs):
                    g = torch.Generator().manual_seed(
                        seed + update_idx * 100000 + t * num_envs + i)
                    states[i] = step_fn(states[i], actions_list[i], g)
            t_step += time.time() - _t0

            # ── Collect rewards + buffer data ──
            for i in range(num_envs):
                s = states[i]
                masks_b.append(masks_list[i].cpu())
                reward = s.rewards.clone() / MAX_REWARD
                rewards_b.append(reward)
                obs_batch.append(obs_list[i])
                actions_b.append(actions_list[i])
                log_probs_b.append(log_probs_list[i])
                values_b.append(values_list[i])
                dones_b.append(bool(dones_list[i]))
                cps_b.append(cps_list[i])

            buffer.observations.append(obs_batch)
            buffer.actions.append(actions_b)
            buffer.log_probs.append(log_probs_b)
            buffer.values.append(values_b)
            buffer.rewards.append(rewards_b)
            buffer.dones.append(dones_b)
            buffer.current_players.append(cps_b)
            buffer.action_masks.append(masks_b)

            # Log every 16 steps: show avg per step with breakdown
            if t > 0 and t % 16 == 0:
                block_elapsed = time.time() - block_t0
                n_env_steps = 16 * num_envs
                logger.info(f"  step {t:3d}/{num_steps}: "
                            f"obs={t_obs*1000:.0f}ms net={t_net*1000:.0f}ms "
                            f"step={t_step*1000:.0f}ms "
                            f"({n_env_steps} env-calls in {block_elapsed:.1f}s, "
                            f"avg={block_elapsed*1000/n_env_steps:.1f}ms/env-step "
                            f"step/step={t_step*1000/16:.0f}ms/step)")
                block_t0 = time.time()
                t_obs = t_net = t_step = 0.0


        rollout_elapsed = time.time() - t_start
        logger.info(f"  Rollout done in {rollout_elapsed:.1f}s, computing GAE...")

        # ── 2. GAE ──
        t0 = time.time()
        obs_stack, acts, log_probs, values, rewards, dones, cps, masks = buffer.get_batch()
        advantages, targets, valid_mask = compute_gae(
            rewards, values, dones, cps, gamma, gae_lambda)

        # Advantage normalization (matching JAX process_trajectory)
        # Whitens advantages to zero-mean, unit-variance using only valid entries
        vf = valid_mask.float()
        adv_mean = (advantages * vf).sum() / vf.sum().clamp(min=1.0)
        adv_var = ((advantages - adv_mean) ** 2 * vf).sum() / vf.sum().clamp(min=1.0)
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

        # Flatten
        T, B = num_steps, num_envs
        obs_flat = {k: v.reshape(T * B, *v.shape[2:]) for k, v in obs_stack.items()}
        acts_flat = acts.reshape(-1)
        log_probs_flat = log_probs.reshape(-1)
        values_flat = values.reshape(-1)
        advantages_flat = advantages.reshape(-1, NUM_PLAYERS)
        targets_flat = targets.reshape(-1, NUM_PLAYERS)
        valid_mask_flat = valid_mask.reshape(-1, NUM_PLAYERS)
        masks_flat = masks.reshape(-1, NUM_ACTIONS)
        gae_elapsed = time.time() - t0
        logger.info(f"  GAE done in {gae_elapsed:.1f}s, starting PPO update...")

        # ── 3. PPO Update ──
        t0 = time.time()
        network.train()
        for epoch in range(update_epochs):
            perm = torch.randperm(T * B)
            for start in range(0, T * B, minibatch_size):
                idx = perm[start:start + minibatch_size]

                obs_mb = {k: v[idx].to(device) for k, v in obs_flat.items()}
                act_mb = acts_flat[idx].to(device)
                logp_old = log_probs_flat[idx].to(device)
                adv_mb = advantages_flat[idx].to(device)
                tgt_mb = targets_flat[idx].to(device)
                vmask_mb = valid_mask_flat[idx].to(device)
                amask_mb = masks_flat[idx].to(device)
                val_old = values_flat[idx].to(device)

                logits, values = network(obs_mb)
                logits = torch.where(amask_mb, logits, torch.full_like(logits, NEG))
                dist = torch.distributions.Categorical(logits=logits)
                logp_new = dist.log_prob(act_mb)
                entropy = dist.entropy()

                ratio = torch.exp(logp_new - logp_old).unsqueeze(-1)
                adv = adv_mb.gather(1, cps.reshape(-1, 1)[idx].to(device))
                vmask = vmask_mb.gather(1, cps.reshape(-1, 1)[idx].to(device))

                # PPO clip loss
                ppo_loss = -masked_mean(
                    torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv),
                    vmask)

                # Magnet KL regularization
                mag_kl = 0.0
                if magnet_net is not None and mag_coef > 0:
                    with torch.no_grad():
                        mag_logits, _ = magnet_net(obs_mb)
                    mag_logits = torch.where(amask_mb, mag_logits, torch.full_like(mag_logits, NEG))
                    mag_dist = torch.distributions.Categorical(logits=mag_logits)
                    mag_kl = masked_mean(
                        torch.distributions.kl.kl_divergence(dist, mag_dist).unsqueeze(-1), vmask)

                # Critic loss
                vt = values.unsqueeze(-1)
                val_clipped = val_old.unsqueeze(-1) + torch.clamp(
                    vt - val_old.unsqueeze(-1), -clip_eps, clip_eps)
                tgt = tgt_mb.gather(1, cps.reshape(-1, 1)[idx].to(device))
                loss_critic = 0.5 * vf_coef * masked_mean(
                    torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)

                loss = ppo_loss - ent_coef * masked_mean(entropy.unsqueeze(-1), vmask) \
                    + mag_coef * mag_kl + loss_critic

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        update_elapsed = time.time() - t0
        logger.info(f"  Update done in {update_elapsed:.1f}s "
                    f"(total: {time.time()-t_start:.1f}s)")

        # ── Logging ──
        avg_reward = rewards.mean().item() * MAX_REWARD
        n_nonzero = int((rewards != 0).sum().item())
        n_total = int(rewards.numel())
        rew_abs_mean = rewards.abs().mean().item() * MAX_REWARD  # avg reward magnitude
        rew_max = rewards.max().item() * MAX_REWARD
        rew_min = rewards.min().item() * MAX_REWARD
        logger.info(f"Update {update_idx + 1}/{num_updates} | "
              f"reward: mean={avg_reward:.3f} abs_mean={rew_abs_mean:.2f} "
              f"max={rew_max:.0f} min={rew_min:.0f} (nonzero={n_nonzero}/{n_total}) | "
              f"loss: {loss.item():.4f} | "
              f"entropy: {masked_mean(entropy.unsqueeze(-1), vmask).item():.4f}")

    # Save final model
    if save_model:
        save_path = default_rl_params_path(env_name, seed)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(network.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    return network


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="red_mahjong")
    parser.add_argument("--round_mode", default="single")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--total_timesteps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrained_model_path", default=None)
    parser.add_argument("--device", default=None, help="cpu, cuda:0, npu:0")
    args = parser.parse_args()
    train_ppo(**vars(args))
