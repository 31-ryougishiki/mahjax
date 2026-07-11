#!/usr/bin/env python3
"""PPO + Magnet regularization trainer (PyTorch, BatchState-native).

Uses RedMahjongParallel with BatchState for GPU-accelerated batched RL training.

Key differences from the old per-env implementation:
- BatchState end-to-end: init_batch → observe_batch → step_batch
- Vectorized GAE: no Python loop over the B dimension
- Pre-allocated rollout buffer: direct (T, B, ...) writes
- Integrated evaluation: 1-vs-3 matchup stats every N updates
- Optional WandB logging
- Periodic checkpointing with optimizer state for resume

Usage:
    python mahjax_pt/examples/ppo_with_reg.py --num_envs 128 --device cuda:0
    python mahjax_pt/examples/ppo_with_reg.py --num_envs 1024 --device cuda:0 --use_wandb
"""

import os
import sys
import time
import logging
import argparse
from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ppo")

# ── Optional wandb ──
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.batch_state import BatchState
from mahjax_pt.examples.common import (
    default_bc_params_path,
    default_rl_params_path,
    get_network_cls,
)
from mahjax_pt.examples.utils import make_eval_fn

# ══════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════

MAX_REWARD = 320.0
NEG = -1e9


# ══════════════════════════════════════════════════════════════════════════
# Vectorized GAE — no per-environment Python loop
# ══════════════════════════════════════════════════════════════════════════

def compute_gae_vectorized(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    current_players: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
):
    """Compute GAE with per-player accumulator tensors, vectorized over B.

    Args:
        rewards:  (T, B, 4) — 4-player reward vector (already / MAX_REWARD)
        values:   (T, B)    — value estimate for the acting player
        dones:    (T, B)    — is_new_episode flag (captured BEFORE step)
        current_players: (T, B) — which player acted at each step

    Returns:
        advantages:  (T, B, 4)
        targets:     (T, B, 4)
        valid_mask:  (T, B, 4) bool
    """
    T, B, P = rewards.shape
    device = rewards.device

    advantages = torch.zeros(T, B, P, device=device)
    targets = torch.zeros(T, B, P, device=device)
    valid_mask = torch.zeros(T, B, P, dtype=torch.bool, device=device)

    # Per-environment, per-player accumulators — all processed in parallel
    gae_acc = torch.zeros(B, P, device=device)
    reward_accum = torch.zeros(B, P, device=device)
    next_value = torch.zeros(B, P, device=device)
    has_next_value = torch.zeros(B, P, dtype=torch.bool, device=device)
    next_valid = torch.zeros(B, P, dtype=torch.bool, device=device)

    b_idx = torch.arange(B, device=device)

    for t in reversed(range(T)):
        cp = current_players[t]   # (B,)
        done = dones[t]           # (B,)

        # Reset accumulators on episode boundaries
        # NOTE: next_valid is NOT reset — JAX preserves it across boundaries
        gae_acc[done] = 0.0
        reward_accum[done] = 0.0
        has_next_value[done] = False
        next_value[done] = 0.0

        # Accumulate rewards, extract acting player's share
        reward_accum = reward_accum + rewards[t]
        player_reward = reward_accum[b_idx, cp].clone()
        reward_accum[b_idx, cp] = 0.0

        not_done = (~done).float()
        td_error = player_reward + gamma * next_value[b_idx, cp] * not_done - values[t]
        new_gae = td_error + gamma * gae_lambda * gae_acc[b_idx, cp] * not_done
        gae_acc[b_idx, cp] = new_gae

        is_valid = has_next_value[b_idx, cp] | done | next_valid[b_idx, cp]

        advantages[t, b_idx, cp] = torch.where(
            is_valid, new_gae, torch.zeros_like(new_gae))
        targets[t, b_idx, cp] = torch.where(
            is_valid, new_gae + values[t], values[t])
        valid_mask[t, b_idx, cp] = is_valid

        next_value[b_idx, cp] = values[t]
        has_next_value[b_idx, cp] = True
        # JAX: next_valid.at[player].set(is_valid) | done
        # When done=True, bool-scalar | array broadcasts True to ALL players
        next_valid[done] = True
        next_valid[b_idx, cp] = is_valid | done

    return advantages, targets, valid_mask


# ══════════════════════════════════════════════════════════════════════════
# Pre-allocated rollout buffer
# ══════════════════════════════════════════════════════════════════════════

class PPOBuffer:
    """Pre-allocated buffer.  Writes directly to (T, B, ...) tensors."""

    def __init__(self, num_steps: int, num_envs: int,
                 num_actions: int = 87, num_players: int = 4,
                 device: torch.device = torch.device("cpu")):
        self.T = num_steps
        self.B = num_envs
        self.device = device

        # Fixed-shape fields
        self.actions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, num_players, device=device)
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        self.current_players = torch.zeros(num_steps, num_envs, dtype=torch.long, device=device)
        self.masks = torch.zeros(num_steps, num_envs, num_actions, device=device)

        # Observations — allocated lazily on first store
        self._obs: Dict[str, torch.Tensor] = {}
        self._obs_keys = None

    def _init_obs(self, obs: Dict[str, torch.Tensor]):
        """Allocate observation tensors from first observation dict."""
        self._obs = {}
        for k, v in obs.items():
            shape = v.shape[1:]  # strip batch dim
            self._obs[k] = torch.zeros(
                self.T, self.B, *shape, dtype=v.dtype, device=self.device)
        self._obs_keys = list(obs.keys())

    def store(self, t: int, obs: Dict[str, torch.Tensor],
              actions: torch.Tensor, log_probs: torch.Tensor,
              values: torch.Tensor, rewards: torch.Tensor,
              dones: torch.Tensor, cps: torch.Tensor,
              masks: torch.Tensor):
        """Store one timestep.  All inputs are (B, ...) tensors."""
        if self._obs_keys is None:
            self._init_obs(obs)

        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.current_players[t] = cps
        self.masks[t] = masks
        for k in self._obs_keys:
            self._obs[k][t] = obs[k]

    def get_batch(self):
        """Return all buffers as (T, B, ...) tensors (no copies)."""
        return (self._obs, self.actions, self.log_probs, self.values,
                self.rewards, self.dones, self.current_players, self.masks)


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where mask is True."""
    denom = mask.float().sum().clamp(min=1.0)
    return (x * mask.float()).sum() / denom


# ══════════════════════════════════════════════════════════════════════════
# Main training
# ══════════════════════════════════════════════════════════════════════════

def train_ppo(
    # Environment
    env_name: str = "red_mahjong",
    round_mode: str = "single",
    seed: int = 0,
    # Scale
    num_envs: int = 1024,
    num_steps: int = 256,
    total_timesteps: int = 100_000_000,
    # PPO
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    ent_coef: float = 0.01,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    update_epochs: int = 4,
    minibatch_size: int = 4096,
    # Magnet regularization
    mag_coef: float = 0.2,
    # Paths
    pretrained_model_path: Optional[str] = None,
    save_model: bool = True,
    # Evaluation
    eval_interval: int = 10,
    eval_num_envs: int = 1000,
    # Infrastructure
    device: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "mahjax-ppo",
    checkpoint_dir: Optional[str] = None,
    resume_from: Optional[str] = None,
):
    # ── Device ──────────────────────────────────────────────────────
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if device.type == "npu":
        import torch_npu  # noqa: F401

    if pretrained_model_path is None:
        pretrained_model_path = default_bc_params_path(env_name)

    # ── Environment ─────────────────────────────────────────────────
    env = make_env(env_name, backend="parallel", round_mode=round_mode,
                   observe_type="dict")
    NUM_PLAYERS = env.num_players
    NUM_ACTIONS = env.num_actions
    BATCH_SIZE = num_envs * num_steps
    NUM_UPDATES = total_timesteps // BATCH_SIZE

    logger.info(f"Device: {device}, env={env_name}, round={round_mode}, seed={seed}")
    logger.info(f"Scale: num_envs={num_envs}, num_steps={num_steps}, "
                f"total_timesteps={total_timesteps}, updates={NUM_UPDATES}")
    logger.info(f"Batch: {BATCH_SIZE}, minibatch={minibatch_size}, lr={lr}")

    # ── Network & pretrained weights ────────────────────────────────
    net_cls = get_network_cls(env_name)
    network = net_cls().to(device)
    torch.manual_seed(seed)

    baseline_net = None
    magnet_net = None

    if os.path.exists(pretrained_model_path):
        logger.info(f"Loading BC params: {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location=device)
        network.load_state_dict(state_dict)

        # Baseline network (frozen, for evaluation opponent)
        baseline_net = net_cls().to(device)
        baseline_net.load_state_dict(state_dict)
        for p in baseline_net.parameters():
            p.requires_grad = False
        baseline_net.eval()

        # Magnet network (frozen, for KL regularization)
        if mag_coef > 0:
            magnet_net = net_cls().to(device)
            magnet_net.load_state_dict(state_dict)
            for p in magnet_net.parameters():
                p.requires_grad = False
            magnet_net.eval()
    else:
        logger.warning(f"BC params not found at {pretrained_model_path}, "
                       "training from scratch")

    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, eps=1e-5,
                                  weight_decay=0.0)

    # ── Resume from checkpoint ──────────────────────────────────────
    start_update = 0
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        network.load_state_dict(ckpt["network"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt.get("update", 0)
        logger.info(f"  Resumed at update {start_update}")

    # ── WandB ───────────────────────────────────────────────────────
    if use_wandb and HAS_WANDB:
        wandb.init(project=wandb_project, config={
            "env_name": env_name, "round_mode": round_mode, "seed": seed,
            "num_envs": num_envs, "num_steps": num_steps,
            "total_timesteps": total_timesteps,
            "lr": lr, "ent_coef": ent_coef, "clip_eps": clip_eps,
            "vf_coef": vf_coef, "mag_coef": mag_coef,
            "update_epochs": update_epochs, "minibatch_size": minibatch_size,
            "device": str(device),
        })
    elif use_wandb and not HAS_WANDB:
        logger.warning("wandb not installed — logging to stdout only")

    # ── Initialize environments ─────────────────────────────────────
    logger.info(f"Initializing {num_envs} envs on {device} ...")
    t_init = time.time()
    bs = env.init_batch(num_envs=num_envs, device=device)
    logger.info(f"  Done in {time.time() - t_init:.1f}s")

    buffer = PPOBuffer(num_steps, num_envs, NUM_ACTIONS, NUM_PLAYERS,
                       device=device)

    # ── Evaluation setup (serial env, small scale — fine for eval) ──
    eval_env = make_env(env_name, backend="serial", round_mode=round_mode,
                        observe_type="dict")
    one_vs_three = make_eval_fn(eval_env, eval_num_envs)

    def _make_deterministic_actor(net):
        @torch.no_grad()
        def actor(state, rng=None):
            del rng
            obs = eval_env.observe(state)
            obs_b = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor)
                         else torch.tensor(v, device=device).unsqueeze(0))
                     for k, v in obs.items()}
            logits, _ = net(obs_b)
            logits = logits.squeeze(0)
            mask = state.legal_action_mask.to(device)
            logits = torch.where(mask, logits, torch.full_like(logits, NEG))
            return int(torch.argmax(logits).item())
        return actor

    def _random_actor(state, rng=None):
        mask = state.legal_action_mask.float()
        probs = mask / mask.sum().clamp(min=1.0)
        return int(torch.multinomial(probs, 1).item())

    baseline_actor = (_make_deterministic_actor(baseline_net)
                      if baseline_net else _random_actor)

    def run_eval():
        agent_actor = _make_deterministic_actor(network)
        metrics = {}
        metrics.update({
            f"eval/vs_rand/{k}": v
            for k, v in one_vs_three(agent_actor, _random_actor)().items()
        })
        if baseline_net:
            metrics.update({
                f"eval/vs_baseline/{k}": v
                for k, v in one_vs_three(agent_actor, baseline_actor)().items()
            })
        return metrics

    # ═════════════════════════════════════════════════════════════════
    # Training loop
    # ═════════════════════════════════════════════════════════════════
    logger.info(f"Training: {NUM_UPDATES} updates starting from update {start_update}")

    for update_idx in range(start_update, NUM_UPDATES):
        t_start = time.time()
        network.eval()

        # ── 1. Collect rollout ──────────────────────────────────────
        for t in range(num_steps):
            # Capture is_new_episode BEFORE reinit (from previous step's
            # terminated/truncated flags).  Must happen before reinit
            # replaces terminated states with fresh ones.
            # Matches JAX: is_new_episode = state.terminated | state.truncated
            is_new_episode = bs.terminated | bs.truncated

            # Re-initialize terminated envs before step
            bs = env.reinit_terminated_batch(bs)

            # Observe — directly returns dict of (B, ...) tensors
            obs = env.observe_batch(bs)

            # Network forward
            with torch.no_grad():
                logits, values = network(obs)
                mask = bs.legal_action_mask
                logits = torch.where(mask, logits, torch.full_like(logits, NEG))
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            # Capture current_player BEFORE step
            cp = bs.current_player.clone()

            # Step (produces rewards for this action, matching JAX's
            # next_state.rewards)
            bs = env.step_batch(bs, actions)

            # Store with reward AFTER step (matching JAX's next_state.rewards)
            buffer.store(
                t, obs, actions, log_probs, values,
                bs.rewards.clone() / MAX_REWARD,
                is_new_episode,
                cp,
                mask,
            )

        rollout_elapsed = time.time() - t_start
        logger.info(f"  Rollout: {rollout_elapsed:.1f}s "
                    f"({num_envs * num_steps / rollout_elapsed:.0f} env-steps/s)")

        # ── 2. GAE ──────────────────────────────────────────────────
        t0 = time.time()
        (obs, acts, log_probs, values, rewards,
         dones, cps, masks) = buffer.get_batch()

        advantages, targets, valid_mask = compute_gae_vectorized(
            rewards, values, dones, cps, gamma, gae_lambda)

        # Whitened advantage normalization (only over valid entries)
        vf = valid_mask.float()
        adv_sum = (advantages * vf).sum()
        adv_count = vf.sum().clamp(min=1.0)
        adv_mean = adv_sum / adv_count
        adv_var = ((advantages - adv_mean) ** 2 * vf).sum() / adv_count
        advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

        # Flatten (T, B, ...) → (T*B, ...)
        T, B = num_steps, num_envs

        def _flatten(x):
            return x.reshape(T * B, *x.shape[2:])

        obs_flat = {k: _flatten(v) for k, v in obs.items()}
        acts_flat = _flatten(acts)
        log_probs_flat = _flatten(log_probs)
        values_flat = _flatten(values)
        advantages_flat = _flatten(advantages)
        targets_flat = _flatten(targets)
        valid_mask_flat = _flatten(valid_mask)
        masks_flat = _flatten(masks)
        cps_flat = _flatten(cps)

        gae_elapsed = time.time() - t0

        # ── 3. PPO update ───────────────────────────────────────────
        t0 = time.time()
        network.train()

        total_loss_val = 0.0
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_frac = 0.0
        total_explained_var = 0.0
        n_minibatches = 0

        for epoch in range(update_epochs):
            perm = torch.randperm(T * B)
            for start in range(0, T * B, minibatch_size):
                idx = perm[start:start + minibatch_size]
                n_minibatches += 1

                obs_mb = {k: v[idx].to(device) for k, v in obs_flat.items()}
                act_mb = acts_flat[idx].to(device)
                logp_old = log_probs_flat[idx].to(device)
                adv_mb = advantages_flat[idx].to(device)
                tgt_mb = targets_flat[idx].to(device)
                vmask_mb = valid_mask_flat[idx].to(device)
                amask_mb = masks_flat[idx].to(device)
                val_old = values_flat[idx].to(device)
                cp_mb = cps_flat[idx].to(device)

                logits, values_new = network(obs_mb)
                logits = torch.where(amask_mb, logits,
                                     torch.full_like(logits, NEG))
                dist = torch.distributions.Categorical(logits=logits)
                logp_new = dist.log_prob(act_mb)
                entropy = dist.entropy()

                # Ratio
                log_ratio = logp_new - logp_old
                ratio = torch.exp(log_ratio).unsqueeze(-1)

                # Per-player advantage & validity
                adv = adv_mb.gather(1, cp_mb.unsqueeze(-1))
                vmask = vmask_mb.gather(1, cp_mb.unsqueeze(-1))

                # PPO clipped actor loss
                clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                ppo_loss = -masked_mean(
                    torch.min(ratio * adv, clip_adv), vmask)

                # Magnet KL regularization
                mag_kl = 0.0
                if magnet_net is not None and mag_coef > 0:
                    with torch.no_grad():
                        mag_logits, _ = magnet_net(obs_mb)
                    mag_logits = torch.where(
                        amask_mb, mag_logits,
                        torch.full_like(mag_logits, NEG))
                    mag_dist = torch.distributions.Categorical(logits=mag_logits)
                    mag_kl = masked_mean(
                        torch.distributions.kl.kl_divergence(dist, mag_dist)
                        .unsqueeze(-1), vmask)

                # Clipped critic loss
                vt = values_new.unsqueeze(-1)
                val_clipped = val_old.unsqueeze(-1) + torch.clamp(
                    vt - val_old.unsqueeze(-1), -clip_eps, clip_eps)
                tgt = tgt_mb.gather(1, cp_mb.unsqueeze(-1))
                loss_critic = 0.5 * vf_coef * masked_mean(
                    torch.max((vt - tgt) ** 2, (val_clipped - tgt) ** 2), vmask)

                # Diagnostics (matching JAX)
                # Taylor approximation: KL ≈ (r-1) - log(r) ≈ 0.5·log(r)²
                approx_kl = masked_mean(
                    (ratio - 1.0) - log_ratio.unsqueeze(-1), vmask)
                clip_frac = masked_mean(
                    (torch.abs(ratio - 1.0) > clip_eps).float(), vmask)
                explained_var = (
                    1.0 - masked_mean((tgt - vt) ** 2, vmask)
                    / (masked_mean((tgt - masked_mean(tgt, vmask)) ** 2, vmask) + 1e-8)
                ).clamp(min=0.0)

                # Total loss
                loss = (ppo_loss
                        - ent_coef * masked_mean(entropy.unsqueeze(-1), vmask)
                        + mag_coef * mag_kl
                        + loss_critic)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_val += loss.item()
                total_actor_loss += ppo_loss.item()
                total_critic_loss += loss_critic.item()
                total_entropy += masked_mean(entropy.unsqueeze(-1), vmask).item()
                total_approx_kl += approx_kl.item()
                total_clip_frac += clip_frac.item()
                total_explained_var += explained_var.item()

        update_elapsed = time.time() - t0

        # ── 4. Logging ──────────────────────────────────────────────
        n = max(n_minibatches, 1)
        avg_reward = rewards.mean().item() * MAX_REWARD
        n_nonzero = int((rewards != 0).sum().item())
        n_total = int(rewards.numel())
        rew_abs_mean = rewards.abs().mean().item() * MAX_REWARD
        rew_max = rewards.max().item() * MAX_REWARD
        rew_min = rewards.min().item() * MAX_REWARD
        eps_len = 1.0 / max(dones.float().mean().item(), 1e-8)

        log_dict = {
            "steps": (update_idx + 1) * BATCH_SIZE,
            "update": update_idx + 1,
            "train/loss_total": total_loss_val / n,
            "train/loss_actor": total_actor_loss / n,
            "train/loss_critic": total_critic_loss / n,
            "train/entropy": total_entropy / n,
            "train/approx_kl": total_approx_kl / n,
            "train/clip_frac": total_clip_frac / n,
            "train/explained_var": total_explained_var / n,
            "train/avg_reward": avg_reward,
            "train/rew_abs_mean": rew_abs_mean,
            "train/rew_max": rew_max,
            "train/rew_min": rew_min,
            "train/nonzero_reward_pct": n_nonzero / max(n_total, 1) * 100,
            "train/avg_eps_len": eps_len,
            "time/rollout_s": rollout_elapsed,
            "time/gae_s": gae_elapsed,
            "time/update_s": update_elapsed,
            "time/total_s": time.time() - t_start,
        }

        logger.info(
            f"Update {update_idx + 1}/{NUM_UPDATES} | "
            f"loss={total_loss_val / n:.4f} "
            f"actor={total_actor_loss / n:.4f} "
            f"critic={total_critic_loss / n:.4f} "
            f"ent={total_entropy / n:.4f} | "
            f"rew={avg_reward:.1f} (abs={rew_abs_mean:.1f} "
            f"max={rew_max:.0f} min={rew_min:.0f} "
            f"nonzero={n_nonzero}/{n_total}) | "
            f"kl={total_approx_kl / n:.4f} "
            f"clip={total_clip_frac / n:.3f} "
            f"expl_var={total_explained_var / n:.3f} | "
            f"time: rollout={rollout_elapsed:.1f}s "
            f"gae={gae_elapsed:.1f}s "
            f"update={update_elapsed:.1f}s"
        )

        # ── 5. Evaluation ───────────────────────────────────────────
        if (update_idx + 1) % eval_interval == 0 or update_idx + 1 == NUM_UPDATES:
            t_eval = time.time()
            network.eval()
            eval_metrics = run_eval()
            log_dict.update(eval_metrics)

            eval_str = " | ".join(
                f"{k}={v:.3f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in sorted(eval_metrics.items()))
            logger.info(f"  Eval ({time.time() - t_eval:.1f}s): {eval_str}")

        # ── 6. WandB ────────────────────────────────────────────────
        if use_wandb and HAS_WANDB:
            wandb.log(log_dict)

        # ── 7. Checkpoint ───────────────────────────────────────────
        if save_model and checkpoint_dir and (
            (update_idx + 1) % eval_interval == 0
            or update_idx + 1 == NUM_UPDATES
        ):
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir,
                                     f"ppo_ckpt_{update_idx + 1}.pt")
            torch.save({
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update": update_idx + 1,
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    # ═════════════════════════════════════════════════════════════════
    # Save final model
    # ═════════════════════════════════════════════════════════════════
    if save_model:
        save_path = default_rl_params_path(env_name, seed)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        torch.save(network.state_dict(), save_path)
        logger.info(f"Model saved to {save_path}")

    if use_wandb and HAS_WANDB:
        wandb.finish()

    return network


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PPO + Magnet regularization (BatchState-native)")
    # Environment
    parser.add_argument("--env_name", default="red_mahjong")
    parser.add_argument("--round_mode", default="single",
                        choices=["single", "east", "half"])
    parser.add_argument("--seed", type=int, default=0)
    # Scale
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--total_timesteps", type=int,
                        default=100_000_000,
                        help="total env steps across all envs")
    # PPO hyperparams
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=4096)
    parser.add_argument("--mag_coef", type=float, default=0.2)
    # Paths
    parser.add_argument("--pretrained_model_path", default=None)
    parser.add_argument("--checkpoint_dir", default=None,
                        help="directory for periodic checkpoint saves")
    parser.add_argument("--resume_from", default=None,
                        help="checkpoint path to resume from")
    parser.add_argument("--no_save", action="store_true",
                        help="skip saving the final model")
    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="evaluate every N updates (0 = disable)")
    parser.add_argument("--eval_num_envs", type=int, default=1000)
    # Infrastructure
    parser.add_argument("--device", default=None,
                        help="cpu, cuda:0, npu:0")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="mahjax-ppo")

    args = parser.parse_args()

    if args.eval_interval == 0:
        args.eval_interval = args.total_timesteps // (args.num_envs * args.num_steps) + 1

    train_ppo(
        env_name=args.env_name,
        round_mode=args.round_mode,
        seed=args.seed,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        total_timesteps=args.total_timesteps,
        lr=args.lr,
        ent_coef=args.ent_coef,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        mag_coef=args.mag_coef,
        pretrained_model_path=args.pretrained_model_path,
        save_model=not args.no_save,
        eval_interval=args.eval_interval,
        eval_num_envs=args.eval_num_envs,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )
