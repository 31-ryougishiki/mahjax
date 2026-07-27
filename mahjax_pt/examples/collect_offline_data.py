#!/usr/bin/env python3
"""Offline data collector using rule-based players (PyTorch port, batch backend).

Uses the parallel (batch) backend for vectorized env stepping.
Player actions are still computed per-env via unstack_state, but all env
transitions are batched into a single step_batch call.

Usage:
    python mahjax_pt/examples/collect_offline_data.py \
        --num_samples 200000 --num_envs 1024 --num_steps 32 --seed 0
"""

import os
import sys
import pickle
import time
import logging
import numpy as np

import torch

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.players import rule_based_player_batch
from mahjax_pt.examples.common import default_dataset_path, attach_dataset_metadata

# ── Logging setup ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect")


def collect_data(
    env_name="red_mahjong",
    num_samples=200_000,
    num_envs=1024,
    num_steps=32,
    seed=0,
    gamma=0.99,
    max_reward=320.0,
    dataset_path=None,
):
    if dataset_path is None:
        dataset_path = default_dataset_path(env_name)

    # ── 1. Init parallel env ──
    logger.info(f"Creating env: {env_name} (round_mode=single, backend=parallel)")
    t0 = time.time()
    env = make_env(env_name, round_mode="single", observe_type="dict", backend="parallel")
    logger.info(f"Env created in {time.time() - t0:.1f}s | "
                f"num_players={env.num_players} num_actions={env.num_actions}")

    # ── 2. Init all envs at once ──
    logger.info(f"Initializing {num_envs} environments with seed={seed} (batch)")
    keys = [seed + i for i in range(num_envs)]
    t0 = time.time()
    states = env.init_batch(keys=keys)
    logger.info(f"Batch init done in {time.time() - t0:.1f}s")

    # ── 3. Collect ──
    chunk_size = num_envs * num_steps
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    total_steps = 0
    start_time = time.time()

    logger.info(f"Collection: {num_chunks} chunks × {num_envs} envs × {num_steps} steps "
                f"= {num_chunks * chunk_size} total | target: {num_samples}")

    data_obs = []
    data_act = []
    data_mask = []
    data_ret = []

    for chunk_idx in range(num_chunks):
        chunk_start = time.time()
        obs_seq = []   # list of batched obs dicts: dict of (B, ...) tensors
        act_seq = []   # list of (B,) int tensors
        mask_seq = []  # list of (B, num_actions) bool tensors
        rew_seq = []   # list of (B, 4) float tensors
        done_seq = []  # list of (B,) bool tensors
        cp_seq = []    # list of (B,) int tensors

        log_time = time.time()
        for step in range(num_steps):
            # ── Observe (batched) ──
            obs = env.observe_batch(states)  # dict of (B, ...) tensors

            # ── Player actions (batched, no per-env unstack) ──
            actions_t = rule_based_player_batch(
                states, seed=seed + chunk_idx * 10000 + step * num_envs)

            # ── Step (batched) ──
            states = env.step_batch(states, actions_t)

            # ── Collect transition data ──
            done = states.terminated | states.truncated
            reward = states.rewards.clone()
            mask = states.legal_action_mask.clone()

            obs_seq.append(obs)
            act_seq.append(actions_t)
            mask_seq.append(mask)
            rew_seq.append(reward)
            done_seq.append(done)
            cp_seq.append(states.current_player.clone())

            # ── Reinit terminated envs ──
            term_count = done.sum().item()
            if term_count > 0:
                states = env.reinit_terminated_batch(states)

            # Log every 8 steps, measuring the 8-step window
            if step % 8 == 0 and step > 0:
                elapsed = time.time() - log_time
                logger.info(f"  Chunk {chunk_idx+1}/{num_chunks} step {step}/{num_steps} "
                           f"({elapsed:.1f}s for 8 steps, ~{elapsed/8*1000:.0f}ms/step "
                           f"| {term_count} resets)")
                log_time = time.time()

        # ── GAE (vectorized across batch) ──
        T, B = num_steps, num_envs
        returns = np.zeros((T, B), dtype=np.float32)
        for b in range(B):
            running_ret = np.zeros(4, dtype=np.float32)
            for t in reversed(range(T)):
                if done_seq[t][b]:
                    running_ret = np.zeros(4, dtype=np.float32)
                r_t = rew_seq[t][b].numpy()
                running_ret = r_t + gamma * running_ret
                p = int(cp_seq[t][b].item())
                returns[t, b] = running_ret[p]

        returns = returns / max_reward

        # ── Flatten & store (skip samples where action is not in mask) ──
        skipped = 0
        for b in range(B):
            for t in range(T):
                mask = mask_seq[t][b]
                action = act_seq[t][b]
                if not mask[action]:
                    skipped += 1
                    continue
                # Extract single-env observation from batched dict
                obs_single = {k: v[b] for k, v in obs_seq[t].items()}
                data_obs.append(obs_single)
                data_act.append(int(action.item()))
                data_mask.append(mask)
                data_ret.append(returns[t, b])
        if skipped > 0:
            logger.warning(f"  Skipped {skipped} bad samples (action not in legal mask)")

        total_steps += T * B

        chunk_elapsed = time.time() - chunk_start
        progress_pct = min(100.0, 100.0 * total_steps / num_samples)
        sps = total_steps / (time.time() - start_time) if total_steps > 0 else 0
        logger.info(f"  Chunk {chunk_idx+1}/{num_chunks} done in {chunk_elapsed:.1f}s | "
                    f"samples: {len(data_obs)}/{num_samples} ({progress_pct:.0f}%) | "
                    f"{sps:.0f} samples/s")

        if len(data_obs) >= num_samples:
            break

    # ── 4. Save ──
    logger.info(f"Saving {num_samples} samples to {dataset_path}...")
    N = num_samples
    data_obs = data_obs[:N]
    data_act = np.array(data_act[:N], dtype=np.int32)
    data_mask = torch.stack([m.clone().detach() for m in data_mask[:N]])
    data_ret = np.array(data_ret[:N], dtype=np.float32)

    def _to_tensor(v):
        """Convert observation value to a tensor for stacking."""
        if isinstance(v, torch.Tensor):
            return v
        if isinstance(v, bool):
            return torch.tensor(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return torch.tensor(v)
        return torch.tensor(v)

    stacked_obs = {}
    for key in data_obs[0].keys():
        stacked_obs[key] = torch.stack([_to_tensor(o[key]) for o in data_obs])

    dataset = attach_dataset_metadata({
        "observation": stacked_obs,
        "action": data_act,
        "legal_action_mask": data_mask,
        "return": data_ret,
    }, env_name)

    os.makedirs(os.path.dirname(dataset_path) or ".", exist_ok=True)
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)

    elapsed = time.time() - start_time
    logger.info(f"Done: {N} samples in {elapsed:.1f}s ({N/elapsed:.0f} samples/s) → {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="red_mahjong")
    parser.add_argument("--num_samples", type=int, default=200_000)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    collect_data(**{k: v for k, v in vars(args).items() if k != "debug"})
