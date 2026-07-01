#!/usr/bin/env python3
"""Offline data collector using rule-based players (PyTorch port).

Usage:
    python mahjax_pt/examples/collect_offline_data.py \
        --num_samples 2000 --num_envs 4 --num_steps 32 --seed 0
"""

import os
import sys
import pickle
import time
import logging
import numpy as np

import torch

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.auto_reset_wrapper import auto_reset
from mahjax_pt.red_mahjong.players import rule_based_player
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
    num_envs=4,
    num_steps=32,
    seed=0,
    gamma=0.99,
    max_reward=320.0,
    dataset_path=None,
):
    if dataset_path is None:
        dataset_path = default_dataset_path(env_name)

    # ── 1. Init env ──
    logger.info(f"Creating env: {env_name} (round_mode=single)")
    t0 = time.time()
    env = make_env(env_name, round_mode="single", observe_type="dict")
    step_env = auto_reset(env.step, env.init)
    logger.info(f"Env created in {time.time() - t0:.1f}s | "
                f"num_players={env.num_players} num_actions={env.num_actions}")

    # ── 2. Init states ──
    logger.info(f"Initializing {num_envs} environment(s) with seed={seed}")
    states = []
    for i in range(num_envs):
        g = torch.Generator().manual_seed(seed + i)
        s = env.init(g)
        states.append(s)
        logger.debug(f"  Env {i}: init OK, dealer={s.current_player}, "
                     f"legal_actions={s.legal_action_mask.sum().item()}")

    # ── 3. Collect ──
    chunk_size = num_envs * num_steps
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    total_steps = 0
    start_time = time.time()

    # Track per-env stats for debugging
    env_step_count = [0] * num_envs        # steps in current game
    env_game_count = [0] * num_envs         # how many games played

    logger.info(f"Collection: {num_chunks} chunks × {chunk_size} steps/chunk "
                f"= {num_chunks * chunk_size} total | target: {num_samples}")

    data_obs = []
    data_act = []
    data_mask = []
    data_ret = []

    for chunk_idx in range(num_chunks):
        chunk_start = time.time()
        obs_seq = []
        act_seq = []
        mask_seq = []
        rew_seq = []
        done_seq = []
        cp_seq = []

        for step in range(num_steps):
            step_start = time.time()
            o_list = []
            a_list = []
            m_list = []
            r_list = []
            d_list = []
            c_list = []

            for i in range(num_envs):
                # ══ Per-env step ══
                env_start = time.time()
                s = states[i]
                cp = s.current_player

                # Check for abnormal state
                if s.terminated and not s.truncated:
                    # Track game end, but auto_reset should handle this
                    env_game_count[i] += 1
                    logger.debug(f"  Env {i}: game #{env_game_count[i]} ended "
                                f"at step {env_step_count[i]}, scores={s.round_state.score.tolist()}")
                    env_step_count[i] = 0

                # Observe
                try:
                    obs = env.observe(s)
                except Exception as e:
                    logger.error(f"  Env {i}: observe failed: {e}", exc_info=True)
                    raise

                mask = s.legal_action_mask
                n_legal = mask.sum().item()

                # Player action
                g = torch.Generator().manual_seed(seed + chunk_idx * 10000 + step * num_envs + i)
                try:
                    action = rule_based_player(s, g)
                except Exception as e:
                    logger.error(f"  Env {i} step {step}: player failed "
                                f"(legal_actions={n_legal}): {e}", exc_info=True)
                    raise

                # Env step
                try:
                    next_s = step_env(s, action, g)
                except Exception as e:
                    logger.error(f"  Env {i} step {step}: env.step failed "
                                f"(action={action}): {e}", exc_info=True)
                    raise

                done = next_s.terminated or next_s.truncated
                reward = next_s.rewards.clone()

                env_step_count[i] += 1
                env_elapsed = time.time() - env_start

                # Warn if single step takes too long
                if env_elapsed > 2.0:
                    logger.warning(f"  Env {i} step {step}: SLOW ({env_elapsed:.1f}s) "
                                   f"action={action}, game_step={env_step_count[i]}, "
                                   f"done={done}, legal={n_legal}")

                states[i] = next_s
                o_list.append(obs)
                a_list.append(action)
                m_list.append(mask)
                r_list.append(reward)
                d_list.append(done)
                c_list.append(cp)

            obs_seq.append(o_list)
            act_seq.append(a_list)
            mask_seq.append(m_list)
            rew_seq.append(r_list)
            done_seq.append(d_list)
            cp_seq.append(c_list)

            step_elapsed = time.time() - step_start
            if step > 0 and step % 8 == 0:
                logger.info(f"  Chunk {chunk_idx+1}/{num_chunks} step {step}/{num_steps} "
                           f"({step_elapsed:.2f}s for 8 steps, ~{step_elapsed/8*1000:.0f}ms/step)")

        # ── GAE for this chunk ──
        logger.debug(f"  Computing GAE for chunk {chunk_idx+1}...")
        T, B, P = num_steps, num_envs, 4
        returns = np.zeros((T, B), dtype=np.float32)
        for b in range(B):
            running_ret = np.zeros(P, dtype=np.float32)
            for t in reversed(range(T)):
                if done_seq[t][b]:
                    running_ret = np.zeros(P, dtype=np.float32)
                r_t = rew_seq[t][b].numpy()
                running_ret = r_t + gamma * running_ret
                p = cp_seq[t][b]
                returns[t, b] = running_ret[p]

        returns = returns / max_reward

        # Flatten & store (skip samples where action is not in mask)
        skipped = 0
        for b in range(B):
            for t in range(T):
                mask = mask_seq[t][b]
                action = act_seq[t][b]
                if not mask[action]:
                    skipped += 1
                    continue
                data_obs.append(obs_seq[t][b])
                data_act.append(action)
                data_mask.append(mask)
                data_ret.append(returns[t, b])
        if skipped > 0:
            logger.warning(f"  Skipped {skipped} bad samples (action not in legal mask)")

        total_steps += T * B

        chunk_elapsed = time.time() - chunk_start
        progress_pct = min(100.0, 100.0 * total_steps / num_samples)
        logger.info(f"  Chunk {chunk_idx+1}/{num_chunks} done in {chunk_elapsed:.1f}s | "
                    f"samples: {total_steps}/{num_samples} ({progress_pct:.0f}%) | "
                    f"games/env: {env_game_count} | "
                    f"current_env_steps: {env_step_count}")

        if total_steps >= num_samples:
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
    logger.info(f"✓ Collected {N} samples in {elapsed:.1f}s ({N/elapsed:.0f} samples/s) → {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="red_mahjong")
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    collect_data(**{k: v for k, v in vars(args).items() if k != "debug"})
