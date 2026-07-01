#!/usr/bin/env python3
"""Offline data collector using rule-based players (PyTorch port).

Ported from examples/collect_offline_data.py.
"""

import os
import sys
import pickle
import time
import numpy as np

import torch

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.auto_reset_wrapper import auto_reset
from mahjax_pt.red_mahjong.players import rule_based_player
from mahjax_pt.examples.common import default_dataset_path, attach_dataset_metadata


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

    env = make_env(env_name, round_mode="single", observe_type="dict")
    step_env = auto_reset(env.step, env.init)

    gen = torch.Generator().manual_seed(seed)

    # Initialize parallel envs
    states = []
    for i in range(num_envs):
        g = torch.Generator().manual_seed(seed + i)
        states.append(env.init(g))

    data_obs = []
    data_act = []
    data_mask = []
    data_ret = []

    chunk_size = num_envs * num_steps
    num_chunks = (num_samples + chunk_size - 1) // chunk_size
    total_steps = 0
    start_time = time.time()

    for chunk_idx in range(num_chunks):
        obs_seq = []
        act_seq = []
        mask_seq = []
        rew_seq = []
        done_seq = []
        cp_seq = []

        for step in range(num_steps):
            o_list = []
            a_list = []
            m_list = []
            r_list = []
            d_list = []
            c_list = []
            for i in range(num_envs):
                s = states[i]
                obs = env.observe(s)
                mask = s.legal_action_mask
                cp = s.current_player

                g = torch.Generator().manual_seed(seed + chunk_idx * 10000 + step * num_envs + i)
                action = rule_based_player(s, g)

                next_s = step_env(s, action, g)
                done = next_s.terminated or next_s.truncated
                reward = next_s.rewards.clone()

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

        # Compute returns (reverse MC)
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

        # Flatten & store
        for b in range(B):
            for t in range(T):
                data_obs.append(obs_seq[t][b])
                data_act.append(act_seq[t][b])
                data_mask.append(mask_seq[t][b])
                data_ret.append(returns[t, b])

        total_steps += T * B
        if total_steps >= num_samples:
            break

    # Trim and pack
    N = num_samples
    data_obs = data_obs[:N]
    data_act = np.array(data_act[:N], dtype=np.int32)
    data_mask = torch.stack([m.clone().detach() for m in data_mask[:N]])
    data_ret = np.array(data_ret[:N], dtype=np.float32)

    # Stack observations
    stacked_obs = {}
    for key in data_obs[0].keys():
        stacked_obs[key] = torch.stack([o[key] for o in data_obs])

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
    print(f"Collected {N} samples in {elapsed:.1f}s → {dataset_path}")
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
    args = parser.parse_args()
    collect_data(**vars(args))
