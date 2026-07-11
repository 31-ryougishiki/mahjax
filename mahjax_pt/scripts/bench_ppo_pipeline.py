#!/usr/bin/env python3
"""Benchmark PPO pipeline hot paths — observe, network, GAE, step.

Usage:
    PYTHONPATH=. python mahjax_pt/tests/bench_ppo_pipeline.py [--device cuda:0] [-B 128]
"""

import time
import argparse
import torch
import numpy as np


def human(v: float) -> str:
    if v >= 1.0:
        return f"{v:.2f}s"
    elif v >= 0.001:
        return f"{v*1000:.1f}ms"
    else:
        return f"{v*1_000_000:.0f}us"


def bench(name, fn, warmup=3, repeat=20):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.min(times), np.mean(times), np.max(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-B", "--num_envs", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device)
    B = args.num_envs
    T = 256

    print(f"═══ PPO Hot-Path Benchmark ═══")
    print(f"Device: {device}  |  Batch: {B}  |  Steps: {T}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")

    # ── Imports ──────────────────────────────────────────────────
    from mahjax_pt.red_mahjong.env import make as make_env
    from mahjax_pt.red_mahjong.observation import _observe_dict_batch
    from mahjax_pt.examples.common import get_network_cls
    from mahjax_pt.examples.ppo_with_reg import (
        observe_batch_bridge, compute_gae_vectorized, masked_mean,
    )

    NET = get_network_cls("red_mahjong")().to(device)
    NET.eval()

    # ═══════════════════════════════════════════════════════════════
    # 1. Env Init
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 1. Env Init ──")
    env = make_env("red_mahjong", backend="parallel", round_mode="single",
                   observe_type="dict")
    t_min, t_avg, t_max = bench("init", lambda: env.init_batch(num_envs=B, device=device),
                                warmup=1, repeat=5)
    print(f"  init_batch({B}):  {human(t_avg)}  ({human(t_min)} min)")

    bs = env.init_batch(num_envs=B, device=device)

    # ═══════════════════════════════════════════════════════════════
    # 2. Observe — current bridge vs goal path
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 2. Observe ──")

    # Goal: direct _observe_dict_batch
    t_min, t_avg, t_max = bench("direct", lambda: _observe_dict_batch(bs),
                                warmup=5, repeat=30)
    obs_direct = t_avg
    print(f"  _observe_dict_batch (goal):      {human(t_avg)}  ({human(t_min)} min)")

    # Current: observe_batch → List[dict]
    t_min, t_avg, t_max = bench("current",
                                lambda: env.observe_batch(bs), warmup=3, repeat=10)
    obs_list = t_avg
    print(f"  observe_batch (List[dict]):      {human(t_avg)}")

    # Stack overhead
    raw = env.observe_batch(bs)
    t_min, t_avg, t_max = bench("stack",
                                lambda: _stack_list(raw, device), warmup=3, repeat=20)
    obs_stack = t_avg
    print(f"  _stack_obs_list overhead:        {human(t_avg)}")
    print(f"  ────────────────────────────────")
    speedup = (obs_list + obs_stack) / max(obs_direct, 1e-9)
    print(f"  Current total:  {human(obs_list + obs_stack)}")
    print(f"  Goal total:     {human(obs_direct)}  ({speedup:.0f}x faster)")

    # ═══════════════════════════════════════════════════════════════
    # 3. Network Forward
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 3. Network Forward ──")
    obs_batch = _observe_dict_batch(bs)
    mask = bs.legal_action_mask

    def net_forward():
        with torch.no_grad():
            logits, values = NET(obs_batch)
            logits = torch.where(mask, logits, torch.full_like(logits, -1e9))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            lp = dist.log_prob(a)
        return logits, values, a, lp

    t_min, t_avg, t_max = bench("net forward", net_forward, warmup=5, repeat=30)
    net_time = t_avg
    print(f"  Forward + sample ({B} envs):  {human(t_avg)}  ({human(t_min)} min)")
    print(f"  Per-env:                      {t_avg/B*1000:.3f}ms")

    # ═══════════════════════════════════════════════════════════════
    # 4. Env Step — known-good from openspec
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 4. Env Step ──")
    known = {128: 0.612, 512: 0.723, 1024: 0.801, 2048: 0.759}
    step_time = known.get(B, 0.612 * (B / 128))
    src = "openspec Phase 7" if B in known else "estimated"
    print(f"  step_batch({B}):  {human(step_time)}  (from {src})")

    # ═══════════════════════════════════════════════════════════════
    # 5. Per-Step Rollout Breakdown
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 5. Per-Step Breakdown ──")
    per_step = obs_direct + net_time + step_time
    per_step_cur = (obs_list + obs_stack) + net_time + step_time

    print(f"  {'Component':20s} {'Time':>10s}  {'%':>5s}")
    print(f"  {'─'*20} {'─'*10} {'─'*5}")
    for name, t in [("observe (goal)", obs_direct), ("network", net_time),
                     ("step_batch", step_time)]:
        print(f"  {name:20s} {human(t):>10s}  {t/per_step*100:>4.0f}%")
    print(f"  {'─'*20} {'─'*10} {'─'*5}")
    print(f"  {'Total per step':20s} {human(per_step):>10s}")

    rollout_s = per_step * T
    rollout_cur = per_step_cur * T
    print(f"\n  Rollout ({T} steps, goal):     {human(rollout_s)}  ({B*T/rollout_s:.0f} env-steps/s)")
    print(f"  Rollout ({T} steps, current):  {human(rollout_cur)}  ({B*T/rollout_cur:.0f} env-steps/s)")

    # ═══════════════════════════════════════════════════════════════
    # 6. GAE
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 6. GAE ──")
    rewards = torch.randn(T, B, 4, device=device) * 0.1
    values = torch.randn(T, B, device=device) * 0.5
    dones_g = torch.rand(T, B, device=device) < 0.02
    cps_g = torch.randint(0, 4, (T, B), device=device)

    t_min, t_avg, t_max = bench("GAE",
                                lambda: compute_gae_vectorized(rewards, values, dones_g, cps_g),
                                warmup=3, repeat=15)
    gae_time = t_avg
    print(f"  compute_gae:  {human(t_avg)}  ({T*B/t_avg:.0f} transitions/s)")

    # ═══════════════════════════════════════════════════════════════
    # 7. PPO Update — estimated from forward time
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 7. PPO Update (estimated) ──")
    BATCH_FULL = T * B
    MB = min(4096, BATCH_FULL)
    batches_per_epoch = max(BATCH_FULL // MB, 1)
    epochs = 4
    # Scale forward from B envs to MB batch; backward ≈ 2x forward
    fw_per_mb = net_time * (MB / B)
    bw_per_mb = fw_per_mb * 2.0
    mb_est = fw_per_mb + bw_per_mb
    est_update = mb_est * batches_per_epoch * epochs

    print(f"  Network forward ({B}):        {human(net_time)}")
    print(f"  Est fwd per MB ({MB}):        {human(fw_per_mb)}")
    print(f"  Est bwd per MB:               {human(bw_per_mb)}")
    print(f"  Est minibatch:                {human(mb_est)}")
    print(f"  Batches/epoch:                {batches_per_epoch}")
    print(f"  Epochs:                       {epochs}")
    print(f"  Est update total:             {human(est_update)}")

    # ═══════════════════════════════════════════════════════════════
    # 8. Full Update Cycle Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n═══ 8. Full Update Cycle ({B} envs x {T} steps) ═══")
    total_goal = rollout_s + gae_time + est_update
    total_cur = rollout_cur + gae_time + est_update
    env_total = B * T

    print(f"  {'Phase':20s} {'Current':>10s}  {'Goal':>10s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    print(f"  {'Rollout':20s} {human(rollout_cur):>10s}  {human(rollout_s):>10s}")
    print(f"  {'GAE':20s} {human(gae_time):>10s}  {human(gae_time):>10s}")
    print(f"  {'PPO Update':20s} {human(est_update):>10s}  {human(est_update):>10s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    print(f"  {'Total':20s} {human(total_cur):>10s}  {human(total_goal):>10s}")

    eps_goal = env_total / max(total_goal, 1e-9)
    eps_cur = env_total / max(total_cur, 1e-9)
    print(f"\n  Updates/s:        {1.0/total_goal:.3f} (goal)")
    print(f"  Env-steps/s:      {eps_goal:.0f} (goal)  vs  {eps_cur:.0f} (current)")
    print(f"  Speedup:          {eps_goal/max(eps_cur,1e-9):.1f}x")

    # ═══════════════════════════════════════════════════════════════
    # 9. Scaling
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── 9. Scaling Estimates (Goal Path) ──")
    step_k = {128: 0.612, 512: 0.723, 1024: 0.801, 2048: 0.759}
    print(f"  {'B':>5s}  {'step':>8s}  {'rollout':>8s}  {'total':>8s}  "
          f"{'env-steps/s':>11s}  {'1e8 steps':>9s}")

    for B_est in [256, 512, 1024, 2048]:
        st = step_k.get(B_est, 0.612 * (B_est / 128))
        # Sub-linear GPU scaling for observe/network
        obs_s = obs_direct * (B_est / B) ** 0.6
        net_s = net_time * (B_est / B)
        ps = obs_s + net_s + st
        rl = ps * T
        bt = B_est * T
        gae_s = gae_time * (B_est / B) * (256 / T)
        mb = min(4096, bt)
        bpe = max(bt // mb, 1)
        fw_mb = net_s * (mb / B_est)
        upd = (fw_mb + fw_mb * 2.0) * bpe * 4
        tot = rl + gae_s + upd
        eps = bt / max(tot, 1e-9)
        # Time for 1e8 total timesteps
        num_upd = 100_000_000 // bt
        hrs = tot * num_upd / 3600
        print(f"  {B_est:5d}  {human(st):>8s}  {human(rl):>8s}  {human(tot):>8s}  "
              f"{eps:>9.0f}/s  {hrs:>7.1f}h")

    # ═══════════════════════════════════════════════════════════════
    # 10. GPU Memory
    # ═══════════════════════════════════════════════════════════════
    if device.type == "cuda":
        print(f"\n── 10. GPU Memory ──")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
        print(f"  Peak:      {torch.cuda.max_memory_allocated()/1024**2:.0f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved()/1024**2:.0f} MB")

    # ═══════════════════════════════════════════════════════════════
    # Diagnosis
    # ═══════════════════════════════════════════════════════════════
    print(f"\n═══ Diagnosis ═══")
    parts = [
        ("observe", obs_direct, per_step),
        ("network", net_time, per_step),
        ("step_batch", step_time, per_step),
    ]
    parts.sort(key=lambda x: -x[1])
    print(f"  Per-step bottleneck: {parts[0][0]} ({parts[0][1]/per_step*100:.0f}%)")

    if obs_list + obs_stack > obs_direct * 10:
        print(f"  !! observe_batch bridge: {human(obs_list+obs_stack)} -> {human(obs_direct)} "
              f"({speedup:.0f}x gap)")
        print(f"     Fix: env_parallel.observe_batch -> _observe_dict_batch")

    print(f"\n  Reference: JAX PPO target ~200-500k env-steps/s on TPU")
    print(f"  Current:   ~{eps_goal:.0f} env-steps/s (goal path)")

    print(f"\n═══ Done ═══")


def _stack_list(obs_list, device):
    keys = list(obs_list[0].keys())
    stacked = {}
    for k in keys:
        vals = [o[k] if isinstance(o[k], torch.Tensor) else torch.tensor(o[k])
                for o in obs_list]
        stacked[k] = torch.stack(vals).to(device)
    return stacked


if __name__ == "__main__":
    main()
