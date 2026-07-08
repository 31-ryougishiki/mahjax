#!/usr/bin/env python3
"""Replay PT parallel env against recorded JAX golden data.

Usage:
  python replay_parallel_against_golden.py                    # all seeds
  python replay_parallel_against_golden.py -s 13 42           # specific seeds
  python replay_parallel_against_golden.py --list             # list available seeds

This is the parallel-env counterpart of replay_pt_against_golden.py.
It verifies that RedMahjongParallel produces identical results to JAX.
"""
import os, sys, time, pickle
import numpy as np, torch
from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel as PtEnv

# Reuse the field-check definitions and helpers from the serial replay script
from replay_pt_against_golden import (
    CHECKS, pt_val, compare_one, _copy_golden_to_pt,
)


def replay_seed(seed, init_state, records, verbose=False):
    """Replay one seed's records against PT parallel env. Returns (seed, ok, details)."""
    t0 = time.time()
    penv = PtEnv(round_mode='single')
    state = penv.init(key=0)
    _copy_golden_to_pt(init_state, state)

    ok_steps = 0
    first_fail = None

    for step_idx, rec in enumerate(records):
        action = rec['action']
        golden = rec['state']

        # Use step_batch with a single-env list — exercises the full
        # action-classification + batch-handler + serial-fallback pipeline.
        try:
            result_list = penv.step_batch([state], [action])
            state = result_list[0]
        except Exception as e:
            first_fail = (step_idx, action, [f"EXCEPTION: {e}"])
            break

        diffs = []
        for name, accessor_fn, tol in CHECKS:
            if name not in golden:
                continue
            gv = golden[name]
            try:
                pv = pt_val(accessor_fn(state, name))
            except Exception:
                diffs.append(f"{name}:accessor_error")
                continue
            if not compare_one(gv, pv, tol):
                diffs.append(name)

        if diffs:
            first_fail = (step_idx, action, diffs)
            if verbose:
                amap = dict((n, a) for n, a, t in CHECKS)
                for name in diffs[:8]:
                    if name.endswith(':accessor_error') or 'EXCEPTION' in name:
                        sys.stderr.write(f"  {name}\n")
                        continue
                    gv = golden[name]
                    pv = pt_val(amap[name](state, name))
                    gv_np = np.asarray(gv); pv_np = np.asarray(pv)
                    if gv_np.shape != pv_np.shape:
                        sys.stderr.write(f"  {name}: SHAPE MISMATCH G={gv_np.shape} P={pv_np.shape}\n")
                    elif gv_np.size > 20:
                        n_diff = int(np.sum(gv_np != pv_np))
                        idx = np.where(gv_np != pv_np)
                        if 'mask' in name.lower() and n_diff < 30:
                            sys.stderr.write(f"  {name}: {n_diff}/{gv_np.size} diffs\n")
                            for i in range(min(n_diff, 15)):
                                if gv_np.ndim == 1:
                                    ai = int(idx[0][i])
                                    sys.stderr.write(f"    act{ai}: G={gv_np[ai]} P={pv_np[ai]}\n")
                                else:
                                    pi, ai = int(idx[0][i]), int(idx[1][i])
                                    sys.stderr.write(f"    P{pi} act{ai}: G={gv_np[pi,ai]} P={pv_np[pi,ai]}\n")
                        else:
                            if gv_np.ndim >= 2:
                                sys.stderr.write(f"  {name}: {n_diff}/{gv_np.size} diffs, first at {list(zip(idx[0][:5], idx[1][:5]))}\n")
                            else:
                                sys.stderr.write(f"  {name}: {n_diff}/{gv_np.size} diffs, first at {idx[0][:10].tolist()}\n")
                    else:
                        sys.stderr.write(f"  {name}: G={gv_np.tolist()} P={pv_np.tolist()}\n")
            break
        ok_steps += 1

    dt = time.time() - t0
    if first_fail is None:
        return (seed, True, {'steps': ok_steps, 'time': dt})
    else:
        s, a, f = first_fail
        return (seed, False, {'steps': ok_steps, 'time': dt, 'fail_step': s, 'fail_act': a, 'fail_fields': f})


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data-dir', default='golden_data', help='Golden data directory')
    p.add_argument('-s', '--seeds', nargs='*', type=int, help='Specific seeds')
    p.add_argument('--list', action='store_true', help='List available seeds')
    args = p.parse_args()

    if args.list:
        for f in sorted(os.listdir(args.data_dir)):
            if f.startswith('golden_seed_'):
                print(f)
        sys.exit(0)

    if args.seeds:
        files = [f'golden_seed_{s:04d}.pkl' for s in args.seeds]
    else:
        files = sorted(f for f in os.listdir(args.data_dir) if f.startswith('golden_seed_'))

    if not files:
        print(f"No golden data found in {args.data_dir}/. Run record_jax_golden.py first.")
        sys.exit(1)

    results = []
    n_pass = 0
    for fname in files:
        path = os.path.join(args.data_dir, fname)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        seed = data['seed']
        records = data['records']
        verbose = len(args.seeds or []) <= 3
        seed, ok, d = replay_seed(seed, data['init_state'], records, verbose=verbose)
        if ok:
            print(f"seed={seed:4d}: OK ({d['steps']} steps, {d['time']:.1f}s)", flush=True)
            n_pass += 1
        else:
            print(f"seed={seed:4d}: FAIL step={d['fail_step']} act={d['fail_act']} fields={d['fail_fields']} ({d['time']:.1f}s)", flush=True)
        results.append((seed, ok, d))

    print(f"\nPassed: {n_pass}/{len(files)} seeds", flush=True)
    if n_pass == len(files):
        print("ALL SEEDS PASS!", flush=True)
    else:
        for seed, ok, d in results:
            if not ok:
                print(f"  seed={seed} FAILED at step {d['fail_step']}: {d['fail_fields']}", flush=True)
        sys.exit(1)
