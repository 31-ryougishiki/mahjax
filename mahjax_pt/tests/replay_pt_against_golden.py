#!/usr/bin/env python3
"""Replay PT against recorded JAX golden data. No JAX needed at runtime.

Usage:
  python replay_pt_against_golden.py                    # all seeds in golden_data/
  python replay_pt_against_golden.py -d ./golden_data   # custom dir
  python replay_pt_against_golden.py -s 13 42           # specific seeds
  python replay_pt_against_golden.py --list             # list available seeds
"""
import os, sys, time, pickle
import numpy as np, torch
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv


# ── Field comparison setup ──
# Map golden dict keys → (pt_accessor, tolerance)
CHECKS = [
    ('current_player',           lambda s,n: s.current_player,             'exact'),
    ('terminated',               lambda s,n: s.terminated,                 'exact'),
    ('legal_action_mask',        lambda s,n: s.legal_action_mask,         'exact'),
    ('rewards',                  lambda s,n: s.rewards,                   'close'),
    ('players.hand',             lambda s,n: s.players.hand,              'exact'),
    ('players.hand_with_red',    lambda s,n: s.players.hand_with_red,     'exact'),
    ('players.melds',            lambda s,n: s.players.melds,             'exact'),
    ('players.meld_counts',      lambda s,n: s.players.meld_counts,       'exact'),
    ('players.discard_counts',   lambda s,n: s.players.discard_counts,    'exact'),
    ('players.river',            lambda s,n: s.players.river,             'exact'),
    ('players.riichi',           lambda s,n: s.players.riichi,            'exact'),
    ('players.riichi_declared',  lambda s,n: s.players.riichi_declared,   'exact'),
    ('players.has_won',          lambda s,n: s.players.has_won,           'exact'),
    ('players.n_kan',            lambda s,n: s.players.n_kan,             'exact'),
    ('players.has_yaku',         lambda s,n: s.players.has_yaku,          'exact'),
    ('players.is_hand_concealed',lambda s,n: s.players.is_hand_concealed, 'exact'),
    ('players.furiten_by_discard',lambda s,n: s.players.furiten_by_discard,'exact'),
    ('players.furiten_by_pass',  lambda s,n: s.players.furiten_by_pass,   'exact'),
    ('players.ippatsu',          lambda s,n: s.players.ippatsu,           'exact'),
    ('players.fan',              lambda s,n: s.players.fan,               'exact'),
    ('players.fu',               lambda s,n: s.players.fu,                'exact'),
    ('round_state.round',        lambda s,n: s.round_state.round,         'exact'),
    ('round_state.dealer',       lambda s,n: s.round_state.dealer,        'exact'),
    ('round_state.next_deck_ix', lambda s,n: s.round_state.next_deck_ix,  'exact'),
    ('round_state.last_deck_ix', lambda s,n: s.round_state.last_deck_ix,  'exact'),
    ('round_state.last_draw',    lambda s,n: s.round_state.last_draw,     'exact'),
    ('round_state.last_player',  lambda s,n: s.round_state.last_player,   'exact'),
    ('round_state.target',       lambda s,n: s.round_state.target,        'exact'),
    ('round_state.draw_next',    lambda s,n: s.round_state.draw_next,     'exact'),
    ('round_state.is_haitei',    lambda s,n: s.round_state.is_haitei,     'exact'),
    ('round_state.is_abortive_draw_normal', lambda s,n: s.round_state.is_abortive_draw_normal, 'exact'),
    ('round_state.terminated_round', lambda s,n: s.round_state.terminated_round, 'exact'),
    ('round_state.score',        lambda s,n: s.round_state.score,         'exact'),
    ('round_state.deck',         lambda s,n: s.round_state.deck,          'exact'),
    ('round_state.dora_indicators',lambda s,n: s.round_state.dora_indicators, 'exact'),
    ('round_state.n_kan_doras',  lambda s,n: s.round_state.n_kan_doras,   'exact'),
    ('round_state.order_points', lambda s,n: s.round_state.order_points,  'exact'),
]


def pt_val(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def compare_one(golden_val, pt_val, tolerance):
    if tolerance == 'exact':
        return np.array_equal(np.asarray(golden_val), np.asarray(pt_val))
    else:
        return np.allclose(np.asarray(golden_val), np.asarray(pt_val), rtol=1e-3, atol=1e-5)


def replay_seed(seed, records, verbose=False):
    """Replay one seed's records against PT. Returns (seed, ok, details)."""
    t0 = time.time()
    penv = PtEnv(round_mode='single')
    state = penv.init(key=0)  # dummy init, state will diverge but actions force alignment

    ok_steps = 0
    first_fail = None

    for step, rec in enumerate(records):
        action = rec['action']
        golden = rec['state']

        state = penv.step(state, action)

        diffs = []
        for name, accessor, tol in CHECKS:
            if name not in golden:
                continue
            gv = golden[name]
            pv = pt_val(accessor(state, name))
            if not compare_one(gv, pv, tol):
                diffs.append(name)

        if diffs:
            first_fail = (step, action, diffs)
            if verbose:
                for name in diffs[:5]:
                    gv = golden[name]
                    pv = pt_val(accessor(state, name))
                    sys.stderr.write(f"  {name}: G={np.asarray(gv).tolist() if np.asarray(gv).size<10 else '...'} P={np.asarray(pv).tolist() if np.asarray(pv).size<10 else '...'}\n")
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

    # Find golden files
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
        verbose = len(args.seeds or []) <= 3  # detail for small runs
        seed, ok, d = replay_seed(seed, records, verbose=verbose)
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
