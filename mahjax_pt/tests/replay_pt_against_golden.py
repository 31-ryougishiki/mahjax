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
    ('players.legal_action_mask', lambda s,n: s.players.legal_action_mask, 'exact'),
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


def _copy_golden_to_pt(golden, state):
    """Copy JAX golden init state into a PT state — explicit per-field to avoid bugs."""
    # Top-level env fields
    state.current_player = int(golden['current_player'])
    state.terminated = bool(golden['terminated'])
    # Env-level mask must match the current player's per-player mask from JAX
    cp = state.current_player
    state.legal_action_mask = torch.from_numpy(golden['players.legal_action_mask'][cp].copy()).bool()
    state.truncated = bool(golden.get('truncated', False))
    state.step_count = int(golden.get('step_count', 0))
    state.rewards = torch.from_numpy(np.array(golden['rewards'], dtype=np.float32)).float()

    # Player fields — manually copy each to ensure correctness
    G = golden
    pp = state.players
    pp.hand[:] = torch.from_numpy(G['players.hand']).to(pp.hand.dtype)
    pp.hand_with_red[:] = torch.from_numpy(G['players.hand_with_red']).to(pp.hand_with_red.dtype)
    pp.melds[:] = torch.from_numpy(G['players.melds']).to(pp.melds.dtype)
    pp.meld_counts[:] = torch.from_numpy(G['players.meld_counts']).to(pp.meld_counts.dtype)
    pp.discard_counts[:] = torch.from_numpy(G['players.discard_counts']).to(pp.discard_counts.dtype)
    pp.river[:] = torch.from_numpy(G['players.river']).to(pp.river.dtype)
    pp.riichi[:] = torch.from_numpy(G['players.riichi']).to(pp.riichi.dtype)
    pp.riichi_declared[:] = torch.from_numpy(G['players.riichi_declared']).to(pp.riichi_declared.dtype)
    pp.has_won[:] = torch.from_numpy(G['players.has_won']).to(pp.has_won.dtype)
    pp.n_kan[:] = torch.from_numpy(G['players.n_kan']).to(pp.n_kan.dtype)
    pp.has_yaku[:] = torch.from_numpy(G['players.has_yaku']).to(pp.has_yaku.dtype)
    pp.is_hand_concealed[:] = torch.from_numpy(G['players.is_hand_concealed']).to(pp.is_hand_concealed.dtype)
    pp.furiten_by_discard[:] = torch.from_numpy(G['players.furiten_by_discard']).to(pp.furiten_by_discard.dtype)
    pp.furiten_by_pass[:] = torch.from_numpy(G['players.furiten_by_pass']).to(pp.furiten_by_pass.dtype)
    pp.ippatsu[:] = torch.from_numpy(G['players.ippatsu']).to(pp.ippatsu.dtype)
    pp.fan[:] = torch.from_numpy(G['players.fan']).to(pp.fan.dtype)
    pp.fu[:] = torch.from_numpy(G['players.fu']).to(pp.fu.dtype)
    pp.legal_action_mask[:] = torch.from_numpy(G['players.legal_action_mask']).to(pp.legal_action_mask.dtype)

    # Round state fields
    pr = state.round_state
    pr.round = int(G['round_state.round'])
    pr.honba = int(G['round_state.honba'])
    pr.kyotaku = int(G['round_state.kyotaku'])
    pr.dealer = int(G['round_state.dealer'])
    pr.next_deck_ix = int(G['round_state.next_deck_ix'])
    pr.last_deck_ix = int(G['round_state.last_deck_ix'])
    pr.last_draw = int(G['round_state.last_draw'])
    pr.last_player = int(G['round_state.last_player'])
    pr.target = int(G['round_state.target'])
    pr.n_kan_doras = int(G.get('round_state.n_kan_doras', 0))
    pr.terminated_round = bool(G['round_state.terminated_round'])
    pr.draw_next = bool(G['round_state.draw_next'])
    pr.is_haitei = bool(G['round_state.is_haitei'])
    pr.is_abortive_draw_normal = bool(G['round_state.is_abortive_draw_normal'])
    pr.kan_declared = bool(G.get('round_state.kan_declared', False))
    pr.can_after_kan = bool(G.get('round_state.can_after_kan', False))
    pr.deck[:] = torch.from_numpy(G['round_state.deck']).to(pr.deck.dtype)
    pr.score[:] = torch.from_numpy(G['round_state.score']).to(pr.score.dtype)
    pr.dora_indicators[:] = torch.from_numpy(G['round_state.dora_indicators']).to(pr.dora_indicators.dtype)
    pr.order_points[:] = torch.from_numpy(G['round_state.order_points']).to(pr.order_points.dtype)
    pr.seat_wind[:] = torch.from_numpy(G['round_state.seat_wind']).to(pr.seat_wind.dtype)


def compare_one(golden_val, pt_val, tolerance):
    if tolerance == 'exact':
        return np.array_equal(np.asarray(golden_val), np.asarray(pt_val))
    else:
        return np.allclose(np.asarray(golden_val), np.asarray(pt_val), rtol=1e-3, atol=1e-5)


def replay_seed(seed, init_state, records, verbose=False):
    """Replay one seed's records against PT. Returns (seed, ok, details)."""
    t0 = time.time()
    penv = PtEnv(round_mode='single')
    state = penv.init(key=0)
    # Copy JAX init state into PT so we start from the same point
    _copy_golden_to_pt(init_state, state)

    ok_steps = 0
    first_fail = None

    for step, rec in enumerate(records):
        action = rec['action']
        golden = rec['state']

        state = penv.step(state, action)

        diffs = []
        for name, accessor_fn, tol in CHECKS:
            if name not in golden:
                continue
            gv = golden[name]
            pv = pt_val(accessor_fn(state, name))
            if not compare_one(gv, pv, tol):
                diffs.append(name)

        if diffs:
            first_fail = (step, action, diffs)
            if verbose:
                # Build accessor map to avoid loop variable capture bug
                amap = dict((n, a) for n, a, t in CHECKS)
                for name in diffs[:8]:
                    gv = golden[name]
                    pv = pt_val(amap[name](state, name))
                    gv_np = np.asarray(gv); pv_np = np.asarray(pv)
                    if gv_np.shape != pv_np.shape:
                        sys.stderr.write(f"  {name}: SHAPE MISMATCH G={gv_np.shape} P={pv_np.shape}\n")
                    elif gv_np.size > 20:
                        n_diff = int(np.sum(gv_np != pv_np))
                        idx = np.where(gv_np != pv_np)
                        # For mask: print actual True/False values at diff positions
                        if 'mask' in name.lower() and n_diff < 30:
                            sys.stderr.write(f"  {name}: {n_diff}/{gv_np.size} diffs\n")
                            for i in range(min(n_diff, 15)):
                                pi, ai = idx[0][i], idx[1][i]
                                sys.stderr.write(f"    P{pi} act{ai}: G={gv_np[pi,ai]} P={pv_np[pi,ai]}\n")
                        else:
                            sys.stderr.write(f"  {name}: {n_diff}/{gv_np.size} diffs, first at {list(zip(idx[0][:5], idx[1][:5]))}\n")
                        n_diff = int(np.sum(gv_np != pv_np))
                        idx = np.where(gv_np != pv_np)
                        first_idx = list(zip(idx[0][:5].tolist(), idx[1][:5].tolist())) if gv_np.ndim > 1 else idx[0][:10].tolist()
                        sys.stderr.write(f"  {name}: {n_diff}/{gv_np.size} diffs, first at {first_idx}\n")
                        if n_diff < 20:
                            for i in first_idx if gv_np.ndim==1 else range(min(5, n_diff)):
                                if gv_np.ndim == 1:
                                    j = idx[0][i]
                                    sys.stderr.write(f"    [{j}] G={gv_np[j]} P={pv_np[j]}\n")
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
