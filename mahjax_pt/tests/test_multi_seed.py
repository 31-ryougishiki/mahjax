#!/usr/bin/env python3
"""Multi-seed JAX vs PT alignment test with multiprocessing support.

Usage:
  python test_multi_seed.py              # all 10 seeds, serial
  python test_multi_seed.py -j 4         # all 10 seeds, 4 workers
  python test_multi_seed.py 13           # seed 13 only
  python test_multi_seed.py 13 42 99     # specific seeds
"""
import jax; jax.config.update('jax_disable_jit', True)
import numpy as np, torch, sys, time
from multiprocessing import cpu_count
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv


def copy_to_pt(js, ps):
    ps.current_player = int(js.current_player)
    ps.terminated = bool(js.terminated)
    ps.legal_action_mask = torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    ps.rewards = torch.from_numpy(np.array(js.rewards, dtype=np.float32).copy()).float()
    jp, pp = js.players, ps.players
    for f, dt in [('hand', torch.int8), ('hand_with_red', torch.int8),
                  ('hand_ids', torch.int16), ('hand_counts', torch.int8),
                  ('drawn_tile', torch.int16),
                  ('melds', torch.int32), ('meld_counts', torch.int8),
                  ('meld_tiles', torch.int16), ('meld_info', torch.int8),
                  ('riichi', torch.bool), ('riichi_declared', torch.bool),
                  ('riichi_step', torch.int8), ('double_riichi', torch.bool),
                  ('ippatsu', torch.bool),
                  ('furiten_by_discard', torch.bool), ('furiten_by_pass', torch.bool),
                  ('is_hand_concealed', torch.bool), ('has_won', torch.bool),
                  ('n_kan', torch.int8), ('discard_counts', torch.int8),
                  ('discards', torch.int16), ('discard_info', torch.int8),
                  ('river', torch.int32),
                  ('has_yaku', torch.bool), ('fan', torch.int32), ('fu', torch.int32),
                  ('can_win', torch.bool), ('pon', torch.int32),
                  ('has_nagashi_mangan', torch.bool)]:
        setattr(pp, f, torch.from_numpy(np.array(getattr(jp, f)).copy()).to(dt))
    jr, pr = js.round_state, ps.round_state
    for f in ['round', 'honba', 'kyotaku', 'dealer', 'next_deck_ix', 'last_deck_ix',
              'last_draw', 'last_player', 'target', 'n_kan_doras',
              'shanten_current_player', 'dummy_count', 'round_limit']:
        setattr(pr, f, int(getattr(jr, f)))
    for f in ['terminated_round', 'draw_next', 'is_haitei', 'kan_declared', 'can_after_kan',
              'can_robbing_kan', 'is_abortive_draw_normal']:
        setattr(pr, f, bool(getattr(jr, f)))
    for f, dt in [('deck', torch.int8), ('score', torch.int32), ('dora_indicators', torch.int8),
                  ('ura_dora_indicators', torch.int8),
                  ('seat_wind', torch.int8), ('init_wind', torch.int8),
                  ('order_points', torch.int32), ('action_history', torch.int8)]:
        setattr(pr, f, torch.from_numpy(np.array(getattr(jr, f)).copy()).to(dt))
    return ps


def jv(val):
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if hasattr(val, '__array__'):
        return np.array(val)
    return np.asarray(val)


CHECKS = [
    ('hand',              lambda s: s.players.hand),
    ('deck',              lambda s: s.round_state.deck),
    ('river',             lambda s: s.players.river),
    ('melds',             lambda s: s.players.melds),
    ('meld_counts',       lambda s: s.players.meld_counts),
    ('discard_counts',    lambda s: s.players.discard_counts),
    ('score',             lambda s: s.round_state.score),
    ('current_player',    lambda s: s.current_player),
    ('terminated_round',  lambda s: s.round_state.terminated_round),
    ('terminated',        lambda s: s.terminated),
    ('draw_next',         lambda s: s.round_state.draw_next),
    ('is_haitei',         lambda s: s.round_state.is_haitei),
    ('is_abortive',       lambda s: s.round_state.is_abortive_draw_normal),
    ('target',            lambda s: s.round_state.target),
    ('last_draw',         lambda s: s.round_state.last_draw),
    ('last_player',       lambda s: s.round_state.last_player),
    ('riichi',            lambda s: s.players.riichi),
    ('riichi_declared',   lambda s: s.players.riichi_declared),
    ('has_won',           lambda s: s.players.has_won),
    ('n_kan',             lambda s: s.players.n_kan),
    ('has_yaku',          lambda s: s.players.has_yaku),
    ('rewards',           lambda s: s.rewards),
]


def compare_all(js, ps):
    diffs = []
    for name, fn in CHECKS:
        jv_val = jv(fn(js))
        pv_val = jv(fn(ps))
        if not np.array_equal(jv_val, pv_val):
            diffs.append(name)
    return diffs


def describe_diff(name, jv_val, pv_val):
    jv_val = np.asarray(jv_val)
    pv_val = np.asarray(pv_val)
    if jv_val.size < 20:
        return f"JAX={jv_val.tolist()} PT={pv_val.tolist()}"
    else:
        n = int(np.sum(jv_val != pv_val))
        return f"{n} elems differ, JAX range [{jv_val.min()},{jv_val.max()}] PT range [{pv_val.min()},{pv_val.max()}]"


def test_seed(seed):
    """Run a single seed test. Returns (seed, ok, details_dict)."""
    import os as _os
    try:
        t0 = time.time()
        sys.stderr.write(f"[pid={_os.getpid()}] seed={seed} starting...\n")
        sys.stderr.flush()
        jenv = JaxEnv(round_mode='single')
        penv = PtEnv(round_mode='single')
        sys.stderr.write(f"[pid={_os.getpid()}] seed={seed} init...\n"); sys.stderr.flush()
        js = jenv.init(jax.random.PRNGKey(seed))
        ps = penv.init(key=0)
        ps = copy_to_pt(js, ps)
        sys.stderr.write(f"[pid={_os.getpid()}] seed={seed} started, running steps...\n"); sys.stderr.flush()

        ok_steps = 0
        fail_step = fail_act = fail_fields = None
        prev_action = -1

        for step in range(200):
            if bool(js.terminated) or bool(js.round_state.terminated_round):
                break
            legal = np.where(np.array(js.legal_action_mask))[0]
            # Prioritize all action types for coverage (mirrors rule_based_player):
            #   win > kan > meld > riichi > kyushu > discard
            a = None
            for candidate in [
                73, 74, 72,                          # RON, TSUMO, RIICHI (win)
                77,                                   # OPEN_KAN
                75, 76,                               # PON, PON_RED
                78, 79, 80, 81, 82, 83,              # CHI_L..CHI_R_RED
            ] + list(range(70, 36, -1)):              # selfkan (70..37)
                if candidate in legal:
                    a = candidate; break
            if a is None:
                # kyushu or discard or anything else
                a = int(legal[0])
            prev_action = a
            a = int(a)
            js = jenv.step(js, a)
            ps = penv.step(ps, a)
            diffs = compare_all(js, ps)
            if diffs:
                fail_step = step
                fail_act = a
                fail_fields = diffs
                # Dump detailed context for the failing step
                sys.stderr.write(f"[FAIL] step={step} act={a} prev_act={prev_action} cp={int(js.current_player)} fields={diffs}\n")
                sys.stderr.write(f"  legal: {list(legal)}\n")
                for name in diffs:
                    fn = dict(CHECKS)[name]
                    jv_val, pv_val = jv(fn(js)), jv(fn(ps))
                    sys.stderr.write(f"  {name}: {describe_diff(name, jv_val, pv_val)}\n")
                sys.stderr.flush()
                break
            ok_steps += 1
            if step % 20 == 19:
                sys.stderr.write(f"[pid={_os.getpid()}] seed={seed} step={step+1}\n"); sys.stderr.flush()

        if fail_step is None:
            diffs = compare_all(js, ps)
            if diffs:
                fail_step = 'END'
                fail_act = '-'
                fail_fields = diffs

        dt = time.time() - t0
        sys.stderr.write(f"[pid={_os.getpid()}] seed={seed} done ({ok_steps} steps, {dt:.0f}s)\n"); sys.stderr.flush()
        if fail_step is None:
            return (seed, True, {'steps': ok_steps, 'time': dt, 'error': None})
        else:
            details = {
                'steps': ok_steps,
                'time': dt,
                'fail_step': fail_step,
                'fail_act': fail_act,
                'fail_fields': fail_fields,
            }
            for name in fail_fields:
                fn = dict(CHECKS)[name]
                jv_val, pv_val = jv(fn(js)), jv(fn(ps))
                details[f'diff_{name}'] = describe_diff(name, jv_val, pv_val)
            return (seed, False, details)
    except Exception as e:
        import traceback
        sys.stderr.write(f"[pid={_os.getpid()}] seed={seed} CRASH: {e}\n{traceback.format_exc()}\n")
        sys.stderr.flush()
        return (seed, False, {'steps': 0, 'time': 0, 'error': str(e)})


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('seeds', nargs='*', type=int, help='Seeds to test (default: all 10)')
    p.add_argument('-j', '--jobs', type=int, default=1, help='Number of parallel workers')
    args = p.parse_args()

    if args.seeds:
        SEEDS = args.seeds
    else:
        SEEDS = [1, 7, 13, 42, 99, 123, 256, 512, 1024, 2048]

    if args.jobs > 1:
        import multiprocessing as _mp
        n_workers = min(args.jobs, len(SEEDS))
        print(f"Running {len(SEEDS)} seeds with {n_workers} workers...", flush=True)
        sys.stderr.write(f"Spawning {n_workers} workers...\n"); sys.stderr.flush()
        with _mp.get_context('spawn').Pool(n_workers) as pool:
            sys.stderr.write(f"Pool created, mapping seeds...\n"); sys.stderr.flush()
            results = pool.map(test_seed, SEEDS)
            sys.stderr.write(f"All workers done.\n"); sys.stderr.flush()
    else:
        print(f"Testing {len(SEEDS)} seeds (serial): {SEEDS}", flush=True)
        results = [test_seed(s) for s in SEEDS]

    # Print results in seed order
    n_pass = 0
    for seed, ok, d in results:
        if ok:
            print(f"seed={seed:4d}: OK ({d['steps']} steps, {d['time']:.0f}s)", flush=True)
            n_pass += 1
        else:
            print(f"seed={seed:4d}: FAIL step={d['fail_step']} act={d['fail_act']} fields={d['fail_fields']} ({d['time']:.0f}s)", flush=True)
            for name in d['fail_fields']:
                key = f'diff_{name}'
                if key in d:
                    print(f"  {name}: {d[key]}", flush=True)

    print(f"\nPassed: {n_pass}/{len(SEEDS)} seeds", flush=True)
    if n_pass == len(SEEDS):
        print("ALL SEEDS PASS!", flush=True)
    else:
        for seed, ok, d in results:
            if not ok:
                print(f"  seed={seed} FAILED: {d['fail_fields']}", flush=True)
        sys.exit(1)
