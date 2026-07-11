#!/usr/bin/env python3
"""Replay PT parallel env against recorded JAX golden data.

Usage:
  python replay_parallel_against_golden.py                    # all seeds (serial)
  python replay_parallel_against_golden.py -j 4               # all seeds, 4 workers
  python replay_parallel_against_golden.py -d ./golden_data   # custom dir
  python replay_parallel_against_golden.py -s 13 42           # specific seeds
  python replay_parallel_against_golden.py --list             # list available seeds
  python replay_parallel_against_golden.py -s 13 42 -v        # verbose diff output

This is the parallel-env counterpart of replay_pt_against_golden.py.
It verifies that RedMahjongParallel produces identical results to JAX.
"""
import os, sys, time, pickle, traceback
import numpy as np, torch
from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel as PtEnv

# Reuse the field-check definitions and helpers from the serial replay script
from replay_pt_against_golden import (
    CHECKS, pt_val, compare_one, _copy_golden_to_pt, _format_diffs,
)

# ── BatchState-compatible accessors ──
# Instead of unstack_state (which does ~40 GPU .clone() calls per seed),
# we index directly into BatchState tensors. Indexing returns a view (no copy),
# then pt_val does a single GPU→CPU transfer per field.
#
# Each entry: (golden_key, lambda bs,i: tensor_slice, tolerance)
BATCH_CHECKS = [
    # Top-level env fields
    ('current_player',           lambda bs,i: bs.current_player[i],             'exact'),
    ('terminated',               lambda bs,i: bs.terminated[i],                 'exact'),
    ('legal_action_mask',        lambda bs,i: bs.legal_action_mask[i],         'exact'),
    ('rewards',                  lambda bs,i: bs.rewards[i],                   'close'),
    # Player fields — batch shape (B, 4, ...), index into B dim
    ('players.hand',             lambda bs,i: bs.players.hand[i],              'exact'),
    ('players.hand_with_red',    lambda bs,i: bs.players.hand_with_red[i],     'exact'),
    ('players.melds',            lambda bs,i: bs.players.melds[i],             'exact'),
    ('players.meld_counts',      lambda bs,i: bs.players.meld_counts[i],       'exact'),
    ('players.discard_counts',   lambda bs,i: bs.players.discard_counts[i],    'exact'),
    ('players.river',            lambda bs,i: bs.players.river[i],             'exact'),
    ('players.riichi',           lambda bs,i: bs.players.riichi[i],            'exact'),
    ('players.riichi_declared',  lambda bs,i: bs.players.riichi_declared[i],   'exact'),
    ('players.has_won',          lambda bs,i: bs.players.has_won[i],           'exact'),
    ('players.n_kan',            lambda bs,i: bs.players.n_kan[i],             'exact'),
    ('players.has_yaku',         lambda bs,i: bs.players.has_yaku[i],          'exact'),
    ('players.is_hand_concealed',lambda bs,i: bs.players.is_hand_concealed[i], 'exact'),
    ('players.furiten_by_discard',lambda bs,i: bs.players.furiten_by_discard[i],'exact'),
    ('players.furiten_by_pass',  lambda bs,i: bs.players.furiten_by_pass[i],   'exact'),
    ('players.ippatsu',          lambda bs,i: bs.players.ippatsu[i],           'exact'),
    ('players.fan',              lambda bs,i: bs.players.fan[i],               'exact'),
    ('players.fu',               lambda bs,i: bs.players.fu[i],                'exact'),
    # Round state fields — batch shape (B,) or (B, N)
    ('round_state.round',        lambda bs,i: bs.round_state.round[i],         'exact'),
    ('round_state.dealer',       lambda bs,i: bs.round_state.dealer[i],        'exact'),
    ('round_state.next_deck_ix', lambda bs,i: bs.round_state.next_deck_ix[i],  'exact'),
    ('round_state.last_deck_ix', lambda bs,i: bs.round_state.last_deck_ix[i],  'exact'),
    ('round_state.last_draw',    lambda bs,i: bs.round_state.last_draw[i],     'exact'),
    ('round_state.last_player',  lambda bs,i: bs.round_state.last_player[i],   'exact'),
    ('round_state.target',       lambda bs,i: bs.round_state.target[i],        'exact'),
    ('round_state.draw_next',    lambda bs,i: bs.round_state.draw_next[i],     'exact'),
    ('round_state.is_haitei',    lambda bs,i: bs.round_state.is_haitei[i],     'exact'),
    ('round_state.is_abortive_draw_normal', lambda bs,i: bs.round_state.is_abortive_draw_normal[i], 'exact'),
    ('round_state.terminated_round', lambda bs,i: bs.round_state.terminated_round[i], 'exact'),
    ('round_state.score',        lambda bs,i: bs.round_state.score[i],         'exact'),
    ('round_state.deck',         lambda bs,i: bs.round_state.deck[i],          'exact'),
    ('round_state.dora_indicators',lambda bs,i: bs.round_state.dora_indicators[i], 'exact'),
    ('round_state.n_kan_doras',  lambda bs,i: bs.round_state.n_kan_doras[i],   'exact'),
    ('round_state.order_points', lambda bs,i: bs.round_state.order_points[i],  'exact'),
]

# ── Batch GPU comparison fields ──
# Same fields as BATCH_CHECKS but with full-batch getters (lambda bs → (B,...) tensor).
# Used by _compare_step_gpu() to compare ALL active seeds against golden in one
# vectorized GPU operation per field, with a single CPU→GPU transfer per field.
BATCH_COMPARE_FIELDS = [
    # Top-level (B,) or (B, N)
    ('current_player',           lambda bs: bs.current_player,             'exact'),
    ('terminated',               lambda bs: bs.terminated,                 'exact'),
    ('legal_action_mask',        lambda bs: bs.legal_action_mask,         'exact'),
    ('rewards',                  lambda bs: bs.rewards,                   'close'),
    # Players (B, 4, ...)
    ('players.hand',             lambda bs: bs.players.hand,              'exact'),
    ('players.hand_with_red',    lambda bs: bs.players.hand_with_red,     'exact'),
    ('players.melds',            lambda bs: bs.players.melds,             'exact'),
    ('players.meld_counts',      lambda bs: bs.players.meld_counts,       'exact'),
    ('players.discard_counts',   lambda bs: bs.players.discard_counts,    'exact'),
    ('players.river',            lambda bs: bs.players.river,             'exact'),
    ('players.riichi',           lambda bs: bs.players.riichi,            'exact'),
    ('players.riichi_declared',  lambda bs: bs.players.riichi_declared,   'exact'),
    ('players.has_won',          lambda bs: bs.players.has_won,           'exact'),
    ('players.n_kan',            lambda bs: bs.players.n_kan,             'exact'),
    ('players.has_yaku',         lambda bs: bs.players.has_yaku,          'exact'),
    ('players.is_hand_concealed',lambda bs: bs.players.is_hand_concealed, 'exact'),
    ('players.furiten_by_discard',lambda bs: bs.players.furiten_by_discard,'exact'),
    ('players.furiten_by_pass',  lambda bs: bs.players.furiten_by_pass,   'exact'),
    ('players.ippatsu',          lambda bs: bs.players.ippatsu,           'exact'),
    ('players.fan',              lambda bs: bs.players.fan,               'exact'),
    ('players.fu',               lambda bs: bs.players.fu,                'exact'),
    # Round state (B,) or (B, N)
    ('round_state.round',        lambda bs: bs.round_state.round,         'exact'),
    ('round_state.dealer',       lambda bs: bs.round_state.dealer,        'exact'),
    ('round_state.next_deck_ix', lambda bs: bs.round_state.next_deck_ix,  'exact'),
    ('round_state.last_deck_ix', lambda bs: bs.round_state.last_deck_ix,  'exact'),
    ('round_state.last_draw',    lambda bs: bs.round_state.last_draw,     'exact'),
    ('round_state.last_player',  lambda bs: bs.round_state.last_player,   'exact'),
    ('round_state.target',       lambda bs: bs.round_state.target,        'exact'),
    ('round_state.draw_next',    lambda bs: bs.round_state.draw_next,     'exact'),
    ('round_state.is_haitei',    lambda bs: bs.round_state.is_haitei,     'exact'),
    ('round_state.is_abortive_draw_normal', lambda bs: bs.round_state.is_abortive_draw_normal, 'exact'),
    ('round_state.terminated_round', lambda bs: bs.round_state.terminated_round, 'exact'),
    ('round_state.score',        lambda bs: bs.round_state.score,         'exact'),
    ('round_state.deck',         lambda bs: bs.round_state.deck,          'exact'),
    ('round_state.dora_indicators',lambda bs: bs.round_state.dora_indicators, 'exact'),
    ('round_state.n_kan_doras',  lambda bs: bs.round_state.n_kan_doras,   'exact'),
    ('round_state.order_points', lambda bs: bs.round_state.order_points,  'exact'),
]

# Map torch dtypes to numpy dtypes for golden tensor construction
_TORCH_TO_NP_DTYPE = {
    torch.int8: np.int8, torch.int16: np.int16, torch.int32: np.int32,
    torch.int64: np.int64, torch.uint8: np.uint8,
    torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64,
    torch.bool: np.bool_,
}


def replay_seed(seed, init_state, records, verbose=False, device=None):
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
        # action-classification + batch-handler pipeline.
        try:
            result_list = penv.step_batch([state], [action], device=device)
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
                # Print error-type diffs directly, then format field-name diffs
                err_diffs = [d for d in diffs if d.endswith(':accessor_error') or 'EXCEPTION' in d]
                field_diffs = [d for d in diffs if d not in err_diffs]
                for name in err_diffs:
                    sys.stderr.write(f"  {name}\n")
                if field_diffs:
                    for line in _format_diffs(state, golden, field_diffs):
                        sys.stderr.write(line + '\n')
            break
        ok_steps += 1

    dt = time.time() - t0
    if first_fail is None:
        return (seed, True, {'steps': ok_steps, 'time': dt})
    else:
        s, a, f = first_fail
        return (seed, False, {'steps': ok_steps, 'time': dt, 'fail_step': s, 'fail_act': a, 'fail_fields': f})


# ── Batch GPU replay ──

def replay_seeds_batch(filepaths, device='cuda'):
    """Replay multiple seeds in one GPU batch. Returns list of (seed, ok, details)."""
    import time as _time
    t0 = _time.time()
    B = len(filepaths)

    # ── Timing accumulators ──
    t_load = 0.0
    t_init = 0.0
    t_step_total = 0.0
    t_compare_total = 0.0
    t_field_checks_total = 0.0

    # Load all golden data
    print(f"[perf] loading {B} golden files...", flush=True)
    _t = _time.time()
    all_data = []
    load_report_every = max(1, B // 10)
    for i, fp in enumerate(filepaths):
        with open(fp, 'rb') as f:
            all_data.append(pickle.load(f))
        if i > 0 and i % load_report_every == 0:
            print(f"[perf]   loaded {i}/{B} files ({_time.time() - _t:.1f}s)", flush=True)
    t_load = _time.time() - _t
    print(f"[perf] load {B} golden files: {t_load:.2f}s", flush=True)

    seeds = [d['seed'] for d in all_data]
    records_list = [d['records'] for d in all_data]
    max_steps = max(len(r) for r in records_list)
    total_steps = sum(len(r) for r in records_list)
    print(f"[perf] seeds={B} max_steps={max_steps} total_steps={total_steps}", flush=True)

    # Init batch on GPU
    _t = _time.time()
    penv = PtEnv(round_mode='single')
    penv._perf = {}  # enable handler-level profiling
    bs = penv.init_batch(num_envs=B, device=device)
    for i, data in enumerate(all_data):
        s = penv.init(key=0)
        _copy_golden_to_pt(data['init_state'], s)
        penv._copy_state_into_batch(bs, i, s)
    t_init = _time.time() - _t
    print(f"[perf] batch init ({B} envs on {device}): {t_init:.2f}s", flush=True)
    if device == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[perf] GPU memory allocated after init: {mem_mb:.1f} MB", flush=True)

    # Tracking
    active = torch.ones(B, dtype=torch.bool, device=device)  # still running
    results = [None] * B
    ok_counts = torch.zeros(B, dtype=torch.int32, device=device)

    # ── Progress reporting ──
    progress_interval = max(1, max_steps // 20)  # ~5% increments
    _last_report_t = _time.time()

    for step_k in range(max_steps):
        if not active.any():
            break

        # Build action tensor: default 0 for finished seeds
        actions_list = [0] * B
        for i in range(B):
            if not active[i].item() if isinstance(active[i], torch.Tensor) else not active[i]:
                continue
            if step_k < len(records_list[i]):
                actions_list[i] = records_list[i][step_k]['action']
            else:
                active[i] = False

        if not active.any():
            break

        actions_t = torch.tensor(actions_list, dtype=torch.int32, device=device)

        # KYUUSHU: inject JAX-generated decks
        kyuushu_mask = actions_t == 85
        if kyuushu_mask.any():
            overrides = {}
            for i in kyuushu_mask.nonzero(as_tuple=False).flatten().tolist():
                if step_k < len(records_list[i]):
                    golden_deck = records_list[i][step_k]['state']['round_state.deck']
                    overrides[i] = torch.from_numpy(golden_deck)
            penv._kyuushu_deck_overrides = overrides

        # ── Step timing (sync GPU first for accurate wall-clock) ──
        if device == 'cuda':
            torch.cuda.synchronize()
        _t_step = _time.time()
        try:
            bs = penv._step_batch_bs(bs, actions_t)
        except Exception as e:
            for i in range(B):
                if active[i] if isinstance(active[i], torch.Tensor) else active[i]:
                    results[i] = (seeds[i], False, {'steps': int(ok_counts[i].item()),
                                  'time': 0, 'fail_step': step_k, 'fail_act': actions_list[i],
                                  'fail_fields': [f'EXCEPTION: {e}']})
            active[:] = False
            break
        t_step_total += _time.time() - _t_step

        # ── Batch GPU compare: one transfer + one vectorized op per field ──
        # Build active comparison mask once (B,)
        if device == 'cuda':
            torch.cuda.synchronize()
        _t_comp = _time.time()
        active_mask = active.clone()
        for i in range(B):
            if step_k >= len(records_list[i]):
                active_mask[i] = False
        n_active = int(active_mask.sum().item())

        if n_active > 0:
            # Determine which golden fields are present (check first active seed)
            first_i = int(active_mask.nonzero(as_tuple=False)[0].item())
            golden_keys = set(records_list[first_i][step_k]['state'].keys())
            # Convert active_mask to Python list once (avoids per-seed GPU→CPU .item() calls)
            active_list = active_mask.cpu().tolist()

            for name, batch_getter, tol in BATCH_COMPARE_FIELDS:
                if name not in golden_keys:
                    continue

                pt_t = batch_getter(bs)  # (B, ...) or (B, 4, ...) on GPU
                shape = tuple(pt_t.shape)
                np_dtype = _TORCH_TO_NP_DTYPE.get(pt_t.dtype, np.float32)

                # Build golden numpy array for all B seeds (zeros for inactive)
                golden_np = np.zeros(shape, dtype=np_dtype)
                for i in range(B):
                    if active_list[i]:
                        golden_np[i] = np.asarray(
                            records_list[i][step_k]['state'][name])
                golden_t = torch.from_numpy(golden_np).to(device=device, dtype=pt_t.dtype)

                # Reshape active_mask for broadcasting: (B,) → (B, 1, ...)
                ndim = pt_t.ndim
                mask_view = active_mask.view(B, *([1] * (ndim - 1)))

                # Vectorized GPU comparison
                if tol == 'exact':
                    field_diff = (pt_t != golden_t) & mask_view
                else:
                    field_diff = ~torch.isclose(
                        pt_t.float(), golden_t.float(),
                        rtol=1e-3, atol=1e-5) & mask_view

                # Seeds with any diff on this field
                seed_diff = field_diff.reshape(B, -1).any(dim=1)  # (B,) bool

                # Record failures
                for i in seed_diff.nonzero(as_tuple=False).flatten().tolist():
                    if results[i] is None:
                        results[i] = (seeds[i], False, {
                            'steps': int(ok_counts[i].item()), 'time': 0,
                            'fail_step': step_k, 'fail_act': actions_list[i],
                            'fail_fields': [name],
                        })
                        active[i] = False

            # Batch update ok_counts for seeds that passed this step
            passed = active_mask & active  # active was set False for failures
            ok_counts += passed.to(torch.int32)

        t_field_checks_total += _time.time() - _t_comp
        t_compare_total += _time.time() - _t_comp

        # ── Progress report ──
        if step_k > 0 and (step_k % progress_interval == 0 or step_k == max_steps - 1):
            _now = _time.time()
            elapsed = _now - t0
            steps_done = int(ok_counts.sum().item())
            n_active_now = int(active.sum().item()) if active.is_cuda else int(active.sum())
            sps = steps_done / elapsed if elapsed > 0 else 0
            print(f"[perf] step={step_k}/{max_steps} "
                  f"active_seeds={n_active_now}/{B} "
                  f"steps_done={steps_done}/{total_steps} "
                  f"elapsed={elapsed:.1f}s sps={sps:.1f} "
                  f"step_t={t_step_total:.1f}s cmp_t={t_compare_total:.1f}s "
                  f"field_chk={t_field_checks_total:.1f}s",
                  flush=True)
            _last_report_t = _now

    # Wrap up any seeds that didn't finish
    dt = _time.time() - t0
    for i in range(B):
        if results[i] is None:
            results[i] = (seeds[i], True,
                         {'steps': int(ok_counts[i].item()), 'time': dt / B})

    # ── Final perf summary ──
    print(f"\n[perf] ===== SUMMARY =====", flush=True)
    print(f"[perf] total time: {dt:.1f}s", flush=True)
    print(f"[perf] load:     {t_load:8.1f}s ({t_load/dt*100:5.1f}%)", flush=True)
    print(f"[perf] init:     {t_init:8.1f}s ({t_init/dt*100:5.1f}%)", flush=True)
    print(f"[perf] step_batch: {t_step_total:8.1f}s ({t_step_total/dt*100:5.1f}%)", flush=True)
    print(f"[perf] compare:  {t_compare_total:8.1f}s ({t_compare_total/dt*100:5.1f}%)", flush=True)
    print(f"[perf]   field_chk: {t_field_checks_total:8.1f}s ({t_field_checks_total/dt*100:5.1f}%)", flush=True)
    total_steps_done = sum(r[2]['steps'] for r in results if r is not None)
    print(f"[perf] total replay steps: {total_steps_done}", flush=True)
    if device == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[perf] GPU peak memory: {mem_mb:.1f} MB", flush=True)

    # ── Handler-level breakdown ──
    perf_items = penv.get_perf_summary()
    if perf_items:
        total_perf_time = sum(t for _, t, _, _, _ in perf_items)
        print(f"\n[perf] ===== HANDLER BREAKDOWN ({total_perf_time:.1f}s total traced) =====", flush=True)
        print(f"[perf] {'handler':<35} {'time':>8} {'%':>6} {'calls':>7} {'active':>7} {'envs':>10}", flush=True)
        print(f"[perf] {'-'*35} {'-'*8} {'-'*6} {'-'*7} {'-'*7} {'-'*10}", flush=True)
        for name, t, calls, active_calls, total_envs in perf_items:
            pct = t / dt * 100
            print(f"[perf] {name:<35} {t:8.1f}s {pct:5.1f}% {calls:7d} {active_calls:7d} {total_envs:10d}", flush=True)

    return results


# ── Worker (module-level for multiprocessing) ──

def _worker_replay(args):
    """Replay a single golden file. Returns (seed, ok, details, verbose_lines)."""
    filepath, verbose, device = args
    t0 = time.time()
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        seed = data['seed']
        records = data['records']
        seed, ok, d = replay_seed(seed, data['init_state'], records, verbose=verbose, device=device)
        d['time'] = time.time() - t0
        return (seed, ok, d, [])
    except Exception:
        dt = time.time() - t0
        return (None, False,
                {'steps': 0, 'time': dt, 'fail_step': 0, 'fail_act': -1,
                 'fail_fields': [f'CRASH: {traceback.format_exc()}']}, [])


# ── Main ──

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--data-dir', default='golden_data', help='Golden data directory')
    p.add_argument('-s', '--seeds', nargs='*', type=int, help='Specific seeds')
    p.add_argument('-j', '--jobs', type=int, default=1, help='Parallel workers (default: 1 = serial)')
    p.add_argument('-v', '--verbose', action='store_true', help='Always print diff details')
    p.add_argument('--gpu', action='store_true', help='Run on CUDA GPU')
    p.add_argument('--list', action='store_true', help='List available seeds')
    args = p.parse_args()

    device = 'cuda' if args.gpu else None

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

    n_seeds = len(files)
    verbose = args.verbose or (n_seeds <= 3 and args.jobs <= 1)

    # Build work items
    work = [(os.path.join(args.data_dir, f), verbose, device) for f in files]

    # ── Execute ──
    n_workers = min(args.jobs, n_seeds) if args.jobs > 1 else 1

    if device == 'cuda' and n_workers <= 1:
        # GPU batch mode: run all seeds in one BatchState
        filepaths = [os.path.join(args.data_dir, f) for f in files]
        print(f"GPU batch replay: {n_seeds} seeds...", flush=True)
        results = replay_seeds_batch(filepaths, device=device)
    elif n_workers > 1:
        from multiprocessing import get_context
        print(f"Replaying {n_seeds} seeds with {n_workers} workers...", flush=True)
        with get_context('spawn').Pool(n_workers) as pool:
            results = pool.map(_worker_replay, work)
    else:
        if n_seeds > 1:
            print(f"Replaying {n_seeds} seeds (serial)...", flush=True)
        results = [_worker_replay(w) for w in work]

    # ── Report ──
    n_pass = 0
    for entry in results:
        if len(entry) == 4:
            seed, ok, d, _vl = entry
        else:
            seed, ok, d = entry
        if ok:
            print(f"seed={seed:4d}: OK ({d['steps']} steps, {d['time']:.1f}s)", flush=True)
            n_pass += 1
        else:
            print(f"seed={seed:4d}: FAIL step={d['fail_step']} act={d['fail_act']} fields={d['fail_fields']} ({d['time']:.1f}s)", flush=True)

    print(f"\nPassed: {n_pass}/{n_seeds} seeds", flush=True)
    if n_pass == n_seeds:
        print("ALL SEEDS PASS!", flush=True)
    else:
        for entry in results:
            seed, ok, d = entry[:3]
            if not ok:
                print(f"  seed={seed} FAILED at step {d['fail_step']}: {d['fail_fields']}", flush=True)
        sys.exit(1)
