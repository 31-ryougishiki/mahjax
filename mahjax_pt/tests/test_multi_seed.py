#!/usr/bin/env python3
"""Multi-seed JAX vs PT alignment test with detailed failure logging."""
import jax; jax.config.update('jax_disable_jit', True)
import numpy as np, torch, sys, time
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv


def copy_to_pt(js, ps):
    ps.current_player = int(js.current_player)
    ps.terminated = bool(js.terminated)
    ps.legal_action_mask = torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    ps.rewards = torch.from_numpy(np.array(js.rewards, dtype=np.float32).copy()).float()
    jp, pp = js.players, ps.players
    for f, dt in [('hand', torch.int8), ('hand_with_red', torch.int8), ('melds', torch.int32),
                  ('meld_counts', torch.int8),
                  ('riichi', torch.bool), ('riichi_declared', torch.bool),
                  ('furiten_by_discard', torch.bool), ('furiten_by_pass', torch.bool),
                  ('is_hand_concealed', torch.bool), ('has_won', torch.bool),
                  ('n_kan', torch.int8), ('discard_counts', torch.int8),
                  ('river', torch.int32), ('has_yaku', torch.bool), ('ippatsu', torch.bool)]:
        setattr(pp, f, torch.from_numpy(np.array(getattr(jp, f)).copy()).to(dt))
    jr, pr = js.round_state, ps.round_state
    for f in ['round', 'honba', 'kyotaku', 'dealer', 'next_deck_ix', 'last_deck_ix',
              'last_draw', 'last_player', 'target']:
        setattr(pr, f, int(getattr(jr, f)))
    for f in ['terminated_round', 'draw_next', 'is_haitei', 'kan_declared', 'can_after_kan',
              'is_abortive_draw_normal']:
        setattr(pr, f, bool(getattr(jr, f)))
    for f, dt in [('deck', torch.int8), ('score', torch.int32), ('dora_indicators', torch.int8),
                  ('seat_wind', torch.int8), ('order_points', torch.int32)]:
        setattr(pr, f, torch.from_numpy(np.array(getattr(jr, f)).copy()).to(dt))
    return ps


def jv(val):
    """Convert to numpy for comparison."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy()
    if hasattr(val, '__array__'):
        return np.array(val)
    return np.asarray(val)


# All fields to compare, with readable names
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
    """Return list of field names that differ."""
    diffs = []
    for name, fn in CHECKS:
        jv_val = jv(fn(js))
        pv_val = jv(fn(ps))
        if not np.array_equal(jv_val, pv_val):
            diffs.append(name)
    return diffs


def describe_diff(name, jv_val, pv_val):
    """Debug description of a single field difference."""
    jv_val = np.asarray(jv_val)
    pv_val = np.asarray(pv_val)
    if jv_val.size < 20:
        return f"JAX={jv_val.tolist()} PT={pv_val.tolist()}"
    else:
        n = int(np.sum(jv_val != pv_val))
        return f"{n} elems differ, JAX range [{jv_val.min()},{jv_val.max()}] PT range [{pv_val.min()},{pv_val.max()}]"


if len(sys.argv) > 1:
    SEEDS = [int(s) for s in sys.argv[1:]]
else:
    SEEDS = [1, 7, 13, 42, 99, 123, 256, 512, 1024, 2048]
print(f"Testing {len(SEEDS)} seeds: {SEEDS}", flush=True)

results = []
for seed in SEEDS:
    sys.stdout.write(f"seed={seed:4d}: ")
    sys.stdout.flush()
    t0 = time.time()

    jenv = JaxEnv(round_mode='single')
    penv = PtEnv(round_mode='single')
    js = jenv.init(jax.random.PRNGKey(seed))
    ps = penv.init(key=0)
    ps = copy_to_pt(js, ps)

    ok = 0
    fail_step = fail_act = fail_fields = None
    for step in range(200):
        if bool(js.terminated) or bool(js.round_state.terminated_round):
            break
        legal = np.where(np.array(js.legal_action_mask))[0]
        discards = [a for a in legal if a < 37]
        a = int(discards[step % len(discards)] if discards else legal[0])
        js = jenv.step(js, a)
        ps = penv.step(ps, a)
        diffs = compare_all(js, ps)
        if diffs:
            fail_step = step
            fail_act = a
            fail_fields = diffs
            break
        ok += 1

    # Also check final state
    if fail_step is None:
        diffs = compare_all(js, ps)
        if diffs:
            fail_step = 'END'
            fail_act = '-'
            fail_fields = diffs

    dt = time.time() - t0
    if fail_step is None:
        print(f"OK ({ok} steps, {dt:.0f}s)", flush=True)
        results.append((seed, True, None))
    else:
        print(f"FAIL step={fail_step} act={fail_act} fields={fail_fields} ({dt:.0f}s)", flush=True)
        # Detailed dump for each failing field
        for name in fail_fields:
            fn = dict(CHECKS)[name]
            jv, pv = jv(fn(js)), jv(fn(ps))
            print(f"  {name}: {describe_diff(name, jv, pv)}", flush=True)
        results.append((seed, False, fail_fields))

n_pass = sum(1 for _, ok, _ in results if ok)
print(f"\nPassed: {n_pass}/{len(SEEDS)} seeds", flush=True)
if n_pass == len(SEEDS):
    print("ALL SEEDS PASS!", flush=True)
else:
    for seed, ok, fields in results:
        if not ok:
            print(f"  seed={seed} FAILED: {fields}", flush=True)
    sys.exit(1)
