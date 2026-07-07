#!/usr/bin/env python3
"""Multi-seed JAX vs PT alignment test."""
import jax; jax.config.update('jax_disable_jit', True)
import numpy as np, torch, sys, time
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv

def copy_to_pt(js, ps):
    ps.current_player = int(js.current_player)
    ps.terminated = bool(js.terminated)
    ps.legal_action_mask = torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    jp, pp = js.players, ps.players
    for f, dt in [('hand', torch.int8), ('hand_with_red', torch.int8), ('melds', torch.int32),
                  ('riichi', torch.bool), ('furiten_by_discard', torch.bool),
                  ('is_hand_concealed', torch.bool), ('has_won', torch.bool),
                  ('n_kan', torch.int8), ('discard_counts', torch.int8),
                  ('river', torch.int32), ('has_yaku', torch.bool)]:
        setattr(pp, f, torch.from_numpy(np.array(getattr(jp, f)).copy()).to(dt))
    jr, pr = js.round_state, ps.round_state
    for f in ['round', 'honba', 'kyotaku', 'dealer', 'next_deck_ix', 'last_deck_ix',
              'last_draw', 'last_player', 'target']:
        setattr(pr, f, int(getattr(jr, f)))
    for f in ['terminated_round', 'draw_next', 'is_haitei', 'kan_declared', 'can_after_kan',
              'is_abortive_draw_normal']:
        setattr(pr, f, bool(getattr(jr, f)))
    for f, dt in [('deck', torch.int8), ('score', torch.int32), ('dora_indicators', torch.int8),
                  ('seat_wind', torch.int8)]:
        setattr(pr, f, torch.from_numpy(np.array(getattr(jr, f)).copy()).to(dt))
    return ps


def compare(js, ps):
    jh = np.array(js.players.hand)
    ph = ps.players.hand.numpy()
    if not np.array_equal(jh, ph): return 'hand'
    jd = np.array(js.round_state.deck)
    pd = ps.round_state.deck.numpy()
    if not np.array_equal(jd, pd): return 'deck'
    jr = np.array(js.players.river)
    pr = ps.players.river.numpy()
    if not np.array_equal(jr, pr): return 'river'
    jm = np.array(js.players.melds)
    pm = ps.players.melds.numpy()
    if not np.array_equal(jm, pm): return 'melds'
    js_ = np.array(js.round_state.score)
    ps_ = ps.round_state.score.numpy()
    if not np.array_equal(js_, ps_): return 'score'
    if int(js.current_player) != ps.current_player: return 'cp'
    if bool(js.round_state.terminated_round) != ps.round_state.terminated_round: return 'term_round'
    if bool(js.terminated) != ps.terminated: return 'terminated'
    if bool(js.round_state.draw_next) != ps.round_state.draw_next: return 'draw_next'
    return None


SEEDS = [1, 7, 13, 42, 99, 123, 256, 512, 1024, 2048]
print(f"Testing {len(SEEDS)} seeds...", flush=True)

results = []
for seed in SEEDS:
    sys.stdout.write(f"seed={seed:4d}: ", ); sys.stdout.flush()
    t0 = time.time()
    jenv = JaxEnv(round_mode='single')
    penv = PtEnv(round_mode='single')
    js = jenv.init(jax.random.PRNGKey(seed))
    ps = penv.init(key=0)
    ps = copy_to_pt(js, ps)

    ok = fail = 0
    first_fail = None
    for step in range(200):
        if bool(js.terminated) or bool(js.round_state.terminated_round):
            break
        legal = np.where(np.array(js.legal_action_mask))[0]
        discards = [a for a in legal if a < 37]
        a = int(discards[step % len(discards)] if discards else legal[0])
        js = jenv.step(js, a)
        ps = penv.step(ps, a)
        err = compare(js, ps)
        if err:
            fail += 1
            first_fail = (step, a, err)
            break
        ok += 1

    if fail == 0:
        err = compare(js, ps)
        if err:
            fail += 1
            first_fail = ('END', '-', err)

    dt = time.time() - t0
    if fail == 0:
        print(f"OK {ok} steps ({dt:.0f}s)", flush=True)
    else:
        print(f"FAIL step={first_fail[0]} act={first_fail[1]} field={first_fail[2]} ({dt:.0f}s)", flush=True)
    results.append((seed, fail))

n_pass = sum(1 for _, f in results if f == 0)
print(f"\nPassed: {n_pass}/{len(SEEDS)} seeds", flush=True)
if n_pass == len(SEEDS):
    print("ALL SEEDS PASS!", flush=True)
else:
    for seed, fail in results:
        if fail > 0:
            print(f"  seed={seed} FAILED", flush=True)
    sys.exit(1)
