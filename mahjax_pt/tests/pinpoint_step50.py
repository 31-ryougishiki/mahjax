#!/usr/bin/env python3
"""Find FIRST field that diverges between JAX and PT, with full context."""
import jax; jax.config.update('jax_disable_jit', True)
import numpy as np, torch
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv

def j2n(v):
    if hasattr(v, '__array__'): return np.array(v)
    if hasattr(v, 'item'): return np.array(v)
    return v

def copy_to_pt(js, ps):
    ps.current_player=int(js.current_player); ps.terminated=bool(js.terminated)
    ps.legal_action_mask=torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    ps.rewards=torch.from_numpy(np.array(js.rewards,dtype=np.float32).copy()).float()
    ps.step_count=int(js.step_count)
    jp,pp=js.players,ps.players
    for f,dt in [('hand',torch.int8),('hand_with_red',torch.int8),('meld_counts',torch.int8),
        ('melds',torch.int32),('riichi',torch.bool),('riichi_declared',torch.bool),
        ('furiten_by_discard',torch.bool),('furiten_by_pass',torch.bool),
        ('is_hand_concealed',torch.bool),('has_won',torch.bool),('n_kan',torch.int8),
        ('discard_counts',torch.int8),('river',torch.int32),('ippatsu',torch.bool),
        ('double_riichi',torch.bool),('has_yaku',torch.bool),('fan',torch.int32),
        ('fu',torch.int32),('can_win',torch.bool),('has_nagashi_mangan',torch.bool)]:
        setattr(pp,f,torch.from_numpy(np.array(getattr(jp,f)).copy()).to(dt))
    jr,pr=js.round_state,ps.round_state
    for f in ['round','honba','kyotaku','dealer','next_deck_ix','last_deck_ix',
        'last_draw','last_player','target','n_kan_doras','shanten_current_player',
        'dummy_count']:
        setattr(pr,f,int(getattr(jr,f)))
    for f in ['terminated_round','draw_next','is_haitei','kan_declared','can_after_kan',
        'can_robbing_kan','is_abortive_draw_normal']:
        setattr(pr,f,bool(getattr(jr,f)))
    for f,dt in [('deck',torch.int8),('score',torch.int32),('dora_indicators',torch.int8),
        ('seat_wind',torch.int8),('init_wind',torch.int8),('order_points',torch.int32),
        ('ura_dora_indicators',torch.int8)]:
        setattr(pr,f,torch.from_numpy(np.array(getattr(jr,f)).copy()).to(dt))
    ah=np.array(jr.action_history); pr.action_history=torch.from_numpy(ah.copy()).to(torch.int8)
    return ps

def compare_all(js, ps, label):
    """Return list of (field_name, jax_val, pt_val) that differ."""
    diffs = []

    def chk(name, jv, pv):
        if isinstance(pv, torch.Tensor): pv = pv.detach().cpu().numpy()
        jv, pv = np.asarray(jv), np.asarray(pv)
        if not np.array_equal(jv, pv):
            diffs.append((name, jv, pv))

    chk("current_player", int(js.current_player), ps.current_player)
    chk("terminated", bool(js.terminated), ps.terminated)

    jp, pp = js.players, ps.players
    chk("hand", np.array(jp.hand), pp.hand)
    chk("hand_with_red", np.array(jp.hand_with_red), pp.hand_with_red)
    chk("discard_counts", np.array(jp.discard_counts), pp.discard_counts)
    chk("meld_counts", np.array(jp.meld_counts), pp.meld_counts)
    chk("melds", np.array(jp.melds), pp.melds)
    chk("river", np.array(jp.river), pp.river)
    chk("riichi", np.array(jp.riichi), pp.riichi)
    chk("riichi_declared", np.array(jp.riichi_declared), pp.riichi_declared)
    chk("ippatsu", np.array(jp.ippatsu), pp.ippatsu)
    chk("furiten_by_discard", np.array(jp.furiten_by_discard), pp.furiten_by_discard)
    chk("furiten_by_pass", np.array(jp.furiten_by_pass), pp.furiten_by_pass)
    chk("is_hand_concealed", np.array(jp.is_hand_concealed), pp.is_hand_concealed)
    chk("has_won", np.array(jp.has_won), pp.has_won)
    chk("n_kan", np.array(jp.n_kan), pp.n_kan)
    chk("has_yaku", np.array(jp.has_yaku), pp.has_yaku)
    chk("fan", np.array(jp.fan), pp.fan)
    chk("fu", np.array(jp.fu), pp.fu)

    jr, pr = js.round_state, ps.round_state
    chk("round", int(jr.round), pr.round)
    chk("honba", int(jr.honba), pr.honba)
    chk("dealer", int(jr.dealer), pr.dealer)
    chk("next_deck_ix", int(jr.next_deck_ix), pr.next_deck_ix)
    chk("last_deck_ix", int(jr.last_deck_ix), pr.last_deck_ix)
    chk("last_draw", int(jr.last_draw), pr.last_draw)
    chk("last_player", int(jr.last_player), pr.last_player)
    chk("target", int(jr.target), pr.target)
    chk("draw_next", bool(jr.draw_next), pr.draw_next)
    chk("is_haitei", bool(jr.is_haitei), pr.is_haitei)
    chk("kan_declared", bool(jr.kan_declared), pr.kan_declared)
    chk("can_after_kan", bool(jr.can_after_kan), pr.can_after_kan)
    chk("is_abortive_draw_normal", bool(jr.is_abortive_draw_normal), pr.is_abortive_draw_normal)
    chk("terminated_round", bool(jr.terminated_round), pr.terminated_round)
    chk("score", np.array(jr.score), pr.score)
    chk("deck", np.array(jr.deck), pr.deck)
    chk("dora_indicators", np.array(jr.dora_indicators), pr.dora_indicators)
    chk("seat_wind", np.array(jr.seat_wind), pr.seat_wind)

    chk("legal_action_mask", np.array(js.legal_action_mask), ps.legal_action_mask)
    chk("rewards", np.array(js.rewards), ps.rewards)

    return diffs

# ═════════════════════════════════════════════════════════
print("=" * 70)
print("Finding FIRST divergence before step 50")
print("=" * 70)

jenv = JaxEnv(round_mode="single")
penv = PtEnv(round_mode="single")
js = jenv.init(jax.random.PRNGKey(42))
ps = penv.init(key=0)
ps = copy_to_pt(js, ps)

prev_diff_names = set()

for step in range(51):
    if bool(js.terminated) or bool(js.round_state.terminated_round):
        print(f"Game ended at step {step}")
        break

    # Compare BEFORE step
    diffs = compare_all(js, ps, f"before step {step}")
    diff_names = set(d[0] for d in diffs)

    new_diffs = diff_names - prev_diff_names
    if new_diffs:
        print(f"\n>>> BEFORE step {step}: NEW DIFF: {new_diffs}")
        for name, jv, pv in diffs:
            if name in new_diffs:
                n = int(np.sum(jv != pv))
                if jv.size < 30:
                    print(f"    {name}: JAX={jv} PT={pv}")
                else:
                    print(f"    {name}: {n} elems differ, JAX range [{jv.min()},{jv.max()}] PT range [{pv.min()},{pv.max()}]")
    elif step % 10 == 0:
        print(f"  step {step}: all {len(diff_names)} diffs same as before" if diff_names else f"  step {step}: CLEAN")

    prev_diff_names = diff_names

    # Select & execute action
    legal = np.where(np.array(js.legal_action_mask))[0]
    discards = [a for a in legal if a < 37]
    action = int(discards[step % len(discards)] if discards else legal[0])
    js = jenv.step(js, action)
    ps = penv.step(ps, action)

# Final report
diffs_after = compare_all(js, ps, "after final step")
if diffs_after:
    print(f"\nFinal diffs: {set(d[0] for d in diffs_after)}")
else:
    print(f"\nFULLY ALIGNED after {step+1} steps")
