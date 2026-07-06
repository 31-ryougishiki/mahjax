#!/usr/bin/env python3
"""Find exact step where last_player diverges between JAX and PT."""
import jax; jax.config.update('jax_disable_jit', True)
import numpy as np, torch
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv

def j2n(v):
    if hasattr(v, '__array__'): return np.array(v)
    return v

def copy_to_pt(js, ps):
    ps.current_player = int(js.current_player)
    ps.terminated = bool(js.terminated)
    ps.legal_action_mask = torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    jp, pp = js.players, ps.players
    for f, dt in [('hand',torch.int8),('hand_with_red',torch.int8),('meld_counts',torch.int8),
        ('melds',torch.int32),('riichi',torch.bool),('furiten_by_discard',torch.bool),
        ('furiten_by_pass',torch.bool),('is_hand_concealed',torch.bool),
        ('has_won',torch.bool),('n_kan',torch.int8),('discard_counts',torch.int8),
        ('river',torch.int32),('ippatsu',torch.bool)]:
        setattr(pp, f, torch.from_numpy(np.array(getattr(jp,f)).copy()).to(dt))
    jr, pr = js.round_state, ps.round_state
    for f in ['round','honba','kyotaku','dealer','next_deck_ix','last_deck_ix',
        'last_draw','last_player','target','n_kan_doras']:
        setattr(pr, f, int(getattr(jr,f)))
    for f in ['terminated_round','draw_next','is_haitei','kan_declared','can_after_kan',
        'is_abortive_draw_normal']:
        setattr(pr, f, bool(getattr(jr,f)))
    for f, dt in [('deck',torch.int8),('score',torch.int32),('dora_indicators',torch.int8),
        ('seat_wind',torch.int8)]:
        setattr(pr, f, torch.from_numpy(np.array(getattr(jr,f)).copy()).to(dt))
    return ps

def compare_step(name, js, ps):
    """Compare all fields, return list of mismatches."""
    fails = []

    if int(js.current_player) != ps.current_player:
        fails.append(f"current_player: J={int(js.current_player)} P={ps.current_player}")
    if bool(js.terminated) != ps.terminated:
        fails.append(f"terminated")
    if bool(js.round_state.terminated_round) != ps.round_state.terminated_round:
        fails.append(f"terminated_round")
    if bool(js.round_state.draw_next) != ps.round_state.draw_next:
        fails.append(f"draw_next: J={bool(js.round_state.draw_next)} P={ps.round_state.draw_next}")
    if int(js.round_state.last_player) != ps.round_state.last_player:
        fails.append(f"last_player: J={int(js.round_state.last_player)} P={ps.round_state.last_player}")
    if int(js.round_state.target) != ps.round_state.target:
        fails.append(f"target: J={int(js.round_state.target)} P={ps.round_state.target}")
    if not np.array_equal(np.array(js.players.hand), ps.players.hand.numpy()):
        fails.append("hand")
    if not np.array_equal(np.array(js.players.river), ps.players.river.numpy()):
        fails.append("river")
    if not np.array_equal(np.array(js.players.melds), ps.players.melds.numpy()):
        fails.append("melds")
    if not np.array_equal(np.array(js.players.discard_counts), ps.players.discard_counts.numpy()):
        fails.append("discard_counts")
    if not np.array_equal(np.array(js.round_state.score), ps.round_state.score.numpy()):
        fails.append("score")

    return fails

def action_name(a):
    from mahjax_pt.red_mahjong.action import Action
    if a < 37: return f"discard({a})"
    if a == 71: return "tsumogiri"
    if Action.is_selfkan(a): return f"selfkan({a-37})"
    return {72:'riichi',73:'RON',74:'TSUMO',75:'PON',76:'PON_RED',
            77:'OPEN_KAN',84:'PASS',85:'KYUUSHU',86:'DUMMY'}.get(a, f'CHI({a})')

print("=" * 60)
print("Tracking last_player divergence step by step")
print("=" * 60)

jax_env = JaxEnv(round_mode="single")
pt_env = PtEnv(round_mode="single")
js = jax_env.init(jax.random.PRNGKey(42))
ps = pt_env.init(key=0)
ps = copy_to_pt(js, ps)

print("Step   Action         JAX-lp PT-lp   JAX-dn PT-dn   JAX-target PT-target   Note")
print("-" * 85)

last_ok_step = -1
for step in range(100):
    if bool(js.terminated) or bool(js.round_state.terminated_round):
        print(f"  Game ended at step {step}")
        break

    legal = np.where(np.array(js.legal_action_mask))[0]
    if len(legal) == 0:
        break

    discards = [a for a in legal if a < 37]
    action = int(discards[step % len(discards)] if discards else legal[0])

    # Record state BEFORE step
    j_lp_before = int(js.round_state.last_player)
    p_lp_before = ps.round_state.last_player
    j_target_before = int(js.round_state.target)
    p_target_before = ps.round_state.target
    j_dn_before = bool(js.round_state.draw_next)
    p_dn_before = ps.round_state.draw_next

    # Step
    js = jax_env.step(js, action)
    ps = pt_env.step(ps, action)

    # Compare AFTER step
    fails = compare_step(f"step{step}", js, ps)

    j_lp = int(js.round_state.last_player)
    p_lp = ps.round_state.last_player
    j_dn = bool(js.round_state.draw_next)
    p_dn = ps.round_state.draw_next
    j_target = int(js.round_state.target)
    p_target = ps.round_state.target

    # Print detail when something changes
    lp_ok = "OK" if j_lp == p_lp else "DIFF!"
    dn_ok = "OK" if j_dn == p_dn else "DIFF!"
    target_ok = "OK" if j_target == p_target else "DIFF!"

    if fails:
        print(f"{step:4d}  {action_name(action):<14s} J:{j_lp} P:{p_lp}  {lp_ok:<5s}  J:{j_dn} P:{p_dn}  {dn_ok:<5s}  J:{j_target:3d} P:{p_target:3d}  {target_ok:<5s}  *** FAIL: {fails}")
        # Detailed dump on first failure
        if last_ok_step >= 0:
            print(f"\n  === First failure after step {last_ok_step} ===")
            print(f"  State BEFORE step {step}:")
            print(f"    JAX: last_player={j_lp_before} target={j_target_before} draw_next={j_dn_before}")
            print(f"    PT:  last_player={p_lp_before} target={p_target_before} draw_next={p_dn_before}")
            print(f"    JAX legal_actions: {legal}")
            print(f"    Action taken: {action_name(action)}")
            print(f"  State AFTER step {step}:")
            # Dump detailed state
            print(f"    JAX cp={int(js.current_player)} last_p={int(js.round_state.last_player)} target={int(js.round_state.target)}")
            print(f"    PT  cp={ps.current_player} last_p={ps.round_state.last_player} target={ps.round_state.target}")
            print(f"    JAX draw_next={bool(js.round_state.draw_next)}")
            print(f"    PT  draw_next={ps.round_state.draw_next}")
            print(f"    JAX discard_counts={np.array(js.players.discard_counts)}")
            print(f"    PT  discard_counts={ps.players.discard_counts.numpy()}")
            print(f"    JAX is_haitei={bool(js.round_state.is_haitei)}")
            print(f"    PT  is_haitei={ps.round_state.is_haitei}")
            print(f"    JAX is_abortive={bool(js.round_state.is_abortive_draw_normal)}")
            print(f"    PT  is_abortive={ps.round_state.is_abortive_draw_normal}")

        # Show legal action masks of both
        j_mask = np.array(js.legal_action_mask)
        p_mask = ps.legal_action_mask.numpy()
        mask_diff = np.where(j_mask != p_mask)[0]
        if len(mask_diff) > 0:
            print(f"    Legal mask diffs ({len(mask_diff)} positions):")
            for a in mask_diff[:10]:
                print(f"      action {a}: JAX={j_mask[a]} PT={p_mask[a]}")
        last_ok_step = step
        # Continue tracking but don't break — let's see how it compounds
    else:
        lp_ok_str = f"J:{j_lp} P:{p_lp}  {lp_ok}"
        print(f"{step:4d}  {action_name(action):<14s} {lp_ok_str:<16s} J:{j_dn} P:{p_dn}  {dn_ok:<5s}  J:{j_target:3d} P:{p_target:3d}  {target_ok}")
        last_ok_step = step

print(f"\nLast OK step: {last_ok_step}")
