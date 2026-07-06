#!/usr/bin/env python3
"""Debug: find root cause of river divergence at step 71."""
import jax; jax.config.update('jax_disable_jit', True)
import time, numpy as np, torch
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv

def j2n(v):
    if hasattr(v, '__array__'): return np.array(v)
    return v

def copy_to_pt(js, ps):
    for f, dt in [('hand',torch.int8),('hand_with_red',torch.int8),('meld_counts',torch.int8),
        ('melds',torch.int32),('riichi',torch.bool),('furiten_by_discard',torch.bool),
        ('furiten_by_pass',torch.bool),('is_hand_concealed',torch.bool),
        ('has_won',torch.bool),('n_kan',torch.int8),('discard_counts',torch.int8),
        ('river',torch.int32)]:
        setattr(ps.players, f, torch.from_numpy(np.array(getattr(js.players,f)).copy()).to(dt))
    for f in ['round','honba','kyotaku','dealer','next_deck_ix','last_deck_ix',
        'last_draw','last_player','target']:
        setattr(ps.round_state, f, int(getattr(js.round_state,f)))
    for f in ['terminated_round','draw_next','is_haitei','kan_declared','can_after_kan',
        'is_abortive_draw_normal']:
        setattr(ps.round_state, f, bool(getattr(js.round_state,f)))
    for f, dt in [('deck',torch.int8),('score',torch.int32),('dora_indicators',torch.int8),
        ('seat_wind',torch.int8)]:
        setattr(ps.round_state, f, torch.from_numpy(np.array(getattr(js.round_state,f)).copy()).to(dt))
    ps.current_player = int(js.current_player)
    ps.terminated = bool(js.terminated)
    ps.legal_action_mask = torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    return ps

print("Init & run 70 steps to reach divergence point...")
jax_env = JaxEnv(round_mode="single")
pt_env = PtEnv(round_mode="single")
js = jax_env.init(jax.random.PRNGKey(42))
ps = pt_env.init(key=0)
ps = copy_to_pt(js, ps)

for step in range(70):
    legal = np.where(np.array(js.legal_action_mask))[0]
    discards = [a for a in legal if a < 37]
    action = int(discards[step % len(discards)] if discards else legal[0])
    js = jax_env.step(js, action)
    ps = pt_env.step(ps, action)

# Check state before step 71
print("\n=== Before step 71 ===")
print(f"JAX cp={int(js.current_player)} terminated={bool(js.terminated)}")
print(f"PT  cp={ps.current_player} terminated={ps.terminated}")
print(f"JAX terminated_round={bool(js.round_state.terminated_round)}")
print(f"PT  terminated_round={ps.round_state.terminated_round}")
print(f"JAX draw_next={bool(js.round_state.draw_next)}")
print(f"PT  draw_next={ps.round_state.draw_next}")
print(f"JAX is_haitei={bool(js.round_state.is_haitei)}")
print(f"PT  is_haitei={ps.round_state.is_haitei}")
print(f"JAX is_abortive_draw_normal={bool(js.round_state.is_abortive_draw_normal)}")
print(f"PT  is_abortive_draw_normal={ps.round_state.is_abortive_draw_normal}")
print(f"JAX last_player={int(js.round_state.last_player)} target={int(js.round_state.target)}")
print(f"PT  last_player={ps.round_state.last_player} target={ps.round_state.target}")
print(f"JAX next_deck_ix={int(js.round_state.next_deck_ix)} last_deck_ix={int(js.round_state.last_deck_ix)}")
print(f"PT  next_deck_ix={ps.round_state.next_deck_ix} last_deck_ix={ps.round_state.last_deck_ix}")
print(f"JAX discard_counts={np.array(js.players.discard_counts)}")
print(f"PT  discard_counts={ps.players.discard_counts.numpy()}")
print(f"JAX n_kan={np.array(js.players.n_kan)}")
print(f"PT  n_kan={ps.players.n_kan.numpy()}")
print(f"JAX has_won={np.array(js.players.has_won)}")
print(f"PT  has_won={ps.players.has_won.numpy()}")

# Compare rivers
jr = np.array(js.players.river)
pr = ps.players.river.numpy()
diff = (jr != pr)
if diff.any():
    print(f"\nRIVER DIFF before step 71: {diff.sum()} elements")
    dp = np.where(diff)
    for pi, si in zip(dp[0][:5], dp[1][:5]):
        print(f"  player{pi} slot{si}: JAX={jr[pi,si]} PT={pr[pi,si]}")
else:
    print("\nRiver: MATCH")

# Now do step 71
print(f"\n=== Step 71 ===")
legal = np.where(np.array(js.legal_action_mask))[0]
discards = [a for a in legal if a < 37]
action = int(discards[70 % len(discards)] if discards else legal[0])
print(f"Action: {action} (legal={len(legal)})")

# Check what the action type is
from mahjax_pt.red_mahjong.action import Action
if action < 37: atype = 'discard'
elif action == 71: atype = 'tsumogiri'
elif Action.is_selfkan(action): atype = 'selfkan'
elif action == 72: atype = 'riichi'
elif action == 73: atype = 'ron'
elif action == 74: atype = 'tsumo'
elif action in (75,76): atype = 'pon'
elif action == 77: atype = 'open_kan'
elif 78 <= action <= 83: atype = 'chi'
elif action == 84: atype = 'pass'
else: atype = f'other({action})'
print(f"Action type: {atype}")

js2 = jax_env.step(js, action)
ps2 = pt_env.step(ps, action)

# Compare rivers after step 71
jr2 = np.array(js2.players.river)
pr2 = ps2.players.river.numpy()
diff2 = (jr2 != pr2)
if diff2.any():
    print(f"\nRIVER DIFF after step 71: {diff2.sum()} elements")
    dp = np.where(diff2)
    for pi, si in zip(dp[0][:10], dp[1][:10]):
        print(f"  player{pi} slot{si}: JAX={jr2[pi,si]} PT={pr2[pi,si]}")

    # Show what changed vs before
    changed_jax = (jr2 != jr)
    changed_pt = (pr2 != pr)
    print(f"\nChanged in JAX: {changed_jax.sum()} slots")
    print(f"Changed in PT: {changed_pt.sum()} slots")

    cj = np.where(changed_jax)
    cp = np.where(changed_pt)
    print("JAX changes:")
    for pi, si in zip(cj[0][:5], cj[1][:5]):
        print(f"  player{pi} slot{si}: {jr[pi,si]} -> {jr2[pi,si]}")
    print("PT changes:")
    for pi, si in zip(cp[0][:5], cp[1][:5]):
        print(f"  player{pi} slot{si}: {pr[pi,si]} -> {pr2[pi,si]}")
else:
    print("\nRiver: MATCH after step 71")

# Compare key fields
print(f"\nJAX terminated_round={bool(js2.round_state.terminated_round)}")
print(f"PT  terminated_round={ps2.round_state.terminated_round}")
print(f"JAX draw_next={bool(js2.round_state.draw_next)}")
print(f"PT  draw_next={ps2.round_state.draw_next}")
print(f"JAX is_haitei={bool(js2.round_state.is_haitei)}")
print(f"PT  is_haitei={ps2.round_state.is_haitei}")
print(f"JAX is_abortive={bool(js2.round_state.is_abortive_draw_normal)}")
print(f"PT  is_abortive={ps2.round_state.is_abortive_draw_normal}")
