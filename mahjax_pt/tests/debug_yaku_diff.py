#!/usr/bin/env python3
"""Directly compare JAX Yaku.judge vs PT Yaku.judge on identical inputs."""
import jax; jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp
import numpy as np, torch
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv

def copy_to_pt(js, ps):
    ps.current_player=int(js.current_player); ps.terminated=bool(js.terminated)
    ps.legal_action_mask=torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    ps.rewards=torch.from_numpy(np.array(js.rewards,dtype=np.float32).copy()).float()
    jp,pp=js.players,ps.players
    for f,dt in [('hand',torch.int8),('hand_with_red',torch.int8),('meld_counts',torch.int8),
        ('melds',torch.int32),('riichi',torch.bool),('riichi_declared',torch.bool),
        ('furiten_by_discard',torch.bool),('furiten_by_pass',torch.bool),
        ('is_hand_concealed',torch.bool),('has_won',torch.bool),('n_kan',torch.int8),
        ('discard_counts',torch.int8),('river',torch.int32),('ippatsu',torch.bool),
        ('double_riichi',torch.bool),('has_yaku',torch.bool),('fan',torch.int32),
        ('fu',torch.int32),('can_win',torch.bool)]:
        setattr(pp,f,torch.from_numpy(np.array(getattr(jp,f)).copy()).to(dt))
    jr,pr=js.round_state,ps.round_state
    for f in ['round','honba','kyotaku','dealer','next_deck_ix','last_deck_ix',
        'last_draw','last_player','target','n_kan_doras']:
        setattr(pr,f,int(getattr(jr,f)))
    for f in ['terminated_round','draw_next','is_haitei','kan_declared','can_after_kan',
        'is_abortive_draw_normal']:
        setattr(pr,f,bool(getattr(jr,f)))
    for f,dt in [('deck',torch.int8),('score',torch.int32),('dora_indicators',torch.int8),
        ('seat_wind',torch.int8),('init_wind',torch.int8),('order_points',torch.int32),
        ('ura_dora_indicators',torch.int8)]:
        setattr(pr,f,torch.from_numpy(np.array(getattr(jr,f)).copy()).to(dt))
    return ps

def short_hand(h37):
    suits = ['m','p','s','z']; parts = []
    for t in range(34):
        c = int(h37[t])
        if c > 0:
            n = (t % 9) + 1; s = suits[t // 9]
            parts.append(f"{n}{s}" * c)
    for t in [34, 35, 36]:
        c = int(h37[t])
        if c > 0: parts.append(f"0{suits[t-34]}" * c)
    return ''.join(parts)

# ── Init + step 0 to get state ──
import sys
def log(msg): print(msg, flush=True)

log("init...")
jenv = JaxEnv(round_mode="single")
penv = PtEnv(round_mode="single")
js = jenv.init(jax.random.PRNGKey(42))
ps = penv.init(key=0)
ps = copy_to_pt(js, ps)

# Step 0
legal = np.where(np.array(js.legal_action_mask))[0]
discarded = int([x for x in legal if x < 37][0])
js = jenv.step(js, discarded)
ps = penv.step(ps, discarded)

# ── Now directly compare Yaku.judge for a specific test case ──
# Use player 0's hand + the discarded tile for RON
# And player 0's hand + next deck tile for TSUMO

p = 0  # Focus on player 0
jh37 = np.array(js.players.hand_with_red[p])
ph37 = ps.players.hand_with_red[p].numpy()

disc_tile = discarded           # tile 4 = 5m
next_tile = int(js.round_state.deck[int(js.round_state.next_deck_ix)])  # tile 16

log(f"\n=== Test: Player {p} ===")
log(f"Hand (13 tiles): {short_hand(jh37)}")
log(f"Discarded tile: {disc_tile}")
log(f"Next deck tile: {next_tile}")
log(f"State: round={int(js.round_state.round)} dealer={int(js.round_state.dealer)}")
log(f"  seat_wind={np.array(js.round_state.seat_wind)}")
log(f"  dora_indicators={np.array(js.round_state.dora_indicators)}")

# ── RON test: hand + discarded tile ──
log(f"\n--- RON: hand + discarded tile ({disc_tile}) ---")
from mahjax_pt.red_mahjong.hand import Hand as PtHand
from mahjax.red_mahjong.hand import Hand as JaxHand
from mahjax_pt.red_mahjong.yaku import Yaku as PtYaku
from mahjax.red_mahjong.yaku import Yaku as JaxYaku

# PT RON — set target like JAX's ron_state
pt_hand_ron = PtHand.add(torch.from_numpy(ph37.copy()), disc_tile)
log(f"PT hand for RON ({int(pt_hand_ron.sum().item())} tiles): {short_hand(pt_hand_ron.numpy())}")
ps.round_state.target = disc_tile  # ← match JAX ron_state

# JAX RON — convert to jnp array first, then add tile
jax_hand_ron = JaxHand.add(jnp.array(jh37), disc_tile)
log(f"JAX hand for RON ({int(jnp.sum(jax_hand_ron))} tiles): {short_hand(np.array(jax_hand_ron))}")

# JAX state target was reset to -1 after step 0's draw — set to discard like ron_state
js_ron = js.replace(round_state=js.round_state.replace(target=jnp.int8(disc_tile)))
log(f"  JAX state target (set to disc_tile): {int(js_ron.round_state.target)}")
log(f"  PT  state target (set to disc_tile): {ps.round_state.target}")

log("Calling JAX Yaku.judge for RON...")
jax_fan_ron, jax_fu_ron = None, None
try:
    jax_yaku_ron, jax_fan_ron, jax_fu_ron = JaxYaku.judge(
        jax_hand_ron, jnp.bool_(True), p, js_ron)
    log(f"  JAX: fan={jax_fan_ron} fu={jax_fu_ron}")
    log(f"  JAX yaku indices: {[i for i,v in enumerate(np.array(jax_yaku_ron)) if v]}")
except Exception as e:
    log(f"  JAX ERROR: {e}")

log("Calling PT Yaku.judge for RON...")
pt_yaku_ron, pt_fan_ron, pt_fu_ron = PtYaku.judge(pt_hand_ron, True, p, ps)
log(f"  PT:  fan={pt_fan_ron} fu={pt_fu_ron}")
pt_ron_idx = [i for i,v in enumerate(pt_yaku_ron) if v]
log(f"  PT  yaku indices: {pt_ron_idx}")

# ── TSUMO test: hand + next deck tile ──
log(f"\n--- TSUMO: hand + next deck tile ({next_tile}) ---")
pt_hand_tsumo = PtHand.add(torch.from_numpy(ph37.copy()), next_tile)
jax_hand_tsumo = JaxHand.add(jnp.array(jh37), next_tile)
log(f"PT hand for TSUMO ({int(pt_hand_tsumo.sum().item())} tiles): {short_hand(pt_hand_tsumo.numpy())}")
log(f"JAX hand for TSUMO ({int(jnp.sum(jax_hand_tsumo))} tiles): {short_hand(np.array(jax_hand_tsumo))}")

# For TSUMO, set last_draw to next_tile (match JAX tsumo_state)
ps.round_state.last_draw = next_tile
log(f"  PT  state last_draw (set to next_tile): {ps.round_state.last_draw}")

# For JAX: the state needs last_draw set to next_tile too
# (JAX creates tsumo_state with last_draw=next_tile)
js_tsumo = js.replace(round_state=js.round_state.replace(last_draw=jnp.int8(next_tile)))
log(f"  JAX state last_draw (set to next_tile): {int(js_tsumo.round_state.last_draw)}")

log("Calling JAX Yaku.judge for TSUMO...")
jax_fan_tsumo, jax_fu_tsumo = None, None
try:
    jax_yaku_tsumo, jax_fan_tsumo, jax_fu_tsumo = JaxYaku.judge(
        jax_hand_tsumo, jnp.bool_(False), p, js_tsumo)
    log(f"  JAX: fan={jax_fan_tsumo} fu={jax_fu_tsumo}")
    log(f"  JAX yaku indices: {[i for i,v in enumerate(np.array(jax_yaku_tsumo)) if v]}")
except Exception as e:
    log(f"  JAX ERROR: {e}")

log("Calling PT Yaku.judge for TSUMO...")
pt_yaku_tsumo, pt_fan_tsumo, pt_fu_tsumo = PtYaku.judge(pt_hand_tsumo, False, p, ps)
log(f"  PT:  fan={pt_fan_tsumo} fu={pt_fu_tsumo}")
pt_tsumo_idx = [i for i,v in enumerate(pt_yaku_tsumo) if v]
log(f"  PT  yaku indices: {pt_tsumo_idx}")

# ── Summary ──
log(f"\n=== SUMMARY ===")
log(f"RON:  JAX fan={jax_fan_ron}  PT fan={pt_fan_ron}")
log(f"TSUMO: JAX fan={jax_fan_tsumo}  PT fan={pt_fan_tsumo}")
