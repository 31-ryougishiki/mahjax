#!/usr/bin/env python3
"""Find FIRST field divergence + dump detailed context for offline debugging."""
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

def short_hand(h37):
    """Convert 37-type hand to readable string like '1m2m3m...'"""
    suits = ['m','p','s','z']
    parts = []
    for t in range(34):
        c = int(h37[t])
        if c > 0:
            n = (t % 9) + 1
            s = suits[t // 9]
            parts.append(f"{n}{s}" * c)
    # Red fives
    for t in [34, 35, 36]:
        c = int(h37[t])
        if c > 0:
            parts.append(f"0{suits[t-34]}" * c)
    return ''.join(parts)

# ═════════════════════════════════════════════════════════
jenv = JaxEnv(round_mode="single")
penv = PtEnv(round_mode="single")
js = jenv.init(jax.random.PRNGKey(42))
ps = penv.init(key=0)
ps = copy_to_pt(js, ps)

# Step 0
legal = np.where(np.array(js.legal_action_mask))[0]
a = int([x for x in legal if x < 37][0])
js = jenv.step(js, a)
ps = penv.step(ps, a)

# ── After step 0, BEFORE step 1 — DUMP CONTEXT ──
print("=" * 70)
print("CONTEXT DUMP: Before step 1 (after init + first discard)")
print("=" * 70)

js_players = js.players
ps_players = ps.players
js_round = js.round_state
ps_round = ps.round_state

print(f"\n--- Round state ---")
print(f"round={int(js_round.round)} dealer={int(js_round.dealer)}")
print(f"next_deck_ix={int(js_round.next_deck_ix)} last_deck_ix={int(js_round.last_deck_ix)}")
print(f"last_draw={int(js_round.last_draw)} last_player={int(js_round.last_player)}")
print(f"target={int(js_round.target)} draw_next={bool(js_round.draw_next)}")
print(f"is_haitei={bool(js_round.is_haitei)}")
print(f"dora_indicators={np.array(js_round.dora_indicators)}")
print(f"seat_wind={np.array(js_round.seat_wind)}")
print(f"score={np.array(js_round.score)}")
print(f"kyotaku={int(js_round.kyotaku)} honba={int(js_round.honba)}")

print(f"\n--- Per-player hands (37-type) ---")
for p in range(4):
    jh = np.array(js_players.hand_with_red[p])
    ph = ps_players.hand_with_red[p].numpy()
    jsh = short_hand(jh)
    psh = short_hand(ph)
    match = "MATCH" if np.array_equal(jh, ph) else "DIFF!"
    print(f"  Player {p}: JAX={jsh}")
    if not np.array_equal(jh, ph):
        print(f"          PT ={psh}  *** {match}")
        print(f"          JAX raw={[int(x) for x in jh]}")
        print(f"          PT  raw={ph.tolist()}")

print(f"\n--- Per-player melds ---")
for p in range(4):
    jm = np.array(js_players.melds[p])
    pm = ps_players.melds[p].numpy()
    jmc = int(js_players.meld_counts[p])
    pmc = int(ps_players.meld_counts[p])
    print(f"  Player {p}: count J={jmc} P={pmc}  melds J={[int(x) for x in jm[:jmc]]} P={pm.tolist()[:pmc]}")

print(f"\n--- Per-player Yaku precompute (after step 0 discard) ---")
print(f"  field: JAX value  |  PT value")
print(f"  has_yaku: {np.array(js_players.has_yaku).tolist()}  |  {ps_players.has_yaku.numpy().tolist()}")
print(f"  fan:      {np.array(js_players.fan).tolist()}  |  {ps_players.fan.numpy().tolist()}")
print(f"  fu:       {np.array(js_players.fu).tolist()}  |  {ps_players.fu.numpy().tolist()}")

# Find the player where fan differs
jfan = np.array(js_players.fan)
pfan = ps_players.fan.numpy()
diff_players = np.where((jfan != pfan).any(axis=1))[0]
print(f"\n--- Fan diff players: {diff_players.tolist()} ---")

# For each diff player, dump the exact input to Yaku.judge
from mahjax_pt.red_mahjong.yaku import Yaku as PtYaku
from mahjax.red_mahjong.yaku import Yaku as JaxYaku

for p in diff_players:
    jh37 = np.array(js_players.hand_with_red[p])
    ph37 = ps_players.hand_with_red[p].numpy()

    print(f"\n=== Player {p} Yaku.judge detail ===")
    print(f"  Hand 37: {[int(x) for x in jh37]}")
    print(f"  Hand readable: {short_hand(jh37)}")

    # Call PT Yaku.judge and dump internals
    print(f"\n  --- PT Yaku.judge(hand, is_ron=True, player={p}) ---")
    try:
        import io, sys
        pt_yaku, pt_fan, pt_fu = PtYaku.judge(torch.from_numpy(ph37.copy()), True, p, ps)
        print(f"  yaku={pt_yaku} fan={pt_fan} fu={pt_fu}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print(f"\n  --- PT Yaku.judge(hand, is_ron=False, player={p}) ---")
    try:
        pt_yaku2, pt_fan2, pt_fu2 = PtYaku.judge(torch.from_numpy(ph37.copy()), False, p, ps)
        print(f"  yaku={pt_yaku2} fan={pt_fan2} fu={pt_fu2}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Also show JAX values
    print(f"\n  --- JAX precomputed ---")
    print(f"  has_yaku = {np.array(js_players.has_yaku[p]).tolist()}")
    print(f"  fan      = {np.array(js_players.fan[p]).tolist()}")
    print(f"  fu       = {np.array(js_players.fu[p]).tolist()}")

# Also dump the discarded tile info
print(f"\n--- Step 0 action info ---")
print(f"  action taken: {a}")
print(f"  discarded_tile (target) before step 0: {int(js_round.target)}")
print(f"  discarded_tile type: the tile that was just discarded in step 0")
print(f"  next deck tile (for tsumo precompute): deck[{int(js_round.next_deck_ix)}] = {int(js_round.deck[int(js_round.next_deck_ix)])}")

# Show first few deck tiles for context
deck = np.array(js_round.deck)
print(f"  Deck[80:84] (recently drawn): {deck[80:84].tolist()}")
print(f"  Deck[next_deck_ix-3:next_deck_ix+3]: {deck[int(js_round.next_deck_ix)-3:int(js_round.next_deck_ix)+3].tolist()}")

print(f"\n{'=' * 70}")
print("END CONTEXT DUMP — send this entire output for debugging")
print(f"{'=' * 70}")
