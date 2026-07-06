#!/usr/bin/env python3
"""
Verify JAX and PT serial implementations produce matching core game logic.

Approach:
  1. Init JAX, copy state to PT (bypass PRNG differences)
  2. Run steps with identical actions
  3. Compare: hand, deck, river, meld, score, current_player, etc.
  4. Skip: legal_action_mask, has_yaku (known: JAX precomputes yaku eagerly)

On CPU: ~15s per JAX step. 15 steps ≈ 4 minutes.
"""
import time, sys, numpy as np, torch
import jax, jax.numpy as jnp
# Use CPU-mode JAX env (no JIT compilation) for reliable CPU testing
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv
from mahjax_pt.red_mahjong.action import Action

# ── Copy helpers ──
def j2n(v):
    if isinstance(v, jnp.ndarray): return np.array(v)
    if hasattr(v, 'item'): return v.item()
    return v

def copy_jax_to_pt(js, ps):
    """Copy JAX state into PT state."""
    ps.current_player = int(js.current_player)
    ps.terminated = bool(js.terminated)
    ps.step_count = int(js.step_count)
    ps.legal_action_mask = torch.from_numpy(np.array(js.legal_action_mask).copy()).bool()
    ps.rewards = torch.from_numpy(np.array(js.rewards, dtype=np.float32).copy())

    # Player state
    jp, pp = js.players, ps.players
    for f, dt in [('hand',torch.int8),('hand_with_red',torch.int8),
        ('meld_counts',torch.int8),('melds',torch.int32),('riichi',torch.bool),
        ('riichi_declared',torch.bool),('furiten_by_discard',torch.bool),
        ('furiten_by_pass',torch.bool),('is_hand_concealed',torch.bool),
        ('has_won',torch.bool),('n_kan',torch.int8),('discard_counts',torch.int8),
        ('river',torch.int32),('ippatsu',torch.bool),('double_riichi',torch.bool)]:
        setattr(pp, f, torch.from_numpy(np.array(getattr(jp,f)).copy()).to(dt))

    # Round state
    jr, pr = js.round_state, ps.round_state
    for f in ['round','honba','kyotaku','dealer','next_deck_ix','last_deck_ix',
              'last_draw','last_player','target','n_kan_doras']:
        setattr(pr, f, int(getattr(jr,f)))
    for f in ['terminated_round','draw_next','is_haitei','kan_declared','can_after_kan']:
        setattr(pr, f, bool(getattr(jr,f)))
    for f, dt in [('deck',torch.int8),('score',torch.int32),('dora_indicators',torch.int8),
                  ('seat_wind',torch.int8)]:
        setattr(pr, f, torch.from_numpy(np.array(getattr(jr,f)).copy()).to(dt))
    return ps

# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("JAX vs PT Serial: Core Game Logic Alignment")
print("=" * 60)

fields = [
    ("hand",            lambda s: s.players.hand),
    ("deck",            lambda s: s.round_state.deck),
    ("river",           lambda s: s.players.river),
    ("melds",           lambda s: s.players.melds),
    ("meld_counts",     lambda s: s.players.meld_counts),
    ("discard_counts",  lambda s: s.players.discard_counts),
    ("score",           lambda s: s.round_state.score),
    ("current_player",  lambda s: s.current_player),
    ("dealer",          lambda s: s.round_state.dealer),
    ("round/honba",     lambda s: (s.round_state.round, s.round_state.honba)),
    ("riichi",          lambda s: s.players.riichi),
    ("riichi_declared", lambda s: s.players.riichi_declared),
    ("has_won",         lambda s: s.players.has_won),
    ("n_kan",           lambda s: s.players.n_kan),
    ("terminated_round",lambda s: s.round_state.terminated_round),
    ("last_draw",       lambda s: s.round_state.last_draw),
    ("last_player",     lambda s: s.round_state.last_player),
    ("target",          lambda s: s.round_state.target),
]

print("\n1. Init + copy...", end=" ", flush=True)
t0 = time.time()
jax_env = JaxEnv(round_mode="single")
pt_env = PtEnv(round_mode="single")
js = jax_env.init(jax.random.PRNGKey(42))
ps = pt_env.init(key=0)
ps = copy_jax_to_pt(js, ps)
print(f"{time.time()-t0:.1f}s")

# Verify initial copy
all_ok = True
for name, fn in fields:
    jv = j2n(fn(js))
    pv = fn(ps)
    if isinstance(pv, torch.Tensor): pv = pv.numpy()
    if not np.array_equal(np.asarray(jv), np.asarray(pv)):
        print(f"  INIT MISMATCH: {name}")
        all_ok = False
print(f"   Init copy: {'ALL MATCH' if all_ok else 'FAILED'}")

# ═══════════════════════════════════════════════════════════════
print(f"\n2. Running steps (JAX ~15s/step on CPU)...")
n_steps = 0
step_ok = 0
step_fail = 0

print(f"\n2. Running up to 128 steps (JAX ~15s/step on CPU, ~{128*15//60}min max)...")
print(f"   Actions: '.' = discard, 'P' = pon, 'C' = chi, 'K' = open_kan, 'S' = selfkan,")
print(f"            'R' = riichi, 'T' = tsumo, 'W' = ron, 'X' = pass, '9' = kyuushu")
print(f"   {'Step':<6} {'Act':<8} {'Time':<8} {'Result':<6}  Note")
print(f"   {'-'*55}")

n_steps = 0
step_ok = 0
step_fail = 0
action_types_seen = set()
prev_action_type = None

MAX_STEPS = 128

for step in range(MAX_STEPS):
    if bool(js.terminated) or bool(js.round_state.terminated_round):
        print(f"   -> Round/game ended at step {step}")
        break

    legal = np.where(np.array(js.legal_action_mask))[0]
    if len(legal) == 0:
        break

    # Pick action: prefer discard, else first legal
    discards = [a for a in legal if a < 37]
    action = int(discards[step % len(discards)] if discards else legal[0])

    # Classify action for display
    if action < 37:
        action_char = '.'
    elif action == 71: action_char = '.'  # tsumogiri
    elif Action.is_selfkan(action): action_char = 'S'
    elif action == 72: action_char = 'R'
    elif action == 73: action_char = 'W'
    elif action == 74: action_char = 'T'
    elif action in (75, 76): action_char = 'P'
    elif action == 77: action_char = 'K'
    elif 78 <= action <= 83: action_char = 'C'
    elif action == 84: action_char = 'X'
    elif action == 85: action_char = '9'
    elif action == 86: action_char = 'D'
    else: action_char = '?'

    action_types_seen.add(action_char)

    # Step both
    t0 = time.time()
    js = jax_env.step(js, action)
    ps = pt_env.step(ps, action)
    dt = time.time() - t0

    # Compare all fields
    n_steps += 1
    failed = []
    for name, fn in fields:
        jv = j2n(fn(js))
        pv = fn(ps)
        if isinstance(pv, torch.Tensor): pv = pv.numpy()
        if not np.array_equal(np.asarray(jv), np.asarray(pv)):
            failed.append(name)

    if failed:
        step_fail += 1
        print(f"   {step:<6} {action_char:<8} {dt:.1f}s     FAIL   {failed}")
    else:
        step_ok += 1

    # Print progress every 10 steps, or when action type changes
    if step % 10 == 0:
        act_summary = ''.join(sorted(action_types_seen))
        print(f"   {step:<6} {action_char:<8} {dt:.1f}s     OK     types seen: [{act_summary}]")
    elif action_char != prev_action_type and action_char not in ('.',):
        print(f"   {step:<6} {action_char:<8} {dt:.1f}s     OK     first {action_char}")

    prev_action_type = action_char

# ═══════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"RESULT: {step_ok}/{n_steps} steps MATCH, {step_fail} steps FAIL")
print(f"Action types covered: {''.join(sorted(action_types_seen))}")
if step_fail == 0:
    print("PASS: Core game logic fully aligns with JAX.")
else:
    print("FAIL: Differences found — see above.")
print(f"{'=' * 60}")
sys.exit(0 if step_fail == 0 else 1)
