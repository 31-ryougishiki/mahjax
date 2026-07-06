#!/usr/bin/env python3
"""Count distinct JIT compilations in JAX mahjax env by measuring elapsed time.

On first call (>1s) = new JIT compile; subsequent same-path calls (<0.01s) = cached.
"""
import time, numpy as np, jax, jax.numpy as jnp
from mahjax.red_mahjong.env import RedMahjong

# Count jax.lax.cond / lax.switch in the env module to estimate max compilations
import mahjax.red_mahjong.env as jenv
import inspect

src = inspect.getsource(jenv)
n_cond = src.count("lax.cond")
n_switch = src.count("lax.switch")
n_jit = src.count("@jax.jit")

print(f"=== JAX mahjax env compilation analysis ===")
print(f"  @jax.jit decorations : {n_jit}")
print(f"  lax.cond calls        : {n_cond}")
print(f"  lax.switch calls      : {n_switch}")
print(f"  (each unique cond/switch branch may trigger a sub-compilation)")
print()

env = RedMahjong(round_mode="single")
key = jax.random.PRNGKey(99)

print("Init...", end=" ", flush=True)
t0 = time.time()
state = env.init(key)
print(f"{time.time()-t0:.1f}s (compile _init)")
print()

# Run steps and track compilation events
print("=== Step-by-step compilation tracking ===")
print(f"{'Step':<6} {'Action':<16} {'Time':<10} {'Note'}")
print("-" * 55)

compile_count = 0
THRESHOLD = 0.5  # seconds — above this = new compilation

for step in range(30):
    if bool(state.terminated):
        print(f"  -> game terminated at step {step}")
        break
    if bool(state.round_state.terminated_round):
        print(f"  -> round terminated at step {step}")
        break

    mask = np.array(state.legal_action_mask)
    legal = np.where(mask)[0]
    if len(legal) == 0:
        break

    # Identify special actions available
    specials = {}
    for name, aid in [("RON",73),("TSUMO",74),("RIICHI",72),("PON",75),
                       ("PON_RED",76),("OPEN_KAN",77),("KYUUSHU",85),
                       ("PASS",84)]:
        if aid in legal:
            specials[name] = aid

    # Pick action: prefer special to trigger diverse compile paths
    if specials:
        # Pick the most interesting (ron > tsumo > riichi > pon > pass)
        for pref in ["RON","TSUMO","RIICHI","PON","PON_RED","OPEN_KAN","PASS","KYUUSHU"]:
            if pref in specials:
                action = specials[pref]
                tag = pref
                break
        else:
            action = list(specials.values())[0]
            tag = list(specials.keys())[0]
    else:
        discards = [a for a in legal if a < 37]
        action = discards[0] if discards else legal[0]
        tag = f"discard({action})" if discards else f"other({action})"

    t0 = time.time()
    state = env.step(state, int(action))
    elapsed = time.time() - t0

    if elapsed > THRESHOLD:
        compile_count += 1
        note = f"*** COMPILE #{compile_count} ***"
    else:
        note = "cached"

    print(f"{step:<6} {tag:<16} {elapsed:.3f}s     {note}")

print()
print(f"Total distinct JIT compilations triggered: {compile_count}")
print()
print("Key insight: each unique path through jax.lax.cond/switch")
print("triggers a separate JAX trace + compilation on first encounter.")
print("On GPU these are 100-500ms each; on CPU 5-20s each.")
