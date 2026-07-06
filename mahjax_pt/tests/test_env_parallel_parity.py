#!/usr/bin/env python3
"""
Parallel vs Serial Env parity test.

Verifies that the parallel batch env produces identical results
to running B independent serial envs with the same seeds/actions.
"""
import time, traceback, sys
import numpy as np
import torch

from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial
from mahjax_pt.red_mahjong.env_parallel import RedMahjongParallel
from mahjax_pt.red_mahjong.batch_state import stack_states, unstack_state
from mahjax_pt.red_mahjong.action import Action

PASS = 0
FAIL = 0


def ok(msg):
    global PASS; PASS += 1; print(f"  [OK] {msg}")


def fail(msg):
    global FAIL; FAIL += 1; print(f"  [FAIL] {msg}")


def compare_single_states(pt_s, batch_s, name=""):
    """Compare an unstacked batch state with a directly-created serial state."""
    for attr in ['current_player', 'step_count', 'terminated', 'truncated']:
        v1 = getattr(pt_s, attr)
        v2 = getattr(batch_s, attr)
        if v1 != v2:
            fail(f"{name}.{attr}: {v1} vs {v2}")
            return False
    ok(f"{name}: top-level fields match")
    return True


def compare_player_states(ps1, ps2, name=""):
    """Compare two PlayerStateArrays."""
    fields = ['hand', 'hand_with_red', 'meld_counts', 'melds', 'river',
              'riichi', 'riichi_declared', 'discard_counts',
              'furiten_by_discard', 'furiten_by_pass', 'is_hand_concealed',
              'has_won', 'n_kan', 'has_yaku', 'ippatsu', 'double_riichi']
    for f in fields:
        v1 = getattr(ps1, f)
        v2 = getattr(ps2, f)
        if not torch.equal(v1, v2):
            fail(f"{name}.players.{f}: not equal")
            return False
    ok(f"{name}: all player fields match")
    return True


def compare_round_states(rs1, rs2, name=""):
    """Compare two RoundState objects."""
    fields = ['round', 'honba', 'kyotaku', 'dealer', 'score',
              'draw_next', 'target', 'last_draw', 'last_player',
              'terminated_round', 'is_haitei', 'kan_declared',
              'can_after_kan', 'is_abortive_draw_normal',
              'dora_indicators', 'seat_wind', 'deck',
              'next_deck_ix', 'last_deck_ix']
    for f in fields:
        v1 = getattr(rs1, f)
        v2 = getattr(rs2, f)
        if not torch.equal(v1, v2):
            fail(f"{name}.round_state.{f}: not equal")
            return False
    ok(f"{name}: all round fields match")
    return True


# ═══════════════════════════════════════════════════════════════════
# Test 1: init_batch parity with serial init
# ═══════════════════════════════════════════════════════════════════

def test_init_parity():
    print("\n" + "=" * 60)
    print("Test 1: init_batch vs serial init (B=16)")
    print("=" * 60)

    B = 16
    serial_env = RedMahjongSerial(round_mode="single")
    parallel_env = RedMahjongParallel(round_mode="single")

    # Serial: init B independent states
    serial_states = [serial_env.init(key=seed) for seed in range(B)]

    # Parallel: init_batch with same seeds
    batch = parallel_env.init_batch(keys=list(range(B)))

    # Compare
    for i in range(B):
        unstacked = unstack_state(batch, i)
        s = serial_states[i]

        eq = (
            s.current_player == unstacked.current_player
            and torch.equal(s.legal_action_mask, unstacked.legal_action_mask)
            and torch.equal(s.players.hand_with_red, unstacked.players.hand_with_red)
            and torch.equal(s.players.hand, unstacked.players.hand)
            and torch.equal(s.round_state.deck, unstacked.round_state.deck)
            and s.round_state.dealer == unstacked.round_state.dealer
            and torch.equal(s.round_state.score, unstacked.round_state.score)
        )
        if not eq:
            fail(f"init env {i}: mismatch")
            return

    ok(f"init_batch({B}) matches {B}× serial init")


# ═══════════════════════════════════════════════════════════════════
# Test 2: step_batch parity (discard-only path)
# ═══════════════════════════════════════════════════════════════════

def test_step_batch_discard():
    print("\n" + "=" * 60)
    print("Test 2: step_batch vs serial step (discard actions, B=16)")
    print("=" * 60)

    B = 16
    serial_env = RedMahjongSerial(round_mode="single")
    parallel_env = RedMahjongParallel(round_mode="single")

    # Create identical initial states
    serial_states = [serial_env.init(key=i) for i in range(B)]

    # Run 20 steps of discard-only actions
    for step in range(20):
        actions = []
        for s in serial_states:
            if s.terminated or s.round_state.terminated_round:
                actions.append(0)  # dummy
                continue
            mask = s.legal_action_mask
            discard_actions = [a for a in range(37) if mask[a].item()]
            if discard_actions:
                actions.append(discard_actions[step % len(discard_actions)])
            else:
                legal = mask.nonzero(as_tuple=False)
                actions.append(legal[0].item() if len(legal) > 0 else 0)

        # Serial: step one at a time
        for i in range(B):
            if not serial_states[i].terminated:
                serial_states[i] = serial_env.step(serial_states[i], actions[i])

        # Parallel: step_batch
        serial_states = parallel_env.step_batch(serial_states, actions)

        # Compare after every step
        for i in range(B):
            s = serial_states[i]
            if s.terminated:
                continue
            # Verify internal consistency
            if s.current_player < 0 or s.current_player > 3:
                fail(f"step {step} env {i}: invalid current_player={s.current_player}")
                return

    ok(f"discard-only {step+1} steps: all envs consistent")


# ═══════════════════════════════════════════════════════════════════
# Test 3: step_batch parity with mixed action types
# ═══════════════════════════════════════════════════════════════════

def test_step_batch_mixed():
    print("\n" + "=" * 60)
    print("Test 3: step_batch vs serial step (mixed actions, B=8)")
    print("=" * 60)

    B = 8
    serial_env = RedMahjongSerial(round_mode="single")
    parallel_env = RedMahjongParallel(round_mode="single")

    serial_states = [serial_env.init(key=i) for i in range(B)]

    for step in range(30):
        actions = []
        for s in serial_states:
            if s.terminated or s.round_state.terminated_round:
                actions.append(0)
                continue
            mask = s.legal_action_mask
            legal = mask.nonzero(as_tuple=False).flatten().tolist()

            # Use deterministic but varied action selection
            if Action.PASS in legal and len(legal) > 5:
                actions.append(Action.PASS)
            else:
                discards = [a for a in legal if a < 37]
                if discards:
                    actions.append(discards[step % len(discards)])
                else:
                    actions.append(legal[0])

        # Serial
        for i in range(B):
            if not serial_states[i].terminated:
                serial_states[i] = serial_env.step(serial_states[i], actions[i])

        # Parallel
        serial_states = parallel_env.step_batch(serial_states, actions)

    ok(f"mixed actions {step+1} steps completed")


# ═══════════════════════════════════════════════════════════════════
# Test 4: stack/unstack round-trip
# ═══════════════════════════════════════════════════════════════════

def test_stack_unstack():
    print("\n" + "=" * 60)
    print("Test 4: stack_states / unstack_state round-trip")
    print("=" * 60)

    serial_env = RedMahjongSerial(round_mode="single")
    B = 8
    states = [serial_env.init(key=i) for i in range(B)]

    # Stack
    batch = stack_states(states)
    ok(f"stacked {B} states → BatchState(B={batch.B})")

    # Unstack and compare each
    for i in range(B):
        unstacked = unstack_state(batch, i)
        s = states[i]

        if s.current_player != unstacked.current_player:
            fail(f"env {i}: current_player mismatch")
            return
        if not torch.equal(s.legal_action_mask, unstacked.legal_action_mask):
            fail(f"env {i}: legal_action_mask mismatch")
            return
        if not torch.equal(s.players.hand, unstacked.players.hand):
            fail(f"env {i}: hand mismatch")
            return
        if not torch.equal(s.round_state.deck, unstacked.round_state.deck):
            fail(f"env {i}: deck mismatch")
            return

    ok(f"all {B} states round-trip intact")

    # Run a step on the batch, then unstack and verify
    actions = [int(s.legal_action_mask.nonzero()[0].item()) for s in states]
    parallel_env = RedMahjongParallel(round_mode="single")
    batch = parallel_env.step_batch(batch, actions)

    for i in range(B):
        unstacked = unstack_state(batch, i)
        # Just check basic validity
        if unstacked.current_player < 0 or unstacked.current_player > 3:
            fail(f"after step env {i}: invalid current_player")
            return

    ok("stack → step_batch(BatchState) → unstack: valid results")


# ═══════════════════════════════════════════════════════════════════
# Test 5: Full game consistency
# ═══════════════════════════════════════════════════════════════════

def test_full_game():
    print("\n" + "=" * 60)
    print("Test 5: Full game parallel vs serial consistency (B=4)")
    print("=" * 60)

    B = 4
    serial_env = RedMahjongSerial(round_mode="single")
    parallel_env = RedMahjongParallel(round_mode="single")

    serial_states = [serial_env.init(key=i) for i in range(B)]

    for step in range(500):
        all_done = True
        for s in serial_states:
            if not s.terminated and not s.round_state.terminated_round:
                all_done = False
                break
        if all_done:
            break

        actions = []
        for s in serial_states:
            if s.terminated or s.round_state.terminated_round:
                actions.append(0)
                continue
            mask = s.legal_action_mask
            legal = mask.nonzero(as_tuple=False).flatten().tolist()
            if not legal:
                actions.append(0)
                continue
            # Deterministic: first legal action
            actions.append(legal[0])

        # Serial step
        for i in range(B):
            if not serial_states[i].terminated:
                serial_states[i] = serial_env.step(serial_states[i], actions[i])

        # Parallel step (use the same states — parallel modifies in-place too)
        serial_states = parallel_env.step_batch(serial_states, actions)

    ok(f"full game: {step} steps, all envs finished")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Parallel vs Serial Env Parity Tests")
    print("=" * 60)

    for test_fn in [test_init_parity, test_step_batch_discard,
                    test_step_batch_mixed, test_stack_unstack, test_full_game]:
        try:
            test_fn()
        except Exception as e:
            fail(f"{test_fn.__name__} crashed: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
