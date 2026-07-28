"""Precision tests: Env init and step via Golden data replay.

Compares PT env state against JAX golden data at every step.

Prerequisites:
    Golden .npz files in tests/golden/ recorded by scripts/record_golden_env.py
    Tests skip gracefully if golden data is missing.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from conftest import has_golden, load_golden, load_golden_playout, assert_state_equal, CHECKS
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _inject_golden_state(env: RedMahjongSerial, golden: dict) -> None:
    """Given an EnvState and a golden dict, overwrite PT state to match golden.

    This is a deterministic way to start PT from a known state without
    calling env.init() — ensuring we replay from exactly the same state
    the JAX reference started from.
    """
    # We use env.init first to get a properly initialized state, then
    # overwrite every field from golden. This is safer than trying to
    # construct the state from scratch.
    pass


def _run_replay_playout(env: RedMahjongSerial, seed: int, max_failures: int = 3):
    """Replay a full golden playout and report mismatches.

    Returns: (n_steps, n_failures, failure_details)
    """
    golden_steps = load_golden_playout(seed)
    if not golden_steps:
        return 0, 0, []

    # Start PT from golden init by calling env.init and then overlaying golden
    state = env.init(key=seed)
    init_golden = load_golden(f"env_init_s{seed}")

    # Overlay init state fields from golden onto PT state
    _overlay_state(state, init_golden)

    # Verify init
    init_failures = assert_state_equal(init_golden, state)
    if init_failures:
        return 0, len(init_failures), [f"INIT: {f}" for f in init_failures]

    failures = []
    for step_idx, step_golden in enumerate(golden_steps):
        action = int(step_golden.get("action", step_golden.get("_action", -1)))
        if action < 0:
            failures.append(f"Step {step_idx}: no action in golden data")
            if len(failures) >= max_failures:
                break
            continue

        try:
            state = env.step(state, action)
        except Exception as e:
            failures.append(f"Step {step_idx}: env.step crashed: {e}")
            break

        step_failures = assert_state_equal(step_golden, state)
        for f in step_failures:
            failures.append(f"Step {step_idx}: {f}")
            if len(failures) >= max_failures:
                break
        if len(failures) >= max_failures:
            break

    return len(golden_steps), len(failures), failures[:max_failures]


def _overlay_state(state, golden: dict):
    """Overlay golden field values onto a PT EnvState.

    This is used to start PT from the exact JAX initial state.
    Only fields present in the golden dict are overwritten.
    """
    # Top-level
    if "current_player" in golden:
        state.current_player = int(golden["current_player"])
    if "terminated" in golden:
        state.terminated = bool(golden["terminated"])
    if "step_count" in golden:
        state.step_count = int(golden["step_count"])

    if "legal_action_mask" in golden:
        state.legal_action_mask = torch.from_numpy(
            golden["legal_action_mask"].copy()).bool()

    if "rewards" in golden:
        state.rewards = torch.from_numpy(
            np.asarray(golden["rewards"], dtype=np.float32)).float()

    # Player fields
    pp = state.players
    G = golden
    _copy_if_present(G, "players.hand", pp.hand)
    _copy_if_present(G, "players.hand_with_red", pp.hand_with_red)
    _copy_if_present(G, "players.melds", pp.melds)
    _copy_if_present(G, "players.meld_counts", pp.meld_counts)
    _copy_if_present(G, "players.discard_counts", pp.discard_counts)
    _copy_if_present(G, "players.river", pp.river)
    _copy_if_present(G, "players.riichi", pp.riichi)
    _copy_if_present(G, "players.riichi_declared", pp.riichi_declared)
    _copy_if_present(G, "players.has_won", pp.has_won)
    _copy_if_present(G, "players.n_kan", pp.n_kan)
    _copy_if_present(G, "players.has_yaku", pp.has_yaku)
    _copy_if_present(G, "players.is_hand_concealed", pp.is_hand_concealed)
    _copy_if_present(G, "players.furiten_by_discard", pp.furiten_by_discard)
    _copy_if_present(G, "players.furiten_by_pass", pp.furiten_by_pass)
    _copy_if_present(G, "players.ippatsu", pp.ippatsu)
    _copy_if_present(G, "players.fan", pp.fan)
    _copy_if_present(G, "players.fu", pp.fu)

    # Round fields
    rs = state.round_state
    _copy_scalar_if_present(G, "round_state.round", rs, "round")
    _copy_scalar_if_present(G, "round_state.dealer", rs, "dealer")
    _copy_scalar_if_present(G, "round_state.next_deck_ix", rs, "next_deck_ix")
    _copy_scalar_if_present(G, "round_state.last_deck_ix", rs, "last_deck_ix")
    _copy_scalar_if_present(G, "round_state.last_draw", rs, "last_draw")
    _copy_scalar_if_present(G, "round_state.last_player", rs, "last_player")
    _copy_scalar_if_present(G, "round_state.target", rs, "target")
    _copy_scalar_if_present(G, "round_state.draw_next", rs, "draw_next")
    _copy_scalar_if_present(G, "round_state.is_haitei", rs, "is_haitei")
    _copy_scalar_if_present(G, "round_state.is_abortive_draw_normal", rs, "is_abortive_draw_normal")
    _copy_scalar_if_present(G, "round_state.terminated_round", rs, "terminated_round")
    _copy_scalar_if_present(G, "round_state.honba", rs, "honba")
    _copy_scalar_if_present(G, "round_state.kyotaku", rs, "kyotaku")
    _copy_scalar_if_present(G, "round_state.n_kan_doras", rs, "n_kan_doras")

    _copy_if_present(G, "round_state.score", rs.score)
    _copy_if_present(G, "round_state.deck", rs.deck)
    _copy_if_present(G, "round_state.dora_indicators", rs.dora_indicators)


def _copy_if_present(golden: dict, key: str, tensor: torch.Tensor):
    if key in golden:
        arr = np.asarray(golden[key])
        tensor[:] = torch.from_numpy(arr).to(tensor.dtype)


def _copy_scalar_if_present(golden: dict, key: str, obj, attr: str):
    if key in golden:
        setattr(obj, attr, type(getattr(obj, attr))(golden[key]))


# ═══════════════════════════════════════════════════════════════════════
# PA01: Init state consistency
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("seed", list(range(100)))
def test_init_precision(seed: int):
    """PA01: PT env.init matches JAX golden init state for 100 seeds."""
    if not has_golden(f"env_init_s{seed}.npz"):
        pytest.skip(f"Golden file env_init_s{seed}.npz not found")

    golden = load_golden(f"env_init_s{seed}")
    env = RedMahjongSerial(round_mode="single")
    state = env.init(key=seed)

    failures = assert_state_equal(golden, state)
    if failures:
        pytest.fail(f"Seed {seed} init mismatch:\n" + "\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA02: Init determinism
# ═══════════════════════════════════════════════════════════════════════

def test_init_deterministic():
    """PA02: Same seed produces identical init states."""
    env = RedMahjongSerial(round_mode="single")
    s1 = env.init(key=42)
    s2 = env.init(key=42)

    # Compare key fields
    assert s1.current_player == s2.current_player
    assert torch.equal(s1.players.hand_with_red, s2.players.hand_with_red)
    assert torch.equal(s1.round_state.deck, s2.round_state.deck)
    assert torch.equal(s1.legal_action_mask, s2.legal_action_mask)
    assert s1.round_state.dealer == s2.round_state.dealer


# ═══════════════════════════════════════════════════════════════════════
# PA03-PA10: Step precision — full playout replay
# ═══════════════════════════════════════════════════════════════════════

_PLAYOUT_SEEDS = list(range(200))


@pytest.mark.slow
@pytest.mark.parametrize("seed", _PLAYOUT_SEEDS[:50])
def test_step_playout_precision(seed: int):
    """PA03/PA10: Full playout replay — golden vs PT step-by-step."""
    if not has_golden(f"env_playout_s{seed}.npz"):
        pytest.skip(f"Golden file env_playout_s{seed}.npz not found")

    env = RedMahjongSerial(round_mode="single")
    n_steps, n_failures, details = _run_replay_playout(env, seed, max_failures=5)

    if n_failures > 0:
        pytest.fail(f"Seed {seed}: {n_failures}/{n_steps} steps failed:\n"
                    + "\n".join(details[:5]))


# ═══════════════════════════════════════════════════════════════════════
# Grouped action-type tests (PA04-PA09)
# These scan golden files for steps containing specific action types
# and verify only those steps.
# ═══════════════════════════════════════════════════════════════════════

# Action constants (mirrored from mahjax_pt.red_mahjong.action)
DISCARD_FIRST, DISCARD_LAST = 0, 36
SELFKAN_FIRST, SELFKAN_LAST = 37, 70
TSUMOGIRI = 71
RIICHI = 72
TSUMO = 73
RON = 74
PON = 75
PON_RED = 76
OPEN_KAN = 77
CHI_L, CHI_L_RED = 78, 79
CHI_M, CHI_M_RED = 80, 81
CHI_R, CHI_R_RED = 82, 83
PASS = 84
KYUUSHU = 85
DUMMY = 86


def _scan_golden_for_actions(action_predicate, min_steps: int = 30,
                             max_seeds: int = 500, start_seed: int = 200):
    """Scan golden playout files for steps matching action_predicate.

    Returns list of (seed, step_idx, action) tuples.
    """
    results = []
    for seed in range(start_seed, start_seed + max_seeds):
        if not has_golden(f"env_playout_s{seed}.npz"):
            continue
        try:
            steps = load_golden_playout(seed)
            for i, step in enumerate(steps):
                action = int(step.get("action", -1))
                if action_predicate(action):
                    results.append((seed, i, action))
                    if len(results) >= min_steps:
                        return results
        except Exception:
            continue
    return results


def _replay_single_step(seed: int, target_step: int) -> list:
    """Replay a playout up to target_step, return failures at that step."""
    env = RedMahjongSerial(round_mode="single")
    golden_steps = load_golden_playout(seed)
    init_golden = load_golden(f"env_init_s{seed}")

    state = env.init(key=seed)
    _overlay_state(state, init_golden)

    # Replay up to target_step
    for i in range(target_step):
        action = int(golden_steps[i].get("action", -1))
        if action < 0:
            return [f"Step {i}: missing action"]
        try:
            state = env.step(state, action)
        except Exception as e:
            return [f"Step {i}: crash: {e}"]

    # Verify target step
    return assert_state_equal(golden_steps[target_step], state)


@pytest.mark.slow
def test_step_pon_chi_precision():
    """PA04: Verify steps containing pon/chi actions match golden."""
    def is_meld(a):
        return a in (PON, PON_RED, CHI_L, CHI_L_RED, CHI_M, CHI_M_RED, CHI_R, CHI_R_RED)

    cases = _scan_golden_for_actions(is_meld, min_steps=30)
    if not cases:
        pytest.skip("No pon/chi steps found in golden data")

    failures = []
    for seed, step_idx, action in cases[:30]:
        step_fails = _replay_single_step(seed, step_idx)
        for f in step_fails:
            failures.append(f"Seed {seed} step {step_idx} (action={action}): {f}")

    if failures:
        pytest.fail(f"{len(failures)} pon/chi step failures:\n" + "\n".join(failures[:10]))


@pytest.mark.slow
def test_step_kan_precision():
    """PA05: Verify steps containing kan actions match golden."""
    def is_kan(a):
        if SELFKAN_FIRST <= a <= SELFKAN_LAST:
            return True
        return a == OPEN_KAN

    cases = _scan_golden_for_actions(is_kan, min_steps=20)
    if not cases:
        pytest.skip("No kan steps found in golden data")

    failures = []
    for seed, step_idx, action in cases[:20]:
        step_fails = _replay_single_step(seed, step_idx)
        for f in step_fails:
            failures.append(f"Seed {seed} step {step_idx} (action={action}): {f}")

    if failures:
        pytest.fail(f"{len(failures)} kan step failures:\n" + "\n".join(failures[:10]))


@pytest.mark.slow
def test_step_riichi_precision():
    """PA06: Verify steps containing riichi actions match golden."""
    cases = _scan_golden_for_actions(lambda a: a == RIICHI, min_steps=20)
    if not cases:
        pytest.skip("No riichi steps found in golden data")

    failures = []
    for seed, step_idx, _ in cases[:20]:
        step_fails = _replay_single_step(seed, step_idx)
        for f in step_fails:
            failures.append(f"Seed {seed} step {step_idx} (riichi): {f}")

    if failures:
        pytest.fail(f"{len(failures)} riichi step failures:\n" + "\n".join(failures[:10]))


@pytest.mark.slow
def test_step_ron_precision():
    """PA07: Verify ron settlement matches golden."""
    cases = _scan_golden_for_actions(lambda a: a == RON, min_steps=20)
    if not cases:
        pytest.skip("No ron steps found in golden data")

    failures = []
    for seed, step_idx, _ in cases[:20]:
        step_fails = _replay_single_step(seed, step_idx)
        for f in step_fails:
            failures.append(f"Seed {seed} step {step_idx} (ron): {f}")

    if failures:
        pytest.fail(f"{len(failures)} ron step failures:\n" + "\n".join(failures[:10]))


@pytest.mark.slow
def test_step_tsumo_precision():
    """PA08: Verify tsumo settlement matches golden."""
    cases = _scan_golden_for_actions(lambda a: a == TSUMO, min_steps=20)
    if not cases:
        pytest.skip("No tsumo steps found in golden data")

    failures = []
    for seed, step_idx, _ in cases[:20]:
        step_fails = _replay_single_step(seed, step_idx)
        for f in step_fails:
            failures.append(f"Seed {seed} step {step_idx} (tsumo): {f}")

    if failures:
        pytest.fail(f"{len(failures)} tsumo step failures:\n" + "\n".join(failures[:10]))


@pytest.mark.slow
def test_step_ryukyoku_precision():
    """PA09: Verify exhaustive draw / abortive draw matches golden."""
    def is_ryukyoku(a):
        return a in (KYUUSHU, DUMMY)

    cases = _scan_golden_for_actions(is_ryukyoku, min_steps=10)
    if not cases:
        pytest.skip("No ryukyoku steps found in golden data")

    failures = []
    for seed, step_idx, action in cases[:10]:
        step_fails = _replay_single_step(seed, step_idx)
        for f in step_fails:
            failures.append(f"Seed {seed} step {step_idx} (action={action}): {f}")

    if failures:
        pytest.fail(f"{len(failures)} ryukyoku step failures:\n" + "\n".join(failures[:10]))
