"""Precision tests: Special hand-crafted Mahjong edge cases via Golden data replay.

Each test verifies a specific rule scenario (red-five interactions, furiten,
pao, double-ron, nagashi, kyuushu, etc.) by replaying a single-step golden
state created for that scenario.

Prerequisites:
    Golden .npz files in tests/golden/env_special/ recorded by
    scripts/record_golden_special.py
"""

import pytest
import torch

from conftest import has_golden, load_golden, assert_state_equal
from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial
from test_precision_env import _overlay_state


def _run_special_case(scene_name: str):
    """Load a special scene golden file, replay it, return failures."""
    path = f"env_special/{scene_name}"
    if not has_golden(f"{path}.npz"):
        pytest.skip(f"Golden file {path}.npz not found")

    golden = load_golden(path)
    env = RedMahjongSerial(round_mode="single")

    # Init and overlay
    state = env.init(key=0)
    _overlay_state(state, golden)

    # Verify init state
    failures = assert_state_equal(golden, state)
    if failures:
        return ["INIT: " + f for f in failures]

    # If golden contains an action, execute it and verify the next state
    action = golden.get("action", golden.get("_action", -1))
    if int(action) >= 0:
        state = env.step(state, int(action))

        # Load the expected post-step state
        post_key = path + "_post"
        if has_golden(f"{post_key}.npz"):
            post_golden = load_golden(post_key)
            failures = assert_state_equal(post_golden, state)
            return ["STEP: " + f for f in failures]

    return []


# ═══════════════════════════════════════════════════════════════════════
# PA18-PA20: Red-five interactions
# ═══════════════════════════════════════════════════════════════════════

def test_red_five_pon():
    """PA18: Red-five pon — can_red_pon == True, pon removes correct tiles."""
    failures = _run_special_case("red_five_pon")
    if failures:
        pytest.fail("\n".join(failures))


def test_red_five_chi():
    """PA19: Red-five chi — can_red_chi detects red in sequence."""
    failures = _run_special_case("red_five_chi")
    if failures:
        pytest.fail("\n".join(failures))


def test_red_five_kan():
    """PA20: Red-five kan — open_kan / closed_kan / added_kan with red."""
    failures = _run_special_case("red_five_kan")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA21: Furiten
# ═══════════════════════════════════════════════════════════════════════

def test_furiten_by_discard():
    """PA21a: Furiten by discard — ron blocked when own discard is a winning tile."""
    failures = _run_special_case("furiten_by_discard")
    if failures:
        pytest.fail("\n".join(failures))


def test_furiten_by_pass():
    """PA21b: Furiten by pass — temporary furiten after declining ron."""
    failures = _run_special_case("furiten_by_pass")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA22: Ippatsu
# ═══════════════════════════════════════════════════════════════════════

def test_ippatsu():
    """PA22: Ippatsu — riichi → ippatsu flag, meld call cancels it."""
    failures = _run_special_case("ippatsu")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA23: Pao (大三元/四喜和 包牌)
# ═══════════════════════════════════════════════════════════════════════

def test_pao():
    """PA23: Pao — player who fed the 3rd dragon/wind pon pays for yakuman."""
    failures = _run_special_case("pao")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA24: Double Ron
# ═══════════════════════════════════════════════════════════════════════

def test_double_ron():
    """PA24: Double ron — two players can ron the same discard."""
    failures = _run_special_case("double_ron")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA25: Nagashi Mangan
# ═══════════════════════════════════════════════════════════════════════

def test_nagashi_mangan():
    """PA25: Nagashi mangan — all discards are terminals/honors and never called."""
    failures = _run_special_case("nagashi_mangan")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA26: Kyuushu (九种九牌)
# ═══════════════════════════════════════════════════════════════════════

def test_kyuushu():
    """PA26: Kyuushu kyuuhai — 9+ unique terminals/honors in starting hand."""
    failures = _run_special_case("kyuushu")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA27: Four Winds Abortive Draw (四風連打)
# ═══════════════════════════════════════════════════════════════════════

def test_four_winds():
    """PA27: Four winds abortive draw — 4 players discard same wind in first turn."""
    failures = _run_special_case("four_winds")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA28: Rinshan Kaihou (岭上开花)
# ═══════════════════════════════════════════════════════════════════════

def test_rinshan():
    """PA28: Rinshan kaihou — win on the tile drawn after declaring a kan."""
    failures = _run_special_case("rinshan")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA29: Haitei / Houtei (海底/河底)
# ═══════════════════════════════════════════════════════════════════════

def test_haitei():
    """PA29a: Haitei — win on the last draw from the wall."""
    failures = _run_special_case("haitei")
    if failures:
        pytest.fail("\n".join(failures))


def test_houtei():
    """PA29b: Houtei — win on the last discard."""
    failures = _run_special_case("houtei")
    if failures:
        pytest.fail("\n".join(failures))


# ═══════════════════════════════════════════════════════════════════════
# PA30: Kokushi / Seven Pairs / Normal win detection
# ═══════════════════════════════════════════════════════════════════════

def test_kokushi_win():
    """PA30a: Kokushi musou — 13-orphan hand detection."""
    failures = _run_special_case("kokushi")
    if failures:
        pytest.fail("\n".join(failures))


def test_seven_pairs_win():
    """PA30b: Seven pairs — chiitoitsu hand detection."""
    failures = _run_special_case("seven_pairs")
    if failures:
        pytest.fail("\n".join(failures))


def test_normal_win():
    """PA30c: Normal win — standard 4 groups + 1 pair."""
    failures = _run_special_case("normal_win")
    if failures:
        pytest.fail("\n".join(failures))
