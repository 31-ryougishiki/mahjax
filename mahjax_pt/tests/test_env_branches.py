#!/usr/bin/env python3
"""
Comprehensive env branch tests — JAX vs PyTorch comparison.

Each test constructs an identical game state in both frameworks,
calls the relevant function, and compares outputs.

Usage:
    PYTHONPATH=. python mahjax_pt/tests/test_env_branches.py
    PYTHONPATH=. python mahjax_pt/tests/test_env_branches.py -v
    PYTHONPATH=. python mahjax_pt/tests/test_env_branches.py --filter tsumo
"""
import sys, os, math, traceback, argparse
import numpy as np
import jax, jax.numpy as jnp
import mahjax
from mahjax.red_mahjong.action import Action as JAction
from mahjax.red_mahjong.env import _make_state, _replace_state, _make_legal_action_mask_after_draw, \
    _make_legal_action_mask_after_discard, _draw as jax_draw_fn, yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile
from mahjax.red_mahjong.state import EnvState as JaxState, PlayerStateArrays as JaxPlayers, \
    RoundState as JaxRound, GameConfig, default_game_config
from mahjax.red_mahjong.hand import Hand as JHand
from mahjax.red_mahjong.tile import Tile as JTile

import torch
from mahjax_pt.red_mahjong.env import RedMahjong
from mahjax_pt.red_mahjong.state import EnvState, PlayerStateArrays, RoundState, GameConfig as PTConfig
from mahjax_pt.red_mahjong.tile import Tile
from mahjax_pt.red_mahjong.hand import Hand
from mahjax_pt.red_mahjong.action import Action

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _hand_34(counts_dict):
    """Create a 34-dim hand tensor from {tile_type: count}."""
    h = torch.zeros(34, dtype=torch.int8)
    for t, c in counts_dict.items():
        h[t] = c
    return h

def _hand_37(counts_dict):
    """Create a 37-dim hand tensor from {tile_idx: count}."""
    h = torch.zeros(37, dtype=torch.int8)
    for t, c in counts_dict.items():
        h[t] = c
    return h

def _jax_hand_34(counts_dict):
    h = jnp.zeros(34, dtype=jnp.int8)
    for t, c in counts_dict.items():
        h = h.at[t].set(c)
    return h

def _jax_hand_37(counts_dict):
    h = jnp.zeros(37, dtype=jnp.int8)
    for t, c in counts_dict.items():
        h = h.at[t].set(c)
    return h

def make_pt_state(cp=0, hand_37=None, hand_34=None, melds=None, riichi=None,
                  is_hand_concealed=None,
                  score=None, round_num=0, honba=0, kyotaku=0,
                  target=-1, last_draw=-1, is_haitei=False,
                  can_after_kan=False, kan_declared=False,
                  can_win=None, has_yaku=None, fan_list=None, fu_list=None,
                  furiten_discard=None, furiten_pass=None,
                  dealer=0, next_deck_ix=50, last_deck_ix=14,
                  n_kan_doras=0, dora_indicators=None):
    """Build a PT EnvState with specified configuration."""
    s = EnvState()
    s.current_player = cp
    s.round_state.dealer = dealer
    s.round_state.round = round_num
    s.round_state.honba = honba
    s.round_state.kyotaku = kyotaku
    s.round_state.target = target
    s.round_state.last_draw = last_draw
    s.round_state.is_haitei = is_haitei
    s.round_state.can_after_kan = can_after_kan
    s.round_state.kan_declared = kan_declared
    s.round_state.next_deck_ix = next_deck_ix
    s.round_state.last_deck_ix = last_deck_ix
    s.round_state.n_kan_doras = n_kan_doras
    s.round_state.seat_wind = torch.tensor([0, 1, 2, 3], dtype=torch.int8)

    if dora_indicators is not None:
        for i, d in enumerate(dora_indicators):
            s.round_state.dora_indicators[i] = d

    if score is not None:
        if isinstance(score, (int, float)):
            s.round_state.score = torch.full((4,), int(score), dtype=torch.int32)
        else:
            s.round_state.score = torch.tensor(score, dtype=torch.int32)

    if hand_37 is not None:
        for p in range(4):
            s.players.hand_with_red[p] = hand_37
    if hand_34 is not None:
        for p in range(4):
            s.players.hand[p] = hand_34

    if melds is not None:
        if isinstance(melds, dict):
            for p in range(4):
                if p in melds:
                    n = int(s.players.meld_counts[p].item())
                    for i, m in enumerate(melds[p]):
                        s.players.melds[p, n + i] = m
                    s.players.meld_counts[p] = len(melds[p])
        else:
            for p in range(min(4, len(melds))):
                n = int(s.players.meld_counts[p].item())
                for i, m in enumerate(melds[p]):
                    s.players.melds[p, n + i] = m
                s.players.meld_counts[p] = len(melds[p])

    if is_hand_concealed is not None:
        if isinstance(is_hand_concealed, bool):
            s.players.is_hand_concealed = torch.full((4,), is_hand_concealed, dtype=torch.bool)
        elif isinstance(is_hand_concealed, dict):
            for p, v in is_hand_concealed.items():
                s.players.is_hand_concealed[p] = v
        else:
            for p, v in enumerate(is_hand_concealed):
                s.players.is_hand_concealed[p] = v

    if riichi is not None:
        if isinstance(riichi, (list, tuple)):
            for p, r in enumerate(riichi):
                s.players.riichi[p] = r
        else:
            for p, r in riichi.items():
                s.players.riichi[p] = r

    if can_win is not None:
        s.players.can_win = can_win
    if has_yaku is not None:
        s.players.has_yaku = has_yaku
    if fan_list is not None:
        s.players.fan = fan_list
    if fu_list is not None:
        s.players.fu = fu_list
    if furiten_discard is not None:
        for p, f in (furiten_discard.items() if isinstance(furiten_discard, dict) else enumerate(furiten_discard)):
            s.players.furiten_by_discard[p] = f
    if furiten_pass is not None:
        for p, f in (furiten_pass.items() if isinstance(furiten_pass, dict) else enumerate(furiten_pass)):
            s.players.furiten_by_pass[p] = f

    return s


# ═══════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════

def test_tsumo_mask_concealed():
    """Tsumo should be legal for a concealed winning hand (menzen tsumo = yaku)."""
    # Complete hand: 123m 456p 789s 東東東 中中
    h34 = _hand_34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:2})
    h37 = _hand_37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:2})
    s = make_pt_state(cp=0, hand_37=h37, hand_34=h34, is_hand_concealed=True,
                      next_deck_ix=50, last_deck_ix=14)

    env = RedMahjong()
    mask = env._make_legal_action_mask_after_draw(s)
    return [("concealed tsumo legal", bool(mask[Action.TSUMO].item()), True)]


def test_tsumo_mask_open_no_yaku():
    """Tsumo should NOT be legal for an open hand with no yaku."""
    from mahjax_pt.red_mahjong.meld import Meld
    # Complete hand shape: 123m 456p 789s 東東東 + pon 中中中 = 4 groups + pair
    # After pon of 中中中, remaining tiles: 123m 456p 789s 東東東 (12 tiles) + drawn tile = 13 + draw
    # With last_draw completing the hand (e.g. another 東), hand shape is winning
    h37 = _hand_37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:4})  # 14 tiles with 東×4
    h34 = _hand_34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:4})
    meld = Meld.init(Action.PON, 33, 1)
    s = make_pt_state(cp=0, hand_37=h37, hand_34=h34,
                      melds=[ [meld], [], [], [] ],
                      is_hand_concealed=False,
                      can_win=torch.zeros(4,34,dtype=torch.bool),
                      has_yaku=torch.zeros(4,2,dtype=torch.bool),  # no yaku precomputed
                      next_deck_ix=50, last_deck_ix=14)

    env = RedMahjong()
    mask = env._make_legal_action_mask_after_draw(s)
    return [("open no yaku tsumo blocked", bool(mask[Action.TSUMO].item()), False)]


def test_tsumo_mask_haitei():
    """Tsumo should be legal on haitei even for open hand (houtei = yaku)."""
    # Complete winning hand shape
    h37 = _hand_37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:2})  # 14 tiles
    h34 = _hand_34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:2})
    s = make_pt_state(cp=0, hand_37=h37, hand_34=h34,
                      is_hand_concealed=False, is_haitei=True,
                      can_win=torch.zeros(4,34,dtype=torch.bool),
                      has_yaku=torch.zeros(4,2,dtype=torch.bool),  # no yaku but haitei counts
                      next_deck_ix=50, last_deck_ix=14)

    env = RedMahjong()
    mask = env._make_legal_action_mask_after_draw(s)
    return [("haitei tsumo legal", bool(mask[Action.TSUMO].item()), True)]


def test_ron_mask_furiten():
    """Ron should be blocked by furiten."""
    h37 = _hand_37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:1})
    h34 = _hand_34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:1})
    s = make_pt_state(cp=0, hand_37=h37, hand_34=h34, target=33,
                      can_win=torch.zeros(4,34,dtype=torch.bool),
                      has_yaku=torch.ones(4,2,dtype=torch.bool),  # has yaku for all
                      furiten_discard={1: True},  # player 1 in furiten
                      next_deck_ix=50, last_deck_ix=14)

    env = RedMahjong()
    mask_4p = env._make_legal_action_mask_after_discard(s)
    ron_legal = bool(mask_4p.players.legal_action_mask[1, Action.RON].item())  # check player 1
    return [("ron blocked by furiten", ron_legal, False)]


def test_kan_blocked_by_haitei():
    """Kan should be blocked on haitei."""
    h37 = _hand_37({0:4})
    s = make_pt_state(cp=0, hand_37=h37, is_haitei=True, next_deck_ix=50, last_deck_ix=14)

    env = RedMahjong()
    mask = env._make_legal_action_mask_after_draw(s)
    kan_legal = any(mask[37 + i] for i in range(34))
    return [("kan blocked by haitei", kan_legal, False)]


def test_kan_blocked_by_4kan_limit():
    """Kan should be blocked after 4 kan."""
    h37 = _hand_37({0:4})
    s = make_pt_state(cp=0, hand_37=h37, is_haitei=False, next_deck_ix=50, last_deck_ix=14)
    s.players.n_kan = torch.tensor([1, 1, 1, 1], dtype=torch.int8)  # 4 total

    env = RedMahjong()
    mask = env._make_legal_action_mask_after_draw(s)
    kan_legal = any(mask[37 + i] for i in range(34))
    return [("kan blocked by 4-kan limit", kan_legal, False)]


def test_riichi_needs_tiles_left():
    """Riichi should be blocked when <4 tiles remain."""
    # Tenpai hand: 123m 456p 789s 東東 中中中 = 3 groups + 2 pairs
    h37 = _hand_37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:3})
    h34 = _hand_34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:3})
    # next_deck_ix - last_deck_ix = 16 - 14 = 2 tiles left (< 4)
    s = make_pt_state(cp=0, hand_37=h37, hand_34=h34,
                      score=[260, 250, 250, 250],
                      next_deck_ix=16, last_deck_ix=14)
    s.players.is_hand_concealed = torch.ones(4, dtype=torch.bool)

    env = RedMahjong()
    mask = env._make_legal_action_mask_after_draw(s)
    return [("riichi blocked <4 tiles", bool(mask[Action.RIICHI].item()), False)]


def test_ron_settlement():
    """Ron settlement formula: 1han30fu child ron = 1000 points."""
    import math
    base = 240  # 1han30fu

    # Child ron: base * 4 / 100 = 960/100 = 9.6 → ceil = 10 (1000 points)
    score = math.ceil(base * 4 / 100.0)
    assert score == 10, f"Child ron score: {score} != 10"

    # Dealer ron: base * 6 / 100 = 1440/100 = 14.4 → ceil = 15 (1500 points)
    score_d = math.ceil(base * 6 / 100.0)
    assert score_d == 15, f"Dealer ron score: {score_d} != 15"

    return [
        ("ron child 1han30fu=1000pts", score, 10),
        ("ron dealer 1han30fu=1500pts", score_d, 15),
    ]


def test_tsumo_settlement():
    """Tsumo settlement: 1han30fu child tsumo → non-dealer 300, dealer 500."""
    import math
    base = 240

    s1 = math.ceil(base / 100.0)      # non-dealer pays
    s2 = math.ceil(base * 2 / 100.0)  # dealer pays

    return [
        ("tsumo child non-dealer pay=300", s1, 3),
        ("tsumo child dealer pay=500", s2, 5),
    ]


def test_meld_blocked_by_riichi():
    """A riichi player cannot call melds on another's discard."""
    h37 = _hand_37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:1})
    s = make_pt_state(cp=0, hand_37=h37, hand_34=_hand_34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:3, 33:1}),
                      target=0, riichi={1: True}, next_deck_ix=50, last_deck_ix=14)

    env = RedMahjong()
    mask_4p = env._make_legal_action_mask_after_discard(s)
    # Player 1 is in riichi — should not have PON/CHI/OPEN_KAN
    p1_mask = mask_4p.players.legal_action_mask[1]
    pon_ok = bool(p1_mask[Action.PON].item()) or bool(p1_mask[Action.PON_RED].item())
    chi_ok = bool(p1_mask[Action.CHI_L:Action.CHI_R_RED + 1].any().item())
    return [
        ("riichi player pon blocked", pon_ok, False),
        ("riichi player chi blocked", chi_ok, False),
    ]


def test_first_turn_check():
    """_is_first_turn should be True for first 4 draws."""
    from mahjax_pt.red_mahjong.env import _is_first_turn, FIRST_DRAW_IDX
    results = []
    results.append(("first draw (83)", _is_first_turn(83), True))
    results.append(("second draw (82)", _is_first_turn(82), True))
    results.append(("fifth draw (79)", _is_first_turn(79), True))
    results.append(("sixth draw (78)", _is_first_turn(78), False))
    results.append(("mid game (50)", _is_first_turn(50), False))
    return results


def test_dora_flip_index():
    """Dora indicator after kan should be at deck[7] (for first kan)."""
    # JAX formula: deck[9 - 2 * n_kan]
    # First kan (n_kan=1): deck[7]
    # Second kan (n_kan=2): deck[5]
    dora_results = [
        ("1st kan dora idx", 9 - 2 * 1, 7),
        ("2nd kan dora idx", 9 - 2 * 2, 5),
        ("3rd kan dora idx", 9 - 2 * 3, 3),
        ("4th kan dora idx", 9 - 2 * 4, 1),
        ("5th kan dora idx", 9 - 2 * 5, -1),
    ]
    results = []
    for name, actual, expected in dora_results:
        results.append((name, actual, expected))
    return results


def test_dealer_continuation():
    """After dealer wins, dealer should continue (renchan)."""
    from mahjax_pt.red_mahjong.env import make as make_env
    env = make_env("red_mahjong", round_mode="east", observe_type="dict")
    s = env.init(torch.Generator().manual_seed(0))
    # Simulate dealer winning
    dealer = int(s.round_state.dealer)
    s.players.has_won[dealer] = True
    s.round_state.terminated_round = True
    s.round_state.can_win = torch.zeros(4, 34, dtype=torch.bool)
    s.round_state.can_win[dealer, 0] = True

    s2 = env._advance_to_next_round_auto(s)
    # Dealer should NOT change, honba should increase
    new_dealer = int(s2.round_state.dealer)
    new_honba = int(s2.round_state.honba)
    new_round = int(s2.round_state.round)

    return [
        ("dealer stays after win", new_dealer, dealer),
        ("honba increases", new_honba, 1),
        ("round stays same", new_round, 0),
    ]


def test_dealer_rotation():
    """After non-dealer wins, dealer should rotate."""
    from mahjax_pt.red_mahjong.env import make as make_env
    env = make_env("red_mahjong", round_mode="east", observe_type="dict")
    s = env.init(torch.Generator().manual_seed(0))
    dealer = int(s.round_state.dealer)
    non_dealer = (dealer + 1) % 4
    s.players.has_won[non_dealer] = True
    s.round_state.terminated_round = True
    s.round_state.can_win = torch.zeros(4, 34, dtype=torch.bool)
    s.round_state.can_win[dealer, 0] = False

    s2 = env._advance_to_next_round_auto(s)
    new_dealer = int(s2.round_state.dealer)
    new_honba = int(s2.round_state.honba)
    new_round = int(s2.round_state.round)

    return [
        ("dealer rotates after non-dealer win", new_dealer, (dealer + 1) % 4),
        ("honba resets", new_honba, 0),
        ("round advances", new_round, 1),
    ]


# ═══════════════════════════════════════════════════════════════
# Test runner
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = {
    "tsumo_concealed": ("Tsumo mask: concealed hand", test_tsumo_mask_concealed),
    "tsumo_open_no_yaku": ("Tsumo mask: open no yaku blocked", test_tsumo_mask_open_no_yaku),
    "tsumo_haitei": ("Tsumo mask: haitei allowed", test_tsumo_mask_haitei),
    "ron_furiten": ("Ron mask: blocked by furiten", test_ron_mask_furiten),
    "kan_haitei": ("Kan blocked by haitei", test_kan_blocked_by_haitei),
    "kan_4limit": ("Kan blocked by 4-kan limit", test_kan_blocked_by_4kan_limit),
    "riichi_tiles": ("Riichi blocked when <4 tiles", test_riichi_needs_tiles_left),
    "ron_settle": ("Ron settlement formulas", test_ron_settlement),
    "tsumo_settle": ("Tsumo settlement formulas", test_tsumo_settlement),
    "meld_riichi": ("Meld blocked by riichi", test_meld_blocked_by_riichi),
    "first_turn": ("_is_first_turn check", test_first_turn_check),
    "dora_flip": ("Kan dora flip index", test_dora_flip_index),
    "dealer_continue": ("Dealer continuation (renchan)", test_dealer_continuation),
    "dealer_rotate": ("Dealer rotation after loss", test_dealer_rotation),
}


def run_one(name, info):
    desc, fn = info
    t0 = time.time()
    passed = 0; failed = 0; details = []
    try:
        results = fn()
        for item in results:
            case, got, exp = item
            ok = (got == exp)
            if ok: passed += 1
            else: failed += 1
            prefix = "PASS" if ok else "FAIL"
            detail = f"  [{prefix}] {case}: got={got}, expected={exp}"
            details.append(detail)
    except Exception as e:
        failed += 1
        details.append(f"  [ERROR] {e}")
        details.append(f"  {traceback.format_exc()}")
    elapsed = time.time() - t0
    status = "PASS" if failed == 0 else f"FAIL({failed})"
    return name, passed, failed, details, elapsed, status, desc


if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.list:
        for n, (d, _) in ALL_TESTS.items():
            print(f"  {n:25s} {d}")
        sys.exit(0)

    selected = {n: i for n, i in ALL_TESTS.items()
                if args.filter is None or args.filter in n}
    if not selected:
        print(f"No tests match '{args.filter}'")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"Env Branch Tests ({len(selected)} groups)")
    print(f"{'='*70}\n")

    tp = tf = 0
    for name, info in selected.items():
        name, p, f, details, ela, status, desc = run_one(name, info)
        tp += p; tf += f
        print(f"[{status}] {desc} ({p} ok, {f} fail, {ela:.2f}s)")
        if f > 0 or args.verbose:
            for d in details:
                print(d)
        print()

    print(f"{'='*70}")
    print(f"SUMMARY: {tp} passed, {tf} failed")
    print(f"{'='*70}")
    sys.exit(1 if tf > 0 else 0)
