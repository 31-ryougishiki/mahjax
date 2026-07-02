#!/usr/bin/env python3
"""
JAX vs PyTorch env function comparison.

For each test, construct equivalent states in both frameworks,
call the same logic function, and compare outputs numerically.
"""
import time, traceback, math
import numpy as np
import jax, jax.numpy as jnp
import mahjax
from mahjax.red_mahjong.env import (_make_state, _replace_state,
    _make_legal_action_mask_after_draw as jax_mask_draw,
    _make_legal_action_mask_after_discard as jax_mask_discard,
    yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile)
from mahjax.red_mahjong.action import Action as JAction
from mahjax.red_mahjong.state import EnvState as JState, PlayerStateArrays as JP, RoundState as JR
from mahjax.red_mahjong.hand import Hand as JHand
from mahjax.red_mahjong.tile import Tile as JTile
from mahjax.red_mahjong.constants import TILE_RANGE as J_TILE_RANGE, NUM_TILE_TYPES as J_NUM_TILE_TYPES

import torch
from mahjax_pt.red_mahjong.env import RedMahjong
from mahjax_pt.red_mahjong.state import EnvState, PlayerStateArrays, RoundState
from mahjax_pt.red_mahjong.tile import Tile
from mahjax_pt.red_mahjong.hand import Hand
from mahjax_pt.red_mahjong.action import Action
from mahjax_pt.red_mahjong.constants import NUM_TILE_TYPES_WITH_RED

env_pt = RedMahjong()


def _jax_hand34(d):
    h = jnp.zeros(34, dtype=jnp.int8)
    for k, v in d.items():
        h = h.at[k].set(v)
    return h


def _jax_hand37(d):
    h = jnp.zeros(37, dtype=jnp.int8)
    for k, v in d.items():
        h = h.at[k].set(v)
    return h


def _pt_hand34(d):
    h = torch.zeros(34, dtype=torch.int8)
    for k, v in d.items():
        h[k] = v
    return h


def _pt_hand37(d):
    h = torch.zeros(37, dtype=torch.int8)
    for k, v in d.items():
        h[k] = v
    return h


def build_jax_mask_draw_state(cp, hand37, hand34, riichi=False, is_haitei=False,
                               can_after_kan=False, is_concealed=True,
                               has_yaku_tsumo=False, can_win_tile=0,
                               n_kan_total=0, next_deck_ix=50, last_deck_ix=14,
                               pon_slots=None):
    """Build a JAX state suitable for _make_legal_action_mask_after_draw."""
    h = jnp.zeros((4, 34), dtype=jnp.int8)
    h = h.at[cp].set(hand34)
    h37 = jnp.zeros((4, 37), dtype=jnp.int8)
    h37 = h37.at[cp].set(hand37)
    hw37 = h37

    can_win = jnp.zeros((4, 34), dtype=jnp.bool_)
    can_win = can_win.at[cp, can_win_tile].set(True)

    has_yaku = jnp.zeros((4, 2), dtype=jnp.bool_)
    has_yaku = has_yaku.at[cp, 0].set(has_yaku_tsumo)
    has_yaku = has_yaku.at[cp, 1].set(True)  # ron yaku

    riichi_arr = jnp.array([riichi] * 4, dtype=jnp.bool_)
    conc_arr = jnp.array([is_concealed] * 4, dtype=jnp.bool_)
    n_kan = jnp.array([n_kan_total // 4] * 4, dtype=jnp.int8)
    pon_arr = jnp.zeros((4, 34), dtype=jnp.int32)
    if pon_slots:
        for p, tile in pon_slots:
            pon_arr = pon_arr.at[p, tile].set(1)

    return _make_state(
        current_player=jnp.int8(cp),
        dealer=jnp.int8(0),
        hand=h,
        hand_with_red=hw37,
        can_win=can_win,
        has_yaku=has_yaku,
        riichi=riichi_arr,
        is_hand_concealed=conc_arr,
        pon=pon_arr,
        n_kan=n_kan,
        is_haitei=jnp.bool_(is_haitei),
        can_after_kan=jnp.bool_(can_after_kan),
        next_deck_ix=jnp.int32(next_deck_ix),
        last_deck_ix=jnp.int8(last_deck_ix),
        score=jnp.full((4,), 250, dtype=jnp.int32),
        seat_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
    )


def build_pt_mask_draw_state(cp, hand37, hand34, riichi=False, is_haitei=False,
                              can_after_kan=False, is_concealed=True,
                              has_yaku_tsumo=False, n_kan_total=0,
                              next_deck_ix=50, last_deck_ix=14):
    """Build equivalent PT state."""
    s = EnvState()
    s.current_player = cp
    s.round_state.dealer = 0
    s.round_state.is_haitei = is_haitei
    s.round_state.can_after_kan = can_after_kan
    s.round_state.next_deck_ix = next_deck_ix
    s.round_state.last_deck_ix = last_deck_ix
    s.round_state.score = torch.full((4,), 250, dtype=torch.int32)
    s.round_state.seat_wind = torch.tensor([0, 1, 2, 3], dtype=torch.int8)

    for p in range(4):
        s.players.hand_with_red[p] = hand37
        s.players.hand[p] = hand34
        s.players.riichi[p] = riichi
        s.players.is_hand_concealed[p] = is_concealed
        s.players.n_kan[p] = n_kan_total // 4
    s.players.n_kan[0] += n_kan_total % 4  # handle remainder

    if has_yaku_tsumo:
        s.players.has_yaku[cp, 0] = True
    s.players.has_yaku[cp, 1] = True  # ron

    return s


def build_jax_mask_discard_state(cp, hand34, target_tile, riichi_flags=None,
                                  furiten_discard=None, furiten_pass=None,
                                  has_yaku_ron=None, is_haitei=False,
                                  n_kan_total=0, next_deck_ix=50, last_deck_ix=14):
    """Build JAX state using real env.init + _replace_state for reliability."""
    jenv = mahjax.make("red_mahjong", round_mode="single", observe_type="dict")
    rng = jax.random.PRNGKey(0)
    base = jenv.init(rng)

    # Build the exact hand arrays we want
    h34 = jnp.tile(hand34[None, :], (4, 1))  # (4, 34)
    h37 = jnp.zeros((4, 37), dtype=jnp.int8)
    for t in range(34):
        h37 = h37.at[:, t].set(hand34[t])

    can_win = jnp.zeros((4, 34), dtype=jnp.bool_)
    can_win = can_win.at[:, int(JTile.to_tile_type(target_tile))].set(True)

    has_yaku = jnp.zeros((4, 2), dtype=jnp.bool_)
    if has_yaku_ron is not None:
        for p, v in has_yaku_ron.items():
            has_yaku = has_yaku.at[p, 0].set(v)

    riichi_arr = jnp.zeros(4, dtype=jnp.bool_)
    if riichi_flags:
        for p, v in riichi_flags.items():
            riichi_arr = riichi_arr.at[p].set(v)

    furiten_d = jnp.zeros(4, dtype=jnp.bool_)
    if furiten_discard:
        for p, v in furiten_discard.items():
            furiten_d = furiten_d.at[p].set(v)

    furiten_p = jnp.zeros(4, dtype=jnp.bool_)
    if furiten_pass:
        for p, v in furiten_pass.items():
            furiten_p = furiten_p.at[p].set(v)

    return _replace_state(base,
        current_player=jnp.int8(cp),
        hand=h34, hand_with_red=h37,
        can_win=can_win, has_yaku=has_yaku,
        riichi=riichi_arr,
        furiten_by_discard=furiten_d, furiten_by_pass=furiten_p,
        n_kan=jnp.full((4,), int(n_kan_total // 4), dtype=jnp.int8),
        is_haitei=jnp.bool_(is_haitei),
        next_deck_ix=jnp.int32(next_deck_ix),
        last_deck_ix=jnp.int8(last_deck_ix),
    )


def build_pt_mask_discard_state(cp, hand37, hand34, target_tile, riichi_flags=None,
                                 furiten_discard=None, furiten_pass=None,
                                 has_yaku_ron=None, is_haitei=False,
                                 n_kan_total=0, next_deck_ix=50, last_deck_ix=14):
    """Build equivalent PT state."""
    s = EnvState()
    s.current_player = cp
    s.round_state.dealer = 0
    s.round_state.target = target_tile
    s.round_state.is_haitei = is_haitei
    s.round_state.next_deck_ix = next_deck_ix
    s.round_state.last_deck_ix = last_deck_ix
    s.round_state.score = torch.full((4,), 250, dtype=torch.int32)
    s.round_state.seat_wind = torch.tensor([0, 1, 2, 3], dtype=torch.int8)

    for p in range(4):
        s.players.hand_with_red[p] = hand37
        s.players.hand[p] = hand34
        s.players.n_kan[p] = n_kan_total // 4

    if riichi_flags:
        for p, v in riichi_flags.items():
            s.players.riichi[p] = v
    if furiten_discard:
        for p, v in furiten_discard.items():
            s.players.furiten_by_discard[p] = v
    if furiten_pass:
        for p, v in furiten_pass.items():
            s.players.furiten_by_pass[p] = v
    if has_yaku_ron is not None:
        for p, v in has_yaku_ron.items():
            s.players.has_yaku[p, 0] = v

    return s


# ═══════════════════════════════════════════════════════════════
# JAX vs PT comparison tests
# ═══════════════════════════════════════════════════════════════

def jax_vs_pt_mask_draw_normal():
    """Normal draw mask: JAX vs PT."""
    h34 = _pt_hand34({0:1,1:1,2:1, 10:1,11:1,12:1, 20:1,21:1,22:1, 28:1,29:1,30:1, 31:1, 32:1})
    h37 = _pt_hand37({0:1,1:1,2:1, 10:1,11:1,12:1, 20:1,21:1,22:1, 28:1,29:1,30:1, 31:1, 32:1})

    # JAX
    jax_h34 = _jax_hand34({0:1,1:1,2:1, 10:1,11:1,12:1, 20:1,21:1,22:1, 28:1,29:1,30:1, 31:1, 32:1})
    jax_h37 = _jax_hand37({0:1,1:1,2:1, 10:1,11:1,12:1, 20:1,21:1,22:1, 28:1,29:1,30:1, 31:1, 32:1})
    jax_s = build_jax_mask_draw_state(0, jax_h37, jax_h34)
    jax_mask = jax_mask_draw(jax_s, jax_s.players.hand_with_red, 0,
                              jnp.int8(5), None)  # new_tile=5

    # PT
    pt_s = build_pt_mask_draw_state(0, h37, h34)
    pt_s.round_state.last_draw = 5
    pt_mask = env_pt._make_legal_action_mask_after_draw(pt_s)

    jax_arr = np.array(jax_mask)
    pt_arr = pt_mask.numpy()
    matches = (jax_arr == pt_arr).sum()
    total = len(jax_arr)

    results = [("mask draw normal match", matches, total)]
    # Specific checks
    results.append(("tsumogiri in both", jax_arr[71]==pt_arr[71], True))
    results.append(("discard counts match", jax_arr[:37].sum(), pt_arr[:37].sum()))
    return results


def jax_vs_pt_mask_discard_ron():
    """Discard mask: ron check. JAX vs PT."""
    # Hand that can ron on target=5: 123m 456p 789s 東東 中中 + 5m
    h34 = _pt_hand34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    h37 = _pt_hand37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    jax_h34 = _jax_hand34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    jax_h37 = _jax_hand37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})

    # Player 0 discards, player 1 can ron
    jax_s = build_jax_mask_discard_state(0, jax_h34, 4,
                                          has_yaku_ron={1: True})
    jax_mask = jax_mask_discard(jax_s, jax_s.players.hand_with_red, 1, 4)

    pt_s = build_pt_mask_discard_state(0, h37, h34, 4, has_yaku_ron={1: True})
    pt_mask_4p = env_pt._make_legal_action_mask_after_discard(pt_s)
    pt_mask = pt_mask_4p.players.legal_action_mask[1]

    jax_arr = np.array(jax_mask)
    pt_arr = pt_mask.numpy()
    matches = (jax_arr == pt_arr).sum()
    total = len(jax_arr)

    results = [("mask discard ron match", matches, total)]
    results.append(("ron in both", bool(jax_arr[74]) == bool(pt_arr[74]), True))
    return results


def jax_vs_pt_mask_discard_furiten():
    """Discard mask: furiten blocks ron."""
    h34 = _pt_hand34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    h37 = _pt_hand37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    jax_h34 = _jax_hand34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    jax_h37 = _jax_hand37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})

    # Player 1 has yaku but is in furiten
    jax_s = build_jax_mask_discard_state(0, jax_h34, 4,
                                          has_yaku_ron={1: True},
                                          furiten_discard={1: True})
    jax_mask = jax_mask_discard(jax_s, jax_s.players.hand_with_red, 1, 4)

    pt_s = build_pt_mask_discard_state(0, h37, h34, 4,
                                        has_yaku_ron={1: True},
                                        furiten_discard={1: True})
    pt_mask_4p = env_pt._make_legal_action_mask_after_discard(pt_s)
    pt_mask = pt_mask_4p.players.legal_action_mask[1]

    jax_arr = np.array(jax_mask)
    pt_arr = pt_mask.numpy()
    matches = (jax_arr == pt_arr).sum()
    total = len(jax_arr)

    return [
        ("mask furiten match", matches, total),
        ("ron blocked in both", bool(jax_arr[74]) == False and bool(pt_arr[74]) == False, True),
    ]


def jax_vs_pt_mask_discard_meld_riichi():
    """Discard mask: riichi player cannot meld."""
    h34 = _pt_hand34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    h37 = _pt_hand37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    jax_h34 = _jax_hand34({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})
    jax_h37 = _jax_hand37({0:1,1:1,2:1, 9:1,10:1,11:1, 24:1,25:1,26:1, 27:2, 33:2})

    # Player 1 is in riichi
    jax_s = build_jax_mask_discard_state(0, jax_h34, 4, riichi_flags={1: True})
    jax_mask = jax_mask_discard(jax_s, jax_s.players.hand_with_red, 1, 4)

    pt_s = build_pt_mask_discard_state(0, h37, h34, 4, riichi_flags={1: True})
    pt_mask_4p = env_pt._make_legal_action_mask_after_discard(pt_s)
    pt_mask = pt_mask_4p.players.legal_action_mask[1]

    jax_arr = np.array(jax_mask)
    pt_arr = pt_mask.numpy()
    matches = (jax_arr == pt_arr).sum()
    total = len(jax_arr)

    return [
        ("mask riichi meld match", matches, total),
        ("pon blocked in both", bool(jax_arr[75]) == False and bool(pt_arr[75]) == False, True),
        ("chi blocked in both", not bool(jax_arr[78:84].any()) and not bool(pt_arr[78:84].any()), True),
    ]


def jax_vs_pt_mask_draw_kan():
    """Draw mask: kan availability. JAX vs PT."""
    # 4 copies of tile 0
    h34 = _pt_hand34({0:4})
    h37 = _pt_hand37({0:4})
    jax_h34 = _jax_hand34({0:4})
    jax_h37 = _jax_hand37({0:4})

    jax_s = build_jax_mask_draw_state(0, jax_h37, jax_h34)
    jax_mask = jax_mask_draw(jax_s, jax_s.players.hand_with_red, 0, jnp.int8(1), None)

    pt_s = build_pt_mask_draw_state(0, h37, h34)
    pt_s.round_state.last_draw = 1
    pt_mask = env_pt._make_legal_action_mask_after_draw(pt_s)

    jax_arr = np.array(jax_mask)
    pt_arr = pt_mask.numpy()
    matches = (jax_arr == pt_arr).sum()
    total = len(jax_arr)

    return [
        ("mask kan match", matches, total),
        ("closed kan in both (37)", jax_arr[37]==pt_arr[37], True),
    ]


def jax_vs_pt_mask_draw_kan_haitei():
    """Draw mask: kan blocked on haitei."""
    h34 = _pt_hand34({0:4})
    h37 = _pt_hand37({0:4})
    jax_h34 = _jax_hand34({0:4})
    jax_h37 = _jax_hand37({0:4})

    jax_s = build_jax_mask_draw_state(0, jax_h37, jax_h34, is_haitei=True)
    jax_mask = jax_mask_draw(jax_s, jax_s.players.hand_with_red, 0, jnp.int8(1), None)

    pt_s = build_pt_mask_draw_state(0, h37, h34, is_haitei=True)
    pt_s.round_state.last_draw = 1
    pt_mask = env_pt._make_legal_action_mask_after_draw(pt_s)

    jax_arr = np.array(jax_mask)
    pt_arr = pt_mask.numpy()
    matches = (jax_arr == pt_arr).sum()
    return [
        ("mask kan haitei match", matches, len(jax_arr)),
        ("kan blocked in both", not jax_arr[37:71].any() and not pt_arr[37:71].any(), True),
    ]


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = {
    "mask_draw_normal": ("Draw mask: normal (JAX vs PT)", jax_vs_pt_mask_draw_normal),
    "mask_discard_ron": ("Discard mask: ron (JAX vs PT)", jax_vs_pt_mask_discard_ron),
    "mask_discard_furiten": ("Discard mask: furiten (JAX vs PT)", jax_vs_pt_mask_discard_furiten),
    "mask_discard_meld_riichi": ("Discard mask: riichi blocks meld (JAX vs PT)", jax_vs_pt_mask_discard_meld_riichi),
    "mask_draw_kan": ("Draw mask: kan (JAX vs PT)", jax_vs_pt_mask_draw_kan),
    "mask_draw_kan_haitei": ("Draw mask: kan+haitei (JAX vs PT)", jax_vs_pt_mask_draw_kan_haitei),
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=None)
    args = parser.parse_args()

    selected = {n: i for n, i in ALL_TESTS.items()
                if args.filter is None or args.filter in n}
    print(f"JAX vs PT Comparison ({len(selected)} tests)\n")
    print(f"{'='*60}")

    tp = tf = 0
    for name, (desc, fn) in selected.items():
        t0 = time.time()
        try:
            results = fn()
            p = sum(1 for _, g, e in results if g == e)
            f = len(results) - p
            tp += p; tf += f
            status = "PASS" if f == 0 else f"FAIL({f})"
            print(f"[{status}] {desc} ({p}/{p+f} ok, {time.time()-t0:.2f}s)")
            if f > 0:
                for item in results:
                    if item[1] != item[2]:
                        print(f"  FAIL {item[0]}: got={item[1]}, expected={item[2]}")
        except Exception as e:
            tf += 1
            print(f"[ERROR] {desc}: {e}")

    print(f"{'='*60}")
    print(f"SUMMARY: {tp} passed, {tf} failed")
    exit(1 if tf > 0 else 0)
