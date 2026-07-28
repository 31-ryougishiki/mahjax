#!/usr/bin/env python3
"""Record JAX golden data for hand-crafted Mahjong edge cases.

Each function below sets up a specific Mahjong scenario, runs it through
the JAX env, and saves the state as a .npz file.

Usage:
    python scripts/record_golden_special.py
    python scripts/record_golden_special.py --output ./mahjax_pt/tests/golden/env_special
    python scripts/record_golden_special.py --scene red_five_pon

Requires: JAX, mahjax
"""

import os, sys, argparse
import numpy as np

_JAX = None
_JaxEnv = None


def _init_jax():
    global _JAX, _JaxEnv
    if _JAX is not None:
        return
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_disable_jit', True)
    jax.config.update('jax_platform_name', 'cpu')
    from mahjax.red_mahjong.env import RedMahjong as _EnvCls
    _JAX = jax
    _JaxEnv = _EnvCls()


def _extract_state(state) -> dict:
    """Same field extraction as record_golden_env.py."""
    G = {}
    G["current_player"] = int(state.current_player)
    G["terminated"] = bool(state.terminated)
    G["truncated"] = bool(state.truncated)
    G["step_count"] = int(state.step_count)
    G["legal_action_mask"] = np.asarray(state.legal_action_mask)
    G["rewards"] = np.asarray(state.rewards, dtype=np.float32)

    pp = state.players
    G["players.hand"] = np.asarray(pp.hand)
    G["players.hand_with_red"] = np.asarray(pp.hand_with_red)
    G["players.melds"] = np.asarray(pp.melds)
    G["players.meld_counts"] = np.asarray(pp.meld_counts)
    G["players.discard_counts"] = np.asarray(pp.discard_counts)
    G["players.river"] = np.asarray(pp.river)
    G["players.riichi"] = np.asarray(pp.riichi)
    G["players.riichi_declared"] = np.asarray(pp.riichi_declared)
    G["players.has_won"] = np.asarray(pp.has_won)
    G["players.n_kan"] = np.asarray(pp.n_kan)
    G["players.has_yaku"] = np.asarray(pp.has_yaku)
    G["players.is_hand_concealed"] = np.asarray(pp.is_hand_concealed)
    G["players.furiten_by_discard"] = np.asarray(pp.furiten_by_discard)
    G["players.furiten_by_pass"] = np.asarray(pp.furiten_by_pass)
    G["players.ippatsu"] = np.asarray(pp.ippatsu)
    G["players.fan"] = np.asarray(pp.fan)
    G["players.fu"] = np.asarray(pp.fu)

    rs = state.round_state
    G["round_state.round"] = int(rs.round)
    G["round_state.dealer"] = int(rs.dealer)
    G["round_state.next_deck_ix"] = int(rs.next_deck_ix)
    G["round_state.last_deck_ix"] = int(rs.last_deck_ix)
    G["round_state.last_draw"] = int(rs.last_draw)
    G["round_state.last_player"] = int(rs.last_player)
    G["round_state.target"] = int(rs.target)
    G["round_state.draw_next"] = bool(rs.draw_next)
    G["round_state.is_haitei"] = bool(rs.is_haitei)
    G["round_state.is_abortive_draw_normal"] = bool(rs.is_abortive_draw_normal)
    G["round_state.terminated_round"] = bool(rs.terminated_round)
    G["round_state.honba"] = int(rs.honba)
    G["round_state.kyotaku"] = int(rs.kyotaku)
    G["round_state.n_kan_doras"] = int(rs.n_kan_doras)
    G["round_state.score"] = np.asarray(rs.score)
    G["round_state.deck"] = np.asarray(rs.deck)
    G["round_state.dora_indicators"] = np.asarray(rs.dora_indicators)

    return G


def _save_scene(output_dir, name, state, action=None, post_state=None):
    """Save a scene to .npz."""
    path = os.path.join(output_dir, f"{name}.npz")
    golden = _extract_state(state)
    if action is not None:
        golden["action"] = int(action)
    np.savez_compressed(path, **golden)
    print(f"  ✓ {name}")

    if post_state is not None:
        post_path = os.path.join(output_dir, f"{name}_post.npz")
        post_golden = _extract_state(post_state)
        np.savez_compressed(post_path, **post_golden)
        print(f"  ✓ {name}_post")


# ═══════════════════════════════════════════════════════════════════════
# Scene generators
# ═══════════════════════════════════════════════════════════════════════

def scene_red_five_pon():
    """PA18: Hand with 1 black 5m + 1 red 5m, can PON_RED."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.action import Action

    state = default_state()
    # Set up: player 0 has 1x5m + 1xred5m in hand
    pp = state.players
    hand = pp.hand_with_red.at[0, 4].set(1)     # 1x black 5m
    hand = hand.at[0, 34].set(1)                  # 1x red 5m
    hand = hand.at[0, 0].set(1)                   # filler tile
    state = state.replace(players=pp.replace(hand_with_red=hand))

    return state, None


def scene_red_five_chi():
    """PA19: Hand that can chi with red five substitution."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.tile import Tile

    state = default_state()
    pp = state.players
    hand = pp.hand_with_red.at[0, :].set(0)
    # 3p + 4p in hand (can chi 2p or 5p)
    hand = hand.at[0, 10].set(1)   # 3p
    hand = hand.at[0, 11].set(1)   # 4p
    hand = hand.at[0, 0].set(1)    # filler
    state = state.replace(players=pp.replace(hand_with_red=hand))

    return state, None


def scene_red_five_kan():
    """PA20: Hand with red-five kan scenarios."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    hand = pp.hand_with_red.at[0, :].set(0)
    # 3x black 5p + 1x red 5p → can closed_kan
    hand = hand.at[0, 13].set(3)
    hand = hand.at[0, 35].set(1)
    state = state.replace(players=pp.replace(hand_with_red=hand))

    return state, None


def scene_furiten_by_discard():
    """PA21a: Player discarded a tile they're now waiting on."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    # Set furiten flag on player 1
    state = state.replace(players=pp.replace(
        furiten_by_discard=pp.furiten_by_discard.at[1].set(True)))

    return state, None


def scene_furiten_by_pass():
    """PA21b: Player passed on a ron, now temporarily furiten."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    state = state.replace(players=pp.replace(
        furiten_by_pass=pp.furiten_by_pass.at[0].set(True)))

    return state, None


def scene_ippatsu():
    """PA22: Player in riichi with ippatsu flag still active."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    state = state.replace(players=pp.replace(
        riichi=pp.riichi.at[0].set(True),
        riichi_declared=pp.riichi_declared.at[0].set(True),
        ippatsu=pp.ippatsu.at[0].set(True)))

    return state, None


def scene_pao():
    """PA23: Player has fed 2 dragon pons and declared the 3rd as open kan."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.action import Action
    from mahjax.red_mahjong.meld import Meld

    state = default_state()
    pp = state.players
    melds = pp.melds
    melds = melds.at[0, 0].set(Meld.init(Action.PON, 31, 1))   # haku pon
    melds = melds.at[0, 1].set(Meld.init(Action.PON, 32, 3))   # hatsu pon
    melds = melds.at[0, 2].set(Meld.init(Action.OPEN_KAN, 33, 2))  # chun open kan
    state = state.replace(players=pp.replace(
        melds=melds, meld_counts=pp.meld_counts.at[0].set(3)))

    return state, None


def scene_double_ron():
    """PA24: Both player 1 and player 2 can ron the same discard."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.action import Action

    state = default_state()
    pp = state.players
    mask = pp.legal_action_mask
    mask = mask.at[1, Action.RON].set(True)
    mask = mask.at[2, Action.RON].set(True)
    state = state.replace(
        current_player=0,
        legal_action_mask=mask[0],
        players=pp.replace(legal_action_mask=mask))

    return state, None


def scene_nagashi_mangan():
    """PA25: Player has nagashi mangan (all discards are terminals/honors)."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    state = state.replace(players=pp.replace(
        has_nagashi_mangan=pp.has_nagashi_mangan.at[0].set(True)))

    return state, None


def scene_kyuushu():
    """PA26: Starting hand with 9+ unique terminals/honors."""
    _init_jax()
    import jax.numpy as jnp
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.tile import Tile

    state = default_state()
    pp = state.players
    # Set up a hand with 9 yaochu types
    hand = jnp.zeros(37, dtype=jnp.int8)
    yaochu = [0, 8, 9, 17, 18, 26, 27, 28, 29]  # 1m,9m,1p,9p,1s,9s,E,S,W
    for t in yaochu:
        hand = hand.at[t].set(1)
    hand = hand.at[0].set(2)  # pair of 1m
    # Fill remaining
    hand = hand.at[1].set(1)
    hand = hand.at[2].set(1)
    hand = hand.at[3].set(1)
    state = state.replace(players=pp.replace(hand_with_red=hand.at[0].set(hand[0])))

    return state, None


def scene_four_winds():
    """PA27: All 4 players discarded same wind in first turn."""
    _init_jax()
    import jax.numpy as jnp
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.tile import River

    state = default_state()
    # Simulate 4 players discarding East (27) in first turn
    river = state.players.river
    for p in range(4):
        river = River.add_discard(river, jnp.int8(27), jnp.int8(p),
                                   jnp.int8(0), False, False)
    state = state.replace(players=state.players.replace(river=river))

    return state, None


def scene_rinshan():
    """PA28: After declaring kan, the draw comes from the dead wall."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.action import Action
    from mahjax.red_mahjong.meld import Meld

    state = default_state()
    pp = state.players
    # Set up: player 0 just declared a closed kan, can_after_kan=True
    rs = state.round_state
    state = state.replace(
        current_player=0,
        round_state=type(rs)(**{**rs.__dict__,
            "can_after_kan": True,
            "kan_declared": True}))

    return state, None


def scene_haitei():
    """PA29a: Last tile from wall — is_haitei=True."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    rs = state.round_state
    # Set next_deck_ix just above last_deck_ix
    state = state.replace(
        round_state=type(rs)(**{**rs.__dict__,
            "next_deck_ix": 15,
            "is_haitei": True}))

    return state, None


def scene_houtei():
    """PA29b: Last discard scenario."""
    _init_jax()
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    rs = state.round_state
    state = state.replace(
        round_state=type(rs)(**{**rs.__dict__,
            "next_deck_ix": 14,
            "is_haitei": True,
            "draw_next": False}))

    return state, None


def scene_kokushi():
    """PA30a: 13-orphan hand (1 of each terminal + honor, plus a pair)."""
    _init_jax()
    import jax.numpy as jnp
    from mahjax.red_mahjong.state import default_state
    from mahjax.red_mahjong.hand import THIRTEEN_ORPHAN_IDX

    state = default_state()
    pp = state.players
    hand_34 = jnp.zeros(34, dtype=jnp.int8)
    for t in THIRTEEN_ORPHAN_IDX:
        hand_34 = hand_34.at[t].set(1)
    hand_34 = hand_34.at[0].set(2)  # pair of 1m
    state = state.replace(players=pp.replace(hand=hand_34.at[0].set(hand_34)))

    return state, None


def scene_seven_pairs():
    """PA30b: Chiitoitsu — 7 pairs."""
    _init_jax()
    import jax.numpy as jnp
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    hand_34 = jnp.zeros(34, dtype=jnp.int8)
    # 7 pairs at positions 0,1,2,3,4,5,6
    for t in range(7):
        hand_34 = hand_34.at[t].set(2)
    state = state.replace(players=pp.replace(hand=hand_34.at[0].set(hand_34)))

    return state, None


def scene_normal_win():
    """PA30c: Standard 4 groups + 1 pair (123m 456p 789s 東東東 中中)."""
    _init_jax()
    import jax.numpy as jnp
    from mahjax.red_mahjong.state import default_state

    state = default_state()
    pp = state.players
    hand_34 = jnp.zeros(34, dtype=jnp.int8)
    # 123m
    hand_34 = hand_34.at[0].set(1).at[1].set(1).at[2].set(1)
    # 456p
    hand_34 = hand_34.at[9].set(1).at[10].set(1).at[11].set(1)
    # 789s
    hand_34 = hand_34.at[24].set(1).at[25].set(1).at[26].set(1)
    # 東東東
    hand_34 = hand_34.at[27].set(3)
    # 中中
    hand_34 = hand_34.at[33].set(2)
    state = state.replace(players=pp.replace(hand=hand_34.at[0].set(hand_34)))

    return state, None


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

SCENES = {
    "red_five_pon": scene_red_five_pon,
    "red_five_chi": scene_red_five_chi,
    "red_five_kan": scene_red_five_kan,
    "furiten_by_discard": scene_furiten_by_discard,
    "furiten_by_pass": scene_furiten_by_pass,
    "ippatsu": scene_ippatsu,
    "pao": scene_pao,
    "double_ron": scene_double_ron,
    "nagashi_mangan": scene_nagashi_mangan,
    "kyuushu": scene_kyuushu,
    "four_winds": scene_four_winds,
    "rinshan": scene_rinshan,
    "haitei": scene_haitei,
    "houtei": scene_houtei,
    "kokushi": scene_kokushi,
    "seven_pairs": scene_seven_pairs,
    "normal_win": scene_normal_win,
}


def main():
    parser = argparse.ArgumentParser(description="Record special scene golden data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--scene", default=None,
                        help="Record a specific scene (default: all)")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "mahjax_pt", "tests",
            "golden", "env_special")
    args.output = os.path.abspath(args.output)
    os.makedirs(args.output, exist_ok=True)

    print(f"Recording special scene golden data to: {args.output}\n")

    scenes_to_run = [args.scene] if args.scene else SCENES.keys()

    for name in scenes_to_run:
        if name not in SCENES:
            print(f"  Unknown scene: {name}")
            continue
        try:
            state, extra = SCENES[name]()
            _save_scene(args.output, name, state,
                       action=extra[0] if extra else None,
                       post_state=extra[1] if extra and len(extra) > 1 else None)
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
