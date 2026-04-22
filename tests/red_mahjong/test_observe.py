import numpy as np
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import _observe_2D, _observe_dict
from mahjax.red_mahjong.meld import Meld
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import River, Tile


HAND_END = 7
GAME_END = HAND_END + 15
RIVER_END = GAME_END + 96 * 3
MELD_END = RIVER_END + 48


def _make_state():
    state = default_state()

    hand = state.players.hand
    hand = hand.at[0, 1].set(1)
    hand = hand.at[0, 2].set(2)
    hand = hand.at[0, 3].set(3)
    hand = hand.at[0, 4].set(4)
    hand = hand.at[1, 9].set(1)
    hand = hand.at[2, 18].set(1)
    hand = hand.at[3, 27].set(1)

    can_win = state.players.can_win
    can_win = can_win.at[0, jnp.array([1, 3], dtype=jnp.int32)].set(True)
    can_win = can_win.at[1, 9].set(True)

    river = state.players.river
    river = River.add_discard(river, jnp.int8(0), jnp.int8(0), jnp.int8(0), False, False)
    river = River.add_discard(river, jnp.int8(1), jnp.int8(1), jnp.int8(0), False, False)
    river = River.add_discard(river, jnp.int8(2), jnp.int8(2), jnp.int8(0), True, False)
    river = River.add_discard(river, jnp.int8(3), jnp.int8(3), jnp.int8(0), False, True)

    melds = state.players.melds
    melds = melds.at[0, 0].set(Meld.init(Action.PON, 1, 2))
    melds = melds.at[0, 1].set(Meld.init(Action.CHI_L, 2, 3))

    legal_action_mask = state.players.legal_action_mask
    legal_action_mask = legal_action_mask.at[0, 1].set(True)
    legal_action_mask = legal_action_mask.at[0, 37 + 5].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.OPEN_KAN].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.PON].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.CHI_L].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.RON].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.PASS].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.TSUMO].set(True)
    legal_action_mask = legal_action_mask.at[0, Action.RIICHI].set(True)

    action_history = state.round_state.action_history
    action_history = action_history.at[0, :3].set(jnp.array([0, 1, 3], dtype=jnp.int8))
    action_history = action_history.at[1, :3].set(jnp.array([10, 11, 12], dtype=jnp.int8))

    return state.replace(
        current_player=jnp.int8(0),
        legal_action_mask=legal_action_mask[0],
        players=state.players.replace(
            hand=hand,
            can_win=can_win,
            legal_action_mask=legal_action_mask,
            melds=melds,
            meld_counts=state.players.meld_counts.at[0].set(jnp.int8(2)),
            river=river,
            discard_counts=jnp.ones((4,), dtype=jnp.int8),
            riichi=state.players.riichi.at[0].set(True),
            furiten_by_discard=state.players.furiten_by_discard.at[0].set(True),
        ),
        round_state=state.round_state.replace(
            action_history=action_history,
            shanten_current_player=jnp.int8(2),
            round=jnp.int8(5),
            honba=jnp.int8(2),
            kyotaku=jnp.int8(4),
            last_draw=jnp.int8(Tile.RED_FIVE["p"]),
            score=jnp.array([260, 240, 270, 230], dtype=jnp.int32),
            dora_indicators=jnp.array([0, Tile.RED_FIVE["p"], -1, -1, -1], dtype=jnp.int8),
        ),
    )


def _one_hot(tile: int) -> np.ndarray:
    out = np.zeros((34,), dtype=np.float32)
    out[tile] = 1.0
    return out


def test_observe_dict_returns_relative_view() -> None:
    state = _make_state().replace(
        current_player=jnp.int8(1),
        legal_action_mask=_make_state().players.legal_action_mask[1],
        round_state=_make_state().round_state.replace(shanten_current_player=jnp.int8(1)),
    )

    obs = _observe_dict(state)

    np.testing.assert_array_equal(np.array(obs["hand"]), np.array([9] + [-1] * 13, dtype=np.int32))
    assert obs["last_draw"].item() == Tile.RED_FIVE["p"]
    np.testing.assert_array_equal(
        np.array(obs["scores"]),
        np.array([240, 270, 230, 260], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        np.array(obs["action_history"])[0, :3],
        np.array([3, 0, 2], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        np.array(obs["action_history"])[1, :3],
        np.array([10, 11, 12], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        np.array(obs["action_history"])[2, :4],
        np.array([-1, -1, -1, -1], dtype=np.int8),
    )
    np.testing.assert_array_equal(
        np.array(obs["dora_indicators"]),
        np.array([0, 13, -1, -1], dtype=np.int8),
    )
    assert obs["shanten_count"].item() == 1
    assert obs["furiten"].item() == 0
    assert obs["round"].item() == 5
    assert obs["honba"].item() == 2
    assert obs["kyotaku"].item() == 4
    assert obs["prevalent_wind"].item() == 1
    assert obs["seat_wind"].item() == 1


def test_observe_2d_hand_and_game_blocks_match_state() -> None:
    state = _make_state()
    obs = np.array(_observe_2D(state))

    expected_hand = np.zeros((7, 34), dtype=np.float32)
    expected_hand[0, [1, 2, 3, 4]] = 1.0
    expected_hand[1, [2, 3, 4]] = 1.0
    expected_hand[2, [3, 4]] = 1.0
    expected_hand[3, [4]] = 1.0
    expected_hand[4, [1, 3]] = 1.0
    expected_hand[5, :] = 1.0
    expected_hand[6, :] = 2.0 / 6.0

    expected_game = np.zeros((15, 34), dtype=np.float32)
    expected_game[0, :] = (260 + 250) / 1250.0
    expected_game[1, :] = (240 + 250) / 1250.0
    expected_game[2, :] = (270 + 250) / 1250.0
    expected_game[3, :] = (230 + 250) / 1250.0
    expected_game[4, :] = 1.0 / 3.0
    expected_game[5, :] = 5.0 / 7.0
    expected_game[6, :] = 2.0 / 10.0
    expected_game[7, :] = 4.0 / 10.0
    expected_game[8, :] = 0.0
    expected_game[9, :] = 1.0 / 3.0
    expected_game[10, :] = 1.0
    expected_game[11, 0] = 1.0
    assert obs.shape == (381, 34)
    np.testing.assert_allclose(obs[:HAND_END], expected_hand)
    np.testing.assert_allclose(obs[HAND_END:GAME_END], expected_game, atol=1e-6)


def test_observe_2d_river_meld_and_strategic_blocks_match_state() -> None:
    state = _make_state()
    obs = np.array(_observe_2D(state))

    np.testing.assert_array_equal(obs[GAME_END + 0], _one_hot(0))
    np.testing.assert_array_equal(obs[GAME_END + 24], _one_hot(1))
    np.testing.assert_array_equal(obs[GAME_END + 48], _one_hot(2))
    np.testing.assert_array_equal(obs[GAME_END + 72], _one_hot(3))
    np.testing.assert_array_equal(obs[GAME_END + 96 + 72], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(obs[GAME_END + 192 + 48], np.ones((34,), dtype=np.float32))

    meld_block = obs[RIVER_END:MELD_END]
    np.testing.assert_array_equal(meld_block[0], np.full((34,), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(meld_block[1], np.full((34,), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(meld_block[16], np.full((34,), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(meld_block[17], np.full((34,), 2.0, dtype=np.float32))
    np.testing.assert_allclose(
        meld_block[32],
        np.full((34,), Action.PON / Action.NUM_ACTION, dtype=np.float32),
    )
    np.testing.assert_allclose(
        meld_block[33],
        np.full((34,), Action.CHI_L / Action.NUM_ACTION, dtype=np.float32),
    )

    strategic_block = obs[MELD_END:]
    np.testing.assert_array_equal(strategic_block[0], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[1], np.zeros((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[7], _one_hot(3))
    np.testing.assert_array_equal(strategic_block[12], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[13], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[14], np.zeros((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[15], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[16], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[17], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[18], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[19], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[20], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[21], np.ones((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[22], np.zeros((34,), dtype=np.float32))


def test_observe_2d_reorders_player_axes_from_current_player() -> None:
    base_state = _make_state()
    state = base_state.replace(
        current_player=jnp.int8(1),
        legal_action_mask=base_state.players.legal_action_mask[1],
        round_state=base_state.round_state.replace(shanten_current_player=jnp.int8(1)),
    )

    obs = np.array(_observe_2D(state))
    strategic_block = obs[MELD_END:]

    np.testing.assert_array_equal(obs[GAME_END + 0], _one_hot(1))
    np.testing.assert_array_equal(obs[GAME_END + 24], _one_hot(2))
    np.testing.assert_array_equal(obs[GAME_END + 48], _one_hot(3))
    np.testing.assert_array_equal(obs[GAME_END + 72], _one_hot(0))
    np.testing.assert_array_equal(strategic_block[0], np.zeros((34,), dtype=np.float32))
    np.testing.assert_array_equal(strategic_block[3], np.ones((34,), dtype=np.float32))
