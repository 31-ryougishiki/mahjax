import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.constants import FIRST_DRAW_IDX
from mahjax.red_mahjong.env import (
    _abortive_draw_normal,
    _draw,
    _init,
    _make_legal_action_mask_after_draw,
    _next_meld_player,
    _next_ron_player,
    _step,
)
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import Tile


def test_init_shape_and_first_draw_position() -> None:
    state = _init(jax.random.PRNGKey(1))
    assert bool(jnp.all(state.rewards == 0))
    assert not bool(state.terminated)
    assert not bool(state.truncated)
    assert int(state.round_state.next_deck_ix) == FIRST_DRAW_IDX - 1
    assert state.round_state.deck.shape == (136,)
    assert state.players.hand.shape == (4, Tile.NUM_TILE_TYPE)


def test_draw_decrements_deck_and_sets_last_draw() -> None:
    state = _init(jax.random.PRNGKey(2))
    before_ix = int(state.round_state.next_deck_ix)
    state = _draw(state)
    assert int(state.round_state.next_deck_ix) == before_ix - 1
    assert int(state.round_state.last_draw) >= 0


def test_mask_after_draw_has_discard_or_tsumogiri() -> None:
    state = _init(jax.random.PRNGKey(3))
    c_p = int(state.current_player)
    hand = state.players.hand_with_red
    new_tile = int(state.round_state.last_draw)
    mask = _make_legal_action_mask_after_draw(state, hand, c_p, new_tile)
    assert bool(mask[Action.TSUMOGIRI] | jnp.any(mask[: Tile.NUM_TILE_TYPE_WITH_RED]))


def test_tsumogiri_action_history_records_actual_tile_and_flag() -> None:
    state = _init(jax.random.PRNGKey(4))
    last_draw = int(state.round_state.last_draw)
    next_state = _step(state, jnp.int8(Action.TSUMOGIRI))

    assert int(next_state.round_state.action_history[0, 0]) == int(state.current_player)
    assert int(next_state.round_state.action_history[1, 0]) == last_draw
    assert int(next_state.round_state.action_history[2, 0]) == 1


def test_next_meld_player_prioritizes_ron_then_distance() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[0, Action.PON].set(True)
    legal = legal.at[1, Action.RON].set(True)
    legal = legal.at[3, Action.RON].set(True)
    nxt, can_any = _next_meld_player(legal, jnp.int8(0))
    assert bool(can_any)
    assert int(nxt) == 1


def test_next_ron_player_returns_closest() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[1, Action.RON].set(True)
    legal = legal.at[2, Action.RON].set(True)
    nxt, can_any = _next_ron_player(legal, jnp.int8(0))
    assert bool(can_any)
    assert int(nxt) == 1


def test_abortive_draw_payments_shape() -> None:
    state = default_state()
    state = state.replace(
        players=state.players.replace(
            can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True)
        )
    )
    next_state = _abortive_draw_normal(state)
    assert bool(next_state.round_state.terminated_round)
    assert next_state.rewards.shape == (4,)
