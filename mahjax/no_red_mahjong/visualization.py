from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import jax.numpy as jnp

from mahjax.red_mahjong.state import State as RedState
from mahjax.red_mahjong.state import default_state as default_red_state
from mahjax.red_mahjong.visualization import (
    render_round_svg as render_red_round_svg,
)
from mahjax.red_mahjong.visualization import (
    render_svg_animation as render_red_svg_animation,
)

from .state import State

Language = Literal["ja", "en"]


def to_red_visual_state(state: State) -> RedState:
    rs = default_red_state()
    hand34 = state._hand.astype(jnp.int8)
    hand37 = jnp.zeros((4, 37), dtype=jnp.int8).at[:, :34].set(hand34)
    legal_4p = state._legal_action_mask_4p
    if legal_4p.shape[1] < rs.players.legal_action_mask.shape[1]:
        pad = rs.players.legal_action_mask.shape[1] - legal_4p.shape[1]
        legal_4p = jnp.pad(legal_4p, ((0, 0), (0, pad)), constant_values=False)
    legal_1p = state.legal_action_mask
    if legal_1p.shape[0] < rs.legal_action_mask.shape[0]:
        pad = rs.legal_action_mask.shape[0] - legal_1p.shape[0]
        legal_1p = jnp.pad(legal_1p, (0, pad), constant_values=False)
    return rs.replace(
        current_player=state.current_player,
        legal_action_mask=legal_1p,
        players=rs.players.replace(
            hand=hand34,
            hand_with_red=hand37,
            legal_action_mask=legal_4p,
            melds=state._melds,
            meld_counts=state._n_meld,
            river=state._river,
            discard_counts=state._n_river,
            riichi=state._riichi,
        ),
        round_state=rs.round_state.replace(
            round=state._round,
            honba=state._honba,
            dealer=state._dealer,
            score=state._score.astype(jnp.int32),
            next_deck_ix=state._next_deck_ix.astype(jnp.int32),
            last_deck_ix=state._last_deck_ix.astype(jnp.int8),
            dora_indicators=state._dora_indicators,
            last_draw=state._last_draw,
            last_player=state._last_player,
            target=state._target,
            terminated_round=state._terminated_round,
            seat_wind=state._seat_wind,
            init_wind=state._init_wind,
            round_limit=state._round_limit,
            kyotaku=state._kyotaku,
            can_after_kan=state._can_after_kan,
            kan_declared=state._kan_declared,
            can_robbing_kan=state._can_robbing_kan,
            is_haitei=state._is_haitei,
        ),
        rewards=state.rewards,
        terminated=state.terminated,
        truncated=state.truncated,
    )


def render_round_svg(
    state: State,
    show_all_hands: bool = True,
    visible_player: int = 0,
    language: Language = "ja",
) -> str:
    return render_red_round_svg(
        to_red_visual_state(state),
        show_all_hands=show_all_hands,
        visible_player=visible_player,
        language=language,
    )


def save_svg(
    state: State,
    filename: str | Path,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_round_svg(
            state,
            show_all_hands=show_all_hands,
            language=language,
        ),
        encoding="utf-8",
    )


def render_svg_animation(
    states: Sequence[State],
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> str:
    return render_red_svg_animation(
        [to_red_visual_state(state) for state in states],
        frame_duration_seconds=frame_duration_seconds,
        show_all_hands=show_all_hands,
        language=language,
    )


def save_svg_animation(
    states: Sequence[State],
    filename: str | Path,
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_svg_animation(
            states,
            frame_duration_seconds=frame_duration_seconds,
            show_all_hands=show_all_hands,
            language=language,
        ),
        encoding="utf-8",
    )
