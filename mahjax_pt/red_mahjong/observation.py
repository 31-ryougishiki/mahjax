from typing import Dict
import torch

from .state import EnvState


def hand_counts_to_idx(counts: torch.Tensor, fill: int = -1, hand_size: int = 14) -> torch.Tensor:
    """Convert a 37-type histogram into a fixed-size list of tile indices.

    Args:
        counts: (37,) tensor — how many copies of each tile type
        fill: value for empty slots
        hand_size: output length (default 14)
    """
    counts = counts.to(torch.int32)
    # Each of 37 tile types can have 0..3 copies
    col = torch.arange(4, dtype=torch.int32).unsqueeze(0)  # (1, 4)
    mask = col < counts.unsqueeze(1)  # (37, 4) bool

    tile_ids = torch.arange(37, dtype=torch.int32).unsqueeze(1).repeat(1, 4)  # (37, 4)
    vals = torch.where(mask, tile_ids.to(torch.int32), torch.tensor(fill, dtype=torch.int32))
    vals = vals.reshape(-1)  # (148,)

    key = mask.reshape(-1).to(torch.int32)  # (148,)
    order = torch.argsort(-key, stable=True)
    sorted_vals = vals[order]

    out = sorted_vals[:hand_size].to(torch.int32)
    out = torch.where(out == fill, torch.tensor(fill, dtype=torch.int32), out)
    return out


def _observe_dict(state: EnvState) -> Dict:
    """Build the observation dictionary from the current player's perspective.

    Returns:
        hand: (14,) player's hand [0-36], -1 = empty
        last_draw: scalar — last drawn tile [0-36], -1 = no draw
        action_history: (3, 200) [player(relative), action(tile/id), tsumogiri]
        shanten_count: scalar (0-6)
        furiten: scalar bool
        scores: (4,) scores from current player's perspective
        round: scalar
        honba: scalar
        kyotaku: scalar
        prevalent_wind: scalar [0-3]
        seat_wind: scalar [0-3]
        dora_indicators: (5,) [0-36], -1 = no dora
    """
    c_p = state.current_player
    c_p_based_order = (torch.arange(4) + c_p) % 4

    # Hand features: 37-type histogram → sorted list of 14 tile indices
    hand_c_p_37 = state.players.hand_with_red[c_p]
    hand_c_p_14 = hand_counts_to_idx(hand_c_p_37)

    # Action history — translate player indices to relative
    player_history = state.round_state.action_history[0, :].to(torch.int32)  # (200,)
    valid_history = player_history >= 0
    relative_player_history = torch.remainder(player_history - c_p, 4).to(
        state.round_state.action_history.dtype
    )
    relative_player_history = torch.where(
        valid_history, relative_player_history, state.round_state.action_history[0, :]
    )
    action_history = state.round_state.action_history.clone()
    action_history[0, :] = relative_player_history

    # Game features
    shanten_c_p = state.round_state.shanten_current_player
    furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    scores = state.round_state.score[c_p_based_order]
    round_num = state.round_state.round
    honba = state.round_state.honba
    kyotaku = state.round_state.kyotaku
    prevalent_wind = int(state.round_state.round) // 4
    seat_wind = int(state.round_state.seat_wind[c_p])
    dora_indicators = state.round_state.dora_indicators[:5]  # (5,)

    return {
        "hand": hand_c_p_14,
        "last_draw": state.round_state.last_draw,
        "action_history": action_history,
        "shanten_count": shanten_c_p,
        "furiten": furiten,
        "scores": scores,
        "round": round_num,
        "honba": honba,
        "kyotaku": kyotaku,
        "prevalent_wind": prevalent_wind,
        "seat_wind": seat_wind,
        "dora_indicators": dora_indicators,
    }


def _observe_2D(state: EnvState):
    """TBD — 2D observation not yet implemented."""
    pass
