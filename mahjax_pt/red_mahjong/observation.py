from typing import Dict, TYPE_CHECKING
import torch

from .state import EnvState

if TYPE_CHECKING:
    from .batch_state import BatchState


def hand_counts_to_idx_batch(counts: torch.Tensor, fill: int = -1, hand_size: int = 14) -> torch.Tensor:
    """Vectorized conversion: (B, 37) histograms → (B, hand_size) tile index lists.

    Args:
        counts: (B, 37) tensor — how many copies of each tile type per env
        fill: value for empty slots
        hand_size: output length (default 14)
    """
    B, NT = counts.shape
    device = counts.device
    counts = counts.to(torch.int32)

    # (B, 37, 4) — each tile type has 0..4 copies
    col = torch.arange(4, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
    mask = col < counts.unsqueeze(-1)  # (B, 37, 4)

    tile_ids = torch.arange(NT, device=device).unsqueeze(0).unsqueeze(-1)  # (1, 37, 1)
    vals = torch.where(mask, tile_ids.to(torch.int32),
                       torch.tensor(fill, dtype=torch.int32, device=device))
    vals = vals.reshape(B, -1)  # (B, 148)

    # Sort: valid tiles before padding
    key = mask.reshape(B, -1).to(torch.int32)  # (B, 148)
    order = torch.argsort(-key, stable=True, dim=1)  # (B, 148)
    sorted_vals = vals.gather(1, order)

    out = sorted_vals[:, :hand_size].to(torch.int32)
    # Ensure padding slots are fill (they should be, but guard against all tiles being valid)
    out = torch.where(out == fill, torch.tensor(fill, dtype=torch.int32, device=device), out)
    return out


def hand_counts_to_idx(counts: torch.Tensor, fill: int = -1, hand_size: int = 14) -> torch.Tensor:
    """Convert a 37-type histogram into a fixed-size list of tile indices.

    Args:
        counts: (37,) tensor — how many copies of each tile type
        fill: value for empty slots
        hand_size: output length (default 14)
    """
    device = counts.device
    counts = counts.to(torch.int32)
    # Each of 37 tile types can have 0..3 copies
    col = torch.arange(4, dtype=torch.int32, device=device).unsqueeze(0)  # (1, 4)
    mask = col < counts.unsqueeze(1)  # (37, 4) bool

    tile_ids = torch.arange(37, dtype=torch.int32, device=device).unsqueeze(1).repeat(1, 4)  # (37, 4)
    vals = torch.where(mask, tile_ids.to(torch.int32),
                       torch.tensor(fill, dtype=torch.int32, device=device))
    vals = vals.reshape(-1)  # (148,)

    key = mask.reshape(-1).to(torch.int32)  # (148,)
    order = torch.argsort(-key, stable=True)
    sorted_vals = vals[order]

    out = sorted_vals[:hand_size].to(torch.int32)
    out = torch.where(out == fill, torch.tensor(fill, dtype=torch.int32, device=device), out)
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
    device = state.players.hand_with_red.device
    c_p_based_order = (torch.arange(4, device=device) + c_p) % 4

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
    shanten_c_p = torch.tensor(state.round_state.shanten_current_player, dtype=torch.int32, device=device)
    furiten = torch.tensor(
        bool(state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]),
        dtype=torch.bool, device=device)
    scores = state.round_state.score[c_p_based_order]
    round_num = torch.tensor(int(state.round_state.round), dtype=torch.int32, device=device)
    honba = torch.tensor(int(state.round_state.honba), dtype=torch.int32, device=device)
    kyotaku = torch.tensor(int(state.round_state.kyotaku), dtype=torch.int32, device=device)
    prevalent_wind = torch.tensor(int(state.round_state.round) // 4, dtype=torch.int32, device=device)
    seat_wind = torch.tensor(int(state.round_state.seat_wind[c_p]), dtype=torch.int32, device=device)
    dora_indicators = state.round_state.dora_indicators[:5]  # (5,)

    return {
        "hand": hand_c_p_14,
        "last_draw": torch.tensor(state.round_state.last_draw, dtype=torch.int32),
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


def _observe_dict_batch(bs: "BatchState") -> Dict[str, torch.Tensor]:
    """Build batched observations directly from BatchState tensors.

    No per-env unstack/stack — all operations on (B, ...) tensors.
    Output dict can be passed directly to ACNet.forward().

    Returns dict of (B, ...) tensors on the same device as the BatchState.
    """
    B = bs.B
    device = bs.players.hand.device
    cp = bs.current_player  # (B,)
    b_idx = torch.arange(B, device=device)

    # ── Hand: (B, 14) from histogram ──
    hand_37 = bs.players.hand_with_red[b_idx, cp]  # (B, 37)
    hand_14 = hand_counts_to_idx_batch(hand_37)  # (B, 14)

    # ── Action history with relative player indices: (B, 3, 200) ──
    ah = bs.round_state.action_history.clone()  # (B, 3, 200)
    player_history = ah[:, 0, :]  # (B, 200)
    valid = player_history >= 0
    relative_player = torch.remainder(player_history - cp.unsqueeze(1), 4)
    ah[:, 0, :] = torch.where(valid, relative_player.to(ah.dtype), player_history)

    # ── Furiten: (B,) bool ──
    furiten = (
        bs.players.furiten_by_discard[b_idx, cp]
        | bs.players.furiten_by_pass[b_idx, cp]
    )

    # ── Scores from current player's perspective: (B, 4) ──
    cp_order = (torch.arange(4, device=device).unsqueeze(0) + cp.unsqueeze(1)) % 4
    scores = bs.round_state.score[b_idx.unsqueeze(1), cp_order]

    # ── Dora indicators: (B, 5) ──
    dora_indicators = bs.round_state.dora_indicators[:, :5]

    return {
        "hand": hand_14,
        "last_draw": bs.round_state.last_draw,
        "action_history": ah,
        "shanten_count": bs.round_state.shanten_current_player,
        "furiten": furiten,
        "scores": scores,
        "round": bs.round_state.round,
        "honba": bs.round_state.honba,
        "kyotaku": bs.round_state.kyotaku,
        "prevalent_wind": bs.round_state.round // 4,
        "seat_wind": bs.round_state.seat_wind[b_idx, cp],
        "dora_indicators": dora_indicators,
    }


def _observe_2D(state: EnvState):
    """TBD — 2D observation not yet implemented."""
    pass
