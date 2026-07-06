# Copyright 2025 The Mahjax Authors.
# Batch state definitions for the parallel (vectorized) environment.
#
# BatchState packs B independent EnvState instances into a single
# batch-first tensor structure. All fields have leading batch dim (B,)
# or (B, 4, ...) for per-player data.

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import List, Optional

from .constants import (
    LEGAL_ACTION_SIZE,
    MAX_DISCARDS_PER_PLAYER,
    MAX_DORA_INDICATORS,
    MAX_HAND_TILES,
    MAX_MELDS_PER_PLAYER,
    NUM_PHYSICAL_TILES,
    NUM_PLAYERS,
    NUM_TILE_TYPES,
    NUM_TILE_TYPES_WITH_RED,
    SENTINEL_DISCARD_VALUE,
    SENTINEL_MELD_VALUE,
    SENTINEL_TILE_ID,
)
from .meld import EMPTY_MELD
from .tile import EMPTY_RIVER
from .state import EnvState, PlayerStateArrays, RoundState


@dataclass
class BatchPlayerState:
    """Per-player state batched across B environments.

    All tensors have shape (B, 4, ...) where axis-1 is the player index.
    """
    hand: torch.Tensor = field(default=None)              # (B, 4, 34) int8
    hand_with_red: torch.Tensor = field(default=None)     # (B, 4, 37) int8
    hand_ids: torch.Tensor = field(default=None)          # (B, 4, MAX_HAND_TILES) int16
    hand_counts: torch.Tensor = field(default=None)       # (B, 4) int8
    drawn_tile: torch.Tensor = field(default=None)        # (B, 4) int16
    legal_action_mask: torch.Tensor = field(default=None) # (B, 4, LEGAL_ACTION_SIZE) bool
    can_win: torch.Tensor = field(default=None)           # (B, 4, 34) bool
    has_yaku: torch.Tensor = field(default=None)          # (B, 4, 2) bool
    fan: torch.Tensor = field(default=None)               # (B, 4, 2) int32
    fu: torch.Tensor = field(default=None)                # (B, 4, 2) int32
    melds: torch.Tensor = field(default=None)             # (B, 4, MAX_MELDS_PER_PLAYER) int32
    meld_tiles: torch.Tensor = field(default=None)        # (B, 4, MAX_MELDS_PER_PLAYER, 4) int16
    meld_info: torch.Tensor = field(default=None)         # (B, 4, MAX_MELDS_PER_PLAYER, 3) int8
    meld_counts: torch.Tensor = field(default=None)       # (B, 4) int8
    river: torch.Tensor = field(default=None)             # (B, 4, MAX_DISCARDS_PER_PLAYER) int32
    discards: torch.Tensor = field(default=None)          # (B, 4, MAX_DISCARDS_PER_PLAYER) int16
    discard_info: torch.Tensor = field(default=None)      # (B, 4, MAX_DISCARDS_PER_PLAYER, 4) int8
    discard_counts: torch.Tensor = field(default=None)    # (B, 4) int8
    riichi: torch.Tensor = field(default=None)            # (B, 4) bool
    riichi_declared: torch.Tensor = field(default=None)   # (B, 4) bool
    riichi_step: torch.Tensor = field(default=None)       # (B, 4) int8
    double_riichi: torch.Tensor = field(default=None)     # (B, 4) bool
    ippatsu: torch.Tensor = field(default=None)           # (B, 4) bool
    furiten_by_discard: torch.Tensor = field(default=None)# (B, 4) bool
    furiten_by_pass: torch.Tensor = field(default=None)   # (B, 4) bool
    is_hand_concealed: torch.Tensor = field(default=None) # (B, 4) bool
    pon: torch.Tensor = field(default=None)               # (B, 4, 34) int32
    has_won: torch.Tensor = field(default=None)           # (B, 4) bool
    n_kan: torch.Tensor = field(default=None)             # (B, 4) int8
    has_nagashi_mangan: torch.Tensor = field(default=None)# (B, 4) bool


@dataclass
class BatchRoundState:
    """Round-level state batched across B environments.

    Scalar fields become (B,) tensors.
    """
    action_history: torch.Tensor = field(default=None)    # (B, 3, 200) int8
    shanten_current_player: torch.Tensor = field(default=None)  # (B,) int
    round: torch.Tensor = field(default=None)              # (B,) int8
    round_limit: torch.Tensor = field(default=None)        # (B,) int8
    terminated_round: torch.Tensor = field(default=None)   # (B,) bool
    honba: torch.Tensor = field(default=None)              # (B,) int8
    kyotaku: torch.Tensor = field(default=None)            # (B,) int8
    init_wind: torch.Tensor = field(default=None)          # (B, 4) int8
    seat_wind: torch.Tensor = field(default=None)          # (B, 4) int8
    dealer: torch.Tensor = field(default=None)             # (B,) int
    order_points: torch.Tensor = field(default=None)       # (B, 4) int32
    score: torch.Tensor = field(default=None)              # (B, 4) int32
    deck: torch.Tensor = field(default=None)               # (B, NUM_PHYSICAL_TILES) int8
    next_deck_ix: torch.Tensor = field(default=None)       # (B,) int
    last_deck_ix: torch.Tensor = field(default=None)       # (B,) int
    draw_next: torch.Tensor = field(default=None)          # (B,) bool
    last_draw: torch.Tensor = field(default=None)          # (B,) int
    last_player: torch.Tensor = field(default=None)        # (B,) int
    dora_indicators: torch.Tensor = field(default=None)    # (B, MAX_DORA_INDICATORS) int8
    ura_dora_indicators: torch.Tensor = field(default=None)# (B, MAX_DORA_INDICATORS) int8
    is_abortive_draw_normal: torch.Tensor = field(default=None)  # (B,) bool
    dummy_count: torch.Tensor = field(default=None)        # (B,) int8
    is_haitei: torch.Tensor = field(default=None)          # (B,) bool
    target: torch.Tensor = field(default=None)             # (B,) int
    n_kan_doras: torch.Tensor = field(default=None)        # (B,) int8
    kan_declared: torch.Tensor = field(default=None)       # (B,) bool
    can_after_kan: torch.Tensor = field(default=None)      # (B,) bool
    can_robbing_kan: torch.Tensor = field(default=None)    # (B,) bool


@dataclass
class BatchState:
    """Top-level batched environment state — B environments packed together.

    All operations in env_parallel.py operate on this structure.
    Fields have leading batch dimension (B,) or (B, 4, ...) for per-player data.
    """
    B: int = 1
    current_player: Optional[torch.Tensor] = None          # (B,) int
    legal_action_mask: Optional[torch.Tensor] = None       # (B, LEGAL_ACTION_SIZE) bool
    players: Optional[BatchPlayerState] = None
    round_state: Optional[BatchRoundState] = None
    step_count: Optional[torch.Tensor] = None              # (B,) int
    rewards: Optional[torch.Tensor] = None                 # (B, 4) float
    terminated: Optional[torch.Tensor] = None              # (B,) bool
    truncated: Optional[torch.Tensor] = None               # (B,) bool


# ─── Conversion utilities ─────────────────────────────────────────

def _default_batch_state(B: int, device: torch.device = torch.device("cpu")) -> BatchState:
    """Create a default BatchState with all tensors initialized."""
    ps = BatchPlayerState(
        hand=torch.zeros(B, NUM_PLAYERS, NUM_TILE_TYPES, dtype=torch.int8, device=device),
        hand_with_red=torch.zeros(B, NUM_PLAYERS, NUM_TILE_TYPES_WITH_RED, dtype=torch.int8, device=device),
        hand_ids=torch.full((B, NUM_PLAYERS, MAX_HAND_TILES), SENTINEL_TILE_ID, dtype=torch.int16, device=device),
        hand_counts=torch.zeros(B, NUM_PLAYERS, dtype=torch.int8, device=device),
        drawn_tile=torch.full((B, NUM_PLAYERS,), SENTINEL_TILE_ID, dtype=torch.int16, device=device),
        legal_action_mask=torch.zeros(B, NUM_PLAYERS, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device),
        can_win=torch.zeros(B, NUM_PLAYERS, NUM_TILE_TYPES, dtype=torch.bool, device=device),
        has_yaku=torch.zeros(B, NUM_PLAYERS, 2, dtype=torch.bool, device=device),
        fan=torch.zeros(B, NUM_PLAYERS, 2, dtype=torch.int32, device=device),
        fu=torch.zeros(B, NUM_PLAYERS, 2, dtype=torch.int32, device=device),
        melds=torch.full((B, NUM_PLAYERS, MAX_MELDS_PER_PLAYER), EMPTY_MELD, dtype=torch.int32, device=device),
        meld_tiles=torch.full((B, NUM_PLAYERS, MAX_MELDS_PER_PLAYER, 4), SENTINEL_TILE_ID, dtype=torch.int16, device=device),
        meld_info=torch.full((B, NUM_PLAYERS, MAX_MELDS_PER_PLAYER, 3), SENTINEL_MELD_VALUE, dtype=torch.int8, device=device),
        meld_counts=torch.zeros(B, NUM_PLAYERS, dtype=torch.int8, device=device),
        river=torch.full((B, NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), EMPTY_RIVER, dtype=torch.int32, device=device),
        discards=torch.full((B, NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), SENTINEL_DISCARD_VALUE, dtype=torch.int16, device=device),
        discard_info=torch.full((B, NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER, 4), SENTINEL_MELD_VALUE, dtype=torch.int8, device=device),
        discard_counts=torch.zeros(B, NUM_PLAYERS, dtype=torch.int8, device=device),
        riichi=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        riichi_declared=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        riichi_step=torch.zeros(B, NUM_PLAYERS, dtype=torch.int8, device=device),
        double_riichi=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        ippatsu=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        furiten_by_discard=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        furiten_by_pass=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        is_hand_concealed=torch.ones(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        pon=torch.zeros(B, NUM_PLAYERS, NUM_TILE_TYPES, dtype=torch.int32, device=device),
        has_won=torch.zeros(B, NUM_PLAYERS, dtype=torch.bool, device=device),
        n_kan=torch.zeros(B, NUM_PLAYERS, dtype=torch.int8, device=device),
        has_nagashi_mangan=torch.ones(B, NUM_PLAYERS, dtype=torch.bool, device=device),
    )

    rs = BatchRoundState(
        action_history=torch.full((B, 3, 200), -1, dtype=torch.int8, device=device),
        shanten_current_player=torch.zeros(B, dtype=torch.int32, device=device),
        round=torch.zeros(B, dtype=torch.int8, device=device),
        round_limit=torch.full((B,), 7, dtype=torch.int8, device=device),
        terminated_round=torch.zeros(B, dtype=torch.bool, device=device),
        honba=torch.zeros(B, dtype=torch.int8, device=device),
        kyotaku=torch.zeros(B, dtype=torch.int8, device=device),
        init_wind=torch.tensor([0, 1, 2, 3], dtype=torch.int8, device=device).unsqueeze(0).expand(B, -1),
        seat_wind=torch.tensor([0, 1, 2, 3], dtype=torch.int8, device=device).unsqueeze(0).expand(B, -1),
        dealer=torch.zeros(B, dtype=torch.int32, device=device),
        order_points=torch.tensor([30, 10, -10, -30], dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1),
        score=torch.full((B, NUM_PLAYERS), 250, dtype=torch.int32, device=device),
        deck=torch.zeros(B, NUM_PHYSICAL_TILES, dtype=torch.int8, device=device),
        next_deck_ix=torch.full((B,), 83, dtype=torch.int32, device=device),
        last_deck_ix=torch.full((B,), 14, dtype=torch.int32, device=device),
        draw_next=torch.zeros(B, dtype=torch.bool, device=device),
        last_draw=torch.full((B,), -1, dtype=torch.int32, device=device),
        last_player=torch.full((B,), -1, dtype=torch.int32, device=device),
        dora_indicators=torch.full((B, MAX_DORA_INDICATORS), -1, dtype=torch.int8, device=device),
        ura_dora_indicators=torch.full((B, MAX_DORA_INDICATORS), -1, dtype=torch.int8, device=device),
        is_abortive_draw_normal=torch.zeros(B, dtype=torch.bool, device=device),
        dummy_count=torch.zeros(B, dtype=torch.int8, device=device),
        is_haitei=torch.zeros(B, dtype=torch.bool, device=device),
        target=torch.full((B,), -1, dtype=torch.int32, device=device),
        n_kan_doras=torch.zeros(B, dtype=torch.int8, device=device),
        kan_declared=torch.zeros(B, dtype=torch.bool, device=device),
        can_after_kan=torch.zeros(B, dtype=torch.bool, device=device),
        can_robbing_kan=torch.zeros(B, dtype=torch.bool, device=device),
    )

    return BatchState(
        B=B,
        current_player=torch.zeros(B, dtype=torch.int32, device=device),
        legal_action_mask=torch.zeros(B, LEGAL_ACTION_SIZE, dtype=torch.bool, device=device),
        players=ps,
        round_state=rs,
        step_count=torch.zeros(B, dtype=torch.int32, device=device),
        rewards=torch.zeros(B, NUM_PLAYERS, dtype=torch.float32, device=device),
        terminated=torch.zeros(B, dtype=torch.bool, device=device),
        truncated=torch.zeros(B, dtype=torch.bool, device=device),
    )


def stack_states(states: List[EnvState]) -> BatchState:
    """Pack B independent EnvState instances into a single BatchState."""
    B = len(states)
    device = states[0].players.hand.device
    bs = _default_batch_state(B, device)

    for i, s in enumerate(states):
        bs.current_player[i] = s.current_player
        bs.legal_action_mask[i] = s.legal_action_mask
        bs.step_count[i] = s.step_count
        bs.rewards[i] = s.rewards
        bs.terminated[i] = s.terminated
        bs.truncated[i] = s.truncated

        # Player state
        ps = s.players
        bps = bs.players
        bps.hand[i] = ps.hand
        bps.hand_with_red[i] = ps.hand_with_red
        bps.hand_ids[i] = ps.hand_ids
        bps.hand_counts[i] = ps.hand_counts
        bps.drawn_tile[i] = ps.drawn_tile
        bps.legal_action_mask[i] = ps.legal_action_mask
        bps.can_win[i] = ps.can_win
        bps.has_yaku[i] = ps.has_yaku
        bps.fan[i] = ps.fan
        bps.fu[i] = ps.fu
        bps.melds[i] = ps.melds
        bps.meld_tiles[i] = ps.meld_tiles
        bps.meld_info[i] = ps.meld_info
        bps.meld_counts[i] = ps.meld_counts
        bps.river[i] = ps.river
        bps.discards[i] = ps.discards
        bps.discard_info[i] = ps.discard_info
        bps.discard_counts[i] = ps.discard_counts
        bps.riichi[i] = ps.riichi
        bps.riichi_declared[i] = ps.riichi_declared
        bps.riichi_step[i] = ps.riichi_step
        bps.double_riichi[i] = ps.double_riichi
        bps.ippatsu[i] = ps.ippatsu
        bps.furiten_by_discard[i] = ps.furiten_by_discard
        bps.furiten_by_pass[i] = ps.furiten_by_pass
        bps.is_hand_concealed[i] = ps.is_hand_concealed
        bps.pon[i] = ps.pon
        bps.has_won[i] = ps.has_won
        bps.n_kan[i] = ps.n_kan
        bps.has_nagashi_mangan[i] = ps.has_nagashi_mangan

        # Round state
        rs = s.round_state
        brs = bs.round_state
        brs.action_history[i] = rs.action_history
        brs.shanten_current_player[i] = rs.shanten_current_player
        brs.round[i] = rs.round
        brs.round_limit[i] = rs.round_limit
        brs.terminated_round[i] = rs.terminated_round
        brs.honba[i] = rs.honba
        brs.kyotaku[i] = rs.kyotaku
        brs.init_wind[i] = rs.init_wind
        brs.seat_wind[i] = rs.seat_wind
        brs.dealer[i] = rs.dealer
        brs.order_points[i] = rs.order_points
        brs.score[i] = rs.score
        brs.deck[i] = rs.deck
        brs.next_deck_ix[i] = rs.next_deck_ix
        brs.last_deck_ix[i] = rs.last_deck_ix
        brs.draw_next[i] = rs.draw_next
        brs.last_draw[i] = rs.last_draw
        brs.last_player[i] = rs.last_player
        brs.dora_indicators[i] = rs.dora_indicators
        brs.ura_dora_indicators[i] = rs.ura_dora_indicators
        brs.is_abortive_draw_normal[i] = rs.is_abortive_draw_normal
        brs.dummy_count[i] = rs.dummy_count
        brs.is_haitei[i] = rs.is_haitei
        brs.target[i] = rs.target
        brs.n_kan_doras[i] = rs.n_kan_doras
        brs.kan_declared[i] = rs.kan_declared
        brs.can_after_kan[i] = rs.can_after_kan
        brs.can_robbing_kan[i] = rs.can_robbing_kan

    return bs


def unstack_state(batch_state: BatchState, index: int) -> EnvState:
    """Extract the index-th EnvState from a BatchState."""
    from .state import PlayerStateArrays, RoundState, EnvState as ES

    i = index
    ps = batch_state.players
    rs = batch_state.round_state

    players = PlayerStateArrays()
    players.hand = ps.hand[i].clone()
    players.hand_with_red = ps.hand_with_red[i].clone()
    players.hand_ids = ps.hand_ids[i].clone()
    players.hand_counts = ps.hand_counts[i].clone()
    players.drawn_tile = ps.drawn_tile[i].clone()
    players.legal_action_mask = ps.legal_action_mask[i].clone()
    players.can_win = ps.can_win[i].clone()
    players.has_yaku = ps.has_yaku[i].clone()
    players.fan = ps.fan[i].clone()
    players.fu = ps.fu[i].clone()
    players.melds = ps.melds[i].clone()
    players.meld_tiles = ps.meld_tiles[i].clone()
    players.meld_info = ps.meld_info[i].clone()
    players.meld_counts = ps.meld_counts[i].clone()
    players.river = ps.river[i].clone()
    players.discards = ps.discards[i].clone()
    players.discard_info = ps.discard_info[i].clone()
    players.discard_counts = ps.discard_counts[i].clone()
    players.riichi = ps.riichi[i].clone()
    players.riichi_declared = ps.riichi_declared[i].clone()
    players.riichi_step = ps.riichi_step[i].clone()
    players.double_riichi = ps.double_riichi[i].clone()
    players.ippatsu = ps.ippatsu[i].clone()
    players.furiten_by_discard = ps.furiten_by_discard[i].clone()
    players.furiten_by_pass = ps.furiten_by_pass[i].clone()
    players.is_hand_concealed = ps.is_hand_concealed[i].clone()
    players.pon = ps.pon[i].clone()
    players.has_won = ps.has_won[i].clone()
    players.n_kan = ps.n_kan[i].clone()
    players.has_nagashi_mangan = ps.has_nagashi_mangan[i].clone()

    round_state = RoundState()
    round_state.action_history = rs.action_history[i].clone()
    round_state.shanten_current_player = int(rs.shanten_current_player[i].item())
    round_state.round = int(rs.round[i].item())
    round_state.round_limit = int(rs.round_limit[i].item())
    round_state.terminated_round = bool(rs.terminated_round[i].item())
    round_state.honba = int(rs.honba[i].item())
    round_state.kyotaku = int(rs.kyotaku[i].item())
    round_state.init_wind = rs.init_wind[i].clone()
    round_state.seat_wind = rs.seat_wind[i].clone()
    round_state.dealer = int(rs.dealer[i].item())
    round_state.order_points = rs.order_points[i].clone()
    round_state.score = rs.score[i].clone()
    round_state.deck = rs.deck[i].clone()
    round_state.next_deck_ix = int(rs.next_deck_ix[i].item())
    round_state.last_deck_ix = int(rs.last_deck_ix[i].item())
    round_state.draw_next = bool(rs.draw_next[i].item())
    round_state.last_draw = int(rs.last_draw[i].item())
    round_state.last_player = int(rs.last_player[i].item())
    round_state.dora_indicators = rs.dora_indicators[i].clone()
    round_state.ura_dora_indicators = rs.ura_dora_indicators[i].clone()
    round_state.is_abortive_draw_normal = bool(rs.is_abortive_draw_normal[i].item())
    round_state.dummy_count = int(rs.dummy_count[i].item())
    round_state.is_haitei = bool(rs.is_haitei[i].item())
    round_state.target = int(rs.target[i].item())
    round_state.n_kan_doras = int(rs.n_kan_doras[i].item())
    round_state.kan_declared = bool(rs.kan_declared[i].item())
    round_state.can_after_kan = bool(rs.can_after_kan[i].item())
    round_state.can_robbing_kan = bool(rs.can_robbing_kan[i].item())

    state = ES()
    state.current_player = int(batch_state.current_player[i].item())
    state.legal_action_mask = batch_state.legal_action_mask[i].clone()
    state.players = players
    state.round_state = round_state
    state.step_count = int(batch_state.step_count[i].item())
    state.rewards = batch_state.rewards[i].clone()
    state.terminated = bool(batch_state.terminated[i].item())
    state.truncated = bool(batch_state.truncated[i].item())

    return state
