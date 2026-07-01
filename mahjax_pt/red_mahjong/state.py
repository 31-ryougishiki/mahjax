from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    COPIES_PER_TILE,
    HONBA_BONUS,
    LEGAL_ACTION_SIZE,
    MAX_DISCARDS_PER_PLAYER,
    MAX_DORA_INDICATORS,
    MAX_HAND_TILES,
    MAX_MELDS_PER_PLAYER,
    NUM_PHYSICAL_TILES,
    NUM_PLAYERS,
    NUM_TILE_TYPES,
    NUM_TILE_TYPES_WITH_RED,
    RIICHI_BET,
    SENTINEL_DISCARD_VALUE,
    SENTINEL_MELD_VALUE,
    SENTINEL_TILE_ID,
    STARTING_POINTS,
    TARGET_POINTS,
)
from .meld import EMPTY_MELD
from .tile import EMPTY_RIVER


@dataclass
class GameConfig:
    """Configuration knobs for red_mahjong rules."""
    allow_open_tanyao: bool = True
    allow_kuikae: bool = False
    use_red_fives: bool = True
    allow_double_ron: bool = True
    enable_special_abortive_draw: bool = True
    enable_pao: bool = True
    seed_wall_from_key: bool = True
    starting_points: int = STARTING_POINTS
    target_points: int = TARGET_POINTS
    honba_bonus: int = HONBA_BONUS
    riichi_bet: int = RIICHI_BET


def default_game_config() -> GameConfig:
    return GameConfig()


@dataclass
class PlayerStateArrays:
    """Per-player state — all tensors have leading axis 4 (one per seat)."""
    hand: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=torch.int8))
    hand_with_red: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, NUM_TILE_TYPES_WITH_RED), dtype=torch.int8))
    hand_ids: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_HAND_TILES), SENTINEL_TILE_ID, dtype=torch.int16))
    hand_counts: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.int8))
    drawn_tile: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS,), SENTINEL_TILE_ID, dtype=torch.int16))
    legal_action_mask: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, LEGAL_ACTION_SIZE), dtype=torch.bool))
    can_win: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=torch.bool))
    has_yaku: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, 2), dtype=torch.bool))
    fan: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, 2), dtype=torch.int32))
    fu: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, 2), dtype=torch.int32))
    melds: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER), EMPTY_MELD, dtype=torch.int32))
    meld_tiles: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER, 4), SENTINEL_TILE_ID, dtype=torch.int16))
    meld_info: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER, 3), SENTINEL_MELD_VALUE, dtype=torch.int8))
    meld_counts: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.int8))
    river: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), EMPTY_RIVER, dtype=torch.int32))
    discards: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), SENTINEL_DISCARD_VALUE, dtype=torch.int16))
    discard_info: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER, 4), SENTINEL_MELD_VALUE, dtype=torch.int8))
    discard_counts: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.int8))
    riichi: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    riichi_declared: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    riichi_step: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.int8))
    double_riichi: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    ippatsu: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    furiten_by_discard: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    furiten_by_pass: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    is_hand_concealed: torch.Tensor = field(default_factory=lambda: torch.ones((NUM_PLAYERS,), dtype=torch.bool))
    pon: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=torch.int32))
    has_won: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.bool))
    n_kan: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.int8))
    has_nagashi_mangan: torch.Tensor = field(default_factory=lambda: torch.ones((NUM_PLAYERS,), dtype=torch.bool))


@dataclass
class RoundState:
    """Round-level state — shared across all four players."""
    rng_key: Optional[torch.Generator] = None
    action_history: torch.Tensor = field(default_factory=lambda: torch.full((3, 200), -1, dtype=torch.int8))
    shanten_current_player: int = 0
    round: int = 0
    round_limit: int = 7
    terminated_round: bool = False
    honba: int = 0
    kyotaku: int = 0
    init_wind: torch.Tensor = field(default_factory=lambda: torch.tensor([0, 1, 2, 3], dtype=torch.int8))
    seat_wind: torch.Tensor = field(default_factory=lambda: torch.tensor([0, 1, 2, 3], dtype=torch.int8))
    dealer: int = 0
    order_points: torch.Tensor = field(default_factory=lambda: torch.tensor([30, 10, -10, -30], dtype=torch.int32))
    score: torch.Tensor = field(default_factory=lambda: torch.full((NUM_PLAYERS,), 250, dtype=torch.int32))
    deck: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PHYSICAL_TILES,), dtype=torch.int8))
    next_deck_ix: int = 83
    last_deck_ix: int = 14
    draw_next: bool = False
    last_draw: int = -1
    last_player: int = -1
    dora_indicators: torch.Tensor = field(default_factory=lambda: torch.full((MAX_DORA_INDICATORS,), -1, dtype=torch.int8))
    ura_dora_indicators: torch.Tensor = field(default_factory=lambda: torch.full((MAX_DORA_INDICATORS,), -1, dtype=torch.int8))
    is_abortive_draw_normal: bool = False
    dummy_count: int = 0
    is_haitei: bool = False
    target: int = -1
    n_kan_doras: int = 0
    kan_declared: bool = False
    can_after_kan: bool = False
    can_robbing_kan: bool = False


@dataclass
class EnvState:
    """Top-level environment state — compatible with the mahjax.core.State API.

    Differs from JAX version:
    - Plain Python dataclass (mutable) instead of frozen flax dataclass
    - Scalar fields are plain Python int/bool instead of 0-d jnp arrays
    """
    current_player: int = 0
    legal_action_mask: torch.Tensor = field(default_factory=lambda: torch.zeros((LEGAL_ACTION_SIZE,), dtype=torch.bool))
    players: PlayerStateArrays = field(default_factory=PlayerStateArrays)
    round_state: RoundState = field(default_factory=RoundState)
    step_count: int = 0
    rewards: torch.Tensor = field(default_factory=lambda: torch.zeros((NUM_PLAYERS,), dtype=torch.float32))
    terminated: bool = False
    truncated: bool = False

    @property
    def env_id(self) -> str:
        return "red_mahjong"


State = EnvState


def default_state() -> EnvState:
    return EnvState()
