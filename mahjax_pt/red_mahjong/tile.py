# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import torch

from .types import Array
from .action import Action
from .constants import RED_FIVE_TILE_IDS, RED_FIVE_TILE_TYPES


class Tile:
    """
    tile_id: when all 136 tiles are distinguished (0..135)
    tile_type: canonical tile type 0..33 (man 0-8, pin 9-17, sou 18-26, winds 27-30, dragons 31-33)
    tile: local tile index 0..36 where 34=m5r, 35=p5r, 36=s5r
    """

    NUM_TILE_ID = 136
    NUM_TILE_TYPE = 34
    NUM_TILE_TYPE_WITH_RED = 37
    BLACK_FIVE = {"m": 4, "p": 13, "s": 22}
    RED_FIVE = {"m": 34, "p": 35, "s": 36}

    # Build lookup from tile_id (0..135) → tile index (0..36).  3 red ids are overwritten.
    _from_tile_id = (torch.arange(136, dtype=torch.int32) // 4).to(torch.int8)
    _red_ids = torch.tensor(RED_FIVE_TILE_IDS, dtype=torch.int32)
    _red_vals = torch.tensor([RED_FIVE["m"], RED_FIVE["p"], RED_FIVE["s"]], dtype=torch.int8)
    _from_tile_id[_red_ids] = _red_vals
    FROM_TILE_ID_TO_TILE = _from_tile_id

    @staticmethod
    def from_tile_id_to_tile(tile_id: Array) -> Array:
        return Tile.FROM_TILE_ID_TO_TILE[tile_id.long()]

    @staticmethod
    def is_tile_red(tile) -> bool:
        if isinstance(tile, torch.Tensor):
            return bool((tile >= Tile.NUM_TILE_TYPE).item())
        return tile >= Tile.NUM_TILE_TYPE

    @staticmethod
    def to_tile_type(tile) -> int:
        """Map tile index (0..36) → canonical tile type (0..33).

        Accepts int, float, or torch scalar/0-d tensor. Returns Python int.
        """
        if isinstance(tile, torch.Tensor):
            t = int(tile.item())
        else:
            t = int(tile)
        if t == Tile.RED_FIVE["m"]:
            return Tile.BLACK_FIVE["m"]
        elif t == Tile.RED_FIVE["p"]:
            return Tile.BLACK_FIVE["p"]
        elif t == Tile.RED_FIVE["s"]:
            return Tile.BLACK_FIVE["s"]
        return t

    @staticmethod
    def to_red(tile_type) -> int:
        """Map canonical tile type → red-five variant if it is a 5.

        Accepts int, float, or torch scalar. Returns Python int.
        """
        if isinstance(tile_type, torch.Tensor):
            t = int(tile_type.item())
        else:
            t = int(tile_type)
        if t == Tile.BLACK_FIVE["m"]:
            return Tile.RED_FIVE["m"]
        elif t == Tile.BLACK_FIVE["p"]:
            return Tile.RED_FIVE["p"]
        elif t == Tile.BLACK_FIVE["s"]:
            return Tile.RED_FIVE["s"]
        return t

    @staticmethod
    def is_tile_type_five(tile_type) -> bool:
        if isinstance(tile_type, torch.Tensor):
            t = int(tile_type.item())
        else:
            t = int(tile_type)
        return t in RED_FIVE_TILE_TYPES

    @staticmethod
    def is_tile_type_seven(tile_type) -> bool:
        t = Tile.to_tile_type(tile_type)
        return (t % 9 == 6) and (t < 27)

    @staticmethod
    def is_tile_type_three(tile_type) -> bool:
        t = Tile.to_tile_type(tile_type)
        return (t % 9 == 2) and (t < 27)

    @staticmethod
    def is_tile_four_wind(tile) -> bool:
        t = Tile.to_tile_type(tile)
        return 27 <= t < 31

    @staticmethod
    def is_yaochu(tile) -> bool:
        t = Tile.to_tile_type(tile)
        num = t % 9
        return (t >= 27) or (num == 0) or (num == 8)


# --- River bit-packing (int32 used in place of JAX uint16) ---
# PyTorch lacks uint16; we use int32 and mask to 16-bit.
_TILE_MASK = 0b0000000000111111         # 6 bits for tile index
_BIT_RIICHI = 1 << 6
_BIT_GRAY = 1 << 7
_BIT_TSUMOGIRI = 1 << 8
_SRC_SHIFT = 9
_MT_SHIFT = 11
_SRC_MASK = 0b11 << _SRC_SHIFT
_MT_MASK = 0b111 << _MT_SHIFT
EMPTY_RIVER = 0xFFFF                    # Sentinel: empty slot


class River:
    @staticmethod
    def add_discard(
        river: Array,
        tile: Array,
        player: Array,
        idx: Array,
        is_tsumogiri: bool,
        is_riichi: bool,
    ) -> Array:
        tile_u16 = (int(tile) & _TILE_MASK)
        if is_tsumogiri:
            tile_u16 |= _BIT_TSUMOGIRI
        if is_riichi:
            tile_u16 |= _BIT_RIICHI
        # GRAY is always False for normal discards; SRC=0, MT=0
        river = river.clone()
        river[player, idx] = tile_u16
        return river

    @staticmethod
    def add_meld(
        river: Array, action: int, player: Array, idx: Array, src: int
    ) -> Array:
        tile_u16 = int(river[player, idx].item())

        if action in (Action.PON, Action.PON_RED):
            meld_type = 1
        elif action == Action.OPEN_KAN:
            meld_type = 2
        elif action in (Action.CHI_L, Action.CHI_L_RED):
            meld_type = 3
        elif action in (Action.CHI_M, Action.CHI_M_RED):
            meld_type = 4
        elif action in (Action.CHI_R, Action.CHI_R_RED):
            meld_type = 5
        else:
            meld_type = 0

        tile_u16 &= ~_BIT_GRAY
        tile_u16 &= ~_SRC_MASK
        tile_u16 &= ~_MT_MASK
        tile_u16 |= _BIT_GRAY
        tile_u16 |= ((src & 0b11) << _SRC_SHIFT)
        tile_u16 |= ((meld_type & 0b111) << _MT_SHIFT)

        river = river.clone()
        river[player, idx] = tile_u16
        return river

    @staticmethod
    def decode_river(river: Array) -> Array:
        """Decode river row → (6, N) tensor: tile, riichi, gray, tsumogiri, src, meld_type."""
        empty = river == EMPTY_RIVER
        tile = (river & _TILE_MASK).to(torch.int32)
        riichi = (river & _BIT_RIICHI) != 0
        gray = (river & _BIT_GRAY) != 0
        tsumogiri = (river & _BIT_TSUMOGIRI) != 0
        src = ((river & _SRC_MASK) >> _SRC_SHIFT).to(torch.int32)
        meld_type = ((river & _MT_MASK) >> _MT_SHIFT).to(torch.int32)

        tile = torch.where(empty, torch.tensor(-1, dtype=torch.int32), tile)
        riichi_i = torch.where(empty, torch.tensor(0, dtype=torch.int32), riichi.to(torch.int32))
        gray_i = torch.where(empty, torch.tensor(0, dtype=torch.int32), gray.to(torch.int32))
        tsumog_i = torch.where(empty, torch.tensor(0, dtype=torch.int32), tsumogiri.to(torch.int32))
        src_i = torch.where(empty, torch.tensor(0, dtype=torch.int32), src)
        mt_i = torch.where(empty, torch.tensor(0, dtype=torch.int32), meld_type)
        return torch.stack([tile, riichi_i, gray_i, tsumog_i, src_i, mt_i], dim=0)

    @staticmethod
    def decode_tile(river: Array) -> Array:
        empty = river == EMPTY_RIVER
        tile = (river & _TILE_MASK).to(torch.int32)
        return torch.where(empty, torch.tensor(-1, dtype=torch.int32), tile)
