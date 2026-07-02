# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import torch
import numpy as np
from pathlib import Path
import importlib.resources as resources

from .types import Array
from .action import Action
from .tile import Tile


def _load_hand_cache():
    """Load the hand-cache from the local mahjax_pt package."""
    with resources.as_file(resources.files("mahjax_pt._src.cache").joinpath("hand_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return torch.from_numpy(data["data"].astype(np.int64))


THIRTEEN_ORPHAN_IDX = torch.tensor([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
POWERS_OF_5_FULL = torch.cat([
    5 ** torch.arange(8, -1, -1),
    5 ** torch.arange(8, -1, -1),
    5 ** torch.arange(8, -1, -1),
])


class Hand:
    CACHE = _load_hand_cache()

    @staticmethod
    def _is_red_chi_action(action):
        return (action == Action.CHI_L_RED) | (action == Action.CHI_M_RED) | (action == Action.CHI_R_RED)

    @staticmethod
    def _chi_index(action):
        if action in (Action.CHI_L, Action.CHI_L_RED):
            return 0
        elif action in (Action.CHI_M, Action.CHI_M_RED):
            return 1
        elif action in (Action.CHI_R, Action.CHI_R_RED):
            return 2
        return -1

    @staticmethod
    def _base_chi_action(chi_idx):
        if chi_idx == 0:
            return Action.CHI_L
        elif chi_idx == 1:
            return Action.CHI_M
        else:
            return Action.CHI_R

    KYUUSHU_MASK = torch.tensor(
        [
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1,
        ],
        dtype=torch.int32,
    )

    @staticmethod
    def make_init_hand(deck):
        """Deal initial hands from the deck (last 52 tiles → 4 players × 13)."""
        hand = torch.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=torch.int8)
        hand_ids = deck[-(13 * 4):].reshape(4, 13)
        for p in range(4):
            for t in hand_ids[p]:
                t = int(t)
                hand[p, t] += 1
        return hand

    @staticmethod
    def to_34(hand):
        """Convert a 37-type hand (with red fives) → 34-type hand."""
        if hand.shape[0] == Tile.NUM_TILE_TYPE:
            return hand
        hand_34 = hand[:Tile.NUM_TILE_TYPE].clone()
        hand_34[Tile.BLACK_FIVE["m"]] += hand[Tile.RED_FIVE["m"]]
        hand_34[Tile.BLACK_FIVE["p"]] += hand[Tile.RED_FIVE["p"]]
        hand_34[Tile.BLACK_FIVE["s"]] += hand[Tile.RED_FIVE["s"]]
        return hand_34

    @staticmethod
    def cache(code):
        """Look up a code in the precomputed cache."""
        c = int(code)
        return (Hand.CACHE[c >> 5] >> (c & 0b11111)) & 1

    @staticmethod
    def has_red_of(hand, tile_type):
        if Tile.is_tile_type_five(tile_type):
            return hand[Tile.to_red(tile_type)] > 0
        return False

    @staticmethod
    def can_chi(hand, tile, action):
        is_red_chi = Hand._is_red_chi_action(action)
        if is_red_chi:
            return Hand.can_red_chi(hand, tile, action)
        else:
            return Hand.can_no_red_chi(hand, tile, action)

    @staticmethod
    def can_no_red_chi(hand, tile, action):
        hand_nr = hand if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED else Hand.to_34(hand)
        tile_type = int(Tile.to_tile_type(tile))
        chi_idx = Hand._chi_index(action)

        # Can't chi honor tiles or anything beyond sou
        if tile_type >= 27:
            return False

        if chi_idx == 0:
            can = (tile_type % 9 < 7) and (hand_nr[tile_type + 1] > 0) and (hand_nr[tile_type + 2] > 0)
        elif chi_idx == 1:
            can = (
                (tile_type % 9 < 8)
                and (tile_type % 9 > 0)
                and (hand_nr[tile_type - 1] > 0)
                and (hand_nr[tile_type + 1] > 0)
            )
        else:
            can = (tile_type % 9 > 1) and (hand_nr[tile_type - 2] > 0) and (hand_nr[tile_type - 1] > 0)
        return can

    @staticmethod
    def can_red_chi(hand, tile, action):
        hand_34 = Hand.to_34(hand)
        tile_type = int(Tile.to_tile_type(tile))
        chi_idx = Hand._chi_index(action)
        base_action = Hand._base_chi_action(chi_idx)
        can_black_chi = Hand.can_no_red_chi(hand_34, tile_type, base_action)

        if chi_idx == 0:
            has_red = Hand.has_red_of(hand, tile_type + 1) | Hand.has_red_of(hand, tile_type + 2)
        elif chi_idx == 1:
            has_red = Hand.has_red_of(hand, tile_type - 1) | Hand.has_red_of(hand, tile_type + 1)
        else:
            has_red = Hand.has_red_of(hand, tile_type - 2) | Hand.has_red_of(hand, tile_type - 1)
        return can_black_chi & has_red & (tile_type < 27)

    @staticmethod
    def can_pon(hand, tile):
        return Hand.can_no_red_pon(hand, tile) | Hand.can_red_pon(hand, tile)

    @staticmethod
    def can_no_red_pon(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        hand_nr = hand if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED else Hand.to_34(hand)
        return hand_nr[tile_type] >= 2

    @staticmethod
    def can_red_pon(hand, tile):
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return False
        tile_type = int(Tile.to_tile_type(tile))
        return Tile.is_tile_type_five(tile_type) & (hand[tile_type] > 0) & (hand[Tile.to_red(tile_type)] > 0)

    @staticmethod
    def can_open_kan(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        hand_34 = Hand.to_34(hand)
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return hand_34[tile_type] == 3
        if Tile.is_tile_type_five(tile_type):
            return (hand[tile_type] == 3) | ((hand[Tile.to_red(tile_type)] == 1) & (hand[tile_type] == 2))
        return hand[tile_type] == 3

    @staticmethod
    def can_added_kan(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        hand_34 = Hand.to_34(hand)
        return hand_34[tile_type] == 1

    @staticmethod
    def can_closed_kan(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        hand_34 = Hand.to_34(hand)
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return hand_34[tile_type] == 4
        if Tile.is_tile_type_five(tile_type):
            return (hand[tile_type] == 3) & (hand[Tile.to_red(tile_type)] == 1)
        return hand[tile_type] == 4

    @staticmethod
    def can_closed_kan_after_riichi(hand, tile, original_can_win):
        tile_type = int(Tile.to_tile_type(tile))

        if not Hand.can_closed_kan(hand, tile_type):
            return False

        new_hand = Hand.closed_kan(hand, tile_type)
        new_can_win = torch.tensor([
            Hand.can_ron(new_hand, t) for t in range(Tile.NUM_TILE_TYPE)
        ])
        return bool((original_can_win == new_can_win).all().item())

    @staticmethod
    def can_tsumo(hand):
        hand_34 = Hand.to_34(hand)
        thirteen_orphan = (hand_34[THIRTEEN_ORPHAN_IDX] > 0).all() & (hand_34[THIRTEEN_ORPHAN_IDX].sum() == 14)
        seven_pairs = (hand_34 == 2).sum() == 7

        codes = (hand_34[:27].to(torch.int32) * POWERS_OF_5_FULL).reshape(3, 9).sum(dim=1)

        valid = True
        for s in range(3):
            valid = valid & bool(Hand.cache(codes[s]))

        suit_sums = hand_34[:27].reshape(3, 9).sum(dim=1)
        heads = int(((suit_sums % 3) == 2).sum().item())
        heads_honors = int((hand_34[27:34] == 2).sum().item())
        heads += heads_honors
        valid = valid & bool(((hand_34[27:34] != 1) & (hand_34[27:34] != 4)).all().item())
        return ((valid & (heads == 1)) | thirteen_orphan | seven_pairs) == 1

    @staticmethod
    def can_ron(hand, tile):
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            tile_for_hand = tile
        else:
            tile_for_hand = Tile.to_tile_type(tile)
        return Hand.can_tsumo(Hand.add(hand, tile_for_hand))

    @staticmethod
    def is_tenpai(hand):
        hand_34 = Hand.to_34(hand)
        for tile_type in range(Tile.NUM_TILE_TYPE):
            if hand_34[tile_type] != 4 and Hand.can_ron(hand_34, tile_type):
                return True
        return False

    @staticmethod
    def can_riichi(hand):
        """Check if any discard leaves a tenpai hand.

        Uses Shanten.discard which computes shanten for all 34 tiles in one pass
        (O(34) instead of O(34×37) via nested loops).
        """
        from .shanten import Shanten
        h34 = Hand.to_34(hand)
        discards = Shanten.discard(h34)  # (34,) shanten after discarding each tile
        # shanten <= 0 means tenpai or complete after discard
        # Only consider tiles we actually have
        return bool(((discards <= 0) & (h34 > 0)).any().item())

    @staticmethod
    def can_kyuushu(hand):
        hand_34 = Hand.to_34(hand)
        return bool(((hand_34 * Hand.KYUUSHU_MASK) > 0).sum().item() >= 9)

    @staticmethod
    def add(hand, tile, x: int = 1):
        """Return a new hand with `x` added at `tile` position."""
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            target = int(Tile.to_tile_type(tile))
        else:
            target = int(tile) if isinstance(tile, (int, float)) else int(tile.item())
        hand = hand.clone()
        hand[target] += x
        return hand

    @staticmethod
    def sub(hand, tile, x: int = 1):
        return Hand.add(hand, tile, -x)

    @staticmethod
    def _remove_one_of_tile_type(hand, tile_type):
        """Remove one tile of the given type, preferring red five if available."""
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand.sub(hand, tile_type)
        if Tile.is_tile_type_five(tile_type):
            if hand[tile_type] > 0:
                return Hand.sub(hand, tile_type)
            else:
                return Hand.sub(hand, Tile.to_red(tile_type))
        else:
            return Hand.sub(hand, tile_type)

    @staticmethod
    def chi(hand, tile, action):
        is_red_chi = Hand._is_red_chi_action(action)
        if is_red_chi:
            return Hand.chi_red(hand, tile, action)
        else:
            return Hand.chi_no_red(hand, tile, action)

    @staticmethod
    def chi_no_red(hand, tile, action):
        tile_type = int(Tile.to_tile_type(tile))
        chi_idx = Hand._chi_index(action)
        start = tile_type - chi_idx
        for i in range(start, start + 3):
            if i != tile_type:
                hand = Hand.sub(hand, i)
        return hand

    @staticmethod
    def chi_red(hand, tile, action):
        tile_type = int(Tile.to_tile_type(tile))
        chi_idx = Hand._chi_index(action)
        start = tile_type - chi_idx
        for i in range(start, start + 3):
            if i != tile_type:
                remove_tile = Tile.to_red(i) if Hand.has_red_of(hand, i) else i
                hand = Hand.sub(hand, remove_tile)
        return hand

    @staticmethod
    def pon(hand, tile, action):
        if action == Action.PON_RED:
            return Hand.pon_red(hand, tile)
        else:
            return Hand.pon_no_red(hand, tile)

    @staticmethod
    def pon_no_red(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        return Hand.sub(hand, tile_type, 2)

    @staticmethod
    def pon_red(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        hand = Hand.sub(hand, tile_type)
        return Hand.sub(hand, Tile.to_red(tile_type))

    @staticmethod
    def open_kan(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand.sub(hand, tile_type, 3)

        if not Tile.is_tile_type_five(tile_type):
            return Hand.sub(hand, tile_type, 3)

        if Tile.is_tile_red(tile):
            return Hand.sub(hand, tile_type, 3)
        else:
            if hand[Tile.to_red(tile_type)] > 0:
                hand = Hand.sub(hand, tile_type, 2)
                return Hand.sub(hand, Tile.to_red(tile_type))
            else:
                return Hand.sub(hand, tile_type, 3)

    @staticmethod
    def added_kan(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand._remove_one_of_tile_type(hand, tile_type)
        return Hand.sub(hand, tile_type)

    @staticmethod
    def closed_kan(hand, tile):
        tile_type = int(Tile.to_tile_type(tile))
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand.sub(hand, tile_type, 4)
        if Tile.is_tile_type_five(tile_type):
            hand = Hand.sub(hand, Tile.to_red(tile_type))
            return Hand.sub(hand, tile_type, 3)
        return Hand.sub(hand, tile_type, 4)

    # ── Batch operations (tensorized for multi-env parallelism) ──

    @staticmethod
    def add_batch(hands, tiles, x=1):
        """hands: (B, 34) or (B, 37), tiles: (B,) int — batch tile add."""
        B = hands.shape[0]
        hands = hands.clone()
        for b in range(B):
            t = int(tiles[b].item()) if isinstance(tiles[b], torch.Tensor) else int(tiles[b])
            if t >= 0 and t < hands.shape[1]:
                hands[b, t] += x
        return hands

    @staticmethod
    def sub_batch(hands, tiles, x=1):
        return Hand.add_batch(hands, tiles, -x)

    @staticmethod
    def can_ron_batch(hands, tiles):
        """hands: (B, 34) or (B, 37), tiles: (B,) int — batch can_ron check."""
        B = hands.shape[0]
        results = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            results[b] = Hand.can_ron(hands[b], int(tiles[b].item()))
        return results

    @staticmethod
    def can_pon_batch(hands, tiles):
        """hands: (B, 37), tiles: (B,) int."""
        B = hands.shape[0]
        results = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            results[b] = Hand.can_pon(hands[b], int(tiles[b].item()))
        return results

    @staticmethod
    def can_chi_any_batch(hands, tiles):
        """hands: (B, 37), tiles: (B,) int — can chi at all (for mask, don't care which type)."""
        B = hands.shape[0]; results = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            h, t = hands[b], int(tiles[b].item())
            for a in (Action.CHI_L, Action.CHI_M, Action.CHI_R,
                       Action.CHI_L_RED, Action.CHI_M_RED, Action.CHI_R_RED):
                if Hand.can_chi(h, t, a):
                    results[b] = True; break
        return results

    @staticmethod
    def can_open_kan_batch(hands, tiles):
        B = hands.shape[0]; results = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            results[b] = Hand.can_open_kan(hands[b], int(tiles[b].item()))
        return results

    @staticmethod
    def to_34_batch(hands):
        """hands: (B, 37) or (B, 34) — batch to_34."""
        if hands.shape[1] == Tile.NUM_TILE_TYPE:
            return hands
        B = hands.shape[0]
        out = hands[:, :Tile.NUM_TILE_TYPE].clone()
        out[:, Tile.BLACK_FIVE["m"]] += hands[:, Tile.RED_FIVE["m"]]
        out[:, Tile.BLACK_FIVE["p"]] += hands[:, Tile.RED_FIVE["p"]]
        out[:, Tile.BLACK_FIVE["s"]] += hands[:, Tile.RED_FIVE["s"]]
        return out

    @staticmethod
    def is_tenpai_batch(hands_34):
        """hands_34: (B, 34) — batch tenpai check."""
        B = hands_34.shape[0]
        results = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            results[b] = Hand.is_tenpai(hands_34[b])
        return results
