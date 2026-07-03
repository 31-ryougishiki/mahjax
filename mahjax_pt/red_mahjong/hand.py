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
    _CACHE_DEVICE = None  # device-cached copy of CACHE
    _CACHE_LAST_DEVICE = None  # which device the cached copy is on

    @staticmethod
    def _get_cache(device):
        """Return CACHE on the given device, caching for performance."""
        if Hand._CACHE_DEVICE is None or Hand._CACHE_LAST_DEVICE != device:
            Hand._CACHE_DEVICE = Hand.CACHE.to(device)
            Hand._CACHE_LAST_DEVICE = device
        return Hand._CACHE_DEVICE

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

    # ── Batch operations (fully vectorized for multi-env parallelism) ──

    @staticmethod
    def add_batch(hands, tiles, x=1):
        """Vectorized: hands (..., 37), tiles (B,) int → hands with tiles[b] += x."""
        hands = hands.clone()
        valid = (tiles >= 0) & (tiles < hands.shape[-1])
        if valid.any():
            idx = torch.arange(hands.shape[0], device=hands.device)[valid]
            hands[idx, tiles[valid].long()] += x
        return hands

    @staticmethod
    def sub_batch(hands, tiles, x=1):
        return Hand.add_batch(hands, tiles, -x)

    @staticmethod
    def to_34_batch(hands):
        """Vectorized: works on any shape (..., 37) → (..., 34)."""
        if hands.shape[-1] == Tile.NUM_TILE_TYPE:
            return hands
        out = hands[..., :Tile.NUM_TILE_TYPE].clone()
        out[..., Tile.BLACK_FIVE["m"]] += hands[..., Tile.RED_FIVE["m"]]
        out[..., Tile.BLACK_FIVE["p"]] += hands[..., Tile.RED_FIVE["p"]]
        out[..., Tile.BLACK_FIVE["s"]] += hands[..., Tile.RED_FIVE["s"]]
        return out

    @staticmethod
    def can_tsumo_batch(hands):
        """Vectorized: hands (B, 37) → (B,) bool — complete hand check using cache."""
        B = hands.shape[0]
        device = hands.device
        h34 = Hand.to_34_batch(hands)  # (B, 34)

        # 1. Thirteen orphans
        orphan_idx = THIRTEEN_ORPHAN_IDX.to(device)
        th = h34[:, orphan_idx]  # (B, 13)
        thirteen = (th > 0).all(dim=1) & (th.sum(dim=1) == 14)  # (B,)

        # 2. Seven pairs
        seven_p = (h34 == 2).sum(dim=1) == 7  # (B,)

        # 3. Normal hand: base-5 encoding + cache lookup
        POW9 = torch.tensor([5**8, 5**7, 5**6, 5**5, 5**4, 5**3, 5**2, 5**1, 5**0],
                            dtype=torch.int32, device=device)
        suits = h34[:, :27].reshape(B, 3, 9).to(torch.int32)  # (B, 3, 9)
        suit_codes = (suits * POW9).sum(dim=2)  # (B, 3)

        POW7 = torch.tensor([5**6, 5**5, 5**4, 5**3, 5**2, 5**1, 5**0],
                            dtype=torch.int32, device=device)
        honors = h34[:, 27:34].to(torch.int32)  # (B, 7)
        honor_codes = (honors * POW7).sum(dim=1) + 1953125  # (B,)

        codes = torch.cat([suit_codes, honor_codes.unsqueeze(1)], dim=1)  # (B, 4)

        # Batch cache lookup: (CACHE[code >> 5] >> (code & 31)) & 1
        CACHE = Hand._get_cache(device)
        cache_idx = (codes >> 5).long().clamp(0, CACHE.shape[0] - 1)  # (B, 4)
        bit_idx = codes & 0b11111  # (B, 4)
        cache_vals = CACHE[cache_idx]  # (B, 4)  fancy-index gather
        valid_suits = ((cache_vals >> bit_idx) & 1).bool()  # (B, 4)

        # Head count
        suit_sums = suits.sum(dim=2)  # (B, 3)
        heads_suits = ((suit_sums % 3) == 2).sum(dim=1)  # (B,)
        heads_honors = (h34[:, 27:34] == 2).sum(dim=1)  # (B,)
        heads = heads_suits + heads_honors  # (B,)

        # Honor constraint: no 1 or 4 copies
        honor_ok = ((h34[:, 27:34] != 1) & (h34[:, 27:34] != 4)).all(dim=1)  # (B,)

        normal = valid_suits.all(dim=1) & (heads == 1) & honor_ok  # (B,)
        return normal | thirteen | seven_p  # (B,)

    @staticmethod
    def can_ron_batch(hands, tiles):
        """Vectorized: hands (B, 37), tiles (B,) → (B,) bool."""
        hands_with = Hand.add_batch(hands, tiles, 1)
        return Hand.can_tsumo_batch(hands_with)

    @staticmethod
    def can_closed_kan_batch(hands):
        """Vectorized: hands (B, 37) → (B, 34) bool mask for all tile types."""
        B = hands.shape[0]
        h34 = Hand.to_34_batch(hands)  # (B, 34)
        result = h34 == 4  # (B, 34)  basic: need 4 copies
        # Red-five adjustment: 3 normal + 1 red
        for tt in (Tile.BLACK_FIVE["m"], Tile.BLACK_FIVE["p"], Tile.BLACK_FIVE["s"]):
            red = Tile.to_red(tt)
            result[:, tt] = (hands[:, tt] == 3) & (hands[:, red] == 1)
        return result  # (B, 34)

    @staticmethod
    def can_added_kan_batch(hands):
        """Vectorized: hands (B, 37) → (B, 34) bool — need exactly 1 of the tile type."""
        h34 = Hand.to_34_batch(hands)  # (B, 34)
        return h34 == 1  # (B, 34)

    @staticmethod
    def can_kyuushu_batch(hands):
        """Vectorized: hands (B, 37) → (B,) bool."""
        h34 = Hand.to_34_batch(hands)
        mask = Hand.KYUUSHU_MASK.to(hands.device)
        return (h34 * mask > 0).sum(dim=1) >= 9  # (B,)

    # ── Multi-player batch helpers (used by meld mask phase) ──

    @staticmethod
    def can_no_red_pon_batch_4p(hands, target_tts):
        """hands: (B, 4, 37), target_tts: (B,) → returns (B, 4) bool.

        target_tts[b] is the tile-type of the discarded tile in env b (same for all 4 players).
        """
        B, P = hands.shape[0], hands.shape[1]
        h34 = Hand.to_34_batch(hands)  # (B, 4, 34)
        idx = target_tts.view(B, 1, 1).expand(B, P, 1)  # (B, 4, 1)
        counts = h34.gather(2, idx.long()).squeeze(2)  # (B, 4)
        return counts >= 2

    @staticmethod
    def can_red_pon_batch_4p(hands, target_tts):
        """hands: (B, 4, 37), target_tts: (B,) → returns (B, 4) bool."""
        B, P = hands.shape[0], hands.shape[1]
        device = hands.device
        result = torch.zeros(B, P, dtype=torch.bool, device=device)
        for tt in (4, 13, 22):
            is_target = (target_tts == tt).unsqueeze(1)  # (B, 1)
            if is_target.any():
                red = Tile.to_red(tt)
                result = result | (is_target & (hands[:, :, tt] > 0) & (hands[:, :, red] > 0))
        return result

    @staticmethod
    def can_open_kan_batch_4p(hands, target_tts):
        """hands: (B, 4, 37), target_tts: (B,) → returns (B, 4) bool.

        Need 3 copies of the target tile type (with red-five handling).
        """
        B, P = hands.shape[0], hands.shape[1]
        device = hands.device
        h34 = Hand.to_34_batch(hands)  # (B, 4, 34)
        idx = target_tts.view(B, 1, 1).expand(B, P, 1)  # (B, 4, 1)
        counts34 = h34.gather(2, idx.long()).squeeze(2)  # (B, 4)
        result = counts34 == 3  # basic: need exactly 3
        # Red-five adjustment for five types
        for tt in (4, 13, 22):
            is_target = (target_tts == tt).unsqueeze(1)  # (B, 1)
            if is_target.any():
                red = Tile.to_red(tt)
                # 3 copies total: either 3 normal, or 2 normal + 1 red
                alt_ok = ((hands[:, :, tt] == 2) & (hands[:, :, red] == 1))
                result = result | (is_target & alt_ok)
        return result

    @staticmethod
    def can_chi_matrix_batch_4p(hands, targets, src_mask):
        """hands: (B, 4, 37), targets: (B,) discard tiles, src_mask: (B, 4) bool.

        Returns (B, 4, 6) bool: chi_L, chi_L_RED, chi_M, chi_M_RED, chi_R, chi_R_RED.
        Only checks entries where src_mask[b, p] is True (src == 3, player to left).
        """
        B, P = hands.shape[0], hands.shape[1]
        device = hands.device
        result = torch.zeros(B, P, 6, dtype=torch.bool, device=device)

        target_tts = Tile.to_tile_type_tensor(targets)  # (B,)
        tt = target_tts  # (B,)  canonical tile type (0-33)

        # Only applicable to non-honor tiles (tt < 27)
        valid_tt = (tt < 27).unsqueeze(1)  # (B, 1)

        if not valid_tt.any():
            return result

        # Gather adjacent tile counts from the 37-type hand
        # For each chi direction, gather the required tiles
        for chi_idx in range(3):
            col_base = chi_idx * 2  # 0=CHI_L, 2=CHI_M, 4=CHI_R
            if chi_idx == 0:  # left: need tt+1, tt+2
                cond = valid_tt & (tt % 9 < 7).unsqueeze(1)  # (B, 1)
                t1 = (tt + 1).clamp(0, 36)
                t2 = (tt + 2).clamp(0, 36)
            elif chi_idx == 1:  # middle: need tt-1, tt+1
                cond = valid_tt & (tt % 9 > 0).unsqueeze(1) & (tt % 9 < 8).unsqueeze(1)  # (B, 1)
                t1 = (tt - 1).clamp(0, 36)
                t2 = (tt + 1).clamp(0, 36)
            else:  # right: need tt-2, tt-1
                cond = valid_tt & (tt % 9 > 1).unsqueeze(1)  # (B, 1)
                t1 = (tt - 2).clamp(0, 36)
                t2 = (tt - 1).clamp(0, 36)

            idx1 = t1.view(B, 1, 1).expand(B, P, 1)  # (B, 4, 1)
            idx2 = t2.view(B, 1, 1).expand(B, P, 1)  # (B, 4, 1)

            has_t1 = hands.gather(2, idx1.long()).squeeze(2) > 0  # (B, 4)
            has_t2 = hands.gather(2, idx2.long()).squeeze(2) > 0  # (B, 4)

            base_ok = cond & src_mask & has_t1 & has_t2  # (B, 4)
            result[:, :, col_base] = base_ok  # non-red chi

            # Red chi: additionally need red five in either t1 or t2 position
            has_red = torch.zeros(B, P, dtype=torch.bool, device=device)
            for tt_check in (4, 13, 22):
                red = Tile.to_red(tt_check)
                # Check if tt_check matches t1 or t2
                is_t1_target = (t1 == tt_check).unsqueeze(1)  # (B, 1)
                is_t2_target = (t2 == tt_check).unsqueeze(1)  # (B, 1)
                has_red = has_red | (is_t1_target & (hands[:, :, red] > 0))
                has_red = has_red | (is_t2_target & (hands[:, :, red] > 0))
            result[:, :, col_base + 1] = base_ok & has_red  # red chi

        return result  # (B, 4, 6)

    # ── Legacy single-env batch wrappers (kept for compatibility) ──

    @staticmethod
    def can_pon_batch(hands, tiles):
        """Vectorized: hands (B, 37), tiles (B,) → (B,) bool (pon = no_red_pon | red_pon)."""
        B = hands.shape[0]
        h34 = Hand.to_34_batch(hands)
        tt = Tile.to_tile_type_tensor(tiles)
        idx = tt.view(B, 1)  # (B, 1)
        no_red = h34.gather(1, idx.long()).squeeze(1) >= 2  # (B,)
        red = torch.zeros(B, dtype=torch.bool, device=hands.device)
        for tt5 in (4, 13, 22):
            is_target = tt == tt5
            if is_target.any():
                red5 = Tile.to_red(tt5)
                red = red | (is_target & (hands[:, tt5] > 0) & (hands[:, red5] > 0))
        return no_red | red

    @staticmethod
    def can_chi_any_batch(hands, tiles):
        """Vectorized: hands (B, 37), tiles (B,) → (B,) bool."""
        B = hands.shape[0]
        device = hands.device
        result = torch.zeros(B, dtype=torch.bool, device=device)
        tt = Tile.to_tile_type_tensor(tiles)
        # Only non-honor targets can be chi'd
        valid = tt < 27
        if not valid.any():
            return result
        # Check all 6 chi actions using per-tile adjacency
        for chi_idx in range(3):
            if chi_idx == 0:
                cond = valid & (tt % 9 < 7)
                t1, t2 = tt + 1, tt + 2
            elif chi_idx == 1:
                cond = valid & (tt % 9 > 0) & (tt % 9 < 8)
                t1, t2 = tt - 1, tt + 1
            else:
                cond = valid & (tt % 9 > 1)
                t1, t2 = tt - 2, tt - 1
            if cond.any():
                idx1 = t1.clamp(0, 36).view(B, 1)
                idx2 = t2.clamp(0, 36).view(B, 1)
                has_t1 = hands.gather(1, idx1.long()).squeeze(1) > 0
                has_t2 = hands.gather(1, idx2.long()).squeeze(1) > 0
                result = result | (cond & has_t1 & has_t2)
        return result

    @staticmethod
    def can_open_kan_batch(hands, tiles):
        """Vectorized: hands (B, 37), tiles (B,) → (B,) bool."""
        B = hands.shape[0]
        h34 = Hand.to_34_batch(hands)
        tt = Tile.to_tile_type_tensor(tiles)
        idx = tt.view(B, 1)
        counts = h34.gather(1, idx.long()).squeeze(1)  # (B,)
        result = counts == 3
        for tt5 in (4, 13, 22):
            is_target = tt == tt5
            if is_target.any():
                red5 = Tile.to_red(tt5)
                result = result | (is_target & (hands[:, tt5] == 2) & (hands[:, red5] == 1))
        return result

    @staticmethod
    def can_riichi_batch(hands_37):
        """Vectorized: hands_37 (B, 37) → (B,) bool.

        Checks if any discard from the hand leaves it tenpai.
        Uses batched Shanten.number_batch with early exit.
        """
        B = hands_37.shape[0]
        device = hands_37.device
        h34 = Hand.to_34_batch(hands_37)  # (B, 34)
        from .shanten import Shanten

        results = torch.zeros(B, dtype=torch.bool, device=device)

        # For each tile type present in any hand, try discarding one
        for t in range(34):
            has_t = h34[:, t] > 0
            check = has_t & ~results  # only hands not yet confirmed
            if not check.any():
                continue

            # Create hands with one of tile t removed
            alt = h34.clone()
            alt[check, t] -= 1
            shanten = Shanten.number_batch(alt)  # (B,) int
            results = results | (check & (shanten <= 0))
            if results.all():
                break

        return results

    @staticmethod
    def is_tenpai_batch(hands_34):
        """Vectorized: hands_34 (B, 34) → (B,) bool — check tenpai by trying each discard."""
        B = hands_34.shape[0]
        device = hands_34.device
        results = torch.zeros(B, dtype=torch.bool, device=device)
        # can_ron for each tile type: hand_34[t] < 4 and can_ron(hand_34, t)
        # We compute can_tsumo(add(hand_34, t)) in batch for all 34 tiles
        for t in range(34):
            can_add = (hands_34[:, t] < 4)
            if can_add.any():
                alt = hands_34.clone()
                alt[:, t] += 1
                # For can_ron check we need 37-type hands; convert by adding zeros for red slots
                alt_37 = torch.zeros(B, 37, dtype=hands_34.dtype, device=device)
                alt_37[:, :34] = alt
                can_ron_t = Hand.can_tsumo_batch(alt_37)
                results = results | (can_add & can_ron_t)
                if results.all():
                    break
        return results
