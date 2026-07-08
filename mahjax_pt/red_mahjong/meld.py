# Copyright 2025 The Mahjax Authors.
# PyTorch eager-mode port of meld.py

import torch

from .types import Array
from .action import Action
from .tile import Tile

EMPTY_MELD = 0xFFFF


def _scalar(x):
    """If x is a tensor, return .item(), else return as-is."""
    if isinstance(x, torch.Tensor):
        return x.item()
    return x


class Meld:
    """Meld encoding: 16-bit packed integer (stored as int32 in PyTorch)."""

    @staticmethod
    def init(action: int, target: int, src: int) -> int:
        target_is_red = int(Tile.is_tile_red(target))
        target_tile_type = int(Tile.to_tile_type(target))
        return (
            (target_is_red << 15)
            | ((src & 0b11) << 13)
            | ((target_tile_type & 0b111111) << 7)
            | (action & 0b1111111)
        )

    @staticmethod
    def empty() -> int:
        return EMPTY_MELD

    @staticmethod
    def is_empty(meld) -> bool:
        return _scalar(meld) == EMPTY_MELD

    @staticmethod
    def is_target_red(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        return bool((m >> 15) & 0b1)

    @staticmethod
    def src(meld) -> int:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return -1
        return (m >> 13) & 0b11

    @staticmethod
    def target(meld) -> int:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return -1
        return (m >> 7) & 0b111111

    @staticmethod
    def target_tile(meld) -> int:
        target = Meld.target(meld)
        if Meld.is_target_red(meld):
            return Tile.to_red(target)
        return target

    @staticmethod
    def action(meld) -> int:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return -1
        return m & 0b1111111

    @staticmethod
    def _chi_index(action) -> int:
        a = _scalar(action)
        if a in (Action.CHI_L, Action.CHI_L_RED):
            return 0
        elif a in (Action.CHI_M, Action.CHI_M_RED):
            return 1
        elif a in (Action.CHI_R, Action.CHI_R_RED):
            return 2
        return -1

    @staticmethod
    def suited_pung(meld) -> int:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return 0
        action = Meld.action(m)
        target = Meld.target(m)
        is_pung = action in (Action.PON, Action.PON_RED, Action.OPEN_KAN) or Action.is_selfkan(action)
        if is_pung and target < 27:
            return 1 << target
        return 0

    @staticmethod
    def chow(meld) -> int:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return 0
        is_chi = Meld.is_chi(m)
        if not is_chi:
            return 0
        action = Meld.action(m)
        pos = Meld.target(m) - Meld._chi_index(action)
        return 1 << pos

    @staticmethod
    def is_kan(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        a = Meld.action(m)
        return a == Action.OPEN_KAN or Action.is_selfkan(a)

    @staticmethod
    def is_closed_kan(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        a = Meld.action(m)
        src = Meld.src(m)
        return src == 0 and Action.is_selfkan(a)

    @staticmethod
    def is_added_kan(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        src = Meld.src(m)
        a = Meld.action(m)
        return Action.is_selfkan(a) and src != 0

    @staticmethod
    def is_chi(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        a = Meld.action(m)
        return (Action.CHI_L <= a <= Action.CHI_R) or (Action.CHI_L_RED <= a <= Action.CHI_R_RED)

    @staticmethod
    def is_pon(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        a = Meld.action(m)
        return a in (Action.PON, Action.PON_RED)

    @staticmethod
    def is_outside(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return False
        is_pon_or_kan = Meld.is_pon(m) or Meld.is_kan(m)
        if not is_pon_or_kan:
            return False
        target = Meld.target(m)
        num = target % 9
        return target >= 27 or num == 0 or num == 8

    @staticmethod
    def has_outside(meld) -> bool:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return True  # empty melds don't prevent outside check
        if Meld.is_outside(m):
            return True
        target = Meld.target(m)
        action = Meld.action(m)
        num = target % 9
        chi_index = Meld._chi_index(action)
        if Meld.is_chi(m):
            if chi_index == 0 and (num == 0 or num == 6):
                return True
            if chi_index == 1 and (num == 1 or num == 7):
                return True
            if chi_index == 2 and (num == 2 or num == 8):
                return True
        return False

    @staticmethod
    def fu(meld) -> int:
        m = _scalar(meld)
        if m == EMPTY_MELD:
            return 0
        action = Meld.action(m)
        base = 0
        if Meld.is_pon(m):
            base = 2
        elif action == Action.OPEN_KAN:
            base = 8
        elif Action.is_selfkan(action):
            src = Meld.src(m)
            base = 8 * (2 if src == 0 else 1)
        return base * (2 if Meld.is_outside(m) else 1)

    @staticmethod
    def contains_red(meld) -> bool:
        m = _scalar(meld)
        action = Meld.action(m)
        target = Meld.target(m)
        if Meld.is_target_red(m):
            return True
        if action == Action.PON_RED:
            return True
        if action in (Action.OPEN_KAN,) or Action.is_selfkan(action):
            if Tile.is_tile_type_five(target):
                return True
        if action in (Action.CHI_L_RED, Action.CHI_M_RED, Action.CHI_R_RED):
            return True
        return False

    @staticmethod
    def exist_prohibitive_tile_type_after_chi(action, target) -> bool:
        chi_index = Meld._chi_index(action)
        if chi_index < 0:
            return False
        tt = Tile.to_tile_type(target)
        if chi_index == 0:
            return not Tile.is_tile_type_seven(tt)
        if chi_index == 2:
            return not Tile.is_tile_type_three(tt)
        return False

    @staticmethod
    def prohibitive_tile_type_after_chi(action, target) -> int:
        chi_index = Meld._chi_index(action)
        tt = int(Tile.to_tile_type(target))
        if Meld.exist_prohibitive_tile_type_after_chi(action, target):
            if chi_index == 0:
                return tt + 3
            elif chi_index == 2:
                return tt - 3
        return -1

    # ── Batch (tensor) versions — operate directly on tensors, no _scalar() ──

    @staticmethod
    def _chi_index_batch(actions):
        """Vectorized: actions (...,) int → (...,) int (0/1/2/-1)."""
        out = torch.full_like(actions, -1, dtype=torch.int32)
        out = torch.where((actions == Action.CHI_L) | (actions == Action.CHI_L_RED), 0, out)
        out = torch.where((actions == Action.CHI_M) | (actions == Action.CHI_M_RED), 1, out)
        out = torch.where((actions == Action.CHI_R) | (actions == Action.CHI_R_RED), 2, out)
        return out

    @staticmethod
    def init_batch(actions, targets, srcs):
        """Vectorized: (B,) actions/targets/srcs → (B,) int32 packed meld."""
        target_tt = Tile.to_tile_type_tensor(targets)
        is_red = Tile.is_tile_red_batch(targets).to(torch.int32)
        return ((is_red << 15) | ((srcs.int() & 0b11) << 13) |
                ((target_tt.int() & 0b111111) << 7) | (actions.int() & 0b1111111)).to(torch.int32)

    @staticmethod
    def is_empty_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        return melds == EMPTY_MELD

    @staticmethod
    def action_batch(melds):
        """Vectorized: melds (...,) → (...,) int (action code)."""
        return melds & 0b1111111

    @staticmethod
    def target_batch(melds):
        """Vectorized: melds (...,) → (...,) int (tile type 0-33)."""
        empty = melds == EMPTY_MELD
        return torch.where(empty, torch.full_like(melds, -1), (melds >> 7) & 0b111111)

    @staticmethod
    def src_batch(melds):
        """Vectorized: melds (...,) → (...,) int (0-3)."""
        empty = melds == EMPTY_MELD
        return torch.where(empty, torch.full_like(melds, -1), (melds >> 13) & 0b11)

    @staticmethod
    def is_target_red_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        empty = melds == EMPTY_MELD
        red = ((melds >> 15) & 0b1).bool()
        return red & ~empty

    @staticmethod
    def is_pon_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        a = melds & 0b1111111
        empty = melds == EMPTY_MELD
        return ((a == Action.PON) | (a == Action.PON_RED)) & ~empty

    @staticmethod
    def is_kan_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        a = melds & 0b1111111
        empty = melds == EMPTY_MELD
        return ((a == Action.OPEN_KAN) | ((a >= 37) & (a < 71))) & ~empty

    @staticmethod
    def is_closed_kan_batch(melds):
        """Vectorized: melds (...,) → (...,) bool (src=0, action=selfkan)."""
        a = melds & 0b1111111
        s = (melds >> 13) & 0b11
        empty = melds == EMPTY_MELD
        return (s == 0) & ((a >= 37) & (a < 71)) & ~empty

    @staticmethod
    def is_added_kan_batch(melds):
        """Vectorized: melds (...,) → (...,) bool (src!=0, action=selfkan)."""
        a = melds & 0b1111111
        s = (melds >> 13) & 0b11
        empty = melds == EMPTY_MELD
        return (s != 0) & ((a >= 37) & (a < 71)) & ~empty

    @staticmethod
    def is_chi_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        a = melds & 0b1111111
        empty = melds == EMPTY_MELD
        return ((a >= Action.CHI_L) & (a <= Action.CHI_R_RED)) & ~empty

    @staticmethod
    def is_outside_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        empty = melds == EMPTY_MELD
        is_pk = Meld.is_pon_batch(melds) | Meld.is_kan_batch(melds)
        target = (melds >> 7) & 0b111111
        num = target % 9
        return ((target >= 27) | (num == 0) | (num == 8)) & is_pk & ~empty

    @staticmethod
    def suited_pung_batch(melds):
        """Vectorized: melds (...,) → (...,) int (bitmask)."""
        empty = melds == EMPTY_MELD
        a = melds & 0b1111111
        target = (melds >> 7) & 0b111111
        is_pung = ((a == Action.PON) | (a == Action.PON_RED) | (a == Action.OPEN_KAN) |
                    ((a >= 37) & (a < 71)))
        suited = target < 27
        bit = torch.where(suited & is_pung, 1 << target, torch.zeros_like(melds))
        return torch.where(empty, torch.zeros_like(melds), bit)

    @staticmethod
    def chow_batch(melds):
        """Vectorized: melds (...,) → (...,) int (bitmask)."""
        empty = melds == EMPTY_MELD
        is_chi = Meld.is_chi_batch(melds)
        action = melds & 0b1111111
        target = (melds >> 7) & 0b111111
        chi_idx = Meld._chi_index_batch(action)
        pos = target - chi_idx
        return torch.where(is_chi & ~empty, 1 << pos, torch.zeros_like(melds))

    @staticmethod
    def fu_batch(melds):
        """Vectorized: melds (...,) → (...,) int."""
        empty = melds == EMPTY_MELD
        is_pon = Meld.is_pon_batch(melds)
        is_open_kan = (melds & 0b1111111) == Action.OPEN_KAN
        is_selfkan = ((melds & 0b1111111) >= 37) & ((melds & 0b1111111) < 71)
        src = (melds >> 13) & 0b11

        base = torch.zeros_like(melds, dtype=torch.int32)
        base = torch.where(is_pon, torch.full_like(base, 2), base)
        base = torch.where(is_open_kan, torch.full_like(base, 8), base)
        base = torch.where(is_selfkan, torch.where(src == 0, torch.full_like(base, 16), torch.full_like(base, 8)), base)

        is_out = Meld.is_outside_batch(melds)
        result = torch.where(is_out, base * 2, base)
        return torch.where(empty, torch.zeros_like(result), result)

    @staticmethod
    def contains_red_batch(melds):
        """Vectorized: melds (...,) → (...,) bool."""
        empty = melds == EMPTY_MELD
        is_red = Meld.is_target_red_batch(melds)
        a = melds & 0b1111111
        target = (melds >> 7) & 0b111111
        is_five = Tile.is_tile_type_five_batch(target)
        pon_red = a == Action.PON_RED
        is_open_kan = a == Action.OPEN_KAN
        is_selfkan = (a >= 37) & (a < 71)
        chi_red = ((a == Action.CHI_L_RED) | (a == Action.CHI_M_RED) | (a == Action.CHI_R_RED))
        kan_five = (is_open_kan | is_selfkan) & is_five
        return (is_red | pon_red | kan_five | chi_red) & ~empty

    @staticmethod
    def has_outside_batch(melds):
        """Vectorized: melds (...,) → (...,) bool. Empty melds → True (don't prevent outside)."""
        empty = melds == EMPTY_MELD
        is_out = Meld.is_outside_batch(melds)
        target = (melds >> 7) & 0b111111
        action = melds & 0b1111111
        num = target % 9
        chi_idx = Meld._chi_index_batch(action)
        is_chi = Meld.is_chi_batch(melds)
        # chi L (idx=0): outside if num==0 or num==6
        # chi M (idx=1): outside if num==1 or num==7
        # chi R (idx=2): outside if num==2 or num==8
        chi_outside = is_chi & (
            ((chi_idx == 0) & ((num == 0) | (num == 6))) |
            ((chi_idx == 1) & ((num == 1) | (num == 7))) |
            ((chi_idx == 2) & ((num == 2) | (num == 8)))
        )
        return (is_out | chi_outside | empty)

    @staticmethod
    def _calc_addition_batch(meld_vals):
        """Vectorized: meld_vals (B, MAX_MELDS) or (B,) → (B, 34) int8."""
        shape = meld_vals.shape
        if meld_vals.ndim == 1:
            meld_vals = meld_vals.unsqueeze(0)
        B, M = meld_vals.shape
        device = meld_vals.device
        addition = torch.zeros(B, 34, dtype=torch.int8, device=device)

        empty = meld_vals == EMPTY_MELD
        targets = (meld_vals >> 7) & 0b111111
        is_pon = Meld.is_pon_batch(meld_vals)
        is_kan = Meld.is_kan_batch(meld_vals)
        is_chi = Meld.is_chi_batch(meld_vals)
        actions = meld_vals & 0b1111111
        chi_idx = Meld._chi_index_batch(actions)

        # Pon: 3 tiles, Kan: 4 tiles
        add_count = torch.zeros_like(meld_vals, dtype=torch.int8)
        add_count = torch.where(is_pon, 3, add_count)
        add_count = torch.where(is_kan, 4, add_count)

        # Scatter pon/kan additions
        for j in range(M):
            valid = ~empty[:, j]
            if valid.any():
                b_idx = torch.arange(B, device=device)[valid]
                tgt = targets[valid, j]
                cnt = add_count[valid, j]
                addition[b_idx, tgt] += cnt.to(torch.int8)

        # Chi: 3 tiles starting at target - chi_idx
        for j in range(M):
            valid = is_chi[:, j] & ~empty[:, j]
            if valid.any():
                b_idx = torch.arange(B, device=device)[valid]
                tgt = targets[valid, j]
                cidx = chi_idx[valid, j]
                start = (tgt - cidx).clamp(0, 31)
                addition[b_idx, start] += 1
                addition[b_idx, start + 1] += 1
                addition[b_idx, start + 2] += 1

        return addition if shape[0] != B else addition
