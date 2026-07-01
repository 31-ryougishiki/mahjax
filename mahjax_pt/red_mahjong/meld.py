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
