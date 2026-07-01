# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

import torch
import numpy as np
from pathlib import Path
import importlib.resources as resources

from .types import Array
from .hand import THIRTEEN_ORPHAN_IDX


def _load_shanten_cache():
    """Load the shanten-cache from the local mahjax_pt package."""
    with resources.as_file(resources.files("mahjax_pt._src.cache").joinpath("shanten_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return torch.from_numpy(data["data"].astype(np.int64))


class Shanten:
    # Algorithm: https://github.com/sotetsuk/pgx/pull/123
    CACHE = _load_shanten_cache()

    @staticmethod
    def discard(hand):
        """For each of the 34 tile types, return the shanten number after discarding one."""
        results = torch.full((34,), 6, dtype=torch.int32)
        for i in range(34):
            if hand[i] != 0:
                h = hand.clone()
                h[i] -= 1
                results[i] = Shanten.number(h)
        return results  # (34,)

    @staticmethod
    def detailed_discard(hand):
        """For each of the 34 tile types, return (normal, 7pairs, 13orphan) after discarding one."""
        results = torch.full((34, 3), 6, dtype=torch.int32)
        for i in range(34):
            if hand[i] != 0:
                h = hand.clone()
                h[i] -= 1
                results[i] = Shanten.detailed_number(h)
        return results  # (34, 3)

    @staticmethod
    def number(hand):
        """Standard shanten number (0 = tenpai, -1 = complete)."""
        return min(
            Shanten.normal(hand),
            Shanten.seven_pairs(hand),
            Shanten.thirteen_orphan(hand),
        ) - 1

    @staticmethod
    def detailed_number(hand):
        return torch.tensor([
            Shanten.normal(hand),
            Shanten.seven_pairs(hand),
            Shanten.thirteen_orphan(hand),
        ], dtype=torch.int32)

    @staticmethod
    def seven_pairs(hand):
        n_pair = int((hand >= 2).sum().item())
        n_kind = int((hand > 0).sum().item())
        return 7 - n_pair + max(7 - n_kind, 0)

    @staticmethod
    def thirteen_orphan(hand):
        n_pair = int((hand[THIRTEEN_ORPHAN_IDX] >= 2).sum().item())
        n_kind = int((hand[THIRTEEN_ORPHAN_IDX] > 0).sum().item())
        return 14 - n_kind - (1 if n_pair > 0 else 0)

    @staticmethod
    def normal(hand):
        """Compute normal shanten number using the precomputed cache."""
        CACHE = Shanten.CACHE
        J = CACHE.shape[1]  # == 9

        def encode_suit(suit):
            if suit == 3:
                # Honors: run over indices 27..33
                code = 0
                for i in range(27, 34):
                    code = code * 5 + int(hand[i])
                return code + 1953125  # 5**9
            else:
                code = 0
                start = 9 * suit
                for i in range(start, start + 9):
                    code = code * 5 + int(hand[i])
                return code

        codes = [encode_suit(s) for s in range(4)]  # length 4

        n_set = min(int(hand.sum().item()) // 3, 4)

        def gather_elem(c, idx):
            lin = c * J + idx
            return int(CACHE.reshape(-1)[lin].item())

        # Base costs for 4 suits
        costs = [gather_elem(codes[s], 4) for s in range(4)]

        # For each set, pick the suit that gives the minimum cost increase
        idx = [[5 if s == v else 0 for s in range(4)] for v in range(4)]  # (4 variants, 4 suits)

        for t in range(4):
            if t >= n_set:
                break
            # Get candidate costs for all 4 variants × 4 suits
            cand = [[gather_elem(codes[s], idx[v][s]) for s in range(4)] for v in range(4)]
            for v in range(4):
                pick = min(range(4), key=lambda s: cand[v][s])
                delta = cand[v][pick]
                costs[v] += delta
                idx[v][pick] += 1

        return min(costs)
