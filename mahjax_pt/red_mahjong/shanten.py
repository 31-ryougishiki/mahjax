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

    # ── Batch methods (vectorized) ──

    _SCACHE_DEVICE = None
    _SCACHE_LAST_DEVICE = None

    @staticmethod
    def _get_cache(device):
        if Shanten._SCACHE_DEVICE is None or Shanten._SCACHE_LAST_DEVICE != device:
            Shanten._SCACHE_DEVICE = Shanten.CACHE.to(device)
            Shanten._SCACHE_LAST_DEVICE = device
        return Shanten._SCACHE_DEVICE

    @staticmethod
    def seven_pairs_batch(hands_34):
        """Vectorized: hands_34 (B, 34) → (B,) int."""
        n_pair = (hands_34 >= 2).sum(dim=1)  # (B,)
        n_kind = (hands_34 > 0).sum(dim=1)   # (B,)
        return 7 - n_pair + torch.clamp(7 - n_kind, 0, 99)

    @staticmethod
    def thirteen_orphan_batch(hands_34):
        """Vectorized: hands_34 (B, 34) → (B,) int."""
        orphan_idx = THIRTEEN_ORPHAN_IDX.to(hands_34.device)
        th = hands_34[:, orphan_idx]  # (B, 13)
        n_pair = (th >= 2).sum(dim=1)  # (B,)
        n_kind = (th > 0).sum(dim=1)   # (B,)
        return 14 - n_kind - (n_pair > 0).int()

    @staticmethod
    def normal_batch(hands_34):
        """Vectorized: hands_34 (B, 34) → (B,) int — normal shanten using cache."""
        B = hands_34.shape[0]
        device = hands_34.device
        CACHE = Shanten._get_cache(device)
        J = CACHE.shape[1]  # == 9
        cache_flat = CACHE.reshape(-1)  # (N*9,)
        CACHE_N = CACHE.shape[0]

        # Encode 4 suits into base-5 codes: (B, 4)
        POW9 = torch.tensor([5**8, 5**7, 5**6, 5**5, 5**4, 5**3, 5**2, 5**1, 5**0],
                            dtype=torch.int32, device=device)
        suits = hands_34[:, :27].reshape(B, 3, 9).to(torch.int32)
        suit_codes = (suits * POW9).sum(dim=2)  # (B, 3)

        POW7 = torch.tensor([5**6, 5**5, 5**4, 5**3, 5**2, 5**1, 5**0],
                            dtype=torch.int32, device=device)
        honors = hands_34[:, 27:34].to(torch.int32)
        honor_codes = (honors * POW7).sum(dim=1) + 1953125  # (B,)

        codes = torch.cat([suit_codes, honor_codes.unsqueeze(1)], dim=1)  # (B, 4)

        # Gather from cache: lin = code * J + offset
        def _gather(offset):
            lin = (codes * J + offset).long().clamp(0, cache_flat.shape[0] - 1)
            return cache_flat[lin]  # (B, 4)

        # n_set = min(hand.sum() // 3, 4)
        n_sets = torch.clamp(hands_34.sum(dim=1).to(torch.int32) // 3, 0, 4)  # (B,)
        max_n_set = int(n_sets.max().item())

        # 4 variants: variant v means suit v is the "head" suit (idx starts at 5)
        best_costs = torch.full((B,), 99, dtype=torch.int32, device=device)

        for v in range(4):
            # Initial cost for variant v: gather base costs (idx=4 for all suits)
            var_cost = _gather(4)[:, v].clone()  # (B,)

            if max_n_set == 0:
                best_costs = torch.minimum(best_costs, var_cost)
                continue

            # Track idx per suit for ALL B envs simultaneously — but argmin picks
            # different suits per env. Unroll the set iteration (max 4).
            # Use (B, 4) idx tensor updated via advanced indexing.
            idx_tpl = torch.where(
                torch.arange(4, device=device).unsqueeze(0) == v,
                torch.tensor(5, dtype=torch.int32, device=device),
                torch.tensor(0, dtype=torch.int32, device=device),
            )  # (1, 4)
            var_idx = idx_tpl.expand(B, 4).clone()  # (B, 4)

            for t in range(max_n_set):
                # Gather candidate costs at current indices
                lin = (codes * J + var_idx).long().clamp(0, cache_flat.shape[0] - 1)
                cand = cache_flat[lin]  # (B, 4) — candidate cost for each suit

                # Best suit (min cost) for each env
                cand_min, best_suit = cand.min(dim=1)  # (B,), (B,)

                # Only apply delta for envs that still need this set
                active = (t < n_sets)  # (B,)
                var_cost = torch.where(active, var_cost + cand_min, var_cost)

                # Increment idx for the chosen suit
                b_idx = torch.arange(B, device=device)[active]
                chosen = best_suit[active]  # (num_active,)
                var_idx[b_idx, chosen] += 1

            best_costs = torch.minimum(best_costs, var_cost)

        return best_costs  # (B,)

    @staticmethod
    def number_batch(hands_34):
        """Vectorized: hands_34 (B, 34) → (B,) int — shanten number (0=tenpai, -1=complete)."""
        normal = Shanten.normal_batch(hands_34)
        seven = Shanten.seven_pairs_batch(hands_34)
        orphan = Shanten.thirteen_orphan_batch(hands_34)
        stacked = torch.stack([normal, seven, orphan], dim=1)  # (B, 3)
        return stacked.min(dim=1).values - 1  # (B,)

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
