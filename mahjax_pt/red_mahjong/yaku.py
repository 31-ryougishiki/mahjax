"""Yaku (役) judgement and score calculation — PyTorch eager port.

Ported from mahjax/red_mahjong/yaku.py (JAX).

Key tensor shapes:
  - MAX_PATTERNS = 3: three hand decomposition patterns processed in parallel
  - NUM_TENHOU_YAKU = 52: all possible yaku
  - Pattern vectors: (3,) tensors for per-pattern values
  - Yaku vectors: (52,) or (52, 3) for per-yaku booleans
"""

import torch
import numpy as np
from pathlib import Path
from functools import reduce
import operator
import importlib.resources as resources

from .constants import DORA_ARRAY
from .hand import Hand
from .meld import EMPTY_MELD, Meld
from .tile import Tile
from .types import Array

# ── Cache loading ────────────────────────────────────────────
def _load_yaku_cache():
    with resources.as_file(resources.files("mahjax_pt._src.cache").joinpath("yaku_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return torch.from_numpy(data["data"].astype(np.int64))

# ── Constants ────────────────────────────────────────────────
# Powers of 5 for hand-pattern encoding
POWERS_OF_5 = torch.cat([5 ** torch.arange(8, -1, -1)] * 3)  # (27,)

WIND_TILE = torch.tensor([27, 28, 29, 30], dtype=torch.long)
OUTSIDE_TILE = torch.tensor([0, 8, 9, 17, 18, 26], dtype=torch.long)
TANYAO_TILE = torch.tensor(
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25],
    dtype=torch.long,
)
KOKUSHI_TILE = torch.tensor([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=torch.long)
ALL_GREEN_TILE = torch.tensor([19, 20, 21, 23, 25, 32], dtype=torch.long)
SCORES = torch.tensor(
    [2000, 2000, 3000, 3000, 4000, 4000, 4000, 6000, 6000, 8000, 8000, 8000],
    dtype=torch.int32,
)
NUM_TENHOU_YAKU = 52


def _dora_array_from_state(state):
    """Compute dora + ura-dora count vectors from round state: returns (2, 34)."""
    dora_counts = torch.zeros(Tile.NUM_TILE_TYPE, dtype=torch.int8)
    for dora_indicator in state.round_state.dora_indicators:
        di = int(dora_indicator)
        if di != -1:
            dt = int(Tile.to_tile_type(di))
            dora_counts[int(DORA_ARRAY[dt])] += 1

    ura_dora_counts = torch.zeros(Tile.NUM_TILE_TYPE, dtype=torch.int8)
    for di in state.round_state.ura_dora_indicators:
        di = int(di)
        if di != -1:
            dt = int(Tile.to_tile_type(di))
            ura_dora_counts[int(DORA_ARRAY[dt])] += 1

    return torch.stack([dora_counts, ura_dora_counts], dim=0)


# ── Yaku index constants ─────────────────────────────────────
class YI:
    """Yaku indices — mirrors JAX _Internal."""
    FullyConcealedHand = 0
    Riichi = 1
    Ippatsu = 2
    RobbingKan = 3
    DrawAfterKan = 4
    BottomOfTheSea = 5
    BottomOfTheRiver = 6
    Pinfu = 7
    AllSimples = 8
    PureDoubleChis = 9
    SeatWindEast = 10
    SeatWindSouth = 11
    SeatWindWest = 12
    SeatWindNorth = 13
    PrevalentWindEast = 14
    PrevalentWindSouth = 15
    PrevalentWindWest = 16
    PrevalentWindNorth = 17
    WhiteDragon = 18
    GreenDragon = 19
    RedDragon = 20
    DoubleRiichi = 21
    SevenPairs = 22
    OutsideHand = 23
    PureStraight = 24
    MixedTripleChis = 25
    TriplePons = 26
    ThreeKans = 27
    AllPons = 28
    ThreeConcealedPons = 29
    LittleThreeDragons = 30
    AllTerminalsAndHonors = 31
    TwicePureDoubleChis = 32
    TerminalsInAllSets = 33
    HalfFlush = 34
    FullFlush = 35
    Renhou = 36
    BlessingOfHeaven = 37
    BlessingOfEarth = 38
    BigThreeDragons = 39
    FourConcealedPons = 40
    CompletedFourConcealedPons = 41
    AllHonors = 42
    AllGreen = 43
    AllTerminals = 44
    NineGates = 45
    PureNineGates = 46
    ThirteenOrphans = 47
    CompletedThirteenOrphans = 48
    BigFourWinds = 49
    LittleFourWinds = 50
    FourKans = 51
    MAX_PATTERNS = 3

# ── Fan / Yakuman tables ─────────────────────────────────────
def _make_fan_table():
    _FAN_OPEN = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.int32)
    _FAN_OPEN[YI.OutsideHand] = 1
    _FAN_OPEN[YI.TerminalsInAllSets] = 2
    _FAN_OPEN[YI.PureStraight] = 1
    _FAN_OPEN[YI.MixedTripleChis] = 1
    _FAN_OPEN[YI.TriplePons] = 2
    _FAN_OPEN[YI.AllPons] = 2
    _FAN_OPEN[YI.ThreeConcealedPons] = 2
    _FAN_OPEN[YI.ThreeKans] = 2
    _FAN_OPEN[YI.SevenPairs] = 2
    _FAN_OPEN[YI.AllSimples] = 1
    _FAN_OPEN[YI.HalfFlush] = 2
    _FAN_OPEN[YI.FullFlush] = 5
    _FAN_OPEN[YI.AllTerminalsAndHonors] = 2
    _FAN_OPEN[YI.LittleThreeDragons] = 2
    _FAN_OPEN[YI.WhiteDragon] = 1
    _FAN_OPEN[YI.GreenDragon] = 1
    _FAN_OPEN[YI.RedDragon] = 1
    for w in (YI.SeatWindEast, YI.SeatWindSouth, YI.SeatWindWest, YI.SeatWindNorth):
        _FAN_OPEN[w] = 1
    for w in (YI.PrevalentWindEast, YI.PrevalentWindSouth, YI.PrevalentWindWest, YI.PrevalentWindNorth):
        _FAN_OPEN[w] = 1

    _FAN_CLOSED = _FAN_OPEN.clone()
    _FAN_CLOSED[YI.FullyConcealedHand] = 1
    _FAN_CLOSED[YI.Riichi] = 1
    _FAN_CLOSED[YI.Pinfu] = 1
    _FAN_CLOSED[YI.PureDoubleChis] = 1
    _FAN_CLOSED[YI.TwicePureDoubleChis] = 3
    _FAN_CLOSED[YI.OutsideHand] = 2
    _FAN_CLOSED[YI.TerminalsInAllSets] = 3
    _FAN_CLOSED[YI.PureStraight] = 2
    _FAN_CLOSED[YI.MixedTripleChis] = 2
    _FAN_CLOSED[YI.HalfFlush] = 3
    _FAN_CLOSED[YI.FullFlush] = 6
    FAN = torch.stack([_FAN_OPEN, _FAN_CLOSED], dim=0)  # (2, 52)
    return FAN


def _make_yakuman_table():
    YM = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.int32)
    YM[YI.BigThreeDragons] = 1
    YM[YI.FourConcealedPons] = 1
    YM[YI.CompletedFourConcealedPons] = 1
    YM[YI.AllHonors] = 1
    YM[YI.AllGreen] = 1
    YM[YI.AllTerminals] = 1
    YM[YI.NineGates] = 1
    YM[YI.PureNineGates] = 1
    YM[YI.ThirteenOrphans] = 1
    YM[YI.CompletedThirteenOrphans] = 1
    YM[YI.BigFourWinds] = 2
    YM[YI.LittleFourWinds] = 1
    YM[YI.FourKans] = 1
    return YM


_FAN = _make_fan_table()           # (2, 52)
_YAKUMAN = _make_yakuman_table()   # (52,)

YAKU_UPDATE_INDICES = torch.tensor([
    YI.Pinfu, YI.PureDoubleChis, YI.TwicePureDoubleChis,
    YI.OutsideHand, YI.TerminalsInAllSets, YI.PureStraight,
    YI.MixedTripleChis, YI.TriplePons, YI.AllPons,
    YI.ThreeConcealedPons, YI.ThreeKans], dtype=torch.long)

YAKU_BEST_UPDATE_INDICES = torch.tensor([
    YI.AllSimples, YI.HalfFlush, YI.FullFlush, YI.AllTerminalsAndHonors,
    YI.WhiteDragon, YI.GreenDragon, YI.RedDragon, YI.LittleThreeDragons,
    YI.FullyConcealedHand, YI.Riichi], dtype=torch.long)

YAKUMAN_UPDATE_INDICES = torch.tensor([
    YI.BigThreeDragons, YI.BigFourWinds, YI.LittleFourWinds,
    YI.NineGates, YI.ThirteenOrphans, YI.AllTerminals,
    YI.AllHonors, YI.AllGreen, YI.FourConcealedPons, YI.FourKans], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════
# Yaku class — cache access & hand evaluation
# ═══════════════════════════════════════════════════════════════

class Yaku:
    CACHE = _load_yaku_cache()          # (1953125, 3)
    MAX_PATTERNS = YI.MAX_PATTERNS

    # ── cache field extractors ──
    # CACHE[code] yields (3,) row. All fields return (3,) int32 tensors.

    @staticmethod
    def head(code):
        """(3,) int — head position within the suit (0-8)."""
        return Yaku.CACHE[code] & 0b1111

    @staticmethod
    def chow(code):
        return (Yaku.CACHE[code] >> 4) & 0b1111111

    @staticmethod
    def pung(code):
        return (Yaku.CACHE[code] >> 11) & 0b111111111

    @staticmethod
    def n_pung(code):
        return (Yaku.CACHE[code] >> 20) & 0b111

    @staticmethod
    def n_double_chow(code):
        return (Yaku.CACHE[code] >> 23) & 0b11

    @staticmethod
    def outside(code):
        return (Yaku.CACHE[code] >> 25) & 1

    @staticmethod
    def nine_gates(code):
        return Yaku.CACHE[code] >> 26  # (3,) — gate pattern

    @staticmethod
    def is_pure_straight(chow_bits):
        """chow_bits: (3,) int32 — bit vector from chow().  Returns (3,) bool."""
        b = chow_bits.to(torch.int32)
        return ((b & 0b1001001) == 0b1001001) \
            | ((b >> 9 & 0b1001001) == 0b1001001) \
            | ((b >> 18 & 0b1001001) == 0b1001001)

    @staticmethod
    def is_triple_chow(chow_bits):
        """chow_bits: (3,) int32. Returns (3,) bool."""
        b = chow_bits.to(torch.int32)
        pat = 0b1000000001000000001
        out = (b & pat) == pat
        for s in range(1, 8):
            out = out | ((b >> s & pat) == pat)
        return out

    @staticmethod
    def is_triple_pung(pung_bits):
        """pung_bits: (3,) int32. Returns (3,) bool."""
        b = pung_bits.to(torch.int32)
        pat = 0b1000000001000000001
        out = (b & pat) == pat
        for s in range(1, 9):
            out = out | ((b >> s & pat) == pat)
        return out

    # ── meld helpers ──
    @staticmethod
    def _chi_index(action):
        if action in (78, 79): return 0
        if action in (80, 81): return 1
        if action in (82, 83): return 2
        return -1

    @staticmethod
    def _calc_addition(meld_val):
        """Return (34,) int8: extra tile counts contributed by one meld."""
        m = int(meld_val) if isinstance(meld_val, torch.Tensor) else meld_val
        if m == EMPTY_MELD:
            return torch.zeros(34, dtype=torch.int8)
        target = Meld.target(m)
        action = Meld.action(m)
        addition = torch.zeros(34, dtype=torch.int8)
        add = 3 if Meld.is_pon(m) else (4 if Meld.is_kan(m) else 0)
        addition[target] = add
        chi_idx = Yaku._chi_index(action)
        if Meld.is_chi(m):
            start = max(min(target - chi_idx, 31), 0)
            addition[start] += 1
            addition[start + 1] += 1
            addition[start + 2] += 1
        return addition

    @staticmethod
    def flatten(hand, melds, n_meld):
        """Combine concealed hand + melded tiles → 34-type count."""
        addition = torch.zeros(34, dtype=torch.int8)
        for i in range(melds.shape[0]):
            addition += Yaku._calc_addition(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i])
        return Hand.to_34(hand).to(torch.int32) + addition.to(torch.int32)

    @staticmethod
    def score(fan, fu):
        """Calculate payment amount from fan and fu."""
        fan = int(fan)
        fu = int(fu)
        if fu == 0:
            return 8000 * fan
        raw = fu * (1 << (fan + 2))
        if raw < 2000:
            return raw
        idx = min(max(fan - 4, 0), 11)
        return int(SCORES[idx].item())

    # ── Core pattern update ──────────────────────────────────
    @staticmethod
    def update(is_pinfu, has_outside, n_double_chow, all_chow, all_pung,
               n_concealed_pung, nine_gates, fu, code, suit, last_tile_type, is_ron):
        """Update pattern vectors with one suit's decomposition.

        All inputs/outputs except code, suit, last_tile_type, is_ron are (3,) tensors.
        Returns updated (3,) tensors.
        """
        code = int(code)
        suit = int(suit)
        lt = int(last_tile_type)
        ir = bool(is_ron)

        chow = Yaku.chow(code)        # (3,)
        pung = Yaku.pung(code)        # (3,)
        head = Yaku.head(code)        # (3,)
        n_pung = Yaku.n_pung(code)    # (3,)
        n_dc = Yaku.n_double_chow(code)  # (3,)
        outside_flag = Yaku.outside(code)  # (3,)
        ng = Yaku.nine_gates(code)    # (3,)

        open_end = ((chow ^ (chow & 1)) << 2) | (chow ^ (chow & 0b1000000))

        in_range = suit == (lt // 9)
        pos = lt % 9

        # Pinfu update — parentheses must match JAX: (~in_range | open_end_check) & (pung == 0)
        is_pinfu = is_pinfu & ((~in_range | (((open_end >> pos) & 1) == 1)) & (pung == 0))

        # Outside
        has_outside = has_outside & (outside_flag == 1)

        # Double chow count
        n_double_chow = n_double_chow + n_dc

        # Accumulate chow / pung bits across suits
        all_chow = all_chow | (chow << (9 * suit))
        all_pung = all_pung | (pung << (9 * suit))

        chow_range = chow | (chow << 1) | (chow << 2)
        loss = ir & in_range & (((chow_range >> pos) & 1) == 0) & (((pung >> pos) & 1) == 1)
        n_concealed_pung = n_concealed_pung + n_pung - loss.to(torch.int32)

        nine_gates = nine_gates | (ng == 1)

        outside_pung = pung & 0b100000001
        n_outside_pung = (outside_pung & 1) + ((outside_pung >> 8) & 1)

        strong = (
            in_range
            & ((1 << head) | ((chow & 1) << 2) | (chow & 0b1000000) | (chow << 1))
            >> pos & 1
        )
        outside_loss = loss & ((outside_pung >> pos) & 1)

        fu = fu + 4 * n_pung + 4 * n_outside_pung \
             - 2 * loss.to(torch.int32) - 2 * outside_loss.to(torch.int32) \
             + 2 * strong.to(torch.int32)

        return is_pinfu, has_outside, n_double_chow, all_chow, all_pung, \
               n_concealed_pung, nine_gates, fu

    # ── Batch cache extractors ──
    # CACHE[code] → (3,) packed int. codes: (B,) → cached: (B, 3).

    @staticmethod
    def head_batch(codes):
        return Yaku.CACHE[codes] & 0b1111

    @staticmethod
    def chow_batch(codes):
        return (Yaku.CACHE[codes] >> 4) & 0b1111111

    @staticmethod
    def pung_batch(codes):
        return (Yaku.CACHE[codes] >> 11) & 0b111111111

    @staticmethod
    def n_pung_batch(codes):
        return (Yaku.CACHE[codes] >> 20) & 0b111

    @staticmethod
    def n_double_chow_batch(codes):
        return (Yaku.CACHE[codes] >> 23) & 0b11

    @staticmethod
    def outside_batch(codes):
        return (Yaku.CACHE[codes] >> 25) & 1

    @staticmethod
    def nine_gates_batch(codes):
        return Yaku.CACHE[codes] >> 26

    # ── Batch pattern update (one suit, B envs) ──

    @staticmethod
    def update_batch(is_pinfu, has_outside, n_double_chow, all_chow, all_pung,
                     n_concealed_pung, nine_gates, fu, codes, suit, last_tile_type, is_ron):
        """Batch version of update. All pattern fields: (B, 3). codes: (B,) int."""
        B = codes.shape[0]
        device = codes.device

        chow = Yaku.chow_batch(codes)          # (B, 3)
        pung = Yaku.pung_batch(codes)          # (B, 3)
        head = Yaku.head_batch(codes)          # (B, 3)
        n_pung = Yaku.n_pung_batch(codes)      # (B, 3)
        n_dc = Yaku.n_double_chow_batch(codes) # (B, 3)
        outside_flag = Yaku.outside_batch(codes)  # (B, 3)
        ng = Yaku.nine_gates_batch(codes)      # (B, 3)

        open_end = ((chow ^ (chow & 1)) << 2) | (chow ^ (chow & 0b1000000))

        in_range = (suit == (last_tile_type // 9)).unsqueeze(1)  # (B, 1) → broadcast
        pos = last_tile_type % 9  # (B,)

        # Pinfu
        open_end_check = ((open_end >> pos.unsqueeze(1)) & 1) == 1  # (B, 3)
        is_pinfu = is_pinfu & ((~in_range | open_end_check) & (pung == 0))

        # Outside
        has_outside = has_outside & (outside_flag == 1)

        # Double chow count
        n_double_chow = n_double_chow + n_dc

        # Accumulate chow / pung bits
        all_chow = all_chow | (chow << (9 * suit))
        all_pung = all_pung | (pung << (9 * suit))

        chow_range = chow | (chow << 1) | (chow << 2)
        loss = is_ron.unsqueeze(1) & in_range & (((chow_range >> pos.unsqueeze(1)) & 1) == 0) & (((pung >> pos.unsqueeze(1)) & 1) == 1)
        n_concealed_pung = n_concealed_pung + n_pung - loss.to(torch.int32)

        nine_gates = nine_gates | (ng == 1)

        outside_pung = pung & 0b100000001
        n_outside_pung = (outside_pung & 1) + ((outside_pung >> 8) & 1)

        strong = (
            in_range
            & ((1 << head) | ((chow & 1) << 2) | (chow & 0b1000000) | (chow << 1))
            >> pos.unsqueeze(1) & 1
        )
        outside_loss = loss & ((outside_pung >> pos.unsqueeze(1)) & 1)

        fu = fu + 4 * n_pung + 4 * n_outside_pung \
             - 2 * loss.to(torch.int32) - 2 * outside_loss.to(torch.int32) \
             + 2 * strong.to(torch.int32)

        return is_pinfu, has_outside, n_double_chow, all_chow, all_pung, \
               n_concealed_pung, nine_gates, fu

    # ── Batch flatten ──

    @staticmethod
    def flatten_batch(hands_34, melds, n_meld):
        """Combine concealed hand + melded tiles → (B, 34) counts.

        hands_34: (B, 34) int32 — concealed hand
        melds: (B, MAX_MELDS) int32 — packed melds
        n_meld: (B,) int — number of melds
        """
        B = hands_34.shape[0]
        addition = Meld._calc_addition_batch(melds)  # (B, 34) int8
        return hands_34.to(torch.int32) + addition.to(torch.int32)

    # ═════════════════════════════════════════════════════════
    # Batch judge_hand_related
    # ═════════════════════════════════════════════════════════

    @staticmethod
    def judge_hand_related_batch(hands_37, melds, n_meld, last_tiles, riichi, is_ron,
                                  prevalent_winds, seat_winds, dora_indicators, ura_dora_indicators):
        """Batch yaku judgement for B hands.

        Args:
            hands_37: (B, 37) int8 — hand tile counts
            melds: (B, 4) int32 — packed meld values
            n_meld: (B,) int32 — number of melds
            last_tiles: (B,) int32 — winning tile (target for ron, last_draw for tsumo)
            riichi: (B,) bool
            is_ron: (B,) bool
            prevalent_winds: (B,) int32 — round wind (0-3)
            seat_winds: (B,) int32 — player seat wind (0-3)
            dora_indicators: (B, 5) int8 — dora indicator tiles
            ura_dora_indicators: (B, 5) int8 — ura dora indicator tiles

        Returns:
            yaku_vecs: (B, 52) bool
            fan: (B,) int32
            fu: (B,) int32
        """
        B = hands_37.shape[0]
        device = hands_37.device
        b_idx = torch.arange(B, device=device)

        # ── 1. Add winning tile to hand ──
        hands_37 = Hand.add_batch(hands_37, last_tiles)

        # ── 2. Red fan ──
        red_fan = hands_37[:, Tile.NUM_TILE_TYPE:].sum(dim=1).to(torch.int32)  # (B,)
        # Add red from melds
        for j in range(melds.shape[1]):
            m = melds[:, j]
            valid = ~(m == EMPTY_MELD)
            if valid.any():
                red_fan = red_fan + Meld.contains_red_batch(m).to(torch.int32) * valid.to(torch.int32)

        last_tile_types = Tile.to_tile_type_tensor(last_tiles).long()  # (B,)
        hands_34 = Hand.to_34_batch(hands_37)  # (B, 34)

        # ── 3. Dora ──
        dora = torch.zeros(B, 2, 34, dtype=torch.int8, device=device)
        for k in range(5):  # up to 5 dora indicators
            di = dora_indicators[:, k]  # (B,)
            valid = di != -1
            if valid.any():
                dt = Tile.to_tile_type_tensor(torch.where(valid, di, torch.zeros_like(di)))
                dora_idx = DORA_ARRAY[dt.long()]  # (B,)
                # Scatter: dora[b, 0, dora_idx[b]] += 1
                valid_idx = b_idx[valid]
                dora[valid_idx, 0, dora_idx[valid_idx].long()] += 1

            udi = ura_dora_indicators[:, k]  # (B,)
            valid_u = udi != -1
            if valid_u.any():
                dt_u = Tile.to_tile_type_tensor(torch.where(valid_u, udi, torch.zeros_like(udi)))
                dora_idx_u = DORA_ARRAY[dt_u.long()]
                valid_u_idx = b_idx[valid_u]
                dora[valid_u_idx, 1, dora_idx_u[valid_u_idx].long()] += 1

        # Riichi: sum normal + ura dora (riichi.unsqueeze(1) for per-env select)
        dora_vec = torch.where(riichi.unsqueeze(1),
                              dora.sum(dim=1),  # (B, 34)
                              dora[:, 0, :])    # (B, 34)

        wind_tile_t = WIND_TILE.to(device)
        seat_wind_tt = wind_tile_t[seat_winds.long()]  # (B,)
        prevalent_wind_tt = wind_tile_t[prevalent_winds.long()]  # (B,)

        # ── 4. Concealed check ──
        is_hand_concealed = torch.ones(B, dtype=torch.bool, device=device)
        for j in range(melds.shape[1]):
            m = melds[:, j]
            not_empty = ~(m == EMPTY_MELD)
            not_closed = ~Meld.is_closed_kan_batch(m)
            is_hand_concealed = is_hand_concealed & ~(not_empty & not_closed)

        # ── 5. Initial pattern vectors (B, 3) ──
        is_pinfu_init = (
            is_hand_concealed
            & (n_meld == 0)
            & (last_tile_types < 27)
            & (hands_34[:, 27:31] < 3).all(dim=1)
            & (hands_34[b_idx, seat_wind_tt] == 0)
            & (hands_34[b_idx, prevalent_wind_tt] == 0)
            & (hands_34[:, 31:34] == 0).all(dim=1)
        ).unsqueeze(1).expand(B, 3)  # (B, 3)

        # has_outside: all melds have outside (empty melds → True)
        has_outside_init = torch.ones(B, dtype=torch.bool, device=device)
        for j in range(melds.shape[1]):
            m = melds[:, j]
            has_outside_init = has_outside_init & Meld.has_outside_batch(m)
        has_outside_init = has_outside_init.unsqueeze(1).expand(B, 3)  # (B, 3)

        # Meld chow/pung bits
        meld_chow_bits = torch.zeros(B, dtype=torch.int32, device=device)
        meld_pung_bits = torch.zeros(B, dtype=torch.int32, device=device)
        for j in range(melds.shape[1]):
            m = melds[:, j]
            valid = ~(m == EMPTY_MELD)
            if valid.any():
                meld_chow_bits = meld_chow_bits | Meld.chow_batch(m)
                meld_pung_bits = meld_pung_bits | Meld.suited_pung_batch(m)

        all_chow = meld_chow_bits.unsqueeze(1).expand(B, 3)  # (B, 3)
        all_pung = meld_pung_bits.unsqueeze(1).expand(B, 3)  # (B, 3)

        # n_kan, n_closed_kan
        n_kan = torch.zeros(B, dtype=torch.int32, device=device)
        n_closed_kan = torch.zeros(B, dtype=torch.int32, device=device)
        for j in range(melds.shape[1]):
            m = melds[:, j]
            valid = ~(m == EMPTY_MELD)
            if valid.any():
                n_kan = n_kan + Meld.is_kan_batch(m).to(torch.int32)
                n_closed_kan = n_closed_kan + Meld.is_closed_kan_batch(m).to(torch.int32)

        # n_concealed_pung
        base_concealed = (hands_34[:, 27:] >= 3).sum(dim=1)  # (B,)
        penalty = is_ron & (last_tile_types >= 27) & (hands_34[b_idx, last_tile_types] >= 3)
        n_concealed_pung_val = base_concealed - penalty.to(torch.int32) + n_closed_kan
        n_concealed_pung = n_concealed_pung_val.unsqueeze(1).expand(B, 3)  # (B, 3)

        nine_gates = torch.zeros(B, 3, dtype=torch.bool, device=device)

        # ── 6. Wind pair / dragon pair / honor pon fu ──
        seat_pair = hands_34[b_idx, seat_wind_tt] == 2  # (B,)
        prev_pair = hands_34[b_idx, prevalent_wind_tt] == 2  # (B,)
        renfu_pair = (seat_wind_tt == prevalent_wind_tt) & seat_pair
        # If renfu: 4 fu. Otherwise: 2 per wind pair (both can apply).
        base_wind_fu = (torch.where(seat_pair, 2, 0) + torch.where(prev_pair, 2, 0))
        wind_pair_fu = torch.where(renfu_pair, torch.full_like(base_wind_fu, 4), base_wind_fu)

        dragon_pair_fu = torch.where((hands_34[:, 31:34] == 2).any(dim=1), 2,
                                     torch.zeros(B, dtype=torch.int32, device=device))

        honor_pon_fu = torch.zeros(B, dtype=torch.int32, device=device)
        for i in range(27, 34):
            count = hands_34[:, i]  # (B,)
            is_three = count == 3
            if is_three.any():
                # Serial: 4 * (2 - penalty), penalty=1 if ron on this tile else 0
                # → match=4, no_match=8
                mult = torch.where(is_ron & (last_tile_types == i),
                                   torch.tensor(4, dtype=torch.int32, device=device),
                                   torch.tensor(8, dtype=torch.int32, device=device))
                honor_pon_fu = honor_pon_fu + is_three.to(torch.int32) * mult

        # Base fu
        meld_fu_sum = torch.zeros(B, dtype=torch.int32, device=device)
        for j in range(melds.shape[1]):
            m = melds[:, j]
            valid = ~(m == EMPTY_MELD)
            if valid.any():
                meld_fu_sum = meld_fu_sum + Meld.fu_batch(m).to(torch.int32)

        base_fu = (torch.where(~is_ron, 2, 0)
                   + meld_fu_sum
                   + wind_pair_fu
                   + dragon_pair_fu
                   + honor_pon_fu
                   + torch.where((last_tile_types >= 27) & (hands_34[b_idx, last_tile_types] == 2), 1, 0))

        fu = base_fu.unsqueeze(1).expand(B, 3)  # (B, 3)

        # ── 7. Suit codes ──
        POW9 = torch.tensor([5**8, 5**7, 5**6, 5**5, 5**4, 5**3, 5**2, 5**1, 5**0],
                            dtype=torch.int32, device=device)
        suits = hands_34[:, :27].reshape(B, 3, 9).to(torch.int32)  # (B, 3, 9)
        codes = (suits * POW9).sum(dim=2)  # (B, 3)

        n_double_chow = torch.zeros(B, 3, dtype=torch.int32, device=device)

        # Process each suit
        for suit in range(3):
            c = codes[:, suit]  # (B,)
            is_pinfu_init, has_outside_init, n_double_chow, all_chow, all_pung, \
                n_concealed_pung, nine_gates, fu = Yaku.update_batch(
                    is_pinfu_init, has_outside_init, n_double_chow, all_chow, all_pung,
                    n_concealed_pung, nine_gates, fu,
                    c, suit, last_tile_types, is_ron)

        # ── 8. Fu cleanup ──
        fu = fu * (is_pinfu_init == 0).to(torch.int32)
        fu = fu + 20 + 10 * (is_hand_concealed & is_ron).unsqueeze(1).to(torch.int32)
        # Kuipin: open hand with fu==20 → min 30 fu
        is_open = (~is_hand_concealed).unsqueeze(1)
        fu = fu + 10 * (is_open & (fu == 20)).to(torch.int32)

        # ── 9. Global shape features ──
        flatten = Yaku.flatten_batch(hands_34, melds, n_meld)  # (B, 34)

        four_winds = (flatten[:, 27:31] >= 3).sum(dim=1)  # (B,)
        three_dragons = (flatten[:, 31:34] >= 3).sum(dim=1)  # (B,)
        has_tanyao = (flatten[:, TANYAO_TILE] > 0).any(dim=1)  # (B,)
        has_honor = (flatten[:, 27:34] > 0).any(dim=1)  # (B,)
        has_outside_in_flatten = (flatten[:, OUTSIDE_TILE] > 0).any(dim=1)  # (B,)

        suit_count = ((flatten[:, 0:9] > 0).any(dim=1).to(torch.int32)
                      + (flatten[:, 9:18] > 0).any(dim=1).to(torch.int32)
                      + (flatten[:, 18:27] > 0).any(dim=1).to(torch.int32))
        is_flush = suit_count == 1  # (B,)

        # ── 10. Build yaku matrix (B, 52, 3) ──
        yaku = torch.zeros(B, NUM_TENHOU_YAKU, 3, dtype=torch.bool, device=device)

        yaku[:, YI.Pinfu] = is_pinfu_init  # (B, 3)
        yaku[:, YI.PureDoubleChis] = is_hand_concealed.unsqueeze(1).expand(B, 3) & (n_double_chow == 1)
        yaku[:, YI.TwicePureDoubleChis] = n_double_chow == 2
        yaku[:, YI.OutsideHand] = has_outside_init & has_honor.unsqueeze(1) & has_tanyao.unsqueeze(1)
        yaku[:, YI.TerminalsInAllSets] = has_outside_init & (~has_honor.unsqueeze(1))

        # PureStraight: per-pattern chow check
        for p in range(3):
            cb = all_chow[:, p]  # (B,)
            yaku[:, YI.PureStraight, p] = ((cb & 0b1001001) == 0b1001001) \
                | ((cb >> 9 & 0b1001001) == 0b1001001) \
                | ((cb >> 18 & 0b1001001) == 0b1001001)

        # TripleChow: per-pattern
        for p in range(3):
            cb = all_chow[:, p]
            pat = 0b1000000001000000001
            out = (cb & pat) == pat
            for s in range(1, 8):
                out = out | ((cb >> s & pat) == pat)
            yaku[:, YI.MixedTripleChis, p] = out

        # TriplePung: per-pattern
        for p in range(3):
            pb = all_pung[:, p]
            pat = 0b1000000001000000001
            out = (pb & pat) == pat
            for s in range(1, 9):
                out = out | ((pb >> s & pat) == pat)
            yaku[:, YI.TriplePons, p] = out

        yaku[:, YI.AllPons] = all_chow == 0
        yaku[:, YI.ThreeConcealedPons] = n_concealed_pung == 3
        yaku[:, YI.ThreeKans] = n_kan.unsqueeze(1).expand(B, 3) == 3

        # ── 11. Pick best pattern ──
        fan_open = _FAN[0].unsqueeze(0)  # (1, 52)
        fan_conc = _FAN[1].unsqueeze(0)  # (1, 52)
        fan_selected = torch.where(is_hand_concealed.unsqueeze(1), fan_conc, fan_open)  # (B, 52)

        pattern_score = torch.zeros(B, 3, dtype=torch.int32, device=device)
        for p in range(3):
            pattern_score[:, p] = (fan_selected.to(torch.int32) * yaku[:, :, p].to(torch.int32)).sum(dim=1) * 200 + fu[:, p]

        best_pattern = torch.argmax(pattern_score, dim=1)  # (B,)
        yaku_best = yaku[b_idx, :, best_pattern]  # (B, 52)
        fu_best = fu[b_idx, best_pattern]  # (B,)
        fu_best = fu_best + (-fu_best % 10)  # round up to 10

        # ── 12. Seven-pairs override ──
        # Serial: is_mentsu_hand = twice_double_chis or (pairs < 7)
        # Seven pairs if: no twice_double_chis AND pairs >= 7
        pairs_count = (hands_34 == 2).sum(dim=1)  # (B,)
        is_mentsu_hand = yaku_best[:, YI.TwicePureDoubleChis] | (pairs_count < 7)
        seven_pairs_mask = ~is_mentsu_hand  # (B,)
        if seven_pairs_mask.any():
            si = b_idx[seven_pairs_mask]
            yaku_best[si] = False
            yaku_best[si, YI.SevenPairs] = True
            fu_best[si] = 25

        # ── 13. Best-only yaku updates ──
        yaku_best[:, YI.AllSimples] = ~(has_honor | has_outside_in_flatten)
        yaku_best[:, YI.HalfFlush] = is_flush & has_honor
        yaku_best[:, YI.FullFlush] = is_flush & (~has_honor)
        yaku_best[:, YI.AllTerminalsAndHonors] = ~has_tanyao
        yaku_best[:, YI.WhiteDragon] = flatten[:, 31] >= 3
        yaku_best[:, YI.GreenDragon] = flatten[:, 32] >= 3
        yaku_best[:, YI.RedDragon] = flatten[:, 33] >= 3
        yaku_best[:, YI.LittleThreeDragons] = (flatten[:, 31:34] >= 2).all(dim=1) & (three_dragons >= 2)
        yaku_best[:, YI.FullyConcealedHand] = is_hand_concealed & ~is_ron
        yaku_best[:, YI.Riichi] = riichi

        # Wind yakus
        for w in range(4):
            yaku_best[:, YI.PrevalentWindEast + w] = (prevalent_winds == w) & (flatten[:, WIND_TILE[w]] >= 3)
            yaku_best[:, YI.SeatWindEast + w] = (seat_winds == w) & (flatten[:, WIND_TILE[w]] >= 3)

        # ── 14. Yakuman check ──
        win_tile_count = hands_34[b_idx, last_tile_types]  # (B,)
        four_concealed_tsumo = (n_concealed_pung_val == 4) & (win_tile_count >= 3) & ~is_ron
        four_concealed_single = (n_concealed_pung_val == 4) & (win_tile_count == 2)

        yakuman = torch.zeros(B, NUM_TENHOU_YAKU, dtype=torch.bool, device=device)
        yakuman[:, YI.BigThreeDragons] = three_dragons == 3
        yakuman[:, YI.BigFourWinds] = four_winds == 4
        yakuman[:, YI.LittleFourWinds] = (flatten[:, 27:31] >= 2).all(dim=1) & (four_winds == 3)
        yakuman[:, YI.NineGates] = nine_gates.any(dim=1)
        kokushi_tiles = torch.tensor([0, 8, 9, 17, 18, 26] + list(range(27, 34)), dtype=torch.int64, device=device)
        yakuman[:, YI.ThirteenOrphans] = (hands_34[:, kokushi_tiles] > 0).all(dim=1) & ~has_tanyao
        yakuman[:, YI.AllTerminals] = ~has_tanyao & ~has_honor
        yakuman[:, YI.AllHonors] = (flatten[:, 0:27] == 0).all(dim=1)
        yakuman[:, YI.AllGreen] = flatten[:, ALL_GREEN_TILE].sum(dim=1) == 14
        yakuman[:, YI.FourConcealedPons] = four_concealed_tsumo
        yakuman[:, YI.CompletedFourConcealedPons] = four_concealed_single
        yakuman[:, YI.FourKans] = n_kan == 4

        yakuman_num = (yakuman.to(torch.int32) @ _YAKUMAN).to(torch.int32)  # (B,)

        has_yakuman = yakuman.any(dim=1)  # (B,)
        if has_yakuman.any():
            yi = b_idx[has_yakuman]
            yaku_best[yi] = yakuman[yi]
            fu_best[yi] = 0

        # Compute fan: yaku_fan + dora_fan + red_fan (matches serial L618-622)
        # Note: .sum() returns int64 (Long) in PyTorch, must explicitly cast to int32
        yaku_fan = (fan_selected * yaku_best.to(torch.int32)).sum(dim=1, dtype=torch.int32)  # (B,)
        dora_fan = (flatten.to(torch.int32) * dora_vec.to(torch.int32)).sum(dim=1, dtype=torch.int32)  # (B,)
        fan_val = yaku_fan + dora_fan + red_fan
        fan_val = torch.where(has_yakuman, yakuman_num.to(torch.int32), fan_val)

        return yaku_best, fan_val.to(torch.int32), fu_best.to(torch.int32)

    # ═════════════════════════════════════════════════════════
    # Main judge entry points
    # ═════════════════════════════════════════════════════════

    @staticmethod
    def judge_hand_related(hand, melds, n_meld, last_tile, riichi, is_ron,
                           prevalent_wind, seat_wind, dora):
        """Evaluate hand yaku.

        hand:      (37,) or (34,) tile counts
        melds:     (4,) packed meld ints
        n_meld:    int
        last_tile: int — winning tile
        riichi:    bool
        is_ron:    bool
        prevalent_wind: int
        seat_wind: int
        dora:      (2, 34) dora count matrix [normal, ura]

        Returns: (yaku_vec, fan, fu)
          yaku_vec: (52,) bool
          fan:      int
          fu:       int
        """
        is_ron = bool(is_ron)
        riichi = bool(riichi)
        n_meld = int(n_meld)
        last_tile = int(last_tile)
        prevalent_wind = int(prevalent_wind)
        seat_wind = int(seat_wind)

        # Add winning tile
        hand = Hand.add(hand, last_tile)

        red_fan = 0
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            red_fan = int(hand[Tile.NUM_TILE_TYPE:].sum().item())
            for i in range(melds.shape[0]):
                m = int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]
                if m != EMPTY_MELD and Meld.contains_red(m):
                    red_fan += 1
            hand = Hand.to_34(hand)
            last_tile_type = int(Tile.to_tile_type(last_tile))
        else:
            last_tile_type = int(last_tile)

        # Dora: riichi players also get ura-dora
        if riichi:
            dora_vec = dora.sum(dim=0)  # (34,)
        else:
            dora_vec = dora[0]          # (34,)

        seat_wind_tt = int(WIND_TILE[seat_wind])
        prevalent_wind_tt = int(WIND_TILE[prevalent_wind])

        # ── concealed check ──
        is_hand_concealed = True
        for i in range(melds.shape[0]):
            m = int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]
            if m != EMPTY_MELD and not Meld.is_closed_kan(m):
                is_hand_concealed = False
                break

        # ── initial pattern vectors (3,) ──
        is_pinfu = torch.tensor([
            is_hand_concealed and n_meld == 0 and last_tile_type < 27
            and bool((hand[27:31] < 3).all().item())
            and hand[seat_wind_tt] == 0 and hand[prevalent_wind_tt] == 0
            and bool((hand[31:34] == 0).all().item())
        ] * YI.MAX_PATTERNS)

        has_outside = torch.tensor([
            all(int(Meld.has_outside(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]))
                or (int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]) == EMPTY_MELD
                for i in range(melds.shape[0]))
        ] * YI.MAX_PATTERNS)

        # Meld chow / pung bitwise
        meld_chow_bits = 0
        meld_pung_bits = 0
        for i in range(melds.shape[0]):
            m = int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]
            meld_chow_bits |= Meld.chow(m)
            meld_pung_bits |= Meld.suited_pung(m)

        all_chow = torch.full((YI.MAX_PATTERNS,), meld_chow_bits, dtype=torch.int32)
        all_pung = torch.full((YI.MAX_PATTERNS,), meld_pung_bits, dtype=torch.int32)

        n_kan = sum(1 for i in range(melds.shape[0])
                    if (int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]) != EMPTY_MELD
                    and Meld.is_kan(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]))
        n_closed_kan = sum(1 for i in range(melds.shape[0])
                           if (int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]) != EMPTY_MELD
                           and Meld.is_closed_kan(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]))

        base_concealed = int((hand[27:] >= 3).sum().item())
        penalty = is_ron and last_tile_type >= 27 and hand[last_tile_type] >= 3
        n_concealed_pung_val = base_concealed - int(penalty) + n_closed_kan
        n_concealed_pung = torch.full((YI.MAX_PATTERNS,), n_concealed_pung_val, dtype=torch.int32)

        nine_gates = torch.full((YI.MAX_PATTERNS,), False)

        # ── wind pair fu ──
        seat_pair = hand[seat_wind_tt] == 2
        prev_pair = hand[prevalent_wind_tt] == 2
        renfu_pair = (seat_wind_tt == prevalent_wind_tt) and seat_pair
        if renfu_pair:
            wind_pair_fu = 4
        else:
            wind_pair_fu = (2 if seat_pair else 0) + (2 if prev_pair else 0)

        dragon_pair_fu = 2 if bool((hand[31:34] == 2).any().item()) else 0

        honor_pon_fu = sum(
            (int(hand[i].item()) == 3) * 4 * (2 - (is_ron and last_tile_type == i))
            for i in range(27, 34))

        # Base fu: menzen-kafu (10), tsumo (2), etc.
        base_fu = (2 if not is_ron else 0) \
            + sum(int(Meld.fu(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]))
                  for i in range(melds.shape[0])) \
            + wind_pair_fu \
            + dragon_pair_fu \
            + honor_pon_fu \
            + (1 if 27 <= last_tile_type and hand[last_tile_type] == 2 else 0)

        fu = torch.full((YI.MAX_PATTERNS,), base_fu, dtype=torch.int32)

        # ── suit codes ──
        codes = (hand[:27].to(torch.int32) * POWERS_OF_5).reshape(3, 9).sum(dim=1)  # (3,)

        n_double_chow = torch.zeros(YI.MAX_PATTERNS, dtype=torch.int32)

        # Process each suit
        for suit in range(3):
            code = int(codes[suit].item())
            is_pinfu, has_outside, n_double_chow, all_chow, all_pung, \
                n_concealed_pung, nine_gates, fu = Yaku.update(
                    is_pinfu, has_outside, n_double_chow, all_chow, all_pung,
                    n_concealed_pung, nine_gates, fu,
                    code, suit, last_tile_type, is_ron)

        # ── fu cleanup ──
        fu = fu * (is_pinfu == 0).to(torch.int32)
        fu = fu + 20 + 10 * int(is_hand_concealed and is_ron)
        # Kuipin correction: open hand with fu==20 → min 30 fu (per pattern, match JAX)
        fu = fu + 10 * ((not is_hand_concealed) & (fu == 20))

        # ── global shape features ──
        flatten = Yaku.flatten(hand, melds, n_meld)  # (34,)

        four_winds = int((flatten[27:31] >= 3).sum().item())
        three_dragons = int((flatten[31:34] >= 3).sum().item())
        has_tanyao = bool((flatten[TANYAO_TILE] > 0).any().item())
        has_honor = bool((flatten[27:34] > 0).any().item())
        has_outside_in_flatten = bool((flatten[OUTSIDE_TILE] > 0).any().item())

        suit_count = (int((flatten[0:9] > 0).any().item())
                      + int((flatten[9:18] > 0).any().item())
                      + int((flatten[18:27] > 0).any().item()))
        is_flush = suit_count == 1

        # ── Build yaku matrix (52, 3) ──
        yaku = torch.zeros(NUM_TENHOU_YAKU, YI.MAX_PATTERNS, dtype=torch.bool)

        # Basic shape yakus
        yaku[YI.Pinfu] = is_pinfu
        yaku[YI.PureDoubleChis] = torch.tensor([is_hand_concealed] * 3) & (n_double_chow == 1)
        yaku[YI.TwicePureDoubleChis] = n_double_chow == 2
        yaku[YI.OutsideHand] = has_outside & has_honor & has_tanyao
        yaku[YI.TerminalsInAllSets] = has_outside & (~has_honor)

        chow_bits = all_chow  # (3,)
        yaku[YI.PureStraight] = Yaku.is_pure_straight(chow_bits)
        yaku[YI.MixedTripleChis] = Yaku.is_triple_chow(chow_bits)
        yaku[YI.TriplePons] = Yaku.is_triple_pung(all_pung)
        yaku[YI.AllPons] = all_chow == 0
        yaku[YI.ThreeConcealedPons] = n_concealed_pung == 3
        yaku[YI.ThreeKans] = torch.full((3,), n_kan == 3)

        # ── Pick best pattern ──
        fan_row = _FAN[1 if is_hand_concealed else 0]  # (52,)
        pattern_score = (fan_row.unsqueeze(0) @ yaku.to(torch.int32)).squeeze(0) * 200 + fu  # (3,)
        best_pattern = int(torch.argmax(pattern_score).item())

        yaku_best = yaku[:, best_pattern]  # (52,)
        fu_best = int(fu[best_pattern].item())
        fu_best = fu_best + (-fu_best % 10)  # round up to 10

        # ── Seven-pairs override ──
        is_mentsu_hand = bool(yaku_best[YI.TwicePureDoubleChis].item()) or ((hand == 2).sum() < 7)
        if not is_mentsu_hand:
            yaku_best = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
            yaku_best[YI.SevenPairs] = True
            fu_best = 25

        # ── Best-only yaku updates ──
        yaku_best[YI.AllSimples] = not (has_honor or has_outside_in_flatten)
        yaku_best[YI.HalfFlush] = is_flush and has_honor
        yaku_best[YI.FullFlush] = is_flush and (not has_honor)
        yaku_best[YI.AllTerminalsAndHonors] = not has_tanyao
        yaku_best[YI.WhiteDragon] = flatten[31] >= 3
        yaku_best[YI.GreenDragon] = flatten[32] >= 3
        yaku_best[YI.RedDragon] = flatten[33] >= 3
        yaku_best[YI.LittleThreeDragons] = bool((flatten[31:34] >= 2).all().item()) and three_dragons >= 2
        yaku_best[YI.FullyConcealedHand] = is_hand_concealed and not is_ron
        yaku_best[YI.Riichi] = riichi

        # Wind yakus
        yaku_best[YI.PrevalentWindEast + prevalent_wind] = flatten[prevalent_wind_tt] >= 3
        yaku_best[YI.SeatWindEast + seat_wind] = flatten[seat_wind_tt] >= 3

        # ── Yakuman check ──
        win_tile_count = int(hand[last_tile_type].item())
        four_concealed_tsumo = (n_concealed_pung_val == 4) and win_tile_count >= 3 and not is_ron
        four_concealed_single = (n_concealed_pung_val == 4) and win_tile_count == 2

        yakuman = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        yakuman[YI.BigThreeDragons] = three_dragons == 3
        yakuman[YI.BigFourWinds] = four_winds == 4
        yakuman[YI.LittleFourWinds] = bool((flatten[27:31] >= 2).all().item()) and four_winds == 3
        yakuman[YI.NineGates] = bool(nine_gates.any().item())
        yakuman[YI.ThirteenOrphans] = bool((hand[KOKUSHI_TILE] > 0).all().item()) and not has_tanyao
        yakuman[YI.AllTerminals] = (not has_tanyao) and (not has_honor)
        yakuman[YI.AllHonors] = bool((flatten[0:27] == 0).all().item())
        yakuman[YI.AllGreen] = int(flatten[ALL_GREEN_TILE].sum().item()) == 14
        yakuman[YI.FourConcealedPons] = four_concealed_tsumo
        yakuman[YI.CompletedFourConcealedPons] = four_concealed_single
        yakuman[YI.FourKans] = n_kan == 4

        yakuman_num = int((yakuman.to(torch.int32) @ _YAKUMAN).item())

        if yakuman.any():
            return yakuman, yakuman_num, torch.tensor(0, dtype=torch.int32)

        fan_val = int((fan_row @ yaku_best.to(torch.int32)).item()) \
            + int((flatten * dora_vec).sum().item()) \
            + red_fan
        return yaku_best, fan_val, fu_best

    @staticmethod
    def judge(hand, is_ron, player, rs):
        """Full hand judgement entry point.

        Returns: (yaku_vec, fan, fu)
        """
        p = int(player)
        melds = rs.players.melds[p]
        n_meld = int(rs.players.meld_counts[p].item())
        last_tile = int(rs.round_state.target) if is_ron else int(rs.round_state.last_draw)
        riichi = bool(rs.players.riichi[p].item())
        prevalent_wind = int(rs.round_state.round) // 4
        seat_wind = int(rs.round_state.seat_wind[p].item())
        dora = _dora_array_from_state(rs)

        return Yaku.judge_hand_related(
            hand=hand, melds=melds, n_meld=n_meld, last_tile=last_tile,
            riichi=riichi, is_ron=is_ron, prevalent_wind=prevalent_wind,
            seat_wind=seat_wind, dora=dora)

    @staticmethod
    def judge_other(*, is_ron, is_riichi, is_ippatsu=False, is_robbing_kan=False,
                    is_after_kan=False, is_bottom_of_the_sea=False,
                    is_bottom_of_the_river=False, is_double_riichi=False,
                    is_blessing_of_heaven=False, is_blessing_of_earth=False):
        """Judge environment-triggered yakus (riichi, ippatsu, haitei, etc.).

        Returns: (normal_yaku, yakuman, normal_fan, yakuman_num)
        """
        is_ron = bool(is_ron)
        is_riichi = bool(is_riichi)
        is_ippatsu = bool(is_ippatsu) and is_riichi
        is_robbing_kan = bool(is_robbing_kan) and is_ron
        is_after_kan = bool(is_after_kan) and not is_ron
        is_bottom_of_the_sea = bool(is_bottom_of_the_sea) and not is_ron
        is_bottom_of_the_river = bool(is_bottom_of_the_river) and is_ron
        is_double_riichi = bool(is_double_riichi) and is_riichi
        is_blessing_of_heaven = bool(is_blessing_of_heaven) and not is_ron
        is_blessing_of_earth = bool(is_blessing_of_earth) and not is_ron

        normal = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        normal[YI.Ippatsu] = is_ippatsu
        normal[YI.RobbingKan] = is_robbing_kan
        normal[YI.DrawAfterKan] = is_after_kan
        normal[YI.BottomOfTheSea] = is_bottom_of_the_sea
        normal[YI.BottomOfTheRiver] = is_bottom_of_the_river
        normal[YI.DoubleRiichi] = is_double_riichi

        yakuman = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        yakuman[YI.BlessingOfHeaven] = is_blessing_of_heaven
        yakuman[YI.BlessingOfEarth] = is_blessing_of_earth

        normal_fan = int(normal.sum().item())
        yakuman_num = int(yakuman.sum().item())
        return normal, yakuman, normal_fan, yakuman_num

    @staticmethod
    def judge_yakuman(hand, is_ron, player, rs):
        """Quick yakuman-only check (used during init)."""
        p = int(player)
        melds = rs.players.melds[p]
        last_tile = int(rs.round_state.target) if is_ron else int(rs.round_state.last_draw)
        hand = Hand.add(hand, last_tile)
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            hand = Hand.to_34(hand)
            last_tile_type = int(Tile.to_tile_type(last_tile))
        else:
            last_tile_type = int(last_tile)

        n_kan = sum(1 for i in range(melds.shape[0])
                    if (int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]) != EMPTY_MELD
                    and Meld.is_kan(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]))
        n_closed_kan = sum(1 for i in range(melds.shape[0])
                           if (int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]) != EMPTY_MELD
                           and Meld.is_closed_kan(int(melds[i].item()) if isinstance(melds[i], torch.Tensor) else melds[i]))
        n_concealed_pung = int((hand >= 3).sum().item()) - (is_ron and hand[last_tile_type] >= 3) + n_closed_kan

        codes = (hand[:27].to(torch.int32) * POWERS_OF_5).reshape(3, 9).sum(dim=1)  # (3,)
        # Check all 3 decomposition patterns per suit (match JAX vmap over all columns)
        nine_gates = any(
            any(int(v.item()) >> 26 for v in Yaku.CACHE[int(codes[s].item())])
            for s in range(3))

        flatten = Yaku.flatten(hand, melds, 0)
        four_winds = int((flatten[27:31] >= 3).sum().item())
        three_dragons = int((flatten[31:34] >= 3).sum().item())
        has_tanyao = bool((flatten[TANYAO_TILE] > 0).any().item())
        has_honor = bool((flatten[27:34] > 0).any().item())
        win_tile_count = int(hand[last_tile_type].item())
        four_concealed_tsumo = n_concealed_pung == 4 and win_tile_count >= 3 and not is_ron
        four_concealed_single = n_concealed_pung == 4 and win_tile_count == 2

        yakuman = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        yakuman[YI.BigThreeDragons] = three_dragons == 3
        yakuman[YI.BigFourWinds] = four_winds == 4
        yakuman[YI.LittleFourWinds] = bool((flatten[27:31] >= 2).all().item()) and four_winds == 3
        yakuman[YI.NineGates] = nine_gates
        yakuman[YI.ThirteenOrphans] = bool((hand[KOKUSHI_TILE] > 0).all().item()) and not has_tanyao
        yakuman[YI.AllTerminals] = (not has_tanyao) and (not has_honor)
        yakuman[YI.AllHonors] = bool((flatten[0:27] == 0).all().item())
        yakuman[YI.AllGreen] = int(flatten[ALL_GREEN_TILE].sum().item()) == 14
        yakuman[YI.FourConcealedPons] = four_concealed_tsumo
        yakuman[YI.CompletedFourConcealedPons] = four_concealed_single
        yakuman[YI.FourKans] = n_kan == 4

        yakuman_num = int((yakuman.to(torch.int32) @ _YAKUMAN).item())
        return yakuman, yakuman_num, torch.tensor(0, dtype=torch.int32)


# ── Expose YI constants on Yaku class ──
for _name in dir(YI):
    if not _name.startswith('_'):
        setattr(Yaku, _name, getattr(YI, _name))


__all__ = ['NUM_TENHOU_YAKU', 'Yaku']
