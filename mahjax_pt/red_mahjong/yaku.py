from __future__ import annotations

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

# Powers of 5 for encoding hand patterns (same as in hand.py but used locally)
powers_of_5_full = torch.cat([5 ** torch.arange(8, -1, -1)] * 3)


def _load_yaku_cache():
    """Load the yaku-cache from the local mahjax_pt package."""
    with resources.as_file(resources.files("mahjax_pt._src.cache").joinpath("yaku_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return torch.from_numpy(data["data"].astype(np.int64))


WIND_TILE = torch.tensor([27, 28, 29, 30], dtype=torch.int8)
OUTSIDE_TILE = torch.tensor([0, 8, 9, 17, 18, 26], dtype=torch.int8)
TANYAO_TILE = torch.tensor(
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25],
    dtype=torch.int8,
)
KOKUSHI_TILE = torch.tensor([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=torch.int8)
ALL_GREEN_TILE = torch.tensor([19, 20, 21, 23, 25, 32], dtype=torch.int8)
SCORES = torch.tensor(
    [2000, 2000, 3000, 3000, 4000, 4000, 4000, 6000, 6000, 8000, 8000, 8000],
    dtype=torch.int32,
)
NUM_TENHOU_YAKU = 52


def _dora_array_from_state(state) -> torch.Tensor:
    """Compute dora and ura-dora count vectors from round state."""
    dora_counts = torch.zeros(Tile.NUM_TILE_TYPE, dtype=torch.int8)
    for dora_indicator in state.round_state.dora_indicators:
        dora_indicator = int(dora_indicator)
        if dora_indicator != -1:
            dora_tile_type = int(Tile.to_tile_type(dora_indicator))
            dora_counts[DORA_ARRAY[dora_tile_type]] += 1

    ura_dora_counts = torch.zeros(Tile.NUM_TILE_TYPE, dtype=torch.int8)
    for dora_indicator in state.round_state.ura_dora_indicators:
        dora_indicator = int(dora_indicator)
        if dora_indicator != -1:
            dora_tile_type = int(Tile.to_tile_type(dora_indicator))
            ura_dora_counts[DORA_ARRAY[dora_tile_type]] += 1

    return torch.stack([dora_counts, ura_dora_counts], dim=0)


class _Internal:
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

    # Build fan lookups (open / closed)
    _FAN_OPEN = torch.zeros((NUM_TENHOU_YAKU,), dtype=torch.int32)
    _FAN_OPEN[OutsideHand] = 1
    _FAN_OPEN[TerminalsInAllSets] = 2
    _FAN_OPEN[PureStraight] = 1
    _FAN_OPEN[MixedTripleChis] = 1
    _FAN_OPEN[TriplePons] = 2
    _FAN_OPEN[AllPons] = 2
    _FAN_OPEN[ThreeConcealedPons] = 2
    _FAN_OPEN[ThreeKans] = 2
    _FAN_OPEN[SevenPairs] = 2
    _FAN_OPEN[AllSimples] = 1
    _FAN_OPEN[HalfFlush] = 2
    _FAN_OPEN[FullFlush] = 5
    _FAN_OPEN[AllTerminalsAndHonors] = 2
    _FAN_OPEN[LittleThreeDragons] = 2
    _FAN_OPEN[WhiteDragon] = 1
    _FAN_OPEN[GreenDragon] = 1
    _FAN_OPEN[RedDragon] = 1
    for w in (SeatWindEast, SeatWindSouth, SeatWindWest, SeatWindNorth):
        _FAN_OPEN[w] = 1
    for w in (PrevalentWindEast, PrevalentWindSouth, PrevalentWindWest, PrevalentWindNorth):
        _FAN_OPEN[w] = 1

    _FAN_CLOSED = _FAN_OPEN.clone()
    _FAN_CLOSED[FullyConcealedHand] = 1
    _FAN_CLOSED[Riichi] = 1
    _FAN_CLOSED[Pinfu] = 1
    _FAN_CLOSED[PureDoubleChis] = 1
    _FAN_CLOSED[TwicePureDoubleChis] = 3
    _FAN_CLOSED[OutsideHand] = 2
    _FAN_CLOSED[TerminalsInAllSets] = 3
    _FAN_CLOSED[PureStraight] = 2
    _FAN_CLOSED[MixedTripleChis] = 2
    _FAN_CLOSED[HalfFlush] = 3
    _FAN_CLOSED[FullFlush] = 6
    FAN = torch.stack([_FAN_OPEN, _FAN_CLOSED], dim=0)

    YAKUMAN = torch.zeros((NUM_TENHOU_YAKU,), dtype=torch.int32)
    YAKUMAN[BigThreeDragons] = 1
    YAKUMAN[FourConcealedPons] = 1
    YAKUMAN[CompletedFourConcealedPons] = 1
    YAKUMAN[AllHonors] = 1
    YAKUMAN[AllGreen] = 1
    YAKUMAN[AllTerminals] = 1
    YAKUMAN[NineGates] = 1
    YAKUMAN[PureNineGates] = 1
    YAKUMAN[ThirteenOrphans] = 1
    YAKUMAN[CompletedThirteenOrphans] = 1
    YAKUMAN[BigFourWinds] = 2
    YAKUMAN[LittleFourWinds] = 1
    YAKUMAN[FourKans] = 1

    YAKU_UPDATE_INDICES = torch.tensor(
        [Pinfu, PureDoubleChis, TwicePureDoubleChis, OutsideHand, TerminalsInAllSets,
         PureStraight, MixedTripleChis, TriplePons, AllPons, ThreeConcealedPons, ThreeKans],
        dtype=torch.int32,
    )
    YAKU_BEST_UPDATE_INDICES = torch.tensor(
        [AllSimples, HalfFlush, FullFlush, AllTerminalsAndHonors,
         WhiteDragon, GreenDragon, RedDragon, LittleThreeDragons,
         FullyConcealedHand, Riichi],
        dtype=torch.int32,
    )
    YAKUMAN_UPDATE_INDICES = torch.tensor(
        [BigThreeDragons, BigFourWinds, LittleFourWinds, NineGates, ThirteenOrphans,
         AllTerminals, AllHonors, AllGreen, FourConcealedPons, FourKans],
        dtype=torch.int32,
    )


class Yaku:
    CACHE = _load_yaku_cache()
    MAX_PATTERNS = _Internal.MAX_PATTERNS

    @staticmethod
    def head(code):
        return int(Yaku.CACHE[code].item()) & 0b1111

    @staticmethod
    def chow(code):
        return (int(Yaku.CACHE[code].item()) >> 4) & 0b1111111

    @staticmethod
    def pung(code):
        return (int(Yaku.CACHE[code].item()) >> 11) & 0b111111111

    @staticmethod
    def n_pung(code):
        return (int(Yaku.CACHE[code].item()) >> 20) & 0b111

    @staticmethod
    def n_double_chow(code):
        return (int(Yaku.CACHE[code].item()) >> 23) & 0b11

    @staticmethod
    def outside(code):
        return (int(Yaku.CACHE[code].item()) >> 25) & 1

    @staticmethod
    def nine_gates(code):
        return int(Yaku.CACHE[code].item()) >> 26

    @staticmethod
    def is_pure_straight(chow):
        return int(
            ((chow & 0b1001001) == 0b1001001)
            | ((chow >> 9 & 0b1001001) == 0b1001001)
            | ((chow >> 18 & 0b1001001) == 0b1001001)
        )

    @staticmethod
    def is_triple_chow(chow):
        pat = 0b1000000001000000001
        out = (chow & pat) == pat
        for s in range(1, 8):
            out = out | ((chow >> s & pat) == pat)
        return int(out)

    @staticmethod
    def is_triple_pung(pung):
        pat = 0b1000000001000000001
        out = (pung & pat) == pat
        for s in range(1, 9):
            out = out | ((pung >> s & pat) == pat)
        return int(out)

    @staticmethod
    def update(is_pinfu, has_outside, n_double_chow, all_chow, all_pung,
               n_concealed_pung, nine_gates, fu, code, suit, last_tile_type, is_ron):
        is_ron = bool(is_ron)
        chow = Yaku.chow(code)
        pung = Yaku.pung(code)
        open_end = ((chow ^ (chow & 1)) << 2) | (chow ^ (chow & 0b1000000))
        in_range = suit == (last_tile_type // 9)
        pos = last_tile_type % 9
        is_pinfu = is_pinfu & (not in_range or (((open_end >> pos) & 1) == 1 and pung == 0))
        has_outside = has_outside & (Yaku.outside(code) == 1)
        n_double_chow = n_double_chow + Yaku.n_double_chow(code)
        all_chow = all_chow | (chow << (9 * suit))
        all_pung = all_pung | (pung << (9 * suit))
        n_pung = Yaku.n_pung(code)
        chow_range = chow | (chow << 1) | (chow << 2)
        loss = is_ron & in_range & (((chow_range >> pos) & 1) == 0) & (((pung >> pos) & 1) == 1)
        n_concealed_pung = n_concealed_pung + n_pung - int(loss)
        nine_gates = nine_gates | (Yaku.nine_gates(code) == 1)
        outside_pung = pung & 0b100000001
        n_outside_pung = (outside_pung & 1) + ((outside_pung >> 8) & 1)
        strong = (
            in_range
            & (
                (1 << Yaku.head(code))
                | ((chow & 1) << 2)
                | (chow & 0b1000000)
                | (chow << 1)
            )
            >> pos
            & 1
        )
        outside_loss = loss & ((outside_pung >> pos) & 1)
        fu = fu + 4 * n_pung + 4 * n_outside_pung - 2 * int(loss) - 2 * int(outside_loss) + 2 * int(strong)
        return is_pinfu, has_outside, n_double_chow, all_chow, all_pung, n_concealed_pung, nine_gates, fu

    @staticmethod
    def _chi_index(action):
        if action in (78, 79):
            return 0
        elif action in (80, 81):
            return 1
        elif action in (82, 83):
            return 2
        return -1

    @staticmethod
    def _calc_addition(meld):
        target = int(Meld.target(meld))
        action = int(Meld.action(meld))
        addition = torch.zeros(34, dtype=torch.int8)
        is_pon_val = int(bool(Meld.is_pon(meld)))
        is_kan_val = int(bool(Meld.is_kan(meld)))
        addition[target] = 3 * is_pon_val + 4 * is_kan_val
        chi_idx = Yaku._chi_index(action)
        start = max(target - chi_idx, 0)
        start = min(start, 31)
        is_chi = int(bool(Meld.is_chi(meld)))
        addition[start] += is_chi
        addition[start + 1] += is_chi
        addition[start + 2] += is_chi
        if meld == EMPTY_MELD:
            addition.zero_()
        return addition

    @staticmethod
    def flatten(hand, melds, n_meld):
        del n_meld
        addition = torch.zeros(34, dtype=torch.int8)
        for i in range(melds.shape[0]):
            addition += Yaku._calc_addition(melds[i])
        return Hand.to_34(hand) + addition

    @staticmethod
    def score(fan, fu):
        if fu == 0:
            return 8000 * fan
        raw = fu * (1 << (fan + 2))
        if raw < 2000:
            return raw
        else:
            idx = min(max(fan - 4, 0), 11)
            return int(SCORES[idx].item())

    @staticmethod
    def judge_hand_related(hand, melds, n_meld, last_tile, riichi, is_ron,
                           prevalent_wind, seat_wind, dora):
        hand = Hand.add(hand, last_tile)
        red_fan = 0
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            red_fan_val = int(hand[Tile.NUM_TILE_TYPE:].sum().item())
            for i in range(melds.shape[0]):
                red_fan_val += int(bool(Meld.contains_red(melds[i])))
            red_fan = red_fan_val
            hand = Hand.to_34(hand)
            last_tile_type = int(Tile.to_tile_type(last_tile))
        else:
            last_tile_type = int(last_tile)

        if riichi:
            dora = dora.sum(dim=0)
        else:
            dora = dora[0]

        seat_wind_tile_type = int(WIND_TILE[int(seat_wind)])
        prevalent_wind_tile_type = int(WIND_TILE[int(prevalent_wind)])

        is_hand_concealed = True
        for i in range(melds.shape[0]):
            m = melds[i]
            if not (bool(Meld.is_closed_kan(m)) or m == EMPTY_MELD):
                is_hand_concealed = False
                break

        is_ron_int = bool(is_ron)

        is_pinfu = torch.full((Yaku.MAX_PATTERNS,),
            is_hand_concealed
            & (n_meld == 0)
            & (last_tile_type < 27)
            & bool((hand[27:31] < 3).all().item())
            & (hand[seat_wind_tile_type] == 0)
            & (hand[prevalent_wind_tile_type] == 0)
            & bool((hand[31:34] == 0).all().item()),
            dtype=torch.bool)

        has_outside = torch.full((Yaku.MAX_PATTERNS,),
            all(bool(Meld.has_outside(melds[i])) or melds[i] == EMPTY_MELD for i in range(melds.shape[0])),
            dtype=torch.bool)

        meld_chow_bits = reduce(operator.or_,
            [int(Meld.chow(melds[i]).item()) for i in range(melds.shape[0])], 0)
        meld_pung_bits = reduce(operator.or_,
            [int(Meld.suited_pung(melds[i]).item()) for i in range(melds.shape[0])], 0)

        all_chow = torch.full((Yaku.MAX_PATTERNS,), meld_chow_bits, dtype=torch.int32)
        all_pung = torch.full((Yaku.MAX_PATTERNS,), meld_pung_bits, dtype=torch.int32)

        n_kan = sum(1 for i in range(melds.shape[0]) if bool(Meld.is_kan(melds[i])) and melds[i] != EMPTY_MELD)
        n_closed_kan = sum(1 for i in range(melds.shape[0]) if bool(Meld.is_closed_kan(melds[i])) and melds[i] != EMPTY_MELD)

        base_concealed = int((hand[27:] >= 3).sum().item())
        penalty = is_ron_int and last_tile_type >= 27 and hand[last_tile_type] >= 3
        n_concealed_pung = base_concealed - int(penalty) + n_closed_kan

        honor_tile_types = torch.arange(27, 34, dtype=torch.int32)
        ron_penalty = (is_ron_int and (honor_tile_types == last_tile_type).any()).item()

        seat_tt = seat_wind_tile_type
        prev_tt = prevalent_wind_tile_type
        seat_pair = hand[seat_tt] == 2
        prev_pair = hand[prev_tt] == 2
        renfu_pair = (seat_tt == prev_tt) and seat_pair
        if renfu_pair:
            wind_pair_fu = 4
        elif not renfu_pair:
            wind_pair_fu = (2 if seat_pair else 0) + (2 if prev_pair else 0)
        else:
            wind_pair_fu = 0

        fu = torch.full((Yaku.MAX_PATTERNS,),
            2 * (not is_ron_int)
            + sum(int(Meld.fu(melds[i]).item()) for i in range(melds.shape[0]))
            + wind_pair_fu
            + 2 * int(bool((hand[31:] == 2).any().item()))
            + int(sum(int(hand[i].item() == 3) * 4 * (2 - int(ron_penalty)) for i in range(27, 34)))
            + int(is_ron_int and 27 <= last_tile_type and hand[last_tile_type] == 2),
            dtype=torch.int32)

        codes = (hand[:27].to(torch.int32) * powers_of_5_full).reshape(3, 9).sum(dim=1)

        n_double_chow = torch.zeros((Yaku.MAX_PATTERNS,), dtype=torch.int32)
        nine_gates = torch.full((Yaku.MAX_PATTERNS,), False)

        for suit in range(3):
            code = int(codes[suit].item())
            is_pinfu, has_outside, n_double_chow, all_chow, all_pung, \
                n_concealed_pung, nine_gates, fu = Yaku.update(
                    is_pinfu, has_outside, n_double_chow, all_chow, all_pung,
                    n_concealed_pung, nine_gates, fu, code, suit, last_tile_type, is_ron_int)

        fu = fu * (is_pinfu == 0).to(torch.int32)
        fu = fu + 20 + 10 * int(is_hand_concealed and is_ron_int)
        fu = fu + 10 * int(not is_hand_concealed and int(fu[0].item()) == 20)

        flatten = Yaku.flatten(hand, melds, n_meld)
        four_winds = int((flatten[27:31] >= 3).sum().item())
        three_dragons = int((flatten[31:34] >= 3).sum().item())
        has_tanyao = bool((flatten[TANYAO_TILE] > 0).any().item())
        has_honor = bool((flatten[27:] > 0).any().item())
        has_honor_ts = bool(has_honor)
        suit_count = (
            int((flatten[0:9] > 0).any().item())
            + int((flatten[9:18] > 0).any().item())
            + int((flatten[18:27] > 0).any().item())
        )
        is_flush = suit_count == 1

        yaku_update_values = torch.stack([
            is_pinfu,
            torch.tensor([is_hand_concealed and int(n_double_chow[i].item()) == 1 for i in range(Yaku.MAX_PATTERNS)]),
            torch.tensor([int(n_double_chow[i].item()) == 2 for i in range(Yaku.MAX_PATTERNS)]),
            has_outside & has_honor_ts & has_tanyao,
            has_outside & (not has_honor_ts),
            torch.full((Yaku.MAX_PATTERNS,), Yaku.is_pure_straight(int(all_chow[0].item()))),
            torch.full((Yaku.MAX_PATTERNS,), Yaku.is_triple_chow(int(all_chow[0].item()))),
            torch.full((Yaku.MAX_PATTERNS,), Yaku.is_triple_pung(int(all_pung[0].item()))),
            torch.full((Yaku.MAX_PATTERNS,), int(all_chow[0].item()) == 0),
            torch.tensor([int(n_concealed_pung[i].item()) == 3 for i in range(Yaku.MAX_PATTERNS)]),
            torch.full((Yaku.MAX_PATTERNS,), n_kan == 3),
        ])

        yaku = torch.zeros((NUM_TENHOU_YAKU, Yaku.MAX_PATTERNS), dtype=torch.bool)
        yaku[_Internal.YAKU_UPDATE_INDICES, :] = yaku_update_values

        fan_row = _Internal.FAN[1 if is_hand_concealed else 0]
        best_pattern = int(torch.argmax((fan_row @ yaku.to(torch.int32)) * 200 + fu).item())
        yaku_best = yaku[:, best_pattern]
        fu_best = int(fu[best_pattern].item())
        fu_best = fu_best + (-fu_best % 10)

        is_mentsu_hand = yaku_best[_Internal.TwicePureDoubleChis] | ((hand == 2).sum() < 7)
        if not is_mentsu_hand:
            yaku_best = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
            yaku_best[_Internal.SevenPairs] = True
            fu_best = 25

        has_outside_in_flatten = bool((flatten[OUTSIDE_TILE] > 0).any().item())

        yaku_best_update = torch.tensor([
            not (has_honor_ts or has_outside_in_flatten),
            is_flush and has_honor_ts,
            is_flush and not has_honor_ts,
            not has_tanyao,
            flatten[31] >= 3,
            flatten[32] >= 3,
            flatten[33] >= 3,
            bool((flatten[31:34] >= 2).all().item()) and three_dragons >= 2,
            is_hand_concealed and not is_ron_int,
            riichi,
        ], dtype=torch.bool)
        yaku_best[_Internal.YAKU_BEST_UPDATE_INDICES] = yaku_best_update
        yaku_best[_Internal.PrevalentWindEast + prevalent_wind] = flatten[prevalent_wind_tile_type] >= 3
        yaku_best[_Internal.SeatWindEast + seat_wind] = flatten[seat_wind_tile_type] >= 3

        win_tile_count = int(hand[last_tile_type].item())
        four_concealed_tsumo = any(
            int(n_concealed_pung[i].item()) == 4 for i in range(Yaku.MAX_PATTERNS)
        ) and win_tile_count >= 3 and not is_ron_int
        four_concealed_single = any(
            int(n_concealed_pung[i].item()) == 4 for i in range(Yaku.MAX_PATTERNS)
        ) and win_tile_count == 2

        yakuman_update_values = torch.tensor([
            three_dragons == 3,
            four_winds == 4,
            bool((flatten[27:31] >= 2).all().item()) and four_winds == 3,
            any(bool(nine_gates[i]) for i in range(Yaku.MAX_PATTERNS)),
            bool((hand[KOKUSHI_TILE] > 0).all().item()) and not has_tanyao,
            not has_tanyao and not has_honor_ts,
            bool((flatten[0:27] == 0).all().item()),
            int(flatten[ALL_GREEN_TILE].sum().item()) == 14,
            four_concealed_tsumo,
            n_kan == 4,
        ], dtype=torch.bool)

        yakuman = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        yakuman[_Internal.YAKUMAN_UPDATE_INDICES] = yakuman_update_values
        yakuman[_Internal.CompletedFourConcealedPons] = four_concealed_single
        yakuman_num = int((yakuman.to(torch.int32) @ _Internal.YAKUMAN).item())

        if yakuman.any():
            return yakuman, yakuman_num, 0
        else:
            fan_val = int((fan_row @ yaku_best.to(torch.int32)).item()) + int(flatten.dot(dora).item()) + red_fan
            return yaku_best, fan_val, fu_best

    @staticmethod
    def judge(hand, is_ron, player, rs):
        p = int(player)
        melds = rs.players.melds[p]
        n_meld = int(rs.players.meld_counts[p].item())
        if is_ron:
            last_tile = int(rs.round_state.target)
        else:
            last_tile = int(rs.round_state.last_draw)
        riichi = bool(rs.players.riichi[p])
        prevalent_wind = int(rs.round_state.round) // 4
        seat_wind = int(rs.round_state.seat_wind[p])
        dora = _dora_array_from_state(rs)
        return Yaku.judge_hand_related(
            hand=hand, melds=melds, n_meld=n_meld, last_tile=last_tile,
            riichi=riichi, is_ron=is_ron, prevalent_wind=prevalent_wind,
            seat_wind=seat_wind, dora=dora,
        )

    @staticmethod
    def judge_other(*, is_ron, is_riichi, is_ippatsu=False, is_robbing_kan=False,
                    is_after_kan=False, is_bottom_of_the_sea=False,
                    is_bottom_of_the_river=False, is_double_riichi=False,
                    is_blessing_of_heaven=False, is_blessing_of_earth=False):
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
        normal[_Internal.Ippatsu] = is_ippatsu
        normal[_Internal.RobbingKan] = is_robbing_kan
        normal[_Internal.DrawAfterKan] = is_after_kan
        normal[_Internal.BottomOfTheSea] = is_bottom_of_the_sea
        normal[_Internal.BottomOfTheRiver] = is_bottom_of_the_river
        normal[_Internal.DoubleRiichi] = is_double_riichi

        yakuman = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        yakuman[_Internal.BlessingOfHeaven] = is_blessing_of_heaven
        yakuman[_Internal.BlessingOfEarth] = is_blessing_of_earth

        normal_fan = int(normal.sum().item())
        yakuman_num = int(yakuman.sum().item())
        return normal, yakuman, normal_fan, yakuman_num

    @staticmethod
    def judge_yakuman(hand, is_ron, player, rs):
        p = int(player)
        melds = rs.players.melds[p]
        last_tile = int(rs.round_state.target) if is_ron else int(rs.round_state.last_draw)
        hand = Hand.add(hand, last_tile)
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            hand = Hand.to_34(hand)
            last_tile_type = int(Tile.to_tile_type(last_tile))
        else:
            last_tile_type = int(last_tile)

        n_kan = sum(1 for i in range(melds.shape[0]) if bool(Meld.is_kan(melds[i])) and melds[i] != EMPTY_MELD)
        n_closed_kan = sum(1 for i in range(melds.shape[0]) if bool(Meld.is_closed_kan(melds[i])) and melds[i] != EMPTY_MELD)
        n_concealed_pung = int((hand >= 3).sum().item()) - (is_ron and hand[last_tile_type] >= 3) + n_closed_kan

        codes = (hand[:27].to(torch.int32) * powers_of_5_full).reshape(3, 9).sum(dim=1)
        nine_gates = any(Yaku.nine_gates(int(codes[s].item())) for s in range(3))

        flatten = Yaku.flatten(hand, melds, 0)
        four_winds = int((flatten[27:31] >= 3).sum().item())
        three_dragons = int((flatten[31:34] >= 3).sum().item())
        has_tanyao = bool((flatten[TANYAO_TILE] > 0).any().item())
        has_honor = bool((flatten[27:] > 0).any().item())
        win_tile_count = int(hand[last_tile_type].item())
        four_concealed_tsumo = n_concealed_pung == 4 and win_tile_count >= 3 and not is_ron
        four_concealed_single = n_concealed_pung == 4 and win_tile_count == 2

        yakuman_update_values = torch.tensor([
            three_dragons == 3,
            four_winds == 4,
            bool((flatten[27:31] >= 2).all().item()) and four_winds == 3,
            nine_gates,
            bool((hand[KOKUSHI_TILE] > 0).all().item()) and not has_tanyao,
            not has_tanyao and not has_honor,
            bool((flatten[0:27] == 0).all().item()),
            int(flatten[ALL_GREEN_TILE].sum().item()) == 14,
            four_concealed_tsumo,
            n_kan == 4,
        ], dtype=torch.bool)

        yakuman = torch.zeros(NUM_TENHOU_YAKU, dtype=torch.bool)
        yakuman[_Internal.YAKUMAN_UPDATE_INDICES] = yakuman_update_values
        yakuman[_Internal.CompletedFourConcealedPons] = four_concealed_single
        yakuman_num = int((yakuman.to(torch.int32) @ _Internal.YAKUMAN).item())
        return yakuman, yakuman_num, 0


# Copy class-level constants from _Internal to Yaku for easy access
for _name in dir(_Internal):
    if _name.startswith('_'):
        continue
    _value = getattr(_Internal, _name)
    if isinstance(_value, int):
        setattr(Yaku, _name, _value)


__all__ = ['NUM_TENHOU_YAKU', 'Yaku']
