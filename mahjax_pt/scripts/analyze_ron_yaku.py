#!/usr/bin/env python3
"""Analyze which yaku types were achieved in RON seeds.

Replays each RON seed through the PT serial env, captures the yaku vector
at the moment of ron, and prints a summary of which yaku appeared.

Usage:
  python analyze_ron_yaku.py                  # all RON seeds from scan
  python analyze_ron_yaku.py -s 3000 3078     # specific seeds
  python analyze_ron_yaku.py -d golden_data   # custom golden dir
"""

import os, sys, pickle, argparse
from collections import Counter
import numpy as np

# Allow importing from tests/ (for _copy_golden_to_pt)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tests'))

# ── Yaku names — order must match yaku.py YI (YakuIndex) indices exactly ──
YAKU_NAMES_EN = [
    "menzen",              #  0  YI.FullyConcealedHand
    "riichi",              #  1  YI.Riichi
    "ippatsu",             #  2  YI.Ippatsu
    "robbing_kan",         #  3  YI.RobbingKan
    "after_kan",           #  4  YI.DrawAfterKan
    "haitei",              #  5  YI.BottomOfTheSea
    "houtei",              #  6  YI.BottomOfTheRiver
    "pinfu",               #  7  YI.Pinfu
    "tanyao",              #  8  YI.AllSimples
    "iipeikou",            #  9  YI.PureDoubleChis
    "yakuhai_jikaze_east", # 10  YI.SeatWindEast
    "yakuhai_jikaze_south",# 11  YI.SeatWindSouth
    "yakuhai_jikaze_west", # 12  YI.SeatWindWest
    "yakuhai_jikaze_north",# 13  YI.SeatWindNorth
    "yakuhai_bakaze_east", # 14  YI.PrevalentWindEast
    "yakuhai_bakaze_south",# 15  YI.PrevalentWindSouth
    "yakuhai_bakaze_west", # 16  YI.PrevalentWindWest
    "yakuhai_bakaze_north",# 17  YI.PrevalentWindNorth
    "yakuhai_haku",        # 18  YI.WhiteDragon
    "yakuhai_hatsu",       # 19  YI.GreenDragon
    "yakuhai_chun",        # 20  YI.RedDragon
    "double_riichi",       # 21  YI.DoubleRiichi
    "chiitoitsu",          # 22  YI.SevenPairs
    "chanta",              # 23  YI.OutsideHand
    "sanshoku_doujun",     # 24  YI.PureStraight
    "ikkitsuukan",         # 25  YI.MixedTripleChis
    "sanshoku_doukou",     # 26  YI.TriplePons
    "sankantsu",           # 27  YI.ThreeKans
    "toitoi",              # 28  YI.AllPons
    "sanankou",            # 29  YI.ThreeConcealedPons
    "shousangen",          # 30  YI.LittleThreeDragons
    "honroutou",           # 31  YI.AllTerminalsAndHonors
    "ryanpeikou",          # 32  YI.TwicePureDoubleChis
    "junchanta",           # 33  YI.TerminalsInAllSets
    "honitsu",             # 34  YI.HalfFlush
    "chinitsu",            # 35  YI.FullFlush
    "renhou",              # 36  YI.Renhou
    "tenhou",              # 37  YI.BlessingOfHeaven
    "chiihou",             # 38  YI.BlessingOfEarth
    "daisangen",           # 39  YI.BigThreeDragons
    "suuankou",            # 40  YI.FourConcealedPons
    "suuankou_tanki",      # 41  YI.CompletedFourConcealedPons
    "tsuuiisou",           # 42  YI.AllHonors
    "ryuuiisou",           # 43  YI.AllGreen
    "chinroutou",          # 44  YI.AllTerminals
    "chuuren_poutou",      # 45  YI.NineGates
    "junsei_chuuren",      # 46  YI.PureNineGates
    "kokushi_musou",       # 47  YI.ThirteenOrphans
    "dora",                # 48  (dora indicator count)
    "ura_dora",            # 49  (ura-dora count)
    "aka_dora",            # 50  (red-five count)
    "kita_dora",           # 51  (north-dora count)
]

YAKU_NAMES_JP = [
    "門前清自摸和",       #  0
    "立直",               #  1
    "一発",               #  2
    "槍槓",               #  3
    "嶺上開花",           #  4
    "海底撈月",           #  5
    "河底撈魚",           #  6
    "平和",               #  7
    "断幺九",             #  8
    "一盃口",             #  9
    "自風 東",            # 10
    "自風 南",            # 11
    "自風 西",            # 12
    "自風 北",            # 13
    "場風 東",            # 14
    "場風 南",            # 15
    "場風 西",            # 16
    "場風 北",            # 17
    "役牌 白",            # 18
    "役牌 發",            # 19
    "役牌 中",            # 20
    "ダブル立直",         # 21
    "七対子",             # 22
    "混全帯么九",         # 23
    "三色同順",           # 24
    "一気通貫",           # 25
    "三色同刻",           # 26
    "三槓子",             # 27
    "対々和",             # 28
    "三暗刻",             # 29
    "小三元",             # 30
    "混老頭",             # 31
    "二盃口",             # 32
    "純全帯么九",         # 33
    "混一色",             # 34
    "清一色",             # 35
    "人和",               # 36
    "天和",               # 37
    "地和",               # 38
    "大三元",             # 39
    "四暗刻",             # 40
    "四暗刻単騎",         # 41
    "字一色",             # 42
    "緑一色",             # 43
    "清老頭",             # 44
    "九蓮宝燈",           # 45
    "純正九蓮宝燈",       # 46
    "国士無双",           # 47
    "ドラ",               # 48
    "裏ドラ",             # 49
    "赤ドラ",             # 50
    "北ドラ",             # 51
]

# Han values
YAKU_HAN = {
    "menzen": 1, "riichi": 1, "ippatsu": 1, "double_riichi": 2,
    "robbing_kan": 1, "after_kan": 1, "haitei": 1, "houtei": 1,
    "pinfu": 1, "tanyao": 1, "iipeikou": 1, "ryanpeikou": 3,
    "yakuhai_haku": 1, "yakuhai_hatsu": 1, "yakuhai_chun": 1,
    "yakuhai_bakaze_east": 1, "yakuhai_bakaze_west": 1,
    "yakuhai_bakaze_south": 1, "yakuhai_bakaze_north": 1,
    "yakuhai_jikaze_east": 1, "yakuhai_jikaze_west": 1,
    "yakuhai_jikaze_south": 1, "yakuhai_jikaze_north": 1,
    "chanta": (2, 1), "junchanta": (3, 2), "honroutou": 2,
    "sanshoku_doujun": (2, 1), "ikkitsuukan": (2, 1),
    "toitoi": 2, "sanankou": 2, "sanshoku_doukou": 2,
    "sankantsu": 2, "shousangen": 2, "chiitoitsu": 2,
    "honitsu": (3, 2), "chinitsu": (6, 5),
    "dora": 1, "ura_dora": 1, "aka_dora": 1, "kita_dora": 1,
}


def replay_and_capture(seed, init_state, records):
    """Replay one seed through PT serial env and capture yaku at ron step.

    At the ron step we call Yaku.judge directly, since the env only stores
    has_yaku/fan/fu but not the yaku vector itself.
    """
    import torch
    from mahjax_pt.red_mahjong.env_serial import RedMahjongSerial as PtEnv
    from mahjax_pt.red_mahjong.yaku import Yaku
    from mahjax_pt.red_mahjong.constants import NUM_TILE_TYPES_WITH_RED
    from replay_pt_against_golden import _copy_golden_to_pt

    penv = PtEnv(round_mode='single')
    state = penv.init(key=0)
    _copy_golden_to_pt(init_state, state)

    results = []
    for step_idx, rec in enumerate(records):
        action = rec['action']

        if action == 74:  # Action.RON
            cp = int(state.current_player)

            # Yaku.judge(hand, is_ron, player_idx, state) expects the 13-tile
            # hand_with_red (NOT 14).  It reads state.round_state.target to add
            # the winning tile internally.
            hand = state.players.hand_with_red[cp].clone()

            try:
                yaku_vec, judge_fan, judge_fu = Yaku.judge(hand, True, cp, state)
                yaku_np = yaku_vec.detach().cpu().numpy() if hasattr(yaku_vec, 'detach') else np.asarray(yaku_vec)
                fan = int(judge_fan.item()) if hasattr(judge_fan, 'item') else int(judge_fan)
                fu = int(judge_fu.item()) if hasattr(judge_fu, 'item') else int(judge_fu)
            except Exception as e:
                yaku_np = None
                fan = int(state.players.fan[cp]) if hasattr(state.players.fan[cp], 'item') else int(state.players.fan[cp])
                fu = int(state.players.fu[cp]) if hasattr(state.players.fu[cp], 'item') else int(state.players.fu[cp])

            riichi = bool(state.players.riichi[cp])
            concealed = bool(state.players.is_hand_concealed[cp])
            n_melds = int(state.players.meld_counts[cp].sum())

            results.append({
                'seed': seed,
                'step': step_idx,
                'fan': fan,
                'fu': fu,
                'riichi': riichi,
                'concealed': concealed,
                'n_melds': n_melds,
                'yaku_vec': yaku_np,
            })

        state = penv.step(state, action)

    return results


def decode_yaku(yaku_vec):
    """Given a binary yaku vector, return list of (name_en, name_jp) tuples."""
    if yaku_vec is None:
        return []
    active = []
    for i, (en, jp) in enumerate(zip(YAKU_NAMES_EN, YAKU_NAMES_JP)):
        if i < len(yaku_vec) and yaku_vec[i]:
            active.append((en, jp))
    return active


def print_results(all_results):
    """Print a summary of yaku achieved across all ron seeds."""
    yaku_counter = Counter()  # yaku_name -> count
    fan_dist = Counter()
    fu_dist = Counter()
    riichi_count = 0
    concealed_count = 0
    total = 0

    for r in all_results:
        total += 1
        fan_dist[r['fan']] += 1
        fu_dist[r['fu']] += 1
        if r['riichi']:
            riichi_count += 1
        if r['concealed']:
            concealed_count += 1

        if r['yaku_vec'] is not None and len(r['yaku_vec']) > 0:
            for en, _jp in decode_yaku(r['yaku_vec']):
                yaku_counter[en] += 1

    print(f"\n{'='*60}")
    print(f"RON analysis: {total} ron events across {len(set(r['seed'] for r in all_results))} seeds")
    print(f"{'='*60}")

    print(f"\n── 基本统计 ──")
    print(f"  立直 (riichi):    {riichi_count}/{total}")
    print(f"  门前清 (concealed): {concealed_count}/{total}")
    print(f"  平均 fan:          {sum(k*v for k,v in fan_dist.items())/total:.1f}")
    print(f"  平均 fu:           {sum(k*v for k,v in fu_dist.items())/total:.1f}")
    print(f"\n  翻数分布: {dict(sorted(fan_dist.items()))}")
    print(f"  符数分布: {dict(sorted(fu_dist.items()))}")

    if yaku_counter:
        print(f"\n── 役种出现次数 ──")
        for (en, count) in yaku_counter.most_common():
            jp = dict(zip(YAKU_NAMES_EN, YAKU_NAMES_JP)).get(en, en)
            han = YAKU_HAN.get(en, '?')
            print(f"  {jp:<16s} ({en:<22s})  han={str(han):>6s}  x{count}")

        # Group by han value
        print(f"\n── 按役种等级分类 ──")
        groups = {1: [], 2: [], 3: [], 6: []}
        for (en, count) in yaku_counter.most_common():
            han = YAKU_HAN.get(en, 0)
            if isinstance(han, tuple):
                han = han[0]  # use closed value
            if han in groups:
                groups[han].append((en, count))
        for han in [1, 2, 3, 6]:
            if groups[han]:
                names = [f"{dict(zip(YAKU_NAMES_EN, YAKU_NAMES_JP)).get(en, en)}({count})"
                         for en, count in groups[han]]
                print(f"  {han}翻: {', '.join(names)}")

    # Per-seed detail
    print(f"\n── 每种子的荣和详情 ──")
    for r in sorted(all_results, key=lambda x: (x['fan'], x['seed'])):
        yaku_list = decode_yaku(r['yaku_vec']) if r['yaku_vec'] is not None else []
        yaku_str = ', '.join(jp for _en, jp in yaku_list) if yaku_list else '(no yaku data)'
        tags = []
        if r['riichi']: tags.append('立直')
        if r['concealed']: tags.append('门前')
        if r['n_melds'] > 0: tags.append(f'副露{r["n_melds"]}')
        print(f"  seed={r['seed']:>5d}  step={r['step']:>3d}  "
              f"{r['fan']}翻{r['fu']}符  [{', '.join(tags)}]  {yaku_str}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Analyze yaku types in RON seeds')
    p.add_argument('-d', '--data-dir', default='golden_data',
                   help='Golden data directory (default: golden_data)')
    p.add_argument('-s', '--seeds', nargs='*', type=int,
                   help='Specific seeds to analyze (default: known RON seeds)')
    args = p.parse_args()

    # Default: known RON seeds from scan_rare_paths
    if args.seeds:
        ron_seeds = args.seeds
    else:
        ron_seeds = [3000, 3016, 3078, 3096, 4010, 4077, 4099, 10038, 10039,
                     10042, 10053, 10059, 10063, 10087, 10088, 10097, 10156,
                     10168, 10187, 10264, 10278, 10282, 10283, 10289, 10311,
                     10338, 10349, 10368, 10376, 10400, 10418, 10433, 10452,
                     10483, 10496]

    all_results = []
    for seed in ron_seeds:
        path = os.path.join(args.data_dir, f'golden_seed_{seed:04d}.pkl')
        if not os.path.exists(path):
            print(f"  [skip] seed={seed}: file not found: {path}", file=sys.stderr)
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            results = replay_and_capture(seed, data['init_state'], data['records'])
            all_results.extend(results)
        except Exception as e:
            print(f"  [error] seed={seed}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    print_results(all_results)
