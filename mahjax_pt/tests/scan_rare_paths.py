#!/usr/bin/env python3
"""Scan golden data for seeds that exercise rare/edge-case paths.

Reads golden pickle files directly (no JAX, no PT env needed) and reports
which seeds hit each rare action type or state condition.  Useful for:

  - Building a regression seed set that covers ALL action types
  - Identifying coverage gaps (which rare paths have zero examples)
  - Understanding action-type distribution across the seed corpus

Usage:
  python scan_rare_paths.py                          # scan all seeds in default dir
  python scan_rare_paths.py -d ./golden_data          # custom dir
  python scan_rare_paths.py -o regression_seeds.json  # output JSON
  python scan_rare_paths.py --print-seeds             # print seed lists per category
  python scan_rare_paths.py --summary-only            # only print summary table
"""

import os, sys, time, pickle, json, argparse
from collections import defaultdict, Counter
import numpy as np

# ── Action constants (mirrored from action.py) ──

# Discard: 0..36 (37 tile types with red fives)
DISCARD_FIRST, DISCARD_LAST = 0, 36

# Self-kan: 37..70  (tile_idx * 2 + closed(0)/added(1))
SELFKAN_FIRST, SELFKAN_LAST = 37, 70

# Special actions
TSUMOGIRI = 71
RIICHI    = 72
TSUMO     = 73
RON       = 74
PON       = 75
PON_RED   = 76
OPEN_KAN  = 77
CHI_L     = 78
CHI_L_RED = 79
CHI_M     = 80
CHI_M_RED = 81
CHI_R     = 82
CHI_R_RED = 83
PASS      = 84
KYUUSHU   = 85
DUMMY     = 86

def action_type(action: int) -> str:
    """Classify an action into a human-readable type."""
    if DISCARD_FIRST <= action <= DISCARD_LAST:
        return 'discard'
    if SELFKAN_FIRST <= action <= SELFKAN_LAST:
        offset = action - SELFKAN_FIRST
        return 'added_kan' if (offset % 2 == 1) else 'closed_kan'
    return {
        TSUMOGIRI: 'tsumogiri',
        RIICHI:    'riichi',
        TSUMO:     'tsumo',
        RON:       'ron',
        PON:       'pon',
        PON_RED:   'pon_red',
        OPEN_KAN:  'open_kan',
        CHI_L:     'chi',
        CHI_L_RED: 'chi_red',
        CHI_M:     'chi',
        CHI_M_RED: 'chi_red',
        CHI_R:     'chi',
        CHI_R_RED: 'chi_red',
        PASS:      'pass',
        KYUUSHU:   'kyuushu',
        DUMMY:     'dummy',
    }.get(action, f'unknown({action})')


def action_family(action: int) -> str:
    """Coarse action family (merging red variants)."""
    t = action_type(action)
    if t in ('discard', 'tsumogiri'):
        return 'discard'
    if t in ('pon', 'pon_red'):
        return 'pon'
    if t in ('chi', 'chi_red'):
        return 'chi'
    return t  # closed_kan, added_kan, open_kan, riichi, tsumo, ron, pass, kyuushu, dummy


# ── Rare-path detectors ──
# Each detector returns a list of "path tags" that were hit in this seed.
# Tags with zero examples after scanning indicate coverage gaps.

def detect_from_records(records: list) -> dict:
    """Analyse a seed's action/state sequence and return per-tag counts.

    Returns a dict mapping tag → count (how many steps hit the tag).
    Also includes a special 'action_families' tag with the set of families seen.
    """
    tags = Counter()

    families_seen = set()
    max_n_kan = 0
    had_riichi_before_win = False
    riichi_active = False

    for step_idx, rec in enumerate(records):
        action = rec['action']
        state = rec['state']

        fam = action_family(action)
        families_seen.add(fam)

        # ── Action-type tags ──
        at = action_type(action)
        tags[f'action:{at}'] += 1

        # ── State-condition tags ──
        # Haitei (last tile draw)
        if state.get('round_state.is_haitei', False):
            tags['state:haitei'] += 1

        # Abortive draw
        if state.get('round_state.is_abortive_draw_normal', False):
            tags['state:abortive_draw'] += 1

        # After-kan state
        if state.get('round_state.can_after_kan', False):
            tags['state:can_after_kan'] += 1

        # Kan declared this step
        if state.get('round_state.kan_declared', False):
            tags['state:kan_declared'] += 1

        # Furiten active
        players_fd = state.get('players.furiten_by_discard')
        if players_fd is not None and np.any(np.asarray(players_fd)):
            tags['state:furiten_by_discard'] += 1

        players_fp = state.get('players.furiten_by_pass')
        if players_fp is not None and np.any(np.asarray(players_fp)):
            tags['state:furiten_by_pass'] += 1

        # Ippatsu
        players_ip = state.get('players.ippatsu')
        if players_ip is not None and np.any(np.asarray(players_ip)):
            tags['state:ippatsu_active'] += 1

        # Kan dora revealed
        n_kd = int(state.get('round_state.n_kan_doras', 0))
        if n_kd > 0:
            tags['state:n_kan_doras'] += 1
        max_n_kan = max(max_n_kan, n_kd)

        # Riichi declared
        players_rd = state.get('players.riichi_declared')
        if players_rd is not None:
            rd_arr = np.asarray(players_rd)
            if np.any(rd_arr):
                tags['state:riichi_declared'] += 1
                # Track if riichi was active before a win
                if riichi_active:
                    had_riichi_before_win = True

        # Riichi active (bet placed)
        players_ri = state.get('players.riichi')
        if players_ri is not None:
            riichi_active = bool(np.any(np.asarray(players_ri)))

        # Kyotaku (riichi bets on table)
        kyotaku = int(state.get('round_state.kyotaku', 0))
        if kyotaku > 0:
            tags['state:kyotaku'] += 1

        # Honba > 0 (dealer repeat)
        honba = int(state.get('round_state.honba', 0))
        if honba > 0:
            tags['state:honba'] += 1

        # Ura dora present
        if 'round_state.ura_dora_indicators' in state:
            ura = np.asarray(state['round_state.ura_dora_indicators'])
            if np.any(ura >= 0):  # -1 = unused
                tags['state:ura_dora'] += 1

        # Someone has won this step
        players_hw = state.get('players.has_won')
        if players_hw is not None and np.any(np.asarray(players_hw)):
            tags['state:has_won'] += 1

        # Terminated round (non-standard ending)
        if state.get('round_state.terminated_round', False):
            tags['state:terminated_round'] += 1

        # Can robbing kan
        if state.get('round_state.can_robbing_kan', False):
            tags['state:can_robbing_kan'] += 1

        # Double riichi
        players_dr = state.get('players.double_riichi')
        if players_dr is not None and np.any(np.asarray(players_dr)):
            tags['state:double_riichi'] += 1

        # Check for nagashi mangan
        players_nm = state.get('players.has_nagashi_mangan')
        if players_nm is not None and np.any(np.asarray(players_nm)):
            tags['state:has_nagashi_mangan'] += 1

        # Four-kan draw: n_kan >= 4, and we see multiple players with kan
        # (approximation: n_kan_doras >= 4 means 4 kans happened)
        if n_kd >= 4:
            tags['state:four_kan_draw_candidate'] += 1

        # Check if this is a redeal (step count resets or round changes without
        # round advancement from terminated_round).  Heuristic: kyuushu + round
        # stays same.
        if at == 'kyuushu':
            # After kyuushu, check if next step's round is same (redeal)
            if step_idx + 1 < len(records):
                next_state = records[step_idx + 1]['state']
                curr_round = int(state.get('round_state.round', -1))
                next_round = int(next_state.get('round_state.round', -2))
                if curr_round == next_round:
                    tags['state:kyuushu_redeal'] += 1

    # ── Seed-level tags ──
    tags['seed:max_n_kan'] = max_n_kan
    tags['seed:action_families'] = ','.join(sorted(families_seen))
    tags['seed:had_riichi_win'] = 1 if had_riichi_before_win else 0
    tags['seed:n_steps'] = len(records)

    return dict(tags)


# ── Main scan ──

def scan_seed(filepath: str) -> dict:
    """Scan one golden file. Returns {seed, tags, path} dict."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    seed = data['seed']
    records = data['records']
    tags = detect_from_records(records)

    return {'seed': seed, 'path': filepath, 'tags': tags, 'n_steps': len(records)}


# ── Worker ──

def _worker_scan(filepath: str) -> dict:
    try:
        return scan_seed(filepath)
    except Exception:
        return {'seed': None, 'path': filepath, 'tags': {},
                'n_steps': 0, 'error': str(sys.exc_info()[1])}


# ── Reporting ──

# Define the rare paths we care about, grouped by category.
# Each entry: (tag_prefix, description, is_action_tag)
RARE_CATEGORIES = [
    # Action types (each individual)
    ('action:kyuushu',       '九種九牌',       'action'),
    ('action:open_kan',      '大明槓',         'action'),
    ('action:added_kan',     '加槓',           'action'),
    ('action:closed_kan',    '暗槓',           'action'),
    ('action:riichi',        '立直',           'action'),
    ('action:tsumo',         '自摸',           'action'),
    ('action:ron',           '栄和',           'action'),
    ('action:pon',           '碰',             'action'),
    ('action:pon_red',       '碰(赤)',         'action'),
    ('action:chi',           '吃',             'action'),
    ('action:chi_red',       '吃(赤)',         'action'),
    ('action:dummy',         'Dummy(局終)',    'action'),

    # State flags (binary: hit or not)
    ('state:haitei',              '海底摸月',           'state'),
    ('state:abortive_draw',       '荒牌流局',           'state'),
    ('state:can_after_kan',       '槍槓可能',           'state'),
    ('state:kan_declared',        '槓宣言中',           'state'),
    ('state:furiten_by_discard',  '振聴(河)',           'state'),
    ('state:furiten_by_pass',     '振聴(見逃)',         'state'),
    ('state:ippatsu_active',      '一発有効中',         'state'),
    ('state:n_kan_doras',         '裏宝牌あり',         'state'),
    ('state:riichi_declared',     '立直宣言済',         'state'),
    ('state:kyotaku',             '供託あり',           'state'),
    ('state:honba',               '本場継続',           'state'),
    ('state:ura_dora',            '裏ドラあり',         'state'),
    ('state:has_won',             '和了',               'state'),
    ('state:terminated_round',    '局終了',             'state'),
    ('state:can_robbing_kan',     '槍槓可能(明示)',     'state'),
    ('state:double_riichi',       'ダブル立直',         'state'),
    ('state:has_nagashi_mangan',  '流し満貫',           'state'),
    ('state:four_kan_draw_candidate', '四槓流れ候補',   'state'),
    ('state:kyuushu_redeal',      '九種九牌→連荘再配牌','state'),
]


def build_report(results: list) -> dict:
    """Aggregate scan results into a structured report."""
    all_tags_global = Counter()         # tag -> total steps across all seeds
    seed_hits = defaultdict(set)        # tag -> set of seed ids that hit it
    action_family_counts = Counter()    # action_family -> total occurrences
    seeds_by_action_family = defaultdict(set)  # family -> seeds that used it
    total_steps = 0

    seed_tags = {}  # seed -> tags dict
    errors = []

    for r in results:
        if r.get('error'):
            errors.append(r)
            continue
        seed = r['seed']
        tags = r['tags']
        seed_tags[seed] = tags
        total_steps += tags.get('seed:n_steps', 0)

        for tag, count in tags.items():
            if tag.startswith('action:'):
                all_tags_global[tag] += count
                seed_hits[tag].add(seed)
            elif tag.startswith('state:') and count > 0:
                all_tags_global[tag] += count
                seed_hits[tag].add(seed)

        # Action family coverage
        families_str = tags.get('seed:action_families', '')
        if families_str:
            for fam in families_str.split(','):
                action_family_counts[f'among_seeds:{fam}'] += 1
                seeds_by_action_family[fam].add(seed)

    return {
        'n_seeds': len([r for r in results if r.get('seed') is not None]),
        'n_errors': len(errors),
        'total_steps': total_steps,
        'tag_counts': dict(all_tags_global),
        'seed_hits': {tag: sorted(seeds) for tag, seeds in seed_hits.items()},
        'action_family_counts': dict(action_family_counts),
        'seeds_by_family': {fam: sorted(seeds) for fam, seeds in seeds_by_action_family.items()},
        'seed_tags': seed_tags,
        'errors': errors,
    }


def print_report(report: dict, print_seeds: bool = False):
    """Print a human-readable report."""
    print(f"\n{'='*70}")
    print(f"Scan complete: {report['n_seeds']} seeds, {report['total_steps']} total steps")
    if report['n_errors']:
        print(f"ERRORS: {report['n_errors']} files failed to load")
        for e in report['errors']:
            print(f"  {e['path']}: {e['error']}")
    print(f"{'='*70}")

    # ── Action family coverage ──
    print(f"\n── 动作类型覆盖 (action family coverage) ──")
    ALL_FAMILIES = ['discard', 'pon', 'chi', 'closed_kan', 'added_kan',
                    'open_kan', 'riichi', 'tsumo', 'ron', 'pass', 'kyuushu', 'dummy']
    for fam in ALL_FAMILIES:
        n_seeds = report['action_family_counts'].get(f'among_seeds:{fam}', 0)
        n_hits = report['tag_counts'].get(f'action:{fam}', 0)
        bar = '#' * min(40, n_seeds // max(1, report['n_seeds'] // 40))
        status = '[OK]' if n_seeds > 0 else '[!!] MISSING'
        print(f"  {status} {fam:<14s}  seeds={n_seeds:>4d}  hits={n_hits:>6d}  {bar}")

    # ── Detailed rare-path table ──
    print(f"\n── 稀有路径覆盖 (rare path coverage) ──")
    print(f"  {'标签':<32s} {'描述':<22s} {'触发种子':>6s} {'触发步骤':>6s}")
    print(f"  {'-'*32}  {'-'*22}  {'-'*6}  {'-'*6}")

    for tag, desc, _kind in RARE_CATEGORIES:
        n_seeds = len(report['seed_hits'].get(tag, set()))
        n_steps = report['tag_counts'].get(tag, 0)
        status = '[OK]' if n_seeds > 0 else '[!!]'
        print(f"  {status} {tag:<32s}  {desc:<22s}  {n_seeds:>4d}   {n_steps:>5d}")

    # ── Action histogram ──
    print(f"\n── 动作频率排行 (top 15) ──")
    action_tags = [(k, v) for k, v in report['tag_counts'].items() if k.startswith('action:')]
    action_tags.sort(key=lambda x: -x[1])
    for tag, count in action_tags[:15]:
        n_seeds = len(report['seed_hits'].get(tag, set()))
        pct = 100.0 * count / max(1, report['total_steps'])
        print(f"  {tag:<30s}  {count:>6d} steps ({pct:5.1f}%)  {n_seeds:>4d} seeds")

    # ── Print seed lists if requested ──
    if print_seeds:
        print(f"\n── 各路径对应种子列表 ──")
        for tag, desc, _kind in RARE_CATEGORIES:
            seeds = report['seed_hits'].get(tag, [])
            if seeds:
                print(f"\n  [{desc}] {tag}  ({len(seeds)} seeds):")
                # pretty-print in rows of 10
                for i in range(0, len(seeds), 10):
                    chunk = seeds[i:i+10]
                    print(f"    {', '.join(str(s) for s in chunk)}")

    # ── Missing coverage (gaps) ──
    missing = [tag for tag, _desc, _kind in RARE_CATEGORIES
               if len(report['seed_hits'].get(tag, set())) == 0]
    if missing:
        print(f"\n── [!] 未覆盖路径 (COVERAGE GAPS) ──")
        for tag in missing:
            desc = next((d for t, d, _ in RARE_CATEGORIES if t == tag), tag)
            print(f"  [!!] {tag}  — {desc}")


def export_seed_lists(report: dict, output_path: str):
    """Export seed lists as JSON, keyed by category tag."""
    export = {}
    for tag, desc, _kind in RARE_CATEGORIES:
        seeds = report['seed_hits'].get(tag, [])
        export[tag] = {'description': desc, 'n_seeds': len(seeds), 'seeds': seeds}

    # Also export by action family
    for fam, seeds in report['seeds_by_family'].items():
        export[f'family:{fam}'] = {'description': f'Any {fam} action',
                                   'n_seeds': len(seeds), 'seeds': seeds}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, indent=2, ensure_ascii=False)
    print(f"\nSeed lists exported to {output_path}")


def export_regression_seeds(report: dict, output_path: str):
    """Export a minimal Python module defining regression seed lists."""
    lines = [
        '"""Regression seed lists — auto-generated by scan_rare_paths.py.',
        '',
        'Seeds that cover every action type and rare path.',
        'Import and use with record_jax_golden.py / replay_*_against_golden.py.',
        '"""',
        '',
        '# Seeds that cover ALL 12 action families',
    ]
    # Collect seeds that together cover all families
    all_seeds = set()
    for fam in ['discard', 'pon', 'chi', 'closed_kan', 'added_kan',
                'open_kan', 'riichi', 'tsumo', 'ron', 'pass', 'kyuushu', 'dummy']:
        fam_seeds = report['seeds_by_family'].get(fam, [])
        if fam_seeds:
            # Take first 3 seeds for each family
            for s in fam_seeds[:3]:
                all_seeds.add(s)
    lines.append(f'FULL_COVERAGE = {sorted(all_seeds)}')
    lines.append('')

    for fam in ['discard', 'pon', 'chi', 'closed_kan', 'added_kan',
                'open_kan', 'riichi', 'tsumo', 'ron', 'pass', 'kyuushu', 'dummy']:
        seeds = report['seeds_by_family'].get(fam, [])
        lines.append(f'{fam.upper()}_SEEDS = {seeds}')

    lines.append('')
    lines.append('# Rare-path seeds')
    for tag, desc, _kind in RARE_CATEGORIES:
        seeds = report['seed_hits'].get(tag, [])
        if seeds:
            varname = tag.replace(':', '_').replace('-', '_').upper()
            lines.append(f'{varname} = {seeds}  # {desc}')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Regression seed module exported to {output_path}")


# ── Main ──

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Scan golden data for rare-path coverage')
    p.add_argument('-d', '--data-dir', default='golden_data',
                   help='Golden data directory (default: golden_data)')
    p.add_argument('-o', '--output', default=None,
                   help='Export seed lists as JSON')
    p.add_argument('--py-output', default=None,
                   help='Export regression seed lists as a Python module')
    p.add_argument('-j', '--jobs', type=int, default=1,
                   help='Parallel workers (default: 1)')
    p.add_argument('--print-seeds', action='store_true',
                   help='Print seed lists per category')
    p.add_argument('--summary-only', action='store_true',
                   help='Only print summary table (no per-seed output)')
    p.add_argument('--min-seeds', type=int, default=3,
                   help='Minimum seeds to report per category (default: 3)')
    args = p.parse_args()

    data_dir = args.data_dir

    # Find files
    files = sorted(f for f in os.listdir(data_dir) if f.startswith('golden_seed_'))
    if not files:
        print(f"No golden data found in {data_dir}/", file=sys.stderr)
        sys.exit(1)

    filepaths = [os.path.join(data_dir, f) for f in files]
    print(f"Scanning {len(filepaths)} seeds from {data_dir}/ ...", flush=True)

    t0 = time.time()

    # ── Execute ──
    n_workers = min(args.jobs, len(filepaths)) if args.jobs > 1 else 1

    if n_workers > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(n_workers) as pool:
            results = pool.map(_worker_scan, filepaths)
    else:
        results = [_worker_scan(fp) for fp in filepaths]

    dt = time.time() - t0

    # ── Report ──
    report = build_report(results)
    print_report(report, print_seeds=args.print_seeds)

    if args.output:
        export_seed_lists(report, args.output)

    if args.py_output:
        export_regression_seeds(report, args.py_output)

    print(f"\nScanned {report['n_seeds']} seeds in {dt:.1f}s")
