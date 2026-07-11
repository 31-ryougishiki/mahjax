#!/usr/bin/env python3
"""Record JAX golden states for seeds. Run once, then replay against PT.

By default uses two exploration mechanisms (15% each) to increase path diversity:
  --pass-epsilon 0.15  →  15% chance of PASS instead of greedy meld
  --no-meld-prob  0.15  →  15% chance of refusing pon/chi/open-kan (→ menzen)

Pass --pass-epsilon 0 --no-meld-prob 0 for pure greedy behaviour.

Usage:
  # Record all 10 default seeds (15% PASS + 15% no-meld exploration)
  python record_jax_golden.py

  # Record specific seeds
  python record_jax_golden.py 13 42 99

  # Record with custom output dir
  python record_jax_golden.py -o ./golden_data

  # Parallel (Windows-safe): each worker lazily initializes JAX
  python record_jax_golden.py -j 10 4000 4001 ... -o ./golden_data

  # Pure greedy (no exploration)
  python record_jax_golden.py --pass-epsilon 0 --no-meld-prob 0

  # Aggressive menzen bias (30% refuse meld) + fixed action seed
  python record_jax_golden.py --no-meld-prob 0.3 --action-seed 42
"""
import gc, os, sys, time, pickle, threading
import numpy as np

# ── stack size ─────────────────────────────────────────────────────
# Windows default 1 MB can overflow during XLA eager execution.
# Set early so that any threads created afterwards inherit the larger
# limit.  Must happen before JAX / XLA starts its thread pool.
if sys.platform == 'win32':
    try:
        threading.stack_size(8 * 1024 * 1024)  # 8 MB
    except Exception:
        pass

# ── lazy JAX init ──────────────────────────────────────────────────
# JAX imports are deferred so that multiprocessing spawn workers
# can each initialize XLA in their own process without racing.
_JAX = None
_JaxEnv = None
_Shanten = None


def _init_jax():
    """Init JAX (once per process). Safe to call multiple times."""
    global _JAX, _JaxEnv, _Shanten
    if _JAX is not None:
        return
    import jax
    jax.config.update('jax_disable_jit', True)
    from mahjax.red_mahjong.env_optim import RedMahjongOptim as _JaxEnvCls
    from mahjax.red_mahjong.shanten import Shanten as _ShantenCls
    _JAX = jax
    _JaxEnv = _JaxEnvCls
    _Shanten = _ShantenCls


def _get_jax():
    _init_jax()
    return _JAX, _JaxEnv


def jax_to_dict(state):
    """Convert JAX state to a flat numpy dict for serialization."""
    d = {}
    # Top-level
    d['current_player'] = int(state.current_player)
    d['terminated'] = bool(state.terminated)
    d['truncated'] = bool(state.truncated)
    d['step_count'] = int(state.step_count)
    d['legal_action_mask'] = np.array(state.legal_action_mask)  # env-level (87,)
    d['players.legal_action_mask'] = np.array(state.players.legal_action_mask)  # (4, 87)
    d['rewards'] = np.array(state.rewards)

    # Player state
    jp = state.players
    for f in ['hand', 'hand_with_red', 'melds', 'meld_counts', 'riichi',
              'riichi_declared', 'furiten_by_discard', 'furiten_by_pass',
              'is_hand_concealed', 'has_won', 'n_kan', 'discard_counts',
              'river', 'has_yaku', 'ippatsu', 'double_riichi', 'fan', 'fu',
              'can_win', 'pon', 'has_nagashi_mangan', 'hand_ids', 'hand_counts',
              'drawn_tile', 'meld_tiles', 'meld_info', 'discards', 'discard_info',
              'riichi_step']:
        if hasattr(jp, f):
            d[f'players.{f}'] = np.array(getattr(jp, f))

    # Round state
    jr = state.round_state
    for f in ['round', 'honba', 'kyotaku', 'dealer', 'next_deck_ix', 'last_deck_ix',
              'last_draw', 'last_player', 'target', 'n_kan_doras', 'shanten_current_player',
              'dummy_count', 'round_limit']:
        if hasattr(jr, f):
            d[f'round_state.{f}'] = int(getattr(jr, f))
    for f in ['terminated_round', 'draw_next', 'is_haitei', 'kan_declared',
              'can_after_kan', 'can_robbing_kan', 'is_abortive_draw_normal']:
        if hasattr(jr, f):
            d[f'round_state.{f}'] = bool(getattr(jr, f))
    for f in ['deck', 'score', 'dora_indicators', 'ura_dora_indicators',
              'seat_wind', 'init_wind', 'order_points', 'action_history']:
        if hasattr(jr, f):
            d[f'round_state.{f}'] = np.array(getattr(jr, f))

    return d


# Actions that consume another player's discard (pon / chi / open-kan).
# Skipping these when exploring forces the hand toward concealed (menzen)
# development, covering riichi, pinfu, iipeikou, ikkitsuukan, etc.
_MELD_ACTIONS = frozenset({75, 76, 77, 78, 79, 80, 81, 82, 83})
# 75=PON, 76=PON_RED, 77=OPEN_KAN, 78=CHI_L, 79=CHI_L_RED,
# 80=CHI_M, 81=CHI_M_RED, 82=CHI_R, 83=CHI_R_RED


def select_action(state, rng=None, pass_epsilon=0.0, no_meld_prob=0.0):
    """Action selection with optional exploration.

    Two independent exploration mechanisms:

    1. **pass_epsilon** — with this probability, choose PASS (84) over the
       greedy pick when PASS is legal.  Covers *furiten_by_pass* and
       multi-player-response paths.

    2. **no_meld_prob** — with this probability, refuse to call pon / chi /
       open-kan even when available, forcing the hand toward concealed
       development.  Covers *riichi, pinfu, iipeikou, ikkitsuukan* and
       other menzen-only yaku.

    Args:
        state:        JAX EnvState
        rng:          numpy RandomState (created per-seed for reproducibility)
        pass_epsilon: float in [0, 1) — prob of choosing PASS over greedy.
        no_meld_prob: float in [0, 1) — prob of skipping pon/chi/open-kan.
    """
    legal = np.where(np.array(state.legal_action_mask))[0]
    legal_set = set(legal)

    # ── epsilon-PASS exploration (takes priority) ──
    if pass_epsilon > 0 and rng is not None and 84 in legal_set:
        if rng.random() < pass_epsilon:
            return 84  # Action.PASS

    # ── Decide whether to suppress melds this step ──
    reject_melds = (
        no_meld_prob > 0
        and rng is not None
        and bool(legal_set & _MELD_ACTIONS)        # only when melds are actually legal
        and rng.random() < no_meld_prob
    )

    # ── Greedy priority ──
    if reject_melds:
        # Skip pon/chi/open-kan → fall through to selfkan, riichi,
        # tsumo/ron, then discard.  This simulates a "closed-hand" player.
        priority = [73, 74, 72] + list(range(70, 36, -1))  # tsumo/ron/riichi, selfkans
    else:
        priority = [73, 74, 72, 77, 75, 76, 78, 79, 80, 81, 82, 83] + list(range(70, 36, -1))

    for c in priority:
        if c in legal_set:
            return int(c)

    # ── Shanten-guided discard selection ──
    # Fallback: any legal action (discards, pass, kyuushu, dummy).
    # For discard actions, pick the tile that minimises shanten so the hand
    # stays on track toward tenpai — this dramatically increases riichi,
    # ron, and tsumo coverage vs the old behaviour of always discarding
    # the lowest-indexed tile (legal[0]).
    legal_discards = [a for a in legal if 0 <= a <= 36 or a == 71]
    if legal_discards:
        cp = int(state.current_player)
        hand_14 = state.players.hand[cp]                 # (34,) after draw

        # Compute pre-discard shanten once to decide strategy.
        # When far from tenpai, use a cheap isolation heuristic (no JAX calls).
        # When close (shanten <= 2), do full per-discard shanten evaluation.
        pre_shanten = int(_Shanten.number(hand_14))

        if pre_shanten <= 2:
            # ── Full shanten evaluation (close to tenpai) ──
            best_a = None
            best_s = 999
            for a in legal_discards:
                if a <= 33:
                    base = a
                elif a <= 36:
                    base = {34: 4, 35: 13, 36: 22}[a]
                else:  # a == 71 (tsumogiri)
                    base = int(state.round_state.last_draw) % 34
                hand_13 = hand_14.at[base].set(hand_14[base] - 1)
                s = int(_Shanten.number(hand_13))
                if s < best_s:
                    best_s = s
                    best_a = a
            return best_a
        else:
            # ── Isolation heuristic (far from tenpai) ──
            # Discard tiles that have fewest same-suit neighbours in hand.
            # This efficiently clears isolated honours and terminals without
            # paying the cost of full shanten computation.
            hand_np = np.array(hand_14)
            best_a = None
            best_score = -999  # higher = more isolated (better to discard)

            for a in legal_discards:
                if a <= 33:
                    base = a
                elif a <= 36:
                    base = {34: 4, 35: 13, 36: 22}[a]
                else:  # a == 71 (tsumogiri)
                    base = int(state.round_state.last_draw) % 34

                suit = base // 9
                if suit == 3:
                    # Honour: isolated unless paired
                    score = 100 if hand_np[base] == 1 else -100
                else:
                    # Numbered tile: count same-suit neighbours within ±2
                    suit_start = suit * 9
                    suit_end = suit_start + 9
                    pos = base % 9
                    neighbours = 0
                    for offset in [-2, -1, 1, 2]:
                        ni = suit_start + pos + offset
                        if suit_start <= ni < suit_end and hand_np[ni] > 0:
                            neighbours += 1
                    # High score = isolated (good to discard)
                    # Low score = connected (keep)
                    score = 10 - neighbours

                if score > best_score:
                    best_score = score
                    best_a = a
            return best_a

    # pass (84), kyuushu (85), dummy (86) — keep original behaviour
    return int(legal[0])


def record_seed(seed, output_dir, pass_epsilon=0.0, no_meld_prob=0.0, action_seed=None):
    """Record a single seed, return (seed, path, n_steps, elapsed).

    Args:
        seed:          JAX PRNG seed for game initialisation.
        output_dir:    Directory to write golden_seed_XXXX.pkl.
        pass_epsilon:  Probability of choosing PASS over greedy when PASS is legal.
        no_meld_prob:  Probability of skipping pon/chi/open-kan to force menzen.
        action_seed:   Seed for the action-selection RNG (None = use ``seed``).
    """
    jax, JaxEnv = _get_jax()
    rng = np.random.RandomState(seed if action_seed is None else action_seed)
    try:
        t0 = time.time()
        sys.stderr.write(f"[pid={os.getpid()}] seed={seed} starting...\n"); sys.stderr.flush()
        jenv = JaxEnv(round_mode='single')
        sys.stderr.write(f"[pid={os.getpid()}] seed={seed} init...\n"); sys.stderr.flush()
        state = jenv.init(jax.random.PRNGKey(seed))
        sys.stderr.write(f"[pid={os.getpid()}] seed={seed} running steps...\n"); sys.stderr.flush()

        # Save initial state first (right after init)
        init_state = jax_to_dict(state)
        records = []
        for step in range(200):
            if bool(state.terminated) or bool(state.round_state.terminated_round):
                break
            action = select_action(state, rng=rng, pass_epsilon=pass_epsilon,
                                    no_meld_prob=no_meld_prob)
            state = jenv.step(state, action)
            records.append({'action': action, 'state': jax_to_dict(state)})
            if step % 20 == 19:
                sys.stderr.write(f"[pid={os.getpid()}] seed={seed} step={step+1}\n"); sys.stderr.flush()

        path = os.path.join(output_dir, f'golden_seed_{seed:04d}.pkl')
        with open(path, 'wb') as f:
            pickle.dump({'seed': seed, 'init_state': init_state, 'records': records}, f)

        dt = time.time() - t0
        sys.stderr.write(f"[pid={os.getpid()}] seed={seed} done: {len(records)} steps ({dt:.0f}s)\n"); sys.stderr.flush()
        print(f"  seed={seed:4d}: {len(records)} steps saved → {path} ({dt:.0f}s)", flush=True)
        return seed, path, len(records), dt
    except Exception as e:
        import traceback
        sys.stderr.write(f"[pid={os.getpid()}] seed={seed} CRASH: {e}\n{traceback.format_exc()}\n"); sys.stderr.flush()
        raise
    finally:
        # Help JAX / XLA release per-seed allocations promptly.
        del jenv, state, records, init_state
        gc.collect()


def _worker_initializer():
    """Pool initializer: init JAX once per worker, before any seed is processed."""
    _init_jax()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Record JAX golden states for seeds. Run once, then replay against PT.')
    p.add_argument('seeds', nargs='*', type=int, help='Seeds to record (default: all 10)')
    p.add_argument('-o', '--output', default='golden_data', help='Output directory')
    p.add_argument('-j', '--jobs', type=int, default=1, help='Parallel workers')
    p.add_argument('--pass-epsilon', type=float, default=0.15,
                   help='Probability of choosing PASS (84) over greedy when legal (0.0–1.0). '
                        'Use 0.15 to trigger furiten_by_pass and pass codepaths.')
    p.add_argument('--no-meld-prob', type=float, default=0.15,
                   help='Probability of skipping pon/chi/open-kan to force menzen '
                        'development.  Set to 0.0 for pure greedy meld behaviour.')
    p.add_argument('--action-seed', type=int, default=None,
                   help='Seed for the action-selection RNG (default: same as game seed). '
                        'Use a fixed value to reproduce the same action perturbations across seeds.')
    args = p.parse_args()

    SEEDS = args.seeds if args.seeds else [1, 7, 13, 42, 99, 123, 256, 512, 1024, 2048]
    os.makedirs(args.output, exist_ok=True)

    # Build work items: (seed, output_dir, pass_epsilon, no_meld_prob, action_seed)
    work_items = [(s, args.output, args.pass_epsilon, args.no_meld_prob, args.action_seed)
                  for s in SEEDS]

    if getattr(args, 'jobs', 1) > 1:
        from multiprocessing import get_context
        n_workers = min(args.jobs, len(SEEDS))
        print(f"Recording {len(SEEDS)} seeds with {n_workers} workers...", flush=True)
        if args.pass_epsilon > 0:
            print(f"  pass-epsilon={args.pass_epsilon:.2f}", flush=True)
        if args.no_meld_prob > 0:
            print(f"  no-meld-prob={args.no_meld_prob:.2f}", flush=True)
        # maxtasksperchild=1: restart each worker after every seed so
        # XLA arena / stack growth can never accumulate across seeds.
        with get_context('spawn').Pool(
            n_workers,
            initializer=_worker_initializer,
            maxtasksperchild=1,
        ) as pool:
            results = pool.starmap(record_seed, work_items)
        total_steps = sum(r[2] for r in results)
    else:
        print(f"Recording {len(SEEDS)} seeds (serial)...", flush=True)
        if args.pass_epsilon > 0:
            print(f"  pass-epsilon={args.pass_epsilon:.2f}", flush=True)
        if args.no_meld_prob > 0:
            print(f"  no-meld-prob={args.no_meld_prob:.2f}", flush=True)
        _init_jax()  # init in main process before the loop
        total_steps = 0
        for seed, od, pe, nm, a_seed in work_items:
            _, _, n, dt = record_seed(seed, od, pass_epsilon=pe, no_meld_prob=nm,
                                      action_seed=a_seed)
            total_steps += n
            gc.collect()  # aggressive cleanup between serial seeds

    print(f"\nDone: {total_steps} total steps")
    print(f"Golden data saved to {args.output}/")
