#!/usr/bin/env python3
"""Record JAX golden states for seeds. Run once, then replay against PT.

Usage:
  # Record all 10 default seeds
  python record_jax_golden.py

  # Record specific seeds
  python record_jax_golden.py 13 42 99

  # Record with custom output dir
  python record_jax_golden.py -o ./golden_data
"""
import os, sys, time, pickle
import jax; jax.config.update('jax_disable_jit', True)
import jax.numpy as jnp
import numpy as np
from mahjax.red_mahjong.cpu_env import RedMahjong as JaxEnv


def jax_to_dict(state):
    """Convert JAX state to a flat numpy dict for serialization."""
    d = {}
    # Top-level
    d['current_player'] = int(state.current_player)
    d['terminated'] = bool(state.terminated)
    d['truncated'] = bool(state.truncated)
    d['step_count'] = int(state.step_count)
    d['legal_action_mask'] = np.array(state.legal_action_mask)
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


def select_action(state):
    """Same action selection as test_multi_seed."""
    import numpy as _np
    legal = _np.where(_np.array(state.legal_action_mask))[0]
    for c in [73, 74, 72, 77, 75, 76, 78, 79, 80, 81, 82, 83] + list(range(70, 36, -1)):
        if c in legal:
            return int(c)
    return int(legal[0])


def record_seed(seed, output_dir):
    """Record a single seed, return (seed, path, n_steps, elapsed)."""
    t0 = time.time()
    jenv = JaxEnv(round_mode='single')
    state = jenv.init(jax.random.PRNGKey(seed))

    records = []
    for step in range(200):
        if bool(state.terminated) or bool(state.round_state.terminated_round):
            break
        action = select_action(state)
        state = jenv.step(state, action)
        records.append({'action': action, 'state': jax_to_dict(state)})

    path = os.path.join(output_dir, f'golden_seed_{seed:04d}.pkl')
    with open(path, 'wb') as f:
        pickle.dump({'seed': seed, 'records': records}, f)

    dt = time.time() - t0
    print(f"  seed={seed:4d}: {len(records)} steps saved → {path} ({dt:.0f}s)", flush=True)
    return seed, path, len(records), dt


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('seeds', nargs='*', type=int, help='Seeds to record (default: all 10)')
    p.add_argument('-o', '--output', default='golden_data', help='Output directory')
    args = p.parse_args()

    SEEDS = args.seeds if args.seeds else [1, 7, 13, 42, 99, 123, 256, 512, 1024, 2048]
    os.makedirs(args.output, exist_ok=True)

    print(f"Recording {len(SEEDS)} seeds to {args.output}/ ...", flush=True)
    total_steps = 0
    t0 = time.time()

    for seed in SEEDS:
        _, _, n, dt = record_seed(seed, args.output)
        total_steps += n

    total_t = time.time() - t0
    print(f"\nDone: {total_steps} total steps in {total_t:.0f}s")
    print(f"Golden data saved to {args.output}/")
