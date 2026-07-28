#!/usr/bin/env python3
"""Record JAX env golden data for mahjax_pt precision tests.

Generates .npz files containing JAX reference state traces.
These are replayed by PT tests without requiring JAX at test time.

Usage:
    python scripts/record_golden_env.py                    # 100 init + 200 playout seeds
    python scripts/record_golden_env.py --init-seeds 500   # 500 init seeds
    python scripts/record_golden_env.py --playout-seeds 0 50 100  # specific playout seeds
    python scripts/record_golden_env.py --output ./mahjax_pt/tests/golden

Requires: JAX, mahjax
"""

import os, sys, argparse, gc, time
import numpy as np

# ── Late JAX import (allows --help without JAX installed) ──
_JAX = None
_JaxEnv = None


def _init_jax():
    global _JAX, _JaxEnv
    if _JAX is not None:
        return
    import jax
    jax.config.update('jax_disable_jit', True)
    jax.config.update('jax_platform_name', 'cpu')
    from mahjax.red_mahjong.env import make as make_jax_env
    _JAX = jax
    _JaxEnv = make_jax_env("red_mahjong", round_mode="single", observe_type="dict")


def _extract_state_fields(state) -> dict:
    """Extract all comparable state fields into a flat dict of numpy arrays."""
    G = {}
    # Top-level
    G["current_player"] = np.asarray(int(state.current_player))
    G["terminated"] = np.asarray(bool(state.terminated))
    G["truncated"] = np.asarray(bool(state.truncated))
    G["step_count"] = np.asarray(int(state.step_count))
    G["legal_action_mask"] = np.asarray(state.legal_action_mask)
    G["rewards"] = np.asarray(state.rewards, dtype=np.float32)

    # PlayerStateArrays
    pp = state.players
    G["players.hand"] = np.asarray(pp.hand)
    G["players.hand_with_red"] = np.asarray(pp.hand_with_red)
    G["players.melds"] = np.asarray(pp.melds)
    G["players.meld_counts"] = np.asarray(pp.meld_counts)
    G["players.discard_counts"] = np.asarray(pp.discard_counts)
    G["players.river"] = np.asarray(pp.river)
    G["players.riichi"] = np.asarray(pp.riichi)
    G["players.riichi_declared"] = np.asarray(pp.riichi_declared)
    G["players.has_won"] = np.asarray(pp.has_won)
    G["players.n_kan"] = np.asarray(pp.n_kan)
    G["players.has_yaku"] = np.asarray(pp.has_yaku)
    G["players.is_hand_concealed"] = np.asarray(pp.is_hand_concealed)
    G["players.furiten_by_discard"] = np.asarray(pp.furiten_by_discard)
    G["players.furiten_by_pass"] = np.asarray(pp.furiten_by_pass)
    G["players.ippatsu"] = np.asarray(pp.ippatsu)
    G["players.fan"] = np.asarray(pp.fan)
    G["players.fu"] = np.asarray(pp.fu)

    # RoundState
    rs = state.round_state
    G["round_state.round"] = np.asarray(int(rs.round))
    G["round_state.dealer"] = np.asarray(int(rs.dealer))
    G["round_state.next_deck_ix"] = np.asarray(int(rs.next_deck_ix))
    G["round_state.last_deck_ix"] = np.asarray(int(rs.last_deck_ix))
    G["round_state.last_draw"] = np.asarray(int(rs.last_draw))
    G["round_state.last_player"] = np.asarray(int(rs.last_player))
    G["round_state.target"] = np.asarray(int(rs.target))
    G["round_state.draw_next"] = np.asarray(bool(rs.draw_next))
    G["round_state.is_haitei"] = np.asarray(bool(rs.is_haitei))
    G["round_state.is_abortive_draw_normal"] = np.asarray(bool(rs.is_abortive_draw_normal))
    G["round_state.terminated_round"] = np.asarray(bool(rs.terminated_round))
    G["round_state.honba"] = np.asarray(int(rs.honba))
    G["round_state.kyotaku"] = np.asarray(int(rs.kyotaku))
    G["round_state.n_kan_doras"] = np.asarray(int(rs.n_kan_doras))
    G["round_state.score"] = np.asarray(rs.score)
    G["round_state.deck"] = np.asarray(rs.deck)
    G["round_state.dora_indicators"] = np.asarray(rs.dora_indicators)

    return G


def record_init(output_dir: str, seeds: list):
    """Record JAX env.init state for each seed."""
    _init_jax()
    import jax

    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        path = os.path.join(output_dir, f"env_init_s{seed}.npz")
        if os.path.exists(path):
            print(f"  Skip seed {seed} (already exists)")
            continue

        key = jax.random.PRNGKey(seed)
        state = _JaxEnv.init(key)
        golden = _extract_state_fields(state)

        np.savez_compressed(path, **golden)
        print(f"  Seed {seed}: {len(golden)} fields saved")


def record_playout(output_dir: str, seeds: list, max_steps: int = 500):
    """Record full JAX env playout for each seed using rule_based_player."""
    _init_jax()
    import jax
    from mahjax.red_mahjong.players import rule_based_player

    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        path = os.path.join(output_dir, f"env_playout_s{seed}.npz")
        if os.path.exists(path):
            print(f"  Skip seed {seed} (already exists)")
            continue

        key = jax.random.PRNGKey(seed)
        state = _JaxEnv.init(key)

        steps_data = {}
        step_idx = 0

        while not bool(state.terminated) and step_idx < max_steps:
            action = rule_based_player(state, key)
            state = _JaxEnv.step(state, action)

            prefix = f"{step_idx}/"
            fields = _extract_state_fields(state)
            fields["action"] = np.asarray(int(action))
            for k, v in fields.items():
                steps_data[prefix + k] = v

            step_idx += 1

        steps_data["steps"] = np.asarray(step_idx)
        np.savez_compressed(path, **steps_data)
        print(f"  Seed {seed}: {step_idx} steps saved")


def main():
    parser = argparse.ArgumentParser(description="Record JAX env golden data")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: mahjax_pt/tests/golden)")
    parser.add_argument("--init-seeds", type=int, default=100,
                        help="Number of init seeds to record")
    parser.add_argument("--playout-seeds", type=int, default=200,
                        help="Number of playout seeds to record")
    parser.add_argument("--playout-offset", type=int, default=0,
                        help="Starting seed offset for playouts")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per playout")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(
            os.path.dirname(__file__), "..", "mahjax_pt", "tests", "golden")
    args.output = os.path.abspath(args.output)

    print(f"Recording JAX golden data to: {args.output}\n")

    # ── Init ──
    print(f"═══ Init ({args.init_seeds} seeds) ═══")
    t0 = time.time()
    record_init(args.output, list(range(args.init_seeds)))
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── Playout ──
    print(f"═══ Playout ({args.playout_seeds} seeds, offset={args.playout_offset}) ═══")
    t0 = time.time()
    seeds = list(range(args.playout_offset, args.playout_offset + args.playout_seeds))
    record_playout(args.output, seeds, args.max_steps)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("All golden data recorded.")


if __name__ == "__main__":
    main()
