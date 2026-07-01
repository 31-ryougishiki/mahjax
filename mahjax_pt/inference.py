#!/usr/bin/env python3
"""Inference script: use a trained PPO/BC model to play mahjong.

Three usage modes:
  1. Interactive:     human vs trained agent (1v3, you control one seat)
  2. Auto-play:       one trained agent vs three baselines, print game log
  3. Batch eval:      run many games and report win-rate / avg-rank stats
"""

import torch
import argparse
from collections import defaultdict

from mahjax_pt.red_mahjong.env import make as make_env
from mahjax_pt.red_mahjong.players import random_player, rule_based_player
from mahjax_pt.examples.networks.red_network import ACNet


# ═══════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════

def load_model(model_path, device="cpu"):
    """Load a trained ACNet from a saved state_dict."""
    net = ACNet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net


# ═══════════════════════════════════════════════════════
# Policy function
# ═══════════════════════════════════════════════════════

@torch.no_grad()
def model_policy(net, device):
    """Return an actor_fn(state, rng) that uses the network for action selection.

    Uses argmax (greedy) by default. For stochastic play, set sample=True.
    """
    def actor(state, rng=None, sample=False):
        obs = net._env_observe(state) if hasattr(net, '_env_observe') else _make_obs(state)
        obs_dev = {k: v.to(device) for k, v in obs.items()}
        logits, _ = net(obs_dev)

        mask = state.legal_action_mask.to(device)
        logits = torch.where(mask, logits, torch.full_like(logits, -1e9))

        if sample:
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs.squeeze(0), 1).item()
        else:
            action = int(torch.argmax(logits).item())
        return action
    return actor


def _make_obs(state):
    """Minimal observation dict for network input (mirrors _observe_dict)."""
    c_p = state.current_player
    from mahjax_pt.red_mahjong.observation import hand_counts_to_idx
    hand_37 = state.players.hand_with_red[c_p]
    hand_14 = hand_counts_to_idx(hand_37)

    ah = state.round_state.action_history.clone()
    player_hist = ah[0, :].to(torch.int32)
    valid = player_hist >= 0
    rel_player = torch.remainder(player_hist - c_p, 4).to(ah.dtype)
    rel_player = torch.where(valid, rel_player, ah[0, :])
    ah[0, :] = rel_player

    scores_ordered = state.round_state.score[(torch.arange(4) + c_p) % 4]

    return {
        "hand": hand_14,
        "action_history": ah,
        "shanten_count": state.round_state.shanten_current_player,
        "furiten": state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p],
        "scores": scores_ordered,
        "round": state.round_state.round,
        "honba": state.round_state.honba,
        "kyotaku": state.round_state.kyotaku,
        "prevalent_wind": int(state.round_state.round) // 4,
        "seat_wind": int(state.round_state.seat_wind[c_p]),
        "dora_indicators": state.round_state.dora_indicators[:5],
    }


# ═══════════════════════════════════════════════════════
# Play one game
# ═══════════════════════════════════════════════════════

def play_one_game(env, actor_fns, seed=0, max_steps=2000, verbose=True):
    """Run a single game. actor_fns = {seat: actor_fn} for each of 4 seats.

    Returns: dict with final scores, step count, winner seats
    """
    gen = torch.Generator().manual_seed(seed)
    state = env.init(gen)
    step_count = 0

    while not state.terminated and step_count < max_steps:
        cp = state.current_player
        actor = actor_fns.get(cp, random_player)
        action = actor(state, gen)
        prev_terminated_round = state.round_state.terminated_round

        state = env.step(state, action)
        step_count += 1

        if verbose and state.rewards.abs().sum() > 0:
            print(f"  Step {step_count:4d} | Player {cp} | action={action:3d} | "
                  f"rewards={state.rewards.tolist()}")

    scores = state.round_state.score.tolist()
    if verbose:
        print(f"\n  Game over after {step_count} steps")
        print(f"  Final scores: {scores}")
    return {"scores": scores, "steps": step_count}


# ═══════════════════════════════════════════════════════
# Batch evaluation
# ═══════════════════════════════════════════════════════

def batch_eval(model_path, num_games=100, device="cpu", opponent="random"):
    """Run many games and report stats: 1 trained agent vs 3 opponents."""
    env = make_env("red_mahjong", round_mode="single", observe_type="dict")
    net = load_model(model_path, device)
    agent_actor = model_policy(net, device)

    if opponent == "rule_based":
        baseline_actor = rule_based_player
    else:
        baseline_actor = random_player

    stats = defaultdict(float)
    for i in range(num_games):
        seed = 1000 + i
        agent_seat = seed % 4
        actors = {s: baseline_actor for s in range(4)}
        actors[agent_seat] = agent_actor

        result = play_one_game(env, actors, seed=seed, verbose=False)
        scores = result["scores"]

        # Rank the agent (1=best, 4=worst)
        agent_score = scores[agent_seat]
        sorted_scores = sorted(scores, reverse=True)
        rank = sorted_scores.index(agent_score) + 1

        stats["games"] += 1
        stats["avg_rank"] += rank
        stats["total_score"] += agent_score
        if rank == 1:
            stats["wins"] += 1
        if rank <= 2:
            stats["top2"] += 1

    n = stats["games"]
    print(f"\n=== Evaluation: {model_path} ===")
    print(f"  Games:          {int(n)}")
    print(f"  vs:             {opponent} × 3")
    print(f"  Win rate:       {stats['wins']/n:.2%}")
    print(f"  Top-2 rate:     {stats['top2']/n:.2%}")
    print(f"  Avg rank:       {stats['avg_rank']/n:.3f}")
    print(f"  Avg score diff: {(stats['total_score']/n - 250):.1f}")
    return stats


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a trained mahjong agent")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model (.pt)")
    parser.add_argument("--mode", choices=["eval", "play"], default="eval",
                        help="eval=multi-game stats, play=single game with log")
    parser.add_argument("--num_games", type=int, default=100,
                        help="Number of games for eval mode")
    parser.add_argument("--opponent", choices=["random", "rule_based"], default="random",
                        help="Opponent type")
    parser.add_argument("--device", default="cpu",
                        help="Device: cpu, cuda:0, npu:0, etc.")
    args = parser.parse_args()

    if args.mode == "eval":
        batch_eval(args.model, num_games=args.num_games,
                   device=args.device, opponent=args.opponent)
    else:
        env = make_env("red_mahjong", round_mode="single", observe_type="dict")
        net = load_model(args.model, args.device)
        agent = model_policy(net, args.device)

        baseline = rule_based_player if args.opponent == "rule_based" else random_player
        agent_seat = 0
        actors = {s: baseline for s in range(4)}
        actors[agent_seat] = agent

        print(f"Agent (seat {agent_seat}) vs 3× {args.opponent}")
        play_one_game(env, actors, seed=42, verbose=True)
