"""Verify script: run a single game of red_mahjong with random players."""

import torch
from mahjax_pt.red_mahjong.env import make
from mahjax_pt.red_mahjong.players import random_player

def main():
    env = make("red_mahjong", round_mode="single", observe_type="dict")
    print(f"Env: {env.id} | num_players={env.num_players} | num_actions={env.num_actions}")
    print(f"Round mode: {env.round_mode} | Next round style: {env.next_round_style}")

    state = env.init(42)
    gen = torch.Generator().manual_seed(123)

    step_count = 0
    total_rewards = torch.zeros(4)

    while not state.terminated and step_count < 2000:
        action = random_player(state, gen)
        state = env.step(state, action)

        if state.rewards.abs().sum() > 0:
            print(f"Step {step_count:4d} | player={state.current_player} | "
                  f"action={action:3d} | rewards={state.rewards.tolist()}")

        total_rewards += state.rewards
        step_count += 1

    print(f"\n=== Game Over ===")
    print(f"Steps: {step_count}")
    print(f"Terminated: {state.terminated}")
    print(f"Final scores: {state.round_state.score.tolist()}")
    print(f"Cumulative rewards: {total_rewards.tolist()}")
    print(f"Success!")


if __name__ == "__main__":
    main()
