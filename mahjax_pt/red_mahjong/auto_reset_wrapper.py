"""Auto-reset wrapper for RL training (PyTorch eager port).

When the environment terminates, automatically reset to a new game,
preserving the final reward for the transition step.
"""

import torch


def auto_reset(step_fn, init_fn):
    """Wrap env.step + env.init to auto-reset on termination.

    Args:
        step_fn: env.step(state, action, key) -> state
        init_fn: env.init(key) -> state

    Returns:
        wrapped_step_fn(state, action, key) -> state
    """

    def wrapped_step_fn(state, action, key=None):
        # If state is already terminated, clear flags before stepping
        if state.terminated or state.truncated:
            state.terminated = False
            state.truncated = False
            state.rewards.zero_()

        # Execute the step
        state = step_fn(state, action, key)

        # If the step produced a terminal state, transition to a fresh one
        if state.terminated or state.truncated:
            new_state = init_fn(key)
            # Preserve the terminal signal and reward from the ending state
            new_state.terminated = state.terminated
            new_state.truncated = state.truncated
            new_state.rewards = state.rewards
            state = new_state

        return state

    return wrapped_step_fn
