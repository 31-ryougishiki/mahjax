# Copyright 2025 The Mahjax Authors.
# PyTorch eager-mode port of the red_mahjong environment.
#
# This is the COMPATIBILITY FACADE.
# - Use env_serial.RedMahjongSerial for correctness verification (vs JAX).
# - Use env_parallel.RedMahjongParallel for GPU/NPU-accelerated batched training.
#
# The RedMahjong class here delegates to the chosen backend.
# Default is 'serial' for deterministic single-env operation.

from typing import List, Literal, Optional
import torch

from .action import Action
from .state import GameConfig, EnvState

# Re-export base classes
from .env_serial import Env, RedMahjongSerial
from .env_parallel import RedMahjongParallel


class RedMahjong(Env):
    """Facade environment class — delegates to serial or parallel backend.

    Args:
        backend: 'serial' (default) or 'parallel'.
        round_mode: 'single', 'east', or 'half'.
        observe_type: 'dict' or '2D'.
        order_points: uma point list.
        game_config: GameConfig instance.
        next_round_style: 'auto' or 'dummy_share'.
    """

    def __init__(
        self,
        backend: Literal["serial", "parallel"] = "serial",
        round_mode: Literal["single", "east", "half"] = "half",
        observe_type: str = "dict",
        order_points: List[int] = [30, 10, -10, -30],
        game_config: Optional[GameConfig] = None,
        next_round_style: Literal["auto", "dummy_share"] = "auto",
    ):
        self._backend = backend

        if backend == "parallel":
            self._impl = RedMahjongParallel(
                round_mode=round_mode,
                observe_type=observe_type,
                order_points=order_points,
                game_config=game_config,
                next_round_style=next_round_style,
            )
        else:
            self._impl = RedMahjongSerial(
                round_mode=round_mode,
                observe_type=observe_type,
                order_points=order_points,
                game_config=game_config,
                next_round_style=next_round_style,
            )

        # Forward commonly accessed attributes
        self.round_mode = self._impl.round_mode
        self.one_round = self._impl.one_round
        self.round_limit = self._impl.round_limit
        self.observe_type = self._impl.observe_type
        self.next_round_style = self._impl.next_round_style
        self.order_points = self._impl.order_points
        self.game_config = self._impl.game_config

    @property
    def id(self):
        return self._impl.id

    @property
    def version(self):
        return self._impl.version

    @property
    def num_players(self):
        return self._impl.num_players

    @property
    def num_actions(self):
        return self._impl.num_actions

    @property
    def observation_shape(self):
        return self._impl.observation_shape

    def init(self, key=None):
        return self._impl.init(key)

    def step(self, state, action, key=None, profile=False):
        return self._impl.step(state, action, key=key, profile=profile)

    def observe(self, state):
        return self._impl.observe(state)

    def observe_batch(self, batch_state):
        """Batch observe — available when backend='parallel'."""
        if hasattr(self._impl, 'observe_batch'):
            return self._impl.observe_batch(batch_state)
        raise NotImplementedError("observe_batch is only available with backend='parallel'")

    def step_batch(self, states, actions, profile=False):
        """Batch step — delegates to parallel backend's step_batch."""
        if hasattr(self._impl, 'step_batch'):
            return self._impl.step_batch(states, actions, profile=profile)
        # Fallback for serial backend: iterate
        result_states = []
        for s, a in zip(states, actions):
            result_states.append(self._impl.step(s, a))
        return result_states

    def init_batch(self, keys=None, num_envs=None, device=None):
        """Batch init — available when backend='parallel'."""
        if hasattr(self._impl, 'init_batch'):
            return self._impl.init_batch(keys=keys, num_envs=num_envs, device=device)
        raise NotImplementedError("init_batch is only available with backend='parallel'")

    def reinit_terminated_batch(self, batch_state, keys=None):
        """Re-initialize terminated envs in a BatchState — parallel backend only."""
        if hasattr(self._impl, 'reinit_terminated_batch'):
            return self._impl.reinit_terminated_batch(batch_state, keys=keys)
        raise NotImplementedError(
            "reinit_terminated_batch is only available with backend='parallel'")


def make(env_name="red_mahjong", backend="serial", **kwargs):
    """Factory function — creates a RedMahjong environment.

    Args:
        env_name: 'red_mahjong' (default).
        backend: 'serial' (default) or 'parallel'.
        **kwargs: passed to RedMahjong constructor.
    """
    if env_name == "red_mahjong":
        return RedMahjong(backend=backend, **kwargs)
    raise ValueError(f"Unknown env: {env_name}")
