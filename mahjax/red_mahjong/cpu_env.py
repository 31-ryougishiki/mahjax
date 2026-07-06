# Copyright 2025 The Mahjax Authors.
# Pure CPU version of the red_mahjong environment.
#
# Disables JIT compilation so JAX runs in eager mode without XLA compilation.
# This avoids the compilation hangs/slowness on non-GPU environments.
#
# Usage:
#   from mahjax.red_mahjong.cpu_env import RedMahjong
#   # Or: export MAHJAX_CPU_MODE=0 to keep JIT enabled
#
# Performance on CPU:
#   init: ~7s, step: ~2.7s (vs 7s/15s with JIT, vs <0.01s for PT eager)
#   JAX's immutable state copies dominate the remaining 2.7s overhead.

import os

_DISABLE_JIT = os.environ.get('MAHJAX_CPU_MODE', '1') != '0'

if _DISABLE_JIT:
    import jax
    jax.config.update('jax_disable_jit', True)

# Re-export main classes and factory
from mahjax.red_mahjong.env import RedMahjong
from mahjax.red_mahjong.state import (
    GameConfig, State, EnvState, PlayerStateArrays, RoundState,
    default_state, default_game_config,
)
from mahjax.red_mahjong.action import Action


def make(round_mode="single", **kwargs):
    """Create a RedMahjong CPU-mode environment."""
    return RedMahjong(round_mode=round_mode, **kwargs)
