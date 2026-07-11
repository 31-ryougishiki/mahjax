# Copyright 2025 The Mahjax Authors.
# PyTorch eager-mode port of the red_mahjong environment — FULLY VECTORIZED.
#
# All operations use batch-first BatchState tensors.
# Each action handler mirrors env_serial 1:1 but operates on B envs at once.
#
# Architecture (after mixin split):
#   env_parallel.py            ← this file: orchestrator (init / step / dispatch)
#   env_parallel_handlers.py   ← HandlersMixin: 11 action handlers + draw
#   env_parallel_internals.py  ← InternalsMixin: mask builders + settlement + yaku
#
# For correctness verification, compare with env_serial.py (reference).

from typing import Dict, List, Literal, Optional, Tuple
import torch
import numpy as np
import dataclasses

from .action import Action
from .constants import (
    FIRST_DRAW_IDX, MAX_DISCARDS_PER_PLAYER, NUM_PLAYERS,
    NUM_TILE_TYPES, NUM_TILE_TYPES_WITH_RED,
    DEAD_WALL_TILES, LEGAL_ACTION_SIZE, STARTING_POINTS,
    RIICHI_BET, MAX_MELDS_PER_PLAYER, MAX_DORA_INDICATORS,
)
from .meld import Meld, EMPTY_MELD
from .tile import River, Tile
from .hand import Hand
from .state import GameConfig, EnvState
from .observation import _observe_dict, _observe_2D, _observe_dict_batch
from .yaku import Yaku

# Import serial helpers (pure functions) and the reference implementation
from .env_serial import (
    Env, _resolve_game_config, _resolve_env_config,
    _is_first_turn, _trigger_special_abortive_draw, _set_tile_type_action,
)
from .shanten import Shanten

# Import batch state
from .batch_state import (
    BatchState, stack_states, unstack_state,
)

# Import mixin classes (split from this file for readability)
from .env_parallel_internals import InternalsMixin, _copy_dataclass_row
from .env_parallel_handlers import HandlersMixin


def _batch_state_to_device(bs: BatchState, device: torch.device) -> BatchState:
    """Move all tensors in a BatchState to the target device (recursively)."""
    changes = {}
    for field in dataclasses.fields(bs):
        val = getattr(bs, field.name)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            changes[field.name] = val.to(device)
        elif dataclasses.is_dataclass(val):
            # Recurse into nested dataclass (BatchPlayerState / BatchRoundState)
            sub = {}
            for sf in dataclasses.fields(val):
                sv = getattr(val, sf.name)
                if sv is not None and isinstance(sv, torch.Tensor):
                    sub[sf.name] = sv.to(device)
            if sub:
                changes[field.name] = dataclasses.replace(val, **sub)
    return dataclasses.replace(bs, **changes)


class RedMahjongParallel(HandlersMixin, InternalsMixin, Env):
    """Fully vectorized parallel mahjong environment.

    All operations use BatchState tensors with batch-first layout.
    Each action handler operates on all B envs simultaneously using
    boolean masks and advanced tensor indexing.

    Action handlers live in HandlersMixin (env_parallel_handlers.py).
    Mask builders, settlement, and yaku live in InternalsMixin (env_parallel_internals.py).
    """

    def __init__(
        self,
        round_mode: Literal["single", "east", "half"] = "half",
        observe_type: str = "dict",
        order_points: List[int] = [30, 10, -10, -30],
        game_config: Optional[GameConfig] = None,
        next_round_style: Literal["auto", "dummy_share"] = "auto",
    ):
        self.__dict__.update(_resolve_env_config(
            round_mode, observe_type, next_round_style, order_points, game_config))

        # Internal serial env for delegate calls (complex mutations, init)
        from .env_serial import RedMahjongSerial
        self._serial = RedMahjongSerial(
            round_mode=self.round_mode,
            observe_type=self.observe_type,
            order_points=self.order_points,
            game_config=self.game_config,
            next_round_style=self.next_round_style,
        )

        # Perf profiling (enabled by _step_batch_bs when called with profile=True)
        self._perf = None  # set to {} when profiling starts

    # ── properties ──
    @property
    def id(self):
        return "red_mahjong_parallel"

    @property
    def version(self):
        return "pt-parallel-0.3"

    @property
    def num_players(self):
        return 4

    @property
    def num_actions(self):
        return Action.NUM_ACTION

    @property
    def observation_shape(self):
        return (37,)

    @property
    def _illegal_action_penalty(self):
        return -10.0

    # ── observe ──
    def observe(self, state):
        return self._serial.observe(state)

    def observe_batch(self, batch_state: BatchState):
        """Batch observation — directly builds (B, ...) tensors, no unstack/stack.

        Returns dict of (B, ...) tensors on the same device as the BatchState,
        ready for direct network consumption.
        """
        if self.observe_type == "dict":
            return _observe_dict_batch(batch_state)
        raise NotImplementedError(
            f"observe_batch only supports observe_type='dict', got '{self.observe_type}'")

    # ── init ──
    def init(self, key=None):
        return self._serial.init(key)

    def init_batch(self, keys=None, num_envs=None, device=None) -> BatchState:
        """Batch initialize B environments.

        Args:
            keys: List of seeds/Generators, or None to auto-generate.
            num_envs: Number of environments (used if keys is None).
            device: torch.device for output tensors (default: cpu).
        """
        if keys is None:
            if num_envs is None:
                raise ValueError("Either keys or num_envs must be provided")
            B = num_envs
            keys = [None] * B
        else:
            B = len(keys)
            if num_envs is not None and num_envs != B:
                raise ValueError(f"keys length {B} != num_envs {num_envs}")
        states = [self._serial.init(key=k) for k in keys]
        bs = stack_states(states)
        if device is not None:
            device = torch.device(device)
            if device.type != 'cpu':
                bs = _batch_state_to_device(bs, device)
        return bs

    # ── step (single) ──
    def step(self, state: EnvState, action, key=None, profile=False):
        return self._serial.step(state, action, key=key, profile=profile)

    # ── Perf profiling helpers ──
    def _perf_add(self, name, dt, n_envs=0):
        """Accumulate timing for a named sub-operation (used by mixin handlers)."""
        if self._perf is None:
            return
        entry = self._perf.setdefault(name, {'time': 0.0, 'calls': 0, 'active_calls': 0, 'total_envs': 0})
        entry['time'] += dt
        entry['calls'] += 1
        if n_envs > 0:
            entry['active_calls'] += 1
            entry['total_envs'] += n_envs

    def get_perf_summary(self):
        """Return sorted list of (name, time, calls, active_calls, total_envs) tuples."""
        if not self._perf:
            return []
        items = []
        for name, e in self._perf.items():
            items.append((name, e['time'], e['calls'], e['active_calls'], e['total_envs']))
        items.sort(key=lambda x: -x[1])  # sort by time descending
        return items

    def reinit_terminated_batch(self, bs: BatchState,
                                 keys: Optional[List] = None) -> BatchState:
        """Re-initialize terminated environments in-place within a BatchState.

        Creates fresh game states for terminated slots and splices them into
        the main BatchState.  Non-terminated envs are untouched.

        Args:
            bs: BatchState — modified in-place and returned.
            keys: Optional list of seeds (one per terminated env).

        Returns:
            The same BatchState with terminated slots replaced by fresh games.
        """
        if not bs.terminated.any():
            return bs

        device = bs.players.hand.device
        term_idx = torch.where(bs.terminated)[0]
        K = len(term_idx)

        # Generate keys for re-initialized envs
        if keys is None:
            keys = [None] * K

        # Create fresh BatchState for terminated subset
        new_bs = self.init_batch(keys=keys, num_envs=K)
        if device.type != 'cpu':
            new_bs = _batch_state_to_device(new_bs, device)

        # Splice new state into terminated slots
        from .env_parallel_internals import _copy_dataclass_row
        new_idx = torch.arange(K, device=device)
        _copy_dataclass_row(bs, term_idx, new_bs, new_idx)

        return bs

    # ═════════════════════════════════════════════════════════════
    # step_batch — main entry point
    # ═════════════════════════════════════════════════════════════

    def step_batch(self, states, actions, profile=False, device=None):
        """Process multiple env steps.

        Args:
            states: List[EnvState] or BatchState.
            actions: List/Tensor of action ints (length B).
            device: torch.device — if set and states is List[EnvState],
                move the internal BatchState to this device before processing.

        Returns:
            Same type as input (List[EnvState] or BatchState).
        """
        if isinstance(states, BatchState):
            return self._step_batch_bs(states, actions, profile=profile)

        # List[EnvState]: stack → process → unstack
        B = len(states)
        actions_t = torch.tensor([int(a) if isinstance(a, (torch.Tensor, np.generic)) else a
                                   for a in actions], dtype=torch.int32)
        bs = stack_states(states)
        if device is not None:
            device = torch.device(device)
            if device.type != 'cpu':
                bs = _batch_state_to_device(bs, device)
                actions_t = actions_t.to(device)
        bs = self._step_batch_bs(bs, actions_t, profile=profile)
        return [unstack_state(bs, i) for i in range(B)]

    def _step_batch_bs(self, bs: BatchState, actions: torch.Tensor, profile=False):
        """Core batched step on BatchState. actions: (B,) int32."""
        import time as _time
        _ts0 = _time.time() if profile else 0
        B = bs.B
        device = bs.players.hand.device

        # ── 1. Handle terminated envs ──
        term = bs.terminated
        if term.any():
            bs.rewards[term] = 0.0

        # ── 2. Handle terminated rounds (advance before step) ──
        if self.next_round_style == "auto" and not self.one_round:
            need_adv = bs.round_state.terminated_round & ~bs.terminated
            if need_adv.any():
                bs = self._advance_round_batch(bs, need_adv)

        # ── 3. Classify actions (fully vectorized, no per-env loops) ──
        active = ~bs.terminated & ~bs.round_state.terminated_round
        a = actions

        is_discard = active & (a < Tile.NUM_TILE_TYPE_WITH_RED)
        is_tsumogiri = active & (a == Action.TSUMOGIRI)
        is_selfkan = active & (a >= 37) & (a < 71)
        is_riichi = active & (a == Action.RIICHI)
        is_ron = active & (a == Action.RON)
        is_tsumo = active & (a == Action.TSUMO)
        is_pon = active & ((a == Action.PON) | (a == Action.PON_RED))
        is_open_kan = active & (a == Action.OPEN_KAN)
        is_chi = active & ((a >= Action.CHI_L) & (a <= Action.CHI_R_RED))
        is_pass = active & (a == Action.PASS)
        is_kyuushu = active & (a == Action.KYUUSHU)
        is_dummy = active & (a == Action.DUMMY)

        # Merge discard+tsumogiri for handlers that treat them similarly
        is_any_discard = is_discard | is_tsumogiri

        # ── 4. Process actions in game-logic order ──
        if self._perf is not None:
            _t = _time.time
            _p = self._perf
            def _timed(name, fn, *args):
                _t0 = _t()
                result = fn(*args)
                dt = _t() - _t0
                entry = _p.setdefault(name, {'time': 0.0, 'calls': 0, 'active_calls': 0, 'total_envs': 0})
                entry['time'] += dt
                entry['calls'] += 1
                # args[0] is always the mask tensor
                mask = args[0] if args else None
                if mask is not None and isinstance(mask, torch.Tensor):
                    n = int(mask.sum().item())
                    if n > 0:
                        entry['active_calls'] += 1
                        entry['total_envs'] += n
                return result

            bs = _timed('riichi',      self._riichi_batch, bs, is_riichi)
            bs = _timed('ron',         self._ron_batch, bs, is_ron)
            bs = _timed('tsumo',       self._tsumo_batch, bs, is_tsumo)
            bs = _timed('pon',         self._pon_batch, bs, is_pon, a)
            bs = _timed('open_kan',    self._open_kan_batch, bs, is_open_kan)
            bs = _timed('chi',         self._chi_batch, bs, is_chi, a)
            bs = _timed('selfkan',     self._selfkan_batch, bs, is_selfkan, a)
            bs = _timed('discard',     self._discard_batch, bs, is_any_discard, a)
            bs = _timed('pass',        self._pass_batch, bs, is_pass)
            bs = _timed('kyuushu',     self._kyuushu_batch, bs, is_kyuushu)
            bs = _timed('dummy',       self._dummy_batch, bs, is_dummy)
        else:
            bs = self._riichi_batch(bs, is_riichi)
            bs = self._ron_batch(bs, is_ron)
            bs = self._tsumo_batch(bs, is_tsumo)
            bs = self._pon_batch(bs, is_pon, a)
            bs = self._open_kan_batch(bs, is_open_kan)
            bs = self._chi_batch(bs, is_chi, a)
            bs = self._selfkan_batch(bs, is_selfkan, a)
            bs = self._discard_batch(bs, is_any_discard, a)
            bs = self._pass_batch(bs, is_pass)
            bs = self._kyuushu_batch(bs, is_kyuushu)
            bs = self._dummy_batch(bs, is_dummy)

        # ── 5. Update step counters ──
        bs.step_count[active] += 1

        # ── 6. Single-round termination ──
        if self.one_round:
            bs.terminated |= bs.round_state.terminated_round

        # ── 7. Auto round advance for multi-round ──
        if self.next_round_style == "auto" and not self.one_round:
            need_adv = bs.round_state.terminated_round & ~bs.terminated
            if need_adv.any():
                bs = self._advance_round_batch(bs, need_adv)

        # ── 8. Set legal mask for terminated envs ──
        if bs.terminated.any():
            bs.legal_action_mask[bs.terminated] = True

        if profile:
            _t_total = 1000 * (_time.time() - _ts0)
            import logging
            _log = logging.getLogger("ppo")
            _log.info(f"step_batch (B={B}): total={_t_total:.0f}ms")

        return bs
