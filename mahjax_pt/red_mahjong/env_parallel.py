# Copyright 2025 The Mahjax Authors.
# PyTorch eager-mode port of the red_mahjong environment — PURE PARALLEL version.
#
# All operations are batch-first: B environments processed simultaneously.
# Eager mode only — no torch.compile, no JIT.
# Control flow divergence handled via boolean masks.

from typing import Dict, List, Literal, Optional, Tuple
import torch
import numpy as np
import math

from .action import Action
from .constants import (
    FIRST_DRAW_IDX, MAX_DISCARDS_PER_PLAYER, NUM_PLAYERS,
    NUM_TILE_TYPES, NUM_TILE_TYPES_WITH_RED, NUM_PHYSICAL_TILES,
    DEAD_WALL_TILES, LEGAL_ACTION_SIZE, STARTING_POINTS,
    RIICHI_BET, MAX_MELDS_PER_PLAYER, MAX_DORA_INDICATORS,
    ZERO_MASK_1D, ZERO_MASK_2D,
)
from .meld import Meld, EMPTY_MELD
from .tile import River, Tile, EMPTY_RIVER
from .hand import Hand
from .shanten import Shanten
from .state import GameConfig, EnvState, default_state, default_game_config
from .types import Array, PRNGKey
from .yaku import Yaku
from .observation import _observe_dict, _observe_2D

# Import serial helpers (pure functions shared between versions)
from .env_serial import (
    Env,
    _resolve_game_config, _calc_wind, _is_first_turn,
    _accept_riichi, _append_meld_to_player, _is_waiting_tile,
    _append_action_history, _trigger_special_abortive_draw,
    CHI_ACTIONS,
)

# Import batch state
from .batch_state import (
    BatchState, BatchPlayerState, BatchRoundState,
    _default_batch_state, stack_states, unstack_state,
)


class RedMahjongParallel(Env):
    """Pure parallel mahjong environment — all B envs processed simultaneously.

    Uses BatchState for batch-first tensor operations.
    All control flow handled via boolean masks (zero Python loops in hot paths).

    For correctness verification, compare with env_serial.py.
    """

    def __init__(
        self,
        round_mode: Literal["single", "east", "half"] = "half",
        observe_type: str = "dict",
        order_points: List[int] = [30, 10, -10, -30],
        game_config: Optional[GameConfig] = None,
        next_round_style: Literal["auto", "dummy_share"] = "auto",
    ):
        if round_mode not in ("single", "east", "half"):
            raise ValueError(f"round_mode must be 'single', 'east', or 'half', got: {round_mode}")
        if observe_type == "2D":
            raise ValueError("observe_type '2D' is not yet implemented")
        if next_round_style not in ("auto", "dummy_share"):
            raise ValueError(f"next_round_style must be 'auto' or 'dummy_share', got: {next_round_style}")

        self.round_mode = round_mode
        self.one_round = round_mode == "single"
        self.round_limit = 4 if round_mode == "east" else 8
        self.observe_type = observe_type
        self.next_round_style = next_round_style
        self.order_points = order_points
        self.game_config = _resolve_game_config(game_config)

    # ── properties ──
    @property
    def id(self):
        return "red_mahjong_parallel"

    @property
    def version(self):
        return "pt-parallel-0.1"

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

    def observe(self, state):
        """Observe a single state (delegates to serial observe)."""
        if self.observe_type == "dict":
            return _observe_dict(state)
        elif self.observe_type == "2D":
            return _observe_2D(state)
        else:
            raise ValueError(f"Unknown observe_type: {self.observe_type}")

    def observe_batch(self, batch_state: BatchState):
        """Build observations for all B environments.

        Returns list of dicts (compatible with existing PPO pipeline).
        """
        observations = []
        for i in range(batch_state.B):
            s = unstack_state(batch_state, i)
            observations.append(self.observe(s))
        return observations

    # ── init / init_batch ──

    def init(self, key=None):
        """Initialize a single game state (for compatibility)."""
        # Delegate to serial init
        from .env_serial import RedMahjongSerial
        serial_env = RedMahjongSerial(
            round_mode=self.round_mode,
            observe_type=self.observe_type,
            order_points=self.order_points,
            game_config=self.game_config,
            next_round_style=self.next_round_style,
        )
        return serial_env.init(key)

    def init_batch(self, keys=None, num_envs=None) -> BatchState:
        """Initialize B game states in batch.

        Args:
            keys: List of torch.Generator or int seeds (length B), or None.
            num_envs: B — batch size (required if keys is None).
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

        # Initialize serial env for generating individual states
        from .env_serial import RedMahjongSerial
        serial_env = RedMahjongSerial(
            round_mode=self.round_mode,
            observe_type=self.observe_type,
            order_points=self.order_points,
            game_config=self.game_config,
            next_round_style=self.next_round_style,
        )

        states = [serial_env.init(key=k) for k in keys]
        return stack_states(states)

    # ── step (single, for compatibility) ──

    def step(self, state: EnvState, action, key=None, profile=False):
        """Single-env step (delegates to serial)."""
        from .env_serial import RedMahjongSerial
        serial_env = RedMahjongSerial(
            round_mode=self.round_mode,
            observe_type=self.observe_type,
            order_points=self.order_points,
            game_config=self.game_config,
            next_round_style=self.next_round_style,
        )
        return serial_env.step(state, action, key=key, profile=profile)

    # ── step_batch ──
    # The main parallel entry point. Processes all B envs simultaneously.

    def step_batch(self, states, actions, profile=False):
        """Process multiple env steps in batched tensor operations.

        Accepts either List[EnvState] (backward compat) or BatchState.
        Returns same type as input.

        Groups envs by action type: all actions are processed via
        dedicated batch handlers. No serial fallback for rare actions.
        """
        import time as _time

        if isinstance(states, BatchState):
            return self._step_batch_batchstate(states, actions, profile)

        # Backward compat: List[EnvState]
        _ts0 = _time.time() if profile else 0
        B = len(states)

        # Classify actions into groups
        groups = self._classify_actions(states, actions)

        if profile:
            _t_class = 1000 * (_time.time() - _ts0)

        # Process each action group with dedicated batch handler
        # Order matters: process discard first (largest group), then others
        if groups['discard'][0]:
            states = self._discard_batch_list(
                states, groups['discard'][0], groups['discard'][1], profile=profile)

        if groups['selfkan'][0]:
            for idx, action in zip(groups['selfkan'][0], groups['selfkan'][1]):
                states[idx] = self.step(states[idx], action)

        if groups['riichi'][0]:
            for idx in groups['riichi'][0]:
                states[idx] = self.step(states[idx], Action.RIICHI)

        if groups['ron'][0]:
            for idx in groups['ron'][0]:
                states[idx] = self.step(states[idx], Action.RON)

        if groups['tsumo'][0]:
            for idx in groups['tsumo'][0]:
                states[idx] = self.step(states[idx], Action.TSUMO)

        if groups['pon'][0]:
            for idx, action in zip(groups['pon'][0], groups['pon'][1]):
                states[idx] = self.step(states[idx], action)

        if groups['open_kan'][0]:
            for idx in groups['open_kan'][0]:
                states[idx] = self.step(states[idx], Action.OPEN_KAN)

        if groups['chi'][0]:
            for idx, action in zip(groups['chi'][0], groups['chi'][1]):
                states[idx] = self.step(states[idx], action)

        if groups['pass'][0]:
            for idx in groups['pass'][0]:
                states[idx] = self.step(states[idx], Action.PASS)

        if groups['kyuushu'][0]:
            for idx in groups['kyuushu'][0]:
                states[idx] = self.step(states[idx], Action.KYUUSHU)

        if groups['dummy'][0]:
            for idx in groups['dummy'][0]:
                states[idx] = self.step(states[idx], Action.DUMMY)

        # After all actions: advance terminated rounds
        if self.next_round_style == "auto" and not self.one_round:
            for i in range(B):
                if states[i].round_state.terminated_round and not states[i].terminated:
                    from .env_serial import RedMahjongSerial
                    s_env = RedMahjongSerial(
                        round_mode=self.round_mode, observe_type=self.observe_type,
                        order_points=self.order_points, game_config=self.game_config,
                        next_round_style=self.next_round_style)
                    states[i] = s_env._advance_to_next_round_auto(states[i])

        if profile:
            _t_total = 1000 * (_time.time() - _ts0)
            import logging
            _log = logging.getLogger("ppo")
            _log.info(f"step_batch (B={B}): "
                      f"discard={len(groups['discard'][0])} "
                      f"selfkan={len(groups['selfkan'][0])} "
                      f"riichi={len(groups['riichi'][0])} "
                      f"ron={len(groups['ron'][0])} "
                      f"tsumo={len(groups['tsumo'][0])} "
                      f"pon={len(groups['pon'][0])} "
                      f"total={_t_total:.0f}ms")

        return states

    def _step_batch_batchstate(self, batch_state: BatchState, actions, profile=False):
        """Step batch using BatchState (fully vectorized path)."""
        # For now, unstack, process, restack
        # This will be fully vectorized in later tasks (3.3-3.9)
        state_list = [unstack_state(batch_state, i) for i in range(batch_state.B)]
        state_list = self.step_batch(state_list, actions, profile=profile)
        return stack_states(state_list)

    # ── Action classification ──

    def _classify_actions(self, states, actions):
        """Classify actions by type for grouped batch processing."""
        groups = {
            'discard': ([], []),    # (indices, actions)
            'selfkan': ([], []),
            'riichi': ([], []),
            'ron': ([], []),
            'tsumo': ([], []),
            'pon': ([], []),
            'open_kan': ([], []),
            'chi': ([], []),
            'pass': ([], []),
            'kyuushu': ([], []),
            'dummy': ([], []),
        }

        for i, a in enumerate(actions):
            a = int(a) if isinstance(a, (torch.Tensor, np.generic)) else a

            # Handle terminated rounds first
            if states[i].round_state.terminated_round and not states[i].terminated:
                if self.next_round_style == "auto" and not self.one_round:
                    from .env_serial import RedMahjongSerial
                    s_env = RedMahjongSerial(
                        round_mode=self.round_mode, observe_type=self.observe_type,
                        order_points=self.order_points, game_config=self.game_config,
                        next_round_style=self.next_round_style)
                    states[i] = s_env._advance_to_next_round_auto(states[i])
                continue

            if a < Tile.NUM_TILE_TYPE_WITH_RED or a == Action.TSUMOGIRI:
                groups['discard'][0].append(i)
                groups['discard'][1].append(a)
            elif Action.is_selfkan(a):
                groups['selfkan'][0].append(i)
                groups['selfkan'][1].append(a)
            elif a == Action.RIICHI:
                groups['riichi'][0].append(i)
            elif a == Action.RON:
                groups['ron'][0].append(i)
            elif a == Action.TSUMO:
                groups['tsumo'][0].append(i)
            elif a in (Action.PON, Action.PON_RED):
                groups['pon'][0].append(i)
                groups['pon'][1].append(a)
            elif a == Action.OPEN_KAN:
                groups['open_kan'][0].append(i)
            elif Action.CHI_L <= a <= Action.CHI_R_RED:
                groups['chi'][0].append(i)
                groups['chi'][1].append(a)
            elif a == Action.PASS:
                groups['pass'][0].append(i)
            elif a == Action.KYUUSHU:
                groups['kyuushu'][0].append(i)
            elif a == Action.DUMMY:
                groups['dummy'][0].append(i)

        return groups

    # ── Batch discard (ported from old env.py) ──

    def _discard_batch_list(self, states, indices, actions, profile=False):
        """Batch discard for List[EnvState] — ported from old env.py _discard_batch."""
        import time as _time
        _t = {} if profile else None
        B = len(indices); P = 4; NUM_ACT = Action.NUM_ACTION
        device = states[0].players.hand.device
        _t0 = _time.time() if profile else 0

        # Safety filter
        safe_indices = []
        safe_actions = []
        for j, i in enumerate(indices):
            cp = states[i].current_player
            dc = int(states[i].players.discard_counts[cp].item())
            if dc >= MAX_DISCARDS_PER_PLAYER:
                continue
            if states[i].round_state.terminated_round and not states[i].terminated:
                if self.next_round_style == "auto" and not self.one_round:
                    from .env_serial import RedMahjongSerial
                    s_env = RedMahjongSerial(
                        round_mode=self.round_mode, observe_type=self.observe_type,
                        order_points=self.order_points, game_config=self.game_config,
                        next_round_style=self.next_round_style)
                    states[i] = s_env._advance_to_next_round_auto(states[i])
                continue
            if states[i].terminated:
                continue
            safe_indices.append(i)
            safe_actions.append(actions[j])

        if not safe_indices:
            return states

        indices = safe_indices
        actions = safe_actions
        B = len(indices)

        # Collect per-env scalars
        cps = [states[i].current_player for i in indices]
        tiles_l = [actions[j] if actions[j] < Tile.NUM_TILE_TYPE_WITH_RED
                    else int(states[indices[j]].round_state.last_draw) for j in range(B)]
        tiles = torch.tensor(tiles_l, dtype=torch.int32)
        d_counts = torch.tensor([int(states[indices[j]].players.discard_counts[cps[j]].item()) for j in range(B)], dtype=torch.int32)
        is_riichi_flags = torch.tensor([bool(states[indices[j]].players.riichi_declared[cps[j]].item()) for j in range(B)], dtype=torch.bool)
        is_tsumo_flags = torch.tensor([int(tiles_l[j]) == int(states[indices[j]].round_state.last_draw) for j in range(B)], dtype=torch.bool)
        cps_t = torch.tensor(cps, dtype=torch.int32)

        # Pre-stack hands and rivers
        hands_37 = torch.stack([states[i].players.hand_with_red for i in indices])  # (B, 4, 37)
        rivers_b = torch.stack([states[i].players.river.clone() for i in indices])  # (B, 4, 24)

        # ── 1. Hand sub + river add + AH update ──
        b_idx_full = torch.arange(B, device=device)
        hands_37[b_idx_full, cps_t, tiles.clamp(0, 36)] -= 1
        for j, i in enumerate(indices):
            cp = cps[j]
            states[i].players.hand_with_red[cp] = hands_37[j, cp]
            states[i].players.hand[cp] = Hand.to_34(hands_37[j, cp])

        # River add
        rivers_b = River.add_discard_batch(
            rivers_b, tiles, cps_t, d_counts, is_tsumo_flags, is_riichi_flags)
        for j, i in enumerate(indices):
            cp = cps[j]; d = min(int(d_counts[j].item()), MAX_DISCARDS_PER_PLAYER - 1)
            states[i].players.river = rivers_b[j]
            states[i].players.discards[cp, d] = tiles_l[j]
            states[i].players.discard_counts[cp] = min(int(states[i].players.discard_counts[cp].item()) + 1, MAX_DISCARDS_PER_PLAYER)
            states[i].players.riichi_declared[cp] = False
            states[i].players.ippatsu[cp] = False
            states[i].round_state.target = Tile.to_tile_type(tiles_l[j])
            states[i].round_state.last_player = cp

        # AH update
        ah_b = torch.stack([states[i].round_state.action_history.clone() for i in indices])
        is_empty = ah_b[:, 0, :] == -1
        empty_exists = is_empty.any(dim=1)
        first_empty = is_empty.int().argmax(dim=1)
        full_envs = ~empty_exists
        if full_envs.any():
            full_idx = b_idx_full[full_envs]
            ah_b[full_idx, :, :-1] = ah_b[full_idx, :, 1:].clone()
            first_empty = torch.where(full_envs, torch.tensor(199, device=device), first_empty)

        is_true_tsumo = torch.tensor([actions[j] == Action.TSUMOGIRI for j in range(B)],
                                     dtype=torch.bool, device=device)
        action_vals = tiles.clone()
        action_vals[is_true_tsumo] = Action.TSUMOGIRI
        tsumo_ints = is_tsumo_flags.to(torch.int8)

        ah_b[b_idx_full, 0, first_empty] = cps_t.to(torch.int8)
        ah_b[b_idx_full, 1, first_empty] = action_vals.to(torch.int8)
        ah_b[b_idx_full, 2, first_empty] = tsumo_ints

        for j, i in enumerate(indices):
            states[i].round_state.action_history = ah_b[j]

        # Furiten check
        for j, i in enumerate(indices):
            cp = cps[j]; t = tiles_l[j]
            h_after = hands_37[j, cp]
            shanten = int(states[i].round_state.shanten_current_player) if cp == states[i].current_player else -1
            if shanten <= 0:
                cr = torch.tensor([Hand.can_ron(h_after, tt) for tt in range(34)], dtype=torch.bool)
                if _is_waiting_tile(cr, t):
                    states[i].players.furiten_by_discard[cp] = True

        # ── 2. Haitei ──
        for i in indices:
            if states[i].round_state.is_haitei:
                if int(states[i].round_state.next_deck_ix) < int(states[i].round_state.last_deck_ix):
                    states[i].round_state.is_abortive_draw_normal = True

        # ── 3. Meld/ron mask ──
        targets = torch.tensor([int(states[i].round_state.target) for i in indices], dtype=torch.int32)
        target_tts = Tile.to_tile_type_tensor(targets)
        target_not_honor = target_tts < 27
        mask_4p = torch.zeros(B, P, NUM_ACT, dtype=torch.bool)

        has_yaku_all = torch.stack([states[i].players.has_yaku for i in indices])
        is_furiten_all = torch.stack([
            torch.stack([states[i].players.furiten_by_discard[p] | states[i].players.furiten_by_pass[p]
                         for p in range(P)]) for i in indices])
        is_riichi_all = torch.stack([states[i].players.riichi for i in indices])
        meld_counts_all = torch.stack([states[i].players.meld_counts for i in indices])
        n_kan_all = torch.tensor([int(states[i].players.n_kan.sum().item()) for i in indices])
        is_haitei_all = torch.tensor([bool(states[i].round_state.is_haitei) for i in indices])

        src_all = (cps_t.unsqueeze(1) - torch.arange(P, device=device).unsqueeze(0)) % 4

        # RON
        for p in range(P):
            is_discard = (cps_t == p)
            if is_discard.all():
                continue
            p_h37 = hands_37[:, p, :]
            has_yaku_p = has_yaku_all[:, p, 0].bool()
            is_furiten_p = is_furiten_all[:, p].bool()
            haitei_p = is_haitei_all
            can_ron_p = Hand.can_ron_batch(p_h37, target_tts)
            ron_ok = (has_yaku_p | haitei_p) & can_ron_p & ~is_furiten_p
            mask_4p[~is_discard, p, Action.RON] = ron_ok[~is_discard]

        # MELD
        cannot_meld_all = is_riichi_all | is_haitei_all.unsqueeze(1) | (meld_counts_all >= MAX_MELDS_PER_PLAYER)
        cannot_kan_all = (n_kan_all >= 4).unsqueeze(1)
        is_discard_all = (cps_t.unsqueeze(1) == torch.arange(P, device=device).unsqueeze(0))
        meld_ok = ~cannot_meld_all & ~is_discard_all

        # Chi
        src_is_3 = (src_all == 3)
        chi_ok = meld_ok & src_is_3 & target_not_honor.unsqueeze(1)
        if chi_ok.any():
            chi_matrix = Hand.can_chi_matrix_batch_4p(hands_37, targets, chi_ok)
            for chi_col in range(6):
                act = int(CHI_ACTIONS[chi_col].item())
                mask_4p[:, :, act] = chi_matrix[:, :, chi_col]

        # Pon
        pon_ok = meld_ok
        if pon_ok.any():
            no_red_pon = Hand.can_no_red_pon_batch_4p(hands_37, target_tts)
            red_pon = Hand.can_red_pon_batch_4p(hands_37, target_tts)
            mask_4p[:, :, Action.PON] = pon_ok & no_red_pon
            mask_4p[:, :, Action.PON_RED] = pon_ok & red_pon

        # Open Kan
        kan_ok = meld_ok & ~cannot_kan_all
        if kan_ok.any():
            open_kan = Hand.can_open_kan_batch_4p(hands_37, target_tts)
            mask_4p[:, :, Action.OPEN_KAN] = kan_ok & open_kan

        # PASS
        any_act = mask_4p.any(dim=2)
        mask_4p[any_act] = mask_4p[any_act] | torch.tensor(
            [a == Action.PASS for a in range(NUM_ACT)], dtype=torch.bool, device=device).unsqueeze(0)

        # ── 4. Next player + draw ──
        need_draw = []
        for bidx, i in enumerate(indices):
            cp = cps[bidx]
            mask_4p[bidx, cp] = False
            can_ron_v = mask_4p[bidx, :, Action.RON]
            can_pon_v = mask_4p[bidx, :, Action.PON] | mask_4p[bidx, :, Action.PON_RED]
            can_kan_v = mask_4p[bidx, :, Action.OPEN_KAN]
            can_chi_v = mask_4p[bidx, :, Action.CHI_L:Action.CHI_R_RED+1].any(dim=1)
            can_any_v = can_ron_v | can_pon_v | can_kan_v | can_chi_v
            if not can_any_v.any():
                states[i].current_player = (cp + 1) % 4
                states[i].round_state.target = -1
                states[i].round_state.draw_next = True
                states[i].round_state.last_player = cp
                if int(states[i].round_state.next_deck_ix) < int(states[i].round_state.last_deck_ix):
                    states[i].round_state.is_abortive_draw_normal = True
                    from .env_serial import RedMahjongSerial
                    s_env = RedMahjongSerial(
                        round_mode=self.round_mode, observe_type=self.observe_type,
                        order_points=self.order_points, game_config=self.game_config,
                        next_round_style=self.next_round_style)
                    states[i] = s_env._abortive_draw_normal(states[i])
                else:
                    need_draw.append(i)
            else:
                priority = torch.where(can_ron_v, 3, torch.where(can_kan_v, 2,
                            torch.where(can_pon_v, 1, torch.where(can_chi_v, 0, -1))))
                next_p = int(torch.argmax(priority).item())
                if can_ron_v.sum() > 1:
                    d = (torch.arange(4) - cp) % 4
                    d = torch.where(can_ron_v, d, torch.tensor(float('inf')))
                    next_p = int(torch.argmin(d).item())
                states[i].current_player = next_p
                states[i].legal_action_mask = mask_4p[bidx, next_p]
                states[i].players.legal_action_mask = mask_4p[bidx]
                states[i].round_state.last_player = cp
                states[i].round_state.draw_next = False
            states[i].round_state.can_after_kan = False

        # Batch draw
        if need_draw:
            states = self._draw_batch_list(states, need_draw, profile=profile)

        return states

    # ── Batch draw ──

    def _draw_batch_list(self, states, indices, profile=False):
        """Batch draw for List[EnvState] — ported from old env.py _draw_batch."""
        import time as _time
        B = len(indices)
        device = states[0].players.hand.device

        cp_hands_list = []
        is_haitei_list = []
        n_kan_list = []
        nxt_ixs_list = []
        can_riichi_bools = []
        cp_melds_list = []
        cp_meld_counts_list = []

        for j, i in enumerate(indices):
            states[i] = _accept_riichi(states[i])
            cp = states[i].current_player
            is_haitei = int(states[i].round_state.next_deck_ix) == int(states[i].round_state.last_deck_ix)
            states[i].round_state.is_haitei = is_haitei
            ix = int(states[i].round_state.next_deck_ix)
            new_tile = int(states[i].round_state.deck[ix].item())
            states[i].round_state.next_deck_ix = ix - 1
            states[i].round_state.last_draw = new_tile
            states[i].round_state.last_player = cp
            states[i].players.hand_with_red[cp] = Hand.add(states[i].players.hand_with_red[cp], new_tile)
            states[i].players.hand[cp] = Hand.to_34(states[i].players.hand_with_red[cp])

            cp_hands_list.append(states[i].players.hand_with_red[cp])
            is_haitei_list.append(is_haitei)
            n_kan_list.append(int(states[i].players.n_kan.sum().item()))
            nxt_ixs_list.append(int(states[i].round_state.next_deck_ix))

            nxt = int(states[i].round_state.next_deck_ix)
            lst = int(states[i].round_state.last_deck_ix)
            can_riichi_bools.append(
                not states[i].players.riichi[cp]
                and int(states[i].round_state.score[cp].item()) >= RIICHI_BET // 100
                and bool(states[i].players.is_hand_concealed[cp].item())
                and nxt - lst >= 4
            )
            cp_melds_list.append(states[i].players.melds[cp].clone())
            cp_meld_counts_list.append(int(states[i].players.meld_counts[cp].item()))

        # Batched checks
        cp_hands = torch.stack(cp_hands_list)
        is_haitei_t = torch.tensor(is_haitei_list, dtype=torch.bool, device=device)
        n_kan_t = torch.tensor(n_kan_list, dtype=torch.int32, device=device)
        nxt_ixs_t = torch.tensor(nxt_ixs_list, dtype=torch.int32, device=device)

        kan_allowed = ~is_haitei_t & (n_kan_t < 4)
        closed_kan_b = Hand.can_closed_kan_batch(cp_hands)

        added_kan_base_b = Hand.can_added_kan_batch(cp_hands)
        has_pon_meld = torch.zeros(B, 34, dtype=torch.bool, device=device)
        for j in range(B):
            for m_idx in range(cp_meld_counts_list[j]):
                m = int(cp_melds_list[j][m_idx].item())
                if m != EMPTY_MELD and Meld.is_pon(m):
                    tgt = int(Meld.target(m))
                    has_pon_meld[j, tgt] = True
        added_kan_b = added_kan_base_b & has_pon_meld
        kan_all_b = (closed_kan_b | added_kan_b) & kan_allowed.unsqueeze(1)

        can_tsumo_b = Hand.can_tsumo_batch(cp_hands)
        can_kyuushu_b = Hand.can_kyuushu_batch(cp_hands)
        is_first_turn_b = (nxt_ixs_t >= FIRST_DRAW_IDX - 4)

        can_riichi_b = torch.zeros(B, dtype=torch.bool, device=device)
        riichi_eligible = torch.tensor(can_riichi_bools, dtype=torch.bool, device=device)
        if riichi_eligible.any():
            eligible_hands = cp_hands[riichi_eligible]
            can_riichi_b[riichi_eligible] = Hand.can_riichi_batch(eligible_hands)

        # Per-env mask construction
        for j, i in enumerate(indices):
            cp = states[i].current_player
            hand = cp_hands[j]
            mask = torch.zeros(LEGAL_ACTION_SIZE, dtype=torch.bool)

            mask[:Tile.NUM_TILE_TYPE_WITH_RED] = (hand > 0)
            ld = int(states[i].round_state.last_draw)
            mask[Action.TSUMOGIRI] = (ld >= 0 and ld < Tile.NUM_TILE_TYPE_WITH_RED and hand[ld] > 0)

            if kan_allowed[j]:
                mask[37:71] = kan_all_b[j]

            if can_tsumo_b[j]:
                mask[Action.TSUMO] = True

            if can_riichi_b[j]:
                mask[Action.RIICHI] = True

            if is_first_turn_b[j] and can_kyuushu_b[j]:
                mask[Action.KYUUSHU] = True

            states[i].legal_action_mask = mask
            states[i].round_state.draw_next = False
            states[i].round_state.kan_declared = False
            states[i].round_state.target = -1
            states[i].round_state.shanten_current_player = Shanten.number(
                Hand.to_34(states[i].players.hand_with_red[cp]))
            if not bool(states[i].players.riichi[cp].item()):
                states[i].players.furiten_by_pass[cp] = False

        return states
