# Copyright 2025 The Mahjax Authors.
# PyTorch eager-mode port of the red_mahjong environment — PURE SERIAL version.
#
# This is the reference implementation for correctness verification.
# Code structure mirrors JAX mahjax/red_mahjong/env.py 1:1.
# Every key method is annotated with the corresponding JAX line range.
#
# No batch/parallel methods here — see env_parallel.py for the vectorized version.

from typing import Dict, List, Literal, Optional, Tuple
import torch
import random as py_random
import numpy as np
import math

from .action import Action
from .constants import (
    DORA_ARRAY, FALSE, FIRST_DRAW_IDX, TILE_RANGE, TRUE,
    ZERO_MASK_1D, ZERO_MASK_2D, MAX_DISCARDS_PER_PLAYER,
    NUM_PLAYERS, NUM_TILE_TYPES, NUM_TILE_TYPES_WITH_RED,
    NUM_PHYSICAL_TILES, DEAD_WALL_TILES, LEGAL_ACTION_SIZE,
    SENTINEL_TILE_ID, SENTINEL_MELD_VALUE, COPIES_PER_TILE,
    STARTING_POINTS, TARGET_POINTS, HONBA_BONUS, RIICHI_BET,
    MAX_HAND_TILES, MAX_MELDS_PER_PLAYER, MAX_DORA_INDICATORS,
)
from .hand import Hand
from .meld import Meld, EMPTY_MELD
from .shanten import Shanten
from .state import GameConfig, State, PlayerStateArrays, RoundState, EnvState, default_state, default_game_config
from .tile import River, Tile, EMPTY_RIVER
from .types import Array, PRNGKey
from .yaku import Yaku
from .observation import _observe_dict, _observe_2D


class Env:
    """Base class mimicking mahjax.core.Env API."""
    def init(self, key=None):
        raise NotImplementedError

    def step(self, state, action, key=None):
        raise NotImplementedError

    def observe(self, state):
        raise NotImplementedError

    @property
    def num_players(self):
        raise NotImplementedError

    @property
    def num_actions(self):
        raise NotImplementedError


# ─── helpers ──────────────────────────────────────────────────────

def _resolve_game_config(game_config=None):
    return default_game_config() if game_config is None else game_config


def _live_wall_end_ix(state):
    return int(state.round_state.last_deck_ix)


def _set_tile_type_action(mask, tile_type, value):
    """Set mask[tile_type] = value; also set red counterpart for fives."""
    tt = Tile.to_tile_type(tile_type)
    mask = mask.clone()
    mask[tt] = value
    if Tile.is_tile_type_five(tt):
        mask[Tile.to_red(tt)] = value
    return mask


def _has_red_discard_action(mask):
    return mask[Action.PON] | mask[Action.PON_RED]


CHI_ACTIONS = torch.tensor(
    [Action.CHI_L, Action.CHI_L_RED, Action.CHI_M, Action.CHI_M_RED,
     Action.CHI_R, Action.CHI_R_RED], dtype=torch.int32)


def _special_abortive_draw_mask():
    mask = ZERO_MASK_2D.clone()
    mask[:, Action.KYUUSHU] = True
    return mask


def _trigger_special_abortive_draw(state):
    state.legal_action_mask = _special_abortive_draw_mask()
    state.round_state.draw_next = False
    state.round_state.kan_declared = False
    state.round_state.is_abortive_draw_normal = False
    return state


def _append_meld_to_player(state, meld_packed, caller, discard_idx, discarder):
    """Record a meld on the caller's meld list and mark the discarder's river.

    JAX: river marker goes on the DISCARDER's river slot (the tile being claimed).
    The SRC field encodes the relative position (discarder - caller) % 4.
    """
    p = int(caller)
    n = int(state.players.meld_counts[p].item())
    state.players.melds[p, n] = meld_packed
    state.players.meld_counts[p] += 1

    disc = int(discarder) if discarder is not None else 0
    # Relative source: who did we take the tile from?
    rel_src = (disc - p) % 4

    state.players.river = River.add_meld(
        state.players.river, Meld.action(meld_packed),
        torch.tensor(disc), torch.tensor(discard_idx),
        rel_src)
    return state


def _accept_riichi(state):
    """Accept riichi during _draw (matches JAX _accept_riichi at _draw top).

    JAX _accept_riichi uses `last_player`, not `current_player`, because
    the riichi declaration was made on the PREVIOUS step. The actual payment
    and flag-setting happens HERE during the next draw.
    """
    # JAX: env.py L797
    lp = int(state.round_state.last_player)
    already_riichi = bool(state.players.riichi[lp].item())
    has_declared = (not already_riichi) and bool(state.players.riichi_declared[lp].item())

    if not already_riichi and has_declared:
        # Pay riichi bet
        state.round_state.score[lp] -= RIICHI_BET // 100
        state.rewards[lp] += -10.0  # -1000 points in reward units
        state.round_state.kyotaku += 1

        # Set riichi flag
        state.players.riichi[lp] = True
        state.players.riichi_declared[lp] = False

        # Ippatsu and Double Riichi
        state.players.ippatsu[lp] = True
        is_double = _is_first_turn(state.round_state.next_deck_ix) and \
            (int(state.players.meld_counts.sum().item()) == 0)
        state.players.double_riichi[lp] = is_double

    return state


def _is_waiting_tile(can_ron, tile):
    """Check if a tile is among the waiting tiles (can_ron is 34-bool vector)."""
    tt = Tile.to_tile_type(tile)
    return bool(can_ron[tt].item())


def _calc_wind(east_player):
    """Calculate seat winds from east player index."""
    return (torch.arange(4) - east_player) % 4


def _is_first_turn(next_deck_ix):
    """True if within first 4 draws of the round (kyuushu / double riichi window)."""
    return next_deck_ix >= FIRST_DRAW_IDX - 4  # >= 79, matching JAX


def _append_action_history(state, action):
    """Append one entry to the action history (3,200). Shift left, write at last occupied slot."""
    ah = state.round_state.action_history
    for col in range(ah.shape[1]):
        if ah[0, col] == -1:
            ah[0, col] = state.current_player
            ah[1, col] = action
            ah[2, col] = -1  # tsumogiri will be set by caller
            return state
    # Full; shift left
    ah[:, :-1] = ah[:, 1:].clone()
    ah[0, -1] = state.current_player
    ah[1, -1] = action
    ah[2, -1] = -1
    return state


# ═══════════════════════════════════════════════════════════════════
# Pure Serial Environment
# ═══════════════════════════════════════════════════════════════════

class RedMahjongSerial(Env):
    """Pure serial mahjong environment — one env, one step at a time.

    This is the REFERENCE implementation for correctness verification.
    Code structure mirrors JAX mahjax/red_mahjong/env.py 1:1.

    Use env_parallel.py for GPU/NPU-accelerated batched training.
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
        return "red_mahjong_serial"

    @property
    def version(self):
        return "pt-serial-0.1"

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
        if self.observe_type == "dict":
            return _observe_dict(state)
        elif self.observe_type == "2D":
            return _observe_2D(state)
        else:
            raise ValueError(f"Unknown observe_type: {self.observe_type}")

    # ── init ──
    # JAX: RedMahjong.init → _init lines 460-540
    def init(self, key=None):
        """Initialize a new game state."""
        if key is None:
            gen = torch.Generator()
        elif isinstance(key, torch.Generator):
            gen = key
        else:
            gen = torch.Generator().manual_seed(int(key) if isinstance(key, (int, float)) else int(key.item()))

        state = default_state()
        gc = self.game_config
        state.round_state.round_limit = self.round_limit

        # Build wall: 4 copies of each tile type
        deck_values = []
        for t in range(34):
            for _ in range(4):
                deck_values.append(t)

        deck = torch.tensor(deck_values, dtype=torch.int8)
        perm = torch.randperm(136, generator=gen)
        deck = deck[perm]

        state.round_state.deck = deck
        state.round_state.next_deck_ix = FIRST_DRAW_IDX  # 83
        state.round_state.last_deck_ix = DEAD_WALL_TILES
        state.round_state.draw_next = True

        # Seat winds — random dealer (matching JAX)
        if isinstance(key, torch.Generator):
            dealer = int(torch.randint(0, 4, (1,), generator=key).item())
        else:
            dealer = 0
        state.round_state.init_wind = _calc_wind(dealer)
        state.round_state.seat_wind = _calc_wind(dealer)
        state.round_state.dealer = dealer
        state.round_state.order_points = torch.tensor(self.order_points, dtype=torch.int32)
        state.round_state.score = torch.full((4,), STARTING_POINTS // 100, dtype=torch.int32)

        # Initial dora indicators
        state.round_state.dora_indicators[0] = int(deck[9].item())
        state.round_state.ura_dora_indicators[0] = int(deck[8].item())

        # Deal: last 52 tiles form the initial 4x13 hands
        state.players.hand_with_red = Hand.make_init_hand(deck)
        for p in range(4):
            state.players.hand[p] = Hand.to_34(state.players.hand_with_red[p])
            state.players.is_hand_concealed[p] = True

        # First draw for the dealer
        state.current_player = state.round_state.dealer
        state = self._draw(state)
        state.round_state.shanten_current_player = \
            Shanten.number(Hand.to_34(state.players.hand_with_red[state.current_player]))
        return state

    # ═════════════════════════════════════════════════════════════
    # STEP
    # ═════════════════════════════════════════════════════════════
    # JAX: _step_auto / _step_dummy_share lines 1886-2050

    def step(self, state: EnvState, action, key=None, profile=False):
        """Execute one step in the environment."""
        action = int(action) if isinstance(action, (torch.Tensor, np.generic)) else action

        if profile:
            import time as _time
            _t = {}

        # Check if terminated
        if state.terminated:
            state.rewards.zero_()
            return state

        # Check for illegal action
        if not state.legal_action_mask[action]:
            return self._step_with_illegal_action(state, state.current_player)

        # Append to action history
        state = _append_action_history(state, action)

        # Dispatch based on action type
        is_terminal_before = state.terminated

        if action < Tile.NUM_TILE_TYPE_WITH_RED:
            if profile: _t0 = _time.time()
            state = self._discard(state, action)
            if profile: _t['discard'] = _time.time() - _t0
        elif Action.is_selfkan(action):
            if profile: _t0 = _time.time()
            tile_type = action - 37
            cp = state.current_player
            is_added = False
            for m_idx in range(int(state.players.meld_counts[cp].item())):
                m = int(state.players.melds[cp, m_idx].item())
                if m != EMPTY_MELD and Meld.is_pon(m) and Meld.target(m) == tile_type:
                    is_added = True
                    break
            import sys
            sys.stderr.write(f'[SELFKAN] cp={cp} action={action} tile_type={tile_type} is_added={is_added} n_kan_sum={int(state.players.n_kan.sum().item())} meld_counts_cp={int(state.players.meld_counts[cp].item())}\n')
            sys.stderr.flush()
            state = self._selfkan(state, action, is_added)
            sys.stderr.write(f'[SELFKAN after] hand_cp_sum={int(state.players.hand_with_red[cp].sum().item())} n_kan_sum={int(state.players.n_kan.sum().item())} last_draw={int(state.round_state.last_draw)} last_player={int(state.round_state.last_player)}\n')
            sys.stderr.flush()
            if profile: _t['selfkan'] = _time.time() - _t0
        elif action == Action.TSUMOGIRI:
            if profile: _t0 = _time.time()
            state = self._discard(state, state.round_state.last_draw, is_tsumogiri_override=True)
            if profile: _t['discard'] = _t.get('discard', 0) + _time.time() - _t0
        elif action == Action.RIICHI:
            if profile: _t0 = _time.time()
            state = self._riichi(state)
            if profile: _t['riichi'] = _time.time() - _t0
        elif action == Action.RON:
            if profile: _t0 = _time.time()
            state = self._ron(state)
            if profile: _t['ron'] = _time.time() - _t0
        elif action == Action.TSUMO:
            if profile: _t0 = _time.time()
            state = self._tsumo(state)
            if profile: _t['tsumo'] = _time.time() - _t0
        elif action in (Action.PON, Action.PON_RED):
            if profile: _t0 = _time.time()
            state = self._pon(state, action)
            if profile: _t['pon'] = _time.time() - _t0
        elif action == Action.OPEN_KAN:
            if profile: _t0 = _time.time()
            state = self._open_kan(state)
            if profile: _t['open_kan'] = _time.time() - _t0
        elif Action.CHI_L <= action <= Action.CHI_R_RED:
            if profile: _t0 = _time.time()
            state = self._chi(state, action)
            if profile: _t['chi'] = _time.time() - _t0
        elif action == Action.PASS:
            if profile: _t0 = _time.time()
            state = self._pass(state)
            if profile: _t['pass'] = _time.time() - _t0
        elif action == Action.KYUUSHU:
            if profile: _t0 = _time.time()
            state = self._kyuushu(state)
            if profile: _t['kyuushu'] = _time.time() - _t0
        elif action == Action.DUMMY:
            if profile: _t0 = _time.time()
            state = self._dummy(state)
            if profile: _t['dummy'] = _time.time() - _t0

        state.step_count += 1

        # Single-round termination (JAX: terminated_round & one_round → terminated=TRUE)
        if self.one_round and state.round_state.terminated_round:
            state.terminated = True

        # Auto round transition (multi-round only)
        if self.next_round_style == "auto" and not self.one_round and not is_terminal_before:
            if state.round_state.terminated_round and not state.terminated:
                if profile: _t0 = _time.time()
                state = self._advance_to_next_round_auto(state)
                if profile: _t['advance'] = _time.time() - _t0

        if profile:
            state._profile = _t

        return state

    # ═════════════════════════════════════════════════════════════
    # Action Handlers — serial, one env at a time
    # ═════════════════════════════════════════════════════════════

    def _step_with_illegal_action(self, state, loser):
        """Penalize illegal actions: game ends, loser gets penalty."""
        state.terminated = True
        state.rewards.zero_()
        state.rewards[loser] = self._illegal_action_penalty
        return state

    # ── _draw ──
    # JAX: _draw lines 788-853

    def _draw(self, state):
        """Draw a tile from the wall (matches JAX _draw lines 788-853)."""
        # 1. Accept pending riichi (JAX line 797)
        state = _accept_riichi(state)

        cp = state.current_player

        # 2. Check special abortive draws (JAX lines 800-816)
        first_discards_exist = all(int(state.players.discard_counts[i].item()) > 0 for i in range(4))
        is_four_wind = False
        if first_discards_exist:
            first_tiles = []
            for i in range(4):
                dec = River.decode_tile(state.players.river[i])
                first_tiles.append(int(dec[0].item()))
            is_four_wind = all(Tile.is_tile_four_wind(t) for t in first_tiles) and all(t == first_tiles[0] for t in first_tiles)

        is_pure_first_turn = (int(state.round_state.next_deck_ix) >= FIRST_DRAW_IDX - 5) and \
            (int(state.players.meld_counts.sum().item()) == 0)
        is_four_wind_draw = is_four_wind and is_pure_first_turn
        is_four_riichi_draw = int(state.players.riichi.sum().item()) == 4
        config = self.game_config
        is_special = config.enable_special_abortive_draw and (is_four_wind_draw or is_four_riichi_draw)

        if is_special:
            return _trigger_special_abortive_draw(state)

        # 3. Move deck pointer and draw (JAX lines 817-820)
        is_haitei = int(state.round_state.next_deck_ix) == int(state.round_state.last_deck_ix)
        state.round_state.is_haitei = is_haitei

        ix = int(state.round_state.next_deck_ix)
        new_tile = int(state.round_state.deck[ix].item())
        state.round_state.next_deck_ix = ix - 1
        state.round_state.last_draw = new_tile
        # Note: do NOT set last_player here — JAX _draw does not.
        # last_player is set by the action handler (e.g. _discard) and
        # represents who last took an action others can respond to.

        # 4. Add tile to hand (JAX lines 819-824)
        state.players.hand_with_red[cp] = Hand.add(state.players.hand_with_red[cp], new_tile)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])

        # Copy tsumo yaku precompute → col 0 for mask check (JAX _draw L834-836)
        state.players.has_yaku[cp, 0] = state.players.has_yaku[cp, 1].clone()
        state.players.fan[cp, 0] = state.players.fan[cp, 1].clone()
        state.players.fu[cp, 0] = state.players.fu[cp, 1].clone()

        # 5. Build legal action mask (JAX lines 825-829)
        if bool(state.players.riichi[cp].item()):
            mask = self._make_legal_action_mask_after_draw_riichi(state, cp)
        else:
            mask = self._make_legal_action_mask_after_draw(state)

        state.legal_action_mask = mask
        state.round_state.draw_next = False
        state.round_state.kan_declared = False
        state.round_state.target = -1
        state.round_state.shanten_current_player = Shanten.number(
            Hand.to_34(state.players.hand_with_red[cp]))

        # 6. Clear furiten_by_pass for non-riichi players (JAX lines 842-844)
        if not bool(state.players.riichi[cp].item()):
            state.players.furiten_by_pass[cp] = False

        return state

    # ── Legal action masks ──
    # JAX: _make_legal_action_mask_after_draw lines 855-900

    def _make_legal_action_mask_after_draw_riichi(self, state, cp):
        """Legal actions after draw for a riichi player (JAX _w_riichi variant)."""
        hand = state.players.hand_with_red[cp]
        mask = ZERO_MASK_1D.clone()

        for i in range(Tile.NUM_TILE_TYPE_WITH_RED):
            mask[i] = hand[i] > 0

        ld = int(state.round_state.last_draw)
        if ld >= 0 and ld < Tile.NUM_TILE_TYPE_WITH_RED and hand[ld] >= 2:
            mask[Action.TSUMOGIRI] = True

        for i in range(34):
            if Hand.can_closed_kan(hand, i):
                mask[37 + i] = True

        return mask

    def _make_legal_action_mask_after_draw(self, state):
        """Build legal action masks for current player after drawing a tile (matches JAX)."""
        cp = state.current_player
        hand = state.players.hand_with_red[cp]
        mask = ZERO_MASK_1D.clone()

        # Discard actions
        for i in range(Tile.NUM_TILE_TYPE_WITH_RED):
            if hand[i] > 0:
                mask[i] = True

        # Tsumogiri
        ld = int(state.round_state.last_draw)
        mask[Action.TSUMOGIRI] = (ld >= 0 and ld < Tile.NUM_TILE_TYPE_WITH_RED and hand[ld] > 0)

        # Self kan — blocked by haitei or 4-kan limit
        cannot_kan = state.round_state.is_haitei or (int(state.players.n_kan.sum().item()) >= 4)
        if not cannot_kan:
            for i in range(34):
                if Hand.can_closed_kan(hand, i):
                    mask[37 + i] = True
                n_melds = int(state.players.meld_counts[cp].item())
                for m_idx in range(n_melds):
                    m = int(state.players.melds[cp, m_idx].item())
                    if m != EMPTY_MELD and Meld.is_pon(m) and Meld.target(m) == i:
                        if Hand.can_added_kan(hand, i):
                            mask[37 + i] = True

        # Tsumo
        is_conc = bool(state.players.is_hand_concealed[cp].item())
        can_tsumo = Hand.can_tsumo(hand)
        can_after_kan = state.round_state.can_after_kan
        is_haitei = state.round_state.is_haitei
        has_yaku_tsumo = bool(state.players.has_yaku[cp, 0].item())
        if can_tsumo and (is_conc or can_after_kan or is_haitei or has_yaku_tsumo):
            mask[Action.TSUMO] = True

        # Riichi
        nxt = int(state.round_state.next_deck_ix)
        lst = int(state.round_state.last_deck_ix)
        tiles_left = nxt - lst
        if (not state.players.riichi[cp]
            and state.round_state.score[cp] >= RIICHI_BET // 100
            and state.players.is_hand_concealed[cp]
            and tiles_left >= 4
            and Hand.can_riichi(hand)):
            mask[Action.RIICHI] = True

        # Kyuushu on first turn
        if _is_first_turn(state.round_state.next_deck_ix) and Hand.can_kyuushu(hand):
            mask[Action.KYUUSHU] = True

        return mask

    # ── _discard ──
    # JAX: _discard lines 988-1051

    def _discard(self, state, tile, is_tsumogiri_override=None):
        """Handle a discard action.

        Args:
            is_tsumogiri_override: if not None, override is_tsumogiri flag.
                JAX uses (action == TSUMOGIRI), PT passes it explicitly.
        """
        cp = state.current_player
        hand = state.players.hand_with_red[cp]
        tile = int(tile) if isinstance(tile, (torch.Tensor, np.generic)) else tile
        tt = Tile.to_tile_type(tile)

        # Mark tsumogiri in action history
        ah = state.round_state.action_history
        for col in reversed(range(ah.shape[1])):
            if ah[0, col] != -1:
                ah[2, col] = 1 if tile == state.round_state.last_draw else 0
                break

        # Remove tile from hand
        state.players.hand_with_red[cp] = Hand.sub(hand, tile)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])

        # Add to river
        d_count = int(state.players.discard_counts[cp].item())
        if is_tsumogiri_override is not None:
            is_tsumogiri = is_tsumogiri_override
        else:
            is_tsumogiri = False  # regular discard (JAX: action == TSUMOGIRI)
        is_riichi = bool(state.players.riichi_declared[cp])
        state.players.river = River.add_discard(
            state.players.river, torch.tensor(tile), torch.tensor(cp),
            torch.tensor(d_count), is_tsumogiri, is_riichi)
        state.players.discards[cp, d_count] = tile
        state.players.discard_counts[cp] += 1

        # Clear riichi_declared
        state.players.riichi_declared[cp] = False

        # Furiten
        h_after = state.players.hand_with_red[cp]
        if Hand.is_tenpai(Hand.to_34(h_after)):
            can_ron = torch.tensor([Hand.can_ron(h_after, t) for t in range(34)], dtype=torch.bool)
            if _is_waiting_tile(can_ron, tile):
                state.players.furiten_by_discard[cp] = True
                state.players.furiten_by_pass[cp] = False

        # Clear per-discard flags (JAX _discard L965-966, L990, L1020)
        state.round_state.last_draw = -1
        state.players.ippatsu[cp] = False
        state.round_state.can_after_kan = False

        # Build meld/ron masks for other players
        # JAX sets target = tile (raw action), not to_tile_type(tile)
        state.round_state.target = tile
        state.round_state.last_player = cp

        # JAX: is_haitei = is_haitei | is_abortive_draw_normal
        if state.round_state.next_deck_ix < state.round_state.last_deck_ix:
            state.round_state.is_abortive_draw_normal = True
            state.round_state.is_haitei = True
        elif state.round_state.is_haitei:
            state.round_state.is_abortive_draw_normal = True

        # Precompute yaku for all 4 players (matches JAX
        # yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile).
        # col 0 = RON on discarded tile, col 1 = TSUMO on next deck tile.
        self._precompute_yaku(state, tile)

        state = self._make_legal_action_mask_after_discard(state)
        return state

    # ── _make_legal_action_mask_after_discard ──
    # JAX: lines 1053-1164

    def _make_legal_action_mask_after_discard(self, state):
        """After a discard, set per-player legal action masks for ron/meld (matches JAX)."""
        cp = state.current_player
        discarded_player = cp
        target = state.round_state.target

        haitei = bool(state.round_state.is_haitei) or \
            (int(state.round_state.next_deck_ix) < int(state.round_state.last_deck_ix))

        mask_4p = ZERO_MASK_2D.clone()
        target_tt = Tile.to_tile_type(target)
        target_is_honor = target_tt >= 27

        for p in range(4):
            if p == discarded_player:
                continue
            hand = state.players.hand_with_red[p]
            src = (discarded_player - p) % 4

            is_riichi = bool(state.players.riichi[p].item())
            meld_full = int(state.players.meld_counts[p].item()) >= MAX_MELDS_PER_PLAYER
            cannot_meld = is_riichi or haitei or meld_full
            cannot_kan = int(state.players.n_kan.sum().item()) >= 4

            m = mask_4p[p]

            # Chi: only from left, not honor
            if not cannot_meld and src == 3 and not target_is_honor:
                for chi_a in CHI_ACTIONS:
                    if Hand.can_chi(hand, target, int(chi_a.item())):
                        m[int(chi_a.item())] = True

            # Pon / Open Kan
            if not cannot_meld:
                h34 = Hand.to_34(hand)
                if h34[target_tt] >= 2:
                    if Hand.can_no_red_pon(hand, target): m[Action.PON] = True
                    if Hand.can_red_pon(hand, target): m[Action.PON_RED] = True
                if h34[target_tt] >= 3 and not cannot_kan:
                    if Hand.can_open_kan(hand, target): m[Action.OPEN_KAN] = True

            # Ron
            has_yaku = bool(state.players.has_yaku[p, 0].item())
            if has_yaku or haitei:
                is_furiten = bool(state.players.furiten_by_discard[p] | state.players.furiten_by_pass[p])
                if not is_furiten and Hand.can_ron(hand, target):
                    m[Action.RON] = True

            if m.any():
                m[Action.PASS] = True

            mask_4p[p] = m

        mask_4p[discarded_player] = False

        # Find next player to act
        can_ron_vec = mask_4p[:, Action.RON]
        can_pon_vec = mask_4p[:, Action.PON] | mask_4p[:, Action.PON_RED]
        can_open_kan_vec = mask_4p[:, Action.OPEN_KAN]
        can_chi_vec = torch.tensor([
            mask_4p[i, Action.CHI_L:Action.CHI_R_RED + 1].any().item() for i in range(4)
        ])

        can_any = can_ron_vec | can_pon_vec | can_open_kan_vec | can_chi_vec
        no_meld_player = not can_any.any().item()

        if no_meld_player:
            state.current_player = (discarded_player + 1) % 4
            state.round_state.target = -1
            state.round_state.draw_next = True
            state.round_state.last_player = discarded_player
            if int(state.round_state.next_deck_ix) < int(state.round_state.last_deck_ix):
                state.round_state.is_abortive_draw_normal = True
                state = self._abortive_draw_normal(state)
            else:
                state = self._draw(state)
        else:
            # Priority: RON > OPEN_KAN > PON > CHI (JAX lines 1147-1164)
            priority = torch.where(can_ron_vec, 3,
                        torch.where(can_open_kan_vec, 2,
                        torch.where(can_pon_vec, 1,
                        torch.where(can_chi_vec, 0, -1))))
            next_player = int(torch.argmax(priority).item())

            if can_ron_vec.sum() > 1:
                distances = (torch.arange(4) - discarded_player) % 4
                distances = torch.where(can_ron_vec, distances, torch.tensor(float('inf')))
                next_player = int(torch.argmin(distances).item())

            state.current_player = next_player
            state.legal_action_mask = mask_4p[next_player]
            state.round_state.last_player = discarded_player
            state.round_state.target = target
            state.round_state.draw_next = False

        state.players.legal_action_mask = mask_4p
        return state

    # ── _riichi ──
    # JAX: _riichi lines 1662-1690

    def _riichi(self, state):
        """Riichi declaration step (matches JAX _riichi)."""
        cp = state.current_player
        hand = state.players.hand_with_red[cp]

        discard_ok = torch.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=torch.bool)
        for i in range(Tile.NUM_TILE_TYPE_WITH_RED):
            if hand[i] > 0:
                sub_hand = Hand.sub(hand, i)
                discard_ok[i] = Hand.is_tenpai(Hand.to_34(sub_hand))

        mask = torch.zeros(LEGAL_ACTION_SIZE, dtype=torch.bool)
        mask[:Tile.NUM_TILE_TYPE_WITH_RED] = discard_ok

        ld = int(state.round_state.last_draw)
        if ld >= 0 and ld < Tile.NUM_TILE_TYPE_WITH_RED:
            mask[Action.TSUMOGIRI] = (hand[ld] >= 2) and discard_ok[ld]

        state.legal_action_mask = mask
        state.players.riichi_declared[cp] = True
        state.round_state.draw_next = False
        return state

    # ── _ron ──
    # JAX: _ron lines 1695-1745

    def _ron(self, state):
        """Handle a RON (win by discard) action."""
        cp = state.current_player
        discarded_player = state.round_state.last_player

        hand = state.players.hand_with_red[cp]
        yaku, fan, fu = Yaku.judge(hand, True, cp, state)
        fan = int(fan) if isinstance(fan, torch.Tensor) else fan
        fu = int(fu) if isinstance(fu, torch.Tensor) else fu

        is_ippatsu = bool(state.players.ippatsu[cp].item()) and bool(state.players.riichi[cp].item())
        is_double_riichi = bool(state.players.double_riichi[cp].item())
        can_robbing_kan = bool(state.round_state.kan_declared)
        is_houtei = bool(state.round_state.is_haitei) and not can_robbing_kan
        is_yakuman = (fu == 0)
        if not is_yakuman:
            fan += int(is_ippatsu) + int(is_double_riichi) + int(can_robbing_kan) + int(is_houtei)

        self._settle_ron(state, cp, discarded_player, fan, fu)

        kyotaku_bonus = 10 * int(state.round_state.kyotaku)
        state.rewards[cp] += float(kyotaku_bonus)
        state.round_state.score[cp] += kyotaku_bonus
        state.round_state.kyotaku = 0

        state.players.has_won[cp] = True
        state.round_state.terminated_round = True
        return state

    # ── _tsumo ──
    # JAX: _tsumo lines 1800-1880

    def _tsumo(self, state):
        """Handle a TSUMO (self-draw win) action."""
        cp = state.current_player
        hand = state.players.hand_with_red[cp]
        yaku, fan, fu = Yaku.judge(hand, False, cp, state)
        fan = int(fan) if isinstance(fan, torch.Tensor) else fan
        fu = int(fu) if isinstance(fu, torch.Tensor) else fu

        is_ippatsu = bool(state.players.ippatsu[cp].item()) and bool(state.players.riichi[cp].item())
        is_double_riichi = bool(state.players.double_riichi[cp].item())
        can_after_kan = bool(state.round_state.can_after_kan)
        is_haitei = bool(state.round_state.is_haitei) and not can_after_kan
        is_yakuman = (fu == 0)
        if not is_yakuman:
            fan += int(can_after_kan) + int(is_ippatsu) + int(is_double_riichi) + int(is_haitei)

        self._settle_tsumo(state, cp, fan, fu)

        kyotaku_bonus = 10 * int(state.round_state.kyotaku)
        state.rewards[cp] += float(kyotaku_bonus)
        state.round_state.score[cp] += kyotaku_bonus
        state.round_state.kyotaku = 0

        state.players.has_won[cp] = True
        state.round_state.terminated_round = True
        return state

    # ── Settlement ──
    # JAX: _settle lines 2020-2120

    def _settle_ron(self, state, winner, loser, fan, fu):
        """Settle payments for a ron win."""
        fan = int(fan) if isinstance(fan, torch.Tensor) else fan
        fu = int(fu) if isinstance(fu, torch.Tensor) else fu
        base = Yaku.score(fan, fu)
        is_dealer = winner == int(state.round_state.dealer)
        score = base * 6 if is_dealer else base * 4
        score = math.ceil(score / 100.0)
        honba_points = int(state.round_state.honba) * 3
        total = int(score) + honba_points
        state.round_state.score[winner] += total
        state.round_state.score[loser] -= total
        state.rewards[winner] = float(total)
        state.rewards[loser] = float(-total)

    def _settle_tsumo(self, state, winner, fan, fu):
        """Settle payments for a tsumo win."""
        fan = int(fan) if isinstance(fan, (torch.Tensor, np.generic)) else fan
        fu = int(fu) if isinstance(fu, (torch.Tensor, np.generic)) else fu
        base = Yaku.score(fan, fu)
        is_dealer = winner == int(state.round_state.dealer)
        honba = int(state.round_state.honba)

        if is_dealer:
            payment = int(math.ceil(base * 2 / 100.0)) + honba
            for p in range(4):
                if p != winner:
                    state.round_state.score[p] -= payment
                    state.round_state.score[winner] += payment
                    state.rewards[p] -= float(payment)
                    state.rewards[winner] += float(payment)
        else:
            non_dealer_pay = int(math.ceil(base / 100.0)) + honba
            dealer_pay = int(math.ceil(base * 2 / 100.0)) + honba
            for p in range(4):
                if p == winner:
                    continue
                pay = dealer_pay if p == int(state.round_state.dealer) else non_dealer_pay
                state.round_state.score[p] -= pay
                state.round_state.score[winner] += pay
                state.rewards[p] -= float(pay)
                state.rewards[winner] += float(pay)

    # ── Melds ──
    # JAX: _pon lines 1240-1290, _chi lines 1295-1345, _kan lines 1350-1500

    def _pon(self, state, action):
        """Handle a PON action."""
        cp = state.current_player
        target = state.round_state.target
        hand = state.players.hand_with_red[cp]
        discarder = state.round_state.last_player

        if int(state.players.meld_counts[cp].item()) >= MAX_MELDS_PER_PLAYER:
            return state

        rel_src = (discarder - cp) % 4
        meld = Meld.init(action, target, rel_src)
        _append_meld_to_player(state, meld, cp, int(state.players.discard_counts[discarder].item()) - 1, discarder)

        state.players.hand_with_red[cp] = Hand.pon(hand, target, action)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.is_hand_concealed[cp] = False

        state.round_state.target = -1     # JAX _pon L1526
        state.round_state.draw_next = False
        state.legal_action_mask = self._make_legal_action_mask_after_draw(state)
        state.current_player = cp
        return state

    def _open_kan(self, state):
        """Handle an OPEN KAN action (matches JAX _kan → open_kan branch)."""
        cp = state.current_player
        target = state.round_state.target
        discarded_player = state.round_state.last_player
        hand = state.players.hand_with_red[cp]
        src = (discarded_player - cp) % 4

        if int(state.players.meld_counts[cp].item()) >= MAX_MELDS_PER_PLAYER:
            return state

        meld = Meld.init(Action.OPEN_KAN, target, src)
        d_idx = int(state.players.discard_counts[discarded_player].item()) - 1
        _append_meld_to_player(state, meld, cp, d_idx, discarded_player)

        state.players.hand_with_red[cp] = Hand.open_kan(hand, target)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.is_hand_concealed[cp] = False
        # n_kan is now incremented in _draw_after_kan (JAX: after reading rinshan tile)

        state.round_state.target = -1  # JAX _open_kan L1490
        state = self._flip_dora(state)
        state = self._draw_after_kan(state)
        return state

    def _selfkan(self, state, action, is_added):
        """Handle a closed or added kan.

        Melds from own hand — no discarder's river to mark.

        JAX: added_kan REPLACES the existing PON meld (meld_counts unchanged).
             closed_kan APPENDS a new meld (meld_counts incremented).
        """
        cp = state.current_player
        hand = state.players.hand_with_red[cp]
        tile_type = action - 37

        if is_added:
            # Find the existing PON meld slot and its src (JAX _added_kan uses pon field)
            pon_idx = -1
            pon_src = 1  # default
            for m_idx in range(int(state.players.meld_counts[cp].item())):
                m = int(state.players.melds[cp, m_idx].item())
                if m != EMPTY_MELD and Meld.is_pon(m) and Meld.target(m) == tile_type:
                    pon_idx = m_idx
                    pon_src = Meld.src(m) if hasattr(Meld, 'src') else 1
                    break
            meld = Meld.init(action, tile_type, pon_src)  # use original PON's src
            state.players.hand_with_red[cp] = Hand.added_kan(hand, tile_type)
            if pon_idx >= 0:
                state.players.melds[cp, pon_idx] = meld  # replace, don't increment count
            else:
                n = int(state.players.meld_counts[cp].item())
                state.players.melds[cp, n] = meld
                state.players.meld_counts[cp] += 1
        else:
            meld = Meld.init(action, tile_type, 0)
            state.players.hand_with_red[cp] = Hand.closed_kan(hand, tile_type)
            # Closed kan: append new meld
            n = int(state.players.meld_counts[cp].item())
            state.players.melds[cp, n] = meld
            state.players.meld_counts[cp] += 1
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        # Note: n_kan is NOT incremented here — JAX _draw_after_kan increments it
        # AFTER reading the rinshan tile from deck[10 + n_kan]. Incrementing here
        # would shift the rinshan index by 1, causing a different tile to be drawn.

        # JAX _kan does NOT set last_player — it stays at the previous value (discarder)
        state.round_state.target = -1  # JAX _draw_after_kan non-robbing branch L1404

        state = self._flip_dora(state)
        state = self._draw_after_kan(state)
        return state

    def _chi(self, state, action):
        """Handle a CHI action."""
        cp = state.current_player
        target = state.round_state.target
        hand = state.players.hand_with_red[cp]
        discarder = state.round_state.last_player

        if int(state.players.meld_counts[cp].item()) >= MAX_MELDS_PER_PLAYER:
            return state

        rel_src = (discarder - cp) % 4
        meld = Meld.init(action, target, rel_src)
        _append_meld_to_player(state, meld, cp, int(state.players.discard_counts[discarder].item()) - 1, discarder)

        state.players.hand_with_red[cp] = Hand.chi(hand, target, action)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.is_hand_concealed[cp] = False

        state.round_state.target = -1  # JAX _chi L1565
        state.round_state.draw_next = False
        state.legal_action_mask = self._make_legal_action_mask_after_draw(state)
        state.current_player = cp
        return state

    # ── _pass / _kyuushu / _dummy ──
    # JAX: _pass lines 1500-1550

    def _pass(self, state):
        """PASS action: move to next player who can act, or draw if nobody can."""
        cp = state.current_player

        if state.players.legal_action_mask[cp, Action.RON]:
            state.players.furiten_by_pass[cp] = True

        mask_4p = state.players.legal_action_mask
        found = False
        for offset in range(1, 4):
            p = (cp + offset) % 4
            if mask_4p[p].any():
                state.current_player = p
                state.legal_action_mask = mask_4p[p]
                found = True
                break

        if not found:
            state.current_player = (state.round_state.last_player + 1) % 4
            if state.round_state.is_abortive_draw_normal:
                state = self._abortive_draw_normal(state)
            else:
                state = self._draw(state)

        return state

    def _kyuushu(self, state):
        """Nine-terminal abortive draw."""
        state.round_state.terminated_round = True
        state.round_state.draw_next = False
        state.round_state.is_abortive_draw_normal = True
        state.rewards.zero_()
        if self.one_round:
            self._finalize_game(state)
        return state

    def _dummy(self, state):
        """Dummy step for round transition (dummy_share mode)."""
        state.round_state.dummy_count += 1
        if state.round_state.dummy_count >= 4:
            state = self._advance_to_next_round_auto(state)
        return state

    # ═════════════════════════════════════════════════════════════
    # Round management
    # ═════════════════════════════════════════════════════════════

    def _flip_dora(self, state):
        """Flip the next dora indicator after a kan (matches JAX deck indexing)."""
        n_dora = int(state.round_state.n_kan_doras)
        dora_idx = 9 - 2 * n_dora
        ura_idx = 8 - 2 * n_dora
        if n_dora < MAX_DORA_INDICATORS and dora_idx >= 0:
            state.round_state.dora_indicators[n_dora] = int(state.round_state.deck[dora_idx].item())
            state.round_state.ura_dora_indicators[n_dora] = int(state.round_state.deck[ura_idx].item())
            state.round_state.n_kan_doras += 1
        return state

    def _draw_after_kan(self, state):
        """Draw replacement tile after a kan (rinshan draw)."""
        cp = state.current_player
        n_kan = int(state.players.n_kan.sum().item())
        ix = 10 + n_kan
        tile = int(state.round_state.deck[ix].item())
        state.players.n_kan[cp] += 1  # JAX _draw_after_kan increments n_kan AFTER reading rinshan
        state.round_state.last_draw = tile
        state.round_state.kan_declared = True
        state.round_state.can_after_kan = True
        state.round_state.can_robbing_kan = True

        state.players.hand_with_red[cp] = Hand.add(state.players.hand_with_red[cp], tile)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])

        state.round_state.draw_next = False
        state.legal_action_mask = self._make_legal_action_mask_after_draw(state)
        return state

    def _abortive_draw_normal(self, state):
        """Handle exhaustive draw (ryukyoku)."""
        state.round_state.terminated_round = True
        state.round_state.draw_next = False  # JAX L1935
        tenpai_players = []
        noten_players = []
        for p in range(4):
            h = state.players.hand_with_red[p]
            if Hand.is_tenpai(Hand.to_34(h)):
                tenpai_players.append(p)
            else:
                noten_players.append(p)

        # JAX formula: total=30 (3000/100), tenpai gets 30//n_tenpai, noten pays 30//n_noten
        n_tenpai = len(tenpai_players)
        n_noten = len(noten_players)
        if n_tenpai > 0 and n_tenpai < 4:
            total_reward = 30
            for p in tenpai_players:
                state.round_state.score[p] += total_reward // n_tenpai
                state.rewards[p] += float(total_reward // n_tenpai)
            for p in noten_players:
                state.round_state.score[p] -= total_reward // n_noten
                state.rewards[p] -= float(total_reward // n_noten)

        return state

    # ── _advance_to_next_round_auto ──
    # JAX: lines 2155-2230

    def _advance_to_next_round_auto(self, state):
        """Advance to the next round (auto mode). Matches JAX _advance_to_next_round_auto."""
        hora = state.players.has_won
        is_tenpai = state.players.can_win.any(dim=-1)
        dealer = int(state.round_state.dealer)

        scores = state.round_state.score.clone().float()
        order = torch.argsort(-scores)
        rank_points = torch.zeros(4, dtype=torch.int32)
        for rank_idx, seat in enumerate(order):
            rank_points[seat] = state.round_state.order_points[rank_idx]
        score_with_rank = state.round_state.score + rank_points
        top_player = int(torch.argmax(score_with_rank).item())
        final_score = score_with_rank.clone()
        final_score[top_player] += 10 * int(state.round_state.kyotaku)

        has_other_than_dealer_won = bool(hora.any().item()) and not bool(hora[dealer].item())
        is_eight_consecutive = int(state.round_state.honba) >= 8
        will_dealer_continue = (bool(is_tenpai[dealer].item()) and not has_other_than_dealer_won) \
            or bool(hora[dealer].item())
        will_dealer_continue = will_dealer_continue and not is_eight_consecutive

        is_final_round = int(state.round_state.round) == int(state.round_state.round_limit)
        has_dealer_end = not will_dealer_continue
        top_pre_rank = int(torch.argmax(state.round_state.score).item())
        is_dealer_top = top_pre_rank == dealer
        has_minus = bool((state.round_state.score < 0).any().item())
        is_game_end = (is_final_round and has_dealer_end) or has_minus or (is_final_round and is_dealer_top)

        if is_game_end:
            state.round_state.score = final_score
            state = self._finalize_game(state)
            return state

        next_round = int(state.round_state.round) if will_dealer_continue else int(state.round_state.round) + 1
        next_honba = (int(state.round_state.honba) + 1) if (not bool(hora.any().item()) or will_dealer_continue) else 0
        next_dealer = dealer if will_dealer_continue else (dealer + 1) % 4

        # Reset per-player round state
        for p in range(4):
            state.players.has_won[p] = False
            state.players.furiten_by_discard[p] = False
            state.players.furiten_by_pass[p] = False
            state.players.riichi_declared[p] = False
            state.players.riichi[p] = False
            state.players.ippatsu[p] = False
            state.players.double_riichi[p] = False
            state.players.n_kan[p] = 0
            state.players.discard_counts[p] = 0
            state.players.meld_counts[p] = 0
            state.players.melds[p].fill_(EMPTY_MELD)
            state.players.river[p].fill_(EMPTY_RIVER)
            state.players.is_hand_concealed[p] = True
        state.players.has_nagashi_mangan.fill_(True)

        # Reset round-level state
        state.round_state.terminated_round = False
        state.round_state.draw_next = True
        state.round_state.dummy_count = 0
        state.round_state.kan_declared = False
        state.round_state.can_after_kan = False
        state.round_state.can_robbing_kan = False
        state.round_state.is_haitei = False
        state.round_state.is_abortive_draw_normal = False
        state.round_state.target = -1
        state.round_state.last_draw = -1
        state.round_state.last_player = -1
        state.round_state.n_kan_doras = 0

        state.round_state.round = next_round
        state.round_state.honba = next_honba
        state.round_state.dealer = next_dealer
        state.round_state.seat_wind = _calc_wind(next_dealer)

        # New deck
        deck = torch.zeros(136, dtype=torch.int8)
        tile_idx = 0
        for t in range(34):
            for _ in range(4):
                deck[tile_idx] = t
                tile_idx += 1
        perm = torch.randperm(136)
        deck = deck[perm]
        state.round_state.deck = deck
        state.round_state.next_deck_ix = FIRST_DRAW_IDX
        state.round_state.last_deck_ix = DEAD_WALL_TILES
        state.round_state.dora_indicators[0] = int(deck[9].item())
        state.round_state.ura_dora_indicators[0] = int(deck[8].item())
        for i in range(1, 5):
            state.round_state.dora_indicators[i] = -1
            state.round_state.ura_dora_indicators[i] = -1

        # Deal
        state.players.hand_with_red = Hand.make_init_hand(deck)
        for p in range(4):
            state.players.hand[p] = Hand.to_34(state.players.hand_with_red[p])

        state.current_player = next_dealer
        state = self._draw(state)
        return state

    # ── Yaku precompute ──
    # JAX: yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile (L225-252)

    def _precompute_yaku(self, state, discarded_tile):
        """Precompute has_yaku/fan/fu for all 4 players after a discard.

        Matches JAX's yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile.
        Column layout (same as JAX):
          col 0 = RON  on the discarded tile
          col 1 = TSUMO on the next deck tile

        JAX creates ron_state (target=discard) and tsumo_state (last_draw=next_tile)
        before calling Yaku.judge. We simulate this by temporarily mutating the state.
        """
        nxt = int(state.round_state.next_deck_ix)
        # JAX always computes yaku regardless of wall state.
        # When wall is exhausted, deck[nxt] is still valid.
        next_tile = int(state.round_state.deck[nxt]) if nxt >= 0 else 0
        disc_tile = int(discarded_tile) if isinstance(discarded_tile, (torch.Tensor, np.generic)) else discarded_tile

        # Save original state fields that we'll temporarily mutate
        orig_target = state.round_state.target
        orig_last_draw = state.round_state.last_draw

        for p in range(4):
            hand_p = state.players.hand_with_red[p]

            # RON on the discarded tile — col 0
            # JAX: has_yaku = yaku42.any(axis=-1), checks yaku TYPES not fan (fan includes dora)
            state.round_state.target = disc_tile
            try:
                yaku_vec_r, fan_ron, fu_ron = Yaku.judge(hand_p, True, p, state)
                fan_r = int(fan_ron.item()) if isinstance(fan_ron, torch.Tensor) else int(fan_ron)
                fu_r = int(fu_ron.item()) if isinstance(fu_ron, torch.Tensor) else int(fu_ron)
                state.players.has_yaku[p, 0] = bool(yaku_vec_r.any().item())
                state.players.fan[p, 0] = fan_r
                state.players.fu[p, 0] = fu_r
            except (IndexError, Exception):
                pass

            # TSUMO on the next draw tile — col 1
            state.round_state.target = orig_target
            state.round_state.last_draw = next_tile
            try:
                yaku_vec_t, fan_tsumo, fu_tsumo = Yaku.judge(hand_p, False, p, state)
                fan_t = int(fan_tsumo.item()) if isinstance(fan_tsumo, torch.Tensor) else int(fan_tsumo)
                fu_t = int(fu_tsumo.item()) if isinstance(fu_tsumo, torch.Tensor) else int(fu_tsumo)
                state.players.has_yaku[p, 1] = bool(yaku_vec_t.any().item())
                state.players.fan[p, 1] = fan_t
                state.players.fu[p, 1] = fu_t
            except (IndexError, Exception):
                pass

        # Restore original state
        state.round_state.target = orig_target
        state.round_state.last_draw = orig_last_draw

    def _finalize_game(self, state):
        """Mark game as terminated (JAX: just sets terminated=True).

        Score/reward finalization only happens in _advance_to_next_round_auto
        for multi-round mode, not here.
        """
        state.terminated = True
        return state

        return state
