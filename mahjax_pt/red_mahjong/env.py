# Copyright 2025 The Mahjax Authors.
# PyTorch eager-mode port of the red_mahjong environment.

from typing import Dict, List, Literal, Optional, Tuple
import torch
import random as py_random
import numpy as np

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


def _append_meld_to_player(state, meld_packed, player, discard_idx, src):
    """Record a meld on the player's meld list and update the river."""
    p = int(player)
    n = int(state.players.meld_counts[p].item())
    state.players.melds[p, n] = meld_packed
    state.players.meld_counts[p] += 1
    # Note: meld_tiles, meld_info population is deferred / embedded in step handlers
    # River update
    state.players.river = River.add_meld(
        state.players.river, Meld.action(meld_packed),
        torch.tensor(p), torch.tensor(discard_idx),
        int(src) if src is not None else 0)
    return state


def _accept_riichi(state):
    """Process a riichi declaration: pay stick, toggle flags."""
    cp = state.current_player
    state.players.riichi[cp] = True
    state.players.riichi_declared[cp] = True
    state.players.double_riichi[cp] = state.round_state.next_deck_ix == 83
    state.players.ippatsu[cp] = True
    state.round_state.kyotaku += 1
    state.round_state.score[cp] -= RIICHI_BET // 100
    return state


def _is_waiting_tile(can_ron, tile):
    """Check if a tile is among the waiting tiles (can_ron is 34-bool vector)."""
    tt = Tile.to_tile_type(tile)
    return bool(can_ron[tt].item())


def _calc_wind(east_player):
    """Calculate seat winds from east player index."""
    return (torch.arange(4) - east_player) % 4


def _is_first_turn(next_deck_ix):
    return next_deck_ix == 83


def _append_action_history(state, action):
    """Append one entry to the action history (3,200). Shift left, write at last occupied slot."""
    ah = state.round_state.action_history
    # find first -1 slot
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
# Main environment class
# ═══════════════════════════════════════════════════════════════════

class RedMahjong(Env):
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
        return "red_mahjong"

    @property
    def version(self):
        return "pt-0.1"

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
    def init(self, key=None):
        """Initialize a new game state. If key is a torch.Generator, use it for seeding."""
        if key is None:
            gen = torch.Generator()
        elif isinstance(key, torch.Generator):
            gen = key
        else:
            # Accept an int seed as well
            gen = torch.Generator().manual_seed(int(key) if isinstance(key, (int, float)) else int(key.item()))

        state = default_state()
        gc = self.game_config
        state.round_state.round_limit = self.round_limit

        # Build wall: 4 copies of each tile type
        deck_values = []
        for t in range(34):
            for _ in range(4):
                deck_values.append(t)
        if gc.use_red_fives:
            # Replace one 5m, 5p, 5s with red variants
            pass  # Already handled: the 37-type system has red fives as separate indices

        # Create the deck with proper tile ids
        deck = torch.tensor(deck_values, dtype=torch.int8)
        # Shuffle
        perm = torch.randperm(136, generator=gen)
        deck = deck[perm]

        state.round_state.deck = deck
        state.round_state.next_deck_ix = FIRST_DRAW_IDX  # 83
        state.round_state.last_deck_ix = DEAD_WALL_TILES
        state.round_state.draw_next = True

        # Seat winds
        dealer = 0  # Use a random dealer
        state.round_state.init_wind = _calc_wind(dealer)
        state.round_state.seat_wind = _calc_wind(dealer)
        state.round_state.dealer = dealer
        state.round_state.order_points = torch.tensor(self.order_points, dtype=torch.int32)
        state.round_state.score = torch.full((4,), STARTING_POINTS // 100, dtype=torch.int32)

        # Initial dora indicators (deck indices 9 and 8 counting from the end of the live wall)
        state.round_state.dora_indicators[0] = int(deck[9].item())
        state.round_state.ura_dora_indicators[0] = int(deck[8].item())

        # Deal: last 52 tiles (indices 84..135) form the initial 4x13 hands
        state.players.hand_with_red = Hand.make_init_hand(deck)
        for p in range(4):
            state.players.hand[p] = Hand.to_34(state.players.hand_with_red[p])
            state.players.is_hand_concealed[p] = True

        # First draw for the dealer: draw from next_deck_ix (which starts at 83),
        # then decrement (deck is consumed right-to-left from 83 down to last_deck_ix)
        state.current_player = state.round_state.dealer
        state = self._draw(state)
        state = self._make_legal_action_mask_after_draw(state)

        # Shanten for current player
        h34 = Hand.to_34(state.players.hand_with_red[state.current_player])
        state.round_state.shanten_current_player = Shanten.number(h34)
        return state

    # ═════════════════════════════════════════════════════════════
    # STEP
    # ═════════════════════════════════════════════════════════════

    def step(self, state: EnvState, action, key=None):
        """Execute one step in the environment."""
        gc = self.game_config
        action = int(action) if isinstance(action, (torch.Tensor, np.generic)) else action

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
            # Discard (0..36)
            state = self._discard(state, action)
        elif Action.is_selfkan(action):
            # Closed / Added kan (action 37..70 → tile_type = action - 37)
            tile_type = action - 37
            is_added = Hand.can_added_kan(
                state.players.hand_with_red[state.current_player], tile_type)
            state = self._selfkan(state, action, is_added)
        elif action == Action.TSUMOGIRI:
            state = self._discard(state, state.round_state.last_draw)
        elif action == Action.RIICHI:
            state = self._riichi(state)
        elif action == Action.RON:
            state = self._ron(state)
        elif action == Action.TSUMO:
            state = self._tsumo(state)
        elif action in (Action.PON, Action.PON_RED):
            state = self._pon(state, action)
        elif action == Action.OPEN_KAN:
            state = self._open_kan(state)
        elif Action.CHI_L <= action <= Action.CHI_R_RED:
            state = self._chi(state, action)
        elif action == Action.PASS:
            state = self._pass(state)
        elif action == Action.KYUUSHU:
            state = self._kyuushu(state)
        elif action == Action.DUMMY:
            state = self._dummy(state)

        state.step_count += 1

        # Auto round transition
        if self.next_round_style == "auto" and not self.one_round and not is_terminal_before:
            if state.round_state.terminated_round and not state.terminated:
                state = self._advance_to_next_round_auto(state)

        return state

    def _step_with_illegal_action(self, state, loser):
        """Penalize illegal actions: game ends, loser gets penalty."""
        state.terminated = True
        state.rewards.zero_()
        state.rewards[loser] = self._illegal_action_penalty
        return state

    # ═════════════════════════════════════════════════════════════
    # Action Handlers
    # ═════════════════════════════════════════════════════════════

    def _draw(self, state):
        """Draw a tile from the wall for the current player.

        The deck is consumed right-to-left: next_deck_ix starts at 83 and decrements.
        When it falls below last_deck_ix, the wall is exhausted (haitei).
        """
        cp = state.current_player
        ix = int(state.round_state.next_deck_ix)
        tile = int(state.round_state.deck[ix].item())
        state.round_state.next_deck_ix = ix - 1  # Move down toward the dead wall
        state.round_state.last_draw = tile
        state.round_state.last_player = cp
        state.players.hand_with_red[cp] = Hand.add(state.players.hand_with_red[cp], tile)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        # Haitei: next draw position has entered the dead wall
        if state.round_state.next_deck_ix < state.round_state.last_deck_ix:
            state.round_state.is_haitei = True
        state.round_state.draw_next = False
        state.round_state.kan_declared = False
        return state

    def _make_legal_action_mask_after_draw(self, state):
        """Build legal action masks for current player after drawing a tile."""
        cp = state.current_player
        hand = state.players.hand_with_red[cp]
        mask = ZERO_MASK_1D.clone()

        # Discard actions
        for i in range(Tile.NUM_TILE_TYPE_WITH_RED):
            if hand[i] > 0:
                mask[i] = True

        # Tsumogiri
        ld = state.round_state.last_draw
        if ld >= 0 and ld < Tile.NUM_TILE_TYPE_WITH_RED and hand[ld] > 0:
            mask[Action.TSUMOGIRI] = True

        # Self kan (closed + added) — actions 37..70 (34 slots, one per tile type)
        for i in range(34):
            if Hand.can_closed_kan(hand, i):
                mask[37 + i] = True
            # Added kan: requires an existing pon meld
            n_melds = int(state.players.meld_counts[cp].item())
            for m_idx in range(n_melds):
                m = int(state.players.melds[cp, m_idx].item())
                if m != EMPTY_MELD and Meld.is_pon(m) and Meld.target(m) == i:
                    if Hand.can_added_kan(hand, i):
                        mask[37 + i] = True

        # Tsumo
        if Hand.can_tsumo(hand):
            mask[Action.TSUMO] = True

        # Riichi
        if (not state.players.riichi[cp]
            and state.round_state.score[cp] >= RIICHI_BET // 100
            and state.players.is_hand_concealed[cp]
            and Hand.can_riichi(hand)):
            mask[Action.RIICHI] = True

        # Kyuushu on first turn
        if _is_first_turn(state.round_state.next_deck_ix) and Hand.can_kyuushu(hand):
            mask[Action.KYUUSHU] = True

        state.legal_action_mask = mask
        return state

    def _discard(self, state, tile):
        """Handle a discard action."""
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
        is_tsumogiri = (tile == state.round_state.last_draw)
        is_riichi = bool(state.players.riichi_declared[cp])
        state.players.river = River.add_discard(
            state.players.river, torch.tensor(tile), torch.tensor(cp),
            torch.tensor(d_count), is_tsumogiri, is_riichi)
        state.players.discards[cp, d_count] = tile
        state.players.discard_counts[cp] += 1

        # Clear riichi_declared
        state.players.riichi_declared[cp] = False

        # Furiten: if this tile is among the waiting tiles
        h_after = state.players.hand_with_red[cp]
        if Hand.is_tenpai(Hand.to_34(h_after)):
            can_ron = torch.tensor([Hand.can_ron(h_after, t) for t in range(34)], dtype=torch.bool)
            if _is_waiting_tile(can_ron, tile):
                state.players.furiten_by_discard[cp] = True
                state.players.furiten_by_pass[cp] = False

        # Build meld/ron masks for other players
        state.round_state.target = tt
        state.round_state.last_player = cp

        # Haitei check: if nobody could act on this discard and wall is exhausted
        if state.round_state.is_haitei:
            state.round_state.is_abortive_draw_normal = True

        state = self._make_legal_action_mask_after_discard(state)
        return state

    def _make_legal_action_mask_after_discard(self, state):
        """After a discard, set legal action masks for ron/meld for all players."""
        cp = state.current_player
        discarded_player = cp
        target = state.round_state.target

        # Give the current player a pass-only mask (they'll be overwritten by round logic)
        mask_4p = ZERO_MASK_2D.clone()

        for p in range(4):
            if p == discarded_player:
                continue
            hand = state.players.hand_with_red[p]
            m = mask_4p[p]

            # Ron
            if Hand.can_ron(hand, target):
                m[Action.RON] = True

            # Pon / Open Kan
            if Hand.can_open_kan(hand, target):
                m[Action.OPEN_KAN] = True
            if Hand.can_pon(hand, target):
                if Hand.can_no_red_pon(hand, target):
                    m[Action.PON] = True
                if Hand.can_red_pon(hand, target):
                    m[Action.PON_RED] = True

            # Chi
            for chi_a in CHI_ACTIONS:
                a = int(chi_a.item())
                if Hand.can_chi(hand, target, a):
                    m[a] = True

            # Double ron check
            if state.round_state.terminated_round and m[Action.RON]:
                # A ron has already been declared on this discard
                pass  # Allow double ron if configured

            mask_4p[p] = m

        # Find next player to act (first non-empty mask after discarded player)
        found = False
        for offset in range(1, 4):
            p = (discarded_player + offset) % 4
            if mask_4p[p].any():
                state.current_player = p
                state.legal_action_mask = mask_4p[p]
                found = True
                break

        if not found:
            # Nobody can call; either abortive draw (wall exhausted) or draw next tile
            state.current_player = (discarded_player + 1) % 4
            if state.round_state.is_abortive_draw_normal:
                state = self._abortive_draw_normal(state)
            else:
                state = self._draw(state)
                state = self._make_legal_action_mask_after_draw(state)

        state.players.legal_action_mask = mask_4p
        return state

    def _riichi(self, state):
        """Riichi declaration step."""
        cp = state.current_player
        # The riichi action just sets flags; the next discard will be riichi-stamped
        state.players.riichi[cp] = True
        # Wait - in mahjax, RIICHI action is followed by discard.
        # The original code has riichi as a separate step that modifies
        # the legal mask to force a discard. Let me handle this properly.
        # After calling RIICHI, the player must discard (only discard actions are legal).
        state = _accept_riichi(state)
        # Set legal action mask: only discard actions
        mask = torch.zeros(LEGAL_ACTION_SIZE, dtype=torch.bool)
        hand = state.players.hand_with_red[cp]
        for i in range(Tile.NUM_TILE_TYPE_WITH_RED):
            if hand[i] > 0:
                mask[i] = True
        ld = state.round_state.last_draw
        if ld >= 0 and ld < Tile.NUM_TILE_TYPE_WITH_RED and hand[ld] > 0:
            mask[Action.TSUMOGIRI] = True
        state.legal_action_mask = mask
        return state

    def _ron(self, state):
        """Handle a RON (win by discard) action."""
        cp = state.current_player
        discarded_player = state.round_state.last_player
        target = state.round_state.target

        hand = state.players.hand_with_red[cp]
        yaku, fan, fu = Yaku.judge(hand, True, cp, state)
        state.players.has_yaku[cp, 1] = True  # 1 = ron
        state.players.fan[cp, 1] = fan
        state.players.fu[cp, 1] = fu
        state.players.has_won[cp] = True

        # Calculate score
        self._settle_ron(state, cp, discarded_player, fan, fu)

        state.round_state.terminated_round = True

        # Check if game is over
        if self.one_round:
            self._finalize_game(state)
        elif state.round_state.round + 1 >= self.round_limit:
            self._finalize_game(state)

        return state

    def _tsumo(self, state):
        """Handle a TSUMO (self-draw win) action."""
        cp = state.current_player
        hand = state.players.hand_with_red[cp]
        yaku, fan, fu = Yaku.judge(hand, False, cp, state)
        state.players.has_yaku[cp, 0] = True  # 0 = tsumo
        state.players.fan[cp, 0] = fan
        state.players.fu[cp, 0] = fu
        state.players.has_won[cp] = True

        self._settle_tsumo(state, cp, fan, fu)
        state.round_state.terminated_round = True

        if self.one_round:
            self._finalize_game(state)
        elif state.round_state.round + 1 >= self.round_limit:
            self._finalize_game(state)

        return state

    def _settle_ron(self, state, winner, loser, fan, fu):
        """Settle payments for a ron win."""
        fan = int(fan) if isinstance(fan, (torch.Tensor, np.generic)) else fan
        fu = int(fu) if isinstance(fu, (torch.Tensor, np.generic)) else fu
        base = Yaku.score(fan, fu)

        dealer = state.round_state.dealer
        honba = int(state.round_state.honba)

        payment = base + honba * 300
        if winner == dealer:
            payment = (payment * 3 + 199) // 200 * 200  # round up
        else:
            payment = (base * 2 + 99) // 200 * 200 if winner == dealer else base  # simplified
            payment += honba * 300

        # Simplified: winner gets base, loser pays base
        state.round_state.score[winner] += payment // 100
        state.round_state.score[loser] -= payment // 100
        state.rewards[winner] = payment / 100.0
        state.rewards[loser] = -payment / 100.0

    def _settle_tsumo(self, state, winner, fan, fu):
        """Settle payments for a tsumo win."""
        fan = int(fan) if isinstance(fan, (torch.Tensor, np.generic)) else fan
        fu = int(fu) if isinstance(fu, (torch.Tensor, np.generic)) else fu
        base = Yaku.score(fan, fu)
        dealer = state.round_state.dealer
        honba = int(state.round_state.honba)

        if winner == dealer:
            payment = (base + honba * 100) // 400 * 100
            for p in range(4):
                if p != winner:
                    state.round_state.score[p] -= payment
                    state.round_state.score[winner] += payment
                    state.rewards[p] -= payment / 100.0
                    state.rewards[winner] += payment / 100.0
        else:
            payment = (base + honba * 100) // 400 * 100
            dealer_payment = (2 * base + honba * 100) // 400 * 100
            for p in range(4):
                if p == winner:
                    continue
                elif p == dealer:
                    state.round_state.score[p] -= dealer_payment
                    state.round_state.score[winner] += dealer_payment
                    state.rewards[p] -= dealer_payment / 100.0
                    state.rewards[winner] += dealer_payment / 100.0
                else:
                    state.round_state.score[p] -= payment
                    state.round_state.score[winner] += payment
                    state.rewards[p] -= payment / 100.0
                    state.rewards[winner] += payment / 100.0

    def _pon(self, state, action):
        """Handle a PON action."""
        cp = state.current_player
        target = state.round_state.target
        hand = state.players.hand_with_red[cp]
        src = state.round_state.last_player

        # Pack meld
        meld = Meld.init(action, target, 1)  # src=1 means from right player (relative)
        _append_meld_to_player(state, meld, cp, int(state.players.discard_counts[src].item()) - 1, src)

        # Remove tiles from hand
        state.players.hand_with_red[cp] = Hand.pon(hand, target, action)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.is_hand_concealed[cp] = False

        # Clear abortive draw flags
        state.round_state.draw_next = False
        state = self._make_legal_action_mask_after_draw(state)
        state.current_player = cp
        return state

    def _open_kan(self, state):
        """Handle an OPEN KAN action."""
        cp = state.current_player
        target = state.round_state.target
        hand = state.players.hand_with_red[cp]
        src = state.round_state.last_player

        meld = Meld.init(Action.OPEN_KAN, target, 2)  # src=2 means from across (relative simplified)
        _append_meld_to_player(state, meld, cp, int(state.players.discard_counts[src].item()) - 1, src)

        state.players.hand_with_red[cp] = Hand.open_kan(hand, target)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.is_hand_concealed[cp] = False
        state.players.n_kan[cp] += 1

        # Flip dora
        state = self._flip_dora(state)
        state = self._draw_after_kan(state)
        return state

    def _selfkan(self, state, action, is_added):
        """Handle a closed or added kan."""
        cp = state.current_player
        hand = state.players.hand_with_red[cp]

        # Kan action 37..70 maps to tile_type = action - 37
        tile_type = action - 37

        if is_added:
            meld = Meld.init(action, tile_type, 1)
            _append_meld_to_player(state, meld, cp, 0, 0)
            state.players.hand_with_red[cp] = Hand.added_kan(hand, tile_type)
        else:
            meld = Meld.init(action, tile_type, 0)
            _append_meld_to_player(state, meld, cp, 0, 0)
            state.players.hand_with_red[cp] = Hand.closed_kan(hand, tile_type)

        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.n_kan[cp] += 1

        state = self._flip_dora(state)
        state = self._draw_after_kan(state)
        return state

    def _chi(self, state, action):
        """Handle a CHI action."""
        cp = state.current_player
        target = state.round_state.target
        hand = state.players.hand_with_red[cp]
        src = state.round_state.last_player

        meld = Meld.init(action, target, 1)
        _append_meld_to_player(state, meld, cp, int(state.players.discard_counts[src].item()) - 1, src)

        state.players.hand_with_red[cp] = Hand.chi(hand, target, action)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])
        state.players.is_hand_concealed[cp] = False

        state.round_state.draw_next = False
        state = self._make_legal_action_mask_after_draw(state)
        state.current_player = cp
        return state

    def _pass(self, state):
        """PASS action: move to next player who can act, or draw if nobody can."""
        cp = state.current_player
        target = state.round_state.target

        # Set furiten_by_pass for players who passed on a ron chance
        for p in range(4):
            if p != cp and state.legal_action_mask[p, Action.RON]:
                state.players.furiten_by_pass[p] = True

        # Find next player with a valid meld/ron action
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
            # Nobody can call; next draw
            state.current_player = (state.round_state.last_player + 1) % 4
            # Check if abortive draw
            if state.round_state.is_abortive_draw_normal:
                state = self._abortive_draw_normal(state)
            else:
                state = self._draw(state)
                state = self._make_legal_action_mask_after_draw(state)

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
        """Flip the next dora indicator after a kan."""
        n_dora = state.round_state.n_kan_doras
        dora_idx = int(state.round_state.last_deck_ix) - n_dora - 1
        if dora_idx >= 0 and n_dora < MAX_DORA_INDICATORS:
            state.round_state.dora_indicators[n_dora] = int(state.round_state.deck[dora_idx].item())
            state.round_state.ura_dora_indicators[n_dora] = int(state.round_state.deck[dora_idx - 1].item())
            state.round_state.n_kan_doras += 1
            state.round_state.last_deck_ix += 1
        return state

    def _draw_after_kan(self, state):
        """Draw replacement tile after a kan (rinshan draw)."""
        cp = state.current_player
        # Draw from end of wall
        ix = int(state.round_state.last_deck_ix) - 1
        tile = int(state.round_state.deck[ix].item())
        state.round_state.last_deck_ix -= 1
        state.round_state.last_draw = tile
        state.round_state.last_player = cp
        state.round_state.kan_declared = True
        state.round_state.can_after_kan = True
        state.round_state.can_robbing_kan = True

        state.players.hand_with_red[cp] = Hand.add(state.players.hand_with_red[cp], tile)
        state.players.hand[cp] = Hand.to_34(state.players.hand_with_red[cp])

        state.round_state.draw_next = False
        state = self._make_legal_action_mask_after_draw(state)
        return state

    def _abortive_draw_normal(self, state):
        """Handle exhaustive draw (ryukyoku)."""
        state.round_state.terminated_round = True
        # Tenpai / noten settlement
        tenpai_players = []
        noten_players = []
        for p in range(4):
            h = state.players.hand_with_red[p]
            if Hand.is_tenpai(Hand.to_34(h)):
                tenpai_players.append(p)
            else:
                noten_players.append(p)

        points_per_tenpai = 3000 // len(tenpai_players) if tenpai_players else 0
        for p in tenpai_players:
            total = points_per_tenpai * len(noten_players) // 100
            state.round_state.score[p] += total
            state.rewards[p] += total
        for p in noten_players:
            total = points_per_tenpai // 100
            state.round_state.score[p] -= total
            state.rewards[p] -= total

        if self.one_round:
            self._finalize_game(state)
        return state

    def _advance_to_next_round_auto(self, state):
        """Advance to the next round (auto mode)."""
        # Determine if dealer stays
        dealer_won = bool(state.players.has_won[state.round_state.dealer])
        tenpai = False
        if not state.players.has_won.any():
            tenpai = Hand.is_tenpai(Hand.to_34(
                state.players.hand_with_red[state.round_state.dealer]))

        if dealer_won or tenpai:
            # Dealer stays
            state.round_state.honba += 1
        else:
            # Dealer rotates
            state.round_state.dealer = (state.round_state.dealer + 1) % 4
            state.round_state.round += 1
            state.round_state.honba = 0

        # Check game end
        if state.round_state.round >= self.round_limit:
            self._finalize_game(state)
            return state

        # Reset round state
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

        # Reset per-player round state
        for p in range(4):
            state.players.has_won[p] = False
            state.players.furiten_by_discard[p] = False
            state.players.furiten_by_pass[p] = False
            state.players.riichi[p] = False
            state.players.riichi_declared[p] = False
            state.players.ippatsu[p] = False
            state.players.n_kan[p] = 0
            state.players.discard_counts[p] = 0
            state.players.meld_counts[p] = 0
            state.players.melds[p].fill_(EMPTY_MELD)
            state.players.river[p].fill_(EMPTY_RIVER)
            state.players.is_hand_concealed[p] = True

        state.players.has_nagashi_mangan.fill_(True)

        # Seat winds
        state.round_state.seat_wind = _calc_wind(state.round_state.dealer)

        # Shuffle and deal new round
        gc = self.game_config
        deck_values = []
        for t in range(34):
            for _ in range(4):
                deck_values.append(t)
        deck = torch.tensor(deck_values, dtype=torch.int8)
        perm = torch.randperm(136)
        deck = deck[perm]
        state.round_state.deck = deck
        state.round_state.next_deck_ix = FIRST_DRAW_IDX
        state.round_state.last_deck_ix = DEAD_WALL_TILES

        state.players.hand_with_red = Hand.make_init_hand(deck)
        for p in range(4):
            state.players.hand[p] = Hand.to_34(state.players.hand_with_red[p])

        # Init dora indicator
        state.round_state.dora_indicators.fill_(-1)
        state.round_state.ura_dora_indicators.fill_(-1)
        dora_ix = int(state.round_state.last_deck_ix)
        state.round_state.dora_indicators[0] = int(deck[dora_ix].item())
        state.round_state.ura_dora_indicators[0] = int(deck[dora_ix - 1].item())

        state.current_player = state.round_state.dealer
        state = self._draw(state)
        state = self._make_legal_action_mask_after_draw(state)

        return state

    def _finalize_game(self, state):
        """Apply final placement bonuses and mark game as terminated."""
        state.terminated = True
        scores = state.round_state.score
        for p in range(4):
            scores[p] += state.round_state.order_points[p]

        # Calculate final rewards as score deltas from start
        base = STARTING_POINTS // 100
        for p in range(4):
            delta = int(scores[p].item()) - base + state.round_state.order_points[p].item()
            state.rewards[p] = float(delta)

        return state


def make(env_name="red_mahjong", **kwargs):
    if env_name == "red_mahjong":
        return RedMahjong(**kwargs)
    raise ValueError(f"Unknown env: {env_name}")
