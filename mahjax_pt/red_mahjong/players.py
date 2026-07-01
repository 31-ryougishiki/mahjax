"""Rule-based and random players for red_mahjong (PyTorch eager mode).

Ported from the JAX original: mahjax/red_mahjong/players.py
"""

import torch
import math
import random as py_random

from .action import Action
from .hand import Hand
from .meld import Meld
from .shanten import Shanten
from .state import State
from .tile import River, Tile
from .yaku import Yaku

# ── Constants ─────────────────────────────────────────────────

PRIORITY_MASK = torch.tensor(
    [5, 4, 3, 2, 1, 2, 3, 4, 5,
     5, 4, 3, 2, 1, 2, 3, 4, 5,
     5, 4, 3, 2, 1, 2, 3, 4, 5,
     6, 7, 7, 7, 6, 6, 6]
)

OUTSIDE_MASK = torch.tensor(
    [1, 0, 0, 0, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 0, 0, 0, 1,
     1, 1, 1, 1, 1, 1, 1]
)

YAKU_TILES = torch.tensor([27, 31, 32, 33])

BASIC_PON_PROB = 0.04
YAKU_PON_PROB = 0.7
YAKU_MELD_PON_PROB = 0.6
HAS_PUNG_PON_PROB = 0.05
BASIC_CHI_PROB = 0.02
YAKU_MELD_CHI_PROB = 0.5
HAS_PUNG_CHI_PROB = 0.05
OPEN_KAN_PROB = 0.05
RIICHI_PROB = 0.9


# ── Helpers ───────────────────────────────────────────────────

def _bernoulli(prob, rng=None):
    """Random boolean with probability `prob` (0..1)."""
    p = float(prob)
    p = max(0.0, min(1.0, p))
    if rng is not None:
        return float(torch.rand(1, generator=rng).item()) < p
    return py_random.random() < p


def _categorical_logits(logits, rng=None):
    """Sample from categorical distribution given (unnormalized) logits."""
    logits = logits.to(torch.float32)
    # Replace -inf with large negative to avoid NaN
    logits = torch.where(torch.isinf(logits), torch.full_like(logits, -1e9), logits)
    probs = torch.softmax(logits, dim=0)
    if rng is not None:
        return int(torch.multinomial(probs, 1, generator=rng).item())
    return int(torch.multinomial(probs, 1).item())


def _rng_split(rng, n=2):
    """Split a torch.Generator into n independent generators (approximate)."""
    if rng is None:
        return [None] * n
    gens = []
    for _ in range(n):
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        gens.append(torch.Generator().manual_seed(seed))
    return gens


# ── Discard Logic ─────────────────────────────────────────────

def _has_river_tile(hand, tile):
    """Return a 34-element vector indicating whether `tile` is in the hand."""
    tile_type = Tile.to_tile_type(tile)
    out = torch.zeros(34, dtype=torch.int32)
    if tile != -1:
        out[tile_type] = int(hand[tile_type].item())
    return out


def _discard_action_from_tile_type(unflatten_hand, tile_type):
    """Map tile_type (0..34) back to the actual tile index for discard (may be red)."""
    t = int(tile_type)
    has_black = unflatten_hand[t] > 0
    if Tile.is_tile_type_five(t) and unflatten_hand[Tile.to_red(t)] > 0:
        if not has_black:
            return Tile.to_red(t)
    return t


def _discard_logic(state, unflatten_hand, flatten_hand):
    """Choose which tile to discard based on shanten-number minimization.

    - Discard the tile that minimizes shanten.
    - Prioritize outside tiles.
    - When tenpai, discard the tile that gives the widest wait.
    - Defense: if another player is in riichi and shanten >= 2, discard genbutsu.
    """
    cp = state.current_player
    n_meld = int(state.players.meld_counts[cp].item())

    # Detailed shanten for each possible discard
    detailed = Shanten.detailed_discard(flatten_hand)  # (34, 3)
    normal_shantens = detailed[:, 0]
    seven_shantens = detailed[:, 1]
    orphan_shantens = detailed[:, 2]

    best_normal = int(normal_shantens.min().item())
    best_7pairs = int(seven_shantens.min().item())
    best_orphan = int(orphan_shantens.min().item())

    # Choose best shanten type
    if best_normal < best_7pairs or best_7pairs <= 3:
        best_shanten = best_normal
        shantens = normal_shantens
    else:
        best_shanten = best_7pairs
        shantens = seven_shantens

    if best_shanten < best_orphan + 2:
        pass  # keep
    else:
        best_shanten = best_orphan
        shantens = orphan_shantens

    if n_meld > 0:
        best_shanten = best_normal
        shantens = normal_shantens

    best_shanten_mask = shantens == best_shanten
    priority = best_shanten_mask.to(torch.int32) * PRIORITY_MASK * (flatten_hand > 0).to(torch.int32)
    best_shanten_action = int(torch.argmax(priority).item())

    # Tenpai waiting logic
    is_tenpai = best_shanten == 0

    can_rons = torch.full((34,), -1, dtype=torch.int32)
    for i in range(34):
        if flatten_hand[i] == 0 or shantens[i] != 0:
            continue
        h = flatten_hand.clone()
        h[i] -= 1
        count = 0
        for t in range(34):
            if Hand.can_ron(h, t):
                count += 1
        can_rons[i] = count

    best_waiting_action = int(torch.argmax(can_rons).item())
    action = best_waiting_action if is_tenpai else best_shanten_action

    # Defense: if someone is in riichi, discard genbutsu (safe tile)
    other_riichi = state.players.riichi.clone()
    other_riichi[cp] = False
    is_other_riichi = bool(other_riichi.any().item())

    if is_other_riichi and best_shanten >= 2:
        riichi_player = int(torch.argmax(other_riichi.to(torch.int32)).item())
        river_tiles = River.decode_tile(state.players.river[riichi_player])

        hand_in_river = torch.zeros(34, dtype=torch.int32)
        for t in river_tiles:
            hand_in_river += _has_river_tile(flatten_hand, t)

        if hand_in_river.sum() > 0:
            # Discard a genbutsu tile (safe against riichi player)
            defense_action = int(torch.argmax(hand_in_river).item())
            action = defense_action

    return _discard_action_from_tile_type(unflatten_hand, action)


# ── Meld Logics ───────────────────────────────────────────────

def _pon_logic(state, unflatten_hand, flatten_hand, rng=None):
    """Decide whether to pon and which type (normal or red)."""
    target_tile = Tile.to_tile_type(state.round_state.target)
    cp = state.current_player

    is_global_yaku = bool((target_tile == YAKU_TILES).any().item())
    is_wind_yaku = target_tile == 27 + int(state.round_state.seat_wind[cp].item())
    is_yaku = is_global_yaku or is_wind_yaku

    # Check if yaku tile is already melded
    melds = state.players.melds[cp]
    n_melds = int(state.players.meld_counts[cp].item())
    is_yaku_meld = False
    for i in range(n_melds):
        m = int(melds[i].item())
        if m == 0xFFFF:
            continue
        mt = Meld.target(m)
        if int(mt) in (27, 31, 32, 33):
            is_yaku_meld = True
        if int(mt) == 27 + int(state.round_state.seat_wind[cp].item()):
            is_yaku_meld = True

    has_pung = flatten_hand[target_tile] >= 3

    basic_prob = float((flatten_hand.to(torch.int32) * (1 - OUTSIDE_MASK)).sum().item()) * BASIC_PON_PROB
    prob = YAKU_PON_PROB if is_yaku else basic_prob
    prob = YAKU_MELD_PON_PROB if is_yaku_meld else prob
    prob = HAS_PUNG_PON_PROB if has_pung else prob

    do_pon = _bernoulli(prob, rng)
    if do_pon:
        # Choose between normal and red pon
        if state.legal_action_mask[Action.PON_RED]:
            return Action.PON_RED
        return Action.PON
    return Action.PASS


def _chi_logic(state, unflatten_hand, flatten_hand, rng=None):
    """Decide whether to chi and which type."""
    target_tile = Tile.to_tile_type(state.round_state.target)
    cp = state.current_player

    melds = state.players.melds[cp]
    n_melds = int(state.players.meld_counts[cp].item())
    is_yaku_meld = False
    for i in range(n_melds):
        m = int(melds[i].item())
        if m == 0xFFFF:
            continue
        mt = Meld.target(m)
        if int(mt) in (27, 31, 32, 33):
            is_yaku_meld = True
        if int(mt) == 27 + int(state.round_state.seat_wind[cp].item()):
            is_yaku_meld = True

    has_pung = flatten_hand[target_tile] >= 3

    basic_prob = float((flatten_hand.to(torch.int32) * (1 - OUTSIDE_MASK)).sum().item()) * BASIC_CHI_PROB
    prob = YAKU_MELD_CHI_PROB if is_yaku_meld else basic_prob
    prob = HAS_PUNG_CHI_PROB if has_pung else prob

    do_chi = _bernoulli(prob, rng)
    if do_chi:
        # Pick a legal chi action randomly
        chi_start = Action.CHI_L
        chi_end = Action.CHI_R_RED + 1
        chi_mask = state.legal_action_mask.clone()
        for i in range(chi_start):
            chi_mask[i] = False
        for i in range(chi_end, Action.NUM_ACTION):
            chi_mask[i] = False
        if chi_mask.any():
            logits = torch.where(chi_mask, torch.tensor(0.0), torch.tensor(float('-inf')))
            return _categorical_logits(logits, rng)
    return Action.PASS


def _open_kan_logic(state, unflatten_hand, flatten_hand, rng=None):
    """Decide whether to open-kan (5% probability)."""
    do_kan = _bernoulli(OPEN_KAN_PROB, rng)
    return Action.OPEN_KAN if do_kan else Action.PASS


def _riichi_logic(state, current_action, rng=None):
    """Decide whether to declare riichi (90% probability)."""
    do_riichi = _bernoulli(RIICHI_PROB, rng)
    return Action.RIICHI if do_riichi else current_action


# ── Main Player Function ──────────────────────────────────────

def rule_based_player(state, rng=None):
    """Rule-based player: minimize shanten, prefer yaku, defend vs riichi.

    Ported from the JAX original. Uses heuristics for discard, meld calls,
    riichi declaration, and win declaration.
    """
    cp = state.current_player
    unflatten_hand = state.players.hand_with_red[cp]
    melds = state.players.melds[cp]
    n_meld = state.players.meld_counts[cp]
    legal_action_mask = state.legal_action_mask
    flatten_hand = Yaku.flatten(unflatten_hand, melds, n_meld)

    # 1. Discard choice: minimize shanten
    discard_action = _discard_logic(state, unflatten_hand, flatten_hand)
    last_draw = int(state.round_state.last_draw) if isinstance(state.round_state.last_draw, (torch.Tensor,)) else state.round_state.last_draw
    if discard_action == last_draw:
        discard_action = Action.TSUMOGIRI

    # 2. If discard_action is not legal, fall back to random
    if not legal_action_mask[discard_action]:
        logits = torch.where(legal_action_mask, torch.tensor(0.0), torch.tensor(float('-inf')))
        action = _categorical_logits(logits, rng)
    else:
        action = discard_action

    # 3. Meld decisions (pon / chi / open-kan override discard)
    is_chi = bool(legal_action_mask[Action.CHI_L:Action.CHI_R_RED + 1].any().item())
    is_pon = bool(legal_action_mask[Action.PON].item()) or bool(legal_action_mask[Action.PON_RED].item())
    is_open_kan = bool(legal_action_mask[Action.OPEN_KAN].item())

    if is_chi:
        action = _chi_logic(state, unflatten_hand, flatten_hand, rng)
    if is_pon:
        action = _pon_logic(state, unflatten_hand, flatten_hand, rng)
    if is_open_kan:
        action = _open_kan_logic(state, unflatten_hand, flatten_hand, rng)

    # 4. Riichi
    if legal_action_mask[Action.RIICHI]:
        action = _riichi_logic(state, action, rng)

    # 5. Override everything: always win if possible
    if legal_action_mask[Action.RIICHI]:
        action = Action.RIICHI
    if legal_action_mask[Action.TSUMO]:
        action = Action.TSUMO
    if legal_action_mask[Action.RON]:
        action = Action.RON

    return int(action)


def random_player(state, rng=None):
    """Return a random legal action."""
    mask = state.legal_action_mask
    logits = torch.where(
        mask,
        torch.tensor(0.0, dtype=torch.float32),
        torch.tensor(float('-inf'), dtype=torch.float32),
    )
    return _categorical_logits(logits, rng)
