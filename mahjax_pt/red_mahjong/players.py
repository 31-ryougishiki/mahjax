"""Rule-based and random players for red_mahjong (PyTorch eager mode)."""
import torch

from .action import Action
from .tile import Tile
from .types import Array, PRNGKey


def random_player(state, rng=None):
    """Return a random legal action."""
    mask = state.legal_action_mask
    # Convert mask to logits
    logits = torch.where(mask, torch.zeros_like(mask, dtype=torch.float32),
                         torch.full_like(mask, float('-inf'), dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1)
    if rng is None:
        return torch.multinomial(probs, 1).item()
    return torch.multinomial(probs, 1, generator=rng).item()


def rule_based_player(state, rng=None):
    """Simple rule-based player. Falls back to random if heuristic fails.

    This is a simplified version. The JAX rule_based_player is complex (~370 lines)
    and will be ported in a follow-up.
    """
    # For now: always riichi if possible, tsumo if possible, ron if possible,
    # otherwise discard the tile closest to tsumogiri
    mask = state.legal_action_mask

    if mask[Action.RON]:
        return Action.RON
    if mask[Action.TSUMO]:
        return Action.TSUMO
    if mask[Action.RIICHI]:
        return Action.RIICHI
    if mask[Action.TSUMOGIRI]:
        return Action.TSUMOGIRI
    # Pick any discard
    for i in range(Tile.NUM_TILE_TYPE_WITH_RED):
        if mask[i]:
            return i
    # Fallback to random
    return random_player(state, rng)
