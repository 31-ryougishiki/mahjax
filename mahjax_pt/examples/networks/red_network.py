"""Actor-Critic Network for red_mahjong (PyTorch port).

Ported from examples/networks/red_network.py (Flax → PyTorch nn.Module).

Observation keys (from _observe_dict):
    hand: (14,) tile indices 0..36, -1=empty
    action_history: (3, 200) [player, action, tsumogiri]
    shanten_count: scalar
    furiten: scalar
    scores: (4,) from current player's perspective
    round: scalar
    honba: scalar
    kyotaku: scalar
    prevalent_wind: scalar
    seat_wind: scalar
    dora_indicators: (5,)

Although _observe_dict may return some scalars as plain Python types,
the network always works with tensors. Callers should batch dims as needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .transformer import TransformerBlock, orthogonal_init_

# ── Constants (matching JAX original) ──────────────────────
HAND_EMB_SIZE = 128
HISTORY_EMB_SIZE = 192
GLOBAL_EMB_SIZE = 64
FINAL_MLP_DIM = 256
TRANFORMER_MLP_DIM = 256
NUM_HAND_LAYER = 2
NUM_HISTORY_LAYER = 2

MAX_SHANTEN = 6.0
SCORE_OFFSET = 250.0
SCORE_SCALE = 1250.0
MAX_ROUND_VALUE = 12.0
MAX_HONBA = 10.0
MAX_KYOTAKU = 10.0
MAX_WIND_VALUE = 3.0

NUM_PLAYERS = 4
MAX_HISTORY_LENGTH = 200
NUM_TILE_TYPE_WITH_RED = 37
NUM_ACTIONS = 87


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Hand embedding
        self.hand_embed = nn.Embedding(NUM_TILE_TYPE_WITH_RED + 1, HAND_EMB_SIZE, padding_idx=0)
        self.hand_transformers = nn.ModuleList([
            TransformerBlock(HAND_EMB_SIZE, num_heads=4, mlp_dim=TRANFORMER_MLP_DIM)
            for _ in range(NUM_HAND_LAYER)
        ])

        # Action history embedding
        self.hist_player_emb = nn.Embedding(NUM_PLAYERS + 1, HISTORY_EMB_SIZE, padding_idx=0)
        self.hist_action_emb = nn.Embedding(NUM_ACTIONS + 1, HISTORY_EMB_SIZE, padding_idx=0)
        self.hist_tsumogiri_emb = nn.Embedding(3, HISTORY_EMB_SIZE, padding_idx=0)  # -1,0,1 → 0,1,2
        self.hist_pos_emb = nn.Embedding(MAX_HISTORY_LENGTH, HISTORY_EMB_SIZE)
        self.history_transformers = nn.ModuleList([
            TransformerBlock(HISTORY_EMB_SIZE, num_heads=4, mlp_dim=TRANFORMER_MLP_DIM)
            for _ in range(NUM_HISTORY_LAYER)
        ])

        # Global feature processing
        self.dora_embed = nn.Embedding(NUM_TILE_TYPE_WITH_RED + 1, HAND_EMB_SIZE, padding_idx=0)
        self.dora_dense = nn.Linear(HAND_EMB_SIZE, GLOBAL_EMB_SIZE)
        # Input: scores(4) + shanten(1) + furiten(1) + round(1) + honba(1) + kyotaku(1) + pwind(1) + swind(1) = 11
        GLOBAL_SCALARS = 11
        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_SCALARS + GLOBAL_EMB_SIZE, GLOBAL_EMB_SIZE),
            nn.ReLU(),
            nn.Linear(GLOBAL_EMB_SIZE, GLOBAL_EMB_SIZE),
        )
        self.apply(orthogonal_init_)

    def _ensure_batch_dim(self, x, base_ndim, feature_dim=1):
        """Return (B, feature_dim) or (B, D) tensor regardless of input shape.

        base_ndim: expected non-batch dims.
        feature_dim: size of the last dimension after batching (default 1 for scalars).
        """
        if not isinstance(x, torch.Tensor):
            t = torch.tensor(x, dtype=torch.float32)
            return t.view(1, feature_dim).expand(-1, feature_dim)

        # Ensure float
        if x.dtype not in (torch.float32, torch.float64):
            x = x.to(torch.float32)

        if x.dim() == 0:
            # scalar tensor → (1, 1)
            return x.view(1, 1)
        elif x.dim() == base_ndim:
            # Missing batch dim: (D,) → (1, D)
            return x.view(1, -1)
        elif x.dim() == base_ndim + 1 and base_ndim == 0:
            # (B,) → (B, 1)
            return x.view(-1, 1)
        else:
            # Already correct: (B, D)
            return x

    def forward(self, obs):
        # ── Hand ──
        hand = obs["hand"]
        if hand.dim() == 1:
            hand = hand.unsqueeze(0)
        hand = hand.to(torch.long).clamp(min=-1) + 1  # -1→0 (pad idx)
        hand_emb = self.hand_embed(hand)  # (B, 14, E)
        hand_mask = (hand > 0).float()
        x_hand = hand_emb * hand_mask.unsqueeze(-1)
        for blk in self.hand_transformers:
            x_hand = blk(x_hand, mask=hand_mask)
        token_count = hand_mask.sum(dim=1, keepdim=True).clamp(min=1)
        hand_feat = (x_hand * hand_mask.unsqueeze(-1)).sum(dim=1) / token_count  # (B, E)

        # ── Action History ──
        ah = obs["action_history"]
        if ah.dim() == 2:
            ah = ah.unsqueeze(0)  # (1, 3, 200)
        players = ah[:, 0, :].long()  # (B, 200)
        actions = ah[:, 1, :].long()
        tsumogiri = ah[:, 2, :].long()
        hist_mask = (actions >= 0).float()

        players_emb = self.hist_player_emb((players + 1).clamp(min=0))
        actions_emb = self.hist_action_emb((actions + 1).clamp(min=0))
        tsumogiri_emb = self.hist_tsumogiri_emb((tsumogiri + 1).clamp(min=0))
        positions = torch.arange(MAX_HISTORY_LENGTH, device=ah.device).unsqueeze(0)
        pos_emb = self.hist_pos_emb(positions)

        x_hist = players_emb + actions_emb + tsumogiri_emb + pos_emb
        x_hist = x_hist * hist_mask.unsqueeze(-1)
        for blk in self.history_transformers:
            x_hist = blk(x_hist, mask=hist_mask)
        hist_token_count = hist_mask.sum(dim=1, keepdim=True).clamp(min=1)
        hist_feat = (x_hist * hist_mask.unsqueeze(-1)).sum(dim=1) / hist_token_count

        # ── Global Scalars ──
        B = hand.shape[0]
        shanten = self._ensure_batch_dim(obs.get("shanten_count", 0), 0) / MAX_SHANTEN
        furiten = self._ensure_batch_dim(obs.get("furiten", False), 0)
        scores = (self._ensure_batch_dim(obs.get("scores", torch.zeros(4)), 1) + SCORE_OFFSET) / SCORE_SCALE
        round_n = self._ensure_batch_dim(obs.get("round", 0), 0) / MAX_ROUND_VALUE
        honba = self._ensure_batch_dim(obs.get("honba", 0), 0) / MAX_HONBA
        kyotaku = self._ensure_batch_dim(obs.get("kyotaku", 0), 0) / MAX_KYOTAKU
        pwind = self._ensure_batch_dim(obs.get("prevalent_wind", 0), 0) / MAX_WIND_VALUE
        swind = self._ensure_batch_dim(obs.get("seat_wind", 0), 0) / MAX_WIND_VALUE

        global_scalar = torch.cat([scores, shanten, furiten, round_n, honba, kyotaku, pwind, swind], dim=-1)

        # ── Dora ──
        dora_indicators = self._ensure_batch_dim(obs.get("dora_indicators", torch.zeros(5)), 1).long()
        dora = (dora_indicators + 1).clamp(min=0)  # -1 → 0
        dora_mask = (dora_indicators >= 0).float()
        dora_emb = self.dora_embed(dora) * dora_mask.unsqueeze(-1)
        dora_n = dora_mask.sum(dim=1, keepdim=True).clamp(min=1)
        dora_summary = dora_emb.sum(dim=1) / dora_n
        dora_feat = self.dora_dense(dora_summary)

        # ── Fuse ──
        global_in = torch.cat([global_scalar, dora_feat], dim=-1)
        global_out = self.global_mlp(global_in)

        return torch.cat([hand_feat, hist_feat, global_out], dim=-1)


class ACNet(nn.Module):
    """Actor-Critic network for red_mahjong.

    Returns (action_logits, value) on __call__.
    Also provides get_action_logits and get_value separately.
    """

    def __init__(self):
        super().__init__()
        self.shared_extractor = FeatureExtractor()
        FEATURE_DIM = HAND_EMB_SIZE + HISTORY_EMB_SIZE + GLOBAL_EMB_SIZE
        self.policy_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, FINAL_MLP_DIM),
            nn.ReLU(),
            nn.Linear(FINAL_MLP_DIM, NUM_ACTIONS),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(FEATURE_DIM, FINAL_MLP_DIM),
            nn.ReLU(),
            nn.Linear(FINAL_MLP_DIM, 1),
        )
        self.apply(orthogonal_init_)
        nn.init.orthogonal_(self.policy_mlp[-1].weight, gain=0.01)
        if self.policy_mlp[-1].bias is not None:
            nn.init.zeros_(self.policy_mlp[-1].bias)

    @staticmethod
    def _remap_legacy_state_dict(state_dict):
        """Remap old two-extractor keys to shared extractor format.

        Old format: policy_extractor.xxx / critic_extractor.xxx
        New format: shared_extractor.xxx
        """
        new_dict = {}
        policy_keys = []
        critic_keys = []
        for k, v in state_dict.items():
            if k.startswith('policy_extractor.'):
                policy_keys.append((k[len('policy_extractor.'):], v))
            elif k.startswith('critic_extractor.'):
                critic_keys.append((k[len('critic_extractor.'):], v))
            else:
                new_dict[k] = v

        if policy_keys:
            # Use policy extractor weights for shared extractor
            for sub_key, v in policy_keys:
                new_dict['shared_extractor.' + sub_key] = v
            # Discard critic extractor weights (now unused)

        return new_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = self._remap_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, obs):
        features = self.shared_extractor(obs)
        return self.policy_mlp(features), self.value_mlp(features).squeeze(-1)

    def get_action_logits(self, obs):
        return self.policy_mlp(self.shared_extractor(obs))

    def get_value(self, obs):
        return self.value_mlp(self.shared_extractor(obs)).squeeze(-1)
