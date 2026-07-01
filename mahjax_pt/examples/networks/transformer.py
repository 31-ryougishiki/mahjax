"""Transformer utilities for MahJax PyTorch port.

Ported from examples/networks/transformer.py (Flax → PyTorch).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init_(module, scale=None):
    """Pytorch-style orthogonal initialization matching Flax orthogonal_init.

    Flax: nn.initializers.orthogonal(scale) — scale is the gain for 2D weights,
    and 1D weights get std=scale (JAX default is scale=sqrt(2) ≈ 1.414).

    PyTorch: nn.init.orthogonal_(w, gain=scale) — gain is the scale factor.
    1D weights: nn.init.normal_(w, std=scale).
    """
    gain = scale if scale is not None else math.sqrt(2.0)
    if not hasattr(module, 'weight'):
        return
    w = module.weight
    if w.ndim >= 2:
        nn.init.orthogonal_(w, gain=gain)
    elif w.ndim == 1:
        nn.init.normal_(w, std=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0.0)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head dot-product self-attention matching Flax's behavior."""

    def __init__(self, features, num_heads):
        super().__init__()
        assert features % num_heads == 0
        self.features = features
        self.num_heads = num_heads
        self.head_dim = features // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(features, features, bias=False)
        self.k_proj = nn.Linear(features, features, bias=False)
        self.v_proj = nn.Linear(features, features, bias=False)
        self.out_proj = nn.Linear(features, features, bias=False)

        self.apply(orthogonal_init_)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn_weights = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, T, T)

        if mask is not None:
            if mask.dim() == 2:
                attn_mask = mask[:, None, None, :]  # (B, 1, 1, T)
            else:
                attn_mask = mask
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))

        # Stable softmax: clamp extreme values to avoid NaN from all -inf rows
        attn_weights = torch.clamp(attn_weights, min=-1e9, max=1e9)
        attn = F.softmax(attn_weights, dim=-1)
        # Replace NaN (all-masked rows) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)
        out = attn @ v  # (B, H, T, D)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block matching the Flax original."""

    def __init__(self, features, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(features)
        self.attn = MultiHeadSelfAttention(features, num_heads)
        self.ln2 = nn.LayerNorm(features)
        self.mlp = nn.Sequential(
            nn.Linear(features, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, features),
        )
        self.apply(orthogonal_init_)

    def forward(self, x, mask=None):
        # Attention sub-layer (Pre-Norm)
        y = self.ln1(x)
        y = self.attn(y, mask=mask)
        x = x + y

        # MLP sub-layer (Pre-Norm)
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x
