import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────
# 1. 2D Rotary Positional Embeddings (Vision)
# ─────────────────────────────────────────────

def build_2d_rope(height, width, head_dim, device, dtype):
    assert head_dim % 4 == 0
    # Each axis (y and x) gets HALF of the head_dim
    half_dim = head_dim // 2

    # freq_seq should be based on half_dim // 2 because we use sin/cos pairs
    freq_seq = torch.arange(0, half_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)

    y_freq = torch.einsum("i,j->ij", y, inv_freq)
    x_freq = torch.einsum("i,j->ij", x, inv_freq)

    # These are (H, half_dim//2) and (W, half_dim//2)
    # We expand them to (H, W, half_dim//2)
    y_freq = y_freq[:, None, :].expand(height, width, -1)
    x_freq = x_freq[None, :, :].expand(height, width, -1)

    # Combine them: (H, W, half_dim)
    combined_freq = torch.cat([y_freq, x_freq], dim=-1)

    # Now create sin and cos: (L, head_dim)
    # We repeat each frequency to match the complex rotation pattern [cos, cos, sin, sin]
    sin = combined_freq.repeat_interleave(2, dim=-1).sin().reshape(-1, head_dim)
    cos = combined_freq.repeat_interleave(2, dim=-1).cos().reshape(-1, head_dim)

    return sin, cos


def apply_rope(q, k, sin, cos):
    # q, k shape: (B, heads, L, head_dim)
    # sin, cos shape: (L, head_dim) -> Reshape to (1, 1, L, head_dim)
    sin = sin[None, None, :, :]
    cos = cos[None, None, :, :]

    def rotate_half(x):
        # Split the last dimension
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    # Standard RoPE formula
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


# ─────────────────────────────────────────────
# 2. Components (Attention & Transformer)
# ─────────────────────────────────────────────

class SigLIPAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin=None, cos=None, causal=False):
        B, L, C = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Inject RoPE (skipping CLS token if present)
        if sin is not None:
            if L == sin.shape[0] + 1:
                q_cls, q_patch = q[:, :, :1], q[:, :, 1:]
                k_cls, k_patch = k[:, :, :1], k[:, :, 1:]
                q_patch, k_patch = apply_rope(q_patch, k_patch, sin, cos)
                q, k = torch.cat([q_cls, q_patch], dim=2), torch.cat([k_cls, k_patch], dim=2)
            else:
                q, k = apply_rope(q, k, sin, cos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if causal:
            mask = torch.tril(torch.ones(L, L, device=x.device)).view(1, 1, L, L)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = attn.softmax(dim=-1)
        x = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, L, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SigLIPAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )
        self.causal = causal

    def forward(self, x, sin=None, cos=None):
        x = x + self.attn(self.norm1(x), sin, cos, self.causal)
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 3. Encoders
# ─────────────────────────────────────────────

class VisionEncoder(nn.Module):
    def __init__(self, patch_size=16, dim=768, depth=12, heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head_dim = dim // heads

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        gh, gw = H // self.patch_size, W // self.patch_size

        sin, cos = build_2d_rope(gh, gw, self.head_dim, x.device, x.dtype)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        for blk in self.blocks:
            x = blk(x, sin, cos)
        return self.norm(x[:, 0])


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=32000, max_len=77, dim=768, depth=12, heads=12):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, causal=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        x = self.token_emb(input_ids) + self.pos_emb[:, :input_ids.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, -1])


# ─────────────────────────────────────────────
# 4. SigLIP2 Wrapper & Loss
# ─────────────────────────────────────────────

class SigLIP2(nn.Module):
    def __init__(self, vision_cfg, text_cfg, proj_dim=512):
        super().__init__()
        self.vision = VisionEncoder(**vision_cfg)
        self.text = TextEncoder(**text_cfg)
        self.v_proj = nn.Linear(vision_cfg["dim"], proj_dim, bias=False)
        self.t_proj = nn.Linear(text_cfg["dim"], proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def forward(self, images, input_ids):
        v = F.normalize(self.v_proj(self.vision(images)), dim=-1)
        t = F.normalize(self.t_proj(self.text(input_ids)), dim=-1)
        logits = (v @ t.t()) * self.logit_scale.exp() + self.logit_bias
        return logits


def siglip2_loss(logits):
    n = logits.size(0)
    labels = 2 * torch.eye(n, device=logits.device) - 1
    return -F.logsigmoid(labels * logits).mean()


def get_siglip2_base():
    v_cfg = {'patch_size': 16, 'dim': 768, 'depth': 12, 'heads': 12}
    t_cfg = {'vocab_size': 32000, 'dim': 512, 'depth': 12, 'heads': 8}
    return SigLIP2(v_cfg, t_cfg)