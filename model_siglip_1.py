# model_siglip_1.py
# Minimal SigLIP v1 implementation
# Dual encoder + Sigmoid contrastive loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ------------------------------------------------------------
# Transformer Block
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------
# Vision Encoder (ViT)
# ------------------------------------------------------------

class VisionEncoder(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 dim=768,
                 depth=12,
                 heads=12):

        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch_size, patch_size)
        num_patches = (image_size // patch_size) ** 2

        self.pos = nn.Parameter(torch.randn(1, num_patches, dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x.mean(dim=1)  # global average pooling


# ------------------------------------------------------------
# Text Encoder
# ------------------------------------------------------------

class TextEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 dim=768,
                 depth=12,
                 heads=12,
                 max_len=77):

        super().__init__()
        self.token = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.randn(1, max_len, dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, input_ids):
        x = self.token(input_ids)
        x = x + self.pos[:, :x.size(1)]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


# ------------------------------------------------------------
# SigLIP v1 Model
# ------------------------------------------------------------

class SigLIP1(nn.Module):
    def __init__(self,
                 vocab_size,
                 dim=768,
                 proj_dim=512):

        super().__init__()

        self.vision = VisionEncoder(dim=dim)
        self.text = TextEncoder(vocab_size=vocab_size, dim=dim)

        self.v_proj = nn.Linear(dim, proj_dim, bias=False)
        self.t_proj = nn.Linear(dim, proj_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))
        self.logit_bias = nn.Parameter(torch.zeros([]))

    def forward(self, images, text_ids):
        v = self.vision(images)
        t = self.text(text_ids)

        v = F.normalize(self.v_proj(v), dim=-1)
        t = F.normalize(self.t_proj(t), dim=-1)

        logits = (v @ t.t()) * self.logit_scale.exp() + self.logit_bias
        return logits


# ------------------------------------------------------------
# SigLIP v1 Loss
# ------------------------------------------------------------

def siglip1_loss(logits):
    n = logits.size(0)
    labels = 2 * torch.eye(n, device=logits.device) - 1
    return -F.logsigmoid(labels * logits).mean()
