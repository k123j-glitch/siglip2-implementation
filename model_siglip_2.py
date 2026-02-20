# model_siglip_2.py
# Minimal SigLIP v2 style implementation
# Adds captioning + EMA + masked modeling + distillation

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# ------------------------------------------------------------
# Transformer Block (supports causal mode)
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

        self.causal = causal

    def forward(self, x):
        attn_mask = None
        if self.causal:
            n = x.size(1)
            attn_mask = torch.triu(
                torch.ones(n, n, device=x.device), 1
            ).bool()

        h, _ = self.attn(self.norm1(x),
                         self.norm1(x),
                         self.norm1(x),
                         attn_mask=attn_mask)

        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------
# Vision Encoder (with masking)
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

    def forward(self, x, mask_ratio=0.0):
        x = self.patch(x)
        x = x.flatten(2).transpose(1, 2)

        if mask_ratio > 0:
            mask = torch.rand(x.shape[:2], device=x.device) < mask_ratio
            x = x.masked_fill(mask.unsqueeze(-1), 0)

        x = x + self.pos

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)


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

        return self.norm(x[:, 0])


# ------------------------------------------------------------
# Caption Decoder
# ------------------------------------------------------------

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, dim=768, depth=6, heads=8):
        super().__init__()

        self.token = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList()
        self.cross = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                TransformerBlock(dim, heads, causal=True)
            )
            self.cross.append(
                nn.MultiheadAttention(dim, heads, batch_first=True)
            )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, image_tokens):
        x = self.token(input_ids)

        for blk, cross in zip(self.blocks, self.cross):
            x = blk(x)
            h, _ = cross(x, image_tokens, image_tokens)
            x = x + h

        return self.head(self.norm(x))


# ------------------------------------------------------------
# SigLIP v2 Model
# ------------------------------------------------------------

class SigLIP2(nn.Module):
    def __init__(self,
                 vocab_size,
                 dim=768,
                 proj_dim=512):

        super().__init__()

        self.vision = VisionEncoder(dim=dim)
        self.text = TextEncoder(vocab_size, dim=dim)

        self.v_proj = nn.Linear(dim, proj_dim, bias=False)
        self.t_proj = nn.Linear(dim, proj_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))
        self.logit_bias = nn.Parameter(torch.zeros([]))

        self.decoder = CaptionDecoder(vocab_size, dim=dim)

        # EMA teacher
        self.teacher = copy.deepcopy(self.vision)
        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, momentum=0.999):
        for t, s in zip(self.teacher.parameters(),
                        self.vision.parameters()):
            t.data = momentum * t.data + (1 - momentum) * s.data

    def forward(self,
                images,
                text_ids,
                caption_ids=None,
                mask_ratio=0.0):

        image_tokens = self.vision(images, mask_ratio)
        text_features = self.text(text_ids)

        v = F.normalize(
            self.v_proj(image_tokens.mean(1)), dim=-1
        )
        t = F.normalize(
            self.t_proj(text_features), dim=-1
        )

        logits = (v @ t.t()) * self.logit_scale.exp() + self.logit_bias

        outputs = {
            "logits": logits,
            "image_tokens": image_tokens
        }

        if caption_ids is not None:
            cap_logits = self.decoder(
                caption_ids[:, :-1],
                image_tokens
            )
            outputs["caption_logits"] = cap_logits

        return outputs


# ------------------------------------------------------------
# SigLIP v2 Loss
# ------------------------------------------------------------

def siglip2_loss(model,
                 outputs,
                 images,
                 caption_ids=None):

    logits = outputs["logits"]
    n = logits.size(0)

    # --------------------------------
    # Contrastive Sigmoid Loss
    # --------------------------------
    labels = 2 * torch.eye(n, device=logits.device) - 1
    contrastive = -F.logsigmoid(labels * logits).mean()

    total = contrastive

    # --------------------------------
    # Captioning Loss (optional)
    # --------------------------------
    if caption_ids is not None:
        cap_logits = outputs["caption_logits"]
        caption_loss = F.cross_entropy(
            cap_logits.reshape(-1, cap_logits.size(-1)),
            caption_ids[:, 1:].reshape(-1),
            ignore_index=0
        )
        total += caption_loss

    # --------------------------------
    # Vision Distillation (FIXED)
    # --------------------------------
    with torch.no_grad():
        teacher_tokens = model.teacher(images)  # ✅ PASS IMAGES

    distill = F.mse_loss(
        outputs["image_tokens"],
        teacher_tokens
    )

    total += 0.1 * distill

    return total