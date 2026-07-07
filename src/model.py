"""The model: an MM-DiT-style transformer trained from random init.

Every weight starts from scratch. The design mirrors Zero-1-to-3's
conditioning, DiT-style:

- Appearance: the reference-view latent (4 ch) is channel-concatenated with
  the noisy target latent (4 ch). The patch embed takes 8 channels, so the
  network can copy texture directly.
- Semantics + viewpoint: [CLIP(768) || pose(4)] -> MLP -> added to the
  timestep embedding -> drives AdaLN-Zero modulation in every block. This is
  the DiT-native equivalent of Zero123's single cross-attn token.
- Objective: SD3-style rectified flow. x_sigma = (1-sigma)*x1 + sigma*eps,
  regress v = eps - x1, sigma sampled logit-normal. No sigma shift: at a
  32x32 latent, shift 1.0 is the right regime.

About 150M params. The right size to actually train from scratch on 20k objects.
"""

import math

import torch
import torch.nn as nn

from config import CONFIG, LATENT


def timestep_embedding(t, dim=256):
    """t = sigma in [0,1] -> sinusoidal embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000.0 * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(4 * dim, dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)   # AdaLN-Zero
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.adaLN(c)[:, None, :].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1) + b1
        x = x + g1 * self.attn(h, h, h, need_weights=False)[0]
        h = self.norm2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class NVSDiT(nn.Module):
    def __init__(self, dim, depth, heads, patch, latent, in_ch=8, out_ch=4):
        super().__init__()
        self.patch, self.latent, self.out_ch = patch, latent, out_ch
        g = latent // patch
        self.x_embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, g * g, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.c_mlp = nn.Sequential(nn.Linear(768 + 4, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DiTBlock(dim, heads) for _ in range(depth)])
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.proj_out = nn.Linear(dim, patch * patch * out_ch)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, sigma, cond_vec):
        # x (B,8,32,32) = [noisy || ref-latent]
        # sigma (B,)
        # cond_vec (B,772) = [CLIP || pose], zeros when dropped for CFG
        B = x.shape[0]
        c = self.t_mlp(timestep_embedding(sigma)) + self.c_mlp(cond_vec)
        h = self.x_embed(x).flatten(2).transpose(1, 2) + self.pos
        for blk in self.blocks:
            h = blk(h, c)
        sh, sc = self.ada_out(c)[:, None, :].chunk(2, dim=-1)
        h = self.proj_out(self.norm_out(h) * (1 + sc) + sh)
        p, g = self.patch, self.latent // self.patch
        h = h.reshape(B, g, g, p, p, self.out_ch).permute(0, 5, 1, 3, 2, 4)
        return h.reshape(B, self.out_ch, self.latent, self.latent)


def build_model(device):
    model = NVSDiT(CONFIG["dim"], CONFIG["depth"], CONFIG["heads"],
                   CONFIG["patch"], LATENT).to(device)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"trainable params: {n:.1f}M (all from random init)")
    return model
