"""Frozen public encoders. No HF token needed.

VAE  : stabilityai/sd-vae-ft-mse   -> 4 x 32 x 32 latent at 256px
CLIP : openai/clip-vit-large-patch14 -> 768-d image embedding
"""

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import CLIPVisionModelWithProjection

from config import DEVICE

SCALE = 0.18215

_vae = None
_clip = None
_clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def load_encoders(device: str = DEVICE):
    """Load once, reuse everywhere."""
    global _vae, _clip, _clip_mean, _clip_std
    if _vae is None:
        _vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
        ).to(device).eval().requires_grad_(False)
        _clip = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=torch.float16
        ).to(device).eval().requires_grad_(False)
        _clip_mean = _clip_mean.to(device)
        _clip_std = _clip_std.to(device)
    return _vae, _clip


@torch.no_grad()
def encode_vae(x01):
    """(B,3,H,W) in [0,1] -> (B,4,32,32) scaled latent."""
    vae, _ = load_encoders()
    z = vae.encode(x01.half() * 2 - 1).latent_dist.mode()
    return z * SCALE


@torch.no_grad()
def encode_clip(x01):
    """(B,3,H,W) in [0,1] -> (B,768)."""
    _, clip = load_encoders()
    x = F.interpolate(x01, size=224, mode="bilinear", align_corners=False)
    return clip(pixel_values=((x - _clip_mean) / _clip_std).half()).image_embeds


@torch.no_grad()
def decode(z):
    """Scaled latent -> image in [0,1], (B,3,H,W)."""
    vae, _ = load_encoders()
    img = vae.decode((z / SCALE).half()).sample
    return (img.float() / 2 + 0.5).clamp(0, 1)
