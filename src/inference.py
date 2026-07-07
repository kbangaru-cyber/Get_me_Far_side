"""Generate a novel view from a single image.

Loads the EMA weights (locally or from the HF Hub), takes a reference image
and a relative camera, and writes the generated view to disk.

The model was trained on centered objects over plain backgrounds. Busy photos
will look much worse.

Usage:
    python inference.py --image chair.png --az 45 --el 10 --out out.png
    python inference.py --image chair.png --az 90 --from-hub
"""

import argparse
import math

import numpy as np
import torch
from PIL import Image

from config import CONFIG, setup
from sampling import load_ema_model, sample_novel_view


def prep_input(pil, size):
    """PIL -> (3,H,W) tensor in [0,1], RGBA composited onto white."""
    img = pil.convert("RGBA").resize((size, size), Image.BICUBIC)
    a = np.asarray(img, np.float32) / 255.0
    rgb = a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4])
    return torch.from_numpy(rgb).permute(2, 0, 1).float()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="input image path")
    ap.add_argument("--az", type=float, default=0.0, help="azimuth delta in degrees")
    ap.add_argument("--el", type=float, default=0.0, help="elevation delta in degrees")
    ap.add_argument("--zoom", type=float, default=0.0, help="zoom (d_radius = -zoom)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    ap.add_argument("--ckpt", type=str, default=None, help="local checkpoint path")
    ap.add_argument("--from-hub", action="store_true",
                    help="download weights from karthick-hug/Get_me_Far_side")
    ap.add_argument("--out", type=str, default="novel_view.png")
    args = ap.parse_args()

    setup(make_dirs=False)
    ema_model = load_ema_model(ckpt_path=args.ckpt, from_hub=args.from_hub)

    cond = prep_input(Image.open(args.image), CONFIG["image_size"])
    pose = torch.tensor([math.radians(args.el),
                         math.sin(math.radians(args.az)),
                         math.cos(math.radians(args.az)),
                         -float(args.zoom)], dtype=torch.float32)

    img = sample_novel_view(ema_model, cond, pose,
                            steps=args.steps, guidance=args.guidance)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
