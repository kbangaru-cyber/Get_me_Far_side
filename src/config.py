"""Central configuration for the NVS from-scratch DiT project.

Every stage (download, pre-encode, train, evaluate) reads from this one dict.
Edit paths here to match your machine or Modal volume layout.
"""

import os
import random

import numpy as np
import torch

CONFIG = dict(
    # ---- data ----
    data_url    = "https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz",
    DATA_DIR    = "/mnt/zero123-data/views",
    CACHE_DIR   = "/mnt/zero123-data/cache256",
    CKPT_DIR    = "/mnt/checkpoint",
    n_objects   = 20001,
    image_size  = 256,          # native resolution of the renders. Do NOT upsample.
    val_objects = 64,           # held out from training
    # ---- model (from scratch) ----
    dim   = 768,
    depth = 14,
    heads = 12,
    patch = 2,
    # ---- training ----
    batch_size    = 64,         # micro-batch; lower on smaller GPUs
    grad_accum    = 2,          # effective batch = 128
    lr            = 1e-4,
    warmup        = 1000,
    steps         = 40000,      # optimizer steps; more = better
    cfg_drop_prob = 0.1,
    ema_decay     = 0.999,
    logit_mean    = 0.0,        # logit-normal sigma sampling
    logit_std     = 1.0,
    log_every     = 50,
    eval_every    = 1000,
    save_every    = 1000,
    # ---- inference ----
    sample_steps = 50,
    guidance     = 3.0,
    seed         = 0,
    # ---- hugging face ----
    hf_repo      = "karthick-hug/Get_me_Far_side",
    hf_ckpt_name = "nvs_scratch_dit.pt",
)

LATENT = CONFIG["image_size"] // 8   # 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_PATH = os.path.join(CONFIG["CKPT_DIR"], CONFIG["hf_ckpt_name"])


def setup(make_dirs: bool = True) -> str:
    """Seed everything and create the working directories. Returns the device."""
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    if make_dirs:
        for k in ("DATA_DIR", "CACHE_DIR", "CKPT_DIR"):
            os.makedirs(CONFIG[k], exist_ok=True)
    return DEVICE
