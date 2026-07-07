"""ODE sampling with classifier-free guidance, plus checkpoint loading.

Checkpoints can come from a local path or straight from the Hugging Face Hub
(karthick-hug/Get_me_Far_side).
"""

import os

import torch

from config import CKPT_PATH, CONFIG, DEVICE, LATENT
from encoders import decode, encode_clip, encode_vae
from model import build_model


@torch.no_grad()
def ode_sample(net, ref, cvec, steps, guidance):
    """Integrate the velocity field from sigma=1 to 0 with Euler steps and CFG."""
    x = torch.randn(ref.shape[0], 4, LATENT, LATENT, device=DEVICE)
    zr, zc = torch.zeros_like(ref), torch.zeros_like(cvec)
    sig = torch.linspace(1, 0, steps + 1, device=DEVICE)
    for i in range(steps):
        s = sig[i].expand(x.shape[0])
        with torch.autocast("cuda", torch.bfloat16):
            vc = net(torch.cat([x, ref], 1), s, cvec)
            vu = net(torch.cat([x, zr], 1), s, zc)
        x = x + (sig[i + 1] - sig[i]) * (vu + guidance * (vc - vu)).float()
    return x


def load_ema_model(ckpt_path: str = None, from_hub: bool = False):
    """Build the model and load the EMA weights.

    ckpt_path: local checkpoint. Defaults to CKPT_DIR/nvs_scratch_dit.pt.
    from_hub:  download the checkpoint from the HF Hub instead.
    """
    if from_hub:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(repo_id=CONFIG["hf_repo"],
                                    filename=CONFIG["hf_ckpt_name"])
    ckpt_path = ckpt_path or CKPT_PATH
    assert os.path.exists(ckpt_path), f"checkpoint not found: {ckpt_path}"

    sd = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(DEVICE)
    model.load_state_dict({k: v for k, v in sd["ema"].items()})
    model.eval().requires_grad_(False)
    print(f"loaded EMA weights from {ckpt_path} (step {sd.get('step', '?')})")
    return model


@torch.no_grad()
def sample_novel_view(ema_model, cond01, pose, steps=None, guidance=None):
    """Reference image + relative pose -> novel view as an HWC numpy array in [0,1].

    cond01: (3,H,W) or (B,3,H,W) image tensor in [0,1]
    pose:   4-vector [d_theta, sin(d_az), cos(d_az), d_radius]
    """
    steps = steps or CONFIG["sample_steps"]
    guidance = guidance if guidance is not None else CONFIG["guidance"]
    if cond01.dim() == 3:
        cond01 = cond01[None]
    cond01 = cond01.to(DEVICE)
    ref = encode_vae(cond01).float()
    cvec = torch.cat([encode_clip(cond01).float(), pose.view(1, 4).to(DEVICE)], dim=1)
    z = ode_sample(ema_model, ref, cvec, steps, guidance)
    return decode(z)[0].cpu().permute(1, 2, 0).numpy()
