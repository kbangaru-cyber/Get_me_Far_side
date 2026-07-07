"""Speed + fidelity: the from-scratch DiT vs the pretrained Zero123.

Runs both models over the same fixed set of (object, input view, target view)
triples with the same step budget, then reports latency, PSNR, and SSIM, and
saves a 4-panel comparison chart.

Usage:
    python evaluate.py                       # 100 pairs, checkpoint from CKPT_DIR
    python evaluate.py --from-hub            # pull EMA weights from the HF Hub
    python evaluate.py --n 50 --held-out     # sample pairs from the val split
"""

import argparse
import importlib.util
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from kornia.metrics import ssim as k_ssim
from PIL import Image
from tqdm.auto import tqdm

from config import CONFIG, DEVICE, setup
from preencode import load_cache
from sampling import load_ema_model, sample_novel_view
from utils import load_image, psnr, z123_pose_deg

Z123_PIPE_URL = ("https://raw.githubusercontent.com/huggingface/diffusers/"
                 "main/examples/community/pipeline_zero1to3.py")
Z123_REPO = "kxic/zero123-xl"   # or "kxic/zero123-165000"


def load_zero123():
    """Download the community pipeline file and load the original Zero123 weights."""
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    path = "/tmp/pipeline_zero1to3.py"
    open(path, "w").write(requests.get(Z123_PIPE_URL, timeout=60).text)
    spec = importlib.util.spec_from_file_location("pipeline_zero1to3", path)
    z123 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(z123)

    dt16 = torch.float16
    try:
        fe = CLIPImageProcessor.from_pretrained(Z123_REPO, subfolder="feature_extractor")
    except Exception:
        fe = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    pipe = z123.Zero1to3StableDiffusionPipeline(
        vae=AutoencoderKL.from_pretrained(Z123_REPO, subfolder="vae", torch_dtype=dt16),
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            Z123_REPO, subfolder="image_encoder", torch_dtype=dt16),
        unet=UNet2DConditionModel.from_pretrained(Z123_REPO, subfolder="unet", torch_dtype=dt16),
        scheduler=DDIMScheduler.from_pretrained(Z123_REPO, subfolder="scheduler"),
        cc_projection=z123.CCProjection.from_pretrained(
            Z123_REPO, subfolder="cc_projection", torch_dtype=dt16),
        feature_extractor=fe, safety_checker=None, requires_safety_checker=False,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def ssim_np(a, b):
    """HWC [0,1] -> scalar SSIM."""
    ta = torch.from_numpy(a).permute(2, 0, 1)[None].float()
    tb = torch.from_numpy(b).permute(2, 0, 1)[None].float()
    return float(k_ssim(ta, tb, window_size=11).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100, help="number of evaluation pairs")
    ap.add_argument("--from-hub", action="store_true",
                    help="download EMA weights from karthick-hug/Get_me_Far_side")
    ap.add_argument("--ckpt", type=str, default=None, help="local checkpoint path")
    ap.add_argument("--held-out", action="store_true",
                    help="sample pairs from the validation split instead of train")
    args = ap.parse_args()

    setup()
    obj_dirs, LAT, CLIPF, POSE = load_cache()
    n_all = LAT.shape[0]
    n_train = n_all - CONFIG["val_objects"]

    ema_model = load_ema_model(ckpt_path=args.ckpt, from_hub=args.from_hub)
    zero123 = load_zero123()

    steps = CONFIG["sample_steps"]   # same budget for both models
    guidance = CONFIG["guidance"]
    size = CONFIG["image_size"]
    rng = random.Random(0)

    # 1) Fixed evaluation set of (object, input, target) triples.
    lo, hi = (n_train, n_all) if args.held_out else (0, n_train)
    triples = [(rng.randrange(lo, hi), *rng.sample(range(12), 2))
               for _ in range(args.n)]

    @torch.no_grad()
    def run_z123(pil, pose):
        return zero123(input_imgs=pil, prompt_imgs=pil, poses=[pose],
                       height=256, width=256, num_inference_steps=steps,
                       guidance_scale=guidance, output_type="np").images[0]

    def run_mine(cond_t, o, ci, ti):
        return sample_novel_view(ema_model, cond_t, POSE[o, ci, ti],
                                 steps=steps, guidance=guidance)

    # 2) Warmup once each, then evaluate.
    d0 = obj_dirs[triples[0][0]]
    inp0 = load_image(os.path.join(d0, "000.png"), size)
    run_mine(torch.from_numpy(inp0).permute(2, 0, 1).float(), triples[0][0], 0, 6)
    run_z123(Image.fromarray((inp0 * 255).astype(np.uint8)), [0.0, 30.0, 0.0])
    torch.cuda.synchronize()

    res = {"mine": {"t": [], "psnr": [], "ssim": []},
           "z123": {"t": [], "psnr": [], "ssim": []}}

    for o, ci, ti in tqdm(triples, desc="evaluating"):
        d = obj_dirs[o]
        inp = load_image(os.path.join(d, f"{ci:03d}.png"), size)
        gt = load_image(os.path.join(d, f"{ti:03d}.png"), size)
        cond_t = torch.from_numpy(inp).permute(2, 0, 1).float()
        pil = Image.fromarray((inp * 255).astype(np.uint8))

        t0 = time.perf_counter()
        pm = run_mine(cond_t, o, ci, ti)
        torch.cuda.synchronize()
        res["mine"]["t"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pz = run_z123(pil, z123_pose_deg(d, ci, ti))
        torch.cuda.synchronize()
        res["z123"]["t"].append(time.perf_counter() - t0)

        res["mine"]["psnr"].append(psnr(pm, gt))
        res["mine"]["ssim"].append(ssim_np(pm, gt))
        res["z123"]["psnr"].append(psnr(pz, gt))
        res["z123"]["ssim"].append(ssim_np(pz, gt))

    for k in res:
        for m in res[k]:
            res[k][m] = np.array(res[k][m])

    # 3) Summary.
    print(f"\n{'':24s}{'my DiT':>14s}{'Zero123':>14s}")
    print(f"{'latency (ms/img)':24s}"
          f"{res['mine']['t'].mean() * 1e3:>11.0f} ms{res['z123']['t'].mean() * 1e3:>11.0f} ms")
    print(f"{'PSNR (dB)':24s}"
          f"{res['mine']['psnr'].mean():>10.2f} +-{res['mine']['psnr'].std():.1f}"
          f"{res['z123']['psnr'].mean():>9.2f} +-{res['z123']['psnr'].std():.1f}")
    print(f"{'SSIM':24s}{res['mine']['ssim'].mean():>13.3f}{res['z123']['ssim'].mean():>14.3f}")
    print(f"\nmy model is {res['z123']['t'].mean() / res['mine']['t'].mean():.1f}x faster; "
          f"PSNR gap {res['mine']['psnr'].mean() - res['z123']['psnr'].mean():+.2f} dB "
          f"(positive = mine better) over {args.n} pairs @ {steps} steps")

    # 4) Charts.
    c_mine, c_z = "#2a7fba", "#d1495b"
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))

    bp = axs[0].boxplot([res["mine"]["t"] * 1e3, res["z123"]["t"] * 1e3],
                        tick_labels=["my DiT", "Zero123"],
                        patch_artist=True, showfliers=False)
    for p, c in zip(bp["boxes"], [c_mine, c_z]):
        p.set_facecolor(c)
        p.set_alpha(0.6)
    axs[0].set_ylabel("latency (ms / image)")
    axs[0].set_title("speed")

    bins = np.linspace(min(res["mine"]["psnr"].min(), res["z123"]["psnr"].min()),
                       max(res["mine"]["psnr"].max(), res["z123"]["psnr"].max()), 25)
    axs[1].hist(res["mine"]["psnr"], bins, alpha=0.6, color=c_mine, label="my DiT")
    axs[1].hist(res["z123"]["psnr"], bins, alpha=0.6, color=c_z, label="Zero123")
    axs[1].axvline(res["mine"]["psnr"].mean(), color=c_mine, ls="--")
    axs[1].axvline(res["z123"]["psnr"].mean(), color=c_z, ls="--")
    axs[1].set_xlabel("PSNR (dB)")
    axs[1].set_title("fidelity distribution")
    axs[1].legend()

    lo = min(res["mine"]["psnr"].min(), res["z123"]["psnr"].min()) - 1
    hi = max(res["mine"]["psnr"].max(), res["z123"]["psnr"].max()) + 1
    axs[2].scatter(res["z123"]["psnr"], res["mine"]["psnr"], s=14, alpha=0.6, color="#444")
    axs[2].plot([lo, hi], [lo, hi], "k--", lw=1)
    axs[2].set_xlim(lo, hi)
    axs[2].set_ylim(lo, hi)
    axs[2].set_xlabel("Zero123 PSNR (dB)")
    axs[2].set_ylabel("my DiT PSNR (dB)")
    axs[2].set_title("per-sample PSNR (above line = mine wins)")

    for k, c, lbl in [("mine", c_mine, "my DiT"), ("z123", c_z, "Zero123")]:
        axs[3].scatter(res[k]["t"].mean() * 1e3, res[k]["psnr"].mean(),
                       s=120, color=c, zorder=3)
        axs[3].errorbar(res[k]["t"].mean() * 1e3, res[k]["psnr"].mean(),
                        yerr=res[k]["psnr"].std(), color=c, capsize=4)
        axs[3].annotate(lbl, (res[k]["t"].mean() * 1e3, res[k]["psnr"].mean()),
                        textcoords="offset points", xytext=(10, 6))
    axs[3].set_xlabel("mean latency (ms / image)")
    axs[3].set_ylabel("mean PSNR (dB)")
    axs[3].set_title("speed vs fidelity (top-left is best)")

    plt.suptitle(f"my DiT vs Zero123 over {args.n} pairs, {steps} steps, guidance {guidance}")
    plt.tight_layout()
    out = os.path.join(CONFIG["CKPT_DIR"], "eval_vs_zero123.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print("saved:", out)


if __name__ == "__main__":
    main()
