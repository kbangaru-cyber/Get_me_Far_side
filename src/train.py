"""Train the DiT from random init with rectified flow.

Per step: sample a batch of (object, view i -> view j) pairs straight from
the RAM cache, form the rectified-flow pair, regress the velocity in bf16
autocast. Conditioning (CLIP||pose and the ref latent) is dropped 10% of the
time for CFG. Warmup -> constant LR, gradient clipping, EMA of the weights,
and a rolling checkpoint with model + EMA + optimizer + step so resume is exact.

Every eval_every steps: generate 3 held-out validation views with the EMA
weights, save them against ground truth, and print PSNR. This, not the loss
curve, is the signal that training is working.

Usage:
    python train.py                    # train (resumes if a checkpoint exists)
    python train.py --fresh            # delete the old checkpoint and start over
    python train.py --steps 60000      # override the step budget
"""

import argparse
import copy
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import CKPT_PATH, CONFIG, DEVICE, setup
from encoders import decode, load_encoders
from model import build_model
from preencode import load_cache
from sampling import ode_sample
from utils import load_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true", help="delete checkpoint and restart")
    ap.add_argument("--steps", type=int, default=None, help="override CONFIG['steps']")
    args = ap.parse_args()
    if args.steps:
        CONFIG["steps"] = args.steps

    setup()
    load_encoders()  # frozen VAE needed only for eval decoding

    obj_dirs, LAT, CLIPF, POSE = load_cache()
    n_all = LAT.shape[0]
    n_train = n_all - CONFIG["val_objects"]
    print(f"train objects: {n_train} | val objects: {CONFIG['val_objects']}")

    model = build_model(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.0)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    def lr_at(s):
        return CONFIG["lr"] * min(1.0, (s + 1) / CONFIG["warmup"])

    def sample_batch(B):
        obj = torch.randint(n_train, (B,))
        vi = torch.randint(12, (B,))
        vj = (vi + 1 + torch.randint(11, (B,))) % 12   # guaranteed != vi
        x1 = LAT[obj, vj].to(DEVICE).float()           # target latent
        ref = LAT[obj, vi].to(DEVICE).float()          # reference latent
        cvec = torch.cat([CLIPF[obj, vi].float(), POSE[obj, vi, vj]], dim=1).to(DEVICE)
        return x1, ref, cvec

    # Fixed validation pairs (held-out objects), with ground-truth pixels for PSNR.
    VAL = [(n_train + k, 0, 6) for k in range(3)]
    val_gt = [load_image(os.path.join(obj_dirs[o], f"{vj:03d}.png"), CONFIG["image_size"])
              for o, vi, vj in VAL]

    @torch.no_grad()
    def run_eval(step):
        ema_model.load_state_dict(ema)
        obj = torch.tensor([o for o, _, _ in VAL])
        vi = torch.tensor([a for _, a, _ in VAL])
        vj = torch.tensor([b for _, _, b in VAL])
        ref = LAT[obj, vi].to(DEVICE).float()
        cvec = torch.cat([CLIPF[obj, vi].float(), POSE[obj, vi, vj]], dim=1).to(DEVICE)
        pred = decode(ode_sample(ema_model, ref, cvec, 16, CONFIG["guidance"])
                      ).cpu().permute(0, 2, 3, 1).numpy()
        fig, axs = plt.subplots(3, 3, figsize=(7.5, 7.5))
        psnrs = []
        for k in range(3):
            inp = load_image(os.path.join(obj_dirs[int(obj[k])], f"{int(vi[k]):03d}.png"),
                             CONFIG["image_size"])
            mse = float(np.mean((pred[k] - val_gt[k]) ** 2))
            psnrs.append(-10 * math.log10(max(mse, 1e-10)))
            titles = ["input", f"pred (PSNR {psnrs[-1]:.1f})", "ground truth"]
            for c, (im, ttl) in enumerate(zip([inp, pred[k], val_gt[k]], titles)):
                axs[k, c].imshow(im)
                axs[k, c].set_title(ttl, fontsize=9)
                axs[k, c].axis("off")
        plt.suptitle(f"validation @ step {step}")
        plt.tight_layout()
        path = os.path.join(CONFIG["CKPT_DIR"], f"val_step{step:06d}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"step {step} | val PSNR {np.mean(psnrs):.2f} dB | saved {path}")

    # ---- resume / fresh start ----
    if args.fresh and os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
        print("fresh start -> removed old checkpoint")

    step, losses = 0, []
    if os.path.exists(CKPT_PATH):
        sd = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(sd["model"])
        ema = {k: v.to(DEVICE) for k, v in sd["ema"].items()}
        opt.load_state_dict(sd["opt"])
        step = int(sd["step"])
        print("resumed @ step", step)
    else:
        print("training from random init")

    def save_ckpt(s):
        torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(),
                    "step": s, "config": CONFIG}, CKPT_PATH)
        os.sync()

    # ---- training loop ----
    accum = CONFIG["grad_accum"]
    model.train()
    opt.zero_grad()
    micro = 0
    pbar = tqdm(total=CONFIG["steps"], initial=step, desc="training")
    while step < CONFIG["steps"]:
        x1, ref, cvec = sample_batch(CONFIG["batch_size"])
        B = x1.shape[0]

        keep = (torch.rand(B, device=DEVICE) >= CONFIG["cfg_drop_prob"]).float()
        cvec = cvec * keep[:, None]
        ref = ref * keep[:, None, None, None]

        noise = torch.randn_like(x1)
        sigma = torch.sigmoid(CONFIG["logit_mean"]
                              + CONFIG["logit_std"] * torch.randn(B, device=DEVICE))
        s = sigma.view(B, 1, 1, 1)
        noisy = (1 - s) * x1 + s * noise
        v_target = noise - x1

        with torch.autocast("cuda", torch.bfloat16):
            pred = model(torch.cat([noisy, ref], 1), sigma, cvec)
        loss = F.mse_loss(pred.float(), v_target) / accum
        loss.backward()
        losses.append(loss.item() * accum)
        micro += 1

        if micro % accum == 0:
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                d = CONFIG["ema_decay"]
                for k, v in model.state_dict().items():
                    ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            step += 1
            pbar.update(1)
            if step % CONFIG["log_every"] == 0:
                pbar.set_postfix(loss=f"{np.mean(losses[-accum * 50:]):.4f}",
                                 lr=f"{lr_at(step):.1e}")
            if step % CONFIG["eval_every"] == 0:
                model.eval()
                run_eval(step)
                model.train()
            if step % CONFIG["save_every"] == 0:
                save_ckpt(step)
    pbar.close()
    save_ckpt(step)
    print("saved ->", CKPT_PATH, "@ step", step)

    # ---- loss curve ----
    if losses:
        w = 200
        sm = np.convolve(losses, np.ones(w) / w, mode="valid")
        plt.figure(figsize=(7, 3))
        plt.plot(losses, alpha=0.25)
        plt.plot(np.arange(len(sm)) + w - 1, sm)
        plt.xlabel("micro-step")
        plt.ylabel("FM loss")
        plt.title("training loss (raw + smoothed)")
        plt.tight_layout()
        loss_path = os.path.join(CONFIG["CKPT_DIR"], "loss_curve.png")
        plt.savefig(loss_path, dpi=120)
        plt.close()
        print("saved:", loss_path)
    print("Reminder: FM loss plateaus early. Trust the validation grids, not this curve.")


if __name__ == "__main__":
    main()
