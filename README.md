# Get me Far side: Novel View Synthesis with a from-scratch DiT

Give the model one image of an object and a relative camera. It generates the object from that new view.

The model is a compact MM-DiT-style transformer trained from random init with rectified flow. No pretrained backbone. No gated repos. No HF token needed. The frozen VAE and CLIP encoders are public.

Pretrained EMA weights: [karthick-hug/Get_me_Far_side](https://huggingface.co/karthick-hug/Get_me_Far_side)

## How it works

- **Appearance.** The reference-view latent (4 ch) is channel-concatenated with the noisy target latent (4 ch). The patch embed takes 8 channels, so the network can copy texture directly.
- **Semantics and viewpoint.** `[CLIP(768) || pose(4)]` goes through an MLP, is added to the timestep embedding, and drives AdaLN-Zero modulation in every block. This is the DiT-native equivalent of Zero123's single cross-attention token.
- **Objective.** SD3-style rectified flow. `x_sigma = (1-sigma) * x1 + sigma * eps`, regress `v = eps - x1`, sigma sampled logit-normal.
- **Size.** About 150M parameters. The right size to actually train from scratch on 20k objects.

The relative camera is encoded as `[d_theta, sin(d_az), cos(d_az), d_radius]`. The sin/cos pair handles azimuth wrap-around.

## Repository layout

```
src/
  config.py         # one CONFIG dict shared by every stage
  utils.py          # camera math, image loading, dataset discovery
  encoders.py       # frozen VAE (sd-vae-ft-mse) and CLIP (ViT-L/14)
  download_data.py  # stream-download the dataset and inspect samples
  preencode.py      # one-time cache: latents + CLIP + all relative poses
  model.py          # the NVSDiT architecture
  sampling.py       # ODE sampling with CFG, checkpoint loading (local or HF Hub)
  train.py          # training loop: warmup, EMA, resume, periodic visual eval
  inference.py      # generate a novel view from one image
  evaluate.py       # speed and fidelity comparison against Zero123
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Edit the paths in `src/config.py` to match your machine. The defaults target Modal Volumes (`/mnt/zero123-data`, `/mnt/checkpoint`). Sized for one H100/A100 80GB. Smaller GPUs work at a lower `batch_size`.

## Pipeline

Run the stages in order from `src/`.

### 1. Download and inspect the dataset

The dataset is Zero123's `views_release` (about 12 PNGs plus 12 `.npy` cameras per object, 2D only, no meshes). Objects without a complete set of 12 views are filtered out.

```bash
python download_data.py --download            # stream the tar, extract a subset
python download_data.py --inspect --obj 11    # save a 12-view grid + a pose pair
python download_data.py --gif                 # save an orbiting views.gif
```

### 2. Pre-encode the dataset

One pass encodes every view with the frozen VAE (4x32x32 latent) and CLIP (768-d embedding) and precomputes all 12x12 relative poses per object. The cache is about 2.5 GB in 4 files. Training then never touches PNGs or the VAE, so the GPU spends its time on the DiT. Steps get roughly 5 to 10x faster.

```bash
python preencode.py
```

One-time cost is about 10 to 20 minutes for 20k objects on an H100. The script skips itself if the cache already exists.

### 3. Train

```bash
python train.py                # resumes automatically if a checkpoint exists
python train.py --fresh       # wipe the checkpoint and start over
```

The recipe: effective batch 128 (64 x 2 accumulation), 1k-step LR warmup, gradient clipping, EMA weights, 10% conditioning dropout for classifier-free guidance. The checkpoint stores model + EMA + optimizer + step, so resume is exact. The last 64 objects are held out as a validation split.

Every 1k steps the script generates 3 held-out views with the EMA weights, saves them next to ground truth, and prints PSNR. Judge training by these grids, not the loss curve. The flow-matching loss flattens early while quality keeps improving for tens of thousands of steps.

Rough timeline from scratch:

| Steps | What to expect |
|---|---|
| 1k to 3k | coarse blobs in roughly the right pose |
| 5k to 10k | recognizable objects that follow the camera |
| 30k+ | decent shapes, still softer than papers |

That remaining gap is the billion-image prior a pretrained backbone brings, not a bug. On 20k objects the model also generalizes poorly to wild photos. Validation objects are the honest test.

### 4. Inference

Use the local checkpoint or pull the released weights from the Hub:

```bash
python inference.py --image chair.png --az 45 --el 10 --out out.png
python inference.py --image chair.png --az 90 --from-hub
```

Inputs should be a centered object on a plain background. That matches the training distribution.

To load the weights in your own code:

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download("karthick-hug/Get_me_Far_side", "nvs_scratch_dit.pt")
ckpt = torch.load(path, map_location="cpu")   # keys: model, ema, opt, step, config
```

Sample from the EMA weights, not the raw model weights.

### 5. Evaluate against Zero123

Runs this DiT and the pretrained Zero123 (`kxic/zero123-xl`) over the same fixed pairs at the same step budget, then reports latency, PSNR, and SSIM and saves a 4-panel chart (latency boxplot, PSNR histogram, per-sample scatter, speed vs fidelity).

```bash
python evaluate.py --n 100
python evaluate.py --n 100 --held-out --from-hub
```

## Design notes

- **Native 256px.** The renders are 256x256. Training at 512 upsampled them and taught the model blur.
- **No sigma shift.** At a 32x32 latent, shift 1.0 is the right regime.
- **Multi-view consistency** is architectural. Each view is generated independently. The EscherNet-style fix is joint multi-view generation with per-view camera encodings, not more training.
- **Scaling knobs, in order of payoff:** more steps, then more objects (`n_objects`, the cache pass extends), then a bigger model (`dim=1024, depth=20` is about 400M), then swap in a pretrained rectified-flow backbone and keep this exact training loop.

## Acknowledgements

- Dataset and the pretrained baseline: [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) (`views_release`, `kxic/zero123-xl`)
- Frozen encoders: `stabilityai/sd-vae-ft-mse`, `openai/clip-vit-large-patch14`
- Architecture and recipe references: DiT (AdaLN-Zero), SD3 (rectified flow, logit-normal sigma)
