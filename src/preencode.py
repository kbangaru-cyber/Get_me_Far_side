"""Pre-encode the whole dataset once: latents + CLIP + relative poses.

One pass encodes every view with the frozen VAE (-> 4x32x32 latent) and CLIP
(-> 768-d embedding), and precomputes all 12x12 relative poses per object.
Results are saved to disk (~2.5 GB total, 4 files). Training then loads them
into CPU RAM and never touches PNGs or the VAE again. The GPU spends its time
on the DiT, which makes steps roughly 5 to 10x faster.

Skips itself if the cache already exists. One-time cost is about 10 to 20 min
for 20k objects on an H100.

Usage:
    python preencode.py                # build (or verify) the cache
    python preencode.py --no-local     # skip the copy to local NVMe
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from config import CONFIG, DEVICE, LATENT, setup
from encoders import encode_clip, encode_vae, load_encoders
from utils import find_object_dirs, load_image, load_RT, rel_pose_matrix

CD = CONFIG["CACHE_DIR"]
FILES = {k: os.path.join(CD, f"{k}.pt") for k in ("lat", "clip", "pose")}
DIRS_JSON = os.path.join(CD, "dirs.json")


class _CacheDS(Dataset):
    """DataLoader workers do the PIL decoding in parallel."""

    def __init__(self, dirs, size):
        self.dirs, self.size = dirs, size

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        imgs = np.stack([load_image(os.path.join(d, f"{v:03d}.png"), self.size)
                         for v in range(12)])
        cams = np.stack([load_RT(os.path.join(d, f"{v:03d}.npy")) for v in range(12)])
        return torch.from_numpy(imgs).permute(0, 3, 1, 2).float(), torch.from_numpy(cams), i


def cache_exists():
    return all(os.path.exists(p) for p in FILES.values()) and os.path.exists(DIRS_JSON)


def load_cache():
    """Load the pre-encoded tensors into CPU RAM. Returns (obj_dirs, LAT, CLIPF, POSE)."""
    assert cache_exists(), "Cache not found. Run preencode.py first."
    obj_dirs = json.load(open(DIRS_JSON))
    lat = torch.load(FILES["lat"])     # (N,12,4,32,32) fp16, CPU
    clipf = torch.load(FILES["clip"])  # (N,12,768)     fp16, CPU
    pose = torch.load(FILES["pose"])   # (N,12,12,4)    fp32, CPU
    print(f"cache loaded: {lat.shape[0]} objects")
    return obj_dirs, lat, clipf, pose


def build_cache(local_copy=True):
    obj_dirs = find_object_dirs(CONFIG["DATA_DIR"])[:CONFIG["n_objects"]]
    assert obj_dirs, "No data found. Run download_data.py --download first."
    load_encoders()

    # 1) Optionally parallel-copy the dataset to local NVMe.
    #    On Modal, volume latency is the bottleneck, not the GPU.
    if local_copy:
        local_root = "/tmp/views"
        os.makedirs(local_root, exist_ok=True)

        def _copy_one(d):
            dst = os.path.join(local_root, os.path.relpath(d, CONFIG["DATA_DIR"]))
            if not (os.path.isdir(dst) and len(os.listdir(dst)) == 24):
                shutil.copytree(d, dst, dirs_exist_ok=True)
            return dst

        with ThreadPoolExecutor(max_workers=64) as ex:
            local = list(tqdm(ex.map(_copy_one, obj_dirs), total=len(obj_dirs),
                              desc="copy to local NVMe"))
    else:
        local_root = CONFIG["DATA_DIR"]
        local = obj_dirs

    # 2) Encode from local disk. The GPU is now the bottleneck, as it should be.
    n = len(local)
    lat = torch.empty(n, 12, 4, LATENT, LATENT, dtype=torch.float16)
    clipf = torch.empty(n, 12, 768, dtype=torch.float16)
    pose = torch.empty(n, 12, 12, 4, dtype=torch.float32)
    loader = DataLoader(_CacheDS(local, CONFIG["image_size"]),
                        batch_size=8, num_workers=16, prefetch_factor=4)
    for imgs, cams, idx in tqdm(loader, desc="encoding views"):
        b, V = imgs.shape[:2]
        flat = imgs.reshape(b * V, *imgs.shape[2:]).to(DEVICE)
        lat[idx] = encode_vae(flat).reshape(b, V, 4, LATENT, LATENT).cpu()
        clipf[idx] = encode_clip(flat).reshape(b, V, 768).cpu()
        for k in range(b):
            pose[idx[k]] = rel_pose_matrix(cams[k])

    # 3) Save. dirs.json stores DATA_DIR paths so eval and inference still work.
    obj_dirs = [os.path.join(CONFIG["DATA_DIR"], os.path.relpath(d, local_root))
                for d in local]
    torch.save(lat, FILES["lat"])
    torch.save(clipf, FILES["clip"])
    torch.save(pose, FILES["pose"])
    json.dump(obj_dirs, open(DIRS_JSON, "w"))
    os.sync()
    print(f"cache built: {n} objects -> {CD}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-local", action="store_true",
                    help="encode straight from DATA_DIR without a local NVMe copy")
    args = ap.parse_args()

    setup()
    if cache_exists():
        obj_dirs, lat, _, _ = load_cache()
        print(f"cache already present ({lat.shape[0]} objects), nothing to do")
    else:
        build_cache(local_copy=not args.no_local)


if __name__ == "__main__":
    main()
