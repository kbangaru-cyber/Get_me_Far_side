"""Download and inspect the Zero123 views_release dataset.

The dataset is 2D only: 12 PNGs + 12 .npy cameras per object, no meshes.
Objects without a complete set of 12 views are filtered out.

Usage:
    python download_data.py --download            # stream the tar and extract a subset
    python download_data.py --inspect             # save inspection figures
    python download_data.py --inspect --obj 11    # inspect a specific object
    python download_data.py --gif                 # save an orbiting views.gif
"""

import argparse
import io
import math
import os
import re
import tarfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from PIL import Image
from tqdm.auto import tqdm

from config import CONFIG, setup
from utils import (cam_center, find_object_dirs, load_image, load_RT,
                   relative_pose, spherical)


def stream_download_subset(url, dest, n_objects):
    """Stream the tar.gz and extract only view files, stopping after n_objects."""
    is_view = re.compile(r"(?:^|/)\d{1,3}\.(png|npy)$")
    seen, n_files = set(), 0
    with requests.get(url, stream=True, headers={"User-Agent": "nvs-dit"}, timeout=60) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            pbar = tqdm(tar, desc="streaming tar")
            for m in pbar:
                if not m.isfile() or not is_view.search(m.name):
                    continue
                d = os.path.dirname(m.name)
                if d not in seen:
                    if len(seen) >= n_objects:
                        break
                    seen.add(d)
                tar.extract(m, dest)
                n_files += 1
                if n_files % 200 == 0:
                    pbar.set_postfix(objects=len(seen), files=n_files)
    print(f"done: {len(seen)} objects, {n_files} files -> {dest}")


def inspect_object(obj_dirs, idx, out_dir):
    """Save a 12-view grid and one (input -> target) pair with its relative pose."""
    d0 = obj_dirs[idx]

    fig, axs = plt.subplots(2, 6, figsize=(14, 5))
    for ax, v in zip(axs.ravel(), range(12)):
        ax.imshow(load_image(os.path.join(d0, f"{v:03d}.png"), 256))
        ax.set_title(f"view {v}")
        ax.axis("off")
    plt.suptitle("12 rendered views of one object")
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"inspect_obj{idx}_views.png")
    fig.savefig(grid_path, dpi=120)
    plt.close(fig)

    ci, ti = 0, 6
    pose = relative_pose(load_RT(os.path.join(d0, f"{ti:03d}.npy")),
                         load_RT(os.path.join(d0, f"{ci:03d}.npy")))
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(load_image(os.path.join(d0, f"{ci:03d}.png"), 256))
    axs[0].set_title("input view")
    axs[0].axis("off")
    axs[1].imshow(load_image(os.path.join(d0, f"{ti:03d}.png"), 256))
    axs[1].set_title("target view")
    axs[1].axis("off")
    plt.tight_layout()
    pair_path = os.path.join(out_dir, f"inspect_obj{idx}_pair.png")
    fig.savefig(pair_path, dpi=120)
    plt.close(fig)

    print("saved:", grid_path)
    print("saved:", pair_path)
    print("relative pose [dtheta, sin daz, cos daz, dr]:",
          [round(float(x), 3) for x in pose])


def save_views_gif(obj_dirs, idx, out_dir):
    """Save an animated gif orbiting one object, annotated with camera angles."""
    d0 = obj_dirs[idx]
    frames = []
    for v in range(12):
        img = load_image(os.path.join(d0, f"{v:03d}.png"), 256)
        th, az, r = spherical(cam_center(load_RT(os.path.join(d0, f"{v:03d}.npy"))))
        fig, ax = plt.subplots(figsize=(3, 3.4))
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"view {v}\ntheta={math.degrees(th):.1f}  az={math.degrees(az):.1f}  r={r:.2f}",
                     fontsize=9)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    gif_path = os.path.join(out_dir, "views.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=500, loop=0)
    print("saved:", gif_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="stream and extract the dataset")
    ap.add_argument("--inspect", action="store_true", help="save inspection figures")
    ap.add_argument("--gif", action="store_true", help="save an orbiting gif of one object")
    ap.add_argument("--obj", type=int, default=11, help="object index to inspect")
    args = ap.parse_args()

    setup()

    if args.download and len(find_object_dirs(CONFIG["DATA_DIR"])) < CONFIG["n_objects"]:
        stream_download_subset(CONFIG["data_url"], CONFIG["DATA_DIR"], CONFIG["n_objects"])

    obj_dirs = find_object_dirs(CONFIG["DATA_DIR"])[:CONFIG["n_objects"]]
    print("usable objects (12 complete views each):", len(obj_dirs))
    assert len(obj_dirs) > CONFIG["val_objects"], \
        "Not enough data. Run with --download or check DATA_DIR."

    if args.inspect:
        inspect_object(obj_dirs, args.obj, CONFIG["CKPT_DIR"])
    if args.gif:
        save_views_gif(obj_dirs, args.obj, CONFIG["CKPT_DIR"])


if __name__ == "__main__":
    main()
