"""Shared helpers: camera math, image loading, dataset directory discovery."""

import json
import math
import os
import re

import numpy as np
import torch
from PIL import Image

from config import CONFIG


# ---------------------------------------------------------------- camera math

def spherical(xyz):
    """Cartesian camera center -> (theta, azimuth, radius)."""
    x, y, z = xyz
    return (math.atan2(math.sqrt(x * x + y * y), z),   # theta (polar)
            math.atan2(y, x),                          # azimuth
            math.sqrt(x * x + y * y + z * z))          # radius


def cam_center(RT):
    """RT is a world->cam (3,4) matrix. Returns the camera center in world space."""
    R, T = RT[:3, :3], RT[:3, 3]
    return -R.T @ T


def relative_pose(target_RT, cond_RT):
    """Relative camera between input and target view.

    Encoded as [d_theta, sin(d_az), cos(d_az), d_radius].
    sin/cos handle azimuth wrap-around.
    """
    th_t, az_t, r_t = spherical(cam_center(target_RT))
    th_c, az_c, r_c = spherical(cam_center(cond_RT))
    d_az = (az_t - az_c) % (2 * math.pi)
    return torch.tensor([th_t - th_c, math.sin(d_az), math.cos(d_az), r_t - r_c],
                        dtype=torch.float32)


def rel_pose_matrix(cams):
    """(12,3,4) camera stack -> (12,12,4) pose grid. [i,j] = pose from cond i to target j."""
    th, az, r = np.zeros(12), np.zeros(12), np.zeros(12)
    for v in range(12):
        c = cams[v].numpy() if torch.is_tensor(cams[v]) else cams[v]
        th[v], az[v], r[v] = spherical(cam_center(c))
    dth = th[None, :] - th[:, None]
    daz = (az[None, :] - az[:, None]) % (2 * np.pi)
    dr  = r[None, :] - r[:, None]
    return torch.from_numpy(
        np.stack([dth, np.sin(daz), np.cos(daz), dr], -1).astype(np.float32))


def z123_pose_deg(d, ci, ti):
    """Relative pose in the [elevation_deg, azimuth_deg, d_radius] format Zero123 expects."""
    th_t, az_t, r_t = spherical(cam_center(load_RT(os.path.join(d, f"{ti:03d}.npy"))))
    th_c, az_c, r_c = spherical(cam_center(load_RT(os.path.join(d, f"{ci:03d}.npy"))))
    return [math.degrees(th_t - th_c),
            math.degrees((az_t - az_c) % (2 * math.pi)),
            r_t - r_c]


# ---------------------------------------------------------------- file loading

def load_image(path, size):
    """RGBA -> RGB composited on white, float [0,1], HWC."""
    img = Image.open(path).convert("RGBA").resize((size, size), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return a[..., :3] * a[..., 3:4] + (1.0 - a[..., 3:4])


def load_RT(path):
    return np.load(path).astype(np.float32)[:3, :4]


def find_object_dirs(root):
    """Return directories that hold a complete set of 12 PNGs and 12 .npy cameras."""
    dirs = []
    for dp, _, fns in os.walk(root):
        pngs = {f for f in fns if re.match(r"\d{1,3}\.png$", f)}
        npys = {f for f in fns if re.match(r"\d{1,3}\.npy$", f)}
        if len(pngs) == 12 and len(npys) == 12:
            dirs.append(dp)
    return sorted(dirs)


def get_obj_dirs():
    """Prefer the cached dirs.json (matches the pre-encoded tensors), else scan the disk."""
    dirs_json = os.path.join(CONFIG["CACHE_DIR"], "dirs.json")
    if os.path.exists(dirs_json):
        return json.load(open(dirs_json))
    return find_object_dirs(CONFIG["DATA_DIR"])[:CONFIG["n_objects"]]


# ---------------------------------------------------------------- misc

def psnr(a, b):
    return -10 * math.log10(max(float(np.mean((a - b) ** 2)), 1e-10))
