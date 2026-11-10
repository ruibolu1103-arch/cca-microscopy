"""Datasets. Prepared arrays live in <root>/<dataset>/<split>_{streams,label,cls}.npy."""
import hashlib
import os

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

import degrade

SIGMA = 2.0
_R = int(3 * SIGMA)
_YY, _XX = np.mgrid[-_R:_R + 1, -_R:_R + 1]
_STAMP = np.exp(-(_YY ** 2 + _XX ** 2) / (2 * SIGMA ** 2)).astype(np.float32)  # peak value is exactly 1


def relabel(label):
    ids = np.unique(label)
    lut = np.zeros(int(label.max()) + 1, np.int32)
    lut[ids[ids > 0]] = np.arange(1, (ids > 0).sum() + 1)
    return lut[label]


def make_targets(label, inst_cls, n_classes):
    """Centre heatmaps [K,H,W], foreground [1,H,W], per-instance normalised inner distance [1,H,W]."""
    h, w = label.shape
    heat = np.zeros((n_classes, h, w), np.float32)
    ids = np.unique(label)
    ids = ids[ids > 0]
    fg = (label > 0).astype(np.float32)
    if len(ids) == 0:
        return heat, fg[None], np.zeros((1, h, w), np.float32)
    pts = np.array(ndi.center_of_mass(np.ones_like(label), label, ids)).reshape(-1, 2)
    for i, (cy, cx) in zip(ids, pts):
        k = 0 if inst_cls is None else int(inst_cls[i])
        y0, x0 = int(round(cy)), int(round(cx))
        a0, a1, b0, b1 = max(y0 - _R, 0), min(y0 + _R + 1, h), max(x0 - _R, 0), min(x0 + _R + 1, w)
        sub = heat[k, a0:a1, b0:b1]
        np.maximum(sub, _STAMP[a0 - y0 + _R:a1 - y0 + _R, b0 - x0 + _R:b1 - x0 + _R], out=sub)
    edt = ndi.distance_transform_edt(~find_boundaries(label, mode="inner") & (label > 0)).astype(np.float32)
    maxima = np.zeros(int(label.max()) + 1, np.float32)
    maxima[ids] = ndi.maximum(edt, label, ids)
    dist = edt / np.maximum(maxima[label], 1.0)
    return heat, fg[None], dist[None]


def seed_of(idx, cond, target):
    return int(hashlib.md5(f"{idx}|{cond}|{target}".encode()).hexdigest()[:8], 16)


def instance_classes(label, cmap):
    """Instance id -> class index (0 positive, 1 negative) from a pixel class map (0 bg, 1, 2)."""
    if cmap is None:
        return None
    ids = np.unique(label)
    ids = ids[ids > 0]
    out = np.zeros(int(label.max()) + 1, int)
    if len(ids):
        out[ids] = np.round(ndi.median(cmap, label, ids)).astype(int) - 1
    return out


class CellData(torch.utils.data.Dataset):
    """Streams are stored as uint16 [N, M, 3, H, W] in 1/65535 units."""

    def __init__(self, root, dataset, split, crop=None, subset=None, cond="clean", target=0, return_label=False):
        d = os.path.join(root, dataset)
        self.streams = np.load(os.path.join(d, f"{split}_streams.npy"), mmap_mode="r")
        self.label = np.load(os.path.join(d, f"{split}_label.npy"), mmap_mode="r")
        path = os.path.join(d, f"{split}_cls.npy")
        self.cls = np.load(path, mmap_mode="r") if os.path.exists(path) else None
        self.n_classes = 2 if self.cls is not None else 1
        self.n_streams = self.streams.shape[1]
        self.crop, self.cond, self.target, self.return_label = crop, cond, target, return_label
        n = len(self.streams)
        keep = max(1, int(round(subset * n))) if subset is not None else n
        self.idx = np.arange(n) if subset is None else np.random.default_rng(2026).permutation(n)[:keep]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = np.asarray(self.streams[j], np.float32) / 65535.0
        label = np.asarray(self.label[j]).astype(np.int32)
        cmap = None if self.cls is None else np.asarray(self.cls[j])
        if self.crop:  # training path only
            h, w = label.shape
            y0, x0 = np.random.randint(0, h - self.crop + 1), np.random.randint(0, w - self.crop + 1)
            ys, xs = slice(y0, y0 + self.crop), slice(x0, x0 + self.crop)
            x, label = x[..., ys, xs], label[ys, xs]
            cmap = None if cmap is None else cmap[ys, xs]
            k, flip = np.random.randint(4), np.random.rand() < 0.5

            def geo(a):
                a = np.rot90(a, k, axes=(-2, -1))
                return np.flip(a, -1).copy() if flip else a.copy()

            x, label = geo(x), geo(label)
            cmap = None if cmap is None else geo(cmap)
            gamma = np.exp(np.random.uniform(np.log(0.7), np.log(1.4), size=(x.shape[0], 1, 1, 1)))
            x = np.clip(x, 0, 1) ** gamma
            label = relabel(label)
        inst_cls = instance_classes(label, cmap)
        streams = torch.from_numpy(x.astype(np.float32))
        if self.cond != "clean":
            gen = torch.Generator().manual_seed(seed_of(j, self.cond, self.target))
            streams = degrade.apply_condition(streams[None], self.cond, self.target, gen)[0]
        out = dict(streams=streams, index=j)
        if self.crop:
            heat, fg, dist = make_targets(label, inst_cls, self.n_classes)
            out.update(heat=torch.from_numpy(heat), fg=torch.from_numpy(fg), dist=torch.from_numpy(dist))
        if self.return_label:
            out.update(label=torch.from_numpy(label.copy()))
            if inst_cls is not None:
                out["inst_cls"] = inst_cls
        return out


def pad_multiple(x, m=7):
    """Reflect-pad the last two dims of [B, M, 3, H, W] to a multiple of m."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph == 0 and pw == 0:
        return x
    b, s, c = x.shape[:3]
    y = torch.nn.functional.pad(x.reshape(b * s, c, h, w), (0, pw, 0, ph), mode="reflect")
    return y.reshape(b, s, c, h + ph, w + pw)
