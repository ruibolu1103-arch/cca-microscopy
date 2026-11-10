"""Convert the downloaded archives into <out>/<dataset>/<split>_{streams,label,cls}.npy.

Numeric arrays are read with allow_pickle=False. Run from outside the data directory with `python -I`-safe
imports only (numpy, scipy, PIL).
"""
import argparse
import io
import json
import os
import re
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def percentile_norm(x, lo=0.1, hi=99.9):
    a, b = np.percentile(x, [lo, hi])
    if b - a < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x.astype(np.float32) - a) / (b - a), 0, 1)


def to_u16(x):
    return np.round(x * 65535).astype(np.uint16)


def prepare_tissuenet(archive, out):
    d = os.path.join(out, "tissuenet")
    os.makedirs(d, exist_ok=True)
    report = {}
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith(".npz")]
        print(names, flush=True)
        for split in ("train", "val", "test"):
            cands = [n for n in names if re.search(rf"{split}[^/]*\.npz$", os.path.basename(n))]
            if not cands:
                raise SystemExit(f"no {split} archive in {names}")
            with z.open(cands[0]) as f:
                npz = np.load(io.BytesIO(f.read()), allow_pickle=False)
                x, y = npz["X"], npz["y"]
            print(split, x.shape, y.shape, x.dtype, y.dtype, flush=True)
            n, h, w, c = x.shape
            if c != 2:
                raise SystemExit(f"expected two channels, got {c}")
            s = np.lib.format.open_memmap(os.path.join(d, f"{split}_streams.npy"), mode="w+", dtype=np.uint16, shape=(n, 2, 3, h, w))
            blank = [0, 0]
            for i in range(n):
                for k in range(2):
                    v = percentile_norm(x[i, ..., k])
                    blank[k] += int(v.max() == 0)
                    s[i, k] = to_u16(v)[None]
            s.flush()
            label = y[..., 0].astype(np.int32)  # channel 0 is the whole-cell label
            np.save(os.path.join(d, f"{split}_label.npy"), label)
            cells = [len(np.unique(l)) - 1 for l in label[:: max(1, n // 200)]]
            report[split] = dict(n=n, h=h, w=w, blank_channels=blank, mean_cells_sampled=float(np.mean(cells)))
    json.dump(report, open(os.path.join(d, "prepare_report.json"), "w"), indent=1)
    print(report, flush=True)


def decode_seg(seg):
    """Official DeepLIIF seg panel: red positive, blue negative, green boundary."""
    r, g, b = seg[..., 0].astype(int), seg[..., 1].astype(int), seg[..., 2].astype(int)
    pos, neg = (r > 128) & (r > g) & (r > b), (b > 128) & (b > g) & (b > r)
    label = np.zeros(seg.shape[:2], np.int32)
    cls = np.zeros(seg.shape[:2], np.uint8)
    next_id = 1
    for mask, c in ((pos, 1), (neg, 2)):
        comp, n = ndi.label(mask)
        if n == 0:
            continue
        sizes = ndi.sum(mask, comp, np.arange(1, n + 1))
        for i in np.nonzero(sizes >= 10)[0] + 1:
            region = comp == i
            label[region] = next_id
            cls[region] = c
            next_id += 1
    return label, cls


def prepare_deepliif(raw, out):
    """Panels: IHC, Hematoxylin, DAPI, Lap2, Marker, Seg. Hematoxylin derives from IHC and is unused."""
    d = os.path.join(out, "deepliif")
    os.makedirs(d, exist_ok=True)
    report = {}
    for split, zip_name in (("train", "DeepLIIF_Training_Set.zip"), ("val", "DeepLIIF_Validation_Set.zip"), ("test", "DeepLIIF_Testing_Set.zip")):
        with zipfile.ZipFile(os.path.join(raw, zip_name)) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".png") and "__MACOSX" not in n)
            first = np.array(Image.open(io.BytesIO(z.read(names[0]))).convert("RGB"))
            hs, ws = first.shape[:2]
            panels = ws // hs
            print(split, len(names), first.shape, "panels", panels, flush=True)
            if panels < 6:
                raise SystemExit(f"{zip_name}: expected at least six panels, found {panels}")
            n = len(names)
            s = np.lib.format.open_memmap(os.path.join(d, f"{split}_streams.npy"), mode="w+", dtype=np.uint16, shape=(n, 2, 3, hs, hs))
            labels = np.lib.format.open_memmap(os.path.join(d, f"{split}_label.npy"), mode="w+", dtype=np.int32, shape=(n, hs, hs))
            cls = np.lib.format.open_memmap(os.path.join(d, f"{split}_cls.npy"), mode="w+", dtype=np.uint8, shape=(n, hs, hs))
            pos, neg = 0, 0
            for i, name in enumerate(names):
                im = np.array(Image.open(io.BytesIO(z.read(name))).convert("RGB"))
                p = [im[:, j * hs:(j + 1) * hs] for j in range(panels)]
                s[i, 0] = to_u16(p[0].astype(np.float32) / 255).transpose(2, 0, 1)
                channels = [percentile_norm(q.max(2)) for q in (p[2], p[3], p[4])]
                s[i, 1] = to_u16(np.stack(channels))
                labels[i], cls[i] = decode_seg(p[panels - 1])
                pos += len(np.unique(labels[i][cls[i] == 1]))
                neg += len(np.unique(labels[i][cls[i] == 2]))
            s.flush(), labels.flush(), cls.flush()
            json.dump(names, open(os.path.join(d, f"{split}_names.json"), "w"))
            report[split] = dict(n=n, size=hs, panels=panels, positive=pos, negative=neg)
    json.dump(report, open(os.path.join(d, "prepare_report.json"), "w"), indent=1)
    print(report, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--which", nargs="+", default=["tissuenet", "deepliif"])
    a = p.parse_args()
    if "tissuenet" in a.which:
        prepare_tissuenet(os.path.join(a.raw, "tissuenet_v1.1.zip"), a.out)
    if "deepliif" in a.which:
        prepare_deepliif(a.raw, a.out)
