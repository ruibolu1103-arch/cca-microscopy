"""Real-data check on MCMICRO exemplar-001 (CyCIF, cycles 6-8, 6 tiles, DNA + 3 markers per cycle).

Every cycle has its own DNA stain, so the shift of a cycle's markers relative to cycle 6 is measured by same-stain DNA
registration, window by window. CCA then has to align a cycle-k marker to cycle-6 DNA (stream 1 moved), or cycle-k DNA to a
cycle-6 marker (stream 0 moved), without seeing cycle-k DNA. The real stage offsets (about 45 px) exceed the search range,
so moving windows are cut to leave a controlled residual (0, 4, 8 or 16 px diagonal, as in the synthetic scenes); the
truth is still measured from DNA. Within-cycle pairs (same acquisition) have truth zero and measure false shifts.
No labels exist here, so the score is shift error only. Criterion R is frozen in the design log.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import re

import numpy as np
import tifffile
import torch
from scipy.ndimage import zoom
from skimage.registration import phase_cross_correlation

from cca import CCA
from cca_v2_eval import raw, variant
from reg_diag import estimate
from reg_gate import job

W, MARGIN, MI_TAU = 224, 96, 0.1
CYCLES = (6, 7, 8)


def norm(x, lo=0.1, hi=99.9):
    a, b = np.percentile(x, [lo, hi])
    return np.zeros_like(x, np.float32) if b - a < 1e-6 else np.clip((x - a) / (b - a), 0, 1).astype(np.float32)


def load(root, scale):
    """{cycle: (tiles [T, 4, H, W] float32 illumination-corrected and resampled, marker names)}."""
    names = [l.strip().split(",") for l in open(os.path.join(root, "markers.csv"))][1:]
    out = {}
    for c in CYCLES:
        f = tifffile.TiffFile(os.path.join(root, "raw", f"exemplar-001-cycle-0{c}.ome.tiff"))
        ffp = tifffile.imread(os.path.join(root, "illumination", f"exemplar-001-cycle-0{c}-ffp.tif")).astype(np.float32)
        dfp = tifffile.imread(os.path.join(root, "illumination", f"exemplar-001-cycle-0{c}-dfp.tif")).astype(np.float32)
        tiles = []
        for s in f.series:
            x = np.clip((s.asarray().astype(np.float32) - dfp) / np.maximum(ffp, 1e-3), 0, None)
            tiles.append(np.stack([zoom(ch, scale, order=1) for ch in x]) if scale != 1 else x)
        out[c] = (np.stack(tiles), [n[2] for n in names if int(n[1]) == c])
    return out


def phase(a, b):
    return phase_cross_correlation(norm(a).astype(np.float64), norm(b).astype(np.float64), upsample_factor=10)[0]


def residual(tile, pos, k, t):
    h = hashlib.md5(f"{tile}|{pos}|{k}|{t}".encode()).digest()
    s = (0, 4, 8, 16)[h[0] % 4]
    return (s if h[1] & 1 else -s), (s if h[2] & 1 else -s)


def pair(ref_dna, ref_marker, mov, t):
    """Two streams [2, W, W]: stream 0 nuclear (DNA), stream 1 marker; stream t is the moving one."""
    x = (mov, ref_marker) if t == 0 else (ref_dna, mov)
    return torch.from_numpy(np.stack([norm(v) for v in x]))


class Pairs(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return dict(streams=self.items[i][:, None].expand(-1, 3, -1, -1).contiguous())


def build(data, keep=None):
    """Returns pair tensors and metadata rows."""
    tiles = data[6][0]
    n, _, h, w = tiles.shape
    grid = [(y, x) for y in range(MARGIN, h - MARGIN - W + 1, W) for x in range(MARGIN, w - MARGIN - W + 1, W)]
    pairs, meta, dna_check = [], [], []
    for i in range(n):
        if keep is not None and i not in keep:
            continue
        for k in (7, 8):
            g = phase(tiles[i, 0], data[k][0][i, 0])
            gi = np.round(g).astype(int)
            for j, (y, x) in enumerate(grid):
                ref = tiles[i, :, y:y + W, x:x + W]
                for t in (0, 1):
                    o = residual(i, j, k, t)
                    qy, qx = y - gi[0] + o[0], x - gi[1] + o[1]
                    if not (0 <= qy <= h - W and 0 <= qx <= w - W):
                        continue
                    mov = data[k][0][i, :, qy:qy + W, qx:qx + W]
                    truth = phase(ref[0], mov[0])
                    expect = np.array(o) + (g - gi)
                    ok = bool(np.abs(truth - expect).max() <= 1.5)
                    dna_check.append(dict(tile=i, pos=j, cycle=k, t=t, truth=truth.tolist(), expect=expect.tolist(), ok=ok))
                    if not ok:
                        continue
                    for m in (1, 2, 3):
                        if t == 1:  # cycle-k marker moves onto cycle-6 DNA
                            pairs.append(pair(ref[0], None, mov[m], 1))
                            name = data[k][1][m]
                        else:  # cycle-k DNA moves onto a cycle-6 marker
                            pairs.append(pair(None, ref[m], mov[0], 0))
                            name = data[6][1][m]
                        meta.append(dict(kind="cross", tile=i, pos=j, cycle=k, t=t, s=abs(o[0]), marker=name, truth=truth.tolist()))
        for c in CYCLES:
            for j, (y, x) in enumerate(grid):
                win = data[c][0][i, :, y:y + W, x:x + W]
                for m in (1, 2, 3):
                    for t in (0, 1):
                        pairs.append(pair(win[0], win[m], win[t * m], t))
                        meta.append(dict(kind="within", tile=i, pos=j, cycle=c, t=t, s=0, marker=data[c][1][m], truth=[0.0, 0.0]))
    return pairs, meta, dna_check


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="exemplar-001 folder with raw/, illumination/, markers.csv")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--cca", required=True, nargs="+")
    p.add_argument("--variant", default="tta_dead")
    p.add_argument("--scale", type=float, default=1.3, help="0.65 um/px to the assumed 0.5 um/px of TissueNet")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--comir", default="", help="optional CoMIR weights (descriptive)")
    p.add_argument("--repo", default="")
    p.add_argument("--tiles", type=int, nargs="*", default=None, help="pilot: evaluate these tiles only")
    p.add_argument("--extra", nargs="*", default=[], help="pilot: name=path extra CCA models (also scored with MI fallback)")
    p.add_argument("--comir_extra", nargs="*", default=[], help="pilot: name=path extra CoMIR weights")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda")
    data = load(a.data, a.scale)
    pairs, meta, dna_check = build(data, a.tiles)
    print("pairs", len(pairs), "dna windows kept", sum(d["ok"] for d in dna_check), "of", len(dna_check), flush=True)
    est = {"none": [(0, 0)] * len(pairs)}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        jobs = [("mi", x[1 - r["t"]].numpy().astype(np.float64), x[r["t"]].numpy().astype(np.float64)) for x, r in zip(pairs, meta)]
        mi = pool.map(job, jobs, chunksize=8)
        est["mi"] = [s if gain >= MI_TAU else (0, 0) for s, gain in mi]
        est["ncc"] = pool.map(estimate, [("ncc",) + j[1:] for j in jobs], chunksize=8)
    for k, path in enumerate(a.cca):
        m = CCA(a.snapshot)
        m.load_state_dict(torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
        m.to(device).eval()
        shifts = [None] * len(pairs)
        for t in (0, 1):
            idx = [i for i, r in enumerate(meta) if r["t"] == t]
            for i, s in zip(idx, variant(raw(m, Pairs([pairs[i] for i in idx]), t, a.batch, 2, device), a.variant)):
                shifts[i] = s
        est[f"cca{k}"] = shifts
    if a.comir:
        from comir import load as load_comir, shifts as comir_shifts
        nets = load_comir(a.comir, a.repo, device)
        est["comir"] = [None] * len(pairs)
        for t in (0, 1):
            idx = [i for i, r in enumerate(meta) if r["t"] == t]
            for i, s in zip(idx, comir_shifts(nets, Pairs([pairs[i] for i in idx]), t, a.batch, 2, device)):
                est["comir"][i] = s
    from comir import load as load_comir2, shifts as comir_shifts2
    for spec in a.extra + [f"cca{k}={p}" for k, p in enumerate(a.cca)]:
        name, path = spec.split("=")
        m = CCA(a.snapshot)
        m.load_state_dict(torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
        m.to(device).eval()
        plain, hyb = [None] * len(pairs), [None] * len(pairs)
        for t in (0, 1):
            idx = [i for i, r in enumerate(meta) if r["t"] == t]
            r_ = raw(m, Pairs([pairs[i] for i in idx]), t, a.batch, 2, device)
            spread = (r_.max(1) - r_.min(1)).max(1)
            for i, s, w_ in zip(idx, variant(r_, a.variant), spread):
                plain[i], hyb[i] = s, (est["mi"][i] if w_ > 6 else s)
        est[name], est[f"{name}+mi"] = plain, hyb
    for spec in a.comir_extra:
        name, path = spec.split("=")
        nets2 = load_comir2(path, a.repo, device)
        est[name] = [None] * len(pairs)
        for t in (0, 1):
            idx = [i for i, r in enumerate(meta) if r["t"] == t]
            for i, s in zip(idx, comir_shifts2(nets2, Pairs([pairs[i] for i in idx]), t, a.batch, 2, device)):
                est[name][i] = s
    seeds = [f"cca{k}" for k in range(len(a.cca))]
    truth = np.array([r["truth"] for r in meta])
    err = {name: np.abs(np.array(v, float) - truth).max(1) for name, v in est.items()}
    mag = {name: np.abs(np.array(v, float)).max(1) for name, v in est.items()}
    cross_moved = np.array([r["kind"] == "cross" and r["s"] > 0 for r in meta])
    within = np.array([r["kind"] == "within" for r in meta])
    cluster = np.array([r["tile"] * 1000 + r["pos"] for r in meta])
    ids = np.unique(cluster)
    members = {c: np.flatnonzero(cluster == c) for c in ids}

    def r1(sel):
        idx = sel[cross_moved[sel]]
        cca = np.mean([(err[s][idx] <= 2).mean() for s in seeds])
        return 100 * (cca - (err["mi"][idx] <= 2).mean())

    rng = np.random.default_rng(0)
    point = r1(np.arange(len(meta)))
    draws = [r1(np.concatenate([members[c] for c in rng.choice(ids, len(ids))])) for _ in range(a.boot)]
    r2 = float(np.mean([(mag[s][within] <= 2).mean() for s in seeds]))

    def summary(sel):
        return {name: dict(n=int(sel.sum()), within1=float((e[sel] <= 1).mean()), within2=float((e[sel] <= 2).mean()),
                           median=float(np.median(e[sel])), shifted=float((mag[name][sel] > 2).mean())) for name, e in err.items()}

    groups = {"cross_moved": cross_moved, "cross_s0": np.array([r["kind"] == "cross" and r["s"] == 0 for r in meta]), "within": within}
    for r in sorted({(x["kind"], x["t"], x["s"]) for x in meta}):
        groups[f"{r[0]}/t{r[1]}/s{r[2]}"] = np.array([(x["kind"], x["t"], x["s"]) == r for x in meta])
    for name in sorted({x["marker"] for x in meta}):
        groups[f"cross_moved/{name}"] = cross_moved & np.array([x["marker"] == name for x in meta])
        groups[f"within/{name}"] = within & np.array([x["marker"] == name for x in meta])
    out = dict(scale=a.scale, variant=a.variant, seeds=len(seeds), pairs=len(meta),
               dna_windows=dict(kept=int(sum(d["ok"] for d in dna_check)), total=len(dna_check)),
               R1=dict(within2_gain_vs_mi=float(point), ci=[float(v) for v in np.percentile(draws, [2.5, 97.5])]),
               R2=dict(cca_not_shifted_within=r2),
               groups={k: summary(v) for k, v in groups.items() if v.sum()}, dna_check=dna_check)
    out["passed"] = dict(R1=bool(point >= 10 and out["R1"]["ci"][0] > 0), R2=bool(r2 >= 0.9))
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, f"real_exemplar001{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("dna_windows", "R1", "R2", "passed")}, indent=1))
    for g in ("cross_moved", "cross_s0", "within"):
        print(g, {n: (round(v["within2"], 3), round(v["median"], 2)) for n, v in out["groups"][g].items()})


if __name__ == "__main__":
    main()
