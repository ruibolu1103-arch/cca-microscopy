"""Test whether classical registration repairs channel misregistration before any learned alignment is designed.

Validation images only, the 600 pilot images not used for threshold selection. Historical diagnostic; see results/README.md for released result coverage.
The shift-direction convention is: an estimate (dy, dx) moves the moving stream onto its reference.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from skimage.registration import phase_cross_correlation

from data import CellData
from evaluate import _score, predict
from metrics import summarise
from model import build

SHIFT = 16
METHODS = ("phase", "ncc", "mi")
# name: (condition, displaced stream, stream that gets moved, true shift of the moved stream)
CASES = {"known0": ("misreg16", 0, 0, (-SHIFT, -SHIFT)), "known1": ("misreg16", 1, 1, (-SHIFT, -SHIFT)),
         "fixed1": ("misreg16", 1, 0, (SHIFT, SHIFT)), "clean_m0": ("clean", 0, 0, (0, 0)), "clean_m1": ("clean", 0, 1, (0, 0))}
COLS = ("inst_tp", "inst_fp", "inst_fn")


def move(x, dy, dx):
    """Integer translation with zero borders, x [..., H, W]."""
    h, w = x.shape[-2:]
    y = torch.zeros_like(x)
    y[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = x[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return y


def mutual_information(a, b, bins):
    joint = np.bincount(a.ravel() * bins + b.ravel(), minlength=bins * bins).reshape(bins, bins).astype(np.float64)
    p = joint / joint.sum()
    q = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    keep = p > 0
    return float((p[keep] * np.log(p[keep] / q[keep])).sum())


def overlap(ref, mov, dy, dx):
    h, w = ref.shape
    return (ref[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)],
            mov[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)])


def search_mi(ref, mov, centre, radius, bins):
    best, best_score = centre, -1.0
    for dy in range(centre[0] - radius, centre[0] + radius + 1):
        for dx in range(centre[1] - radius, centre[1] + radius + 1):
            score = mutual_information(*overlap(ref, mov, dy, dx), bins)
            if score > best_score:
                best, best_score = (dy, dx), score
    return best


def estimate_mi(ref, mov, bins=32):
    """Exhaustive mutual information within +-24 px on half-resolution images, refined at full resolution."""
    level = lambda v: np.minimum((v * bins).astype(np.int64), bins - 1)
    half = lambda v: 0.25 * (v[:-1:2, :-1:2] + v[1::2, :-1:2] + v[:-1:2, 1::2] + v[1::2, 1::2])
    dy, dx = search_mi(level(half(ref)), level(half(mov)), (0, 0), 12, bins)
    return search_mi(level(ref), level(mov), (2 * dy, 2 * dx), 2, bins)


def estimate(job):
    method, ref, mov = job
    if method == "mi":
        return estimate_mi(ref, mov)
    if method == "phase":
        shift = phase_cross_correlation(ref, mov, normalization="phase")[0]
    else:  # masked normalised cross-correlation (Padfield 2012) with full masks: zero-mean and overlap-normalised
        full = np.ones_like(ref, dtype=bool)
        shift = phase_cross_correlation(ref, mov, reference_mask=full, moving_mask=full, overlap_ratio=0.8)[0]
    return int(round(float(shift[0]))), int(round(float(shift[1])))


class Moved(torch.utils.data.Dataset):
    """Moves one stream of every image by its own integer shift."""

    def __init__(self, base, stream=None, shifts=None):
        self.base, self.stream, self.shifts = base, stream, shifts

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        if self.stream is None:
            return item
        streams = item["streams"].clone()
        streams[self.stream] = move(streams[self.stream], *self.shifts[i])
        return dict(item, streams=streams)


def records(ds, model, thresholds, a, device, pool):
    ht, ft = thresholds
    return pool.map(_score, [(p[0], p[1], p[2], p[3], p[4], ht, ft) for p in predict(model, ds, a.batch, a.loaders, device)], chunksize=8)


def f1(rows, idx=None):
    s = (rows if idx is None else rows[idx]).sum(0)
    return 100 * 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1)


def recovery_ci(method, base, oracle, boot=2000):
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(boot):
        idx = rng.integers(0, len(method), len(method))
        draws.append((f1(method, idx) - f1(base, idx)) / max(f1(oracle, idx) - f1(base, idx), 1e-6))
    return [float(v) for v in np.percentile(draws, [2.5, 97.5])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    device = torch.device("cuda")
    ck = torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=True)
    probe = CellData(a.root, "tissuenet", "val")
    model, _ = build(ck["config"], a.snapshot, probe.n_streams, probe.n_classes)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]

    def dataset(cond, target):
        ds = CellData(a.root, "tissuenet", "val", cond=cond, target=target, return_label=True)
        ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[300:300 + a.limit]]
        return ds

    out = dict(thresholds=thresholds, images=a.limit, evals={}, shift_error={})
    per_image = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        def evaluate(name, ds):
            recs = records(ds, model, thresholds, a, device, pool)
            per_image[name] = recs
            out["evals"][name] = summarise(recs)
            print(name, {k: round(out["evals"][name][k], 3) for k in ("inst_f1", "count_rel")}, flush=True)

        n = a.limit
        evaluate("clean", Moved(dataset("clean", 0)))
        for t in (0, 1):
            evaluate(f"none{t}", Moved(dataset("misreg16", t)))
            evaluate(f"oracle{t}", Moved(dataset("misreg16", t), t, [(-SHIFT, -SHIFT)] * n))
        for case, (cond, target, moved, truth) in CASES.items():
            base = dataset(cond, target)
            pairs = []
            for i in range(n):
                s = base[i]["streams"].numpy().mean(1)  # [M, H, W]
                pairs.append((s[1 - moved].astype(np.float64), s[moved].astype(np.float64)))
            for method in METHODS:
                shifts = pool.map(estimate, [(method, r, m) for r, m in pairs], chunksize=8)
                err = np.array([max(abs(s[0] - truth[0]), abs(s[1] - truth[1])) for s in shifts])
                out["shift_error"][f"{method}/{case}"] = dict(median=float(np.median(err)), within2=float((err <= 2).mean()))
                print(method, case, out["shift_error"][f"{method}/{case}"], flush=True)
                evaluate(f"{method}/{case}", Moved(base, moved, shifts))

    rows = {k: np.array([[r[c] for c in COLS] for r in v], float) for k, v in per_image.items()}
    table = {}
    for method in METHODS:
        rec = [(f1(rows[f"{method}/known{t}"]) - f1(rows[f"none{t}"])) / max(f1(rows[f"oracle{t}"]) - f1(rows[f"none{t}"]), 1e-6) for t in (0, 1)]
        harm = f1(rows["clean"]) - min(f1(rows[f"{method}/clean_m0"]), f1(rows[f"{method}/clean_m1"]))
        table[method] = dict(recovery0=float(rec[0]), recovery1=float(rec[1]), clean_harm=float(harm),
                             ci0=recovery_ci(rows[f"{method}/known0"], rows["none0"], rows["oracle0"]),
                             ci1=recovery_ci(rows[f"{method}/known1"], rows["none1"], rows["oracle1"]))
    best = max(table, key=lambda k: table[k]["recovery0"] + table[k]["recovery1"])
    low = min(table[best]["recovery0"], table[best]["recovery1"])
    if low >= 0.9 and table[best]["clean_harm"] <= 0.5:
        verdict = "classical_sufficient"
    elif low < 0.5:
        verdict = "room_for_module"
    else:
        verdict = "grey_user_decides"
    out.update(table=table, best=best, verdict=verdict)
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.limit == 600 else f"_limit{a.limit}"
    json.dump(out, open(os.path.join(a.out, f"reg_diag_tissuenet{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("table", "best", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
