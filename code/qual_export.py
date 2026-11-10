"""Material for the qualitative figure. Descriptive; subset and selection rule documented in results/README.md.

TissueNet test, first `--n` images of a seed-0 permutation. Scenes: nuclear stream moved 16 px, membrane stream moved 16 px, and an
aligned clean scene. Methods: no alignment, gated MI (tau 0.1), CCA (one seed, tta_dead), oracle shift (misaligned scenes only).
Selection: per misaligned scene, images with >= 30 cells at the 25/50/75th percentile of CCA instance F1; in the aligned scene, the
image at the median MI harm among images where gated MI applied a non-zero shift; one CCA failure (shift error > 2 px) at the median
cell count over both misaligned scenes. Saves an npz with the picked images and a json summary; rendering happens locally.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from b2_eval import load
from cca import CCA
from cca_v2_eval import raw, variant
from data import CellData
from evaluate import predict
from metrics import decode, image_record
from reg_gate import Scene, direction, job

SCENES = [("clean", 0, 16), ("clean", 1, 16), ("clean", 0, 0)]
MI_TAU = 0.1


def _decode(args):
    heat, fg, dist, label, ht, ft = args
    pred, pts, _ = decode(heat, fg, dist, ht, ft)
    return pred.astype(np.int32), image_record(label, pred, pts)


def instances(model, ds, thresholds, a, device, pool):
    ht, ft = thresholds
    jobs = [(p[0], p[1], p[2], p[3], ht, ft) for p in predict(model, ds, a.batch, a.loaders, device)]
    return pool.map(_decode, jobs, chunksize=8)


def inst_f1(rec):
    return 100 * 2 * rec["inst_tp"] / max(2 * rec["inst_tp"] + rec["inst_fp"] + rec["inst_fn"], 1)


def nearest(values, candidates, target):
    return int(candidates[np.argmin(np.abs(values[candidates] - target))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    device = torch.device("cuda")
    probe = CellData(a.root, "tissuenet", "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]
    aligner = CCA(a.snapshot)
    aligner.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    aligner.to(device).eval()

    def dataset(cond, t):
        ds = CellData(a.root, "tissuenet", "test", cond=cond, target=t, return_label=True)
        ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[: a.n]]
        return ds

    data = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for cond, t, s in SCENES:
            key = f"{cond}@{t}/s{s}"
            base = dataset(cond, t)
            n = len(base)
            scene = Scene(base, t, s)
            truth = [tuple(-v for v in direction(base[i]["index"], t, s)) if s else (0, 0) for i in range(n)]
            pairs = []
            for i in range(n):
                x = scene[i]["streams"].numpy().mean(1)
                pairs.append(("mi", x[1 - t].astype(np.float64), x[t].astype(np.float64)))
            est = pool.map(job, pairs, chunksize=8)
            shifts = dict(none=[(0, 0)] * n, mi=[e[0] if e[1] >= MI_TAU else (0, 0) for e in est],
                          cca=variant(raw(aligner, scene, t, a.batch, a.loaders, device), "tta_dead"))
            if s:
                shifts["oracle"] = truth
            res = {name: instances(full, Scene(base, t, s, sh), thresholds, a, device, pool) for name, sh in shifts.items()}
            data[key] = dict(base=base, t=t, s=s, truth=truth, shifts=shifts, res=res)
            print(key, {k: round(float(np.mean([inst_f1(r) for _, r in v])), 2) for k, v in res.items()}, flush=True)

    picks, failures = [], []
    for key in ("clean@0/s16", "clean@1/s16"):
        d = data[key]
        cells = np.array([r["gt"] for _, r in d["res"]["none"]])
        f = np.array([inst_f1(r) for _, r in d["res"]["cca"]])
        ok = np.where(cells >= 30)[0]
        for q in (25, 50, 75):
            picks.append((key, nearest(f, ok, np.percentile(f[ok], q)), f"cca_p{q}"))
        err = np.array([max(abs(u[0] - w[0]), abs(u[1] - w[1])) for u, w in zip(d["shifts"]["cca"], d["truth"])])
        failures += [(key, int(i), int(cells[i])) for i in np.where(err > 2)[0]]
    if failures:
        counts = np.array([c for _, _, c in failures], float)
        k = int(np.argmin(np.abs(counts - np.median(counts))))
        picks.append((failures[k][0], failures[k][1], "cca_failure"))
    d = data["clean@0/s0"]
    applied = np.array([sh != (0, 0) for sh in d["shifts"]["mi"]])
    harm = np.array([inst_f1(r0) - inst_f1(r1) for (_, r0), (_, r1) in zip(d["res"]["none"], d["res"]["mi"])])
    if applied.any():
        cand = np.where(applied)[0]
        picks.append(("clean@0/s0", nearest(harm, cand, np.median(harm[cand])), "mi_harm_median"))

    arrays, summary = {}, dict(n=a.n, thresholds=thresholds, mi_tau=MI_TAU, picks=[], scene_means={}, mi_applied_aligned=float(applied.mean()))
    for key, d in data.items():
        summary["scene_means"][key] = {k: float(np.mean([inst_f1(r) for _, r in v])) for k, v in d["res"].items()}
    for j, (key, i, why) in enumerate(picks):
        d = data[key]
        item = Scene(d["base"], d["t"], d["s"])[i]
        tag = f"p{j}"
        arrays[f"{tag}_input"] = item["streams"].numpy().mean(1).astype(np.float16)
        arrays[f"{tag}_label"] = item["label"].numpy().astype(np.int32)
        info = dict(tag=tag, scene=key, moved_stream=d["t"], shift_px=d["s"], position=i, image_index=int(item["index"]), reason=why,
                    cells=int(d["res"]["none"][i][1]["gt"]), truth=list(d["truth"][i]), methods={})
        for name in d["res"]:
            pred, rec = d["res"][name][i]
            arrays[f"{tag}_{name}_pred"] = pred
            arrays[f"{tag}_{name}_streams"] = Scene(d["base"], d["t"], d["s"], d["shifts"][name])[i]["streams"].numpy().mean(1).astype(np.float16)
            info["methods"][name] = dict(shift=[int(v) for v in d["shifts"][name][i]], inst_f1=inst_f1(rec))
        summary["picks"].append(info)
        print(tag, key, why, {k: round(v["inst_f1"], 1) for k, v in info["methods"].items()}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "qual_tissuenet.npz"), **arrays)
    json.dump(summary, open(os.path.join(a.out, "qual_tissuenet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
