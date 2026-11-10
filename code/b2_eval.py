"""Stage 2 kill test (criterion B2): CCA in front of `full` against `full` retrained end to end with misalignment augmentation.

Validation images only; same scenes and image subsets as reg_gate.py and cca_eval.py. The CCA confidence threshold stays as chosen in stage 1.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from cca import CCA
from cca_eval import estimate
from data import CellData
from evaluate import select_thresholds
from model import build
from reg_diag import f1, records
from reg_gate import ALIGNED, MISALIGNED, Scene, as_rows, gated


class Told(torch.nn.Module):
    """Tells the control which stream defines the coordinate frame."""

    def __init__(self, model, stream):
        super().__init__()
        self.model, self.stream = model, stream

    def forward(self, x):
        return self.model(x, torch.full((len(x),), self.stream, device=x.device, dtype=torch.long))


def load(path, snapshot, n_streams, n_classes, device):
    ck = torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)
    model, _ = build(ck["config"], snapshot, n_streams, n_classes)
    model.load_state_dict(ck["model"], strict=False)
    return model.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda")
    probe = CellData(a.root, "tissuenet", "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)
    control = load(a.control, a.snapshot, probe.n_streams, probe.n_classes, device)
    aligner = CCA(a.snapshot)
    aligner.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    aligner.to(device).eval()
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]
    pilot = np.arange(a.limit)

    def dataset(cond, target):
        ds = CellData(a.root, "tissuenet", "val", cond=cond, target=target, return_label=True)
        ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[a.tune:a.tune + a.limit]]
        return ds

    rows = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        control_thresholds, val_f1 = select_thresholds(Told(control, 0), a.root, "tissuenet", a.batch, a.loaders, device, pool)
        print("control thresholds", control_thresholds, val_f1, flush=True)
        for cond, t, s in MISALIGNED + ALIGNED:
            key = f"{cond}@{t}/s{s}"
            base = dataset(cond, t)
            scene = Scene(base, t, s)
            none = as_rows(records(scene, full, thresholds, a, device, pool))
            shifts, conf = estimate(aligner, scene, t, a.batch, a.loaders, device)
            aligned = as_rows(records(Scene(base, t, s, shifts), full, thresholds, a, device, pool))
            rows[key] = dict(none=none, cca=gated(aligned, none, conf, a.tau),
                             control=as_rows(records(scene, Told(control, 1 - t), control_thresholds, a, device, pool)))
            print(key, {k: round(f1(v), 2) for k, v in rows[key].items()}, flush=True)

    mis = [f"{c}@{t}/s{s}" for c, t, s in MISALIGNED]
    ali = [f"{c}@{t}/s{s}" for c, t, s in ALIGNED]

    def score(idx):
        return (float(np.mean([f1(rows[k]["cca"][idx]) - f1(rows[k]["control"][idx]) for k in mis])),
                float(np.mean([f1(rows[k]["cca"][idx]) - f1(rows[k]["control"][idx]) for k in ali])))

    gain, aligned = score(pilot)
    rng = np.random.default_rng(0)
    draws = np.array([score(pilot[rng.integers(0, len(pilot), len(pilot))]) for _ in range(a.boot)])
    out = dict(images=a.limit, tau=a.tau, control_thresholds=control_thresholds, gain=gain, gain_ci=[float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
               aligned=aligned, aligned_ci=[float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])],
               per_scene={k: {m: f1(v) for m, v in rows[k].items()} for k in mis + ali})
    out["passed"] = bool(gain >= 2 and out["gain_ci"][0] > 0 and aligned >= -1)
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.limit == 600 else f"_limit{a.limit}"
    json.dump(out, open(os.path.join(a.out, f"b2_tissuenet{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("gain", "gain_ci", "aligned", "aligned_ci", "passed")}, indent=1))


if __name__ == "__main__":
    main()
