"""Is the score drop under channel displacement lost cells or only a coordinate mismatch? Validation pilot images by default; --split test uses every test image.

Each displaced scene is scored twice: labels fixed, and labels moved together with the displaced stream. The second score stays
high if predictions simply follow that stream. Descriptive, not a gate.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from data import CellData
from metrics import summarise
from model import build
from reg_diag import move, records
from reg_gate import CROP, Scene, direction


class MovedLabel(Scene):
    def __getitem__(self, i):
        item = self.base[i]
        x = item["streams"].clone()
        dy, dx = direction(item["index"], self.target, self.s)
        x[self.target] = move(x[self.target], dy, dx)
        label = move(item["label"], dy, dx)
        return dict(item, streams=x[..., CROP:-CROP, CROP:-CROP], label=label[CROP:-CROP, CROP:-CROP])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", default="val", choices=("val", "test"), help="test uses every test image")
    a = p.parse_args()
    device = torch.device("cuda")
    ck = torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=True)
    probe = CellData(a.root, "tissuenet", "val")
    model, _ = build(ck["config"], a.snapshot, probe.n_streams, probe.n_classes)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]

    def dataset(target):
        ds = CellData(a.root, "tissuenet", a.split, target=target, return_label=True)
        if a.split == "val":
            ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[a.tune:a.tune + a.limit]]
        return ds

    out = {}
    keys = ("inst_f1", "det_f1", "count_rel")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        out["clean"] = summarise(records(Scene(dataset(0), 0, 0), model, thresholds, a, device, pool))
        print("clean", {k: round(out["clean"][k], 3) for k in keys}, flush=True)
        for t in (0, 1):
            for s in (4, 8, 16):
                fixed = summarise(records(Scene(dataset(t), t, s), model, thresholds, a, device, pool))
                moved = summarise(records(MovedLabel(dataset(t), t, s), model, thresholds, a, device, pool))
                out[f"{t}/s{s}"] = dict(fixed=fixed, moved=moved)
                print(t, s, "fixed", {k: round(fixed[k], 3) for k in keys}, "moved", {k: round(moved[k], 3) for k in keys}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    out["split"], out["images"] = a.split, len(dataset(0))
    name = "frame_test_tissuenet.json" if a.split == "val" else "frame_test_tissuenet_test.json"
    json.dump(out, open(os.path.join(a.out, name), "w"), indent=1)


if __name__ == "__main__":
    main()
