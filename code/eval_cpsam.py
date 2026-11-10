"""Evaluate the pretrained Cellpose-SAM model under the same test conditions (no retraining).

Input mapping: TissueNet -> (nuclear, membrane, empty). DeepLIIF -> (IHC luminance, DAPI, Lap2), because the
model takes three channels. Output is reduced to the same per-image records; there are no class outputs.
"""
import argparse
import json
import os

import numpy as np

from data import CellData
from evaluate import conditions
from metrics import centroids, image_record, summarise


def to_input(streams, dataset):
    s = streams.numpy()
    if dataset == "tissuenet":
        return np.stack([s[0, 0], s[1, 0], np.zeros_like(s[0, 0])], -1)
    return np.stack([s[0].mean(0), s[1, 0], s[1, 1]], -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--model", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0, help="validation pilot images, same subset as evaluate.py")
    a = p.parse_args()
    from cellpose import models

    model = models.CellposeModel(gpu=True, pretrained_model=a.model)
    os.makedirs(a.out, exist_ok=True)
    probe = CellData(a.root, a.dataset, "test")
    per_image, summary = {}, {}
    for name, target in conditions(probe.n_streams, False):
        key = f"{name}@{target}"
        ds = CellData(a.root, a.dataset, a.split, cond=name, target=target, return_label=True)
        if a.limit:
            ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[300:300 + a.limit]]
        recs = []
        for i in range(len(ds)):
            item = ds[i]
            mask = model.eval(to_input(item["streams"], a.dataset), batch_size=a.batch)[0]
            _, pts = centroids(mask)
            recs.append(image_record(item["label"].numpy(), mask, pts.round().astype(int) if len(pts) else pts))
        per_image[key] = recs
        summary[key] = summarise(recs)
        print(key, summary[key], flush=True)
    json.dump(dict(config="cpsam", per_condition=summary), open(os.path.join(a.out, f"{a.split}_summary.json"), "w"), indent=1)
    json.dump(per_image, open(os.path.join(a.out, f"{a.split}_per_image.json"), "w"))


if __name__ == "__main__":
    main()
