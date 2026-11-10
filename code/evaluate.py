"""Evaluate a checkpoint. Thresholds come from clean validation images; test labels are read only here."""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch
from scipy.special import expit

from data import CellData, pad_multiple
from degrade import SEVERITY_CURVES, SINGLE_STREAM_CONDITIONS
from metrics import decode, image_record, summarise
from model import build

HEAT_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
FG_GRID = (0.3, 0.4, 0.5, 0.6)


def collate(items):
    keys = ("streams", "index")
    out = {k: torch.utils.data.default_collate([i[k] for i in items]) for k in keys}
    out["label"] = [i["label"].numpy() for i in items]
    out["inst_cls"] = [i.get("inst_cls") for i in items]
    return out


def _score(job):
    heat, fg, dist, label, inst_cls, heat_thr, fg_thr = job
    pred, pts, cls = decode(heat, fg, dist, heat_thr, fg_thr)
    return image_record(label, pred, pts, gt_class=inst_cls, pred_class=cls if inst_cls is not None else None)


@torch.no_grad()
def predict(model, ds, batch, workers, device):
    dl = torch.utils.data.DataLoader(ds, batch, num_workers=workers, collate_fn=collate)
    for b in dl:
        h, w = b["streams"].shape[-2:]
        x = pad_multiple(b["streams"]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x)
        heat = out["heat"][..., :h, :w].float().cpu().numpy()
        fg = out["fg"][:, 0, :h, :w].float().cpu().numpy()
        dist = out["dist"][:, 0, :h, :w].float().cpu().numpy()
        for i in range(len(heat)):
            yield expit(heat[i]), expit(fg[i]), expit(dist[i]), b["label"][i], b["inst_cls"][i], int(b["index"][i])


def select_thresholds(model, root, dataset, batch, workers, device, pool, max_images=300):
    """Choose decoding thresholds on clean validation images by instance F1 (same rule for all methods)."""
    ds = CellData(root, dataset, "val", return_label=True)
    ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[:max_images]]
    preds = list(predict(model, ds, batch, workers, device))
    best, best_f1 = None, -1.0
    for ht in HEAT_GRID:
        for ft in FG_GRID:
            recs = pool.map(_score, [(p[0], p[1], p[2], p[3], p[4], ht, ft) for p in preds], chunksize=8)
            f1 = summarise(recs)["inst_f1"]
            if f1 > best_f1:
                best, best_f1 = (ht, ft), f1
    return best, best_f1


def conditions(n_streams, curves):
    conds = [("clean", 0)]
    for name in SINGLE_STREAM_CONDITIONS:
        conds += [(name, t) for t in range(n_streams)]
    conds.append(("all_photon10", 0))
    if curves:
        for name in SEVERITY_CURVES:
            conds += [(name, t) for t in range(n_streams)]
    return conds


def run(a):
    device = torch.device(a.device)
    ck = torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=True)
    probe = CellData(a.root, a.dataset, "val")
    model, _ = build(ck["config"], a.snapshot, probe.n_streams, probe.n_classes)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if unexpected or any(not k.startswith("backbone.") for k in missing):
        raise RuntimeError(f"bad checkpoint: {missing} {unexpected}")
    model.to(device).eval()
    os.makedirs(a.out, exist_ok=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        (ht, ft), val_f1 = select_thresholds(model, a.root, a.dataset, a.batch, a.workers, device, pool)
        result = dict(config=ck["config"], thresholds=[ht, ft], val_clean_f1=val_f1, n_train=ck["n_train"], per_condition={})
        per_image = {}
        for name, target in conditions(probe.n_streams, a.curves):
            key = f"{name}@{target}"
            ds = CellData(a.root, a.dataset, a.split, cond=name, target=target, return_label=True)
            if a.limit:  # pilot on validation images not used for threshold selection
                ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[300:300 + a.limit]]
            recs = pool.map(_score, [(p[0], p[1], p[2], p[3], p[4], ht, ft)
                                     for p in predict(model, ds, a.batch, a.workers, device)], chunksize=8)
            per_image[key] = recs
            result["per_condition"][key] = summarise(recs)
            print(key, result["per_condition"][key], flush=True)
    json.dump(result, open(os.path.join(a.out, f"{a.split}_summary.json"), "w"), indent=1)
    json.dump(per_image, open(os.path.join(a.out, f"{a.split}_per_image.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--curves", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    run(p.parse_args())
