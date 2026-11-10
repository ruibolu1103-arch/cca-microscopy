"""Descriptive component ablation of CCA on the test split, same 20 scenes as final_eval.py, CCA seed 0 in front of `full`.

coarse   : coarse level only (DINOv2 token correlation), no rotation averaging, no dead zone
single   : coarse + fine, one view
tta      : + median of the four rotations
tta_dead : + dead zone (|d| <= 1 -> 0); the adopted version
short    : adopted inference on the 6000-iteration model (effect of longer training)
Paired image bootstrap of each variant against tta_dead.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from b2_eval import load
from cca import CCA
from cca_eval import collate
from cca_v2_eval import raw, variant
from data import CellData
from reg_diag import f1, records
from reg_gate import ALIGNED, MISALIGNED, Scene, as_rows, direction


@torch.inference_mode()
def coarse(model, scene, target, batch, loaders, device):
    out = []
    for x in torch.utils.data.DataLoader(scene, batch, num_workers=loaders, collate_fn=collate):
        x = x.to(device)
        t = torch.full((len(x),), target, device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d = model(x, t)["coarse_shift"].float()
        out += [(int(round(float(v[0]))), int(round(float(v[1])))) for v in d.cpu()]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cca", required=True, help="adopted 18000-iteration model")
    p.add_argument("--short", required=True, help="6000-iteration model")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda")
    probe = CellData(a.root, "tissuenet", "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)
    models = {}
    for name, path in (("long", a.cca), ("short", a.short)):
        m = CCA(a.snapshot)
        m.load_state_dict(torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
        models[name] = m.to(device).eval()
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]
    rows, errors, n = {}, {}, None
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for cond, t, s in MISALIGNED + ALIGNED:
            key = f"{cond}@{t}/s{s}"
            base = CellData(a.root, "tissuenet", "test", cond=cond, target=t, return_label=True)
            if a.limit:
                base.idx = base.idx[: a.limit]
            n = len(base)
            scene = Scene(base, t, s)
            truth = [tuple(-v for v in direction(base[i]["index"], t, s)) if s else (0, 0) for i in range(n)]
            est = raw(models["long"], scene, t, a.batch, a.loaders, device)
            shifts = dict(none=[(0, 0)] * n, coarse=coarse(models["long"], scene, t, a.batch, a.loaders, device),
                          single=variant(est, "single"), tta=variant(est, "tta"), tta_dead=variant(est, "tta_dead"),
                          short=variant(raw(models["short"], scene, t, a.batch, a.loaders, device), "tta_dead"))
            r = {}
            for name, sh in shifts.items():
                err = np.array([max(abs(u[0] - w[0]), abs(u[1] - w[1])) for u, w in zip(sh, truth)])
                errors[f"{name}/{key}"] = dict(median=float(np.median(err)), within1=float((err <= 1).mean()), within2=float((err <= 2).mean()))
                r[name] = as_rows(records(Scene(base, t, s, sh), full, thresholds, a, device, pool))
            rows[key] = r
            print(key, {k: round(f1(v), 2) for k, v in r.items()}, flush=True)
    mis = [f"{c}@{t}/s{s}" for c, t, s in MISALIGNED]
    ali = [f"{c}@{t}/s{s}" for c, t, s in ALIGNED]
    names = [k for k in rows[mis[0]] if k != "none"]

    def stat(idx):
        out = {}
        for v in names:
            out[f"{v}/mis"] = float(np.mean([f1(rows[k][v][idx]) for k in mis]))
            out[f"{v}/harm"] = float(np.mean([f1(rows[k]["none"][idx]) - f1(rows[k][v][idx]) for k in ali]))
            out[f"{v}/mis_vs_adopted"] = float(np.mean([f1(rows[k][v][idx]) - f1(rows[k]["tta_dead"][idx]) for k in mis]))
            out[f"{v}/ali_vs_adopted"] = float(np.mean([f1(rows[k][v][idx]) - f1(rows[k]["tta_dead"][idx]) for k in ali]))
        return out

    idx = np.arange(n)
    point = stat(idx)
    rng = np.random.default_rng(0)
    draws = [stat(idx[rng.integers(0, n, n)]) for _ in range(a.boot)]
    ci = {k: [float(v) for v in np.percentile([d[k] for d in draws], [2.5, 97.5])] for k in point}
    out = dict(images=n, point=point, ci=ci, shift_error=errors, per_scene={k: {m: f1(v) for m, v in rows[k].items()} for k in mis + ali})
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    json.dump(out, open(os.path.join(a.out, f"ablation_tissuenet{suffix}.json"), "w"), indent=1)
    print(json.dumps(point, indent=1))


if __name__ == "__main__":
    main()
