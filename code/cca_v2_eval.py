"""CCA variants on validation images: longer training, rotation averaging, dead zone, disagreement gate. Descriptive; rule frozen in design log."""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from b2_eval import load
from cca import CCA
from cca_eval import collate
from data import CellData
from reg_diag import f1, records
from reg_gate import Scene, as_rows, direction

SCENES = [(t, s) for t in (0, 1) for s in (4, 8, 16)]
VARIANTS = ("single", "tta", "tta_dead", "tta_dead_gate")


@torch.inference_mode()
def raw(model, scene, target, batch, loaders, device):
    """Float shifts for the four rotations, [N, 4, 2], expressed in the original frame."""
    out = []
    for x in torch.utils.data.DataLoader(scene, batch, num_workers=loaders, collate_fn=collate):
        x = x.to(device)
        t = torch.full((len(x),), target, device=device, dtype=torch.long)
        per = []
        for k in range(4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d = model(torch.rot90(x, k, (-2, -1)), t)["shift"].float()
            for _ in range(k):  # one counter-clockwise view turns (dy, dx) into (-dx, dy); undo it once per rotation
                d = torch.stack([d[:, 1], -d[:, 0]], 1)
            per.append(d)
        out.append(torch.stack(per, 1).cpu().numpy())
    return np.concatenate(out)


def variant(est, name):
    if name == "single":
        return [(int(round(v[0])), int(round(v[1]))) for v in est[:, 0]]
    med = np.median(est, 1)
    spread = (est.max(1) - est.min(1)).max(1)
    shifts = []
    for v, w in zip(med, spread):
        d = (int(round(v[0])), int(round(v[1])))
        if name != "tta" and max(abs(d[0]), abs(d[1])) <= 1:
            d = (0, 0)
        if name == "tta_dead_gate" and w > 6:
            d = (0, 0)
        shifts.append(d)
    return shifts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dataset", default="tissuenet")
    p.add_argument("--run", required=True)
    p.add_argument("--cca", required=True, nargs="+", help="name=path")
    p.add_argument("--part", choices=["tune", "pilot", "all"], default="tune")
    p.add_argument("--pick", nargs="*", default=[], help="model:variant pairs to score")
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    device = torch.device("cuda")
    probe = CellData(a.root, a.dataset, "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)
    summary = "val_summary.json" if os.path.exists(os.path.join(a.run, "val_summary.json")) else "test_summary.json"
    thresholds = json.load(open(os.path.join(a.run, summary)))["thresholds"]
    aligners = {}
    for c in a.cca:
        name, path = c.split("=")
        m = CCA(a.snapshot)
        m.load_state_dict(torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
        aligners[name] = m.to(device).eval()

    def dataset(t):
        ds = CellData(a.root, a.dataset, "val", target=t, return_label=True)
        order = np.random.default_rng(0).permutation(len(ds.idx))
        if a.part == "tune":
            ds.idx = ds.idx[order[:a.tune]]
        elif a.part == "pilot":
            ds.idx = ds.idx[order[a.tune:a.tune + a.limit]]
        elif a.limit:
            ds.idx = ds.idx[order[:a.limit]]
        return ds

    wanted = [tuple(v.split(":")) for v in a.pick] or [(n, v) for n in aligners for v in VARIANTS]
    res = {f"{n}:{v}": dict(mis={}, ali={}, err={}, clean={}) for n, v in wanted}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for t, s in SCENES + [(0, 0), (1, 0)]:
            base = dataset(t)
            scene = Scene(base, t, s)
            truth = [tuple(-v for v in direction(base[i]["index"], t, s)) if s else (0, 0) for i in range(len(base))]
            ests = {n: raw(m, scene, t, a.batch, a.loaders, device) for n, m in aligners.items()}
            for n, v in wanted:
                shifts = variant(ests[n], v)
                err = np.array([max(abs(u[0] - w[0]), abs(u[1] - w[1])) for u, w in zip(shifts, truth)])
                score = f1(as_rows(records(Scene(base, t, s, shifts), full, thresholds, a, device, pool)))
                row = res[f"{n}:{v}"]
                row["mis" if s else "ali"][f"{t}/s{s}"] = score
                row["err"][f"{t}/s{s}"] = dict(median=float(np.median(err)), within1=float((err <= 1).mean()), within2=float((err <= 2).mean()))
                print(n, v, t, s, round(score, 2), row["err"][f"{t}/s{s}"], flush=True)
            if not s:
                clean = f1(as_rows(records(scene, full, thresholds, a, device, pool)))
                for row in res.values():
                    row["clean"][str(t)] = clean
    for key, row in res.items():
        row["mis_mean"] = float(np.mean(list(row["mis"].values())))
        row["harm"] = float(np.mean([row["clean"][k.split("/")[0]] - v for k, v in row["ali"].items()]))
        print(key, "mis_mean", round(row["mis_mean"], 2), "harm", round(row["harm"], 2), flush=True)
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_limit{a.limit}" if (a.part == "all" and a.limit) or a.limit not in (0, 600) else ""
    json.dump(res, open(os.path.join(a.out, f"cca_v2_{a.dataset}_{a.part}{suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
