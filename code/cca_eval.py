"""Stage 1 of the alignment module: CCA in front of a fixed `full` checkpoint against gated mutual information (criterion B1).

Validation images only. Same 20 scenes, tuning images and pilot images as reg_gate.py. The mutual information method and its
threshold stay as selected under criterion A2; the CCA confidence threshold is chosen on the tuning images with the same objective.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from cca import CCA
from data import CellData
from model import build
from reg_diag import f1, records
from reg_gate import ALIGNED, MISALIGNED, Scene, as_rows, direction, gated, job, mean_f1

TAUS = (-np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, np.inf)
MI_TAU = 0.1


def collate(items):
    return torch.stack([i["streams"] for i in items])


@torch.no_grad()
def estimate(model, scene, target, batch, loaders, device):
    shifts, conf = [], []
    for x in torch.utils.data.DataLoader(scene, batch, num_workers=loaders, collate_fn=collate):
        t = torch.full((len(x),), target, device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x.to(device), t)
        shifts += [(int(round(v[0])), int(round(v[1]))) for v in out["shift"].float().cpu().tolist()]
        conf += out["conf"].float().sigmoid().cpu().tolist()
    return shifts, np.array(conf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda")
    ck = torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=True)
    probe = CellData(a.root, "tissuenet", "val")
    model, _ = build(ck["config"], a.snapshot, probe.n_streams, probe.n_classes)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    aligner = CCA(a.snapshot)
    aligner.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    aligner.to(device).eval()
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]
    n = a.tune + a.limit
    tune, pilot = np.arange(a.tune), np.arange(a.tune, n)

    def dataset(cond, target):
        ds = CellData(a.root, "tissuenet", "val", cond=cond, target=target, return_label=True)
        ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[:n]]
        return ds

    rows, conf, gains, errors = {}, {}, {}, {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for cond, t, s in MISALIGNED + ALIGNED:
            key = f"{cond}@{t}/s{s}"
            base = dataset(cond, t)
            scene = Scene(base, t, s)
            rows[f"none/{key}"] = as_rows(records(scene, model, thresholds, a, device, pool))
            pairs, truth = [], []
            for i in range(n):
                item = scene[i]
                x = item["streams"].numpy().mean(1)
                pairs.append((x[1 - t].astype(np.float64), x[t].astype(np.float64)))
                dy, dx = direction(item["index"], t, s) if s else (0, 0)
                truth.append((-dy, -dx))
            if s:
                rows[f"oracle/{key}"] = as_rows(records(Scene(base, t, s, truth), model, thresholds, a, device, pool))
            est = pool.map(job, [("mi", r, m) for r, m in pairs], chunksize=8)
            gains[key] = np.array([e[1] for e in est])
            rows[f"mi/{key}"] = as_rows(records(Scene(base, t, s, [e[0] for e in est]), model, thresholds, a, device, pool))
            shifts, conf[key] = estimate(aligner, scene, t, a.batch, a.loaders, device)
            rows[f"cca/{key}"] = as_rows(records(Scene(base, t, s, shifts), model, thresholds, a, device, pool))
            for name, values in (("cca", shifts), ("mi", [e[0] for e in est])):
                err = np.array([max(abs(u[0] - v[0]), abs(u[1] - v[1])) for u, v in zip(values, truth)])
                errors[f"{name}/{key}"] = dict(median=float(np.median(err[pilot])), within2=float((err[pilot] <= 2).mean()))
            print(key, {k.split("/")[0]: round(f1(v[pilot]), 2) for k, v in rows.items() if k.endswith("/" + key)},
                  "err", errors[f"cca/{key}"], errors[f"mi/{key}"], flush=True)

    mis = [f"{c}@{t}/s{s}" for c, t, s in MISALIGNED]
    ali = [f"{c}@{t}/s{s}" for c, t, s in ALIGNED]
    none = {k: rows[f"none/{k}"] for k in mis + ali}
    oracle = {k: rows[f"oracle/{k}"] for k in mis}
    mi = {k: gated(rows[f"mi/{k}"], none[k], gains[k], MI_TAU) for k in mis + ali}

    def apply(tau):
        return {k: gated(rows[f"cca/{k}"], none[k], conf[k], tau) for k in mis + ali}

    search = {str(tau): 0.5 * mean_f1(apply(tau), mis, tune) + 0.5 * mean_f1(apply(tau), ali, tune) for tau in TAUS}
    tau = float(max(search, key=search.get))
    cca = apply(tau)

    def score(idx):
        gain = float(np.mean([f1(cca[k][idx]) - f1(mi[k][idx]) for k in mis]))
        harm = float(np.mean([f1(none[k][idx]) - f1(cca[k][idx]) for k in ali]))
        room = float(np.mean([f1(oracle[k][idx]) - f1(cca[k][idx]) for k in mis]))
        return gain, harm, room

    gain, harm, room = score(pilot)
    rng = np.random.default_rng(0)
    draws = np.array([score(pilot[rng.integers(0, len(pilot), len(pilot))]) for _ in range(a.boot)])
    ci = lambda c: [float(v) for v in np.percentile(draws[:, c], [2.5, 97.5])]
    passed = bool(gain >= 3 and ci(0)[0] > 0 and harm <= 1.0)
    out = dict(images=dict(tune=a.tune, pilot=a.limit), tau=tau, search=search, gain=gain, gain_ci=ci(0), harm=harm, harm_ci=ci(1), room=room, room_ci=ci(2),
               mi_harm=float(np.mean([f1(none[k][pilot]) - f1(mi[k][pilot]) for k in ali])), passed=passed, shift_error=errors,
               per_scene={k: dict(none=f1(none[k][pilot]), mi=f1(mi[k][pilot]), cca=f1(cca[k][pilot]), cca_always=f1(rows[f"cca/{k}"][pilot]),
                                  oracle=f1(oracle[k][pilot]) if k in oracle else None, applied=float((conf[k][pilot] >= tau).mean())) for k in mis + ali})
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.limit == 600 and a.tune == 300 else f"_limit{a.limit}"
    json.dump(out, open(os.path.join(a.out, f"cca_stage1_tissuenet{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("tau", "gain", "gain_ci", "harm", "harm_ci", "mi_harm", "room", "room_ci", "passed")}, indent=1))
    for k, v in out["per_scene"].items():
        print(k, {x: (round(y, 2) if y is not None else None) for x, y in v.items()})


if __name__ == "__main__":
    main()
