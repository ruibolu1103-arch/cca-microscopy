"""Criterion T: frozen test-set evaluation of CCA (several seeds) in front of `full`, against end-to-end controls and gated MI.

Same 20 scenes as reg_gate.py, on every test image. Thresholds of each model are chosen on clean validation images.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from b2_eval import Told, load
from cca import CCA
from cca_eval import estimate
from cca_v2_eval import raw, variant
from data import CellData
from evaluate import select_thresholds
from reg_diag import f1, records
from reg_gate import ALIGNED, MISALIGNED, Scene, as_rows, direction, gated, job

MI_TAU = 0.1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dataset", default="tissuenet")
    p.add_argument("--run", required=True)
    p.add_argument("--controls", nargs="*", default=[], help="name=path")
    p.add_argument("--cca", required=True, nargs="+", help="one path per seed")
    p.add_argument("--variant", default="tta_dead", help="tta_dead, or gated:<tau> for the stage-one confidence gate")
    p.add_argument("--primary", default="full_xa")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda")
    probe = CellData(a.root, a.dataset, "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)
    controls = {c.split("=")[0]: load(c.split("=")[1], a.snapshot, probe.n_streams, probe.n_classes, device) for c in a.controls}
    aligners = []
    for path in a.cca:
        m = CCA(a.snapshot)
        m.load_state_dict(torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
        aligners.append(m.to(device).eval())
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]

    def dataset(cond, target):
        ds = CellData(a.root, a.dataset, "test", cond=cond, target=target, return_label=True)
        if a.limit:
            ds.idx = ds.idx[: a.limit]
        return ds

    def shifts_of(model, scene, t):
        if a.variant.startswith("gated"):
            tau = float(a.variant.split(":")[1])
            s, conf = estimate(model, scene, t, a.batch, a.loaders, device)
            return [v if c >= tau else (0, 0) for v, c in zip(s, conf)]
        return variant(raw(model, scene, t, a.batch, a.loaders, device), a.variant)

    rows, chosen, errors, n = {}, {}, {}, None
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, model in controls.items():
            chosen[name] = select_thresholds(Told(model, 0), a.root, a.dataset, a.batch, a.loaders, device, pool)[0]
            print(name, "thresholds", chosen[name], flush=True)
        for cond, t, s in MISALIGNED + ALIGNED:
            key = f"{cond}@{t}/s{s}"
            base = dataset(cond, t)
            n = len(base)
            scene = Scene(base, t, s)
            truth = [tuple(-v for v in direction(base[i]["index"], t, s)) if s else (0, 0) for i in range(n)]
            r = dict(none=as_rows(records(scene, full, thresholds, a, device, pool)))
            if s:
                r["oracle"] = as_rows(records(Scene(base, t, s, truth), full, thresholds, a, device, pool))
            pairs = []
            for i in range(n):
                x = scene[i]["streams"].numpy().mean(1)
                pairs.append(("mi", x[1 - t].astype(np.float64), x[t].astype(np.float64)))
            est = pool.map(job, pairs, chunksize=8)
            mi = as_rows(records(Scene(base, t, s, [e[0] for e in est]), full, thresholds, a, device, pool))
            r["mi"] = gated(mi, r["none"], np.array([e[1] for e in est]), MI_TAU)
            for k, model in enumerate(aligners):
                shifts = shifts_of(model, scene, t)
                err = np.array([max(abs(u[0] - w[0]), abs(u[1] - w[1])) for u, w in zip(shifts, truth)])
                errors[f"cca{k}/{key}"] = dict(median=float(np.median(err)), within1=float((err <= 1).mean()), within2=float((err <= 2).mean()))
                r[f"cca{k}"] = as_rows(records(Scene(base, t, s, shifts), full, thresholds, a, device, pool))
            for name, model in controls.items():
                r[name] = as_rows(records(scene, Told(model, 1 - t), chosen[name], a, device, pool))
            rows[key] = r
            print(key, {k: round(f1(v), 2) for k, v in r.items()}, flush=True)

    mis = [f"{c}@{t}/s{s}" for c, t, s in MISALIGNED]
    ali = [f"{c}@{t}/s{s}" for c, t, s in ALIGNED]
    seeds = [f"cca{k}" for k in range(len(aligners))]

    def cca(k, idx):
        return float(np.mean([f1(rows[k][s][idx]) for s in seeds]))

    def stat(idx):
        out = dict(harm=float(np.mean([f1(rows[k]["none"][idx]) - cca(k, idx) for k in ali])),
                   room=float(np.mean([f1(rows[k]["oracle"][idx]) - cca(k, idx) for k in mis])),
                   vs_none=float(np.mean([cca(k, idx) - f1(rows[k]["none"][idx]) for k in mis])),
                   vs_mi=float(np.mean([cca(k, idx) - f1(rows[k]["mi"][idx]) for k in mis])))
        for name in controls:
            out[f"vs_{name}"] = float(np.mean([cca(k, idx) - f1(rows[k][name][idx]) for k in mis]))
            out[f"vs_{name}_aligned"] = float(np.mean([cca(k, idx) - f1(rows[k][name][idx]) for k in ali]))
        return out

    idx = np.arange(n)
    point = stat(idx)
    rng = np.random.default_rng(0)
    draws = [stat(idx[rng.integers(0, n, n)]) for _ in range(a.boot)]
    ci = {k: [float(v) for v in np.percentile([d[k] for d in draws], [2.5, 97.5])] for k in point}
    per_seed = {s: dict(vs_mi=float(np.mean([f1(rows[k][s]) - f1(rows[k]["mi"]) for k in mis])),
                        harm=float(np.mean([f1(rows[k]["none"]) - f1(rows[k][s]) for k in ali]))) for s in seeds}
    for name in controls:
        for s in seeds:
            per_seed[s][f"vs_{name}"] = float(np.mean([f1(rows[k][s]) - f1(rows[k][name]) for k in mis]))
    out = dict(dataset=a.dataset, images=n, variant=a.variant, seeds=len(aligners), thresholds=thresholds, control_thresholds=chosen,
               point=point, ci=ci, per_seed=per_seed, shift_error=errors,
               per_scene={k: {m: f1(v) for m, v in rows[k].items()} | {"cca_mean": cca(k, idx)} for k in mis + ali})
    if a.primary in controls:
        out["passed"] = dict(T1=bool(point[f"vs_{a.primary}"] >= 2 and ci[f"vs_{a.primary}"][0] > 0), T2=bool(point["harm"] <= 1.0),
                             T3=bool(point["vs_mi"] >= 3 and ci["vs_mi"][0] > 0))
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    json.dump(out, open(os.path.join(a.out, f"final_{a.dataset}{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("point", "ci", "per_seed") + (("passed",) if "passed" in out else ())}, indent=1))


if __name__ == "__main__":
    main()
