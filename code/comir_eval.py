"""Descriptive: CoMIR baseline against CCA (same seeds and variant as criterion T) in front of `full`, on the test split,
same 20 scenes as final_eval.py. Paired image bootstrap. Not part of any frozen criterion."""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from b2_eval import load
from cca import CCA
from cca_v2_eval import raw, variant
from comir import load as load_comir, shifts as comir_shifts
from data import CellData
from reg_diag import f1, records
from reg_gate import ALIGNED, MISALIGNED, Scene, as_rows, direction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--dataset", default="tissuenet")
    p.add_argument("--run", required=True)
    p.add_argument("--cca", required=True, nargs="+")
    p.add_argument("--comir", required=True, nargs="+", help="one CoMIR weight file per seed")
    p.add_argument("--repo", required=True)
    p.add_argument("--variant", default="tta_dead")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--dead", action="store_true", help="post-hoc check: CoMIR with the same <=1 px dead zone as CCA, CoMIR only")
    a = p.parse_args()
    device = torch.device("cuda")
    probe = CellData(a.root, a.dataset, "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)
    nets = [load_comir(c, a.repo, device) for c in a.comir]
    aligners = []
    for path in a.cca:
        m = CCA(a.snapshot)
        m.load_state_dict(torch.load(os.path.join(path, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
        aligners.append(m.to(device).eval())
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]
    rows, errors, n = {}, {}, None
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for cond, t, s in MISALIGNED + ALIGNED:
            key = f"{cond}@{t}/s{s}"
            base = CellData(a.root, a.dataset, "test", cond=cond, target=t, return_label=True)
            if a.limit:
                base.idx = base.idx[: a.limit]
            n = len(base)
            scene = Scene(base, t, s)
            truth = [tuple(-v for v in direction(base[i]["index"], t, s)) if s else (0, 0) for i in range(n)]
            ests = {("comir" if j == 0 and len(nets) == 1 else f"comir{j}"): comir_shifts(nt, scene, t, a.batch, a.loaders, device) for j, nt in enumerate(nets)}
            if a.dead:
                ests = {"comir_dead": [(0, 0) if max(abs(d[0]), abs(d[1])) <= 1 else d for d in next(iter(ests.values()))]}
            for k, model in enumerate(aligners if not a.dead else []):
                ests[f"cca{k}"] = variant(raw(model, scene, t, a.batch, a.loaders, device), a.variant)
            r = {}
            for name, sh in ests.items():
                err = np.array([max(abs(u[0] - w[0]), abs(u[1] - w[1])) for u, w in zip(sh, truth)])
                errors[f"{name}/{key}"] = dict(median=float(np.median(err)), within1=float((err <= 1).mean()), within2=float((err <= 2).mean()))
                r[name] = as_rows(records(Scene(base, t, s, sh), full, thresholds, a, device, pool))
            rows[key] = r
            print(key, {k: round(f1(v), 2) for k, v in r.items()}, {k: errors[f"{k}/{key}"]["within2"] for k in r}, flush=True)
    mis = [f"{c}@{t}/s{s}" for c, t, s in MISALIGNED]
    if a.dead:
        out = dict(dataset=a.dataset, images=n, shift_error=errors, per_scene={k: {m: f1(v) for m, v in rows[k].items()} for k in rows})
        json.dump(out, open(os.path.join(a.out, f"comir_dead_{a.dataset}.json"), "w"), indent=1)
        return
    ali = [f"{c}@{t}/s{s}" for c, t, s in ALIGNED]
    seeds = [f"cca{k}" for k in range(len(aligners))]
    comirs = [k for k in rows[mis[0]] if k.startswith("comir")]

    def stat(idx):
        cca = lambda k: float(np.mean([f1(rows[k][s][idx]) for s in seeds]))
        com = lambda k: float(np.mean([f1(rows[k][c][idx]) for c in comirs]))
        return dict(cca_minus_comir=float(np.mean([cca(k) - com(k) for k in mis])),
                    cca_minus_comir_aligned=float(np.mean([cca(k) - com(k) for k in ali])))

    idx = np.arange(n)
    point = stat(idx)
    rng = np.random.default_rng(0)
    draws = [stat(idx[rng.integers(0, n, n)]) for _ in range(a.boot)]
    ci = {k: [float(v) for v in np.percentile([d[k] for d in draws], [2.5, 97.5])] for k in point}
    out = dict(dataset=a.dataset, images=n, point=point, ci=ci, shift_error=errors,
               per_scene={k: {m: f1(v) for m, v in rows[k].items()} for k in mis + ali})
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    tag = f"comir{len(nets)}seeds_" if len(nets) > 1 else "comir_"
    json.dump(out, open(os.path.join(a.out, f"{tag}{a.dataset}{suffix}.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, f"{tag}{a.dataset}{suffix}_rows.npz"), **{f"{k}|{m}": v for k, r in rows.items() for m, v in r.items()})
    print(json.dumps(dict(point=point, ci=ci), indent=1))


if __name__ == "__main__":
    main()
