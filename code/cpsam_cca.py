"""CCA in front of the pretrained Cellpose-SAM on displaced TissueNet validation scenes. Descriptive under criterion B2."""
import argparse
import json
import os

import numpy as np
import torch

from cca import CCA
from cca_eval import estimate
from data import CellData
from eval_cpsam import to_input
from metrics import centroids, image_record, summarise
from reg_diag import COLS, f1
from reg_gate import Scene, direction

MISALIGNED = [(t, s) for t in (0, 1) for s in (4, 8, 16)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--tau", type=float, default=0.3)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--split", default="val", help="test: every test image, adopted variant, optional CoMIR (descriptive)")
    p.add_argument("--variant", default="gated", help="gated (stage one, tau) or tta_dead")
    p.add_argument("--comir", default="")
    p.add_argument("--repo", default="")
    a = p.parse_args()
    from cellpose import models

    device = torch.device("cuda")
    cp = models.CellposeModel(gpu=True, pretrained_model=a.model)
    aligner = CCA(a.snapshot)
    aligner.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    aligner.to(device).eval()

    def dataset(target):
        ds = CellData(a.root, "tissuenet", a.split, target=target, return_label=True)
        if a.split == "val":
            ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[a.tune:a.tune + a.limit]]
        elif a.limit:  # fixed random test subset (seed 0), so the run fits in the server time
            ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[:a.limit]]
        return ds

    nets = None
    if a.comir:
        from comir import load as load_comir, shifts as comir_shifts
        nets = load_comir(a.comir, a.repo, device)

    def score(scene):
        recs = []
        for i in range(len(scene)):
            item = scene[i]
            mask = cp.eval(to_input(item["streams"], "tissuenet"), batch_size=a.batch)[0]
            _, pts = centroids(mask)
            recs.append(image_record(item["label"].numpy(), mask, pts.round().astype(int) if len(pts) else pts))
        return recs

    def rows(recs):
        return np.array([[r[c] for c in COLS] for r in recs], float)

    out, table = {}, {}
    for t, s in MISALIGNED + [(0, 0), (1, 0)]:
        key = f"clean@{t}/s{s}"
        base = dataset(t)
        scene = Scene(base, t, s)
        if a.variant == "gated":
            shifts, conf = estimate(aligner, scene, t, 16, 2, device)
            shifts = [v if c >= a.tau else (0, 0) for v, c in zip(shifts, conf)]
        else:
            from cca_v2_eval import raw, variant
            shifts = variant(raw(aligner, scene, t, 16, 2, device), a.variant)
        methods = dict(none=scene, cca=Scene(base, t, s, shifts))
        if nets is not None:
            methods["comir"] = Scene(base, t, s, comir_shifts(nets, scene, t, 16, 2, device))
        if s:
            truth = [tuple(-v for v in direction(base[i]["index"], t, s)) for i in range(len(base))]
            methods["oracle"] = Scene(base, t, s, truth)
        table[key] = {}
        for name, ds in methods.items():
            recs = score(ds)
            table[key][name] = rows(recs)
            out.setdefault(key, {})[name] = summarise(recs)
        print(key, {k: round(f1(v), 2) for k, v in table[key].items()}, flush=True)

    mis = [f"clean@{t}/s{s}" for t, s in MISALIGNED]
    ali = ["clean@0/s0", "clean@1/s0"]

    def stat(idx):
        return (float(np.mean([f1(table[k]["cca"][idx]) - f1(table[k]["none"][idx]) for k in mis])),
                float(np.mean([f1(table[k]["none"][idx]) - f1(table[k]["cca"][idx]) for k in ali])))

    n = len(next(iter(table.values()))["none"])
    idx = np.arange(n)
    gain, harm = stat(idx)
    rng = np.random.default_rng(0)
    draws = np.array([stat(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(a.boot)])
    comir = {}
    if nets is not None:
        def vs(idx):
            return (float(np.mean([f1(table[k]["cca"][idx]) - f1(table[k]["comir"][idx]) for k in mis])),
                    float(np.mean([f1(table[k]["cca"][idx]) - f1(table[k]["comir"][idx]) for k in ali])))
        pt = vs(idx)
        dr = np.array([vs(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(a.boot)])
        comir = dict(cca_minus_comir=pt[0], ci=[float(v) for v in np.percentile(dr[:, 0], [2.5, 97.5])],
                     cca_minus_comir_aligned=pt[1], aligned_ci=[float(v) for v in np.percentile(dr[:, 1], [2.5, 97.5])])
    result = dict(images=n, split=a.split, variant=a.variant, comir=comir, tau=a.tau, gain=gain, gain_ci=[float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
                  harm=harm, harm_ci=[float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])], per_scene=out)
    os.makedirs(a.out, exist_ok=True)
    suffix = ("" if a.limit == 600 else f"_limit{a.limit}") if a.split == "val" else f"_{a.split}" + (f"{a.limit}" if a.limit else "")
    json.dump(result, open(os.path.join(a.out, f"cpsam_cca_tissuenet{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: result[k] for k in ("gain", "gain_ci", "harm", "harm_ci")}, indent=1))


if __name__ == "__main__":
    main()
