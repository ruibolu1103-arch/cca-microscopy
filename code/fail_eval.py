"""Descriptive failure analysis: which test images defeat the aligners (clean, 16 px, both moving streams), and do they share
low signal? Per-image shift error of CCA (adopted, seed 0), CoMIR (seed 0) and gated MI against simple image statistics of the
moving and reference streams (before displacement) and the number of labelled cells (evaluation-only use of test labels)."""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from cca import CCA
from cca_v2_eval import raw, variant
from comir import load as load_comir, shifts as comir_shifts
from data import CellData
from reg_gate import Scene, direction, job


def stats(v):
    """v [H, W] in 0..1: fraction of bright pixels, contrast (p99 - p50), std."""
    return dict(bright=float((v > 0.3).mean()), contrast=float(np.percentile(v, 99) - np.percentile(v, 50)), std=float(v.std()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--comir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    device = torch.device("cuda")
    cca = CCA(a.snapshot)
    cca.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    cca = cca.to(device).eval()
    nets = load_comir(a.comir, a.repo, device)
    out = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for t in (0, 1):
            base = CellData(a.root, "tissuenet", "test", target=t, return_label=True)
            n = len(base)
            scene = Scene(base, t, 16)
            truth = np.array([[-v for v in direction(base[i]["index"], t, 16)] for i in range(n)], float)
            est = dict(cca=variant(raw(cca, scene, t, 16, 2, device), "tta_dead"), comir=comir_shifts(nets, scene, t, 16, 2, device))
            pairs, feats = [], []
            for i in range(n):
                item = base[i]
                x = item["streams"].numpy().mean(1)
                cells = len(np.unique(item["label"].numpy())) - 1
                feats.append(dict(cells=cells, **{f"moving_{k}": v for k, v in stats(x[t]).items()}, **{f"ref_{k}": v for k, v in stats(x[1 - t]).items()}))
                y = scene[i]["streams"].numpy().mean(1)
                pairs.append(("mi", y[1 - t].astype(np.float64), y[t].astype(np.float64)))
            mi = pool.map(job, pairs, chunksize=8)
            est["mi"] = [m[0] if m[1] >= 0.1 else (0, 0) for m in mi]
            fail = {k: np.abs(np.array(v, float) - truth).max(1) > 2 for k, v in est.items()}
            all_fail = fail["cca"] & fail["comir"] & fail["mi"]
            keys = list(feats[0])
            table = {k: np.array([f[k] for f in feats]) for k in keys}
            res = dict(n=n, fail_rate={k: float(v.mean()) for k, v in fail.items()}, all_three_fail=float(all_fail.mean()),
                       cca_fail_also_comir=float((fail["cca"] & fail["comir"]).sum() / max(fail["cca"].sum(), 1)),
                       median_when_cca_fails={k: float(np.median(table[k][fail["cca"]])) for k in keys},
                       median_when_cca_ok={k: float(np.median(table[k][~fail["cca"]])) for k in keys},
                       cca_fail_by_quartile={})
            for k in keys:
                q = np.quantile(table[k], [0.25, 0.5, 0.75])
                bins = np.digitize(table[k], q)
                res["cca_fail_by_quartile"][k] = [float(fail["cca"][bins == b].mean()) if (bins == b).any() else None for b in range(4)]
            res["comir_fail_by_quartile"] = {k: [float(fail["comir"][np.digitize(table[k], np.quantile(table[k], [0.25, 0.5, 0.75])) == b].mean()) for b in range(4)]
                                             for k in ("cells", "moving_contrast", "ref_contrast")}
            out[f"t{t}"] = res
            print(t, json.dumps({k: res[k] for k in ("fail_rate", "all_three_fail", "cca_fail_also_comir")}), flush=True)
            print(t, "cca fail by quartile", {k: [round(v, 3) if v is not None else None for v in vs] for k, vs in res["cca_fail_by_quartile"].items()}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "fail_tissuenet.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
