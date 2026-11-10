"""Descriptive failure boundary: alignment accuracy against shift size, clean TissueNet test images, both moving streams.

Diagonal shifts s in {4, 8, 16, 20, 24, 28, 32, 40} px. Search ranges by design: CCA coarse level +-28 px, gated MI and the
CoMIR search +-24 px, so the curve shows where each method stops working. Shift error only (no segmentation)."""
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

SIZES = (4, 8, 16, 20, 24, 28, 32, 40)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--comir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    device = torch.device("cuda")
    cca = CCA(a.snapshot)
    cca.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    cca = cca.to(device).eval()
    nets = load_comir(a.comir, a.repo, device)
    res = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for t in (0, 1):
            base = CellData(a.root, "tissuenet", "test", target=t, return_label=True)
            if a.limit:
                base.idx = base.idx[: a.limit]
            n = len(base)
            for s in SIZES:
                scene = Scene(base, t, s)
                truth = np.array([[-v for v in direction(base[i]["index"], t, s)] for i in range(n)], float)
                est = dict(cca=variant(raw(cca, scene, t, a.batch, 2, device), "tta_dead"),
                           comir=comir_shifts(nets, scene, t, a.batch, 2, device))
                pairs = []
                for i in range(n):
                    x = scene[i]["streams"].numpy().mean(1)
                    pairs.append(("mi", x[1 - t].astype(np.float64), x[t].astype(np.float64)))
                mi = pool.map(job, pairs, chunksize=8)
                est["mi"] = [m[0] if m[1] >= 0.1 else (0, 0) for m in mi]
                for name, v in est.items():
                    err = np.abs(np.array(v, float) - truth).max(1)
                    res[f"{name}/t{t}/s{s}"] = dict(within1=float((err <= 1).mean()), within2=float((err <= 2).mean()), median=float(np.median(err)))
                print(t, s, {k: round(res[f"{k}/t{t}/s{s}"]["within2"], 3) for k in est}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    suffix = f"_limit{a.limit}" if a.limit else ""
    json.dump(dict(sizes=SIZES, results=res), open(os.path.join(a.out, f"range_tissuenet{suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
