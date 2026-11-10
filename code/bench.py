"""Descriptive efficiency table: parameters and inference time per image pair (224 x 224 view, TissueNet test images).

GPU methods are timed with CUDA synchronisation after warm-up, at batch 1 and batch 16; gated MI is timed per pair on one CPU
thread (its exhaustive search is how it is run in the experiments). The segmenter `full` is timed too, for scale."""
import argparse
import json
import os
import time

import numpy as np
import torch

from b2_eval import load
from cca import CCA
from comir import load as load_comir, represent, search
from data import CellData
from reg_gate import Scene, job


def params(m, trainable_only=False):
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only))


def timed(fn, x, reps):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps / len(x) * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cca", required=True)
    p.add_argument("--comir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=48)
    a = p.parse_args()
    torch.set_num_threads(1)
    device = torch.device("cuda")
    base = CellData(a.root, "tissuenet", "test", target=1, return_label=True)
    scene = Scene(base, 1, 16)
    xs = torch.stack([scene[i]["streams"] for i in range(a.pairs)])
    cca = CCA(a.snapshot)
    cca.load_state_dict(torch.load(os.path.join(a.cca, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    cca = cca.to(device).eval()
    nets = load_comir(a.comir, a.repo, device)
    probe = CellData(a.root, "tissuenet", "val")
    full = load(a.run, a.snapshot, probe.n_streams, probe.n_classes, device)

    @torch.inference_mode()
    def cca_single(x):
        t = torch.ones(len(x), dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return cca(x, t)["shift"]

    @torch.inference_mode()
    def cca_tta(x):
        t = torch.ones(len(x), dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return [cca(torch.rot90(x, k, (-2, -1)), t)["shift"] for k in range(4)]

    @torch.inference_mode()
    def comir_fn(x):
        z = represent(nets, x).float()
        return search(z[:, 0], z[:, 1])

    @torch.inference_mode()
    def full_fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return full(x)

    out = dict(params=dict(cca_trainable=params(cca, True), cca_total=params(cca), comir=params(nets), full_total=params(full)), ms_per_pair={})
    gpu = xs.to(device)
    for name, fn in (("cca_single", cca_single), ("cca_tta", cca_tta), ("comir", comir_fn), ("full_segmenter", full_fn)):
        for b in (1, 16):
            try:
                out["ms_per_pair"][f"{name}/b{b}"] = timed(fn, gpu[:b], max(3, 48 // b))
            except Exception as e:  # noqa: BLE001 - record and continue, it is a benchmark
                out["ms_per_pair"][f"{name}/b{b}"] = f"error: {e}"
            print(name, b, out["ms_per_pair"][f"{name}/b{b}"], flush=True)
    mi = []
    for i in range(min(a.pairs, 24)):
        x = xs[i].numpy().mean(1)
        t = time.perf_counter()
        job(("mi", x[0].astype(np.float64), x[1].astype(np.float64)))
        mi.append((time.perf_counter() - t) * 1000)
    out["ms_per_pair"]["gated_mi/cpu1"] = float(np.mean(mi))
    out["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(out, indent=1), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "bench.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
