"""How much room does confidence-gated classical registration leave for a learned alignment module?

Validation images only. Method and gate threshold are chosen on the 300 threshold-selection images; numbers are
reported on the 600 pilot images. Historical gate diagnostic; see results/README.md for released result coverage. Scenes are central 224x224 crops, so a
displaced stream shows no zero border.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from data import CellData
from metrics import summarise
from model import build
from reg_diag import COLS, estimate, f1, move, mutual_information, overlap, records

CROP, BINS, LIMIT = 16, 32, 24
METHODS = ("mi", "phase")
TAUS = (-np.inf, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf)
MISALIGNED = [(c, t, s) for t in (0, 1) for c, s in (("clean", 4), ("clean", 8), ("clean", 16), ("photon10", 16), ("defocus4", 16), ("all_photon10", 16))]
ALIGNED = [(c, t, 0) for t in (0, 1) for c in ("clean", "photon10", "defocus4", "all_photon10")]


def direction(index, target, s):
    """Fixed random diagonal per image, target and size."""
    h = hashlib.md5(f"{index}|{target}|{s}".encode()).digest()
    return (s if h[0] & 1 else -s), (s if h[1] & 1 else -s)


class Scene(torch.utils.data.Dataset):
    """Degrade, displace the target stream, crop centrally, then optionally move it back by a per-image shift."""

    def __init__(self, base, target, s, shifts=None):
        self.base, self.target, self.s, self.shifts = base, target, s, shifts

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        x = item["streams"].clone()
        if self.s:
            x[self.target] = move(x[self.target], *direction(item["index"], self.target, self.s))
        x = x[..., CROP:-CROP, CROP:-CROP]
        if self.shifts is not None:
            x[self.target] = move(x[self.target], *self.shifts[i])
        return dict(item, streams=x, label=item["label"][CROP:-CROP, CROP:-CROP])


def level(v):
    return np.minimum((v * BINS).astype(np.int64), BINS - 1)


def job(args):
    method, ref, mov = args
    shift = estimate((method, ref, mov))
    if max(abs(shift[0]), abs(shift[1])) > LIMIT:
        shift = (0, 0)
    a, b = level(ref), level(mov)
    zero = mutual_information(*overlap(a, b, 0, 0), BINS)
    return shift, (mutual_information(*overlap(a, b, *shift), BINS) - zero) / max(zero, 1e-6)


def as_rows(recs):
    return np.array([[r[c] for c in COLS] for r in recs], float)


def gated(aligned, none, gain, tau):
    return np.where((gain >= tau)[:, None], aligned, none)


def mean_f1(rows, scenes, part):
    return float(np.mean([f1(rows[k][part]) for k in scenes]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tune", type=int, default=300)
    p.add_argument("--limit", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--loaders", type=int, default=2)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--boot", type=int, default=1000)
    a = p.parse_args()
    device = torch.device("cuda")
    ck = torch.load(os.path.join(a.run, "last.pt"), map_location="cpu", weights_only=True)
    probe = CellData(a.root, "tissuenet", "val")
    model, _ = build(ck["config"], a.snapshot, probe.n_streams, probe.n_classes)
    model.load_state_dict(ck["model"], strict=False)
    model.to(device).eval()
    thresholds = json.load(open(os.path.join(a.run, "val_summary.json")))["thresholds"]
    n = a.tune + a.limit
    tune, pilot = np.arange(a.tune), np.arange(a.tune, n)

    def dataset(cond, target):
        ds = CellData(a.root, "tissuenet", "val", cond=cond, target=target, return_label=True)
        ds.idx = ds.idx[np.random.default_rng(0).permutation(len(ds.idx))[:n]]
        return ds

    rows, gains, errors = {}, {}, {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for cond, t, s in MISALIGNED + ALIGNED:
            key = f"{cond}@{t}/s{s}"
            base = dataset(cond, t)
            rows[f"none/{key}"] = as_rows(records(Scene(base, t, s), model, thresholds, a, device, pool))
            scene = Scene(base, t, s)
            pairs, truth = [], []
            for i in range(n):
                item = scene[i]
                x = item["streams"].numpy().mean(1)
                pairs.append((x[1 - t].astype(np.float64), x[t].astype(np.float64)))
                dy, dx = direction(item["index"], t, s) if s else (0, 0)
                truth.append((-dy, -dx))
            if s:
                rows[f"oracle/{key}"] = as_rows(records(Scene(base, t, s, truth), model, thresholds, a, device, pool))
            for method in METHODS:
                est = pool.map(job, [(method, r, m) for r, m in pairs], chunksize=8)
                shifts = [e[0] for e in est]
                gains[f"{method}/{key}"] = np.array([e[1] for e in est])
                err = np.array([max(abs(u[0] - v[0]), abs(u[1] - v[1])) for u, v in zip(shifts, truth)])
                errors[f"{method}/{key}"] = dict(median=float(np.median(err[pilot])), within2=float((err[pilot] <= 2).mean()),
                                                 gain_median=float(np.median(gains[f"{method}/{key}"][pilot])))
                rows[f"{method}/{key}"] = as_rows(records(Scene(base, t, s, shifts), model, thresholds, a, device, pool))
            print(key, {k: round(f1(v[pilot]), 2) for k, v in rows.items() if k.endswith("/" + key)}, flush=True)

    mis = [f"{c}@{t}/s{s}" for c, t, s in MISALIGNED]
    ali = [f"{c}@{t}/s{s}" for c, t, s in ALIGNED]

    def apply(method, tau):
        return {k: gated(rows[f"{method}/{k}"], rows[f"none/{k}"], gains[f"{method}/{k}"], tau) for k in mis + ali}

    search = {}
    for method in METHODS:
        for tau in TAUS:
            g = apply(method, tau)
            search[f"{method}|{tau}"] = 0.5 * mean_f1(g, mis, tune) + 0.5 * mean_f1(g, ali, tune)
    best_key = max(search, key=search.get)
    method, tau = best_key.split("|")[0], float(best_key.split("|")[1])
    g = apply(method, tau)
    none = {k: rows[f"none/{k}"] for k in mis + ali}
    oracle = {k: rows[f"oracle/{k}"] for k in mis}

    def score(idx):
        room = float(np.mean([f1(oracle[k][idx]) - f1(g[k][idx]) for k in mis]))
        harm = float(np.mean([f1(none[k][idx]) - f1(g[k][idx]) for k in ali]))
        return room, harm

    room, harm = score(pilot)
    rng = np.random.default_rng(0)
    draws = np.array([score(pilot[rng.integers(0, len(pilot), len(pilot))]) for _ in range(a.boot)])
    pure = [k for k in mis if k.startswith("clean")]
    compound = [k for k in mis if not k.startswith("clean")]
    out = dict(images=dict(tune=a.tune, pilot=a.limit), thresholds=thresholds, method=method, tau=tau, search=search,
               room=room, harm=harm, room_ci=[float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
               harm_ci=[float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])],
               room_pure=float(np.mean([f1(oracle[k][pilot]) - f1(g[k][pilot]) for k in pure])),
               room_compound=float(np.mean([f1(oracle[k][pilot]) - f1(g[k][pilot]) for k in compound])),
               per_scene={k: dict(none=f1(none[k][pilot]), gated=f1(g[k][pilot]), always=f1(rows[f"{method}/{k}"][pilot]),
                                  oracle=f1(oracle[k][pilot]) if k in oracle else None,
                                  applied=float((gains[f"{method}/{k}"][pilot] >= tau).mean())) for k in mis + ali},
               shift_error=errors)
    if room >= 5:
        out["verdict"] = "room_for_module"
    elif room < 2 and harm <= 0.5:
        out["verdict"] = "classical_sufficient"
    else:
        out["verdict"] = "grey_user_decides"
    os.makedirs(a.out, exist_ok=True)
    suffix = "" if a.limit == 600 and a.tune == 300 else f"_limit{a.limit}"
    json.dump(out, open(os.path.join(a.out, f"reg_gate_tissuenet{suffix}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("method", "tau", "room", "room_ci", "room_pure", "room_compound", "harm", "harm_ci", "verdict")}, indent=1))
    for k, v in out["per_scene"].items():
        print(k, {x: (round(y, 2) if y is not None else None) for x, y in v.items()})


if __name__ == "__main__":
    main()
