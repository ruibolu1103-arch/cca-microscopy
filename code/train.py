"""Train one configuration. Only the train split is read here."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import degrade
from data import CellData
from model import build


def focal(logit, target, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.9999).float()
    loss = -(pos * (1 - p) ** alpha * torch.log(p) + (1 - pos) * (1 - target) ** beta * p ** alpha * torch.log(1 - p))
    return loss.sum() / pos.sum().clamp_min(1.0)


def main_loss(out, b):
    heat = focal(out["heat"], b["heat"])
    bce = F.binary_cross_entropy_with_logits(out["fg"], b["fg"])
    p = torch.sigmoid(out["fg"])
    dice = 1 - (2 * (p * b["fg"]).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + b["fg"].sum((1, 2, 3)) + 1)
    dist = F.l1_loss(torch.sigmoid(out["dist"]), b["dist"], reduction="none")
    dist = (dist * b["fg"]).sum() / b["fg"].sum().clamp_min(1.0)
    return heat + bce + dice.mean() + dist, dict(heat=heat.item(), bce=bce.item(), dist=dist.item())


def aux_and_trust(out, b, cfg, kappa, lam_aux, lam_trust):
    """Stream-specific auxiliary loss and, for the trust gate, the error-calibrated trust loss."""
    aux = out["aux"]  # [B, M, K+1, H, W]
    m, h, w = aux.shape[1], aux.shape[3], aux.shape[4]
    t_heat = F.adaptive_max_pool2d(b["heat"], (h, w))
    t_fg = F.adaptive_avg_pool2d(b["fg"], (h, w))
    target = torch.cat([t_heat, t_fg], 1)[:, None].expand_as(aux)
    err = F.binary_cross_entropy_with_logits(aux, target, reduction="none").mean(2)  # [B, M, H, W]
    if cfg.get("aux") == "hard":  # hard-pixel supervision of the source paper
        with torch.no_grad():
            pred = F.adaptive_avg_pool2d(torch.sigmoid(out["fg"]), (h, w)) > 0.5
            hard = (pred != (t_fg > 0.5)).float()
        loss = (err * hard).sum() / (hard.sum().clamp_min(1.0) * m)
    else:
        loss = err.mean()
    total = lam_aux * loss
    if cfg.get("trust"):
        e = err.detach()
        tau = F.softmax(-(e / e.mean((1, 2, 3), keepdim=True).clamp_min(1e-4)) / kappa, 1)
        kl = 0.0
        for r in out["rhos"]:
            log_q = F.log_softmax(r[:, :, 0], 1)
            kl = kl + (tau * (tau.clamp_min(1e-8).log() - log_q)).sum(1).mean()
        total = total + lam_trust * kl / len(out["rhos"])
    return total


def misalign(x, limit=24):
    """Displace one random stream of each sample (circular, so no zero border); labels stay in the other stream's frame."""
    b, m = x.shape[:2]
    x = x.clone()
    moved = torch.randint(0, m, (b,), device=x.device)
    for i in range(b):
        u = torch.rand(1).item()
        r = 0 if u < 0.25 else (limit if u < 0.75 else 6)
        dy, dx = torch.randint(-r, r + 1, (2,)).tolist()
        x[i, moved[i]] = torch.roll(x[i, moved[i]], (dy, dx), (-2, -1))
    return x, (moved + 1) % m


def augment(streams, kind):
    if kind == "physics":
        return degrade.augment_physics(streams)
    if kind == "mcrm":
        return degrade.augment_mcrm(streams)
    if kind == "dropout":
        return degrade.augment_channel_dropout(streams)
    return streams


def run(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device(a.device)
    ds = CellData(a.root, a.dataset, "train", crop=a.crop, subset=a.subset)

    def worker_init(w):
        np.random.seed((a.seed * 1000 + w + torch.initial_seed()) % (2 ** 32))
        torch.set_num_threads(1)

    dl = torch.utils.data.DataLoader(ds, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(ds) >= a.batch,
                                     persistent_workers=a.workers > 0, worker_init_fn=worker_init,
                                     pin_memory=device.type == "cuda")
    model, cfg = build(a.config, a.snapshot, ds.n_streams, ds.n_classes)
    model.to(device)
    params = model.trainable_parameters()
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    warm = max(1, int(0.05 * a.iters))

    def lr_at(i):
        if i < warm:
            return (i + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (i - warm) / max(a.iters - warm, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w", encoding="utf8")
    it, t0, stream = 0, time.time(), iter(dl)
    model.train()
    while it < a.iters:
        try:
            b = next(stream)
        except StopIteration:
            stream = iter(dl)
            b = next(stream)
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        x = augment(b["streams"], cfg["aug"])
        analysis = None
        if cfg.get("mis"):
            x, analysis = misalign(x)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(x) if analysis is None else model(x, analysis)
        # Losses run in float32: in bfloat16, 1 - 1e-4 rounds to 1 and the focal term hits log(0).
        out = {k: ([r.float() if r is not None else None for r in v] if isinstance(v, list) else v.float()) for k, v in out.items()}
        with torch.autocast(device.type, enabled=False):
            loss, parts = main_loss(out, b)
            if "aux" in out and cfg.get("aux"):
                loss = loss + aux_and_trust(out, b, cfg, a.kappa, a.lam_aux, a.lam_trust)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        it += 1
        if it % a.log_every == 0 or it == a.iters:
            rec = dict(it=it, loss=loss.item(), sec=time.time() - t0, **parts)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save(dict(model=state, config=a.config, args=vars(a), n_train=n_train, n_total=n_total),
               os.path.join(a.out, "last.pt"))
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    json.dump(dict(n_train=n_train, n_total=n_total, sec=time.time() - t0, peak_mem=peak),
              open(os.path.join(a.out, "train_done.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=12000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--crop", type=int, default=252)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--kappa", type=float, default=0.5)
    p.add_argument("--lam-aux", dest="lam_aux", type=float, default=0.4)
    p.add_argument("--lam-trust", dest="lam_trust", type=float, default=0.1)
    p.add_argument("--subset", type=float, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log-every", dest="log_every", type=int, default=100)
    run(p.parse_args())
