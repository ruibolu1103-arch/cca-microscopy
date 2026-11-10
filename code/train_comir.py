"""Train the CoMIR baseline on aligned training pairs (nuclear, membrane) with the same physics degradation as CCA."""
import argparse
import json
import os
import random
import time

import numpy as np
import torch

import degrade
from comir import networks, represent
from train_cca import Images


def pairwise_loss(z, tau):
    """InfoNCE of the authors' compute_pairwise_loss with the MSE critic, vectorised. z [B, 2, C, H, W]."""
    n = len(z)
    p = torch.cat([z[:, 0], z[:, 1]]).flatten(1).float()
    sq = p.pow(2).sum(1)
    sim = -(sq[:, None] + sq[None] - 2 * p @ p.T) / p.shape[1] / tau
    i = torch.arange(2 * n, device=z.device)
    pos = sim[i, (i + n) % (2 * n)]
    neg = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    return (-pos + torch.logsumexp(neg, 1)).mean()


def rotate(x, ks):
    return torch.stack([torch.rot90(v, int(k), (-2, -1)) for v, k in zip(x, ks)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", default="tissuenet")
    p.add_argument("--repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default="", help="pilot: start from these weights")
    p.add_argument("--real", default="", help="pilot: exemplar-001 folder; mixes real CyCIF pairs into training")
    p.add_argument("--real_tiles", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--real_frac", type=float, default=0.5)
    a = p.parse_args()
    if a.real:
        p.error("--real is an unpublished fine-tuning option; real_data.py is not distributed. See results/README.md.")
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda")
    nets = networks(a.repo, a.seed)
    if a.init:
        nets.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True))
    nets = nets.to(device)
    opt = [torch.optim.Adam(n.parameters(), lr=1e-3, weight_decay=1e-4) for n in nets]
    data = Images(a.root, a.dataset, crop=a.crop)
    dl = torch.utils.data.DataLoader(data, a.batch, shuffle=True, num_workers=a.workers, drop_last=True,
                                     persistent_workers=True)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w", encoding="utf8")
    it, t0, stream = 0, time.time(), iter(dl)
    nets.train()
    while it < a.iters:
        try:
            x = next(stream)
        except StopIteration:
            stream = iter(dl)
            x = next(stream)
        x = degrade.augment_physics(x.to(device, non_blocking=True))
        ks = [np.random.randint(4, size=len(x)) for _ in range(2)]
        z = torch.stack([rotate(nets[m](rotate(x[:, m, :1], ks[m])), -ks[m]) for m in range(2)], 1)  # C4 equivariance as in the notebook
        loss = pairwise_loss(z, a.tau)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}")
        for o in opt:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for n, o in zip(nets, opt):
            torch.nn.utils.clip_grad_norm_(n.parameters(), 1.0)
            o.step()
        it += 1
        if it % 100 == 0 or it == a.iters:
            log.write(json.dumps(dict(it=it, loss=loss.item(), sec=time.time() - t0)) + "\n")
            log.flush()
            print(it, round(loss.item(), 4), flush=True)
    torch.save(nets.state_dict(), os.path.join(a.out, "last.pt"))
    json.dump(dict(iters=a.iters, batch=a.batch, crop=a.crop, tau=a.tau), open(os.path.join(a.out, "train_done.json"), "w"))


if __name__ == "__main__":
    main()
