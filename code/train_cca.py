"""Train CCA on training images with synthetic known translations and physics degradation. No cell labels are used."""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import degrade
from cca import CCA
from reg_diag import move

MARGIN, LIMIT = 28, 24


class Images(torch.utils.data.Dataset):
    def __init__(self, root, dataset, crop=280):
        self.streams = np.load(os.path.join(root, dataset, "train_streams.npy"), mmap_mode="r")
        self.crop = crop

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, i):
        x = np.asarray(self.streams[i], np.float32) / 65535.0
        h, w = x.shape[-2:]
        y0, x0 = np.random.randint(0, h - self.crop + 1), np.random.randint(0, w - self.crop + 1)
        x = np.rot90(x[..., y0:y0 + self.crop, x0:x0 + self.crop], np.random.randint(4), axes=(-2, -1))
        if np.random.rand() < 0.5:
            x = np.flip(x, -1)
        gamma = np.exp(np.random.uniform(np.log(0.7), np.log(1.4), size=(x.shape[0], 1, 1, 1)))
        return torch.from_numpy((np.clip(x, 0, 1) ** gamma).astype(np.float32))


def sample_shift():
    u = random.random()
    if u < 0.25:
        return 0, 0
    r = LIMIT if u < 0.75 else 6
    return random.randint(-r, r), random.randint(-r, r)


def scene(x):
    """Degrade, displace one random stream and crop centrally. Returns streams, target stream and the corrective shift."""
    x = degrade.augment_physics(x)
    b = len(x)
    for i in torch.nonzero(torch.rand(b) < 0.15).flatten().tolist():  # both streams at low photon count
        peak = math.exp(random.uniform(math.log(5.0), math.log(60.0)))
        for m in range(x.shape[1]):
            x[i, m] = degrade.low_photon(x[i, m][None], peak=peak)[0]
    target = torch.randint(0, x.shape[1], (b,), device=x.device)
    truth = torch.zeros(b, 2, device=x.device)
    for i in range(b):
        dy, dx = sample_shift()
        t = int(target[i])
        x[i, t] = move(x[i, t], dy, dx)
        truth[i, 0], truth[i, 1] = -dy, -dx
    return x[..., MARGIN:-MARGIN, MARGIN:-MARGIN], target, truth


def loss_fn(out, truth, offsets):
    cell = torch.cdist(truth / 7.0, offsets).argmin(1)
    reg = F.smooth_l1_loss(out["shift"], truth) + 0.25 * F.smooth_l1_loss(out["coarse_shift"], truth)
    ce = F.cross_entropy(out["coarse"] * 10, cell)
    good = ((out["shift"].detach() - truth).abs().max(1)[0] <= 2).float()
    conf = F.binary_cross_entropy_with_logits(out["conf"], good)
    return reg + 0.5 * ce + 0.5 * conf, dict(reg=reg.item(), ce=ce.item(), conf=conf.item(), within2=good.mean().item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--dataset", default="tissuenet")
    p.add_argument("--snapshot", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
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
    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    model = CCA(a.snapshot)
    if a.init:
        model.load_state_dict(torch.load(os.path.join(a.init, "last.pt"), map_location="cpu", weights_only=True)["model"], strict=False)
    model = model.to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(i, a.iters) / a.iters)))
    data = Images(a.root, a.dataset)
    dl = torch.utils.data.DataLoader(data, a.batch, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "w", encoding="utf8")
    it, t0, stream = 0, time.time(), iter(dl)
    model.train()
    while it < a.iters:
        try:
            x = next(stream)
        except StopIteration:
            stream = iter(dl)
            x = next(stream)
        x, target, truth = scene(x.to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, target)
        out = {k: v.float() for k, v in out.items()}
        with torch.autocast("cuda", enabled=False):
            loss, parts = loss_fn(out, truth, model.offsets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iter {it}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        it += 1
        if it % 100 == 0 or it == a.iters:
            rec = dict(it=it, loss=loss.item(), sec=time.time() - t0, **parts)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save(dict(model=state, iters=a.iters, seed=a.seed, dataset=a.dataset), os.path.join(a.out, "last.pt"))


if __name__ == "__main__":
    main()
