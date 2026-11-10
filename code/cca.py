"""Cross-channel alignment (CCA): estimate the translation that moves one stream onto its analysis stream.

Coarse step: correlation of learned projections of frozen DINOv2 tokens (stride 7 px, +-28 px). Fine step: correlation of a
small per-stream CNN on the coarsely warped image (stride 2 px, +-8 px). A confidence head decides later whether to apply it.
Convention follows reg_diag.py: an estimate (dy, dx) moves the moving stream onto its reference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import IMAGENET_MEAN, IMAGENET_STD, FrozenDino


def offsets(radius):
    return torch.tensor([(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)], dtype=torch.float32)


def correlate(ref, mov, radius):
    """Mean cosine similarity of [B, C, H, W] maps over the overlap for every integer offset; returns [B, K]."""
    h, w = ref.shape[-2:]
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            a = ref[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)]
            b = mov[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
            out.append((a * b).sum(1).mean((1, 2)))
    return torch.stack(out, 1)


def warp(x, d):
    """Move [B, C, H, W] by float shifts d [B, 2] (dy, dx) with zero borders."""
    b, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    gy = (ys[None] - d[:, 0, None, None]) * 2 / (h - 1) - 1
    gx = (xs[None] - d[:, 1, None, None]) * 2 / (w - 1) - 1
    return F.grid_sample(x, torch.stack([gx, gy], -1), mode="bilinear", padding_mode="zeros", align_corners=True)


class CCA(nn.Module):
    COARSE, FINE = 7.0, 2.0

    def __init__(self, snapshot, n_streams=2, dim=128, width=32, radius=4, scale=2):
        super().__init__()
        self.backbone = FrozenDino(snapshot)
        self.radius, self.scale = radius, scale
        self.proj = nn.ModuleList(nn.Sequential(nn.Linear(self.backbone.dim, 256), nn.GELU(), nn.Linear(256, dim)) for _ in range(n_streams))
        self.cnn = nn.ModuleList(
            nn.Sequential(nn.Conv2d(3, width, 5, padding=2), nn.GELU(), nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
                          nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.GELU(), nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.GELU(),
                          nn.Conv2d(width, width, 3, padding=1))
            for _ in range(n_streams))
        self.temp = nn.Parameter(torch.tensor([3.0, 3.0]))
        k = (2 * radius + 1) ** 2
        self.conf = nn.Sequential(nn.Linear(2 * (k + 3), 64), nn.GELU(), nn.Linear(64, 1))
        self.register_buffer("offsets", offsets(radius))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))

    @staticmethod
    def pick(features, target):
        """Split per-stream maps [B, C, H, W] into reference and moving maps for each sample."""
        mask = target.view(-1, 1, 1, 1).bool()
        return torch.where(mask, features[0], features[1]), torch.where(mask, features[1], features[0])

    def tokens(self, x):
        b, c, _, _ = x.shape
        y = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        with torch.no_grad():
            t = self.backbone.forward_plain(y)[:, 1:]
        h, w = y.shape[-2] // self.backbone.patch, y.shape[-1] // self.backbone.patch
        return t.reshape(b, h, w, -1)

    @staticmethod
    def stats(s):
        return torch.cat([s, s.max(1, keepdim=True)[0], s.max(1, keepdim=True)[0] - s.mean(1, keepdim=True), s.std(1, keepdim=True)], 1)

    def forward(self, streams, target):
        """streams [B, 2, 3, H, W] in 0..1, target [B] the stream to move. Returns the shift, confidence logit and coarse scores."""
        x = (streams - self.mean) / self.std
        features = [F.normalize(p(self.tokens(x[:, m])).permute(0, 3, 1, 2), dim=1) for m, p in enumerate(self.proj)]
        ref, mov = self.pick(features, target)
        coarse = correlate(ref, mov, self.radius)
        d0 = (coarse.mul(self.temp[0].exp()).softmax(1)[..., None] * self.offsets).sum(1) * self.COARSE
        moved = x.clone()
        i = torch.arange(len(x), device=x.device)
        moved[i, target] = warp(x[i, target], d0.detach())
        features = [F.normalize(c(moved[:, m]), dim=1) for m, c in enumerate(self.cnn)]
        ref, mov = self.pick(features, target)
        fine = correlate(ref, mov, self.radius)
        d1 = (fine.mul(self.temp[1].exp()).softmax(1)[..., None] * self.offsets).sum(1) * self.FINE
        return dict(shift=d0.detach() + d1, coarse_shift=d0, conf=self.conf(torch.cat([self.stats(coarse), self.stats(fine)], 1))[:, 0], coarse=coarse)
