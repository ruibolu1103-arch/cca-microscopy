"""Channel degradations for multi-stream microscopy input.

All functions take a stream tensor x of shape [B, 3, H, W] with values in [0, 1], where every
stream was percentile-normalised beforehand. Randomness comes from an explicit generator so the
test conditions are reproducible per image.
"""
import math

import torch
import torch.nn.functional as F


def renorm(x):
    """Re-apply per-image percentile normalisation, as a real acquisition pipeline would."""
    b = x.shape[0]
    flat = x.reshape(b, -1)
    k = max(1, int(flat.shape[1] * 0.001))
    lo = flat.kthvalue(k, dim=1).values.view(b, 1, 1, 1)
    hi = flat.kthvalue(flat.shape[1] - k + 1, dim=1).values.view(b, 1, 1, 1)
    return ((x - lo) / (hi - lo).clamp_min(1e-3)).clamp(0, 1)


def missing(x, **_):
    return torch.zeros_like(x)


def low_photon(x, peak=10.0, read_noise=1.0, gen=None):
    """Poisson shot noise plus Gaussian read noise at the given peak photon count."""
    photons = torch.poisson(x * peak, generator=gen)
    photons = photons + read_noise * torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return renorm((photons / peak).clamp_min(0))


def _gauss_kernel(sigma, device, dtype):
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum(), radius


def defocus(x, sigma=4.0, **_):
    """Separable Gaussian point-spread function."""
    k, r = _gauss_kernel(sigma, x.device, x.dtype)
    c = x.shape[1]
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return renorm(y)


def regional(x, n=3, area=(0.04, 0.15), gen=None, **_):
    """Zero n random rectangles per image (uneven staining, debris, field-of-view loss).

    This is also the masking operator of MCRM in the source paper.
    """
    b, _, h, w = x.shape
    y = x.clone()
    u = torch.rand((b, n, 4), generator=gen).tolist()
    for i in range(b):
        for a_, r_, cx, cy in u[i]:
            s = (area[0] + (area[1] - area[0]) * a_) * h * w
            ar = math.exp(math.log(0.5) + (math.log(2.0) - math.log(0.5)) * r_)
            rh, rw = int(round(math.sqrt(s * ar))), int(round(math.sqrt(s / ar)))
            y0, x0 = int(cy * max(1, h - rh)), int(cx * max(1, w - rw))
            y[i, :, y0:y0 + rh, x0:x0 + rw] = 0.0
    return y


def crosstalk(x, other, alpha=0.5, **_):
    """Bleed-through: the stream receives a fraction of another stream's signal."""
    return renorm((1 - alpha) * x + alpha * other)


def misregister(x, shift=16, **_):
    """Translate the stream relative to the others; exposed borders are zero."""
    h, w = x.shape[-2:]
    s = min(shift, h - 1, w - 1)
    y = torch.zeros_like(x)
    y[..., s:, s:] = x[..., : h - s, : w - s]
    return y


# Name -> (function, kwargs). "seen" conditions are used by physics augmentation in training;
# crosstalk and misregistration stay held out for testing.
SINGLE_STREAM_CONDITIONS = {
    "missing": (missing, {}),
    "photon10": (low_photon, {"peak": 10.0}),
    "defocus4": (defocus, {"sigma": 4.0}),
    "regional": (regional, {}),
    "crosstalk": (crosstalk, {"alpha": 0.5}),
    "misreg16": (misregister, {"shift": 16}),
}
SEVERITY_CURVES = {
    "photon30": (low_photon, {"peak": 30.0}),
    "photon3": (low_photon, {"peak": 3.0}),
    "defocus2": (defocus, {"sigma": 2.0}),
    "defocus8": (defocus, {"sigma": 8.0}),
}


def apply_condition(streams, name, target, gen=None):
    """Apply a named test condition to stream `target` (or to all streams for 'all_photon10').

    streams: [B, M, 3, H, W].
    """
    out = streams.clone()
    m = streams.shape[1]
    if name == "clean":
        return out
    if name == "all_photon10":
        for k in range(m):
            out[:, k] = low_photon(streams[:, k], peak=10.0, gen=gen)
        return out
    fn, kw = {**SINGLE_STREAM_CONDITIONS, **SEVERITY_CURVES}[name]
    if fn is crosstalk:
        out[:, target] = crosstalk(streams[:, target], streams[:, (target + 1) % m], **kw)
    else:
        out[:, target] = fn(streams[:, target], gen=gen, **kw)
    return out


def augment_physics(streams, p=0.6):
    """Training augmentation: for each sample, with probability p degrade one random stream.

    Severity is sampled continuously so training does not contain the exact test settings.
    """
    b, m = streams.shape[:2]
    device = streams.device
    out = streams.clone()
    which = torch.randint(0, m, (b,), device=device)
    kinds = torch.randint(0, 4, (b,), device=device)
    active = torch.rand(b, device=device) < p
    for i in torch.nonzero(active).flatten().tolist():
        k = which[i].item()
        x = streams[i, k][None]
        kind = kinds[i].item()
        if kind == 0:
            y = missing(x)
        elif kind == 1:
            peak = math.exp(torch.empty(1).uniform_(math.log(2.0), math.log(60.0)).item())
            y = low_photon(x, peak=peak)
        elif kind == 2:
            y = defocus(x, sigma=torch.empty(1).uniform_(1.0, 10.0).item())
        else:
            y = regional(x)
        out[i, k] = y[0]
    return out


def augment_mcrm(streams, r=0.5):
    """MCRM of the source paper: mask local rectangles on one stream, balanced over streams."""
    b, m = streams.shape[:2]
    out = streams.clone()
    idx = torch.randperm(b)[: int(r * b)].tolist()
    for j, i in enumerate(idx):
        k = j % m
        out[i, k] = regional(streams[i, k][None])[0]
    return out


def augment_channel_dropout(streams, p=0.3):
    """Zero one whole stream per sample with probability p."""
    b, m = streams.shape[:2]
    out = streams.clone()
    for i in torch.nonzero(torch.rand(b) < p).flatten().tolist():
        out[i, torch.randint(0, m, (1,)).item()] = 0.0
    return out
