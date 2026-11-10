"""TrustFuse and its controls.

A frozen DINOv2 ViT is shared by M streams. Trainable parts: cross-stream adapters, per-stage fusion
gates, a detail stream and a light decoder. Controls are the same class with switches, plus a U-Net.
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- frozen DINOv2 (HF weight layout)
class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q, self.k, self.v, self.proj = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = (f(x).reshape(b, n, self.heads, self.head_dim).transpose(1, 2) for f in (self.q, self.k, self.v))
        y = F.scaled_dot_product_attention(q, k, v)
        return self.proj(y.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.fc1, self.fc2 = nn.Linear(dim, 4 * dim), nn.Linear(4 * dim, dim)
        self.ls1, self.ls2 = nn.Parameter(torch.ones(dim)), nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        return x + self.ls2 * self.fc2(F.gelu(self.fc1(self.norm2(x))))


class FrozenDino(nn.Module):
    """DINOv2 ViT loaded from a Hugging Face snapshot directory (config.json + model.safetensors)."""

    def __init__(self, snapshot):
        super().__init__()
        from safetensors.torch import load_file

        cfg = json.load(open(os.path.join(snapshot, "config.json")))
        self.dim, self.patch = cfg["hidden_size"], cfg["patch_size"]
        n, heads = cfg["num_hidden_layers"], cfg["num_attention_heads"]
        self.patch_proj = nn.Conv2d(3, self.dim, self.patch, self.patch)
        self.blocks = nn.ModuleList(Block(self.dim, heads) for _ in range(n))
        self.layernorm = nn.LayerNorm(self.dim, eps=1e-6)
        sd = load_file(os.path.join(snapshot, "model.safetensors"))
        self.cls_token = nn.Parameter(sd["embeddings.cls_token"])
        self.pos_embed = nn.Parameter(sd["embeddings.position_embeddings"])
        self.patch_proj.load_state_dict({
            "weight": sd["embeddings.patch_embeddings.projection.weight"],
            "bias": sd["embeddings.patch_embeddings.projection.bias"],
        })
        self.layernorm.load_state_dict({"weight": sd["layernorm.weight"], "bias": sd["layernorm.bias"]})
        pairs = {
            "norm1": "norm1", "norm2": "norm2", "attn.q": "attention.attention.query", "attn.k": "attention.attention.key",
            "attn.v": "attention.attention.value", "attn.proj": "attention.output.dense", "fc1": "mlp.fc1", "fc2": "mlp.fc2",
        }
        for i, blk in enumerate(self.blocks):
            p = f"encoder.layer.{i}."
            own = {}
            for mine, theirs in pairs.items():
                for w in ("weight", "bias"):
                    own[f"{mine}.{w}"] = sd[f"{p}{theirs}.{w}"]
            own["ls1"], own["ls2"] = sd[p + "layer_scale1.lambda1"], sd[p + "layer_scale2.lambda1"]
            blk.load_state_dict(own)
        for q in self.parameters():
            q.requires_grad_(False)
        self.eval()
        self.taps = [n // 4, n // 2, 3 * n // 4, n]

    def train(self, mode=True):
        return super().train(False)  # always in eval mode

    def embed(self, x, proj=None):
        h, w = x.shape[-2:]
        t = (proj or self.patch_proj)(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(t.shape[0], -1, -1)
        side = int(math.sqrt(self.pos_embed.shape[1] - 1))
        grid = self.pos_embed[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)
        size = (h // self.patch, w // self.patch)
        if size != (side, side):
            grid = F.interpolate(grid.float(), size=size, mode="bicubic", align_corners=False)
        pos = grid.permute(0, 2, 3, 1).reshape(1, size[0] * size[1], -1)
        return torch.cat([cls + self.pos_embed[:, :1], t + pos.to(t.dtype)], dim=1)

    def forward_plain(self, x):
        """Plain forward, used only to self-check agreement with the reference implementation."""
        t = self.embed(x)
        for blk in self.blocks:
            t = blk(t)
        return self.layernorm(t)


# ---------------------------------------------------------------- cross-stream adapter
class CrossStreamAdapter(nn.Module):
    """Prompt-injected adapter (structure after CPIA of the source paper, for M streams)."""

    def __init__(self, dim, n_streams, rp=0.25, ra=0.25, drop=0.1):
        super().__init__()
        cp, d = int(dim * rp), int(dim * ra)
        self.down_p = nn.Linear(dim, cp)
        self.fuse = nn.Linear(cp, cp)
        self.up_p = nn.Linear(cp, dim)
        self.gamma = nn.Parameter(torch.zeros(n_streams, dim))
        self.beta = nn.Parameter(torch.zeros(n_streams, dim))
        self.w_down, self.w_prompt = nn.Linear(dim, d), nn.Linear(dim, d)
        self.w_up = nn.Linear(d, dim)
        self.drop = nn.Dropout(drop)
        nn.init.zeros_(self.w_up.weight)  # starts as identity so the frozen features are untouched
        nn.init.zeros_(self.w_up.bias)

    def forward(self, tokens, weight=None):
        # tokens: [B, M, N, C]; weight: [B, M, N, 1] sums to 1 over M, or None for a uniform mean.
        p = self.down_p(tokens)
        z = p.mean(1) if weight is None else (p * weight).sum(1)
        z = self.up_p(F.relu(self.fuse(z)))  # [B, N, C]
        out = []
        for m in range(tokens.shape[1]):
            prompt = z + z * self.gamma[m] + self.beta[m]
            y = self.w_up(self.drop(F.relu(self.w_down(tokens[:, m]) + self.w_prompt(prompt))))
            out.append(tokens[:, m] + y)
        return torch.stack(out, 1)


# ---------------------------------------------------------------- fusion gates
class TrustGate(nn.Module):
    """Spatial trust rho (supervised by branch errors) plus a per-feature residual delta."""

    def __init__(self, dim, reduced=64):
        super().__init__()
        self.phi = nn.Conv2d(dim, reduced, 1)
        self.g = nn.Sequential(nn.Conv2d(3 * reduced, reduced, 1), nn.GroupNorm(8, reduced), nn.GELU())
        self.rho = nn.Conv2d(reduced, 1, 1)
        self.delta = nn.Conv2d(reduced, dim, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, feats):
        # feats: [B, M, C, H, W]
        b, m, c, h, w = feats.shape
        r = self.phi(feats.flatten(0, 1)).reshape(b, m, -1, h, w)
        mean = r.mean(1, keepdim=True)
        x = torch.cat([r, mean.expand_as(r), (r - mean).abs()], 2)
        hid = self.g(x.flatten(0, 1))
        rho = self.rho(hid).reshape(b, m, 1, h, w)
        delta = self.delta(hid).reshape(b, m, c, h, w)
        gate = F.softmax(rho + delta, dim=1)
        return (gate * feats).sum(1), rho


class DGFM(nn.Module):
    """Difference-guided gated fusion of the source paper; folded pairwise when M > 2."""

    def __init__(self, dim, reduced=64):
        super().__init__()
        self.px, self.py = nn.Conv2d(dim, reduced, 1), nn.Conv2d(dim, reduced, 1)
        self.g = nn.Sequential(nn.Conv2d(3 * reduced, reduced, 1), nn.GroupNorm(8, reduced), nn.GELU(), nn.Conv2d(reduced, dim, 1))

    def forward(self, feats):
        f = feats[:, 0]
        for m in range(1, feats.shape[1]):
            y = feats[:, m]
            rx, ry = self.px(f), self.py(y)
            gate = torch.sigmoid(self.g(torch.cat([rx, ry, (rx - ry).abs()], 1)))
            f = gate * f + (1 - gate) * y
        return f, None


class MeanFusion(nn.Module):
    def forward(self, feats):
        return feats.mean(1), None


# ---------------------------------------------------------------- decoder and heads
def conv_block(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU(), nn.Conv2d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU())


class Heads(nn.Module):
    def __init__(self, ch, n_classes):
        super().__init__()
        self.heat, self.fg, self.dist = nn.Conv2d(ch, n_classes, 1), nn.Conv2d(ch, 1, 1), nn.Conv2d(ch, 1, 1)
        nn.init.constant_(self.heat.bias, -4.6)  # prior of about 1% positives for focal loss

    def forward(self, x):
        return {"heat": self.heat(x), "fg": self.fg(x), "dist": self.dist(x)}


class UPerDecoder(nn.Module):
    """Pyramid pooling on the last stage, top-down fusion, then learned upsampling to 4x the token grid."""

    def __init__(self, ch, out_ch, pools=(1, 2, 3, 6)):
        super().__init__()
        def conv(a, b, k=3):
            return nn.Sequential(nn.Conv2d(a, b, k, padding=k // 2), nn.GroupNorm(8, b), nn.GELU())
        self.ppm = nn.ModuleList(nn.Sequential(nn.AdaptiveAvgPool2d(s), conv(ch, ch, 1)) for s in pools)
        self.ppm_out = conv(ch * (1 + len(pools)), ch)
        self.fpn = nn.ModuleList(conv(ch, ch) for _ in range(3))
        self.bottleneck = conv(4 * ch, ch)
        self.refine1, self.refine2 = conv_block(ch, ch), conv_block(ch, out_ch)

    def forward(self, p):
        size = p[3].shape[-2:]
        top = self.ppm_out(torch.cat([p[3]] + [F.interpolate(m(p[3]), size=size, mode="bilinear", align_corners=False) for m in self.ppm], 1))
        down, outs = top, [top]
        for i in (2, 1, 0):  # all ViT stages share one resolution, so the top-down sum needs no resizing
            down = p[i] + down
            outs.append(self.fpn[i](down))
        y = self.bottleneck(torch.cat(outs, 1))
        y = self.refine1(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))
        return self.refine2(F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False))


class CrossAlign(nn.Module):
    """Global cross-attention: analysis-stream tokens fetch the other stream's content wherever it sits. End-to-end control for CCA."""

    def __init__(self, dim, heads=8, inner=384):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = nn.Linear(dim, inner), nn.Linear(dim, inner), nn.Linear(dim, inner)
        self.out = nn.Linear(inner, dim)

    def split(self, t):
        return t.reshape(*t.shape[:2], self.heads, -1).transpose(1, 2)

    def forward(self, feat, analysis):
        b, _, c, h, w = feat.shape
        i = torch.arange(b, device=feat.device)
        ref, other = feat[i, analysis].flatten(2).transpose(1, 2), feat[i, 1 - analysis].flatten(2).transpose(1, 2)
        y = F.scaled_dot_product_attention(self.split(self.q(ref)), self.split(self.k(other)), self.split(self.v(other)))
        y = self.out(y.transpose(1, 2).flatten(2)).transpose(1, 2).reshape(b, c, h, w)
        feat = feat.clone()
        feat[i, 1 - analysis] = y.to(feat.dtype)
        return feat


class TrustFuse(nn.Module):
    def __init__(self, snapshot, n_streams=2, n_classes=1, fusion="trust", adapter=True, detail=True, dec=128, scale=2,
                 decoder="light", extra_patch=False, analysis=False, cross=False):
        super().__init__()
        self.backbone = FrozenDino(snapshot)
        self.scale, self.n_streams = scale, n_streams
        dim = self.backbone.dim
        self.extra_patch = None
        if extra_patch:  # streams after the first get their own trainable patch embedding
            self.extra_patch = nn.Conv2d(3, dim, self.backbone.patch, self.backbone.patch)
            self.extra_patch.load_state_dict(self.backbone.patch_proj.state_dict())
        self.fusion_kind = fusion
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.stream_embed = nn.Parameter(torch.zeros(n_streams, dim))
        nn.init.trunc_normal_(self.stream_embed, std=0.02)
        self.analysis_embed = nn.Parameter(torch.zeros(dim)) if analysis else None  # marks the stream that defines the coordinate frame
        self.adapters = nn.ModuleList(CrossStreamAdapter(dim, n_streams) for _ in range(4)) if adapter else None
        make = {"trust": lambda: TrustGate(dim), "dgfm": lambda: DGFM(dim), "mean": lambda: MeanFusion()}[fusion]
        self.fusers = nn.ModuleList(make() for _ in range(4))
        self.cross = nn.ModuleList(CrossAlign(dim) for _ in range(4)) if cross else None
        self.proj = nn.ModuleList(nn.Conv2d(dim, dec, 1) for _ in range(4))
        self.decoder_kind = decoder
        if decoder == "upernet":
            self.uper = UPerDecoder(dec, dec // 2)
        else:
            self.up1, self.up2, self.up3 = conv_block(2 * dec, dec), conv_block(2 * dec, dec), conv_block(2 * dec, dec // 2)
        self.use_detail = detail
        out_ch = dec // 2
        if detail:
            self.det1 = conv_block(3, 16)
            self.det2 = nn.Sequential(nn.AvgPool2d(2), conv_block(16, 32))
            self.mix2, self.mix1 = conv_block(dec // 2 + 32, dec // 2), conv_block(dec // 2 + 16, dec // 2)
        self.heads = Heads(out_ch, n_classes)
        self.aux_head = nn.Sequential(nn.Conv2d(dim, 64, 1), nn.GELU(), nn.Conv2d(64, n_classes + 1, 1))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _run_stage(self, t, s):
        a, b = ([0] + self.backbone.taps)[s], self.backbone.taps[s]
        def fn(x):
            for blk in self.backbone.blocks[a:b]:
                x = blk(x)
            return x
        use_ckpt = os.environ.get("VIT_CHECKPOINT", "0") == "1"  # only needed on small GPUs
        return ckpt.checkpoint(fn, t, use_reentrant=False) if use_ckpt and self.training and t.requires_grad else fn(t)

    def forward(self, streams, analysis=None):
        # streams: [B, M, 3, H, W], values in [0, 1]; H, W multiples of 7 at scale 2. analysis: [B] frame-defining stream.
        b, m, c, h, w = streams.shape
        x = (streams.flatten(0, 1) - self.mean) / self.std
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        if self.extra_patch is None:
            t = self.backbone.embed(x)  # [B*M, 1+N, C]
        else:
            per_stream = x.reshape(b, m, *x.shape[1:])
            t = torch.cat([self.backbone.embed(per_stream[:, k], None if k == 0 else self.extra_patch) for k in range(m)], 0)
            t = t.reshape(m, b, *t.shape[1:]).transpose(0, 1).flatten(0, 1)  # same order as the shared path
        n, dim = t.shape[1], t.shape[2]
        gh, gw = x.shape[-2] // self.backbone.patch, x.shape[-1] // self.backbone.patch
        t = t.reshape(b, m, n, dim) + self.stream_embed[:, None]
        if self.analysis_embed is not None and analysis is not None:
            t = t.clone()
            t[torch.arange(b, device=t.device), analysis] += self.analysis_embed
        fused, rhos, weight, last = [], [], None, None
        for s in range(4):
            if self.adapters is not None:
                t = self.adapters[s](t, weight)
            t = self._run_stage(t.flatten(0, 1), s).reshape(b, m, n, dim)
            feat = self.backbone.layernorm(t[:, :, 1:]).transpose(2, 3).reshape(b, m, dim, gh, gw)
            if self.cross is not None:
                feat = self.cross[s](feat, analysis)
            f, rho = self.fusers[s](feat)
            fused.append(f)
            rhos.append(rho)
            last = feat
            if rho is not None:
                w_ = F.softmax(rho.detach(), 1).flatten(3).transpose(2, 3)  # [B, M, N-1, 1]
                weight = torch.cat([w_.mean(2, keepdim=True), w_], 2)
        p = [self.proj[i](fused[i]) for i in range(4)]
        if self.decoder_kind == "upernet":
            y = self.uper(p)
        else:
            y = self.up1(torch.cat([p[3], p[2]], 1))
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = self.up2(torch.cat([y, F.interpolate(p[1], size=y.shape[-2:], mode="bilinear", align_corners=False)], 1))
            y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
            y = self.up3(torch.cat([y, F.interpolate(p[0], size=y.shape[-2:], mode="bilinear", align_corners=False)], 1))
        if self.use_detail:
            flat = streams.flatten(0, 1)
            d1 = self.det1(flat)
            d2 = self.det2(d1)
            if self.cross is not None:  # raw pixels of the other stream are not aligned, so details come from the analysis stream
                wgt = F.one_hot(analysis, m).to(x.dtype).view(b, m, 1, 1, 1).expand(b, m, 1, gh, gw)
            elif rhos[-1] is not None:
                wgt = F.softmax(rhos[-1], 1)
            else:
                wgt = torch.full((b, m, 1, gh, gw), 1.0 / m, device=x.device, dtype=x.dtype)
            def merge(d):
                sz = d.shape[-2:]
                ww = F.interpolate(wgt.flatten(0, 1), size=sz, mode="bilinear", align_corners=False).reshape(b, m, 1, *sz)
                return (d.reshape(b, m, -1, *sz) * ww).sum(1)
            y = F.interpolate(y, size=d2.shape[-2:], mode="bilinear", align_corners=False)
            y = self.mix2(torch.cat([y, merge(d2)], 1))
            y = F.interpolate(y, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            y = self.mix1(torch.cat([y, merge(d1)], 1))
        else:
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        out = self.heads(y)
        if self.training:
            aux = self.aux_head(last.flatten(0, 1)).reshape(b, m, -1, gh, gw)
            out.update(aux=aux, rhos=rhos)
        return out


# ---------------------------------------------------------------- fully trainable control
class UNet(nn.Module):
    """Standard U-Net with early channel concatenation; every parameter is trainable."""

    def __init__(self, in_ch, n_classes=1, base=32):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]
        self.enc = nn.ModuleList([conv_block(in_ch, ch[0])] + [conv_block(ch[i], ch[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList(conv_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(4)))
        self.heads = Heads(ch[0], n_classes)

    def trainable_parameters(self):
        return list(self.parameters())

    def forward(self, streams, analysis=None):
        b, m, c, h, w = streams.shape
        x = streams.reshape(b, m * c, h, w)
        if analysis is not None:  # one constant map per stream marks the frame-defining stream
            x = torch.cat([x, F.one_hot(analysis, m).to(x.dtype).view(b, m, 1, 1).expand(b, m, h, w)], 1)
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for d, s in zip(self.dec, reversed(skips[:-1])):
            x = d(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return self.heads(x)


CONFIGS = {
    # name: model kwargs, augmentation, trust supervision, auxiliary-loss mode
    "unet_none": dict(model="unet", aug="none"),
    "unet_drop": dict(model="unet", aug="dropout"),
    "unet_phys": dict(model="unet", aug="physics"),
    "base_phys": dict(model="vit", fusion="mean", adapter=False, detail=True, aug="physics"),
    "moba": dict(model="vit", fusion="dgfm", adapter=True, detail=False, aug="mcrm", aux="hard"),
    "full": dict(model="vit", fusion="trust", adapter=True, detail=True, aug="physics", aux="all", trust=True),
    "abl_noada": dict(model="vit", fusion="trust", adapter=False, detail=True, aug="physics", aux="all", trust=True),
    "abl_nosup": dict(model="vit", fusion="trust", adapter=True, detail=True, aug="physics", aux="all", trust=False),
    "abl_mcrm": dict(model="vit", fusion="trust", adapter=True, detail=True, aug="mcrm", aux="all", trust=True),
    "abl_nodetail": dict(model="vit", fusion="trust", adapter=True, detail=False, aug="physics", aux="all", trust=True),
    # Exploratory, added after the core gate: TrustGate replaced by a plain mean, everything else as in full.
    # Control for the alignment module: end-to-end misalignment training, told which stream defines the frame.
    "full_mis": dict(model="vit", fusion="trust", adapter=True, detail=True, aug="physics", aux="all", trust=True, mis=True, analysis=True),
    # Stronger end-to-end controls for criterion B3: global cross-stream attention, and a fully trainable CNN.
    "full_xa": dict(model="vit", fusion="trust", adapter=True, detail=True, aug="physics", aux="all", trust=True, mis=True, analysis=True, cross=True),
    "unet_mis": dict(model="unet", aug="physics", mis=True, analysis=True),
    "abl_mean": dict(model="vit", fusion="mean", adapter=True, detail=True, aug="physics", aux="all", trust=False),
    # Exploratory, added after the core gate: run with a ViT-L snapshot and more training.
    "strong": dict(model="vit", fusion="trust", adapter=True, detail=True, aug="physics", aux="all", trust=True,
                   decoder="upernet", extra_patch=True),
}


def build(name, snapshot, n_streams, n_classes):
    cfg = CONFIGS[name]
    if cfg["model"] == "unet":
        return UNet(n_streams * 3 + (n_streams if cfg.get("analysis") else 0), n_classes), cfg
    return TrustFuse(snapshot, n_streams, n_classes, cfg["fusion"], cfg["adapter"], cfg["detail"],
                     decoder=cfg.get("decoder", "light"), extra_patch=cfg.get("extra_patch", False), analysis=cfg.get("analysis", False), cross=cfg.get("cross", False)), cfg
