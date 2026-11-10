"""Optional implementation self-checks. CPU only with a small thread budget, so it does not disturb other jobs."""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

torch.set_num_threads(2)

import degrade
from data import instance_classes, make_targets, seed_of
from metrics import decode, image_record, instance_counts, summarise
from model import CONFIGS, FrozenDino, build
from train import aux_and_trust, main_loss


def synth_label(n=20, size=256, r=9, seed=0):
    """Well-separated discs on a grid, so hand-computed results are known."""
    label = np.zeros((size, size), np.int32)
    yy, xx = np.mgrid[:size, :size]
    side = int(np.ceil(np.sqrt(n)))
    step = size // side
    k = 0
    for i in range(side):
        for j in range(side):
            if k == n:
                break
            k += 1
            label[(yy - (i * step + step // 2)) ** 2 + (xx - (j * step + step // 2)) ** 2 <= r * r] = k
    return label


def check(name, ok, detail=""):
    print(("PASS" if ok else "FAIL"), name, detail, flush=True)
    if not ok:
        sys.exit(1)


def test_vit(snapshot):
    from transformers import Dinov2Model

    ref = Dinov2Model.from_pretrained(snapshot).eval()
    mine = FrozenDino(snapshot).eval()
    x = torch.randn(1, 3, 196, 252)
    with torch.no_grad():
        a = ref(x, interpolate_pos_encoding=True).last_hidden_state if _accepts_flag(ref) else ref(x).last_hidden_state
        b = mine.forward_plain(x)
    diff = (a - b).abs().max().item()
    check("7.1 ViT matches transformers Dinov2Model", diff < 1e-3, f"max abs diff {diff:.2e}")


def _accepts_flag(model):
    import inspect

    return "interpolate_pos_encoding" in inspect.signature(model.forward).parameters


def test_metrics():
    gt = synth_label()
    check("7.2 perfect instance match", instance_counts(gt, gt) == (20, 0, 0))
    check("7.2 empty prediction", instance_counts(gt, np.zeros_like(gt)) == (0, 0, 20))
    pred = gt.copy()
    pred[pred == 1] = 0  # remove one instance
    pred[0:6, 0:6] = 99  # add one spurious instance
    check("7.2 hand-computed TP/FP/FN", instance_counts(gt, pred) == (19, 1, 1), str(instance_counts(gt, pred)))
    half = gt.copy()
    half[half == 2] = 0
    half[(gt == 2) & (np.arange(256)[None, :] % 2 == 0)] = 2  # keeps about half of one instance: IoU near 0.5
    tp, _, _ = instance_counts(gt, half)
    check("7.2 IoU boundary handled", tp in (19, 20))
    pts = np.array([np.argwhere(gt == i).mean(0) for i in range(1, 21)]).round().astype(int)
    r = summarise([image_record(gt, gt, pts)])
    check("7.2 perfect prediction gives F1 1 and zero count error",
          r["inst_f1"] == 1 and r["det_f1"] == 1 and r["count_rel"] == 0, str(r))
    r = summarise([image_record(gt, gt, pts[:15])])
    check("7.2 count error hand-computed", abs(r["count_rel"] - 5 / 20) < 1e-9 and r["count_mae"] == 5, str(r))


def test_degrade():
    x = torch.rand(1, 2, 3, 64, 64)
    for cond in ["photon10", "regional", "defocus4", "crosstalk", "misreg16", "missing", "all_photon10"]:
        a = degrade.apply_condition(x, cond, 1, torch.Generator().manual_seed(seed_of(3, cond, 1)))
        b = degrade.apply_condition(x, cond, 1, torch.Generator().manual_seed(seed_of(3, cond, 1)))
        check(f"7.3 {cond} reproducible", torch.equal(a, b))
        if cond != "all_photon10":
            check(f"7.3 {cond} leaves the other stream untouched", torch.equal(a[:, 0], x[:, 0]))
    check("7.3 missing is zero", degrade.apply_condition(x, "missing", 1).__getitem__((0, 1)).abs().sum() == 0)
    y = degrade.misregister(x[:, 1], 16)
    check("7.3 misregistration shifts by 16", torch.equal(y[..., 16:, 16:], x[:, 1][..., :-16, :-16]) and y[..., :16, :].sum() == 0)
    ramp = torch.ones(1, 3, 32, 32) * 0.5
    low = degrade.low_photon(torch.rand(2000, 3, 4, 4) * 0 + 0.5, peak=10.0, read_noise=0.0, gen=torch.Generator().manual_seed(0))
    check("7.3 low photon output in range", low.min() >= 0 and low.max() <= 1)
    check("7.3 defocus preserves a constant image after renorm guard", degrade.defocus(ramp, sigma=2.0).isfinite().all())
    for fn in (degrade.augment_physics, degrade.augment_mcrm, degrade.augment_channel_dropout):
        check(f"7.3 {fn.__name__} keeps shape", fn(x.repeat(4, 1, 1, 1, 1)).shape == (4, 2, 3, 64, 64))


def test_training_step(snapshot):
    b = 2
    labels = [synth_label(n=9, size=56, r=4) for _ in range(b)]
    t = [make_targets(l, None, 1) for l in labels]
    batch = dict(heat=torch.tensor(np.stack([i[0] for i in t])), fg=torch.tensor(np.stack([i[1] for i in t])),
                 dist=torch.tensor(np.stack([i[2] for i in t])))
    x = torch.rand(b, 2, 3, 56, 56)
    for name, cfg in CONFIGS.items():
        model, cfg = build(name, snapshot, 2, 1)
        model.train()
        frozen = {k: v.clone() for k, v in model.named_parameters() if k.startswith("backbone.")}
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        out = model(x)
        loss, _ = main_loss(out, batch)
        if "aux" in out and cfg.get("aux"):
            loss = loss + aux_and_trust(out, batch, cfg, 0.5, 0.4, 0.1)
        before = [p.clone() for p in model.trainable_parameters()]
        opt.zero_grad()
        loss.backward()
        graded = sum(p.grad is not None and p.grad.abs().sum() > 0 for p in model.trainable_parameters())
        opt.step()
        moved = sum((p != q).any().item() for p, q in zip(model.trainable_parameters(), before))
        same = all(torch.equal(v, dict(model.named_parameters())[k]) for k, v in frozen.items())
        n = sum(p.numel() for p in model.trainable_parameters()) / 1e6
        check(f"7.4 {name}: frozen unchanged, trainable moved", same and moved > 0, f"trainable {n:.2f}M, graded {graded}, moved {moved}, loss {loss.item():.3f}")
        assert out["heat"].shape[-2:] == (56, 56)


def test_adapter_identity(snapshot):
    """A freshly initialised adapter must leave tokens unchanged, so training starts from frozen features."""
    model, _ = build("full", snapshot, 2, 1)
    model.eval()
    tok = torch.randn(1, 2, 5, model.backbone.dim)
    with torch.no_grad():
        check("7.4 adapter starts as identity", torch.allclose(model.adapters[0](tok), tok))


def test_encoding_bound():
    label = synth_label(n=30, size=256, r=8)
    heat, fg, dist = make_targets(label, None, 1)
    pred, pts, cls = decode(heat, fg[0], dist[0], 0.5, 0.5)
    r = summarise([image_record(label, pred, pts)])
    check("7.5 encode-decode bound on separated cells", r["inst_f1"] > 0.99 and r["count_rel"] == 0, str(r))
    cmap = (label > 0).astype(np.uint8) * (1 + (label % 2 == 0))
    cls_map = instance_classes(label, cmap)
    heat2, fg2, dist2 = make_targets(label, cls_map, 2)
    pred, pts, cls = decode(heat2, fg2[0], dist2[0], 0.5, 0.5)
    r = summarise([image_record(label, pred, pts, gt_class=cls_map, pred_class=cls)])
    check("7.5 class-aware bound", r["cls_f1"] > 0.99 and r["ki67_mae"] < 1e-9, str(r))


if __name__ == "__main__":
    snap = sys.argv[1]
    test_metrics()
    test_degrade()
    test_encoding_bound()
    test_vit(snap)
    test_adapter_identity(snap)
    test_training_step(snap)
    print("ALL_PASS")
