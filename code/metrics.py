"""Post-processing and metrics. Works on numpy arrays at native image resolution."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed


def centroids(label):
    ids = np.unique(label)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return ids, np.zeros((0, 2))
    return ids, np.array(ndi.center_of_mass(np.ones_like(label), label, ids)).reshape(-1, 2)


def find_peaks(heat, thr, window=5):
    """heat: [K, H, W] probabilities. Returns points [N, 2] and class index [N]."""
    merged = heat.max(0)
    keep = (merged == ndi.maximum_filter(merged, size=window)) & (merged > thr)
    pts = np.argwhere(keep)
    cls = heat[:, pts[:, 0], pts[:, 1]].argmax(0) if len(pts) else np.zeros(0, int)
    return pts, cls


def decode(heat, fg, dist, heat_thr, fg_thr):
    """Seeded watershed. Returns instance map, peak points and peak classes."""
    pts, cls = find_peaks(heat, heat_thr)
    markers = np.zeros(fg.shape, np.int32)
    if len(pts) == 0:
        return markers, pts, cls
    markers[pts[:, 0], pts[:, 1]] = np.arange(1, len(pts) + 1)
    mask = fg > fg_thr
    mask[pts[:, 0], pts[:, 1]] = True
    return watershed(-dist, markers, mask=mask), pts, cls


def instance_counts(gt, pred, iou_thr=0.5):
    """TP, FP, FN at IoU > thr; with thr >= 0.5 each matched pair is unique."""
    gt_ids, pred_ids = np.unique(gt[gt > 0]), np.unique(pred[pred > 0])
    n_g, n_p = len(gt_ids), len(pred_ids)
    if n_g == 0 or n_p == 0:
        return 0, n_p, n_g
    g = np.searchsorted(gt_ids, gt) + 1
    g[gt == 0] = 0
    p = np.searchsorted(pred_ids, pred) + 1
    p[pred == 0] = 0
    joint = g.ravel().astype(np.int64) * (n_p + 1) + p.ravel()
    inter = np.bincount(joint, minlength=(n_g + 1) * (n_p + 1)).reshape(n_g + 1, n_p + 1)[1:, 1:]
    a_g = np.bincount(g.ravel(), minlength=n_g + 1)[1:]
    a_p = np.bincount(p.ravel(), minlength=n_p + 1)[1:]
    iou = inter / np.maximum(a_g[:, None] + a_p[None] - inter, 1)
    tp = int((iou > iou_thr).sum())
    return tp, n_p - tp, n_g - tp


def point_matches(a, b, radius):
    """Hungarian matching of points within radius. Returns matched index pairs."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    r, c = linear_sum_assignment(np.where(d <= radius, d, 1e6))
    ok = d[r, c] <= radius
    return r[ok], c[ok]


def image_record(gt_label, pred_label, pred_pts, radius=6.0, gt_class=None, pred_class=None):
    """All per-image numbers needed to recompute every metric under bootstrap.

    gt_class maps instance id to class index when classes exist; pred_class is per point.
    """
    tp, fp, fn = instance_counts(gt_label, pred_label)
    ids, gt_pts = centroids(gt_label)
    r, c = point_matches(gt_pts, pred_pts, radius)
    rec = dict(inst_tp=tp, inst_fp=fp, inst_fn=fn, det_tp=len(r), det_fp=len(pred_pts) - len(r),
               det_fn=len(gt_pts) - len(r), pred=len(pred_pts), gt=len(gt_pts))
    if gt_class is not None:
        g_cls = np.array([gt_class[i] for i in ids], int) if len(ids) else np.zeros(0, int)
        same = int((g_cls[r] == pred_class[c]).sum()) if len(r) else 0
        rec.update(cls_tp=same, cls_fp=len(pred_pts) - same, cls_fn=len(gt_pts) - same,
                   pred_pos=int((pred_class == 0).sum()), gt_pos=int((g_cls == 0).sum()))
    return rec


def summarise(records):
    """Micro-averaged metrics over a list of per-image records."""
    s = {k: sum(r[k] for r in records) for k in records[0]}

    def f1(t, p, n):
        return 2 * t / max(2 * t + p + n, 1)

    err = sum(abs(r["pred"] - r["gt"]) for r in records)
    out = dict(inst_f1=f1(s["inst_tp"], s["inst_fp"], s["inst_fn"]), det_f1=f1(s["det_tp"], s["det_fp"], s["det_fn"]),
               count_mae=err / len(records), count_rel=err / max(s["gt"], 1))
    if "cls_tp" in s:
        out["cls_f1"] = f1(s["cls_tp"], s["cls_fp"], s["cls_fn"])
        idx = [abs(r["pred_pos"] / max(r["pred"], 1) - r["gt_pos"] / max(r["gt"], 1)) for r in records if r["gt"] > 0]
        out["ki67_mae"] = float(np.mean(idx)) if idx else 0.0
    return out
