"""Phase 6B prep patch (run once, zero-training):
1) evaluation/metrics.py  -> append evaluate_detection_persize (COCO-style
   size-stratified AP; additive, evaluate_detection untouched).
2) evaluation/evaluate_baseline.py -> hook persize into metrics.json
   (det_AP50_small/medium/large, det_AP5095_*, det_recall50_*, det_ngt_*).
Idempotent. Required by EXP-07 per-size AP / recall collection."""
import sys

PERSIZE = '''

# --- Phase 6B: COCO-convention size-stratified detection AP (additive) ---
_SIZE_NAMES = ("small", "medium", "large")
_SIZE_EDGES = ((0.0, 1024.0), (1024.0, 9216.0), (9216.0, float("inf")))


def evaluate_detection_persize(preds, gts):
    # Size-stratified AP50 / AP50-95 / recall at 640-input scale:
    # small <32^2, medium 32^2..96^2, large >=96^2 (GT box area).
    # Matching replicates evaluate_detection (greedy one-to-one, IoU 0.5).
    # TP dets inherit the bucket of their matched GT; unmatched dets count as
    # FP in every bucket (COCO convention); dets matched to another bucket's
    # GT are excluded from that bucket. Additive only.
    iouv = torch.linspace(0.5, 0.95, 10)
    acc = {n: {"correct": [], "conf": [], "n_gt": 0} for n in _SIZE_NAMES}
    for pred, gt in zip(preds, gts):
        areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) if len(gt) else torch.zeros(0)
        for name, (lo, hi) in zip(_SIZE_NAMES, _SIZE_EDGES):
            acc[name]["n_gt"] += int(((areas >= lo) & (areas < hi)).sum())
        if pred is None or len(pred) == 0:
            continue
        xyxy, conf, cls = pred
        n = len(gt)
        correct = torch.zeros(len(xyxy), len(iouv), dtype=torch.bool)
        matched = torch.full((len(xyxy),), -1, dtype=torch.long)
        if n:
            ious = box_iou(xyxy, gt)
            iou_max, match = ious.max(1)
            used = set()
            for j in (iou_max > iouv[0]).nonzero(as_tuple=False).flatten():
                t = int(match[j])
                if t not in used:
                    used.add(t)
                    correct[j] = iou_max[j] > iouv
                    matched[j] = t
                    if len(used) == n:
                        break
        marea = areas[matched.clamp(min=0)] if n else torch.zeros(0)
        for name, (lo, hi) in zip(_SIZE_NAMES, _SIZE_EDGES):
            tp_in = (matched >= 0) & (marea >= lo) & (marea < hi)
            idx = tp_in | (matched < 0)
            if idx.any():
                acc[name]["correct"].append(correct[idx])
                acc[name]["conf"].append(conf[idx])
    res = {}
    for name in _SIZE_NAMES:
        ngt = acc[name]["n_gt"]
        cl = acc[name]["correct"]
        if ngt == 0 or not cl or sum(c.shape[0] for c in cl) == 0:
            res[name] = {"AP50": 0.0, "AP5095": 0.0, "recall50": 0.0, "n_gt": ngt}
            continue
        correct = torch.cat(cl)
        conf = torch.cat(acc[name]["conf"])
        pred_cls = torch.zeros(correct.shape[0], dtype=torch.long)
        target_cls = torch.zeros(max(ngt, 1), dtype=torch.long)
        ap = ap_per_class(correct, conf, pred_cls, target_cls, iouv)
        res[name] = {"AP50": float(ap[:, 0].mean()), "AP5095": float(ap.mean()),
                     "recall50": float(correct[:, 0].sum()) / max(ngt, 1), "n_gt": ngt}
    return res
'''

ANCHOR = '''    if det_preds:
        m = evaluate_detection(det_preds, det_gts)
        metrics.update({"mAP50": round(m["mAP50"], 4), "mAP50_95": round(m["mAP50_95"], 4),
                        "det_n_gt": m["n_gt"], "det_n_pred": m["n_pred"]})
'''
HOOK = ANCHOR + '''        try:
            from evaluation.metrics import evaluate_detection_persize
            pm = evaluate_detection_persize(det_preds, det_gts)
            for _b in ("small", "medium", "large"):
                metrics["det_AP50_" + _b] = round(pm[_b]["AP50"], 4)
                metrics["det_AP5095_" + _b] = round(pm[_b]["AP5095"], 4)
                metrics["det_recall50_" + _b] = round(pm[_b]["recall50"], 4)
                metrics["det_ngt_" + _b] = pm[_b]["n_gt"]
        except Exception as _e:
            print(f"[persize] failed: {_e}")
'''

mp = "evaluation/metrics.py"
src = open(mp).read()
if "evaluate_detection_persize" in src:
    print("[patch] metrics.py already patched")
else:
    open(mp, "w").write(src + PERSIZE)
    print("[patch] metrics.py: appended evaluate_detection_persize")

eb = "evaluation/evaluate_baseline.py"
src = open(eb).read()
if "det_AP50_small" in src:
    print("[patch] evaluate_baseline.py already patched")
elif ANCHOR in src:
    open(eb, "w").write(src.replace(ANCHOR, HOOK))
    print("[patch] evaluate_baseline.py: hooked persize metrics")
else:
    print("[patch] ERROR: anchor not found in evaluate_baseline.py")
    sys.exit(1)
