"""Unified evaluation metrics for the three tasks.

Metric definitions match the official light-model baselines
(YOLOP / TriLiteNet validation code):

- Detection: COCO-style mAP@0.5 and mAP@0.5:0.95 via precision-recall AP per
  class (continuous max-precision interpolation), single class ('car').
- Drivable Area / Lane: pixel accuracy, foreground IoU and mIoU (mean over the
  2 classes bg/fg) from a binary confusion matrix; lane also reports
  lineAccuracy = (sensitivity + specificity) / 2 (TwinLiteNet/TriLiteNet style).
"""
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Detection: AP / mAP
# ---------------------------------------------------------------------------
def compute_ap(recall, precision):
    """VOC/COCO-style AP: max-precision interpolation over 101 recall levels."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ap = 0.0
    for i in range(mrec.size - 1):
        ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def ap_per_class(tp, conf, pred_cls, target_cls, iouv):
    """Per-class AP over an IoU-threshold vector. tp: (n, len(iouv)) bool."""
    tp = tp.cpu().numpy()
    conf = conf.cpu().numpy()
    pred_cls = pred_cls.cpu().numpy().astype(int)
    target_cls = np.asarray(target_cls, dtype=int)

    unique = np.unique(target_cls)
    ap = np.zeros((len(unique), len(iouv)))
    for ci, cls in enumerate(unique):
        pm = pred_cls == cls
        tm = target_cls == cls
        npos = tm.sum()
        if npos == 0:
            continue
        order = np.argsort(-conf[pm])
        cp = tp[pm][order]
        conf_c = conf[pm][order]
        tp_c = np.cumsum(cp, axis=0)
        fp_c = np.cumsum(1 - cp, axis=0)
        for j in range(len(iouv)):
            rec = tp_c[:, j] / (npos + 1e-16)
            prec = tp_c[:, j] / np.maximum(tp_c[:, j] + fp_c[:, j], 1e-16)
            ap[ci, j] = compute_ap(rec, prec)
    return ap  # (n_class, n_iou)


def box_iou(boxes1, boxes2):
    """IoU of xyxy boxes. boxes1: (N,4), boxes2: (M,4) -> (N,M)."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-16)


def evaluate_detection(preds, gts, iou_thres=0.6):
    """preds: list of (xyxy_tensor, conf_tensor, cls_tensor) per image (already NMS'd).
    gts: list of xyxy tensors per image (single class). Returns dict with mAP50, mAP50-95."""
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []  # per-image (correct, conf, pred_cls, tcls)
    for pred, gt in zip(preds, gts):
        if pred is None or len(pred) == 0:
            stats.append((torch.zeros(0, len(iouv), dtype=torch.bool),
                          torch.zeros(0), torch.zeros(0, dtype=torch.long),
                          torch.zeros(0, dtype=torch.long)))
            continue
        xyxy, conf, cls = pred
        n = len(gt)
        correct = torch.zeros(len(xyxy), len(iouv), dtype=torch.bool)
        if n:
            ious = box_iou(xyxy, gt)
            iou_max, match = ious.max(1)
            used = set()
            for j in (iou_max > iouv[0]).nonzero(as_tuple=False).flatten():
                t = int(match[j])
                if t not in used:
                    used.add(t)
                    correct[j] = iou_max[j] > iouv
                    if len(used) == n:
                        break
        stats.append((correct, conf, cls, torch.zeros(n, dtype=torch.long)))
    if not stats or all(s[0].shape[0] == 0 for s in stats):
        return {"mAP50": 0.0, "mAP50_95": 0.0, "n_gt": int(sum(len(g) for g in gts))}
    correct = torch.cat([s[0] for s in stats])
    conf = torch.cat([s[1] for s in stats])
    pred_cls = torch.cat([s[2] for s in stats])
    target_cls = torch.cat([s[3] for s in stats])
    ap = ap_per_class(correct, conf, pred_cls, target_cls, iouv)
    return {
        "mAP50": float(ap[:, 0].mean()),
        "mAP50_95": float(ap.mean()),
        "n_gt": int(len(target_cls)),
        "n_pred": int(len(conf)),
    }


# ---------------------------------------------------------------------------
# Segmentation: pixel Acc / fg IoU / mIoU / lineAccuracy
# ---------------------------------------------------------------------------
class SegmentationMetric:
    """Binary-class confusion-matrix metrics, identical to TriLiteNet's
    SegmentationMetric(numClass=2)."""

    def __init__(self, num_class=2):
        self.num_class = num_class
        self.conf = np.zeros((num_class, num_class), dtype=np.int64)

    def reset(self):
        self.conf *= 0

    def add_batch(self, pred, gt):
        mask = (gt >= 0) & (gt < self.num_class)
        flat = self.num_class * gt[mask].astype(np.int64) + pred[mask].astype(np.int64)
        self.conf += np.bincount(flat, minlength=self.num_class ** 2).reshape(
            self.num_class, self.num_class)

    def pixel_accuracy(self):
        return np.diag(self.conf).sum() / max(self.conf.sum(), 1)

    def per_class_iou(self):
        inter = np.diag(self.conf)
        union = self.conf.sum(1) + self.conf.sum(0) - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        return iou

    def fg_iou(self):
        return float(self.per_class_iou()[1])

    def m_iou(self):
        return float(np.mean(self.per_class_iou()))

    def sensitivity(self):
        tp, fn = self.conf[1, 1], self.conf[1, 0]
        return tp / max(tp + fn, 1)

    def specificity(self):
        tn, fp = self.conf[0, 0], self.conf[0, 1]
        return tn / max(tn + fp, 1)

    def line_accuracy(self):
        """TwinLiteNet/TriLiteNet lane metric: (sens + spec) / 2."""
        return (self.sensitivity() + self.specificity()) / 2


def evaluate_segmentation(pred_maps, gt_maps, lane=False):
    """pred_maps, gt_maps: lists of binary (H,W) numpy arrays (already pad-cropped)."""
    metric = SegmentationMetric(2)
    for p, g in zip(pred_maps, gt_maps):
        metric.add_batch(p, g)
    res = {
        "pixel_acc": float(metric.pixel_accuracy()),
        "fg_iou": metric.fg_iou(),
        "m_iou": metric.m_iou(),
    }
    if lane:
        res["line_acc"] = float(metric.line_accuracy())
    return res
