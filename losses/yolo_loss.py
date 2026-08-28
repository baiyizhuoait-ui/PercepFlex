"""YOLOv5-style detection loss (single class), adapted for the adaptive model.

Takes the head's per-scale raw outputs (B, na, ny, nx, 5+nc) and per-image
targets (n, 5): [cls, cx, cy, w, h] normalized. Anchor assignment + box/obj/cls
losses follow the standard YOLOv5 formulation (simplified, single class).
"""
import math

import torch
import torch.nn as nn


def bbox_iou(box1, box2, xywh=True, eps=1e-7):
    """IoU for (n,4) boxes, xywh or xyxy."""
    if xywh:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_x2, b1_y1, b1_y2 = box1[..., 0], box1[..., 2], box1[..., 1], box1[..., 3]
        b2_x1, b2_x2, b2_y1, b2_y2 = box2[..., 0], box2[..., 2], box2[..., 1], box2[..., 3]
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter + eps)
    return inter / union


class YOLOLoss(nn.Module):
    def __init__(self, anchors, nc=1, img_size=640, strides=(8, 16, 32)):
        super().__init__()
        self.nc = nc
        self.na = anchors.shape[1]
        self.nl = anchors.shape[0]
        self.anchors = anchors  # (nl, na, 2) in input pixels
        self.strides = torch.tensor(strides, dtype=torch.float)
        self.img_size = img_size
        self.balance = [4.0, 1.0, 0.4]

    def forward(self, preds, targets, img_size=None):
        """preds: list of nl tensors (B, na, ny, nx, no); targets: (nt, 5) cls,xywh(norm)."""
        img_size = img_size or self.img_size
        device = preds[0].device
        bs = preds[0].shape[0]
        tcls, tbox, indices, anchors = self.build_targets(preds, targets, img_size)

        loss_box = torch.zeros((), device=device)
        loss_obj = torch.zeros((), device=device)
        loss_cls = torch.zeros((), device=device)
        for i, pi in enumerate(preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=torch.float, device=device)
            n = b.shape[0]
            if n:
                pxy = pi[b, a, gj, gi][:, :2].sigmoid() * 2 - 0.5
                pwh = (pi[b, a, gj, gi][:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], xywh=True)
                loss_box += (1.0 - iou).mean() * self.balance[i]
                tobj[b, a, gj, gi] = 1.0
                if self.nc > 1:
                    loss_cls += nn.functional.binary_cross_entropy_with_logits(
                        pi[b, a, gj, gi][:, 5:], nn.functional.one_hot(tcls[i].long(), self.nc).float())
            else:
                loss_box += torch.zeros(1, device=device).squeeze() * self.balance[i]
            # objectness: all anchors in this scale
            loss_obj += nn.functional.binary_cross_entropy_with_logits(
                pi[..., 4], tobj, reduction="mean") * self.balance[i]

        loss_box *= 0.05  # box gain (YOLOv5)
        loss_obj *= 1.0
        loss_cls *= 0.5 if self.nc > 1 else 0.0
        return loss_box + loss_obj + loss_cls, dict(box=loss_box.item(), obj=loss_obj.item(), cls=loss_cls.item())

    def build_targets(self, preds, targets, img_size):
        """YOLOv5-style anchor assignment (simplified: no multi-scale-aware filtering)."""
        device = preds[0].device
        nl, na = self.nl, self.na
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=device)  # [0,1,2,3,4,5,6] = [img_id, cls, cx, cy, w, h, anchor_id]
        ai = torch.arange(na, device=device).float().view(na, 1).repeat(1, targets.shape[0])
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # (na, nt, 7)

        for i in range(nl):
            ny, nx = preds[i].shape[2], preds[i].shape[3]
            gain[2:6] = torch.tensor([nx, ny, nx, ny], device=device)
            t = targets * gain
            if t.shape[1]:
                gxy = t[:, :, 2:4]
                gwh = t[:, :, 4:6]
                j = torch.ones(t.shape[0], t.shape[1], dtype=torch.bool, device=device)  # keep all (no grid-edge filtering for MVP)
                t = t[j]
                b, c = t[:, :2].long().T
                gxy, gwh = t[:, 2:4], t[:, 4:6]
                gij = gxy.long()
                gi, gj = gij.T
                a = t[:, 6].long()
                indices.append((b, a, gj.clamp_(0, ny - 1), gi.clamp_(0, nx - 1)))
                tbox.append(torch.cat((gxy - gxy.floor(), gwh), 1))
                anch.append(self.anchors[i].to(device)[a])
                tcls.append(c)
            else:
                indices.append((torch.zeros(0, device=device, dtype=torch.long),) * 4)
                tbox.append(torch.zeros(0, 4, device=device))
                anch.append(torch.zeros(0, 2, device=device))
                tcls.append(torch.zeros(0, device=device))
        return tcls, tbox, indices, anch
