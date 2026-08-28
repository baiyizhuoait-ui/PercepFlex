"""YOLOv5-style NMS for decoded detection outputs (N, 25200, 6): xywh, conf, cls."""
import torch
import torchvision


def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.6, max_det=300):
    """prediction: (1, 25200, 6) decoded (xywh, obj_conf, cls_conf). -> list of (N,6) xyxy.

    Score = obj_conf * cls_conf (same as YOLOP / TriLiteNet lib.core.general)."""
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres
    output = []
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            output.append(torch.zeros((0, 6)))
            continue
        x[:, 5:] *= x[:, 4:5]  # conf = obj * cls
        # box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            output.append(torch.zeros((0, 6)))
            continue
        c = x[:, 5:6] * 0  # agnostic NMS (single class anyway)
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        if keep.shape[0] > max_det:
            keep = keep[:max_det]
        output.append(x[keep])
    return output
