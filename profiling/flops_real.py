"""Accurate FLOPs counter for the nested-prefix dynamic model.

thop/profiler count full-conv FLOPs and miss the sliced F.conv2d calls, so
dynamic-width FLOPs were wrong. This hooks torch.nn.functional.conv2d and
accumulates MACs from the ACTUAL (sliced) weight/input shapes.
"""
import torch
import torch.nn.functional as F


def count_flops(model, x, reps=1):
    """Return GFLOPs (2*MACs) for one forward of model(x)."""
    orig_conv = F.conv2d
    macs = [0]

    def counting_conv(input, weight, bias=None, stride=1, padding=0,
                      dilation=1, groups=1):
        out = orig_conv(input, weight, bias, stride, padding, dilation, groups)
        n = input.shape[0]
        out_h, out_w = out.shape[2], out.shape[3]
        in_c = input.shape[1]
        out_c = weight.shape[0]
        k = weight.shape[2] * weight.shape[3]
        macs[0] += n * out_h * out_w * out_c * (in_c // groups) * k
        return out

    F.conv2d = counting_conv
    try:
        with torch.no_grad():
            for _ in range(reps):
                model(x)
    finally:
        F.conv2d = orig_conv
    return 2 * macs[0] / reps
