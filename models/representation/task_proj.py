"""Per-task projection off a shared Compact Z (Phase 4A Level 2, probe A).

Purpose
-------
Under R2 the three tasks do not see Z symmetrically:

    detection : Z -> 1x1 (inside DetFromZ, det_ch=32) -> pyramid
    DA        : Z (raw, z_channels)                   -> DynamicSegHead
    lane      : Z (raw, z_channels)                   -> DynamicSegHead

So detection already owns a projection while DA and lane do not, and the two
segmentation heads are the only place where the shared Z width leaks straight
into a task. `TaskProj` gives DA and lane their own explicit projection so the
width each task sees can be set independently of the shared Z width.

Design rules (kept deliberately strict so the probe stays interpretable)
-----------------------------------------------------------------------
1. Widths are ABSOLUTE targets, not multiples of z. The same targets
   {det: 32, lane: 32, da: 16} are used at every shared-Z width, so across the
   z16 / z32 arms the only thing that changes is the shared Z itself. That is
   what makes "does it still depend on z?" a clean question.
2. `in_ch == out_ch` uses nn.Identity, so a projection that does not change the
   width is a genuine no-op rather than an extra learned remix. Keeping it a
   no-op is what makes R3 comparable to R2 on the arms where nothing should move.
3. Default off. With `task_proj` absent from the config, every existing R0/R1/R2
   model builds bit-identically to before.

Cost is negligible: two 1x1 convs, ~800 params at z=16 and ~1.6k at z=32, i.e.
under 1% of the model. The probe is therefore close to parameter-matched.
"""
import torch
import torch.nn as nn


class TaskProj(nn.Module):
    """1x1 conv + BN + ReLU from the shared Z to one task's own width.

    Identity when in_ch == out_ch (see rule 2 above).
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        if self.in_ch == self.out_ch:
            self.op = nn.Identity()
        else:
            self.op = nn.Sequential(
                nn.Conv2d(self.in_ch, self.out_ch, 1, bias=False),
                nn.BatchNorm2d(self.out_ch),
                nn.ReLU(inplace=True),
            )

    def forward(self, z):
        return self.op(z)

    def extra_repr(self):
        return "in=%d out=%d%s" % (
            self.in_ch, self.out_ch,
            " (identity)" if self.in_ch == self.out_ch else "")
