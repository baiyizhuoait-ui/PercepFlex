# P5-STEP6/7 - Lane causal resolution probe: pre-registration

Committed before the runner produced any number, so the reading of the result
cannot be tuned after seeing it.

## What is already established, and what is still missing

The resolution ladder (phase5_resolution.csv, 300 images, model-free) shows the
block-fill reference for lane is 0.1853 at 1/8, 0.3220 at 1/4 and 0.5721 at
1/2, while the trained models sit at 0.1843 / 0.1882 / 0.1845. They are sitting
exactly on the 1/8 rung. Lane targets are 3-4 px wide (erosion survival 0.0107
at k=1, 0.0007 at k=2) and the smallest addressable unit at 1/8 is a 64 px
block, so the model paints 3.37x the true lane area with recall 0.689 and
precision around 0.20.

That is a measurement, not yet a cause. Two things could produce it, and they
are architecturally different:

- **H-05a** the head simply operates too coarsely - the fix is a better
  upsampler (DUpsampling, Tian et al. CVPR 2019, argues bilinear upsampling is
  data-independent and that a learned replacement recovers detail).
- **H-05b** the information is not in Z at all - Z is fused from F2/F3/F4, all
  1/8 or deeper, so sub-cell lane position was discarded before the head ever
  ran. The fix is to give the lane branch the encoder's 1/4 map.

Both predict an improvement, so a single cell cannot separate them. Two cells
can.

## Cells (Z=16, E-base, seed 0, 4 epochs, 640x640, batch 16)

| cell | lane input | new information | params | ~extra GFLOPs |
|---|---|---|---|---|
| `r2u_z16` | Z at 1/8 | - | 201366 | baseline (reused from P4A-STEP2 sanity) |
| `l14up_z16` | bilinear(Z) at 1/4 | none | 201366 (+0) | ~+0.53 (head FLOPs x4) |
| `l14f1_z16` | bilinear(Z) + 1x1 lateral from encoder s1 (1/4, 32 ch) | yes | 201878 (+512) | ~+0.53 |
| `lch64_z16` | Z at 1/8, head hidden 32 -> 64 | none | 233814 (+32448) | ~+0.41 |

`l14up` and `l14f1` differ in exactly one thing: whether real 1/4 information
reaches the lane branch. Everything else - parameters up to the 512-channel
lateral, head architecture, head operating resolution, training protocol - is
held constant. That is what makes the difference between them attributable to
information rather than to resolution.

`lch64` is the channel arm. Its FLOP cost (+0.41) is deliberately close to the
resolution arm (+0.53) so the two can be compared as gain per unit compute,
which is the actual question in section 7 of the Phase 5 brief. It is not
there to find a better model.

The baseline is not retrained: `r2u_z16` at 4 epochs is the P4A-STEP2
sanity run and the model builds to identical parameters with the new options
defaulted off (asserted by phase4b_compat.py).

## Readability threshold

2x the external Phase 2-C noise floor: lane_fg 0.0064, mAP50 0.0146,
da_fg 0.0404. This probe is single-seed and estimates no variance of its own.

Baseline (4ep, from the sanity run): mAP50 0.2411, da_mIoU 0.8169,
da_fg 0.7146, lane_mIoU 0.5761, lane_fg 0.1765.

## Predictions, registered in advance

- **L1 (H-05b, primary)** `l14f1_z16` lane_fg >= 0.1829, i.e. baseline + 2x
  noise. The ladder opened +0.137 of headroom between 1/8 and 1/4; capturing
  even a tenth of it clears this bar.
- **L2 (H-05a)** `l14up_z16` lane_fg gain >= 1x noise (0.0032) but smaller than
  the L1 gain. DUpsampling says a learned upsampler should help somewhat; it
  does not say it should help as much as real high-resolution information.
- **L3 (spatial vs channel, the point of the probe)**
  gain(l14f1) / dGFLOPs(l14f1) > gain(lch64) / dGFLOPs(lch64).
  This is the claim that lane is spatially constrained rather than
  channel constrained. It can hold even if L1 alone is unimpressive.
- **L4 (control)** detection and DA move by less than 2x noise, because only
  the lane branch is touched. Probe A already showed this control can fail -
  changing only the lane branch moved detection by -2.53x noise at 20 epochs -
  because the tasks are coupled through Z. If L4 fails here too, it is recorded
  as coupling evidence, not as a broken cell.

## Decision rule, fixed now

- **L1 and L3** -> H-05b supported. Extend `l14f1_z16` to 20 epochs. Do not yet
  run the 1/2 rung.
- **L1, not L3** -> resolution helps but buys no more per FLOP than channels.
  H-05b is supported weakly, H-06 is not subordinate, and "lane is spatial" cannot
  be claimed.
- **L2 only** -> the gain is upsampling, not information. H-05b is not supported;
  the 1/4 lateral is not worth its complexity.
- **Nothing reaches 1x noise** -> H-05b not supported at Z=16. Do not extend to
  20 epochs on this branch; the next question becomes whether the 1/4 map
  itself is too shallow (32 ch, one block) to carry lane, which is a different
  experiment needing its own registration.
- **Between 1x and 2x** -> unresolved. The correct action is to extend the two
  most informative cells to 20 epochs, not to declare a result.

## Standing caveat, restated because it binds the reading

4-epoch readings have twice failed to survive 20 epochs in this project (probe
A's detection control: +0.26x at 4ep, -2.53x at 20ep; reconstruction depth:
+4.47x at 4ep, -0.10x at 20ep). Therefore, even if L1 passes here, the result
is treated as provisional until a 20-epoch confirmation exists. A 4-epoch
result cannot establish H-05b; it can only justify spending the 20-epoch run.
