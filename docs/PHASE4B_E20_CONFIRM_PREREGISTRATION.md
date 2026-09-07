# Phase 4A - Level 2 Probe B: 20-epoch confirmation, pre-registration

Status: the two 20ep runs this document governs were **launched before this file
was written**, and no result had been observed when it was written. It is
therefore a pre-registration, not a post-hoc rationalisation. The 4-epoch
numbers quoted below are already committed in `phase4B_probeB_results.csv`.

## Why this run exists

Probe B produced two findings, and only one of them was predicted:

1. The predicted one failed. `r2d_z16` (dilated reconstruction, dilation 2 and 4)
   did not beat the baseline by 2x noise: +0.82x on mAP50, +1.50x on mAP50_95
   (unresolved). Q1 fails, so by the registered rule B = NO for the question
   "is receptive field the binding constraint".
2. The **control arm** won, and it was not predicted to. `r2p_z16` (identical
   blocks, dilation 1 and 1, byte-identical parameter count 204246) beat the
   baseline by **+4.47x noise on mAP50** and **+4.78x on mAP50_95**, and beat
   the dilated cell by +3.64x / +3.28x.

Finding 1 is a clean negative and needs no confirmation. Finding 2 is a
positive, and this project already has direct evidence that 4-epoch readings
are not safe: probe A's detection control was inert at 4 epochs (+0.26x noise)
and cost -2.53x noise at 20 epochs. A 4-epoch positive that has not been
re-checked is exactly the mistake that run demonstrated. So `r2p_z16` is
re-checked at 20 epochs before it is allowed to influence any design decision.

`r2d_z16` is included because the dilation question is worth one clean answer at
convergence: if dilation is harmful at 20 epochs too, the whole YOLOF-style
direction is closed for this model, and that is worth knowing cheaply.

## Cells

20 epochs, seed 0, E-base, 640x640, batch 16 - identical to the R2 baseline runs.

| cell | shared Z | reconstruction | params | vs R2 baseline |
|---|---|---|---|---|
| `r2p_z16` | 16 | 2 depthwise residual blocks, dilation **1, 1** | 204246 | +2880 (+1.43%) |
| `r2d_z16` | 16 | identical blocks, dilation **2, 4** | 204246 | +2880 (+1.43%) |

Baseline is **not** retrained: `r2_z16` in `phase4A_results.csv` (20 ep, seed 0,
201366 params, mAP50 0.3543, mAP50_95 0.1273, da_mIoU 0.8488, da_fg 0.7612,
lane_mIoU 0.5847, lane_fg 0.1943) is the same model with a 1x1 reconstruction.
Epoch and seed match exactly, so reuse is legitimate.

## Readability threshold

External Phase 2-C noise floor, used unchanged:

| metric | 1 noise | 2 noise |
|---|---|---|
| mAP50 | 0.0073 | 0.0146 |
| mAP50_95 | 0.0032 | 0.0064 |
| da_mIoU | 0.0142 | 0.0284 |
| da_fg | 0.0202 | 0.0404 |
| lane_mIoU | 0.0021 | 0.0042 |
| lane_fg | 0.0032 | 0.0064 |

Single seed, no variance estimated by this run. Only effects at >= 2x noise are
treated as readable. Effects between 1x and 2x are recorded as *unresolved*.

## Predictions

- **B1 (primary).** `r2p_z16` beats `r2_z16` by >= 2x noise on mAP50 (+0.0146)
  or mAP50_95 (+0.0064). At 4 epochs the same comparison gave +0.0326 and
  +0.0153.
- **B2 (the discriminator, carried over from 4ep).** `r2p_z16` beats `r2d_z16`
  by >= 2x noise on mAP50 or mAP50_95. At 4 epochs: +0.0266 and +0.0105.
- **B3 (control).** The lane task does not pay for it: `lane_mIoU` and
  `lane_fg` must not fall by >= 2x noise versus `r2_z16`. At 4 epochs both were
  inside 1x noise against the baseline (-0.90x and -0.59x). If the detection
  gain is bought with a lane loss at 20 epochs, that must be reported as a
  trade, exactly as it was for probe A.

## Decision rule (fixed now)

| outcome | verdict | next step |
|---|---|---|
| B1 and B2 hold | **depth confirmed** | adopt plain-depth reconstruction as the new R2 baseline; close the dilation direction; re-examine z width under the new reconstruction before choosing final candidates |
| B1 holds, B2 fails | **depth confirmed, dilation moot** | same as above; dilation is simply not worth the hyperparameter |
| B1 fails | **4ep artefact** | the +4.47x was a 4-epoch reading, like probe A's P4; reconstruction depth is not a lever; close this branch and go to the next item on the ranked list |

Anything between 1x and 2x noise is reported as *unresolved* and calls for more
epochs or more seeds, not for declaring a result.

## Interaction with the rest of Level 2

Probe C (already complete) rejected gradient magnitude imbalance as a lever:
sacrificing ~13x noise of mAP50 bought 0.27x noise of DA and 1.25x of lane.
Probe A rejected per-task projection width. If B1 holds here, the Level 2
picture becomes coherent and asymmetric - detection's bottleneck is the
reconstruction, while DA and lane are not limited by the shared Z at all. That
is the case for a task-aware *final* design, not for a task-aware *shared* Z.

If B1 holds, one follow-up becomes the obvious next experiment and should be
run before any final candidate is fixed: `r2p_z32`, to check whether the
reconstruction gain and the z-width gain add or overlap. It is **not** part of
this pre-registration and will not be launched without its own.
