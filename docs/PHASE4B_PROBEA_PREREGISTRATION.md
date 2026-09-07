# Phase 4A - Level 2 Probe A: pre-registration

**Written before any probe result was seen.** Committed at `03c5014`, before the
runner produced a single number. Nothing in this file has been revised since.

## Question

Does giving each task its own projection off the shared Z help, and in
particular does it decouple what a task can use from the width of the shared Z?

## Design

Four cells, E-base, seed 0, 4 epochs, batch 16, 640x640, `tri_train` 69863.

| cell | variant | shared Z | det | lane | da | params |
|---|---|---|---|---|---|---|
| `r2u_z16` | R2 uniform | 16 | 32 (via `z_proj`) | raw 16 | raw 16 | 201366 |
| `r2u_z32` | R2 uniform | 32 | 32 (via `z_proj`) | raw 32 | raw 32 | 215734 |
| `r3tp_z16` | R3 taskproj | 16 | 32 | **32** (16->32) | 16 (identity) | 206550 |
| `r3tp_z32` | R3 taskproj | 32 | 32 (identity) | 32 (identity) | **16** (32->16) | 211670 |

`r2u_z16` is **reused** from the Phase 4A STEP 2 sanity run (`exp4A_r2_z16_e4`).
That is legitimate only because the model builds to the identical 201366
parameters with `task_proj` defaulted off, verified numerically by
`scripts/phase4b_compat.py`. Its provenance is recorded in the results CSV as
`source=reused:phase4a-sanity`.

Two design rules, both load-bearing:

1. **Targets are absolute, not multiples of z.** `{det: 32, lane: 32, da: 16}` at
   *both* widths. Consequence: under R3 every head sees the same input width at
   z16 and at z32. Only the shared Z differs. That is what makes "does it still
   depend on z?" a clean question.
2. **`in_ch == out_ch` is `nn.Identity`.** A projection that does not change
   width is a genuine no-op, not an extra learned remix.

## What each arm can actually test

Because of rule 2, R3 differs from R2 in exactly one place per arm:

- **z16 arm**: lane 16->32 is the only real change. det is already 32 in both, da
  is 16 in both. So a detection or DA movement here would be a *shared-Z
  training interaction*, not a projection effect, and must be reported as such.
- **z32 arm**: da 32->16 is the only real change (and it *removes* 4064 params).
  det and lane are identical to R2.

Confound, stated up front: the z16 arm also adds 5184 lane-head params (+2.6%).
So a lane gain there is "lane got more capacity", which is the thing being
tested, but it is not separable from "lane got an extra conv layer".

## Noise floor and power

External reference only (Phase 2-C, 3 seeds, 4 epochs). This probe is
**single-seed**, so it estimates no variance of its own.

| metric | 1x noise | 2x noise (readability threshold) |
|---|---|---|
| mAP50 | 0.0073 | 0.0146 |
| mAP50_95 | 0.0032 | 0.0064 |
| da_mIoU | 0.0142 | 0.0284 |
| da_fg | 0.0202 | 0.0404 |
| lane_mIoU | 0.0021 | 0.0042 |
| lane_fg | 0.0032 | 0.0064 |

**Anything below 2x noise is recorded as unresolved, not as a result.** A
4-epoch single-seed probe is deliberately cheap and therefore deliberately
underpowered; a null here means "not visible at this resolution", not "absent".

## Pre-registered predictions

- **P1 (z16, lane).** `r3tp_z16.lane_fg - r2u_z16.lane_fg >= 0.0064` (2x noise).
  Rationale: lane showed 2.25x-noise z demand under R2 at 20 epochs.
- **P2 (z32, DA).** `|r3tp_z32.da_fg - r2u_z32.da_fg| < 0.0404` (2x noise):
  squeezing DA from 32 to 16 costs nothing. Rationale: DA demand was 0.21x noise.
- **P3 (cross-arm, primary).** The R3 z-gap is smaller than the R2 z-gap by at
  least 2x noise, on mAP50 and lane_fg:
  `|r3tp_z32 - r3tp_z16| < |r2u_z32 - r2u_z16| - 2x_noise`.
  This is the real question. If it holds, what R2 was paying for at wide z was
  largely head input width, not shared representational capacity.
- **P4 (detection, control).** `|r3tp - r2u|` on mAP50 within 1x noise on both
  arms, since det is held at 32 in all four cells. If detection moves beyond
  2x noise, that is unexpected and must be reported, not explained away.

## Decision rule (fixed now)

- **A = YES** if **P3** holds (primary), or if **P1 and P2** both hold (secondary).
  -> go to task-aware allocation / projection in depth, then pick the final
  structure, then 3-seed confirmation.
- **A = PARTIAL** if P1/P2 hold but P3 fails: per-task width helps, but the shared
  Z width still matters. -> task-aware allocation *with a wider shared Z* (the
  priority-2 direction), still not B.
- **A = NO** if none of P1-P3 reach 2x noise, i.e. R3 is indistinguishable from R2
  everywhere. -> switch to **B** (second reconstruction) to test whether the
  reconstruction, not the allocation, is the binding constraint.

If any quantity lands between 1x and 2x noise, it is **unresolved** and the right
response is to extend the two most informative cells to 20 epochs, not to declare
a result in either direction.

## Explicitly not in this probe

z256/z512, denser z sweeps, large encoder sweeps under R2, pruning, INT8, NPU
deployment, multi-seed confirmation. Those stay blocked until a final structure
is chosen.
