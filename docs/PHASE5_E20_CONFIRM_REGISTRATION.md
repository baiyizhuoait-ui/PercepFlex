# P5-STEP7b — 20-epoch confirmation of l14f1_z16
## Registration, written and committed BEFORE the run

Date: 2026-09-08 12:35 CST. No number from this run exists yet.

## What is being confirmed

The 4-epoch probe passed L1+L3 of the P5-STEP7 pre-registration
(`PHASE5_LANE_PROBE_PREREGISTRATION.md`): adding the 1/4 lateral
(`l14f1`) moved lane_fg +0.0151 (2.36x external noise) while the
equal-FLOP channel control (`lch64`) moved +0.0013 (0.20x), and the
det/DA controls stayed inside 1x noise. Per that registration, this
authorises (does not establish) H-05b: the lane bottleneck at Z=16 is
the absence of 1/4-resolution information, not channel capacity.

## Literature pre-check (user rule: search before every step)

- YOLOPX (Pattern Recognition 148, 2024): lane head consumes C2 (1/4)
  high-resolution features + long-distance context; SOTA lane IoU 27.2
  on BDD100K at full training budget. The 1/4 benefit is not a
  short-training artifact in the prior art.
- Lo et al., FSS/DD-Block (EDANet variants): thin structures are erased
  by early downsampling; keeping feature extraction at 1/2-1/4 rescues
  them, with gains that persist at convergence.
- Implication: the 4-epoch probe direction is expected to survive 20
  epochs, but magnitude may shrink (probe A precedent: +0.26x at 4ep,
  -2.53x at 20ep). The run exists precisely to test that.

## Comparison target (already committed, not rerun)

Baseline: `r2_z16` at 20 epochs from `phase4A_results.csv`
(same architecture family; `r2u_z16` in the probe CSV is the renamed
4-epoch row of this cell):

    lane_fg 0.1943   mAP50 0.3543   da_fg 0.7612

## Decision rule, fixed now

Noise floor: Phase 2-C external 2x lane_fg = 0.0128 (4-epoch
measurement). No 20-epoch replicate exists; using the 4-epoch floor at
20 epochs is conservative because seed variance shrinks with training.
Applying it on purpose.

- d_lane = lane_fg(l14f1_z16, 20ep) - 0.1943
- **d_lane >= +0.0256 (2x)** -> H-05b CONFIRMED at 20 epochs. P5-STEP7
  closes; bottleneck profile locks "spatial resolution (1/4 missing)"
  as the lane bottleneck.
- **+0.0128 (1x) <= d_lane < +0.0256** -> WEAK SUPPORT. H-05b stays
  provisional; do not build on it; record in the map as unresolved.
- **d_lane < +0.0128** -> NOT CONFIRMED. The 4-epoch gain was a
  short-training artifact (probe A precedent). H-05b returns to OPEN and
  the lane bottleneck question stays open.
- Controls: det (mAP50) and DA (da_fg) of l14f1_z16 e20 must sit
  within 2x their respective noise floors of the r2_z16 e20 values.
  A det move beyond 2x is recorded as task coupling (as in probe A),
  not as a broken cell, but it caps any claim of a "free" lane gain.

## Sanity guards

- Config `phase4b_l14f1_z16.yaml` is identical to the probe run
  (audited in `phase5_lane_probe_config_audit.txt`); only epochs
  change (4 -> 20).
- No other file is modified for this run.
