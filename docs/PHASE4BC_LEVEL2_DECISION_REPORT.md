# Phase 4A - Level 2: consolidated decision report

Scope: the three Level 2 probes (A per-task projection width, B reconstruction,
C gradient magnitude balance). Every number is recomputed from the result CSVs.
The 20-epoch re-check of probe B's unpredicted positive is still running; it is
reported as pending and no conclusion here depends on it.

Noise floor throughout is the external Phase 2-C estimate (3 seeds @ 4ep):
mAP50 0.0073, mAP50_95 0.0032, da_mIoU 0.0142, da_fg 0.0202, lane_mIoU 0.0021,
lane_fg 0.0032. All probes are single-seed, so only effects at >= 2x noise are
called readable; 1x-2x is recorded as unresolved.

---

## 1. Verdicts

| probe | question | epochs | verdict | key number |
|---|---|---|---|---|
| **A** | does each task need its own projection width off the shared Z? | 20 | **NO** | lane +1.16x noise, detection -2.53x |
| **B** | is the reconstruction's *receptive field* the constraint? | 4 | **NO** | dilation costs 3.64x noise vs plain depth |
| **B'** | does reconstruction *depth* help? (unpredicted) | 20 | **NO - 4ep artefact** | +4.47x at 4ep becomes **-0.10x** at 20ep |
| **C** | is detection's gradient dominance starving the seg tasks? | 20 | **REJECTED** | give up 13.07x noise mAP50, buy 0.27x DA |

All three probes were designed to find a task-specific capacity lever. All three
failed to find one on the shared Z. One of them found a detection-specific lever
by accident.

---

## 2. Probe A recap (20 epochs)

`r3tp_z16` vs `r2_z16`: lane gets its own 16->32 projection (+5184 params, all
of it in the lane branch; the detection path is byte-identical because
`task_proj` sets `DetFromZ.det_ch`, which R2 already had at 32).

| metric | r2_z16 | r3tp_z16 | delta | x noise |
|---|---|---|---|---|
| mAP50 | 0.3543 | 0.3420 | -0.0123 | -1.68 (unresolved) |
| mAP50_95 | 0.1273 | 0.1192 | -0.0081 | **-2.53 (loss)** |
| da_mIoU | 0.8488 | 0.8556 | +0.0068 | +0.48 |
| da_fg | 0.7612 | 0.7712 | +0.0100 | +0.50 |
| lane_mIoU | 0.5847 | 0.5870 | +0.0023 | +1.10 |
| lane_fg | 0.1943 | 0.1980 | +0.0037 | +1.16 |

Two readings, both bad for the hypothesis: either task-specific projection buys
an unresolved lane gain with a readable detection loss, or - the more
interesting half - the three tasks are strongly coupled *through* Z, since
touching only the lane branch moved detection.

Methodological note that shaped everything after it: at 4 epochs the detection
control was inert (+0.26x) and at 20 epochs it cost -2.53x. **A 4-epoch control
that passes says nothing about 20 epochs.** This is why probe C was run at 20
epochs and why probe B's positive is not being accepted at 4.

---

## 3. Probe B (4 epochs) - reconstruction: receptive field vs depth

The reconstruction that feeds the detection head from Z was a single 1x1
convolution, i.e. a one-pixel receptive field at 1/8 scale. YOLOF (CVPR 2021)
diagnoses exactly this failure mode in single-level detectors and fixes it with
a dilated encoder, which is why this probe was built. Three cells, including a
control that differs from the dilated cell only in the dilation rates.

| cell | z | reconstruction | params | mAP50 | mAP50_95 |
|---|---|---|---|---|---|
| `r2u_z16` | 16 | 1x1 (baseline) | 201366 | 0.2411 | 0.0728 |
| `r2d_z16` | 16 | 2 dw resid blocks, dilation 2,4 | 204246 | 0.2471 | 0.0776 |
| `r2p_z16` | 16 | identical blocks, dilation 1,1 | 204246 | **0.2737** | **0.0881** |
| `r2d_z32` | 32 | dilation 2,4 | 218614 | 0.2497 | 0.0802 |

**Q1 FAIL.** The dilated cell gains +0.82x noise on mAP50 over baseline
(+1.50x, unresolved, on mAP50_95). Far short of the 2x gate.

**Q2 FAIL, sign reversed.** `r2d_z16` loses to `r2p_z16` by 3.64x noise on
mAP50 and 3.28x on mAP50_95, at byte-identical parameter count. Dilation does
not merely add nothing on top of depth - it removes most of what depth bought.

**Q3 marginal FAIL.** lane_mIoU moved +1.19x at z16, just over the 1x gate; DA
stayed at 0.26x.

**Verdict: B = NO.** Receptive field is not the binding constraint, and the
YOLOF-style dilated direction is closed for this model at 1/8 resolution.

### The unpredicted result

The control arm won, and it was not predicted to. Decomposing the two effects:

```
depth (plain blocks, dilation 1)   +4.47x noise mAP50   <- real lever
dilation on top of that depth      -3.64x noise mAP50   <- actively harmful
------------------------------------------------------------------
net dilated cell vs baseline       +0.82x noise
```

+2880 params (+1.43%) for +4.47x noise of mAP50 is the best cost-benefit number
measured anywhere in this project. It is also a 4-epoch single-seed reading,
which probe A just demonstrated is unsafe. `r2p_z16` and `r2d_z16` are running
at 20 epochs now; see section 6.

---

## 4. Probe C (20 epochs) - H-11, gradient magnitude imbalance

Motivation was a measurement already in hand: at the R2 20ep checkpoints,
detection owns 73.2% (z16) and 76.9% (z32) of the gradient on Z, with all
pairwise cosines at ~0. GradNorm's central claim is that magnitude imbalance
alone - no conflict required - is enough for one task to monopolise shared
parameters. Intervention: `lambda_det = 0.2`, model otherwise bit-identical to
R2.

| cell | z | mAP50 | mAP50_95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|
| `r2_z16` | 16 | 0.3543 | 0.1273 | 0.8488 | 0.7612 | 0.5847 | 0.1943 |
| `rw_z16` | 16 | 0.2589 | 0.0817 | 0.8526 | 0.7667 | 0.5872 | 0.1983 |
| `r2_z32` | 32 | 0.3553 | 0.1248 | 0.8573 | 0.7736 | 0.5875 | 0.1992 |
| `rw_z32` | 32 | 0.2614 | 0.0826 | 0.8573 | 0.7738 | 0.5871 | 0.1988 |

### C4 manipulation check - with a correction to my own pre-registration

The committed diagnostic backpropagates each task loss **separately**, so it
never sees `lambda_det`. Its `share_det` column is therefore *unweighted* and
cannot be compared with the 0.50 gate as written - the raw column still reads
0.707 and 0.818, which would have looked like a failed manipulation. Applying
the weights to the same measured norms, which is what the gate describes in
words:

| cell | g_det | g_da | g_lane | raw det share | **effective** det share | gate |
|---|---|---|---|---|---|---|
| `rw_z16` | 0.03062 | 0.00717 | 0.00552 | 0.707 | **0.326** | PASS |
| `rw_z32` | 0.03529 | 0.00414 | 0.00372 | 0.818 | **0.473** | PASS |

Baseline effective shares were 0.732 and 0.770. The gate (<= 0.50) passes on
both arms. This is recorded as an instrument error in the pre-registration, not
silently patched.

Secondary evidence that the intervention bit: the *unweighted* norms also
moved. g_da rose 0.00587 -> 0.00717 (+22%) at z16 and 0.00322 -> 0.00414 (+29%)
at z32. The rebalanced model learned a Z that the segmentation tasks push
harder. The solution changed, not just the loss scale.

### Predictions

**C1 FAIL.** Best segmentation movement anywhere is +1.25x noise (`rw_z16`
lane_fg). At z32 both DA metrics moved by +0.00x and +0.01x - literally not at
all.

**C2 FAIL.** Rebalancing made the segmentation tasks *less* z-sensitive, not
more: the z32-z16 gaps are +0.16x (lane_fg) and +0.33x (da_mIoU), versus R2's
+1.53x and +0.60x.

**C3 PASS, hugely.** Detection fell 13.07x noise at z16 and 12.86x at z32.

### The exchange rate

```
rw_z16:  gave up  13.07x noise of mAP50
         bought    0.27x noise of da_fg,   1.25x noise of lane_fg
rw_z32:  gave up  12.86x noise of mAP50
         bought    0.01x noise of da_fg,  -0.12x noise of lane_fg
```

**Verdict: H-11 rejected as a lever.** Detection's dominance of the Z gradient
is real, but it is not what caps DA and lane. Crushing detection by an order of
magnitude more than the threshold bought a quarter of one noise unit.

### One more measurement

Effective rank of Z went *up* under rebalancing - 6.79 -> 7.85 effective
dimensions at z16 (42.5% -> 49.1%), 11.88 -> 13.43 at z32 (37.1% -> 42.0%) -
with no segmentation gain at all. More of Z is being used and none of the
additional use reaches DA or lane.

---

## 5. What the three probes say together

Three independent attempts to find a **task-specific capacity lever on the
shared Z** all failed:

- give the tasks different amounts of Z (A) - no
- give them more Z to share (Phase 3B/3C, z16->z128) - no
- give them a bigger share of the gradient on Z (C) - no

And one lever was found by accident, and it is **detection-specific** (B'):

- give the detection reconstruction depth - yes, +4.47x noise for +1.43% params

The coherent picture: **detection and the segmentation tasks have different
bottlenecks, and only detection's bottleneck is in the shared-Z pathway.** The
segmentation tasks are not starved of Z; they are saturated with respect to what
Z contains. That is an argument for a task-aware *final* design (different input
resolutions or feature sources per task) and against a task-aware *shared* Z.

### Candidate explanations for what actually caps DA and lane

These are hypotheses, not results. Listed in the order I would test them.

1. **Spatial resolution, not capacity (leading candidate for lane).** Every task
   reads Z at 1/8. A lane marking a few pixels wide becomes a sub-pixel feature
   at 1/8, and the head upsamples 8x to produce the mask. lane_fg IoU of 0.19 is
   exactly what thin-structure-at-low-resolution looks like. Probe A gave lane
   more *channels* and got nothing, which is consistent: the limit is spatial,
   not channel-wise. Testable with **no training** - see section 6.
2. **DA is genuinely saturated.** da_mIoU 0.849-0.857 and da_fg 0.76-0.77 have
   not moved under any intervention in Phase 3A, 3B, 3C, 4A or any Level 2
   probe. Phase 3A already noted DA saturating at E-large. Remaining errors are
   probably boundary and occlusion cases, which capacity does not fix.
3. **Label noise / evaluation ceiling for lane.** BDD100K lane annotations are
   sparse and inconsistent; published lane numbers on this dataset are
   commonly reported as accuracy rather than IoU because IoU is harsh on thin
   structures. Part of the 0.19 may simply be irreducible for this annotation
   set.
4. **Loss-side, not architecture-side.** If the segmentation loss has no class
   weighting, background dominates the gradient and thin foreground structures
   are under-optimised. This has not been checked and is cheap to check.

### Ruled out by this phase

- wider shared Z (Phase 3B/3C: no reliable effect; effective rank shows the
  extra channels are correlated, not independent)
- per-task projection width off Z (A)
- dilated / larger-receptive-field reconstruction (B)
- gradient magnitude rebalancing (C)
- task projection as a channel-remixing operation (A, with the caveat that
  probe A tested the weakest form: a 1x1, spatially uniform, input-independent
  remix)

### Still open

- H-07b, bottleneck *placement* (never tested)
- whether lane can be rescued by features at 1/4 or 1/2 - the one intervention
  the error-geometry result actually motivates, and it exits the shared-Z design
- whether R2's far-field DA regression versus R0 replicates (exploratory, 6c)
- DA's residual 0.11 gap to its own resolution ceiling - far-field cases, likely
  depth ambiguity rather than capacity

*Resolved since this list was first written:* reconstruction depth (6a, artefact)
and "is lane resolution-limited" (6b, yes).

---

## 6. Resolution: 20-epoch confirmation and the error-geometry diagnostic

### 6a. The +4.47x was a 4-epoch artefact

| cell | params | mAP50 | mAP50_95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|
| `r2_z16` | 201366 | 0.3543 | 0.1273 | 0.8488 | 0.7612 | 0.5847 | 0.1943 |
| `r2p_z16` | 204246 | 0.3536 | 0.1261 | 0.8543 | 0.7693 | 0.5863 | 0.1967 |
| `r2d_z16` | 204246 | 0.3474 | 0.1263 | 0.8481 | 0.7601 | 0.5870 | 0.1977 |

- **B1 FAIL** - -0.10x noise on mAP50, -0.38x on mAP50_95.
- **B2 FAIL** - r2p over r2d is +0.85x, unresolved.
- **B3 PASS** - lane +0.76x / +0.75x, no cost.

Registered outcome: **4ep ARTEFACT. Reconstruction depth is not a lever.**

The gain did not shrink, it vanished, which identifies the mechanism: this was a
**convergence-speed effect, not a capacity effect**. The deeper reconstruction
gets to the same place sooner and the 1x1 baseline catches up by epoch 20. All
three cells are indistinguishable at convergence. This is now the second time a
4-epoch reading has failed to survive 20 epochs in this project (probe A:
detection control +0.26x at 4ep, -2.53x at 20ep), so the rule is upgraded from a
caveat to a standing protocol: **4-epoch probes here may falsify a large
predicted effect, they may not establish one.**

Not affected: the dilated cell is still the lowest of the three on mAP50
(-0.95x). Dilation is not rescued by convergence, it is just not resolvable at
this noise level.

### 6b. Lane is resolution-bound, and the number is unambiguous

Zero-training error geometry on 300 val images, 640x640 output. Three
measurements, the first of which involves no model at all.

| quantity | value | meaning |
|---|---|---|
| `lane_res_ceiling_1over8` | **0.1853** | fg IoU of a naive max-pool-by-8-then-upsample of the **ground truth** |
| achieved lane fg IoU | 0.1843 / 0.1882 / 0.1845 | r2_z16 / r2_z128 / r0_z16 |
| `lane_erode1_ratio` | **0.0107** | only 1.1% of lane pixels survive a 1px erosion - lane lines are 1-2 px wide |
| `lane_pred_over_gt_area` | **3.37x** | the model paints 3.4x more lane pixels than exist |
| `lane_recall@0` | 0.689 | it still covers 69% of the true lane pixels |
| lane fg IoU @ tol 0 / 1 / 2 px | 0.184 / 0.382 / **0.460** | 2 px of slack more than doubles IoU |

**The model's lane output is statistically indistinguishable from a crude 1/8
block representation of the ground truth** (difference 0.001-0.003, i.e. inside
one noise unit of 0.0032). Everything this project has added - wider Z, per-task
projections, balanced gradients, a deeper reconstruction - has moved lane fg IoU
by amounts smaller than the gap to that naive ceiling.

The mechanism is visible in the area ratio. At 1/8 the minimum addressable unit
is an 8x8 block = 64 input pixels, while a lane line is 1-2 px wide. To cover a
line the model must light up whole blocks, so it predicts 3.4x the true area:
recall 0.689 but precision only ~0.20. IoU 0.184 follows arithmetically. The
model finds the lanes; it cannot place them. This is a **geometric** limit and
no amount of Z capacity addresses it.

### 6c. DA is a different story, and part of it was hidden

| quantity | r2_z16 | r2_z128 | r0_z16 |
|---|---|---|---|
| `da_res_ceiling_1over8` | 0.8753 | 0.8753 | 0.8753 |
| achieved da fg IoU | 0.7555 | 0.7647 | 0.7699 |
| `da_erode1_ratio` | 0.9558 | 0.9558 | 0.9558 |
| DA IoU, top third (far) | **0.3933** | **0.3073** | **0.6377** |
| DA IoU, middle third | 0.7628 | 0.7773 | 0.7778 |
| DA IoU, bottom third (near) | 0.6378 | 0.6453 | 0.6454 |

DA is blob-like (96% survives erosion) and sits **0.11 below** its own 1/8
resolution ceiling - so unlike lane it is *not* resolution-bound, and there is
real headroom that resolution alone would permit.

The band breakdown surfaces something the aggregate metric hides, and it is
exploratory rather than pre-registered so it is stated as an observation: the
top third of the image (far field / horizon) is where DA fails, and **R0 does
substantially better there than R2** (0.638 vs 0.393 at z16, 0.307 at z128).
Routing detection through the shared Z appears to have cost the far-field
drivable area roughly 0.24-0.33 IoU while leaving the aggregate da_mIoU looking
flat. If that survives a second seed it is a real cost of the R2 design that
Phase 4A's headline numbers did not capture.

### 6d. What this changes

The leading hypothesis in section 5 is confirmed for lane and refuted for DA.
Lane's ceiling is geometric; the intervention that would address it is giving
the lane head features at 1/4 or 1/2, which is outside the shared-Z premise
entirely. DA has genuine headroom and its residual errors are far-field.

**Net state of Level 2: every lever tested on the shared Z has failed, and the
one remaining explanation for the segmentation tasks is not about Z at all.**
That is a complete answer to the Level 2 question, and it is a negative one.

---

## 7. Threats to validity

1. **Single seed everywhere.** The noise floor is an external Phase 2-C
   estimate from 3 seeds at 4 epochs, not from these runs. Effects near 2x
   noise are genuinely uncertain; only the large ones (13x, 4.5x, 3.6x) should
   be treated as solid.
2. **Probe B is at 4 epochs** and is not accepted as a result until section 6
   completes. Probe A is the existence proof that this matters.
3. **C4 was measured with an instrument that cannot see the manipulation** and
   required an explicit correction. Any future pre-registration that gates on a
   quantity must check that the instrument actually observes it.
4. **Probe C's detection collapse is large enough to change the training
   dynamics**, not just the weighting - the final train loss dropped from 0.2150
   to 0.1322. The null on C1 is therefore a statement about a specific, fairly
   aggressive reweighting, and a milder one (0.5) was not tested.
5. **`r2p_z16`'s control role and its positive result are conflated.** It was
   designed to isolate dilation, so its win is real but its mechanism is
   untested: it could be depth, it could be the extra nonlinearity, it could be
   a favourable optimisation change. Section 6 tests survival, not mechanism.
