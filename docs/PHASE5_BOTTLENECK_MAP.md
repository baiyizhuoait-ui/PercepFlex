# Phase 5 - Bottleneck Map

This is the deliverable that answers Phase 5's question: "what is each task
bottlenecked by, and is it task-specific, reproducible, causal?".

The map is built from four layers, each one independently checkable against
the artefacts in `experiments/phase5/`:

  L1  Model-free geometric and structural measurements. No model involved;
      the ceiling and the structure of the labels are measured directly.
      Source: `phase5_resolution.csv`, `phase5_da_analysis.csv`,
      `phase5_detection_analysis.csv`, `phase5_error_geometry.csv`,
      `phase5_label_geometry.csv`.

  L2  Pre-registered diagnostic interventions. Where the cheap measurement
      could not separate two hypotheses, an intervention was designed with
      the prediction, control and falsification written down BEFORE the
      number was read. Source: `phase5_hypothesis_matrix.csv`,
      `docs/PHASE5_LANE_PROBE_PREREGISTRATION.md`,
      `docs/PHASE5_LITERATURE_REGISTER.md`.

  L3  The probe (intervention experiment). A single reading cannot establish
      a finding in this project (two prior 4-epoch readings reversed at
      20). Where the probe is the source, the standing caveat is repeated.
      Source: `phase5_lane_probe_results.csv`,
      `phase5_lane_probe_errorgeom.csv`,
      `phase5_lane_probe_decision.txt`.

  L4  Phase 4B - remaining-factor exclusion. Anything Phase 5 could not
      separate is queued for an ordered set of probes, each with its own
      literature check. Source: `docs/PHASE4B_PLAN.md`.

## 1. Per-task bottleneck profile

### 1.1 Lane

| property | value | source |
|---|---|---|
| target stroke width (model input) | 1.99 px | label geometry, n=287 val |
| target stroke width (train vs val) | 3.98 / 3.97 px (ratio 0.997) | label geometry |
| 1/8 block-fill reference | 0.1853 | resolution ladder |
| 1/4 block-fill reference | 0.3220 | resolution ladder |
| 1/2 block-fill reference | 0.5721 | resolution ladder |
| 4 px perfectly-localised predictor | 0.97 | label geometry |
| baseline r2u_z16: prec / rec / IoU / area | 0.2049 / 0.6706 / 0.1845 / 3.55x | error geometry, r0 |
| published BDD100K SOTA range | 26.5-28.8 (YOLOP, A-YOLOM(s), YOLOPv3) | literature register |

**Bottleneck: the operating grid, 1/8.**

Two pieces of evidence. First, the model sits statistically on the 1/8
block-fill ceiling across all three training regimes (r0 / r2 z16 / r2
z128). Second, painting the lane at 1/4 would jump the ceiling to 0.3220
and at 1/2 to 0.5721 - and the SOTA models that score 26.5-28.8 operate
their lane head finer than 1/8, putting the realistic target for a 1/4 head
at 0.20-0.26 (not 0.3220, which is the oracle bound).

**Second-order bottleneck: continuity, captured in H17.** Precision 0.2049
> 1/8 block-fill 0.1854 means the model is slightly more parsimonious than
the trivial strategy; recall 0.6706 means it pays for that parsimony by
missing a third of the target. The 1/4 probe splits these: if precision
rises while recall stays at 0.67, the residual is a separate (continuity)
problem - exactly the regime YOLOPX's Polarized Self-Attention module is
designed for ("long-range spatial dependencies critical for lane line
continuity"), and SCNN's slice-by-slice message passing addresses
("particularly effective for capturing long and narrow structures").

**What is ruled out.**
- Channel capacity at 1/8 (H6): the lch64 cell at equal FLOP.
- Annotation width mismatch between splits (H8 in its train/val form):
  train and val strokes agree to 0.3%, and a correctly sized perfectly
  localised stroke scores 0.97 against the label.
- Incompatible per-class widths: all 11-14 lane classes measure 3.3-4.1 px;
  the heterogeneity is semantic, not geometric.

**Decision of the probe (L1-L4, all PASS) and its standing caveat.**
All three cells are in (`phase5_lane_probe_decision.txt`):

    cell        lane_fg   d_lane  x_noise   mAP50   da_fg
    r2u_z16      0.1765   +0.0000   0.00   0.2411  0.7146
    l14up_z16    0.1894   +0.0129   2.02   0.2523  0.7283
    l14f1_z16    0.1916   +0.0151   2.36   0.2402  0.7370
    lch64_z16    0.1778   +0.0013   0.20   0.2373  0.7292

- L1 (primary): l14f1 crosses the pre-set threshold 0.1829. PASS.
- L2: l14up (upsample-only, no new information) gains 2.02x noise -
  operating-point resolution alone moves lane. PASS.
- L3 (spatial vs channel per FLOP): l14f1 +0.0270 lane_fg/GFLOP vs
  lch64 +0.0031 - 8.7x. PASS: lane is spatially constrained, not
  channel-constrained at Z=16.
- L4: det and DA controls all inside 1x noise. PASS.

Verdict: **H5b SUPPORTED (provisional)** - the lane bottleneck is the
absence of 1/4-resolution information. H6 (channel capacity) is
subordinate: 64 channels at 1/8 buy 0.20x noise. ACTION per prereg:
l14f1_z16 extended to 20 epochs under a rule registered before the run
(`PHASE5_E20_CONFIRM_REGISTRATION.md`: >= 2x noise vs the committed
r2_z16 e20 baseline 0.1943 confirms H5b; 1x-2x is weak; < 1x reverts
H5b to OPEN, probe A precedent). Standing caveat binds: two prior
4-epoch findings reversed at 20 epochs; nothing here is established
until that run lands.

**Probe error geometry (content region, e4 checkpoints, internally
comparable - the STEP 3 baseline row is a 20-epoch model and must not be
subtracted from these):**

    cell        fg_iou  prec   recall  area    k=1    k=2
    l14up_z16   0.1756  0.2073 0.5765  3.09x   0.3429 0.3875
    l14f1_z16   0.1807  0.2019 0.6396  3.62x   0.3585 0.4016
    lch64_z16   0.1626  0.1881 0.5898  3.46x   0.3290 0.3932

The 1/4 lateral buys RECALL (+0.063 over upsample-only) at flat
precision - it finds lane pixels the 1/8 head cannot see, it does not
localise them better (precision unchanged, area inflates 3.09x ->
3.62x). This is the continuity half of H17 showing a pulse; the
quantisation half (precision ~0.20 everywhere) is untouched, as
predicted: 2 px strokes still cannot be drawn thinner than the head
allows.

**20-epoch confirmation (registered in f775718 before the run).**
Eval metric: lane_fg 0.2192 vs committed baseline 0.1943 = +0.0249 =
1.95x noise -> **WEAK SUPPORT** per the pre-set bar (CONFIRMED needed
2x = +0.0256; missed by 0.0007). Controls clean: det +0.24x, DA
+0.08x. Same-budget error geometry (both e20, identical code):

    metric          r2_z16 e20   l14f1 e20    delta
    EG fg_iou       0.1843       0.2074       +0.0231
    vs block-fill   -0.0012      +0.0220      FIRST model above it
    precision       0.2036       0.2294       +0.0258
    recall          0.6886       0.7034       +0.0148
    pred/gt area    3.73x        3.41x        tighter
    fg_iou tol1     0.3821       0.4122       +0.0301

At 4 epochs the gain was recall-only; at 20 both recall and precision
rose and the model clears the 1/8 block-fill reference for the first
time. Direction and rough magnitude survived (probe A precedent did
not recur). The eval-metric verdict stays WEAK because the rule was
fixed in advance - the honest next question is whether 1.95x is seed
noise (2 more seeds, ~5h GPU) or a true effect just under the bar.

**4B-2 feature provenance (linear probe on the frozen e20 baseline
encoder, 150 tri_val images, cell-level IoU at the 1/8 grid):**

    probe level      lane     da
    enc_1/4          0.0786   0.2925
    enc_1/8          0.1136   0.3439
    enc_1/16         0.1904   0.5060
    enc_1/32         0.2129   0.7538
    enc_1/4 native   0.0450   0.2634

Counter-intuitive and decisive: lane location is LINEARLY MORE
decodable from the deep maps than from the 1/4 map - the naive story
("the information is already in f1 and the head fails to read it") is
wrong. Lane location is a global, context-level property (long thin
structures spanning the image); deep features with large receptive
fields carry it; local 1/4 texture barely does (native probe 0.0450).
DA shows the same monotone shape with 1/32 at 0.7538 - independent
confirmation of H18 (DA is region-level semantic).

Refined mechanism for the l14f1 gain: the encoder already knows WHERE
lanes are (deep maps, through Z); what the 1/8 output grid cannot do
is DRAW them at ~2 px width. The 1/4 lateral supplies drawing
resolution, not lane knowledge. This matches the 20-epoch EG: the
block-fill reference crossed with precision AND recall both up.

Caveat: a linear probe UNDERESTIMATES information in early features
(it cannot form nonlinear edge-grouping). The claim recorded here is
deliberately narrow: lane location is not LINEARLY present in the 1/4
map. Direction this opens: a lane head that reads deep context at a
high-resolution output grid (deep-upsampled + lateral) rather than
reading early features - a Phase 6 architecture candidate, not a
Phase 4B experiment.

**STEP 7c seed replication (rule registered in a185e48 before the runs).**

    seed  lane_fg   d_lane   x_noise   mAP50   da_fg
    0     0.2192   +0.0249    1.95   0.3612  0.7678
    1     0.2136   +0.0193    1.51   0.3553  0.7698
    2     0.2212   +0.0269    2.10   0.3600  0.7726
    mean            +0.0237    1.85   sd 0.0039  sem 0.0023

VERDICT: **WEAK SUPPORT**. The bar (2x = +0.0256) is missed by 0.84 SEM.
But the replication bought a sharper answer than pass/fail: the effect is
**not seed noise**. d_mean sits 10.42 SEM away from zero and every one of
the three seeds clears the 1x floor. The honest reading is a real, small
effect whose magnitude happens to line up with an arbitrary 2x threshold -
not a 4-epoch reading inflated by luck. H5b stays provisional and nothing
is built on it alone.

### 1.2 Drivable area

| property | value | source |
|---|---|---|
| 1/8 block-fill reference | 0.8801 | resolution ladder |
| 1/4 block-fill reference | 0.9410 | resolution ladder |
| baseline r0_z16: fg / prec / rec / area | 0.7699 / 0.7795 / 0.9841 / 1.319x | error geometry |
| 1 px boundary band share | 2.4% | label geometry |
| 1/8 cell (16 native px) boundary band share | 28.4% | label geometry |
| far / mid / near band gap (post letterbox crop) | within noise | DA bands, Step 2 |
| tolerance curve tol0 -> tol8 | 0.7699 -> 0.7951 (+0.025) | error geometry |

**Bottleneck: region-level semantic identification (H18), not resolution
and not boundary localisation.**

The model is 0.11 **below** its own 1/8 block-fill reference (0.7699 vs
0.8801). It cannot reach the score of the trivial strategy "paint every
touched 1/8 cell". Raising the bar (a finer head) cannot be the fix for a
model that has not reached the existing bar.

Recall is saturated at 0.9841; missing regions is not the failure.
Tolerance curve is flat - a 8 px slack only adds 0.025. A thin misaligned
rim is not the failure either; the boundary band at the model's real
granularity holds only 28.4% of the foreground.

That leaves over-inclusion: distinct regions that the model paints as
drivable but the label does not, e.g. opposite lanes beyond a median,
turn pockets, road-like shoulders. Phase 4B STEP 4 (anatomy of false
positives) measures this directly and is the next step before any training.

**What is ruled out.**
- Resolution (H10): the model has not reached the resolution-defined bar.
- Channel capacity (H9): same argument.
- Far-field semantics (H11): the earlier apparent far-field collapse was
  a letterbox-padding artefact. After cropping to the content rows
  140-500, far / mid / near bands agree within noise and R0 is not better
  than R2 in the far band (H11-alt SUPPORTED).
- Boundary refinement as the operative fix (not formally hypothesised
  but the obvious next guess): the flat tolerance curve says a thin
  rim is not the failure.

### 1.3 Detection

| property | value | source |
|---|---|---|
| small / medium / large recall | 0.64-0.69 / 0.90 / 0.95 | detection analysis |
| crowding effect on recall | none | detection analysis |
| effect of Z 16 -> 128 on small recall | +0.0235 | detection analysis |
| effective rank of Z (z16 / z32 / z128) | 42.5% / 37.1% / 24.9% | Phase 4A |
| gradient-magnitude imbalance (det / total) | 73-77% | Phase 4A, probe C |
| baseline mAP50 / mAP50_95 (r0_z16, 4ep) | 0.2411 / 0.0728 | error geometry |

**Bottleneck: small-object feature resolution, not capacity.**

The largest gap in the diagnostic is between small and medium recall
(+0.21-0.26). Crowding is irrelevant. Z width is a weak lever (+0.0235
on small for a ~10x increase in shared capacity). Effective rank goes
**down** with Z width, against the "more capacity" reading.

The most documented published lever is a P2 (1/4) detection head, which the
R0 head here does not have - it reads encoder F2/F3/F4 directly, but the
finest is 1/8. RSO-YOLO on BDD100K: P2 head +4.6 mAP50 / +1.8 mAP50:95,
the largest single-module gain in their ablation. MST-YOLO on BDD100K:
APsmall 0.124 -> 0.209, ARsmall 0.251 -> 0.423. YOLOPv3 argues
multi-task driving perception "fail[s] to fully leverage multi-scale
high-resolution features... not conducive for the network to detect small
objects that are prevalent in intelligent driving scenarios". The
intervention is published, well-quantified, and the strongest remaining
candidate.

**What is ruled out.**
- Z capacity (H1): a 4x increase in shared width buys only +0.0235 small
  recall and reduces effective rank.
- Gradient conflict (H13): probe C rebalanced detection to 20% weight and
  saw -13x noise of mAP50 for +0.27x of DA. Whatever caps DA, it is not
  detection's gradient share.
- Crowding (H3): not a factor.
- Receptive field (H3): probe B at 4-epoch showed no effect and 20-epoch
  showed -0.10x; H3 is closed.

## 2. Per-task intervention matrix

```
task  intervention                          gain / cost / status
lane  1/4 head, no new information           lane_fg 0.1765 -> 0.1894 (+0.0129, 2.02x noise);
                                             +0.534 GFLOPs; L4 control passes (1 of 3)
lane  1/4 head + 1/4 lateral                 expected gain from the lateral; probe running
lane  channels 32 -> 64 at 1/8               equal-FLOP control; probe running
lane  edge-aware / continuity (YOLOPX PSA)   outside the scope of the Phase 5 budget; Phase 4B
DA    finer head / more channels             rejected: 0.11 below the 1/8 bar
DA    edge-aware aux supervision             outside the scope; Phase 4B (conditional)
DA    boundary refinement (AURASeg-style)    not applicable: flat tolerance, not a thin rim
DA    semantic / context                     Phase 4B STEP 4
det   P2 (1/4) detection head                MST-YOLO +68% ARsmall; RSO-YOLO +4.6 mAP50; Phase 4B
det   Z width 16 -> 128                      +0.0235 small recall, -17.5% effective rank; not worth it
det   detection gradient share (probe C)     -13x mAP50 for +0.27x DA; rejected
```

## 3. Where the answer still cannot be read

1. The 1/4 lane probe is the cleanest experiment Phase 5 has. Two of the
   three cells are still training. The 4-epoch verdict is provisional; the
   20-epoch confirmation must follow any 4-epoch pass.
2. DA's region-level semantic hypothesis (H18) is the only DA factor that
   survives; it is not yet measured. 4B-4 will measure it (size, distance
   to GT, border-touching).
3. Detection has the largest published lever (P2 head) and the smallest
   project-side data: ARsmall by category and by Z is recorded, but
   whether the 1/4 head here buys what the literature reports is a
   separate experiment (4B-3), to be registered before running.
4. Lane continuity (H17) is opened by the precision/recall split; the
   experiment to test it (SCNN message passing, YOLOPX PSA) lies outside
   the Phase 5 budget. It is logged as a follow-up.

## 4. What this is not

- It is not a "winner" architecture. The Phase 5 brief forbids it
  ("最终架构只能后置").
- It is not a paper that says "lane is X, DA is Y, det is Z". Every
  "is" in this map is "the cheapest available measurement says, and the
  next probe will test". Where the probe has not yet run, the verdict
  row is blank.
- It is not a contribution that any of the three interventions is novel.
  They are all published. The contribution is the measurement - the
  numbers attached to the per-task profile and the decision rules that
  attached them.

---

## 4B-3. Detection — the binding constraint was on the supervision side

Added after `danc_z16` (4 epochs) was measured. All of the diagnosis below was
made **without training anything**; the training run only tested the
prediction.

### What the zero-training diagnostics found

- **Assignment coverage.** Under this project's own rule
  (`0.5 < box_px / anchor_px < 2.0` on both axes, the ratio is stride-free),
  **48.5% of GT boxes receive no positive assignment at all** (49.6% of small
  boxes). Re-clustering the anchors with IoU-k-means on `tri_train` drops that
  to 3.7%.
- **Honesty guard.** By the community standard (YOLOv5 AutoAnchor, ratio
  threshold 4.0, target BPR > 0.98) the same anchors give only **0.4%**
  zero-match, i.e. they are *adequate*. The hole is produced by the
  interaction of a weak prior with this project's stricter threshold. Both
  numbers are reported; quoting only one would overstate or hide the finding.
- **Prior quality.** Default 9 anchors: mean best IoU 0.407, 15.0% of boxes
  below 0.30, and **aspect-flipped with respect to the data** (anchors are
  h/w = 2.5-3.0 tall; BDD100K small vehicles are h/w = 0.85 near-square).
  k-means 9: mean best IoU 0.680, 1.1% below 0.30.

### What the intervention showed (D1 / D4 / D5)

| cell | mAP50 | mAP50_95 | da_fg | lane_fg | params | GFLOPs |
|---|---|---|---|---|---|---|
| `r2u_z16` (baseline, committed) | 0.2411 | 0.0728 | 0.7146 | 0.1765 | 0.2014 M | 1.0796 |
| `danc_z16` (anchors only) | **0.3954** | **0.1491** | 0.7219 | 0.1756→0.1764 | 0.2014 M | 1.0796 |

- **D1: +0.1543 mAP50 = 10.57x noise** (noise 0.0146). mAP50_95 doubles.
- **D4:** lane_fg -0.02x noise, da_fg +0.18x noise — the change stayed
  detection-only, as required.
- **D5:** identical params and FLOPs. The win is genuinely free.

A zero-cost, zero-parameter change buying 10x noise is categorically different
from every capacity-side intervention tried in this project (Z width, encoder
capacity, lane resolution). It says the detection number was being held down by
**how the task was supervised**, not by what the model could represent.

### The mechanism is NOT the one predicted (D2)

The preregistration predicted the gain would land on small-object recall.
Measured with `scripts/phase5_det_size_recall.py` over all 108,363 GT boxes:

| bucket | n_gt | base recall@.5 | danc recall@.5 | base AP@.5 | danc AP@.5 | share of weighted AP gain |
|---|---|---|---|---|---|---|
| small (<32 px) | 67,566 | 0.6217 | 0.6486 | 0.1326 | 0.1496 | **14.4%** |
| medium (32-96 px) | 29,559 | 0.8366 | 0.9334 | 0.2315 | 0.4263 | **72.1%** |
| large (>=96 px) | 11,238 | 0.9132 | 0.9588 | 0.2919 | 0.3879 | 13.5% |

Small recall does rise (+0.0269), so **D2 formally passes** by the rule as
written. But **72% of the gain is on medium objects and only 14% on small** —
the boxes that were supposedly never supervised are not where the improvement
landed. The honest label is *formally confirmed, substantively weak*.
Interpretation: the dominant channel is **assignment/box quality**, not the
small-object coverage hole. This is consistent with ATSS (L6), which finds that
how positives are defined dominates everything else, and with the anchor-shape
counter-evidence in L6 (aspect-ratio changes wash out because regression pulls
boxes back onto the GT).

Two consequences carried forward:

1. **H20 (grid resolution binds) is now a priori less likely.** The sub-cell
   statistic motivated a stride-4 level for small objects; if small objects are
   not where the gain is, a stride-4 level should buy little. `dp2a_z16` is the
   test and is read mechanically by D3.
2. **Previous detection numbers are confounded.** Every earlier phase measured
   detection with the default anchors. Any conclusion that used detection as
   its most-sensitive metric (Phase 3A encoder monotonicity, Phase 3C budget
   allocation) was measured through a partially broken supervision channel. The
   direction of those conclusions is not necessarily wrong — the bias was
   constant across cells — but their *effect sizes* are not trustworthy, and
   this is stated here rather than quietly carried forward.

### Cost note

`dp2a` (stride-4 level) costs +594 params. The literature baselines quoted in
L6 pay +29% GFLOPs for a P2 head; this model's head is small enough that the
cost is a few percent.
