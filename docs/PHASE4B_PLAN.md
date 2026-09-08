# Phase 4B - remaining-factor exclusion

Standing instruction for this phase: every experiment is preceded by a
literature check that predicts the outcome and records whether the technique is
already published. Those checks are in `docs/PHASE5_LITERATURE_REGISTER.md`
and are dated before the runs they cover. A technique being published kills the
novelty claim, not the experiment.

This phase runs **after** the Phase 5 deliverables. Phase 5 asks what each task
is limited by; Phase 4B takes the factors Phase 5 could not separate and rules
them out one at a time until only one is left per task.

---

## 0. What Phase 5 already closed, so 4B does not re-test it

| task | closed | how |
|---|---|---|
| lane | resolution-bound at 1/8 | block-fill reference 0.1853 vs trained models 0.1843-0.1882; targets 3-4 px against an 8 px addressable unit |
| lane | not channel-bound at 1/8 | `lch64` control at equal FLOP (result pending) |
| DA | **not** resolution-bound | 1/8 -> 1/4 block-fill headroom is only 0.11; recall already 0.97-0.99 |
| DA | far field is not the problem | the earlier "far-field collapse" was letterbox padding; after cropping to rows 140-500 the bands agree within noise |
| det | small objects dominate | recall 0.64-0.69 small vs 0.90 medium / 0.95 large |
| det | crowding is not the problem | isolated / near / overlapping recall differ by less than noise |
| det | Z capacity is a weak lever | z 16 -> 128 moves small by +0.0235 and nothing else |
| all | gradient conflict is not the bottleneck | probe C: cos ~ 0 between task gradients; rebalancing changed nothing |

---

## 1. Probes, in the order they will be run

Ordering rule: anything answerable without training runs first, and is only
escalated to a training run if the cheap answer says the factor is live.

### 4B-1 Lane label geometry - is part of the lane deficit the label?

- **Question.** Are BDD100K lane strokes the same width in train and val? If
  not, a perfectly localised model trained on the train width is bounded at
  `min/max` of the two widths when scored on val, and no architecture can pass
  that bound.
- **Prediction from literature.** The 8 px-train / 2 px-test figure is quoted
  in the lane-detection literature as a known evaluation unfairness; CULane
  sidesteps it by drawing both prediction and GT as 30 px curves before
  matching. So a mismatch is plausible and would be large.
- **Method.** Distance transform over 300 sampled masks per split; estimators
  `4 * mean(EDT)` and `2 * p99(EDT)`; plus a perfect-localisation IoU curve
  over assumed predicted stroke widths. No model, no GPU.
- **Falsification of the label hypothesis.** train and val p99 widths agree
  within ~10%.
- **Status.** Running. Script `scripts/phase5_label_geometry.py`, output
  `experiments/phase5/phase5_label_geometry.csv`.

### 4B-2 Feature provenance by linear readout - is 1/4 information actually there?

- **Question.** H5b says the 1/4 information never reaches the lane branch. But
  "the encoder has a 1/4 map" is not the same as "that map supports lane". A
  frozen-feature linear readout separates them: if a ridge/logistic classifier
  on s1 (1/4) predicts lane far better than one on F2 (1/8), the information is
  present and the architecture is throwing it away. If both readouts are equal,
  s1 does not carry lane and giving it to the head cannot help.
- **Why it matters.** It gives an **upper bound** for the `l14f1` result before
  the 20-epoch confirmation. If the readout says s1 supports lane at IoU 0.30
  and `l14f1` lands at 0.20, the gap is head/optimisation; if the readout says
  0.20 and `l14f1` lands at 0.20, the lateral is already delivering everything
  s1 has and going finer is pointless.
- **Same test, free, for detection.** Run the readout for small-object presence
  at 1/4 vs 1/8. This is the cheap version of the P2 question in 4B-3.
- **Prediction.** A-YOLOM argues lane "requires low-level and multi-scale
  features", so the 1/4 readout should beat the 1/8 readout for lane. For DA
  the two should be close, because DA is blob-like (YOLOPv2) and the block-fill
  ladder already showed only 0.11 of resolution headroom.
- **Falsification.** 1/4 readout <= 1/8 readout + noise for lane.
- **Cost.** CPU only, one pass to cache features. No training of the model.

### 4B-3 Detection: a 1/4 rung for the detection head

- **Question.** Is small-object recall limited by the absence of a 1/4 feature
  in the detection head? The head currently reads F2/F3/F4 - uncompressed, but
  finest is 1/8. There is no P2.
- **Prediction from literature, and it is a strong one.** RSO-YOLO's BDD100K
  ablation: the P2 head is the largest single-module gain in the whole table
  (+4.6 mAP50, +1.8 mAP50:95, vs +0.8/+0.7 for BiFPN). MST-YOLO on BDD100K:
  `APsmall 0.124 -> 0.209`, `ARsmall 0.251 -> 0.423`, miss rate 5.62 -> 3.13.
  YOLOPv3 makes the same argument inside multi-task driving perception.
  Expect small-recall to move more per FLOP than widening Z did.
- **Cells.** `detp2_z16` (add the 1/4 rung), `detch_z16` (equal-FLOP channel
  control so the comparison is per unit compute, not per module), baseline
  `r2u_z16` reused.
- **Control and confound.** Adding a rung adds parameters and FLOPs, so the
  channel arm is mandatory. Also watch detection's effect on DA and lane
  through Z - probe A already showed a lane-only change moved detection at 20
  epochs, so the reverse coupling is expected and will be recorded as coupling,
  not as a broken cell.
- **Falsification.** small recall moves less than 1x noise, or the channel
  control matches it per FLOP.
- **Cost.** 2 cells x 4 epochs, ~1 h. 20-epoch confirmation only if the 4-epoch
  reading clears 2x noise - and per project rule, a 4-epoch result can justify
  the 20-epoch run but cannot establish the finding.

### 4B-4 DA: anatomy of the false positives

- **Question.** DA loses on precision (0.72-0.87) with recall saturated
  (0.97-0.99) and paints 1.32-1.37x the true area, yet its tolerance curve is
  flat. Flat tolerance means the error is **not** a thin misaligned rim. So
  what are the false-positive pixels? Distinct blobs (a context/semantics
  error) or something structured?
- **Method, zero-training.** Connected-component size distribution of the FP
  regions, distance from each FP component to the nearest GT foreground, and
  whether FP components touch the image border. Blobs far from GT = semantic
  confusion; border-touching blobs = boundary/context.
- **Prediction from literature.** GDA-RoadSeg names "weak edge-region
  segmentation" and says decoders that fuse with a plain 1x1 concat "weaken
  shallow-detail representations... which inflates the estimated drivable
  area"; AURASeg splits the problem into region reconstruction and boundary
  refinement. Both predict region-level inflation rather than a thin rim, which
  is what our flat tolerance curve already shows.
- **Ruled out already.** The "alternatively drivable" class is not a hidden
  ceiling: `datasets/bdd100k.py` maps foreground to `>0`, so direct (1) and
  alternative (2) are both positive. No class-mediated penalty exists.
- **Falsification.** FPs are dominated by a <=2 px rim around GT - then it is
  boundary precision after all and the boundary-suppression direction is live.
- **Cost.** CPU only, reuses existing checkpoints' predictions if cached, else
  one inference pass.

### 4B-5 Conditional, only if 4B-4 says boundary: DA edge-aware auxiliary supervision

- **Precedent.** GDA-RoadSeg's edge-aware auxiliary branch with generated edge
  supervision; AURASeg's Sobel-informed residual boundary refinement module.
- **Position.** This is a *probe*, registered in advance with a falsification
  bar, not a model improvement. If it moves DA foreground IoU by less than 2x
  noise, the boundary story dies and DA is declared near its ceiling for this
  budget.
- **Cost.** 2 cells x 4 epochs.

### 4B-6 Conditional: detection label assignment for small objects

- **Precedent.** YOLOPv3 argues hand-crafted assignment "results in ambiguous
  matching between the prior anchors and the ground truth, thus impairing
  detection performance" and replaces it with dynamic assignment.
- **Only if 4B-3 underperforms its literature prediction.** Then the small
  object may be present at 1/4 but not being assigned a target during training,
  which is a supervision problem rather than a resolution one.
- **Cost.** 1 cell x 4 epochs plus a diagnostic on assignment statistics, which
  is free and should be done first: count, per GT small box, how many
  locations are assigned positive. If small boxes receive systematically fewer
  positives, that is measurable before spending a run.

---

## 2. GPU budget and discipline

| step | cells | epochs | approx |
|---|---|---|---|
| 4B-1, 4B-2, 4B-4 | - | - | CPU only |
| 4B-3 | 2 | 4 | ~1 h |
| 4B-5 (conditional) | 2 | 4 | ~1 h |
| 4B-6 (conditional) | 1 | 4 | ~0.5 h |
| confirmation of whatever survives | <=2 | 20 | ~4 h |
| final 3-seed | 1 | 20 | ~6 h |

Rules carried from the brief and from two prior failures in this project:
4-epoch readings do not establish findings (probe A: +0.26x at 4 epochs,
-2.53x at 20; reconstruction depth: +4.47x at 4 epochs, -0.10x at 20). Three
seeds are spent once, at the end, on the single surviving change. No seed is
added to rescue a result.

## 3. What would make this phase a failure

If every probe lands inside noise, the honest conclusion is that at 0.2 M
parameters and this budget the three tasks are each at a different kind of
ceiling - lane at geometry, DA at region semantics, detection at supervision -
and none of them is movable by the cheap levers. That is a publishable
negative result only if every probe was registered in advance with its
falsification bar, which is why section 1 is written before section 1's numbers
exist.
