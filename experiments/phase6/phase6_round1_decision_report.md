# Phase 6A — Decision Report (overnight 2026-09-09 22:00 → 09-10 05:20)

**Decision**: PROCEED with the bottleneck-aware asymmetric architecture (combo)
as the single candidate for 3-seed confirmation. Lane-spatial is the Phase 6B
deepening direction. DA far-field modeling is REJECTED as a design target.

---

## 1. Evidence → Architecture chain (what Phase 6A established)

### 1.1 Composition result (EXP-04/GATE-6A.1/P6A-STEP2c) — the Phase 6A headline

At 20 epochs, seed 0, R2 family:

| model | params | FLOPs | det mAP50 | lane mIoU | DA mIoU |
|---|---|---|---|---|---|
| r2_z16 baseline (uniform) | 0.2014M | 1.0796G | 0.3543 | 0.5847 | 0.8488 |
| + supervision only (danc anchors) | 0.2014M | 1.0796G | **0.4982** / 0.4960 (s2) | 0.5824 | 0.8567 |
| + spatial only (l14f1 lane 1/4) | 0.2019M | 1.6391G | 0.3612 | **0.5988** | 0.8533 |
| **combo (both)** | 0.2019M | 1.6391G | **0.5047** | 0.5962 | 0.8551 |

The combo matches or exceeds the element-wise max of the two single
interventions on all three tasks, at +0.56G FLOPs total (+52% vs baseline,
all of it on the lane branch, none on det/DA). **Task-directed asymmetric
resources compose without interference.** det even ticks up (0.5047 vs
0.4982/0.4960) — within seed noise (range 0.4960-0.5047 ≈ 1.4 SEM),
consistent with zero det-side change.

### 1.2 EXP-01: spatial vs channel capacity (strict FLOPs parity)

Two channel-arm controls, both losing to the spatial arm on lane_fg:

| arm | FLOPs | lane_fg (4ep) | lane_mIoU (4ep) |
|---|---|---|---|
| l14f1 (fewer ch @ 1/4) | 1.6391G | **0.1916** | **0.5839** |
| lch72 (more ch @ 1/8, 0.3% parity) | 1.6336G | 0.1831 | 0.5797 |
| lch64 (cheaper arm, Phase 5) | 1.4933G | 0.1778 | 0.5765 |

Individually each gap is sub-noise at 4ep (0.0085 ≈ 0.66x noise; lch64 gap
0.0138 ≈ 1.1x noise), but the sign is consistent across both controls and
the 20ep l14f1 row is positive (1.95x noise). Supporting negative result:
lane label widening HURTS (lane8 20ep: 0.5455 vs 0.5847 baseline, -3.7pt vs
danc) — the lane bottleneck is NOT supervision width and NOT channels; the
residual mechanism is spatial addressability.

**Working statement (H-35, provisional)**: under extreme compression,
channel capacity and spatial capacity are non-interchangeable per task;
tasks have a binding order (which dimension saturates first). Detection
binds on supervision/assignment; lane binds on spatial precision; DA binds
on neither (saturated).

### 1.3 EXP-03: assignment analysis (zero-training)

Old aspect-flipped anchors left ~50% zero-positive holes at ALL sizes
(small 51.1%, medium 44.3%, large 51.7%); k-means set: 5.4/1.6/0.6%.
Mean best-anchor IoU 0.405 → 0.678. The hole is a uniform supervision
handicap, not a small-object-only issue — consistent with D2 (gain lands in
medium bucket where headroom existed).

**H-34 (capacity–supervision coupling)** remains a HYPOTHESIS: we have the
assignment fix (+0.14 mAP50 abs at 20ep, zero cost) and its cross-config
replication (lane8 0.4870, da14 0.4076@4ep), but no anchor×capacity
factorial. Do not promote without it.

### 1.4 EXP-05: DA — do not build far-field machinery

Phase 5 established (letterbox-crop corrected): far/mid/near bands agree
within noise; the model sits 0.11 BELOW its own 1/8 block-fill reference
(0.7699 vs 0.8801) — it fails region-level semantic identification, not
resolution, not boundary, not channels, not far-field. Any DA-side
architecture work before closing H-18 is unjustified.

## 2. Gates and discipline

- GATE-6A.1 (combo 4ep det≥0.30 ∧ lane≥0.55): PASSED (0.3984 / 0.5861) → 20ep ran.
- GATE-6A.2 (lch72 4ep lane_fg>0.1916 → 20ep): NOT triggered (0.1831) → no run. Saved ~2.5h.
- No 4ep result is cited as a conclusion. 20ep single-seed = confirmation tier;
  3-seed not yet run (next step).
- Literature collision (Round 0): k-means anchors are YOLO-standard → not
  claimed as novelty; YOLOPv2 assigns tasks to feature levels heuristically →
  our difference is measurement-driven derivation + budget-parity controls +
  composition test; EfficientNet balances dims single-task → per-task binding
  order under multi-task budget is the open territory.

## 3. Level self-assessment (per §31)

- L1 new bottleneck relationship: YES (task-conditioned dimension binding;
  composition validity) — 20ep single-seed.
- L2 controlled-intervention reproduction: YES for both ingredients
  (danc 2 seeds; l14f1 4ep+20ep); combo 1 seed.
- L3 cross-architecture: PARTIAL (R2 family only).
- L4 architecture principle derived: YES (asymmetric, measurement-driven).
- L5 model beats baseline at fixed budget: YES (combo > baseline on all
  three tasks; det +42.5% rel, lane +2.0%, DA flat) — but vs *uniform*
  alternatives at parity, only the lane-arm comparison is complete.

## 4. Next actions (ordered)

0. ~~combo 3-seed~~ **DONE (10:03)**: det {0.5047, 0.4931, 0.4964} mean 0.4981;
   lane {0.5962, 0.5938, 0.5952} mean 0.5951; DA mean 0.8550. det mean equals
   danc-only range (0.4982/0.4960); lane mean within 0.29x noise of l14f1's
   single-seed 0.5988. **Composition confirmed at the 3-seed tier (L2 strong).**
2. Anchor×capacity factorial (z16 vs z32 with/without k-means, 4ep×4) → H-34
   tier test (zero-training pre-analysis already done).
3. Lane: EXP-02 second variant only if a strictly cheaper 1/4 path exists
   (l14f1 costs +52% FLOPs; a prune-then-refine variant could halve it).
4. DA: freeze architecture work; H-18 semantic-identification study is the
   only open lane (data-side).
5. Paper-track decision after (1)+(2): mechanism review fallback remains
   available and fully evidenced.

## 5. Artifacts

- experiments/phase6/phase6_results.csv (combo4/combo20/lch72 rows)
- experiments/phase6/phase6_exp3_assignment.csv
- experiments/phase6/phase6_candidate_cost.csv, phase6_experiment_registry.csv
- experiments/phase6/phase6_architecture_hypotheses.csv, phase6_novelty_matrix.csv
- docs/PHASE6_BOTTLENECK_TO_ARCHITECTURE.md
- Invalid-run forensics preserved: experiments/phase4a/*.INVALID_anchor_mismatch
