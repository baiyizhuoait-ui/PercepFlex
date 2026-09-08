# Phase 4B-3 preregistration — detection: anchor prior / assignment coverage

Registered **before** any detection cell is trained. The decision rule is
fixed here and is applied mechanically afterwards.

## 1. What the zero-training diagnostics found

All of the following was measured without training a single step
(`scripts/phase5_anchor_coverage.py`, `scripts/phase5_anchor_verify.py`,
`scripts/phase5_anchor_kmeans.py`, `scripts/phase5_det_anchorsize.py`).
The anchor-coverage number was computed twice, once by re-deriving the
assignment rule and once by calling `YOLOLoss.build_targets` directly; the
two agree to within 0.6 points.

**Box population** (tri_val, 500 images; tri_train, 3000 images):

- 68.1% of GT boxes are "small" (side < 32 px on the 640x640 canvas).
- median small object = 14.1 x 12.0 px, aspect h/w = 0.85 (near square).
- at stride 8 (the finest level the detector has) a small box spans on
  average 2.09 x 1.71 cells; **29.8% have their minimum dimension under one
  cell**. At stride 4 that becomes 4.18 x 3.42 and 2.5%.

**Anchor prior quality:**

| anchor set | mean best anchor IoU | frac boxes < 0.30 |
|---|---|---|
| current 9 (project default) | 0.407 | 15.0% |
| IoU-k-means on tri_train | **0.680** | 1.1% |

**Assignment coverage** — `build_targets` keeps a (level, anchor) pair only
when `0.5 < box_px / anchor_px < 2.0` on both axes. The stride cancels in
that ratio, so coverage depends on the anchor's PIXEL size only:

| anchor set | zero-match, thr=2.0 (project) | zero-match, thr=4.0 (YOLOv5 BPR) |
|---|---|---|
| current 9, all boxes | **48.5%** | 0.4% |
| current 9, small only | **49.6%** | 0.6% |
| k-means 9, all boxes | 3.7% | 0.0% |
| k-means 9, small only | 5.0% | 0.1% |

**The honest reading, and the reason both thresholds are reported.** By the
community standard (YOLOv5 AutoAnchor uses a ratio threshold of 4.0 and
targets BPR > 0.98) the current anchors are *adequate*: 0.4% zero-match.
The 48.5% hole is produced by the **interaction** of the anchor set with
this project's stricter threshold of 2.0. Reporting only the 2.0 column
would overstate the finding; reporting only the 4.0 column would hide it.
The anchors are not "wrong" — they are a poor *prior* (mean best IoU 0.407,
and aspect-flipped with respect to the data: anchors are h/w = 2.5-3.0 tall
while BDD100K small vehicles are h/w = 0.80 near-square), and the strict
assignment rule turns a poor prior into an absent one for half the boxes.

## 2. Hypotheses under test

- **H19 (new) — detection is supervision/assignment-limited, not
  capacity-limited.** Half the GT boxes receive no positive assignment under
  the project's own rule. If this is the binding constraint, replacing the
  anchors with data-driven ones buys detection accuracy at **zero parameter
  and zero FLOP cost** — which would make it categorically different from
  every capacity-side intervention tried so far (Z width, encoder capacity,
  lane resolution).
- **H20 — on top of a correct prior, grid resolution still binds.** The
  29.8% sub-cell statistic is the grid-resolution argument. It is tested
  only *after* the prior is fixed, so the two effects do not confound.

## 3. Cells

| cell | change vs baseline | params | GFLOPs |
|---|---|---|---|
| `r2u_z16` (baseline, already committed) | — | 0.2014 M | 1.0796 |
| `danc_z16` | anchors only: default 9 -> IoU-k-means 9 | identical | identical |
| `dp2a_z16` | + a stride-4 level fed by `upsample(Z)`, k-means 12 anchors | +head only | small |

Both cells set `model.detection.anchors` **and** `train.anchors` to the same
value; the model decodes with the former and assigns with the latter, so
they must not diverge.

k-means anchors (IoU distance, tri_train, 32421 boxes, seed 0):

    stride  8: [9, 8]  [18, 15]  [32, 24]
    stride 16: [49, 38] [80, 52] [65, 102]
    stride 32: [124, 82] [166, 136] [237, 214]

4-level set for `dp2a` (k=12, smallest three on the new stride-4 level):

    stride  4: [8, 7]  [15, 12]  [20, 21]
    stride  8: [35, 19] [38, 32] [64, 41]
    stride 16: [41, 72] [87, 63] [135, 83]
    stride 32: [98, 128] [180, 139] [241, 221]

## 4. Decision rule (fixed now, applied mechanically later)

Noise floors are the external Phase 2-C figures already used by the lane
probe: `mAP50 0.0146`, `lane_fg 0.0064`, `da_fg 0.0404`. Baseline is the
**committed** `r2u_z16` 4-epoch row (mAP50 0.2411), not a rerun.

- **D1 (primary, H19)** `danc` mAP50 gain >= 2x noise (+0.0292) ->
  H19 SUPPORTED. 1x-2x -> WEAK. < 1x -> H19 REJECTED at this budget.
- **D2 (mechanism)** the gain must land on small-object recall. If mAP50
  rises while small recall does not, the mechanism is not the one predicted
  and the finding is reported as mechanism-unconfirmed regardless of D1.
- **D3 (H20, sequential)** `dp2a` mAP50 - `danc` mAP50 >= 1x noise
  (+0.0146) -> grid resolution binds on top of the prior. Negative or
  inside noise -> the sub-cell statistic is not the binding constraint.
  D3 is only read if D1 passes; if D1 fails, `dp2a` is still run but is
  interpreted as a test of H20 alone against the baseline.
- **D4 (control)** lane_fg and da_fg move < 2x noise in both cells.
- **D5 (cost)** `danc` params and GFLOPs equal to baseline within rounding.
  A "free" win that is not free is not a free win.

## 5. Standing caveats

- **4-epoch readings do not establish anything.** Two prior findings in this
  project reversed between 4 and 20 epochs (probe A; reconstruction depth).
  A D1 pass authorises a 20-epoch confirmation, nothing more.
- Changing anchors changes BOTH the training assignment and the eval-time
  decode. That is inherent to the intervention and is not separable here.
- The k-means set is seed-dependent. It is reported with its seed (0) and
  the script is committed; a different seed is a different prior, not a
  replication.

## 6. Literature position

- YOLOv5 AutoAnchor (k-means + genetic evolution, BPR metric, thr=4.0) is
  the standard practice this copies. **No novelty is claimed for
  re-clustering anchors.** The claim is diagnostic: that the *reason*
  detection sits where it does in this project is a supervision-side prior
  mismatch, which is a different statement from "re-clustering helps".
- RSO-YOLO (BDD100K): adding a P2 head gives +4.6 mAP50 for +29% GFLOPs
  and -44% FPS. MHD-Net: P2 plus dilated context, +2.6 mAP at negligible
  cost. SPTD-YOLO: P2 must combine *upsampled semantics* with shallow
  detail rather than use raw shallow features — the same
  semantics-from-deep / resolution-from-grid principle the 4B-2 lane probe
  found independently. Our head is far smaller than those baselines, so the
  FLOP cost of a stride-4 level here is expected to be a few percent, not
  29%.
