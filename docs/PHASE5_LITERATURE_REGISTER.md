# Phase 5 / 4B literature register

Purpose: the brief requires that every experiment is preceded by a literature
check that (a) predicts the outcome and (b) flags whether the technique is
already published. This file is that record. It is written before the numbers
are read for the experiments it covers, so that a prediction cannot be
retrofitted to a result.

Rule carried over from the brief: a technique being published is not a failure
of this project. Only the novelty claim dies. The diagnosis is what we are
after, and the diagnosis does not depend on the intervention being new.

---

## L1. Lane: is the bottleneck the operating resolution of the head, or the absence of 1/4 information?

**Experiment (running):** `l14up_z16` (bilinear Z -> 1/4, no new information)
vs `l14f1_z16` (same plus a 1x1 lateral from encoder s1, which is real 1/4).
Control `lch64_z16` (channels 32 -> 64 at 1/8).

**Prediction from literature: both help, `l14f1` more, and the control helps
least per FLOP.**

- A-YOLOM (Wang et al., IEEE T-VT 2024, arXiv 2310.01641) states the asymmetry
  directly: drivable area "covers most of the image but there are only a few
  regions; lane lines are the opposite - many instances, each small, thin and
  narrow", and that lane segmentation "requires low-level and multi-scale
  features for effective analysis". It then builds **two separate segmentation
  necks**, one per task. This is the clearest statement in the literature that
  lane and DA do not want the same feature level.
- YOLOPX (jiaoZ7688) exposes `use_C2` and routes **C2 (H/4) into the lane
  head**, while C3 (H/8) serves segmentation and C4 (H/16) serves drivable
  area. Its lane head additionally carries a Polarized Self-Attention module
  "for capturing long-range spatial dependencies critical for lane line
  continuity".
- YOLOPv3 (MDPI Remote Sensing 16(10):1774) argues YOLOP/YOLOPv2 "fail to
  utilize high-resolution features" and HybridNets/YOLOPX "only employ them for
  semantic segmentation", and reports that its own lane head gains from a
  lightweight self-attention refinement module.

**Collisions, recorded up front.** The intervention in `l14f1` - feed the lane
head a 1/4 feature - is **published**. YOLOPX does it, YOLOPv3 does it,
A-YOLOM does it with a separate neck. We claim no novelty for it. What is not,
as far as the register can establish, published is the *measurement*: a
model-free ladder showing that a 0.2 M-parameter model sits exactly on the 1/8
block-fill reference (0.1853 vs models at 0.1843-0.1882), i.e. that the lane
head is not under-trained but geometrically incapable at that rung.

**Quantitative anchor, which is the most useful thing here.** Published BDD100K
lane IoU, collated from the A-YOLOM README: ENet 14.64, SCNN 15.84,
ENet-SAD 16.02, YOLOv8n(seg) 22.9, YOLOP 26.5, A-YOLOM(n) 28.2,
A-YOLOM(s) 28.8, YOLOPv3 28.0. Our 1/8-rung models sit at 17.6-18.8.

Read that against our own ladder (block-fill reference: 1/8 = 0.1853,
1/4 = 0.3220, 1/2 = 0.5721). Every published model above ~22 IoU operates its
lane head finer than 1/8; every model at or below ~16 operates at 1/8 or
coarser and is in the same band we are in. So the realistic expectation for
`l14f1` is **not** the 0.3220 ladder value - that is an oracle bound - but
something in the 0.20-0.26 region, i.e. closing part of the gap toward YOLOP's
26.5 with a 0.2 M model. If `l14f1` lands near 0.20+, that is consistent with
the literature. If it lands at 0.19, the 1/4 lateral is not reaching the head
usefully and the next question is whether s1 (32 ch, one block) is too shallow
to carry lane - which A-YOLOM's separate-neck design suggests it may be.

**Caveat that binds the metric itself.** CULane's official protocol draws both
prediction and ground truth as **30-pixel-wide** curves and matches at IoU 0.5
(Pan et al. 2018; restated in the LLFLD and ORANet papers). BDD100K lane labels
are reported at 8 px stroke for train and 2 px for test. The field does not, in
general, score thin-line pixel IoU at native width - it widens first. Combined
with our own erosion result (survival 0.0107 at k=1, 0.0007 at k=2 on 3-4 px
targets), lane fg-IoU at 1/8 is a metric with very little dynamic range. Any
claim about lane must be reported with this in view.

---

## L2. Drivable area: boundary precision, not resolution

**Established without training:** DA block-fill reference is 0.8753 at 1/8 and
0.9410 at 1/4, so only ~0.11 of headroom exists from resolution alone, while
the trained models sit at 0.71-0.73 fg with recall 0.97-0.99 and precision
0.72-0.87 and an area ratio of 1.32-1.37x. The tolerance curve is flat, unlike
lane's, which is steep.

**Literature says the same thing and names it.**

- GDA-RoadSeg (Complex & Intelligent Systems 2025) lists "weak edge-region
  segmentation" as a named failure mode: because the road interior dominates
  the pixel count, "semantic segmentation models often bias training toward
  optimizing this dominant region and neglect the smaller edge regions", and
  decoders that fuse with a plain 1x1 concat "weaken shallow-detail
  representations... yielding missed or merged thin structures, which inflates
  the estimated drivable area". Their fix is an edge-aware auxiliary branch
  with generated edge supervision.
- AURASeg (arXiv 2510.21536) opens with the same observation - "conventional
  encoder-decoder models often recover coarse region masks while losing the
  fine spatial information needed to localize drivable-area boundaries
  accurately" - and separates the problem into *region reconstruction* and
  *boundary refinement*, adding a Sobel-informed residual boundary refinement
  module. It reports that region-level metrics stay competitive while boundary
  quality improves, which is exactly our recall-saturated / precision-limited
  signature.
- PIDNet keeps an explicit boundary branch alongside detail and context
  branches. YOLOPv2's own analysis treats DA as a blob-like task needing
  global context rather than fine detail.

**Prediction:** resolution-side interventions on DA will be near-noise;
boundary-side interventions (edge-aware auxiliary supervision, or a boundary
tolerance-aware evaluation) are where the residual lives. This is why Phase 5
does not spend a training run on DA resolution.

**Note for the write-up:** the published DA numbers are much higher than ours
(YOLOP 91.6 mIoU, A-YOLOM(s) 91.0), but those are mIoU over both classes with
background dominating, whereas we report foreground IoU. Not comparable; do not
put them side by side without saying so.

---

## L3. Detection: small objects, and the P2 head is the published lever

**Established without training:** over 3184 GT boxes, recall is 0.64-0.69 for
small (<32^2), 0.90 medium, 0.95 large; crowding does not depress recall; and
widening Z from 16 to 128 helps small only (+0.0235).

**Literature predicts a large effect from a 1/4 detection head, and quantifies
it.**

- RSO-YOLO (MDPI Sensors 25(21):6703), BDD100K ablation, single-module gains:
  BiFPN +0.8 mAP50 / +0.7 mAP50:95; **P2 detection head +4.6 mAP50 /
  +1.8 mAP50:95**, the largest single-module gain in the table; combining
  BiFPN+P2 gives +6.1 mAP50 over baseline.
- MST-YOLO (MDPI Sensors 24(22):7347), on BDD100K: adding an ST-P2Neck moves
  **APsmall 0.124 -> 0.209 and ARsmall 0.251 -> 0.423** (recall +8.15 points
  overall), and cuts the miss-detection rate from 5.62 to 3.13.
- YOLOPv3 (op. cit.) makes the same argument inside multi-task driving
  perception: prior work "fail[s] to fully leverage multi-scale high-resolution
  features... not conducive for the network to detect small objects that are
  prevalent in intelligent driving scenarios".
- YOLO-VD (Elsevier Array 30, 2026) reports an "additional P2-level detection
  head ... to enhance small-object recognition" as one of its four components.

**Relevance to this architecture, and an important asymmetry.** Our detection
head already bypasses the Z bottleneck and reads encoder F2/F3/F4 directly (R0
design), i.e. it already has the *uncompressed* features - but the finest of
those is 1/8. There is no P2 (1/4) rung feeding detection. So the published
lever is available and untested here.

**Prediction for a P2-style probe:** a 1/4 detection rung should move small
recall more than anything else tried so far, and should move it more per FLOP
than widening Z (which bought +0.0235 small for a large compute increase).
This is a Phase 4B candidate, not a Phase 5 one - the brief's Phase 5 detection
list is reconstruction decomposition, spatial-resolution diagnosis and
receptive-field diagnosis, all zero-training, and those are already done.

**Collision:** the P2 head is textbook at this point (it is a standard YOLO
option). No novelty. Same rule as L1: the value is in the measurement, not the
module.

---

## L4. Cross-cutting: is per-task feature level a known result?

Yes, and this is worth stating plainly because it bounds what can be claimed.

- LOMT / MTI-Net argue tasks should be served at different depths and scales;
  A-YOLOM instantiates it with two segmentation necks; YOLOPX instantiates it
  with `use_C2` for lane. The *architectural conclusion* "lane wants 1/4, DA
  does not, detection wants 1/4 for small" is therefore not new in outline.
- What is not published, and what this project can actually contribute, is a
  **bottleneck profile with numbers attached**: per task, the model-free
  resolution ceiling, the gap to the trained model, the error geometry, and the
  gain per GFLOP of the spatially-targeted intervention versus a
  channel-spending control at equal FLOP. That is a measurement contribution,
  and it is what the Phase 5 deliverables are built to carry.

## L5. What this changes about the plan

1. Lane probe: read `l14f1` against 0.20-0.26, not against the 0.3220 oracle
   bound. A result of 0.20 is consistent with the literature and clears the
   pre-registered L1 bar (0.1829); it is not a disappointment.
2. DA: no training run on resolution. The boundary hypothesis is the one with
   literature support; it needs its own registration before any run.
3. Detection: the P2-style probe is the strongest remaining Phase 4B candidate
   by published effect size (ARsmall +68% in MST-YOLO). It must be registered
   before running, with the control being equal-FLOP spending elsewhere.
4. Everywhere: report lane fg-IoU with the CULane-widening / BDD100K
   train-8px-test-2px caveat attached, or the number will be read as worse than
   it is.

---

## L6. Detection supervision side (added for Phase 4B-3, before reading D1-D3)

Registered *before* the anchor cells were decided, so it can constrain the
reading of the numbers rather than rationalise them afterwards.

**ATSS (Zhang et al., CVPR 2020, arXiv:1912.02424).** The essential difference
between anchor-based and anchor-free detection is **how positive and negative
training samples are defined**, not whether anchors exist and not whether
regression starts from a box or a point. Swapping only the assignment rule
moves RetinaNet 37.0 -> 37.8 AP and drops FCOS 37.8 -> 36.9. Two consequences
that bear directly on this project:

1. It predicts that a large mAP gain can come from the **assignment** side
   without any change in capacity, parameters or FLOPs - which is exactly the
   shape of the `danc` intervention (anchors only, 0 params, 0 FLOPs).
2. It also predicts **diminishing returns**: with an adaptive assignment rule,
   RetinaNet with 9 anchors per location and with 1 anchor per location perform
   about the same ("tiling multiple anchors is not necessary"). So re-clustering
   anchors is a one-shot repair, not a direction to iterate on. If a further
   anchor-design sweep is ever proposed, ATSS says the expected gain is small
   and the honest move is to change the assignment rule instead.

**Counter-evidence on anchor shape specifically.** In an anchor-shape ablation
(PLOS ONE 16(11):e0260609, zebrafish cell detection), adding a fourth aspect
ratio did not improve final mAP - the authors attribute this to the box
regression head eventually pulling predictions back onto the GT. That is a
reason to be sceptical of the *aspect-ratio mismatch* story in isolation: it
supports reading our gain as **assignment coverage** (48.5% of boxes had zero
positive under the project rule) rather than as "the anchors were the wrong
shape". The two are confounded in `danc` by construction; this register entry
is the note that they are confounded.

**Prediction carried into the D2 reading.** If the gain is assignment-driven
rather than resolution-driven, mAP50 should rise more than small-object
*recall*: recall@0.5 of small boxes is limited by whether a box is found at
all, which is a resolution/capacity question, whereas mAP rewards the quality
and ranking of the boxes that are found. D2 was written to catch exactly this:
a rise in mAP50 with no rise in small recall is a mechanism-unconfirmed
outcome, not a pass.

**P2 / high-resolution detection head (carried from the 4B-3 preregistration).**
RSO-YOLO (BDD100K): a P2 head gives +4.6 mAP50 for +29% GFLOPs and -44% FPS.
MHD-Net: P2 plus dilated context, +2.6 mAP at negligible cost. SPTD-YOLO: P2
must combine upsampled semantics with shallow detail rather than use raw
shallow features - the same semantics-from-deep / resolution-from-grid
principle the 4B-2 lane probe found independently. Our head is far smaller than
those baselines, so the FLOP cost of a stride-4 level here is expected to be a
few percent, not 29%.
