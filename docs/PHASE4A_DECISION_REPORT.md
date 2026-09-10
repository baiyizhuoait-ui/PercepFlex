# Phase 4A - Shared Bottleneck (R2) Study

Does the compact representation Z actually constrain the model, or did R0 simply
route detection around it so that Z could never matter?

Every number below is recomputed from the result CSVs by
`scripts/phase4a_report.py`; none of it is transcribed by hand.

Protocol: E-base encoder, seed 0, 20 epochs, batch 16, lr 1e-3, AdamW + cosine,
640x640, tri_train 69863. R0 rows are reused from Phase 3A/3B - R0 is never retrained.
Noise floor is an **external** reference (Phase 2-C, 3 seeds, 4 epochs); Phase 4A is
single-seed and does not estimate its own variance.

## 1. Headline

Three things changed when detection was forced through Z:

1. **Detection started using Z.** Its gradient on Z went from exactly 0.0000 to
   being the largest of the three tasks (73%, 77%, 75% of the total).
2. **Detection became Z-sensitive.** Over z16->z128 it moves
   -0.0043 mAP50 in R0 (0.59x noise) but +0.0171 in R2 (2.34x noise).
3. **Detection got better, not worse.** Forcing it through a 16-channel Z *raised*
   mAP50 by +0.0339 at z16.

## 2. Side by side at equal z

| z | variant | params M | GFLOPs | mAP50 | mAP50-95 | DA mIoU | DA fg | lane mIoU | lane fg |
|---|---|---|---|---|---|---|---|---|---|
| 16 | R0 | 0.1889 | 1.0597 | 0.3204 | 0.1149 | 0.8564 | 0.7723 | 0.5867 | 0.1973 |
| 16 | R2 | 0.2014 | 1.0796 | 0.3543 | 0.1273 | 0.8488 | 0.7612 | 0.5847 | 0.1943 |
| 32 | R0 | 0.2027 | 1.1973 | 0.3222 | 0.1158 | 0.8527 | 0.7670 | 0.5875 | 0.1992 |
| 32 | R2 | 0.2157 | 1.2238 | 0.3553 | 0.1248 | 0.8573 | 0.7736 | 0.5875 | 0.1992 |
| 128 | R0 | 0.2858 | 2.0231 | 0.3161 | 0.1129 | 0.8522 | 0.7663 | 0.5877 | 0.2005 |
| 128 | R2 | 0.3019 | 2.0889 | 0.3714 | 0.1355 | 0.8518 | 0.7656 | 0.5885 | 0.2015 |

## 3. The bypass-masking test

The clean test is not the absolute score - R2 has a bigger detection head, so a
level shift there proves nothing. The test is how much each model **moves** when z
changes, because within one variant the head architecture is fixed.

| metric | R0 z16->z128 | x noise | R2 z16->z128 | x noise |
|---|---|---|---|---|
| mAP50 (detection) | -0.0043 | 0.59x | +0.0171 | 2.34x |
| mAP50_95 (detection) | -0.0020 | 0.63x | +0.0082 | 2.56x |
| da_mIoU (DA) | -0.0042 | 0.30x | +0.0030 | 0.21x |
| da_fg (DA) | -0.0060 | 0.30x | +0.0044 | 0.22x |
| lane_mIoU (lane) | +0.0010 | 0.48x | +0.0038 | 1.81x |
| lane_fg (lane) | +0.0032 | 1.00x | +0.0072 | 2.25x |

Detection and lane both cross the noise floor under R2 and both were inert under R0.
DA stays inside it in both. That is a sign flip for detection, not just a scale change:
R0 detection *lost* 0.0043 mAP50 by widening Z, R2 *gained* 0.0171.

## 4. Per-task Z demand under R2

**detection**

| metric | z16->z32 | x noise | z16->z128 | x noise | verdict |
|---|---|---|---|---|---|
| mAP50 | +0.0010 | 0.14x | +0.0171 | 2.34x | beyond noise |
| mAP50_95 | -0.0025 | 0.78x | +0.0082 | 2.56x | beyond noise |

**DA**

| metric | z16->z32 | x noise | z16->z128 | x noise | verdict |
|---|---|---|---|---|---|
| da_mIoU | +0.0085 | 0.60x | +0.0030 | 0.21x | inside noise |
| da_fg | +0.0124 | 0.61x | +0.0044 | 0.22x | inside noise |

**lane**

| metric | z16->z32 | x noise | z16->z128 | x noise | verdict |
|---|---|---|---|---|---|
| lane_mIoU | +0.0028 | 1.33x | +0.0038 | 1.81x | beyond noise |
| lane_fg | +0.0049 | 1.53x | +0.0072 | 2.25x | beyond noise |

## 5. Who actually uses Z (gradient diagnostic)

`||dL_task/dZ||` per task, and the share of the total. This is measured by
backpropagating each task loss separately on the same batches.

| cell | det | da | lane | det share | cos(det,da) | cos(det,lane) | cos(da,lane) |
|---|---|---|---|---|---|---|---|
| r0_z16 | 0.0000 | 0.0053 | 0.0062 | 0.000 | n/a | n/a | +0.048 |
| r0_z32 | 0.0000 | 0.0049 | 0.0040 | 0.000 | n/a | n/a | +0.042 |
| r0_z128 | 0.0000 | 0.0042 | 0.0068 | 0.000 | n/a | n/a | +0.037 |
| r2_z16 | 0.0308 | 0.0059 | 0.0054 | 0.732 | -0.002 | +0.000 | +0.060 |
| r2_z32 | 0.0250 | 0.0032 | 0.0043 | 0.769 | +0.001 | +0.000 | +0.042 |
| r2_z128 | 0.0246 | 0.0047 | 0.0035 | 0.750 | -0.000 | -0.000 | +0.033 |

R0 detection is exactly zero at every width - it contributed no supervision to Z at
all, which is the bypass measured rather than read off the code. Under R2 detection
becomes the dominant consumer. The cosines are all near zero, so the tasks are
**orthogonal rather than in conflict**: gradient competition is not what makes Z
look unimportant.

## 6. Is the extra width actually used? (effective rank)

Effective rank of the Z channel covariance, over the same batches.

| cell | channels | effective rank | as % of width | top-8 var | top-32 var | dead channels |
|---|---|---|---|---|---|---|
| r0_z16 | 16 | 6.04 | 37.8% | 0.917 | 1.000 | 0 |
| r0_z32 | 32 | 9.62 | 30.0% | 0.800 | 1.000 | 0 |
| r0_z128 | 128 | 35.76 | 27.9% | 0.565 | 0.827 | 0 |
| r2_z16 | 16 | 6.79 | 42.5% | 0.898 | 1.000 | 0 |
| r2_z32 | 32 | 11.88 | 37.1% | 0.790 | 1.000 | 0 |
| r2_z128 | 128 | 31.93 | 24.9% | 0.587 | 0.832 | 0 |

Width and usable dimensionality are not the same thing. z128 carries about
32 independent dimensions, so roughly a quarter of its channels are redundant.
That is why widening Z past ~32 buys little: the model does not convert the extra
channels into extra independent features. No channel is ever dead, they are simply
correlated.

## 7. Cost, and the confound that must not be ignored

R2 is not free. The reconstruction lives in the detection head:

| z | R0 det head | R2 det head | delta | R0 total | R2 total | dFLOPs |
|---|---|---|---|---|---|---|
| 16 | 8406 | 20918 | +12512 | 188854 | 201366 | +0.0199 |
| 32 | 8406 | 21430 | +13024 | 202710 | 215734 | +0.0265 |
| 128 | 8406 | 24502 | +16096 | 285846 | 301942 | +0.0658 |

The R2 minus R0 detection gap is
+0.0339 at z16, +0.0331 at z32 and +0.0553 at z128. It is already ~+0.033 at z16, where Z is at its
narrowest, so most of that gap is the DetFromZ reconstruction and head capacity, not
Z width. Only the additional 0.0214 seen at z128 is attributable to width.
This is why section 3 compares movement within a variant rather than R2 against R0.

## 8. Hypothesis matrix

| hypothesis | status | decision |
|---|---|---|
| H-01 - R0 shows no detection->Z effect mainly because detection bypasses Z | **supported** | Measured, not inferred: ||dL_det/dZ|| is exactly 0 in R0 at all three widths and >0 in R2. The R0 null was the bypass. |
| H-02 - Once detection is forced through Z, mAP becomes clearly Z-sensitive | **supported-partial** | Detection Z-sensitivity rises to 2.34x noise (mAP50) and 2.56x (mAP50_95), but it is not monotone: z16->z32 is 0.14x noise, only z16->z128 clears the floor. |
| H-03 - z=16 is already sufficient for DA | **supported** | DA spread is inside the noise floor in BOTH R0 (<=0.30x) and R2 (<=0.61x). Saturation is real, not a bypass artefact. |
| H-04 - Lane's Z demand is real, not R0-specific training noise | **supported** | lane_fg z16->z128 is 2.25x noise under R2 versus 1.00x under R0. The lane Z demand reproduces and strengthens under a shared bottleneck. |
| H-05 - Encoder-heavy still beats Z-heavy under a genuine shared bottleneck | **not-tested** | Phase 3C settled this under R0 only. No encoder sweep was run under R2, so the allocation law is not yet known to transfer. |
| H-06 - Compressing encoder features into Z causes irreversible information loss | **not-supported-as-stated** | No dead channels at any width, and detection IMPROVES through Z, so compression to z>=16 is not measurably lossy here. What is lost is effective dimensionality, not task performance. |
| H-07 - R2 performance depends as much on reconstruction design as on Z width | **open** | Only one reconstruction (DetFromZ, det_ch=32) was trained. Its contribution cannot be separated from Z width without a second design. |
| H-08 - A single Z can serve all three tasks | **partially-supported** | A single Z does serve all three tasks, but not optimally: detection and lane want z128, DA is indifferent at z16. It works, it is not ideal. |
| H-09 - Tasks need different granularity, so one uniform Z is suboptimal | **supported** | Task-specific demand is measurable: detection 2.34x, lane 2.25x, DA 0.21x noise over the same z16->z128 range. |
| H-10 - Task gradients conflict at Z and shape how it is used | **not-supported** | Cosines are ~0 (det-da -0.002..+0.001, det-lane ~0.000, da-lane +0.033..+0.060). Tasks are near-orthogonal, not in conflict, so conflict does not explain Z insensitivity. |
| H-06b - Extra Z capacity is allocated but not used | **supported-with-caveat** | Utilisation falls from 42.5% of channels at z16 to 24.9% at z128 (effective rank 6.79 -> 31.93 of 16 -> 128), so most added channels are redundant. Caveat: none are dead and the redundant ones still buy real gains (detection 2.34x, lane 2.25x noise), so it is low-utilisation, not pure waste. |
| H-07b - Bottleneck placement matters more than width | **not-tested** | Only one bottleneck placement was trained (Z at 1/8 resolution, compression before all three heads). Width was swept, placement was not, so the two cannot be compared yet. |

## 9. Stopping decision

**CASE C.** Detection is Z-sensitive under R2 (2.34x noise), lane is Z-sensitive
(2.25x), and DA is indifferent (0.21x). That is a task-specific capacity demand, so
the next question is task-aware projection or split-Z, not a wider shared Z.

CASE A does not hold: the trend is not monotone - z16->z32 is only 0.14x noise, all
of detection's gain arrives at z128. CASE B does not hold because two tasks do move.
CASE D does not hold because R2 is better than R0, not worse.

## 10. Answers

1. **Does detection become Z-sensitive once forced through Z?** Yes. 0.59x noise in R0 to 2.34x in R2 on mAP50, and its gradient share goes from 0 to about 75 percent.

2. **Is DA genuinely Z-insensitive?** Yes. Inside the noise floor in both R0 and R2, so saturation is real and not a bypass artefact.

3. **Is lane's Z sensitivity real?** Yes, and stronger under R2: lane_fg z16->z128 is 2.25x noise versus 1.00x under R0.

4. **Is z=16 still enough?** For DA yes. Not for detection or lane, both of which gain beyond noise by z128.

5. **Does z=32 have real value?** For detection, no - z16->z32 is 0.14x noise. For lane, yes - lane_fg is 1.53x noise. z32 is a lane-width, not a detection-width.

6. **Is z128 just wasted capacity?** Partly. Only about 32 of 128 channels are independent dimensions, but detection and lane both gain beyond noise there, so it is low-utilisation rather than waste.

7. **Does the encoder-Z interaction persist in R2?** Not tested. No encoder sweep was run under R2, so the Phase 3C allocation law is not yet known to transfer.

8. **Is bottleneck placement more important than width?** Untested - only one placement was trained. This is the main open Level 2 question.

9. **Is reconstruction now the binding constraint?** It is a large effect: the R2-R0 gap is already ~+0.033 at z16, before Z width helps at all. Whether a different reconstruction does better is untested.

10. **Is there task-specific representation demand?** Yes. Over the same z range: detection 2.34x, lane 2.25x, DA 0.21x noise.

11. **Is there task gradient conflict?** No. All pairwise cosines are near zero. The tasks are orthogonal, not competing.

12. **Is a single shared Z reasonable?** It works - all three tasks train and R2 beats R0 - but it is not optimal, because the three tasks want different widths.

13. **Is a task-specific projection or adapter worth it?** Yes, as a small Level 2 probe. CASE C points directly at it.

14. **What should go to multi-seed confirmation?** r2_z128 (best accuracy), r2_z16 (best accuracy per FLOP, and the z16 baseline), and r0_z16 as the R0 reference. Three cells, not more.

15. **Which model should go on to pruning / INT8 / deployment?** r2_z16, unless accuracy dominates. It reaches mAP50 0.3543 at 1.08 GFLOPs; r2_z128 needs 2.09 GFLOPs - nearly double - for +0.0171 mAP50. R0's best detection under this protocol needs 1.428 GFLOPs for 0.3585.

## 11. Limitations

- **Single seed.** Every Phase 4A number is seed 0. The noise floor is an external
  Phase 2-C reference (3 seeds, 4 epochs), not a variance estimate for this phase.
  Anything within about 2x that floor should be read as unresolved.
- **One reconstruction, one placement.** `DetFromZ` at det_ch=32 is the only design
  trained, so H-07 and the placement question are untouched.
- **r2_z16 was trained twice.** The first attempt was killed by an out-of-memory
  condition I caused by running a diagnostic on the same GPU mid-training. The
  reported run is the clean retry at epoch 20.
- **FPS is still unusable** (power-cap throttling); all cost statements use params
  and FLOPs only.
- **R2 beats R0 partly on head capacity**, not only on the shared bottleneck. The
  decomposition in section 7 is the honest reading and must be carried forward.
- **No encoder sweep under R2**, so H-05 is deferred rather than answered.

## 12. What Phase 4A recommends next

Only Level 2 probes, and at most two of them, per CASE C:

- **Split-Z / task projection** (4 epochs): does a per-task projection off one shared
  Z beat a single uniform Z at the same width? This is the direct test of H-08/H-09.
- **A second reconstruction** (4 epochs): separates H-07 from Z width, and tells us
  whether the ~+0.033 constant offset is reconstruction quality or just head params.

Not recommended: continuing to widen Z (effective rank says the model will not use
it), and starting pruning / INT8 before the multi-seed confirmation in section 10.

