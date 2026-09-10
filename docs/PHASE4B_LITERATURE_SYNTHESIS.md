# Phase 4A Level 2 - literature check and what it implies

Written while the 20-epoch confirmation of `r3tp_z16` was training. Purpose: put
the probe A null in context, and decide what probe B should actually be.

Sources are listed with what they measured, not just what they claim.

## 1. YOLOF (Chen et al., CVPR 2021) - the closest thing to a prior for our setup

`arXiv:2103.09460`. They decouple the two jobs an FPN does and test all four
combinations inside RetinaNet:

| encoder | in | out | mAP |
|---|---|---|---|
| MiMo | multi | multi | **35.9** |
| **SiMo** | **single** | **multi** | **35.0** |
| MiSo | multi | single | 23.9 |
| SiSo | single | single | 23.7 |

**Our R0 detection is MiMo** (reads encoder F2/F3/F4). **Our R2 detection is
SiMo** (one Z at 1/8, three output scales). SiMo costs 0.9 mAP against MiMo.

Two consequences, and they cut in opposite directions:

- It was never surprising that forcing detection through Z did not hurt. Phase 4A
  measured R2 detection as *better* than R0 (+0.0331 at z32, +0.0553 at z128).
  The literature says single-in / multi-out is nearly free, so H-06 ("compression
  to Z is lossy") was a weak hypothesis to begin with. We now have both our own
  measurement and an external prior saying the same thing.
- It also means **the shared Z was never the likely bottleneck for detection.**
  Searching for detection gains by widening Z was looking in the wrong place.
  That is consistent with the effective-rank result: z128 carries ~32 independent
  dimensions and the model does not convert extra channels into extra features.

What YOLOF says the single-level feature *does* lack:

1. **Limited scale range / receptive field.** A single feature map has a fixed
   receptive field. Their fix is the **Dilated Encoder**: a 1x1 projector followed
   by four dilated residual blocks, restoring most of the gap.
2. **Positive-anchor imbalance.** Does not apply to us - our Z is at 1/8, not 1/32,
   so anchors stay dense.

Our reconstruction is a **single 1x1 convolution**. It has a receptive field of
one pixel at 1/8 resolution. YOLOF's diagnosis applies to us almost verbatim.

## 2. What the driving-perception models actually do

YOLOP, YOLOPv2/v3, HybridNets, A-YOLOM, TriLiteNet (arXiv 2509.04092) all use
shared encoder + three task decoders, and they split the information path by task:

- **Detection** gets multi-scale fusion - BiFPN (HybridNets), or SPP + LitePAN
  (TriLiteNet), built from several encoder levels.
- **Segmentation** gets the highest-resolution feature (C3, 1/8 for TriLiteNet -
  the same resolution as our Z), plus an attention module (PCAA) before the two
  segmentation decoders.

So the field's implicit answer to "what does each task need" is: detection needs
**multi-scale fusion and receptive field**, segmentation needs **high-resolution
spatial features plus attention**. Nobody in this family feeds detection a single
1/8 map through one 1x1 and calls it done - except us.

TriLiteNet also documents the trade-off explicitly: their base config gets DA mIoU
92.4 and lane Acc 82.3 but detection mAP 72.3, and they attribute it to the
architecture prioritising segmentation. Task-preference is a design knob in this
literature, which is exactly the ECMT framing below.

## 3. Effective rank: why widening Z stops paying

- **RRAE** (arXiv 2405.13980): the bottleneck is determined by the *singular
  values* of the latent, not its dimensionality. An over-estimated latent dimension
  produces "holes"; the model uses fewer directions than it is given.
- **Inductive Bottleneck** (arXiv 2512.07331): ViTs spontaneously compress
  effective dimensionality in mid-layers - a U-shaped EED profile - and the depth
  of that compression adapts to the data, not to the architecture.

Our measurement (Z channel utilisation 42.5% at z16, 37.1% at z32, 24.9% at z128,
no dead channels) is the same phenomenon seen from the CNN side. The implication
is not "widen more": it is **the extra width is not the constraint, and adding
width will keep being under-used.** Capacity has to be spent somewhere the
optimizer will actually convert into independent directions - depth, receptive
field, or an explicit pressure to decorrelate.

## 4. Re-reading the probe A null

Probe A gave lane its own 16->32 projection and lane_fg moved 0.25x noise.

The projection we tested was a **1x1 convolution**: channel-only, spatially
uniform, input-independent. That is the weakest possible form of task-specific
adaptation. Recent work on task-specific projection in dense multi-task vision
(MLORE / TSAP, IEEE AMCAI 2025) does not use a plain 1x1 - it uses an attention
mechanism adapting **spatial, channel and long-range** dependencies per task, and
reports that this is what makes compact task decoders work on top of a rich shared
representation.

So the honest reading of probe A is narrower than "task-specific projection does
not work": **channel re-mixing at a fixed shared Z does not work.** What was not
tested is a *spatial or attentive* per-task projection. That is a real limitation
of the probe, and it should be stated alongside the null rather than buried.

## 5. ECMT (Aich et al., ICCV 2023) - which knob does what

Shared encoder + task decoders, both slimmable. Their result: **task preference is
controlled by decoder capacity; total compute is controlled by encoder capacity.**
Their search rule gives the larger budget to the preferred task's decoder while
searching the encoder for overall performance.

This matches Phase 3C (encoder-heavy beats Z-heavy at fixed total budget) and
reframes probe A: decoder width is the *task-preference* knob, not the accuracy
knob. Moving it mostly reshuffles which task wins - which is more or less what we
saw (lane flat, DA slightly down, detection flat).

## 6. Gradient magnitude: an axis we measured but never followed up

PCGrad (Yu et al., NeurIPS 2020) names the "tragic triad": conflicting gradients,
high curvature, and **magnitude imbalance**. Conflict is only one of the three.

Our own diagnostic already measured the magnitudes at Z:

| cell | det | da | lane |
|---|---|---|---|
| r2_z16 | 0.0308 | 0.0059 | 0.0054 |
| r2_z32 | 0.0250 | 0.0032 | 0.0043 |
| r2_z128 | 0.0246 | 0.0047 | 0.0035 |

**Detection's gradient on Z is 5-8x the segmentation tasks', while the cosines are
~0.** By PCGrad's framing, that is magnitude imbalance *without* conflict - and
imbalance alone is enough for one task to dominate the shared parameter update.
This is a cheap, non-architectural hypothesis we can test without training a new
model: **H-11 - under a summed loss, Z is effectively a detection representation and
DA/lane are passengers.** It would explain at once why DA is z-insensitive (it barely
pushes on Z) and why widening Z helps detection.

## 7. Budget: the confound running underneath all of this

- Chinchilla (Hoffmann et al., 2022): most models are undertrained for their size;
  a model too large for its token budget never reaches its potential.
- "Where should I spend my FLOPs?" (Koppula et al., 2022): for a fixed budget,
  train a smaller backbone longer rather than a larger one briefly.
- Standard fair-comparison practice: match optimizer steps, batch, and schedule
  definition; report mean and spread across seeds; a single run can be an outlier.

Our project has already been bitten by this once: Phase 3B found the z-width effect
was dominated by training budget (5-37x differences). Probe A at 4 epochs sits at
68% of the 20-epoch mAP50. **Architecture rankings in this regime are known to be
unstable**, which is the principled version of the power caveat in the probe A
report.

## 8. Ranked list of what is still plausible

| # | possibility | why | cost to test | prior |
|---|---|---|---|---|
| 1 | **Receptive field / multi-scale reconstruction** (YOLOF dilated encoder on Z) | our reconstruction has a 1-pixel receptive field; YOLOF names exactly this failure | 1x1 replaced by 2 dilated depthwise residual blocks, ~0.7k params | high |
| 2 | **Attentive / spatial per-task projection** (TSAP-style) instead of 1x1 | probe A tested the weakest form; attention adapts spatial+channel+long range | small; needs a spatial attention block per task | medium-high |
| 3 | **Loss / gradient rebalancing** (GradNorm, uncertainty weighting) | measured 5-8x magnitude imbalance at Z, cosines ~0 | no new architecture; retrain only | medium-high |
| 4 | **Force rank usage** (decorrelation / orthogonality pressure on Z) | 25% utilisation at z128 is the bottleneck, not the width | one regularizer term | medium |
| 5 | **Bottleneck placement** (move Z off 1/8, or compress later) | H-07b, still untested | new configs | medium |
| 6 | **Encoder depth over width at fixed budget** | Phase 3C already leans this way; Chinchilla supports it | new encoders | medium |
| 7 | **More training budget before concluding anything** | every effect here is small; 4ep -> 20ep has already moved things | expensive but honest | high as a *check*, not a fix |

## 9. What this means for probe B

B was defined as "is the reconstruction the binding constraint?" Row 1 is the
highest-prior version of that question, and it is near-parameter-matched, so a
result cannot be explained away by extra capacity.

Design:

- `rec=dw_dilated`: 1x1 projector (unchanged) then **two depthwise 3x3 residual
  blocks with dilation 2 and 4** at width 32. Adds roughly 0.7k parameters
  regardless of z. Only the receptive field changes.
- `rec=dw_plain`: **identical blocks at dilation 1.** Same parameters, same depth,
  same everything except dilation. This is the control that separates
  "it needed a bigger receptive field" from "it needed more layers".

Cells at 4 epochs, seed 0: `r2d_z16`, `r2d_z32` (dilated) and `r2p_z16` (plain
control). Predictions are registered before launch in the probe B pre-registration.
