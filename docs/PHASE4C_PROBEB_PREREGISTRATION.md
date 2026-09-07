# Phase 4A - Level 2 Probe B: pre-registration

**Written before any probe B result exists.** Probe A already came back without a
positive signal (`docs/PHASE4B_PROBEA_PREREGISTRATION.md`, results in
`experiments/phase4a/phase4B_probeA_analysis.txt`), so per the plan fixed earlier
this is the next branch.

## Why this design and not a generic "second reconstruction"

`docs/PHASE4B_LITERATURE_SYNTHESIS.md` is the full argument. The short version:

- YOLOF (CVPR 2021) measured MiMo 35.9 / **SiMo 35.0** / MiSo 23.9 / SiSo 23.7 mAP.
  Our R2 detection is SiMo (one Z at 1/8, three output scales), so detection from a
  shared Z was never expected to be expensive. That also means the shared Z was
  never the likely bottleneck, which is consistent with our effective-rank finding.
- YOLOF's diagnosis of what a single-level feature *does* lack is a **fixed
  receptive field**, and their fix is a Dilated Encoder.
- Our reconstruction is a **single 1x1 convolution** - a one-pixel receptive field
  at 1/8 resolution. The diagnosis applies to us almost verbatim.

Every driving-perception model in this family (YOLOP, HybridNets, TriLiteNet) gives
detection multi-scale fusion (BiFPN / SPP + PAN) rather than a single map.

## Cells

4 epochs, seed 0, E-base, 640x640, batch 16.

| cell | shared Z | reconstruction | params | vs baseline |
|---|---|---|---|---|
| `r2d_z16` | 16 | 2 depthwise residual blocks, dilation **2, 4** | 204246 | +2880 (+1.43%) |
| `r2d_z32` | 32 | same | 218614 | +2880 (+1.33%) |
| `r2p_z16` | 16 | identical blocks, dilation **1, 1** | 204246 | +2880 (+1.43%) |

Baselines are the probe A cells already measured: `r2u_z16` (mAP50 0.2411) and
`r2u_z32` (mAP50 0.2496).

**The control is the point.** `r2d_z16` and `r2p_z16` have byte-identical parameter
counts (verified: 204246 both) and identical depth; the only difference is the
dilation rates. So:

- `r2d > r2p` -> the gain is **receptive field**.
- `r2d == r2p` -> any gain was just **more layers / capacity**, not scale range.

## Predictions

- **Q1 (primary).** `r2d_z16` beats `r2u_z16` on mAP50 or mAP50_95 by at least 2x
  noise (0.0146 / 0.0064). Rationale: the reconstruction currently has a one-pixel
  receptive field; dilation 2 and 4 extend it to roughly 9 and 17 pixels at 1/8
  scale, i.e. 72 and 136 pixels at input scale.
- **Q2 (the discriminator).** `r2d_z16` beats `r2p_z16` by at least 2x noise on
  mAP50 or mAP50_95. This is what separates receptive field from depth.
- **Q3 (control).** DA and lane move less than 1x noise under both `r2d_*` cells.
  The reconstruction is detection-only; if the segmentation tasks move more than
  that, it is a shared-Z training interaction and must be reported as such, not
  folded into the reconstruction story.

## Decision rule (fixed now)

- **B = YES** if Q1 and Q2 both hold -> receptive field is the binding constraint.
  Go deep on reconstruction: more blocks, larger rates, or a full YOLOF-style
  dilated encoder.
- **B = PARTIAL** if Q1 holds but Q2 fails -> reconstruction *capacity* matters but
  not scale range. Try a wider or deeper reconstruction instead of dilation.
- **B = NO** if Q1 fails (Q2 then being moot).
- Between 1x and 2x noise is **unresolved**; the response is to extend `r2d_z16` to
  20 epochs, not to declare either direction.

## What happens if B is also negative

Probe A (per-task width) and probe B (reconstruction) would both be null, which
means neither of the two obvious levers is the one. The ranked remaining
possibilities from the literature review, in the order I would spend GPU on them:

1. **H11 - gradient magnitude imbalance (free to test, no new architecture).**
   We already measured `||dL_det/dZ||` at 5-8x the DA and lane values with cosines
   near zero. PCGrad's framing says magnitude imbalance alone - with no conflict at
   all - is enough for one task to own a shared parameter. Test: loss reweighting
   (GradNorm or uncertainty weighting) at 4 epochs. If DA/lane are passengers on Z,
   this is the cheapest way to find out.
2. **Force rank usage.** Z utilisation is 24.9% at z128. A decorrelation or
   orthogonality pressure on Z, rather than more channels.
3. **Bottleneck placement** (H7b, still untested): move Z off 1/8, or compress
   later in the encoder.
4. **Budget.** 4 epochs sits at 68% of the 20-epoch mAP50, and Phase 3B already
   showed z-width effects dominated by training budget. Before declaring any
   architecture conclusion, the two most informative cells should be re-checked at
   the final budget.

## Explicitly still blocked

z256/z512, denser z sweeps, large encoder sweeps under R2, pruning, INT8, NPU
deployment, multi-seed confirmation.
