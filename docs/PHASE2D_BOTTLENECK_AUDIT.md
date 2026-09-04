# Phase 2-D · Z-Bottleneck Structural Audit

**Status:** 0-training-cost analysis (read-only). Forbids conclusion-hacking — only code, configs, and (later) measured numbers are referenced.

## 1. Where is Z produced?

`models/representation/compact_z.py::CompactRepresentation.forward`:

```python
def forward(self, feats):
    f2, f3, f4 = feats                         # encoder multi-scale, 1/8 / 1/16 / 1/32
    z = self.lat4(f4)                          # 1x1 conv → zc channels
    z = upsample_to_f3(z); z = z + self.lat3(f3)
    z = upsample_to_f2(z); z = z + self.lat2(f2)
    z = self.act(self.bn(z))                   # Z at 1/8 resolution
    return {"z": z, "multi_scale": feats}      # ← both are exposed
```

`z_channels` is the ONLY capacity knob: `lat2/lat3/lat4` are `Conv2d(c, zc, 1)` lateral
projections, and the result is a 1/8 feature map of width `zc`. There is no other
compression in the model.

**The encoder is fixed** for the entire Phase 2-B/D Z-sweep family
(`stem=16, stages=[32,64,96,128], blocks=[2,2,2]`). So `f2/f3/f4` have channel widths
64 / 96 / 128, independent of z.

## 2. Which task heads actually see Z?

`models/static_model.py::StaticMultiTaskModel.__init__`:

```python
self.encoder = LightEncoder(...)
self.representation = CompactRepresentation(...)

if det_cfg.get("from_z", False):               # default OFF in Phase 2-B zsweep
    self.det_head = DetFromZ(zc, ...)
    self.det_from_z = True
else:
    self.det_head = DynamicDetHead(enc_cfg["stages"][1:], ...)   # ← uses [F2,F3,F4]
    self.det_from_z = False

self.da_head  = DynamicSegHead(zc, ...)
self.lane_head = DynamicSegHead(zc, ...)
```

`StaticMultiTaskModel.forward`:

```python
def forward(self, x, return_z=False):
    feats = self.encoder(x)
    z = self.representation(feats)["z"]
    if self.det_from_z:
        det = self.det_head(z, hw)             # R1/R2 — Z only
    else:
        det = self.det_head([feats[0], feats[1], feats[2]])    # R0 — bypass
    da   = self.da_head(z, hw)
    lane = self.lane_head(z, hw)
    ...
```

| Task       | R0 (current zsweep)   | R1 (`from_z:false`)  | R2 (`from_z:true`)    |
|------------|-----------------------|----------------------|-----------------------|
| detection  | encoder F2/F3/F4 directly (64/96/128 ch) | Z raw (zc @ 1/8) | Z → 1x1→BN→ReLU→32ch (det_ch fixed) |
| drivable   | Z (zc ch @ 1/8)       | Z (zc ch @ 1/8)      | Z (zc ch @ 1/8)       |
| lane       | Z (zc ch @ 1/8)       | Z (zc ch @ 1/8)      | Z (zc ch @ 1/8)       |

The `phase2b_zsweep_z{16,32,128}.yaml` files all set `detection: {nc: 1}` only —
no `from_z` flag, so `det_from_z = False`. **R0 is what we are actually running in
the budget experiment.**

## 3. Is there a high-dim bypass for the segmentation heads?

`DynamicSegHead.forward(z, target_hw)`: only `z` is the entry point. There is no
skip-connection from the encoder to the segmentation heads. The `z` it sees is the
fused-and-1x1-projected output of `compact_z.py`, at 1/8 resolution.

So for **drivable / lane**, Z is the *only* information path. Compression at
`zc ∈ {16, 32, 128}` is a real, hard bottleneck for those two tasks.

## 4. Is there a high-dim bypass for detection (R0)?

Yes. In R0:

- The detection head consumes the encoder's three multi-scale feature maps
  (`F2(64ch, 1/8)`, `F3(96ch, 1/16)`, `F4(128ch, 1/32)`) directly.
- These features are produced **upstream** of the Z lateral projections and
  contain the encoder's full information content.
- The only place Z could affect detection in R0 is the shared multi-task loss
  (and therefore the gradients that flow back into the encoder).

This means: **in R0, varying `z_channels` is a real ablation for drivable / lane
but only a soft, gradient-mediated perturbation for detection.**

## 5. Why the Phase 2-C Z-sweep could not find a knee

`experiments/phase2c/expA_multiseed.csv` (3 seeds × {16, 32, 128} @ 4ep) shows:

- **mAP50 (detection, R0 bypass path):** 0.2349 ± 0.0015, 0.2394 ± 0.0101, 0.2425 ± 0.0021
  — within-seed std (largest 0.0101) is comparable to cross-z mean gap (0.0076),
  and the cross-z ordering is non-monotonic in two of three seeds.
- **da_mIoU (R0 hits Z bottleneck):** 0.8202 ± 0.0112, 0.8266 ± 0.0048, 0.8196 ± 0.0213
  — `z=128` is *lower* on average than `z=32` (seed2 outlier 0.7950 drives this).
- **lane_mIoU (R0 hits Z bottleneck):** 0.5756, 0.5770, 0.5790 — *monotonic*,
  but the cross-z spread (0.0034) is on the same order as the within-z noise
  (typical σ ≈ 0.003 from the 3-seed sample).

Two things are happening simultaneously:

1. **Z really is a hard bottleneck for segmentation** — but `z=16` is *already*
   large enough to express the BDD100K 2-class DA and lane fg/bg logits at 1/8
   resolution. Once the information rate of the task is below `zc · 80 · 80` bits
   per image (very loose bound), additional channels just add unused capacity.
2. **Detection is decoupled from Z in R0** — so it cannot be used as an
   out-of-sample signal that Z is too small. The within-z noise for mAP50 reflects
   the encoder's stochasticity, not Z.

## 6. Implication for the budget experiment

The Phase 2-D budget experiment is testing a *legitimate* question — does more
training move the per-task mean so the cross-z gap grows? — but the answer is
*structurally* bounded:

- For DA / lane, the gap can only grow if `z=16` is actually below the task's
  information rate at 4ep. At 4ep, mIoU is still rising (`final_avg_loss` is
  still ~0.246 at ep4 vs ~0.05 plateau for a more converged model). It is
  possible that with more epochs, `z=16` saturates first and the gap opens.
- For detection, the gap is *not structurally present in R0*, so even 200ep
  will not show a Z effect on mAP50. (Only via gradient coupling; expected
  to be small.)

This audit means the decision report's "z-vs-task sensitivity" question has a
known answer before the experiments even run: **lane ≥ DA ≫ det in their
ability to reveal Z capacity.** Any future re-design should make detection
read Z (R2) to get a real three-way sensitivity.

## 7. Open caveat (not a finding, a flag)

- The above is read directly from `static_model.py` and the three zsweep YAMLs.
  If anyone modifies those, this audit must be re-run.
- The encoder's effective capacity is bounded by its **residual paths** in
  `DWSBlock` (the `+ res` in `light_encoder.py:61`). These are not part of Z and
  are not ablated here.
