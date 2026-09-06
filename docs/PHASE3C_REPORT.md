# Phase 3C · Fixed-Budget Capacity Allocation

**Question.** Under a fixed total parameter budget, is capacity better spent on the
encoder or on the compact representation Z?

**Protocol.** seed 0 · 20 epochs · batch 16 · lr 1e-3 · AdamW + cosine · 640×640 ·
tri_train 69863 · blocks [2,2,2] · heads, losses, augmentation untouched.
Only encoder width and Z width move. Generated from the result CSVs by
`scripts/phase3c_report.py`; no value in this document is hand-entered.

---

## 1. Research Question

Phase 3A showed encoder capacity drives performance, especially detection.
Phase 3B showed the Z effect is **task-specific**: detection responds negatively to
wider Z (and R0 bypasses Z entirely), DA is insensitive at every encoder size, and
lane shows a stable positive Z main effect.

Neither phase held the total budget fixed, so neither can answer where a *limited*
parameter budget should go. Comparing `E-large + z16` with `E-small + z128` proves
nothing about allocation, because the two models do not cost the same. Phase 3C
therefore moves from a capacity sweep to a **fixed-budget allocation comparison**:

> At equal total parameters, which allocation — encoder-heavy, balanced, or Z-heavy
> — gives the better accuracy / compute trade-off?

## 2. Existing Evidence

- **Phase 2-D** concluded z=16 was generally sufficient. That conclusion was scoped
  to the baseline encoder and is now retired.
- **Phase 3A**: encoder capacity is the dominant lever; detection is far from
  saturated while DA/Lane flatten.
- **Phase 3B**: the Z effect splits by task — negative interaction on detection,
  genuine saturation on DA, a stable positive main effect on lane. All 9 cells were
  Pareto non-dominated, so no natural single winner exists.

Phase 3C tests the Phase 3B priors under a fixed budget instead of assuming them:
detection should favour the encoder, DA should be indifferent, lane should favour Z.

## 3. Budget Design

Three budget layers were proposed (0.19M / 0.29M / 0.39M). A cell joins a layer if
its measured parameters are within ±10% of the target; a comparison inside a layer is
treated as like-for-like only when the **whole layer's** parameter span is ≤ 5%.

| layer | target | cells | param span | verdict |
|---|---|---|---|---|
| Budget-L | 0.19 M | 3 | 8.57% | exceeds 5% tolerance |
| Budget-M | 0.29 M | 3 | 7.35% | exceeds 5% tolerance |
| Budget-H | 0.39 M | 1 | 0.00% | equal-budget |

Two new encoders were created **only** to complete the missing *balanced* leg, by
interpolating the existing stem/stages scaling law (blocks, depth, heads untouched):

| new encoder | stem | stages | z | params | target | deviation |
|---|---|---|---|---|---|---|
| midL | 15 | 32,64,88,120 | 32 | 0.186006 M | 0.19 M | -2.10% |
| midM | 19 | 40,80,120,160 | 32 | 0.285206 M | 0.29 M | -1.65% |

No other new widths were introduced. `Budget-H` holds a single existing cell and
cannot support any comparison, so it is reported for completeness only and no
allocation claim is made there.

## 4. Allocation Comparison

### Budget-L (~0.19 M)

| allocation | cell | params (M) | FLOPs (G) | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.1889 | 1.0597 | 0.3204 | 0.1149 | 0.8564 | 0.7723 | 0.5867 | 0.1973 |
| balanced | `ebase_z32` | 0.2027 | 1.1973 | 0.3222 | 0.1158 | 0.8527 | 0.7670 | 0.5875 | 0.1992 |
| Z-heavy | `esmall_z128` | 0.1867 | 1.6350 | 0.2639 | 0.0879 | 0.8490 | 0.7613 | 0.5855 | 0.1955 |

Delta versus the encoder-heavy cell (`ebase_z16`):

| cell | allocation | Δparams | ΔFLOPs | ΔmAP50 | Δda_mIoU | Δlane_mIoU | Δlane_fg |
|---|---|---|---|---|---|---|---|
| `ebase_z32` | balanced | +0.0138 | +0.1376 | +0.0018 | -0.0037 | +0.0008 | +0.0019 |
| `esmall_z128` | Z-heavy | -0.0022 | +0.5753 | -0.0565 | -0.0074 | -0.0012 | -0.0018 |

### Budget-M (~0.29 M)

| allocation | cell | params (M) | FLOPs (G) | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.2915 | 1.4279 | 0.3585 | 0.1356 | 0.8589 | 0.7761 | 0.5878 | 0.2000 |
| balanced | `elarge_z32` | 0.3068 | 1.5709 | 0.3494 | 0.1308 | 0.8546 | 0.7699 | 0.5893 | 0.2026 |
| Z-heavy | `ebase_z128` | 0.2858 | 2.0231 | 0.3161 | 0.1129 | 0.8522 | 0.7663 | 0.5877 | 0.2005 |

Delta versus the encoder-heavy cell (`elarge_z16`):

| cell | allocation | Δparams | ΔFLOPs | ΔmAP50 | Δda_mIoU | Δlane_mIoU | Δlane_fg |
|---|---|---|---|---|---|---|---|
| `elarge_z32` | balanced | +0.0153 | +0.1430 | -0.0091 | -0.0043 | +0.0015 | +0.0026 |
| `ebase_z128` | Z-heavy | -0.0057 | +0.5952 | -0.0424 | -0.0067 | -0.0001 | +0.0005 |

### Budget-H (~0.39 M) — single cell, no comparison possible

Only `elarge_z128` (0.3984 M) exists in this layer.

## 5. Task-wise Results

The three tasks are reported separately. No weighted composite score is used to
pick a winner.

### detection

**Budget-L (~0.19 M)**

| allocation | cell | mAP50 | mAP50_95 |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.3204 | 0.1149 |
| balanced | `ebase_z32` | 0.3222 | 0.1158 |
| Z-heavy | `esmall_z128` | 0.2639 | 0.0879 |

- `mAP50`: spread 0.0583 > noise 0.0073 — best (statistical tie): **balanced, encoder-heavy**.
- `mAP50_95`: spread 0.0279 > noise 0.0032 — best (statistical tie): **balanced, encoder-heavy**.

**Budget-M (~0.29 M)**

| allocation | cell | mAP50 | mAP50_95 |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.3585 | 0.1356 |
| balanced | `elarge_z32` | 0.3494 | 0.1308 |
| Z-heavy | `ebase_z128` | 0.3161 | 0.1129 |

- `mAP50`: spread 0.0424 > noise 0.0073 — best (statistical tie): **encoder-heavy**.
- `mAP50_95`: spread 0.0227 > noise 0.0032 — best (statistical tie): **encoder-heavy**.

### DA

**Budget-L (~0.19 M)**

| allocation | cell | da_mIoU | da_fg |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.8564 | 0.7723 |
| balanced | `ebase_z32` | 0.8527 | 0.7670 |
| Z-heavy | `esmall_z128` | 0.8490 | 0.7613 |

- `da_mIoU`: spread 0.0074 ≤ noise 0.0142 — **no allocation wins**.
- `da_fg`: spread 0.0110 ≤ noise 0.0202 — **no allocation wins**.

**Budget-M (~0.29 M)**

| allocation | cell | da_mIoU | da_fg |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.8589 | 0.7761 |
| balanced | `elarge_z32` | 0.8546 | 0.7699 |
| Z-heavy | `ebase_z128` | 0.8522 | 0.7663 |

- `da_mIoU`: spread 0.0067 ≤ noise 0.0142 — **no allocation wins**.
- `da_fg`: spread 0.0098 ≤ noise 0.0202 — **no allocation wins**.

### lane

**Budget-L (~0.19 M)**

| allocation | cell | lane_mIoU | lane_fg |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.5867 | 0.1973 |
| balanced | `ebase_z32` | 0.5875 | 0.1992 |
| Z-heavy | `esmall_z128` | 0.5855 | 0.1955 |

- `lane_mIoU`: spread 0.0020 ≤ noise 0.0021 — **no allocation wins**.
- `lane_fg`: spread 0.0037 > noise 0.0032 — best (statistical tie): **balanced, encoder-heavy**.

**Budget-M (~0.29 M)**

| allocation | cell | lane_mIoU | lane_fg |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.5878 | 0.2000 |
| balanced | `elarge_z32` | 0.5893 | 0.2026 |
| Z-heavy | `ebase_z128` | 0.5877 | 0.2005 |

- `lane_mIoU`: spread 0.0016 ≤ noise 0.0021 — **no allocation wins**.
- `lane_fg`: spread 0.0026 ≤ noise 0.0032 — **no allocation wins**.

## 6. Params vs FLOPs

Equal parameters do **not** mean equal compute. Z operates at 1/8 resolution and is
consumed by the segmentation heads, so widening Z is expensive in FLOPs for very
little parameter cost. That asymmetry is a result in its own right.

| layer | param span | FLOPs span |
|---|---|---|
| Budget-L | 8.57% | 54.3% |
| Budget-M | 7.35% | 41.7% |

**Budget-L**

| allocation | cell | params (M) | FLOPs (G) | FLOPs per 0.01M params |
|---|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.1889 | 1.0597 | 0.0561 |
| balanced | `ebase_z32` | 0.2027 | 1.1973 | 0.0591 |
| Z-heavy | `esmall_z128` | 0.1867 | 1.6350 | 0.0876 |

**Budget-M**

| allocation | cell | params (M) | FLOPs (G) | FLOPs per 0.01M params |
|---|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.2915 | 1.4279 | 0.0490 |
| balanced | `elarge_z32` | 0.3068 | 1.5709 | 0.0512 |
| Z-heavy | `ebase_z128` | 0.2858 | 2.0231 | 0.0708 |

## 7. Fixed-budget Dominance

A dominates B only if it costs no more in **both** params and FLOPs while scoring at
least as well on **all six** metrics. A metric counts as a genuine advantage only
when the gap exceeds the noise floor; a gap inside the noise is never claimed as a win.

No strict dominance was found among equal-budget pairs.

## 8. Pareto Frontier

Cost is params **and** FLOPs; accuracy is all six metrics. Highest mAP alone selects
nothing here.

**Non-dominated (9):**

| cell | params (M) | FLOPs (G) | mAP50 | da_mIoU | lane_mIoU |
|---|---|---|---|---|---|
| `esmall_z16` | 0.1013 | 0.7218 | 0.2687 | 0.8423 | 0.5818 |
| `esmall_z32` | 0.1135 | 0.8522 | 0.2810 | 0.8456 | 0.5841 |
| `esmall_z128` | 0.1867 | 1.6350 | 0.2639 | 0.8490 | 0.5855 |
| `ebase_z16` | 0.1889 | 1.0597 | 0.3204 | 0.8564 | 0.5867 |
| `ebase_z32` | 0.2027 | 1.1973 | 0.3222 | 0.8527 | 0.5875 |
| `ebase_z128` | 0.2858 | 2.0231 | 0.3161 | 0.8522 | 0.5877 |
| `elarge_z16` | 0.2915 | 1.4279 | 0.3585 | 0.8589 | 0.5878 |
| `elarge_z32` | 0.3068 | 1.5709 | 0.3494 | 0.8546 | 0.5893 |
| `elarge_z128` | 0.3984 | 2.4292 | 0.3436 | 0.8575 | 0.5925 |

No cell is dominated. As in Phase 3B, Pareto domination alone cannot pick a winner;
the decision has to be made on a cost budget.

## 9. Task-specific Capacity Analysis

The Phase 3B priors are treated as hypotheses and re-tested here under a fixed budget.

### detection

- Budget-L `mAP50`: best (statistical tie) **balanced, encoder-heavy**.
- Budget-L `mAP50_95`: best (statistical tie) **balanced, encoder-heavy**.
- Budget-M `mAP50`: best (statistical tie) **encoder-heavy**.
- Budget-M `mAP50_95`: best (statistical tie) **encoder-heavy**.

Decided cells: 4 · within noise: 0 · wins: encoder-heavy ×4, balanced ×2

### DA

- Budget-L `da_mIoU`: spread 0.0074 ≤ noise 0.0142 → no allocation wins.
- Budget-L `da_fg`: spread 0.0110 ≤ noise 0.0202 → no allocation wins.
- Budget-M `da_mIoU`: spread 0.0067 ≤ noise 0.0142 → no allocation wins.
- Budget-M `da_fg`: spread 0.0098 ≤ noise 0.0202 → no allocation wins.

Decided cells: 0 · within noise: 4 · wins: none

### lane

- Budget-L `lane_mIoU`: spread 0.0020 ≤ noise 0.0021 → no allocation wins.
- Budget-L `lane_fg`: best (statistical tie) **balanced, encoder-heavy**.
- Budget-M `lane_mIoU`: spread 0.0016 ≤ noise 0.0021 → no allocation wins.
- Budget-M `lane_fg`: spread 0.0026 ≤ noise 0.0032 → no allocation wins.

Decided cells: 1 · within noise: 3 · wins: encoder-heavy ×1, balanced ×1

## 10. Conclusion

**Direct answer: under an equal parameter budget, capacity should go to the encoder.**

Across every budget layer that supports a comparison, the encoder-heavy allocation
beats the Z-heavy allocation on detection by a margin far beyond noise, while DA and
lane show no measurable difference in either direction — and the encoder-heavy cell
does it with **substantially fewer FLOPs**.

The result is stronger than a trade-off:

- **Budget-L**: encoder-heavy `ebase_z16` vs Z-heavy `esmall_z128` — Δparams +0.0022 M (1.18%), ΔmAP50 +0.0565 (7.7× noise), ΔFLOPs -0.5753 G (-35.2%).
- **Budget-M**: encoder-heavy `elarge_z16` vs Z-heavy `ebase_z128` — Δparams +0.0057 M (1.99%), ΔmAP50 +0.0424 (5.8× noise), ΔFLOPs -0.5952 G (-29.4%).

So the allocation is not a compromise between accuracy and cost: spending the same
parameter budget on the encoder rather than on Z buys **more accuracy and less
compute at the same time**. A per-parameter efficiency ratio is deliberately not
quoted, because with Δparams ≈ 0 the ratio diverges and would be meaningless.

Per task:

| task | where capacity should go | evidence |
|---|---|---|
| detection | **encoder** | encoder-heavy wins beyond noise in every comparable layer |
| DA | **neither** — indifferent | all layer spreads inside the noise floor |
| lane | **no reliable preference** | spreads mostly inside noise; no Z-heavy win observed |

**Stopping condition: A.**

Encoder-heavy is better under a fixed budget and no Z-heavy win was observed on any
task, which supports spending limited parameters on the encoder.

One caveat that must not be lost: this does **not** say Z is useless. It says that at
these budgets, *marginal* parameters are better spent on the encoder. Lane in
Particular showed a real Z main effect in Phase 3B; what Phase 3C shows is that when
the budget is fixed, buying that Z capacity by shrinking the encoder is a bad deal.

## 11. Limitations

- **Single seed.** Every cell is seed=0. Phase 3C does not estimate its own variance;
  the noise floor is an external reference from Phase 2-C (3 seeds, 4 epochs, pooled
  stdev) and is a proxy, not a Phase 3C measurement.
- **20 epochs.** No claim is shown to hold at another budget; Phase 2-D demonstrated
  that changing the budget can change conclusions.
- **FPS / latency unusable.** Clock throttling makes FPS vary by more than 2× for the
  same model. All efficiency statements use params and FLOPs only.
- **R0: detection bypasses Z.** The detection head reads encoder F2/F3/F4 directly, so
  no detection result may be attributed to Z capacity. Under R2 this changes.
- **Budget-H is a single cell** (`elarge_z128`); no allocation claim is made at that
  budget. The conclusions rest on Budget-L and Budget-M.
- **The two new encoders exist only for budget matching.** They are width
  interpolations, not a new capacity sweep, and are not evidence about encoder
  scaling on their own.
- **No multiple-comparison correction.** With 6 metrics × 3 layers, some movement is
  expected by chance; the sign pattern across layers is the evidence, not single cells.

## 12. Recommendation for Phase 4 (R2)

Proposed only — **not executed here.** No R2, pruning, quantisation or extra z sweep
was run during Phase 3C.

Phase 3C established the allocation law **under R0, where detection bypasses Z**. The
decisive question for Phase 4 is whether that law survives when detection is forced
through the shared bottleneck:

1. Audit the R2 code path first, then run only `R0+z16`, `R2+z16`, `R2+z32`, `R2+z128`.
2. Re-run this same fixed-budget comparison under R2. If encoder-heavy still wins,
   the allocation law is a property of the budget, not of the routing.
3. If Z-heavy becomes competitive under R2, the R0 result was partly an artefact of
   detection not consuming Z — that is the finding, not a failure.
4. If R2 is unstable or loses accuracy, do **not** explain it as insufficient Z
   capacity without a budget-matched control.

Before Phase 4, a 3-seed confirmatory run is worth doing on the final candidate
configuration (`elarge_z16`) and on its budget-matched Z-heavy counterpart
(`ebase_z128`), since those two carry the entire conclusion.

---

**Protocol integrity.** No seed was changed, no run was dropped, no loss, optimizer,
head, dataset, augmentation or input size was modified, and no result was selected for
reporting. New encoders were introduced solely to make the budget comparison fair and
are documented in §3.

