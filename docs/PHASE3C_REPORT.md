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
its measured parameters are within ±5% of the target. Membership uses the same
tolerance as the like-for-like test, so a layer can never contain a cell that then
fails the comparison it was admitted for.

Cells that sit near a layer but outside the tolerance are excluded from the triad
(they are still real measurements and still appear in the Pareto frontier):

- `ebase_z32` — +6.68% from the Budget-L target (0.2027 M).
- `elarge_z32` — +5.79% from the Budget-M target (0.3068 M).


| layer | target | cells | param span | verdict |
|---|---|---|---|---|
| Budget-L | 0.19 M | 3 | 1.56% | equal-budget |
| Budget-M | 0.29 M | 3 | 2.21% | equal-budget |
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
| balanced | `midL_z32` | 0.1860 | 1.1603 | 0.3152 | 0.1109 | 0.8532 | 0.7676 | 0.5862 | 0.1972 |
| Z-heavy | `esmall_z128` | 0.1867 | 1.6350 | 0.2639 | 0.0879 | 0.8490 | 0.7613 | 0.5855 | 0.1955 |

Delta versus the encoder-heavy cell (`ebase_z16`):

| cell | allocation | Δparams | ΔFLOPs | ΔmAP50 | Δda_mIoU | Δlane_mIoU | Δlane_fg |
|---|---|---|---|---|---|---|---|
| `midL_z32` | balanced | -0.0029 | +0.1006 | -0.0052 | -0.0032 | -0.0005 | -0.0001 |
| `esmall_z128` | Z-heavy | -0.0022 | +0.5753 | -0.0565 | -0.0074 | -0.0012 | -0.0018 |

### Budget-M (~0.29 M)

| allocation | cell | params (M) | FLOPs (G) | mAP50 | mAP50-95 | da_mIoU | da_fg | lane_mIoU | lane_fg |
|---|---|---|---|---|---|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.2915 | 1.4279 | 0.3585 | 0.1356 | 0.8589 | 0.7761 | 0.5878 | 0.2000 |
| balanced | `midM_z32` | 0.2852 | 1.5249 | 0.3462 | 0.1279 | 0.8572 | 0.7737 | 0.5900 | 0.2037 |
| Z-heavy | `ebase_z128` | 0.2858 | 2.0231 | 0.3161 | 0.1129 | 0.8522 | 0.7663 | 0.5877 | 0.2005 |

Delta versus the encoder-heavy cell (`elarge_z16`):

| cell | allocation | Δparams | ΔFLOPs | ΔmAP50 | Δda_mIoU | Δlane_mIoU | Δlane_fg |
|---|---|---|---|---|---|---|---|
| `midM_z32` | balanced | -0.0063 | +0.0970 | -0.0123 | -0.0017 | +0.0022 | +0.0037 |
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
| balanced | `midL_z32` | 0.3152 | 0.1109 |
| Z-heavy | `esmall_z128` | 0.2639 | 0.0879 |

- `mAP50`: spread 0.0565 > noise 0.0073 — best (statistical tie): **balanced, encoder-heavy**.
- `mAP50_95`: spread 0.0270 > noise 0.0032 — best (statistical tie): **encoder-heavy**.

**Budget-M (~0.29 M)**

| allocation | cell | mAP50 | mAP50_95 |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.3585 | 0.1356 |
| balanced | `midM_z32` | 0.3462 | 0.1279 |
| Z-heavy | `ebase_z128` | 0.3161 | 0.1129 |

- `mAP50`: spread 0.0424 > noise 0.0073 — best (statistical tie): **encoder-heavy**.
- `mAP50_95`: spread 0.0227 > noise 0.0032 — best (statistical tie): **encoder-heavy**.

### DA

**Budget-L (~0.19 M)**

| allocation | cell | da_mIoU | da_fg |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.8564 | 0.7723 |
| balanced | `midL_z32` | 0.8532 | 0.7676 |
| Z-heavy | `esmall_z128` | 0.8490 | 0.7613 |

- `da_mIoU`: spread 0.0074 ≤ noise 0.0142 — **no allocation wins**.
- `da_fg`: spread 0.0110 ≤ noise 0.0202 — **no allocation wins**.

**Budget-M (~0.29 M)**

| allocation | cell | da_mIoU | da_fg |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.8589 | 0.7761 |
| balanced | `midM_z32` | 0.8572 | 0.7737 |
| Z-heavy | `ebase_z128` | 0.8522 | 0.7663 |

- `da_mIoU`: spread 0.0067 ≤ noise 0.0142 — **no allocation wins**.
- `da_fg`: spread 0.0098 ≤ noise 0.0202 — **no allocation wins**.

### lane

**Budget-L (~0.19 M)**

| allocation | cell | lane_mIoU | lane_fg |
|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.5867 | 0.1973 |
| balanced | `midL_z32` | 0.5862 | 0.1972 |
| Z-heavy | `esmall_z128` | 0.5855 | 0.1955 |

- `lane_mIoU`: spread 0.0012 ≤ noise 0.0021 — **no allocation wins**.
- `lane_fg`: spread 0.0018 ≤ noise 0.0032 — **no allocation wins**.

**Budget-M (~0.29 M)**

| allocation | cell | lane_mIoU | lane_fg |
|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.5878 | 0.2000 |
| balanced | `midM_z32` | 0.5900 | 0.2037 |
| Z-heavy | `ebase_z128` | 0.5877 | 0.2005 |

- `lane_mIoU`: spread 0.0023 > noise 0.0021 — best (statistical tie): **balanced**.
- `lane_fg`: spread 0.0037 > noise 0.0032 — best (statistical tie): **Z-heavy, balanced**.

## 6. Params vs FLOPs

Equal parameters do **not** mean equal compute. Z operates at 1/8 resolution and is
consumed by the segmentation heads, so widening Z is expensive in FLOPs for very
little parameter cost. That asymmetry is a result in its own right.

| layer | param span | FLOPs span |
|---|---|---|
| Budget-L | 1.56% | 54.3% |
| Budget-M | 2.21% | 41.7% |

**Budget-L**

| allocation | cell | params (M) | FLOPs (G) | FLOPs per 0.01M params |
|---|---|---|---|---|
| encoder-heavy | `ebase_z16` | 0.1889 | 1.0597 | 0.0561 |
| balanced | `midL_z32` | 0.1860 | 1.1603 | 0.0624 |
| Z-heavy | `esmall_z128` | 0.1867 | 1.6350 | 0.0876 |

**Budget-M**

| allocation | cell | params (M) | FLOPs (G) | FLOPs per 0.01M params |
|---|---|---|---|---|
| encoder-heavy | `elarge_z16` | 0.2915 | 1.4279 | 0.0490 |
| balanced | `midM_z32` | 0.2852 | 1.5249 | 0.0535 |
| Z-heavy | `ebase_z128` | 0.2858 | 2.0231 | 0.0708 |

## 7. Fixed-budget Dominance

Only pairs whose parameter gap is within 5% are tested at all. Inside such a pair
the two cells are treated as sharing one budget by construction, so the residual 1-2%
gap is neither an advantage nor a disadvantage.

**Tier 1 (strict).** params ≤, FLOPs ≤, all six metrics ≥ raw value.
**Tier 2 (noise-aware).** same budget, FLOPs ≤, no metric worse by more than the
noise floor, at least one metric better beyond it.

Tier 2 is the rule that applies here. Tier 1 is shown only for transparency: it fails
on this data purely because of sub-tolerance and sub-noise residuals, not because any
allocation is genuinely competitive.

**Tier 1 — strict**

| dominating | dominated | Δparams | ΔFLOPs | better beyond noise |
|---|---|---|---|---|
| `midL_z32` | `esmall_z128` | -0.0007 (-0.37%) | -0.4747 (-29.0%) | mAP50, mAP50_95 |
| `midM_z32` | `ebase_z128` | -0.0006 (-0.21%) | -0.4982 (-24.6%) | mAP50, mAP50_95, lane_mIoU |

**Tier 2 — noise-aware (the applicable rule)**

| dominating | dominated | Δparams | ΔFLOPs | better beyond noise |
|---|---|---|---|---|
| `ebase_z16` | `esmall_z128` | +0.0022 (+1.18%) | -0.5753 (-35.2%) | mAP50, mAP50_95 |
| `ebase_z16` | `midL_z32` | +0.0029 (+1.56%) | -0.1006 (-8.7%) | mAP50_95 |
| `elarge_z16` | `ebase_z128` | +0.0057 (+1.99%) | -0.5952 (-29.4%) | mAP50, mAP50_95 |

**Near miss** — blocked only by metrics whose shortfall is at most 1.5× noise:

- `elarge_z16` vs `midM_z32`: wins mAP50, mAP50_95, blocked by `lane_mIoU` (1.05× noise), `lane_fg` (1.16× noise).

A margin that small is not a loss, it is an unresolved measurement. It is reported
as unresolved rather than as a win for either side.

Under tier 2 the cells that are never dominated by anything are: `ebase_z16`, `elarge_z16`.

## 8. Pareto Frontier

Cost is params **and** FLOPs; accuracy is all six metrics. Highest mAP alone selects
nothing here.

**Non-dominated (9):**

| cell | params (M) | FLOPs (G) | mAP50 | da_mIoU | lane_mIoU |
|---|---|---|---|---|---|
| `esmall_z16` | 0.1013 | 0.7218 | 0.2687 | 0.8423 | 0.5818 |
| `esmall_z32` | 0.1135 | 0.8522 | 0.2810 | 0.8456 | 0.5841 |
| `midL_z32` | 0.1860 | 1.1603 | 0.3152 | 0.8532 | 0.5862 |
| `ebase_z16` | 0.1889 | 1.0597 | 0.3204 | 0.8564 | 0.5867 |
| `ebase_z32` | 0.2027 | 1.1973 | 0.3222 | 0.8527 | 0.5875 |
| `midM_z32` | 0.2852 | 1.5249 | 0.3462 | 0.8572 | 0.5900 |
| `elarge_z16` | 0.2915 | 1.4279 | 0.3585 | 0.8589 | 0.5878 |
| `elarge_z32` | 0.3068 | 1.5709 | 0.3494 | 0.8546 | 0.5893 |
| `elarge_z128` | 0.3984 | 2.4292 | 0.3436 | 0.8575 | 0.5925 |

**Dominated (2):** `esmall_z128`, `ebase_z128`

## 9. Task-specific Capacity Analysis

The Phase 3B priors are treated as hypotheses and re-tested here under a fixed budget.

### detection

- Budget-L `mAP50`: best (statistical tie) **balanced, encoder-heavy**.
- Budget-L `mAP50_95`: best (statistical tie) **encoder-heavy**.
- Budget-M `mAP50`: best (statistical tie) **encoder-heavy**.
- Budget-M `mAP50_95`: best (statistical tie) **encoder-heavy**.

Decided cells: 4 · within noise: 0 · wins: encoder-heavy ×4, balanced ×1

### DA

- Budget-L `da_mIoU`: spread 0.0074 ≤ noise 0.0142 → no allocation wins.
- Budget-L `da_fg`: spread 0.0110 ≤ noise 0.0202 → no allocation wins.
- Budget-M `da_mIoU`: spread 0.0067 ≤ noise 0.0142 → no allocation wins.
- Budget-M `da_fg`: spread 0.0098 ≤ noise 0.0202 → no allocation wins.

Decided cells: 0 · within noise: 4 · wins: none

### lane

- Budget-L `lane_mIoU`: spread 0.0012 ≤ noise 0.0021 → no allocation wins.
- Budget-L `lane_fg`: spread 0.0018 ≤ noise 0.0032 → no allocation wins.
- Budget-M `lane_mIoU`: best (statistical tie) **balanced**.
- Budget-M `lane_fg`: best (statistical tie) **Z-heavy, balanced**.

Decided cells: 2 · within noise: 2 · wins: balanced ×2, Z-heavy ×1

## 10. Conclusion

**Direct answer: for detection, yes — under an equal parameter budget capacity should
go to the encoder. For DA and lane the answer is not established, because moving the
allocation barely moves them at all.**

Across every budget layer that supports a comparison, the encoder-heavy allocation
beats the Z-heavy allocation on detection by a margin far beyond noise, and it does it
with **substantially fewer FLOPs**. DA is indifferent to the allocation at every layer.
Lane is indifferent at Budget-L, and at Budget-M shows a marginal lead for the
*balanced* cell — not for Z-heavy — at roughly 1.05-1.16× noise, which one seed cannot
resolve.

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
| detection | **encoder** | encoder-heavy clearly wins in 2/2 comparable layers |
| DA | **neither** — indifferent | every layer spread inside the noise floor |
| lane | **no reliable preference** | 1/2 layers inside noise; where a winner does appear it is *balanced*, never Z-heavy |

**Stopping condition: none of A/B/C/D matches exactly.**

No rule matches, and choosing one anyway would be the wrong move. Concretely:

- **Rule A fails its second clause.** Encoder-heavy clearly wins detection in
  2/2 layers, but it wins **no** segmentation task anywhere: DA is inside noise
  at every layer, and at Budget-M the balanced cell is the one that is ahead on lane.
- **Rule B fails.** Z-heavy never wins lane, on any layer.
- **Rule C fails.** Balanced does win lane at Budget-M, but it is a clear detection
  loser at both layers, so it is not the best all-round trade-off.
- **Rule D fails.** The detection margin is 5.8-7.7× the noise floor.

The honest reading is therefore a **detection-only rule A**: encoder-first is
established for detection and is *not* established for DA or lane. The lane
counter-signal is real but sits at only ~1.05-1.16× noise, which is exactly the
regime a single seed cannot adjudicate. It is recorded as unresolved, not as a
win for either allocation.

One caveat that must not be lost: this does **not** say Z is useless. It says that at
these budgets, *marginal* parameters are better spent on the encoder. Lane in
particular showed a real Z main effect in Phase 3B; what Phase 3C shows is that when
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

