# Phase-2-B Task 2 — Capacity / Sensitivity / Utility Analysis (`auto` mode)


## Raw table

| label | z | params(M) | flops(G) | fps | mAP50 | da_mIoU | lane_fg | lane_mIoU |
|---|---|---|---|---|---|---|---|---|
| z16 | 16 | 0.189 | 1.06 | 139 | 0.2368 | 0.8199 | 0.1818 | 0.5792 |
| z32 | 32 | 0.203 | 1.20 | 416 | 0.2421 | 0.8317 | 0.1802 | 0.5775 |
| z48 | 48 | 0.217 | 1.33 | 375 | 0.2360 | 0.8328 | 0.1799 | 0.5772 |
| z64 | 64 | 0.230 | 1.47 | 172 | 0.2424 | 0.8258 | 0.1836 | 0.5796 |
| z96 | 96 | 0.258 | 1.75 | 151 | 0.2437 | 0.8247 | 0.1809 | 0.5776 |
| z128 | 128 | 0.286 | 2.02 | 153 | 0.2403 | 0.8318 | 0.1841 | 0.5798 |

## (d) Per-task retention vs best (non-synchronous degradation)

| label | mAP50 ret% | da_mIoU ret% | lane_fg ret% | lane_mIoU ret% |
|---|---|---|---|---|
| z16 | 97.2 | 98.5 | 98.8 | 99.9 |
| z32 | 99.3 | 99.9 | 97.9 | 99.6 |
| z48 | 96.8 | 100.0 | 97.7 | 99.6 |
| z64 | 99.5 | 99.2 | 99.7 | 100.0 |
| z96 | 100.0 | 99.0 | 98.3 | 99.6 |
| z128 | 98.6 | 99.9 | 100.0 | 100.0 |

> **Derived from the table above (not asserted a priori).** At the lowest-capacity point (**z16**) the least-retained task is **mAP50** (97.2% of its best) and the most-retained is **lane_mIoU** (99.9%). Spread = 2.7 pp.

> ⚠️ **Spread is only 2.7 pp — retention is nearly SYNCHRONOUS across this capacity range.** The non-synchronous-degradation claim is therefore NOT supported by these data; do not report it as a finding.


## (b) Per-step capacity sensitivity (Δ per capacity/Z step)

| step | ΔmAP50 | Δda_mIoU | Δlane_fg | Δflops(G) | ΔmAP per +1 GFLOP |
|---|---|---|---|---|---|
| z32->z16 | +0.0053 | +0.0118 | -0.0016 | +0.14 | +0.0385 |
| z48->z32 | -0.0061 | +0.0011 | -0.0003 | +0.14 | -0.0443 |
| z64->z48 | +0.0064 | -0.0070 | +0.0037 | +0.14 | +0.0465 |
| z96->z64 | +0.0013 | -0.0011 | -0.0027 | +0.28 | +0.0047 |
| z128->z96 | -0.0034 | +0.0071 | +0.0032 | +0.28 | -0.0124 |

> **Derived from the table above (not asserted a priori).** Largest total change across the sweep: **da_mIoU** (+0.0119). A task has saturated where its Δ turns ~0 or negative while FLOPs keep rising; `ΔmAP per +1 GFLOP` falling toward 0 is the diminishing-return signal.


## Empirical noise floor (from a control metric)

`mAP50` does not receive the swept quantity under the current routing, so its variation across these 6 runs is **run-to-run noise, not a capacity effect**.

| control metric | mean | std (σ) | 2σ (min claimable Δ) | range |
|---|---|---|---|---|
| mAP50 | 0.2402 | 0.0032 | **0.0063** | 0.0077 |

> Any per-step Δ smaller than **2σ = 0.0063** should be read as noise. Compare the per-step table above against this threshold before calling any difference an effect. With n=6 runs this σ is itself a rough estimate.


## (a) Utility, efficiency, and Pareto

| label | Utility(0-1) | U/params(1/M) | U/flops(1/G) | mAP/flops |
|---|---|---|---|---|
| z16 | 0.981 | 5.20 | 0.926 | 0.223 |
| z32 | 0.990 | 4.89 | 0.827 | 0.202 |
| z48 | 0.982 | 4.53 | 0.735 | 0.177 |
| z64 | 0.995 | 4.32 | 0.675 | 0.165 |
| z96 | 0.991 | 3.84 | 0.567 | 0.139 |
| z128 | 0.995 | 3.48 | 0.492 | 0.119 |

- **Pareto frontier (mAP vs FLOPs):** z16, z32, z64, z96
- **Pareto frontier (mAP vs Params):** z16, z32, z64, z96
- **Knee (mAP vs FLOPs, max-distance-to-chord):** z32 (dist 0.568)
- **Knee (mAP vs Params, max-distance-to-chord):** z32 (dist 0.568)

> The knee is the capacity point beyond which extra capacity buys progressively less accuracy. Treat it as a **descriptive** summary of this single sweep, not a fitted model.


---

## ⚠️ Statistical caveats (read before quoting any number above)

- **Single seed (0), no repeats.** Every cell above is ONE run. There is no confidence interval and no significance test. Small adjacent-step deltas (especially in da_mIoU / lane_mIoU, which move by ~0.001–0.01) are very likely within run-to-run noise and **must not** be reported as effects.
- **4-epoch protocol.** Absolute accuracies are far from convergence; they are comparable *across rows* (same protocol) but not against published fully-trained baselines.
- **To make a claim about the saturation point**, re-run at least the two candidate Z points with 2–3 seeds and report mean ± std. Until then, describe the curve's shape, not its exact inflection.

