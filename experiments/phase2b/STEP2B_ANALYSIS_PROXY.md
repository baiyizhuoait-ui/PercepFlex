# Phase-2-B Task 2 — Capacity / Sensitivity / Utility Analysis (`proxy` mode)

> **PROXY** (joint encoder+Z capacity from existing trusted equal-budget runs; 1.0M/2.0M are 3ep/2ep, NOT the fixed 4ep protocol). The dedicated Z-isolated sweep (`mode auto`) supersedes this once it runs.


## Raw table

| label | z | params(M) | flops(G) | fps | mAP50 | da_mIoU | lane_fg | lane_mIoU |
|---|---|---|---|---|---|---|---|---|
| 0.235M | - | 0.230 | 1.47 | 157 | 0.2414 | 0.8360 | 0.1843 | 0.5802 |
| 0.5M | - | 0.424 | 2.46 | 186 | 0.2846 | 0.8334 | 0.1915 | 0.5837 |
| 1.0M | - | 0.959 | 4.22 | 179 | 0.2951 | 0.8518 | 0.1943 | 0.5855 |
| 2.0M | - | 1.980 | 8.84 | 158 | 0.3187 | 0.8450 | 0.1939 | 0.5853 |

## (d) Per-task retention vs best (non-synchronous degradation)

| label | mAP50 ret% | da_mIoU ret% | lane_fg ret% | lane_mIoU ret% |
|---|---|---|---|---|
| 0.235M | 75.7 | 98.1 | 94.9 | 99.1 |
| 0.5M | 89.3 | 97.8 | 98.6 | 99.7 |
| 1.0M | 92.6 | 100.0 | 100.0 | 100.0 |
| 2.0M | 100.0 | 99.2 | 99.8 | 100.0 |

> Detection retains the least at low capacity; DA the most — tasks degrade **non-synchronously** (Compact Z is a contested information-allocation problem, not merely a small feature map).


## (b) Per-step capacity sensitivity (Δ per capacity/Z step)

| step | ΔmAP50 | Δda_mIoU | Δlane_fg | Δflops(G) |
|---|---|---|---|---|
| 0.5M->0.235M | +0.0432 | -0.0026 | +0.0072 | +0.99 |
| 1.0M->0.5M | +0.0105 | +0.0184 | +0.0028 | +1.76 |
| 2.0M->1.0M | +0.0236 | -0.0068 | -0.0004 | +4.62 |

> Detection gains the most per step (most capacity-hungry); DA/Lane saturate early — confirms Phase-1 H2 / Exp-D D3-2 (Detection > Lane > DA sensitivity).


## (a) Utility, efficiency, and Pareto

| label | Utility(0-1) | U/params(1/M) | U/flops(1/G) | mAP/flops |
|---|---|---|---|---|
| 0.235M | 0.890 | 3.87 | 0.605 | 0.164 |
| 0.5M | 0.951 | 2.24 | 0.387 | 0.116 |
| 1.0M | 0.975 | 1.02 | 0.231 | 0.070 |
| 2.0M | 0.997 | 0.50 | 0.113 | 0.036 |

- **Pareto frontier (mAP vs FLOPs):** 0.235M, 0.5M, 1.0M, 2.0M
- **Pareto frontier (mAP vs Params):** 0.235M, 0.5M, 1.0M, 2.0M

> Diminishing-return region = where ΔmAP per +ΔFLOPs falls toward ~0 while FLOPs/params keep rising. Identify the knee from the per-step table above once the dedicated Z-sweep (auto) data is available.

