# Phase 2 Step 1 — Equal-Budget Baseline (运行日志)

## Fixed Stage-A protocol
epochs=4, bs16, AdamW lr1e-3 wd5e-4, cosine, tri_train全量, seed0.
（train.py auto num_workers=8/pin_memory — 已修复数据瓶颈，gpu% 75-89, ~100ms/step）

## A0: 0.235M static no-KD (锚定参考) — DONE
| Params | eval_fwd | mAP50 | mAP50_95 | DA mIoU | DA fgIoU | Lane fgIoU | Lane mIoU | Lane lineAcc |
|---|---|---|---|---|---|---|---|---|
| 0.230M | 6.38ms | 0.2414 | 0.0785 | 0.836 | 0.7418 | 0.1843 | 0.5802 | 0.8138 |
(全量 tri_val, 10k imgs, det_n_gt=108363)

### 观察
- 4ep 的 A0 mAP(0.241) < Phase1 6ep no-KD(0.268) → Detection 对训练量也敏感。
  固定 protocol 需权衡：epochs 越大 A0 越强但矩阵成本×epochs。
  暂定 4ep 保持矩阵内部一致；若 Route 判定(对 TriLite/TwinLite 同预算)需要更强基线再上调。
- eval 工具链验证通过：train→eval 端到端跑通，指标合理。mAP@.5:.95 在 10k 图较慢(~min级)但非挂起。
- det_n_pred 2.76M/10k (276/img) — NMS conf=0.001 正常，非上次 3M/图 的病态。

### A0 profile (eager, bs1)
params=0.230M  flops=1.473G  (专用 latency P50/P95 见 Step-9 harness)

### Budget chain (fixed 4ep protocol)
- A_budget_0.5M: DONE (avg_loss 0.2347, ep4) @ 14:32
- A_budget_1.0M / 2.0M: 后台训练中 (串行)

### Step-2 single-teacher KD recheck (fixed 4ep, SAME 9999-train-subset paired)
- s2_nokd: DONE (control)
- s2_kd_YOLOP: DONE (avg_loss 0.4637)
- s2_kd_TLP / s2_kd_TriLite: 后台训练中

## 等预算表（统一评测 tri_val; A0/0.5M=固定4ep, 1.0/2.0M=Phase1 3/2ep 注记）
| budget | params | FLOPs | mAP50 | DA_mIoU | DA_fg | Lane_fg | Lane_m | fps |
|---|---|---|---|---|---|---|---|---|
| A0 0.235M | 0.230M | 1.47G | 0.2414 | 0.8360 | 0.7418 | 0.1843 | 0.5802 | 157 |
| 0.5M | 0.424M | 2.46G | 0.2846 | 0.8334 | 0.7384 | 0.1915 | 0.5837 | 186 |
| 1.0M | 0.959M | 4.22G | 0.2951 | 0.8518 | 0.7654 | 0.1943 | 0.5855 | 179 |
| 2.0M | 1.980M | 8.84G | 0.3187 | 0.8450 | 0.7555 | 0.1939 | 0.5853 | 158 |
趋势：mAP 随容量↑(0.24→0.32)，Detection 最敏感；DA/Lane 早饱和(0.5M)。确认 Phase1 H2。
评测: experiments/phase2/s1_eval/{A_budget_0.5M,ph_1.0M,ph_2.0M}; A0: A0_static_nokd_eval

### A0 latency (bs1, 640, RTX5060, bench_latency.py)
- eager: mean 5.45ms, p50 3.73ms, p95 10.44ms, fps 184, peak_mem 29MiB(forward)
- torch.compile: 在此环境(RTX5060/Blackwell + torch2.11) inductor 报错 → 记录为部署限制，待服务器/稳定环境再测

## 同预算对齐参考（官方 pretrained, tri_val）
| 模型 | params | FLOPs | mAP50 | DA_mIoU | Lane_fg |
|---|---|---|---|---|---|
| Ours 0.235M | 0.230M | 1.47G | 0.2414 | 0.8360 | 0.1843 |
| TriLite-tiny | 0.151M | 1.83G | 0.4953 | 0.8796 | 0.1952 |
| TriLite-small | 0.592M | 6.61G | 0.6326 | 0.9053 | 0.2198 |
| TriLite-base | 2.350M | 25.4G | 0.7235 | 0.9203 | 0.2371 |
| Ours 0.5M | 0.424M | 2.46G | 0.2846 | 0.8334 | 0.1915 |
| TLP-medium | 0.479M | 15.4G | (n/a) | 0.9191 | 0.2297 |
| TLP-large | 1.944M | 58.6G | (n/a) | 0.9279 | 0.2450 |
| Ours 2.0M | 1.980M | 8.84G | 0.3187 | 0.8450 | 0.1939 |

## 诚实解读 (影响 §35 Route 判定)
- 我们的 FLOPs 效率远高：同 params 下 FLOPs 低 ~6-10x（紧凑 1/8 表示）。
- **但同参数下 mAP/DA/Lane 均明显落后官方 pretrained TriLite/TLP**，尤其检测(0.24 vs 0.50 tiny)。
- **不对等注记**：baseline 官方权重 = 全量 BDD100K + 充分训练；我们 = tri_train(69863) + 固定4ep。
  公平同数据对比需把 baseline 也按我们 protocol 重训(代价大)。
- 当前方向性：**不太可能走 Route A(同预算超 TriLite/TwinLite)**；更可能 Route B
  (紧凑+低FLOPs+弹性的价值，而非绝对精度对标既有轻量模型)。待用户决策。
### latency (eager bs1/640, RTX5060)
- A0 0.235M: p50 3.73ms / p95 10.44ms / fps 184 / peak 29MiB
- 0.5M: p50 3.94ms / p95 10.95ms / fps 178 / peak 36MiB
