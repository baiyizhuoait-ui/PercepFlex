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
