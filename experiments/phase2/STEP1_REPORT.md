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
