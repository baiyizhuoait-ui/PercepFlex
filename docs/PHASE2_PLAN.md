# PHASE 2 — Execution Plan & Fixed Protocol

> 依据任务书 §38 顺序执行。本文档定义 Phase 2 全程统一的训练/评测 protocol，
> 映射 A0–A9 矩阵到具体 config/脚本/命令，并记录每一步状态。
> 全程遵守 novelty 边界（§23-24）：不复制 D2BNet/Sparse U-PDP/SCAM 结构，
> 不声称已有工作的"首发"。无效模块即删（§29/§36）。

## 0. 结论基线（来自 Phase 1 / 1-B，勿重跑）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | Lane fgIoU | 备注 |
|---|---|---|---|---|---|---|
| ours_0.23M (static, no-KD) | 0.235M | 1.57G | 0.268 | 0.842 | 0.180 | 全 tri_val |
| ours + KD(YOLOP) | 0.235M | 1.57G | 0.280 | 0.850 | 0.193 | 单 teacher 最佳 |
| ours_0.5M | 0.424M | 2.46G | 0.282 | 0.840 | 0.193 | 容量档（4ep）|
| ours_1.0M | 0.959M | 4.22G | 0.295 | 0.852 | 0.194 | 容量档（3ep）|
| ours_2.0M | 1.980M | 8.83G | 0.319 | 0.845 | 0.194 | 容量档（2ep）|
| YOLOP (teacher) | — | — | 0.766 | 0.912 | 0.225 | 官方权重 |
| TwinLiteNet | — | — | — | 0.911 DA | — | 官方权重 |
| TriLiteNet-tiny | — | — | 0.495 | 0.880 | 0.195 | 官方权重 |

**容量档结论（H2）**：Detection > Lane > DA 对容量敏感；DA/Lane 在 0.5M 即饱和。
注意：上述容量档用 4/3/2 非一致 epoch，仅作容量趋势上下文，不参与 Phase-2 等预算表。

## 1. Fixed Stage-A Protocol（Phase 2 全矩阵统一）

目标：A0–A9 与容量档在同 protocol 下可比。

| 项 | 值 |
|---|---|
| data | `tri_train` 全量(69,863) 训练；`tri_val` 全量(10,000) 评测 |
| batch_size | 16（drop_last）|
| optimizer | AdamW, lr 1e-3, wd 5e-4 |
| scheduler | CosineAnnealingLR over `epochs × steps` |
| epochs (Stage A) | **4**（与 Phase1 容量 4ep 可比、算力可控）|
| num_workers | auto ~ cores（本地 24 核 → 8）；pin_memory=True；prefetch=4 |
| seed | 0（主结果）；Step A 加 2 额外 seed 估方差 |
| device | cuda（脚本自动，拒绝静默 CPU）|
| grad clip | 10.0 |

推理基线统一评测：640×640 letterbox、单类车辆 mAP@0.5、DA/Lane 二值 Acc/fgIoU/mIoU、
CUDA-sync、bs=1、全 tri_val。评测脚本 `evaluation/evaluate_baseline.py`。

## 2. Step 1 — Equal-Budget Baseline

需要在一个固定 protocol 下重训 4 档容量（现有档 epoch 不一致，仅作趋势参考）：
- configs: `train_stageA_eb.yaml` 基础上派生 0.25/0.5/1.0/2.0M（模型段扩 encoder+z 到目标 params）
- 每档跑 3 seeds 估方差；评测 + 分层 latency
- 对齐基线：TwinLiteNet+ / TriLiteNet（官方权重，评测脚本现成）
- 报告列：Params, FLOPs, mAP50, DA mIoU, Lane fgIoU, FPS, P50, P95, Memory
- 比较：parameter-matched + FLOPs-matched

**状态：规划完成，待启动训练。**

## 3. Step 2 — Single-Teacher KD 复核（固定 protocol）

复用已有 cross-arch KD 机制，固定 protocol 复核 4 行：
- A0 no-KD（= Step1 0.235M 参考）
- A1 +YOLOP KD
- A2 +TwinLiteNet+ KD
- A3 +TriLiteNet KD
记录 ΔmAP / ΔDA / ΔLane。KD 走离线 cache（`scripts/precompute_teacher.py`）避免每步 teacher 开销。

**状态：待跑。**

## 4. Step 3 — Multi-Teacher KD（Exp H，本次真正新建）

- Teacher：YOLOP / TwinLiteNet+ / TriLiteNet
- 组合：Y+Tw, Y+Tr, Tw+Tr, Y+Tw+Tr
- 每 teacher 走 `Teacher-specific projection P_i → common latent`，融合（首版 3 个极简方法）：
  M1 Average；M2 Fixed task-aware weight（YOLOP→det，Twin/Tri→seg）；M3 confidence-weighted
- 第一版**不做** attention fusion / 复杂 Teacher Fusion Network
- KD 损失施加于：student Z（feature 对齐到 common latent）+ 输出（seg sigmoid-MSE）
- 必须回答 Q1（互补性）/Q2（冗余）/Q3（task-wise 最佳 teacher）
- 可能结果：multi 反而 ≤ YOLOP-only（§28）——记录 teacher contribution/conflict/task-wise gain

**状态：需新建 config + loss + teacher-projection 模块。**

## 5. Step 4/5 — Static Task Adapter（+ KD）

- Adapter 极小：先 1×1 conv（每任务一个）；不够再 dw3×3+1×1。绝非 dynamic routing，是"极小预算下给三个头的固定专属容量"。
- 挂在 shared Compact Z 之后、每 head 之前。
- 必须 equal-parameter 消融：Adapter 增加 ~0.05M 时，用同等 generic 加宽模型对照，判断是"任务特化有效"还是"纯变大"。
- Step 5 四格：A=Shared no-KD, B=Shared+KD, C=Adapter no-KD, D=Adapter+KD。
- D>B 且 D>C 才保留；否则删（§29）。

**状态：需新建 adapter 模块 + equal-param 对照 config。**

## 6. Step 6-8 — Elastic Student

- **禁止 task-wise width**（router→每任务独立宽）——旧问题。只做 global capacity：router 输出单标量档位 Tiny/Medium/Large，三任务共享同一容量状态。
- 只保留原 Router 作 negative baseline。
- 训练顺序（§18，避免旧 Dynamic+KD 崩溃）：
  1) 先稳定 KD student（Step2 产出）
  2) 在该 student 上训练 elastic subnet（各 width 充分采样）
- Adapter 必须兼容 0.25×/0.5×/1.0× width。

**状态：待新建 global-capacity elastic 模型 + elastic 训练脚本。**

## 7. Step 9 — 全 Pareto / Latency 分层

- 指标：Accuracy–Params / Accuracy–FLOPs / Accuracy–Latency
- latency 分层记录：PyTorch eager → torch.compile → （条件允许）ONNX / TensorRT
- FLOPs↓ 不直接等同 latency↓（eager 已观察失真），必须实跑 latency
- 借鉴 SCAM-P 测法（1000 frames + 100 warmup），但本机只有 RTX 5060，不虚构 edge 数据

**状态：待 Step1 后建 benchmark 脚本。**

## 8. A0–A9 完整矩阵（最终报告用）

| id | 模型 | KD | Multi-T | Adapter | Elastic |
|---|---|---|---|---|---|
| A0 | Ours static | × | × | × | × |
| A1 | +YOLOP KD | ✓ | × | × | × |
| A2 | +TwinLite KD | ✓ | × | × | × |
| A3 | +TriLite KD | ✓ | × | × | × |
| A4 | +Multi-Teacher KD | ✓ | ✓ | × | × |
| A5 | +Task Adapter | × | × | ✓ | × |
| A6 | +KD+Adapter | ✓ | ✓ | ✓ | × |
| A7 | Ours Elastic | × | × | × | ✓ |
| A8 | +KD | ✓ | ✓ | × | ✓ |
| A9 | +KD+Adapter | ✓ | ✓ | ✓ | ✓ |

## 9. 退出路线（§31-35，避免强行复杂化）

- Multi-teacher 无增益 → `Compact + best single + Elastic`
- Adapter 无增益 → `Compact + KD + Elastic`
- Elastic 无真实 latency 收益 → 不宣称 speedup，宣称 "configurable computational capacity"（deployment limitation 如实分析）
- 若同预算超 TriLite/TwinLite → Route A；否则 Route B（single compact elastic MT model）

## 10. 每步汇报格式（§37，17 项）
每步产出：目的 / 相关论文 / 实现 / Params / FLOPs / mAP50 / DA mIoU / Lane fgIoU /
FPS / P50 / P95 / Memory / vs previous baseline / 是否支持假设 / 新 collision /
是否保留 / 下一步。
