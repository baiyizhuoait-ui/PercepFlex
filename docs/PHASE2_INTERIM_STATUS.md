# Phase 2 — 中间状态与决策点（供审阅）

> 依据任务书 §38 顺序推进。本文汇总至 Round 7 的实际进展、已产出的受控实验、
> 以及一个需要你决策的关键发现。GPU：本地 RTX 5060（8GB）单卡。

---

## A. 目前完成 / 产物

### Step 1 — Equal-Budget baseline（部分）
统一固定 protocol：Stage-A **4ep**、bs16、AdamW lr1e-3、cosine、全量 tri_train、
评测全量 tri_val。已修复数据加载瓶颈（gpu% 个位数→80%，~100ms/step）。

| 模型 | Params | FLOPs | mAP50 | DA_mIoU | Lane_fg | 状态 |
|---|---|---|---|---|---|---|
| A0 (ours 0.235M, no-KD) | 0.230M | 1.47G | 0.2414 | 0.836 | 0.184 | 新训 4ep，已评测 |
| ours 0.5M | ~0.42M | — | — | — | — | 新训 4ep 完成，未评测 |
| ours 1.0M / 2.0M | — | — | — | — | — | 用 Phase1 已有(3ep/2ep) 待评测 |

> 注：4ep 的 A0 mAP(0.241) < Phase1 6ep no-KD(0.268) → Detection 对训练量敏感。
> 决定省 ~2.5h GPU：1.0/2.0M 不重训，复用 Phase1 模型（epoch 差异如实注明）。

### Step 2 — Single-Teacher KD 受控复核（已完成，见下节）
同 9999-tri_train 子集配对（no-KD 对照 vs YOLOP/TLP/TriLite KD），固定 4ep，
全量 tri_val 评测。4 模型 train+eval 全部完成。

### 代码/基建（已提交）
- `docs/PHASE2_PLAN.md`（完整计划+protocol+矩阵+novelty 边界）
- `configs/phase2_stageA_0.235M.yaml`（固定 protocol）
- `distillation/multi_teacher.py`（Step-3：per-teacher projection→common latent→student Z，无 attention）
- `scripts/bench_latency.py`（Step-9：eager+compile，P50/P95/FPS/Mem）
- `scripts/phase2_step1_budget.sh` / `phase2_step2_singleKD.sh` / `phase2_step2_eval.sh`
- `evaluate_baseline._profile_ours`（真实 FLOPs）修复

---

## B. 关键发现（Step 2）— 需要你决策

### 干净配对结果（同 9999 子集、固定 4ep、全量 tri_val 评测）
| 模型 | mAP50 | DA_mIoU | Lane_fg |
|---|---|---|---|
| no-KD 对照 | 0.0763 | 0.7770 | 0.1352 |
| +YOLOP KD | 0.0788 | 0.7678 | 0.1372 |
| +TwinLite+ KD | 0.0689 | 0.7634 | 0.1330 |
| +TriLite KD | 0.0636 | 0.7742 | 0.1362 |

**单 teacher KD 在本受控设置下增益≈0 甚至略负**（YOLOP mAP +0.0025、TLP −0.007、TriLite −0.013；DA 略降、Lane 中性）。
与 Phase1 声称 +0.012 mAP 不符 → Phase1 那次提升被"KD 用 10k / no-KD 用 69k"混淆。

### 解读（为何可能不帮）
1. **9999 子集模型欠训练**（mAP 0.076 vs 全量 0.268）：数据稀缺时 feature-KD 强正则干扰弱学生拟合。
2. **DA/Lane 早已饱和**（Phase1 H2：0.5M 即饱和）→ 对它们 KD 无益（delta≈0）一致。
3. **feature 对齐只作用于 student Z（喂 seg）**；det 读多尺度不经 Z → 对 det 增益微弱。
4. Teacher 与极小 student 在数据饥渴下统计不对齐。

### 两种解读
- (i) KD 在真正受控下本就不可靠（尤其分割任务已饱和、检测不经共享瓶颈）；
- (ii) 9999 病态，不代表全量——需全量受控复核（给 69863 生成 teacher cache ~50GB 或在线 KD）调和。

---

## C. 决策点（供你后续决定，不影响已记录结果）
1. **KD 是否作为 Phase-2 基石**：决定 Step-3 multi-teacher(A4) / Step-5 KD+Adapter(A6) /
   Step-7/8 elastic+KD(A8/A9) 是否值得做。
   - 路径 α：接受负结果 → 降级/跳过 KD 组合，主攻 **Compact + Adapter + Elastic(global)**
     （任务书 §31/§34 Route B）。
   - 路径 β：全量受控 KD 复核调和后再定（严谨、耗时长）。
   - 路径 γ：视 9999 为伪影，用 Phase1 全量数字当参考继续 multi-teacher（接受小数据 regime）。
2. **Step-1 剩余**：0.5M 评测 + 1.0/2.0M(Phase1) 评测 + latency 分层。

## D. 下一步（待你决策后）
按你选择的路径推进；无论哪条，先把 Step-1 剩余评测 + latency 补齐，形成等预算表与 Pareto，
再进入你拍板的 KD/Adapter/Elastic 主攻路线。

## 补充决策信息（Round 9）
### Step-1 同预算对齐：我们并不领先既有轻量模型（绝对精度）
同参数下 ours 的 mAP/DA/Lane **均落后**官方 pretrained TriLiteNet / TwinLiteNet+
(检测 0.24 vs TriLite-tiny 0.50; DA 0.84 vs TLP 0.92)，但 **FLOPs 低 6-10×**
(ours 0.235M=1.47G vs TriLite-tiny=1.83G vs TLP-medium=15G)。注：baseline 官方权重
=全量+充分训练，我们=tri_train 4ep，非完全同protocol。
→ 倾向 Route B(紧凑+低FLOPs+弹性价值)，基本排除 Route A(同预算超 TriLite/TwinLite)。
### 两处关键证据汇总(供 Route 决策)
- Step2: 受控 9999 下 KD 无稳定增益 → KD 不宜当基石
- Step1: 同参数绝对精度落后既有轻量模型 → 不走 Route A 精度竞赛
两者共同指向 Route B：**Compact + Adapter + Elastic(global)**, 价值=紧凑/低算力/一模型多档。
