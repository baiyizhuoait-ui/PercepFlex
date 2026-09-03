# AGENT 交接提示词 — PercepFlex Phase 2-B（Compact Representation Core Discovery）

> 用途：把当前阶段完整交接给新 agent。新 agent 无本会话上下文，请以下文为准，
> 先读指定文档/代码再动手。遵守文末"硬约束"，不要臆造结果。

---

## 一、项目背景

**任务**：BDD100K 三任务驾驶感知 —— ① Object Detection（单类车）② Drivable Area 分割 ③ Lane 分割。
**目标**：极小模型、实时、三任务保精度、面向真实边缘部署；EI/工程类论文；核心是找"不撞车"的论文创新点，
不是堆模块。
**现状路线**：Phase 2-B = "Compact Representation Core Discovery"——研究一个极小共享瓶颈表示 Compact Z，
是否是三任务共同、参数/算力高效的信息瓶颈；重点找**可测量的科学关系**（容量↔每任务精度↔饱和点↔算力），
而非"我做了一个共享 feature"。

**代码/数据/环境（全部已验证可用）**
- 项目根：`/home/mycode/ai_study/trac`（WSL Ubuntu 24 核；不要用 `ai_study/trac` 以外的旧路径）
- 数据：`data/bdd100k/`（11G；tri_train 69863 / tri_val 10000 交集划分已就位）
- Python：`/home/mycode/ai_study/gpu_env/bin/python`（torch 2.11.0+cu130，RTX 5060 8GB，cuda OK）
- 训练：`training/train.py --config <cfg> [--epochs 4] [--num-images N] [--init ckpt] [--outdir ...]`
  （脚本自带 GPU 横幅+实时进度+拒绝静默 CPU；DataLoader 已 auto-workers，gpu 利用率 ~80%）
- 评测：`evaluation/evaluate_baseline.py --baseline OursStatic --preset <checkpoint路径> --outdir <dir>`
  → 产出 metrics.json（mAP50/mAP95/DA mIoU/DA fgIoU/Lane fgIoU/Lane mIoU/fps）且 log 内含
  `[profile] params=..M flops=..G`（真实切片感知 FLOPs）。全量 tri_val 一次 ~5–8 分钟。
- 固定 Stage-A protocol：4ep、bs16、AdamW lr1e-3、cosine、seed0。config 见 `configs/phase2_stageA_0.235M.yaml`
  （0.235M 结构：encoder stem16/stages[32,64,96,128]/z64）；容量档参考 `configs/train_stageA_{0.5M,1.0M,2.0M}.yaml`。
- 离线 teacher cache：`experiments/expF_single_teacher/kd_cache/`（YOLOP/TLP/TriLite 各 9999 图 feat+da+lane）。
- Git：独立仓库，分支 `main`，remote `git@github.com:baiyizhuoait-ui/PercepFlex.git`（推送用
  `GIT_SSH_COMMAND="ssh -i /home/mycode/.ssh/id_ed25519_trac -o IdentitiesOnly=yes -o BatchMode=yes"`）。
- 产物目录：`experiments/phase2/`（Phase2 各步结果）。

**Phase 1 关键结论（背景，勿重跑）**
1. Compact Representation 是最稳定、最值得继续的方向。
2. task-wise dynamic router 失败（collapse/不稳定）→ **禁止再研究**。
3. 紧凑 Z 容量↑时：Detection 提升最明显、Lane 次之、DA 早饱和 → 任务对共享表示容量敏感度不同。
4. 旧 KD 结论有 数据量/epoch 混淆（KD 用 10k、no-KD 用 69k）→ 不可作最终结论。
5. 结构问题：**Detection 路径当前不经 Compact Z**（det 直接读 encoder 多尺度 F2/F3/F4；
   DA/Lane 读 Z）→ 对 Z 做 KD 无法解释 det 提升。

---

## 二、已经做完的任务（可信、勿重做；相关文档先读）

1. **Phase 2 Step 1 等预算 baseline**（固定 4ep 协议，全量 tri_val）→ `experiments/phase2/STEP1_REPORT.md`
   | 0.235M(0.23M/1.47G): mAP .2414 DA .836 Lane .184 | 0.5M(0.42M/2.46G): .2846/.833/.192 |
   | 1.0M(0.96M/4.22G): .2951/.852/.194 | 2.0M(1.98M/8.84G): .3187/.845/.194（1.0/2.0M 为 Phase1 3ep/2ep 模型）
   - A0 eager latency: p50 3.73ms p95 10.44ms fps184；0.5M p50 3.94ms p95 10.95ms fps178。
2. **Phase 2 Step 2 受控 KD 复核（重要负结果）** → `experiments/phase2/STEP2_REPORT.md`
   - 同一 9999 训练子集配对 no-KD vs YOLOP/TLP/TriLite KD（固定4ep）：**KD 增益≈0 甚至略负**
     （YOLOP mAP +.0025 / TLP −.007 / TriLite −.013；DA 略降 Lane 中性）。9999 下模型欠训练(mAP .076)，
     解释待定；**未经全量受控复核前不宣称 KD 稳定有益**。
3. **同预算对齐参考（诚实结论）** → `STEP1_REPORT.md` 尾部：同参数下我们 mAP/DA/Lane 落后官方
   pretrained TriLite/TwinLite+（det 0.24 vs 0.50），但 FLOPs 低 6–10× → 基本排除 Route A（同预算精度竞赛）。
4. **Literature Novelty Audit（Phase 2-B Task 1，已完成）** → `docs/PHASE2B_LITERATURE_NOVELTY.md`
   - 18 篇碰撞矩阵 + A/B/C/D：**D 类空缺=在 BDD100K 三任务极低算力下对同一紧凑瓶颈做容量扫描 +
     每任务饱和点 + 等预算 shared-vs-widening 效率实证**；C 类雷区=别把"共享表示/IB-in-MTL/跨架构KD/
     驾驶+pruning+KD(arXiv:2511.05557)"当首发。
   - 初始假设（一句话）与 interim 推荐已写入该文件（倾向 CONTINUE，实验失败即 PIVOT）。
5. **已建基建（可复用，勿重建）**：`distillation/multi_teacher.py`（MultiTeacherKD，暂不用）、
   `scripts/bench_latency.py`（eager/compile latency）、`scripts/phase2_step2_singleKD.sh`、
   `evaluate_baseline._profile_ours`（真实 FLOPs）、`configs/phase2_stageA_0.235M.yaml`。
   Git 已同步 `main`（最新 `f33552a`）。

---

## 三、接下来的任务（按此顺序；先别做 Task1 之外的事）

### Task 2（本轮核心）— Compact Z 实验设计（均需 R0/R1/R2 前先建代码）
按 `docs/PHASE2B_LITERATURE_NOVELTY.md` 的 D 类空缺设计，**不要先堆模块**：

**(a) Representation Capacity Sweep**：固定其余一切，扫 Z channels ∈ {16,32,48,64,96,128}
（改 `configs/` 的 z_channels，或新增 sweep config；每档训 4ep 全量 tri_train → 评测）。
记录：Params/FLOPs/FPS/mAP50/DA mIoU/Lane IoU/每任务相对增益/综合 utility。
找：**diminishing-return 区间**（不是"越大越好"）；画 Z vs accuracy / FLOPs / FPS / utility / Pareto。

**(b) 每任务容量敏感度分析**：ΔDetection/ΔDA/ΔLane per Z-step（Z16→32→48→…）；
验证"不同任务对共享表示容量需求不同、饱和点不同"（此观察是方法动机，本身非创新）。

**(c) R0/R1/R2 — Detection 是否应进 Compact Z（最关键结构问题）**：
- R0 现状：Encoder →[det 读多尺度；Z→DA+Lane]
- R1 Fully Shared：Encoder → Z → det+DA+Lane 全走 Z
- R2 Hybrid（**优先**）：Encoder → Z；det 只经**极轻量 projection 从 Z 取**（禁止 det 直接用大 feature）
回答：一个统一 Compact Z 能否承担三任务主要信息交换？（此决定 KD 是否在逻辑上成立——见 (e)）

**(d) Representation Utility / 信息损失**：定义简单 Utility(Z)=归一化三任务加权（或 det·DA·Lane 组合），
报告 Utility/FLOPs、Utility/Params、每任务 drop vs Z 缩小；找**非同步退化/degradation pattern**
（"compact rep 是受限信息分配问题，不只是小 feature"）。先实证，别引入复杂 IB 数学。

**(e) Cross-Architecture KD（仅辅助，且必须先修好 (c)）**：若单 teacher KD 在受控下无稳定增益 →
**停止 Multi-Teacher**。仅当 det 真正使用 Z 后，才做 YOLOP→student 的 No KD / Output KD / Feature KD /
Feature+Output KD 四格；无效即停。

**(f) Parameter-matched 效率对比（论文价值所在）**：0.25/0.5/1M 三档下设计
A 普通 widening（同参） / B Compact Z / C Compact Z+轻头 / D (+KD 若有效)，参数量对齐后比较
accuracy / FLOPs / FPS。验证："同样预算下 Compact Representation 是否比单纯加宽更有效利用参数"。

### Task 3 — 收尾输出（实验完成后，不要提前）
最终给出（用户 §13 要求 7 项）：① Novelty Report（≥15 篇）② Collision Matrix ③ Compact
Representation Hypothesis（一句话，可用审计版）④ 实验汇总表 ⑤ 明确推荐 **STOP / CONTINUE / PIVOT** 及理由
⑥ 2–3 个候选论文标题（暂定）⑦ 下一阶段仅保留 2–4 个最有价值实验。

---

## 四、硬约束（务必遵守）

1. **Novelty 边界**：禁止把以下包装成"首次提出"——共享 feature/统一多任务表示(Sparse U-PDP)、
   task/channel 动态路由(D2BNet)、轻量 attention 三头(SCAM-P)、轻量多档配置(TwinLite/TriLite/YOLOP)、
   弹性宽度(OFA/SN-Net/DS-Net)、预算感知 MTL(EC-MT)、跨架构/异构 KD、驾驶多任务+pruning+KD(arXiv:2511.05557)。
   审计发现高度撞车 → 立即 PIVOT，不要围绕已解决问题堆模块。
2. **诚实记录**：新结果与旧结果冲突时**不覆盖旧结果**，注明 old/new/explanation；不臆造数字。
3. **先实验后结论**：不允许 task-wise dynamic router / MoE / VQ / Transformer 大模块 / 大型 attention /
   temporal / optical flow / dynamic resolution / 复杂 pruning / NAS / RL routing / 大规模 Multi-Teacher。
4. **每个实验记录**：Params/FLOPs/FPS/三任务指标/checkpoint/seed/epochs/**git commit hash**。
5. 一次 bash 调用约 120s 上限：长 GPU 训练放后台(run_in_background)并轮询日志；后台任务跨轮持续。
6. 失败不阻塞：某个机制无益就删除并如实报告（任务书 §29/§36 精神）。
7. 单卡 RTX 5060 8GB：全量 4ep 训练 0.235M ≈ 27min，0.5M ≈ 36min；评测 ≈ 5–8min/模型；
   大训练勿并行；显存 >7GB 风险时降 batch 到 8。
