# Phase 1 进度状态（2026-08-29 凌晨）

## Step 完成情况（任务书 §27）

| Step | 内容 | 状态 |
|------|------|------|
| 1-2 | 读现有项目/数据，确认可复用代码 | ✅ YOLOP/TwinLiteNet(+)/TriLiteNet 全部接入 |
| 3 | 统一 baseline evaluation | 🔄 全量 val 评测运行中（9 配置）|
| 4 | Compact Representation（Module A） | ✅ LightEncoder + CompactZ（0.166M/0.82G）|
| 5 | Static multi-task model | ✅ 0.227M / 1.57G，输出与 YOLOP 同构 |
| 6 | Task-Resource Router（Module B） | ✅ Z→Pool→MLP→task logits→离散档位；shared 变体 |
| 7 | 3 档 dynamic width（nested prefix） | ✅ 0.25/0.5/1.0，FLOPs 0.82G→1.57G 随宽度变化 |
| 8 | task-wise allocation | ✅ 逐帧异构分配（CSV 验证）|
| 9 | budget-aware 训练 loss | ✅ L_task + λ_budget·L_budget |
| 10 | Static vs Dynamic 实验 | ⏳ 待训练完成 |
| 11 | Shared vs Task-wise 实验 | ⏳ 待训练（shared 变体）|
| 12 | allocation 可视化 + Pareto | 🛠 工具就绪（visualization/plots.py, exp5_pareto.py）|
| 13 | 失败模式分析 | 🛠 工具就绪（failure_stats）|
| 14 | 阶段评估 | ⏳ |

## Baseline 评测（Step 3，全量 tri_val 10k）

| 模型 | Params | FLOPs | FPS | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|---|---|---|
| YOLOP | 7.94M | 31.3G | 84.1 | 0.766（论文 76.5 ✓）| 0.912（91.5 ✓）| 0.225 |
| TwinLiteNet | 0.44M | 14.1G | 130.9 | — | 0.911（91.5 ✓）| 0.228 |
| TLP nano/small/med/large | … | … | … | … | … | … |
| TriLiteNet tiny/small/base | … | … | … | … | … | … |

## 下一步

1. 等 baseline 全量评测完成（~40 min）
2. **Stage A 训练**（subset 10k 快速验证 → 全量），确认 loss 收敛 + val 指标提升
3. Stage B（冻结 encoder 训 router+heads）→ C（joint）→ D（budget-aware）
4. 训练 task-wise 与 shared 两个 router 变体（Exp2）
5. 跑 Exp1-5 + 失败模式 + 可视化
6. 论文级对比（含 Equal-Budget：TwinLiteNet+ / TriLiteNet / Ours static / Ours dynamic）

## 关键风险
- 训练时长：全量 69,863 张 × 4 阶段，估计 4-6 小时/轮；先用 10k 子集快速迭代
- Router 不收敛到异构分配（失败模式 C）：budget loss 与任务 loss 的平衡需调
- 动态宽度在真实硬件上的 latency 收益待验证（失败模式 D）
