# 驾驶感知多任务轻量模型研究 —— 完整工作报告

> 覆盖区间：Phase 1（MVP）→ Phase 1-B（Compact + Dynamic + KD 验证）全部完成
> 撰写日期：2026-08-30 ｜ 环境：WSL2 / RTX 5060 Laptop 8GB / Python 3.12 / torch 2.11+cu130

---

## 一、项目背景与目标

研究**面向资源受限设备的轻量多任务驾驶感知**（Detection + Drivable Area + Lane）：

> 能否学习一个 Compact Driving Representation，并由 Task-Resource Router 按「任务 × 资源 × 表示」动态分配模型容量，让**同一个模型**在不同计算预算下运行？

核心假设：**H1**（task-wise 动态分配优于统一宽度）、**H2**（紧凑表示保留三任务信息）、**H3**（单模型覆盖 Pareto 曲线）、**H4**（跨架构 teacher 蒸馏提升紧凑表示）。

---

## 二、基础设施（Phase 0）

- 工作区 `trac/`（独立 git，53 commits），数据 BDD100K 全量（tri_train 69,863 / tri_val 10,000）
- 基线全部接入并复现论文：YOLOP、TwinLiteNet、TwinLiteNet+(nano/small/medium/large)、TriLiteNet(tiny/small/base)
- 统一评测/训练协议：640×640、bs=1、CUDA sync、全量 tri_val；模型（Ours）0.235M 参数

---

## 三、Phase 1（MVP）结果

### 模型结构
`Input → LightEncoder(0.166M) → Compact Z(64ch@1/8) → Task-Resource Router → 动态三头(nested-prefix 0.25/0.5/1.0)`，全模型 **0.235M / 1.57G**。

### 统一基线（全量 tri_val，含论文复现）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|---|---|
| YOLOP（B1，复现论文 76.5/91.5 ✓） | 7.94M | 31.3G | 0.766 | 0.912 | 0.225 |
| TwinLiteNet（B2，复现 91.5 ✓） | 0.44M | 14.1G | — | 0.911 | 0.228 |
| TwinLiteNet+ nano | 0.03M | 1.9G | — | 0.863 | 0.186 |
| TriLiteNet tiny/small/base | 0.15/0.59/2.35M | 1.8/6.6/25.4G | 0.495/0.633/0.724 | 0.880/0.905/0.920 | 0.195/0.220/0.237 |

### Ours 三档 + Dynamic（全量 val）

| 模型 | FLOPs | mAP50 | DA mIoU | Lane fgIoU | FPS |
|---|---|---|---|---|---|
| Ours Static 0.25/0.5/1.0 | 0.78/0.92/1.47G | 0.234/0.254/0.261 | 0.805/0.867/0.854 | 0.125/0.194/0.184 | 129/131/141 |
| Ours Dynamic（平均 0.90G） | 0.90G | 0.243 | 0.837 | 0.167 | 136 |

**Phase 1 关键发现**：Dynamic ≈ Static 插值（H3 支持）；但 **H1 弱**（learned<random<fixed），
oracle 上界只有 +0.0125 DA。→ 进入 Phase 1-B 深入验证。

---

## 四、Phase 1-B 四个实验

### Experiment A：等预算 Static vs Dynamic（3 seeds，全量）

| 模型（mean±std） | mAP50 | DA mIoU |
|---|---|---|
| Static-EB（独立训练 @0.45=0.90G） | **0.2555±0.003** | 0.8364±0.003 |
| Dynamic（A6-B2-C2-D2 管线，0.90G） | 0.1176±0.109 | 0.766±0.081 |

**Case C 确认**：等平均 FLOPs 下独立训练的 Static 全面优于 Dynamic，且 Dynamic 训练高度不稳定
（部分种子 router 对 DA 塌缩到 tiny → 头弱化）。**H1 不成立**。

### Experiment D：紧凑表示容量（0.23→2.0M，全量）

| 模型 | Params | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|---|
| ours_0.23M | 0.235M | 0.252 | 0.834 | 0.174 |
| ours_0.5M | 0.424M | 0.282 | 0.840 | 0.193 |
| ours_1.0M | 0.959M | 0.295 | 0.852 | 0.194 |
| ours_2.0M | 1.980M | 0.319 | 0.845 | 0.194 |

**H2 支持**：0.23M 保留 2.0M 的 mAP 0.79× / DA 0.99× / Lane 0.90×；容量敏感性 **Detection > Lane > DA**
（DA 在 0.5M 即饱和）——解释了路由收益上界小的原因。

### Experiment F/G：单 teacher 跨架构 KD（全量在线 + 10k×10 公平离线）

全量在线（YOLOP teacher，6 epoch）：
| 模型 | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|
| no-KD | 0.2682 | 0.8423 | 0.1801 |
| **+YOLOP KD** | **0.2798** (+0.012) | **0.8495** (+0.007) | **0.1932** (+0.013) |

公平离线（10k×10，全量 val 评测）：
| 模型 | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|
| no_kd | 0.1425 | 0.7989 | 0.1538 |
| +YOLOP | 0.1559 (+0.013) | 0.8135 (+0.015) | 0.1556 |
| +TwinLiteNet+ | 0.1434 (+0.001) | 0.8197 (+0.021) | 0.1615 (+0.008) |
| +TriLiteNet | 0.1443 (+0.002) | 0.8206 (+0.022) | 0.1589 (+0.005) |

**H4 支持**：三 teacher 均提升 student（DA +0.015~0.022）；YOLOP 检测最佳、综合最稳 → 选为 Exp I teacher。

### Experiment I：KD 后重测 Dynamic（四模型）

| 模型 | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|
| A Static no-KD | 0.2682 | 0.8423 | 0.1801 |
| B Dynamic no-KD | 0.2427 | 0.8372 | 0.1666 |
| **C Static+KD** | **0.2798** | **0.8495** | **0.1932** |
| D Dynamic+KD（全量公平） | 0.0458 | 0.4406 | 0.1094 |

**判定**：C-A = **+0.012/+0.007/+0.013（KD 有效 ✓）**；B-A 负；D-C 严重负——诊断确认
**router 对 DA 塌缩到 tiny（57%），窄宽度 DA 头过度预测前景（cls1 0.37 vs 正常 0.15-0.25）**，
动态训练管线系统性降低质量（ExpA s1/s2、ExpI D 三处复现）。

---

## 五、最终结论（§26 决策树 → **Route B：通用弹性模型**）

> **Compact Representation（H2 ✓，信息效率高）+ Cross-Architecture KD（H4 ✓，accuracy 引擎）+ Runtime Elasticity（多档位部署灵活性，不承诺等预算 accuracy 增益）**

论文候选结论（全部诚实测量）：
> "紧凑表示 + 跨架构蒸馏让 0.235M 模型在轻量 baseline 水平运行并支持多计算档位弹性部署；
> 学习式逐任务路由在当前极小模型区间内不提供额外 accuracy 收益（oracle 上界 +0.0125 DA），
> 其价值限于部署灵活性。"

| 假设 | 结论 |
|---|---|
| H1（task-wise 动态分配 > 统一宽度） | ✗ 不成立（oracle 上界小 + 动态训练不稳定） |
| H2（紧凑表示保留三任务信息） | ✓ 支持 |
| H3（单模型覆盖 Pareto） | ✓ 支持（Dynamic≈Static 插值） |
| H4（跨架构 KD 提升 student） | ✓ 支持 |

---

## 六、关键工程贡献

1. **离线 KD（Plan A）**：teacher 目标一次性预计算（fp16 缓存）+ student 从缓存训练
   → teacher 成本每步 527-966ms → 0，F/G 从 ~15h → ~2h
2. **真实切片感知 FLOPs 计数**：修正 thop 对动态宽度的高估（Dynamic 平均 0.90G 而非 1.06G）
3. 统一评测/训练协议、3-seed 统计、失败模式记录（router 塌缩分析）

---

## 七、交付物

- 各实验报告：`experiments/expA_equal_budget/EXP_A_REPORT.md`、`expD_compact_z/EXP_D_REPORT.md`、
  `expF_single_teacher/EXP_FG_REPORT.md`、`expI_kd_dynamic/`（metrics）、`PHASE1B_REPORT.md`
- 图表（§22 Figure 1-6）：`experiments/figures/`（Pareto×2、容量曲线、KD 效果、KD 变体、预算分布）
- 服务器复现手册：`docs/RUNBOOK_SERVER.md`
- 全部代码/配置/脚本在 `trac/`（53 commits）

---

## 八、下一步建议

1. **Multi-Teacher KD（Exp H）**：验证三 teacher 互补性（自然延伸）
2. **服务器全量复现**：按 RUNBOOK 跑 70k 最终数字（本地 10k 筛选结论方向已明确）
3. **论文写作**：以「紧凑表示 + KD + 弹性部署」为主线，dynamic router 作 negative result 分析
