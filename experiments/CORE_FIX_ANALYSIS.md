# Phase 1 核心算法修复记录（Exp2/Exp3 分析 → 修订训练）

## 问题（2026-08-29 实测）

Exp3（1000 张 val）：**learned router (0.248) < random (0.256) < fixed_medium (0.259)**
Exp2：shared (0.2463) ≈ taskwise (0.2424)，计算量相当（0.467 vs 0.460 avg width）

诊断：
1. `corr(router 选择宽度, 该帧 large−tiny 增益) ≈ 0.03` —— Router 分配与难度无关
2. **Oracle 上界**：完美难度 Router（预算匹配）只比 uniform_medium 好 **−0.019**（更差！）
   —— 因为该模型的**宽度-质量曲线非单调**（DA/Lane: medium > large），large 子网欠训练
3. 根本原因：Stage B/C/D 的宽度分布由（未学好的）Router 决定 → 少被选中的宽度（large）训练不足
   → "大宽度≠更好" → Router 无信号可学（use-it-or-lose-it 死循环）

## 修复（核心算法修改）

1. **Stage U：均匀宽度训练头**（每 batch 随机采样宽度，所有宽度同等训练）
   → 验证：宽度-质量曲线变单调（DA fgIoU: 0.608 → 0.727 → 0.742 at 0.25/0.5/1.0）✓
2. **Soft routing 训练**（头输出按 soft 概率混合，Router 获得所有宽度的平滑梯度）
3. **Stage C：冻结 encoder+heads，只训 Router**（在稳定的头上学难度映射）
4. 管线：A（静态满宽）→ U（均匀宽度）→ C（Router）→ D（joint budget-aware）

## 待验证（修订训练完成后）

- [ ] corr(chosen_width, gain) 是否显著 > 0.05
- [ ] Exp3: learned 是否 > random / fixed
- [ ] Exp1: dynamic 是否 ≥ static 插值
- [ ] Oracle 上界在新模型（单调曲线）上是否 > uniform

## 修订训练结果（2026-08-29 深夜）

### 尝试与结果
1. **Stage U（均匀宽度）**：修复了宽度-质量曲线（DA 单调：0.61→0.71→0.74）✓
2. **Soft routing**：Router 获得多宽度梯度，但 corr(choice, gain) 仍 ≈ 0
3. **Difficulty Router（多尺度特征 + 显式难度监督）**：探针 AUC 0.69（信号存在），
   但训练后 corr 仍 ≈ 0，且 joint 阶段检测头 large 宽度崩溃（mAP 0.024）——训练不稳定

### 核心科学结论（诚实记录）

**逐帧任务级路由在当前模型/数据上的收益上界很小**：
- 预算匹配 oracle：DA +0.0125，Lane +0.0065（完美难度估计也只能提升这么多）
- 难度可预测性：多尺度特征探针 AUC 0.69（中等）
- 当前 Router 实际收益 ≈ 0（与 random 相当）

**Phase 1 成立的部分**：
- H3（Pareto 覆盖）：dynamic ≈ static 插值，平均 FLOPs -32%（Exp1 全量验证）
- 单一模型多运行档位（Tiny/Medium/Large/Dynamic 一键切换）
- H2 待 Exp4（等预算对比轻量 baseline）验证

**H1（task-wise > 统一宽度）在当前设置下弱支持**——上界小是主因，
非 Router 训练不足（已试 hard-STE / soft / difficulty-supervision 三种）。

### 建议
- Phase 1 收尾：以稳定模型（exp_train_D）为准，跑 Exp4（H2 等预算对比），
  完整报告如实呈现上界分析
- Phase 2 方向：Cross-Architecture KD 增强 Compact Z（H4），使宽度敏感性更高、
  路由收益上界更大——这可能让 H1 在更强的表示下成立
