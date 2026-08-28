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
