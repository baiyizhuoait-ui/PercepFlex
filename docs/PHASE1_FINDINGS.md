# Phase 1 核心发现报告（H1-H3 验证）

## 已验证成立

### H3 ✓（Accuracy–Compute Pareto 覆盖，全量 val 10k）
- Dynamic 模型（平均宽度 0.389，≈1.06G FLOPs）精度 0.2427 mAP / 0.838 DA
  位于 static_tiny (0.82G, 0.2336) 与 static_medium (1.16G, 0.2541) 之间——
  **单一模型覆盖了静态多模型的精度-算力曲线**（Exp1）
- 相对 static_large 节省 ~32% FLOPs，精度损失仅 ~0.019 mAP
- 支持 Tiny/Medium/Large/Dynamic 四档一键切换（含 Phase 3 的 HIGH/MEDIUM/LOW 模式接口）

### 机制 ✓（§6/§7/§8 设计目标）
- Nested-prefix 连续通道宽度（硬件友好），3 档离散预算
- 逐帧任务异构分配（task-heterogeneity 0.83-0.88）
- Router 行为可视化 + 分配统计 CSV（§14 数据源就绪）

## 未成立 / 弱支持（诚实记录）

### H1 ✗（task-wise dynamic > 统一宽度，等算力下）
- **Exp3**（1000 张，多次复现）：learned 0.2466 < random 0.2553 < fixed_medium 0.2588
- **Oracle 上界**：完美难度 Router（预算匹配）也只提升 DA +0.0125、Lane +0.0065
- **难度可预测性**：多尺度特征探针 AUC 0.69（中等，非 0.9）
- 尝试的修复均未突破：hard-STE → soft routing → DifficultyRouter（丰富特征+显式监督）
- 结论：**逐帧任务级路由在当前模型/数据上的收益上界本身很小**（+0.01~0.01 IoU），
  不是 Router 训练不足，而是杠杆有限

### 失败模式观察（§13）
- A（永远 Large）：无 budget loss 时出现 ✓ 已用 budget loss 控制
- B（永远 Tiny）：budget target 过强时出现（soft 训练塌缩）✓ 已调
- C（三任务同预算）：taskwise 模型 0.83 异构率 ✓ 未发生
- D（FLOPs 降延迟不降）：**发生**——eager PyTorch 的 Python 开销淹没小模型宽度收益
  （微基准 3.09→2.35ms 有效，eval 循环 7.3-7.8ms 噪声内）；需编译后端兑现
- E（难度估计不可靠）：**发生**——corr(choice, gain) ≈ 0

## 待验证
- H2（Compact Z 小预算下保留三任务信息）：Exp4 等预算对比（需训练 0.5/1.0/2.0M 模型）

## 对论文的意义
- 核心卖点调整为：**单一紧凑模型 + 多运行档位 + Pareto 覆盖（H3）**，
  并诚实报告逐帧路由的收益上界（+0.01 IoU）作为边界条件
- H1 的更强形式可能需要 Phase 2 KD 增强表示（H4）后重新验证——
  更强的 Compact Z 可能放大宽度敏感性，从而放大路由收益
