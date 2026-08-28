# 算法开发任务书（存档）

> 项目：Task-Resource Adaptive Compact Representation for Lightweight Multi-Task Driving Perception
> 来源：用户提供（2026-08-28），本文件为完整存档，供各阶段开发 agent 使用。

## 1. 项目目标
不是简单改进 YOLOP / 替换 Backbone / 增加 Attention / 普通 pruning。目标是研究面向资源受限设备的轻量多任务驾驶感知框架：
在 Detection + Drivable Area + Lane 三任务下，学习紧凑共享表示，并根据任务需求与计算预算动态分配模型容量，使**同一个模型**能在不同计算预算下运行，在保持精度的同时降低实际计算量。

最终目标：
1. 参数量明显低于传统 YOLOP/HybridNets。
2. 相同/接近参数、FLOPs 预算下性能不低于 TwinLiteNet+/TriLiteNet，争取更好的 Accuracy–Latency Pareto Frontier。
3. 同一模型支持多个运行档位，而非训练多个独立模型。
4. 后续扩展到不同硬件和视频时序推理。
5. 形成可写论文的完整研究框架，而非工程拼装。

## 2. 研究核心问题
极小模型同时执行 Det/DA/Lane 三任务时，不同任务所需信息不完全相同。与其静态给所有任务分配固定容量，能否学习 Compact Driving Representation，并由 Task-Resource Router 运行时决定给哪些任务、哪些特征、多少计算容量？
核心概念必须是 **Task × Resource × Representation**；Dynamic Width/Depth/Conv、Attention、Pruning、KD 只能作为实现工具，不能作为唯一创新点。

## 3. 必须避免的简单撞车方案
ESPNet+PCAA+TwinLiteNet；TwinLiteNet 换 Backbone；YOLOP+CBAM/SE/ECA；普通 Dynamic Conv；普通 Pruning；普通 KD；单纯 OFA 弹性宽度；单纯 DS-Net 动态宽度；单纯 task routing；单纯 task-specific feature sharing。
（已有工作：YOLOP、TwinLiteNet(+)、TriLiteNet、UF-Net、D2BNet、MDANet、OFA、DS-Net、SN-Net、跨架构蒸馏等。）

## 4. Baseline 必须首先建立（先不改网络）
- B1 YOLOP；B2 TwinLiteNet 或 TwinLiteNet+；B3 TriLiteNet；B4 普通轻量 CNN backbone + 三任务 heads。
统一测试：Parameters / FLOPs / Model Size / Detection mAP / DA mIoU / Lane IoU / FPS / Latency / GPU memory / CPU memory / FP32-FP16。不接受只看 accuracy。

## 5. Phase 1 两个核心模块
- **Module A Compact Driving Representation**：Input → Lightweight Encoder → Multi-scale Features → Compact Z。Encoder 可从 ESPNet 风格 block / DW Conv / PW Conv / 轻量 FPN-PAN 起步，不堆模块。目标：足够小、稳定、三任务可用的 Compact Feature Space。
- **Module B Task-Resource Adaptive Router**（最重要创新）：不是判断"大/小模型"，而是根据当前输入特征与运行预算，为每个任务决定 feature allocation / width-channel budget / computation level。每个任务获得 α_det / α_da / α_lane。

## 6. Router 第一版结构约束
优先 **Nested/Prefix Channel 结构**（1.0x/0.75x/0.5x/0.25x 连续前缀通道），不用随机 channel mask（稀疏 mask 减 FLOPs 但不减真实 latency）。保证 TensorRT/PyTorch 可直接执行。

## 7. 第一版动态机制
至少 3 个档位 Tiny/Medium/Large（如 0.25/0.5、0.5/0.75、1.0，比例由实验确定）。Router(Z, budget) → task-specific allocation → dynamic sub-network。budget 输入如 LOW/MEDIUM/HIGH。

## 8. 关键设计要求：task-wise heterogeneous allocation
Router 不能只学"这张图难不难"，要学"这张图对哪个任务难"。必须允许 Det=Tiny & DA=Tiny & Lane=Large 这类异构分配（本阶段最重要实验之一）。

## 9. Loss
L_task = L_det + λ_da·L_da + λ_lane·L_lane；L_total = L_task + λ_budget·L_budget（约束 expected FLOPs / active channels / latency proxy）。先做简单可运行版本，最终研究 accuracy–computation Pareto。

## 10. Router 训练策略
Stage A 训练固定最大容量模型 → Stage B 冻结/部分冻结 Encoder，训 Router+Heads → Stage C Joint Fine-tune → Stage D budget-aware training。不要一开始全量同时训练。

## 11. 第一阶段实验
1. Static vs Dynamic（Static Tiny/Medium/Large vs Dynamic，接近平均 FLOPs 下 Dynamic 不能显著损失性能）
2. Shared Dynamic vs Task-wise Dynamic（全任务一起切 vs 任务独立，证明 task-wise 有效）
3. Random Router vs Learned Router（random/fixed/global-difficulty/task-aware）
4. Equal Budget Comparison（0.5M/1.0M/2.0M 参数等效预算下比 TwinLiteNet+/TriLiteNet/Our Static/Our Dynamic——同样的计算预算谁性能更高）
5. Pareto Frontier（x=latency/FLOPs, y=overall task performance；Tiny/Medium/Large/Dynamic）

## 12. 综合性能
同时报告 mAP_det / mIoU_DA / IoU_lane 三个独立指标；Normalized Performance 只作辅助，不能用 composite score 掩盖单任务退化。

## 13. 失败模式监控（训练与测试均记录统计）
A. Router 永远选 Large（没学会省算力）；B. 永远选 Tiny（budget 过强或 supervision 不足）；C. 三任务永远相同 budget（没学会 task-specific）；D. FLOPs 降但真实 latency 不降（动态结构硬件不友好）；E. 平均精度好但困难场景崩溃（difficulty estimation 不可靠）。

## 14. Router 行为可视化
Per-frame allocation（Det/DA/Lane 每帧档位）；Scene-level statistics（晴天/阴天/夜间/拥堵/高速/城市/复杂 lane，条件允许按标签划分，否则用 uncertainty/confidence 构造 difficulty bins）。证明 Router 行为有语义意义而非随机。

## 15-16. Phase 2（Phase 1 稳定后才做）
Cross-Architecture Distillation：teacher 可选 YOLOP/TwinLiteNet+/TriLiteNet 任一或组合。比较 KD-A 单 teacher / KD-B 多 teacher output KD / KD-C feature KD / KD-D 跨架构 Compact Representation Distillation。核心约束：不简单平均 teacher feature（channel/spatial/semantic/statistics 均不同），需 lightweight projection/alignment，区分 architecture-specific / task-relevant / shared 信息，避免把 teacher 冗余结构蒸给学生。

## 17. Phase 3 Runtime Adaptation
Input → Scene Difficulty → Task Demand → Available Compute Budget → Router → Dynamic Model。三个运行场景 Mode1 High Accuracy / Mode2 Balanced / Mode3 Low Latency，**同一 trained model** 不重训。

## 18. Phase 4 Video Adaptive Inference
利用 Frame(t-1)+Frame(t) 预测 feature change / task uncertainty / scene change；变化小则 Reuse 上一帧特征，变化明显则 Recompute（如 t1=Full, t2=Reuse, t3=Reuse, t4=Medium, t5=Full）。Phase 1 不实现。

## 19. Hardware-aware
区分 theoretical vs actual efficiency。至少记录 GPU latency/FPS、CPU latency（可行时）、memory、power/energy（设备允许时）。目标 Desktop GPU / Jetson-Edge / CPU；无 Jetson 时先完成 PC 真实 latency benchmark，**不允许虚构 edge-device 结果**。

## 20. 第一阶段网络结构
Input → Lightweight Encoder → Multi-scale Features → Compact Z → Task-Resource Router → {Det Head, DA Head, Lane Head}。
Router：Z → Global Pool → Small MLP → task-specific allocation logits → Discrete budget selection。
禁止一开始加入：Transformer / MoE / PCAA / CBAM / 多种 Attention / 动态卷积 / 大型 FPN / 复杂 NAS。

## 21. 论文级对比对象
YOLOP、TwinLiteNet+、TriLiteNet、MDANet（可获得时）、Our Static、Our Dynamic、Our Dynamic+KD、Our Dynamic+KD+Temporal（最终）。无法严格复现的论文在 README 标注 `Not reproduced`，不得估算对方指标冒充实验。

## 22. 实验自动记录
统一 experiment log：experiment_id / model / commit_hash / dataset_version / input_size / parameters / FLOPs / average_latency / P50 / P95 / FPS / GPU_memory / mAP / DA_mIoU / Lane_IoU / budget_distribution / seed。
目录 experiments/exp_001/...，每实验自动保存 config.yaml / metrics.json / training_log.txt / checkpoint.pt / allocation_statistics.csv。

## 23. 代码结构
models/{encoder,representation,router,heads,adaptive_model.py}、losses/、distillation/、datasets/、training/、evaluation/、profiling/、visualization/、configs/、experiments/、scripts/。不许全写进一个 model.py；baseline 与 ours 可通过 config 切换。

## 24. Phase 1 禁止
同时加 pruning / INT8 / temporal reuse / dynamic resolution / MoE / 多 backbone 切换 / 大量 attention / 为降参数破坏 baseline 可比性 / 先追求论文结构 / 无法解释的"创新"模块。

## 25. Phase 1 成功标准
Architecture：单模型 ≥3 个 compute budgets；支持 task-wise allocation；不需训练三个独立模型。
Efficiency：Dynamic 平均 latency/FLOPs 明显低于 Large 固定模型；Router 无明显额外开销。
Accuracy：相近平均计算预算下 Our Dynamic ≥ Our Static，最好 > lightweight baseline；至少一个关键任务不崩溃。
Behavior：不同场景→不同分配；不同任务→不同容量分配。
Reproducibility：全部结果单命令复现。

## 26. 论文假设
H1 任务间共享特征需求不同 ⇒ task-wise dynamic allocation 比统一宽度有效。
H2 Compact Representation 在小参数预算下保留三任务信息。
H3 动态信息分配获得更好 Accuracy–Compute Pareto Frontier，无需多模型。
H4（Phase 2）Heterogeneous teachers 提供互补 architecture-aware knowledge。
H5（Phase 3/4）Runtime adaptation + temporal reuse 降低视频流平均计算成本。

## 27. 工作顺序
1 读现有项目与数据集结构 → 2 确认可复用代码 → 3 统一 baseline evaluation → 4 实现 Compact Representation → 5 Static multi-task 模型 → 6 Task-Resource Router → 7 多级 dynamic width → 8 task-wise allocation → 9 budget-aware training → 10 Static vs Dynamic 实验 → 11 Shared vs Task-wise 实验 → 12 allocation 可视化与 Pareto 曲线 → 13 失败模式分析 → 14 Phase 1 有效后开始 Cross-Architecture Distillation。

## 28. 汇报要求
每阶段汇报：A 修改文件 B 网络结构 C Parameters/FLOPs D 训练结果 E 各任务指标 F Dynamic allocation statistics G 实际 latency H 与 baseline 差距 I 失败问题 J 下阶段建议。必须同时报 Best/Average/Worst，不只报最好实验。

## 29. 最终研究路线
Heterogeneous Teachers → Cross-Architecture Distillation → Compact Driving Z → Task-Resource Router → {Det/DA/Lane} heads → Dynamic Width/Depth → Runtime Adaptation → Temporal Video Reuse → Edge Deployment。
论文最终回答："在资源受限的多任务驾驶感知中，能否通过学习紧凑表示并动态分配有限的信息与计算预算，在不牺牲关键任务性能的情况下，让一个模型适应不同场景、任务需求和计算约束？"

## 30. 当前任务（Phase 0 + Phase 1）
不提前实现 Cross-Architecture KD / Temporal Adaptation / Dynamic Resolution / INT8 / Hardware Runtime Controller。
第一目标：证明 Task-Resource Adaptive Compact Representation 核心思想成立。若 Phase 1 无法证明 Dynamic Task-wise Allocation 优于 Static，停止堆功能、分析原因、修改核心算法；成功后再进 Phase 2。
