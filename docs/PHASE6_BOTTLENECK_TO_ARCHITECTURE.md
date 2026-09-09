# Phase 6 · Bottleneck-to-Architecture Discovery

> 开启：2026-09-09 晚 ｜ 纪律：NEW KNOWLEDGE > NEW ARCHITECTURE > PERFORMANCE
> 本文档记录 Evidence → Bottleneck → Hypothesis → Intervention → Result → Architecture implication 链条
> 配套：`experiments/phase6/phase6_architecture_hypotheses.csv`（H-A~H-N）、`phase6_novelty_matrix.csv`

---

## 0. 文献碰撞检查结论（2026-09-09，搜索执行）

| 方向 | 碰撞源 | 结论 |
|---|---|---|
| k-means anchors | YOLOv3+ 标准、YOLOPv2 已用 scene-adaptive k-means anchors；ATSS/SimOTA/OTA 是动态分配主线 | **danc 本身不是创新**。可主张的只有 capacity–supervision coupling（H-M） |
| 任务×特征层级 | YOLOPv2：DA 接 FPN 前浅层、lane 接 FPN 末端+反卷积（启发式）；HybridNets/GDMNet 类似 | "任务用不同层特征"= 已知。新颖点只能在：**bottleneck 测量驱动 + 严格预算配平 + 分辨率（不只层级）维度** |
| channel vs resolution | EfficientNet：单任务分类中 depth/width/resolution 需均衡缩放（compound scaling） | "维度需均衡"已知；**未解问题 = 极端压缩多任务下，不同任务在不同维度上先触顶（binding order）**——这是 EXP-1 要检验的潜在经验定律 |
| 动态分辨率 | DRNet (NeurIPS21, 逐图输入分辨率, 分类)；空间自适应采样 (ECCV20)；FoveatedSeg (CVPR25) | 分类域已知。任务条件化（仅 lane 触发 1/4）+ 驾驶 MTL 未检——但必须先做 offline oracle，否则不启动 |
| 梯度冲突 | GradNorm/PCGrad/CAGrad | 本项目已证 REJECTED，Phase 6 禁止再碰 |

**核心防碰撞设计**：所有架构主张以「预算配平对照」+「预注册判据」+「测量驱动的推导链」与文献区分；"模块没人叫过这个名字"不作为新颖性依据。

## 1. Evidence → Bottleneck（已确认的输入）

| 任务 | 确认的瓶颈 | 关键证据 | 状态 |
|---|---|---|---|
| Detection | 监督/assignment + encoder 表征质量 | danc 20ep 跨-seed +9.7~9.9x noise（零成本）；dp2a/dp2b REJECTED；3A encoder 单调 | H19 CONFIRMED |
| Lane | spatial addressability（1/8 硬天花板） | model-free ceiling 0.1853 被精确贴顶；l14f1 三-seed +1.85x（方向稳定幅度小）；lch64 +0.20x | H5b WEAK SUPPORT |
| DA | 远场语义/标注分布（非容量） | z16/32/128 非单调；分辨率非主导；远场 GT canvas 异常 | H11-alt SUPPORTED |
| Multi-task | 梯度冲突不存在 | cos≈0；rebalance/project/split-Z 全部无效 | H13 REJECTED |

## 2. Phase 6 核心假设（唯一 Level-1 候选）

> **H-N：在极端轻量多任务模型中，capacity 不是标量，而是 {channel, spatial, context, supervision} 四个资源维度；不同任务在不同维度上先触顶（binding order）：det→supervision，lane→spatial，DA→(远场)语义。**

该假设若成立且跨架构复现（EXP-4），即为项目的核心 scientific contribution；若不成立，Phase 2-5 证据链转入 mechanism-oriented review（fallback 已获用户批准）。

## 3. 第一轮实验登记（预算纪律：no-training 优先，4ep 只做 falsification）

| EXP | 问题 | 类型 | 预算 | 判据 |
|---|---|---|---|---|
| EXP-1 | spatial capacity 与 channel capacity 可否互换？FLOPs 严格配平下 fewer-ch@1/4 vs more-ch@1/8 谁赢？ | 4ep × 2 cells | ~1.2h GPU | lane_fg 差 ≥1x noise；赢者 20ep 确认 |
| EXP-2 | 最小 1/4 支路能否破 1/8 ceiling？ | **已完成**（= l14f1） | — | WEAK SUPPORT（1.85x，未过 2x） |
| EXP-3 | 极端压缩下 assignment quality 影响是否被放大（capacity–supervision coupling）？ | 零训练分析 | CPU | anchor-IoU 分布 × size-bucket recall × 容量档交叉 |
| EXP-4 | danc 干预跨架构泛化？+ 统一架构第一块砖 | 4ep × 1 cell（danc+l14f1 组合） | ~35min | det 增益保持且 lane 增益保持（两者独立 2x noise 界内不互伤） |
| EXP-5 | DA 真瓶颈是否远场语义/数据？ | 零训练扩展 | CPU | far/mid/near × 边界/语义混淆分解 |
| EXP-6 | 多分辨率架构成本模型 | 零训练模拟 | CPU | params/FLOPs/激活内存 表，供 EXP-1 配平与 H-K 组装 |
| EXP-oracle | 哪些图真正需要 1/4 lane 计算？ | 零训练 | CPU | 若大多数 val 图需要 1/4 → H-F 关闭 |

**GPU 排程（本夜）**：GPU-1 = 4B 收口重跑（lane8 20ep + da14 4ep，锚框 bug 已修复 6399548，22:15 起，ETA ~02:45）→ GPU-2 = EXP-1 双 cell + EXP-4 组合 probe（~02:45-04:30）→ 择优 20ep 确认（~04:30-07:00）→ 缓冲 + 晨间报告 09:30。

## 4. 决议规则（预注册）

- 4ep PASS 只授权 20ep 确认（本项目两次 4ep→20ep 反转在案）
- 判据只能用 SUPPORTED / WEAK SUPPORT / CONFIRMED / REJECTED / UNRESOLVED
- 每个架构机制必须有预算配平对照（resolution vs channel 的对照是 EXP-1 本身）
- 最多保留 3-5 候选；第二轮只深入 1-2 个
- Level 1-4 全过才允许构建正式新模型；否则转机制综述

## 5. Result 追加区（随实验更新）

- [22:15] GPU-1 lane8 重跑启动（锚框修复后）
- （待追加）
