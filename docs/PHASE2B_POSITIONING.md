# Phase 2-B — 方向定位与竞品对照（Direction Positioning）

> 目的：回答三个问题 —— (1) 我们做的方向是什么；(2) 与竞品逐方面区别；(3) 目前结果能否与对手比对、优劣在哪。
> 所有数字来自实测报告（STEP1 / STEP2 / Exp-D / 同预算对齐表），未做任何推断性填充。
> 生成日期：2026-09-03

---

## 一、方向定位（一句话）

我们做的**不是**「又一个更轻的三任务网络」，而是对一个**可测量科学关系**的研究：

> 在 BDD100K 三任务（Detection + Drivable Area + Lane）、**极低算力（<2 GFLOPs）** 下，
> **同一个极小共享瓶颈 Compact Z 的容量，如何决定每个任务各自的精度、饱和点与算力效率**；
> 以及在等参数/等 FLOPs 下，「共享瓶颈」是否比「单纯加宽骨干」更高效。

核心假设（可证伪）：

> Compactness is an **information-allocation mechanism**, not just a smaller feature map.

竞品回答的是「谁更准」，我们回答的是「紧凑为什么有效、什么时候失效、拐点在哪」。

---

## 二、与竞品逐方面区别

| # | 维度 | 竞品（TriLiteNet / TwinLiteNet+ / YOLOPX / YOLOP / Sparse U-PDP / D2BNet / SCAM-P） | 我们（PercepFlex） |
|---|---|---|---|
| 1 | 问题类型 | 架构设计问题：如何设计更准/更快的三任务网络 | **测量问题**：Z 容量 ↔ 每任务精度/饱和点/算力的函数关系 |
| 2 | 核心主张 | 我的结构更好（注意力、双分支、重参数化、动态路由） | 紧凑是信息分配机制（可被证伪的假设） |
| 3 | 优化目标 | 精度↑、参数↓、FLOPs↓ 三者取 Pareto | **拐点定位**：Z 多大开始收益递减、哪个任务先饱和 |
| 4 | 实验方法论 | 报告自己模型的最终分数表 | **受控扫描**：单变量 Z∈{16..128}、等预算 shared-vs-widening、配对对照 |
| 5 | 效率维度 | 主要报 Params / FLOPs，很少报实测延迟 | Params + FLOPs + **实测 p50/p95/FPS**（RTX5060, bs1） |
| 6 | 任务与数据 | 同三任务（BDD100K）；TwinLiteNet+ 仅 DA+lane 双任务 | 同三任务，使用 tri_ 子集（69 863 / 10 000） |
| 7 | 贡献形态 | 新模块/新结构，追求 SOTA | **经验性规律 + 设计准则**，EI/工程论文，不追求 SOTA |

### 2.1 Novelty 边界（来自 18 篇文献审计，务必遵守）

- **A 类（明显撞车）**：无。没有单篇同时覆盖「BDD100K 三任务 + 极小共享瓶颈 + 容量扫描 + 等预算效率」。
- **D 类（空缺，可作核心）**：
  1. 无工作在 BDD100K 三任务、<2 GFLOPs 下对**同一紧凑共享瓶颈做 Z∈{16..128} 系统容量扫描**并报告每任务饱和点；
  2. 无工作在等参数/等 FLOPs 下做 **shared-bottleneck vs 单纯加宽** 的效率对比实证；
  3. 无工作把「三任务对共享表示容量需求不同步」做成**可测量观察**并据此给出信息分配设计（非动态路由）。
- **C 类雷区（不可当卖点）**：
  - 「共享表示喂多头」本身 → 撞 #12（人脸情感域）/ #10 MT-VIB 先例；
  - 「IB 应用于 MTL」→ 撞 #9 Bi-MTDP / #10 MT-VIB / #11 SGW-KEM-IB；
  - 「弹性宽度」→ 撞 OFA / DS-Net / D2BNet；
  - 「驾驶 MTL + pruning + KD」→ 撞 arXiv:2511.05557（且其为训练后剪枝 34→23M，非从头极小瓶颈）。

**结论**：卖点只能是「驾驶三任务对共享表示容量需求的**任务差异** + 等预算下 shared 比 widened 更高效」的**实证与机制解释**，而非结构 novelty。

---

## 三、目前结果（全部为已实测可信数字）

### 3.1 等预算表（固定 Stage-A protocol：4ep / bs16 / AdamW lr1e-3 / cosine / seed0）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | DA fg | Lane fg | Lane mIoU | FPS |
|---|---|---|---|---|---|---|---|---|
| A0 0.235M | 0.230M | 1.47G | 0.2414 | 0.8360 | 0.7418 | 0.1843 | 0.5802 | 157 |
| 0.5M | 0.424M | 2.46G | 0.2846 | 0.8334 | 0.7384 | 0.1915 | 0.5837 | 186 |
| 1.0M | 0.959M | 4.22G | 0.2951 | 0.8518 | 0.7654 | 0.1943 | 0.5855 | 179 |
| 2.0M | 1.980M | 8.84G | 0.3187 | 0.8450 | 0.7555 | 0.1939 | 0.5853 | 158 |

> 注：1.0M / 2.0M 沿用 Phase-1 权重（3ep / 2ep），非 4ep，已在原始报告中如实标注。

### 3.2 实测延迟（bs1 / 640 / RTX5060 / eager）

| 模型 | p50 | p95 | FPS |
|---|---|---|---|
| A0 0.235M | 3.73 ms | 10.44 ms | 184 |
| 0.5M | 3.94 ms | 10.95 ms | 178 |

`torch.compile` 在本环境（Blackwell + torch 2.11）inductor 报错，已记录为部署限制，未强行报数。

### 3.3 已确认的科学观察（Exp-D）

| 任务 | 0.235M 相对 2.0M 保留率 | 饱和行为 |
|---|---|---|
| Detection mAP50 | **79.2%** | 持续爬升，2.0M 才到 100% |
| Drivable Area mIoU | 98.7% | 0.5M 即 ≥99.4% |
| Lane fgIoU | 89.7% | 0.5M 即 ≥99.6% |

→ **非同步退化（asynchronous degradation）**：三任务对共享表示容量的需求显著不同，Detection 是唯一持续受益于容量者。这是我们的核心差异化观察。

### 3.4 负结果（诚实记录）

- **单 teacher KD 无稳定增益**（同 9999 子集配对，固定 4ep）：YOLOP +0.0025 mAP、TwinLite+ −0.0074、TriLite −0.0127。DA 略降、Lane 中性。
- 与 Phase-1 声称的 +0.012 mAP 冲突。冲突解释：Phase-1 那次提升被「KD 用 10k / no-KD 用 69k」的数据量差异混淆。
- → KD 不作为本项目基石；Task 2(e) 已据此降级为辅助手段。

---

## 四、能否与对手比对？—— 目前**不能完全公平比对**

### 4.1 同预算对齐表（官方 pretrained，tri_val）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | Lane fg |
|---|---|---|---|---|---|
| Ours 0.235M | 0.230M | 1.47G | 0.2414 | 0.8360 | 0.1843 |
| TriLite-tiny | 0.151M | 1.83G | 0.4953 | 0.8796 | 0.1952 |
| Ours 0.5M | 0.424M | 2.46G | 0.2846 | 0.8334 | 0.1915 |
| TriLite-small | 0.592M | 6.61G | 0.6326 | 0.9053 | 0.2198 |
| TLP-medium | 0.479M | 15.4G | n/a（无 det） | 0.9191 | 0.2297 |
| Ours 2.0M | 1.980M | 8.84G | 0.3187 | 0.8450 | 0.1939 |
| TriLite-base | 2.350M | 25.4G | 0.7235 | 0.9203 | 0.2371 |
| TLP-large | 1.944M | 58.6G | n/a（无 det） | 0.9279 | 0.2450 |

### 4.2 **不对等声明（必须在论文中显式写出）**

- 官方权重 = 全量 BDD100K + 充分训练；
- 我们 = tri_train（69 863） + 固定 4ep。

因此该表**只能作为风险信号，不能作为最终判决**。

### 4.3 口径修正（重要）

早期文档写的「FLOPs 低 6–10×」是拿 **TwinLiteNet+** 比的，而 TLP 只做**双任务**（无检测头），比 FLOPs 天然占便宜。
**同任务（三任务）公平对照只有 TriLiteNet**，实测 FLOPs 倍数为：

| 对照 | FLOPs 倍数（对手 / 我们） |
|---|---|
| Ours 0.235M vs TriLite-tiny | 1.83 / 1.47 = **1.24×** |
| Ours 0.5M vs TriLite-small | 6.61 / 2.46 = **2.69×** |
| Ours 2.0M vs TriLite-base | 25.4 / 8.84 = **2.87×** |
| Ours 0.5M vs TLP-medium（双任务） | 15.4 / 2.46 = **6.26×** |
| Ours 2.0M vs TLP-large（双任务） | 58.6 / 8.84 = **6.63×** |

→ 对外表述应改为「同参数下 FLOPs 低 1.2–2.9×（三任务对照）/ 6.3–6.6×（双任务对照）」，**不得笼统称 6–10×**。

### 4.4 FLOPs 归一化效率（我们的软肋，必须自曝）

mAP50 per GFLOP：

| 模型 | mAP/GFLOP |
|---|---|
| **TriLite-tiny** | 0.4953 / 1.83 = **0.271** |
| Ours 0.235M | 0.2414 / 1.47 = **0.164** |
| Ours 0.5M | 0.2846 / 2.46 = 0.116 |
| TriLite-small | 0.6326 / 6.61 = 0.096 |
| Ours 1.0M | 0.2951 / 4.22 = 0.070 |
| Ours 2.0M | 0.3187 / 8.84 = 0.036 |
| TriLite-base | 0.7235 / 25.4 = 0.028 |

**即使按 FLOPs 归一化，TriLite-tiny 仍领先我们 1.65×。这个差距无法完全由「训练不充分」解释。**
→ 「我们 FLOPs 效率更高」这个说法在当前数据下**不成立**；唯一站得住的是「**同参数下 FLOPs 绝对值更低**」（紧凑 1/8 表示带来的结构性事实）。

---

## 五、优劣清单

### 5.1 优势（站得住的）

1. **同参数下 FLOPs 绝对值显著更低**（三任务对照 1.2–2.9×）—— 来自 1/8 分辨率的紧凑共享表示，是结构性事实，不依赖调参。
2. **实测延迟真实可用**：A0 p50 3.73 ms / 184 FPS（RTX5060, bs1, eager），竞品少有同条件实测。
3. **变量控制干净**：`z_channels` 是单一连续旋钮，可做纯粹的单变量容量扫描——这是竞品论文不具备的方法学条件。
4. **科学问题无人占据**（D 类空缺 3 项），且已有非同步退化的初步实证。
5. **负结果记录完整**（KD 三条 teacher 全部实测），可作为论文中的 honest negative result，增加可信度。

### 5.2 劣势 / 风险（必须正视）

1. **绝对精度明显落后**：同 FLOPs 下检测 mAP 约为 TriLite-tiny 的一半（0.241 vs 0.495）。
2. **⚠️ 最大发表风险：缺同 protocol 重训的竞品对照。** 现有对照全部是「官方充分训练权重 vs 我们 4ep」，审稿人会直接质疑。**必须在论文前补做**：用我们的 protocol（tri_train 69 863 / 4ep / bs16 / lr1e-3 / cosine）重训 TriLiteNet-tiny（与 small），得到真正同 protocol 的对照点。
3. **「FLOPs 效率更高」目前不成立**（见 §4.4），若强行主张会被数据推翻。
4. **Route A 已排除**：同预算下不太可能超过 TriLite/TwinLite+，因此不能走精度竞赛路线。
5. **KD 辅助路线已堵**：单 teacher 无增益，Multi-Teacher 仅在 (c) 修好后作为辅助，且不得作为卖点。

---

## 六、建议的论文 positioning 表述（避开 C 类雷区）

不可用：
- ~~「我们提出了共享瓶颈表示」~~（撞 MT-VIB / #12）
- ~~「我们将信息瓶颈引入多任务学习」~~（撞 Bi-MTDP / MT-VIB）
- ~~「我们的方法是首个用于驾驶三任务的紧凑模型」~~（撞 TriLiteNet）

可用（以测量与机制为核心）：

> We present a **controlled capacity study** of an extremely small shared bottleneck (Compact Z)
> for BDD100K three-task driving perception at ultra-low compute (<2 GFLOPs). Rather than
> proposing a new architecture, we measure **how** per-task accuracy, saturation points, and
> compute efficiency vary as a function of bottleneck capacity, and show that the three tasks
> exhibit **asynchronous degradation** under a shared representation — detection degrades
> first and most severely, while drivable-area and lane saturate early. We further provide an
> **equal-budget comparison of shared-bottleneck versus backbone widening**, and derive
> practical guidance for information allocation at a fixed compute budget.

---

## 七、决策建议（Task 3 输入）

- **判定：CONTINUE（带明确补救条件）**。D 类空缺确实存在，非同步退化已有初步实证，方向成立。
- **触发 PIVOT 的条件**：
  1. 同 protocol 重训 TriLiteNet 后，我们在等 FLOPs 下仍显著落后且「shared 比 widened 更高效」不成立 → 核心假设被证伪 → PIVOT；
  2. Z 扫描显示三个任务的饱和点几乎重合（无非同步退化）→ 差异化观察消失 → PIVOT。
- **Task 3 之前必须补的 1 项实验**（非原有计划内，但为发表所必需）：
  **TriLiteNet-tiny/small 按我们 protocol 重训**（tri_train 69 863, 4ep, bs16, lr1e-3, cosine, seed0），
  产出同 protocol 对照点。这是把「风险信号」变成「可辩护结论」的唯一途径。
