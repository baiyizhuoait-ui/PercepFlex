# PHASE 2-B — Literature Novelty Audit（综合报告）

> 范围：2023–2026，主题=极小共享瓶颈表示 Compact Z 用于 BDD100K 三任务
> (Detection + Drivable Area + Lane) 驾驶感知的效率/容量研究。
> 检索方式：6 主题 fan-out 子代理 web 检索（T1 通用MTL瓶颈 / T2 IB+MTL / T3 驾驶MTL共享表示 /
> T4 驾驶pruning+KD与弹性容量 / T5 跨架构KD / T6 表示效率与parameter-matched）。
> ⚠️ 标注 [conf?] 的条目为检索所得、venue/年份/ID 需进一步核实（尤其 2026 新 arXiv）。

---

## 一、Collision / Novelty 表（核心 ~18 篇）

图例：sharedRep=共享表示 explicitBottleneck=显式低维瓶颈 kd=蒸馏 dynamic=动态/弹性宽度 budget=研究算力/参数预算

| # | Paper | Year/Venue | Dataset | Tasks | sharedRep | Bottleneck | kd | dynamic | budget | 重叠度判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **TriLiteNet** | 2025 IEEE Access/arXiv:2509.04092 | BDD100K | det+DA+lane | ✓ | ✗ | ✗ | ✗ | ✓ | **B/C**：同任务同数据集、含 0.14M tiny 档；但无显式信息瓶颈机制、无容量-算力关系研究 |
| 2 | **Sparse U-PDP** | 2023 TIV | BDD100K | det+DA+lane | ✓ | ✗(动态kernel表示) | ✗ | ✗ | ✗ | **B**：统一三任务但宽/深共享表示，不为极小瓶颈服务 |
| 3 | **SCAM-P** | 2025 IEEE[conf?] | BDD100K | det+DA+lane | ✓ | ✗ | ✗ | ✗ | ✗ | **B**：加注意力增强共享特征，非压缩瓶颈 |
| 4 | **YOLOPX / CA-YOLOPv8** | 2024 | BDD100K | det+DA+lane | ✓ | ✗ | ✗ | ✗ | ✗ | **B**：统一三任务基线，无瓶颈/预算研究 |
| 5 | **arXiv:2511.05557 pruning+KD 驾驶MTL** | 2025 | BDD100K | det+DA+lane | ✓ | ✗ | ✓ | ✗ | ✓ | **B**：同任务+压缩+KD；但=训练后剪枝(34→23M)，非"从头极小瓶颈"假设 |
| 6 | **TwinLiteNet / +** | 2023/2024 | BDD100K | DA+lane(双任务) | ✓ | ✗ | ✓(+) | ✗ | ✓ | **B**：仅2任务，双分支非瓶颈，无科学化容量问题 |
| 7 | **MultiNet** | 2018 IV | driving | road+vehicle+class | ✓ | ✗ | ✗ | ✗ | ✓ | **B**：共享encoder+轻头"祖先"，未显式尺寸化瓶颈、无等预算Pareto |
| 8 | **D2BNet** | 2023[conf?] | 驾驶MTL | det/DA/lane类 | ✓ | ✗(task路由) | ✗ | ✓ | ✓ | **B**：task/channel动态路由，方向相反(加动态性)，非压缩瓶颈 |
| 9 | **Bi-MTDP** | CVPR 2024 | NYUv2/PASCAL/City | dense预测 | ✓ | ✓(VIB层) | ✓ | ✗ | ✓ | **B/C**：二值化+VIB高斯瓶颈+KD；但瓶颈是"修复正则"，非驱动三任务、非极小连续Z的容量扫描 |
| 10 | **MT-VIB** | 2020 | 分类MTL | 多分类 | ✓ | ✓(VIB) | ✗ | ✗ | ✗ | **C(机制先例)/WEAK(域)**：共享变分瓶颈+多头，但分类尺度、鲁棒性目的 |
| 11 | **SGW/KEM-IB** | ACCV2024 arXiv:2410.03778 | 通用dense | 多dense | ✓ | ✓ | ✗ | ✗ | ✗ | **B**：IB知识提取模块减任务间干扰；非单极小瓶颈喂头、非驾驶低算力 |
| 12 | **共享变分latent 情感识别**[conf? arXiv:2607] | 2026 | 人脸情感 | VA/expr/AU | ✓ | ✓ | ✗ | ✗ | ✗ | **B(结构近)**：单共享变分瓶颈Z喂多头，但域=人脸情感、无算力框架 |
| 13 | **InfoCom (IB协同感知)** | 2025 arXiv:2512.10305 | OPV2V类 | 协同多智能体 | ✓ | ✓ | ✗ | ✗ | ✓ | **B(域远)**：IB压缩的是"车车间通信消息"，非单机3任务共享瓶颈 |
| 14 | **Towards Task-Compatible Compressible Representations** | 2024 ICMEW | COCO/City | det+depth | ✓ | ✓(rate) | ✗ | ✗ | ✗ | **B**：率失真编码共享瓶颈；预算=编码码率BPP非计算/参数 |
| 15 | **Rethinking Resource Competition in MTL (shared rep vs params)** | 2024 IEEE Access | MTL感知 | 多任务 | ✓ | ✗ | ✗ | ✗ | ✗ | **B(框架锚点)**：实证"共享表示才是真资源"；无极小Z、无等预算shared-vs-widening |
| 16 | **Dynamic Feature Interaction (D2BNet类)** | 2023 IJCV | BDD100K/NYU | 驾驶MTL | ✓ | ✗ | ✗ | ✓ | ✗ | **B**：动态交互增强，非压缩瓶颈 |
| 17 | **HeteroAKD / 跨架构KD (分割/分类)** | 2024-25 | 分割/分类 | seg/cls | 对齐空间 | ✗ | ✓ | ✗ | ✗ | **B**：KD期特征对齐空间，非部署瓶颈、非3任务驾驶 |
| 18 | **压缩/率失真MTL (RDC/RPC, Winner-Take-All)[conf?]** | 2024-26 | 压缩/玩具 | 多目标 | ✓ | ✓ | ✗ | ✗ | ✗ | **C(理论近)/域远**：IB/率失真理论多任务容量；源码压缩域，非计算参数效率、非驾驶 |

---

## 二、A/B/C/D 分类结论

- **A（明显撞车，需避开）**：无单篇完全覆盖"BDD100K三任务 + 极小共享瓶颈 + 容量扫描 + 等预算效率"。
  最接近的意图撞车是 #1 TriLiteNet(同任务) 与 #5 arXiv:2511.05557(同任务+压缩+KD)，但机制(剪枝/手工架构)均**非"极小瓶颈是效率来源"的容量研究**。
- **B（部分重叠但可区分）**：#2,#3,#4,#6,#7,#8,#9,#11,#13,#14,#15,#16,#17 —— 共享表示常见、IB-in-MTL(MT-VIB/Bi-MTDP)、跨架构KD、弹性宽度(OFA/DS-Net)、驾驶压缩均已有；**我们的区分=系统研究"单极小共享瓶颈Z的维度→每任务精度/饱和点/参数与算力效率"这一可测量关系**，在严格同预算 + 极低算力下。
- **C（高度接近，需换方向或大幅差异化）**：若把"共享瓶颈喂多头"本身当创新 → 撞 #12(域不同) / #10(MT-VIB先例) / #14(率失真) → **不能当核心**；若把"IB应用于MTL"当创新 → 撞 #9/#10/#11。**C 类=我们最容易踩的雷。**
- **D（明显空缺，可作核心）**：
  1. **无工作**在 BDD100K 三任务、极低算力(<2G FLOPs)下，对**同一个紧凑共享瓶颈做 Z∈{16..128} 系统容量扫描**并报告 每任务精度/饱和点/效用-算力曲线；
  2. **无工作**在等参数/等FLOPs下做 **shared-bottleneck vs 普通加宽** 的效率对比实证；
  3. **无工作**把"三任务对共享表示容量需求不同(degron pattern)"做成**可测量的科学观察**并据此给出信息分配设计(非动态路由)。

---

## 三、初始 Compact Representation Hypothesis（一句话，待实验证实/证伪）
> "We hypothesize that a single extremely small shared bottleneck representation (Compact Z)
> can serve detection / drivable-area / lane with a measurable, task-specific sensitivity to its
> capacity, so that at a fixed ultra-low compute budget a compact shared bottleneck is more
> parameter- and FLOPs-efficient than simply widening the backbone — i.e., compactness is an
> information-allocation mechanism, not just a smaller feature map."

---

## 四、interim 推荐（审计阶段，实验未跑完前为 provisional）
**倾向 CONTINUE（带 pivot 保险）**：审计显示"极小共享瓶颈的容量/效率可测量关系"尚无直接工作(D类空缺)，
但必须**避免把 C 类(共享表示/IB-in-MTL本身)当卖点**。卖点应是"驾驶三任务对共享表示容量需求的
任务差异 + 等预算下 shared 比 widened 更高效"的实证与机制解释。
若后续 R1/R2 + 容量扫描实验表明该关系不成立/无效率优势 → 立即 PIVOT(届时再评估 Route B 弹性或换向)。
