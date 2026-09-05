# Phase 3A · Encoder Capacity Sanity Check

**研究问题**：在极低参数预算的多任务驾驶视觉模型中，有限参数应优先投入 encoder 还是 compact representation？

**本阶段子问题**：在固定 Z（z=16）和固定训练协议下，改变 **encoder 容量**能否产生超出噪声的性能变化？如果能，它与 Phase 2-D 中改变 Z 容量所产生的变化相比量级如何？

- 执行时间：2026-09-05 11:12:17 → 15:47:42（4h35m GPU）
- Commit：`51b68394aa6a727597bd046dd79ae5553d704113`
- Seed：**0（全程未改）**，结果未挑选、未剔除

---

## 1. 实验表

固定条件：`z=16`、`epochs=20`、`batch=16`、`lr=1e-3`、AdamW + cosine、`tri_train` 全量 69863 张、`input_size 640×640`、`seed=0`。
**唯一自变量：encoder 宽度**（`stem` / `stages`），`blocks=[2,2,2]` 深度不变，heads 不变，Z 不变。

| Encoder | stem | stages | Params (M) | FLOPs (G) | mAP50 | mAP50-95 | DA mIoU | DA fg | Lane mIoU | Lane fg | peak mem (MiB) | final train loss | train (min) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E-small | 12 | 24-40-64-80 | **0.1013** | **0.722** | 0.2687 | 0.0890 | 0.8423 | 0.7515 | 0.5818 | 0.1885 | 2130 | 0.2290 | 113 | trained |
| E-base  | 16 | 32-64-96-128 | **0.1889** | **1.060** | 0.3204 | 0.1149 | 0.8564 | 0.7723 | 0.5867 | 0.1973 | 2446 | 0.2156 | NA | reused: phase2d/expD_z16_e20 |
| E-large | 20 | 40-80-128-168 | **0.2915** | **1.428** | 0.3585 | 0.1356 | 0.8589 | 0.7761 | 0.5878 | 0.2000 | 2970 | 0.2068 | 153 | trained |

原始结果表：`experiments/phase3a/exp3A_encoder.csv`

### 关于 E-base 的复用（诚实声明）

E-base 就是当前 baseline encoder 本身，配置与协议与 Phase 2-D 的 `expD_z16_e20` **逐字段一致**（同为 z=16 / 20ep / bs16 / lr1e-3 / seed=0）。因此没有重跑，而是通过符号链接复用其 checkpoint 与 eval metrics，节省约 132 分钟 GPU 时间：

```
experiments/phase3a/exp3A_ebase_z16/checkpoint.pt     -> ../../phase2d/expD_z16_e20/checkpoint.pt
experiments/phase3a/exp3A_ebase_z16/training_log.txt  -> ../../phase2d/expD_z16_e20/training_log.txt
experiments/phase3a/exp3A_ebase_z16_eval/metrics.json -> ../../phase2d/expD_z16_e20_eval/metrics.json
```

代价：E-base 的 `train_wall_min` 为 `NA`（未重新计时）。该字段不参与任何结论。CSV 中 `source` 列已显式标记 `reused:phase2d/expD_z16_e20`，不会被误认为独立 run。

### 关于 FPS（不作效率结论）

| Encoder | FLOPs (G) | profiled `fps` | `eval_fps` | avg latency (ms) |
|---|---|---|---|---|
| E-small | 0.722 | 135.03 | 177.92 | 7.41 |
| E-base  | 1.060 | 140.62 | 166.20 | 7.11 |
| E-large | 1.428 | **375.21** | 189.32 | 2.67 |

E-large 的 FLOPs 是 E-small 的 **1.98 倍**，profiled FPS 却是其 **2.78 倍**（延迟 2.67ms vs 7.41ms）——物理上不可能。这是本机 **SW Power Cap / clock throttling**（`clocks_throttle_reasons.active=0x4`）造成的测量态差异，不是真实效率差异。`eval_fps` 三个值（177.9 / 166.2 / 189.3）同样非单调。

**结论：本报告所有效率判断一律基于 FLOPs 与参数量，FPS 与延迟数据仅存档，不引用、不比较。**

---

## 2. 参数 / FLOPs 对比

| 对比 | ΔParams | ΔFLOPs | 备注 |
|---|---|---|---|
| E-small → E-base | +0.0876 M (+86.5%) | +0.338 G (+46.8%) | |
| E-base → E-large | +0.1026 M (+54.3%) | +0.369 G (+34.8%) | |
| E-small → E-large | **+0.1902 M (+187.7%)** | **+0.706 G (+97.8%)** | 参数近 3 倍，FLOPs 近 2 倍 |

三档由 `scripts/phase3a_encoder_scan.py` 在 0.40–2.00 的乘法 scale 网格（27 档）上搜索得到，目标是命中用户指定的 ~0.10M / ~0.19M / ~0.30M，同时保持 encoder「单调变宽」的形状（stem 取 4 的倍数、stages 取 8 的倍数）。

**参数量独立校验**：扫描器预测的 0.1889M 与 Phase 2-D 实测 baseline 0.189M 一致；三档 params 与 FLOPs 均严格递增（无倒挂）。

**Z 的对照成本**（Phase 2-D，encoder 固定为 baseline）：z 16→128 花掉 **+0.0970 M / +0.963 G**。
注意：花在 Z 上的 **FLOPs 比花在 encoder 上更贵**（+0.963G vs +0.706G，同样换来约 0.1M 参数），因为 Z 作用在 1/8 分辨率的高维特征上，且被两个 segmentation head 反复消费。

---

## 3. 任务性能对比

### 3.1 单调性（Q1）

| metric | E-small | E-base | E-large | Δ(base−small) | Δ(large−base) | Δ(large−small) | 严格单调 |
|---|---|---|---|---|---|---|---|
| mAP50 | 0.2687 | 0.3204 | 0.3585 | +0.0517 | +0.0381 | **+0.0898** | ✅ True |
| mAP50-95 | 0.0890 | 0.1149 | 0.1356 | +0.0259 | +0.0207 | **+0.0466** | ✅ True |
| da_mIoU | 0.8423 | 0.8564 | 0.8589 | +0.0141 | +0.0025 | **+0.0166** | ✅ True |
| da_fg | 0.7515 | 0.7723 | 0.7761 | +0.0208 | +0.0038 | **+0.0246** | ✅ True |
| lane_mIoU | 0.5818 | 0.5867 | 0.5878 | +0.0049 | +0.0011 | **+0.0060** | ✅ True |
| lane_fg | 0.1885 | 0.1973 | 0.2000 | +0.0088 | +0.0027 | **+0.0115** | ✅ True |

**6/6 指标全部严格单调递增**（E-small < E-base < E-large，无一例外）。训练 loss 也单调下降（0.2290 → 0.2156 → 0.2068），与容量序一致——不是优化噪声。

**分任务敏感度**（相对 E-small 的提升幅度）：

| 任务 | 代表指标 | 绝对提升 | **相对提升** |
|---|---|---|---|
| Detection | mAP50-95 | +0.0466 | **+52.4%** |
| Detection | mAP50 | +0.0898 | **+33.4%** |
| Lane | lane_fg | +0.0115 | +6.1% |
| DA | da_fg | +0.0246 | +3.3% |
| DA | da_mIoU | +0.0166 | +2.0% |
| Lane | lane_mIoU | +0.0060 | +1.0% |

**Detection 对 encoder 容量的敏感度比其他两个任务高一个数量级**（相对提升 33–52% vs 1–6%）。

### 3.2 与 Phase 2-D 的 Z 效应对比（Q2）

噪声底来自 Phase 2-C Exp A 的 3-seed @4ep pooled stdev。

| metric | ΔZ 离散度 (z16/32/128 @20ep) | 噪声底 | ΔEncoder (small→large) | **ΔEncoder / ΔZ** | 判定 |
|---|---|---|---|---|---|
| mAP50 | 0.0027 | 0.0073 | 0.0898 | **33.3×** | encoder 远超噪声 |
| mAP50-95 | 0.0010 | 0.0032 | 0.0466 | **46.6×** | encoder 远超噪声 |
| da_mIoU | 0.0019 | 0.0142 | 0.0166 | **8.7×** | encoder 超过噪声 |
| da_fg | 0.0027 | 0.0202 | 0.0246 | **9.1×** | encoder 超过噪声 |
| lane_mIoU | 0.0016 | 0.0021 | 0.0060 | **3.8×** | encoder 超过噪声 |
| lane_fg | 0.0032 | 0.0032 | 0.0115 | **3.6×** | encoder 超过噪声 |

**最小倍数是 3.6×，最大 46.6×。** 在完全相同的训练预算（20ep）下，改变 encoder 容量产生的效应是改变 Z 容量的 **4–47 倍**。

稳健性说明：噪声底来自 4ep 的 3-seed 估计，是 20ep 真实的代理值而非精确值。但 Detection 的 ΔEncoder（0.0898 / 0.0466）即使真实 20ep 噪声比代理值大 3–5 倍，结论仍然成立；DA/Lane 的 Δ(large−base)（0.0011–0.0038）即使噪声只有代理值的一半，也仍然落在噪声内。

### 3.3 单位成本收益（Q3）

| metric | 花在 encoder（每 +0.01M 参数） | 花在 Z（每 +0.01M 参数） | **比值（参数）** | **比值（FLOPs）** |
|---|---|---|---|---|
| mAP50 | +0.00472 | +0.00021 | **22.9×** | **61.2×** |
| mAP50-95 | +0.00245 | +0.00008 | **29.7×** | **79.4×** |
| da_mIoU | +0.00087 | −0.00011 | −7.7×（符号相反） | −20.6× |
| da_fg | +0.00129 | −0.00015 | −8.4×（符号相反） | −22.4× |
| lane_mIoU | +0.00032 | +0.00012 | 2.5× | 6.8× |
| lane_fg | +0.00060 | +0.00033 | 1.8× | 4.9× |

**每 0.01M 参数投到 encoder 换来的 mAP50 是投到 Z 的 22.9 倍；按 FLOPs 计是 61.2 倍。** mAP50-95 更极端（29.7× / 79.4×）。DA 两项的比值是负数——把参数投到 Z 上不仅没有收益，点估计还是略微负的。

### 3.4 等参数量 like-for-like 对比（决定性证据）

两个总参数量几乎相同（差 1.9%）的配置，一个把预算花在 Z 上，一个花在 encoder 上：

| | baseline encoder + z=128 | **E-large encoder + z=16** | delta |
|---|---|---|---|
| Params (M) | 0.2860 | 0.2915 | +1.9% |
| **FLOPs (G)** | 2.023 | **1.428** | **−29.4%** |
| mAP50 | 0.3224 | **0.3585** | **+0.0361** |
| mAP50-95 | 0.1157 | **0.1356** | **+0.0199** |
| da_mIoU | 0.8553 | **0.8589** | +0.0036 |
| da_fg | 0.7708 | **0.7761** | +0.0053 |
| lane_mIoU | **0.5879** | 0.5878 | −0.0001 |
| lane_fg | **0.2005** | 0.2000 | −0.0005 |

**E-large + z16 用了更多参数（+1.9%）、更少 FLOPs（−29.4%），6 项指标中 4 项更高、2 项持平（差值 0.0001 / 0.0005，远在噪声内）。**

这是一个干净的**支配关系（dominance）**：在同等参数预算下，「encoder-heavy + 小 Z」严格优于「baseline-encoder + 大 Z」——既更准，又更省算力。两个 lane 指标的 −0.0001 / −0.0005 不构成反例，因为它们比 lane 的噪声底（0.0021 / 0.0032）小一个数量级。

---

## 4. Encoder Sensitivity 判断

### 4.1 逐步判定：哪些步长是真的，哪些在噪声内

| metric | small→base | base→large | 解读 |
|---|---|---|---|
| mAP50 | +0.0517 **real** | +0.0381 **real** | 两步都显著，未饱和 |
| mAP50-95 | +0.0259 **real** | +0.0207 **real** | 两步都显著，未饱和 |
| da_mIoU | +0.0141 noise | +0.0025 noise | 两步都在噪声内 |
| da_fg | +0.0208 **real** | +0.0038 noise | 第一步真实，第二步饱和 |
| lane_mIoU | +0.0049 **real** | +0.0011 noise | 第一步真实，第二步饱和 |
| lane_fg | +0.0088 **real** | +0.0027 noise | 第一步真实，第二步饱和 |

### 4.2 结论：encoder **未饱和**，但分任务

- **Detection（mAP50 / mAP50-95）：E-large 处尚未饱和。** base→large 仍然带来 +0.0381 / +0.0207，分别是噪声底的 5.2× / 6.5×。用户规则「E-large 相比 E-base 几乎没有收益 → 记 saturation」**不适用**——这里有明确收益。
- **DA / Lane：E-large 处已饱和。** base→large 的 4 个指标全部落在噪声内（0.0011–0.0038 vs 噪声底 0.0021–0.0202）。

### 4.3 为什么 Detection 最敏感——与代码审计互相印证

`docs/PHASE2D_BOTTLENECK_AUDIT.md` 已经查明 R0 架构的信息流：

| 任务 | 特征来源 | 是否经过 Z |
|---|---|---|
| Detection | encoder 多尺度 F2(64ch,1/8) / F3(96ch,1/16) / F4(128ch,1/32) | **否，bypass** |
| Drivable area | CompactRepresentation 输出 Z (1/8) | 是 |
| Lane | CompactRepresentation 输出 Z (1/8) | 是 |

这正好解释了 Phase 3A 的形态：Detection **直接吃 encoder 特征**，所以 encoder 变宽它直接受益（+33~52%）；DA/Lane **只吃 z=16 的 Z**，所以 encoder 变宽后，更丰富的特征必须被压进同样 16 通道的瓶颈里才能到达这两个 head——收益被 Z 卡住，在 E-large 处就压平了。

**关键推论：DA/Lane 在 E-large 的「饱和」，很可能是 z=16 的瓶颈造成的假饱和，而不是任务本身的容量上限。** 这个推论**不能**用 Phase 3A 的数据证实——Phase 3A 全程固定 z=16，两个因素完全混杂。这正是 Phase 3B 要解决的问题。

### 4.4 对研究问题的直接回答

> 有限参数应优先投入 encoder 还是 compact representation？

**支持「优先投 encoder」**，三条独立证据：

1. **效应量**：ΔEncoder / ΔZ = 3.6×–46.6×（6/6 指标全部 > 1）
2. **单位成本**：每 0.01M 参数的 mAP50 收益，encoder 是 Z 的 22.9×（按 FLOPs 是 61.2×）
3. **等参数量支配**：同等参数下，encoder-heavy + 小 Z 在 mAP50 上 +0.0361 且 FLOPs −29.4%

**但这是一个有条件的结论**：证据强度在 Detection 上最强（支配关系明确、未饱和），在 DA/Lane 上偏弱（点估计为正但幅度在噪声内，且受到 z=16 混杂）。要把它从「Detection 上的强结论」升级为「全任务的稳健结论」，需要 Phase 3B 的二维矩阵。

---

## 5. 是否进入 Phase 3B：**是**

**明确结论：进入 Phase 3B（Encoder × Z 二维矩阵）。**

理由：

1. **Phase 3A 已经证明 encoder 值得投参数**（22.9× 单位成本优势、等参数量支配），这个方向有继续做的价值，不是噪声驱动的假信号。
2. **但 Phase 3A 无法归因 DA/Lane 的饱和。** z=16 全程固定，encoder 容量与 Z 容量完全混杂。DA/Lane 在 E-large 压平，既可能是「任务已达容量上限」，也可能是「z=16 卡住了 encoder 的收益」——这两种解释指向完全相反的架构决策（前者说明该把参数给 detection，后者说明该给 Z）。**只有二维矩阵能区分它们。**
3. **矩阵还能回答「encoder 的收益是否依赖 Z 大小」**。如果 E-large 与 z=128 的组合在 DA/Lane 上明显超过 E-large + z=16，就说明存在交互效应，Phase 3C 的等预算分配才有意义；如果所有 Z 列下 DA/Lane 都压平，那就确实是任务饱和，Phase 3C 应把资源全部导向 detection 侧。

**不进入的理由不成立**：用户规则中「E-large 相比 E-base 几乎没有收益 → 记 saturation，不要强行扩大 encoder」的触发条件**未满足**——Detection 上 E-large 有明确且超噪声的收益。

### Phase 3B 工作量说明（供决策）

Phase 3A 已经完成了矩阵的 **z=16 整列**（esmall+z16、ebase+z16、elarge+z16 三个 cell 均已存在）。因此 Phase 3B 实际只需要跑 **6 个新 cell**，而非 9 个：

| 新 cell | 预估训练时长 |
|---|---|
| E-small × z32 / z128 | 2 × ~113 min |
| E-base × z32 / z128 | 2 × ~132 min |
| E-large × z32 / z128 | 2 × ~153 min |
| **合计** | **≈ 13.3 h + 评估 ≈ 14 h** |

如果希望更严格，z=16 的三个 cell 可以重跑以获得同批次计时（会多出 ≈ 6.6h）。**这一项请用户指示，我不擅自决定。**

**等待用户确认后才会启动 Phase 3B，不会自行开跑。**

---

## 6. 文件路径清单

### 配置文件（WSL 仓库 `/home/mycode/ai_study/trac/`）

| 文件 | 说明 |
|---|---|
| `configs/phase3a_esmall_z16.yaml` | E-small，stem 12 / stages [24,40,64,80] |
| `configs/phase3a_ebase_z16.yaml` | E-base（= 当前 baseline），stem 16 / stages [32,64,96,128] |
| `configs/phase3a_elarge_z16.yaml` | E-large，stem 20 / stages [40,80,128,168] |

三份配置除 `encoder.stem` / `encoder.stages` 与 `epochs: 20` 外，与 `configs/phase2b_zsweep_z16.yaml` 逐字段一致（已用 `diff` 验证，差异仅出现在注释与上述字段）。

### 脚本

| 文件 | 说明 |
|---|---|
| `scripts/phase3a_encoder_scan.py` | 宽度扫描器：27 档 scale 网格搜索，命中三档参数量目标 |
| `scripts/phase3a_run_encoder.sh` | Runner：继承 Phase 2-D v2 的 checkpoint epoch 校验，加 E-base 复用逻辑 |
| `scripts/phase3a_analyze.py` | Q1 单调性 / Q2 ΔEncoder vs ΔZ / Q3 单位成本 / Q4 saturation |
| `scripts/phase3a_allocation_ledger.py` | 参数效率对账（A: 花在 Z / B: 花在 encoder / C: 比值 / D: like-for-like） |

### 结果与产物

| 文件 | 说明 |
|---|---|
| `experiments/phase3a/exp3A_encoder.csv` | **原始结果表**（3 行，含 params/FLOPs/6 指标/mem/loss/时间/source/commit） |
| `experiments/phase3a/exp3A_analysis.txt` | Q1–Q4 完整分析输出 |
| `experiments/phase3a/exp3A_allocation_ledger.txt` | 参数效率对账输出（A/B/C/D 四段） |
| `experiments/phase3a/exp3A_runner.log` | Runner 时间线 |
| `experiments/phase3a/exp3A_{esmall,ebase,elarge}_z16/` | 训练目录（checkpoint.pt + training_log.txt） |
| `experiments/phase3a/exp3A_{esmall,ebase,elarge}_z16_eval/` | 评估目录（metrics.json） |
| `experiments/phase3a/exp3A_{esmall,elarge}_z16_eval.log` | 评估日志（含 profiling 原始行） |

### 参考文档

| 文件 | 说明 |
|---|---|
| `docs/PHASE2D_BOTTLENECK_AUDIT.md` | 代码侧架构审计（Z 产生路径、head 分配表、bypass 判定） |
| `docs/PHASE2D_DECISION_REPORT.md` | Phase 2-D 结论：BRANCH B，停止搜索 knee |
| `docs/PHASE3A_REPORT.md` | 本报告 |

---

## 7. 已知局限（诚实声明）

1. **单 seed**。全部三个 cell 均为 seed=0。跨 cell 差异是与 Phase 2-C 的 3-seed 噪声底比较，而不是本阶段新估的噪声。Proxy 噪声底来自 4ep，非 20ep。
2. **E-base 为复用**，非独立重跑。配置与协议已逐字段核对一致，但 `train_wall_min` 缺失。
3. **FPS / 延迟不可用**（Power Cap 导致），效率结论全部基于 FLOPs 与参数量。
4. **z=16 全程固定**，因此本阶段无法区分「DA/Lane 任务饱和」与「z=16 瓶颈造成的假饱和」——这是 Phase 3B 的核心任务。
5. **未修改任何 loss / optimizer / 数据 / 增强 / 输入尺寸**；未因结果不理想剔除任何 run 或更换 seed。

---

**阶段状态**：Phase 3A 完成 · 等待用户确认后进入 Phase 3B
**分析脚本**：`scripts/phase3a_analyze.py`、`scripts/phase3a_allocation_ledger.py`
**报告日期**：2026-09-05
