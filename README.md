# TRAC: Task-Resource Adaptive Compact Representation
## Lightweight Multi-Task Driving Perception (Detection + Drivable Area + Lane)

研究目标：面向资源受限设备，学习紧凑共享表示（Compact Driving Representation Z），
由 Task-Resource Router 按「任务 × 资源 × 表示」动态分配模型容量，
使**同一个模型**在多个计算档位（Tiny/Medium/Large）下运行，
建立优于固定容量模型的 Accuracy–Compute Pareto Frontier。

> Phase 1 核心假设：
> - **H1** 不同任务对共享特征需求不同 ⇒ task-wise dynamic allocation 优于统一宽度。
> - **H2** Compact Representation 在小参数预算下同时保留三任务信息。
> - **H3** 动态信息分配取得更好的 Accuracy–Compute Pareto Frontier，无需训练多个模型。

详细任务书见 `docs/TASKBOOK.md`（算法开发任务书摘要）与 `docs/PHASE0_DEV_PREP.md`（开发准备报告）。

---

## 目录结构

```
trac/
├── models/               # 我们的模型（Phase 1 实现）
│   ├── encoder/          #   轻量共享 Encoder（Module A 起点）
│   ├── representation/   #   Compact Driving Representation Z
│   ├── router/           #   Task-Resource Router（Module B，核心创新）
│   ├── heads/            #   Det / DA / Lane 任务头
│   └── adaptive_model.py #   单模型多档位组装
├── losses/               # L_task + L_budget（budget-aware）
├── distillation/         # Phase 2：Cross-Architecture Distillation（暂为空）
├── datasets/             # BDD100K 三任务数据集加载
├── training/             # 训练管线（Stage A/B/C/D 策略）
├── evaluation/           # 统一指标：mAP / mIoU / Lane IoU / params / FLOPs / latency
├── profiling/            # FLOPs / latency / GPU memory / FPS 硬件友好性测量
├── visualization/        # Router 行为可视化、Pareto 曲线、failure-mode 统计
├── configs/              # 模型/训练/实验配置（baseline ↔ ours 可切换）
├── experiments/          # 每个实验一个目录（exp_001/...），自动记录 §22 全部字段
├── baselines/            # 已克隆的 baseline 仓库（各自保留 .git）
│   ├── TwinLiteNet/      #   B2（DA+Lane，权重已在仓库 pretrained/best.pth）
│   ├── TwinLiteNetPlus/  #   B2+（DA+Lane，nano 权重已从 HF 镜像获取）
│   └── TriLiteNet/       #   B3（Det+DA+Lane，权重待获取/从头训练）
├── scripts/              # 冒烟测试、数据准备、一键复现等
├── data/                 # BDD100K（images/100k, images/10k, segments, lanes, labels）
├── requirements.txt
└── README.md
```

兄弟 baseline 仓库（工作区根目录，各自独立 venv，已完成统一速度基准测量，见
`../experiments/FINAL_REPORT.md`）：
- `../YOLOP`        B1（权重 `weights/End-to-end.pth` ✓，另有 320/640/1280 ONNX）
- `../HybridNets`   参考（权重 `weights/hybridnets.pth` ✓）
- `../YOLOPv2` `../YOLOPX` `../YOLOPv3` `../A-YOLOM` 参考实现

---

## 环境

- 系统：WSL2 Ubuntu，Python 3.12.3，无 conda
- GPU：RTX 5060 Laptop 8GB（sm_120 / Blackwell），CUDA 13.x，驱动 580.95
- 本项目统一使用 `../gpu_env` venv（torch 2.11.0+cu130，CUDA 可用，
  thop/ptflops/albumentations/tensorboardX/yacs/timm 已装）
- 复现依赖清单见 `requirements.txt`

验证命令：
```bash
../gpu_env/bin/python scripts/smoke_baselines.py --device cuda   # baseline 导入/前向/参数/FLOPs
```

---

## Baseline 现状（Phase 0 核实）

| ID | 模型 | 任务 | 代码 | 官方权重 | 冒烟测试（640×640, FP32） |
|----|------|------|------|----------|------------------------------|
| B1 | YOLOP | Det+DA+Lane | ✓ | ✓ End-to-end.pth | ✓（gpu_env 导入正常；速度基准见 ../experiments/FINAL_REPORT.md）|
| B2 | TwinLiteNet | DA+Lane | ✓ | ✓ pretrained/best.pth | ✓ 0.44M / 14.1G |
| B2+ | TwinLiteNetPlus | DA+Lane | ✓ | nano ✓（HF 镜像）；small/medium/large 待获取 | ✓ nano 33K / 1.9G |
| B3 | TriLiteNet | Det+DA+Lane | ✓ | ✗（Google Drive，网络不通，待用户/从头训练）| ✓ tiny 0.15M / 1.8G |
| B4 | 轻量 CNN 三头 baseline | Det+DA+Lane | Phase 1 构建 | — | — |

官方权重下载地址（需用户侧网络）：TriLiteNet:
https://drive.google.com/drive/folders/1wLZqemCxxzwiFeFUGY1zMaqcKoQLHFyK
TwinLiteNetPlus: https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz

---

## 数据（BDD100K）

目标布局：
```
data/bdd100k/
├── images/100k/{train,val}/
├── images/10k/test/
├── segments/{train,val}/   # drivable area 标注
├── lanes/{train,val}/      # lane 标注
└── labels/                 # bdd100k_labels_images_{train,val}.json（检测框）
```

当前状态：
- 图片：`../trac_data/bdd100k_download/`（bitmind/bdd100k-real @ hf-mirror，
  100k×5 部分 + 10k 测试集，下载中；已验证 bdd100k_seg.zip = 1 万张 jpg **图片**，非标注）
- 标注：**未取得**（官方站 bdd-data.berkeley.edu 与 Google Drive 均不通；待用户提供或授权）
- 备选标注来源：OpenDataLab（opendatalab.com，可达，需注册 token）、
  HuggingFace gated `bdd100k/bdd100k`（需账号 + 接受许可，经 hf-mirror）

---

## Phase 1 里程碑（对应任务书 §27 Step 4–14）

1. **Module A** Compact Representation：轻量 Encoder（ESPNet 风格块/深度可分离卷积）
   + 多尺度特征 + 紧凑 Z（`models/encoder` + `models/representation`）
2. **Module B** Task-Resource Router（`models/router`）：
   Z → Global Pool → 小 MLP → 任务级 allocation logits → 离散档位（Tiny/Medium/Large）
   - **Nested/Prefix channel 结构**（1.0x/0.75x/0.5x/0.25x 连续前缀通道，硬件友好）
   - 允许 task-wise heterogeneous allocation（Det=Large, DA=Tiny, Lane=Medium）
3. Static 三档模型 + 统一评估（`evaluation` + `profiling`）
4. budget-aware 训练（`losses`：L_task + λ_budget·L_budget）
5. 实验 1–5（Static vs Dynamic；Shared vs Task-wise；Random vs Learned；Equal Budget；
   Pareto Frontier）+ 失败模式 A–E 统计 + Router 行为可视化

训练策略（§10）：Stage A 固定最大容量 → Stage B 冻结 Encoder 训 Router+Heads
→ Stage C Joint Fine-tune → Stage D budget-aware。

---

## 实验记录规范（§22）

每个实验 `experiments/exp_XXX/` 自动保存：
`config.yaml`、`metrics.json`、`training_log.txt`、`checkpoint.pt`、`allocation_statistics.csv`；
metrics.json 必含：experiment_id / model / commit_hash / dataset_version / input_size /
parameters / FLOPs / average_latency / P50 / P95 / FPS / GPU_memory /
mAP / DA_mIoU / Lane_IoU / budget_distribution / seed。
