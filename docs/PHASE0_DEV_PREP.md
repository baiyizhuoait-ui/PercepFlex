# Phase 0 开发准备报告

日期：2026-08-28 ｜ 目标：为 Phase 1（MVP：Compact Representation + Task-Resource Router）铺平道路

## 0.1 工作区盘点（Step 1–2：读现有项目、确认可复用代码）

### 已有仓库（工作区根目录，均含 .git 与官方权重）
| 仓库 | 状态 | 可复用点 |
|------|------|----------|
| `YOLOP/` | ✅ 官方权重 End-to-end.pth + 320/640/1280 ONNX | **B1 baseline**；三任务多任务范式参考 |
| `YOLOPv2/` `YOLOPX/` `YOLOPv3/` | ✅ 各有权重/演示 | 参考实现（A-YOLOM 风格改进） |
| `HybridNets/` | ✅ hybridnets.pth | 参考（EfficientNet-B3 共享骨干多任务） |
| `A-YOLOM/` | ✅ | YOLOv8 多任务，参考 |

> 上一阶段（2026-08-21~25）已完成 YOLOP 家族 + HybridNets 的**统一速度基准**（FP16/bs=1/CUDA 同步，
> RTX 5060 实测：YOLOP 82 FPS / 7.94M / 131MiB 等），报告在 `../experiments/FINAL_REPORT.md`，
> 其统一评测口径可直接沿用。注意：此前实验**无精度验证**（无 ground truth），Phase 1 必须补上。

### 新克隆 baseline（`trac/baselines/`，各自保留 .git）
| 仓库 | 版本/预设 | 冒烟测试（640×640 FP32, gpu_env） |
|------|-----------|------------------------------------|
| `TwinLiteNet` | DA+Lane 两任务 | ✅ 0.440M / 14.06G / 8.0ms；权重 `pretrained/best.pth` ✅ |
| `TwinLiteNetPlus` | nano/small/medium/large 四档 | ✅ nano 33.4K / 1.89G / 5.5ms；nano 权重已从 HF 镜像获取并验证可加载 ✅ |
| `TriLiteNet` | tiny/small/base 三档，Det+DA+Lane | ✅ tiny 0.151M / 1.83G；small 0.592M / 6.61G；base 2.35M / 25.38G；官方权重待获取 |

### 环境（Step 3 前置）
- 本项目统一用 `../gpu_env`（Python 3.12.3，torch 2.11.0+cu130，CUDA 13.0 可用，
  thop/ptflops/albumentations/tensorboardX/yacs/timm ✓，timm/elephant 本次补装）。
- baseline 仓库官方 requirements（torch==1.8.0 等）在 Python 3.12 + Blackwell 下不可安装，
  沿用上一阶段的决策（torch 2.11/2.13+cu130），已在 README 注明。
- 各兄弟仓库仍用自己的 venv（yolop_env 等），不冲突。

## 0.2 数据现状（BDD100K）

### 已获得
- 开发子集 10,010 张唯一图片（已解压到 `trac/data/bdd100k/images/dev/`）：
  来源 hf-mirror 的 bitmind/bdd100k-real，其中 `10k.zip` 与 `bdd100k_seg.zip` 均为 10,002 张 JPG 且
  99.9% 重复（9990 张相同）——即该镜像只有约 1 万张有效图片。
- 该镜像的 `100k_part01~05.zip` 全部损坏：全新完整下载（无断点续传）仍无法解压
  （无 EOCD 中央目录），判断为上传方分卷错误/被截断，已删除，不再依赖。

### 未获得（用户已确认自行下载后拷入工作区）
1. **全量图片**（images/100k/{train,val}、images/10k/test）
2. **检测框标注**（bdd100k_labels_images_train/val.json）
3. **可行驶区域标注**（segments/*.png）
4. **车道线标注**（lanes/*.png）

原因：官方站 bdd-data.berkeley.edu 与 Google Drive 在当前网络均不可达。
用户选择「自己下载后拷进工作区」，目标结构与下载途径见 `docs/USER_DOWNLOADS.md`。

## 0.3 Phase 1 前需要用户做的事
- [ ] **决策标注来源**（上节 1/2/3/4 选一；若选 1 或 2，请提供 token/确认注册）
- [ ] （可选）TriLiteNet 官方权重（GDrive 文件夹 1wLZqemCxxzwiFeFUGY1zMaqcKoQLHFyK）与
      TwinLiteNetPlus small/medium/large 权重（GDrive 文件夹 1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz），
      可自行下载后放入 `trac_data/weights/`；否则 Phase 1 从零训练 baseline（在统一数据上更公平）。
- [ ] 确认图片下载继续（~6.4GB，已在进行；磁盘剩余 898G 充足）

## 0.4 已建立的开发基础设施
- `trac/` 独立 git 仓库（commit 8147d8e），后续实验记录 commit_hash。
- `scripts/smoke_baselines.py`：B1–B3 冒烟 + 参数/FLOPs/前向耗时（一键验证环境与 baseline 可用）。
- `experiments/README.md`：§22 实验记录规范（metrics.json 字段、Best/Average/Worst 硬规则）。
- `requirements.txt` / `.gitignore` / `README.md` / `docs/TASKBOOK.md`（任务书完整存档）。

## 0.5 下一步（Phase 1 Step 3–4 起）
1. 标注到位后：写 `datasets/`（BDD100K 三任务加载 + YOLOP 式 train/val split 配置）。
2. `evaluation/`：统一评测（mAP + mIoU + Lane IoU + 精度/延迟全套，复用上一阶段口径）。
3. Module A（encoder + representation）→ Module B（router + nested prefix width）→ Static 三档 → Dynamic。
4. 训练 Stage A–D；实验 1–5；失败模式统计与可视化。

## 0.6 风险与注意事项
- bitmind 图片镜像未验证与官方文件一一对应（文件名看起来是 BDD100K 命名）；解压后需抽查校验数量。
- RTX 5060 8GB 显存：训练 batch 需保守（640×640 三任务模型，估计 ≤16）。
- WSL 内存 15GB：大数据加载注意 workers/缓存设置。
- 上一阶段环境为 torch 2.13（yolop_env 等），本项目 gpu_env 为 2.11；若某些 baseline 代码在 2.11 报错
  （如 torch.meshgrid 行为），可在各自 venv 中跑 baseline，ours 跑 gpu_env。
