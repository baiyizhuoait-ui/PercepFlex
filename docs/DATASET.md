# BDD100K 数据集状态（Phase 0 整合后）

> 更新：2026-08-28。来源：用户从 D 盘拷入（官方站/镜像下载），已复制到 `trac/data/bdd100k/`。

## 目录布局（官方结构）

```
trac/data/bdd100k/
├── images/100k/train/      70,000 张
├── images/100k/val/        10,000 张
├── images/10k/test/         2,000 张   ← 官方 test 为 10k，用户只下载了 2k（Phase 1 用不到 test，无碍）
├── labels/
│   ├── bdd100k_labels_images_train.json   1.45GB（2018 v1.0 检测标注）
│   └── bdd100k_labels_images_val.json       208MB
├── segments/masks/train/    7,000 张 DA 掩码（PNG）
├── segments/masks/val/      1,000 张 DA 掩码（PNG）
├── lanes/masks/train/      70,000 张 lane 掩码（PNG）
└── lanes/masks/val/        10,000 张 lane 掩码（PNG）
```

## 三任务可用性核对（2026-08-28，程序化验证）

| 任务 | 标注来源 | 训练集 | 验证集 | 状态 |
|------|----------|--------|--------|------|
| Detection | labels/*.json | 70,000 | 10,000 | ✅ 完整，json 内文件名与图片一一对应 |
| Lane | lanes/masks/* | 70,000 | 10,000 | ✅ 掩码与图片 100% 对应 |
| Drivable Area | segments/masks/* | **2,976** | **454** | ⚠️ 部分可用（见下） |

### ⚠️ Drivable Area 掩码版本不匹配（重要）

`segments/masks/` 的 8,000 张掩码**并非**标注在本仓库 `images/100k/` 的同一批图片上：

- 8,000 张掩码中仅 **3,430 张**能在 `images/100k/{train,val}` 找到对应图片
  （train 2,976 + val 454）；其余 **4,570 张**对应的图片不在本仓库的 100k 图片集中
  （其名字与 hf-mirror bitmind 10k 样本一致 —— 与用户下载的 100k 图片是不同版本/划分）。
- 结论：DA 掩码来自另一版 BDD100K 划分，与本仓库图片集不匹配。

**处理方案（已定）**：
1. `datasets/` 加载器按**实际掩码↔图片交集**构建 DA 标注清单：
   `da_train.txt`（2,976 条）、`da_val.txt`（454 条）—— 合法、无跨 split 泄漏。
2. Phase 1 MVP 用 2,976 张训练 DA 足够（YOLOP 原论文全任务仅用 6,972 张）。
3. 论文最终数字如需完整 7k 训练 DA，需用户重新下载**与本仓库图片版本匹配的**
   `bdd100k_drivable_2020.zip` 掩码（官方站 bdd-data.berkeley.edu / OpenDataLab）。
   在取得前，README 中如实标注 `DA trained on 2,976 images subset`，不冒充全量。

### 图片集版本说明
- `labels/*.json` 时间戳 2018（v1.0 检测标注），与 `images/100k` 匹配 ✅
- `lanes/masks` 与 `images/100k` 100% 匹配 ✅
- `segments/masks` 与 `images/100k` 部分匹配（见上）⚠️
- 三个任务在**同一批图片**上的交集训练清单由 `scripts/build_splits.py`（Phase 1 编写）统一生成，
  保证三任务 loss 在相同样本上计算（与 YOLOP/TriLiteNet 训练协议一致）。

## 开发子集（保留）
- `images/dev/`：10,010 张（hf-mirror 有效部分），其中 5,430 张与 100k 图片重复、
  4,580 张仅存在于该样本。仅用于管线开发调试，不参与官方协议训练/评测。

## 校验命令
```bash
gpu_env/bin/python scripts/verify_dataset.py   # Phase 1 提供：三任务交集、计数、对应关系
```
