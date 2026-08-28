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
| Detection | labels/*.json | 69,863 | 10,000 | ✅ 完整（官方 json 即 69,863 条），与图片一一对应 |
| Lane | lanes/masks/* | 70,000 | 10,000 | ✅ 掩码与图片 100% 对应 |
| Drivable Area | segments/masks/* | 70,000 | 10,000 | ✅ 完整，掩码与图片 100% 对应（用户 2026-08-28 重新下载） |

### ✅ DA 掩码已补齐（2026-08-28）

第一版 `segments/masks/`（7k/1k）与图片版本不匹配（仅 3,430/8,000 可用）。
用户已重新下载**完整版** `segments/`（70,000 train + 10,000 val），
程序化验证：掩码文件名与 `images/100k` **100% 一一对应**。
三任务交集清单：`splits/tri_train.txt`（69,863 条）/ `splits/tri_val.txt`（10,000 条），
即完整官方协议（det json 官方为 69,863 条）。

### 图片集版本说明
- `labels/*.json`、`lanes/masks`、`segments/masks` 与 `images/100k` 全部 100% 匹配 ✅
- 三个任务在**同一批图片**上的交集清单已由 `scripts/verify_dataset.py` 生成
  （`splits/*.txt`），保证三任务 loss 在相同样本上计算（与 YOLOP/TriLiteNet 训练协议一致）。

## 开发子集（保留）
- `images/dev/`：10,010 张（hf-mirror 有效部分），其中 5,430 张与 100k 图片重复、
  4,580 张仅存在于该样本。仅用于管线开发调试，不参与官方协议训练/评测。

## 校验命令
```bash
gpu_env/bin/python scripts/verify_dataset.py   # Phase 1 提供：三任务交集、计数、对应关系
```
