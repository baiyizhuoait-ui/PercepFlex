# 用户需要下载的东西（截至 Phase 0 结束）

> 生成日期：2026-08-28。以下内容本机网络无法访问（官方站/Google Drive 不通），需要你用自己的网络下载后拷入工作区。

## 1. BDD100K 数据集（最重要）

目标目录结构（放在 WSL 里，推荐 `/home/mycode/ai_study/trac/data/bdd100k/`；
如果你从 Windows 拷入，可放 D 盘再告诉我路径，我用软链/拷贝接入）：

```
bdd100k/
├── images/
│   ├── 100k/train/        # 70,000 张
│   ├── 100k/val/          # 10,000 张
│   └── 10k/test/          # 10,000 张
├── labels/                # 检测框标注
│   ├── bdd100k_labels_images_train.json
│   └── bdd100k_labels_images_val.json
├── segments/train/        # 可行驶区域掩码（PNG，train+val 各若干）
├── segments/val/
├── lanes/train/           # 车道线掩码（PNG）
└── lanes/val/
```

> 官方布局：`images/100k/{train,val}`、`images/10k/test`、`drivable/`（即 segments）、`lanes/`、
> `bdd100k_labels_images_{train,val}.json`。YOLOP/TwinLite/TriLite 均用此布局。

下载途径（任选，按推荐排序）：
1. **OpenDataLab**（国内可达）：https://opendatalab.com/OpenDataLab/BDD100K —— 注册后在其页面/CLI 下载 images + labels + seg 全套。
2. **官方站**（需注册）：https://bdd-data.berkeley.edu/ （需邮箱注册获取下载密码）。
3. 你手头已有的任何镜像（百度网盘等）。

> 说明：hf-mirror 上 bitmind/bdd100k-real 镜像的 100k 分卷已确认损坏（全新下载也无法解压），
> 只有约 1 万张图片有效（我已解压为开发子集 `trac/data/bdd100k/images/dev/`，用于 Phase 1 管线开发；
> 正式训练/评测需要你提供全量图片与标注）。

## 2. TriLiteNet 官方权重（可选）

- 地址：https://drive.google.com/drive/folders/1wLZqemCxxzwiFeFUGY1zMaqcKoQLHFyK
- 下载后放到：`/home/mycode/ai_study/trac_data/weights/trilitenet/`
- 用途：可选——Phase 2 作 KD teacher、或校验我们复现的 TriLiteNet 指标范围。
  如果不提供，Phase 1 将从零训练 TriLiteNet（在统一数据/配置下对比反而更公平）。

## 3. TwinLiteNetPlus 权重（small/medium/large，可选）

- 地址：https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz
- 下载后放到：`/home/mycode/ai_study/trac_data/weights/twinlitenetplus/`
- 说明：nano 档权重已从 HF 镜像获取并验证（`trac_data/weights/tlp_nano.safetensors`），
  small/medium/large 用于 Equal-Budget 对比的完整档位覆盖；缺档位则 Phase 1 从零训练补齐。

## 4. 拷入后告诉我

- BDD100K 数据路径（若不在默认位置）
- 权重文件是否已放好

我会：解压/校验 → 写 `datasets/` 加载器 → 开始 Phase 1 Step 3（统一 baseline evaluation）。
