# 服务器运行手册（HIVE：Ubuntu + TITAN RTX，全程免密码）

> 更新：2026-08-30（已按你实际环境二次核对）
> **服务器**：`cmu@HIVE`，Ubuntu 22.04，**NVIDIA TITAN RTX 24GB**（Turing sm_75）
> **项目路径**：`/home/cmu/Desktop/trac/`
> **原则：全程不用 sudo / 密码、不用 venv**。
> 因为系统没有 `python3-venv`（装它要 sudo），改用「系统 python + 用户级依赖」
> （`python3 -m pip install --user`，自动装到 `/home/cmu/.local/`，无需密码）。

---

## 0. 环境确认（已核对 2026-08-30）

| 项 | 值 | 备注 |
|---|---|---|
| GPU | TITAN RTX 24GB（cc 7,5） | ✓ |
| 驱动 / CUDA | 535.288.01 / 12.2 | ✓ |
| torch | **2.1.1+cu121，cuda_avail=True**（系统已装） | ✓ 直接用 |
| Python | 3.10.12 | ✓ |
| 磁盘 | 5.2T 可用 | ✓ |
| 依赖 | torch/numpy/cv2 系统已有；scikit-learn/albumentations/timm/tensorboardX/yacs/ptflops/prefetch_generator/imageio 已装到用户级 | 还差 **thop、seaborn**（见下） |

> ⚠️ 不用 `source trac_venv/bin/activate`（venv 不存在）。所有命令直接 `python3 ...`。
> 若提示 `pip` 不在 PATH，用 `python3 -m pip` 代替 `pip`。

---

## 1. 环境准备（全部 🔓，无需密码）

### 1.1 补装还没装的依赖（用用户级安装）

```bash
python3 -m pip install --user --upgrade pip
# 补装 thop、seaborn（上次列表里缺这两个）
python3 -m pip install --user thop seaborn
# 确认 torch / torchvision / 关键依赖都可用
python3 -c "import torch, torchvision, numpy, cv2, sklearn, albumentations, timm, yacs, thop, ptflops; print('torch', torch.__version__, '| tv', torchvision.__version__, '| cuda', torch.cuda.is_available(), '| cc', torch.cuda.get_device_capability())"
# 期望：torch 2.1.1+cu121 | cuda True | cc (7, 5)
```

> 如果 `thop` 装不上（编译问题），项目里已自带 `profiling/flops_real.py`（更准），
> 可忽略 thop；但 `seaborn` 是 TriLiteNet 可视化用到，建议装。

---

## 2. 项目与数据（全部 🔓）

### 2.1 从 GitHub 拉取项目（含全部权重，无需 GDrive）

```bash
mkdir -p /home/cmu/Desktop
cd /home/cmu/Desktop
git clone https://github.com/baiyizhuoait-ui/PercepFlex.git trac
cd trac
ls weights/   # 应看到 YOLOP/TwinLiteNet/TwinLiteNetPlus/TriLiteNet 全部权重（共 ~118MB，随仓库走）
ls trained_models/   # Static_noKD.pt / Static_KD_YOLOP.pt / Dynamic_noKD.pt
```

### 2.2 拉取 baseline 模型代码（4 个公共仓库，评测用）

> 权重已随主仓库提供；这里只 clone **模型代码**（评测时 `sys.path` 会导入它们）。
> 注意 `baselines/` 目录在仓库里是空的（嵌套仓库不入库），需按下面命令补全：

```bash
cd /home/cmu/Desktop/trac/baselines
git clone --depth 1 https://github.com/chequanghuy/TriLiteNet.git        TriLiteNet
git clone --depth 1 https://github.com/chequanghuy/TwinLiteNet.git       TwinLiteNet
git clone --depth 1 https://github.com/chequanghuy/TwinLiteNetPlus.git   TwinLiteNetPlus
# YOLOP 代码在 trac 的上一层目录（evaluation/evaluate_baseline.py 里写死 ../YOLOP）
cd /home/cmu/Desktop
git clone --depth 1 https://github.com/hustvl/YOLOP.git YOLOP
# 校验 baseline 权重能加载（读 trac/weights/，无需任何额外下载）
cd /home/cmu/Desktop/trac
python3 -c "
from scripts.load_baseline_weights import load_baseline
for n, ps in [('TwinLiteNet',[None]),('TwinLiteNetPlus',['nano','large']),('TriLiteNet',['tiny','base'])]:
    m = load_baseline(n, ps[0] if n!='TwinLiteNetPlus' else ps[-1], 'cpu')
    print(n, 'OK', sum(x.numel() for x in m.parameters())/1e6, 'M')
"
```

### 2.3 数据（BDD100K，~11G，需单独传输）

> 数据不在 GitHub 里（太大）。服务器有两个来源任选：
> **(A) 从你本机打包传过去**（推荐，本机已有完整 `data/bdd100k/`）；
> **(B) 服务器能联网时从 BDD100K 官网重新下载**（若服务器访问不了 GDrive，选 A）。

#### 方式 A：本机打包 → scp → 服务器解压

**① 本机（WSL）打包**（在 `ai_study/trac` 所在机器上）：
```bash
cd /home/mycode/ai_study/trac
# 打包成单个 tar（图片是 jpg 已压缩，gz 压不动，直接 tar 更快）
tar -cf /home/mycode/bdd100k_20260830.tar -C data bdd100k
# 若想更小可加 z 用 gzip：tar -czf ...（慢一些）
ls -lh /home/mycode/bdd100k_20260830.tar   # ~11G
```

**② 本机 → 服务器传输**（scp；若提示输密码是 cmu 登录密码，非 root）：
```bash
scp /home/mycode/bdd100k_20260830.tar cmu@HIVE:/home/cmu/Desktop/
# 或断点续传更快用 rsync：
rsync -avP /home/mycode/bdd100k_20260830.tar cmu@HIVE:/home/cmu/Desktop/
```

**③ 服务器上解压**（解压到 `data/`，得到 `data/bdd100k/...`）：
```bash
cd /home/cmu/Desktop/trac
tar -xf /home/cmu/Desktop/bdd100k_20260830.tar -C data
ls data/bdd100k/
# 期望看到：images/  labels/  segments/  lanes/  splits/
```

**④ 校验**（确认目录结构、数量正确）：
```bash
cd /home/cmu/Desktop/trac
python3 scripts/verify_dataset.py
# 期望：images train=70000 val=10000；三任务交集 tri_train=69863 / tri_val=10000
```

#### 方式 B：OpenDataLab 国内镜像直接下载（免传文件，推荐服务器用）
> OpenDataLab（上海AI实验室）有 BDD100K 官方镜像，国内高速直连。
> 数据页 https://opendatalab.com/OpenDataLab/BDD100K ；需先注册账号 https://opendatalab.org.cn/register
> （浏览器注册，与服务器无关）。

```bash
# 1) 装工具 + 登录（登录用 OpenDataLab 账号，不是服务器密码）
python3 -m pip install --user opendatalab
odl login

# 2) 确认数据集确切名字与文件清单（若名字不同，用 odl search BDD100K 找）
odl search BDD100K
odl ls OpenDataLab/BDD100K          # 看总文件/大小
odl ls OpenDataLab/BDD100K/images   # 看 images 子目录（100k/10k）

# 3) 只下本项目需要的子集（Detection+可行驶区+Lane）
cd /home/cmu/Desktop/trac/data
odl get OpenDataLab/BDD100K/images/100k      # train.zip + val.zip
odl get OpenDataLab/BDD100K/labels           # bdd100k_labels_images_{train,val}.json
odl get OpenDataLab/BDD100K/segments         # 可行驶区分割掩码（若叫 seg_masks 以此为准）
odl get OpenDataLab/BDD100K/lanes            # 车道线掩码（若叫 lane_masks 以此为准）

# 4) 解压 + 整理成项目结构 data/bdd100k/{images,labels,segments,lanes}
cd /home/cmu/Desktop/trac/data
unzip -o bdd100k_images_100k_train.zip -d bdd100k/images/100k/train/
unzip -o bdd100k_images_100k_val.zip   -d bdd100k/images/100k/val/
unzip -o bdd100k_labels_images_train.zip -d bdd100k/labels/
unzip -o bdd100k_labels_images_val.zip   -d bdd100k/labels/
unzip -o seg_masks_train.zip -d bdd100k/segments/masks/train/   # 名字以 odl ls 为准
unzip -o seg_masks_val.zip   -d bdd100k/segments/masks/val/
unzip -o lane_masks_train.zip -d bdd100k/lanes/masks/train/
unzip -o lane_masks_val.zip   -d bdd100k/lanes/masks/val/

# 5) 校验
cd /home/cmu/Desktop/trac
python3 scripts/verify_dataset.py
# 期望：images train=70000 val=10000；三任务交集 tri_train=69863 / tri_val=10000
```
> `splits/{tri_train,tri_val}.txt` 是项目自定义三任务交集划分，已随仓库走，勿覆盖。
> OpenDataLab 下载的是官方原始压缩包，解压后的目录名可能与上述不同，**以 `odl ls` 看到的实际文件名为准**，解压到对应目录即可。

---

## 3. 冒烟测试（确认脚本能跑，🔓）

```bash
cd /home/cmu/Desktop/trac
python3 training/train.py --config configs/train_stageA.yaml --num-images 64 --epochs 1 --outdir /home/cmu/Desktop/trac/experiments/smoke
# 应正常跑完，末尾打印 avg_loss 与 saved checkpoint
```

---

## 4. 正式实验（全部 🔓；outdir 都在项目内）

> TITAN RTX 24GB：建议把 config 里的 `batch_size` 提到 32（约快 2×）。
> 所有命令直接 `python3`，无需 activate。

### Experiment A（等预算 Static vs Dynamic，3 seeds）
```bash
cd /home/cmu/Desktop/trac
EXP=/home/cmu/Desktop/trac/experiments/expA_equal_budget

# static_eb（独立训练 @0.45，等预算 ~0.90G）—— 3 seeds
for s in 0 1 2; do
  python3 training/train.py --config configs/train_stageA_eb.yaml --seed $s --outdir $EXP/static_eb_s$s
done
# dynamic（A6-B2-C2-D2）—— 3 seeds
for s in 0 1 2; do
  python3 training/train.py --config configs/train_stageA.yaml --seed $s --epochs 6 --outdir $EXP/dynA_s$s
  python3 training/train.py --config configs/train_stageB.yaml --seed $s --init $EXP/dynA_s$s/checkpoint.pt --epochs 2 --outdir $EXP/dynB_s$s
  python3 training/train.py --config configs/train_stageC.yaml --seed $s --init $EXP/dynB_s$s/checkpoint.pt --epochs 2 --outdir $EXP/dynC_s$s
  python3 training/train.py --config configs/train_stageD.yaml --seed $s --init $EXP/dynC_s$s/checkpoint.pt --epochs 2 --outdir $EXP/dynD_s$s
done
# 评测
python3 evaluation/evaluate_baseline.py --baseline OursStatic --preset $EXP/static_eb_s0/checkpoint.pt --static-width 0.45 --outdir $EXP/static_eb_s0_eval
python3 evaluation/evaluate_baseline.py --baseline OursDynamic --preset $EXP/dynD_s0/checkpoint.pt --outdir $EXP/dynD_s0_eval --alloc-stats
# 汇总
python3 scripts/expA_analyze.py $EXP
```

### Experiment D（容量扫描 0.5/1.0/2.0M）
```bash
cd /home/cmu/Desktop/trac
EXP=/home/cmu/Desktop/trac/experiments/expD_compact_z
for t in 0.5M 1.0M 2.0M; do
  python3 training/train.py --config configs/train_stageA_$t.yaml --outdir $EXP/ours_$t
  python3 evaluation/evaluate_baseline.py --baseline OursStatic --preset $EXP/ours_$t/checkpoint.pt --outdir $EXP/ours_${t}_eval
done
python3 scripts/expD_analyze.py
```

### Experiment F/G（单 teacher KD，离线方案）
```bash
cd /home/cmu/Desktop/trac
EXP=/home/cmu/Desktop/trac/experiments/expF_single_teacher

# 1) 预计算 teacher 目标（TITAN RTX 可 batch 64；磁盘充足，可全量 70k）
for t in YOLOP TwinLiteNetPlus TriLiteNet; do
  python3 scripts/precompute_teacher.py --teacher $t --num-images 70000 &
done
wait
# 2) 全量离线 KD 训练 + 评测
for t in YOLOP TwinLiteNetPlus TriLiteNet; do
  python3 training/train.py --config configs/train_stageA_offkd_$t.yaml --num-images 70000 --epochs 6 --outdir $EXP/kd_${t}_full
  python3 evaluation/evaluate_baseline.py --baseline OursStatic --preset $EXP/kd_${t}_full/checkpoint.pt --outdir $EXP/kd_${t}_full_eval
done
python3 scripts/expFG_analyze.py
```

### Experiment I（KD 后重测 Dynamic）
```bash
cd /home/cmu/Desktop/trac
EXP=/home/cmu/Desktop/trac/experiments/expI_kd_dynamic
KD=/home/cmu/Desktop/trac/experiments/expF_single_teacher/kd_YOLOP_full/checkpoint.pt
python3 training/train.py --config configs/train_stageB.yaml --init $KD --epochs 2 --outdir $EXP/kdD_B
python3 training/train.py --config configs/train_stageC.yaml --init $EXP/kdD_B/checkpoint.pt --epochs 2 --outdir $EXP/kdD_C
python3 training/train.py --config configs/train_stageD.yaml --init $EXP/kdD_C/checkpoint.pt --epochs 2 --outdir $EXP/kdD_D
python3 evaluation/evaluate_baseline.py --baseline OursDynamic --preset $EXP/kdD_D/checkpoint.pt --outdir $EXP/kdD_eval --alloc-stats
```

### 汇总图表
```bash
cd /home/cmu/Desktop/trac
python3 scripts/gen_figures.py
```

---

## 5. 当前可立即执行的顺序

1. 🔓 第 1 节：`python3 -m pip install --user thop seaborn` + 验证一行
2. 🔓 第 2 节：拷项目 + 数据 + `verify_dataset.py`
3. 🔓 第 3 节：冒烟测试
4. 🔓 第 4 节：正式实验 A→D→F/G→I
5. 🔓 完成后把 `experiments/` 的 metrics.json / 报告拷回本地汇总

> 全程不需要密码。若某条 pip 编译缺系统库，先把报错贴给我，我给**无需密码**的替代。

---

## 6. 注意事项

- 不用 venv（装不了），直接用系统 python + 用户级依赖；`python3` 会自动加载 `/home/cmu/.local`。
- 若 `python3 -m pip` 报「external command not runnable as root」之类，说明权限问题，把报错贴我。
- torch 2.1.1 vs 本地 2.11：state_dict 跨版本兼容；有 API 差异报错就贴给我。
- 服务器 GPU 上已有一个 python 进程在用 586MiB，不影响你训练（24GB 充足）。
