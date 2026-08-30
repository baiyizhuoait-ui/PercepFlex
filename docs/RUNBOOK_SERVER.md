# 服务器运行手册（HIVE：Ubuntu + TITAN RTX，全程免密码）

> 更新：2026-08-30（已按你实际环境核对）
> **服务器**：`cmu@HIVE`，Ubuntu 22.04，**NVIDIA TITAN RTX 24GB**（Turing sm_75）
> **项目路径**：`/home/cmu/Desktop/trac/`
> **原则：本手册所有命令都不需要 sudo / 密码**（venv+pip 装依赖、全部写在家目录、不碰系统 apt）。
> 唯一需要密码的情形（受保护目录拷贝 / 系统 apt）已全部绕开。

---

## 0. 服务器现状（2026-08-30 已核对）

| 项 | 值 | 备注 |
|---|---|---|
| GPU | TITAN RTX 24GB（cc 7,5） | ✓ |
| 驱动 / CUDA | 535.288.01 / 12.2 | ✓ |
| torch | **2.1.1+cu121，cuda_avail=True** | ✓ **已可用，不必重装** |
| Python | 3.10.12 | ✓ |
| 磁盘 | 5.2T 可用 | ✓ |
| 已装 | numpy/cv2/yaml/tqdm/scipy/PIL/matplotlib/pandas | 缺 sklearn 等（下方补装） |

---

## 1. 环境准备（全部 🔓，无需密码）

### 1.1 建 venv（用 `--system-site-packages` 继承现成的 torch 2.1.1）

```bash
cd /home/cmu
python3 -m venv --system-site-packages trac_venv
source trac_venv/bin/activate
python -m pip install --upgrade pip
```

### 1.2 补装缺的依赖

```bash
pip install scikit-learn albumentations timm tensorboardX yacs thop ptflops prefetch_generator imageio seaborn
# 确认 torch 与 torchvision 都能用
python -c "import torch, torchvision; print('torch', torch.__version__, '| tv', torchvision.__version__, '| cuda', torch.cuda.is_available(), '| cc', torch.cuda.get_device_capability())"
# 期望：torch 2.1.1+cu121 | cuda True | cc (7, 5)
```

> 之后每次新开终端先 `source /home/cmu/trac_venv/bin/activate`。

---

## 2. 项目与数据（全部 🔓）

```bash
mkdir -p /home/cmu/Desktop/trac
cd /home/cmu/Desktop/trac
# 把 trac/ 项目代码放进来（不含本地 gpu_env）
# 数据放到：
#   /home/cmu/Desktop/trac/data/bdd100k/
#     images/100k/{train,val}   images/10k/test
#     labels/bdd100k_labels_images_{train,val}.json
#     segments/masks/{train,val}   lanes/masks/{train,val}
#     splits/{tri_train,tri_val}.txt

# 校验数据
source /home/cmu/trac_venv/bin/activate
python scripts/verify_dataset.py
# 期望：images train=70000 val=10000；三任务交集 tri_train=69863 / tri_val=10000
```

---

## 3. 冒烟测试（确认脚本在服务器能跑，🔓）

```bash
cd /home/cmu/Desktop/trac && source /home/cmu/trac_venv/bin/activate
python training/train.py --config configs/train_stageA.yaml --num-images 64 --epochs 1 --outdir /home/cmu/Desktop/trac/experiments/smoke
# 应正常跑完，末尾打印 avg_loss 与 saved checkpoint
```

---

## 4. 正式实验（全部 🔓；outdir 都在项目内）

> TITAN RTX 24GB：建议把 config 里的 `batch_size` 提到 32（约快 2×）。
> 以下命令路径均已换算到 `/home/cmu/Desktop/trac`。

### Experiment A（等预算 Static vs Dynamic，3 seeds）
```bash
cd /home/cmu/Desktop/trac && source /home/cmu/trac_venv/bin/activate
EXP=/home/cmu/Desktop/trac/experiments/expA_equal_budget

# static_eb（独立训练 @0.45，等预算 ~0.90G）—— 3 seeds
for s in 0 1 2; do
  python training/train.py --config configs/train_stageA_eb.yaml --seed $s --outdir $EXP/static_eb_s$s
done
# dynamic（A6-B2-C2-D2）—— 3 seeds
for s in 0 1 2; do
  python training/train.py --config configs/train_stageA.yaml --seed $s --epochs 6 --outdir $EXP/dynA_s$s
  python training/train.py --config configs/train_stageB.yaml --seed $s --init $EXP/dynA_s$s/checkpoint.pt --epochs 2 --outdir $EXP/dynB_s$s
  python training/train.py --config configs/train_stageC.yaml --seed $s --init $EXP/dynB_s$s/checkpoint.pt --epochs 2 --outdir $EXP/dynC_s$s
  python training/train.py --config configs/train_stageD.yaml --seed $s --init $EXP/dynC_s$s/checkpoint.pt --epochs 2 --outdir $EXP/dynD_s$s
done
# 评测（每个 seed）
python evaluation/evaluate_baseline.py --baseline OursStatic --preset $EXP/static_eb_s0/checkpoint.pt --static-width 0.45 --outdir $EXP/static_eb_s0_eval
python evaluation/evaluate_baseline.py --baseline OursDynamic --preset $EXP/dynD_s0/checkpoint.pt --outdir $EXP/dynD_s0_eval --alloc-stats
# 汇总
python scripts/expA_analyze.py $EXP
```

### Experiment D（容量扫描 0.5/1.0/2.0M）
```bash
cd /home/cmu/Desktop/trac && source /home/cmu/trac_venv/bin/activate
EXP=/home/cmu/Desktop/trac/experiments/expD_compact_z
for t in 0.5M 1.0M 2.0M; do
  python training/train.py --config configs/train_stageA_$t.yaml --outdir $EXP/ours_$t
  python evaluation/evaluate_baseline.py --baseline OursStatic --preset $EXP/ours_$t/checkpoint.pt --outdir $EXP/ours_${t}_eval
done
python scripts/expD_analyze.py
```

### Experiment F/G（单 teacher KD，离线方案）
```bash
cd /home/cmu/Desktop/trac && source /home/cmu/trac_venv/bin/activate
EXP=/home/cmu/Desktop/trac/experiments/expF_single_teacher
CACHE=$EXP/kd_cache

# 1) 预计算 teacher 目标（TITAN RTX 可 batch 64；磁盘充足 5.2T，可全量 70k）
for t in YOLOP TwinLiteNetPlus TriLiteNet; do
  python scripts/precompute_teacher.py --teacher $t --num-images 70000 &
done
wait
# 2) 全量离线 KD 训练 + 评测
for t in YOLOP TwinLiteNetPlus TriLiteNet; do
  python training/train.py --config configs/train_stageA_offkd_$t.yaml --num-images 70000 --epochs 6 --outdir $EXP/kd_${t}_full
  python evaluation/evaluate_baseline.py --baseline OursStatic --preset $EXP/kd_${t}_full/checkpoint.pt --outdir $EXP/kd_${t}_full_eval
done
python scripts/expFG_analyze.py
```

### Experiment I（KD 后重测 Dynamic）
```bash
cd /home/cmu/Desktop/trac && source /home/cmu/trac_venv/bin/activate
EXP=/home/cmu/Desktop/trac/experiments/expI_kd_dynamic
KD=/home/cmu/Desktop/trac/experiments/expF_single_teacher/kd_YOLOP_full/checkpoint.pt
python training/train.py --config configs/train_stageB.yaml --init $KD --epochs 2 --outdir $EXP/kdD_B
python training/train.py --config configs/train_stageC.yaml --init $EXP/kdD_B/checkpoint.pt --epochs 2 --outdir $EXP/kdD_C
python training/train.py --config configs/train_stageD.yaml --init $EXP/kdD_C/checkpoint.pt --epochs 2 --outdir $EXP/kdD_D
python evaluation/evaluate_baseline.py --baseline OursDynamic --preset $EXP/kdD_D/checkpoint.pt --outdir $EXP/kdD_eval --alloc-stats
```

### 汇总图表
```bash
cd /home/cmu/Desktop/trac && source /home/cmu/trac_venv/bin/activate
python scripts/gen_figures.py
```

---

## 5. 当前可立即执行的顺序

1. 🔓 第 1 节：建 venv + 补依赖（一次性）
2. 🔓 第 2 节：拷项目 + 数据 + `verify_dataset.py`
3. 🔓 第 3 节：冒烟测试
4. 🔓 第 4 节：正式实验 A→D→F/G→I
5. 🔓 完成后把 `experiments/` 的 metrics.json / 报告拷回本地汇总

> 全程不需要密码。若某条 pip 在编译时缺系统库（少见），先 `pip install` 报错贴给我，
> 我会给**无需密码**的替代（如 `--no-build-isolation`、换 wheel、或把缺的 .so 拷到 venv）。

---

## 6. 需要注意的差异

| 项 | 本地 RTX 5060 | 服务器 TITAN RTX |
|---|---|---|
| 架构 | Blackwell sm_120 | Turing sm_75 |
| torch | 2.11+cu130 | **2.1.1+cu121（已可用，继承使用）** |
| 显存 | 8GB | **24GB（batch 32-64）** |
| Python | 3.12 | 3.10 |

> ⚠️ torch 2.1.1 vs 本地 2.11：state_dict 兼容（跨版本加载没问题）；
> 若有任何 API 差异导致报错，贴给我，我给兼容写法（多为主版本差异，影响很小）。
