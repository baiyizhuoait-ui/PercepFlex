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
cd /home/cmu/Desktop/trac
python3 scripts/verify_dataset.py
# 期望：images train=70000 val=10000；三任务交集 tri_train=69863 / tri_val=10000
```

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
