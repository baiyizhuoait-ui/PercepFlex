# 服务器运行手册（Phase 1-B 全量复现）

> 目的：在更大的 GPU（A100/H100 等）上跑全量 70k 训练的最终数字。
> 本地 8GB 笔记本已完成全部实验与脚本验证；以下步骤在服务器上直接可用。

## 1. 环境准备

```bash
# 1) 拷贝项目与数据
# 项目代码（不含 gpu_env / experiments 大文件）：
#   <workspace>/trac/
# 数据（9.8GB）：
#   <workspace>/trac/data/bdd100k/   （images/100k、labels、segments/masks、lanes/masks、splits/）

# 2) Python 环境（服务器上）
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # 按服务器 CUDA 版本
pip install numpy opencv-python pyyaml tqdm pandas scipy scikit-learn pillow \
            imageio albumentations timm tensorboardX yacs thop ptflops
```

## 2. 数据校验（可选但推荐）

```bash
python scripts/verify_dataset.py          # 三任务交集 = 69,863 train / 10,000 val
```

## 3. 运行顺序（对应任务书 A→D→F/G→I）

### Experiment A（等预算 Static vs Dynamic，3 seeds）
```bash
# static_eb（独立训练 @0.45，等预算 ~0.90G）
python training/train.py --config configs/train_stageA_eb.yaml --seed {0,1,2} \
    --outdir experiments/expA_equal_budget/static_eb_s{0,1,2}
# dynamic（A6-B2-C2-D2 管线）
python training/train.py --config configs/train_stageA.yaml --seed {0,1,2} --epochs 6 --outdir .../dynA_s{0,1,2}
python training/train.py --config configs/train_stageB.yaml --seed {0,1,2} --init .../dynA_s{0,1,2}/checkpoint.pt --epochs 2 --outdir .../dynB_s{0,1,2}
python training/train.py --config configs/train_stageC.yaml --seed {0,1,2} --init .../dynB_s{0,1,2}/checkpoint.pt --epochs 2 --outdir .../dynC_s{0,1,2}
python training/train.py --config configs/train_stageD.yaml --seed {0,1,2} --init .../dynC_s{0,1,2}/checkpoint.pt --epochs 2 --outdir .../dynD_s{0,1,2}
# 评测
python evaluation/evaluate_baseline.py --baseline OursStatic --preset <ckpt> --static-width 0.45 --outdir <out>
python evaluation/evaluate_baseline.py --baseline OursDynamic --preset <ckpt> --outdir <out> --alloc-stats
```

### Experiment D（容量扫描 0.5/1.0/2.0M）
```bash
python training/train.py --config configs/train_stageA_{0.5M,1.0M,2.0M}.yaml --outdir experiments/expD_compact_z/ours_{0.5M,1.0M,2.0M}
# 评测：evaluate_baseline 会自动读 checkpoint 旁的 config.yaml（架构匹配）
```

### Experiment F/G（单 teacher KD）
```bash
# 方案 A（离线 KD，推荐）：先预计算 teacher 目标（可并行）
python scripts/precompute_teacher.py --teacher {YOLOP,TwinLiteNetPlus,TriLiteNet} --num-images 70000
python training/train.py --config configs/train_stageA_offkd_{YOLOP,TwinLiteNetPlus,TriLiteNet}.yaml \
    --num-images 70000 --epochs 6 --outdir experiments/expF_single_teacher/kd_{yolop,twinlite,trilite}_full
# 评测同上（OursStatic）
```
> 说明：`--num-images 70000` = 全量；缓存空间按 teacher 特征约 0.9-1.8 MB/图（fp16 @1/16），
> 70k 约 60-130GB/teacher，服务器磁盘需预留。

### Experiment I（KD 后重测 Dynamic）
```bash
# 用全量 KD student 初始化 dynamic 管线
python training/train.py --config configs/train_stageB.yaml --init <kd_student_ckpt> --epochs 2 --outdir .../kdD_B
python training/train.py --config configs/train_stageC.yaml --init .../kdD_B/checkpoint.pt --epochs 2 --outdir .../kdD_C
python training/train.py --config configs/train_stageD.yaml --init .../kdD_C/checkpoint.pt --epochs 2 --outdir .../kdD_D
```

## 4. 分析 / 图表 / 报告

```bash
python scripts/expA_analyze.py experiments/expA_equal_budget   # A 汇总+种子统计
python scripts/expD_analyze.py                                  # D 容量分析
python scripts/expFG_analyze.py                                 # F/G teacher 对比
python scripts/gen_figures.py                                   # Figure 1/3（其余按需扩展）
```

## 5. 预期结果参考（本地 8GB 笔记本，10k 筛选 / 全量混合）

| 实验 | 本地结论 |
|---|---|
| A | Static-EB ≥ Dynamic（等 0.90G）；Dynamic 种子不稳定 |
| D | 0.23M 保留 mAP 0.79×/DA 0.99×/Lane 0.90× of 2.0M |
| F/G | 三 teacher 均提升；YOLOP-KD 全量 +0.012/+0.007/+0.013 |
| I | KD 有效（C>A）；Dynamic 组合失败（router 塌缩，D<<C） |

> 服务器上的全量数字可能因训练量（70k vs 10k）与 seed 略有不同，以服务器结果为准。
