# 服务器运行手册（Ubuntu + TITAN RTX）

> 更新：2026-08-30 ｜ 目标服务器：**Ubuntu + NVIDIA TITAN RTX（24GB，Turing sm_75）**
> ⚠️ 本手册把每一步明确标注是否需要密码：
> - 🔒 = **需要 sudo / 管理员密码**（你自己拿到密码后单独执行，或让有权限的人跑）
> - 🔓 = **不需要密码**（拿到 shell 后可直接执行；仅在你有权限的目录下操作）

---

## 0. 一次性要点

- **TITAN RTX 是 Turing 架构（sm_75）**，不是本机 RTX 5060（Blackwell sm_120）。
  本机 gpu_env 的 `torch 2.11+cu130` 是为 Blackwell 构建的，**不建议直接搬过去**。
  → 服务器上装 `torch` 的 **cu121 / cu124** 版 wheel（Turing 兼容），或先验证现有 torch。
- 你说大部分 CV 依赖已装好 → 先**检查**有什么（不花密码），缺的**优先用 venv+pip**（不花密码）装，
  尽量不碰系统级 apt（那是 🔒）。
- 24GB 显存 → 可跑更大 batch（建议 32，节省时间），且能跑**全量 70k 训练**。

---

## 1. 环境检查（全 🔓，无需密码）

拿到 shell 后，先跑这些**只读**命令，把输出发给我，我据此给你精确的下一步：

```bash
# 系统与硬件
uname -a
nvidia-smi                          # 看驱动版本、GPU、已用显存
python3 --version

# 已有的深度学习环境（有没有 torch？哪个 CUDA？）
python3 -c "import torch; print('torch', torch.__version__, '| cuda_avail', torch.cuda.is_available(), '| cc', torch.cuda.get_device_capability())" 2>&1 || echo "NO torch in default python"
which nvcc && nvcc --version || echo "NO system nvcc (正常，torch 自带 CUDA runtime 就不需要)"

# 常见 CV 依赖是否已装
python3 -c "import numpy, cv2, yaml, tqdm, scipy, sklearn, PIL, matplotlib, pandas; print('common deps OK')" 2>&1 || echo "SOME COMMON DEPS MISSING"

# 磁盘空间（训练需要 >100GB 数据+缓存）
df -h /home /tmp
```

> 🔓 如果上面显示 `torch` 已装且 `cuda_avail True`，跳过第 2 节的 torch 安装，直接看它版本是否兼容 Turing。

---

## 2. Python 环境（全 🔓，无需密码）

在你自己家目录下建 venv，**不碰系统**：

```bash
cd ~
python3 -m venv trac_venv                 # 不需要密码
source trac_venv/bin/activate             # 激活（之后都在这个环境里）
python -m pip install --upgrade pip
```

### 2a. 安装 torch（Turing 兼容）

> 若第 1 节已确认有可用 torch 且 cc==(7,5)，跳过。否则装 cu121 版（Turing 最稳）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 期望输出: 2.5.1+cu121 True (7, 5)
```

### 2b. 安装其余依赖（全 🔓）

```bash
pip install numpy opencv-python pyyaml tqdm pandas scipy scikit-learn pillow imageio \
            albumentations timm tensorboardX yacs thop ptflops matplotlib
```

---

## 3. 拷贝项目与数据（全 🔓，需你自己有文件/网络）

```bash
cd ~
mkdir -p trac
# 方式 A：从本机（Windows/WSL）拷（scp/sftp/网盘）
# 方式 B：如果有 git 仓库，直接 clone（把 trac/ 推到你自己的私有仓库后）
#   git clone <你的仓库> trac

# 数据（9.8GB）放到 trac/data/bdd100k/ 下，目录结构：
#   trac/data/bdd100k/images/100k/{train,val}   labels/*.json
#   segments/masks/{train,val}  lanes/masks/{train,val}  splits/*.txt

# 校验数据（🔓）
cd ~/trac && python scripts/verify_dataset.py
# 期望：三任务交集 tri_train=69863 / tri_val=10000
```

> 🔒 若需要把数据从 Windows 拷贝，且要用到挂载/权限设置，那部分可能需要密码；
> 单纯 `scp`/拷文件到你家目录不需要密码。

---

## 4. 运行实验（全 🔓，除非写系统路径）

> 所有 `--outdir` 都放 `~/trac/experiments/`（你家目录，不需密码）。

### 4.0 说明
- 24GB 显存可把各 config 的 `batch_size` 提到 32（训练约快 2×）。改 config 或命令行。
- 以下命令按任务书顺序 A→D→F/G→I。

### Experiment A（等预算 Static vs Dynamic，3 seeds）
```bash
cd ~/trac && source ~/trac_venv/bin/activate

# static_eb（独立训练 @0.45，等预算 ~0.90G）—— 3 seeds
for s in 0 1 2; do
  python training/train.py --config configs/train_stageA_eb.yaml --seed $s \
      --outdir experiments/expA_equal_budget/static_eb_s$s
done
# dynamic（A6-B2-C2-D2 管线）—— 3 seeds
for s in 0 1 2; do
  python training/train.py --config configs/train_stageA.yaml --seed $s --epochs 6 --outdir experiments/expA_equal_budget/dynA_s$s
  python training/train.py --config configs/train_stageB.yaml --seed $s --init experiments/expA_equal_budget/dynA_s$s/checkpoint.pt --epochs 2 --outdir experiments/expA_equal_budget/dynB_s$s
  python training/train.py --config configs/train_stageC.yaml --seed $s --init experiments/expA_equal_budget/dynB_s$s/checkpoint.pt --epochs 2 --outdir experiments/expA_equal_budget/dynC_s$s
  python training/train.py --config configs/train_stageD.yaml --seed $s --init experiments/expA_equal_budget/dynC_s$s/checkpoint.pt --epochs 2 --outdir experiments/expA_equal_budget/dynD_s$s
done
# 评测
python evaluation/evaluate_baseline.py --baseline OursStatic --preset experiments/expA_equal_budget/static_eb_s0/checkpoint.pt --static-width 0.45 --outdir experiments/expA_equal_budget/static_eb_s0_eval
python evaluation/evaluate_baseline.py --baseline OursDynamic --preset experiments/expA_equal_budget/dynD_s0/checkpoint.pt --outdir experiments/expA_equal_budget/dynD_s0_eval --alloc-stats
```

### Experiment D（容量扫描）
```bash
for t in 0.5M 1.0M 2.0M; do
  python training/train.py --config configs/train_stageA_$t.yaml --outdir experiments/expD_compact_z/ours_$t
  python evaluation/evaluate_baseline.py --baseline OursStatic --preset experiments/expD_compact_z/ours_$t/checkpoint.pt --outdir experiments/expD_compact_z/ours_${t}_eval
done
# 分析
python scripts/expD_analyze.py
```

### Experiment F/G（单 teacher KD，方案 A 离线 KD）
```bash
# 1) 预计算 teacher 目标（可选并行，TITAN RTX 24GB 可 batch 64）
for t in YOLOP TwinLiteNetPlus TriLiteNet; do
  python scripts/precompute_teacher.py --teacher $t --num-images 70000 &
done
wait
# 2) 全量离线 KD 训练（缓存就绪后，无 teacher 每步开销）
for t in YOLOP TwinLiteNetPlus TriLiteNet; do
  python training/train.py --config configs/train_stageA_offkd_$t.yaml --num-images 70000 --epochs 6 \
      --outdir experiments/expF_single_teacher/kd_${t}_full
  python evaluation/evaluate_baseline.py --baseline OursStatic --preset experiments/expF_single_teacher/kd_${t}_full/checkpoint.pt --outdir experiments/expF_single_teacher/kd_${t}_full_eval
done
python scripts/expFG_analyze.py
```
> 磁盘注意：70k 图的 teacher 缓存约 60-130GB/teacher（fp16 @1/16），三 teacher 共 ~300GB，请确认磁盘空间。
> 若磁盘紧张，可先用 `--num-images 30000` 筛选，最优 teacher 再全量。

### Experiment I（KD 后重测 Dynamic）
```bash
python training/train.py --config configs/train_stageB.yaml --init experiments/expF_single_teacher/kd_YOLOP_full/checkpoint.pt --epochs 2 --outdir experiments/expI_kd_dynamic/kdD_B
python training/train.py --config configs/train_stageC.yaml --init experiments/expI_kd_dynamic/kdD_B/checkpoint.pt --epochs 2 --outdir experiments/expI_kd_dynamic/kdD_C
python training/train.py --config configs/train_stageD.yaml --init experiments/expI_kd_dynamic/kdD_C/checkpoint.pt --epochs 2 --outdir experiments/expI_kd_dynamic/kdD_D
python evaluation/evaluate_baseline.py --baseline OursDynamic --preset experiments/expI_kd_dynamic/kdD_D/checkpoint.pt --outdir experiments/expI_kd_dynamic/kdD_eval --alloc-stats
```

---

## 5. 汇总与图表（全 🔓）
```bash
python scripts/expA_analyze.py experiments/expA_equal_budget
python scripts/expD_analyze.py
python scripts/expFG_analyze.py
python scripts/gen_figures.py
```

---

## 6. 可能需要密码的步骤汇总（🔒 集中列在这）

| 操作 | 为什么需要密码 | 替代方案（🔓） |
|---|---|---|
| `sudo apt-get install ...`（系统级依赖） | apt 需要 root | 尽量用 venv+pip，不装系统包 |
| 系统 CUDA Toolkit / 驱动升级 | 驱动/系统级 | **不要动**——torch cu121 wheel 自带 CUDA runtime，无需系统 nvcc |
| 写 `/root`、`/opt`、`/usr` 等目录 | 权限 | 全部改写到 `~/trac`、`~/trac_venv` |
| nvidia-container / docker GPU 挂载 | 若用 docker | 直接裸机跑 torch 即可，无需 docker |
| 从 Windows 拷数据到受保护目录 | 权限 | 拷到 `~/` 即可 |

> 总结：**真正需要密码的几乎只有「apt 装系统包」**，而本手册已用 venv+pip 绕开。
> 其余全部 🔓。若某个 pip 包需要系统编译（个别包），先 `pip install` 试，失败再考虑 apt（🔒）。

---

## 7. 与本地（RTX 5060）的差异注意

| 项 | 本地 5060 | 服务器 TITAN RTX |
|---|---|---|
| 架构 | Blackwell sm_120 | **Turing sm_75** |
| torch | 2.11+cu130 | 用 **cu121/cu124** 版 |
| 显存 | 8GB（batch 8-16） | **24GB（batch 32-64）** |
| 预期速度 | 全量 6-epoch ~50min/模型 | 快 ~2-3×，且可全量 70k |

> ⚠️ 服务器结果可能与本地（10k 筛选）绝对数值略有差异，以服务器全量为准（数据更大、训练更足）。

---

## 8. 当前可立即执行的顺序（拿到 shell 后）

1. 🔓 跑第 1 节**检查命令**，把输出发给我
2. 🔓 建 venv + 装依赖（第 2 节）
3. 🔓 拷数据 + `verify_dataset.py` 校验（第 3 节）
4. 🔓 按第 4 节跑实验（A→D→F/G→I）
5. 🔒 仅当遇到系统级报错才需要密码（apt）
