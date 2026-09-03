# PercepFlex Phase 2-B 操作手册（GPU 实验全流程）

> 目的：把「启动 / 监控 / 暂停 / 续跑 / 停止 / 提交」每一步都写成可直接复制的命令，
> 使实验控制不依赖 agent 的执行权限。
>
> 环境：WSL2 Ubuntu（host `OI`）· 项目根 `/home/mycode/ai_study/trac` ·
> Python `/home/mycode/ai_study/gpu_env/bin/python`（torch 2.11.0+cu130）·
> RTX 5060 Laptop 8GB · 数据 `data/bdd100k/`（tri_train 69863 / tri_val 10000）

---

## 〇、速查表

| 我想… | 跳到 |
|---|---|
| 确认环境能用 | [§一 环境自检](#一环境自检) |
| 启动 Z 扫描 | [§二 启动扫描](#二启动扫描后台) |
| 看现在跑到哪了 | [§三 监控进度](#三监控进度) |
| **优雅暂停**（跑完当前 Z 再停） | [§四 优雅暂停](#四优雅暂停推荐) |
| **立刻停止**（kill） | [§五 立即停止](#五立即停止kill) |
| 接着上次继续跑 | [§六 断点续跑](#六断点续跑) |
| 只补一个 Z 的评测 | [§七 补单个评测](#七补单个评测) |
| 启动 R1/R2 | [§八 R1R2 实验](#八-r1r2-实验task-2c) |
| 提交代码 | [§九 Git 提交](#九-git-提交) |
| 出问题了 | [§十 故障排查](#十故障排查) |

---

## 一、环境自检

每次开始干活前先跑这一条，一次确认 **WSL 执行 + GPU + Python + 项目路径** 四件事：

```bash
cd /home/mycode/ai_study/trac
echo "--- host/user ---"; hostname; whoami
echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "--- python/torch ---"
/home/mycode/ai_study/gpu_env/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
echo "--- data ---"; ls data/bdd100k/ 2>/dev/null | head
echo "--- git ---"; git log --oneline -1
```

**期望输出**：`gpu` 行显示 `0 MiB`（空闲）或你自己的作业占用；torch 行显示 `2.11.0 True`。

> ⚠️ **显存 > 7GB 时**：单卡 8GB 有 OOM 风险。要么等当前作业结束，要么把
> `configs/*.yaml` 里的 `train.batch_size: 16` 改成 `8`。

---

## 二、启动扫描（后台）

### 2.1 全量 Z 扫描（z=16…128，约 3.3h）

```bash
cd /home/mycode/ai_study/trac
mkdir -p experiments/phase2b
setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
  > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown
sleep 5
tail -5 experiments/phase2b/zsweep_stdout.log
```

### 2.2 只跑指定 Z（推荐先跑子集看曲线）

```bash
cd /home/mycode/ai_study/trac
ZS_OVERRIDE="32 48 64" setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
  > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown
```

> **必须用 `setsid nohup … & disown`**，普通 `&` 会随启动它的 shell 一起死掉。

---

## 三、监控进度

### 3.1 一眼看全（推荐，复制整段）

```bash
cd /home/mycode/ai_study/trac
echo "=== 当前在跑 ==="; tail -1 experiments/phase2b/zsweep_stdout.log
echo "=== 已完成数据点 ==="; column -s, -t experiments/phase2b/zsweep_results.csv
echo "=== 进程 ==="; pgrep -af "train.py|evaluate_baseline|run_zsweep" | grep -v pgrep || echo "无进程"
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### 3.2 持续跟踪（每 30s 刷新，Ctrl+C 退出）

```bash
cd /home/mycode/ai_study/trac
watch -n 30 'tail -1 experiments/phase2b/zsweep_stdout.log; echo "---"; tail -n +2 experiments/phase2b/zsweep_results.csv | wc -l | xargs echo "done points:"'
```

### 3.3 只看训练 loss

```bash
tail -f experiments/phase2b/zsweep_z64.log | grep "ep "
```

### 3.4 判断还剩多久

单个 Z 点 ≈ **28 min 训练 + 8 min 评测 ≈ 36 min**。
看日志里 `[stage A] ep N DONE` 的 N 可知当前 epoch（共 4 个）。

---

## 四、优雅暂停（推荐）

**原理**：runner 每开始一个 Z 点前检查 `experiments/phase2b/STOP` 文件是否存在，
存在则跑完当前 Z 点后干净退出。**不会丢当前 Z 点的训练结果。**

```bash
cd /home/mycode/ai_study/trac
touch experiments/phase2b/STOP
```

然后按 [§三](#三监控进度) 观察，直到：

```
[runner] STOP file detected — exiting after Z=64
```

进程退出后确认：

```bash
pgrep -af "run_zsweep|train.py" | grep -v pgrep || echo "已停止"
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # 应回到 ~0 MiB
```

**恢复前务必删掉 STOP 文件**，否则 restart 后立刻又退出：

```bash
rm -f experiments/phase2b/STOP
```

---

## 五、立即停止（kill）

顺序**不能反** —— 先杀 runner，否则它会立刻拉起下一个 Z 点。

```bash
cd /home/mycode/ai_study/trac
pkill -f "phase2b_run_zsweep"      # 1) 先停调度器
pkill -f "training/train.py"       # 2) 再停训练
pkill -f "evaluate_baseline.py"    # 3) 若此刻在评测
sleep 3
# 确认
pgrep -af "train.py|run_zsweep|evaluate_baseline" | grep -v pgrep || echo "已全部停止"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
```

> 用 `pkill`（默认 SIGTERM），**不要用 `-9`**。SIGTERM 让 torch 有机会写完 checkpoint，
> `-9` 可能留下大小正常但内容截断的权重文件。

### 停止后记录断点（建议）

```bash
cd /home/mycode/ai_study/trac
ls -la experiments/phase2b/zsweep_z*/
echo "rows: $(tail -n +2 experiments/phase2b/zsweep_results.csv | grep -c .)"
```

---

## 六、断点续跑

**直接重跑启动命令即可，v2 runner 会自动识别已完成的部分**：

| 目录状态 | runner 行为 |
|---|---|
| 已有 `*_eval/metrics.json` | 跳过（已完成） |
| 只有 `checkpoint.pt`、无 eval | **跳过训练，直接补 eval** |
| 目录不存在 | 完整训练 + 评测 |

```bash
cd /home/mycode/ai_study/trac
rm -f experiments/phase2b/STOP
setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
  >> experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown
```

> 用 `>>` 追加，别用 `>`，否则会覆盖历史日志。
> v2 对 CSV 也是追加模式，重启不会清空已有数据点（v1 有这个 bug，**不要再用 v1**）。

---

## 七、补单个评测

**场景**：训练完了但 eval 被中断（当前 z=16 就是这种情况，只需 8 min，不用重训 28 min）。

```bash
cd /home/mycode/ai_study/trac
Z=16
/home/mycode/ai_study/gpu_env/bin/python evaluation/evaluate_baseline.py \
  --baseline OursStatic \
  --preset experiments/phase2b/zsweep_z${Z}/checkpoint.pt \
  --outdir experiments/phase2b/zsweep_z${Z}_eval
```

后台版：

```bash
cd /home/mycode/ai_study/trac
setsid nohup /home/mycode/ai_study/gpu_env/bin/python evaluation/evaluate_baseline.py \
  --baseline OursStatic \
  --preset experiments/phase2b/zsweep_z16/checkpoint.pt \
  --outdir experiments/phase2b/zsweep_z16_eval \
  > experiments/phase2b/zsweep_z16_eval.log 2>&1 < /dev/null & disown
```

完成后核对：

```bash
cat experiments/phase2b/zsweep_z16_eval/metrics.json
```

---

## 八、 R1/R2 实验（Task 2(c)）

**前置**：先跑完 [§二](#二启动扫描后台) 的 Z 扫描，根据实测饱和点选 Z。
若已看过曲线，可用 `ZS="32 64 96"` 指定。

```bash
cd /home/mycode/ai_study/trac
ZS="32 64 96" setsid nohup bash scripts/phase2b_run_r1r2.sh \
  > experiments/phase2b/r1r2_stdout.log 2>&1 < /dev/null & disown
sleep 5; tail -3 experiments/phase2b/r1r2_stdout.log
```

停止 / 暂停方式与 Z 扫描相同（`pkill -f "phase2b_run_r1r2"`；该脚本同样支持 STOP 文件）。

---

## 九、 Git 提交

每次改动代码或产出新结果后提交，runner 会把 commit hash 记进 CSV，保证结果可追溯。

```bash
cd /home/mycode/ai_study/trac
export GIT_SSH_COMMAND="ssh -i /home/mycode/.ssh/id_ed25519_trac -o IdentitiesOnly=yes -o BatchMode=yes"

git add -A
git status --porcelain          # 确认将要提交的内容
git commit -m "Phase2-B: <简述改动>"
git log --oneline -1
git push origin main
```

**只提交特定文件**（避免把大权重文件提交进去）：

```bash
git add configs/phase2b_*.yaml models/representation/det_from_z.py \
        models/static_model.py scripts/phase2b_*.sh scripts/phase2b_*.py \
        docs/PHASE2B_*.md
git commit -m "Phase2-B: <简述改动>"
```

> `.gitignore` 应已排除 `*.pt` / `checkpoint.pt` / `data/`。若 `git status` 里出现
> 权重文件，先确认再提交。

---

## 十、故障排查

### 10.1 显存 OOM

**症状**：`CUDA out of memory`。

```bash
nvidia-smi                       # 看谁占着显存
pkill -f "training/train.py"     # 清场
# 然后改小 batch
sed -i 's/batch_size: 16/batch_size: 8/' configs/phase2b_zsweep_z*.yaml
```

### 10.2 想看完整报错

```bash
cd /home/mycode/ai_study/trac
grep -nE "Error|Traceback|error:|Exception" experiments/phase2b/*.log | tail -20
```

### 10.3 eval 报权重加载不匹配

**症状**：日志出现 `missing keys` / `unexpected keys` 非 0。

**原因**：eval 优先读 checkpoint 旁的 `config.yaml` 重建架构。若该文件缺失或与实际不符，
会回退到默认 `ours_static.yaml`（z_channels 就错了，整轮扫描作废）。

```bash
cat experiments/phase2b/zsweep_z16/config.yaml      # 确认 z_channels 正确
ls experiments/phase2b/zsweep_z16/                  # 确认 config.yaml 存在
```

### 10.4 想确认 runner 真的在跑

```bash
cd /home/mycode/ai_study/trac
ps -eo pid,etime,cmd | grep -E "run_zsweep|train\.py" | grep -v grep
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

若 GPU 利用率长期为 0% 但进程在，多半卡在 dataloader —— 检查 `num_workers`（配置里是 2）。

### 10.5 结果 CSV 被清空了

只会发生在使用 **v1** runner 时（它在启动时 `> "$CSV"` 截断文件）。
**请改用 v2**：`scripts/phase2b_run_zsweep_v2.sh`。

### 10.6 agent 说「wsl 被黑名单，无法执行」

这是**误判**，实测为间歇性拦截。重试顺序：

```bash
# 1) 完整路径重试
/c/Windows/System32/wsl.exe -e bash -lc 'echo OK'
# 2) 裸名重试
wsl -e bash -lc 'echo OK'
# 3) PowerShell
#    & "$env:SystemRoot\System32\wsl.exe" -e bash -lc 'echo OK'
```

详见 `~/.workbuddy/skills/wsl-exec-from-windows/SKILL.md`。

---

## 附录：关键路径与耗时

| 项目 | 值 |
|---|---|
| 项目根 | `/home/mycode/ai_study/trac` |
| Python | `/home/mycode/ai_study/gpu_env/bin/python` |
| 主日志 | `experiments/phase2b/zsweep_stdout.log` |
| 结果表 | `experiments/phase2b/zsweep_results.csv` |
| 权重 | `experiments/phase2b/zsweep_z*/checkpoint.pt`（每 epoch 存盘） |
| 评测结果 | `experiments/phase2b/zsweep_z*_eval/metrics.json` |
| STOP 文件 | `experiments/phase2b/STOP` |
| 单 Z 点耗时 | ≈28 min 训练 + ≈8 min 评测 ≈ **36 min** |
| 全量 6 点 | ≈ **3.3 h** |
| 固定 protocol | 4 ep / bs16 / AdamW lr1e-3 / cosine / seed 0 |

---

*最后更新：2026-09-03 21:25（WSL 执行能力已恢复并验证）*
