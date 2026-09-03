# Phase 2-B — Z-Sweep 断点记录（Breakpoint / Resume Guide）

> 用途：记录 Task 2(a) Z 容量扫描被暂停时的状态，以及以后如何恢复。
> 暂停请求时间：2026-09-03 17:06（用户要求）
> 记录撰写时间：2026-09-03 17:08

---

## 一、被暂停时的状态快照

| 项目 | 状态 |
|---|---|
| 正在运行 | `scripts/phase2b_run_zsweep.sh`（v1，串行） |
| 当前 Z 点 | **z=16**，ep4 / step ~1880 / 4366，ETA ≈3.8 min 到本轮结束 |
| 训练损失 | total 0.248（det 0.112 / da 0.0725 / lane 0.0635），稳定下降 |
| GPU | RTX 5060，2400 / 8151 MiB，util ~85% |
| 已落盘权重 | `experiments/phase2b/zsweep_z16/checkpoint.pt`（2.4 MB，**每 epoch 保存一次**） |
| 已完成 Z 点 | **0 个**（`zsweep_results.csv` 只有表头，z=16 的 eval 尚未执行） |
| 剩余 | z=16 收尾 → eval → z=32/48/64/96/128，原估 ≈3.3 h |

### 关键结论：暂停代价很小

`train.py` **每个 epoch 都会写 checkpoint**，因此 z=16 已训练出的权重不会丢。
最坏情况只是丢掉「z=16 最后一个 epoch 之后的部分」和「z=16 的 eval」。
z=16 若被中断在 ep4 中途，保存下来的实际是 **ep3 结束时的权重（3ep）**，
与固定 protocol 的 4ep 不一致 —— 恢复时应按下方 §三.2 处理。

---

## 二、如何停止（在 WSL 终端内执行）

**顺序很重要**：先停 runner，否则它会立刻拉起下一个 Z。

```bash
cd /home/mycode/ai_study/trac

# 1) 先停 runner，防止它启动下一个 Z
pkill -f "phase2b_run_zsweep.sh"
# 2) 再停训练进程
pkill -f "training/train.py"
# 3) 若此刻正在 eval，也停掉
pkill -f "evaluate_baseline.py"
```

确认已停干净：

```bash
pgrep -af "train.py|run_zsweep|evaluate_baseline"   # 应无输出
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader   # 显存应回落到 ~300MiB
```

若 `pkill` 后仍有残留（少见），用 PID 精确终止：

```bash
pgrep -af "training/train.py"      # 记下 PID
kill -TERM <PID>                    # 先礼貌终止，给 torch 收尾机会
# 等待 30s 仍未退出才用：kill -9 <PID>
```

> 不要用 `kill -9` 作为首选 —— 可能留下写了一半的 `checkpoint.pt`（2.4 MB 文件存在但内容损坏）。

停止后建议**重命名**中断点的权重，避免恢复时误当作完整 4ep 结果：

```bash
cd /home/mycode/ai_study/trac/experiments/phase2b
ls -la zsweep_z16/checkpoint.pt
# 若你是在 ep4 完成前停的：
mv zsweep_z16/checkpoint.pt zsweep_z16/checkpoint_ep3_INTERRUPTED.pt
```

---

## 三、如何恢复

### 3.1 使用新的 v2 runner（已带优雅停止 + 断点续跑）

`scripts/phase2b_run_zsweep_v2.sh` 相比 v1 增加：

1. **STOP 文件**：`touch experiments/phase2b/STOP` → runner 在当前 Z 点结束后优雅退出；`rm` 掉即可继续。
2. **断点续跑**：某 Z 点若已存在 `zsweep_z<N>_eval/metrics.json` 则整体跳过；
   若只有 `checkpoint.pt` 而无 eval 结果，则跳过训练、直接补 eval。
3. **Z 列表覆盖**：`ZS_OVERRIDE="32 48 64" bash scripts/phase2b_run_zsweep_v2.sh`
4. **CSV 追加而非清空**（v1 启动时 `> "$CSV"` 会清空，重启即毁掉部分结果 —— 这是 v1 的缺陷）。

### 3.2 恢复命令

```bash
cd /home/mycode/ai_study/trac
export GIT_SSH_COMMAND="ssh -i /home/mycode/.ssh/id_ed25519_trac -o IdentitiesOnly=yes -o BatchMode=yes"

# 情况 A：z=16 的 4ep checkpoint 已完整（ep4 写完）→ 直接续跑，会自动补 eval 再继续
setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
  > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown

# 情况 B：z=16 被中断在 ep3（权重已按要求改名）→ 重训 z=16 并跑完整条链
rm -rf experiments/phase2b/zsweep_z16
setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
  > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown

# 情况 C：只想先补跑几个关键 Z 点（等 Task 2(a) 曲线出来再定 R1/R2 的采样点）
ZS_OVERRIDE="16 32 64" setsid nohup bash scripts/phase2b_run_zsweep_v2.sh \
  > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown
```

> 恢复前请先确认没有旧进程残留（见 §二），单卡 8 GB 不要并行两个训练。

### 3.3 以后想中途暂停

```bash
touch experiments/phase2b/STOP    # 当前 Z 点跑完后停下
rm    experiments/phase2b/STOP    # 解除
```

---

## 四、⚠️ 不要做的事

- **不要在 v1 runner 运行期间编辑 `scripts/phase2b_run_zsweep.sh`。**
  bash 已把 `for` 循环体解析进内存，就地改写文件会让它下次按原 offset 读取到错位字节，
  可能执行出不可预期的内容。要改就改 v2，或用 v2 重新启动。
- 不要在恢复时复用 v1 —— 它会先清空 `zsweep_results.csv`。

---

## 五、未提交的改动（恢复后需补 commit）

以下文件在暂停时**只写入了磁盘、尚未 git commit**（当时 agent 侧 WSL 调用被临时拦截，
现已恢复，见 §六）：

- `scripts/phase2b_run_zsweep_v2.sh`（新建）
- `docs/PHASE2B_SWEEP_BREAKPOINT.md`（本文件）

已提交的：`docs/PHASE2B_POSITIONING.md`（`53a94c7`）。

补提交命令：

```bash
cd /home/mycode/ai_study/trac
export GIT_SSH_COMMAND="ssh -i /home/mycode/.ssh/id_ed25519_trac -o IdentitiesOnly=yes -o BatchMode=yes"
git add scripts/phase2b_run_zsweep_v2.sh docs/PHASE2B_SWEEP_BREAKPOINT.md
git commit -m "Phase2-B: resumable Z-sweep runner (v2) + breakpoint record

v2 adds STOP-file graceful halt, per-Z resume (skip finished / eval-only when
checkpoint exists), ZS_OVERRIDE for partial sweeps, and appends to the CSV
instead of truncating it (v1 wiped partial results on restart)."
```

---

## 六、状态更新与事实修正（2026-09-03 21:25）

### 事实修正

本文档初版写有「agent 已失去 WSL 执行能力」——**该结论错误，现予撤销**。

实测时间线（四种调用形式）：

| 时间 | 裸 `wsl` | System32 完整路径 | PowerShell |
|---|---|---|---|
| 16:15 | 拦截 | **可用** | — |
| 17:06 | 拦截 | 拦截 | 拦截 |
| **21:25** | **可用** | **可用** | **可用** |

17:06 的拦截是**瞬时的**，21:25 未做任何配置变更即自动恢复。教训：单次探测失败
不足以得出「永久不可用」的结论，每轮开始应重新探测。

### 暂停已生效（状态核对于 21:25）

- 残留进程：**无**（`train.py` / `run_zsweep` / `evaluate_baseline` 均无）
- GPU：**0 MiB / 8151 MiB, 0%**（已完全释放）
- `experiments/phase2b/zsweep_z16/checkpoint.pt` — 2,494,368 B，**完整 4 epoch**，mtime 17:12
- `zsweep_z16_eval.log` — **0 字节**，即 eval 在启动瞬间被中断，z=16 无评测数据
- `zsweep_results.csv` — 仅表头，**0 行有效数据**
- `zsweep_runner.log` 末行：`[2026-09-03 17:12:09] Z=16 EVAL`

### 结论

- 暂停目标达成，**未丢失任何已训练权重**。
- z=16 训练完成但**未评测**；恢复后第一个动作应是补 z=16 的 eval（约 8 min），
  而非重训（约 28 min）。v2 runner 的断点续跑逻辑会自动识别这一情况。
- z=32/48/64/96/128 均未启动。
