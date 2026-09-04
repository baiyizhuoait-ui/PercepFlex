# Phase 2-D · 暂停 / 恢复操作说明

**暂停时刻**：2026-09-04 16:13 设置 `experiments/phase2d/STOP`
**暂停点**：z=16 · 10ep 的 eval 完成后停止（不进入 z=16 · 20ep）
**当时状态**：训练进行到 ep9，checkpoint 仅存到 **epoch 8**

---

## 关机前请确认（重要）

runner 的 STOP 检查发生在**每个 cell 开始前**。所以设置 STOP 后：

1. z=16 · 10ep 的**训练会继续跑完**（ep9 → ep10）
2. 训练完成后**正常 eval**，写出 `expD_z16_e10_eval/metrics.json`
3. 写完 CSV 行后，才准备进入下一个 cell，此时检测到 STOP → 优雅退出

**干净关机时间点**：training_log 出现 `[stage A] ep 10 DONE` 且 eval 完成后
（预计 **16:24–16:25**）。

验证命令：

```bash
cd /home/mycode/ai_study/trac
tail -2 experiments/phase2d/expD_z16_e10/training_log.txt   # 应看到 "ep 10 DONE"
tail -4 experiments/phase2d/expD_runner.log                  # 应看到 "HALTED BY STOP"
cat experiments/phase2d/expD_budget.csv                      # 应有 1 行 z=16,10
```

如果提前关机，**不会丢训练数据**（checkpoint 已落盘），但会触发下面
「半成品 checkpoint」的处理流程。

---

## 提前关机的风险与处理（已自动防护）

`train.py` **每个 epoch 都覆盖保存** `checkpoint.pt`。若训练在 ep9 中断，
盘上留下的是 **epoch 8/9 的权重**，而原 runner 的 resume 逻辑只看
「有没有 checkpoint」，会**把一个 8-epoch 模型当成 10ep 结果去 eval 并写进 CSV**
—— 这会静默污染数据集。

**防护**：`scripts/phase2d_run_budget_v2.sh` 增加了 epoch 完整性校验：

```bash
CK_EP=$(python -c "...torch.load(CKPT).get('epoch')...")
if [ -f "$CKPT" ] && [ "$CK_EP" = "$EP" ]; then
    echo "[resume] ckpt epoch=${CK_EP} matches target — eval only"
else
    mv "$CKPT" "${CKPT}.partial_ep${CK_EP}"   # 移走半成品，重新训练
    ...retrain...
fi
```

**下次恢复请务必用 v2**，不要用 v1：

```bash
bash scripts/phase2d_run_budget_v2.sh
```

---

## 恢复步骤（开机后）

```bash
cd /home/mycode/ai_study/trac

# 1. 移除 STOP，否则 runner 会立刻退出
rm -f experiments/phase2d/STOP

# 2. 确认 z16·10ep 的 checkpoint 是完整的 epoch 10
python -c "import torch;print(torch.load('experiments/phase2d/expD_z16_e10/checkpoint.pt',map_location='cpu',weights_only=False)['epoch'])"
# 期望输出 10；若输出 8 或 9，说明当时提前关机了，v2 会自动重跑该 cell

# 3. 用 v2 重启（Windows 侧需用 run_in_background 托管 wsl 调用）
bash scripts/phase2d_run_budget_v2.sh
```

Windows 侧的正确启动方式（**会话常驻，避免被杀**）：

```
run_in_background=true:
wsl.exe -e bash -lc 'cd /home/mycode/ai_study/trac && bash scripts/phase2d_run_budget_v2.sh 2>&1 | tee experiments/phase2d/expD_foreground.log'
```

---

## 剩余 cell 与预估（从恢复时刻起）

| Cell | 预估训练 | 累计 |
|---|---|---|
| z=16 · 20ep | 132 min | 2.2 h |
| z=32 · 10ep | 73 min | 3.4 h |
| z=32 · 20ep | 146 min | 5.9 h |
| z=128 · 10ep | 114 min | 7.8 h |
| z=128 · 20ep | 228 min | **11.6 h** |

（z=16 · 10ep 已完成，resume 时自动跳过）

---

## 中间决策点

- **所有 10ep 完成时**（恢复后约 3.4 h）：可先看 4ep / 10ep 的趋势。
  若 10ep 下 z=16/32/128 仍完全重合且重合度与 4ep 相当，20ep 大概率也重合，
  可提前终止后面两个 20ep cell，省约 6 小时。
- **z=128 · 20ep 是最大单点**（3.8 h，占 33%）；若时间紧张，这是第一个可砍的。
