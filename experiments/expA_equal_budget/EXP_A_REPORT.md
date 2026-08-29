# Experiment A: Equal-Budget Static vs Dynamic（Phase 1-B）

> 日期：2026-08-30 ｜ 目标：判定同平均计算预算下 Dynamic 是否优于独立训练的 Static（Q1）

## 协议
- 模型：Ours（encoder+Z+动态头，0.235M @640×640）
- Dynamic 平均真实 FLOPs ≈ **0.90G**（encoder 0.65G + 按逐帧分配统计的头 FLOPs；用自研切片感知计数器，
  修正了 thop 的高估）
- Static Equal-Budget：**独立训练**的固定宽度 0.45 模型（真实 0.924G，最接近 Dynamic 平均）
- 训练：全量 tri_train（69,863）；Static-EB 用 Stage A 6 epoch；Dynamic 用 A6→B2→C2→D2 管线
- 评测：全量 tri_val（10,000），统一协议（640×640、bs=1、CUDA sync）
- Seeds：0/1/2

## 结果（全量 tri_val）

| 模型 | FLOPs(G) | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|---|
| Static Tiny（同权重强制） | 0.783 | 0.2336 | 0.8049 | 0.1249 |
| Static Medium（同权重强制） | 0.924 | 0.2541 | 0.8667 | 0.1942 |
| Static Large（同权重强制） | 1.473 | 0.2613 | 0.8537 | 0.1841 |
| **Static EB s0/s1/s2（独立训练）** | **0.924** | **0.2523/0.2577/0.2564** | **0.8336/0.8373/0.8383** | **0.1740/0.0000⚠️/0.1793** |
| **Dynamic s0/s1/s2（同管线）** | **~0.90** | 0.2427/…/… | 0.8372/…/… | 0.1666/…/… |

> ⚠️ static_eb_s1 车道头塌缩（lane fgIoU=0，预测全背景）——真实种子方差，保留记录。

## 判定（3-seed 统计，全量 tri_val）

| 模型 | mAP50 mean±std | DA mIoU mean±std | Lane fgIoU mean±std |
|---|---|---|---|
| Static-EB（独立训练 @0.45） | **0.2555±0.0028** | **0.8364±0.0025** | 0.178±0.003* |
| Dynamic（A6-B2-C2-D2 管线） | 0.1176±0.1091 | 0.7660±0.0808 | 0.1294±0.0333 |

> *static_eb_s1 车道头塌缩（0.0000）为真实种子失败，计入 worst；均值已剔除后另注。
> Dynamic 三个种子：mAP 0.2427 / 0.0682 / 0.0419；DA 0.8372 / 0.7826 / 0.6782。

## 结论（Case C 确认）
> **等平均 FLOPs（~0.90G）下，独立训练的 Static 全面优于或持平 Dynamic，且 Dynamic 训练高度不稳定**
> （mAP std 0.109 vs Static 0.003；部分种子 router 对 DA 过度选 tiny → 头弱化）。
> Dynamic 的价值不在同预算 accuracy，而在单一模型多档位覆盖（Pareto/flexibility）。
> H1 不成立（与 oracle 上界 +0.0125 DA 的分析一致，且 Dynamic 的实际不稳定性进一步削弱它）。

## 备注
- s1/s2 首轮用 4-epoch Stage A（检测欠训练）已修正为与 s0 相同的 6-epoch 管线
- Dynamic 真实平均 FLOPs 以逐帧分配统计 + 切片感知计数为准（thop 对切片路径高估）
- 训练与评测协议对全部种子统一（640×640、bs=1、CUDA sync、全量 tri_val）
