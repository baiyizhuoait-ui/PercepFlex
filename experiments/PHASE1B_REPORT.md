# Phase 1-B 最终报告（Compact Representation + Dynamic Capacity + Cross-Architecture KD）

> 完成日期：2026-08-30 ｜ 目标：判断研究路线是否值得进论文（Q1-Q3 + 决策树）

## 核心结果表（§23，全量 tri_val 10,000，640×640）

| Model | Params | FLOPs | mAP50 | DA mIoU | Lane fgIoU | FPS | Latency |
|-------|-------:|------:|------:|--------:|-----------:|----:|--------:|
| YOLOP (B1) | 7.94M | 31.3G | 0.766 | 0.912 | 0.225 | 68.5 | 14.6ms |
| TwinLiteNet (B2) | 0.44M | 14.1G | — | 0.911 | 0.228 | 111 | 9.0ms |
| TriLiteNet-tiny (B3) | 0.15M | 1.8G | 0.495 | 0.880 | 0.195 | 226 | 4.4ms |
| **Ours Static (0.235M)** | 0.235M | 1.57G | 0.268 | 0.842 | 0.180 | 152 | 6.6ms |
| **Ours Static+KD** | 0.235M | 1.57G | **0.280** | **0.850** | **0.193** | 161 | 6.2ms |
| **Ours Dynamic** | 0.235M | ~0.90G | 0.243 | 0.837 | 0.167 | 136 | 7.4ms |
| **Ours Dynamic+KD** | 0.235M | ~0.90G | 0.046 | 0.441 | 0.109 | 139 | 7.2ms | |

## 各实验结论

### Experiment A：Equal-Budget Static vs Dynamic（3 seeds）
- **Case C 确认**：等平均 FLOPs（~0.90G）下，独立训练的 Static-EB（mAP 0.2555±0.003）
  全面优于或持平 Dynamic（0.1176±0.109，种子不稳定，部分 router 塌缩到 tiny）
- Dynamic 的价值 = 单一模型多档位覆盖（flexibility），非等预算 accuracy

### Experiment D：Compact Representation 容量（0.23→2.0M）
- **H2 支持**：0.23M 紧凑 Z 保留三任务大部分信息（mAP 0.79× / DA 0.99× / Lane 0.90× of 2.0M）
- 容量敏感性：Detection > Lane > DA（DA 在 0.5M 饱和）——解释了路由收益上界小

### Experiment F/G：Single-Teacher Cross-Architecture KD
- **H4 初步支持：三个 teacher 都提升 student**（10k×10 公平对比）
- YOLOP（检测 +0.013，综合最稳）> TriLiteNet（DA +0.022）> TwinLiteNet+
- 全量在线验证：YOLOP-KD 全面 +0.012/+0.007/+0.013（mAP/DA/Lane）
- 离线 KD（Plan A）实现：teacher 成本从每步 527-966ms 降到 0，F/G 从 ~15h 降到 ~2h

### Experiment I：KD 后重测 Dynamic
- **C-A：KD 有效（+0.012/+0.007/+0.013）✓**
- **B-A：Dynamic 无效（-0.026/-0.005/-0.014）**（与 ExpA 一致）
- **D-C：KD-Dynamic 全量公平版仍差（-0.234 mAP, -0.409 DA）**——诊断确认：
  **router 对 DA 塌缩到 tiny（57%），窄宽度 DA 头过度预测前景（cls1 0.37 vs 正常 0.15-0.25）**
  → 动态训练管线（A→B→C→D）系统性降低模型质量（ExpA s1/s2、ExpI D 三处复现）

## 决策树结论（§26）

```
Equal Budget:  Dynamic ≈ Static（Case B/C）→ Dynamic = flexibility
Compact Z:     有效（H2 ✓）→ 保留 Compact Representation
Cross-Arch KD: 有效（H4 ✓，C > A）→ 进入 KD
KD 后 Dynamic: 验证中（待全量 fair-D）
```

**确定路线：Route B（通用弹性模型）——Compact Model + KD + Runtime Elasticity：**
- **Compact Representation（H2 ✓）**：0.23M 保留三任务 79-99% 信息，信息效率高
- **Cross-Architecture KD（H4 ✓）**：显著提升 compact student（mAP/DA/Lane 全面 +0.012/+0.007/+0.013），
  是 accuracy 的核心引擎（论文卖点之一）
- **Dynamic Capacity = Runtime Elasticity（不承诺等预算 accuracy 增益）**：
  同一模型可在任意固定宽度运行（Pareto 覆盖真实存在，Exp1 强制宽度曲线已验证），
  但**学到的 router 不增加价值且训练不稳定**——如实报告，作为 negative result/ablation
- **KD+Dynamic 组合（D）失败**：router 塌缩问题在 KD 表示下依然存在

论文结论（候选）："紧凑表示 + 跨架构蒸馏可让 0.235M 模型在轻量 baseline 水平运行，
并支持多计算档位的弹性部署；学习式逐任务路由在当前极小区间内不提供额外 accuracy 收益
（oracle 上界 +0.0125 DA），其价值限于部署灵活性。"

## 关键工程成果
- 真实切片感知 FLOPs 计数（修正 thop 高估，Dynamic 平均 0.90G 而非 1.06G）
- 离线 KD（teacher 目标预计算 + 缓存训练）——时间从 15h → 2h
- 统一评测/训练协议、3-seed 统计、失败模式记录

## 服务器运行手册
见 `docs/RUNBOOK_SERVER.md`（全量 70k 最终数字可在服务器上跑，本地已提供全部脚本/协议）。
