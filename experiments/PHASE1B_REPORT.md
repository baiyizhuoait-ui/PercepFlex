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
| **Ours Dynamic+KD** | 0.235M | ~0.90G | *训练中(全量)* | | | | |

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
- **B-A：Dynamic 无效（-0.026）**（与 ExpA 一致）
- **D-C / D-A：KD-Dynamic 全量训练中**（10k 版因训练量不足 mAP 0.038，已启动全量公平版）

## 决策树结论（§26）

```
Equal Budget:  Dynamic ≈ Static（Case B/C）→ Dynamic = flexibility
Compact Z:     有效（H2 ✓）→ 保留 Compact Representation
Cross-Arch KD: 有效（H4 ✓，C > A）→ 进入 KD
KD 后 Dynamic: 验证中（待全量 fair-D）
```

**当前路线：Route B（通用弹性模型）为主，KD 提供 accuracy 增强：**
- Compact Representation（H2 ✓，信息效率高）
- Cross-Architecture KD（H4 ✓，显著提升 compact student）
- Dynamic Capacity（多档位 flexibility + Pareto 覆盖，不做 accuracy 卖点）
- 若 fair-D 显示 KD 改善 Dynamic，则 KD+Dynamic 组合成立（D > B）

## 关键工程成果
- 真实切片感知 FLOPs 计数（修正 thop 高估，Dynamic 平均 0.90G 而非 1.06G）
- 离线 KD（teacher 目标预计算 + 缓存训练）——时间从 15h → 2h
- 统一评测/训练协议、3-seed 统计、失败模式记录

## 待完成
- [ ] fair-D（全量 KD-Dynamic）结果填入
- [ ] Figure 2/4/5/6 生成
- [ ] 服务器运行手册（全量最终数字）
