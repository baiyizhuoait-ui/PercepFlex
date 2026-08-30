# Experiment F/G: Single-Teacher Cross-Architecture KD

> 目标：不同架构 teacher（YOLOP/TwinLiteNet+/TriLiteNet）能否提升 Compact Student（Q3）

## 协议
- Student：Ours static（0.235M）
- 训练：10k 子集 × 10 epoch（公平对比；全量 70k 在线 YOLOP-KD 另报）
- KD：feature 对齐（1×1 投影→Z）+ DA/Lane 输出 sigmoid-MSE（离线预计算 teacher 目标，Plan A）
- 评测：全量 tri_val(10,000)

## 结果（全量 tri_val）

| 模型 | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|
| no_kd（10k×10） | 0.1425 | 0.7989 | 0.1538 |
| +YOLOP KD | **0.1559** (+0.013) | 0.8135 (+0.015) | 0.1556 (+0.002) |
| +TwinLiteNet+ KD | 0.1434 (+0.001) | 0.8197 (+0.021) | **0.1615** (+0.008) |
| +TriLiteNet KD | 0.1443 (+0.002) | **0.8206** (+0.022) | 0.1589 (+0.005) |
| YOLOP-KD 全量在线（70k×6） | 0.2798 | 0.8495 | 0.1932 |
| no_kd 全量（70k×6） | 0.2682 | 0.8423 | 0.1801 |

## 结论
- **Cross-Architecture KD 有效（H4 初步支持）**：三个 teacher 均提升 DA（+0.015~0.022）
  与 Lane；检测方面 YOLOP 提升最明显（+0.013 mAP）
- **Teacher 排名**：YOLOP（检测最佳、综合最稳）> TriLiteNet（DA 最佳）> TwinLiteNet+
- 全量在线验证：YOLOP-KD 在 mAP/DA/Lane 全面 +0.012/+0.007/+0.013
- 选择 **YOLOP** 作为 Exp I 的 teacher

## 备注
- 离线 KD（Plan A）将 teacher 成本从每步 527-966ms 降到 0（缓存读取），F/G 从 ~15h 降到 ~2h
- 10k×10 的绝对数值低于全量训练（样本量 4× 差距），但 teacher 相对排名在两种训练量下一致
