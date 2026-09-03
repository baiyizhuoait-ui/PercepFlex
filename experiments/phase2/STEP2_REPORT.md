# Phase 2 Step 2 — Single-Teacher KD 受控复核

## 目的
在固定 protocol 下干净地复核单 teacher KD 是否有增益。
修正 Phase1 混淆：Phase1 的 KD 用 10k 训练、no-KD 用 69k 全量 → 数据集大小不同。

## 协议（干净配对）
- 同一固定 9999-tri_train 子集（与现成 teacher cache 对齐）
- no-KD 对照与 3 个 KD 均在该子集、固定 4ep、seed0、bs16
- teacher：YOLOP / TwinLiteNetPlus / TriLiteNet（离线 cache，特征对齐 student Z + seg sigmoid-MSE）
- 评测：全量 tri_val（10k）

## 结果（全量 tri_val）
| 模型 | mAP50 | DA_mIoU | DA_fg | Lane_fg |
|---|---|---|---|---|
| s2_nokd (对照) | 0.0763 | 0.7770 | 0.6575 | 0.1352 |
| s2_kd_YOLOP | 0.0788 | 0.7678 | 0.6454 | 0.1372 |
| s2_kd_TLP | 0.0689 | 0.7634 | 0.6395 | 0.1330 |
| s2_kd_TriLite | 0.0636 | 0.7742 | 0.6536 | 0.1362 |

| KD | ΔmAP50 | ΔDA_mIoU | ΔLane_fg |
|---|---|---|---|
| +YOLOP | +0.0025 | −0.0092 | +0.0020 |
| +TLP | −0.0074 | −0.0136 | −0.0022 |
| +TriLite | −0.0127 | −0.0028 | +0.0010 |

## 结论（重要）
**单 teacher KD 在本受控小数据协议下无稳定增益，甚至略负。**
与 Phase1 声称的 +0.012 mAP 不符 → Phase1 那次的"KD 提升"部分被数据集大小混淆。

### 可能解释
1. **9999-subset 模型严重欠训练**（mAP 0.076 vs 全量 0.268）：数据稀缺时 feature-KD 强正则可能反而干扰 student 拟合。
2. **DA/Lane 早已饱和**（Phase1 H2：DA 0.5M 即饱和）：所以 DA/Lane 的 KD 无增益（delta≈0）一致。
3. **feature 对齐只作用于 student Z（喂 seg）**，而 det 读多尺度（不经 Z）→ 对 det 增益微弱。
4. Teacher 特征统计与极小 student 在数据饥渴下不对齐。

### 对 Phase-2 的启示（taskbook §28 允许"KD 无明显帮助"）
- 不能想当然假设 KD 是可靠模块；A4/A6/A8/A9 若建立在 KD 上需谨慎。
- 需区分"小数据欠训练" vs "全量"两个 regime。受控 9999 是干净但病态的 regime。
- **待办：全量数据受控 KD 复核**（需生成全量 69863 teacher cache 或在线 KD），用于调和此负结果与 Phase1 正结果——这是能否把 KD 当作 Phase-2 基石的判据。

## 数据/产物
- checkpoint: experiments/phase2/s2/{s2_nokd,s2_kd_YOLOP,s2_kd_TLP,s2_kd_TriLite}
- 评测: experiments/phase2/s2_eval/
