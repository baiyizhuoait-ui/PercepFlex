# Experiment D: Compact Representation 等预算实验

> 目标：验证 Compact Z 在极小参数预算下是否保留三任务信息（Q2）

## 协议
- 四个容量档：0.235M(1.57G) / 0.424M(2.46G) / 0.959M(4.22G) / 1.980M(8.83G)
- 同一结构（LightEncoder+CompactZ+三头），只缩放 encoder/Z 容量
- 同一训练管线（Stage A，epochs 按算力匹配：4/4/3/2）、同一数据（tri_train 全量）、同一评测（tri_val 全量）
- 0.23M 为 ExpA 的 static_eb_s0（6 epoch）；0.5/1.0/2.0M 为本实验新训

## 结果（全量 tri_val）

| 模型 | Params | FLOPs | mAP50 | DA mIoU | Lane fgIoU |
|---|---|---|---|---|---|
| ours_0.23M | 0.235M | 1.57G | 0.2523 | 0.8336 | 0.1740 |
| ours_0.5M | 0.424M | 2.46G | 0.2819 | 0.8398 | 0.1931 |
| ours_1.0M | 0.959M | 4.22G | 0.2951 | 0.8518 | 0.1943 |
| ours_2.0M | 1.980M | 8.83G | 0.3187 | 0.8450 | 0.1939 |

## 结论（D3-1/2/3）

- **D3-1：Detection 最先且最严重退化**（0.23M 保留 79% mAP vs 2.0M）；
  DA 几乎不退化（99%），Lane 中等（90%）
- **D3-2：任务容量敏感性：Detection > Lane > DA**（DA 在 0.5M 即饱和，Lane 在 0.5M 饱和，
  Detection 持续受益于容量）
- **D3-3：是——0.23M 紧凑 Z 保留三任务大部分信息**（mAP 0.79× / DA 0.99× / Lane 0.90×），
  且绝对性能合理（mAP 0.25 / DA 0.83 / Lane 0.17）
- 含义：**Compact Representation 的信息效率高（H2 支持）**；瓶颈在检测任务对容量的需求，
  而非表示本身。这也解释了为什么 Dynamic 路由的收益上界小——分割任务对宽度不敏感。

## 备注
- 0.23M 的 params 在分析脚本中因无 config.yaml 显示 0（实际 0.235M）
- 训练算力按容量匹配（大模型少 epoch），相对比较公平
