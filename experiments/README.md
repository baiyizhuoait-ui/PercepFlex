# 实验记录规范（对应任务书 §22）

每个实验在 `experiments/exp_XXX/` 下自动保存：

| 文件 | 说明 |
|------|------|
| `config.yaml` | 完整配置（模型结构、档位、loss 权重、数据、种子） |
| `metrics.json` | 统一指标（字段见下），训练结束后由 evaluation 管线写入 |
| `training_log.txt` | 训练日志（每 epoch 三任务指标 + loss） |
| `checkpoint.pt` | 最优权重 |
| `allocation_statistics.csv` | 每帧 Det/DA/Lane 档位分配（Router 行为） |

## metrics.json 必含字段

```json
{
  "experiment_id": "exp_001",
  "model": "ours_dynamic",
  "commit_hash": "<git rev-parse HEAD>",
  "dataset_version": "bdd100k-v1.0-split-yolop",
  "input_size": [640, 640],
  "parameters": 123456,
  "flops": 1234567890,
  "model_size_mb": 1.2,
  "average_latency_ms": 8.5,
  "p50_latency_ms": 8.2,
  "p95_latency_ms": 9.9,
  "fps": 117.6,
  "gpu_memory_mib": 210,
  "mAP_det": 0.512,
  "mIoU_da": 0.90,
  "IoU_lane": 0.28,
  "budget_distribution": {"tiny": 0.1, "medium": 0.3, "large": 0.6},
  "seed": 0
}
```

三条硬性规则：
1. **不得只报 Best** —— 每个实验记录 Best / Average / Worst（按任务指标或场景分桶）。
2. **不得用 composite score 掩盖单任务退化** —— mAP_det / mIoU_DA / IoU_lane 必须同时上报。
3. **不得虚构未测量数据** —— 复现不了的 baseline 在 README 标注 `Not reproduced`。

## 模板

`experiments/template/` 为新建实验的起始模板（从脚本复制即可）。

## 运行约定

每个实验可由单一命令复现（§25 Reproducibility），命令写入该实验的
`config.yaml` 的 `command` 字段或 `experiments/scripts/` 下对应脚本。
