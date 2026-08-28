# Experiment 1: Static vs Dynamic（全量 tri_val，10,000 张）

模型：Ours（0.235M），训练 A→B→C→D 后同一 checkpoint，仅分配方式不同。

| variant | avg width | FLOPs(G) | mAP50 | DA_mIoU | Lane_fgIoU | eval_fwd_ms | eval_FPS |
|---|---|---|---|---|---|---|---|
| static_tiny (0.25) | 0.25 | ~0.82 | 0.2336 | 0.8049 | 0.1249 | 7.75 | 129.1 |
| static_medium (0.5) | 0.50 | ~1.16 | 0.2541 | 0.8667 | 0.1942 | 7.67 | 130.4 |
| static_large (1.0) | 1.00 | ~1.57 | 0.2613 | 0.8537 | 0.1841 | 7.56 | 132.3 |
| dynamic | 0.389 | ~1.06 | 0.2427 | 0.8372 | 0.1666 | 7.35 | 136.0 |

## 结论（H1-H3 初步）

1. **Dynamic ≈ Static 插值**：dynamic (1.06G, 0.2427) 位于 static_tiny (0.82G, 0.2336) 与
   static_medium (1.16G, 0.2541) 之间——等平均 FLOPs 下 Dynamic 不显著损失性能 ✓（H3）
2. **Router 有任务感知**：DA 平均宽度 0.47 > det/lane 0.41，任务异构分配率 83.2% ✓（H1）
3. **FLOPs 节省真实**：dynamic 平均 1.06G vs static_large 1.57G（-32%）
4. ⚠️ **失败模式 D**：eager PyTorch 下 eval 循环延迟未随宽度下降（7.3-7.8ms 噪声内），
   微基准显示纯模型宽度收益存在（3.09→2.35ms）但被 Python/分发开销淹没；
   需 torch.compile / TensorRT / ONNX 后端兑现硬件延迟收益。已记录，非算法问题。
