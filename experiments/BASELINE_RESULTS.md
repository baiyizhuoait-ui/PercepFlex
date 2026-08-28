# Baseline 统一评测结果（Phase 1 Step 3）

> 协议：BDD100K val（tri_val，10,000 张），640×640 letterbox（seg 指标裁剪 pad），
> 检测为单类车辆（car/bus/truck/train 合并，与 YOLOP/TriLiteNet 官方权重一致），
> mAP@0.5 / mAP@0.5:0.95，NMS conf=0.001/iou=0.6；DA/Lane 为像素级 Acc / fg IoU / 2 类 mIoU。
> 归一化按各模型官方（YOLOP/TriLiteNet: ImageNet；TwinLiteNet 系: /255）。
> 硬件：RTX 5060 Laptop 8GB，FP32，bs=1，CUDA 同步。

## 已完成

| 模型 | Params | FLOPs@640 | 平均延迟 | FPS | mAP50 | mAP50-95 | DA fgIoU | DA mIoU | Lane fgIoU | Lane mIoU | Lane lineAcc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOP (B1) | 7.94M | 31.3G | 11.9ms | 84.1 | **0.7657** | 0.4406 | 0.8556 | **0.9115** | 0.2247 | 0.6038 | 0.7896 |
| TwinLiteNet (B2) | 0.44M | 14.1G | 7.6ms | 130.9 | — | — | 0.8551 | **0.9114** | 0.2281 | 0.6077 | 0.7203 |
| TwinLiteNetPlus nano | ...运行中 | | | | | | | | | | |
| TriLiteNet tiny/small/base | ...运行中 | | | | | | | | | | |

## 论文对照（YOLOP 官方 README）

- YOLOP：Detection mAP50 76.5%、DA mIOU 91.5% → **我们的复现 76.57 / 91.15，一致** ✓
- TwinLiteNet：DA mIoU 91.5 → 91.14 ✓（Lane 指标口径略异）

## 备注

- 检测为单类车辆（car+bus+truck+train），与 YOLOP/TriLiteNet 官方权重训练协议一致；
  若需 8 类官方协议对比，可用 `det_categories` 扩展（后续 Equal-Budget 实验再定）。
- 延迟为纯前向（含 CUDA 同步，不含 NMS）；FP32。
