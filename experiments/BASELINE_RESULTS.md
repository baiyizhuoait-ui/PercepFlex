# Baseline 统一评测结果（Phase 1 Step 3）

> 协议：BDD100K val（tri_val，10,000 张），640×640 letterbox（seg 指标裁剪 pad），
> 检测为单类车辆（car/bus/truck/train 合并，与 YOLOP/TriLiteNet 官方权重一致），
> mAP@0.5 / mAP@0.5:0.95，NMS conf=0.001/iou=0.6；DA/Lane 为像素级 Acc / fg IoU / 2 类 mIoU。
> 归一化按各模型官方（YOLOP/TriLiteNet: ImageNet；TwinLiteNet 系: /255）。
> 硬件：RTX 5060 Laptop 8GB，FP32，bs=1，CUDA 同步。

## 已完成（全量 tri_val 10,000 张，2026-08-29 01:31）

| 模型 | Params | FLOPs@640 | 延迟 | FPS | mAP50 | mAP50-95 | DA fgIoU | DA mIoU | Lane fgIoU | Lane mIoU | Lane lineAcc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOP (B1) | 7.94M | 31.3G | 14.6ms | 68.5 | **0.7657** | 0.4406 | 0.8556 | **0.9115** | 0.2247 | 0.6038 | 0.7896 |
| TwinLiteNet (B2) | 0.44M | 14.1G | 9.0ms | 111.4 | — | — | 0.8551 | **0.9114** | 0.2281 | 0.6077 | 0.7203 |
| TwinLiteNetPlus nano | 0.033M | 1.9G | 5.2ms | 191.4 | — | — | 0.7776 | 0.8634 | 0.1859 | 0.5866 | 0.6710 |
| TwinLiteNetPlus small | 0.122M | 4.7G | 6.1ms | 163.8 | — | — | 0.8336 | 0.8986 | 0.2165 | 0.6019 | 0.7062 |
| TwinLiteNetPlus medium | 0.479M | 15.4G | 9.3ms | 107.3 | — | — | 0.8672 | 0.9191 | 0.2297 | 0.6087 | 0.7175 |
| TwinLiteNetPlus large | 1.944M | 58.6G | 15.0ms | 66.8 | — | — | 0.8815 | 0.9279 | 0.2450 | 0.6161 | 0.7448 |
| TriLiteNet tiny | 0.151M | 1.8G | 4.4ms | 226.4 | 0.4953 | 0.2023 | 0.8045 | 0.8796 | 0.1952 | 0.5914 | 0.6790 |
| TriLiteNet small | 0.592M | 6.6G | 6.8ms | 146.2 | 0.6326 | 0.3131 | 0.8458 | 0.9053 | 0.2198 | 0.6037 | 0.7064 |
| TriLiteNet base | 2.350M | 25.4G | 9.4ms | 106.5 | 0.7235 | 0.3970 | 0.8697 | 0.9203 | 0.2371 | 0.6123 | 0.7301 |

> 注：延迟为 FP32 纯前向（CUDA 同步，不含 NMS/预处理）。TwinLiteNet/TLP 无检测头（mAP 为 —）。

## 论文对照（YOLOP 官方 README）

- YOLOP：Detection mAP50 76.5%、DA mIOU 91.5% → **我们的复现 76.57 / 91.15，一致** ✓
- TwinLiteNet：DA mIoU 91.5 → 91.14 ✓（Lane 指标口径略异）

## 备注

- 检测为单类车辆（car+bus+truck+train），与 YOLOP/TriLiteNet 官方权重训练协议一致；
  若需 8 类官方协议对比，可用 `det_categories` 扩展（后续 Equal-Budget 实验再定）。
- 延迟为纯前向（含 CUDA 同步，不含 NMS）；FP32。
