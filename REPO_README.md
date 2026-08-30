# PercepFlex（原 trac/）

Task-Resource Adaptive Compact Representation for Lightweight Multi-Task Driving Perception
（Detection + Drivable Area + Lane）

> 完整实验报告：`experiments/COMPREHENSIVE_REPORT.md`、`experiments/PHASE1B_REPORT.md`
> 服务器部署手册：`docs/RUNBOOK_SERVER.md`

## 本仓库包含（服务器直接可用）

- **全部代码**：models / losses / datasets / training / evaluation / profiling / visualization / scripts / configs
- **Baseline 官方权重**（服务器无法从 Google Drive 下载，已打包）：
  - `weights/YOLOP_End-to-end.pth`（B1，95MB）
  - `weights/TriLiteNet_{base,small,nano}.pth`（B3）
  - `weights/TwinLiteNetPlus_{large,medium,small,nano}.pth`（B2+）
  - `weights/TwinLiteNet_best.pth`（B2）
- **关键已训练模型**（可直接加载作 init / 对比）：
  - `trained_models/Static_noKD.pt`（全量，mAP 0.268 / DA 0.842 / Lane 0.180）
  - `trained_models/Static_KD_YOLOP.pt`（全量+KD，mAP 0.280 / DA 0.850 / Lane 0.193）
  - `trained_models/Dynamic_noKD.pt`（全量动态）
- **实验报告与图表**：`experiments/*_REPORT.md`、`experiments/figures/`
- **服务器手册**：`docs/RUNBOOK_SERVER.md`（Ubuntu+TITAN RTX 免密码部署）

## 本仓库不含（需要单独获取）

- **BDD100K 数据（9.8GB）**：`data/bdd100k/`（images/100k、labels、segments/masks、lanes/masks、splits/）。
  GitHub 放不下；服务器需自行拷贝（校验命令：`python3 scripts/verify_dataset.py`，
  期望 tri_train=69,863 / tri_val=10,000）。
- **baseline 仓库代码**：TwinLiteNet / TwinLiteNetPlus / TriLiteNet 是公共 GitHub 仓库，服务器自行 clone：
  - https://github.com/chequanghuy/TwinLiteNet
  - https://github.com/chequanghuy/TwinLiteNetPlus
  - https://github.com/chequanghuy/TriLiteNet
  - https://github.com/hustvl/YOLOP
  > clone 位置：前 3 个放到 `baselines/<名字>`（仓库内 baselines/ 为空目录，评测时 sys.path 导入）；
  > YOLOP 放到 **trac 的上一层**（`evaluate_baseline.py` 固定从 `../YOLOP` 导入其 `lib/`）。
  > 权重已全部在 `weights/` 内，clone 代码即可评测，无需任何额外下载。
- **大缓存**（kd_cache 等，可在服务器用 `scripts/precompute_teacher.py` 重新生成）
