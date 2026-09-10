"""Phase 6B prefill (zero-training): populate phase6_equal_budget.csv
with existing 20ep results instead of retraining identical configs.
  B spatial-heavy = l14f1 + OLD anchors, 20ep 3-seed   (phase4a artifacts)
  C asymmetric    = k-means + l14f1 (combo), 20ep 3-seed (phase6 artifacts)
  reference       = r2_z16 baseline 20ep (first matching metrics found)
source column marks every reused row honestly."""
import csv
import glob
import json
import os

OUT = "experiments/phase6/phase6_equal_budget.csv"
COLS = ("variant,cell,z,encoder,epochs,params_M,flops_G,fps,mAP50,mAP50_95,"
        "da_mIoU,da_fg,lane_mIoU,lane_fg,peak_gpu_mem_mib,final_train_loss,"
        "train_wall_min,seed,source,git_commit")

SPECS = []
for s, tag in [(0, ""), (1, "_s1"), (2, "_s2")]:
    p = f"experiments/phase4a/exp4B_l14f1_z16_e20{tag}_eval/metrics.json"
    if os.path.exists(p):
        SPECS.append(("e8_B_spatial_l14f1", "e8", 16, "ebase", 20, s,
                      "reused-phase4b-l14f1-e20", p))
for s, tag in [(0, ""), (1, "_s1"), (2, "_s2")]:
    p = f"experiments/phase6/exp6_combo20{tag}_eval/metrics.json"
    if os.path.exists(p):
        SPECS.append(("e8_C_asym_combo", "e8", 16, "ebase", 20, s,
                      "reused-phase6-combo20-e20", p))
ref = None
for pat in ["experiments/phase4a/*r2_z16*e20*eval/metrics.json",
            "experiments/phase4a/*z16*e20*eval/metrics.json",
            "experiments/phase2b/*z16*e20*eval/metrics.json"]:
    hits = sorted(glob.glob(pat))
    if hits:
        ref = hits[0]
        break
if ref:
    SPECS.append(("e8_ref_r2z16", "ref", 16, "ebase", 20, 0,
                  "reused-r2z16-e20", ref))


def existing_cells():
    if not os.path.exists(OUT):
        return set()
    with open(OUT) as f:
        return {r[1] for r in csv.reader(f) if len(r) > 1}


if not os.path.exists(OUT):
    with open(OUT, "w") as f:
        f.write(COLS + "\n")

have = existing_cells()
for cell, var, z, enc, ep, seed, source, mp in SPECS:
    if cell in have:
        print(f"[prefill] {cell} already present - skip")
        continue
    m = json.load(open(mp))
    g = lambda k: m.get(k)
    row = [var, cell, z, enc, ep,
           round(g("parameters") / 1e6, 4) if g("parameters") else "NA",
           round(g("flops") / 1e9, 4) if g("flops") else "NA",
           g("fps"), g("mAP50"), g("mAP50_95"), g("da_mIoU"), g("da_fg_iou"),
           g("lane_mIoU"), g("lane_fg_iou"), "NA", "NA", "NA", seed, source,
           "reused"]
    with open(OUT, "a", newline="") as f:
        csv.writer(f).writerow(row)
    print(f"[prefill] {cell} seed{seed} <- {mp}")
print("[prefill] done")
