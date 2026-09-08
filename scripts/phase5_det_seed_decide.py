"""Phase 4B-3 detection 20ep seed confirmation - mechanical decision.

Replicates the +0.1543 4ep gain (danc_z16, k-means anchors) at 20 epochs across
3 seeds. The rule is registered in docs/PHASE5_DET_E20_CONFIRM_REGISTRATION.md
(seed0, E1 bar) and the seed replication was authorised by the user on
2026-09-08 night. This script applies the rule mechanically.

Noise floor: mAP50 0.0146. Baseline = committed r2_z16 20ep = 0.3543 (reused).
"""
import csv
import sys

NOISE = 0.0146
BASE = 0.3543  # r2_z16 e20, committed
CONFIRM_2X = BASE + 2 * NOISE   # 0.3835
CONFIRM_1X = BASE + NOISE       # 0.3689

rows = list(csv.DictReader(open("experiments/phase5/phase5_det_e20_results.csv")))
cells = [r for r in rows if r["cell"] == "danc_z16"]
print(f"danc_z16 20ep rows found: {len(cells)} (expect 3: seeds 0/1/2)")
out = ["Detection 20ep seed confirmation (danc_z16, k-means anchors)"]
out.append(f"baseline r2_z16 e20 mAP50 = {BASE}; noise = {NOISE}")
out.append(f"CONFIRM bar = {CONFIRM_2X:.4f} (2x), WEAK floor = {CONFIRM_1X:.4f} (1x)")
out.append("")
out.append(f"{'seed':<6}{'mAP50':>8}{'mAP50_95':>10}  vs base")
seeds = []
for r in sorted(cells, key=lambda x: int(x.get("seed", 0))):
    s = int(r.get("seed", 0))
    m = float(r["mAP50"]); m95 = float(r["mAP50_95"])
    seeds.append(m)
    out.append(f"{s:<6}{m:>8.4f}{m95:>10.4f}  {m-BASE:+.4f} = {(m-BASE)/NOISE:+.2f}x noise")
out.append("")
if len(seeds) < 3:
    out.append("[INCOMPLETE] fewer than 3 seeds present - do NOT decide yet")
else:
    mean = sum(seeds) / 3
    lo = min(seeds); hi = max(seeds)
    spread = hi - lo
    out.append(f"3-seed mean mAP50 = {mean:.4f} (delta {mean-BASE:+.4f} = "
               f"{(mean-BASE)/NOISE:+.2f}x noise)")
    out.append(f"spread {lo:.4f} .. {hi:.4f} (range {spread:.4f} = "
               f"{spread/NOISE:.2f}x noise)")
    out.append("")
    n_above_2x = sum(1 for s in seeds if s >= CONFIRM_2X)
    n_above_1x = sum(1 for s in seeds if s >= CONFIRM_1X)
    out.append(f"seeds >= 2x bar: {n_above_2x}/3 ; >= 1x floor: {n_above_1x}/3")
    out.append("")
    if n_above_2x == 3 and mean - BASE >= 2 * NOISE:
        out.append("VERDICT: CONFIRMED - all 3 seeds clear 2x noise and the mean "
                   "is >= 2x. The +0.15-level detection gain from k-means anchors "
                   "is real and not seed luck.")
    elif n_above_1x == 3:
        out.append("VERDICT: WEAK but real - all 3 seeds clear 1x but not 2x. "
                   "Gain is real (>=1x on all seeds) but sits below the 2x "
                   "confirmation bar.")
    else:
        out.append("VERDICT: NOT CONFIRMED - some seed fails the 1x floor. The "
                   "4ep gain does not replicate at 20 epochs across seeds "
                   "(reversed finding).")
    out.append("")
    out.append("Cross-check against lane/DA (control): a detection-anchor change "
               "must not move lane_fg/da_fg (report any large move).")

txt = "\n".join(out)
print(txt)
with open("experiments/phase5/phase5_det_e20_seed_decision.txt", "w") as fh:
    fh.write(txt + "\n")
print("\n[written] experiments/phase5/phase5_det_e20_seed_decision.txt")
