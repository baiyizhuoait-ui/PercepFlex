"""P4B-EXP-05 lane label widening (lane8_z16) - mechanical decision.

lane8_z16 trains on 8px-widened lane labels, evaluates on raw 2px. Baseline is
the committed r2_z16 20ep lane_fg = 0.1943. Rule registered in
docs/PHASE5_LANE_WIDEN_PREREGISTRATION.md (D5-1..D5-4) BEFORE the run.

Noise floor: lane_fg 0.0064.
"""
import csv

NOISE = 0.0064
BASE = 0.1943  # r2_z16 e20 lane_fg, committed

rows = list(csv.DictReader(open("experiments/phase5/phase5_lane8_e20.csv")))
lane = [r for r in rows if r["cell"] == "lane8_z16"]
print(f"lane8_z16 20ep rows: {len(lane)}")
out = ["P4B-EXP-05 lane label widening (lane8_z16) decision"]
out.append(f"baseline r2_z16 e20 lane_fg = {BASE}; noise = {NOISE}")
if not lane:
    out.append("[INCOMPLETE] lane8_z16 not present - do NOT decide")
else:
    r = lane[0]
    lf = float(r["lane_fg"])
    d = lf - BASE
    x = d / NOISE
    out.append(f"lane8_z16 lane_fg = {lf:.4f} (delta {d:+.4f} = {x:+.2f}x noise)")
    out.append(f"  control mAP50 = {r['mAP50']}, da_fg = {r['da_fg']} (should be ~base)")
    out.append("")
    if abs(d) < NOISE:
        out.append("VERDICT D5-4: widening is NEUTRAL at 20ep (< 1x noise). "
                   "Training on widened lines does not change thin-line IoU -> "
                   "not the binding supervision knob.")
    elif x >= 2:
        out.append("VERDICT D5-2: SUPERVISION SUPPORTED - widening lifts lane IoU "
                   "by >= 2x noise. A pure supervision gain, zero architecture "
                   "cost, consistent with every published BDD100K model. Carry it "
                   "as a default recipe change.")
    elif x >= 1:
        out.append("VERDICT D5-2/WEAK: widening raises lane IoU by 1-2x noise - "
                   "direction positive but below the confirmation bar.")
    elif x <= -2:
        out.append("VERDICT D5-3: widening HURTS (>= 2x noise negative). Confirms "
                   "the precision problem on thin lines is architectural (no "
                   "supervision fix recovers it) -> redirects to the high-res "
                   "output head (P4B-EXP-06).")
    elif x <= -1:
        out.append("VERDICT D5-3/WEAK: widening lowers lane IoU by 1-2x noise - "
                   "direction negative but below the bar.")

txt = "\n".join(out)
print(txt)
with open("experiments/phase5/phase5_lane8_decision.txt", "w") as fh:
    fh.write(txt + "\n")
print("\n[written] experiments/phase5/phase5_lane8_decision.txt")
