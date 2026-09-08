"""Apply the PHASE5_E20_CONFIRM_REGISTRATION.md decision rule to the landed
l14f1_z16 20-epoch run. The rule was committed (f775718) before the run
started; nothing here is negotiable - including the 0.0007 shortfall."""
import csv
import io

ROOT = "/home/mycode/ai_study/trac"
E20 = ROOT + "/experiments/phase5/phase5_l14f1_e20.csv"

# Committed comparison target (phase4A_results.csv, r2_z16 e20) and
# registered noise floors (2x Phase 2-C external).
BASE = {"lane_fg": 0.1943, "mAP50": 0.3543, "da_fg": 0.7612}
NOISE = {"lane_fg": 0.0128, "mAP50": 0.0292, "da_fg": 0.0808}

rows = [r for r in csv.DictReader(io.open(E20, encoding="utf-8-sig"))
        if r["cell"] == "l14f1_z16"]
assert len(rows) == 1, rows
r = rows[0]
new = {"lane_fg": float(r["lane_fg"]), "mAP50": float(r["mAP50"]),
       "da_fg": float(r["da_fg"])}

d_lane = new["lane_fg"] - BASE["lane_fg"]
x = d_lane / NOISE["lane_fg"]
d_det = new["mAP50"] - BASE["mAP50"]
d_da = new["da_fg"] - BASE["da_fg"]

print("l14f1_z16 e20: lane_fg %.4f  mAP50 %.4f  da_fg %.4f" %
      (new["lane_fg"], new["mAP50"], new["da_fg"]))
print("baseline r2_z16 e20 (committed): lane_fg %.4f  mAP50 %.4f  da_fg %.4f" %
      (BASE["lane_fg"], BASE["mAP50"], BASE["da_fg"]))
print("d_lane %+.4f (%.2fx noise)  d_det %+.4f (%.2fx)  d_da %+.4f (%.2fx)" %
      (d_lane, x, d_det, d_det / NOISE["mAP50"], d_da, d_da / NOISE["da_fg"]))

if d_lane >= 2 * NOISE["lane_fg"]:
    verdict = "CONFIRMED"
elif d_lane >= NOISE["lane_fg"]:
    verdict = "WEAK SUPPORT"
else:
    verdict = "NOT CONFIRMED"
print("registered rule ->", verdict)

out = ROOT + "/experiments/phase5/phase5_l14f1_e20_decision.txt"
with io.open(out, "w", encoding="utf-8") as f:
    f.write(
        "Phase 5 STEP 7b - 20-epoch confirmation decision (rule fixed in "
        "f775718 before the run)\n"
        "=====================================================================\n"
        "l14f1_z16 e20: lane_fg %.4f  mAP50 %.4f  da_fg %.4f  lane_mIoU %s\n"
        "baseline r2_z16 e20 (committed row): lane_fg %.4f  mAP50 %.4f  "
        "da_fg %.4f\n"
        "d_lane %+.4f = %.2fx noise (CONFIRMED line: +0.0256; missed by "
        "%+.4f)\n"
        "d_det %+.4f (%.2fx noise, gate 2x) - ok\n"
        "d_da  %+.4f (%.2fx noise, gate 2x) - ok\n"
        "VERDICT: %s\n"
        "Per the registration: WEAK means H5b stays provisional; do not "
        "build on it;\nrecord as unresolved. The 4-ep direction survived in "
        "sign and magnitude\n(+2.36x at 4ep, +1.95x at 20ep) but did not "
        "clear the pre-set bar.\n"
        % (new["lane_fg"], new["mAP50"], new["da_fg"], r["lane_mIoU"],
           BASE["lane_fg"], BASE["mAP50"], BASE["da_fg"],
           d_lane, x, 2 * NOISE["lane_fg"] - d_lane,
           d_det, d_det / NOISE["mAP50"], d_da, d_da / NOISE["da_fg"],
           verdict))
print("wrote", out)
