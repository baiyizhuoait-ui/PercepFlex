"""Fold the L1-L4 decision into the hypothesis matrix WITHOUT waiting for the
probe error geometry (which is slow because it recomputes boundary curves).
The EG enrichment (precision/recall decomposition) is folded later by
phase5_eg_merge_fold.py; this script only sets statuses/decisions."""
import csv
import io
import re

ROOT = "/home/mycode/ai_study/trac"
MATRIX = ROOT + "/experiments/phase5/phase5_hypothesis_matrix.csv"
DECIDE = ROOT + "/experiments/phase5/phase5_lane_probe_decision.txt"

txt = io.open(DECIDE, encoding="utf-8").read()
l1 = bool(re.search(r"L1\b.*PASS", txt))
l2 = bool(re.search(r"L2\b.*PASS", txt))
l3 = bool(re.search(r"\bPASS: lane is spatially constrained", txt))
l4_ok = "control FAILED" not in txt
print("L1", l1, "L2", l2, "L3", l3, "L4", l4_ok)

rows = list(csv.DictReader(io.open(MATRIX, encoding="utf-8-sig")))
fields = list(rows[0].keys())
by_id = {r["id"]: r for r in rows}

if l1 and l3:
    h5b = "SUPPORTED (pre-registered L1+L3, 4ep; provisional until 20ep)"
elif l1:
    h5b = "WEAKLY SUPPORTED (L1 not L3)"
else:
    h5b = "NOT SUPPORTED at 4ep"

ev = ("l14f1 lane_fg 0.1916 (+0.0151, 2.36x noise, threshold 0.1829); "
      "l14up 0.1894 (+2.02x); lch64 0.1778 (+0.20x); "
      "per-FLOP spatial 0.0270 vs channel 0.0031 (8.7x); "
      "L4 det/DA controls < 1x noise")

if "H5b" in by_id:
    by_id["H5b"]["status"] = h5b
    by_id["H5b"]["evidence_available"] = ev
    by_id["H5b"]["decision"] = (
        "extend l14f1_z16 to 20ep under PHASE5_E20_CONFIRM_REGISTRATION.md; "
        "do not run the 1/2 rung; 4ep result provisional"
    )
if "H5a" in by_id:
    by_id["H5a"]["status"] = (
        "PARTIALLY SUPPORTED (L2: upsample-only gains 2.02x noise, "
        "crosses 1/8 block-fill 0.1853)"
        if l2 else "OPEN"
    )
    by_id["H5a"]["decision"] = "subordinate to H5b; no further experiment"
if "H6" in by_id:
    by_id["H6"]["status"] = "REJECTED (lch64: 64ch at 1/8 buys 0.20x noise for +16% params)"
    by_id["H6"]["decision"] = "channel capacity is not the lane bottleneck at Z=16"
if "H17" in by_id:
    by_id["H17"]["status"] = "OPEN (quantisation half now causal; continuity half waits for 20ep + EG)"

with open(MATRIX, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print("matrix updated:", h5b)
