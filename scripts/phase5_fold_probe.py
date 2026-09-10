"""Phase 5 - fold the probe + error-geometry results back into the matrix.

Reads:
  experiments/phase5/phase5_lane_probe_decision.txt   (L1-L4 rule output)
  experiments/phase5/phase5_lane_probe_errorgeom.csv  (precision/recall/
                                                      area/tolerance for the
                                                      three probe cells)

Writes:
  experiments/phase5/phase5_hypothesis_matrix.csv   (H-05a/H-05b/H-06/H-17 updated)
  experiments/phase5/phase5_bottleneck_profile.csv  (per-task per-bottleneck
                                                    verdict)
  experiments/phase5/phase5_intervention_matrix.csv (per-task: intervention,
                                                    cell, gain, cost, status)

The H-17 status uses the precision/recall decomposition to decide whether the
1/4 head moves precision (resolution was the binding constraint) or recall
(continuity was). If precision is the one that moves, H-05/H-05b is the
explanation; if recall also moves, H-17 partly dissolves into H-05.
"""
import csv
import io
import os
import re

ROOT = "/home/mycode/ai_study/trac"
MATRIX = os.path.join(ROOT, "experiments/phase5/phase5_hypothesis_matrix.csv")
DECIDE = os.path.join(ROOT, "experiments/phase5/phase5_lane_probe_decision.txt")
EG = os.path.join(ROOT, "experiments/phase5/phase5_lane_probe_errorgeom.csv")
BP = os.path.join(ROOT, "experiments/phase5/phase5_bottleneck_profile.csv")
IM = os.path.join(ROOT, "experiments/phase5/phase5_intervention_matrix.csv")


def load_decision():
    if not os.path.exists(DECIDE):
        return None
    txt = io.open(DECIDE, encoding="utf-8").read()
    l1 = bool(re.search(r"L1\b.*PASS", txt))
    l2 = bool(re.search(r"L2\b.*PASS", txt))
    l3 = bool(re.search(r"\bPASS: lane is spatially constrained", txt))
    l4_ok = "control FAILED" not in txt
    branch = "?"
    if l1 and l3:
        branch = "H-05b supported; extend l14f1 to 20ep"
    elif l1 and not l3:
        branch = "H-05b weakly supported; spatial claim cannot be made"
    elif l2 and not l1:
        branch = "H-05a supported; H-05b not"
    elif "UNRESOLVED" in txt:
        branch = "unresolved; extend two cells to 20ep"
    else:
        branch = "nothing reaches 1x noise; do not extend"
    return {"l1": l1, "l2": l2, "l3": l3, "l4_ok": l4_ok, "branch": branch, "raw": txt}


def load_errorgeom():
    if not os.path.exists(EG):
        return None
    rows = list(csv.DictReader(io.open(EG, encoding="utf-8-sig")))
    return {r["cell"]: r for r in rows if r.get("task") == "lane"}


def fold(matrix_rows, by_id, decision, eg):
    fields = list(matrix_rows[0].keys())
    if decision and eg:
        baseline = eg.get("r2u_z16") or eg.get("r2_z16")
        l14f1 = eg.get("l14f1_z16")
        if baseline and l14f1:
            b_prec = float(baseline["precision"])
            b_rec = float(baseline["recall"])
            f_prec = float(l14f1["precision"])
            f_rec = float(l14f1["recall"])
            b_iou = float(baseline["fg_iou"])
            f_iou = float(l14f1["fg_iou"])
            decision_summary = (
                f"r2u_z16: prec {b_prec:.4f} rec {b_rec:.4f} IoU {b_iou:.4f}; "
                f"l14f1_z16: prec {f_prec:.4f} rec {f_rec:.4f} IoU {f_iou:.4f}. "
                f"Delta prec {f_prec-b_prec:+.4f}, delta rec {f_rec-b_rec:+.4f}."
            )
        else:
            decision_summary = "EG results missing for the new cells."

        if decision["l1"] and decision["l3"]:
            h5b_state = "SUPPORTED (pre-registered L1+L3)"
        elif decision["l1"] and not decision["l3"]:
            h5b_state = "WEAKLY SUPPORTED (L1 not L3)"
        else:
            h5b_state = "NOT SUPPORTED at 4ep"
        by_id["H-05b"].update(
            status=h5b_state,
            evidence_available=decision_summary + "  " + decision["branch"],
            decision=decision["branch"] + (
                "; 4ep result provisional until 20ep confirmation" if decision["l1"] else ""
            ),
        )
        # H-05a mirrors H-05b roughly: l14up is the H-05a cell.
        l14up = eg.get("l14up_z16")
        if l14up and float(l14up["fg_iou"]) > float(baseline["fg_iou"]) + 0.0032:
            by_id["H-05a"]["status"] = "PARTIALLY SUPPORTED"
        else:
            by_id["H-05a"]["status"] = "WEAKLY SUPPORTED (L2 only or none)"

        # H-17 reads the recall side of the move.
        if l14f1 and l14f1.get("recall") and l14f1["recall"] != baseline["recall"]:
            by_id["H-17"]["status"] = (
                f"OPEN (recall {'moved' if abs(float(l14f1['recall'])-float(baseline['recall'])) > 0.01 else 'did not move'} "
                "between 1/8 and 1/4)"
            )

    with open(MATRIX, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(matrix_rows)


def write_bottleneck_profile(matrix_rows):
    fields = ["task", "bottleneck", "status", "evidence_one_line"]
    with open(BP, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in matrix_rows:
            w.writerow({
                "task": r["task"],
                "bottleneck": r["hypothesis"][:80],
                "status": r["status"],
                "evidence_one_line": r["evidence_available"][:120].replace("\n", " "),
            })


def write_intervention_matrix(matrix_rows):
    fields = ["task", "intervention", "status", "falsification"]
    with open(IM, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in matrix_rows:
            w.writerow({
                "task": r["task"],
                "intervention": r["intervention"][:120].replace("\n", " "),
                "status": r["status"],
                "falsification": r["falsification_criterion"][:120].replace("\n", " "),
            })


def main():
    rows = list(csv.DictReader(io.open(MATRIX, encoding="utf-8-sig")))
    by_id = {r["id"]: r for r in rows}
    decision = load_decision()
    eg = load_errorgeom()
    fold(rows, by_id, decision, eg)
    write_bottleneck_profile(rows)
    write_intervention_matrix(rows)
    print("updated", MATRIX)
    print("wrote", BP)
    print("wrote", IM)


if __name__ == "__main__":
    main()
