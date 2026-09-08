#!/usr/bin/env python
"""STEP 7c seed replication decision.

Rule fixed in docs/PHASE5_SEED_REPLICATION_REGISTRATION.md (commit a185e48),
BEFORE the two extra seeds existed. Nothing here adapts to the numbers;
the script only applies the rule and prints the arithmetic that produced it.

    d_mean >= +0.0256            -> CONFIRMED (H5b upgraded)
    +0.0128 <= d_mean < +0.0256  -> WEAK (unchanged)
    d_mean <  +0.0128            -> NOT CONFIRMED (H5b reverts to OPEN)

No seed may be dropped, and no additional seed may be added, after the
numbers are visible.
"""

import csv
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_LANE_FG = 0.1943          # committed r2_z16 e20 row, not a rerun
NOISE = 0.0128                     # 1x noise on lane_fg at 20 epochs
CONFIRM_BAR = 2.0 * NOISE          # 0.0256
WEAK_FLOOR = 1.0 * NOISE           # 0.0128

SEEDS_CSV = "experiments/phase5/phase5_l14f1_e20_seeds.csv"
SEED0_CSV = "experiments/phase5/phase5_l14f1_e20.csv"
OUT = "experiments/phase5/phase5_l14f1_e20_seeds_decision.txt"


def read_rows(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get(r, key, default=None):
    v = r.get(key, default)
    return None if v in (None, "", "nan") else v


def main():
    # seed 0 lives in its own CSV from the STEP 7b confirmation run;
    # seeds 1 and 2 were appended into the replication CSV.
    observations = []          # (seed, lane_fg, mAP50, da_fg, source)
    for r in read_rows(SEED0_CSV):
        if r.get("cell") == "l14f1_z16" and get(r, "epochs") == "20":
            observations.append((
                int(r.get("seed") or 0),
                float(r["lane_fg"]), float(r["mAP50"]), float(r["da_fg"]),
                os.path.basename(SEED0_CSV),
            ))
    for r in read_rows(SEEDS_CSV):
        if r.get("cell") == "l14f1_z16" and get(r, "epochs") == "20":
            observations.append((
                int(r.get("seed") or 0),
                float(r["lane_fg"]), float(r["mAP50"]), float(r["da_fg"]),
                os.path.basename(SEEDS_CSV),
            ))

    # de-duplicate on seed id, keep the first occurrence
    seen, rows = set(), []
    for o in observations:
        if o[0] in seen:
            continue
        seen.add(o[0])
        rows.append(o)
    rows.sort()

    if not rows:
        print("no l14f1_z16 20-epoch rows found; nothing to decide")
        return 1

    d_lane = [v - BASELINE_LANE_FG for _, v, _, _, _ in rows]
    n = len(d_lane)
    mean = sum(d_lane) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in d_lane) / (n - 1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")

    if mean >= CONFIRM_BAR:
        verdict = "CONFIRMED"
    elif mean >= WEAK_FLOOR:
        verdict = "WEAK SUPPORT"
    else:
        verdict = "NOT CONFIRMED"

    lines = []
    lines.append("Phase 5 STEP 7c - seed replication decision")
    lines.append("=" * 62)
    lines.append("")
    lines.append("Rule fixed before the runs (docs/PHASE5_SEED_REPLICATION_REGISTRATION.md,")
    lines.append("commit a185e48). It is applied here, not renegotiated.")
    lines.append("")
    lines.append(f"baseline: committed r2_z16 e20 lane_fg = {BASELINE_LANE_FG:.4f} (not rerun)")
    lines.append(f"noise (1x, 20 epochs)                  = {NOISE:.4f}")
    lines.append(f"CONFIRMED bar (2x)                     = {CONFIRM_BAR:.4f}")
    lines.append(f"WEAK floor  (1x)                       = {WEAK_FLOOR:.4f}")
    lines.append("")
    lines.append(f"{'seed':>5} {'lane_fg':>8} {'d_lane':>9} {'x_noise':>8} {'mAP50':>7} {'da_fg':>7}  source")
    for (s, lf, mp, da, src), d in zip(rows, d_lane):
        lines.append(f"{s:>5} {lf:>8.4f} {d:>+9.4f} {d/NOISE:>8.2f} {mp:>7.4f} {da:>7.4f}  {src}")
    lines.append("")
    lines.append(f"n seeds      = {n}")
    lines.append(f"d_mean       = {mean:+.4f}  ({mean/NOISE:.2f}x noise)")
    if n > 1:
        lines.append(f"d_sd         = {sd:.4f}")
        lines.append(f"d_sem        = {sem:.4f}")
        lines.append(f"mean vs 0    = {mean/sem:.2f} SEM" if sem > 0 else "mean vs 0    = n/a")
        lines.append(f"mean vs bar  = {(mean-CONFIRM_BAR)/sem:.2f} SEM" if sem > 0 else "")
    lines.append("")
    lines.append(f"VERDICT: {verdict}")
    lines.append("")

    if verdict == "CONFIRMED":
        lines.append("H5b is upgraded from provisional to SUPPORTED. The 1/4 lateral may")
        lines.append("be treated as a real, replicated effect and built on.")
    elif verdict == "WEAK SUPPORT":
        lines.append("H5b stays WEAK SUPPORT. The effect is real in direction and clearly")
        lines.append("non-zero, but its mean sits below the pre-set 2x bar. Treat it as a")
        lines.append("small effect: do not build the final architecture on it alone, and")
        lines.append("do not report it as a confirmed bottleneck finding without the bar")
        lines.append("being restated. Every seed landed above the 1x floor, so the")
        lines.append("direction is stable; the magnitude is the open part.")
    else:
        lines.append("H5b reverts to OPEN. The 4-epoch probe gain did not survive at")
        lines.append("20 epochs once seed variance was accounted for (probe C precedent).")

    lines.append("")
    lines.append("Seeds used, in full, without selection:")
    for s, lf, mp, da, src in rows:
        lines.append(f"  seed {s}: lane_fg {lf:.4f} (from {src})")

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(ROOT, OUT), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
