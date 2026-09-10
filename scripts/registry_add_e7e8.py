#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append the Phase 6B (Phase 6B) experiment rows EXP-07 / EXP-08 to the registry.

Idempotent: re-running replaces existing rows with the same exp_id instead of
duplicating them. Status is deliberately RUNNING / placeholder — no Phase 6B
conclusion is asserted here.
"""
import csv
import io
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
REL = "experiments/phase6/phase6_experiment_registry.csv"
PATH = os.path.join(ROOT, REL)

NEW = [
    {
        "exp_id": "EXP-07",
        "question": "Does assignment/supervision quality change the marginal value of "
                    "representation capacity (capacity x supervision coupling)?",
        "status": "RUNNING (4ep screening DONE; 20ep confirmation GATED, not launched)",
        "key_result": "screening seed0 mAP50: z16_old 0.3879 / z16_km 0.3917 / z32_old 0.4161 / "
                      "z32_km 0.4035 -> dCap(old)=+0.0282, dCap(km)=+0.0118, "
                      "interaction=-0.0164 (lane side -0.0058); direction CONTRADICTS H-34; "
                      "screening-only, formal gate pending",
        "evidence_tier": "SCREENING-ONLY (4ep, not a conclusion)",
        "artifact": "phase6_round2_factorial.csv",
        "phase": "Phase 6",
        "subphase": "6B",
        "step": "P6B-STEP1",
    },
    {
        "exp_id": "EXP-08",
        "question": "At ~equal FLOPs/params, does bottleneck-aware asymmetric allocation beat "
                    "uniform capacity expansion?",
        "status": "RUNNING (A-uniform 20ep seed0 in flight; seeds 1-2 not launched)",
        "key_result": "A-uniform = encoder x1.40 (0.3339M / 1.6650G, +1.6% vs combo); "
                      "B spatial-heavy reused from 20ep 3-seed: det ~0.359 ~= r2_z16 baseline "
                      "0.3543 -> +52% FLOPs buys no detection; C combo det 0.4981 = B +0.139 at "
                      "zero extra cost -> supervision, not FLOPs, moves detection; "
                      "placeholder until A finishes",
        "evidence_tier": "PARTIAL (A seed0 pending; seeds 1-2 open)",
        "artifact": "phase6_equal_budget.csv",
        "phase": "Phase 6",
        "subphase": "6B",
        "step": "P6B-STEP1",
    },
]

with io.open(PATH, encoding="utf-8-sig", newline="") as f:
    rd = csv.DictReader(f)
    fields = list(rd.fieldnames)
    rows = [r for r in rd if r.get("exp_id") not in {n["exp_id"] for n in NEW}]

before = len(rows)
rows.extend(NEW)
with io.open(PATH, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (r.get(k) or "") for k in fields})

print("registry rows: %d -> %d (fields: %s)" % (before, len(rows), ",".join(fields)))
for r in rows:
    print("  %-7s %-5s %-4s %-12s %s" % (r.get("exp_id"), r.get("phase"), r.get("subphase"),
                                         r.get("step"), (r.get("status") or "")[:60]))
