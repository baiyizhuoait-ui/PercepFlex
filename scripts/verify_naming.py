#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the canonical id tables after naming migration (verification helper)."""
import csv
import io
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."


def show(path, cols):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        print("  MISSING", path)
        return
    with io.open(p, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    print("  columns:", list(rows[0].keys()))
    for r in rows:
        print("   ", " | ".join("%s=%s" % (c, (r.get(c) or "-")) for c in cols))
    print("  rows:", len(rows))


print("=== phase6_architecture_hypotheses.csv ===")
show("experiments/phase6/phase6_architecture_hypotheses.csv", ["id", "alias", "status"])

print("\n=== phase6_experiment_registry.csv ===")
show("experiments/phase6/phase6_experiment_registry.csv", ["exp_id", "phase", "subphase", "step"])

print("\n=== residual legacy ids in the 4 LIVE phase6 files ===")
LIVE = [
    "experiments/phase6/phase6_architecture_hypotheses.csv",
    "experiments/phase6/phase6_experiment_registry.csv",
    "experiments/phase6/phase6_novelty_matrix.csv",
    "experiments/phase6/phase6_round1_decision_report.md",
]
RULES = [
    ("H-letter", re.compile(r"\bH-[A-N]\b")),
    ("EXP-unpadded", re.compile(r"(?<![0-9A-Za-z-])EXP-[0-9](?![0-9])")),
    ("bare-STEP", re.compile(r"(?<![A-Za-z0-9-])STEP[0-9]")),
    ("Round-n", re.compile(r"(?<![A-Za-z])Round\s+[12](?![0-9])")),
    ("4B/4C-n", re.compile(r"(?<![A-Za-z0-9])4[BC]-[0-9]")),
]
for rel in LIVE:
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        continue
    txt = io.open(p, encoding="utf-8-sig").read()
    hits = {name: len(rx.findall(txt)) for name, rx in RULES}
    hits = {k: v for k, v in hits.items() if v}
    print("  %-58s %s" % (rel, hits if hits else "CLEAN"))
