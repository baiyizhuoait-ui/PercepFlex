#!/usr/bin/env python
"""P4A-STEP2 - decide whether the 4-epoch R2 sanity run may proceed.

Pass criteria are the ones the protocol lists: it trains, the loss falls, all
three tasks produce output, nothing is NaN, and the metrics file is sane. It
also checks that detection actually produced predictions (n_pred > 0), because
a detection head that silently emits nothing would still 'train' happily.
"""
import csv, json, os, re, sys

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments/phase4a")
SAN = os.path.join(OUT, "phase4A_sanity.csv")
REPORT = os.path.join(OUT, "phase4A_sanity_check.txt")
TAG = sys.argv[1] if len(sys.argv) > 1 else "exp4A_r2_z16_e4"

# metrics.json is FLAT, not nested: the fg scores are 'da_fg_iou' / 'lane_fg_iou'.
# Keeping the canonical analysis names on the left so the printed table matches
# every other phase's output.
METRICS = [("mAP50", "mAP50"), ("mAP50_95", "mAP50_95"),
           ("da_mIoU", "da_mIoU"), ("da_fg", "da_fg_iou"),
           ("lane_mIoU", "lane_mIoU"), ("lane_fg", "lane_fg_iou")]

w = []
def p(s=""):
    w.append(s); print(s)

ok = True

p("=" * 100)
p("P4A-STEP2 SANITY CHECK  (%s)" % TAG)
p("=" * 100)
p()

tlog = os.path.join(OUT, TAG, "training_log.txt")
mp = os.path.join(OUT, TAG + "_eval", "metrics.json")

# ---- 1 training log
if not os.path.exists(tlog):
    p("1. training log                 : MISSING -> FAIL")
    ok = False
else:
    lines = [l for l in open(tlog) if "loss" in l and "ep" in l]
    p("1. training log lines           : %d" % len(lines))
    losses = []
    for l in lines:
        m = re.search(r"loss ([0-9.]+)", l)
        if m:
            losses.append(float(m.group(1)))
    if not losses:
        p("   no loss values parsed        : FAIL")
        ok = False
    else:
        first, last = losses[0], losses[-1]
        nan = any(x != x for x in losses)
        p("   first loss %.4f -> last %.4f" % (first, last))
        p("   contains NaN                 : %s" % ("YES -> FAIL" if nan else "no"))
        p("   loss decreased               : %s" % ("yes -> PASS" if last < first else "NO -> FAIL"))
        ok = ok and (last < first) and not nan
        # per-task losses
        for t in ("det", "da", "lane"):
            vals = [float(m.group(1)) for l in lines
                    for m in [re.search(r"\b%s ([0-9.]+)" % t, l)] if m]
            if vals:
                p("   %-4s loss %.4f -> %.4f" % (t, vals[0], vals[-1]))
p()

# ---- 2 metrics
p("2. eval metrics")
if not os.path.exists(mp):
    p("   metrics.json                 : MISSING -> FAIL")
    ok = False
else:
    m = json.load(open(mp))
    for k, jk in METRICS + [("parameters", "parameters"), ("flops", "flops")]:
        v = m.get(jk, None)
        bad = (v is None) or (isinstance(v, float) and v != v)
        p("   %-10s %s%s" % (k, v, "   -> FAIL (NaN/missing)" if bad else ""))
        ok = ok and not bad
    n_pred = m.get("det_n_pred", 0)
    p("   det_n_pred                   : %s%s" % (
        n_pred, "" if n_pred and n_pred > 0 else "   -> FAIL (detection emitted nothing)"))
    ok = ok and bool(n_pred and n_pred > 0)
p()

# ---- 3 csv row
p("3. result row in phase4A_sanity.csv")
if os.path.exists(SAN):
    rows = list(csv.DictReader(open(SAN)))
    p("   rows: %d" % len(rows))
    for r in rows:
        p("   %s  params %s  flops %s  mAP50 %s  da %s  lane %s" % (
            r.get("cell"), r.get("params_M"), r.get("flops_G"),
            r.get("mAP50"), r.get("da_mIoU"), r.get("lane_mIoU")))
    if not rows:
        p("   no rows -> FAIL")
        ok = False
else:
    p("   csv missing -> FAIL")
    ok = False
p()

p("=" * 100)
p("SANITY: %s" % ("PASS - proceed to the 20-epoch main experiment" if ok
                  else "FAIL - fix the implementation before spending 20 epochs"))
p("=" * 100)

open(REPORT, "w").write("\n".join(w) + "\n")
print()
print("written:", REPORT)
sys.exit(0 if ok else 1)
