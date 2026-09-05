#!/usr/bin/env python
"""
Phase 3B · structural config audit.

Mandatory pre-run gate. For every one of the 9 matrix cells (the 3 new-ish
z16 cells from Phase 3A plus the 6 new Phase 3B cells) it checks the 12
fields the protocol pins down, and refuses to declare PASS unless the ONLY
fields that vary across the matrix are:

    model.encoder.stem
    model.encoder.stages
    model.representation.z_channels

Fields that live in train.py rather than the YAML (optimizer, scheduler) and
in the runner (seed) are verified by grepping the actual source, so a config
cannot smuggle in a change the YAML does not even express.

Writes: experiments/phase3b/phase3B_config_audit.txt
Exits non-zero if any check fails (so the runner can gate on it).
"""
import os
import re
import sys

import yaml

ROOT = "/home/mycode/ai_study/trac"
OUTDIR = os.path.join(ROOT, "experiments", "phase3b")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- pinned spec
SPEC = {
    "blocks": [2, 2, 2],
    "input_size": [640, 640],
    "detection": {"nc": 1},
    "segmentation": {"hidden": 32},
    "stage": "A",
    "train_split": "tri_train",
    "batch_size": 16,
    "epochs": 20,
    "lr": 1e-3,
    "weight_decay": 5e-4,
    "lambda_da": 1.0,
    "lambda_lane": 1.0,
    "lambda_budget": 0.0,
    "grad_clip": 10.0,
    "data_root": "data/bdd100k",
    "seed": 0,
}

EXPECTED_ENCODER = {
    "esmall": dict(stem=12, stages=[24, 40, 64, 80]),
    "ebase": dict(stem=16, stages=[32, 64, 96, 128]),
    "elarge": dict(stem=20, stages=[40, 80, 128, 168]),
}

# (config path, encoder key, expected z)
CELLS = []
for enc in ("esmall", "ebase", "elarge"):
    for z in (16, 32, 128):
        cfg = (f"configs/phase3a_{enc}_z16.yaml" if z == 16
               else f"configs/phase3b_{enc}_z{z}.yaml")
        CELLS.append((cfg, enc, z))

# Fields that ARE allowed to differ across cells (the two experimental axes)
ALLOWED_TO_VARY = {("model", "encoder", "stem"),
                   ("model", "encoder", "stages"),
                   ("model", "representation", "z_channels")}


def flatten(d, prefix=()):
    """Yield (key_path_tuple, value) for every leaf of a nested dict."""
    for k, v in d.items():
        p = prefix + (k,)
        if isinstance(v, dict):
            yield from flatten(v, p)
        else:
            yield p, v


def same(a, b):
    """
    Compare two config values.

    Numeric-looking YAML scalars written as `1e-3` / `5e-4` are parsed by
    PyYAML as STRINGS (YAML 1.1 wants a dot in the mantissa), so a naive
    `==` against a Python float reports a false mismatch. train.py does
    `float(tr.get("lr", 1e-3))` at runtime, so comparing numerically is the
    faithful check. Non-numeric values still compare exactly.
    """
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return False


def fmt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def check_source_invariants():
    """Optimizer / scheduler / seed live outside the YAML; verify at source."""
    results = []

    train_py = os.path.join(ROOT, "training", "train.py")
    src = open(train_py).read() if os.path.exists(train_py) else ""
    results.append(("training/train.py: optimizer is AdamW",
                    bool(re.search(r"optim\.AdamW\(", src)),
                    "found optim.AdamW(" if src else "FILE MISSING"))
    results.append(("training/train.py: scheduler is CosineAnnealingLR",
                    bool(re.search(r"lr_scheduler\.CosineAnnealingLR\(", src)),
                    "found CosineAnnealingLR" if src else "FILE MISSING"))
    results.append(("training/train.py: no LR/optimizer branching on config",
                    "lr=cfg" not in src and 'cfg["train"].get("optimizer"' not in src,
                    "optimizer hyperparameters not config-driven"))

    runner = os.path.join(ROOT, "scripts", "phase3b_run_matrix.sh")
    rsrc = open(runner).read() if os.path.exists(runner) else ""
    results.append(("scripts/phase3b_run_matrix.sh: SEED pinned to 0",
                    bool(re.search(r'SEED=\$\{SEED_OVERRIDE:-0\}', rsrc)),
                    'SEED=${SEED_OVERRIDE:-0}' if rsrc else "FILE MISSING"))
    results.append(("scripts/phase3b_run_matrix.sh: EPOCHS pinned to 20",
                    bool(re.search(r'EP=\$\{EPOCHS_OVERRIDE:-20\}', rsrc)),
                    'EP=${EPOCHS_OVERRIDE:-20}' if rsrc else "FILE MISSING"))
    results.append(("scripts/phase3b_run_matrix.sh: passes --seed to train.py",
                    bool(re.search(r'--seed "\$SEED"', rsrc)),
                    "--seed \"$SEED\"" if rsrc else "FILE MISSING"))
    results.append(("scripts/phase3b_run_matrix.sh: no --batch-size override",
                    "--batch-size" not in rsrc,
                    "batch size comes from the config only"))
    results.append(("scripts/phase3b_run_matrix.sh: no --num-workers override",
                    "--num-workers" not in rsrc,
                    "worker count comes from the config only"))
    return results


def main():
    lines = []
    w = lines.append
    failures = []

    w("=" * 88)
    w("PHASE 3B · STRUCTURAL CONFIG AUDIT")
    w("=" * 88)
    w("")
    w("Rule: across the 3x3 matrix the ONLY permitted differences are")
    w("      model.encoder.stem, model.encoder.stages, model.representation.z_channels.")
    w("      Everything else must be byte-identical to the pinned protocol.")
    w("")
    w(f"pinned spec: {SPEC}")
    w("")

    # ---------------------------------------------------------- per-cell checks
    parsed = {}
    w("-" * 88)
    w("### 1. PER-CELL FIELD CHECK")
    w("-" * 88)
    w(f"{'cell':<16} {'file':<34} {'stem':>5} {'stages':<20} {'z':>4}  result")
    w("-" * 88)

    for cfg_rel, enc, z in CELLS:
        path = os.path.join(ROOT, cfg_rel)
        tag = f"{enc}_z{z}"
        if not os.path.exists(path):
            w(f"{tag:<16} {cfg_rel:<34} {'':>5} {'':<20} {'':>4}  MISSING")
            failures.append(f"{tag}: config file missing")
            continue
        cfg = yaml.safe_load(open(path))
        parsed[tag] = cfg

        m, t = cfg.get("model", {}), cfg.get("train", {})
        stem = m.get("encoder", {}).get("stem")
        stages = m.get("encoder", {}).get("stages")
        zc = m.get("representation", {}).get("z_channels")
        blocks = m.get("encoder", {}).get("blocks")
        stg = "[" + ",".join(str(s) for s in stages) + "]"

        bad = []
        if stem != EXPECTED_ENCODER[enc]["stem"]:
            bad.append(f"stem={stem} want {EXPECTED_ENCODER[enc]['stem']}")
        if list(stages) != EXPECTED_ENCODER[enc]["stages"]:
            bad.append(f"stages={stages} want {EXPECTED_ENCODER[enc]['stages']}")
        if zc != z:
            bad.append(f"z={zc} want {z}")
        if list(blocks) != SPEC["blocks"]:
            bad.append(f"blocks={blocks} want {SPEC['blocks']}")
        if list(cfg.get("input_size", [])) != SPEC["input_size"]:
            bad.append(f"input_size={cfg.get('input_size')}")
        if m.get("detection") != SPEC["detection"]:
            bad.append(f"detection={m.get('detection')}")
        if m.get("segmentation") != SPEC["segmentation"]:
            bad.append(f"segmentation={m.get('segmentation')}")

        for k in ("stage", "train_split", "batch_size", "epochs", "lr",
                  "weight_decay", "lambda_da", "lambda_lane", "lambda_budget",
                  "grad_clip"):
            if not same(t.get(k), SPEC[k]):
                bad.append(f"train.{k}={t.get(k)!r} want {SPEC[k]!r}")
        if cfg.get("data", {}).get("root") != SPEC["data_root"]:
            bad.append(f"data.root={cfg.get('data', {}).get('root')}")

        if bad:
            w(f"{tag:<16} {cfg_rel:<34} {stem:>5} {stg:<20} {zc:>4}  FAIL")
            for b in bad:
                w(f"      - {b}")
            failures.extend(f"{tag}: {b}" for b in bad)
        else:
            w(f"{tag:<16} {cfg_rel:<34} {stem:>5} {stg:<20} {zc:>4}  OK")

    # ------------------------------------------------- cross-cell invariance
    w("")
    w("-" * 88)
    w("### 2. CROSS-CELL INVARIANCE  (which fields actually vary?)")
    w("-" * 88)
    if len(parsed) < 2:
        w("  not enough cells parsed")
        failures.append("cross-cell invariance: too few cells")
    else:
        keys = None
        for tag, cfg in parsed.items():
            ks = {p for p, _ in flatten(cfg)}
            keys = ks if keys is None else (keys | ks)
        varying = []
        for k in sorted(keys):
            vals = {}
            for tag, cfg in parsed.items():
                d = dict(flatten(cfg))
                vals[tag] = d.get(k, "<absent>")
            uniq = {repr(v) for v in vals.values()}
            if len(uniq) > 1:
                varying.append((".".join(k), vals))
        if not varying:
            w("  (no field varies -- suspicious, expected encoder+Z to vary)")
            failures.append("no field varies across the matrix")
        else:
            for name, vals in varying:
                allowed = name in {".".join(a) for a in ALLOWED_TO_VARY}
                flag = "ALLOWED (experimental axis)" if allowed else "*** VIOLATION ***"
                w(f"  {name}  ->  {flag}")
                for tag in sorted(vals):
                    w(f"        {tag:<12} {vals[tag]!r}")
                if not allowed:
                    failures.append(f"forbidden field varies: {name}")

    # ------------------------------------------------------ source invariants
    w("")
    w("-" * 88)
    w("### 3. SOURCE-LEVEL INVARIANTS  (optimizer / scheduler / seed)")
    w("-" * 88)
    w("  These are not expressible in the YAML, so they are checked at source.")
    w("")
    for name, ok, detail in check_source_invariants():
        w(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        w(f"           {detail}")
        if not ok:
            failures.append(f"source invariant: {name}")

    # --------------------------------------------------- runtime log evidence
    w("")
    w("-" * 88)
    w("### 4. RUNTIME EVIDENCE  (what the trainer actually did)")
    w("-" * 88)
    w("  The YAML is the intent; the log is what happened. Read every training")
    w("  log that exists so far and confirm lr / batch / schedule at runtime.")
    w("")
    w(f"  {'cell':<16} {'log':<6} {'lr@step1':<10} {'batch':<7} {'steps/ep':<9} {'ep done':<8} result")
    w("  " + "-" * 70)

    LOG_SOURCES = []
    for enc in ("esmall", "ebase", "elarge"):
        LOG_SOURCES.append((f"{enc}_z16", f"experiments/phase3a/exp3A_{enc}_z16/training_log.txt"))
    for enc in ("esmall", "ebase", "elarge"):
        for z in (32, 128):
            LOG_SOURCES.append((f"{enc}_z{z}", f"experiments/phase3b/exp3B_{enc}_z{z}/training_log.txt"))

    n_logs = 0
    for tag, rel in LOG_SOURCES:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            w(f"  {tag:<16} {'(pending)':<6} {'-':<10} {'-':<7} {'-':<9} {'-':<8} not run yet")
            continue
        n_logs += 1
        txt = open(p, errors="replace").read()

        lr0 = re.search(r"lr ([0-9.eE+-]+)", txt)
        lr0 = lr0.group(1) if lr0 else "NA"
        bs = re.search(r"batch_size=(\d+)", txt)
        bs = bs.group(1) if bs else "NA"
        steps = re.findall(r"step \d+/(\d+)", txt)
        steps = steps[0] if steps else "NA"
        epd = re.findall(r"ep (\d+) DONE", txt)
        epd = max((int(e) for e in epd), default=0)

        bad = []
        try:
            if abs(float(lr0) - SPEC["lr"]) > 1e-9:
                bad.append(f"lr={lr0}")
        except ValueError:
            bad.append(f"lr unparseable ({lr0})")
        if bs != str(SPEC["batch_size"]):
            bad.append(f"batch={bs}")
        if epd != SPEC["epochs"]:
            bad.append(f"completed epochs={epd}")

        if bad:
            w(f"  {tag:<16} {'found':<6} {lr0:<10} {bs:<7} {str(steps):<9} {epd:<8} FAIL")
            for b in bad:
                w(f"        - {b}")
            failures.extend(f"runtime {tag}: {b}" for b in bad)
        else:
            w(f"  {tag:<16} {'found':<6} {lr0:<10} {bs:<7} {str(steps):<9} {epd:<8} OK")
    w("")
    w(f"  ({n_logs}/{len(LOG_SOURCES)} logs present; missing ones are cells not yet run)")
    w("  Note: train.py auto-raises DataLoader workers (2 -> 8) from CPU count.")
    w("        That is identical on every cell (same machine, cores=24) and is")
    w("        not a protocol variable, so it is recorded, not flagged.")

    # ------------------------------------------------------------- verdict
    w("")
    w("=" * 88)
    if failures:
        w("AUDIT RESULT: FAIL")
        w("=" * 88)
        w("")
        for f in failures:
            w(f"  - {f}")
        w("")
        w("DO NOT LAUNCH. Fix the above before running any Phase 3B cell.")
        rc = 1
    else:
        w("AUDIT RESULT: PASS")
        w("=" * 88)
        w("")
        w("PASS: only encoder.stem / encoder.stages / representation.z_channels")
        w("      differ across the 9 cells. Everything else -- blocks, heads,")
        w("      input size, optimizer (AdamW), scheduler (cosine), epochs (20),")
        w("      batch (16), lr (1e-3), losses, seed (0), dataset (tri_train) --")
        w("      is identical to the pinned Phase 3A / Phase 2-D protocol.")
        rc = 0

    txt = "\n".join(lines) + "\n"
    dst = os.path.join(OUTDIR, "phase3B_config_audit.txt")
    with open(dst, "w") as f:
        f.write(txt)
    print(txt)
    print(f"[audit] written -> {dst}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
