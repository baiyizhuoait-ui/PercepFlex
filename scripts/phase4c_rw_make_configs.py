"""Probe C configs: R2 with the detection loss scaled down to 0.2.

Goal (H11): equalise the three tasks' effective pull on the shared Z. Measured
at the Phase 4A 20ep checkpoints, ||dL_t/dZ|| is 0.0308 / 0.0059 / 0.0054 at
z16 (det / da / lane) -> detection owns ~75% of the gradient while the pairwise
cosines sit at ~0. Scaling det by 0.2 puts the three at ~0.0062 / 0.0059 /
0.0054, i.e. near parity, without inflating the total loss scale.

Rule: the ONLY difference from the R2 baseline config is train.lambda_det.
Everything else - architecture, z width, schedule, seed - is copied verbatim.
This script asserts that by deep-comparing the parsed YAML, not by eyeballing.
"""
import copy
import pathlib

import yaml

ROOT = pathlib.Path("/home/mycode/ai_study/trac")
LAMBDA_DET = 0.2
PAIRS = [("r2u_z16", "rw_z16"), ("r2u_z32", "rw_z32")]


def flat_diff(a, b, pre=""):
    d = []
    for k in sorted(set(a) | set(b)):
        ka, kb = pre + str(k), pre + str(k)
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            d += flat_diff(va, vb, ka + ".")
        elif va != vb:
            d.append((ka, va, vb))
    return d


lines = []
ok = True
for base, new in PAIRS:
    src = ROOT / ("configs/phase4b_%s.yaml" % base)
    dst = ROOT / ("configs/phase4c_%s.yaml" % new)
    cfg = yaml.safe_load(src.read_text())
    orig = copy.deepcopy(cfg)

    cfg["train"]["lambda_det"] = LAMBDA_DET
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))

    reloaded = yaml.safe_load(dst.read_text())
    diff = flat_diff(orig, reloaded)
    lines.append("%-10s <- %-10s  lambda_det=%s" % (new, base, LAMBDA_DET))
    lines.append("   diffs vs base: %s" % (diff if diff else "NONE"))
    if diff != [("train.lambda_det", None, LAMBDA_DET)]:
        ok = False
        lines.append("   [FAIL] unexpected diffs")
    # baseline must not already carry the key
    if "lambda_det" in orig.get("train", {}):
        ok = False
        lines.append("   [FAIL] base config already sets lambda_det")

out = ROOT / "experiments/phase4a/phase4C_probeC_config_audit.txt"
out.write_text("Probe C config audit - only train.lambda_det may differ\n"
               "=======================================================\n"
               + "\n".join(lines) + "\n\nRESULT: " + ("PASS" if ok else "FAIL") + "\n")
print("\n".join(lines))
print("\nRESULT:", "PASS" if ok else "FAIL")
