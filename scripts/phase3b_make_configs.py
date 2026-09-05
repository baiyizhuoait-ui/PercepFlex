#!/usr/bin/env python
"""
Phase 3B · generate the six new configs from the Phase 3A templates.

The Phase 3A templates are ALREADY verified (config diff against
configs/phase2b_zsweep_z16.yaml: only encoder.stem / encoder.stages /
epochs differ). Phase 3B must vary one more axis: representation.z_channels.

Rather than hand-writing YAML (risk of typos / drift), we load the matching
Phase 3A template, mutate exactly ONE field (z_channels), and dump it back.
This makes "only encoder + Z change" true BY CONSTRUCTION, which the audit
script then verifies independently.
"""
import os
import sys

import yaml

ROOT = "/home/mycode/ai_study/trac"
CFGDIR = os.path.join(ROOT, "configs")

ENCODERS = {
    "esmall": dict(stem=12, stages=[24, 40, 64, 80], scale="x0.65"),
    "ebase": dict(stem=16, stages=[32, 64, 96, 128], scale="x1.00"),
    "elarge": dict(stem=20, stages=[40, 80, 128, 168], scale="x1.30"),
}

NEW_Z = [32, 128]

HEADER = """# Phase 3B · Encoder x Z Interaction Study — {enc_up} + z{z}
#
# Generated from configs/phase3a_{enc}_z16.yaml by
# scripts/phase3b_make_configs.py: the template is loaded and exactly ONE
# field is mutated (representation.z_channels 16 -> {z}). Everything else is
# byte-identical to the Phase 3A cell that already ran.
#
#   encoder  : stem {stem} / stages {stages}  (width scale {scale} of baseline)
#   blocks   : [2,2,2]                      -> depth untouched
#   Z width  : {z}                          -> THE ONLY variable that changed
#   heads    : detection / DA / lane untouched
#   protocol : epochs 20, batch 16, lr 1e-3, AdamW, cosine, seed 0, tri_train
#
# Purpose: test whether the DA/Lane plateau seen at E-large + z16 in Phase 3A
# is (A) genuine task saturation, or (B) a z=16 bottleneck. If widening Z
# revives DA/Lane at E-large more than at E-small, there is an interaction.
"""


def main():
    made = []
    for enc, meta in ENCODERS.items():
        src = os.path.join(CFGDIR, f"phase3a_{enc}_z16.yaml")
        if not os.path.exists(src):
            print(f"[FATAL] missing template {src}")
            return 1
        with open(src) as f:
            base = yaml.safe_load(f)

        # sanity: template really is the encoder we think it is
        got_stem = base["model"]["encoder"]["stem"]
        got_stages = list(base["model"]["encoder"]["stages"])
        if got_stem != meta["stem"] or got_stages != meta["stages"]:
            print(f"[FATAL] template {src} is stem={got_stem} stages={got_stages}, "
                  f"expected {meta['stem']} {meta['stages']}")
            return 1
        if base["model"]["representation"]["z_channels"] != 16:
            print(f"[FATAL] template {src} z_channels is not 16")
            return 1

        for z in NEW_Z:
            cfg = yaml.safe_load(open(src).read())  # fresh copy
            cfg["model"]["representation"]["z_channels"] = z

            dst = os.path.join(CFGDIR, f"phase3b_{enc}_z{z}.yaml")
            hdr = HEADER.format(
                enc_up=enc.upper(), enc=enc, z=z, stem=meta["stem"],
                stages=meta["stages"], scale=meta["scale"],
            )
            with open(dst, "w") as f:
                f.write(hdr)
                yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
            made.append(dst)
            print(f"[ok] wrote {dst}")

    print(f"\n{len(made)} configs written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
