"""Phase 5 - post-probe orchestration.

Once the three probe cells (l14up_z16, l14f1_z16, lch64_z16) have written
their rows to experiments/phase5/phase5_lane_probe_results.csv, run three
things in order:

  1.  Apply the L1-L4 decision rule (phase5_probe_decide.py). The verdict
      comes from the rule, not from judgement.
  2.  Score the same checkpoints with the error-geometry code path used
      for the Phase 4A cells, so each new cell gets the precision / recall
      / area / tolerance-curve breakdown that turns lane_fg into a two-axis
      story (H5a/H5b vs H17). Uses the env-override seam added to
      phase5_errorgeom.py in 9d5a2a1.
  3.  Fold the result back into phase5_hypothesis_matrix.csv (H5a, H5b,
      H6, H17) and regenerate phase5_bottleneck_profile.csv and
      phase5_intervention_matrix.csv.

Step 2 is GPU work (one forward per cell) but is short - 300 images on a
0.2 M model. It can run in parallel with the next probe cell of Phase 4B
if any is queued.

This script does not invent anything. Each step's logic lives in its own
file; this is just the conductor.
"""
import json
import os
import subprocess
import sys

ROOT = "/home/mycode/ai_study/trac"
PY = "/home/mycode/ai_study/gpu_env/bin/python"

CSV = os.path.join(ROOT, "experiments/phase5/phase5_lane_probe_results.csv")
CHECKPOINTS = [
    ("l14up_z16",  "configs/phase4b_l14up_z16.yaml",
     "experiments/phase4a/exp4B_l14up_z16_e4/checkpoint.pt"),
    ("l14f1_z16",  "configs/phase4b_l14f1_z16.yaml",
     "experiments/phase4a/exp4B_l14f1_z16_e4/checkpoint.pt"),
    ("lch64_z16",  "configs/phase4b_lch64_z16.yaml",
     "experiments/phase4a/exp4B_lch64_z16_e4/checkpoint.pt"),
]


def step1_decide():
    print("== step 1: pre-registered L1-L4 decision ==")
    subprocess.check_call([PY, os.path.join(ROOT, "scripts/phase5_probe_decide.py")])


def step2_errorgeom():
    cells = json.dumps([(c, c, cfg, ckpt) for (c, cfg, ckpt) in CHECKPOINTS])
    out = os.path.join(ROOT, "experiments/phase5/phase5_lane_probe_errorgeom.csv")
    env = dict(os.environ,
               PHASE5_EG_CELLS=cells,
               PHASE5_EG_OUT=out)
    print("== step 2: error geometry on probe cells ==")
    subprocess.check_call(
        [PY, os.path.join(ROOT, "scripts/phase5_errorgeom.py")],
        env=env,
    )


def step3_fold_in():
    print("== step 3: update hypothesis matrix + bottleneck artifacts ==")
    subprocess.check_call(
        [PY, os.path.join(ROOT, "scripts/phase5_fold_probe.py")]
    )


def main():
    if not os.path.exists(CSV):
        sys.exit("no probe CSV yet: " + CSV)
    step1_decide()
    step2_errorgeom()
    step3_fold_in()


if __name__ == "__main__":
    main()
