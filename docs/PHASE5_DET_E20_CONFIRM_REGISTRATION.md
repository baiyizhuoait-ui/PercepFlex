# P4B-EXP-03, 20-epoch confirmation of `danc` — preregistration

Written **before** the 20-epoch run is launched. D1 has already passed at
4 epochs (`phase5_det_probe_decision.txt`: +0.1543 mAP50 = 10.57x noise), and
the standing rule in this project is that a 4-epoch pass authorises a 20-epoch
confirmation and nothing more. This document fixes what the confirmation means.

## 1. Why a confirmation is still required at 10x noise

Two findings in this project reversed between 4 and 20 epochs (probe A;
reconstruction depth). A 10x-noise 4-epoch gain is not exempt from that: the
4-epoch budget is exactly where a *faster-converging* configuration looks best,
and a better anchor prior is precisely a faster-convergence intervention. The
confirmation exists to separate "danc is better" from "danc gets there
sooner".

## 2. Baseline

The **committed** Phase 4A 20-epoch row, not a rerun:

| cell | epochs | params | GFLOPs | mAP50 | mAP50_95 | da_fg | lane_fg |
|---|---|---|---|---|---|---|---|
| `r2_z16` (= `r2u_z16`, architecture verified identical) | 20 | 0.2014 M | 1.0796 | **0.3543** | 0.1273 | 0.7612 | 0.1943 |

Source: `experiments/phase4a/phase4A_results.csv`, checkpoint
`experiments/phase4a/exp4A_r2_z16_e20/checkpoint.pt` (the sibling
`..._FAILED_OOM` directory is an earlier aborted attempt and is not used).

The checkpoint is on disk, so the size-stratified recall probe can be re-run
against it later for a 20-epoch D2 without any additional training.

## 3. Cell

| cell | change | epochs | params | GFLOPs |
|---|---|---|---|---|
| `danc_z16` | anchors only: default 9 -> IoU-k-means 9 | 20 | identical | identical |

Seed 0, same as the 4-epoch run and the baseline.

## 4. Decision rule

Noise floors are the external Phase 2-C figures used throughout:
`mAP50 0.0146`, `lane_fg 0.0064`, `da_fg 0.0404`.

- **E1 (primary)** `danc_20` mAP50 - 0.3543 >= 2x noise (+0.0292), i.e.
  mAP50 >= 0.3835 -> the anchor finding is **confirmed at 20 epochs**.
  1x-2x -> WEAK. < 1x -> the 4-epoch result was a convergence-speed artefact
  and H-19 is REJECTED at full budget.
- **E2 (convergence, not a pass/fail gate)** report `danc_20 - danc_4`.
  A negative value means the 4-epoch number overshot; it is reported, not
  corrected.
- **E3 (control)** `lane_fg` and `da_fg` within 2x noise of the 20-epoch
  baseline (0.1943 / 0.7612). Detection-only change must stay detection-only.
- **E4 (cost)** params and GFLOPs identical to baseline within rounding.

E1 is evaluated with `scripts/phase5_det_decide.py` extended for the 20-epoch
comparison; no number is read by hand.

## 5. Standing caveats

- Changing anchors changes **both** the training assignment and the eval-time
  decode. The two are not separable in this experiment; the confirmation does
  not change that.
- The 20-epoch baseline was produced at an older commit. All source changes
  between it and HEAD are flag-gated defaults (`rec="1x1"`, `task_proj` off,
  `p2` off, `highres=False`) and `evaluation/` is byte-identical across the two
  commits, so the comparison is against the same evaluator. This was verified
  by diff, not assumed.
- The noise floor is a Phase 2-C figure and is reused as-is; it is not
  re-estimated at this budget.

## 6. What a confirmation does and does not license

- It licenses **using** k-means anchors in later cells as the default, if the
  user accepts that the project's detection baseline moves.
- It does **not** license further anchor engineering. ATSS (see
  `PHASE5_LITERATURE_REGISTER.md` L6) reports that with a sound assignment rule
  the number and shape of anchors per location barely matters, so the expected
  return on iterating anchor design is small.
