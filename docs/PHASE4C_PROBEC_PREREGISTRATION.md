# Probe C — Gradient-Magnitude Rebalancing (H-11)

**Pre-registered before any probe C number exists.** The runner had not been
started when this was written; the only numbers below come from P4A-STEP5
(already committed) and from the external Phase 2-C noise floor.

---

## 1. Why this probe exists

Three separate observations in this project have no common explanation yet:

1. DA is essentially insensitive to Z width (0.21× noise, Phase 4A §3).
2. Lane is only weakly sensitive (1.53× noise at 20ep, z16→z32).
3. Widening Z only ever moves detection.

Probe A tested an *architectural* explanation (per-task projection) and was
null. Probe B is testing a second architectural explanation (receptive field of
the reconstruction). Probe C tests an **optimisation** explanation, which is
orthogonal to both: it leaves the architecture untouched.

The measurement that motivates it, from P4A-STEP5 (`phase4A_gradient_diagnostic.csv`,
20ep checkpoints, 8 batches):

| cell | ‖∂L_det/∂Z‖ | ‖∂L_da/∂Z‖ | ‖∂L_lane/∂Z‖ | det share | cos(det,da) | cos(det,lane) |
|---|---|---|---|---|---|---|
| r2_z16 | 0.03080 | 0.00587 | 0.00540 | **0.732** | −0.0017 | +0.0001 |
| r2_z32 | 0.02499 | 0.00322 | 0.00427 | **0.769** | +0.0007 | +0.0005 |

Detection owns ≈75% of the pull on the shared Z, and the gradients are
**orthogonal** (cos ≈ 0), so this is not gradient *conflict*. GradNorm
(Chen et al., ICML 2018) is explicitly built on the claim that magnitude
imbalance alone is enough for one task to dominate the shared parameters —
that the fix is to equalise gradient magnitudes, and that conflict is not a
precondition. Our numbers are a textbook instance of the situation it describes.

**H-11: the shared Z is being monopolised by detection, and this — not Z width
and not head capacity — is why DA and lane do not benefit from Z.**

If H-11 is right, it also supplies a *competing* explanation for probe A's null:
giving lane a wider projection is useless if lane barely pushes Z in the first
place.

## 2. Intervention

`train.lambda_det: 0.2`, `lambda_da = lambda_lane = 1.0`. Architecture, Z width,
schedule and seed are byte-identical to the R2 baseline apart from this one key
(asserted by `scripts/phase4c_rw_make_configs.py`, output in
`experiments/phase4a/phase4C_probeC_config_audit.txt`).

Predicted effective shares with λ_det = 0.2, using the measured norms above:

| cell | det | da | lane |
|---|---|---|---|
| rw_z16 | 0.353 | 0.337 | 0.310 |
| rw_z32 | 0.400 | 0.258 | 0.342 |

**Why scale detection down rather than scale DA/lane up.** Adam normalises each
parameter's update by the running RMS of its gradient, so a *uniform* scale on
the whole loss is largely absorbed. Scaling det down therefore reallocates the
*direction* of the shared-trunk update without inflating the total loss scale,
and it leaves the detection head's own effective learning rate nearly unchanged
(its parameters receive only detection gradient, so the update is
scale-invariant). Scaling DA/lane up 5× would reach the same relative shares but
would also multiply the total loss by ≈2.

## 3. Cells

| cell | Z | epochs | seed | notes |
|---|---|---|---|---|
| `rw_z16` | 16 | 20 | 0 | new |
| `rw_z32` | 32 | 20 | 0 | new |

**Baselines are not retrained.** `r2_z16` and `r2_z32` at 20ep / seed 0 already
exist in `experiments/phase4a/phase4A_results.csv` and are the *same model* with
`lambda_det = 1.0`. Reuse is legitimate here in a way it was not for probe A:
the epoch count and seed match exactly.

Not 4 epochs. Probe A's null was read at 4ep, where mAP50 sits at 68% of its
20ep value; running the next probe at the same under-trained point would
reproduce the same power problem. 20ep is the protocol every Phase 4A main
comparison used.

## 4. Readability threshold

External noise floor from Phase 2-C (3 seeds @ 4ep), used unchanged:

| metric | 1× noise | 2× noise |
|---|---|---|
| mAP50 | 0.0073 | 0.0146 |
| mAP50_95 | 0.0032 | 0.0064 |
| da_mIoU | 0.0142 | 0.0284 |
| da_fg | 0.0202 | 0.0404 |
| lane_mIoU | 0.0021 | 0.0042 |
| lane_fg | 0.0032 | 0.0064 |

This probe is **single-seed** and estimates no variance of its own, so only
effects at **≥ 2× noise** are treated as readable. Effects between 1× and 2× are
recorded as *unresolved*, not as results.

## 5. Predictions

Baselines (R2, 20ep) and their z-gap:

| metric | r2_z16 | r2_z32 | gap | gap in noise units |
|---|---|---|---|---|
| mAP50 | 0.3543 | 0.3553 | +0.0010 | 0.14× |
| mAP50_95 | 0.1273 | 0.1248 | −0.0025 | −0.78× |
| da_mIoU | 0.8488 | 0.8573 | +0.0085 | 0.60× |
| da_fg | 0.7612 | 0.7736 | +0.0124 | 0.61× |
| lane_mIoU | 0.5847 | 0.5875 | +0.0028 | 1.33× |
| lane_fg | 0.1943 | 0.1992 | +0.0049 | 1.53× |

**C1 (primary).** Rebalancing gives a segmentation task a real gain at fixed z:
on at least one arm, `lane_fg` beats the R2 baseline at the same z by ≥ +0.0064
(2× noise), **or** `da_mIoU` by ≥ +0.0284, **or** `da_fg` by ≥ +0.0404.
Lane is the likelier of the two — it has the most headroom (fg IoU 0.19) and it
receives the smallest share of the Z gradient.

**C2 (the interaction that actually discriminates).** Under rebalancing, DA and
lane become *z-sensitive*: the `rw_z32 − rw_z16` gap for lane_fg reaches
≥ +0.0064 and/or da_mIoU reaches ≥ +0.0284, i.e. it clears 2× noise where the
R2 gap only reached 1.53× and 0.60×. This is the strongest form of the
hypothesis — it says Z now carries information for the tasks that were being
starved, not merely that their heads got a better-conditioned loss.

**C3 (the cost).** Detection gives something up: `mAP50` falls by ≥ 0.0146
(2× noise) on at least one arm. Two readings, both informative:
- det falls → its dominance was load-bearing; the sharing is genuinely a
  zero-sum reallocation, and the final model must trade det against seg.
- det does not move at all → dominance was *not* buying detection anything, and
  rebalancing is close to a free gain for the segmentation tasks.

**C4 (manipulation check — gates everything).** Recompute per-task gradient
norms on the two new checkpoints (`scripts/phase4c_probeC_diag.py`, no
training). The check passes only if detection's **effective** share
(0.2 × measured ‖∂L_det/∂Z‖, renormalised against the unweighted da/lane norms)
is ≤ 0.50, down from 0.73–0.77. If the intrinsic norms shift during rebalanced
training such that detection still owns > 50%, the intervention did not do what
this document claims it does and **no scientific conclusion may be drawn** — the
run is reported as an invalid manipulation, not as a null result.

## 6. Decision rule (fixed before the run)

| outcome | verdict | next step |
|---|---|---|
| **C4 fails** | INVALID | report as a failed manipulation; do not touch this axis again without a different intervention (true GradNorm / uncertainty weighting) |
| C4 passes, **C1 or C2** holds | **H-11 supported** | the probe-A null may be an optimisation artefact; re-run the best task-projection cell *under balanced gradients* (A × C interaction), and treat loss balancing as a candidate ingredient of the final model |
| C4 passes, **C3** holds, C1 and C2 fail | **H-11 rejected as a lever** | imbalance is real but reallocating gradient does not help the seg tasks — their ceiling is set by what Z *contains*, which points back at probe B's answer |
| C4 passes, nothing reaches 2× noise | **H-11 not binding** | drop this axis; the shared-Z gradient composition is not where the remaining headroom is |

Anything landing between 1× and 2× noise is reported as *unresolved* and calls
for a follow-up on that specific metric, not for a verdict.

## 7. Known limitations, stated in advance

- **Single seed.** No variance estimate. A null here is weaker than a positive.
- **Static weight.** λ_det = 0.2 is fixed from norms measured at the *end* of
  baseline training. The norms early in training are different, so the
  equalisation is approximate by construction. If C1/C2 hold, the natural
  follow-up is a dynamic scheme (GradNorm), not a finer static grid — a grid
  would be fitting noise at 1 seed.
- **Directional, not architectural.** This cannot fix anything that is wrong
  with Z's *content*. If probe B shows the reconstruction's receptive field is
  the binding constraint, probe C and probe B are answering different questions
  and a negative here does not contradict a positive there.
