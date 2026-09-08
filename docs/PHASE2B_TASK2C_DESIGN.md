# Phase 2-B Task 2(c) — R1/R2: does Detection belong in Compact Z?

Date: 2026-09-04 · written BEFORE the runs, so the design is not fitted to results.

## 1. The question

Under R0 the detection head reads the encoder's multi-scale features
[F2(1/8), F3(1/16), F4(1/32)] and **never touches Z**; only DA and lane read Z.
That is fine engineering, but it weakens the paper's central claim: if the hardest
task bypasses Z, then "one compact shared Z carries all three tasks" has not been
tested — Z is only a segmentation neck.

Task 2(c) forces detection through Z and measures what it costs.

| variant | detection reads | entry projection |
|---|---|---|
| R0 (done) | encoder multi-scale features | — |
| R1 | Compact Z | `z_proj: false` (Z used raw) |
| R2 | Compact Z | `z_proj: true` (1x1 conv + BN + ReLU) |

## 2. Pre-flight finding A: R1 and R2 are nearly the same model

`models/representation/det_from_z.py:76-87`:

```python
if proj:      # R2
    Sequential(Conv2d(zc, det_ch, 1), BatchNorm2d(det_ch), ReLU)
else:         # R1
    Sequential(Conv2d(zc, det_ch, 1) if zc != det_ch else Identity(),
               BatchNorm2d(det_ch)   if zc != det_ch else Identity(),
               Identity())
```

So R1 and R2 differ **only by one BN + ReLU**, at every z.
The exception is **z = 32**, where `zc == det_ch == 32` makes R1's entry conv an
`Identity` as well — the only point where R1 genuinely feeds Z in raw.

**Consequence.** The originally planned "R1 vs R2 at several z" is a micro-ablation
whose expected effect is far below the measured noise floor (2 sigma on mAP50 =
0.0063, see §4). Spending ~32 min per point on it cannot produce a usable result.
Revised: R1 is run at **one** point (z=32), last, and is droppable.

## 3. Pre-flight finding B: R2's detection head is 2.5-3.4x LARGER than R0's

`scripts/phase2b_r1r2_paramaudit.py` (params, encoder fixed):

| z | R0 det head | R2 det head | R2/R0 | R0 total | R2 total | delta |
|---|---|---|---|---|---|---|
| 32 | 8,406 | 21,430 | 2.55x | 202,710 | 215,734 | +13,024 |
| 64 | 8,406 | 22,454 | 2.67x | 230,422 | 244,470 | +14,048 |
| 128 | 8,406 | 24,502 | 2.92x | 285,846 | 301,942 | +16,096 |
| 256 | 8,406 | 28,598 | 3.40x | 396,694 | 416,886 | +20,192 |

R0's head is almost free (three 1x1 convs straight off encoder features). R2 must
first project Z and then *build* a pyramid (two 3x3 stride-2 blocks at width 32),
which costs real parameters.

**This determines how the result is read — and the confound runs in our favour:**

- If R2 detection is **worse** than R0, it is worse *despite* having 2.5-3.4x the
  head parameters. That strengthens the conclusion: the bottleneck, not capacity,
  is the cause.
- If R2 detection is **comparable or better**, we CANNOT attribute it to Z being
  sufficient — the bigger head is an alternative explanation. We would then need a
  param-matched control before claiming anything.

Either way the efficiency story is affected: routing detection through Z **costs**
params/FLOPs here, it does not save them.

Cross-check: the audit's 215,734 params for R2 z=32 matches the smoke run's
reported `parameters=215,734` exactly, so the audit and the real model agree.

## 4. Noise floor — what effect size is believable

From Task 2(a) (`docs/PHASE2B_REPRODUCIBILITY.md`):

- Same-config repeat (z=64 vs Step-1 0.235M): **0.0102 da_mIoU**.
- Control metric mAP50 across 6 R0 runs: sigma = 0.0032, **2 sigma = 0.0063**.

**Important caveat for this task.** Under R0, mAP50 was a *control* (detection did
not read Z). Under R1/R2 it *is* the treatment, so we lose the in-run control. We
carry 2 sigma = 0.0063 over as a first-order estimate, but it was measured on a
different gradient path and with only n=6, so it is an estimate, not a measurement.

Decision rule: **only an R0 vs R2 mAP50 gap larger than ~0.0063 may be called an
effect; anything smaller is noise.** A collapse of the size we would expect if Z
cannot carry detection (say 0.24 -> 0.15) is unambiguous regardless.

## 5. Design

Fixed protocol, identical to the R0 sweep: 4 epochs, bs16, AdamW lr 1e-3, cosine,
seed 0, `tri_train`. Encoder fixed; `det_ch` fixed at 32 so head capacity does not
confound the Z sweep.

Z points, **ordered by information value so early stopping discards the least**:

| order | run | why |
|---|---|---|
| 1 | R2 z=32 | R0's saturation point. Is the capacity that suffices for DA/lane enough for detection? |
| 2 | R2 z=128 | 4x that. Does detection simply need more capacity? |
| 3 | R2 z=64 | fills the curve between 32 and 128 |
| 4 | R2 z=256 | upper bracket; finds break-even if 128 is still short |
| 5 | R1 z=32 | the only genuine R1/R2 ablation (§2); droppable |

~32 min/point (measured from the R0 sweep), so ~2h40m for all five.
Runs 1-2 (~1h05m) already answer the headline question.

## 6. Fairness checks passed before launch

- **Anchors identical**: `train.py:36-38`, `det_head.py:25-27`,
  `dynamic_det_head.py:22-24`, `det_from_z.py:42-44` — same 3x3 anchors. No
  anchor confound between R0 and R2.
- **Scale count identical**: DetFromZ emits 3 scales at strides 8/16/32 with the
  same output format as R0's head, so YOLOLoss sees the same shapes.
- **End-to-end path verified**: `experiments/phase2b/smoke.log` — R2 z=32 trained
  1 epoch and evaluated, `missing=0 unexpected=0`, reached `SMOKE_DONE`.
  (mAP50 = 0.0 there is expected at 1 epoch on 160 images.)

## 7. Launch

```bash
cd /home/mycode/ai_study/trac
nohup bash scripts/phase2b_run_r1r2.sh "R2:32,128,64,256" "R1:32" \
  > experiments/phase2b/r1r2_launch.log 2>&1 &
```

Graceful stop at any point boundary: `touch experiments/phase2b/STOP`
Resumes safely: a point whose `*_eval/metrics.json` exists is skipped; a point with
only a checkpoint goes straight to eval. The CSV is appended, never truncated.

## 8. Analysis rules (agreed before seeing data)

- Compare R2 against the R0 row **at the same z** (zsweep_results.csv), never
  against a different z.
- Report the da_mIoU / lane columns too: if detection degrades while DA/lane hold,
  that is the clean signature of a capacity conflict inside Z.
- Do not report an R1 vs R2 difference unless it exceeds 2 sigma = 0.0063.
- If the R0 vs R2 detection gap is large, confirm at least one point with a second
  seed before writing it into the paper.
