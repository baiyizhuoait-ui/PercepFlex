# H21 / dp2b_z16 — preregistered before running

Registered 2026-09-08 22:4x, before `dp2b_z16` was trained. The rules below are
transcribed into `scripts/phase5_det_decide.py` (section D3b) so that the
verdict is produced mechanically. Nothing here may be changed after the numbers
land.

## Why this cell exists

`dp2a_z16` (D3) lost 0.0447 mAP50 relative to `danc_z16`. That cell varies two
things at once and therefore cannot be interpreted on its own:

1. **grid density** — 25200 -> 102000 anchors, a new stride-4 output level;
2. **stride-4 feature quality** — the new level was fed by bilinearly upsampling
   Z, and Z is a 16-channel bottleneck. Eight-fold upsampling of a 16-channel
   code is close to a blank high-resolution grid: it adds places to predict from
   without adding the shallow semantics that make those places useful.

`dp2b_z16` holds (1) fixed and replaces (2): the stride-4 level now takes a 1x1
lateral from encoder stage-0 `f1`, which is a real stride-4 feature map.

- config: `configs/phase4b_dp2b_z16.yaml` = `configs/phase4b_dp2a_z16.yaml`
  with `model.detection.p2: up -> f1`. Nothing else differs.
- cost: 202984 params (+1618 over the 201366 baseline, +0.80%); 4 levels,
  k-means 12 anchors (identical to dp2a); balance `[4.0, 1.0, 0.25, 0.06]`.
- recipe: 4 epochs, seed 0, identical to every other cell in this probe.

## Verdicts

| id | rule | read as |
|---|---|---|
| **D3b** | `dp2b - danc >= +0.0146` (1x noise) | PASS -> H21 supported: stride-4 grid resolution does bind, but only when the level carries real shallow features. Then a 20ep confirmation is required; this is not established at 4ep. |
| | | FAIL or negative -> H21 rejected. |
| **D3b-2** | report `dp2b - dp2a` | This *is* the confound, measured: how much of dp2a's -0.0447 was feature source rather than grid density. |
| **D4b** | `|lane_fg|`, `|da_fg|` move < 2x noise (0.0128 / 0.0808) | A detection-only change must stay detection-only. |
| **D5b** | report params / FLOPs | Cost transparency. |

## Stop rule (written down now, to be enforced later)

If D3b fails, the detection line **closes**. Specifically: no further pyramid
levels, no further anchor redesign, no further P2 variants. Rationale:

- ATSS (L6) predicts the anchor repair is one-shot and that iterating anchor
  design pays almost nothing once assignment is sound.
- L7 documents that a P2 head which injects shallow noise is *expected* to lose
  accuracy (CAA-YOLO: AP-large drops; YOLOv11-P2-CBAM: recall drops), and that
  the published remedy is an attention module on top — a new architecture
  direction, not a bottleneck diagnosis, and out of scope here.
- TriLiteNet (0.15M params, 0.55 GFLOPs) reaches **mAP50 49.6 on BDD100K with
  only P3/P4/P5 and no stride-4 level at all** (L8), while using the same
  k-means auto-anchor trick that produced our +0.1543. The externally credible
  design at this parameter scale has no P2.

That last point is a prior **against** the cell. It is registered before the run
so that a negative result is not later re-described as a surprise, and so that
running the cell anyway is justified only by its ability to separate the
confound — which it is, at the cost of one 4-epoch run.

## What a pass would and would not mean

A pass would show that grid density helps *given real shallow features*, i.e.
that D3's failure was a feature-source artefact. It would **not** show that
detection is resolution-bound in the way lane is: lane's bottleneck was proven
geometrically (1/8 block-fill ceiling, 1-2 px targets), whereas detection's
dominant channel so far is supervision (D2: 72% of the gain landed on medium
boxes). Any pass at 4ep still requires a 20ep confirmation — two findings in
this project reversed between 4 and 20 epochs.
