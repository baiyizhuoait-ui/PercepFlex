# 4B-5 lane label widening — preregistered before running

Registered 2026-09-08 23:2x, before `lane8_z16` was trained. Decision rules are
fixed here; the verdict comes from the numbers, not from hindsight.

## Hypothesis (H-new, supervision side)

Phase 4B-2 proved lane position is a *deep-map* property, and Phase 4A/5 error
geometry proved lane is geometrically starved: our lane head trains and is
evaluated on the same ~2px thin mask (datasets/bdd100k.py binarises
`0 < m < 255`), while the model paints ~3.37x the true area (precision ~0.20).
Every comparable BDD100K multi-task model instead trains on a WIDENED line and
evaluates on the thin ground truth:

- HybridNets (ENet-SAD convention): train 8px, val 2px, lanes merged to centre.
- TriLiteNet: train 8px, val 2px.
- Registered as L8 before this run.

So the question is not whether widening is the literature norm — it is — but
whether *for our architecture* training on 8px and evaluating on 2px helps or
hurts lane IoU, given that our model already over-paints by 3.37x. Two
outcomes are possible and both are informative.

## Cell

- config: `configs/phase4b_lane8_z16.yaml` = `configs/phase4b_r2u_z16.yaml`
  (the Phase 4B lane baseline) plus a single `train.lane_train_widen: 8`.
  No architecture change, no parameter change, no detection change.
- 20 epochs, seed 0. Baseline = the committed `r2_z16` 20ep row
  (lane_fg 0.1943) — reused, not rerun, per project convention.
- Implementation verified by smoke test: train=True+widen8 gives 1561 lane fg
  px vs 495 for train=True-nowiden and train=False; the widen applies ONLY to
  training, eval always reads the raw ~2px label.

## Verdicts

| id | rule | read as |
|---|---|---|
| **D5-1** | `|lane8 - baseline| >= 2x noise (0.0128)` on lane_fg | effect exists; sign decides direction |
| **D5-2** | if lane8 > baseline by 2x noise | SUPERVISION SUPPORTED: training on widened lines lifts thin-line IoU → a pure supervision gain at zero architecture cost; carry it as a default recipe change |
| **D5-3** | if lane8 < baseline by 2x noise | widening HURTS our over-painting model → confirms the precision problem is architectural (no supervision fix recovers it); a candidate for the high-res output head (4B-6) |
| **D5-4** | |lane8 - baseline| < 1x noise | widening is neutral at 20ep → not the binding supervision knob |
| **D4b** | `|da_fg|`, `|mAP50|` move < 2x noise | control: a lane-only change must stay lane-only |

## Why this is worth one 20-epoch run

If D5-2 lands, we get a real lane gain that requires no architecture change and
is consistent with every published BDD100K model — a direct hit on a task that
has resisted every capacity-side lever. If D5-3 lands, it would be the first
clean demonstration that lane's problem is *localisation on thin lines*, not
missing supervision, which redirects to the high-resolution output head
(4B-6). Either way a single 20ep run disambiguates two live hypotheses.

No seed replication is registered yet: that is a follow-on only if the effect
is large enough to matter, and the GPU budget for the night is shared with the
detection seed confirmation (higher priority per user decision).
