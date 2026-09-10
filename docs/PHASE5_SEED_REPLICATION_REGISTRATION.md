# P5-STEP7c — seed replication of l14f1_z16 e20
## Registration, committed BEFORE the runs

Date: 2026-09-08 15:20 CST. Seeds 1 and 2 do not exist yet.

## Question

P5-STEP7b landed lane_fg +0.0249 (1.95x noise) — WEAK SUPPORT, missing
the CONFIRMED bar (+0.0256 = 2x) by 0.0007. Two more seeds decide
whether 1.95x is seed noise or a true effect just under the bar.

## Design

- Cell: l14f1_z16, 20 epochs, config bit-identical to the seed-0 run
  (`phase4b_l14f1_z16.yaml`), seeds 1 and 2, sequential on one GPU.
- Comparison target stays the committed r2_z16 e20 row (0.1943) — the
  baseline is NOT retrained per seed; the registered noise floor
  (2x external, 0.0128) remains the yardstick.
- Output dirs get `_s1` / `_s2` suffixes (runner patched for this);
  the seed-0 checkpoint is not touched.

## Decision rule, fixed now

d_i = lane_fg(seed i) - 0.1943, for i in {0, 1, 2}; d_mean = mean.

- **d_mean >= +0.0256** -> H-05b CONFIRMED at 20 epochs. P5-STEP7 closes;
  the 1/4 lateral enters the Phase 6 candidate list.
- **+0.0128 <= d_mean < +0.0256** -> stays WEAK. H-05b is recorded as a
  real-but-sub-bar effect; Phase 6 may use it but must say "weak".
- **d_mean < +0.0128** -> NOT CONFIRMED. Seed 0 was favourable noise;
  H-05b reverts to OPEN and the probe EG block-fill crossing is recorded
  as unreplicated.
- Guardrail per seed: det and DA must stay within 2x their noise
  floors of the baseline row in every seed; a violation is recorded as
  task coupling, not a broken cell.
- No seed may be discarded, rerun, or added after seeing its number.
