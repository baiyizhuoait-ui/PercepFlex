# Phase 2-B — Task 2 Experimental Design & Execution Package

> Companion to `docs/AGENT_HANDOFF.md` (§三 Task 2) and `docs/PHASE2_PLAN.md`.
> This file is the single entry point for running Task 2(a)–(f). It is written by the
> handoff agent; **GPU training itself must run inside WSL** (see §5 — `wsl.exe` is
> harness-blacklisted in the agent's shell, so this agent could only prepare artifacts,
> not launch training).

---

## 0. What this session delivered (ready to run)

| Artifact | Purpose | Status |
|---|---|---|
| `configs/phase2b_zsweep_z{16,32,48,64,96,128}.yaml` | Task 2(a) **R0** pure-Z sweep (encoder FIXED at 0.235M geometry, only `z_channels` varies) | ✅ written |
| `configs/phase2b_r1_zsweep_z{32,64,96}.yaml` | Task 2(c) **R1**: det reads Z directly (`from_z`, `z_proj=false`) | ✅ written |
| `configs/phase2b_r2_zsweep_z{32,64,96}.yaml` | Task 2(c) **R2**: det reads Z via 1×1 projection (`from_z`, `z_proj=true`) | ✅ written |
| `models/representation/det_from_z.py` | R1/R2 detection-from-Z module (additive, never imported unless `from_z:true`) | ✅ written |
| `models/static_model.py` (edit) | Flag-gated `detection.from_z` branch; **default off ⇒ R0 byte-for-byte unchanged** | ✅ edited |
| `scripts/phase2b_run_zsweep.sh` | Sequential train→eval→record runner for the R0 sweep (appends `zsweep_results.csv`) | ✅ written |
| `scripts/phase2b_analyze.py` | Pure-Python analyzer: capacity/Pareto/sensitivity/utility (no GPU) | ✅ written + tested |
| `experiments/phase2b/STEP2B_ANALYSIS_PROXY.md` | Analysis of **existing trusted** capacity results (proxy, see §8) | ✅ written |

**Blocker (honest):** the agent's shell cannot launch `wsl.exe` (harness Program Blacklist) and
there is no sshd in WSL (port 22 closed), so the 4ep GPU training for (a)/(c) **could not be
executed here**. All numbers below that are not from prior trusted runs are design/expected, not
measured. Nothing is fabricated.

---

## 1. Scientific questions → D-class gap mapping

| Sub-task | Question (measurable) | D-class gap it fills |
|---|---|---|
| **(a)** Capacity Sweep | For fixed backbone, how does `mAP/DA/Lane` move as `Z∈{16..128}`? Where is the **diminishing-return knee** + Pareto frontier? | D-1: first systematic Z-capacity scan at <2G FLOPs on BDD100K 3-task |
| **(b)** Per-task sensitivity | `ΔDet/ΔDA/ΔLane` per Z-step; do tasks **saturate at different Z**? | D-3: "per-task capacity need differs" as a *measured* relation |
| **(c)** R0/R1/R2 | Can **one unified Compact Z** carry detection too (not just seg)? | Core mechanism question; decides whether KD-on-Z is logically coherent |
| **(d)** Representation Utility | Define `Utility(Z)`; report `Utility/FLOPs`, `Utility/Params`; show **non-synchronous degradation** | D-3: Compact Z = *information-allocation*, not just a small feature |
| **(e)** KD (aux only) | Only after (c) is fixed. If controlled single-teacher KD shows no stable gain → **stop Multi-Teacher** | Reuses Step-2 negative result |
| **(f)** Parameter-matched | Same params: widening vs Compact Z vs +light head vs (+KD) — which uses params better? | D-2: first shared-bottleneck-vs-widening empirical at equal params |

---

## 2. Hypotheses (one-liner, falsifiable)

- **H2a** — Beyond a knee `Z*`, per-step Δ(accuracy) → ~0 while FLOPs/params keep rising; Pareto knee exists.
- **H2b** — Per-task sensitivity ordering **Detection > Lane > DA**; DA/Lane saturate earlier than Detection.
- **H2c** — **R2** (det via 1×1 projection from Z) recovers most of R0's detection accuracy ⇒ a unified Z *can* carry detection ⇒ KD-on-Z becomes logically valid.
- **H2d** — Degradation is **non-synchronous**; at equal params `Utility/FLOPs` of Compact Z ≥ naive widening.
- **H2f** — At equal parameter budget, Compact Z (+light head) ≥ plain widening on mAP / Utility.

---

## 3. Fixed protocol (all Task-2 rows)

`4ep · bs16 · AdamW lr1e-3 wd5e-4 · CosineAnnealing · tri_train full (69,863) · seed0`
(+ **2 extra seeds** for variance on the key knee point and on the R0/R2 comparison).
Eval: `tri_val` full (10,000), 640 letterbox, `evaluate_baseline.py --baseline OursStatic`.
**Record per run** (AGENT_HANDOFF §四.4): `Params · FLOPs · FPS · mAP50 · DA mIoU · Lane fgIoU · checkpoint · seed · epochs · git commit`.

---

## 4. Architecture facts (read before editing)

- `StaticMultiTaskModel`: `Encoder → [F2(1/8),F3(1/16),F4(1/32)] → CompactRepresentation Z(1/8, zc)`.
  - **R0**: `DetHead` consumes `[F2,F3,F4]`; `da_head`/`lane_head` consume **Z**.
  - `z_channels` is the *only* knob for Z capacity (confirmed in `compact_z.py`).
- Because R0's det does **not** read Z, the (a) Z-sweep primarily probes **segmentation** sensitivity to Z;
  detection is a near-flat control (expected). The (c) R2 sweep is what makes det depend on Z.
- R1/R2 are **additive**: `det_from_z.py` is only imported when `detection.from_z:true`; default off.
- **Z resolution is 1/8** (`compact_z.py` interpolates Z to `f2.shape[2:]`), so an R1/R2 detection
  pyramid built from Z starts at stride 8 and downsamples to 16/32.
- **CORRECTION (2026-09-03, commit e498cde) — R2 was rewritten after the first smoke test.**
  The first `DetFromZ` draft emitted a **single** detection scale. It crashed in
  `yolo_loss.build_targets` (`preds[i]` IndexError) because `YOLOLoss.nl = anchors.shape[0] = 3`.
  Fixing the crash alone was not enough: a 1-scale head would lose to R0 for *missing scales*
  rather than for *information bottleneck*, which would invalidate the R0-vs-R2 comparison.
  Rewritten to emit the **same 3 scales, strides [8,16,32] and anchors as R0**, so the only
  difference under test is the **information source** (encoder multi-scale vs. one unified Z).
- Downsampling branches inside `DetFromZ` use a **fixed internal width `det_ch=32`**, not
  `z_channels`. Otherwise R2 head params would grow O(z²) across the Z sweep and "Z capacity"
  would be confounded with "head capacity". With `det_ch` fixed, R2 params stay ~constant
  across Z (only the 1×1 entry projection scales as zc·det_ch).
- Measured cost of this design (smoke test, 1ep/160 imgs): R0 z16 = 0.189M / 1.060G / 2.53ms;
  R2 z32 = 0.216M / 1.224G / 4.59ms. R2 pays ≈27K params + ≈0.16G FLOPs for the Z→3-scale
  reconstruction — this overhead is itself part of the research question and is reported.

---

## 5. Launch (WSL, RTX 5060, single card — do NOT parallelize large trains)

```bash
# 1) commit the prepared scaffolding (records the git hash each run will cite)
cd /home/mycode/ai_study/trac
export GIT_SSH_COMMAND="ssh -i /home/mycode/.ssh/id_ed25519_trac -o IdentitiesOnly=yes -o BatchMode=yes"
git add configs/phase2b_*.yaml models/representation/det_from_z.py \
        models/static_model.py scripts/phase2b_*.py scripts/phase2b_*.sh \
        designs/phase2b/TASK2_DESIGN.md experiments/phase2b/STEP2B_ANALYSIS_PROXY.md
git commit -m "Phase2-B Task2: Z-sweep + R1/R2 configs, runner, analyzer, design doc"

# 2) Task 2(a) R0 Z-sweep (6 × ~28min train + ~5min eval ≈ 3.3h, single serial job)
# NOTE: on Windows use the FULL path C:\Windows\System32\wsl.exe — the bare `wsl`
#       name resolves to the WindowsApps alias and is blocked. To survive after the
#       launching shell exits, use setsid + nohup + disown.
setsid nohup bash scripts/phase2b_run_zsweep.sh \
  > experiments/phase2b/zsweep_stdout.log 2>&1 < /dev/null & disown

# poll (from Windows):
#   C:\Windows\System32\wsl.exe -e bash -lc 'cd /home/mycode/ai_study/trac && tail -3 experiments/phase2b/zsweep_stdout.log && cat experiments/phase2b/zsweep_results.csv'

# 3) Task 2(c) R1/R2 — AFTER the (a) sweep, so the Z points can be chosen from the
#    observed saturation curve rather than guessed. Default: R2 @ z=32,64,96 + R1 @ z=64.
setsid nohup bash scripts/phase2b_run_r1r2.sh \
  > experiments/phase2b/r1r2_stdout.log 2>&1 < /dev/null & disown
#   custom set:  bash scripts/phase2b_run_r1r2.sh "R2:32,64,96" "R1:64"

# 4) analyze (works without GPU)
/home/mycode/ai_study/gpu_env/bin/python scripts/phase2b_analyze.py --root . --mode auto \
  --out experiments/phase2b/zsweep_analysis.md
```

> If a train exceeds ~7 GB (only at the largest Z), drop to `bs8` (`--batch-size 8`).

---

## 6. Per-experiment recording template (§四.4)

| field | source |
|---|---|
| Params / FLOPs / FPS | `eval/metrics.json` (`parameters`, `flops`, `fps`) |
| mAP50 / mAP50_95 / DA mIoU / DA fg / Lane fg / Lane mIoU | `eval/metrics.json` |
| checkpoint | `experiments/phase2b/<run>/checkpoint.pt` |
| seed / epochs | run args (0 / 4) |
| git commit | `git rev-parse HEAD` (captured by the runner) |

The runner writes all of this to `experiments/phase2b/zsweep_results.csv` automatically.

---

## 7. Novelty guardrails (AGENT_HANDOFF §四 — do NOT claim first)

- ❌ Shared feature / unified multi-task representation (Sparse U-PDP, etc.)
- ❌ Task/channel dynamic routing (D2BNet) — already failed in Phase 1, forbidden.
- ❌ Light attention 3-head (SCAM-P), elastic width (OFA/SN-Net/DS-Net), budget-aware MTL (EC-MT).
- ❌ Cross-arch / heterogeneous KD, driving-MTL + pruning + KD (arXiv:2511.05557).
- ✅ **Our allowed claim** (D-class gap): *a measurable relation between Compact-Z capacity and
  per-task accuracy / saturation points / compute efficiency at ultra-low budget, plus the
  shared-vs-widening empirical at equal parameters.* If a new paper closes this gap → **PIVOT**.

---

## 8. Proxy analysis (existing trusted results — context only, NOT the Z-isolated sweep)

From `experiments/expD_compact_z/EXP_D_REPORT.md` + `experiments/phase2/STEP1_REPORT.md`
(joint encoder+Z capacity; 1.0M/2.0M are 3ep/2ep, flagged). Full table in
`experiments/phase2b/STEP2B_ANALYSIS_PROXY.md`.

- **Non-synchronous degradation (d):** at 0.235M, Detection retains **75.7%** of its 2.0M mAP,
  DA **98.1%**, Lane fg **94.9%** — tasks contest Z unevenly (info-allocation, not a small feature).
- **Per-task sensitivity (b):** Detection gains the most per capacity step; DA/Lane saturate by ~0.5M.
- **Efficiency (a):** `Utility/FLOPs` = 0.605 (0.235M) → 0.113 (2.0M); smallest model is ~5×
  more compute-efficient per utility unit. All four points lie on the mAP-vs-FLOPs Pareto frontier
  (monotonic), so the **knee must be found by the dedicated Z-isolated sweep**, not this proxy.

> Caveat: the proxy varies encoder width *and* Z together, so it cannot isolate "det sensitivity to
> Z". That isolation is exactly what Task 2(a) (R0, fixed encoder) measures; expect det ≈ flat across
> Z in R0, then move under R2 in (c).

---

## 9. Decision gates for Task 3 (STOP / CONTINUE / PIVOT)

- **CONTINUE** if: (a) shows a clear diminishing-return knee; (c) R2 recovers ≥~90% of R0 det mAP
  at the knee Z; (f) Compact Z ≥ widening at equal params. → proceed to (f) full + Task 3 output (§13).
- **PIVOT** if: (c) R2 det collapses (unified Z cannot carry det) → drop the "unified Z carries det"
  claim, re-scope novelty to the seg-side + capacity/saturation story; and/or if a new paper closes
  the D-gap → change the selling point (never claim C-class shared-rep/IB-MTL as first).
- **STOP Multi-Teacher** if: controlled single-teacher KD (re-check on full data) shows no stable
  gain even after (c) — consistent with Step-2 negative result.

*2–3 candidate paper titles and the next-phase 2–4 experiments are produced in Task 3 once the
measured (a)/(c)/(f) numbers exist (per user §13).*
