# Phase-2-B — Reproducibility audit: how large is run-to-run noise?

**Status: measured, not assumed. This finding constrains every claim Task 2 can make.**

Date: 2026-09-03 · sweep commit range `ce560f4` … `7b71d4b`

---

## 1. The natural repeat experiment

Task 2(a) sweeps `z_channels` over {16, 32, 48, 64, 96, 128} with the encoder fixed.
Step-1's `0.235M` reference row used **`z_channels: 64`** with the same encoder geometry.
Our z=64 sweep point is therefore the *same model* under the *same protocol* — an
unplanned but genuine repeat experiment.

| | Step-1 `0.235M` | this sweep `z=64` | difference |
|---|---|---|---|
| Params | 0.230 M | 0.230 M | **identical** |
| FLOPs | 1.47 G | 1.473 G | identical (rounding) |
| mAP50 | 0.2414 | 0.2424 | +0.0010 |
| **da_mIoU** | **0.8360** | **0.8258** | **−0.0102** |
| lane_fg | 0.1843 | 0.1836 | −0.0007 |

Configs were diffed field by field and are semantically identical: encoder
`stem 16 / stages [32,64,96,128] / blocks [2,2,2]`, `z_channels: 64`,
`nc: 1`, `segmentation.hidden: 32`, `input_size [640,640]`, `stage: A`,
`train_split: tri_train`, `batch_size 16`, `epochs 4`, `lr 1e-3`,
`weight_decay 5e-4`, `lambda_da/lane 1.0`, `grad_clip 10.0`, `num_workers 2`.

**So a same-config, same-seed repeat differs by 0.0102 da_mIoU.**

## 2. Where the nondeterminism comes from

- `training/train.py:262` sets `torch.manual_seed(args.seed)` and `:349` seeds the
  DataLoader generator — shuffling is reproducible.
- `training/train.py:357` **auto-raises `num_workers` from 2 to 8** on this box
  (`cores=24`). The value is stable on one machine but is *machine dependent*, so a
  run is not portable across hosts.
- `training/train.py` sets **neither `torch.backends.cudnn.deterministic` nor
  `torch.use_deterministic_algorithms`**. The YOLO detection loss uses indexed
  scatter/atomic CUDA ops, which are not bit-reproducible; over 4 epochs the
  resulting gradient differences compound into the observed 0.01 mIoU.
- `datasets/*.py` uses no `random` / `numpy.random`, so augmentation is not an
  additional unseeded source.

## 3. Consequence: the sweep is underpowered

Observed da_mIoU across the sweep: **0.8199 → 0.8317 → 0.8328 → 0.8258** (z = 16/32/48/64).
The series is **non-monotonic** — z=64 falls back *below* z=32 and z=48.

- Total spread across z ≥ 32: **0.0070**
- Spread across all four points: **0.0129**
- Measured repeat noise (§1): **0.0102**

The signal we are trying to resolve is therefore **~1.3× the repeat noise**. Under a
single seed the Z → da_mIoU relation is **not resolvable above z = 16**.

What *is* supported by the data:

- **z = 16 is measurably worse than z ≥ 32** for DA (0.8199 vs ~0.830, a gap of
  ~0.010, i.e. at the scale of the repeat noise but consistent in direction).
- **Above z = 32 there is no resolvable gain** for any of the three tasks.

What is **not** supported and must not be claimed:

- ~~"DA saturates at z = 32"~~ — the 32 → 48 delta is +0.0011, an order of magnitude
  below the noise floor; and z = 64 moves the other way.
- Any ranking among z ∈ {32, 48, 64, 96, 128}.

Detection behaves as designed: under R0 the detection head reads multi-scale encoder
features and never touches Z, so its variation across the sweep
(0.2368 / 0.2421 / 0.2360 / 0.2424, σ ≈ 0.0034) is a **control** — see the analyzer's
`--noise-control` flag.

## 4. Latency is affected by the same class of problem

In-sweep p50 latency: z=16 **7.256 ms**, z=32 **2.412 ms**, z=48 **2.664 ms**,
z=64 **5.810 ms** — for models whose FLOPs differ by only 1.39×. z=16 is the
known cold-clock outlier (see commit `ebe0248`), but z=64 at 5.81 ms shows that even
evals run right after training are not stable. **No in-sweep latency/FPS number is
usable**; all checkpoints must be re-benchmarked with
`scripts/phase2b_rebench_latency.py` before any latency claim.

## 5. Required actions

1. **Do not report per-Z-point rankings from single-seed runs.** Report the shape
   (z=16 deficient; z ≥ 32 flat) with the noise floor stated alongside.
2. **Re-run the decisive points with ≥ 3 seeds.** Minimum viable set: z ∈ {16, 32, 128}
   × seeds {0, 1, 2} = 9 runs ≈ 5.4 h GPU at ~36 min/run. Full 6 points × 3 seeds is
   ~10.8 h. Report mean ± std; only differences exceeding ~2σ may be called effects.
3. **Optionally make training reproducible** so future single runs are meaningful:
   set `torch.use_deterministic_algorithms(True)` and
   `torch.backends.cudnn.deterministic = True`, pin `num_workers` explicitly via
   `--num-workers`, and add a `worker_init_fn` seeding `random`/`numpy` per worker.
   This typically slows training; it should be measured before adopting it.
4. **Re-benchmark latency** for all checkpoints under one consistent warm protocol.

## 6. Silver lining

This is a *negative-control* result, and it is exactly what a rigorous capacity study
needs: it tells us the measurement instrument's resolution before we interpret the
curve. Discovering it now is far cheaper than discovering it in review. It also
directly informs Task 2(c): if we route detection through Z (R1/R2) and then see a
change, we now know the size a change must exceed to be believed.
