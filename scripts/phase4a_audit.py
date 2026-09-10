#!/usr/bin/env python
"""P4A-STEP1 - R2 architecture audit, provenance trace, cost breakdown.

Two things are checked, and the second one is the one that matters:

1. Shapes. Where Z is produced, at what resolution, and what each head consumes.

2. Provenance, by intervention rather than by reading the code. Two forward
   passes are run with independent control of `feats` and `z`:
     - feats-override: keep the real Z, replace the encoder's F2/F3/F4 with
       random noise. A head whose output does NOT move cannot be reading feats.
     - z-override:     keep the real feats, replace Z with random noise.
       A head whose output does NOT move cannot be reading Z.
   A head is Z-only iff it moves under z-override and is inert under
   feats-override. That is a test, not an assertion in a docstring.
"""
import os, sys, copy, json
sys.path.insert(0, "/home/mycode/ai_study/trac")
import yaml, torch
import torch.nn as nn
from models.static_model import StaticMultiTaskModel

ROOT = "/home/mycode/ai_study/trac"
OUT = os.path.join(ROOT, "experiments/phase4a")
DOC = os.path.join(ROOT, "docs/PHASE4A_R2_ARCHITECTURE_AUDIT.md")
TRACE = os.path.join(OUT, "shared_bottleneck_trace.txt")
COST = os.path.join(OUT, "phase4A_cost_breakdown.csv")
os.makedirs(OUT, exist_ok=True)

TPL = {16: "configs/phase3a_ebase_z16.yaml",
       32: "configs/phase3b_ebase_z32.yaml",
       128: "configs/phase3b_ebase_z128.yaml"}
ZS = [16, 32, 128]

torch.manual_seed(0)


def build(cfg):
    flat = dict(cfg["model"])
    flat["input_size"] = cfg["input_size"]
    return StaticMultiTaskModel(flat)


def to_r2(cfg):
    c = copy.deepcopy(cfg)
    c["model"]["detection"]["from_z"] = True
    c["model"]["detection"]["z_proj"] = True
    c["model"]["detection"]["det_ch"] = 32
    return c


def manual_forward(m, x, z_ov=None, feats_ov=None):
    """Replicates StaticMultiTaskModel.forward with feats and z decoupled."""
    feats = m.encoder(x)
    z = m.representation(feats)["z"]
    zz = z if z_ov is None else z_ov
    ff = feats if feats_ov is None else feats_ov
    hw = x.shape[2:]
    if m.det_from_z:
        det = m.det_head(zz, hw)
    else:
        det = m.det_head([ff[0], ff[1], ff[2]])
    da = m.da_head(zz, hw)
    lane = m.lane_head(zz, hw)
    return det, da, lane, z, feats


def flat_diff(a, b):
    """Max |a-b| over a possibly nested output structure."""
    if isinstance(a, (list, tuple)):
        return max(flat_diff(x, y) for x, y in zip(a, b))
    return float((a.float() - b.float()).abs().max())


def params_by_module(m):
    agg = {}
    for n, p in m.named_parameters():
        top = n.split(".")[0]
        agg[top] = agg.get(top, 0) + p.numel()
    return agg


def flops_by_module(m, x):
    counts = {}
    hooks = []

    def make(name):
        def f(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                cin = inp[0].shape[1]
                h, w = out.shape[2], out.shape[3]
                k = mod.kernel_size[0] * mod.kernel_size[1]
                macs = (cin // mod.groups) * mod.out_channels * k * h * w
                counts[name] = counts.get(name, 0) + macs
        return f

    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(make(name)))
    was = m.training
    m.eval()
    with torch.no_grad():
        m(x)
    m.train(was)
    for h in hooks:
        h.remove()
    agg = {}
    for k, v in counts.items():
        agg[k.split(".")[0]] = agg.get(k.split(".")[0], 0) + v
    return agg  # MACs; FLOPs = 2 * MACs


rows = []
lines = []
md = []


def L(s=""):
    lines.append(s); print(s)


L("=" * 100)
L("PHASE 4A · SHARED-BOTTLENECK PROVENANCE TRACE")
L("=" * 100)
L()
L("Method: intervention, not inspection. For each model we run three forwards")
L("on the same input: baseline; Z replaced by N(0,1) noise with the encoder")
L("features kept real; encoder F2/F3/F4 replaced by N(0,1) noise with Z kept real.")
L("A head is Z-only iff it MOVES under the z-override and is INERT under the")
L("feats-override.")
L()

MODULES = ["encoder", "representation", "det_head", "da_head", "lane_head"]

for z in ZS:
    cfg0 = yaml.safe_load(open(os.path.join(ROOT, TPL[z])))
    cfg2 = to_r2(cfg0)
    for tag, cfg in (("R0", cfg0), ("R2", cfg2)):
        m = build(cfg)
        m.eval()
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            det, da, lane, zr, feats = manual_forward(m, x)
            zov = torch.randn_like(zr)
            fov = [torch.randn_like(f) for f in feats]
            det_z, da_z, lane_z, _, _ = manual_forward(m, x, z_ov=zov)
            det_f, da_f, lane_f, _, _ = manual_forward(m, x, feats_ov=fov)
        d_det_z = flat_diff(det, det_z)
        d_da_z = flat_diff(da, da_z)
        d_lane_z = flat_diff(lane, lane_z)
        d_det_f = flat_diff(det, det_f)
        d_da_f = flat_diff(da, da_f)
        d_lane_f = flat_diff(lane, lane_f)

        pbm = params_by_module(m)
        fbm = flops_by_module(m, x)

        L("-" * 100)
        L("  %s  z=%d   det_from_z=%s" % (tag, z, m.det_from_z))
        L("-" * 100)
        L("    Z              : shape %s   (1/%d of input)" % (
            tuple(zr.shape), 640 // zr.shape[2]))
        L("    encoder feats  : F2 %s (1/8)   F3 %s (1/16)   F4 %s (1/32)" % (
            tuple(feats[0].shape), tuple(feats[1].shape), tuple(feats[2].shape)))
        L("    det pyramid    : %s" % (
            "Z(1/8) -> down -> 1/16 -> down -> 1/32  [built BY DOWNSAMPLING Z]" if m.det_from_z
            else "F2/F3/F4 read directly from the encoder"))
        L()
        L("    %-8s %-22s %14s %14s   %s" % ("head", "override", "max|delta|", "threshold", "reads?"))
        THR = 1e-6
        for hname, dz, df in (("detection", d_det_z, d_det_f),
                              ("DA", d_da_z, d_da_f),
                              ("lane", d_lane_z, d_lane_f)):
            L("    %-8s %-22s %14.3e %14.0e   %s" % (
                hname, "Z -> noise", dz, THR, "Z" if dz > THR else "NOT Z"))
            L("    %-8s %-22s %14.3e %14.0e   %s" % (
                hname, "encoder feats -> noise", df, THR,
                "BYPASS (reads feats)" if df > THR else "clean"))
        L()
        verdicts = {}
        for hname, dz, df in (("detection", d_det_z, d_det_f),
                              ("DA", d_da_z, d_da_f),
                              ("lane", d_lane_z, d_lane_f)):
            if dz > THR and df <= THR:
                v = "Z-only"
            elif dz > THR and df > THR:
                v = "Z + raw feats (HYBRID)"
            elif dz <= THR and df > THR:
                v = "BYPASS Z (reads encoder only)"
            else:
                v = "depends on neither (suspect)"
            verdicts[hname] = v
        L("    provenance verdict: detection=%s  DA=%s  lane=%s" % (
            verdicts["detection"], verdicts["DA"], verdicts["lane"]))
        shared = all(v == "Z-only" for v in verdicts.values())
        L("    STRICT SHARED BOTTLENECK: %s" % ("YES" if shared else "NO"))
        L()
        L("    params: " + "  ".join("%s=%d" % (k, pbm.get(k, 0)) for k in MODULES))
        L("    params total %d" % sum(pbm.values()))
        L("    MACs   : " + "  ".join("%s=%.3fM" % (k, fbm.get(k, 0) / 1e6) for k in MODULES))
        L("    MACs total %.3f M  (= %.3f GFLOPs)" % (
            sum(fbm.values()) / 1e6, 2 * sum(fbm.values()) / 1e9))
        L()
        rows.append(dict(variant=tag, z=z,
                         det_from_z=str(bool(m.det_from_z)),
                         det_verdict=verdicts["detection"],
                         da_verdict=verdicts["DA"],
                         lane_verdict=verdicts["lane"],
                         strict_shared="yes" if shared else "no",
                         z_shape=str(tuple(zr.shape)),
                         **{"params_" + k: pbm.get(k, 0) for k in MODULES},
                         params_total=sum(pbm.values()),
                         **{"macs_M_" + k: round(fbm.get(k, 0) / 1e6, 4) for k in MODULES},
                         gflops=round(2 * sum(fbm.values()) / 1e9, 4)))
        del m

with open(TRACE, "w") as f:
    f.write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- cost csv
if rows:
    cols = ["variant", "z", "det_from_z", "det_verdict", "da_verdict", "lane_verdict",
            "strict_shared", "z_shape"] + ["params_" + k for k in MODULES] + \
           ["params_total"] + ["macs_M_" + k for k in MODULES] + ["gflops"]
    with open(COST, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

# ---------------------------------------------------------------- markdown doc
r0 = {r["z"]: r for r in rows if r["variant"] == "R0"}
r2 = {r["z"]: r for r in rows if r["variant"] == "R2"}

md.append("# Phase 4A · R2 Architecture Audit")
md.append("")
md.append("Generated by `scripts/phase4a_audit.py`. Provenance is established by")
md.append("**intervention**: each head is re-run with Z replaced by noise and, separately,")
md.append("with the encoder features replaced by noise. A head that moves when Z moves and")
md.append("stays still when the encoder features move is Z-only. That is a measurement, not")
md.append("a claim about what the code says.")
md.append("")
md.append("---")
md.append("")
md.append("## 1. Information flow")
md.append("")
md.append("```")
md.append("R0   Encoder -> [F2,F3,F4] -> Detection")
md.append("     Encoder -> Z -> DA")
md.append("     Encoder -> Z -> Lane")
md.append("")
md.append("R2   Encoder -> Z -> reconstruction pyramid -> Detection")
md.append("     Encoder -> Z -> DA")
md.append("     Encoder -> Z -> Lane")
md.append("```")
md.append("")
md.append("`Z` is produced by `CompactRepresentation`: 1x1 lateral convs on F2/F3/F4,")
md.append("bilinear upsample, add, BN, ReLU. It lives at **1/8 resolution** with")
md.append("`z_channels` channels.")
md.append("")
md.append("## 2. Provenance table")
md.append("")
md.append("| variant | z | task | input feature | comes from Z? | bypass? | spatial scale | channels |")
md.append("|---|---|---|---|---|---|---|---|")
for z in ZS:
    for tag, src in (("R0", r0), ("R2", r2)):
        r = src.get(z)
        if not r:
            continue
        zc = r["z_shape"].split(",")[1].strip() if "," in r["z_shape"] else "?"
        scales = {"detection": ("1/8, 1/16, 1/32", zc), "DA": ("1/8", zc), "lane": ("1/8", zc)}
        for task, key in (("detection", "det_verdict"), ("DA", "da_verdict"), ("lane", "lane_verdict")):
            v = r[key]
            fromz = "yes" if v == "Z-only" else ("partial" if "HYBRID" in v else "no")
            bypass = "no" if v == "Z-only" else "YES"
            sp, ch = scales[task]
            md.append("| %s | %d | %s | %s | %s | %s | %s | %s |" % (
                tag, z, task,
                "Z" if fromz == "yes" else ("Z + encoder F2/F3/F4" if fromz == "partial" else "encoder F2/F3/F4"),
                fromz, bypass, sp, ch))
md.append("")
md.append("**Result.** R0's detection head is a genuine bypass: it is inert when Z is")
md.append("replaced by noise and moves when the encoder features are. R2 removes it - all")
md.append("three heads are Z-only, so R2 is a strict shared bottleneck.")
md.append("")
md.append("## 3. Structural caveat that is not a bypass but is a real cost")
md.append("")
md.append("Z sits at 1/8. The detection head needs 1/8, 1/16 and 1/32. R2 builds the two")
md.append("coarser scales by **downsampling Z**, and Z itself was built by **upsampling**")
md.append("F3 and F4 to 1/8. The 1/32 detection scale therefore travels")
md.append("")
md.append("```")
md.append("F4 (1/32) -> 1x1 -> upsample to 1/8 -> add into Z -> downsample to 1/32")
md.append("```")
md.append("")
md.append("This round trip is not a bypass (the path is forced through Z) but it is a")
md.append("plausible source of spatial information loss, and it makes R2's detection result")
md.append("partly a statement about the reconstruction, not only about Z width. This is")
md.append("hypothesis H-07 and it is why the reconstruction probe exists.")
md.append("")
md.append("## 4. Cost breakdown")
md.append("")
md.append("| variant | z | encoder | representation | det head | DA head | lane head | total | GFLOPs |")
md.append("|---|---|---|---|---|---|---|---|---|")
for z in ZS:
    for tag, src in (("R0", r0), ("R2", r2)):
        r = src.get(z)
        if not r:
            continue
        md.append("| %s | %d | %d | %d | %d | %d | %d | %d | %.3f |" % (
            tag, z, r["params_encoder"], r["params_representation"], r["params_det_head"],
            r["params_da_head"], r["params_lane_head"], r["params_total"], r["gflops"]))
md.append("")
if r0 and r2:
    md.append("R2 - R0 at equal z (the reconstruction premium you asked to be made explicit):")
    md.append("")
    md.append("| z | R0 params | R2 params | delta | delta % | where the delta sits |")
    md.append("|---|---|---|---|---|---|")
    for z in ZS:
        a, b = r0.get(z), r2.get(z)
        if not (a and b):
            continue
        d = b["params_total"] - a["params_total"]
        md.append("| %d | %d | %d | %+d | %+.1f%% | detection head (reconstruction, det_ch=32 fixed) |" % (
            z, a["params_total"], b["params_total"], d, d / a["params_total"] * 100))
    md.append("")
    md.append("Because `det_ch` is pinned at 32, R2's reconstruction cost is essentially")
    md.append("constant across the z sweep (only the 1x1 entry projection scales with")
    md.append("`z_channels`). R0 vs R2 at equal z is therefore close to like-for-like, and")
    md.append("the residual premium is reported rather than hidden.")
md.append("")
md.append("## 5. Verdict")
md.append("")
ok = all(r2[z]["strict_shared"] == "yes" for z in ZS if z in r2)
byp = all(r0[z]["det_verdict"] == "BYPASS Z (reads encoder only)" for z in ZS if z in r0)
md.append("- R0 detection bypasses Z: **%s**" % ("confirmed" if byp else "NOT confirmed - investigate"))
md.append("- R2 is a strict shared bottleneck: **%s**" % ("confirmed" if ok else "NO - do not proceed"))
md.append("")
md.append("Proceed to the 4-epoch sanity run only if both are confirmed.")
md.append("")

with open(DOC, "w") as f:
    f.write("\n".join(md) + "\n")

print()
print("written:", TRACE)
print("written:", COST)
print("written:", DOC)
print()
print("R0 detection verdicts:", {z: r0[z]["det_verdict"] for z in ZS if z in r0})
print("R2 strict shared     :", {z: r2[z]["strict_shared"] for z in ZS if z in r2})
