#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PercepFlex naming migration v3 — FULL unification incl. historical phases.

v2 -> v3 changes (user decision 2026-09-10: freeze nothing historical):
  * Phase 0-5 UNFROZEN: historical phases are now in scope.
  * NEW  H<n>[suffix]   -> H-<nn>[suffix]     e.g. H1->H-01, H5b->H-05b, H11-alt->H-11-alt
  * NEW  bare STEP<n>   -> P5-STEP<n>         (Phase 5 closure chain steps)
  * NEW  P4B/EXP-0n     -> P4B-EXP-0n         (slash form collapsed to hyphen)
  * frozen list is now the MINIMAL set only (data/binary/tool-self).

Idempotent: a second run must report 0 changes.
Dry-run by default; --apply writes. Use --diff to inspect.

Code-safety note:
  .py/.sh are scanned in FULL (comments, docstrings and string literals all
  transformed) because the audited occurrences are comments, docstrings and
  *internal* dict keys whose rename is behaviour-preserving. Every touched
  .py is py_compile'd and every touched .sh is `bash -n`'d by the caller
  (scripts/verify_naming.py); any syntax failure must be treated as a bug.
"""
import argparse
import csv
import difflib
import fnmatch
import io
import os
import re
import sys

TEXT_EXT = (".md", ".csv", ".py", ".sh", ".yaml", ".yml", ".json", ".txt")

H_LETTER = {chr(ord("A") + i): 22 + i for i in range(14)}  # A->22 .. N->35


def rule_phase_round(m):
    return "Phase 6A" if m.group(1) == "1" else "Phase 6B"


def rule_hyp(m):
    """H1 -> H-01 ; H5b -> H-05b ; H11-alt matched as H11 then '-alt' kept."""
    return "H-%02d%s" % (int(m.group(1)), m.group(2) or "")


RULES = [
    # ---- cross-phase experiment ids (longest-context first) ----
    (re.compile(r"Phase\s+4B-(\d)"), lambda m: "P4B-EXP-0" + m.group(1)),
    (re.compile(r"Phase\s+4C-(\d)"), lambda m: "P4C-EXP-0" + m.group(1)),
    (re.compile(r"\bP4B/EXP-0(\d)\b"), lambda m: "P4B-EXP-0" + m.group(1)),
    (re.compile(r"\bP4C/EXP-0(\d)\b"), lambda m: "P4C-EXP-0" + m.group(1)),
    (re.compile(r"\b4B-(\d)\b"), lambda m: "P4B-EXP-0" + m.group(1)),
    (re.compile(r"\b4C-(\d)\b"), lambda m: "P4C-EXP-0" + m.group(1)),
    # ---- Phase 6 round -> subphase letter ----
    (re.compile(r"Phase\s+6\s*Round[- ]*([12])\b"), rule_phase_round),
    (re.compile(r"\bRound[- ]+([12])\b"), rule_phase_round),
    # ---- zero-pad experiment ids ----
    (re.compile(r"\bEXP-(\d)\b"), lambda m: "EXP-0" + m.group(1)),
    # ---- hypothesis ids: Phase 6 letters -> global numbers ----
    (re.compile(r"\bH-([A-N])\b"), lambda m: "H-%d" % H_LETTER[m.group(1)]),
    # ---- hypothesis ids: Phase 1-5 bare numbers -> canonical dashed/padded ----
    (re.compile(r"\bH(\d{1,2})([a-z]?)\b"), rule_hyp),
    # ---- scheduling aliases -> execution steps ----
    (re.compile(r"\bGPU-([123])\b"), lambda m: "P6A-STEP" + m.group(1)),
    (re.compile(r"\bSTEP\s+([A-D])\b"), lambda m: "P6A-STEP2" + m.group(1).lower()),
    # ---- phase-qualified steps: "Phase 4A STEP 5" -> "P4A-STEP5" ----
    # MUST precede the bare-STEP rule, otherwise the bare rule steals the token
    # and mislabels a Phase-4A/3C/5 step as a Phase-5 closure step.
    (re.compile(r"\b(?:Phase|PHASE)\s+(\d[A-C])\s*[·\-_ ]*\s*STEP\s*(\d+[a-z]?)"),
     lambda m: "P%s-STEP%s" % (m.group(1), m.group(2))),
    (re.compile(r"\b(?:Phase|PHASE)\s+(\d)(?![\dA-C])\s*[·\-_ ]*\s*STEP\s*(\d+[a-z]?)"),
     lambda m: "P%s-STEP%s" % (m.group(1), m.group(2))),
    # ---- Phase 5 own probe steps ("STEP 6", "STEP 7b") ----
    # No other phase numbers its steps >= 6, so these are unambiguous even bare.
    (re.compile(r"(^|[^A-Za-z0-9-])STEP[ _]*([67][a-z]?)\b"),
     lambda m: m.group(1) + "P5-STEP" + m.group(2)),
    # ---- NOTE: bare "STEP 0..5" is NOT handled here ----
    # It is overloaded across phases (Phase 2/3C/4A/4B closure-chain all use it),
    # so it is resolved per-file via SCOPE_PREFIX below. A global rule would
    # mislabel e.g. "Phase 4A STEP 5" as a Phase-5 closure step.
    # ---- surgical: uppercase "ROUND2 STEP1" chain header ----
    (re.compile(r"(?<![A-Za-z0-9-])ROUND\s*-?\s*([12])\s*STEP\s*([0-9])\b"),
     lambda m: ("P6A-STEP" if m.group(1) == "1" else "P6B-STEP") + m.group(2)),
    # ---- surgical: Phase-2 status marker constants (STEP1_BUDGET_CHAIN_DONE ...) ----
    # negative lookbehind is REQUIRED: a plain \b also matches inside the
    # already-migrated "P2-STEP2_..." and would double-prefix it.
    (re.compile(r"(?<![A-Za-z0-9-])STEP([12])_(BUDGET_CHAIN_DONE|SINGLE_KD_DONE|EVAL_DONE)\b"),
     lambda m: "P2-STEP" + m.group(1) + "_" + m.group(2)),
    # ---- gates ----
    (re.compile(r"\bGATE-?([12])\b"), lambda m: "GATE-6A." + m.group(1)),
]


# --------------------------------------------------------------------------- #
# Bare "STEP <0..5>" is phase-ambiguous. Resolve it from the file's owning phase.
# Every entry is evidence-backed (the file's own phase banner / directory).
SCOPE_PREFIX = [
    ("experiments/phase2/", "P2-STEP"),
    ("experiments/phase3c/", "P3C-STEP"),
    ("experiments/phase4a/", "P4A-STEP"),
    ("experiments/phase4b/", "P4B-STEP"),
    # Phase 4B->5 closure chain (danc seeds / dp2b / lane8 / da14 all live here)
    ("experiments/phase5/", "P4B5-STEP"),
    ("docs/PHASE2", "P2-STEP"),
    ("docs/PHASE3C", "P3C-STEP"),
    ("docs/PHASE4A", "P4A-STEP"),
    ("docs/PHASE4B", "P4B-STEP"),
    ("docs/PHASE4BC", "P4B-STEP"),
    ("docs/PHASE4C", "P4C-STEP"),
    ("docs/PHASE5", "P4B5-STEP"),
    ("scripts/clean_master_chain.sh", "P4B5-STEP"),
    ("scripts/phase1b_night_master_chain.sh", "P4B5-STEP"),
    ("scripts/phase4b_recover_step2_then_chain.sh", "P4B5-STEP"),
    ("scripts/phase4b_run.sh", "P4B-STEP"),
    # analysis scripts carry the same step vocabulary as their phase docs
    ("scripts/phase3c_", "P3C-STEP"),
    ("scripts/phase4a_", "P4A-STEP"),
    ("scripts/phase4b_", "P4B-STEP"),
    ("scripts/phase4bc_", "P4B-STEP"),
    ("scripts/phase4c_", "P4C-STEP"),
    ("scripts/phase5_", "P4B5-STEP"),
    ("scripts/phase6_round2_chain.sh", "P6B-STEP"),
    ("scripts/phase6_overnight.sh", "P6A-STEP"),
]


def scope_prefix(rel):
    for pref, lab in SCOPE_PREFIX:
        if rel.startswith(pref):
            return lab
    return None


def transform(text, scope=None):
    n = 0
    for rx, rep in RULES:
        text, k = rx.subn(rep, text)
        n += k
    if scope:
        rx = re.compile(r"(^|[^A-Za-z0-9-])STEP[ _]*([0-5])\b")
        text, k = rx.subn(lambda m: m.group(1) + scope + m.group(2), text)
        n += k
    return text, n


# --------------------------------------------------------------------------- #
def load_frozen(path):
    pats = []
    if not os.path.exists(path):
        return pats
    with io.open(path, encoding="utf-8-sig") as f:   # BOM-safe
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            if line:
                pats.append(line)
    return pats


def is_frozen(rel_path, pats):
    frozen = False
    for p in pats:
        neg = p.startswith("!")
        pat = p[1:] if neg else p
        if fnmatch.fnmatch(rel_path, pat):
            frozen = not neg
    return frozen


# --------------------------------------------------------------------------- #
CSV_COLS = {
    "experiments/phase6/phase6_experiment_registry.csv": ("phase", "subphase", "step"),
}


def registry_meta(exp_id):
    n = int(re.sub(r"\D", "", exp_id) or 0)
    if n <= 6:
        if n in (1, 4):
            return "Phase 6", "6A", "P6A-STEP2"
        if n == 2:
            return "Phase 6", "6A", "P6A-STEP3"
        return "Phase 6", "6A", "-"
    return "Phase 6", "6B", "P6B-STEP1"


def csv_pass(csv_path, rel, apply):
    with io.open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        if not rows:
            return 0
        fields = list(rows[0].keys())
    changed = 0
    out_rows = []
    for r in rows:
        new = {}
        alias = ""
        for k, v in r.items():
            v = v if isinstance(v, str) else ""
            if k == "alias":
                new[k] = v
                continue
            if k == "id":
                m = re.fullmatch(r"H-(\d{2})", v)
                if m and 22 <= int(m.group(1)) <= 35:
                    alias = "H-" + chr(ord("A") + int(m.group(1)) - 22)
            nv, n = transform(v, scope_prefix(rel))
            changed += n
            new[k] = nv
        out_rows.append((new, alias))
    if rel in CSV_COLS:
        for name in CSV_COLS[rel]:
            if name not in fields:
                fields.append(name)
        for new, _ in out_rows:
            new.update(zip(CSV_COLS[rel], registry_meta(new["exp_id"])))
    if rel.endswith("phase6_architecture_hypotheses.csv"):
        if "alias" not in fields:
            fields.insert(fields.index("id") + 1, "alias")
        for new, alias in out_rows:
            new["alias"] = alias
    if not apply:
        return changed
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for new, _ in out_rows:
        w.writerow({k: new.get(k, "") for k in fields})
    with io.open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(buf.getvalue())
    return changed


# --------------------------------------------------------------------------- #
# heavy dirs holding tens of thousands of non-text files (data/ has ~412k files);
# walking them is pointless and makes the scan look hung -> prune by path.
PRUNE_DIRS = (".git", "__pycache__", "data", "datasets", "weights",
              "trained_models", "runs", "logs")


def iter_text_files(root, pats):
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            if d in PRUNE_DIRS:
                continue
            rel_d = os.path.relpath(os.path.join(dirpath, d), root).replace(os.sep, "/")
            # prune any dir covered by a frozen rule (e.g. "data/**")
            if is_frozen(rel_d + "/_probe_", pats):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            if not fn.endswith(TEXT_EXT):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            if is_frozen(rel, pats):
                continue
            yield rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--only", default=None, help="restrict to a comma-separated path prefix list")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    frozen_path = args.frozen or os.path.join(root, "scripts", "naming_frozen.txt")
    pats = load_frozen(frozen_path)
    only = tuple(x.strip() for x in args.only.split(",")) if args.only else None

    total = 0
    touched = []
    for rel in sorted(iter_text_files(root, pats)):
        if only and not rel.startswith(only):
            continue
        ap_ = os.path.join(root, rel)
        try:
            with io.open(ap_, encoding="utf-8-sig") as f:
                original = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        if rel.endswith(".csv"):
            n = csv_pass(ap_, rel, args.apply)
            if n:
                touched.append((rel, n))
                total += n
            continue
        new, n = transform(original, scope_prefix(rel))
        if not n:
            continue
        touched.append((rel, n))
        total += n
        if args.diff and not args.apply:
            d = difflib.unified_diff(
                original.splitlines(), new.splitlines(),
                fromfile=rel + " (before)", tofile=rel + " (after)", lineterm="", n=0)
            sys.stdout.write("\n".join(list(d)[:60]) + "\n")
        if args.apply:
            with io.open(ap_, "w", encoding="utf-8") as f:
                f.write(new)

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print("\n=== %s : %d change(s) in %d file(s) ===" % (mode, total, len(touched)))
    for rel, n in touched:
        print("  %-72s %4d" % (rel, n))
    if not args.apply:
        print("\n(re-run with --apply to write)")

    # frozen files that still hold legacy ids (informational)
    n_frozen_hits = 0
    for rel in sorted(iter_text_files(root, [])):  # no freeze -> scan all
        if not is_frozen(rel, pats):
            continue
        try:
            with io.open(os.path.join(root, rel), encoding="utf-8-sig") as f:
                txt = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        _, n = transform(txt)
        if n:
            n_frozen_hits += 1
    print("\n[frozen] %d file(s) still hold legacy ids by design "
          "(resolved via docs/NAMING_ALIASES.csv)" % n_frozen_hits)


if __name__ == "__main__":
    main()
