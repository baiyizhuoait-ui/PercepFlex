#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refs_pass2.py -- reference synchronisation for renamed paths (existence-proven).

Design constraints discovered the hard way
-----------------------------------------
* `scripts/naming_path_map.csv` is a degenerate audit artefact for this pass
  (95/98 RENAME rows have old_path == new_path) -> unusable as a source.
* `git diff --cached -M` holds the real staged renames, BUT rename detection
  pairs *content-identical* files across unrelated experiments
  (e.g. two byte-identical `config.yaml`), producing bogus dir mappings such as
  `experiments/expA_equal_budget/dynA_s1 -> experiments/phase1b/exp_train_a`.
  Naively derived dir mappings therefore CANNOT be trusted.

So this tool never rewrites on trust.  It resolves each currently-dangling
reference by generating candidates from four ordered rule families, then
**keeps only candidates that exist on disk**.  A ref is rewritten only when the
candidate set collapses to exactly one verified target.

Rules (most specific first)
---------------------------
R1  exact git file pair                       old_file -> new_file
R2  basename-preserving git dir pair          (+ sieve: new ends with old basename)
R3  phase-dir insertion                       experiments/<slug>/r -> experiments/phase*/<slug'>/r
R4  phase-prefixed basename                   scripts/<f> -> scripts/phase*_<f>

Usage:  python3 scripts/refs_pass2.py [--apply]
"""
import glob
import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXCLUDE_FILES = {
    "scripts/naming_path_map.csv",
    "scripts/naming_frozen.txt",
    "scripts/apply_naming_migration.py",
    "scripts/check_naming.sh",
    "scripts/rename_paths.py",
    "scripts/refs_pass2.py",
    "scripts/verify_naming.py",
    "scripts/diag_chk11.py",
}

PRUNE_DIRS = {"data", "datasets", "weights", "trained_models", ".git",
              "outputs", "runs", "__pycache__", ".venv", "node_modules"}

TEXT_EXT = {".md", ".txt", ".py", ".sh", ".yaml", ".yml", ".json", ".csv",
            ".toml", ".cfg", ".ini", ".rst", ".bash"}

REF_ROOTS = ("experiments", "configs", "scripts", "docs", "evaluation",
             "models", "training", "utils", "tools", "assets", "deploy")
# The trailing boundary matters: without it, module-symbol notation such as
# `models/representation/det_from_z.DEFAULT_ANCHORS_3S` matched as ".DEFAU".
REF_RE = re.compile(r"((?:%s)/[A-Za-z0-9_./-]*\.[A-Za-z0-9]{2,5}(?![A-Za-z0-9_]))"
                    % "|".join(REF_ROOTS))


def git(*args):
    p = subprocess.run(["git"] + list(args), cwd=ROOT,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout.decode("utf-8", "replace")


def exists(rel):
    return os.path.exists(os.path.join(ROOT, rel.rstrip(".")))


# ---------------------------------------------------------------- rule sources
def git_file_pairs():
    """R1: exact staged file renames (full path -> full path)."""
    out = []
    for line in git("diff", "--cached", "--name-status", "-M",
                    "--diff-filter=R").splitlines():
        p = line.split("\t")
        if len(p) >= 3 and p[0].startswith("R"):
            out.append((p[1].strip(), p[2].strip()))
    out.sort(key=lambda t: len(t[0]), reverse=True)
    return out


def git_dir_pairs():
    """R2: dir-level pairs with basename-preserving sieve."""
    votes = {}
    for old, new in git_file_pairs():
        if old == new:
            continue
        op, np_ = old.split("/"), new.split("/")
        k = 0
        while k < min(len(op), len(np_)) and op[-1 - k] == np_[-1 - k]:
            k += 1
        if k == 0:
            continue
        mo = "/".join(op[:len(op) - k])
        mn = "/".join(np_[:len(np_) - k])
        if not mo or not mn or mo == mn:
            continue
        votes[(mo, mn)] = votes.get((mo, mn), 0) + 1
    out = []
    for (mo, mn), c in votes.items():
        last = mo.split("/")[-1]
        if not mn.lower().endswith(last.lower()):
            continue                      # kill cross-experiment pairing noise
        if not os.path.isdir(os.path.join(ROOT, mn)):
            continue
        out.append((mo, mn, c))
    out.sort(key=lambda t: (len(t[0]), t[2]), reverse=True)
    return out


def phase_dirs():
    return sorted(d for d in glob.glob(os.path.join(ROOT, "experiments", "phase*"))
                  if os.path.isdir(d))


def phase_insert(rel):
    """R3: experiments/<slug>/rest -> experiments/phase*/<slug'>/rest"""
    parts = rel.split("/")
    if len(parts) < 2 or parts[0] != "experiments":
        return []
    slug, rest = parts[1], parts[2:]
    if slug.startswith("phase"):
        return []
    cands = []
    for pd in phase_dirs():
        base = os.path.basename(pd)
        for d in os.listdir(pd):
            if d.lower() == slug.lower():
                cands.append("/".join(["experiments", base, d] + rest))
    return cands


def phase_basename(rel):
    """R4: scripts/<f> -> scripts/phase*_<f>  (and lowercased variant)"""
    parts = rel.split("/")
    if len(parts) != 2 or parts[0] not in ("scripts", "configs"):
        return []
    d, fn = parts
    cands = []
    stem, ext = os.path.splitext(fn)
    for g in (glob.glob(os.path.join(ROOT, d, "phase*_" + fn)),
              glob.glob(os.path.join(ROOT, d, "phase*_" + fn.lower())),
              glob.glob(os.path.join(ROOT, d, "phase*_" + stem.lower() + ext))):
        for p in g:
            cands.append("%s/%s" % (d, os.path.basename(p)))
    return cands


# --------------------------------------------------------------------- resolve
def resolve(ref, r1, r2):
    """Return ordered unique verified candidates for one dangling ref."""
    cands = []
    for old, new in r1:                      # R1 exact file
        if ref == old and exists(new):
            cands.append(("R1", new))
    for mo, mn, _c in r2:                    # R2 dir prefix
        if ref == mo or ref.startswith(mo + "/"):
            cand = mn + ref[len(mo):]
            if exists(cand):
                cands.append(("R2", cand))
    for cand in phase_insert(ref):           # R3
        if exists(cand):
            cands.append(("R3", cand))
    for cand in phase_basename(ref):         # R4
        if exists(cand):
            cands.append(("R4", cand))
    seen, ordered = set(), []
    for tag, c in cands:
        if c in seen:
            continue
        seen.add(c)
        ordered.append((tag, c))
    return ordered


def iter_text_targets():
    for line in git("ls-files").splitlines():
        rel = line.strip()
        if not rel:
            continue
        if rel.split("/")[0] in PRUNE_DIRS:
            continue
        if rel in EXCLUDE_FILES:
            continue
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        yield rel


def collect_dangling():
    d = {}
    for rel in iter_text_targets():
        try:
            with io.open(os.path.join(ROOT, rel), encoding="utf-8", newline="") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        for m in REF_RE.finditer(text):
            ref = m.group(1).rstrip(".")
            if not exists(ref):
                d.setdefault(ref, []).append(rel)
    return d


def main():
    apply = "--apply" in sys.argv
    r1 = git_file_pairs()
    r2 = git_dir_pairs()
    print("sources: R1=%d exact file pair(s), R2=%d dir pair(s)"
          % (len(r1), len(r2)))

    dangling = collect_dangling()
    print("dangling refs: %d\n" % len(dangling))

    plan, unresolved, ambiguous = {}, [], []
    for ref in sorted(dangling, key=lambda r: (-len(dangling[r]), r)):
        cands = resolve(ref, r1, r2)
        if len(cands) == 1:
            plan[ref] = cands[0]
        elif not cands:
            unresolved.append(ref)
        else:
            ambiguous.append((ref, cands))
            plan[ref] = cands[0]

    for ref, (tag, cand) in sorted(plan.items()):
        print("  FIX  %-68s -> %-64s [%s]" % (ref, cand, tag))
    for ref, cands in ambiguous:
        print("  AMBIG %-67s -> %s" % (ref, "; ".join("%s:%s" % c for c in cands)))
    for ref in unresolved:
        print("  LEFT %-67s -> (no verified target)" % ref)

    if not apply:
        print("\n[DRY-RUN] %d fix(es), %d ambiguous, %d unresolved"
              % (len(plan), len(ambiguous), len(unresolved)))
        return

    # rewrite: longest ref first so nested refs do not shadow each other
    rules = []
    for ref, (tag, cand) in sorted(plan.items(), key=lambda kv: len(kv[0]),
                                  reverse=True):
        pat = re.compile(r"(?<![A-Za-z0-9_.])" + re.escape(ref) +
                         r"(?![A-Za-z0-9_\-])")
        rules.append((pat, cand))
    n_ref, n_file = 0, 0
    for rel in iter_text_targets():
        p = os.path.join(ROOT, rel)
        try:
            with io.open(p, encoding="utf-8", newline="") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        new_text, k = text, 0
        for pat, cand in rules:
            new_text, kk = pat.subn(cand, new_text)
            k += kk
        if k and new_text != text:
            n_ref += k
            n_file += 1
            print("   %-64s %d" % (rel, k))
            with io.open(p, "w", encoding="utf-8", newline="") as f:
                f.write(new_text)
    print("\nrefs_pass2: rewrote %d reference(s) in %d file(s)" % (n_ref, n_file))


if __name__ == "__main__":
    main()
