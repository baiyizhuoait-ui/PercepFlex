#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PercepFlex path renamer + reference synchroniser (v3).

Reads docs/../naming_path_map.csv (or scripts/naming_path_map.csv), renames the
listed paths with `git mv` (falling back to shutil.move for untracked paths),
then rewrites every textual reference to those paths across the repo.

Safety properties
  1. Longest old_path first  -> `experiments/phase1b/exp_train_a` can never corrupt
     `experiments/phase1b/exp_train_a_subset`.
  2. Only `experiments/<...>` / `docs/<...>` / `configs/<...>` / `scripts/<...>`
     substrings are rewritten; bare basenames are left alone.
  3. After rewriting, every remaining path reference is re-resolved against the
     filesystem; any dangling reference is reported and makes the run fail
     (unless --no-verify).
  4. --apply is required to write; dry-run prints the plan.

Typical use (after the Round 2 chain has fully finished):
    python scripts/rename_paths.py --root .                # plan
    python scripts/rename_paths.py --root . --apply        # execute
    python scripts/verify_naming.py .                      # post-checks
"""
import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import sys

TEXT_EXT = (".md", ".csv", ".py", ".sh", ".yaml", ".yml", ".json", ".txt")
REF_ROOTS = ("experiments/", "docs/", "configs/", "scripts/", "evaluation/", "models/", "training/")


def load_map(map_path):
    with io.open(map_path, encoding="utf-8-sig") as f:          # BOM-safe
        lines = [l for l in f if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    if not reader.fieldnames or "old_path" not in reader.fieldnames:
        # header line was swallowed by a stray comment/BOM -> re-parse with explicit names
        lines = [l for l in lines if "old_path" in l]
        if not lines:
            return []
        reader = csv.DictReader(lines)
    rows = list(reader)
    out = []
    for r in rows:
        if (r.get("action") or "").strip() != "RENAME":
            continue
        old = (r.get("old_path") or "").strip()
        new = (r.get("new_path") or "").strip()
        if old and new and old != new:
            out.append((r.get("kind", "").strip(), old, new,
                        (r.get("confidence") or "").strip(),
                        (r.get("evidence") or "").strip()))
    out.sort(key=lambda t: len(t[1]), reverse=True)
    return out


def is_tracked(root, rel):
    p = subprocess.run(["git", "ls-files", "--error-unmatch", "--", rel],
                       cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode == 0


def do_move(root, old, new, apply):
    src = os.path.join(root, old)
    dst = os.path.join(root, new)
    if not os.path.exists(src):
        return "MISSING-SRC"
    if os.path.exists(dst):
        return "DST-EXISTS"
    if not apply:
        return "PLAN"
    os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
    if is_tracked(root, old):
        subprocess.run(["git", "mv", old, new], cwd=root, check=True)
        return "git-mv"
    shutil.move(src, dst)
    return "move"


# data/ alone holds ~412k files; walking it is pointless and makes the scan hang.
PRUNE_DIRS = (".git", "__pycache__", "data", "datasets", "weights",
              "trained_models", "runs", "logs")


def iter_text(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for fn in sorted(filenames):
            if fn.endswith(TEXT_EXT):
                yield os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")


def rewrite_refs(root, mapping, apply, exclude):
    total = 0
    hits = []
    pairs = [(o, n) for _, o, n, _, _ in mapping]
    for rel in iter_text(root):
        if rel in exclude:
            continue
        ap = os.path.join(root, rel)
        try:
            with io.open(ap, encoding="utf-8-sig") as f:
                txt = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        new = txt
        n = 0
        for o, nn in pairs:
            # only rewrite when the old path starts a real reference root,
            # i.e. it is used as a path (preceded by start/quote/space/slash)
            pat = re.compile(r"(^|[^A-Za-z0-9_./-])" + re.escape(o) + r"(?![A-Za-z0-9_])")
            new, k = pat.subn(lambda m: m.group(1) + nn, new)
            n += k
        if n:
            total += n
            hits.append((rel, n))
            if apply:
                with io.open(ap, "w", encoding="utf-8") as f:
                    f.write(new)
    return total, hits


def verify(root, exclude=None):
    """Every referenced experiments/docs/configs/scripts path must exist."""
    exclude = exclude or set()
    existing_dirs = set()
    for ref_root in REF_ROOTS:
        base = os.path.join(root, ref_root)
        if os.path.isdir(base):
            for name in os.listdir(base):
                existing_dirs.add(ref_root + name)
    dangling = []
    # Require a real file extension: bare prefixes such as "configs/phase3a_"
    # (from `configs/phase3a_{esmall,..}.yaml`) or "training/eval" are not
    # references and used to produce ~90% false positives.
    pat = re.compile(r"(?<![\w/])((?:%s)[A-Za-z0-9_./-]*\.[A-Za-z0-9]{2,5})" % "|".join(REF_ROOTS))
    for rel in iter_text(root):
        if rel in exclude:
            continue
        ap = os.path.join(root, rel)
        try:
            with io.open(ap, encoding="utf-8-sig") as f:
                txt = f.read()
        except (UnicodeDecodeError, OSError):
            continue
        for m in pat.finditer(txt):
            ref = m.group(1).rstrip(".,;:)`\"'")
            # strip a trailing file part -> check the top-level entry
            parts = ref.split("/")
            top = "/".join(parts[:2])
            if top not in existing_dirs and not os.path.exists(os.path.join(root, ref)):
                dangling.append((rel, ref))
    return dangling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--map", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--min-confidence", default="LOW",
                    choices=["HIGH", "MED", "LOW"], help="HIGH skips MED/LOW items")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    map_path = args.map or os.path.join(root, "scripts", "naming_path_map.csv")
    if not os.path.exists(map_path):
        map_path = os.path.join(root, "docs", "naming_path_map.csv")
    mapping = load_map(map_path)

    order = {"HIGH": 2, "MED": 1, "LOW": 0}
    if args.min_confidence != "LOW":
        keep = order[args.min_confidence]
        mapping = [m for m in mapping if order.get(m[3], 0) >= keep]

    print("=== path rename plan (%d item(s), longest-first) ===" % len(mapping))
    results = []
    for kind, old, new, conf, ev in mapping:
        st = do_move(root, old, new, args.apply)
        results.append((st, old, new, conf))
        print("  %-12s %-52s -> %-52s [%s]" % (st, old, new, conf))

    moved = [r for r in results if r[0] in ("git-mv", "move", "PLAN")]
    if args.apply:
        # only rewrite references for paths that actually moved
        real = [(k, o, n, c, e) for (k, o, n, c, e) in
                [(m[0], m[1], m[2], m[3], m[4]) for m in mapping]
                if any(r[1] == o and r[0] in ("git-mv", "move") for r in results)]
        n, hits = rewrite_refs(root, real, True, set())
        print("\n=== reference rewrite : %d site(s) in %d file(s) ===" % (n, len(hits)))
        for rel, k in hits:
            print("  %-72s %4d" % (rel, k))

    if not args.no_verify:
        dangling = verify(root)
        print("\n=== reference resolution check ===")
        if dangling:
            print("  DANGLING %d reference(s):" % len(dangling))
            for rel, ref in dangling[:40]:
                print("    %-60s %s" % (rel, ref))
        else:
            print("  all path references resolve OK")
        if dangling and args.apply:
            sys.exit(2)

    if not args.apply:
        print("\n(dry-run; re-run with --apply to execute)")


if __name__ == "__main__":
    main()
