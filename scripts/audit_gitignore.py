#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only .gitignore / repository-hygiene auditor  (Phase 6B hygiene, L2).

WHY THIS EXISTS
---------------
A directory rename silently invalidates the *literal path* rules and the
`!whitelist` entries of .gitignore.  Nothing errors, nothing shows up in the
visible part of `git status`; it surfaces only at `git add -A` -- whose default
action is *commit*.  In Phase 6B this class of failure produced two real
incidents:

  (a) `experiments/*/kd_cache/` no longer matched the renamed path
      `experiments/phase2b/expF_single_teacher/kd_cache/`
      -> 21 GB of *.npy was one `git add -A` away from being committed.
  (b) the `!experiments/phaseX/*.csv` whitelist stopped at phase5
      -> every Phase-6 result table (registry / hypotheses / statistics)
         was silently untracked for days.

This script turns that class into something a machine can see.  It is
READ-ONLY: it never edits .gitignore, never stages, never commits.

REPORTS
-------
  1. DEAD RULES            root-anchored literal rules ('/'-containing, no
                           wildcard) whose target no longer exists on disk.
  2. SHADOWED DELIVERABLES ignored files with a deliverable extension
                           (csv/md/json/yaml/yml/txt) -- a result table hiding
                           behind an ignore rule is a reproducibility bug.
  3. SIZE AUDIT            blobs >LARGE_MB in HEAD, and files >SMALL_MB that
                           the next `git add -A` would introduce.

Usage:  python3 scripts/audit_gitignore.py [--root DIR]
Env:    HYGIENE_GITIGNORE=<path>   audit a different ignore file (self-test hook)
Exit:   always 0 (report only).  scripts/check_repo_hygiene.sh owns the verdict.
"""
import io
import os
import subprocess
import sys

LARGE_MB = 10.0    # blob already in HEAD
SMALL_MB = 1.0     # would-be-added file
DELIV_EXT = (".csv", ".md", ".json", ".yaml", ".yml", ".txt")
# Ignored-but-deliverable-looking files that are in fact intentional bulk data or
# intentional run logs -- not "a result table hiding behind an ignore rule".
SHADOW_EXEMPT_SUFFIX = ("status.txt", "_manifest.txt", "training_log.txt")
SHADOW_EXEMPT_PREFIX = ("data/", "../trac_data/", "baselines/")
SHADOW_EXEMPT_SEG = ("/kd_cache/", "/__pycache__/", "/.ipynb_checkpoints/")


def sh(*args):
    return subprocess.run(args, capture_output=True, text=True).stdout


def read_gitignore(root):
    path = os.environ.get("HYGIENE_GITIGNORE") or os.path.join(root, ".gitignore")
    if not os.path.exists(path):
        return path, []
    return path, io.open(path, encoding="utf-8", errors="replace").read().splitlines()


def _is_root_anchored(pattern):
    """True only for a LITERAL path rule that is anchored at the repo root.

    gitignore semantics: a pattern with an interior (or leading) '/' is resolved
    relative to the ignore file's directory; a pattern with no interior slash
    (e.g. `__pycache__/`, `.vscode/`, `*.pyc`) matches at ANY depth and is NOT a
    literal path rule -- it can never "go stale" from a directory rename.
    """
    if any(c in pattern for c in "*?["):
        return None                     # glob rule: structural, skip
    if pattern.startswith("/"):
        return pattern.lstrip("/").rstrip("/")
    core = pattern.rstrip("/")
    if "/" not in core:
        return None                     # any-depth pattern, skip
    return core


def dead_rules(root, lines):
    """Root-anchored literal rules whose target vanished (the rename hazard)."""
    out = []
    for i, raw in enumerate(lines, 1):
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("!"):
            continue
        target = _is_root_anchored(s)
        if target is None:
            continue
        if not os.path.exists(os.path.join(root, target)):
            out.append((i, s))
    return out


def shadowed_deliverables():
    """Ignored files that look like a deliverable (would silently never ship)."""
    raw = sh("git", "status", "--porcelain", "--ignored", "-uall")
    out = []
    for line in raw.splitlines():
        if not line.startswith("!!"):
            continue
        p = line[3:].strip()
        if not p.lower().endswith(DELIV_EXT):
            continue
        if p.endswith(".log") or p.endswith(SHADOW_EXEMPT_SUFFIX):
            continue
        if p.startswith(SHADOW_EXEMPT_PREFIX):
            continue
        if any(seg in "/" + p for seg in SHADOW_EXEMPT_SEG):
            continue
        out.append(p)
    return sorted(out)


def big_blobs_in_head():
    raw = sh("git", "ls-tree", "-r", "-l", "HEAD")
    out = []
    for line in raw.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            size = int(parts[3])
        except ValueError:
            continue
        if size >= LARGE_MB * 1024 * 1024:
            out.append((size, parts[4]))
    return sorted(out, reverse=True)


def would_be_added_large():
    raw = sh("git", "ls-files", "-z", "--others", "--exclude-standard")
    out = []
    for p in [x for x in raw.split("\0") if x]:
        try:
            size = os.path.getsize(p)
        except OSError:
            continue
        if size >= SMALL_MB * 1024 * 1024:
            out.append((size, p))
    return sorted(out, reverse=True)


def main():
    root = os.getcwd()
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]
    os.chdir(root)

    gi_path, lines = read_gitignore(root)
    dead = dead_rules(root, lines)
    shadow = shadowed_deliverables()
    big = big_blobs_in_head()
    addbig = would_be_added_large()

    print("=" * 72)
    print("audit_gitignore.py  --  read-only .gitignore / hygiene audit")
    print("  root     : %s" % root)
    print("  gitignore: %s (%d rules)" % (gi_path, len(lines)))
    print("=" * 72)

    print("\n[1] DEAD RULES (root-anchored literal, target gone) ......... %d" % len(dead))
    for ln, rule in dead:
        print("     .gitignore:%d   %s" % (ln, rule))
    if not dead:
        print("     none")

    print("\n[2] SHADOWED DELIVERABLES (ignored, but looks like a result)  %d" % len(shadow))
    for p in shadow:
        print("     %s" % p)
    if not shadow:
        print("     none")

    print("\n[3a] HEAD blobs >= %.0f MB .................................. %d" % (LARGE_MB, len(big)))
    for size, p in big:
        print("     %8.2f MB  %s" % (size / 1048576.0, p))
    if not big:
        print("     none")

    print("\n[3b] would-be-added files >= %.0f MB ........................ %d" % (SMALL_MB, len(addbig)))
    for size, p in addbig:
        print("     %8.2f MB  %s" % (size / 1048576.0, p))
    if not addbig:
        print("     none")

    print("\nSUMMARY  dead=%d shadow=%d head_big=%d add_big=%d"
          % (len(dead), len(shadow), len(big), len(addbig)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
