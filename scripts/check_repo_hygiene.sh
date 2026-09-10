#!/bin/bash
# Repository-hygiene gate (Phase 6B hygiene, L3).
#
# Guards the failure CLASS described in the Phase 6B remediation plan:
#   a directory rename silently invalidates .gitignore literal-path rules and
#   the `!whitelist` entries; nothing errors, nothing shows in the visible
#   `git status`; it surfaces only at `git add -A` -- whose default action is
#   COMMIT.  Two real incidents came out of it (21 GB kd_cache near-miss;
#   every Phase-6 result table silently untracked for days).
#
# Assertions
#   A1a  the audit-chain files are actually tracked by git
#   A1b  no on-disk .csv under experiments/ is ignored by .gitignore
#        (a result table must never be invisible to git)          <- incident (b)
#   A2   nothing that the next `git add -A` would introduce is a cache/weight
#        file or >= 1 MB                                        <- incident (a)
#   A3   no blob >= 50 MB sits in HEAD unless registered in hygiene_exceptions.txt
#   A4   no dead root-anchored literal rule in .gitignore (rename leftovers)
#
# Usage:  bash scripts/check_repo_hygiene.sh [--selftest] [--quiet]
# Exit:   0 = all green, 1 = at least one assertion failed.
# NOTE:   --selftest must FAIL on a deliberately planted counterexample.
#         A gate that cannot fail is not a gate.
set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || exit 1

SELFTEST=0
QUIET=0
for a in "$@"; do
  [ "$a" = "--selftest" ] && SELFTEST=1
  [ "$a" = "--quiet" ] && QUIET=1
done

FAIL=0
ok ()   { [ "$QUIET" -eq 1 ] || echo "[OK]   $1"; }
bad ()  { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }

EXC=scripts/hygiene_exceptions.txt
MAXHITS=${MAXHITS:-8}

echo "=============================================================="
echo " repo hygiene check   (root: $(pwd))"
echo "=============================================================="

# ---------------------------------------------------------------- A1a ----- #
H1A_FILES="
experiments/phase6/phase6_experiment_registry.csv
experiments/phase6/phase6_architecture_hypotheses.csv
experiments/phase6/phase6_round2_statistics.csv
docs/NAMING_CONVENTION.md
scripts/check_naming.sh
scripts/check_repo_hygiene.sh
"
MISSING=0
for f in $H1A_FILES; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then :; else
    [ "$MISSING" -lt "$MAXHITS" ] && echo "        not tracked: $f"
    MISSING=$((MISSING+1))
  fi
done
if [ "$MISSING" -gt 0 ]; then
  bad "A1a audit-chain file(s) not tracked by git: $MISSING"
else
  ok "A1a audit-chain files tracked"
fi

# ---------------------------------------------------------------- A1b ----- #
# Any on-disk .csv under experiments/ that git silently ignores is a
# reproducibility bug -- that is how the Phase-6 result tables went missing.
#
# The previous form walked `experiments/phase*/` only, and fired only when such a
# directory's .csv set was ENTIRELY ignored (`ign -ge disk && trk -eq 0`). It
# therefore missed two real shapes:
#   (i)  a .csv in a non-`phase*` top-level dir, e.g. experiments/_figures/ --
#        `experiments/*/*.csv` swallowed it no matter what the whitelist said;
#   (ii) partial swallowing (3 of 5 ignored, 2 tracked) -- `ign -ge disk` false.
# The invariant is simply "a result table is never invisible to git", so state
# exactly that instead of approximating it per directory.
IGN_CSV=$(git status --porcelain --ignored -uall experiments/ 2>/dev/null \
          | sed -n 's/^!! //p' | grep '\.csv$')
IGN_N=$(printf '%s\n' "$IGN_CSV" | grep -c .)
if [ "$IGN_N" -gt 0 ]; then
  printf '%s\n' "$IGN_CSV" | head -"$MAXHITS" | sed 's/^/        ignored: /'
  bad "A1b on-disk .csv under experiments/ ignored by .gitignore: $IGN_N"
else
  ok "A1b no .csv under experiments/ is ignored"
fi

# ----------------------------------------------------------------- A2 ----- #
BANNED_SUFFIX=('.npy' '.npz' '.pt' '.pth' '.onnx' '.safetensors')
H2_BAD=0
H2_SHOWN=0
while IFS= read -r p; do
  [ -z "$p" ] && continue
  hit=0
  case "/$p" in */kd_cache/*) hit=1;; esac
  for s in "${BANNED_SUFFIX[@]}"; do
    case "$p" in *"$s") hit=1;; esac
  done
  if [ "$hit" -eq 0 ]; then
    sz=$(stat -c '%s' "$p" 2>/dev/null || echo 0)
    [ "$sz" -ge 1048576 ] && hit=1
  fi
  if [ "$hit" -eq 1 ]; then
    [ "$H2_SHOWN" -lt "$MAXHITS" ] && echo "        would add: $p"
    H2_SHOWN=$((H2_SHOWN+1)); H2_BAD=$((H2_BAD+1))
  fi
done < <(git ls-files --others --exclude-standard 2>/dev/null)
if [ "$H2_BAD" -gt 0 ]; then
  bad "A2 'git add -A' would introduce cache/weight/large file(s): $H2_BAD"
else
  ok "A2 nothing oversized or cache-like would be added"
fi

# ----------------------------------------------------------------- A3 ----- #
BIG=0
while read -r sz path; do
  [ -z "${path:-}" ] && continue
  [ "$sz" -lt 52428800 ] && continue
  if [ -f "$EXC" ] && grep -qxF "$path" "$EXC" 2>/dev/null; then
    ok "A3 exempt (registered): $(awk -v s="$sz" 'BEGIN{printf "%.1f MB", s/1048576}') $path"
  else
    echo "        >50 MB in HEAD: $(awk -v s="$sz" 'BEGIN{printf "%.1f MB", s/1048576}') $path"
    BIG=$((BIG+1))
  fi
done < <(git ls-tree -r -l HEAD | awk '{print $4, $5}' | sort -rn | head -40 | awk '$1>=52428800')
if [ "$BIG" -gt 0 ]; then
  bad "A3 unregistered blob(s) >= 50 MB in HEAD: $BIG  (register in $EXC with a reason)"
else
  ok "A3 no unregistered large blob in HEAD"
fi

# ----------------------------------------------------------------- A4 ----- #
DEAD_OUT=$(python3 scripts/audit_gitignore.py 2>/dev/null | sed -n '/\[1\] DEAD RULES/,/^$/p')
DEAD_N=$(printf '%s\n' "$DEAD_OUT" | grep -c '\.gitignore:')
if [ "$DEAD_N" -gt 0 ]; then
  printf '%s\n' "$DEAD_OUT" | grep '\.gitignore:' | head -"$MAXHITS" | sed 's/^/        /'
  bad "A4 dead .gitignore literal rule(s): $DEAD_N"
else
  ok "A4 no dead .gitignore literal rules"
fi

# ------------------------------------------------------------- verdict ---- #
echo "--------------------------------------------------------------"
if [ "$FAIL" -gt 0 ]; then
  echo "RESULT: repo hygiene FAILED ($FAIL assertion(s))"
else
  echo "RESULT: repo hygiene OK"
fi

# ------------------------------------------------------------ selftest ---- #
if [ "$SELFTEST" -eq 1 ]; then
  echo "--------------------------------------------------------------"
  echo " SELFTEST: planting counterexamples - the gate MUST report FAIL"
  T=experiments/_hygienetest
  TC=experiments/_hygienetest_csv
  mkdir -p "$T" "$TC"
  head -c 2097152 /dev/zero > "$T/planted.npy"          # un-ignored 2 MB .npy -> A2
  printf 'a,b\n1,2\n' > "$TC/planted.csv"               # ignored result table -> A1b
  cp .gitignore /tmp/.gi.bak.$$
  printf 'experiments/dead_rule_dir_planted_xyz/\n' >> .gitignore      # -> A4
  printf 'experiments/*/*.csv\n' >> .gitignore                         # -> A1b
  OUT=$(bash "$0" 2>&1); RC=$?
  mv /tmp/.gi.bak.$$ .gitignore
  rm -rf "$T" "$TC"
  echo "$OUT" | grep -E '^\[FAIL\]|^RESULT'
  if [ "$RC" -ne 0 ]; then
    echo " SELFTEST: PASS (gate failed as expected, rc=$RC)"
  else
    echo " SELFTEST: *** FAIL *** the gate stayed green on planted counterexamples"
    exit 1
  fi
  # after cleanup the real gate must be green again
  bash "$0" --quiet | grep -E '^RESULT'
fi

exit $(( FAIL > 0 ? 1 : 0 ))
