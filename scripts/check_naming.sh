#!/usr/bin/env bash
# PercepFlex naming linter v3 — full-repo unification (historical phases included).
#
# Usage: bash scripts/check_naming.sh [--strict] [--root DIR]
#   exit 0 = no undeclared violation ; exit 1 = violations ; --strict escalates.
#
# Frozen paths come from scripts/naming_frozen.txt (last-match-wins, '!' unfreezes).
# Declared legacy is read from docs/NAMING_ALIASES.csv (those literals are allowed).
set -u

ROOT="."
STRICT=0
MAXHITS="${MAXHITS:-8}"
while [ $# -gt 0 ]; do
  case "$1" in
    --strict) STRICT=1 ;;
    --root) ROOT="$2"; shift ;;
    *) ;;
  esac
  shift
done
cd "$ROOT" || exit 3

FROZEN_FILE="scripts/naming_frozen.txt"
[ -f "$FROZEN_FILE" ] || { echo "ERROR: $FROZEN_FILE not found"; exit 3; }

# ---- freeze list ---------------------------------------------------------- #
FROZEN=()
while IFS= read -r ln; do
  ln="${ln%%#*}"
  ln="$(printf '%s' "$ln" | sed -e 's/[[:space:]]*$//')"
  [ -n "$ln" ] && FROZEN+=("$ln")
done < "$FROZEN_FILE"

is_frozen() { # $1 = rel path ; return 0 = frozen, 1 = live
  local p="$1" pat st=1
  for pat in "${FROZEN[@]}"; do
    if [ "${pat:0:1}" = '!' ]; then
      case "$p" in ${pat:1}) st=1 ;; esac     # unfreeze -> live
    else
      case "$p" in $pat) st=0 ;; esac         # freeze   -> frozen
    fi
  done
  return $st
}

# ---- live text files ------------------------------------------------------ #
LIVE=()
while IFS= read -r f; do
  f="./${f#./}"
  rel="${f#./}"
  if ! is_frozen "$rel"; then LIVE+=("$f"); fi
done < <(find . -type f \( -name '*.md' -o -name '*.csv' -o -name '*.py' \
  -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.txt' \) \
  -not -path './.git/*' -not -path '*/__pycache__/*' \
  -not -path './data/*' -not -path './datasets/*' \
  -not -path './weights/*' -not -path './trained_models/*' | sort)

echo "scanned: ${#LIVE[@]} live file(s), ${#FROZEN[@]} frozen pattern(s)"
echo

FAIL=0
check_grep() { # $1 label  $2 regex  $3 advice  $4 optional ignore-regex
  local label="$1" rx="$2" advice="$3" ignore="${4:-}" hits n
  if [ -n "$ignore" ]; then
    hits="$(grep -InE "$rx" "${LIVE[@]}" 2>/dev/null | grep -vE "$ignore" || true)"
  else
    hits="$(grep -InE "$rx" "${LIVE[@]}" 2>/dev/null || true)"
  fi
  n="$(printf '%s' "$hits" | grep -c . || true)"
  if [ "${n:-0}" -gt 0 ]; then
    echo "[FAIL] $label : $n hit(s)  ->  $advice"
    printf '%s\n' "$hits" | head -n "$MAXHITS" | cut -c1-150 | sed 's/^/        /'
    FAIL=$((FAIL+1))
  else
    echo "[OK]   $label"
  fi
}

check_grep "CHK1 residual Phase-6 letter hypothesis (H-A .. H-N)" \
  '\bH-[A-N]\b' 'use H-22 .. H-35' ',H-[A-N],'
check_grep "CHK2 unpadded experiment id (EXP-n)" \
  '\bEXP-[0-9]\b' 'use EXP-0n'
check_grep "CHK3 bare execution step (STEPn / STEP 2)" \
  '(^|[^A-Za-z0-9-])STEP[ _]?[0-9]\b' 'use P<phase>-STEP<n>'
check_grep "CHK4 prose round counter (Round 1|2 / Round-1)" \
  '(^|[^A-Za-z])Round[ _-]+[12]\b' 'use Phase 6A / Phase 6B'
check_grep "CHK5 closed-phase local id (4B-n / 4C-n)" \
  '(^|[^A-Za-z0-9])4[BC]-[0-9]' 'use P4B-EXP-0n / P4C-EXP-0n'
check_grep "CHK6 bare Phase 1-5 hypothesis id (H<n>)" \
  '\bH[0-9]{1,2}[a-z]?\b' 'use H-<nn> (zero-padded, dashed)'
check_grep "CHK7 slash-form cross-phase id (P4B/EXP-)" \
  'P4[BC]/EXP-' 'use P4B-EXP-0n'
check_grep "CHK8 letterless gate (GATE1|GATE-1)" \
  '\bGATE-?[0-9]\b' 'use GATE-<phase>.<n>'
check_grep "CHK9 scheduling alias (GPU-1|2|3)" \
  '\bGPU-[123]\b' 'use P6A-STEP1 .. P6A-STEP3'

# ---- machine-facing filenames --------------------------------------------- #
UPPER=""
while IFS= read -r f; do
  rel="${f#./}"
  is_frozen "$rel" && continue
  case "$rel" in
    docs/*|*/README.md|*/TEMPLATE*|scripts/NAMING*) continue ;;
  esac
  printf '%s' "$rel" | grep -qE '/[^/]*[A-Z][^/]*\.(yaml|yml)$' && UPPER="$UPPER$rel\n"
done < <(find . -type f \( -name '*.yaml' -o -name '*.yml' \) -not -path './.git/*')
if [ -n "$UPPER" ]; then
  echo "[WARN] CHK10 machine-facing yaml with uppercase letters:"
  printf "%b" "$UPPER" | sed 's/^/        /'
else
  echo "[OK]   CHK10 machine-facing yaml naming"
fi

# ---- CHK11: stale references to renamed paths ------------------------------ #
# Two failure modes that must not be conflated:
#
#   CHK11  (FAIL) a reference still uses a path that WAS renamed -> a genuine
#                 unification defect.  scripts/refs_pass2.py resolves each
#                 currently-dangling reference against the filesystem, so
#                 "fixable" is exactly "stale old name that still points at a
#                 real file".  Fixable > 0 means refs_pass2 --apply would change
#                 something.
#
#   CHK11w (WARN) a reference points at an artefact that never existed in git
#                 (runtime *.log / *.flag / *.npz / deleted summaries).  These
#                 are historical citations, not naming defects; they are
#                 inventoried in scripts/naming_historical_refs.txt rather than
#                 being "fixed" into invented paths.
#
# Note: only refs whose extension terminates the token count.  Without that
# trailing boundary, module-symbol notation such as
# `models/representation/det_from_z.DEFAULT_ANCHORS_3S` matched as ".DEFAU".
PASS2_OUT="$(python3 scripts/refs_pass2.py 2>&1 || true)"
if ! printf '%s' "$PASS2_OUT" | grep -q '\[DRY-RUN\]'; then
  echo "[FAIL] CHK11 refs_pass2.py did not complete; reference sync unverifiable"
  FAIL=$((FAIL+1))
else
  FIXABLE="$(printf '%s' "$PASS2_OUT" | sed -n 's/.*\[DRY-RUN\] \([0-9][0-9]*\) fix(es).*/\1/p')"
  printf '%s\n' "$PASS2_OUT" | sed -n 's/^  LEFT  *\([^ ]*\).*/\1/p' | sort -u \
      > scripts/naming_historical_refs.txt
  HISTN="$(wc -l < scripts/naming_historical_refs.txt | tr -d ' ')"
  if [ "${FIXABLE:-0}" -gt 0 ]; then
    echo "[FAIL] CHK11 stale reference(s) to renamed path(s): $FIXABLE"
    printf '%s\n' "$PASS2_OUT" | sed -n 's/^  FIX  */        FIX /p' | head -"$MAXHITS"
    FAIL=$((FAIL+1))
  else
    echo "[OK]   CHK11 no stale references to renamed paths"
  fi
  echo "[WARN] CHK11w historical citation(s) to never-committed artefacts: $HISTN"
  [ "$HISTN" -gt 0 ] && head -"$MAXHITS" scripts/naming_historical_refs.txt | sed 's/^/        /'
fi

echo
if [ "$FAIL" -gt 0 ]; then
  echo "RESULT: $FAIL undeclared violation class(es)"
  [ "$STRICT" -eq 1 ] && exit 1
  exit 0
fi
echo "RESULT: naming UNIFIED"
exit 0
