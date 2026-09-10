#!/bin/bash
# One command for every machine-checkable gate in this repository.
#
#   bash scripts/check_all.sh
#
# Aggregates:
#   1. scripts/check_naming.sh        -- identifier / path naming unification
#   2. scripts/check_repo_hygiene.sh  -- .gitignore validity, audit-chain
#                                        tracking, oversized/cached files
#
# Wired into a pre-commit hook by scripts/install_hooks.sh.
# Exit 0 only when EVERY gate is green.
set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || exit 1
rc=0

echo "### 1/2  naming convention"
bash scripts/check_naming.sh || rc=1
echo
echo "### 2/2  repository hygiene"
bash scripts/check_repo_hygiene.sh || rc=1
echo
echo "=============================================================="
if [ "$rc" -eq 0 ]; then
  echo "RESULT: ALL GATES GREEN"
else
  echo "RESULT: GATE FAILURE(S) -- do not commit until they are resolved"
fi
exit "$rc"
