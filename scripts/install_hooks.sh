#!/bin/bash
# Install the repository's TRACKED hooks into .git/hooks.
#
# Why tracked: .git/hooks is not versioned, so a hook that only exists there is
# invisible, unreviewable and lost on a fresh clone. The source of truth lives
# in scripts/hooks/ and this script copies it in.
#
#   bash scripts/install_hooks.sh
set -eu
cd "$(git rev-parse --show-toplevel)"
mkdir -p .git/hooks
for h in scripts/hooks/*; do
  [ -f "$h" ] || continue
  n=$(basename "$h")
  cp "$h" ".git/hooks/$n"
  chmod +x ".git/hooks/$n"
  echo "installed .git/hooks/$n"
done
echo "done. verify with: git commit (it will run scripts/check_all.sh)"
