#!/usr/bin/env bash
# Re-create the vendor/ tree from the pinned upstream repos.
# Usage: scripts/fetch-vendor.sh [--depth N]
# vendor/ is gitignored (too large to commit); vendor/LOCK.json records the
# exact upstream commit each directory was taken from.
set -uo pipefail
DEPTH=1
if [ "${1:-}" = "--depth" ]; then DEPTH="$2"; fi
cd "$(dirname "$0")/.." || exit 1
mkdir -p vendor
FAIL=0
while read -r url dir; do
  [ -z "$url" ] && continue
  case "$url" in \#*) continue;; esac
  if [ -d "vendor/$dir" ]; then echo "== $dir already present, skipping"; continue; fi
  echo "== cloning $dir ($url)"
  if git clone --quiet --depth "$DEPTH" --single-branch "$url" "vendor/$dir"; then
    rm -rf "vendor/$dir/.git"
    echo "   OK $dir"
  else
    echo "   FAILED $dir ($url)"; FAIL=1
  fi
done < vendor/repos.txt
exit $FAIL
