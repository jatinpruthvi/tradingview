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

# Some repos cannot be cloned at all - koala73/worldmonitor resets the connection
# mid-pack ("RPC failed; curl 56"), which is why LOCK.json records it as fetched
# via codeload. Same pinned content, no git protocol.
fetch_tarball() {
  local url="$1" dir="$2" slug branch
  slug=$(printf '%s' "$url" | sed -e 's#^https://github.com/##' -e 's#\.git$##' -e 's#/$##')
  for branch in main master; do
    rm -rf "vendor/$dir"
    mkdir -p "vendor/$dir"
    if curl -fsSL --retry 2 "https://codeload.github.com/$slug/tar.gz/refs/heads/$branch" \
         | tar xz -C "vendor/$dir" --strip-components=1 2>/dev/null; then
      echo "   OK $dir (codeload tarball, $branch)"
      return 0
    fi
    rm -rf "vendor/$dir"
  done
  echo "   FAILED $dir ($url) - clone and codeload both failed"
  return 1
}

# repos.txt is tracked with LF, but a Windows checkout with autocrlf hands us
# CRLF - and a trailing \r in $dir makes every clone fail with "could not
# create work tree dir". Strip it rather than trusting the checkout.
while read -r url dir; do
  [ -z "$url" ] && continue
  case "$url" in \#*) continue;; esac
  if [ -d "vendor/$dir" ]; then echo "== $dir already present, skipping"; continue; fi
  echo "== cloning $dir ($url)"
  if git clone --quiet --depth "$DEPTH" --single-branch "$url" "vendor/$dir"; then
    rm -rf "vendor/$dir/.git"
    echo "   OK $dir"
  else
    rm -rf "vendor/$dir"
    fetch_tarball "$url" "$dir" || FAIL=1
  fi
done < <(tr -d '\r' < vendor/repos.txt)
exit $FAIL
