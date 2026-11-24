#!/usr/bin/env bash
#
# Set up tvlab: virtualenv, dependencies, editable install, vendored repos.
#
#   ./scripts/setup.sh              # full setup
#   ./scripts/setup.sh --no-vendor  # skip the 632 MB vendor/ clone
#
# Safe to re-run. Idempotent steps are guarded so a second invocation is fast.
set -euo pipefail
cd "$(dirname "$0")/.."

WITH_VENDOR=1
for arg in "$@"; do
  case "$arg" in
    --no-vendor) WITH_VENDOR=0 ;;
    -h|--help)   sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

PY=${PYTHON:-python3}

echo "==> virtualenv"
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -V

echo "==> pip tooling"
pip install -q -U pip wheel setuptools

echo "==> dependencies (requirements.txt)"
pip install -q -r requirements.txt

echo "==> editable install (provides the 'tvlab' command)"
# Editable, not a wheel: tvlab resolves config/ and dashboard/static relative to
# the repository root, and you are expected to read and edit those configs.
pip install -q -e . --no-deps

if [ "$WITH_VENDOR" -eq 1 ]; then
  echo "==> vendored reference repos (vendor/, gitignored, ~632 MB)"
  bash scripts/fetch-vendor.sh
else
  echo "==> skipping vendor/ (--no-vendor)"
fi

echo "==> sanity check"
python -c "import tvlab, pandas, numpy; print('   tvlab', tvlab.__version__,
      '| pandas', pandas.__version__, '| numpy', numpy.__version__)"
tvlab --version

cat <<'EOF'

Setup complete. Next:

  . .venv/bin/activate
  tvlab markets                                    # what is configured
  tvlab scan     --market crypto --style swing     # who is in a setup now
  tvlab study    --market crypto --style swing     # full validation run + report
  tvlab serve    --market crypto --style swing     # dashboard on http://localhost:8000
  pytest                                           # 209 tests

Read the "Read this first" section of README.md before trusting any number:
the bundled data is synthetic and the default strategy is NOT validated.

Optional, for real market data:
  pip install -r requirements-optional.txt         # ccxt, yfinance
EOF
