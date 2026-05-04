#!/usr/bin/env bash
# Build with CRAP analysis enabled and compute CRAP scores.
# Usage: scripts/crap.sh [extra cmake flags...]
#   e.g. scripts/crap.sh -DZOO_CRAP_THRESHOLD=20
#
# Requires Python packages: pip install lizard gcovr
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON -DZOO_ENABLE_CRAP=ON "$@"
cmake --build build --target crap
