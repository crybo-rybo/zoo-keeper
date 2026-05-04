#!/usr/bin/env bash
# Build with CRAP analysis enabled and compute CRAP scores.
# Usage: scripts/crap.sh [extra cmake flags...]
#   e.g. scripts/crap.sh -DZOO_CRAP_THRESHOLD=20
#
# Requires Python packages: pip install lizard gcovr
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

cmake_flags=(
    -DZOO_BUILD_TESTS=ON
    -DZOO_BUILD_HUB=ON
    -DZOO_BUILD_INTEGRATION_TESTS=ON
    -DZOO_ENABLE_CRAP=ON
)

if [[ -n "${ZOO_INTEGRATION_MODEL:-}" ]]; then
    cmake_flags+=("-DZOO_INTEGRATION_MODEL=${ZOO_INTEGRATION_MODEL}")
fi

scripts/build.sh "${cmake_flags[@]}" "$@"
cmake --build build --target crap
