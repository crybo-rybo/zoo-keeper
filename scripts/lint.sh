#!/usr/bin/env bash
# Build with strict warnings treated as errors.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
scripts/build-all.sh -DZOO_WARNINGS_AS_ERRORS=ON
cmake --build build
echo "Lint (warning-free build) passed."
