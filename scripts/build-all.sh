#!/usr/bin/env bash
# Configure and build the project with all optional flags set.
# Usage: scripts/build [extra cmake flags...]
#   e.g. scripts/build -DZOO_WARNINGS_AS_ERRORS=ON
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON -DZOO_BUILD_BENCHMARKS=ON -DZOO_BUILD_INTEGRATION_TESTS=ON "$@"
cmake --build build
