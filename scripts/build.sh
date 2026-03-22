#!/usr/bin/env bash
# Configure and build the project with provided cmake flags.
# Usage: scripts/build [extra cmake flags...]
#   e.g. scripts/build -DZOO_BUILD_EXAMPLES=ON
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
cmake -B build "$@"
cmake --build build
