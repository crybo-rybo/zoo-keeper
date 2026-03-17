#!/usr/bin/env bash
# Run the test suite. Pass optional ctest flags (e.g. -R PatternName).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
ctest --test-dir build --output-on-failure "$@"
