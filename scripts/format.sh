#!/usr/bin/env bash
# Run clang-format on all C++ source and header files.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Prefer clang-format-18 to match CI; fall back to whatever is on PATH
if command -v clang-format-18 &>/dev/null; then
    CLANG_FORMAT=clang-format-18
elif [ -x /opt/homebrew/opt/llvm@18/bin/clang-format ]; then
    CLANG_FORMAT=/opt/homebrew/opt/llvm@18/bin/clang-format
else
    CLANG_FORMAT=clang-format
fi

find src include tests examples benchmarks -type f \( -name '*.cpp' -o -name '*.hpp' \) | xargs "$CLANG_FORMAT" -i
echo "Formatting complete."
