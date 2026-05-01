#!/usr/bin/env bash
# Run clang-format on all C++ source and header files.
# Usage:
#   scripts/format.sh           # format in place
#   scripts/format.sh --check   # exit non-zero if formatting would change anything (matches CI)
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

CHECK_MODE=0
if [ "${1:-}" = "--check" ]; then
    CHECK_MODE=1
fi

# Match CI's file set: skip extern/, format owned C/C++ sources and headers.
# Use a read loop instead of `mapfile` for compatibility with macOS's stock Bash 3.2.
files=()
while IFS= read -r f; do
    files+=("$f")
done < <(git ls-files '*.hpp' '*.cpp' '*.h' '*.c' | grep -v '^extern/')

if [ ${#files[@]} -eq 0 ]; then
    echo "No files to format."
    exit 0
fi

if [ "$CHECK_MODE" -eq 1 ]; then
    "$CLANG_FORMAT" --dry-run --Werror "${files[@]}"
    echo "Format check passed."
else
    "$CLANG_FORMAT" -i "${files[@]}"
    echo "Formatting complete."
fi
