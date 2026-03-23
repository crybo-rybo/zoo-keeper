#!/usr/bin/env bash
# Initialize submodules and prepare the repository for building.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
git submodule update --init --recursive
echo "Submodules initialized."
