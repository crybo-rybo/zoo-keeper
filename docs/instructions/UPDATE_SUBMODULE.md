# Updating the llama.cpp Submodule

Zoo-Keeper vendors llama.cpp as a git submodule at `extern/llama.cpp`.
Updating it is a deliberate, multi-step process because llama.cpp API
changes can break compilation across multiple files in `src/core/`.

## Procedure

1. **Create a dedicated branch.** Submodule updates should not be mixed
   with feature work:
   ```bash
   git checkout -b chore/update-llama-cpp
   ```

2. **Update the submodule to the target commit:**
   ```bash
   cd extern/llama.cpp
   git fetch origin
   git checkout <target-tag-or-commit>
   cd ../..
   ```

3. **Build and fix compilation errors.** llama.cpp API changes will
   surface as build failures in `src/core/model*.cpp` — these are the
   only files that should need changes:
   ```bash
   scripts/build.sh
   ```

4. **Run the full test suite:**
   ```bash
   scripts/test.sh
   ```

5. **Run integration tests** if you have a GGUF model available, to
   verify inference behavior hasn't regressed:
   ```bash
   scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON
   ZOO_INTEGRATION_MODEL=/path/to/model.gguf scripts/test.sh
   ```

6. **Stage the submodule pointer and any adaptation changes:**
   ```bash
   git add extern/llama.cpp src/core/
   git commit -m "chore: update llama.cpp to <version/commit>"
   ```

7. **Open a PR.** Submodule updates always go through review.

## Rules

- **Only `src/core/model*.cpp` should need changes.** If the update
  forces changes in tools, agent, or public headers, that indicates a
  layer boundary violation that needs architectural discussion.
- **Never modify files inside `extern/llama.cpp/` directly.** If a
  local patch is needed, it should be documented in an ADR with a plan
  to upstream or remove it.
- **Test on both Linux and macOS** before merging (CI covers this).
