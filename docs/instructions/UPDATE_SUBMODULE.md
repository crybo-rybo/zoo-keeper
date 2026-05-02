# Updating the llama.cpp Submodule

Zoo-Keeper vendors llama.cpp as a git submodule at `extern/llama.cpp`.
Updating it is a deliberate, multi-step process because llama.cpp API and
CMake target changes can affect Core, Hub, and packaging code.

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

3. **Build and fix compilation errors.** llama.cpp API changes most often
   surface in `src/core/model*.cpp`, but updates to `common/` can also require
   CMake/package and Hub download adaptations:
   ```bash
   scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON -DZOO_BUILD_EXAMPLES=ON
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

6. **Run packaging smoke checks** when upstream CMake targets or installed
   archives change. Verify both build-tree and install-tree consumers.

7. **Stage the submodule pointer and any adaptation changes:**
   ```bash
   git add extern/llama.cpp src/core/ src/hub/ cmake/ docs/
   git commit -m "chore: update llama.cpp to <version/commit>"
   ```

8. **Open a PR.** Submodule updates always go through review.

## Rules

- **Keep llama.cpp calls inside the intended layers.** Core model calls stay in
  `src/core/model*.cpp`; optional Hub code may use llama.cpp download/cache
  helpers behind `ZOO_BUILD_HUB=ON`.
- **Treat `common`/`llama-common` as an integration contract.** If upstream
  renames targets or changes download/parser structs, update package configs,
  pkg-config metadata, and docs in the same migration.
- **Never modify files inside `extern/llama.cpp/` directly.** If a
  local patch is needed, it should be documented in an ADR with a plan
  to upstream or remove it.
- **Test on both Linux and macOS** before merging (CI covers this).
