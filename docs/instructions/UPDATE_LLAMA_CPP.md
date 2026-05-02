# Updating llama.cpp

Zoo-Keeper pins llama.cpp by release tag in `cmake/ZooKeeperOptions.cmake`, then
fetches it at CMake configure time via `FetchContent`. Updating it is a
deliberate, multi-step process because llama.cpp API and CMake target changes
can affect Core, Hub, and packaging code.

## Procedure

1. **Create a dedicated branch.** llama.cpp updates should not be mixed
   with feature work:
   ```bash
   git checkout -b chore/update-llama-cpp
   ```

2. **Bump `ZOO_LLAMA_TAG`** in `cmake/ZooKeeperOptions.cmake`:
   ```cmake
   set(ZOO_LLAMA_TAG "<new-release-tag>" CACHE STRING
       "llama.cpp release tag used by FetchContent")
   ```

3. **Force a fresh fetch and build.** FetchContent caches sources under
   `build/_deps/`; clear them before reconfiguring so the new SHA is
   actually picked up:
   ```bash
   rm -rf build/_deps/llama_cpp-* build/_deps/llama_cpp-build
   scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON -DZOO_BUILD_EXAMPLES=ON
   ```

4. **Run the full test suite:**
   ```bash
   scripts/test.sh
   ```

5. **Run integration tests against a real model** (see
   `.secret/integration-testing.md`):
   ```bash
   scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON
   ZOO_INTEGRATION_MODEL=/path/to/model.gguf scripts/test.sh
   ```

6. **Run packaging smoke checks** when upstream CMake targets or installed
   archives change. Verify build-tree, install-tree, and FetchContent
   consumers under `tests/packaging/`.

7. **Stage the version bump and any adaptation changes:**
   ```bash
   git add cmake/ZooKeeperOptions.cmake src/core/ src/hub/ cmake/ docs/
   git commit -m "chore: update llama.cpp to <version/commit>"
   ```

8. **Open a PR.** llama.cpp updates always go through review.

## Rules

- **Keep llama.cpp calls inside the intended layers.** Core model calls stay in
  `src/core/model*.cpp`; optional Hub code may use llama.cpp download/cache
  helpers behind `ZOO_BUILD_HUB=ON`.
- **Treat `llama-common` as an integration contract.** If upstream renames
  targets or changes download/parser structs, update package configs,
  pkg-config metadata, and docs in the same migration.
- **Do not local-patch fetched llama.cpp sources.** FetchContent re-fetches
  the pinned tag on every clean build, so any in-place edits would be silently
  discarded. Carry compatibility workarounds in `cmake/ZooKeeperLlama.cmake`
  instead — see `zoo_apply_llama_common_workarounds()` for the b8992
  `-include algorithm` pattern.
- **Test fixtures are vendored under `tests/fixtures/`.** Do not depend on
  files inside the FetchContent cache (`build/_deps/llama_cpp-src/`); those
  paths are not part of Zoo-Keeper's tracked sources.
- **Test on both Linux and macOS** before merging (CI covers this).
