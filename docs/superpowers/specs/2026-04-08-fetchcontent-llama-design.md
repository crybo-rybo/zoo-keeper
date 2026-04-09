# FetchContent llama.cpp Dependency Design

## Goal

Make `zoo-keeper` easier to consume through `FetchContent` without forcing
downstream users to understand or initialize this repository's `llama.cpp`
submodule manually.

## Constraints

- Keep the existing vendored `extern/llama.cpp` submodule workflow for
  maintainers.
- Do not change the public C++ API.
- Preserve the current packaging targets (`ZooKeeper::zoo`,
  `ZooKeeper::zoo_core`) and installed-package behavior.
- Prefer dependency resolution that lets advanced consumers provide their own
  `llama` target or installed `llama` package.

## Chosen Design

Use a hybrid dependency resolution order for `llama.cpp`:

1. Reuse existing in-configure `llama` and `common` targets if the parent
   project already created both of them.
2. Reuse the vendored `extern/llama.cpp` submodule when present.
3. As a final fallback, fetch a pinned `llama.cpp` revision with
   `FetchContent` when `ZOO_FETCH_LLAMA=ON`.
4. Fail with a clear configure-time error if none of the above succeed.

## Why This Shape

- It gives downstream CMake consumers control first.
- It preserves maintainers' reproducible submodule-based workflow.
- It keeps `FetchContent` available as an opt-in convenience instead of making
  every configure implicitly download a large transitive dependency.
- It aligns with current packaging work that already distinguishes same-tree
  consumers from installed-package consumers.

## Validation

- Add a packaging smoke test that consumes `zoo-keeper` via `FetchContent`.
- Update CI to run that smoke test.
- Keep existing build-tree and install-tree package-consumer smoke tests.
