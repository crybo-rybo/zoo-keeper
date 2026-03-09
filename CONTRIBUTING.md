# Contributing

## Workflow

1. Create a feature branch from `main`.
2. Keep commits small, focused, and reversible.
3. Run the relevant build and test commands before opening a pull request.
4. Open a pull request instead of pushing directly to `main`.

## Development Checklist

- Configure a local build:

  ```bash
  cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON
  ```

- Build all targets:

  ```bash
  cmake --build build -j4
  ```

- Run the unit suite:

  ```bash
  ctest --test-dir build --output-on-failure -L unit
  ```

- Run integration coverage when needed:

  ```bash
  cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_INTEGRATION_TESTS=ON \
      -DZOO_INTEGRATION_MODEL=/absolute/path/to/model.gguf
  cmake --build build -j4
  ctest --test-dir build --output-on-failure -L integration
  ```

- Check formatting:

  ```bash
  git ls-files '*.hpp' '*.cpp' '*.h' '*.c' | grep -v '^extern/' | \
      xargs clang-format --dry-run --Werror
  ```

- Run static analysis:

  ```bash
  cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DZOO_BUILD_TESTS=ON
  git ls-files '*.cpp' | grep -v '^extern/' | \
      xargs clang-tidy -p build --quiet --header-filter='^(include/zoo|src|tests|examples)/'
  ```

## Coding Expectations

- Stay within the public API boundary under `include/zoo/`.
- Keep new code warning-free under the repo's strict compiler flags.
- Add or update tests for every behavior change.
- Preserve deterministic tests and prefer fixtures over ad hoc data.
- Avoid project-local changes under `extern/llama.cpp/` unless intentionally updating the vendored dependency.

## Pull Requests

Each pull request should include:

- A concise summary of what changed and why.
- Linked issue or audit context when applicable.
- Test evidence (`ctest`, sanitizer, or coverage output as appropriate).
- Notes for any new CMake options, workflow changes, or package metadata changes.
