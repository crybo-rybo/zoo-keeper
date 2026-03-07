# Repository Guidelines

## Project Structure & Module Organization
- `include/zoo/`: Public API boundary.
  - `zoo.hpp`: umbrella include for consumers.
  - `agent.hpp`: async orchestration layer.
  - `core/*`: core model/types (`Model`, `Config`, `Message`, `Response`).
  - `tools/*`: tool registry, parsing, validation types/utilities.
- `src/core/`: core llama.cpp wrapper implementation (`model.cpp`).
- `tests/`: GoogleTest suite in `unit/` with reusable data in `fixtures/`.
- `examples/`: demo executable and sample config (`demo_chat.cpp`, `config.example.json`).
- `cmake/`: Dependency and package config helpers.
- `extern/llama.cpp/`: Vendored submodule dependency; avoid project-local changes here unless intentionally updating the submodule.

## Build, Test, and Development Commands
- Initialize dependencies (fresh clone):
  ```bash
  git submodule update --init --recursive
  ```
- Configure dev build (tests + examples):
  ```bash
  cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON
  ```
- Build all targets:
  ```bash
  cmake --build build -j4
  ```
- Run all tests:
  ```bash
  ctest --test-dir build --output-on-failure
  ```
- List discovered tests:
  ```bash
  ctest --test-dir build -N
  ```
- Run one suite by pattern:
  ```bash
  ctest --test-dir build -R ToolRegistryTest --output-on-failure
  ```
- Enable hardening checks when needed:
  ```bash
  cmake -B build -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON
  cmake --build build
  ctest --test-dir build --output-on-failure

  cmake -B build -DZOO_ENABLE_COVERAGE=ON -DZOO_BUILD_TESTS=ON
  cmake --build build
  ctest --test-dir build --output-on-failure
  ```
- Platform toggles:
  ```bash
  # macOS defaults to Metal ON; Linux defaults OFF
  cmake -B build -DZOO_ENABLE_METAL=OFF

  # CUDA path
  cmake -B build -DZOO_ENABLE_CUDA=ON
  ```

## Coding Style & Naming Conventions
- Language standard is C++23 (`CMAKE_CXX_STANDARD 23`).
- Compiler warnings are strict (`-Wall -Wextra -Wpedantic` / `/W4`); keep builds warning-free.
- Follow existing naming patterns:
  - Types/classes: `PascalCase` (e.g., `ToolRegistry`)
  - Functions/methods: `snake_case` (e.g., `register_tool`)
  - Test files: `tests/unit/test_<component>.cpp`
- Keep headers in `include/zoo/` as the public API boundary; avoid leaking internal-only details.

## Testing Guidelines
- Framework: GoogleTest/GoogleMock via CMake.
- Add or update unit tests for every behavior change, especially tool calling/parser behavior, error recovery, and core type/config validation.
- Prefer deterministic tests with reusable fixtures from `tests/fixtures/`.
- Use `ctest --test-dir build -R <Pattern>` for focused runs while iterating.

## Commit & Pull Request Guidelines
- Never do work directly on `main`. Create/use a separate feature branch for all changes.
- Commit often in small, logical increments to make reverting and bisecting easy.
- Commit style in this repo is concise, imperative, and descriptive (e.g., `Fix EOG token detection`, `Update README.md test status`).
- PRs should include:
  - Clear change summary and rationale
  - Linked issue(s) when applicable
  - Test evidence (`ctest` output or equivalent)
  - Notes on config/build flag changes if relevant
