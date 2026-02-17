# Repository Guidelines

## Project Structure & Module Organization
- `include/zoo/`: Public headers and engine/backend interfaces (`agent.hpp`, `engine/*`, `backend/*`).
- `src/backend/`: Concrete backend implementation (`llama_backend.cpp`).
- `tests/`: GoogleTest suite split into `unit/`, `mocks/`, and `fixtures/`.
- `examples/`: Demo executable (`demo_chat.cpp`).
- `cmake/`: Dependency and package config helpers.
- `extern/llama.cpp/`: Vendored submodule dependency; avoid project-local changes here unless intentionally updating the submodule.

## Build, Test, and Development Commands
- Configure dev build:
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
- Run one suite by pattern:
  ```bash
  ctest --test-dir build -R HistoryManagerTest
  ```
- Enable hardening checks when needed:
  ```bash
  cmake -B build -DZOO_ENABLE_SANITIZERS=ON
  cmake -B build -DZOO_ENABLE_COVERAGE=ON
  ```

## Coding Style & Naming Conventions
- Language standard is C++17 (`CMAKE_CXX_STANDARD 17`).
- Compiler warnings are strict (`-Wall -Wextra -Wpedantic` / `/W4`); keep builds warning-free.
- Follow existing naming patterns:
  - Types/classes: `PascalCase` (e.g., `HistoryManager`)
  - Functions/methods: `snake_case` (e.g., `register_tool`)
  - Test files: `tests/unit/test_<component>.cpp`
- Keep headers in `include/zoo/` as the public API boundary; avoid leaking internal-only details.

## Testing Guidelines
- Framework: GoogleTest/GoogleMock via CMake.
- Add or update unit tests for every behavior change, especially engine flow, tool calling, and error recovery paths.
- Prefer deterministic tests using `tests/mocks/mock_backend.*` and reusable fixtures from `tests/fixtures/`.

## Commit & Pull Request Guidelines
- Do not commit directly to `main`; use feature branches and open a PR.
- Commit style in this repo is concise, imperative, and descriptive (e.g., `Fix EOG token detection`, `Update README.md test status`).
- PRs should include:
  - Clear change summary and rationale
  - Linked issue(s) when applicable
  - Test evidence (`ctest` output or equivalent)
  - Notes on config/build flag changes if relevant
