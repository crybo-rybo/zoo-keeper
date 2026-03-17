# AGENTS.md

## Project overview

Zoo-Keeper is a C++23 library on llama.cpp providing model loading, inference,
tool registration, and an async agentic loop for locally hosted LLMs.
Platforms: Linux and macOS only.

## Commands

```bash
# First-time setup
scripts/bootstrap

# Build (with tests)
scripts/build                         # or: scripts/build -DZOO_BUILD_EXAMPLES=ON

# Run all tests
scripts/test                          # or: scripts/test -R PatternName

# Format all C++ files
scripts/format

# Lint (warning-free build)
scripts/lint

# Sanitizers / coverage (manual)
scripts/build -DZOO_ENABLE_SANITIZERS=ON
scripts/build -DZOO_ENABLE_COVERAGE=ON
```

## Project structure

- `include/zoo/` — public API boundary
  - `core/` — `Model`, `Config`, `Message`, `Response`, `types.hpp`
  - `tools/` — `ToolRegistry`, `ToolCallParser`, `ErrorRecovery`
  - `internal/` — private headers (grammar, interceptor, batch, agent runtime)
  - `agent.hpp` — `Agent` async orchestrator
  - `zoo.hpp` — umbrella include
- `src/core/` — all llama.cpp calls (`model*.cpp`)
- `src/agent/` — Agent runtime and backend
- `tests/unit/` — GoogleTest suite; `tests/fixtures/` — reusable data
- `examples/` — demo executables and sample config
- `docs/` — architecture, guides, ADRs
- `cmake/` — build helpers
- `extern/llama.cpp/` — vendored submodule

## Architecture (three layers)

| Layer | Namespace | Depends on |
|-------|-----------|-----------|
| 3 — Agent | `zoo::Agent` | Layer 1 + 2 |
| 2 — Tools | `zoo::tools` | nothing (header-only, no llama.cpp) |
| 1 — Core  | `zoo::core`  | llama.cpp only |

## Code style

- C++23, strict typing (`std::expected`, no exceptions)
- Types/classes: `PascalCase`; functions/methods: `snake_case`
- Test files: `tests/unit/test_<component>.cpp`
- Keep builds warning-free (`-Wall -Wextra -Wpedantic`)

## Testing

- Unit tests = pure logic only (types, tools, validation, parsing)
- Model/Agent testing = integration tests (requires real GGUF model)
- Never `using namespace zoo;` in tests (clashes with `::testing`)
- Fixtures in `tests/fixtures/`

<AgentBoundaries>
## Boundaries

### Always (no permission needed)
- Read any file, run `scripts/build`, `scripts/test`, `scripts/format`
- Run clang-format, ctest, cmake with standard flags

### Ask first
- Adding new dependencies or modifying `CMakeLists.txt` build structure
- Changes to public API headers (`include/zoo/*.hpp`, `include/zoo/core/*.hpp`, `include/zoo/tools/*.hpp`)
- Updating the llama.cpp submodule (`extern/llama.cpp`)
- Schema or breaking behavioral changes

### Never
- Include `llama.h` in any public header (forward-declare llama types)
- Add llama.cpp calls outside `src/core/model*.cpp`
- Use exceptions for error handling (use `std::expected`)
- Push directly to `main`
- Commit `.DS_Store`, build artifacts, or secrets
</AgentBoundaries>

## Commit conventions

- Concise, imperative, descriptive (e.g. `fix EOG token detection`)
- Feature branches + PRs; never push directly to `main`
- Commit often in small logical increments

## Definition of done

Change is done when: behavior is observable, `scripts/test` passes,
`scripts/format` produces no diff, and builds are warning-free.
