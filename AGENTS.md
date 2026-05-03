# AGENTS.md

## Project overview

Zoo-Keeper is a C++23 library on llama.cpp providing model loading, inference,
tool registration, and an async agentic loop for locally hosted LLMs.
Platforms: Linux and macOS only.

## Commands

```bash
# Build (with tests) — llama.cpp is fetched automatically on first configure
scripts/build.sh                      # or: scripts/build.sh -DZOO_BUILD_EXAMPLES=ON

# Run all tests
scripts/test.sh                       # or: scripts/test.sh -R PatternName

# Format all C++ files
scripts/format.sh

# Lint (warning-free build)
scripts/lint.sh

# Sanitizers / coverage (manual)
scripts/build.sh -DZOO_ENABLE_SANITIZERS=ON
scripts/build.sh -DZOO_ENABLE_COVERAGE=ON
```

## Integration testing

See `.secret/integration-testing.md` for local model paths, integration test commands, and `demo_chat` verification steps. The default integration model is `Qwen3-8B-Q4_K_M.gguf`.

## Project structure

- `include/zoo/` — public API boundary
  - `core/` — `Model`, `Config`, `Message`, `Response`, `types.hpp`
  - `tools/` — `ToolRegistry`, `ToolCallParser`, `ErrorRecovery`
  - `hub/` — optional GGUF inspection, HuggingFace, model store
  - `internal/` — private headers (grammar, interceptor, batch, agent runtime)
  - `agent.hpp` — `Agent` async orchestrator
  - `zoo.hpp` — umbrella include
- `src/core/` — all llama.cpp calls (`model*.cpp`)
- `src/agent/` — Agent runtime and backend
- `src/hub/` — hub layer implementation (compiled when `ZOO_BUILD_HUB=ON`)
- `tests/unit/` — GoogleTest suite; `tests/fixtures/` — reusable data
- `examples/` — demo executables and sample config
- `docs/` — architecture, guides, ADRs
- `cmake/` — build helpers (llama.cpp is fetched into `build/_deps/` at configure time)

## Architecture (four layers)

| Layer | Namespace | Depends on |
|-------|-----------|-----------|
| 4 — Hub *(optional)* | `zoo::hub` | Layer 1 + llama.cpp (`ZOO_BUILD_HUB=ON`) |
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
- Read any file, run `scripts/build.sh`, `scripts/test.sh`, `scripts/format.sh`
- Run clang-format, ctest, cmake with standard flags

### Ask first
- Adding new dependencies or modifying `CMakeLists.txt` build structure
- Changes to public API headers (`include/zoo/*.hpp`, `include/zoo/core/*.hpp`, `include/zoo/tools/*.hpp`)
- Updating the pinned llama.cpp version (`ZOO_LLAMA_TAG` in `cmake/ZooKeeperOptions.cmake`)
- Schema or breaking behavioral changes

### Never
- Include `llama.h` in any public header (forward-declare llama types)
- Add llama.cpp calls outside `src/core/model*.cpp`
- Use exceptions for error handling (use `std::expected`)
- Push directly to `main`
- Commit `.DS_Store`, build artifacts, or secrets
</AgentBoundaries>

## Changeset discipline

This is a maturing ~8.5K SLOC codebase. Agents must treat every added line as a cost.

### Hard rules
- **< 150 SLOC added per PR** (excluding tests). Exceeding this requires splitting into multiple PRs
- **One concern per PR** — never bundle a refactor with a feature or fix
- **Read before writing.** Before modifying any file, read it and at least 2 files that interact with it. Do not propose changes to code you haven't read
- **Search for prior art.** `grep`/`glob` the codebase for similar patterns before introducing new ones. If the codebase solves an analogous problem, follow that pattern

### Prefer (in order)
1. Deleting code that is no longer needed
2. Modifying existing code to handle the new case
3. Adding code to an existing file
4. Creating a new file (last resort — justify why existing files won't work)

### Do not add
- New abstractions used in only one place — inline instead
- Wrapper types that forward to an inner type without meaningful logic
- Configuration for behavior with one correct value
- Comments restating what code does (only comment *why*)
- Error handling for states the architecture guarantees cannot occur
- "Improvement" drive-bys — don't clean up code adjacent to your change in the same PR

### How to split large changes
- "Prepare" commits (rename, move, add fixtures) land before "implement" commits
- Public header changes can be a separate PR from their implementation
- Formatting/style fixes go in their own commit, never mixed with logic

### Orientation checklist (before writing any code)
1. Identify which layer (Core / Tools / Agent / Hub) your change belongs to
2. Read the relevant public header(s) in `include/zoo/`
3. Read the implementation file(s) in `src/` you intend to modify
4. Check `tests/unit/` for existing test coverage of that area
5. Confirm your change respects layer boundaries (no upward dependencies)

## Commit conventions

- Concise, imperative, descriptive (e.g. `fix EOG token detection`)
- Feature branches + PRs; never push directly to `main`
- Commit often in small logical increments

## Definition of done

Change is done when: behavior is observable, `scripts/test.sh` passes,
`scripts/format.sh` produces no diff, builds are warning-free, and the
PR is under the SLOC limit with a single logical concern.
