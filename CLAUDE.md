# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# Quick start (build + tests)
scripts/build.sh && scripts/test.sh

# Run a single test by name
scripts/test.sh -R "TestSuiteName.TestName"

# Format all source files
scripts/format.sh

# Integration tests (requires a real GGUF model)
scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON
ZOO_INTEGRATION_MODEL=/path/to/model.gguf scripts/test.sh

# Hub layer (GGUF inspection, HuggingFace downloading, model store)
scripts/build.sh -DZOO_BUILD_HUB=ON

# Sanitizers / coverage
scripts/build.sh -DZOO_ENABLE_SANITIZERS=ON
scripts/build.sh -DZOO_ENABLE_COVERAGE=ON
```

## Integration Testing

See `.secret/integration-testing.md` for local model paths, integration test commands, and `demo_chat` verification steps. The default integration model is `Qwen3-8B-Q4_K_M.gguf`.

## Architecture

C++23 library on llama.cpp (fetched at configure time via CMake `FetchContent`, pinned by `ZOO_LLAMA_TAG` in `cmake/ZooKeeperOptions.cmake`). Four layers — each depends only on layers below it:

| Layer | Namespace | Role |
|-------|-----------|------|
| 4 | `zoo::hub` | **Optional.** GGUF inspection, HuggingFace downloading, local model store, auto-configuration. Requires `ZOO_BUILD_HUB=ON` |
| 3 | `zoo::Agent` | Async orchestration: inference thread, request queue, streaming, agentic tool loop |
| 2 | `zoo::tools` | Tool registry, schema validation, GBNF schema grammar generation. Header-only, zero llama.cpp dependency |
| 1 | `zoo::core` | `Model` — direct synchronous llama.cpp wrapper. Owns all llama resources. Not thread-safe |

**Threading model:** Agent owns the inference thread; callers submit via `chat()` and get `RequestHandle<TextResponse>`. All callbacks run on the inference thread. Model is protected by thread confinement to the inference thread.

**Tool calling:** Model initializes chat templates via `common_chat_templates_init()` (from the llama.cpp `common` library). Prompt rendering uses `common_chat_templates_apply()`. Tool calling is template-driven: `Model::set_tool_calling()` detects the model's native format (29+ formats recognized) and activates a lazy grammar with format-specific triggers. Models without a recognized native tool calling format have tool calling disabled (`set_tool_calling()` returns `false`). Parsed tool calls are returned inside a `ParsedResponse` struct (containing `std::vector<OwnedToolCall>`) via `Model::parse_tool_response()`. The old hardcoded `<tool_call>` sentinel approach and generic fallback format have been removed.

**CMake targets:** `zoo` (static lib), `zoo_core` (interface compat alias). Consumers use `ZooKeeper::zoo`. The build requires `LLAMA_BUILD_COMMON=ON` to link the `common` library from llama.cpp.

## Key Conventions

- All llama.cpp calls live in `src/core/model*.cpp` — nowhere else
- `model.hpp` uses forward declarations for llama types (no `llama.h` in public headers); `common_chat_templates` is also forward-declared there
- Error handling uses `std::expected` (C++23), not exceptions
- `role_to_string()` returns `const char*` (static storage) — safe for `llama_chat_message`
- `ZOO_LOG` is a no-op when `ZOO_LOGGING_ENABLED` is not defined
- `validate_role_sequence()` is a free function in `types.hpp` (pure logic, unit testable)
- `ToolCallInfo` in `types.hpp` carries parsed tool call data (id, name, arguments_json) from model output
- `CoreToolInfo` in `types.hpp` is the Layer 1 tool descriptor — the agent converts `tools::ToolMetadata` to this before calling `Model::set_tool_calling()`

## Testing

- Unit tests cover pure logic only: types, tools, validation, parsing, grammar, interceptor, batch
- Model/Agent testing requires integration tests with a real GGUF model
- Never `using namespace zoo;` in test files — `zoo::testing` clashes with `::testing` (gtest)
- Test binary: `zoo_tests`, discovered via `gtest_discover_tests`

## Pre-PR Checklist

Before opening a Pull Request, always run:

```bash
scripts/format.sh    # CI enforces formatting
scripts/build.sh     # Must compile cleanly
scripts/test.sh      # All tests must pass
```

<AgentBoundaries>
## Boundaries

### Always (no permission needed)
- Read any file, run `scripts/build.sh`, `scripts/test.sh`, `scripts/format.sh`

### Ask first
- Adding new dependencies or modifying CMakeLists.txt build structure
- Changes to public API headers (`include/zoo/*.hpp`, `include/zoo/core/*.hpp`, `include/zoo/tools/*.hpp`)
- Updating the pinned llama.cpp version (`ZOO_LLAMA_TAG` in `cmake/ZooKeeperOptions.cmake`)

### Never
- Include `llama.h` in any public header (forward-declare llama types)
- Add llama.cpp calls outside `src/core/model*.cpp`
- Use exceptions for error handling (use `std::expected`)
- Push directly to `main`
- Commit `.DS_Store`, build artifacts, or secrets
</AgentBoundaries>

## Changeset Discipline

This codebase is approaching maturity. Every change must justify its existence. The default answer to "should I add this?" is **no**.

### Size constraints
- Target **< 150 SLOC added** per PR (excluding tests). If a change is growing beyond this, split it
- One logical concern per changeset — do not bundle refactors with features or fixes
- Refactoring PRs add zero net features. Feature PRs do minimal refactoring

### Prefer modification over addition
- Modify existing files before creating new ones
- Extend existing abstractions before introducing new ones
- Delete dead code rather than working around it
- If a helper/utility would only be used once, inline it

### Before writing code
- **Read first.** Understand the 2–3 files surrounding your change. Check for existing patterns that solve your problem
- Search for prior art: if the codebase already handles a similar case, follow that pattern exactly
- Check if the problem can be solved by removing code instead of adding it

### What not to add
- Abstractions for hypothetical future use
- Wrapper types that just forward to an inner type
- Configuration options for behavior that has one correct value
- Comments restating what the code does — only comment *why*
- Defensive checks for states that internal code guarantees cannot happen

### Splitting work
- Separate "prepare" commits (moving code, renaming, adding test fixtures) from "implement" commits
- When touching a file with poor formatting or style, fix that in a separate commit — not mixed with logic changes
- If a change requires modifying a public header AND its implementation, consider whether the header change can land first as a smaller PR

## Git Workflow

All changes go through feature branches and Pull Requests. Do not push directly to `main`.
