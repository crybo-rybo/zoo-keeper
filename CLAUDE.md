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

## Architecture

C++23 library on llama.cpp (submodule at `extern/llama.cpp`). Four layers — each depends only on layers below it:

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
- Updating the llama.cpp submodule (`extern/llama.cpp`)

### Never
- Include `llama.h` in any public header (forward-declare llama types)
- Add llama.cpp calls outside `src/core/model*.cpp`
- Use exceptions for error handling (use `std::expected`)
- Push directly to `main`
- Commit `.DS_Store`, build artifacts, or secrets
</AgentBoundaries>

## Git Workflow

All changes go through feature branches and Pull Requests. Do not push directly to `main`.
