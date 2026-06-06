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

# Hub layer (HuggingFace downloading, local model store)
scripts/build.sh -DZOO_BUILD_HUB=ON

# Sanitizers / coverage
scripts/build.sh -DZOO_ENABLE_SANITIZERS=ON
scripts/build.sh -DZOO_ENABLE_COVERAGE=ON

# CRAP score (complexity × coverage risk metric) — requires: pip install lizard gcovr
scripts/crap.sh
```

## Integration Testing

See `.secret/integration-testing.md` for local model paths, integration test commands, and `demo_chat` verification steps. The default integration model is `Qwen3-8B-Q4_K_M.gguf`.

## Architecture

Zoo-Keeper is a synchronous llama.cpp **harness** centered on one loaded model session: `zoo::Model`. C++23, built on llama.cpp fetched at configure time via CMake `FetchContent` (pinned by `ZOO_LLAMA_TAG` in `cmake/ZooKeeperOptions.cmake`). There is **no Agent SDK** — automatic tool loops, async request queues, inference threads, and `RequestHandle` were removed in the v2 harness rewrite.

| Layer | Namespace | Role |
|-------|-----------|------|
| Hub *(optional)* | `zoo::hub` | HuggingFace downloading + local model store. Requires `ZOO_BUILD_HUB=ON`. `ModelStore::load_model()` returns `std::unique_ptr<zoo::Model>` |
| Harness | `zoo::Model` (alias for `zoo::core::Model`) | Synchronous load · generate · complete · extract · stream · cancel · history · tool-call parsing. Single-session, not thread-safe |
| Tool utilities | `zoo::tools` | Schema registration (`ToolRegistry`), tool-call parsing (`ToolCallParser`), argument validation (`ToolArgumentsValidator`). Zero llama.cpp dependency. No tool execution |
| Core internals | `zoo::core` | `GgufInspector` (metadata read), `SystemProbe` (RAM/CPU/GPU probe), hardware-aware `auto_configure`, llama.cpp wrapper internals |

`zoo::Model` lives at `include/zoo/model.hpp` as `using Model = core::Model;`. Consumers should use the top-level alias.

**Threading model:** `Model` is synchronous and not internally thread-safe. Callers that need concurrency own separate instances or synchronize externally. There is no inference thread, no request queue, and no callback dispatcher.

**Generation surface:**
- `Model::generate(user_message_or_view, ...)` — appends to retained history, returns `TextResponse`
- `Model::complete(ConversationView, ...)` — stateless: runs against an explicit conversation, then restores the previous retained history
- `Model::extract(schema, ..., ...)` — schema-constrained JSON, returns `ExtractionResponse`
- `Model::generate_from_history(...)` — low-level pass that commits the assistant turn and surfaces any structured tool calls
- Streaming via `TokenCallback`, cancellation via `CancellationCallback` — both `FunctionRef`-typed (non-owning, synchronous)

**Tool calling:** Template-driven and native-only. `Model::set_tool_calling(std::vector<ToolSpec>)` asks llama.cpp's `common_chat_templates` layer to prepare the model's native format (29+ formats recognized) — parser, grammar triggers, stop sequences. Models without a recognized native tool calling format have tool calling disabled (`set_tool_calling()` returns `false`). The old hardcoded `<tool_call>` sentinel approach and generic fallback format were removed.

Zoo-Keeper does **not** run an autonomous tool loop. The harness emits structured `OwnedToolCall` records via `generate_from_history()` or `Model::parse_tool_response()`; the application validates them with `zoo::tools` and dispatches them through its own executor map / service layer / UI workflow, then appends `OwnedMessage::tool(...)` results before another model pass.

**Tool registry:** `tools::ToolRegistry` owns normalized `ToolSpec`s (name + description + JSON Schema) and validation metadata only. It does **not** store handlers. `ToolHandler`, `ToolDefinition`, `make_tool_definition(...)`, `ToolRegistry::invoke(...)`, and `ToolRegistry::find_handler(...)` were removed in v2. Registration is schema-only: `registry.register_tool(name, description, schema)`.

**CMake targets:** `zoo` (static lib). Consumers use `ZooKeeper::zoo`. The build requires `LLAMA_BUILD_COMMON=ON` to link the `common` library from llama.cpp.

## Key Conventions

- All llama.cpp / gguf.h / ggml-backend.h calls live in `src/core/` — nowhere else (currently `model*.cpp`, `gguf_inspector.cpp`, `system_probe.cpp`, `hf_download.cpp`)
- Public headers in `include/zoo/core/` use forward declarations for llama types (no `llama.h`, `gguf.h`, or `ggml-backend.h` in public headers); `common_chat_templates` is forward-declared in `model.hpp`
- Error handling uses `std::expected` (C++23), not exceptions
- `role_to_string()` returns `const char*` (static storage) — safe for `llama_chat_message`
- `ZOO_LOG` is a no-op when `ZOO_LOGGING_ENABLED` is not defined
- `validate_role_sequence()` is a free function in `types.hpp` (pure logic, unit testable) with overloads for `ConversationView`, `HistorySnapshot`, `std::span<const OwnedMessage>`, `std::vector<OwnedMessage>`, and `std::span<const MessageView>`
- `OwnedToolCall` / `ToolCallView` / `ToolCallSpan` in `types.hpp` carry parsed tool call data (id, name, arguments_json)
- `ToolSpec` in `types.hpp` is the model-facing tool descriptor (name + description + `parameters_schema`) passed to `Model::set_tool_calling()`
- `ConversationView` / `MessageView` are non-owning request-scoped views; `OwnedMessage` / `HistorySnapshot` are retained storage. Adapters convert one to the other at API boundaries.
- `GenerationOverride` is the per-call generation policy: either `inherit_defaults()` or `explicit_options(GenerationOptions)`

## Testing

- Unit tests cover pure logic and private runtime seams: types, tools, validation, parsing, grammar, batch, prompt bookkeeping, sampling helpers, streaming filter, token accounting, GGUF inspection, GPU fit, hub catalog, model store internals, and model harness wiring
- Live Model behavior requires integration tests with a real GGUF model (`tests/integration/test_model_harness.cpp`)
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
- Changes to public API headers (`include/zoo/model.hpp`, `include/zoo/zoo.hpp`, `include/zoo/core/*.hpp`, `include/zoo/tools/*.hpp`, `include/zoo/hub/*.hpp`)
- Updating the pinned llama.cpp version (`ZOO_LLAMA_TAG` in `cmake/ZooKeeperOptions.cmake`)
- Reintroducing async / threaded orchestration — the v2 harness is intentionally synchronous

### Never
- Include `llama.h` in any public header (forward-declare llama types)
- Add llama.cpp calls outside `src/core/`
- Use exceptions for error handling (use `std::expected`)
- Push directly to `main`
- Commit `.DS_Store`, build artifacts, or secrets
- Add an automatic tool-execution loop inside Zoo-Keeper — execution belongs to caller code
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
