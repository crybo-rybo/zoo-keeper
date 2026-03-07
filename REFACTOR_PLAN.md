# Zoo-Keeper v2 Refactor Plan

## Goal

Simplify zoo-keeper into a clean, layered llama.cpp wrapper that enables agentic behavior from locally hosted LLMs. Favor simplicity and compartmentalization. Linux/macOS only.

## Architecture: 3 Clean Layers

```
Layer 3: Agent (Orchestration)
  - Agentic loop, tool execution, streaming, async inference
  - Depends on Layer 1 + Layer 2

Layer 2: Tools
  - Tool registry, tool call parsing, validation, schemas
  - Standalone — no dependency on Layer 1

Layer 1: Core (llama.cpp Wrapper)
  - Model loading, inference, tokenization
  - Chat formatting, KV cache, context management
  - Conversation history tracking
  - Zero dependency on Layers 2 or 3
```

## C++ Standard

**C++23** — use `std::expected` instead of `tl::expected`. Drop the FetchContent dependency.

## Layer 1: `zoo::core`

**Responsibilities:**
- Model loading/unloading (GGUF)
- Tokenization
- Chat template formatting (incremental prompt building via llama_chat_apply_template)
- Synchronous inference (generate)
- KV cache management
- Conversation history tracking + token estimation
- Sampling parameters
- Config + validation

**Key design:**
- `core::Model` is the direct llama.cpp wrapper — load a model, feed it messages, get text back
- History management is part of Model (tied to KV cache state)
- Single-threaded, synchronous — no async, no queues
- Usable standalone without tools or agents

**Files:**
- `include/zoo/core/types.hpp` — Message, Role, Error, ErrorCode, Config, SamplingParams, Response, TokenUsage, Metrics
- `include/zoo/core/model.hpp` — Model class declaration
- `src/core/model.cpp` — Model implementation (llama.cpp calls)

## Layer 2: `zoo::tools`

**Responsibilities:**
- ToolRegistry (template-based registration, schema generation, invocation)
- ToolCallParser (detect tool calls in model output)
- ErrorRecovery (argument validation, retry tracking)

**Key design:**
- Zero dependency on Layer 1 — operates on strings and JSON
- Largely unchanged from current code, just moved to new namespace

**Files:**
- `include/zoo/tools/types.hpp` — ToolCall, ToolEntry, ToolHandler
- `include/zoo/tools/registry.hpp` — ToolRegistry
- `include/zoo/tools/parser.hpp` — ToolCallParser
- `include/zoo/tools/validation.hpp` — ErrorRecovery

## Layer 3: `zoo::agent`

**Responsibilities:**
- Async inference loop (request queue, threading)
- Agentic tool loop (detect -> validate -> execute -> inject -> re-generate)
- Streaming callback wiring
- Per-request cancellation
- Metrics collection

**Key design:**
- Thin orchestrator composing core::Model + tools::ToolRegistry
- Much simpler agentic loop since context mgmt is in Layer 1 and tool logic is in Layer 2

**Files:**
- `include/zoo/agent.hpp` — Agent class
- `include/zoo/zoo.hpp` — Convenience header

## What Gets Removed

| Feature | Files | Rationale |
|---------|-------|-----------|
| RAG (in-memory) | `engine/rag_store.hpp` | Half-baked. Users bring their own. |
| ContextDatabase | `engine/context_database.hpp` | SQLite dep adds complexity. |
| MCP Client | `mcp/` (6 files) | Separate concern. Revisit later. |
| GGUF Utils | `gguf_utils.hpp` | Nice-to-have, not core. |
| Memory Estimate | `memory_estimate.hpp` | Depends on gguf_utils, speculative. |
| GPU OOM / SIGABRT | In llama_backend.cpp | Fragile platform-specific edge cases. |
| tl::expected | FetchContent dep | Replaced by std::expected (C++23). |
| Windows support | CI + platform code | Simplify. Linux/macOS only. |
| PromptTemplate enum | types.hpp | Just use llama_chat_apply_template auto-detect. |

## Final File Structure

```
include/zoo/
  core/
    types.hpp        # Message, Role, Error, Config, SamplingParams, Response, validate_role_sequence
    model.hpp        # Model class (direct llama.cpp wrapper)
  tools/
    types.hpp        # ToolCall, ToolEntry
    registry.hpp     # ToolRegistry (template registration + invocation)
    parser.hpp       # ToolCallParser
    validation.hpp   # ErrorRecovery (arg validation + retries)
  agent.hpp          # Agent class (async orchestration + agentic loop)
  zoo.hpp            # Convenience header
src/
  core/
    model.cpp        # Model implementation (all llama.cpp calls)
tests/
  unit/
    test_types.cpp           # Types, config validation, role sequence validation
    test_tool_registry.cpp   # Tool registration, schema, invocation
    test_tool_parser.cpp     # Tool call parsing
    test_error_recovery.cpp  # Argument validation, retries
  fixtures/
    sample_responses.hpp
    tool_definitions.hpp
examples/
  demo_chat.cpp      # Interactive CLI chat app
  config.example.json
```

## CMake Targets

| Target | Type | Links |
|--------|------|-------|
| `zoo` | INTERFACE | nlohmann_json |
| `zoo_core` | STATIC | zoo, llama |

## Testing Philosophy

- Unit tests: pure logic only (54 tests)
- Model/Agent: integration tests with real GGUF models
- No mocks or IBackend abstraction

## Migration Steps (completed)

1. Create feature branch `feature/v2-refactor`
2. Build Layer 1 — core types + Model class
3. Build Layer 2 — move tool files to new namespace
4. Build Layer 3 — rewrite Agent as thin composer
5. Delete removed features (RAG, MCP, gguf_utils, memory_estimate, context_database)
6. Remove IBackend abstraction — merge Model + LlamaBackend into one class
7. Restructure tests to pure logic only (remove MockBackend, test_model, test_agent)
8. Update CMakeLists.txt, docs, CLAUDE.md
