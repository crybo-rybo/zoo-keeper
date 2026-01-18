# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Zoo-Keeper is a header-only C++17 library built on top of llama.cpp that functions as a complete Agent Engine for local LLM inference. It abstracts agentic AI systems by providing automated conversation history management, type-safe tool registration, asynchronous inference with cancellation support, and intelligent context window management.

**Status:** MVP Implementation Complete (Phase 1)

## Git Workflow

**IMPORTANT:** Now that the initial MVP implementation is complete, all code changes MUST follow this workflow:

1. **Create a feature branch** from `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** on the feature branch

3. **Commit your changes** with descriptive messages:
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

4. **Push the branch** to remote:
   ```bash
   git push -u origin feature/your-feature-name
   ```

5. **Open a Pull Request** to merge back into `main`
   - Include a clear description of changes
   - Reference any related issues
   - Ensure all tests pass
   - Request review if needed

6. **Merge via Pull Request** - DO NOT push directly to `main`

**Never commit directly to main** after the initial MVP implementation. This ensures:
- Code review process is followed
- CI/CD checks run before merging
- Clean git history with meaningful PR descriptions
- Easy rollback if needed

## Build Commands

```bash
# Configure (from project root)
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON

# Build
cmake --build build

# Run all tests
ctest --test-dir build

# Run tests with sanitizers
cmake -B build -DZOO_ENABLE_SANITIZERS=ON && cmake --build build && ctest --test-dir build

# Run with coverage
cmake -B build -DZOO_ENABLE_COVERAGE=ON && cmake --build build && ctest --test-dir build
```

### CMake Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Metal acceleration (macOS) | ON (macOS) |
| `ZOO_ENABLE_CUDA` | CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_EXAMPLES` | Build examples | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan/TSan/UBSan | OFF |

## Architecture

### Three-Layer Design

1. **Public API Layer** - Single entry point via `zoo::Agent` class
2. **Engine Layer** - Core logic components (Request Queue, History Manager, Tool Registry, Agentic Loop, Template Engine, Error Recovery)
3. **Backend Layer** - Abstracted llama.cpp interface with production and mock implementations

### Threading Model

- **Calling Thread**: Submits `chat()` requests, receives `std::future<Response>`
- **Inference Thread**: Owns `llama_context`, processes queue, executes tools, manages history

All callbacks (`on_token`, `on_tool_call`) execute on the inference thread. Consumer is responsible for cross-thread synchronization.

### Key Design Decisions

- **Header-only distribution**: Simplifies integration
- **`std::expected` for errors**: Modern composable error handling without exceptions
- **Dependency injection for llama.cpp**: Enables unit testing via Mock Backend
- **Value semantics**: Predictable ownership, fewer lifetime bugs

## Test Structure

```
tests/
├── unit/
│   ├── test_request_queue.cpp
│   ├── test_history_manager.cpp
│   ├── test_tool_registry.cpp
│   ├── test_agentic_loop.cpp
│   ├── test_template_engine.cpp
│   ├── test_error_recovery.cpp
│   └── test_agent.cpp
├── mocks/
│   ├── mock_backend.hpp
│   └── mock_backend.cpp
├── fixtures/
│   ├── sample_responses.hpp
│   ├── tool_definitions.hpp
│   └── template_expectations.hpp
└── CMakeLists.txt
```

Uses GoogleTest/GoogleMock. Follow TDD: Red (failing test) → Green (minimal code to pass) → Refactor.

## Dependencies

| Dependency | Integration |
|------------|-------------|
| llama.cpp | Git submodule |
| nlohmann/json | CMake FetchContent |
| GoogleTest 1.14+ | CMake FetchContent (tests only) |

## Compiler Requirements

- **Windows**: MSVC 2019+ (19.20)
- **macOS**: Clang 13.0+
- **Linux**: GCC 11.0+ or Clang 13.0+

## Key Types

- `zoo::Agent` - Main entry point, owns inference thread
- `zoo::Config` - Immutable configuration (model path, context size, sampling params, template)
- `zoo::Message` - Value type with role (System/User/Assistant/Tool) and content
- `zoo::Response` - Result containing text, tool history, token usage, latency metrics

## Tool Registration

Template-based registration supporting `int`, `float`, `double`, `bool`, `std::string` parameters:
```cpp
agent.register_tool<Func>("name", "description", func);
```

Generates JSON schema automatically from function signature.

## Reference Documents

- `zoo-keeper-prd.md` - Product requirements, goals, success metrics
- `zoo-keeper-trd.md` - Technical architecture, state machines, test plan
