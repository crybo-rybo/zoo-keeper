# Zoo-Keeper Production Readiness Review

**Date:** 2026-02-27
**Scope:** Full codebase audit — public API, engine, backend, MCP, tests, build, docs

### 3.2 Config validation is incomplete

**File:** `include/zoo/types.hpp:282-299`

Missing validations:
- `sampling.temperature` — negative or extremely large values
- `sampling.top_p` — must be in (0.0, 1.0]
- `n_gpu_layers` — negative value
- `request_queue_capacity` — zero or negative
- `kv_cache_type_k/v` — only checks `>= 0`, doesn't validate against known enum values

---

### 3.3 Streaming callback lifetime not enforced

**File:** `include/zoo/types.hpp:278-279`, `include/zoo/engine/agentic_loop.hpp:423-447`

`on_token` is a `std::function<void(std::string_view)>` that can capture arbitrary references. The callback executes on the inference thread, potentially long after the caller's stack frame is gone. The `make_streaming_callback()` helper also captures local variables by `[&]`, creating a fragile lifetime dependency.

Currently safe because `generate()` is synchronous, but this is an implicit contract that's easy to break.

**Fix:** Document the lifetime contract prominently. Consider taking callbacks by value and requiring them to be self-contained.

---

### 3.4 Conditional `#ifdef ZOO_ENABLE_MCP` changes public API surface

**File:** `include/zoo/agent.hpp:9-11, 350+`

When `ZOO_ENABLE_MCP` is off, `add_mcp_server()` doesn't exist and `Agent` has a different binary layout. Libraries compiled with different MCP flags are ABI-incompatible but have the same header name.

**Fix:** Either always expose the method (returning an error when MCP is disabled) or use the pimpl pattern to keep the public class layout stable.

---

### 3.5 Timing-dependent tests are flaky

**File:** `tests/unit/test_request_queue.cpp:181-191, 204-219, 282-307`

Tests use hardcoded sleeps (50ms, 100ms) and assume threads will be scheduled within those windows. On loaded CI machines, these intermittently fail.

**Fix:** Replace `std::this_thread::sleep_for()` with condition variables, latches, or barriers.

---

### 3.6 No test timeouts configured

**File:** `tests/CMakeLists.txt:47-52`

`gtest_discover_tests` has no `TIMEOUT` property. A hanging test (e.g., deadlock from the thread-safety bugs above) blocks CI indefinitely.

**Fix:** Add `PROPERTIES TIMEOUT 30` or similar.

---

### 3.7 Phase 4 features completely undocumented

The following public APIs have no documentation page:
- `zoo::estimate_memory()` and `MemoryEstimate` struct
- `zoo::read_gguf_metadata()` and `GgufModelInfo` struct
- `Config::auto_tune_context`
- `Config::kv_cache_type_k/v`
- `Agent::get_training_context_size()`
- `ErrorCode::GpuOutOfMemory` recovery pattern

---

### 3.8 Token estimation has no overflow protection

**File:** `include/zoo/engine/history_manager.hpp:73-74, 96`

`estimated_tokens_` is an `int`. Adding large token counts without saturation can overflow to negative values, causing the pruning logic to believe the context is empty. Use `size_t` or add saturation arithmetic.

---

### 3.9 Incomplete error propagation in `Session::send_request()`

**File:** `include/zoo/mcp/protocol/session.hpp:172-181`

When transport `send()` fails, the error is converted to a hardcoded JSON-RPC error code (`-32603`) and routed back through the promise. The original transport error code and message are lost. The caller gets a generic "Internal error" instead of "pipe broken" or "process exited".

---

### 3.10 No `LlamaBackend` tests

The real backend (`include/zoo/backend/llama_backend.hpp`) has zero dedicated tests. All tests use `MockBackend`. This means:
- `format_prompt()` incremental formatting logic is untested against real `llama_chat_apply_template()`
- KV cache management is untested
- GPU memory detection (`sysctl` / `/proc/meminfo`) is untested
- The SIGABRT handler path is untested

---

## 4. CLEANUP (Low Priority)

### 4.1 Inconsistent include paths

Some headers use `#include "zoo/types.hpp"` (absolute), others use `#include "types.hpp"` (relative). Pick one convention.

---

### 4.2 `set_system_prompt()` should accept `std::string_view`

**File:** `include/zoo/agent.hpp:239`

Read-only string parameters throughout the API use `const std::string&`. These should be `std::string_view` for consistency with C++17 idioms and to avoid forcing callers to construct `std::string` from literals.

---

### 4.3 `RequestHandle` default constructor creates invalid state

**File:** `include/zoo/types.hpp:481-493`

Default-constructed `RequestHandle` has `id = 0` and a default-constructed `std::future` (which throws on `.get()`). There's no `is_valid()` method. Delete the default constructor or add a validity check.

---

### 4.4 Session dual constructors are code duplication

**File:** `include/zoo/mcp/protocol/session.hpp:41-58`

Two constructors with near-identical bodies. Use a delegating constructor or a private `init()` method.

---

### 4.5 `ToolCall::generate_id()` resets across restarts

**File:** `include/zoo/engine/tool_call_parser.hpp:153-156`

IDs are sequential starting from 0. If the application restarts, IDs reuse, potentially confusing external systems that track tool call IDs.

---

### 4.6 Magic numbers in `AgenticLoop`

**File:** `include/zoo/engine/agentic_loop.hpp:456-459`

```cpp
int max_tool_iterations_ = 5;
int memory_retrieval_top_k_ = 4;
int memory_min_messages_to_keep_ = 6;
double memory_prune_target_ratio_ = 0.75;
```

These should be configurable through `Config` or at least named constants.

---

### 4.7 Formatter buffer over-allocation

**File:** `src/backend/llama_backend.cpp:204-206`

Pre-allocates `context_size * 4` bytes for the format buffer. For a 32k context, that's 128KB upfront. The buffer grows dynamically in `format_prompt()` anyway — start smaller.

---

### 4.8 CLAUDE.md status is outdated

States "Phase 3 (MCP Client) Complete" but Phase 4 PRs (#42-#47) are all merged.

---

### 4.9 Missing `CONTRIBUTING.md`

Project accepts external PRs but has no contributor guide.

---

### 4.10 Silent JSON parse failures in `ToolCallParser`

**File:** `include/zoo/engine/tool_call_parser.hpp:82-84`

Failed JSON parses are silently swallowed. Consider logging or incrementing a diagnostic counter for observability.

---

## Summary

| Priority | Count | Resolved | Remaining | Key Theme |
|----------|-------|----------|-----------|-----------|
| **Consider Changing** | 10 | — | 10 | API ergonomics, validation, documentation, test reliability |
| **Cleanup** | 10 | — | 10 | Conventions, magic numbers, minor inefficiencies |
