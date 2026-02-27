# Zoo-Keeper Production Readiness Review

**Date:** 2026-02-27
**Scope:** Full codebase audit — public API, engine, backend, MCP, tests, build, docs

---

## 1. SHOULD DEFINITELY CHANGE (Critical / Blocking)

### ~~1.1 Use-After-Free: `ErrorRecovery::validate_args()` dereferences pointer after lock release~~ ✅ Fixed

~~**Files:** `include/zoo/engine/error_recovery.hpp:49-86`, `include/zoo/engine/tool_registry.hpp:271-281`~~

~~`get_parameters_schema()` returns a raw pointer to data inside `ToolRegistry::tools_` while a `shared_lock` is held. The lock is released when `get_parameters_schema()` returns. `validate_args()` then dereferences this pointer across ~30 lines without any lock. A concurrent `register_tool()` (which takes a `unique_lock`) could reallocate the internal map, leaving a dangling pointer.~~

~~**Fix:** Return `std::optional<nlohmann::json>` by value from `get_parameters_schema()`, or keep the lock held for the duration of validation.~~

**Resolution:** `get_parameters_schema()` now returns `std::optional<nlohmann::json>` by value. The lock is fully released before `validate_args()` begins inspection. Tests updated accordingly.

---

### ~~1.2 `HistoryManager::get_messages()` returns a reference through a mutex~~ ✅ Fixed

~~**File:** `include/zoo/engine/history_manager.hpp:134-137`~~

~~```cpp~~
~~const std::vector<Message>& get_messages() const {~~
~~    std::lock_guard<std::mutex> lock(mutex_);~~
~~    return messages_;  // Lock released, reference escapes~~
~~}~~
~~```~~

~~The lock is released the instant the function returns, but the caller holds a reference to the internal vector. Any concurrent `add_message()` or `prune_oldest_messages_until()` invalidates the reference. The docstring even says "Copy of message history" but the signature returns `const&`.~~

~~This reference is used heavily in `AgenticLoop` (lines 125, 136, 216, 219, 223, 418) — every one of those call sites is a potential data race.~~

~~**Fix:** Change return type to `std::vector<Message>` (return by value).~~

**Resolution:** Return type changed to `std::vector<Message>`. Callers receive an independent snapshot; the mutex is released before the copy is returned. `GetMessagesReturnsConst` test replaced with `GetMessagesReturnsCopy` which verifies independence.

---

### ~~1.3 Unsafe `siglongjmp` in C++ — destructors skipped, mutexes orphaned~~ ✅ Fixed

~~**File:** `src/backend/llama_backend.cpp:276-317`~~

~~The GPU OOM recovery uses `sigsetjmp`/`siglongjmp` from a SIGABRT handler. When `siglongjmp` fires:~~
~~- C++ destructors for stack objects between `sigsetjmp` and the signal are **not** run.~~
~~- If `llama_decode()` held any internal mutex, it is permanently orphaned — the next call deadlocks.~~
~~- The recovery path constructs an `Error` with `std::string` allocation, which is unsafe if the heap is corrupted.~~

~~The code already documents this risk in comments (lines 272-274) but ships it anyway.~~

~~**Fix:** Replace with a `volatile sig_atomic_t` flag checked after `llama_decode()` returns. If `llama_decode()` itself aborts and never returns, the process is unrecoverable regardless — `siglongjmp` just delays the crash with corrupted state.~~

**Resolution:** Replaced `sigsetjmp`/`siglongjmp` with a plain file-scope signal handler (`gpu_oom_signal_handler`) that sets a `thread_local volatile std::sig_atomic_t` flag. The flag is checked after `llama_decode()` returns non-zero. Removed `<csetjmp>` include.

---

### ~~1.4 Subprocess vector creates dangling `const char*` pointers~~ ✅ Fixed

~~**File:** `include/zoo/mcp/transport/stdio_transport.hpp:68-84`~~

~~```cpp~~
~~std::vector<const char*> cmd_parts;~~
~~cmd_parts.push_back(config_.command.c_str());~~
~~for (const auto& arg : config_.args) {~~
~~    cmd_parts.push_back(arg.c_str());~~
~~}~~
~~```~~

~~The `cmd_parts` vector stores pointers to `config_.args` elements. If `config_` is moved or reassigned between construction and the `subprocess_create()` call, every pointer dangles. More immediately: if any push triggers reallocation of `cmd_parts`, all prior `const char*` values are still valid (they point into `config_`'s strings, not the vector), so this particular scenario is actually safe. However, the code lacks a `reserve()` call, making intent unclear and the pattern fragile.~~

~~Additionally, if `subprocess_create()` fails (line 86), `process_` may be partially initialized with no cleanup.~~

~~**Fix:** Add `cmd_parts.reserve(config_.args.size() + 2)` and document subprocess cleanup on failure.~~

**Resolution:** Added `cmd_parts.reserve(config_.args.size() + 2)` before the push loop with an explanatory comment. No reallocation can occur after the first `push_back`, keeping all `c_str()` pointers valid.

---

### ~~1.5 Documentation shows code that won't compile~~ ✅ Fixed

~~**Files:** `docs/getting-started.md:57-58`, `docs/examples.md` (26+ occurrences)~~

~~All user-facing examples call `.get()` directly on `agent->chat(...)`:~~

~~```cpp~~
~~auto response = agent->chat(zoo::Message::user("Hello")).get();  // WRONG~~
~~```~~

~~`chat()` returns `RequestHandle`, not `std::future`. The correct usage:~~

~~```cpp~~
~~auto handle = agent->chat(zoo::Message::user("Hello"));~~
~~auto response = handle.future.get();~~
~~```~~

~~The actual example binary (`examples/demo_chat.cpp:283`) uses the correct pattern, but every documentation page has the wrong one.~~

**Resolution:** All occurrences in `getting-started.md` and `examples.md` corrected to `.future.get()`. API table updated to show `RequestHandle` as the return type.

---

### ~~1.6 No LICENSE file despite MIT claim~~ ✅ Fixed

~~**File:** README.md line 92 states `## License: MIT` and displays a license badge. No `LICENSE` file exists in the repository. This creates legal ambiguity for any downstream user.~~

~~**Fix:** Add a standard MIT LICENSE file.~~

**Resolution:** `LICENSE` file added with standard MIT license text.

---

## 2. SHOULD CHANGE (High Priority)

### 2.1 FILE* TOCTOU race between `send()` and `disconnect()`

**File:** `include/zoo/mcp/transport/stdio_transport.hpp:112-114, 138-140`

`send()` calls `subprocess_stdin(&process_)` to get a `FILE*`. `disconnect()` calls `fclose()` on the same handle. If both execute concurrently, `send()` writes to a closed `FILE*` — undefined behavior. The `connected_` atomic doesn't fully protect this because the check-then-use window exists.

**Fix:** Store the `FILE*` handles at connect time. Use a mutex around `send()` and `disconnect()`, or use an atomic flag that `send()` checks before writing.

---

### 2.2 Unchecked JSON types from MCP server responses

**File:** `include/zoo/mcp/mcp_client.hpp:119-147`

The `discover_tools()` loop iterates over `(*result)["tools"]` without validating that each element is a JSON object. A malicious or buggy MCP server sending `"tools": [null, 42, "garbage"]` would silently create tools with empty names via `.value("name", "")`.

Similarly, `include/zoo/mcp/protocol/session.hpp:231-260` accesses `result["capabilities"]` without an `is_object()` check. A server returning `"capabilities": null` crashes.

**Fix:** Add `is_object()` guards before accessing fields. Return `McpProtocolError` for malformed responses.

---

### 2.3 Cancel tokens map grows unboundedly

**File:** `include/zoo/agent.hpp:222-228`

`cancel()` sets a flag but never removes the entry from `cancel_tokens_`. Over a long-running session with many requests, this map leaks memory linearly.

**Fix:** Erase the entry in `inference_loop()` after the request completes.

---

### 2.4 `ToolRegistry::register_tool()` throws instead of returning `Expected`

**File:** `include/zoo/engine/tool_registry.hpp:203-207`

Every other API in the library returns `Expected<T>`. This one throws `std::invalid_argument`. Callers wrapping registration in a try-catch are doing busywork that the `Expected` pattern was designed to eliminate.

**Fix:** Return `Expected<void>`.

---

### 2.5 O(n^2) pruning in `HistoryManager::prune_oldest_messages_until()`

**File:** `include/zoo/engine/history_manager.hpp:209-212`

Each `messages_.erase(begin + offset)` is O(n) because it shifts all subsequent elements. In a loop, this becomes O(n^2). For histories of thousands of messages, this is a noticeable stall on the inference thread.

**Fix:** Collect indices to remove, then erase in a single batch (e.g., `std::remove_if` + `erase`).

---

### 2.6 Read thread lifecycle is fragile

**File:** `include/zoo/mcp/transport/stdio_transport.hpp:96-98`

If `std::thread` construction throws (e.g., resource exhaustion), `connected_` was never set to `true`, but `process_` has already been spawned. The destructor path won't call `disconnect()` (because `connected_` is false), leaking the child process.

**Fix:** Wrap thread creation in try-catch. On failure, destroy the subprocess and return an error.

---

### 2.7 CI only tests on Ubuntu

**File:** `.github/workflows/ci.yml`

The project documents support for macOS (Metal), Linux (GCC/Clang), and Windows (MSVC). CI only runs `ubuntu-latest`. Metal acceleration, MSVC compilation, and platform-specific code paths (sysctl on macOS, /proc/meminfo on Linux) are never tested in CI.

**Fix:** Add a build matrix: `{ubuntu-latest, macos-latest, windows-latest}`. Enable Metal on macOS, test both GCC and Clang on Linux.

---

### 2.8 No sanitizers in CI

**File:** `CMakeLists.txt:62-68` (sanitizer flags configured), `.github/workflows/ci.yml` (not used)

ASan, TSan, and UBSan are fully wired into CMake but never exercised in CI. The use-after-free bugs documented above would likely be caught by ASan.

**Fix:** Add a CI job: `cmake -B build -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build`.

---

## 3. CONSIDER CHANGING (Medium Priority)

### 3.1 KV cache type uses magic integers

**File:** `include/zoo/types.hpp:269-275`

```cpp
int kv_cache_type_k = 1;  // 0=F32, 1=F16, 8=Q8_0, 2=Q4_0
int kv_cache_type_v = 1;
```

Users must memorize ggml enum values. A typo (`3` instead of `2`) silently passes validation and produces undefined behavior at runtime.

**Fix:** Define a `KvCacheType` enum: `{ F32 = 0, F16 = 1, Q4_0 = 2, Q8_0 = 8 }`.

---

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
| **Should Definitely Change** | 6 | ✅ 6 | 0 | Memory safety, signal safety, compilation errors in docs, missing LICENSE |
| **Should Change** | 8 | — | 8 | Thread safety, resource leaks, CI gaps, error handling consistency |
| **Consider Changing** | 10 | — | 10 | API ergonomics, validation, documentation, test reliability |
| **Cleanup** | 10 | — | 10 | Conventions, magic numbers, minor inefficiencies |
