# Production Readiness Audit

**Date:** 2026-03-07
**Version:** 0.2.0
**Auditor:** Claude (Opus 4.6)

## Re-evaluation (2026-03-08)

The P1 section of this audit is now stale relative to the current tree on `main`.

- P1-5 is already implemented in [include/zoo/core/types.hpp](/Users/conorrybacki/Programs/zoo-keeper/include/zoo/core/types.hpp).
- P1-6 is already implemented in [src/core/model.cpp](/Users/conorrybacki/Programs/zoo-keeper/src/core/model.cpp).
- P1-7 is already implemented in [include/zoo/core/types.hpp](/Users/conorrybacki/Programs/zoo-keeper/include/zoo/core/types.hpp), but [docs/configuration.md](/Users/conorrybacki/Programs/zoo-keeper/docs/configuration.md) had not been updated.
- P1-8 is already implemented as optional stderr logging in [include/zoo/internal/log.hpp](/Users/conorrybacki/Programs/zoo-keeper/include/zoo/internal/log.hpp) and wired from [CMakeLists.txt](/Users/conorrybacki/Programs/zoo-keeper/CMakeLists.txt).
- P1-9 is already implemented in [include/zoo/core/types.hpp](/Users/conorrybacki/Programs/zoo-keeper/include/zoo/core/types.hpp) and [include/zoo/agent.hpp](/Users/conorrybacki/Programs/zoo-keeper/include/zoo/agent.hpp).
- P1-10 is already partially implemented via [CHANGELOG.md](/Users/conorrybacki/Programs/zoo-keeper/CHANGELOG.md) and generated version constants in [include/zoo/version.hpp.in](/Users/conorrybacki/Programs/zoo-keeper/include/zoo/version.hpp.in). Release tagging remains process work outside this repository change.

---

## P0 — Critical (Must fix before any production use)

### 1. CMake install is broken — `find_package(ZooKeeper)` will fail

`cmake/ZooKeeperConfig.cmake.in:7` includes `ZooKeeperTargets.cmake`, but `CMakeLists.txt` never generates it. The `install(TARGETS ...)` call (line 64) does not use `EXPORT ZooKeeperTargets`, and no `install(EXPORT ...)` rule exists.

**Impact:** Any downstream project using `find_package(ZooKeeper)` after `make install` will get a CMake error.

**Fix:** Add export set to install rules:
```cmake
install(TARGETS zoo zoo_core EXPORT ZooKeeperTargets ...)
install(EXPORT ZooKeeperTargets
    NAMESPACE ZooKeeper::
    DESTINATION lib/cmake/ZooKeeper)
```

### 2. Uncaught exceptions from user callbacks crash the inference thread

`src/core/model.cpp:216-218` — The user-provided `on_token` callback is invoked with no try-catch. If it throws, the exception propagates out of the inference loop and terminates the inference thread. All queued and future requests will hang indefinitely (promises never fulfilled).

`include/zoo/agent.hpp:227-265` — `inference_loop()` has no top-level exception guard either. Any unexpected exception (from callbacks, tool handlers, or std::bad_alloc) kills the thread silently.

**Impact:** One bad callback permanently bricks the Agent.

**Fix:** Wrap callback invocations in try-catch, and add a top-level catch in `inference_loop()` that sets errors on remaining promises.

### 3. Null pointer dereference if model has no chat template

`src/core/model.cpp:117` — `tmpl_` is assigned from `llama_model_chat_template()` which returns `nullptr` when the model lacks a chat template. This `nullptr` is later passed to `llama_chat_apply_template()` (called in `format_prompt()`), causing undefined behavior.

**Impact:** Loading any model without an embedded chat template (some GGUF quantizations) will segfault on first inference.

**Fix:** Check `tmpl_` after assignment; return `Error{ErrorCode::TemplateRenderFailed, "Model has no chat template"}` if null.

### 4. No integration tests for Model or Agent

Per CLAUDE.md: "Model/Agent are tested via integration tests." However, zero integration tests exist. The two core layers of the library (Model wrapping llama.cpp, Agent orchestrating threading + tool loops) have **no automated test coverage**. Unit tests cover only pure-logic helpers (types, tool registry, parser).

**Impact:** Regressions in inference, history management, KV cache, threading, cancellation, and the tool loop will go undetected.

**Fix:** Add integration test suite (can gate behind `ZOO_BUILD_INTEGRATION_TESTS` and a model fixture).

---

## P1 — High (Should fix before production release)

### 5. No sampling parameter validation

`include/zoo/core/types.hpp:163-174` — `Config::validate()` checks `model_path`, `context_size`, and `max_tokens` only. `SamplingParams` is not validated at all. Invalid values silently produce broken behavior:

| Parameter | Risk |
|-----------|------|
| `temperature < 0` | Undefined sampler behavior |
| `top_p < 0` or `top_p > 1` | Probability math fails |
| `top_k = 0` | Empty candidate set, crash or hang |
| `repeat_penalty < 0` | Nonsensical penalties |

**Fix:** Add validation in `Config::validate()` for all sampling parameters.

### 6. Inference loop has no timeout or maximum token safeguard

`src/core/model.cpp:195` — The autoregressive `while (true)` loop only breaks on EOG token, stop sequence, user callback stop, max_tokens, or context exhaustion. With `max_tokens = -1` (the default), a model that never emits EOG will generate until context fills — potentially millions of tokens with large contexts.

**Impact:** Runaway generation consumes unbounded CPU time with no cancellation path from the Model layer.

**Fix:** Add an absolute `max_generation_tokens` safety limit (e.g., `context_size`) even when `max_tokens = -1`.

### 7. Unbounded request queue by default

`include/zoo/core/types.hpp:159` — `request_queue_capacity = 0` means unbounded. Under load, memory grows without limit.

**Impact:** OOM under sustained high request rates.

**Fix:** Default to a reasonable bound (e.g., 64 or 128) and document the `0 = unlimited` behavior.

### 8. No logging or observability

The entire library produces zero diagnostic output. There is no optional logging, no metrics export, and no way to observe internal state (KV cache utilization, queue depth, tool retry counts, inference timing) without instrumenting user code.

**Impact:** Production debugging is extremely difficult. Users have no visibility into why requests are slow, failing, or producing unexpected output.

**Fix:** Add optional structured logging (e.g., callback-based or spdlog). At minimum, log errors and tool loop iterations.

### 9. `max_tool_iterations` is hardcoded

`include/zoo/agent.hpp:288` — `constexpr int max_tool_iterations = 5`. This is not configurable via `Config`. Similarly, `ErrorRecovery` max retries is hardcoded to 2 (`tools/validation.hpp`).

**Impact:** Users cannot tune the agentic loop to their use case. Some tasks legitimately need more iterations; some safety-critical contexts need fewer.

**Fix:** Add `max_tool_iterations` and `max_tool_retries` to `Config`.

### 10. No CHANGELOG or release process

Version is 0.2.0 with no CHANGELOG, no git tags, no GitHub releases, and no version constants exposed in C++ headers. Breaking changes (KV cache default, batch API refactor, interceptor addition) are undocumented for downstream consumers.

**Impact:** Consumers have no way to know what changed between versions or whether an upgrade is safe.

**Fix:** Create `CHANGELOG.md`, tag releases, and generate a `version.hpp` at configure time.

---

## P2 — Medium (Should fix before 1.0)

### 11. Seed entropy is weak

`src/core/model.cpp:500` — When `seed = -1`, uses `time(nullptr)` (second-precision). Multiple Agents created in the same second will share identical seed values.

**Fix:** Use `std::random_device` or `std::chrono::high_resolution_clock`.

### 12. `n_gpu_layers` defaults to -1 (offload everything)

`include/zoo/core/types.hpp:148` — Default `-1` tells llama.cpp to offload all layers to GPU. On systems with insufficient VRAM, this causes allocation failure or OOM-kill with no helpful error.

**Fix:** Default to `0` (CPU-only) and document GPU offloading as opt-in.

### 13. History can grow unbounded

`Model` stores all conversation messages in `messages_` with no cap. Long-running conversations will eventually exhaust memory or exceed context window without clear error.

**Fix:** Add `max_history_messages` config or automatic truncation strategy.

### 14. No code formatting or static analysis enforcement

No `.clang-format` or `.clang-tidy` in the project root. No formatting CI check. Contributors cannot ensure consistent style. The CI runs sanitizers but no static analysis (clang-tidy, cppcheck).

**Fix:** Add `.clang-format`, `.clang-tidy`, and a CI formatting check job.

### 15. `.gitignore` is incomplete

Current `.gitignore`:
```
.vscode/**
.cache/**
.claude/**
build/**
tests/cmake_test_discovery_*
```

Missing entries: `.DS_Store`, `*.swp`, `*.swo`, `.idea/`, `compile_commands.json`, `*.gcda`, `*.gcno`, `*.o`, `*.a`, `*.so`, `*.dylib`.

### 16. No coverage reporting in CI

`ci.yml` does not have a coverage job despite `ZOO_ENABLE_COVERAGE` being a supported option. No coverage is tracked or uploaded.

**Fix:** Add a coverage CI job that uploads to Codecov or similar.

### 17. Cancel flag not checked during prefill

`include/zoo/agent.hpp:295` — Cancellation is only checked between tool loop iterations, not during the potentially long prefill decode (`model.cpp:174-188`) or autoregressive generation. A cancel request during a long generation must wait for it to complete.

**Fix:** Thread the cancel token into the token callback so cancellation is responsive.

### 18. `formatted_` buffer uses context-scaled allocation

`src/core/model.cpp:118` — `formatted_.resize(context_size_ * 4)` allocates `context_size * 4` bytes on the heap. With a 128K context, this is 512KB — fine. But with a hypothetical 1M context, this is 4MB allocated upfront and potentially insufficient for multi-turn conversations anyway.

**Fix:** Use a dynamically growing buffer or document the sizing assumption.

---

## P3 — Low (Nice to have)

### 19. No `noexcept` specifications

Public APIs don't declare `noexcept` even where appropriate (e.g., `get_history()`, `Message` default operations). This prevents move optimizations in containers and makes exception guarantees unclear.

### 20. Tool call ID counter never resets

`include/zoo/tools/parser.hpp:77-80` — Static `std::atomic<int> counter` monotonically increases forever. Not a real problem (32-bit int overflow at ~2 billion), but semantically surprising across parse calls.

### 21. No API reference documentation

Users must read header files directly. No Doxygen configuration, no generated API docs, no `///` doc comments on public interfaces.

### 22. Single example application

Only `demo_chat.cpp` exists. Missing examples for:
- Standalone `Model` usage (without Agent)
- Error handling patterns
- Custom tool implementation with complex types
- Streaming with cancellation

### 23. No `CONTRIBUTING.md` or `SECURITY.md`

Standard open-source project files are missing.

### 24. No pkg-config support

Only CMake `find_package` integration (and that's broken, see P0-1). No `.pc` file for non-CMake build systems.

### 25. Missing GoogleTest license attribution

`README.md` credits llama.cpp and nlohmann/json but not GoogleTest. No `LICENSES/` directory with third-party license copies.

### 26. README claims "54 passing tests" — actual count is 92

README is stale. Test count has grown but documentation was not updated.

---

## Summary

| Priority | Count | Description |
|----------|-------|-------------|
| **P0 — Critical** | 4 | Broken install, crash bugs, zero integration tests |
| **P1 — High** | 6 | Missing validation, no safeguards, no observability |
| **P2 — Medium** | 8 | Defaults, tooling, CI gaps |
| **P3 — Low** | 8 | Documentation, polish, attribution |
| **Total** | **26** | |

### Recommended fix order

1. P0-1: Fix CMake export targets (blocks all downstream users)
2. P0-2: Add exception safety to inference thread (crash bug)
3. P0-3: Guard against null chat template (crash bug)
4. P1-5: Validate sampling parameters (silent corruption)
5. P0-4: Add integration tests (confidence gate)
6. P1-6: Add inference timeout safeguard
7. P1-7: Bound the request queue
8. P1-8: Add basic logging
9. Everything else in priority order
