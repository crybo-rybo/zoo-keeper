# Phase D: Core Model File Decomposition

## Context

`src/core/model.cpp` is a 750-line monolithic file mixing initialization, tokenization, sampling, inference, prompt formatting, grammar handling, history management, and KV cache operations. Phase D (Epic 4) decomposes it into focused files, adds RAII resource wrappers, and extracts testable pure-logic helpers. The public `Model` API remains unchanged.

## Slice 1: RAII Wrappers (Issue 4.2)

Do this first — it changes member types in `model.hpp` that all subsequent `.cpp` files compile against.

### New files
- **`include/zoo/internal/core/raii.hpp`** — Deleter structs + `unique_ptr` aliases (forward-declares llama types, no `<llama.h>`)
  ```cpp
  struct LlamaModelDeleter   { void operator()(llama_model* p) const noexcept; };
  struct LlamaContextDeleter { void operator()(llama_context* p) const noexcept; };
  struct LlamaSamplerDeleter { void operator()(llama_sampler* p) const noexcept; };
  using UniqueLlamaModel   = std::unique_ptr<llama_model, LlamaModelDeleter>;
  using UniqueLlamaContext  = std::unique_ptr<llama_context, LlamaContextDeleter>;
  using UniqueLlamaSampler  = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;
  ```
- **`src/core/raii.cpp`** — 3 one-liner deleter implementations calling `llama_model_free`, `llama_free`, `llama_sampler_free`

### Changes to existing files
- **`include/zoo/core/model.hpp`**: `#include "zoo/internal/core/raii.hpp"`, replace raw pointers with RAII types, destructor becomes `= default`
- **`src/core/model.cpp`**: Replace manual cleanup in destructor, change raw assignments to `.reset()`, sampler swap in `rebuild_sampler_with_grammar` uses move assignment
- **`CMakeLists.txt`**: Add `src/core/raii.cpp` to `add_library(zoo ...)`

### Notes
- `vocab_` stays raw — it's a non-owning borrowed pointer from `llama_model_get_vocab`
- Reverse declaration order of members (model, ctx, sampler) gives correct destruction order (sampler first)

## Slice 2: File Split (Issue 4.1)

Split model.cpp into focused files. Each implements Model member methods and includes `model.hpp` + `<llama.h>`.

| New file | Methods moved | Rationale |
|----------|--------------|-----------|
| `src/core/model.cpp` (slimmed, ~100 lines) | `initialize_global()`, `g_init_flag`, constructor, `~Model() = default`, `Model::load()` | Core lifecycle |
| `src/core/model_init.cpp` | `initialize()`, `tokenize()` | Backend setup + tokenization |
| `src/core/model_inference.cpp` | `run_inference()`, `generate()`, `generate_from_history()` | Generation pipeline (tight call chain) |
| `src/core/model_prompt.cpp` | `build_llama_messages()`, `format_prompt()`, `finalize_response()`, `clear_kv_cache()` | Prompt delta + KV cache state |
| `src/core/model_history.cpp` | `set_system_prompt()`, `add_message()`, `get_history()`, `clear_history()`, `context_size()`, `estimated_tokens()`, `is_context_exceeded()`, `estimate_tokens()`, `find_stop_sequence()`, `trim_history_to_fit()` | History & context bookkeeping |
| `src/core/model_sampling.cpp` | `set_tool_grammar()`, `clear_tool_grammar()`, `rebuild_sampler_with_grammar()`, `create_sampler_chain()`, `add_sampling_stages()`, `add_dist_sampler()` | Sampler chain + grammar |

### Static helper relocation
- `make_sampler_seed()` → anonymous namespace in `model_sampling.cpp` (only caller: `add_dist_sampler`)
- `invoke_token_callback()` → anonymous namespace in `model_inference.cpp` (only caller: `run_inference`)
- `g_init_flag` + `initialize_global()` stay in slimmed `model.cpp` (called by `initialize()` in `model_init.cpp` via the Model member function)

### CMake update
```cmake
add_library(zoo STATIC
    src/agent.cpp
    src/agent/backend_model.cpp
    src/agent/runtime.cpp
    src/core/model.cpp
    src/core/model_init.cpp
    src/core/model_inference.cpp
    src/core/model_prompt.cpp
    src/core/model_history.cpp
    src/core/model_sampling.cpp
    src/core/raii.cpp
)
```

## Slice 3: Extract Pure-Logic Helpers + Tests (Issue 4.3)

Extract testable free functions following the `batch.hpp` pattern (inline, header-only, no llama dependency).

### New internal headers
- **`include/zoo/internal/core/stop_sequence.hpp`** — `find_stop_sequence(string_view text, vector<string> stops) -> size_t`
  - Currently a `const` Model method with no member access — pure function
- **`include/zoo/internal/core/history_trim.hpp`** — `compute_history_trim(vector<Message>, size_t max) -> optional<TrimResult>`
  - Extracts the index-computation logic from `trim_history_to_fit()`; Model method calls it and applies the result

### New test files
- **`tests/unit/test_stop_sequence.cpp`** — suffix matching, empty sequences, no match, multiple matches
- **`tests/unit/test_history_trim.cpp`** — boundary conditions, system prompt preservation, role-boundary alignment

### Changes
- `model_history.cpp`: call extracted free functions instead of inline logic
- `model_inference.cpp`: call `find_stop_sequence` from internal header instead of Model method
- `model.hpp`: remove `find_stop_sequence` from private methods (it becomes a free function)
- `tests/CMakeLists.txt`: add both new test files to `zoo_tests`

## Verification

1. **Build**: `cmake -B build -DZOO_BUILD_TESTS=ON && cmake --build build`
2. **Run tests**: `ctest --test-dir build` — all existing tests pass, new tests included
3. **Sanitizers**: `cmake -B build -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build`
4. **Verify no public API change**: Examples still compile without modification
5. **Verify no installed header change**: `include/zoo/internal/core/raii.hpp` and other internal headers are already excluded from install
