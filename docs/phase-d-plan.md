# Phase D: Model Decomposition and Prompt-State Cleanup

> Status: Completed for 1.0. This document is retained as historical implementation context.

## Summary

Phase D completes Epic 4 from the cleanup roadmap by restructuring `zoo::core::Model` around three goals:

- split `src/core/model.cpp` into smaller responsibility-focused translation units,
- replace manual llama resource cleanup with model-private RAII ownership,
- centralize incremental prompt rendering and KV-cache invalidation so the rules are explicit and testable.

The public `Model` API stays unchanged. No new public installed headers are added.

## Key Changes

### 1. Resource ownership: keep it model-private and install-safe

- Replace the owning raw members for `llama_model`, `llama_context`, and `llama_sampler` with `std::unique_ptr` members using private custom deleters declared inside `include/zoo/core/model.hpp`.
- Keep `vocab_` and `tmpl_` as borrowed non-owning pointers.
- Define the deleter call operators and `Model::~Model()` out-of-line in the slimmed `src/core/model.cpp`, after including `<llama.h>`.
- Do not add `raii.hpp` or `raii.cpp`. If a separate file is needed after implementation starts, use a domain name such as `model_resources.cpp`, not a generic RAII file.
- Update initialization and grammar-rebuild paths to construct new handles first and only replace the active member after success, preserving current failure behavior.

### 2. Prompt/KV bookkeeping: make the state explicit

- Replace the current ad hoc prompt state fields with one private `PromptState` grouping inside `Model`.
- `PromptState` owns:
  - committed prompt prefix length,
  - formatted prompt buffer,
  - cached `llama_chat_message` view,
  - explicit cache invalidation state based on a dirty flag or revision counter.
- Rename internal-only prompt helpers for clarity:
  - `format_prompt()` -> `render_prompt_delta()`
  - `prev_len_` -> `committed_prompt_len_`
  - `formatted_` -> `formatted_prompt_`
- Make `src/core/model_prompt.cpp` the only place that mutates prompt checkpoint or message-cache state.
- Use explicit invalidation rules:
  - any history content change marks cached llama messages dirty,
  - any history shrink or reset clears the KV cache and resets committed prompt length,
  - `finalize_response()` is the only path that advances the committed prompt checkpoint after a successful turn.
- Keep behavior unchanged for generation, rollback on failure, and tool-loop usage.

### 3. File decomposition: split by responsibility, not mechanically

- `src/core/model.cpp`
  - one-time backend init,
  - constructor,
  - destructor,
  - factory `load()`,
  - private deleter definitions.
- `src/core/model_init.cpp`
  - `initialize()`,
  - `tokenize()`.
- `src/core/model_inference.cpp`
  - `run_inference()`,
  - `generate()`,
  - `generate_from_history()`,
  - token-callback helper,
  - stop-sequence helper if it remains inference-local.
- `src/core/model_prompt.cpp`
  - prompt-state helper logic,
  - cached llama-message view rebuild,
  - `render_prompt_delta()`,
  - `finalize_response()`,
  - `clear_kv_cache()`.
- `src/core/model_history.cpp`
  - `set_system_prompt()`,
  - `add_message()`,
  - `get_history()`,
  - `clear_history()`,
  - `estimate_tokens()`,
  - `trim_history_to_fit()`,
  - context bookkeeping accessors.
- `src/core/model_sampling.cpp`
  - sampler-chain construction,
  - grammar sampler rebuild,
  - sampler-seed helper.

Only add a private helper header if it is shared by multiple `.cpp` files or directly unit-tested. Do not make any public header depend on a non-installed internal header.

### 4. Tests and verification

- Add focused unit coverage for the extracted prompt bookkeeping helper or state machine:
  - same-size content mutation invalidates cached llama messages,
  - shorter rendered prompt forces KV reset,
  - `finalize_response()` advances committed prompt length only after successful rendering,
  - clearing history resets prompt checkpoint state.
- Keep `find_stop_sequence` extraction optional. If it is extracted naturally during the split, add a small unit test; do not make it the core deliverable for Issue 4.3.
- Do not add a permanent `history_trim.hpp` unless trim-boundary logic is genuinely reused and worth testing independently.
- Update `tests/CMakeLists.txt` only for new targeted tests that directly support Phase D.
- Verification:
  - `cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON`
  - `cmake --build build -j4`
  - `ctest --test-dir build --output-on-failure`
  - `cmake -B build -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON`
  - `cmake --build build -j4`
  - `ctest --test-dir build --output-on-failure`

## Assumptions

- Public `zoo::core::Model` behavior and signatures remain stable through Phase D.
- Internal names may change for clarity; no new public API is introduced.
- No new installed headers are added.
- Descriptive domain names are preferred over generic utility names; avoid `raii.*`.
