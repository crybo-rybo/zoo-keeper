# Transition Plan: Move Away from `llama_batch_get_one()`

## Goal
Implement prompt prefill and token decode using explicit `llama_batch` construction (`llama_batch_init` / `llama_batch_free`) instead of `llama_batch_get_one()`, while preserving current behavior and improving long-prompt robustness.

## Why Transition
- `llama_batch_get_one()` is documented as a transition helper and marked as a pattern to avoid for new code.
- The current implementation sends the full prompt as one decode batch; this makes `n_batch` tuning less safe and can fail on long prompts.
- Explicit batch construction gives direct control over:
  - chunked prefill (`<= n_batch`)
  - token positions and sequence IDs
  - logits emission policy
  - future expansion to multi-sequence batching
- This aligns `zoo-keeper` with current llama.cpp API direction and reduces reliance on convenience wrappers.

## Current State (Repository)
- Prefill path currently uses a single `llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size())` batch.
- Generation path currently uses `llama_batch_get_one(&new_token, 1)` for iterative decode.
- `n_batch` and `n_ubatch` are configured in context params, but prefill is not chunked against those limits.

## Implementation Plan

### Phase 1: Introduce Local Batch Abstraction
- Add an internal RAII helper in `src/core/model.cpp` or a private translation-unit helper:
  - owns `llama_batch` allocated via `llama_batch_init(max_tokens, 0, 1)`
  - frees with `llama_batch_free` in destructor
  - provides `clear()` and `push(token, pos, logits)` helpers
- Keep helper private to `core::Model` implementation (no public API changes).

Reasoning:
- Encapsulates manual array handling and avoids leaks/invalid memory access.
- Reusable for both prompt prefill and 1-token generation steps.

### Phase 2: Replace Prefill With Chunked Decode
- Replace single prefill decode with a loop over prompt tokens in chunks of `effective_batch_size`.
- Compute `effective_batch_size` as `min(configured_n_batch_or_ctx, llama_n_batch(ctx_))`.
- For each chunk:
  - fill local batch entries with explicit `pos` and seq-id `0`
  - set `logits=true` only on the final token of the final chunk
  - call `llama_decode(ctx_, batch)`
- On decode return code `1`, retry with reduced chunk size (e.g., halve until 1), then return a clear error if still failing.

Reasoning:
- Prevents long prompts from hard-failing solely due to single oversized prefill batch.
- Mirrors resilience pattern used by llama.cpp examples for constrained KV situations.

### Phase 3: Replace 1-Token Decode Path
- Replace `llama_batch_get_one(&new_token, 1)` with explicit single-entry batch fill.
- Preserve existing generation loop semantics and stop conditions.

Reasoning:
- Completes transition and removes dependency on helper API entirely.

### Phase 4: Optional Config Surface Improvements
- Expose `n_batch` and `n_ubatch` in `Config` with safe defaults and validation:
  - default `n_batch = context_size` (current behavior)
  - default `n_ubatch = min(512, n_batch)` (or keep current default where stable)
  - validate positive values and `n_ubatch <= n_batch`
- Keep backward compatibility by maintaining existing defaults when fields are unset.

Reasoning:
- Allows tuning memory/latency tradeoffs without recompilation.

### Phase 5: Tests
- Add/update unit tests around model prefill behavior (and integration tests where feasible):
  - prompt length near and above `n_batch`
  - fallback path when decode returns `1`
  - unchanged behavior for short prompts
  - metrics and stop-sequence behavior remain intact
- If full llama runtime tests are expensive, isolate and test chunking logic in helper-level tests.

Reasoning:
- Most regression risk is in batching edge cases and fallback loops.

## Risks and Mitigations
- Risk: Incorrect `pos`/`logits` population can distort outputs.
  - Mitigation: helper abstraction + focused tests for token position progression.
- Risk: Retry logic on decode warnings may hide true failures.
  - Mitigation: bounded retries, explicit error reporting with return code and chunk size.
- Risk: Performance regression from overly conservative chunking.
  - Mitigation: start with `n_batch` chunks and reduce only on decode failure.

## Acceptance Criteria
- No callsites remain for `llama_batch_get_one()` in project code.
- Long prompts above prior single-batch limits can prefill successfully when context headroom permits.
- Existing tests pass; new batching/chunking tests pass.
- Baseline short-prompt latency regression is negligible (or documented with rationale).

## Rollout
- Step 1: Land internal helper + decode path migration.
- Step 2: Land config-surface extension for `n_batch` / `n_ubatch` (if approved).
- Step 3: Document tuning guidance in `docs/performance-tuning.md` or equivalent.

