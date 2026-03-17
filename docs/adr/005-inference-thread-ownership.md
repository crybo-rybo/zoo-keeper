# ADR-005: Agent Owns the Inference Thread

## Status

Accepted

## Context

Two threading models were considered:
1. **Caller-managed:** The caller runs inference on their own thread, with Model
   providing thread-safety guarantees.
2. **Agent-managed:** Agent owns a dedicated inference thread; callers submit
   requests and receive futures.

## Decision

`zoo::Agent` owns a background inference thread. Callers submit requests via
`chat()` / `complete()` and receive `std::future<Response>`. All callbacks and
tool handlers execute on the inference thread.

`zoo::core::Model` remains single-threaded and not thread-safe. Agent protects
it with `model_mutex_`.

## Rationale

- **Safety:** A single owner thread eliminates data races on llama.cpp state
  (KV cache, sampling context, token buffers) by construction rather than by
  convention.
- **Simplicity for callers:** Callers get a future-based async API without
  needing to understand llama.cpp threading constraints.
- **Agentic loop:** The tool execution loop (generate → detect → execute →
  re-generate) runs entirely on the inference thread, avoiding cross-thread
  state synchronization for tool results and history updates.
- **Model stays simple:** `Model` can remain a straightforward synchronous
  wrapper without thread-safety complexity in its implementation.

## Consequences

- Callers cannot directly access Model while Agent is running.
- All callbacks execute on the inference thread — callers must not block in
  callbacks or they will stall inference.
- Cancellation is cooperative (checked between generation steps).
- `Model` is still usable standalone for simple single-threaded use cases.
