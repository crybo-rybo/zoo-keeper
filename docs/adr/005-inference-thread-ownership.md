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
`chat()` / `complete()` and receive `RequestHandle<TextResponse>`. All callbacks
and tool handlers execute on the inference thread.

`zoo::core::Model` remains single-threaded and not thread-safe. Agent protects
it by confining all access to the single inference thread.

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
- Tool handlers execute on the inference thread — callers must not block in
  tool handlers or they will stall inference. Streaming token callbacks run on
  the CallbackDispatcher thread; blocking in a streaming callback backs up the
  dispatcher queue rather than stalling inference.
- Cancellation is cooperative (checked between generation steps).
- `Model` is still usable standalone for simple single-threaded use cases.
