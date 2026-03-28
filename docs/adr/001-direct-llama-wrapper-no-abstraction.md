# ADR-001: Direct llama.cpp Wrapper Without Backend Abstraction

## Status

Accepted

## Context

Early designs considered an `IBackend` interface to abstract over the llama.cpp
runtime, enabling mock-based testing and potential alternative backends. This
would have placed a virtual dispatch layer between `Model` and llama.cpp.

## Decision

`zoo::core::Model` directly owns and calls llama.cpp resources (`llama_model*`,
`llama_context*`, `llama_sampler*`) and accesses `llama_vocab*` through the
model, without an intervening abstraction layer.

A private `AgentBackend` seam exists **only** in the internal agent layer to
decouple the runtime orchestrator from the model for testing purposes. This seam
is not exposed in the public API.

## Rationale

- **Simplicity:** A single concrete wrapper is easier to reason about, debug,
  and maintain than a virtual interface hierarchy.
- **Performance:** No virtual dispatch overhead on the hot inference path.
- **Honest testing:** Unit tests cover pure logic (types, tools, parsing).
  Model behavior is validated through integration tests against real GGUF models,
  which catches issues that mocks would hide (tokenization edge cases, KV cache
  behavior, sampling distribution).
- **Narrow scope:** Zoo-Keeper targets llama.cpp specifically. Multi-backend
  support is a non-goal.

## Consequences

- Model cannot be mocked in unit tests. Integration tests require a real model file.
- Swapping llama.cpp for another backend would require rewriting `Model`.
- The internal `AgentBackend` seam can be used for runtime-level testing without
  a live model, but this is a private implementation detail.
