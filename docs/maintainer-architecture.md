# Maintainer Architecture

This note documents the private module boundaries behind the public Zoo-Keeper API. It is for contributors working on runtime internals, build surface cleanup, and structural refactors. Public-facing guidance stays in [architecture.md](architecture.md).

## Boundary Rules

- Only headers under `include/zoo/` are part of the supported installed API.
- Headers under `include/zoo/internal/` are private implementation detail and must not be installed or documented as consumer dependencies.
- Public headers should describe API shape, not carry large runtime implementations.
- Prefer domain-specific types and seams over generic utility layers.

## Runtime Ownership

### Public facade

- `zoo::Agent` is a thin public facade.
- Construction, destruction, and public forwarding live in `src/agent.cpp`.
- The facade owns configuration and a private implementation handle only.

### Private runtime

- `internal::agent::AgentRuntime` owns the inference thread and the orchestration loop.
- The runtime owns:
  - the request mailbox
  - request tracking and cancellation state
  - the tool registry used during the tool loop
  - the backend seam used to talk to the model layer
- Calling-thread operations that need model state are routed into the runtime instead of touching the model directly.

### Backend seam

- `internal::agent::AgentBackend` is the only model-facing interface the runtime depends on.
- The production adapter wraps `zoo::core::Model`.
- Tests use fake backends through this seam to validate orchestration behavior without a live model.
- This seam is intentionally private; it exists to test the runtime, not to become a second public model abstraction.

## Internal Agent Modules

| Module | Responsibility |
|--------|----------------|
| `internal/agent/runtime.*` | Worker thread, request processing, tool loop |
| `internal/agent/backend.*` | Runtime-to-model seam |
| `internal/agent/backend_model.*` | Production adapter around `zoo::core::Model` |
| `internal/agent/mailbox.hpp` | Request and command queueing |
| `internal/agent/request_tracker.hpp` | Request ids, futures, cancellation flags, cleanup |
| `internal/agent/command.hpp` | Typed control operations applied on the inference thread |

Keep responsibilities narrow. If a change affects queueing, cancellation, command routing, and request execution at once, it usually belongs in a smaller extracted unit instead.

## Core Model Structure

`zoo::core::Model` remains the direct llama.cpp wrapper. Its public API is stable, but its implementation is split by responsibility:

| File | Responsibility |
|------|----------------|
| `src/core/model.cpp` | construction, destruction, factory, one-time backend setup |
| `src/core/model_init.cpp` | initialization and tokenization |
| `src/core/model_inference.cpp` | generation and inference flow |
| `src/core/model_prompt.cpp` | prompt delta rendering and KV-cache bookkeeping |
| `src/core/model_history.cpp` | history mutation and trimming |
| `src/core/model_sampling.cpp` | sampler construction and grammar updates |

Contributor rules:

- llama resource ownership stays model-private
- prompt and KV-cache bookkeeping stays localized, not spread through unrelated files
- no new public headers should be added for model internals unless the API truly needs them

## Tooling Boundaries

- `include/zoo/tools/*` contains the supported public tool API.
- `ToolRegistry` owns normalized metadata and invocation wiring.
- Parser and validator operate on strings and JSON, not on model internals.
- `include/zoo/internal/tools/*` contains private grammar/interceptor helpers used by the runtime.

Maintain these invariants:

- supported schema behavior must stay explicit and deterministic
- grammar emission must not depend on unstable container iteration
- tool-loop observability should use explicit domain types such as `ToolInvocation`, not overloaded chat-history messages

## Documentation Split

- `architecture.md` explains the public layers, targets, and user-visible threading guarantees
- `maintainer-architecture.md` explains private ownership and implementation seams

If a document starts teaching private command types, mailbox structure, or backend adapter details to normal consumers, that content belongs here instead.
