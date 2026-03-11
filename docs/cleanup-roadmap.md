# Cleanup Roadmap

## Purpose

This roadmap turns the current repository audit into a concrete implementation plan focused on:

- simplifying the public API and project structure,
- reducing architectural ambiguity,
- removing technical debt that slows future changes,
- making the library feel more conventional and easier to contribute to,
- preserving working behavior while making the codebase easier to reason about.

This is intentionally opinionated. It favors a smaller number of clearer concepts over preserving every current abstraction boundary.

## Guiding Principles

1. **One obvious public story** — consumers should not need to understand internal layering details to use the library.
2. **Move implementation out of public headers** — headers should define API, not carry runtime subsystems.
3. **Prefer explicit domain types over overloaded generic containers** — especially for tool execution results and events.
4. **Keep internal mechanisms private** — request queues, worker envelopes, logging helpers, and grammar plumbing should not leak into the installed API surface.
5. **Tighten contract/documentation alignment** — if the docs claim a behavior, the runtime must implement it or the docs must narrow the claim.
6. **Refactor with test seams, not fear** — each major reshape should leave behind better unit-test coverage than before.

## Proposed End State

The target shape for the codebase is:

- a single clear public runtime target for normal consumers,
- private/internal implementation files for agent orchestration and llama backend integration,
- a unified tool API that supports both typed registration and manual schema registration through the same public entry points,
- explicit tool event/result types in responses,
- deterministic tool/schema ordering,
- build/package/install behavior that matches the documented architecture,
- documentation that describes only what the code truly supports.

## Workstreams

The work is organized into epics. Each epic is broken into concrete issues with acceptance criteria.

---

## Epic 1 — Public API and Build Surface Simplification

**Goal:** make the library structure obvious, conventional, and less header-driven.

**Why first:** this sets the shape for every later refactor and immediately reduces confusion for users and contributors.

### Issue 1.1 — Collapse the public target story

**Problem**
- The docs present a clean layered architecture, but the build/package shape exposes a hybrid model that is harder to understand.
- Normal users should not need to reason about `INTERFACE` vs `STATIC` layering to link the library correctly.

**Tasks**
- Choose a single primary public target, recommended: `zoo` as the main compiled library target.
- Rework CMake so the install-tree and build-tree expose the same public target story.
- Keep any internal targets private unless there is a strong, documented reason to expose them.
- Update docs/examples to link the new canonical target name.

**Acceptance criteria**
- There is one documented primary target for consumers.
- `FetchContent`, subdirectory use, and installed-package use all share the same linking guidance.
- Architecture docs and build docs no longer contradict the actual target layout.

### Issue 1.2 — Move `Agent` implementation out of the public header

**Problem**
- `include/zoo/agent.hpp` currently contains a large amount of runtime implementation and internal coordination logic.
- This increases compile cost and leaks internals into the public API surface.

**Tasks**
- Reduce `include/zoo/agent.hpp` to a public declaration-oriented header.
- Move implementation into `src/agent.cpp` and private headers under `src/` or `include/zoo/detail/` only if truly necessary.
- Hide internal types such as request envelopes and queues from the installed surface.

**Acceptance criteria**
- Public `Agent` header mostly describes API, not the worker loop implementation.
- Internal runtime machinery is not installed as part of the public include tree.
- Existing examples continue to compile against the cleaner API.

### Issue 1.3 — Stop installing internal headers and dependency headers directly

**Problem**
- Internal headers are currently part of the installed tree.
- The package currently vendors `nlohmann` headers into the install tree instead of treating dependencies more cleanly.

**Tasks**
- Limit installed headers to the actual supported public API.
- Remove `include/zoo/internal/*` from the public install surface.
- Revisit dependency propagation so install behavior is less ad hoc.

**Acceptance criteria**
- Installed headers match the intended public API boundary.
- Private implementation headers are not published.
- Packaging is simpler and easier to explain.

---

## Epic 2 — Tooling API Unification and Contract Hardening

**Goal:** turn the tool system into one coherent model instead of a set of adjacent mechanisms.

**Why second:** tool calling is central to the library identity, and today it is the most conceptually fragmented subsystem.

### Issue 2.1 — Add manual-schema registration directly to `Agent`

**Problem**
- Typed tool registration is available on `Agent`, but manual-schema registration is only available via `ToolRegistry`.
- This makes the “easy path” and the “advanced path” feel like different APIs.

**Tasks**
- Add `Agent::register_tool(...)` overloads for schema + handler registration.
- Keep grammar refresh behavior consistent across typed and manual registration paths.
- Update docs/examples to show both forms through the same public surface.

**Acceptance criteria**
- Consumers can use typed and manual-schema tool registration without dropping down to registry internals.
- Example code for advanced tools no longer bypasses the main runtime API.

### Issue 2.2 — Define the supported schema subset explicitly

**Problem**
- Docs imply broader schema support than the runtime actually validates or constrains.
- Validation currently handles only required fields and primitive type checks.
- Grammar generation currently assumes a flat required-argument object model.

**Tasks**
- Decide between:
  - implementing a broader JSON Schema subset, or
  - explicitly documenting and enforcing a smaller supported subset.
- Encode that decision in validation and grammar generation.
- Fail clearly when unsupported schema constructs are registered.

**Acceptance criteria**
- Schema support is explicit, deterministic, and documented.
- Unsupported constructs fail fast instead of silently degrading.
- Docs/examples stop implying capabilities the runtime does not guarantee.

### Issue 2.3 — Introduce explicit tool execution domain types

**Problem**
- `Response::tool_calls` is a vector of generic `Message` objects, which obscures what actually happened.
- The field name suggests tool-call records, but the agent currently stores tool result messages.

**Tasks**
- Introduce explicit types such as `ToolInvocation`, `ToolResult`, or `ToolEvent`.
- Record tool name, id, arguments, execution outcome, and serialized result separately from chat history messages.
- Rename response fields to match the new semantics.

**Acceptance criteria**
- Response data makes tool execution understandable without reverse-engineering message history.
- Field names match actual contents.
- Tool observability improves in both examples and tests.

### Issue 2.4 — Make tool/schema ordering deterministic

**Problem**
- Tool ordering currently comes from an `unordered_map`, which can make grammar/schema output nondeterministic.

**Tasks**
- Sort tool names or maintain registration order explicitly.
- Ensure emitted schemas and generated grammar are deterministic.
- Update tests to lock in deterministic behavior.

**Acceptance criteria**
- Repeated runs yield stable grammar/schema output.
- Tests do not rely on hash-map iteration behavior.

---

## Epic 3 — Agent Runtime Simplification

**Goal:** reduce cross-cutting complexity in the async orchestration layer.

**Why third:** once API shape is cleaned up, the internal worker model can be simplified without exposing churn to users.

### Issue 3.1 — Make model access inference-thread-owned

**Problem**
- `Agent::process_request()` holds `model_mutex_` across a large section of work, including generation and tool execution.
- This couples unrelated operations and increases deadlock/latency risk.

**Tasks**
- Refactor so the model is owned by the inference thread for request processing.
- Convert cross-thread operations into commands or snapshots instead of coarse locking.
- Revisit how `set_system_prompt`, `clear_history`, and `get_history` interact with the worker thread.

**Acceptance criteria**
- The model is not protected by one large request-scoped mutex during generation.
- Cross-thread operations are explicit and easier to reason about.
- Cancellation and shutdown paths remain correct.

### Issue 3.2 — Extract request queue and cancellation logic into focused units

**Problem**
- Queueing, cancellation tokens, worker loop control, and request processing are packed tightly into `Agent`.

**Tasks**
- Split queue/cancellation concerns from request execution concerns.
- Introduce smaller internal units with clear responsibilities.
- Move worker lifecycle, command handling, and tool-loop orchestration into private runtime implementation files.
- Isolate the concrete `core::Model` adapter in its own private implementation file instead of keeping it inline with the public Agent facade.
- Keep `src/agent.cpp` focused on the public `Agent` facade, construction, and thin forwarding only.
- Keep the public API unchanged or simpler.

**Acceptance criteria**
- `Agent` has a smaller, more obvious responsibility set.
- `src/agent.cpp` no longer contains the full worker loop and concrete backend adapter implementation.
- Runtime orchestration and llama backend integration live in private implementation files with narrower responsibilities.
- Runtime shutdown and cancellation behavior are easier to test directly.

### Issue 3.3 — Add unit-test seams for agent behavior

**Problem**
- Important agent behavior is mostly covered through integration tests rather than isolated unit tests.

**Tasks**
- Introduce a mockable or fake generation backend seam.
- Add focused tests for queue full handling, cancellation, retry exhaustion, and tool-loop limits.
- Prefer testing through the internal runtime/backend seam; keep any test-only `Agent` construction hooks minimal and transitional.
- Keep live integration tests as smoke coverage, not as the only proof of behavior.

**Acceptance criteria**
- Agent orchestration behavior can be validated without a live model.
- The most failure-prone runtime behaviors have targeted tests.
- The primary fast tests do not depend on the live `core::Model` implementation.

---

## Epic 4 — Core Model File Decomposition

**Goal:** split `Model` implementation into smaller, concept-focused units.

**Why fourth:** this is high value, but safer after the runtime and API direction are settled.

### Issue 4.1 — Split `model.cpp` by responsibility

**Problem**
- `src/core/model.cpp` currently mixes backend setup, tokenization, sampling, inference, prompt formatting, grammar handling, and history management.

**Tasks**
- Split implementation into logical files, for example:
  - `model_init.cpp`
  - `model_inference.cpp`
  - `model_prompt.cpp`
  - `model_history.cpp`
  - `model_sampling.cpp`
- Keep the public `Model` API stable during the split.

**Acceptance criteria**
- No single core implementation file acts as a dumping ground.
- Related logic is easier to navigate and review.

### Issue 4.2 — Introduce RAII wrappers for llama resources

**Problem**
- Raw resource ownership is managed manually today.

**Tasks**
- Wrap sampler/context/model lifetimes in focused RAII helpers.
- Reduce duplicated cleanup logic and failure-path complexity.

**Acceptance criteria**
- Resource ownership becomes simpler and safer.
- Constructor/destructor logic reads linearly.

### Issue 4.3 — Isolate prompt delta/KV-cache bookkeeping

**Problem**
- Incremental prompt state, formatting, and KV-cache invalidation are subtle and currently spread across several methods.

**Tasks**
- Centralize incremental prompt bookkeeping.
- Make cache invalidation rules explicit.
- Add targeted tests where practical.

**Acceptance criteria**
- Prompt-formatting rules are easier to inspect and modify.
- KV-cache reset behavior is localized and documented.

---

## Epic 5 — Config and Example Ergonomics

**Goal:** remove avoidable boilerplate and make examples look like a polished library, not a prototype.

### Issue 5.1 — Add JSON serialization helpers for config types

**Problem**
- The demo manually maps JSON into `Config`, duplicating the public configuration contract.

**Tasks**
- Add `to_json/from_json` for `Config` and `SamplingParams`.
- Reuse those helpers in examples and tests.
- Decide whether path expansion belongs in the example or in library utilities.

**Acceptance criteria**
- Example config loading is concise.
- There is one authoritative JSON mapping for config types.

### Issue 5.2 — Simplify and sharpen examples

**Problem**
- The examples are useful, but some of them currently compensate for missing library ergonomics.

**Tasks**
- Rewrite examples to demonstrate the intended end-state API.
- Keep each example focused on one concept.
- Ensure advanced examples do not teach internal workarounds.

**Acceptance criteria**
- Examples become shorter and more idiomatic.
- The demo code reflects the intended library design.

---

## Epic 6 — Packaging, Install, and Documentation Alignment

**Goal:** make the project feel consistent and conventional from clone to install.

### Issue 6.1 — Export real CMake package targets

**Problem**
- Package config recreates imported targets manually instead of exporting the actual build targets.

**Tasks**
- Use proper CMake export/install target flows.
- Keep build-tree and install-tree consumption consistent.
- Reduce custom package config logic where possible.

**Acceptance criteria**
- Installed package usage is conventional and easy to maintain.
- Package configuration is smaller and less fragile.

### Issue 6.2 — Align docs with actual behavior

**Problem**
- Some docs currently overstate or misstate runtime behavior.

**Tasks**
- Correct tool-call id generation docs.
- Correct tool history/response docs.
- Align architecture text with the final target structure.
- Narrow any claims about schema support to what is actually implemented.

**Acceptance criteria**
- Docs accurately reflect the codebase.
- Contributors can trust the architecture and build guides.

### Issue 6.3 — Add a maintainer-facing architecture note

**Problem**
- Future contributors will need a concise explanation of private runtime boundaries after the refactor.

**Tasks**
- Add a short maintainer architecture document describing internal ownership and module boundaries.
- Distinguish public API from internal implementation modules.

**Acceptance criteria**
- Future structural refactors have a clear reference point.
- Internal module boundaries stay intentional.

---

## Recommended Execution Order

### Phase A — Shape the surface
1. Epic 1 — Public API and Build Surface Simplification
2. Epic 6.1 — Export real CMake package targets
3. Epic 6.2 — Align docs with actual behavior for any surface changes

### Phase B — Unify the tool system
4. Epic 2 — Tooling API Unification and Contract Hardening
5. Add tests for deterministic tool ordering and explicit tool event/result types

Detailed execution notes: [phase-abc-catchup-plan.md](phase-abc-catchup-plan.md)

### Phase C — Simplify the async runtime
6. Epic 3 — Agent Runtime Simplification
7. Expand unit-test coverage for agent orchestration behavior

Detailed execution notes: [phase-abc-catchup-plan.md](phase-abc-catchup-plan.md)

### Phase D — Decompose the model implementation
8. Epic 4 — Core Model File Decomposition

### Phase E — Polish and consistency
9. Epic 5 — Config and Example Ergonomics
10. Epic 6.3 — Maintainer architecture note

## Decision Gates

Before implementation begins, we should align on these decisions:

1. **Primary target name**
   - Keep `zoo_core` as the public target, or rename/collapse to `zoo`?
   - Recommendation: expose one compiled public target named `zoo`.

2. **Schema support strategy**
   - Implement a broader supported subset, or enforce/document a smaller one?
   - Recommendation: enforce a clearly documented subset first, then grow deliberately.

3. **Tool response model**
   - Do we want a minimal `ToolInvocation` record, or a richer event stream?
   - Recommendation: start with a small explicit `ToolInvocation` type.

4. **Concurrency model**
   - Should mutating agent operations become worker-thread commands?
   - Recommendation: yes, if we are serious about simplifying the runtime model.

## Definition of Done for the Cleanup Program

This cleanup effort is successful when:

- the public API surface is smaller and clearer,
- the build/install/package story is conventional,
- major runtime implementation lives in `.cpp` files instead of public headers,
- tool registration is unified and explicit,
- response types describe actual domain behavior clearly,
- docs match real runtime guarantees,
- the refactored architecture is easier for a new contributor to navigate.

## Suggested First Implementation Slice

If we want the highest leverage first move, start with:

1. Epic 1.1 — collapse the public target story,
2. Epic 1.2 — move `Agent` implementation out of the public header,
3. Epic 1.3 — stop installing internals,
4. Epic 6.2 — update docs to match the new surface.

That sequence improves clarity immediately without forcing the deeper tool/runtime changes yet.
