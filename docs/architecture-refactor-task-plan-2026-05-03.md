# Architecture Refactor Task Plan

Date: 2026-05-03

Source review: [architecture-review-2026-05-03.md](architecture-review-2026-05-03.md)

This plan selects the highest-impact implementation items from the architecture
review and converts them into task groups with exit criteria. Treat each task
group as a separate PR unless the change is purely documentation. Public API
changes should be additive first and deliberately reviewed before replacing or
deprecating existing methods.

## Priority Order

1. Stabilize Agent request processing.
2. Make Core grammar/sampler state explicit.
3. Add RAII wrappers for short-lived llama/GGUF resources.
4. Fix tool registry concurrency contract and `Agent::tool_count()`.
5. Improve async API ergonomics: errors, handle cancellation, and stoppable
   streaming callbacks.
6. Make generation override semantics explicit.
7. Stop exposing `tools::detail` as documented user API.
8. Complete public-header Pimpl cleanup.
9. Split Hub store internals and make catalog writes atomic.

## Task Group 1: Stabilize Agent Request Processing

Objective: reduce `AgentRuntime::process_request()` from a broad orchestration
blob into narrow private components while preserving public behavior.

Tasks:

- Add a private `RequestHistoryScope` helper for Append-vs-Replace request
  history handling.
- Move stateless history restore and stateful history trimming into that helper.
- Share `RequestHistoryScope` between text requests and extraction requests.
- Add a private `GenerationPass` or `GenerationRunner` helper that owns callback
  wrapping, cancellation check wiring, prompt/completion token counting, and
  first-token/generation-time metrics.
- Add a private `ToolLoopController` for tool-call validation, retry counts,
  tool execution dispatch, retry messages, and `ToolTrace` assembly.
- Keep `AgentRuntime` responsible for mailbox dispatch, command routing,
  request-slot resolution, and worker lifecycle only.
- Add or update unit tests for history restoration, history trimming, tool retry
  exhaustion, trace creation, callback failure, cancellation, and extraction
  behavior after the extraction/text shared helpers are introduced.

Exit criteria:

- `process_request()` no longer directly owns tool validation, retry counting,
  handler dispatch, trace assembly, and response metrics construction.
- Text and extraction paths use the same history-scope mechanism.
- Existing `AgentRuntime` unit tests pass without weakening assertions.
- New unit tests cover the extracted helpers without loading a GGUF model.
- No public API headers change in this task group.

## Task Group 2: Make Core Grammar/Sampler State Explicit

Objective: replace ad hoc grammar-mode fields with an explicit private policy for
plain generation, native tool calling, and schema-constrained extraction.

Tasks:

- Introduce a private `SamplerPolicy`, `GrammarModeStrategy`, or equivalent
  private value that represents `Plain`, `NativeToolCall`, and `Schema` modes.
- Move sampler rebuild decisions out of scattered `if (grammar_mode == ...)`
  branches and into that policy.
- Add a `ScopedGrammarOverride` for extraction so schema grammar activation and
  restoration are scope-owned.
- Keep llama.cpp and llama-common types private to `src/core/`.
- Preserve native tool-call parser refresh behavior during prompt rendering, but
  make the ownership relationship explicit.
- Add tests for schema extraction after tool calling, tool calling after schema
  extraction, empty grammar fallback, and parser/grammar refresh.

Exit criteria:

- `Model::generate_from_history()` delegates sampler/grammar setup to the policy
  instead of branching over raw grammar session fields.
- Schema extraction cannot accidentally leave the model in schema mode after the
  request finishes.
- Native tool calling still works after an extraction request restores prior
  state.
- Existing Core unit tests and integration extraction tests pass.
- No new public backend abstraction is introduced.

## Task Group 3: Add RAII Wrappers For Short-Lived C Resources

Objective: make all C resource acquire/free pairs scope-owned.

Tasks:

- Add a private `LlamaBatchHandle` wrapper for `llama_batch_init()` /
  `llama_batch_free()`.
- Replace manual `llama_batch_free()` calls in the inference prefill and
  autoregressive loops.
- Add a private `GgufContextHandle` wrapper for `gguf_init_from_file()` /
  `gguf_free()`.
- Add a private vocab-only llama model wrapper for Hub inspection.
- Keep wrappers private under `src/core/` and `src/hub/`.
- Add failure-path tests where practical, or targeted unit tests for wrapper move
  behavior if the wrapper is independently testable.

Exit criteria:

- Direct `llama_batch_free()`, `gguf_free()`, and vocab-only
  `llama_model_free()` calls appear only inside the relevant wrappers.
- Existing inference behavior and Hub inspection behavior are unchanged.
- Early returns in modified paths no longer require manual cleanup.
- `scripts/test.sh` passes.

## Task Group 4: Fix Tool Registry Concurrency Contract

Objective: remove the data-race-shaped edge around `Agent::tool_count()` and make
`ToolRegistry`'s threading model truthful in code and docs.

Tasks:

- Decide the public `ToolRegistry` contract:
  - single-threaded unless externally synchronized, or
  - internally synchronized.
- Prefer external synchronization unless a concrete low-level use case requires
  concurrent direct registry access.
- Route `Agent::tool_count()` through the runtime command lane or maintain an
  atomic count updated on the inference thread.
- Update `docs/tools.md`, README wording if needed, and any Doxygen comments that
  imply `ToolRegistry` internal locking.
- Add a unit test for concurrent `Agent::tool_count()` during queued
  registration.
- Keep tool registration order deterministic.

Exit criteria:

- `Agent::tool_count()` cannot concurrently read `ToolRegistry` containers while
  the inference thread mutates them.
- Public docs accurately describe `ToolRegistry` thread-safety.
- Tests cover the selected synchronization behavior.
- No behavior change to tool ordering, replacement, or schema export.

## Task Group 5: Improve Async API Ergonomics

Objective: make the public async surface align with the library's `Expected<T>`
model and reduce cancellation/streaming friction.

Tasks:

- Add primary `Expected<void>` forms for system-prompt updates where current
  void methods can fail internally.
- Add a primary `Expected<HistorySnapshot> get_history()` form, or otherwise make
  the failure mode explicit without returning an ambiguous empty snapshot.
- Keep current convenience methods only as documented best-effort helpers, or
  mark them for later deprecation.
- Add `RequestHandle<Result>::cancel()` by storing a private cancellation sink in
  request state.
- Preserve `Agent::cancel(RequestId)` for external correlation.
- Add an `AsyncTokenCallback` returning `TokenAction`.
- Adapt existing `AsyncTextCallback` overloads to return Continue.
- Propagate async callback `TokenAction::Stop` to the running request without
  requiring a separate `Agent::cancel(handle.id())` call.
- Update examples and docs to show the new primary paths.

Exit criteria:

- Stopped agents and failed command-lane operations are observable through
  `Expected` on the primary API path.
- A caller can cancel a request through its handle without holding an `Agent&`.
- A streaming callback can stop generation directly.
- Existing callback code using `void(std::string_view)` still compiles.
- New tests cover handle cancellation, stoppable async streaming, and command
  failure reporting.

## Task Group 6: Make Generation Override Semantics Explicit

Objective: remove ambiguity where `GenerationOptions{}` means both "inherit
agent/model defaults" and "literal built-in defaults."

Tasks:

- Add an additive `GenerationOverride` parameter object with `std::optional`
  fields, or add overloads that accept `std::optional<GenerationOptions>`.
- Define explicit semantics:
  - no override means inherit configured defaults,
  - explicit options mean use exactly those options.
- Keep current overloads for source compatibility and document their current
  inheritance behavior.
- Add tests proving a caller can intentionally request built-in defaults when
  the agent/model has non-default defaults configured.
- Update docs and examples to prefer the explicit override path.

Exit criteria:

- Per-call inheritance no longer depends only on structural equality with
  `GenerationOptions{}`.
- Existing code keeps compiling with current overloads.
- New docs clearly distinguish inherited defaults from explicit request options.
- Tests cover inherited, explicit non-default, and explicit built-in-default
  cases.

## Task Group 7: Replace Documented `tools::detail` Usage

Objective: stop teaching consumers to depend on internal helper names.

Tasks:

- Add a public `ToolDefinition::create(...)`, `make_tool_definition(...)`, or
  `ToolDefinitionBuilder` API for batch tool definition construction.
- Keep template-heavy callable traits private where possible.
- Add a public schema normalization entry point only if direct normalized schema
  construction is a supported use case.
- Update `docs/tools.md` batch registration examples to use the new public API.
- Keep old `tools::detail` names available temporarily for source compatibility,
  but remove them from docs and public-facing Doxygen.
- Add tests for the new public construction path.

Exit criteria:

- Consumer docs no longer mention `zoo::tools::detail`.
- Batch tool registration remains possible without internal namespace usage.
- Existing `Agent::register_tool(...)` behavior is unchanged.
- Tests cover typed and JSON-schema tool definition construction through the new
  public API.

## Task Group 8: Complete Public-Header Pimpl Cleanup

Objective: make `include/zoo/core/model.hpp` a true Pimpl public boundary.

Tasks:

- Move llama forward declarations, deleter structs, handle aliases, and private
  llama-specific methods out of the public header where possible.
- Keep only public `Model` methods, special members, `struct Impl`, and
  `std::unique_ptr<Impl>` in the installed header.
- Move private implementation declarations into `src/core/model_impl.hpp` or
  source-local helpers.
- Confirm installed docs do not surface llama implementation members.
- Run packaging consumer tests after the header cleanup.

Exit criteria:

- Public headers do not expose llama.cpp type names unless they are intentionally
  part of the public contract.
- Core implementation still owns all llama resources privately.
- FetchContent and installed CMake consumer tests still build.
- No llama.cpp headers are included from installed public headers.

## Task Group 9: Split Hub Store Internals

Objective: keep `ModelStore` as a public facade while separating persistence,
lookup, import, pull, and factory concerns.

Tasks:

- Add private `CatalogRepository` for JSON load/save.
- Implement atomic catalog save using temp-file write plus rename.
- Add private `ModelResolver` for alias/name/path/id matching.
- Add private `ModelImporter` for local file inspection and catalog entry
  creation.
- Add private `HubPullService` for HuggingFace download, validation, and source
  annotation.
- Keep `ModelStore` public methods source-compatible.
- Add tests for atomic-save behavior, resolver precedence, duplicate aliases, and
  pull annotation persistence.

Exit criteria:

- `ModelStore` remains the public facade, but persistence, resolving, importing,
  and pulling are private units with narrow responsibilities.
- Catalog writes are crash-resistant at the file-operation level.
- Existing Hub tests pass with strengthened resolver and persistence coverage.
- No Core or Agent dependency direction changes.

## Comprehensive Changeset Exit Criteria

The architecture refactor series is complete only when all selected task groups
meet their individual exit criteria and the following whole-program criteria are
true.

Architecture:

- The four-layer boundary remains intact: Hub may depend on Core, Agent may
  depend on Core and Tools, Tools remains llama-free, and Core remains the
  llama.cpp runtime boundary.
- `AgentRuntime` owns worker lifecycle and dispatch, but request-mode logic,
  history scoping, generation-pass metrics, and tool-loop policy live in narrow
  private helpers.
- Core grammar/sampler mode is represented by explicit private policy instead of
  scattered mutable session flags.
- Public installed headers do not expose private llama.cpp implementation details.
- `ToolRegistry` has one documented and tested thread-safety contract.
- `ModelStore` remains a facade over private Hub collaborators.

Behavior:

- Public behavior remains source-compatible except for explicitly approved,
  documented deprecations or breaking changes.
- Existing examples still build and run.
- Tool calling, structured extraction, request cancellation, retained history,
  stateless completion, streaming callbacks, Hub inspection, and model-store
  catalog operations remain observable through tests or examples.
- Any newly added public API has docs and at least one example or focused test.

Verification:

- `scripts/test.sh` passes.
- `scripts/format.sh` produces no diff.
- `scripts/lint.sh` is warning-free.
- Packaging consumer tests pass for FetchContent and installed CMake usage.
- Integration tests in `.secret/integration-testing.md` pass on the default local
  model before declaring the series finished.
- Sanitizer builds pass for the touched concurrency/resource-management areas
  before merging the corresponding PRs.

Performance and operability:

- Token throughput and latency are not materially worse on the default
  integration model; any regression greater than 5% is explained and accepted.
- No new unbounded queues are introduced on hot paths.
- Tool execution and callback dispatch continue to run off the inference thread.
- Shutdown remains deterministic: no worker thread is left running after
  `Agent::stop()` or destruction.

Documentation and migration:

- README, `docs/getting-started.md`, `docs/tools.md`, `docs/extract.md`,
  `docs/architecture.md`, and `docs/maintainer-architecture.md` match the final
  public and private shapes.
- `MIGRATION.md` documents any deprecations or behavioral clarifications.
- `CHANGELOG.md` summarizes user-visible additions and migration notes.

Changeset discipline:

- Each PR stays under the repository's non-test SLOC limit unless explicitly split
  or approved.
- Each PR has one concern: no mixed feature/refactor/doc cleanup bundles.
- Public API header changes are reviewed deliberately before implementation.
- No new dependency, CMake structure change, or llama.cpp pin update is included
  unless separately approved.
