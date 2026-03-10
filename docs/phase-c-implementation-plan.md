# Phase C Implementation Plan

## Evaluation

Phase C is still materially outstanding in the current tree.

- `src/agent.cpp` still contains the full runtime stack: queueing, cancellation bookkeeping, worker lifecycle, tool-loop execution, and model coordination.
- `src/agent.cpp:261` holds `model_mutex` across the entire request lifecycle, including inference and tool execution.
- Cross-thread model operations still happen directly from caller threads in `src/agent.cpp:134`, `src/agent.cpp:153`, `src/agent.cpp:158`, and `src/agent.cpp:481`.
- There is no isolated agent-runtime test coverage today; the only `Agent` tests are live integration checks in `tests/integration/test_model_agent.cpp`.

Two additional findings should be treated as part of Phase C:

- `src/agent.cpp:175` reads `model->has_tool_grammar()` without synchronization, so prompt-mode selection currently depends on unsynchronized model state.
- `src/agent.cpp:29` stores `submitted_at` but never uses it.

## Phase Goals

- Keep the public `zoo::Agent` API stable.
- Move all `core::Model` access onto the inference thread.
- Replace coarse model locking with explicit runtime commands and snapshots.
- Split queueing/cancellation concerns into focused internal units.
- Add unit-test coverage for agent orchestration without introducing a public backend abstraction.

## Recommended Design

### 1. Extract Internal Runtime Units

Create a small internal runtime layer and keep the public `Agent` class as a façade.

Recommended internal units:

- `AgentRuntime`: owns the worker thread and request-processing loop.
- `RuntimeMailbox`: accepts bounded chat requests and unbounded control commands.
- `RequestTracker`: owns request IDs, cancellation flags, and promise cleanup.
- `AgentBackend`: internal interface for the model operations the runtime actually needs.

Suggested file layout:

- `include/zoo/internal/agent/runtime.hpp`
- `include/zoo/internal/agent/mailbox.hpp`
- `include/zoo/internal/agent/request_tracker.hpp`
- `include/zoo/internal/agent/backend.hpp`
- `src/agent.cpp` as the public façade only
- `src/agent/runtime.cpp`
- `src/agent/backend_model.cpp`

The seam should stay internal to the Agent layer. `core::Model` remains the real llama.cpp wrapper and should not be turned into a generic public backend abstraction.

### 2. Make the Model Worker-Owned

Remove `model_mutex` entirely and route model-affecting operations through worker-thread commands.

Recommended control commands:

- `SetSystemPrompt`
- `ClearHistory`
- `GetHistory`
- `RefreshToolGrammar`
- `Stop`

Behavioral rules:

- `chat()` enqueues a request and returns immediately, as it does today.
- `set_system_prompt()`, `clear_history()`, and `get_history()` remain synchronous at the public API boundary, but internally enqueue a command and wait on a promise.
- Control commands should be processed between requests, not in the middle of an active generation pass. That preserves the current practical behavior where these calls block until the in-flight request is done.
- The mailbox should prioritize pending control commands before starting the next queued request.

This gives the runtime one clear rule: if something mutates or snapshots the model, the inference thread does it.

### 3. Publish Prompt/Grammar State Without Touching `Model`

`build_tool_system_prompt()` should stop reading model state directly.

Recommended approach:

- Publish a small runtime snapshot such as `tool_grammar_active_` after each grammar refresh attempt.
- `register_tool()` should still complete only after the corresponding grammar-refresh command finishes, so callers keep the existing “tool is ready when the call returns” behavior.
- `build_tool_system_prompt()` should use the tool registry plus the published prompt-mode snapshot, never the model itself.

This fixes the current race while keeping the API unchanged.

### 4. Extract Queue and Cancellation Responsibilities

Move request admission and cancellation bookkeeping out of `Agent::Impl`.

Recommended responsibilities:

- `RuntimeMailbox`
  - bounded request lane governed by `Config::request_queue_capacity`
  - unbounded control-command lane
  - shutdown/close semantics
- `RequestTracker`
  - assign request IDs
  - create and store cancellation flags
  - remove completed requests in one place
  - fail queued requests during shutdown

Key simplifications:

- remove `Request::submitted_at`
- keep `cancel()` as an atomic flag flip only
- centralize cleanup so request tokens are erased in one completion path instead of being spread across the worker loop

## Backend Test Seam

The unit-test seam should model only what the Agent runtime needs.

Recommended internal backend surface:

- `Expected<void> add_message(const Message&)`
- `Expected<core::Model::GenerationResult> generate_from_history(std::optional<TokenCallback>, CancellationCallback)`
- `void finalize_response()`
- `void set_system_prompt(const std::string&)`
- `std::vector<Message> get_history() const`
- `void clear_history()`
- `bool set_tool_grammar(const std::string&)`
- `void clear_tool_grammar()`

Production code uses a thin adapter around `core::Model`. Unit tests use a fake backend that returns scripted generations, records history mutations, and can block until cancellation is triggered.

## Test Plan

Add a new unit suite, for example `tests/unit/test_agent_runtime.cpp`, plus any focused mailbox/tracker tests if the extracted units justify separate files.

Minimum coverage for Phase C:

- queue full handling with `request_queue_capacity = 1`
- cancel before processing begins
- cancel during generation
- stop drains queued requests with `AgentNotRunning`
- tool validation retry exhaustion
- tool loop limit exhaustion
- system prompt, history clear, and history snapshot commands serialize correctly with queued work
- prompt builder chooses grammar or JSON instructions from published runtime state, not direct model access

The existing integration coverage in `tests/integration/test_model_agent.cpp` should remain as smoke coverage only.

## Recommended Implementation Order

### Slice 1: Structural extraction with no behavior change

- Move `RequestQueue` logic into `RuntimeMailbox`.
- Move request ID and cancellation-token management into `RequestTracker`.
- Keep the old execution flow initially so the refactor is mechanical and easy to review.

### Slice 2: Worker-owned model operations

- Introduce control commands and remove `model_mutex`.
- Convert `set_system_prompt()`, `get_history()`, `clear_history()`, and grammar refresh to worker commands.
- Publish prompt-mode state explicitly.

### Slice 3: Internal backend seam

- Introduce `AgentBackend` and a concrete `ModelAgentBackend`.
- Update the runtime to depend on the backend interface instead of `core::Model` directly.
- Keep this seam internal to the library and test targets.

### Slice 4: Unit tests

- Add fake-backend runtime tests.
- Add targeted shutdown and cancellation tests around the extracted mailbox/tracker units.

### Slice 5: Documentation cleanup

- Update `docs/architecture.md` to describe the new worker-owned model rule and runtime units.
- Update the blocking-behavior comment in `include/zoo/agent.hpp:128` so it no longer references `model_mutex`.

## Acceptance Criteria for Completion

Phase C should be considered done when all of the following are true:

- no request-scoped `model_mutex` remains in the agent runtime
- all model mutations and history snapshots are worker-thread-owned
- queueing/cancellation logic lives in dedicated internal units
- agent orchestration has fast unit coverage without a live model
- architecture and API comments describe the new threading model accurately

## Risks and Guardrails

- Keep the public `Agent` surface unchanged unless a change is clearly necessary. Phase C should be an internal runtime cleanup, not a user-facing redesign.
- Do not create a repo-wide “backend abstraction” for the model layer. The test seam belongs inside Agent only.
- Preserve request ordering semantics: control commands may jump ahead of queued requests, but they should not mutate model state in the middle of an active request.
- Treat shutdown carefully. Once control commands exist, shutdown must resolve both queued request promises and any pending synchronous control waits.
