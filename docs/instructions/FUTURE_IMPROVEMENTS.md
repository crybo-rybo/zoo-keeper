# Zoo-Keeper — Future Improvement Punch List

This document tracks refactoring opportunities surfaced during the architectural
audit that produced PR `refactor/sync-command-and-handle`. That PR shipped item
#1 below (collapse the sync-command boilerplate + replace the `void*`
`RequestHandle`). The remaining items are sized so each could land as its own
small PR, in roughly the order written.

The bar throughout: **subtractive PRs preferred**. Every item below should
remove more code than it adds, or trade equivalent SLOC for materially better
type-safety / performance / clarity.

---

## #2 — Split `runtime.cpp` (now ~520 lines, was 640)

**File:** `src/agent/runtime.cpp`

`.claude/rules/agent-layer.md` says decompose if `runtime.cpp` exceeds ~500
lines. After PR #1 it is ~520 — close enough that the next addition will trip
the rule. Pre-emptively split along responsibility boundaries:

- `runtime_inference.cpp` — `inference_loop`, `handle_request`,
  `process_request` (the hot path).
- `runtime_commands.cpp` — `handle_command`, `resolve_command_on_shutdown`, the
  six `*_impl` helpers and their public-facing overloads.
- `runtime_lifecycle.cpp` — ctor, dtor, `stop`, `is_running`, `fail_pending`.

Mechanical split — no logic changes. Update `cmake/ZooKeeperTargets.cmake`
accordingly.

## #3 — Stop double-copying conversation history on stateless requests

**Files:** `src/agent/runtime.cpp` (`materialize_conversation`),
`include/zoo/core/types.hpp` (`ConversationView`).

Today every `complete()` and stateless `extract()` call materializes the
`ConversationView` into an owned `std::vector<Message>` on the calling thread,
then the payload (containing that vector) is **copied again** into the queued
request slot. For a long conversation this is two O(N) allocations + copies on
the request path.

Two options:

- **Cheap:** keep materialization, but `std::move` the snapshot into the queued
  payload. Eliminates one copy. ~5-line change.
- **Cleaner:** thread `ConversationView` deeper into the runtime so the
  materialized owned-vector only exists once, on the inference thread, when
  prompt rendering actually needs it. Larger change but eliminates the
  call-thread allocation entirely.

Start with the cheap version; revisit the cleaner one if profiling justifies.

## #4 — Replace `Storage`-enum view types with `std::variant`

**File:** `include/zoo/core/types.hpp` (`ToolCallSpan` ~lines 136-162,
`ConversationView` ~lines 301-327).

Both classes use a hand-rolled `enum class Storage { Borrowed, Owned }` plus a
union to discriminate between span-of-views and span-of-owned. `operator[]`
manually branches on the enum. Replace with:

```cpp
std::variant<std::span<const Borrowed>, std::span<const Owned>>
```

…and `std::visit` in `operator[]`. Eliminates the manual branch, makes the
discriminator type-safe, and removes the `Storage` enum entirely. Net SLOC:
slightly negative. Behavior: identical.

## #5 — Unify `register_tool` / `extract` overload zoo

**File:** `include/zoo/agent.hpp` lines ~281-345 (and the matching `extract`
overloads).

Today `register_tool` has 4 templated overloads — `(initializer_list, Func)`,
`(initializer_list, Func, timeout)`, `(span, Func)`, `(span, Func, timeout)` —
plus 2 non-templated overloads for the prebuilt-schema variant. `extract` has a
similar fan-out.

Collapse to one signature per logical operation:

```cpp
template <typename Func>
Expected<void> register_tool(std::string_view name, std::string_view description,
                             std::span<const std::string> param_names, Func func,
                             std::optional<std::chrono::nanoseconds> timeout = {});
```

`std::span<const std::string>` already accepts braced-init lists, so the
`initializer_list` overload becomes redundant. Drops ~60 lines of templates.
Mirror the change in `extract`.

## #6 — Freeze `ToolRegistry` after init; drop the `shared_mutex`

**File:** `include/zoo/tools/registry.hpp`.

`ToolRegistry` currently uses a reader-writer lock. But:

- Tool registration goes through `RegisterTool[s]Cmd` which executes on the
  inference thread, not concurrently with reads.
- After registration, the registry is read-only.

The lock therefore exists for a contention pattern that does not happen.
Replace with a build-then-freeze model: the runtime hands the registry over to
`refresh_tool_calling_state()` once at registration time; reads from the
inference thread need no lock. Removes lock acquisition from every tool
invocation.

## #7 — Move hub `ErrorCode` values into `include/zoo/hub/`

**File:** `include/zoo/core/types.hpp` lines ~357-402.

`ErrorCode` defines values 700-799 for the hub layer (`GgufMetadataNotFound`,
`InvalidModelIdentifier`, `StoreCorrupted`, …). These pollute the core enum
even when `ZOO_BUILD_HUB=OFF`. Two paths:

- Move these enumerators into a hub-only header so they only exist when the hub
  is compiled.
- Or document them as reserved-for-hub and leave them.

Either is acceptable; the first is cleaner.

## #8 — Decompose `model_inference.cpp` (415 lines)

**File:** `src/core/model_inference.cpp`.

Single `run_inference()` mixes prompt tokenization, prefill, decode, stop-
sequence matching, tool stream filtering, and callback dispatch. Extract:

- `class StopSequenceMatcher` — own the matching state machine.
- `struct InferencePhase { prefill, decode, finalize }` — one method each.
- Move the word-trigger filter into `StreamFilter` where it belongs.

Each piece becomes 50-100 SLOC and unit-testable without a real GGUF model.

---

## Polish (medium impact, low risk)

- **`Agent::estimated_tokens()`** — `Model` already exposes
  `estimated_tokens()` internally; thread it through the runtime so callers
  can query context usage.
- **`max_tool_iterations` per-request** — promote from `agent_config_` to
  `GenerationOptions` so individual calls can override the agent-wide cap.
- **`ExtractionResponse::raw_output`** — add an optional `std::string`
  field; on partial extraction the user currently has no way to see what the
  model actually produced.
- **Configurable callback dispatch** — `CallbackDispatcher` always hops to a
  separate thread for token callbacks. For low-latency consumers, expose an
  opt-in same-thread mode.
- **Hub integration test** — `tests/integration/` has no end-to-end test
  exercising HuggingFace download → GGUF inspect → model load. Add a network-
  gated test that catches breakage when `llama.cpp`'s `common` download
  internals shift.

---

## Out of scope

- **Abstract `LlamaBackend` interface.** Tempting (mockable Model, swap
  engines) but speculative. The CLAUDE.md `Changeset Discipline` section is
  explicit: do not introduce abstractions for hypothetical future use. Skip
  unless a concrete second backend appears.
- **CMake consolidation.** The current 16-file `cmake/` setup is annoying but
  stable. Touch only when a real change forces it.
