# Zoo-Keeper 2.0 — Architectural Cleanup Design Document

## Context

Zoo-Keeper is a C++23 SDK on llama.cpp (~17K LOC, 92 files, four layers: core / tools / agent / hub). A comprehensive audit across all four layers surfaced two real concurrency hazards, one feature-correctness gap, multiple half-finished migrations, speculative abstractions with no second user, and supply-chain hygiene issues in the build system.

The codebase is fundamentally sound: layer boundaries are respected, RAII is correct, `std::expected` is used consistently, and the threading model is documented. But the project sits at the seam between "experimental" and "mature," and accumulated drift is the dominant tax on new contributors.

This document consolidates audit findings into a 2.0 cleanup plan. It explicitly waives the `< 150 SLOC per PR` Changeset Discipline rule for the cross-cutting redesigns where piecewise patches would produce worse architecture; smaller mechanical fixes still ship independently.

Goal: a smaller, more coherent codebase with no transitional ballast, no use-after-free hazards, no silent feature gaps, and a public API that fits in a one-page summary.

---

## Audit Findings Summary

### HIGH severity
1. **`stop()` deadlocks on a long-running tool handler.** `~AgentRuntime` joins the inference thread, which is blocked on `ToolExecutor::submit(...).get()`. User code can hang the destructor indefinitely. (`src/agent/tool_executor.hpp:30,58`, `src/agent/runtime_inference.cpp:149`)
2. **Streaming-callback use-after-free.** `CallbackDispatcher::dispatch_async` enqueues entries holding raw `AsyncTokenCallback*` into the request slot. On the error-rethrow path the slot can be cleared while the dispatcher thread still holds the pointer. (`src/agent/callback_dispatcher.hpp:111-124`)
3. **Only the first tool call per assistant turn is executed.** `runtime_inference.cpp:108-122` invokes `structured_tool_calls.front()` only; remaining calls are stored in history without execution, breaking multi-call models (Llama-3.1 JSON, Hermes, Qwen3 XML) across turns.
4. **GBNF grammar accepts ill-formed JSON.** Raw control chars allowed inside strings; integers allow leading zeros (`src/tools/grammar.hpp:163-167`). Model output can be grammar-valid but reject from `nlohmann::json::parse`.
5. **`JsonSchema integer` rejects `3.0`-shaped values** (`src/tools/registry.cpp:58-59`). Many LLMs serialize integers as floats; silently fails validation.
6. **Build supply chain: llama.cpp archive fetched without `URL_HASH`** (`cmake/ZooKeeperDependencies.cmake:45-49`). GitHub-generated tarballs are not byte-stable.
7. **Sanitizer/coverage flags leak through `INTERFACE_LINK_OPTIONS`** to FetchContent consumers (`cmake/Sanitizers.cmake:9-11`, `cmake/Coverage.cmake:7-9`, `cmake/ZooKeeperPackage.cmake:9`).
8. **`find_dependency(llama CONFIG)` is unversioned** (`cmake/ZooKeeperConfig.cmake.in:17`) — installed package silently satisfied by mismatched llama versions.
9. **Path traversal via HuggingFace filename component** (`src/hub/huggingface.cpp:91-154`) — `owner/repo::../../etc/passwd` is accepted.

### MED severity (selected)
- "Transitional" `Message`/`ToolCallInfo` aliases are the entrenched canon (~40 sites).
- `ToolCallView`/`ToolCallSpan` borrowed-branch has no production producer.
- `AgentBackend::replace_history`, `AgentRuntime::extract(json, string_view)`, `trim_history_to_fit`, `PromptState::dirty`, `AsyncTextCallback` — dead code.
- `GenerationResult`/`ParsedResponse` duplicated between `core::Model` and `AgentBackend`.
- 3-overload pattern for `set_system_prompt`/`get_history`/`clear_history`/`add_system_message` on the public Agent.
- `ScopeExit`: `std::function`-backed, allocates per request; move-assign fires old callback before adopting new one (non-idiomatic).
- VRAM headroom math ignores `context_size`; auto-configured GPU offload can OOM at runtime.
- `GgufInspector`/`SystemProbe` live in `zoo::core` but docs attribute them to Hub.
- `core/json.hpp` throws `std::invalid_argument` for parse errors (violates the project's "no exceptions" rule).
- Catalog `"version"` field written but never validated on load; no advisory locking.
- `cmake_minimum_required(3.18)` is below the 3.20+ features actually used.
- Tests redundantly link `llama-common` directly, reaching across the layer boundary.
- README/CHANGELOG/AGENTS.md drift: layer attribution, `LLAMA_BUILD_TOOLS=OFF` claim, 1-arg `Agent::create` example.

### LOW severity
mach port leak, `Error::context` nominally optional but always populated, `FunctionRef` arg-forwarding nuance, examples installed to `BINDIR`, `FetchDependencies.cmake` dead forwarder, etc.

---

## Cross-Cutting Redesigns

These are the changes where SLOC discipline produces worse architecture. They are designed as cohesive multi-file changes for 2.0.

### R1. Unified Cancellation & Lifecycle Protocol

**Problem.** Five separate cancellation surfaces today:
- `AgentRuntime::running_` (atomic, joined-on-shutdown)
- `Slot::cancelled` (per-request atomic)
- `RuntimeMailbox::shutdown_`
- `CallbackDispatcher::shutdown_`
- `ToolExecutor::shutdown_`

`ToolExecutor::submit().get()` blocks unconditionally — the inference thread cannot observe agent shutdown while a user handler runs. This is the root of finding #1.

**Design.** Introduce a single shared `CancellationToken` carried by reference from `AgentRuntime` into every blocking call site:

```cpp
// src/agent/cancellation.hpp (NEW)
namespace zoo::internal::agent {
class CancellationToken {
  public:
    [[nodiscard]] bool stopped() const noexcept {
        return stopped_.load(std::memory_order_acquire);
    }
    void request_stop() noexcept {
        {
            std::lock_guard lock(mutex_);
            stopped_.store(true, std::memory_order_release);
        }
        cv_.notify_all();
    }
    // Wait up to `timeout` for stop OR `pred` to be true.
    template <typename Pred>
    bool wait_for(std::chrono::nanoseconds timeout, Pred pred) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout,
                            [&] { return stopped_.load() || pred(); });
    }
  private:
    std::atomic<bool> stopped_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

// Composed token observed by ToolExecutor / CallbackDispatcher: triggers when
// EITHER the agent is shutting down OR the per-request slot was cancelled.
class CompositeCancellation {
  public:
    CompositeCancellation(const CancellationToken& agent_stop,
                          const std::atomic<bool>& request_cancelled) noexcept
        : agent_stop_(&agent_stop), request_cancelled_(&request_cancelled) {}
    [[nodiscard]] bool cancelled() const noexcept {
        return agent_stop_->stopped() ||
               request_cancelled_->load(std::memory_order_acquire);
    }
  private:
    const CancellationToken* agent_stop_;
    const std::atomic<bool>* request_cancelled_;
};
}
```

**Wiring.**
- `AgentRuntime` owns one `CancellationToken stop_token_` (replaces `running_`).
- `ToolExecutor::submit(handler, args, cancellation)` returns `std::future<Expected<json>>`; the worker thread checks `cancellation.cancelled()` between jobs. Critically, the **caller** of `.get()` waits via `stop_token_.wait_for(...)` in a loop, returning `RequestCancelled` if shutdown observed.
- `CallbackDispatcher` accepts a token at construction and aborts queued entries on stop.
- `RuntimeMailbox` is collapsed into the same protocol: `pop()` returns `nullopt` when `stop_token.stopped()`.

**Resulting `stop()` semantics.**
```cpp
void AgentRuntime::stop() {
    stop_token_.request_stop();              // wakes ToolExecutor, dispatcher,
                                             // mailbox, and any in-flight .get()
    request_mailbox_.notify_shutdown();      // wakes a possibly-idle pop()
    if (inference_thread_.joinable())
        inference_thread_.join();
}
```

A user tool handler that genuinely runs forever still won't terminate (we can't kill C++ code), but the inference thread no longer blocks on it: `.get()` is replaced with a loop that observes `stop_token_` every N ms and returns `RequestCancelled` to drop the request without waiting on the handler. The detached handler is best-effort — document that long-running handlers should themselves check cancellation by accepting a `const CancellationToken&` parameter via a typed `tools::CancellableToolHandler`.

**File changes.**
- New: `src/agent/cancellation.hpp` (~70 SLOC)
- Modified: `src/agent/runtime.hpp/cpp` (replace `running_` with `stop_token_`), `src/agent/tool_executor.hpp` (add token + interruptible loop), `src/agent/callback_dispatcher.hpp` (token-aware shutdown), `src/agent/mailbox.hpp` (use token instead of own `shutdown_`)
- Public addition: `tools::ToolHandler` gains an optional `(json, const CancellationToken&)` overload via concept dispatch in `tools::detail::is_json_handler_like` — old `(json) -> Expected<json>` continues to work.

**Resolves:** Finding #1 (deadlock), Finding #2 (UAF: dispatcher drains-or-aborts on token-stop before any slot clear). Removes three redundant `shutdown_` flags.

---

### R2. Tool-Loop Architectural Redesign

**Problem.** `ToolLoopController::run` (`src/agent/runtime_inference.cpp:32-79`) handles one tool call per iteration, hardcoded to `structured_tool_calls.front()`. Multiple correctness and design issues:
- Multi-call models (Llama-3.1, Hermes, Qwen3) silently lose tool calls.
- The "please respond to the user" nudge at line 64 is a heuristic working around the absence of a clean tool-result sequencing protocol.
- `tool_invoked_`/`retry_count_for(name)` bookkeeping is per-controller but conceptually per-loop.
- `handle_validation_failure` and `handle_tool_call` duplicate history-mutation logic.
- The dispatcher-UAF fix from R1 needs to land in this same area.

**Design.** Replace the controller with a state-machine-flavored runner that processes one *turn* (one model generation) per iteration and executes *all* tool calls in that turn before the next generation:

```cpp
// src/agent/tool_loop.hpp (NEW; replaces inline ToolLoopController)
class ToolLoop {
  public:
    Expected<TextResponse> run(const ActiveRequest& request,
                               std::chrono::steady_clock::time_point start_time);

  private:
    struct Turn {
        std::string assistant_content;
        std::vector<OwnedToolCall> tool_calls;  // all of them, in order
        GenerationStats::PassRecord stats;
    };

    Expected<Turn>                generate_turn(const ActiveRequest&, GenerationStats&);
    Expected<void>                execute_all_tools(const Turn&,
                                                    const ActiveRequest&,
                                                    int iteration,
                                                    bool record_trace);
    Expected<void>                execute_one_tool(const OwnedToolCall&, ...);
    Expected<TextResponse>        finalize(std::string&& content,
                                           const GenerationStats&, bool record_trace);

    AgentBackend&                    backend_;
    const tools::ToolRegistry&       registry_;
    ToolExecutor&                    executor_;
    CallbackDispatcher&              dispatcher_;
    const CancellationToken&         stop_token_;
    const AgentConfig&               config_;
    bool                             native_tool_calling_;

    std::unordered_map<std::string, int>  retry_counts_;     // by tool name
    std::vector<ToolInvocation>           invocations_;
    bool                                  any_tool_invoked_ = false;
};
```

**Per-turn flow.**
```
for iteration in [1..max_tool_iterations]:
    if cancelled: return RequestCancelled
    turn = generate_turn()              # one model pass
    if turn.tool_calls.empty():
        if turn.content.empty() and any_tool_invoked_:
            # nudge once, then finalize even if empty (don't loop forever)
            backend_.add_message(Role::User, "Please respond ...")
            continue
        return finalize(turn.content, ...)
    backend_.add_message(Role::Assistant, turn.content, turn.tool_calls)
    backend_.finalize_response()
    execute_all_tools(turn, ...)        # appends one Tool message per call
    dispatcher_.drain()                 # exactly once per turn, post-tool-results
```

**`execute_all_tools` logic.**
- For each tool call in the turn, in order:
  - Validate args against schema; on failure, append a `Tool` error message and bump `retry_counts_[name]`. If retries exhausted return `ToolRetriesExhausted`. Continue to next tool call (still in the same turn — a malformed call doesn't kill sibling calls).
  - On valid args: submit to `ToolExecutor` with composite cancellation. Append the result as a `Tool` message. Record an invocation in the trace if requested.
- If any tool in the turn was invoked, set `any_tool_invoked_ = true`.

**Why this shape.**
- One generation → all tool calls executed → one drain. Removes per-call drain from the hot path.
- Validation-failure recovery is per-call, not per-turn: the model can fix one bad call while sibling calls succeed.
- The "Please respond" heuristic moves to the no-tool-calls branch only — it never fires while tool calls are still pending.
- Cancellation is checked at the boundary of every blocking operation (generation, tool submit, drain).

**`Model::GenerationResult` and `AgentBackend::GenerationResult` are merged** (`src/agent/backend.hpp:24-33` is deleted; the backend interface uses `core::Model::GenerationResult` directly). Same for `ParsedResponse`. The deliberate decoupling produced two structurally identical types and a field-by-field adapter — negative value.

**Public effect.** The `chat()` / `complete()` return type does not change; the existing `ToolTrace` mechanism captures all invocations correctly because `invocations_` is now a per-turn-aware vector.

**Resolves:** Finding #3 (multi-call), bundles cleanly with Finding #1/#2 (cancellation lands here), removes duplicate types (`AgentBackend::GenerationResult`, `AgentBackend::ParsedToolResponse`), removes single-use `handle_validation_failure`.

---

### R3. Type System Rectification

**Problem.** Several speculative abstractions never grew the second user, and one "transitional" migration entrenched itself as canon.

**Decisions for 2.0.**

| Decision | Rationale |
|---|---|
| `OwnedMessage` is the canonical name; `using Message = OwnedMessage` is **removed**. | The alias was always either canonical or transitional, never both. Pick canonical. ~40 sites get renamed. |
| `OwnedToolCall` is canonical; `using ToolCallInfo = OwnedToolCall` is **removed**. | Same. ~6 sites. |
| `ToolCallView` is **removed**; `MessageView::tool_calls()` returns `std::span<const OwnedToolCall>`. | `ToolCallView` exists only in tests. The variant `ToolCallSpan` falls out. ~80 SLOC. |
| `AsyncTextCallback = AsyncTokenCallback` alias is **removed**. | Zero callers. |
| `ToolCallWordTriggerFilter`'s owned-vector constructor is **removed**; only the borrowed-span ctor survives. | Production always uses borrowed. |
| `ScopeExit<F>` becomes a template `ScopeGuard<F>` (in `runtime_helpers.hpp`); no `std::function` allocation; move-assign uses `release()` semantics. | Cleaner, allocation-free, idiomatic. |
| `core::Model::ParsedResponse` and the agent's `AgentBackend::ParsedToolResponse` are unified — the backend interface uses `core::Model::ParsedResponse` directly. Same for `GenerationResult`. | The decoupling never paid for itself. |
| `Error::context` retains `std::optional` (callers may legitimately pass `nullopt`). | Light churn for marginal benefit. |
| `FunctionRef` is documented in-place: not for move-only arg types; arg-forwarding caveat noted. | Cosmetic. |

**`types.hpp` size after this pass: ~700 lines down from 879**, with all retained types load-bearing.

**Mechanical migration steps** (the kind a single PR handles cleanly when SLOC is waived):
1. Rename `Message` → `OwnedMessage` and `ToolCallInfo` → `OwnedToolCall` across all 40 sites.
2. Replace `MessageView::tool_calls()` return type; delete `ToolCallSpan` and `ToolCallView`.
3. Delete `AsyncTextCallback` and the owned `ToolCallWordTriggerFilter` ctor.
4. Move `Model::ParsedResponse` and `Model::GenerationResult` to a small `core/generation_types.hpp` so both layers reference one definition.
5. Drop `AgentBackend::GenerationResult` and `AgentBackend::ParsedToolResponse`.

**Resolves:** all of §3a-3e/§3g findings, plus removes the ScopeExit allocation finding (§4c).

---

### R4. Public API Surface Simplification

**Problem.** `include/zoo/agent.hpp` exposes a 3-overload pattern for several operations:

```cpp
// Today (excerpt):
void                          set_system_prompt(string_view);
Expected<void>                try_set_system_prompt(string_view);
Expected<void>                set_system_prompt(string_view, nanoseconds timeout);

HistorySnapshot               get_history() const;
Expected<HistorySnapshot>     try_get_history() const;
Expected<HistorySnapshot>     get_history(nanoseconds timeout) const;

void                          clear_history();
Expected<void>                try_clear_history();
Expected<void>                clear_history(nanoseconds timeout);

Expected<void>                add_system_message(string_view);
Expected<void>                add_system_message(string_view, nanoseconds timeout);
```

12 methods for 4 operations.

**Design for 2.0.**

```cpp
// Operation policy — a small typed value, not three overloads.
struct CommandTimeout {
    std::optional<std::chrono::nanoseconds> value;
    static CommandTimeout none()                          { return {}; }
    static CommandTimeout of(std::chrono::nanoseconds t)  { return {t}; }
};

class Agent {
public:
    Expected<void>             set_system_prompt(std::string_view, CommandTimeout = {});
    Expected<void>             add_system_message(std::string_view, CommandTimeout = {});
    Expected<HistorySnapshot>  get_history(CommandTimeout = {}) const;
    Expected<void>             clear_history(CommandTimeout = {});
    // ... same shape for register_tool / register_tools
};
```

- **12 methods → 4.** Default `CommandTimeout{}` means "wait indefinitely" (was previously `try_*` semantics).
- **No more silent best-effort.** Removing the void-returning overloads forces callers to look at the Expected; today many callers call the void overload and lose the error.
- `CommandTimeout` is a typed wrapper rather than `std::optional<duration>` because it's more self-documenting at call sites: `set_system_prompt(p, CommandTimeout::of(500ms))` reads better than `set_system_prompt(p, 500ms)` (which collides with a hypothetical future variadic).

**Facade cleanup, bundled.**
- `Agent::Impl` (single-member wrapper) is **removed**; `Agent` stores `std::unique_ptr<internal::agent::AgentRuntime>` directly.
- `Agent` no longer caches `model_config_`/`agent_config_`/`default_generation_options_`; the runtime owns the single copy and `Agent::model_config()` forwards to it. Public accessors keep their shape.
- The unused `AgentRuntime::extract(json, string_view, ...)` overload is removed (test-only).

**Migration.** Provide a one-page MIGRATION.md section showing the rename map. The audit explicitly recommended this as a single bundled change.

**Resolves:** §4a (API ergonomics), §4b (double-Impl indirection), §3a/§3d residue.

---

### R5. Layer Boundary Correction (Hub vs Core)

**Problem.** `GgufInspector`, `SystemProbe`, and the `auto_configure` JSON resolver live in `zoo::core` and ship unconditionally. README/CHANGELOG attribute them to the Hub layer. The boundaries documented in CLAUDE.md ("Layer 1 = direct llama.cpp wrapper") are violated by the presence of `auto_configure` (which inspects files and probes hardware) and by `core/json.hpp` transitively pulling `gguf_inspector.hpp` into every consumer.

**Design for 2.0.** Move the inspection/probe/auto-configure surface into a new mandatory-but-narrow `zoo::config` layer that sits between core and the optional Hub:

```
Layer 1  zoo::core      llama.cpp wrapper. NO file inspection, NO hardware probe.
Layer 2  zoo::tools     unchanged.
Layer 2b zoo::config    GgufInspector, SystemProbe, auto_configure. Always built.
Layer 3  zoo::Agent     unchanged.
Layer 4  zoo::hub       optional. HF download + ModelStore. Depends on zoo::config.
```

**File moves.**
- `include/zoo/core/{gguf_inspector,system_probe,model_info}.hpp` → `include/zoo/config/`
- `include/zoo/core/json.hpp` → split: `include/zoo/core/json.hpp` (sampling/model/agent/generation JSON only) + `include/zoo/config/auto_configure.hpp` (auto-configure resolver).
- `src/core/{gguf_inspector,system_probe}.cpp` → `src/config/`
- Public umbrella `zoo.hpp` adds `#include "config/auto_configure.hpp"` unconditionally; Hub-gated includes only cover hub.

**`zoo::core` after the move** contains exactly: `Model`, `types.hpp` value types, `auto_configure: true` JSON dispatch (calls into `zoo::config`).

**Resolves:** finding §A4 (Hub vs core mismatch), finding §6d (compile-time bloat), and the unstated layering inconsistency in CLAUDE.md.

---

### R6. Auto-Configure Feedback Loop

**Problem.** `GgufInspector::auto_configure(info, sys)` computes:
1. `context_size` from training context vs RAM
2. `n_gpu_layers` from layer count vs VRAM

These two are independent today, but they shouldn't be: KV-cache scales linearly with `context_size` AND with `n_gpu_layers`. A model configured for `context_size=32768` with the VRAM heuristic's chosen `n_gpu_layers` can OOM at first generation.

**Design for 2.0.** Two-pass resolver in `zoo::config::auto_configure`:

```cpp
struct VramBudget {
    int64_t free_vram_bytes = 0;
    int64_t bytes_per_layer = 0;
    int64_t per_token_kv_bytes = 0;   // derived from ModelInfo (n_kv_heads,
                                       //   head_dim, dtype, n_layers)
};

ModelConfig auto_configure(const ModelInfo& info, const SystemProbe::Snapshot& sys) {
    // Pass 1: tentative context from training-ctx vs available RAM.
    int tentative_ctx = pick_context(info, sys);

    // Pass 2: VRAM budget accounts for KV cache at the resolved context.
    VramBudget budget = compute_vram_budget(info, sys);
    int n_gpu_layers = pick_gpu_layers(info, budget, tentative_ctx);

    // Pass 3: if n_gpu_layers was forced lower than expected, the layers
    // staying on CPU need RAM budget too — adjust context if necessary.
    int final_ctx = reconcile_ram_with_offload(tentative_ctx, info, sys,
                                                n_gpu_layers);
    return { .context_size = final_ctx, .n_gpu_layers = n_gpu_layers, ... };
}
```

Surfaces the math in one place, makes the cost of each knob visible, and unit-tests against synthetic `ModelInfo`/`SystemProbe` snapshots without needing real GGUF files.

Bundled with this:
- Fix `read_gguf_u32_as_i32` → `int64_t` for context_length (§6b).
- Release `mach_host_self()` send-right on macOS (§6c).
- Move `core/json.hpp`'s auto-configure helpers into the new `config` layer.

**Resolves:** §6a (VRAM math), §6b/§6c, and the core-vs-config layering question.

---

## Tactical Fixes (Independent, Small)

These ship as separate small PRs. SLOC discipline applies fully here.

### T1. Concurrency hardening (post-R1, separate)
- Hub catalog advisory file locking (`flock(LOCK_EX)`) around `CatalogRepository::save`.
- `fsync(dir_fd)` after catalog rename for durability.
- `RequestSlots::active_request()` documentation: payload pointers valid until inference thread resolves/clears the slot.

### T2. Grammar correctness (`src/tools/grammar.hpp`)
- `string` rule: `[^"\\\x00-\x1F]*` (no raw control chars).
- Integer rule: `("0" | [1-9][0-9]*)`.
- Number rule: same prefix.
- Add `\uXXXX` escape alternative.
- Enum: canonicalize numeric enum literal via explicit `to_string` rather than `dump()`.

### T3. Validation correctness (`src/tools/registry.cpp`)
- `json_matches_type(Integer)` accepts `is_number_integer() || (is_number_float() && std::trunc(v) == v && v in INT range)`.
- Reject duplicate property keys (silent today via nlohmann dedup).
- Document the `additionalProperties:false` runtime enforcement.

### T4. Hub security
- `parse_identifier`: reject `..` and embedded slashes in filename component.
- `download_file`: require destination to be inside an allowlisted root (configurable via `HuggingFaceClient::Config`).
- Add retry budget to `download_model` (3 attempts, exponential backoff on transient errors).
- `validate_download_status`: treat `< 200 || >= 400` as failure (drops the 3xx-is-success window).
- Catalog: read and validate `"version"` field on load; reject if mismatch (until migration code exists).
- Catalog: reject empty `id` during `validate_catalog_entries`.

### T5. Build supply chain & install
- Add `ZOO_LLAMA_ARCHIVE_SHA256` cache var, pass as `URL_HASH` to llama_cpp `FetchContent_Declare`.
- Update `ZOO_LLAMA_ARCHIVE_BASE_URL` to `github.com/ggml-org/llama.cpp`.
- Move sanitizer/coverage `target_link_options` from `PUBLIC` to `PRIVATE`.
- `find_dependency(llama 0.0.${LLAMA_BUILD_NUMBER} CONFIG)` in installed config.
- `cmake_minimum_required(VERSION 3.21)` (matches presets, allows generator-expr sources).
- Drop `tests/CMakeLists.txt:34` redundant `llama-common` link.
- Drop `cmake/FetchDependencies.cmake` (dead forwarder).
- Drop `zoo_core` interface alias (2.0 breaking change).
- Drop `PATTERN "internal" EXCLUDE` (excludes non-existent directory).
- Drop `examples/CMakeLists.txt` install rules for demo binaries.
- Guard `ZOO_ENABLE_CRAP` behind `ZOO_PROJECT_IS_TOP_LEVEL`.
- Add `message(FATAL_ERROR)` when both coverage and sanitizers are enabled.
- Fix `scripts/build.sh`: enable `ZOO_BUILD_TESTS=ON` by default (matches docs claim).

### T6. Documentation & coherence
- Fix README Quick Start to use 3-arg `Agent::create`.
- Strike the false `LLAMA_BUILD_TOOLS=OFF` claim, or actually set it.
- Update layer attribution table after R5 (`config` layer between core and hub).
- Rename `zoo::testing` fixture namespace to `zoo::test_fixtures`; delete the `using namespace` warning from CLAUDE.md.
- Document `FunctionRef` arg-forwarding semantics.

### T7. Header hygiene
- Replace `<nlohmann/json.hpp>` with `<nlohmann/json_fwd.hpp>` where only the type is referenced (saves ~24K lines per public TU).
- Move template body of `Agent::extract<Message>` to a `agent.tcc` or accept fwd-decl + explicit instantiation cost.

### T8. core/json.hpp → Expected
- `load_model_config` returns `Expected<ModelConfig>` for *all* error paths (today it throws `std::invalid_argument` on schema errors and returns `Expected` on auto-configure failures).
- Keep `from_json`/`to_json` ADL hooks throwing — they must, per nlohmann convention — but wrap them in `Expected` at the entry point.

### T9. Dead code cleanup (already covered by R3 but listed for completeness)
- `Model::trim_history_to_fit` (empty body).
- `PromptState::dirty` (write-only).
- `AgentBackend::replace_history` (unused).
- `AgentRuntime::extract(json, string_view)` (test-only).
- `is_tool_trigger_detected`/`extract_word_triggers` (move to test file).

---

## Sequencing & Risk

Recommended landing order. Each block is an independently shippable cluster; later blocks depend on earlier ones.

| # | Block | Risk | Approx churn |
|---|-------|------|--------------|
| 1 | T5 build supply chain (URL_HASH, sanitizer PRIVATE, llama version pin) | LOW | ~50 SLOC |
| 2 | T2 + T3 grammar/validation tightening (correctness, no API change) | LOW | ~80 SLOC |
| 3 | T4 hub security & robustness | LOW | ~150 SLOC |
| 4 | R3 type system rectification (renames + dead removal) | MED | ~500 SLOC; touches 40 files mechanically |
| 5 | R1 unified cancellation protocol | HIGH | ~250 SLOC, deep concurrency |
| 6 | R2 tool-loop redesign (depends on R1, R3) | HIGH | ~400 SLOC; replaces ~600 |
| 7 | R5 layer reorganization (config layer) | MED | ~300 SLOC moves; rebuilds CMakeLists, public umbrella |
| 8 | R6 auto-configure feedback loop (depends on R5) | MED | ~150 SLOC + tests |
| 9 | R4 public API surface simplification | MED-HIGH | ~200 SLOC; 2.0 breaking change |
| 10 | T1, T6, T7, T8, T9 | LOW | trickle |

**Why this order:**
- Supply chain first because it protects every subsequent build.
- Correctness fixes before architectural moves: easy wins that reduce noise.
- Type rectification before tool-loop redesign because the redesign uses the renamed types.
- Cancellation before tool loop because the loop is built around the cancellation token.
- Layer reorg before auto-configure feedback because the feedback loop lives in the new `config` layer.
- Public API last because it's the largest external-visible break and benefits from internals being settled.

**Compatibility.** User-confirmed: 2.0 may break freely. The `MIGRATION.md` for 2.0 should cover:
- Rename: `Message` → `OwnedMessage`, `ToolCallInfo` → `OwnedToolCall`.
- API collapse: `try_*` and void-returning overloads → single Expected-returning method.
- Namespace move: `zoo::core::GgufInspector` → `zoo::config::GgufInspector`.
- Removed: `ToolCallView`, `ToolCallSpan`, `AsyncTextCallback`, `Agent::Impl`, etc.
- Behavior change: multi-tool-per-turn now executes all calls.

---

## Verification

Per cluster, before merging:

### Build + format + tests (all clusters)
```bash
scripts/format.sh
scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON
scripts/test.sh
scripts/lint.sh                    # warnings-as-errors
```

### Sanitizers (R1, R2, R3)
```bash
scripts/build.sh -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON
scripts/test.sh
```

### Coverage / CRAP (R6, T-cluster)
```bash
scripts/build.sh -DZOO_ENABLE_CRAP=ON
scripts/crap.sh                    # functions over threshold = regression
```

### Integration smoke (R1, R2, R5, R6, R4)
```bash
scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON
ZOO_INTEGRATION_MODEL=/path/to/Qwen3-8B-Q4_K_M.gguf scripts/test.sh
examples/demo_chat                 # real model end-to-end
```

### Targeted new tests
- R1: `test_cancellation.cpp` — verify `stop()` interrupts an in-flight tool handler within bounded time.
- R2: `test_tool_loop_multi_call.cpp` — scripted-backend test emitting 2-3 tool calls per turn, asserting all are executed with one drain per turn and history reflects all of them.
- R5: `test_config_layer.cpp` — `zoo::config::auto_configure` against synthetic `ModelInfo`/`SystemProbe` snapshots.
- R6: parameterized test sweeping `context_size` × `total_vram` × `n_layers` and asserting auto-resolved `n_gpu_layers * (bytes_per_layer + per_token_kv_bytes * context_size) <= vram_budget * 0.85`.
- R4: `test_command_timeout.cpp` — verify `CommandTimeout::none()` waits indefinitely and `CommandTimeout::of(0ns)` returns `RequestTimeout` quickly.
- T4: `test_hub_path_traversal.cpp` — assert `parse_identifier("owner/repo::../../etc/passwd")` returns `InvalidModelIdentifier`.

### Downstream FetchContent smoke (T5)
A separate test repo that pulls zoo-keeper via `FetchContent` and verifies:
- Build succeeds without sanitizer flags leaking into the consumer.
- `ZooKeeper::zoo` target resolves with both build-tree and installed configs.
- Catalog round-trip works (write + reopen + read).

---

## Open Questions for Implementation

1. **`CancellableToolHandler` shape.** Should tool handlers receive `const CancellationToken&` directly, or a typed `tools::TaskContext` wrapper that can carry future fields (request id, logger handle, deadline)? Recommendation: `TaskContext` — the wrapper is cheap and forward-compatible.
2. **Multi-call tool-loop budgeting.** Today `max_tool_iterations` counts turns. After R2 executes all calls within a turn, a turn with 10 calls still counts as 1 iteration. Add a separate `max_tool_calls_per_turn` (default 8) to prevent runaway models?
3. **R5 namespace cost.** `zoo::config::GgufInspector` reads cleanly, but is it worth introducing a new namespace just for two types + a resolver? Alternative: keep them in `zoo::core` and accept the "core is not strictly llama-only" reality, then fix the docs instead. Recommendation: do the move — the docs claim has been a recurring source of audit drift.
4. **Public API: do we keep `Agent::create` taking three args, or introduce a builder?** A builder reads better at call sites for partial configuration; three-arg `create` is fine and matches today's shape. Recommendation: stay with three-arg `create`; introduce a builder only if the constructor grows a fourth parameter.

These can be settled during implementation; none block the document.
