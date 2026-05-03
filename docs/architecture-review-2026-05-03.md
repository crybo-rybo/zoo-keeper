# Zoo-Keeper Architectural Review

Date: 2026-05-03

Scope: whole repository, with two lenses:

- Internal/private implementation architecture across Core, Tools, Agent, and Hub.
- External public API ergonomics for library consumers.

This review uses the current code, docs, and examples as evidence. It also uses the
C++ Core Guidelines and cppreference as a baseline for C++23 resource management,
interface design, and concurrency guidance.

## Executive Verdict

Zoo-Keeper has a sound macro-architecture. The four-layer shape is clear, llama.cpp
is mostly contained behind Core and Hub, `zoo::Agent` is a facade over a private
runtime, and the private `AgentBackend` adapter seam is a good use of the Adapter
pattern for testability.

The main architectural risk is not missing patterns. It is that a few central
implementation units have accumulated too many policies:

- `AgentRuntime::process_request()` owns request history scoping, generation
  passes, tool-call detection, validation retry policy, tool execution, trace
  assembly, callback dispatch, metrics, and final response assembly.
- `core::Model` owns model/session resources, prompt delta rendering, history,
  grammar mode, sampler construction, tool-call parsing, schema extraction
  grammar, token accounting, and response assembly.
- Public headers expose some implementation details that the Pimpl design is
  already trying to hide.

The recommended direction is incremental: preserve the four layers and public
facade, then extract narrow private strategies/builders around request processing,
grammar/sampler policy, resource ownership, and schema/tool definition creation.

## Reference Baseline

- C++ Core Guidelines recommend precise, strongly typed interfaces, Pimpl for
  stable library boundaries, RAII for resource ownership, and short functions that
  perform one logical operation:
  <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines?lang=en>
- The same guidelines recommend treating joining threads as scoped containers and
  preferring a joining-thread abstraction over raw `std::thread` ownership.
- `std::jthread` is available since C++20 and automatically requests stop and joins
  in its destructor:
  <https://en.cppreference.com/cpp/thread/jthread>
- `std::expected` is a C++23 vocabulary type for representing either a value or an
  error:
  <https://en.cppreference.com/cpp/utility/expected>

## What Is Already Sound

- **Layering is coherent.** Public docs and source layout agree on Core, Tools,
  Agent, and optional Hub responsibilities.
- **The public `Agent` facade is thin.** `include/zoo/agent.hpp` forwards to
  `src/agent/agent_facade.cpp` and private runtime state.
- **The private backend seam is appropriate.** `src/agent/backend.hpp` lets agent
  orchestration tests use fake model behavior without loading a GGUF model.
- **Request slots are a good fit for bounded async work.** The fixed-capacity
  `RequestSlots` model gives stable handles and explicit backpressure.
- **Tool metadata normalization is deterministic.** `ToolRegistry` preserves
  registration order and normalizes manual schemas before runtime use.
- **Error codes are cohesive.** `Expected<T>` plus categorized `ErrorCode` gives a
  simple non-exception public error model.

Preserve these decisions during refactors.

## Internal Findings

### P1: Agent request processing is a god workflow

Current shape: one request-processing method drives the whole agentic loop.

Evidence:

- `src/agent/runtime_inference.cpp:78` starts `AgentRuntime::process_request()`.
  The function runs through history scoping, streaming callback wrapping,
  generation, tool-call parsing, validation retries, tool invocation, trace
  assembly, history mutation, metrics, and response construction.
- The extraction path in `src/agent/runtime_extraction.cpp:18` duplicates several
  concerns: history scope, callback metrics, grammar activation/restoration, and
  response assembly.

Impact:

- Adding another request mode, tool policy, timeout policy, or tracing mode will
  require editing a high-risk function.
- Test coverage is forced through broad request scenarios rather than narrow
  unit tests for retry behavior, trace construction, or generation metrics.
- The code has hidden temporal coupling: history restore, callback drain,
  grammar restore, and response finalization must happen in the correct order.

Recommended refactor direction:

- Introduce a private `RequestHistoryScope` RAII helper used by both text and
  extraction processing. It should own Append-vs-Replace behavior, restore
  stateless history, and enforce retained-history trimming.
- Extract a `GenerationPass` helper that returns text, prompt tokens, completion
  tokens, first-token timestamp, and elapsed generation time.
- Extract a `ToolLoop` or `ToolLoopController` class that owns validation retry
  counts, tool execution, and `ToolTrace` assembly. This is a good fit for a
  Strategy/State-style object because retry policy and execution policy are the
  likely variation points.
- Keep `AgentRuntime` as the Active Object: queueing, command dispatch, and worker
  lifecycle stay there.

Incremental path:

1. Extract `RequestHistoryScope` first, with no behavior change.
2. Extract `GenerationMetrics` and `GenerationPass`.
3. Move tool retry/trace logic into `ToolLoopController`.
4. Leave public API untouched.

### P1: Core grammar and sampler state should become an explicit policy object

Current shape: grammar mode, tool state, schema grammar, and sampler rebuilding are
all mutable fields under `Model::Impl::Session`.

Evidence:

- `src/core/model_impl.hpp:68` stores `tool_grammar_str`.
- `src/core/model_impl.hpp:69` stores `grammar_mode`.
- `src/core/model_impl.hpp:70` stores `tool_state`.
- `src/core/model_sampling.cpp:47` activates schema grammar by writing
  `tool_grammar_str`.
- `src/agent/runtime_extraction.cpp:54` temporarily overrides grammar state and
  restores tool calling through a `ScopeExit`.
- `src/core/model_prompt.cpp:99` refreshes tool parser and grammar state during
  prompt rendering.

Impact:

- A new constrained-generation mode will add more branches to `Model` rather than
  a new policy.
- Extraction and tool calling are mutually coordinated by mutable session fields,
  making it easy to forget a restoration path.
- Sampler construction and grammar state are coupled to prompt rendering, which is
  a surprising dependency.

Recommended refactor direction:

- Introduce a private `SamplerPolicy` or `GrammarModeStrategy` with variants for:
  `Plain`, `NativeToolCall`, and `Schema`.
- Make `Model::generate_from_history()` ask the active policy to build or refresh
  the sampler for the current pass.
- Introduce a `ScopedGrammarOverride` RAII type for extraction so the temporary
  schema mode has an explicit owner.
- Keep the policy private to Core. Do not add a public backend abstraction.

Design pattern fit:

- Strategy fits because the axis of variation is "how sampler/grammar behavior is
  configured for a generation pass."
- RAII fits the temporary extraction override.

### P1: llama.cpp resources need stronger RAII wrappers

Current shape: long-lived llama resources use `unique_ptr` deleters, but short-lived
resources still use manual free calls.

Evidence:

- `src/core/model.cpp:21` defines deleters for model/context/sampler/templates.
- `src/core/model_inference.cpp:57` creates a `llama_batch` and frees it manually
  at `src/core/model_inference.cpp:68`.
- `src/core/model_inference.cpp:158` creates another `llama_batch` with manual
  cleanup at several error exits and at `src/core/model_inference.cpp:206`.
- `src/hub/inspector.cpp:137` allocates a `gguf_context` and frees it at
  `src/hub/inspector.cpp:163`.
- `src/hub/inspector.cpp:174` loads a vocab-only model and frees it at
  `src/hub/inspector.cpp:190`.

Impact:

- The code is currently careful, but future early returns or exceptions from
  allocations/string operations can leak short-lived C resources.
- Manual cleanup makes the inference loop harder to review.

Recommended refactor direction:

- Add private RAII wrappers:
  - `LlamaBatchHandle`
  - `GgufContextHandle`
  - `VocabOnlyModelHandle`
- Keep them in private source headers under `src/core/` and `src/hub/`.
- Use `unique_ptr`-style deleters or small move-only classes. This is not a new
  abstraction for business logic; it is resource ownership.

Design pattern fit:

- RAII is the primary pattern. Adapter is secondary: each wrapper adapts a C
  acquire/free pair into C++ scope ownership.

### P1: Agent tool state has a data-race-shaped edge

Current shape: most model-affecting operations are routed through the runtime
command lane, but `tool_count()` reads `ToolRegistry` directly.

Evidence:

- `src/agent/runtime_commands.cpp:149` returns `tool_registry_.size()` directly.
- `src/agent/runtime_commands.cpp:170` and `src/agent/runtime_commands.cpp:179`
  modify `tool_registry_` on the inference thread.
- `src/tools/registry.cpp:305` stores tools in a `std::vector` and
  `std::unordered_map` with no internal synchronization.
- `docs/tools.md:100` says batch registration uses a single lock acquisition, but
  the implementation has no registry mutex.

Impact:

- Concurrent `tool_count()` and `register_tool()` can race.
- The public docs overstate synchronization in the low-level registry.

Recommended refactor direction:

- For `Agent::tool_count()`, either route the read through the command lane or
  maintain an atomic count updated on the inference thread.
- For `ToolRegistry`, choose and document one contract:
  - single-threaded registry with external serialization, or
  - internally synchronized registry with a mutex/shared mutex.
- Given the agent already serializes registration through the inference thread,
  prefer external serialization for `ToolRegistry` and fix docs to avoid implying
  internal locking.

### P2: Thread ownership can use C++20/C++23 joining-thread idioms

Current shape: three private worker objects own `std::thread` plus `shutdown_`
flags and condition variables.

Evidence:

- `src/agent/runtime.hpp:124` stores `std::thread inference_thread_`.
- `src/agent/callback_dispatcher.hpp:141` stores `std::thread thread_`.
- `src/agent/tool_executor.hpp:119` stores `std::thread thread_`.
- Each destructor/stop path manually sets shutdown state, notifies, and joins.

Impact:

- The current implementation is understandable, but lifecycle code is repeated.
- Cooperative cancellation is represented by local flags instead of a standard
  stop token.
- Tool timeout/cancellation is already noted as a TODO in
  `src/agent/tool_executor.hpp:30`.

Recommended refactor direction:

- Replace private worker thread ownership with `std::jthread` where compiler
  support is available.
- Pass `std::stop_token` into worker loops and compose it with mailbox shutdown.
- Keep public cancellation based on `RequestId` for compatibility, but internally
  bridge request cancellation into a standard stop-token-like source.

Design pattern fit:

- Active Object still describes `AgentRuntime`.
- RAII/joining-thread ownership makes the worker lifetime easier to audit.

### P2: Hub `ModelStore` is doing catalog, resolver, downloader, inspector, and factory work

Current shape: `ModelStore` is a public facade and a private all-in-one
implementation.

Evidence:

- `src/hub/store.cpp:98` stores catalog state.
- `src/hub/store.cpp:106` reads JSON.
- `src/hub/store.cpp:136` writes JSON.
- `src/hub/store.cpp:152` implements model lookup/resolution.
- `src/hub/store.cpp:258` inspects and registers local files.
- `src/hub/store.cpp:357` creates agents.
- `src/hub/store.cpp:367` pulls from HuggingFace and annotates source metadata.

Impact:

- Store persistence, model resolution, download integration, and runtime factory
  behavior will evolve at different rates.
- Save operations write the catalog directly, so a process crash during write can
  corrupt the catalog.

Recommended refactor direction:

- Keep `ModelStore` as the public Facade.
- Split private responsibilities:
  - `CatalogRepository`: load/save with atomic temp-file + rename.
  - `ModelResolver`: alias/name/path/id matching policy.
  - `ModelImporter`: inspect + register local files.
  - `HubPullService`: HuggingFace download + validation + source annotation.
- This is a Repository pattern internally, not a public API expansion.

### P2: Public headers expose more private implementation than the Pimpl design needs

Current shape: public headers avoid including `llama.h`, but still expose llama
forward declarations and private llama-specific member declarations.

Evidence:

- `include/zoo/core/model.hpp:14` forward-declares llama/common types.
- `include/zoo/core/model.hpp:161` defines llama deleter types in the private
  section.
- `include/zoo/core/model.hpp:181` through `include/zoo/core/model.hpp:206`
  declares many private implementation methods, including methods taking
  `llama_sampler*`.
- `Model` already stores only `std::unique_ptr<Impl>` at
  `include/zoo/core/model.hpp:208`.

Impact:

- llama.cpp names are still visible to consumers and documentation tooling.
- Private declarations increase public header churn when internals change.
- The class is using Pimpl, but only partially.

Recommended refactor direction:

- Move llama deleters, llama handle aliases, and private implementation methods
  into `src/core/model_impl.hpp` or source-local helpers.
- Leave the public `Model` class with public methods, special members, `struct
  Impl`, and `std::unique_ptr<Impl>`.
- This strengthens the existing Pimpl boundary and improves installed-header
  stability.

## External API Findings

### P1: Per-call `GenerationOptions{}` means both "inherit defaults" and "explicit defaults"

Current shape: a default-constructed `GenerationOptions` is treated as "use the
model or agent default generation options."

Evidence:

- `include/zoo/core/types.hpp:607` defines `GenerationOptions::is_default()`.
- `src/agent/runtime.cpp:123` uses `is_default()` to decide whether to replace
  request options with `default_generation_options_`.
- `src/core/model_inference.cpp:412` does the same for direct `Model` calls.

Impact:

- If an agent was created with non-default generation settings, a caller cannot
  intentionally request the literal built-in defaults for one call.
- A struct equality check is being used as a control signal, which is fragile as
  options grow.

Recommended API direction:

- Add an additive request options type:
  - `GenerationOverride` with `std::optional` fields, or
  - `std::optional<GenerationOptions>` at call sites where `nullopt` means inherit.
- Keep existing overloads for compatibility, but document that
  `GenerationOptions{}` inherits defaults.
- Long term, make inheritance explicit with names like `UseDefaultGeneration`.

Design pattern fit:

- Builder/Parameter Object. The important part is explicit optionality, not a
  heavier abstraction.

### P1: Several public methods silently discard runtime errors

Current shape: some public control methods return `void` or a default value even
though the runtime command lane can fail.

Evidence:

- `include/zoo/agent.hpp:169` declares `void set_system_prompt(...)`.
- `include/zoo/agent.hpp:183` declares `Expected<void> add_system_message(...)`,
  which is better.
- `include/zoo/agent.hpp:211` declares `HistorySnapshot get_history() const`.
- `src/agent/runtime_commands.cpp:60` discards the result of
  `set_system_prompt_impl`.
- `src/agent/runtime_commands.cpp:92` returns an empty `HistorySnapshot` if the
  command fails.

Impact:

- Consumers can believe a command succeeded after the agent has stopped.
- Empty history can mean either "no history" or "failed to query history."
- The API weakens the otherwise good `Expected<T>` story.

Recommended API direction:

- Make `Expected<void> set_system_prompt(...)` the primary API.
- Make `Expected<HistorySnapshot> get_history()` the primary API.
- Retain current convenience methods only as explicitly documented best-effort
  helpers, or deprecate them before a breaking release.

### P1: `RequestHandle` should own a cancellation path

Current shape: cancellation requires passing `handle.id()` back into `Agent`.

Evidence:

- `include/zoo/agent.hpp:164` exposes `void cancel(RequestId id)`.
- `RequestHandle` exposes `id()`, `ready()`, and `await_result()` but no
  cancellation method.
- Docs and README describe cancellation as `agent->cancel(handle.id())`.

Impact:

- The caller must keep both the `Agent` and handle together.
- Cancellation is easy to lose when handles are passed across application layers.
- Handle-based APIs in user code become more verbose than necessary.

Recommended API direction:

- Add `RequestHandle<Result>::cancel()` by storing a small private cancellation
  sink in the request state.
- Keep `Agent::cancel(RequestId)` for external correlation and compatibility.
- Consider `RequestHandle::await_result(timeout)` optionally returning a timeout
  without cancelling, as it does today; cancellation should stay explicit.

Design pattern fit:

- Command/Handle. The handle is already the public command object for awaiting;
  cancellation belongs there as an operation on the same asynchronous request.

### P2: Async streaming callbacks cannot request stop

Current shape: synchronous Core streaming callbacks can return `TokenAction`, but
Agent async callbacks return `void`.

Evidence:

- `include/zoo/core/types.hpp:508` defines `TokenCallback` as returning
  `TokenAction`.
- `include/zoo/core/types.hpp:518` defines `AsyncTextCallback` as
  `std::function<void(std::string_view)>`.
- `src/agent/runtime_inference.cpp:137` dispatches async callbacks and always
  returns `TokenAction::Continue`.

Impact:

- Core and Agent have different streaming control semantics.
- A UI callback cannot stop generation directly when it has enough output; it must
  coordinate cancellation separately.

Recommended API direction:

- Add a new callback alias such as
  `AsyncTokenCallback = std::function<TokenAction(std::string_view)>`.
- Keep the current `void` callback overload as an adapter returning Continue.
- Internally propagate Stop to request cancellation or direct generation stop.

### P2: `tools::detail` is being used as public API

Current shape: helper APIs under `zoo::tools::detail` are exposed by public
headers and recommended in docs for batch registration.

Evidence:

- `include/zoo/tools/registry.hpp:26` declares public `namespace detail`.
- `docs/tools.md:87` shows users calling
  `zoo::tools::detail::make_tool_definition(...)`.
- `include/zoo/agent.hpp:231` and `src/agent/runtime.cpp:76` also depend on
  detail helpers.

Impact:

- Consumers can reasonably depend on names that the library labels as internal.
- It becomes harder to refactor schema normalization or callable traits.

Recommended API direction:

- Add a public `ToolDefinition::create(...)` or `ToolDefinitionBuilder`.
- Add a public `ToolSchema::normalize(...)` or `ToolSchema` value object if schema
  normalization remains useful outside the registry.
- Move `detail` declarations that are not needed by templates out of installed
  headers.

Design pattern fit:

- Builder fits tool definition construction because it has metadata, schema,
  handler, and validation as distinct steps.

### P2: The extraction API and README do not agree

Current shape: most docs and examples use `extract(schema, message)`, but the
README advertises `extract(prompt, schema)`.

Evidence:

- `include/zoo/agent.hpp:141` defines `extract(output_schema, message, ...)`.
- `docs/extract.md:26` uses `agent->extract(schema, "...")`.
- `README.md:217` uses `agent->extract("Extract info: ...", schema)`.
- `README.md:86` describes `agent->extract(prompt, schema)`.

Impact:

- New users following the README will hit a compile error.
- The inconsistency obscures the API's intended argument order.

Recommended API direction:

- Fix README immediately.
- Consider adding an overload with message-first order only if user testing shows
  that message-first is more natural. Avoid supporting both orders permanently if
  it creates overload ambiguity with JSON/string conversions.

### P2: `ToolRegistry` public thread-safety contract should be explicit

Current shape: README says thread-safe tool registration, docs imply locks, but
the registry has no internal synchronization.

Evidence:

- `README.md:83` says "Thread-safe Agent owns inference thread" generally, and the
  feature table presents ToolRegistry as a runtime capability.
- `docs/tools.md:100` says `ToolRegistry::register_tools` inserts under a single
  lock acquisition.
- `src/tools/registry.cpp:305` mutates vectors/maps without locks.

Impact:

- Low-level users may share `ToolRegistry` across threads unsafely.
- The agent path is safer than direct registry use, but the docs do not make that
  distinction sharply enough.

Recommended API direction:

- Document `ToolRegistry` as single-threaded unless externally synchronized.
- If direct multi-threaded registry use is a goal, add a private mutex and state
  clear guarantees for concurrent invoke/register/read.

### P3: Public `Model` could be a cleaner synchronous low-level API

Current shape: `core::Model` exposes low-level generation and history mutation,
including tool-calling grammar methods that normal consumers probably should not
call directly.

Evidence:

- `include/zoo/core/model.hpp:125` exposes `set_tool_calling(...)`.
- `include/zoo/core/model.hpp:130` exposes `set_schema_grammar(...)`.
- `include/zoo/core/model.hpp:147` exposes `parse_tool_response(...)`.

Impact:

- Consumers can put the model into grammar modes that the Agent expects to own.
- The direct Core API is powerful, but the division between stable user workflow
  and integration hooks is blurry.

Recommended API direction:

- Group advanced methods under docs clearly labeled "advanced/integration."
- Longer term, consider an `AdvancedModelControl` or free functions in an
  explicitly advanced header if the library wants to keep `Model` small.

## Recommended Change Sequence

1. **No public API break:** add `RequestHistoryScope`, `GenerationPass`, and
   `ToolLoopController` privately in Agent.
2. **No public API break:** add RAII wrappers for `llama_batch`, `gguf_context`,
   and vocab-only llama model inspection.
3. **No public API break:** fix docs around `ToolRegistry` thread-safety and README
   `extract()` argument order.
4. **Small internal API change:** replace direct `AgentRuntime::tool_count()` read
   with a command-lane read or atomic count.
5. **No public API break:** move public-header private llama implementation
   declarations behind the existing `Model::Impl`.
6. **Additive public API:** add handle-owned cancellation and stoppable async
   callbacks.
7. **Additive public API:** add explicit generation override semantics.
8. **Additive public API:** add public `ToolDefinitionBuilder` or
   `ToolDefinition::create(...)`, then stop documenting `tools::detail`.
9. **Optional Hub cleanup:** split `ModelStore` internals into catalog repository,
   resolver, importer, and pull service with atomic catalog writes.

## Pattern Recommendations Summary

| Area | Recommended pattern | Why |
|------|---------------------|-----|
| `Agent` public type | Facade, already present | Keep user-facing API small while runtime internals evolve |
| `AgentBackend` | Adapter, already present | Test orchestration without real llama.cpp model |
| Runtime worker | Active Object, already present | Serialize model access on one inference thread |
| Request processing | Strategy/State | Separate text, extraction, and future request modes |
| History scoping | RAII | Restore stateless history and trim state reliably |
| llama/GGUF resources | RAII Adapter | Make C resource ownership exception-safe |
| Sampler/grammar mode | Strategy + Scoped Override | Isolate constrained-generation modes |
| Tool definition creation | Builder | Stop exposing `tools::detail` as user API |
| Hub catalog persistence | Repository | Isolate JSON persistence and atomic save policy |

## Test Gaps To Add During Refactors

- Concurrent `tool_count()` while registering tools through `Agent`.
- `GenerationOptions{}` inheritance vs explicit default behavior.
- `RequestHandle::cancel()` once added.
- `AsyncTokenCallback` returning `TokenAction::Stop` once added.
- RAII wrappers under artificial early-return/failure paths.
- Atomic `ModelStore` catalog save behavior.

## Bottom Line

Do not rewrite the library. The existing boundaries are good enough to evolve.
The highest-value changes are private extractions that reduce the edit surface of
`AgentRuntime::process_request()` and `core::Model` grammar/sampler logic, followed
by a few additive API improvements that make error handling, cancellation, and
generation defaults explicit for consumers.
