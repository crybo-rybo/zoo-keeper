# Migration Guide

This document covers what consumers need to know when upgrading Zoo-Keeper.

## v1.1.3 → v1.1.4

### Async Request Controls

`RequestHandle<Result>` now has `cancel()`, which requests cancellation without
requiring the caller to keep an `Agent&`:

```cpp
auto handle = agent->chat("Write a long answer.");
handle.cancel();
```

`Agent::cancel(handle.id())` remains available for callers that correlate
requests externally.

### Streaming Callbacks

The primary async callback type is now `AsyncTokenCallback`, which can return
`TokenAction::Stop` to end generation from inside the callback. Existing
`void(std::string_view)` callback code remains source-compatible and is treated
as `TokenAction::Continue`.

```cpp
auto handle = agent->chat(
    "List three items.",
    zoo::GenerationOverride::inherit_defaults(),
    [](std::string_view token) {
        std::cout << token << std::flush;
        return zoo::TokenAction::Continue;
    });
```

### Explicit Generation Overrides

`GenerationOverride` distinguishes "inherit configured defaults" from "use
these exact request options." Existing overloads that accept `GenerationOptions`
still compile and keep their legacy inheritance behavior when passed
`GenerationOptions{}`.

Use:

```cpp
agent->chat("Use configured defaults.", zoo::GenerationOverride::inherit_defaults());
agent->chat("Use these options exactly.",
            zoo::GenerationOverride::explicit_options(zoo::GenerationOptions{}));
```

### Expected-Based Agent Commands

`try_set_system_prompt()`, `try_get_history()`, and `try_clear_history()` expose
command-lane failures through `Expected<T>`. The existing convenience methods
remain best-effort helpers for source compatibility.

### Tool Definition Construction

Batch tool definitions can be built through the public
`zoo::tools::make_tool_definition(...)` helpers. Code that previously used
`zoo::tools::detail::make_tool_definition(...)` should switch to the public
helper; the old detail name is still present for compatibility.

### ToolRegistry Thread-Safety Clarification

Direct `ToolRegistry` use is single-threaded unless the caller externally
synchronizes overlapping reads and writes. `Agent` owns its runtime registry on
the inference thread and keeps `Agent::tool_count()` synchronized internally.

### Tool Handler Threading

Agent tool handlers now run on a dedicated `ToolExecutor` worker thread. The
tool loop still waits for each handler result before continuing, so request
ordering and tool-loop semantics are unchanged. The practical difference is
that user-supplied tool code no longer executes on the inference thread.

### Hub Store Persistence

`ModelStore` catalog saves now write a temporary catalog file and atomically
rename it over the old catalog. The public API is unchanged, but interrupted
saves are less likely to leave a truncated `catalog.json`.

## v1.1.2 → v1.1.3

### llama.cpp moved to FetchContent

Zoo-Keeper no longer vendors llama.cpp as a git submodule at `extern/llama.cpp`.
CMake's `FetchContent` now downloads llama.cpp at configure time, pinned by
`ZOO_LLAMA_TAG` in `cmake/ZooKeeperOptions.cmake`.

### What changes for contributors

- `git clone` is enough — drop `--recurse-submodules` and any
  `git submodule update --init --recursive` step from your local workflow.
- The first `cmake -B build` (or `scripts/build.sh`) requires network access
  to fetch llama.cpp. Subsequent configures reuse `build/_deps/`.
- To pin a different llama.cpp release tag, edit `ZOO_LLAMA_TAG` (or override
  with `-DZOO_LLAMA_TAG=...`) — see `docs/instructions/UPDATE_LLAMA_CPP.md`.
- Parent projects that already define `llama` and `llama-common` targets are
  unaffected; Zoo-Keeper still reuses them and skips its own fetch.

### What changes for downstream packagers

- The `ZOO_FETCH_LLAMA` CMake option has been removed. FetchContent runs by
  default whenever no parent-provided `llama`/`llama-common` targets exist.
- `scripts/bootstrap.sh` has been removed; it only initialized the submodule.

### llama.cpp b8992 and `llama-common`

Zoo-Keeper now targets llama.cpp release `b8992`, where the upstream `common`
target/archive is named `llama-common`/`libllama-common.a`, with
`libllama-common-base.a` as a required sidecar archive. Parent projects that
provide llama.cpp targets must expose both `llama` and `llama-common`.

Installed Zoo-Keeper packages now link consumers against both common archives.
If you package Zoo-Keeper manually, update any allowlists or archive copy steps
that still mention `libcommon.a`.

### Hub Cache Behavior

HuggingFace repository downloads now use llama.cpp's Hugging Face-style cache
instead of Zoo-Keeper's old owner/repo directory under the store/cache path.
`ModelStore::pull()` still registers the downloaded GGUF in Zoo-Keeper's catalog,
but `ModelEntry::file_path` may now point under a path like:

```text
.../models--owner--repo/snapshots/<commit>/<file>.gguf
```

`CachedModelInfo::size_bytes` is retained for source compatibility but now
reports `0`, because llama.cpp b8992 cache listing entries expose repository and
tag only.

### HuggingFace Identifiers

`HuggingFaceClient::download_model()` accepts the same repo-file form that
`ModelStore::pull()` accepts:

```cpp
hf->download_model("owner/repo::model.Q4_K_M.gguf");
hf->download_model("owner/repo:Q4_K_M");
```

Bearer tokens continue to use `HuggingFaceClient::Config::token`.

## v1.1.1 → v1.1.2

### `add_system_message()` (Additive)

`Agent::add_system_message(message)` appends a system-role message to the active
conversation without replacing the initial system prompt and without flushing the KV
cache. This is distinct from `set_system_prompt()`, which replaces the first message
and forces a full history re-encode.

Use `add_system_message()` to inject context mid-conversation (e.g., tool results,
retrieval snippets, or updated instructions) without disturbing prior turns.

An optional timeout overload is available:

```cpp
// No timeout — blocks until the inference thread processes the command.
agent->add_system_message("Restrict your answer to three sentences.");

// With timeout — returns RequestTimeout if the thread is busy.
auto result = agent->add_system_message("Use metric units.", std::chrono::seconds(2));
```

### `validate_role_sequence()` Relaxed

`validate_role_sequence()` no longer returns an error when a system-role message
appears after the first position. If your code checked for that specific error to
detect sequence violations, remove that branch — system messages anywhere in history
are now permitted.

### CMake: `ZOO_ENABLE_INSTALL` Defaulting

`ZOO_ENABLE_INSTALL` now defaults to `OFF` when zoo-keeper is consumed via
`add_subdirectory` or FetchContent. If you relied on install targets being present in
a subdirectory build, pass `-DZOO_ENABLE_INSTALL=ON` explicitly.

### CMake Module Restructure (Internal)

The build system was refactored into dedicated files under `cmake/`. The public
CMake interface (`ZooKeeper::zoo`, `ZooKeeper::zoo_core`, option names) is unchanged.
`FetchDependencies.cmake` now delegates to `ZooKeeperDependencies.cmake`; both remain
present for backwards compatibility.

## v1.1.0 → v1.1.1

### Hub Layer (Additive)

New optional Layer 4 (`zoo::hub`) adds GGUF inspection, HuggingFace downloading,
and a local model store. Public headers live under `include/zoo/hub/`. The key
types are `GgufInspector`, `HuggingFaceClient`, and `ModelStore`.

This is a purely additive change — non-hub consumers are unaffected. Enable with
`-DZOO_BUILD_HUB=ON` at configure time.

### Batch Tool Registration (Additive)

`ToolRegistry::register_tools()` and `Agent::register_tools()` accept a
collection of tool definitions, registering them in a single call. Existing
per-tool `register_tool()` methods continue to work unchanged.

### Callback Threading Change

Streaming token callbacks now dispatch to a dedicated `CallbackDispatcher` thread
instead of running directly on the inference thread. The guidance to avoid
blocking in callbacks still applies, but the consequence of a slow callback is
now dispatcher queue backup rather than inference thread stall.

Tool handlers are not affected — they still run on the inference thread.

### CMake Flag Independence

`ZOO_BUILD_INTEGRATION_TESTS=ON` now works independently without requiring
`ZOO_BUILD_TESTS=ON`. Previously, integration tests were gated behind the
general test flag.

## v1.0.3 → v1.1.0

Version `1.1.0` is a major API step rather than a small patch. The biggest
changes are the config split, typed request handles, dedicated extraction
results, and the clearer message/history APIs that separate retained state from
request-scoped conversations.

### Config Split

Before, consumers configured the library through one aggregate `zoo::Config`
object. After this release, the configuration concerns are split into
`ModelConfig`, `AgentConfig`, and `GenerationOptions`.

```cpp
// Before: one aggregate config object.
zoo::Config config;
config.model_path = "/models/llama.gguf";
config.context_size = 8192;
config.max_history_messages = 64;
config.max_tokens = 128;

auto agent = zoo::Agent::create(config);
```

```cpp
// After: pass the three configuration concerns explicitly.
zoo::ModelConfig model_config;
model_config.model_path = "/models/llama.gguf";
model_config.context_size = 8192;

zoo::AgentConfig agent_config;
agent_config.max_history_messages = 64;

zoo::GenerationOptions generation;
generation.max_tokens = 128;

auto agent = zoo::Agent::create(model_config, agent_config, generation);
```

### Request Handles

Async calls still return request handles, but the handle surface changed.
Before, the handle exposed `.id` and a `std::future`; now it exposes `id()`,
`ready()`, and `await_result()`.

```cpp
// Before: request completion came from the embedded future.
auto handle = agent->chat(zoo::Message::user("Hello"));
auto response = handle.future.get();
if (response) {
    std::cout << response->text << "\n";
}
```

```cpp
// After: the call returns a request handle.
auto handle = agent->chat("Hello");
auto response = handle.await_result();
if (response) {
    std::cout << response->text << "\n";
}
```

### Extraction Results

Structured extraction now returns `ExtractionResponse`, which separates the raw model
text from the parsed JSON payload.

```cpp
// Before: extracted data lived on the generic response type.
auto handle = agent->extract(schema, zoo::Message::user("Alice is 30."));
auto response = handle.future.get();
if (response) {
    std::cout << response->extracted_data->dump() << "\n";
}
```

```cpp
// After: extraction data is part of the dedicated extraction response type.
auto handle = agent->extract(schema, "Alice is 30.");
auto response = handle.await_result();
if (response) {
    std::cout << response->data.dump() << "\n";
    std::cout << response->text << "\n";
}
```

### Message And History APIs

Request-scoped inputs now use `MessageView` and `ConversationView`, while
retained agent state stays behind `get_history()` and `clear_history()`. Use
`chat()` for appending a new turn and `complete()` for running against a
supplied history without mutating the retained conversation.

```cpp
// Before: scoped history passed owning Message values directly.
std::vector<zoo::Message> history = {
    zoo::Message::user("Hello"),
    zoo::Message::assistant("Hi there"),
    zoo::Message::user("What did I just say?")
};

auto scoped = agent->complete(history);
```

```cpp
// After: choose the request shape explicitly.
agent->chat(zoo::MessageView{zoo::Role::User, "Hello"});

const std::array<zoo::MessageView, 3> history = {
    zoo::MessageView{zoo::Role::User, "Hello"},
    zoo::MessageView{zoo::Role::Assistant, "Hi there"},
    zoo::MessageView{zoo::Role::User, "What did I just say?"},
};

auto scoped = agent->complete(
    zoo::ConversationView{std::span<const zoo::MessageView>(history)});
```

### Tool Calling

Tool calling is now template-driven through llama.cpp's native chat templates. Models
that do not expose a native tool format no longer rely on the old sentinel-based
fallback path.

## 0.2.x → 1.0.0

### Summary

No breaking API changes occurred between 0.2.x and 1.0.0. The public interface was stable throughout the pre-1.0 development cycle. A source-compatible upgrade is expected for any consumer that stayed within the documented public boundary.

### Public Boundary

The supported public API is:

- Headers under `include/zoo/`
- The primary CMake target `ZooKeeper::zoo`

Source files (`src/`) and CMake packaging internals are not part of the compatibility contract and may change in any release.

### CMake Target Changes

`ZooKeeper::zoo` is and has been the primary consumer target throughout 0.2.x and into 1.0.0.

`ZooKeeper::zoo_core` remains available as a compatibility alias that forwards to `ZooKeeper::zoo`. New consumers should use `ZooKeeper::zoo` directly.

### C++ Standard

C++23 is required. This has not changed from 0.2.x.

### llama.cpp Dependency at 1.0.0

At the 1.0.0 release, `extern/llama.cpp` was pinned to a specific commit
(`d1b4757dedbb60a811c8d7012249a96b1b702606`, tagged
`gguf-v0.17.1-2054-gd1b4757de`) for reproducible builds. Current releases use
CMake `FetchContent` instead; see the v1.1.2 → v1.1.3 notes above.

### What Has Not Changed

- All public headers: `zoo/zoo.hpp`, `zoo/agent.hpp`, `zoo/core/model.hpp`, `zoo/core/types.hpp`, `zoo/tools/registry.hpp`, `zoo/tools/parser.hpp`, `zoo/tools/validation.hpp`
- All public types: `zoo::Agent`, `zoo::core::Model`, `zoo::Message`, `zoo::Role`, `zoo::Config`, `zoo::Response`, `zoo::Error` (note: `zoo::Config` and `zoo::Response` were later split in v1.1.0 — see the v1.0.3 → v1.1.0 section for the current API surface)
- Error handling: `std::expected`-based throughout
- Include paths: unchanged

## Future Releases

For post-1.0 migration notes, see [CHANGELOG.md](CHANGELOG.md) and subsequent entries in this file.
