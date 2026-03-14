# Changelog

All notable changes to Zoo-Keeper will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Zoo-Keeper adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2026-03-14

### Changed

- Updated the `CMakeLists.txt` file to reflect the current project version.

## [1.0.1] - 2026-03-13

### Added

- **`Agent::complete(messages, callback)`** — stateless, request-scoped completion API. Accepts a full `std::vector<Message>` history and executes it without mutating the agent's retained conversation state. Streaming, tool execution, and cancellation work identically to `chat()`. Useful for server workloads where a single `Agent` instance serves many independent callers concurrently.
- **`AgentBackend::replace_messages` / `Model::replace_messages`** — new method that replaces the retained message list without flushing the KV cache, enabling efficient history restore after a scoped request.

### Fixed

- After a `complete()` call the agent's retained history is fully restored; subsequent `chat()` calls see no state from the scoped request.
- History restore after `complete()` no longer issues a redundant `llama_memory_clear`: the KV cache is left intact and stale entries are naturally overwritten when the next prompt is decoded from position zero, avoiding unnecessary latency on the first `chat()` turn following a `complete()`.

### Changed

- Internal `Request` payload now carries `std::vector<Message>` and a `HistoryMode` discriminant (`Append` / `Replace`) instead of a single `Message`. The existing `chat()` path is source-compatible and behavior-identical.
- `RequestTracker::prepare` gains a vector overload for replace-mode requests.
- `AgentRuntime` exposes `complete()` in the same pattern as `chat()` with equivalent queue-full, not-running, and empty-history error paths.

### Tests

- Unit: `CompleteUsesScopedHistoryAndRestoresPersistentHistory` — verifies retained history is unchanged after a scoped request.
- Unit: `CompleteRejectsEmptyMessageHistory` — verifies `InvalidMessageSequence` is returned for an empty message list.
- Unit: `PrepareVectorPayloadPreservesMessagesAndHistoryMode` — verifies tracker correctly round-trips the vector payload and `HistoryMode`.
- Unit: mailbox tests updated to validate `Request.messages` vector representation.
- Integration: `AgentCompleteDoesNotMutatePersistentHistory` — live-model end-to-end verification that `complete()` leaves public history unchanged.

## [1.0.0] - 2026-03-11

### Added

- **`zoo::core::Model`** — direct synchronous llama.cpp wrapper with model loading, tokenization, generation, history tracking, and incremental KV-cache management
- **`zoo::tools`** — ToolRegistry (typed + manual JSON tool registration), ToolCallParser (streaming tool-call extraction), ErrorRecovery (validation and retry logic), GrammarBuilder (GBNF grammar generation), ToolCallInterceptor (streaming interception)
- **`zoo::Agent`** — async orchestrator with per-request queue, dedicated inference thread, cancellable `RequestHandle` futures, streaming callbacks, and autonomous tool execution loop
- **`std::expected`-based error handling** throughout the public API — no exceptions required
- **CMake package** with `ZooKeeper::zoo` as the primary consumer target; supports `find_package`, `FetchContent`, and `add_subdirectory`; includes both build-tree and install-tree configurations
- **pkg-config support** via `zoo-keeper.pc` for non-CMake consumers
- **CI pipeline**: build matrix covering Ubuntu (GCC, Clang) and macOS (Clang + Metal), AddressSanitizer, UBSanitizer, code coverage, clang-format/clang-tidy lint, and packaging smoke tests
- **Full Doxygen API reference** generated from public headers
- **Documentation**: getting-started guide, architecture overview, configuration reference, tools guide, example programs, compatibility policy, and building guide
- **142 unit tests** covering types, tool parsing, grammar generation, schema validation, streaming interception, and batch computation
- llama.cpp submodule pinned to `d1b4757dedbb60a811c8d7012249a96b1b702606` (`gguf-v0.17.1-2054-gd1b4757de`) for reproducible builds

### Platform Support

- Linux (GCC 13+, Clang 17+)
- macOS (Clang, Metal acceleration available)
- C++23 required
- Windows is not supported

[1.0.1]: https://github.com/crybo-rybo/zoo-keeper/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/crybo-rybo/zoo-keeper/releases/tag/v1.0.0
