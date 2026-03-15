# Changelog

All notable changes to Zoo-Keeper will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Zoo-Keeper adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] - 2026-03-14

### Added

- **`Agent::extract(output_schema, message, callback)`** — stateful structured extraction API. Constrains model generation to produce JSON conforming to a caller-supplied JSON Schema, parses the output, validates it against the schema, and returns the result via a new `Response::extracted_data` field. Reuses the existing grammar and validation infrastructure with a simplified single-pass flow (no tool loop).
- **`Agent::extract(output_schema, messages, callback)`** — stateless variant that operates on a provided message history without mutating agent state, mirroring the `chat()` / `complete()` pattern.
- **`GrammarBuilder::build_schema(parameters)`** — generates a GBNF grammar for a standalone JSON schema (no `<tool_call>` sentinel wrapping). Shares the same parameter/optional/enum rule generation logic as the tool grammar builder via refactored prefix-parameterized helpers.
- **`validate_json_against_schema(data, parameters)`** — free function in `zoo::tools` that validates a JSON object against a `ToolParameter` vector without requiring a `ToolCall` or `ToolRegistry`.
- **`detail::normalize_schema(schema)`** — public helper that normalizes a JSON Schema into a `vector<ToolParameter>`, extracted from the existing `normalize_manual_tool_metadata()`.
- **`Model::set_schema_grammar(grammar_str)`** — enables non-lazy (immediately active) grammar constraints for schema output, using `llama_sampler_init_grammar()` instead of the lazy sentinel-triggered `llama_sampler_init_grammar_lazy_patterns()` used for tool calling.
- **`AgentBackend::set_schema_grammar(grammar_str)`** — virtual method on the internal backend seam, enabling test fakes to observe schema grammar setup.
- **`Response::extracted_data`** — new `std::optional<nlohmann::json>` field on `Response`, populated only by `extract()` calls.
- **`ErrorCode::InvalidOutputSchema` (600)** and **`ErrorCode::ExtractionFailed` (601)** — new error codes for extraction-specific failures.
- **`Request::extraction_schema`** — internal field on the request payload for routing extraction requests through the runtime.
- **`RequestTracker::prepare` extraction overloads** — accept an extraction schema alongside messages and callbacks.

### Changed

- `Model` grammar state replaced: `bool grammar_active_` → `GrammarMode` enum (`None`, `ToolCall`, `Schema`) to distinguish lazy (tool) from immediate (schema) grammar activation.
- `GrammarBuilder` internals refactored: rule-generation helpers are now prefix-parameterized (`append_prefixed_parameter_rules`, `append_prefixed_optional_rules`, `build_prefixed_required_sequence`), shared by both `build()` and `build_schema()`.
- `model_inference.cpp`: `<tool_call>` sentinel detection is now gated on `GrammarMode::ToolCall`, preventing false tool-call detection during schema-constrained generation.
- `generate_from_history()` dispatches grammar rebuild to `rebuild_sampler_with_grammar()` or `rebuild_sampler_with_schema_grammar()` based on the active grammar mode.
- `AgentRuntime::process_request()` dispatches to `process_extraction_request()` when the request carries an extraction schema.
- CMake project version bumped to 1.0.3.

### Tests

- **10 new schema grammar tests** (`test_schema_grammar.cpp`): empty params, no sentinels, single required property, mixed required/optional, all primitive types (integer, number, string, boolean), enum constraints, all-optional properties, and comparison with tool grammar output.
- **10 new extraction tests** (`test_extraction.cpp`): valid JSON extraction, invalid schema rejection, grammar setup/restore, tool grammar restoration after extraction, streaming callbacks, cancellation, stateful and stateless modes, single-pass routing (no tool loop), malformed JSON output error, and `extracted_data` is `nullopt` on normal `chat()` responses.
- Existing `FakeBackend` in `test_agent_runtime.cpp` updated with `set_schema_grammar` override.
- All 218 tests pass (previously 198).

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

[1.0.3]: https://github.com/crybo-rybo/zoo-keeper/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/crybo-rybo/zoo-keeper/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/crybo-rybo/zoo-keeper/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/crybo-rybo/zoo-keeper/releases/tag/v1.0.0
