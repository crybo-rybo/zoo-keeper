# Changelog

All notable changes to Zoo-Keeper will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Zoo-Keeper adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/crybo-rybo/zoo-keeper/releases/tag/v1.0.0
