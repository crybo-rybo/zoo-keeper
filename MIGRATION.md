# Migration Guide

This document covers what consumers need to know when upgrading Zoo-Keeper.

## 0.2.x → 1.0.0

### Summary

No breaking API changes occurred between 0.2.x and 1.0.0. The public interface was stable throughout the pre-1.0 development cycle. A source-compatible upgrade is expected for any consumer that stayed within the documented public boundary.

### Public Boundary

The supported public API is:

- Headers under `include/zoo/` (excluding `include/zoo/internal/`)
- The primary CMake target `ZooKeeper::zoo`

Internal headers (`include/zoo/internal/`), source files (`src/`), and CMake packaging internals are not part of the compatibility contract and may change in any release.

### CMake Target Changes

`ZooKeeper::zoo` is and has been the primary consumer target throughout 0.2.x and into 1.0.0.

`ZooKeeper::zoo_core` remains available as a compatibility alias that forwards to `ZooKeeper::zoo`. New consumers should use `ZooKeeper::zoo` directly.

### C++ Standard

C++23 is required. This has not changed from 0.2.x.

### llama.cpp Submodule

The `extern/llama.cpp` submodule is now pinned to a specific commit (`d1b4757dedbb60a811c8d7012249a96b1b702606`, tagged `gguf-v0.17.1-2054-gd1b4757de`) for reproducible builds. Consumers embedding Zoo-Keeper via `add_subdirectory` or `FetchContent` will get this pinned version automatically.

### What Has Not Changed

- All public headers: `zoo/zoo.hpp`, `zoo/agent.hpp`, `zoo/core/model.hpp`, `zoo/core/types.hpp`, `zoo/tools/registry.hpp`, `zoo/tools/parser.hpp`, `zoo/tools/validation.hpp`
- All public types: `zoo::Agent`, `zoo::core::Model`, `zoo::Message`, `zoo::Role`, `zoo::Config`, `zoo::Response`, `zoo::Error`
- Error handling: `std::expected`-based throughout
- Include paths: unchanged

## Future Releases

For post-1.0 migration notes, see [CHANGELOG.md](CHANGELOG.md) and subsequent entries in this file.
