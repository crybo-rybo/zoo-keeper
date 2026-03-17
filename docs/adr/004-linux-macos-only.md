# ADR-004: Linux and macOS Only (No Windows Support)

## Status

Accepted

## Context

Supporting Windows would require handling MSVC-specific C++23 gaps, different
threading primitives, Windows-specific path handling, and CI infrastructure.

## Decision

Zoo-Keeper targets Linux and macOS only. Windows is not supported.

## Rationale

- **C++23 readiness:** At the time of this decision, MSVC's C++23 support
  (particularly `std::expected`) was incomplete or had different semantics.
- **llama.cpp alignment:** The primary llama.cpp deployment targets are Linux
  and macOS (including Metal acceleration on Apple Silicon).
- **Maintenance cost:** Supporting a third platform doubles CI matrix complexity
  and conditional compilation paths for a project with limited maintainer time.
- **Target audience:** Users running local LLMs typically use Linux or macOS.

## Consequences

- No MSVC CI, no Windows-specific code paths.
- Hardware acceleration: Metal (macOS), CUDA (Linux).
- Contributors on Windows would need WSL2.
