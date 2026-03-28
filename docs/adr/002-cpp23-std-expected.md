# ADR-002: C++23 with std::expected for Error Handling

## Status

Accepted

## Context

The project needed an error handling strategy. Options considered:
1. C++ exceptions (`throw`/`catch`)
2. `tl::expected` (pre-C++23 polyfill)
3. `std::expected` (C++23 standard)
4. Error codes

## Decision

Use C++23 as the language standard with `std::expected` for all fallible
operations. No exceptions anywhere in the codebase.

## Rationale

- **Value semantics:** `std::expected` makes error paths explicit at the call
  site. Callers cannot accidentally ignore errors the way they can with
  exceptions.
- **No hidden control flow:** Exceptions create invisible goto paths that are
  especially hard for AI agents and new contributors to reason about.
- **Standard, not polyfill:** `std::expected` is part of C++23 and requires no
  external dependency. `tl::expected` was previously used but dropped in favor
  of the standard version.
- **Compiler support:** GCC 13+, Clang 18+, and Apple Clang 16+ all support
  `std::expected`. The project already requires C++23 for other features.

## Consequences

- Minimum compiler requirements are higher (GCC 13+, Clang 18+, Apple Clang 16+).
- Windows support is not a target (MSVC C++23 support was lagging at decision time).
- All public APIs return `std::expected<T, zoo::Error>` rather than throwing.
- Error codes are organized by category (100s=config, 200s=backend, 300s=engine,
  400s=runtime, 500s=tools, 600s=extraction).
