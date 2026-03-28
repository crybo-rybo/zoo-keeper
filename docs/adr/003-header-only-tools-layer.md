# ADR-003: Header-Only Tools Layer with Zero llama.cpp Dependency

## Status

Accepted

## Context

Tool registration, call parsing, schema validation, and grammar generation
could have been implemented as part of the core layer (tightly coupled to Model)
or as an independent layer.

## Decision

The tools layer (`zoo::tools`) is header-only and has zero dependency on
llama.cpp or `zoo::core`. It operates entirely on strings and `nlohmann::json`.

## Rationale

- **Testability:** The entire tools layer is unit-testable without a model,
  llama.cpp build, or GGUF file. This makes the test suite fast and
  deterministic.
- **Reusability:** Tool definitions, schemas, and parsing logic could
  theoretically be used with a different backend.
- **Build isolation:** Changes to llama.cpp or the core layer do not trigger
  recompilation of tools code.
- **Layer discipline:** Strict dependency direction (Agent → {Tools, Core} →
  llama.cpp) prevents circular dependencies and keeps the architecture legible.
  Tools and Core are independent siblings — Tools has no dependency on Core.

## Consequences

- Tools layer cannot directly access model state or tokenization.
- Any model-aware tool behavior (e.g., grammar-constrained generation) must be
  wired through the Agent layer, which composes both.
- JSON schema is generated from C++ function signatures via template
  metaprogramming, supporting: `int`, `float`, `double`, `bool`, `std::string`.
