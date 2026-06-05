# Maintainer Architecture

This note documents private module boundaries behind the public Zoo-Keeper
harness API. Public-facing guidance stays in [architecture.md](architecture.md).

## Boundary Rules

- Only headers under `include/zoo/` are part of the supported installed API.
- `zoo::Model` is the public model-session name; it aliases the implementation
  class in `zoo::core`.
- Private headers live under `src/` and must not be installed or documented as
  consumer dependencies.
- llama resource ownership stays private; never include `llama.h` from public
  headers.

## Core Model Structure

| File | Responsibility |
|------|----------------|
| `src/core/model.cpp` | construction, destruction, factory, one-time backend setup |
| `src/core/model_init.cpp` | initialization and tokenization |
| `src/core/model_inference.cpp` | retained-history generation and raw generation passes |
| `src/core/model_harness.cpp` | stateless completion and schema extraction public harness methods |
| `src/core/model_prompt.cpp` | prompt delta rendering and KV-cache bookkeeping |
| `src/core/model_history.cpp` | history mutation and trimming |
| `src/core/model_sampling.cpp` | sampler construction and grammar updates |
| `src/core/model_tool_calling.cpp` | tool-call setup and response parsing |
| `src/core/stream_filter.*` | streaming token filtering for native tool-call triggers |
| `src/core/model_impl.hpp` | private llama handles, loaded model state, session state, and sampler policy |

Keep prompt rendering, KV-cache resets, and sampler rebuilds localized to core.
Do not spread llama.cpp calls outside `src/core/model*.cpp`.

## Tooling Boundaries

- `include/zoo/tools/*` contains the supported public tool utility API.
- `ToolRegistry` owns normalized metadata and optional handlers.
- Parser and validator operate on strings and JSON, not on llama internals.
- `src/tools/grammar.hpp` is private implementation used by schema extraction.

Tool execution is caller-owned. Do not reintroduce an autonomous tool loop
without a separate product/API decision.

## Hub Internals

`zoo::hub::ModelStore` stays the public facade for catalog operations, local
imports, HuggingFace pulls, and metadata-backed model loading. It reuses
`zoo::core::GgufInspector` for inspection and auto-configuration.

Catalog saves must remain temp-file-plus-rename operations; do not reintroduce
direct truncating writes to `catalog.json`.

## Documentation Split

- `architecture.md` explains the public harness and target structure.
- `maintainer-architecture.md` explains private ownership and implementation seams.
- `maintainer-cmake-packaging.md` explains package config generation and usage.

Before release prep, verify `README.md`, `docs/architecture.md`, and
`docs/spec.md` agree on `zoo::Model`, `zoo::tools`, and optional `zoo::hub`.
