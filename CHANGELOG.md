# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `SamplingParams::validate()` — validates temperature, top_p, top_k, repeat_penalty, repeat_last_n
- `Config::max_tool_iterations` — configurable tool loop iteration limit (default: 5)
- `Config::max_tool_retries` — configurable per-tool retry limit (default: 2)
- `ZOO_ENABLE_LOGGING` CMake option — enables `[zoo:level]` diagnostic output to stderr
- `zoo/version.hpp` — compile-time version constants (`VERSION_MAJOR`, `VERSION_MINOR`, `VERSION_PATCH`, `VERSION_STRING`)
- `CHANGELOG.md`
- `docs/compatibility.md` — intended 1.x public API and deprecation policy
- Inference safety cap: generation is bounded by `context_size` when `max_tokens = -1`
- Exception safety: `inference_loop()` catches exceptions and fulfills promises with errors

### Changed
- `Config::request_queue_capacity` default changed from `0` (unbounded) to `64`
- `Config::validate()` now validates sampling parameters and tool iteration limits
- `ErrorCode::InvalidSamplingParams` (103) added for sampling validation errors
- Build-tree CMake package config now links the full llama/ggml dependency set for actual consumers
- CI now smoke-tests build-tree and installed-package CMake consumers
- README/docs no longer hardcode a stale unit-test count

## [0.2.0] - 2026-03-07

### Added
- `ToolCallInterceptor` — real-time tool call detection during streaming
- `compute_prefill_chunks()` — chunked prefill for large prompts
- Batch processing utilities (`core/batch.hpp`)

### Changed
- KV cache default changed from F16 to Q8_0 for reduced memory footprint

### Fixed
- Empty response after tool call — model now re-prompted when it emits EOG immediately after tool result

## [0.1.0] - 2025-12-01

### Added
- Initial release
- `zoo::core::Model` — direct llama.cpp wrapper (model loading, inference, tokenization, history)
- `zoo::tools::ToolRegistry` — type-safe tool registration with automatic JSON schema generation
- `zoo::tools::ToolCallParser` — tool call detection in model output
- `zoo::tools::ErrorRecovery` — argument validation and retry tracking
- `zoo::Agent` — async orchestrator with inference thread and agentic tool loop
- Chat template support via llama.cpp
- Streaming token callbacks
- Request cancellation
- `demo_chat` example application
