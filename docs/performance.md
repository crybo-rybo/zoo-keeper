# Performance Notes

Zoo-Keeper's default API surface is now designed around the fast path:

- borrowed request inputs (`MessageView`, `ConversationView`, `std::string_view`)
- split config objects (`ModelConfig`, `AgentConfig`, `GenerationOptions`)
- typed request results (`TextResponse`, `ExtractionResponse`)
- slot-backed async handles instead of promise/future-heavy request tracking
- opt-in tool tracing (`GenerationOptions::record_tool_trace`)

## Local Benchmark Harness

Build the repo-local benchmark target:

```bash
scripts/build -DZOO_BUILD_BENCHMARKS=ON
```

Run the real-model benchmark harness:

```bash
build/benchmarks/zoo_benchmarks /path/to/model.gguf
```

Or supply the model path through the environment:

```bash
ZOO_BENCHMARK_MODEL=/path/to/model.gguf build/benchmarks/zoo_benchmarks
```

The harness is intentionally real-model-only. It currently covers:

- `Model::generate()` against the retained stateful conversation path
- `Model::generate_from_history()` against a supplied conversation snapshot

For each benchmark case, the harness reports:

- end-to-end latency
- time to first token
- decode throughput in tokens/second
- effective prefill throughput in prompt tokens/second
- prompt token counts
- completion token counts

If no GGUF path is supplied, the benchmark exits with an error instead of falling back to mocked or synthetic measurements.

## Layout Watchpoints

The benchmark executable prints a few hot type sizes at startup so layout drift is visible in normal development. On the current Apple Silicon reference build, the harness reports:

- `RequestHandle<TextResponse>`: 56 bytes
- `MessageView`: 80 bytes
- `OwnedMessage`: 80 bytes

Treat these as maintainer reference points rather than API guarantees. If these numbers move meaningfully, rerun the benchmarks and inspect whether the change buys enough clarity or functionality to justify the extra footprint.
