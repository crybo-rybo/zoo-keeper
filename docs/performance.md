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

Run the synthetic hot-path benchmarks:

```bash
build/benchmarks/zoo_benchmarks
```

Run the same harness with a live GGUF model to exercise prompt rendering from retained history:

```bash
ZOO_BENCHMARK_MODEL=/path/to/model.gguf build/benchmarks/zoo_benchmarks
```

The harness currently covers:

- stream trigger detection
- stateless runtime completion overhead
- extraction overhead
- tool-loop overhead
- live `Model::generate_from_history()` when a model path is supplied

## Layout Watchpoints

The benchmark executable prints a few hot type sizes at startup so layout drift is visible in normal development. On the current Apple Silicon reference build, the harness reports:

- `RequestHandle<TextResponse>`: 56 bytes
- `MessageView`: 80 bytes
- `OwnedMessage`: 80 bytes

Treat these as maintainer reference points rather than API guarantees. If these numbers move meaningfully, rerun the benchmarks and inspect whether the change buys enough clarity or functionality to justify the extra footprint.
