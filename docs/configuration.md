# Configuration Reference

Runtime setup is split across three value types: `zoo::ModelConfig`, `zoo::AgentConfig`, and `zoo::GenerationOptions`. The library provides opt-in JSON helpers for those types, plus `SamplingParams`, in `zoo/core/json.hpp`.

## JSON Mapping

Each struct maps to a JSON object with the same field names as the public API. The mapping is strict: unknown keys are rejected, required keys are checked, and type mismatches fail during parse.

```cpp
#include <zoo/core/json.hpp>

#include <fstream>

std::ifstream file("config.json");
auto json = nlohmann::json::parse(file);

zoo::ModelConfig model = json.at("model").get<zoo::ModelConfig>();
zoo::AgentConfig agent = json.at("agent").get<zoo::AgentConfig>();
zoo::GenerationOptions generation = json.at("generation").get<zoo::GenerationOptions>();
```

`ModelConfig::validate()`, `AgentConfig::validate()`, and `GenerationOptions::validate()` remain separate from JSON parsing. `Agent::create(model, agent, generation)` runs all three validations before the model is loaded.

## Config Fields

### `zoo::ModelConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `string` | required | Path to the GGUF model file |
| `context_size` | `int` | `8192` | Requested context window size in tokens |
| `n_gpu_layers` | `int` | `0` | Number of layers to offload to GPU |
| `use_mmap` | `bool` | `true` | Memory-map the model file |
| `use_mlock` | `bool` | `false` | Lock model pages in RAM |

### `zoo::AgentConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_history_messages` | `size_t` | `64` | Maximum non-system messages retained in history |
| `request_queue_capacity` | `size_t` | `64` | Maximum queued requests owned by the agent |
| `max_tool_iterations` | `int` | `5` | Detect/execute/respond iterations per request |
| `max_tool_retries` | `int` | `2` | Validation retries for malformed tool calls |

### `zoo::GenerationOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sampling` | `SamplingParams` | default-constructed | Sampling behavior for the request |
| `max_tokens` | `int` | `-1` | Completion cap, or `-1` for the context-limited maximum |
| `stop_sequences` | `vector<string>` | empty | Additional stop strings |
| `record_tool_trace` | `bool` | `false` | Materialize `TextResponse::tool_trace` / `ExtractionResponse::tool_trace` |

### `zoo::SamplingParams`

Configured inside `generation.sampling`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `0.7` | Sampling temperature |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-K sampling limit |
| `repeat_penalty` | `float` | `1.1` | Penalty for repeated tokens |
| `repeat_last_n` | `int` | `64` | Number of tokens considered by the repeat penalty |
| `seed` | `int` | `-1` | Random seed, with `-1` meaning per-request randomness |

## Example: In Code

```cpp
zoo::ModelConfig model;
model.model_path = "models/custom-model.gguf";
model.context_size = 4096;
model.n_gpu_layers = 16;

zoo::AgentConfig agent;
agent.max_history_messages = 32;
agent.request_queue_capacity = 128;

zoo::GenerationOptions generation;
generation.max_tokens = 256;
generation.record_tool_trace = true;
generation.sampling.temperature = 0.8f;
generation.sampling.top_p = 0.95f;

auto runtime = zoo::Agent::create(model, agent, generation);
```

## Example: JSON File

`examples/config.example.json` shows the release-facing shape used by `demo_chat`:

```json
{
  "model": {
    "model_path": "path/to/model.gguf",
    "context_size": 4096,
    "n_gpu_layers": -1,
    "use_mmap": true,
    "use_mlock": false
  },
  "agent": {
    "max_history_messages": 64,
    "request_queue_capacity": 64,
    "max_tool_iterations": 5,
    "max_tool_retries": 2
  },
  "generation": {
    "max_tokens": -1,
    "stop_sequences": [],
    "record_tool_trace": false,
    "sampling": {
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "repeat_penalty": 1.1,
      "repeat_last_n": 64,
      "seed": -1
    }
  },
  "system_prompt": "You are a helpful assistant with access to tools.",
  "tools": true
}
```

The example app wraps the three config blocks in a small top-level struct so it can carry `system_prompt` and the example-only `tools` toggle alongside them.

## Validation

Validation checks run automatically inside `Agent::create()`.

- `ModelConfig`: `model_path` must be set and `context_size` must be positive
- `AgentConfig`: `max_history_messages` and `request_queue_capacity` must be at least 1
- `GenerationOptions`: `max_tokens` must be positive or `-1`, and `sampling` must be valid

If validation fails, construction returns an `Error` with the relevant `ErrorCode`.

## See Also

- [Getting Started](getting-started.md) -- basic setup walkthrough
- [Building](building.md) -- CMake options and platform setup
