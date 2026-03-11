# Configuration Reference

All runtime configuration is provided through `zoo::Config` at Agent or Model creation time. Configuration is immutable after the runtime object is constructed.

## JSON Mapping

Zoo-Keeper provides opt-in JSON serialization helpers for `zoo::Config` and `zoo::SamplingParams` in `zoo/core/json.hpp`.

```cpp
#include <zoo/core/json.hpp>

#include <fstream>

std::ifstream file("config.json");
auto json = nlohmann::json::parse(file);
zoo::Config config = json.get<zoo::Config>();

if (auto validation = config.validate(); !validation) {
    throw std::runtime_error(validation.error().to_string());
}
```

The JSON mapping uses the same field names as the public structs and keeps `sampling` nested under the `sampling` key.

Deserialization is intentionally strict:

- `model_path` is required
- unknown top-level config keys are rejected
- unknown `sampling` keys are rejected
- JSON type mismatches fail during parse

Semantic validation remains separate. `from_json` handles known keys, defaults, and types; `Config::validate()` still checks runtime-valid values such as bounds and numeric ranges.

`Config::to_json` omits `system_prompt` when unset and always omits `on_token`, because callbacks are transient and not serializable.

## Config Fields

### Model Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `string` | (required) | Path to GGUF model file |
| `context_size` | `int` | `8192` | Context window size in tokens. Must be > 0 |
| `n_gpu_layers` | `int` | `0` | GPU layers to offload. Defaults to CPU-only; opt in to GPU offload explicitly |
| `use_mmap` | `bool` | `true` | Memory-map model file for faster loading |
| `use_mlock` | `bool` | `false` | Lock model in RAM (prevents swapping) |

### Sampling Parameters

Configured via `config.sampling`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `0.7` | Sampling temperature. 0.0 = deterministic, higher = more random |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | `int` | `40` | Top-K sampling limit. Must be >= 1 |
| `repeat_penalty` | `float` | `1.1` | Penalty for repeating tokens. 1.0 = no penalty |
| `repeat_last_n` | `int` | `64` | Number of tokens to consider for repeat penalty |
| `seed` | `int` | `-1` | Random seed. -1 = random seed per request |

### Generation Limits

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | `int` | `-1` | Maximum tokens to generate per response. -1 uses a safety cap of `context_size` |
| `stop_sequences` | `vector<string>` | empty | Additional stop strings to halt generation |

### System Prompt

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `optional<string>` | empty | System message prepended to all conversations |
| `max_history_messages` | `size_t` | `64` | Maximum number of non-system messages retained before the oldest turns are trimmed |

The system prompt can also be set/updated after creation via `agent->set_system_prompt()`.

### Request Queue

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `request_queue_capacity` | `size_t` | `64` | Maximum queued requests. Set to `0` to allow an unbounded queue |

### Tool Loop Controls

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tool_iterations` | `int` | `5` | Maximum detect/execute/respond iterations for a single request |
| `max_tool_retries` | `int` | `2` | Maximum validation retries for malformed tool calls |

### Callbacks

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on_token` | `optional<TokenCallback>` | empty | Default per-token streaming callback (returns `Continue` or `Stop`). Used by `Model::generate()` when no callback is passed directly. When using `Agent::chat()`, pass the callback there instead — the per-request callback takes precedence |

## Prompt Template

Zoo-Keeper uses `llama_chat_apply_template()` to auto-detect the chat template from the model's metadata. No manual template configuration is needed -- the correct format (Llama 3, ChatML, etc.) is determined automatically from the GGUF file.

If the selected GGUF does not expose a chat template, model creation now fails fast with `TemplateRenderFailed` instead of deferring the failure to first inference.

## Example: In-Code Configuration

```cpp
zoo::Config config;
config.model_path = "models/custom-model.gguf";
config.context_size = 4096;
config.max_tokens = 256;
config.n_gpu_layers = 32;
config.max_history_messages = 32;

// Sampling
config.sampling.temperature = 0.8f;
config.sampling.top_p = 0.95f;
config.sampling.top_k = 40;
config.sampling.repeat_penalty = 1.1f;
config.sampling.seed = 42;

// Stop sequences
config.stop_sequences = {"\n\n", "User:"};

// System prompt
config.system_prompt = "You are a helpful AI assistant.";

auto agent = std::move(*zoo::Agent::create(config));
```

## Example: JSON Configuration File

```json
{
  "model_path": "path/to/model.gguf",
  "context_size": 4096,
  "max_tokens": 256,
  "n_gpu_layers": 0,
  "use_mmap": true,
  "use_mlock": false,
  "system_prompt": "You are a helpful AI assistant.",
  "stop_sequences": [],
  "max_history_messages": 64,
  "request_queue_capacity": 64,
  "max_tool_iterations": 5,
  "max_tool_retries": 2,
  "sampling": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "repeat_last_n": 64,
    "seed": -1
  }
}
```

## Validation

`Config::validate()` is called automatically during `Agent::create()`. It checks:

- `model_path` is not empty
- `context_size` > 0
- `max_tokens` is positive or -1
- `sampling.temperature` >= 0.0
- `sampling.top_p` is in `[0.0, 1.0]`
- `sampling.top_k` >= 1
- `sampling.repeat_penalty` >= 0.0
- `sampling.repeat_last_n` >= 0
- `max_history_messages` >= 1
- `max_tool_iterations` >= 1
- `max_tool_retries` >= 0

When `max_tokens = -1`, generation is still bounded by `context_size` as a production safety limit.

Invalid configs produce an `Error` with the appropriate `ErrorCode` (100-series).

## See Also

- [Getting Started](getting-started.md) -- basic setup walkthrough
- [Building](building.md) -- CMake options and platform setup
