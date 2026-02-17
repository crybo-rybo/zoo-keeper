# Configuration Reference

All configuration is provided through the `zoo::Config` struct at Agent creation time. Configuration is immutable after the Agent is constructed.

## Config Fields

### Model Settings

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `string` | (required) | Path to GGUF model file |
| `context_size` | `int` | `8192` | Context window size in tokens. Must be > 0 |
| `n_gpu_layers` | `int` | `-1` | GPU layers to offload. -1 = all, 0 = CPU only |
| `use_mmap` | `bool` | `true` | Memory-map model file for faster loading |
| `use_mlock` | `bool` | `false` | Lock model in RAM (prevents swapping) |

### Sampling Parameters

Configured via `config.sampling`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `0.7` | Sampling temperature. 0.0 = deterministic, higher = more random |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold (0.0-1.0) |
| `top_k` | `int` | `40` | Top-K sampling limit. 0 = disabled |
| `repeat_penalty` | `float` | `1.1` | Penalty for repeating tokens. 1.0 = no penalty |
| `repeat_last_n` | `int` | `64` | Number of tokens to consider for repeat penalty |
| `seed` | `int` | `-1` | Random seed. -1 = random seed per request |

### Prompt Template

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_template` | `PromptTemplate` | `Llama3` | Chat template format |
| `custom_template` | `optional<string>` | empty | Custom template string (required if `PromptTemplate::Custom`) |

Available templates:
- `PromptTemplate::Llama3` -- Meta Llama 3 format
- `PromptTemplate::ChatML` -- ChatML format (`<|im_start|>` / `<|im_end|>`)
- `PromptTemplate::Custom` -- User-provided template string

### Generation Limits

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens` | `int` | `512` | Maximum tokens to generate per response. Must be > 0 |
| `stop_sequences` | `vector<string>` | empty | Additional stop strings to halt generation |

### System Prompt

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_prompt` | `optional<string>` | empty | System message prepended to all conversations |

The system prompt can also be set/updated after creation via `agent->set_system_prompt()`.

### Callbacks

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `on_token` | `optional<function<void(string_view)>>` | empty | Global per-token streaming callback (runs on inference thread) |

## Example: Custom Configuration

```cpp
zoo::Config config;
config.model_path = "models/custom-model.gguf";
config.context_size = 4096;
config.max_tokens = 256;
config.n_gpu_layers = 32;

// Sampling
config.sampling.temperature = 0.8f;
config.sampling.top_p = 0.95f;
config.sampling.top_k = 40;
config.sampling.repeat_penalty = 1.1f;
config.sampling.seed = 42;

// Stop sequences
config.stop_sequences = {"\n\n", "User:"};

// Custom template
config.prompt_template = zoo::PromptTemplate::Custom;
config.custom_template = "{{role}}: {{content}}\n";

// System prompt
config.system_prompt = "You are a helpful AI assistant.";

auto agent = std::move(*zoo::Agent::create(config));
```

## ChatOptions (Per-Request)

Per-request options are passed to `chat()`:

```cpp
zoo::ChatOptions options;
options.rag.enabled = true;
options.rag.top_k = 4;

auto future = agent->chat(zoo::Message::user("Hello"), options);
```

### RagOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable RAG retrieval for this request |
| `top_k` | `int` | `4` | Number of chunks to retrieve |
| `context_override` | `optional<string>` | empty | Bypass retriever with precomputed context |

See [RAG Retrieval](rag.md) for details.

## Validation

`Config::validate()` is called automatically during `Agent::create()`. It checks:

- `model_path` is not empty
- `context_size` > 0
- `max_tokens` > 0
- If `prompt_template` is `Custom`, `custom_template` must be provided

Invalid configs produce an `Error` with the appropriate `ErrorCode` (100-series).

## See Also

- [Getting Started](getting-started.md) -- basic setup walkthrough
- [Building](building.md) -- CMake options and platform setup
- [RAG Retrieval](rag.md) -- RagOptions details
