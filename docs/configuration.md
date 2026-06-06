# Configuration Reference

Runtime setup is split across two public value types:
`zoo::ModelConfig` and `zoo::GenerationOptions`. JSON helpers for those types,
plus `SamplingParams`, live in `zoo/core/json.hpp`.

## JSON Mapping

```cpp
#include <zoo/core/json.hpp>

#include <fstream>

std::ifstream file("config.json");
auto json = nlohmann::json::parse(file);

zoo::ModelConfig model = json.at("model").get<zoo::ModelConfig>();
zoo::GenerationOptions generation = json.at("generation").get<zoo::GenerationOptions>();
```

`ModelConfig::validate()` and `GenerationOptions::validate()` remain separate
from JSON parsing. `Model::load(model, generation)` runs both validations before
loading llama.cpp state.

## `zoo::ModelConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_path` | `string` | required | Path to the GGUF model file |
| `context_size` | `int` | `8192` | Requested context window size in tokens |
| `n_batch` | `int` | `2048` | Batch size for prompt processing |
| `n_gpu_layers` | `int` | `0` | Number of layers to offload to GPU. `0` is strict CPU-only; `-1` requests full offload |
| `use_mmap` | `bool` | `true` | Memory-map the model file |
| `use_mlock` | `bool` | `false` | Lock model pages in RAM |

Model JSON may include `"auto_configure": true`. That key is resolved only by
`zoo::load_model_config()`, which inspects the GGUF file, probes hardware, and
then applies explicit JSON overrides.

## `zoo::GenerationOptions`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sampling` | `SamplingParams` | default-constructed | Sampling behavior |
| `max_tokens` | `int` | `-1` | Completion cap, or `-1` for the context-limited maximum |
| `stop_sequences` | `vector<string>` | empty | Additional stop strings |

## `zoo::SamplingParams`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | `float` | `0.7` | Sampling temperature |
| `top_p` | `float` | `0.9` | Nucleus sampling threshold |
| `top_k` | `int` | `40` | Top-K sampling limit |
| `repeat_penalty` | `float` | `1.1` | Penalty for repeated tokens |
| `repeat_last_n` | `int` | `64` | Number of tokens considered by the repeat penalty |
| `seed` | `int` | `-1` | Random seed, with `-1` meaning per-request randomness |

## Example

```cpp
zoo::ModelConfig model;
model.model_path = "models/custom-model.gguf";
model.context_size = 4096;
model.n_gpu_layers = 16;

zoo::GenerationOptions generation;
generation.max_tokens = 256;
generation.sampling.temperature = 0.8f;
generation.sampling.top_p = 0.95f;

auto loaded = zoo::Model::load(model, generation);
```

`examples/config.example.json` shows the release-facing JSON shape used by
`demo_chat`.

## Validation

- `ModelConfig`: `model_path` must be set and `context_size` / `n_batch` must be positive
- `GenerationOptions`: `max_tokens` must be positive or `-1`, and sampling values must be valid

## See Also

- [Getting Started](getting-started.md)
- [Building](building.md)
