# Hub Layer

The hub layer (`zoo::hub`) is an optional Layer 4 that provides GGUF file
inspection, HuggingFace model downloading, and a local model catalog. It is
only compiled when `ZOO_BUILD_HUB=ON`.

```bash
scripts/build.sh -DZOO_BUILD_HUB=ON
```

## GGUF Inspection

`GgufInspector::inspect()` reads model metadata from a GGUF file without
loading tensor weights. It uses a two-phase approach:

1. `gguf_init_from_file()` for raw key-value metadata extraction
2. `llama_model_load_from_file()` with `vocab_only=true` for derived
   statistics (parameter count, file size, layer count)

The result is a `ModelInfo` struct containing architecture, quantization type,
context length, embedding dimensions, and all raw GGUF metadata as a
string-to-string map.

`GgufInspector::auto_configure()` generates a `ModelConfig` with sensible
defaults from inspected metadata:

- `context_size` = min(training context, 8192) to avoid OOM
- `n_gpu_layers` = -1 (offload all layers)
- `use_mmap` = true, `use_mlock` = false

```cpp
auto info = zoo::hub::GgufInspector::inspect("/path/to/model.gguf");
if (!info) { /* handle error */ }

std::cout << info->name << " (" << info->description << ")\n";
std::cout << "Layers: " << info->layer_count << "\n";
std::cout << "Context: " << info->context_length << "\n";

// One-step: inspect + auto-configure
auto config = zoo::hub::GgufInspector::auto_configure("/path/to/model.gguf");
auto model = zoo::core::Model::load(*config).value();
```

## HuggingFace Client

`HuggingFaceClient` wraps llama.cpp's `llama-common` download infrastructure.
HuggingFace repository downloads go into llama.cpp's Hugging Face-style cache,
honoring the same environment variables as llama.cpp:

- `LLAMA_CACHE`
- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `HF_HOME`/`hub`
- `XDG_CACHE_HOME`/`huggingface/hub`
- `~/.cache/huggingface/hub`

Models downloaded by any llama.cpp tool (llama-cli, llama-server, etc.) are
immediately available, and vice versa. The client supports ETag caching,
resume, multi-split GGUF files, and retry with exponential backoff.

```cpp
auto hf = zoo::hub::HuggingFaceClient::create().value();

// Download a model (returns local file path)
auto path = hf->download_model("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M");
if (!path) { /* handle error */ }

std::cout << "Downloaded to: " << *path << "\n";

// Download a specific file from a repository
auto exact = hf->download_model(
    "bartowski/Qwen3-8B-GGUF::Qwen3-8B-Q4_K_M.gguf");

// List models already in the llama.cpp cache
auto cached = zoo::hub::HuggingFaceClient::list_cached_models();
for (const auto& m : cached) {
    std::cout << m.to_string() << "\n";
}
```

`CachedModelInfo::size_bytes` is retained for source compatibility but is
reported as `0`, because llama.cpp's b8992 cache listing exposes repository and
tag only.

For gated models, pass a bearer token via `HuggingFaceClient::Config`:

```cpp
auto hf = zoo::hub::HuggingFaceClient::create({.token = "hf_..."}).value();
```

## Identifier Formats

`HuggingFaceClient::parse_identifier()` accepts three formats:

| Format | Example | Meaning |
|--------|---------|---------|
| `owner/repo::file.gguf` | `bartowski/Qwen3-8B-GGUF::Qwen3-8B-Q4_K_M.gguf` | Specific file in a repository |
| `owner/repo:tag` | `bartowski/Qwen3-8B-GGUF:Q4_K_M` | Repository with quantization tag (ollama/llama.cpp style) |
| `owner/repo` | `bartowski/Qwen3-8B-GGUF` | Repository, resolves to best available GGUF |

## Model Store

`ModelStore` manages a local catalog of downloaded GGUF models, persisted as
JSON in the store directory (default: `~/.zoo-keeper/models/`).

The store supports alias-based lookup, auto-configuration from cached
inspection metadata, and one-liner Model or Agent creation.

```cpp
auto store = zoo::hub::ModelStore::open().value();
auto hf = zoo::hub::HuggingFaceClient::create().value();

// Download and register in one step
store->pull(*hf, "bartowski/Qwen3-8B-GGUF:Q4_K_M", {"qwen3"});

// Or register a local file
store->add("/path/to/model.gguf", {"my-model"});

// Find by alias
auto entry = store->find("qwen3").value();
std::cout << entry.info.name << " at " << entry.file_path << "\n";

// One-liner: alias to running agent
auto agent = store->create_agent("qwen3").value();
agent->set_system_prompt("You are a helpful assistant.");

// Or load a core::Model directly
auto model = store->load_model("qwen3").value();
```

Catalog operations: `add()`, `remove()`, `find()`, `list()`, `add_alias()`.
Resolution order for `find()`: exact alias match, then name substring, then
path match.

## Error Codes

Hub error codes occupy the 700-799 range. Symbolic names live in
`zoo::hub::HubErrorCode`; returned `zoo::Error` values carry the corresponding
core numeric code via `zoo::hub::to_error_code(...)`.

| Code | Name | Description |
|------|------|-------------|
| 700 | `HubErrorCode::GgufReadFailed` | Could not open or parse a GGUF file |
| 701 | `HubErrorCode::GgufMetadataNotFound` | An expected metadata key was missing |
| 702 | `HubErrorCode::ModelNotFound` | No model matched the given name, alias, or path |
| 703 | `HubErrorCode::ModelAlreadyExists` | A model with the same path is already registered |
| 704 | `HubErrorCode::DownloadFailed` | HTTP download failed |
| 706 | `HubErrorCode::HuggingFaceApiError` | The HuggingFace API returned an error |
| 707 | `HubErrorCode::InvalidModelIdentifier` | Could not parse the identifier string |
| 708 | `HubErrorCode::StoreCorrupted` | The catalog JSON is malformed |
| 709 | `HubErrorCode::FilesystemError` | A filesystem operation failed |

## See Also

- [Getting Started](getting-started.md) -- basic Agent setup
- [Architecture](architecture.md) -- layer design and threading model
- [Examples](examples.md) -- complete usage snippets including model store
