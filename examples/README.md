# Zoo-Keeper Examples

Runnable programs that exercise the public API. These binaries are the
compile-checked reference for end-to-end usage; the guides under `docs/` link
here instead of embedding full programs in markdown.

## Prerequisites

- macOS or Linux, C++23 toolchain, CMake 3.18+
- A local GGUF model file (see `.secret/integration-testing.md` for paths used
  in this repo)
- Examples enabled at configure time:

```bash
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

Binaries are written to `build/examples/`.

## Programs

| Binary | Source | What it shows |
|--------|--------|----------------|
| `minimal_model` | [`minimal_model.cpp`](minimal_model.cpp) | Smallest `zoo::Model::load()` + `generate()` flow |
| `demo_chat` | [`demo_chat.cpp`](demo_chat.cpp) | Interactive CLI, JSON config, streaming, metrics |
| `demo_extract` | [`demo_extract.cpp`](demo_extract.cpp) | Stateful, stateless, and streaming `Model::extract()` |
| `model_generate` | [`model_generate.cpp`](model_generate.cpp) | Minimal one-shot generation |
| `error_handling` | [`error_handling.cpp`](error_handling.cpp) | `Expected` error handling on `Model::load()` and `generate()` |
| `stream_cancel` | [`stream_cancel.cpp`](stream_cancel.cpp) | Token streaming callback + cancellation callback |
| `manual_tool_schema` | [`manual_tool_schema.cpp`](manual_tool_schema.cpp) | Manual JSON schema tools + native call parsing |

### `minimal_model`

```bash
./build/examples/minimal_model /path/to/model.gguf
```

Matches the walkthrough in [Getting Started](../docs/getting-started.md).

### `demo_chat`

Edit [`config.example.json`](config.example.json) so `model.model_path` points at
your GGUF, then:

```bash
./build/examples/demo_chat examples/config.example.json
```

Type `/help` in the REPL for commands. Covers multi-turn history, streaming,
cancellation, and response metrics.

[`config.auto.example.json`](config.auto.example.json) shows `auto_configure` in
the model block.

### `demo_extract`

```bash
./build/examples/demo_extract /path/to/model.gguf
```

See [Structured Output](../docs/extract.md) for schema details.

### `model_generate`

```bash
./build/examples/model_generate /path/to/model.gguf "What is the capital of France?"
```

### `error_handling`

```bash
./build/examples/error_handling /path/to/model.gguf
```

### `stream_cancel`

```bash
./build/examples/stream_cancel /path/to/model.gguf
```

### `manual_tool_schema`

```bash
./build/examples/manual_tool_schema /path/to/model.gguf
```

See [Tools](../docs/tools.md) for typed vs manual registration.

## Hub layer

There is no hub-specific example binary yet. Build with `-DZOO_BUILD_HUB=ON` and
see [Hub Layer](../docs/hub.md). Core-layer `GgufInspector` and `SystemProbe`
do not require the hub.

## Documentation map

| Topic | Guide | Runnable reference |
|-------|-------|-------------------|
| First model | [getting-started.md](../docs/getting-started.md) | `minimal_model` |
| Build / CI | [building.md](../docs/building.md) | all binaries above |
| Tools | [tools.md](../docs/tools.md) | `manual_tool_schema` |
| Extraction | [extract.md](../docs/extract.md) | `demo_extract` |
| Config JSON | [configuration.md](../docs/configuration.md) | `config.example.json` |
| Index in docs | [examples.md](../docs/examples.md) | this file |
