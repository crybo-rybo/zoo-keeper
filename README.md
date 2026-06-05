<p align="center">
  <img src="docs/images/zoo_keeper_logo.png" alt="Zoo-Keeper logo" width="220">
</p>

<h1 align="center">Zoo-Keeper</h1>

<p align="center">
  <b>A C++23 llama.cpp LLM harness for local model sessions.</b><br/>
  <sub>Model loading &bull; Streaming inference &bull; Native tool-call parsing &bull; Structured output</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/C%2B%2B-23-blue" alt="C++23" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/tests-ctest%20passing-success" alt="Tests" />
</p>

---

## What is Zoo-Keeper?

Zoo-Keeper is a C++23 LLM harness built tightly on [llama.cpp](https://github.com/ggerganov/llama.cpp). It wraps the raw C API with a type-safe model session for embedding local GGUF-backed inference in desktop apps, tools, games, and edge systems — no server required.

**llama.cpp is the engine. Zoo-Keeper is the harness.**

At a high level, Zoo-Keeper provides:

- **`zoo::Model`** — model loading, retained history, stateless completion, streaming, cancellation, and schema extraction
- **`zoo::tools`** — caller-owned tool metadata, parsing, and schema validation
- **`zoo::hub`** *(optional)* — HuggingFace downloads and a local model store

See [Architecture](docs/architecture.md) for the harness boundary and llama.cpp ownership model.

## Quick Start

```bash
git clone https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
```

llama.cpp is fetched automatically at CMake configure time.

```cpp
#include <zoo/zoo.hpp>

auto model = zoo::Model::load(model_config).value();
model->set_system_prompt("You are a helpful assistant.");
auto response = model->generate("Hello!").value();
```

For CMake integration, configuration, tools, streaming, and error handling, see [Getting Started](docs/getting-started.md) and [Building](docs/building.md). Runnable programs live under [`examples/`](examples/README.md).

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | First build and first model session |
| [Building](docs/building.md) | CMake setup, FetchContent, Metal/CUDA, sanitizers, install/package |
| [Configuration](docs/configuration.md) | Model config, sampling parameters, generation limits, JSON config |
| [Tools](docs/tools.md) | Tool schemas, native call parsing, supported schema subset, error handling |
| [Structured Output](docs/extract.md) | Grammar-constrained extraction, schema reference, stateful vs. stateless |
| [Hub Layer](docs/hub.md) | HuggingFace downloading, local model store, and how hub code uses core inspection |
| [Architecture](docs/architecture.md) | Layer design, runtime ownership, threading model, target structure |
| [Examples](docs/examples.md) | Runnable programs under `examples/`; API sketches in docs |
| [Compatibility](docs/compatibility.md) | Public API boundary, release stability policy, deprecation rules |
| [Migration](MIGRATION.md) | Upgrade notes for major API changes |

## Testing

```bash
scripts/test.sh
```

See [Building](docs/building.md) for hub builds, integration tests, and sanitizers.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov — the inference engine beneath the harness
- [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann
- [GoogleTest](https://github.com/google/googletest) by Google

## License

MIT
