# Building

## Quick Start

```bash
git clone https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
scripts/test.sh
```

llama.cpp is fetched automatically at CMake configure time; no submodule
initialization is required.

## Helper Scripts

Zoo-Keeper ships small wrapper scripts for the common local workflows:

- `scripts/build.sh` - configures and builds the project with the flags you pass
- `scripts/build-all.sh` - enables tests, integration tests, examples, and benchmarks
- `scripts/test.sh` - runs `ctest` against the existing `build/` tree
- `scripts/format.sh` - runs `clang-format` over the repo's C++ files
- `scripts/lint.sh` - performs a warning-free build

`scripts/build-all.sh` is the easiest way to exercise the broad release surface
before packaging, while `scripts/build.sh` stays useful for smaller builds or
custom flag combinations.

## CMake Presets

Presets require CMake 3.21+ (the raw build still works on CMake 3.18+).

Current presets in `CMakePresets.json` are:

- configure: `default`, `dev`, `integration`, `sanitizers`, `coverage`, `docs`
- build: `default`, `dev`, `docs`
- test: `default`, `unit`, `integration`

Examples:

```bash
cmake --preset default
cmake --build --preset default
ctest --preset default
```

```bash
cmake --preset docs
cmake --build --preset docs
```

The docs preset is intended for Doxygen generation only. The coverage preset
turns on coverage instrumentation and integration tests, while the integration
preset enables the live-model test label set.

## CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Metal acceleration (macOS) | ON (macOS only) |
| `ZOO_ENABLE_CUDA` | CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_INTEGRATION_TESTS` | Build Model/Agent integration tests | OFF |
| `ZOO_BUILD_EXAMPLES` | Build example applications | OFF |
| `ZOO_BUILD_BENCHMARKS` | Build the repo-local benchmark harness | OFF |
| `ZOO_BUILD_HUB` | Build optional GGUF inspection, HuggingFace, and model-store APIs | OFF |
| `ZOO_BUILD_DOCS` | Configure the `zoo_docs` Doxygen target | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan + UBSan | OFF |
| `ZOO_ENABLE_INSTALL` | Generate install and package metadata | ON top-level, OFF as subproject |
| `ZOO_ENABLE_LOGGING` | Emit runtime diagnostic logs to stderr | OFF |
| `ZOO_WARNINGS_AS_ERRORS` | Treat warnings in zoo-owned targets as errors | OFF |

## Platform-Specific Setup

### macOS

Metal acceleration is enabled by default on macOS:

```bash
scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON
```

To disable Metal and use CPU only:

```bash
scripts/build.sh -DZOO_ENABLE_METAL=OFF
```

### Linux

```bash
scripts/build.sh -DZOO_ENABLE_CUDA=ON -DZOO_BUILD_TESTS=ON
```

Requires the CUDA toolkit and `nvcc` on PATH.

### CPU Only

```bash
scripts/build.sh -DZOO_ENABLE_METAL=OFF -DZOO_ENABLE_CUDA=OFF
```

## Compiler Requirements

| Platform | Compiler | Minimum Version |
|----------|----------|-----------------|
| macOS | Clang | 16.0+ |
| Linux | GCC | 13.0+ |
| Linux | Clang | 18.0+ |

C++23 support is required (`std::expected`, defaulted comparison operators).

## Dependencies

| Dependency | Version | Integration | Notes |
|------------|---------|-------------|-------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | pinned by `ZOO_LLAMA_TAG` | CMake FetchContent (default) or parent-provided targets | Core inference engine |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11+ | CMake FetchContent | JSON parsing |
| [GoogleTest](https://github.com/google/googletest) | 1.14+ | CMake FetchContent | Tests only |
| [Doxygen](https://www.doxygen.nl/) | host tool | System package | Required only when `ZOO_BUILD_DOCS=ON` |
| [Graphviz](https://graphviz.org/) | host tool | System package | Optional for call graphs and include diagrams |

`nlohmann/json` and `llama.cpp` are both downloaded automatically during CMake
configuration. To pin a different llama.cpp release tag, set `ZOO_LLAMA_TAG` in
`cmake/ZooKeeperOptions.cmake` (or override via `-DZOO_LLAMA_TAG=...`). If your
parent project already defines `llama` and `llama-common` CMake targets,
Zoo-Keeper reuses them and skips its own fetch. Installed-package consumers
still need a discoverable `nlohmann_json` package because the public headers
include `<nlohmann/json.hpp>`.

## Running Tests

```bash
# All tests
scripts/test.sh

# Specific test suite
scripts/test.sh -R ToolRegistryTest

# Verbose output
scripts/test.sh --verbose
```

Hub-layer unit tests (`tests/unit/test_hub.cpp`) are only compiled when the hub
layer is enabled. To include them, configure with `-DZOO_BUILD_HUB=ON`:

```bash
scripts/build.sh -DZOO_BUILD_TESTS=ON -DZOO_BUILD_HUB=ON
scripts/test.sh -R "HuggingFace|ModelStore|AutoConfig|GgufInspector|HubPath"
```

## Integration Tests

The integration target exercises the concrete `Model` and `Agent` layers. Two
failure-path tests run using vendored fixtures. Optional live smoke tests run
when a real GGUF path is provided.

```bash
scripts/build.sh -DZOO_BUILD_INTEGRATION_TESTS=ON \
    -DZOO_INTEGRATION_MODEL=/absolute/path/to/model.gguf
scripts/test.sh -L integration
```

## Benchmarks

Benchmarks are built with the repo-local benchmark target and the
`ZOO_BUILD_BENCHMARKS=ON` flag:

```bash
scripts/build-all.sh
build/benchmarks/zoo_benchmarks /absolute/path/to/model.gguf
```

The benchmark harness is meant for live GGUF-backed runs, not mocked unit tests.

## Sanitizers

```bash
scripts/build.sh -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON
scripts/test.sh
```

Enables AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).

## Coverage

```bash
scripts/build.sh -DZOO_ENABLE_COVERAGE=ON -DZOO_BUILD_TESTS=ON
scripts/test.sh
```

GitHub Actions also captures an `lcov` report and uploads it as a workflow
artifact. Upload to Codecov runs only when `CODECOV_TOKEN` is available in CI.

## API Reference

Install `doxygen` locally before enabling docs. `graphviz` is optional, but if
`dot` is available Doxygen will emit diagrams in the generated HTML.

```bash
cmake --preset docs
cmake --build --preset docs
```

The generated site is written to `build/docs/doxygen/html/index.html`, with XML
output in `build/docs/doxygen/xml`.

GitHub Actions also builds the Doxygen site on every push and pull request,
uploads the generated output as a workflow artifact, and deploys the latest
`main` build to GitHub Pages.

## Using Zoo-Keeper in Your Project

### CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(
    zoo-keeper
    GIT_REPOSITORY https://github.com/crybo-rybo/zoo-keeper.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(zoo-keeper)

target_link_libraries(your_target PRIVATE ZooKeeper::zoo)
```

Zoo-Keeper fetches llama.cpp automatically. If your parent project already
builds llama.cpp and exposes both `llama` and `llama-common` CMake targets,
Zoo-Keeper reuses them and skips its own fetch.

### Git Submodule

```bash
git submodule add https://github.com/crybo-rybo/zoo-keeper.git extern/zoo-keeper
```

```cmake
add_subdirectory(extern/zoo-keeper)
target_link_libraries(your_target PRIVATE ZooKeeper::zoo)
```

### Installed CMake Package

```cmake
find_package(ZooKeeper CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ZooKeeper::zoo)
```

Make sure `nlohmann_json` is also installed and discoverable via
`CMAKE_PREFIX_PATH` or `nlohmann_json_DIR`. `ZooKeeperConfig.cmake` resolves it
transitively with `find_dependency(nlohmann_json CONFIG)`, so consumers do not
need a separate `target_link_libraries(... nlohmann_json::nlohmann_json)` line.

Example:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH="/opt/zoo-keeper;/opt/nlohmann_json"
```

### pkg-config

```bash
pkg-config --cflags --libs zoo-keeper
```

The `zoo-keeper.pc` file declares a dependency on `nlohmann_json`. If
`pkg-config` cannot resolve it, install the `nlohmann_json` pkg-config package
and make sure its `.pc` directory is on `PKG_CONFIG_PATH` alongside
Zoo-Keeper's.

## Running the Demo

```bash
scripts/build.sh -DZOO_BUILD_EXAMPLES=ON
./build/examples/demo_chat examples/config.example.json
```

Additional example executables are built alongside `demo_chat`:

- `demo_extract` - structured extraction examples for stateful, stateless, and streaming flows
- `model_generate` -- standalone `zoo::core::Model` usage
- `error_handling` -- practical error reporting patterns
- `stream_cancel` -- streaming output with cooperative cancellation
- `manual_tool_schema` -- manual-schema registration through `Agent::register_tool(...)`

## See Also

- [Getting Started](getting-started.md) -- first agent walkthrough
- [Configuration](configuration.md) -- config struct fields
- [Maintainer CMake Packaging Notes](maintainer-cmake-packaging.md) -- build-tree vs install-tree package config internals
