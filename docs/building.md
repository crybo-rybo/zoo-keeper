# Building

## Quick Start

```bash
git clone --recurse-submodules https://github.com/crybo-rybo/zoo-keeper.git
cd zoo-keeper
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
ctest --test-dir build
```

## CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Metal acceleration (macOS) | ON (macOS only) |
| `ZOO_ENABLE_CUDA` | CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_INTEGRATION_TESTS` | Build Model/Agent integration tests | OFF |
| `ZOO_BUILD_EXAMPLES` | Build example applications | OFF |
| `ZOO_BUILD_DOCS` | Configure the `zoo_docs` Doxygen target | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan/TSan/UBSan | OFF |
| `ZOO_ENABLE_LOGGING` | Emit runtime diagnostic logs to stderr | OFF |

## Platform-Specific Setup

### macOS (Metal)

Metal acceleration is enabled by default on macOS:

```bash
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_EXAMPLES=ON
cmake --build build
```

To disable Metal and use CPU only:

```bash
cmake -B build -DZOO_ENABLE_METAL=OFF
```

### Linux (CUDA)

```bash
cmake -B build -DZOO_ENABLE_CUDA=ON -DZOO_BUILD_TESTS=ON
cmake --build build
```

Requires CUDA toolkit installed and `nvcc` on PATH.

### CPU Only

```bash
cmake -B build -DZOO_ENABLE_METAL=OFF -DZOO_ENABLE_CUDA=OFF
cmake --build build
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
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | pinned | Git submodule | Core inference engine |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11+ | CMake FetchContent | JSON parsing |
| [GoogleTest](https://github.com/google/googletest) | 1.14+ | CMake FetchContent | Tests only |
| [Doxygen](https://www.doxygen.nl/) | host tool | System package | Required only when `ZOO_BUILD_DOCS=ON` |
| [Graphviz](https://graphviz.org/) | host tool | System package | Optional for call graphs and include diagrams |

All FetchContent dependencies are downloaded automatically during CMake configuration.

## Running Tests

```bash
# All tests
ctest --test-dir build

# Specific test suite
ctest --test-dir build -R ModelTest

# Verbose output
ctest --test-dir build --output-on-failure --verbose
```

## Integration Tests

The integration target exercises the concrete `Model` and `Agent` layers. Two failure-path tests run using vendored fixtures. Optional live smoke tests run when a real GGUF path is provided.

```bash
cmake -B build -DZOO_BUILD_TESTS=ON -DZOO_BUILD_INTEGRATION_TESTS=ON \
    -DZOO_INTEGRATION_MODEL=/absolute/path/to/model.gguf
cmake --build build
ctest --test-dir build --output-on-failure -L integration
```

## Sanitizers

```bash
cmake -B build -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Enables AddressSanitizer (ASan) and UndefinedBehaviorSanitizer (UBSan).

## Coverage

```bash
cmake -B build -DZOO_ENABLE_COVERAGE=ON -DZOO_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

GitHub Actions also captures an `lcov` report and uploads it as a workflow artifact and to Codecov on every `main` push and pull request.

## API Reference

Install `doxygen` locally before enabling docs. `graphviz` is optional, but if `dot` is available Doxygen will emit diagrams in the generated HTML.

```bash
cmake -B build -DZOO_BUILD_DOCS=ON
cmake --build build --target zoo_docs
```

The generated site is written to `build/docs/doxygen/html/index.html`, with XML output in `build/docs/doxygen/xml`.

GitHub Actions also builds the Doxygen site on every push and pull request, uploads the generated output as a workflow artifact, and deploys the latest `main` build to GitHub Pages.

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

target_link_libraries(your_target PRIVATE zoo_core)
```

### Git Submodule

```bash
git submodule add https://github.com/crybo-rybo/zoo-keeper.git extern/zoo-keeper
```

```cmake
add_subdirectory(extern/zoo-keeper)
target_link_libraries(your_target PRIVATE zoo_core)
```

### Installed CMake Package

```cmake
find_package(ZooKeeper CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE ZooKeeper::zoo_core)
```

### pkg-config

```bash
pkg-config --cflags --libs zoo-keeper
```

## Running the Demo

```bash
cmake -B build -DZOO_BUILD_EXAMPLES=ON
cmake --build build
./build/examples/demo_chat models/your-model.gguf
```

Additional example executables are built alongside `demo_chat`:

- `model_generate` -- standalone `zoo::core::Model` usage
- `error_handling` -- practical error reporting patterns
- `stream_cancel` -- streaming output with cooperative cancellation
- `custom_tool_schema` -- manual JSON schema registration for nested tool arguments

## See Also

- [Getting Started](getting-started.md) -- first agent walkthrough
- [Configuration](configuration.md) -- all Config struct fields
