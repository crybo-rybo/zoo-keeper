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
| `ZOO_BUILD_EXAMPLES` | Build example applications | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_MCP` | MCP client support | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan/TSan/UBSan | OFF |

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

### Linux / Windows (CUDA)

```bash
cmake -B build -DZOO_ENABLE_CUDA=ON -DZOO_BUILD_TESTS=ON
cmake --build build
```

Requires CUDA toolkit installed and `nvcc` on PATH.

### MCP Client Support

```bash
cmake -B build -DZOO_ENABLE_MCP=ON -DZOO_BUILD_TESTS=ON
cmake --build build
```

### CPU Only

```bash
cmake -B build -DZOO_ENABLE_METAL=OFF -DZOO_ENABLE_CUDA=OFF
cmake --build build
```

## Compiler Requirements

| Platform | Compiler | Minimum Version |
|----------|----------|-----------------|
| macOS | Clang | 13.0+ |
| Linux | GCC | 11.0+ |
| Linux | Clang | 13.0+ |
| Windows | MSVC | 2019 (19.20+) |

## Dependencies

| Dependency | Version | Integration | Notes |
|------------|---------|-------------|-------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | pinned | Git submodule | Core inference engine |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11+ | CMake FetchContent | JSON parsing |
| [tl::expected](https://github.com/TartanLlama/expected) | latest | CMake FetchContent | Error handling |
| [GoogleTest](https://github.com/google/googletest) | 1.14+ | CMake FetchContent | Tests only |
| SQLite3 | system | System library | Context database |

All FetchContent dependencies are downloaded automatically during CMake configuration.

## Running Tests

```bash
# All tests
ctest --test-dir build

# Specific test suite
ctest --test-dir build -R HistoryManagerTest

# Verbose output
ctest --test-dir build --output-on-failure --verbose
```

## Sanitizers

```bash
cmake -B build -DZOO_ENABLE_SANITIZERS=ON -DZOO_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

Enables AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer (UBSan).

## Coverage

```bash
cmake -B build -DZOO_ENABLE_COVERAGE=ON -DZOO_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build
```

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

target_link_libraries(your_target PRIVATE zoo::zoo)
```

### Git Submodule

```bash
git submodule add https://github.com/crybo-rybo/zoo-keeper.git extern/zoo-keeper
```

```cmake
add_subdirectory(extern/zoo-keeper)
target_link_libraries(your_target PRIVATE zoo::zoo)
```

## Running the Demo

```bash
cmake -B build -DZOO_BUILD_EXAMPLES=ON
cmake --build build
./build/examples/demo_chat models/your-model.gguf
```

The demo supports options like `--temperature`, `--max-tokens`, `--template`, and `--system`. Run with `--help` for details.

## See Also

- [Getting Started](getting-started.md) -- first agent walkthrough
- [Configuration](configuration.md) -- all Config struct fields
