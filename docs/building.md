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

## GPU Out-of-Memory Handling

### Problem

On Apple Silicon (and CUDA), when a model's KV cache or activation memory exceeds available
GPU/unified memory during inference, llama.cpp's Metal (or CUDA) backend calls `ggml_abort()`,
which terminates the host process with SIGABRT. Zoo-keeper provides two layers of defense.

### Prevention (Recommended)

Zoo-keeper performs a pre-load file-size check during `Agent::create()`. If the model file
appears to exceed available system memory, initialization returns `ErrorCode::BackendInitFailed`
with an actionable message before attempting the slow model load.

For tighter memory budgets, reduce KV cache memory pressure by using quantized KV cache types:

```cpp
zoo::Config config;
config.model_path = "model.gguf";
config.context_size = 8192;
// KV cache quantization — reduces memory at a small quality cost
// config.kv_cache_type_k = 8;  // Q8_0: roughly half the default memory
// config.kv_cache_type_v = 8;
auto agent = zoo::Agent::create(config);
```

Additional mitigations:

- Use a smaller quantization (Q4_K_M, Q5_K_M) instead of F16/BF16 weights.
- Reduce `context_size` — KV cache scales linearly with context.
- Reduce `n_gpu_layers` to offload fewer layers to GPU.

### Recovery (Best-Effort)

Zoo-keeper installs a `SIGABRT` handler on the inference thread using `sigsetjmp`/`siglongjmp`.
If `ggml_abort()` fires during inference, the handler intercepts the abort signal and returns
`ErrorCode::GpuOutOfMemory` instead of crashing the process.

**Important caveats:**

- This recovery is **not fully safe**. C++ destructors may not run for objects on the `longjmp`
  path, and the llama.cpp context is in an indeterminate state.
- After receiving `GpuOutOfMemory`, stop the agent and create a new one rather than continuing
  to use the same instance.
- Signal handler interaction: if your application installs its own `SIGABRT` handler, it will
  be temporarily replaced during `generate()` calls and restored afterward.

```cpp
auto handle = agent->chat(Message::user("..."));
auto response = handle.future.get();
if (!response && response.error().code == zoo::ErrorCode::GpuOutOfMemory) {
    // GPU OOM — context is corrupt. Stop this agent and create a new one.
    agent->stop();
    agent.reset();
    // Recreate with smaller context_size or fewer gpu_layers
}
```

## See Also

- [Getting Started](getting-started.md) -- first agent walkthrough
- [Configuration](configuration.md) -- all Config struct fields
