# Zoo-Keeper Agent Engine

[![Tests](https://img.shields.io/badge/tests-230%2F230%20passing-success)]() [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)]() [![License](https://img.shields.io/badge/license-MIT-green)]()

A modern C++17 header-only Agent Engine for local LLM inference, built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp). Zoo-Keeper provides a complete agentic AI framework with automated conversation history management, type-safe tool calling, error recovery, and asynchronous inference with streaming support.

**Status:** Phase 2 (Tool System) Complete

## Features

- **Asynchronous Inference**: Non-blocking chat API with `std::future` support
- **Tool Calling System**: Type-safe tool registration with automatic JSON schema generation
- **Agentic Loop**: Automatic tool call detection, execution, and result injection
- **Error Recovery**: Argument validation with configurable retry logic for failed tool calls
- **RAG (Ephemeral)**: Per-request retrieval context injection without history pollution
- **Conversation Management**: Automatic history tracking with tool call history
- **Multiple Prompt Templates**: Built-in support for Llama3, ChatML, and custom formats
- **Streaming Support**: Token-by-token callbacks for real-time output
- **Type-Safe Errors**: Modern `std::expected` error handling without exceptions
- **Header-Only**: Easy integration with CMake FetchContent
- **Extensible Backend**: Abstract interface for testing and alternative implementations
- **Thread-Safe**: Producer-consumer pattern with dedicated inference thread
- **Hardware Acceleration**: Metal (macOS) and CUDA support via llama.cpp

## Quick Start

```cpp
#include <zoo/zoo.hpp>
#include <iostream>

int main() {
    // Configure the agent
    zoo::Config config;
    config.model_path = "models/llama-3-8b.gguf";
    config.context_size = 8192;
    config.max_tokens = 512;
    config.prompt_template = zoo::PromptTemplate::Llama3;

    // Create agent
    auto agent_result = zoo::Agent::create(config);
    if (!agent_result) {
        std::cerr << "Error: " << agent_result.error().to_string() << std::endl;
        return 1;
    }

    auto agent = std::move(*agent_result);  // std::unique_ptr<Agent>

    // Set system prompt
    agent->set_system_prompt("You are a helpful AI assistant.");

    // Send a message
    auto future = agent->chat(zoo::Message::user("Hello!"));

    // Wait for response
    auto response = future.get();
    if (response) {
        std::cout << "Assistant: " << response->text << std::endl;
        std::cout << "Tokens used: " << response->usage.total_tokens << std::endl;
    } else {
        std::cerr << "Error: " << response.error().to_string() << std::endl;
    }

    return 0;
}
```

## Building

### Prerequisites

- **C++17 compiler**: GCC 11+, Clang 13+, or MSVC 2019+
- **CMake 3.18+**
- **Git** (for submodules)

### Build Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/zoo-keeper.git
cd zoo-keeper

# Initialize submodules
git submodule update --init --recursive

# Configure with CMake
cmake -B build \
  -DZOO_BUILD_TESTS=ON \
  -DZOO_BUILD_EXAMPLES=ON \
  -DZOO_ENABLE_METAL=ON  # macOS only

# Build
cmake --build build -j4

# Run tests
ctest --test-dir build --output-on-failure

# Run demo
build/examples/demo_chat models/your-model.gguf
```

### CMake Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `ZOO_ENABLE_METAL` | Metal acceleration (macOS) | ON (macOS) |
| `ZOO_ENABLE_CUDA` | CUDA acceleration | OFF |
| `ZOO_BUILD_TESTS` | Build test suite | OFF |
| `ZOO_BUILD_EXAMPLES` | Build examples | OFF |
| `ZOO_ENABLE_COVERAGE` | Coverage instrumentation | OFF |
| `ZOO_ENABLE_SANITIZERS` | ASan/TSan/UBSan | OFF |

## Usage Examples

### Streaming Output

```cpp
auto future = agent->chat(
    zoo::Message::user("Write a haiku about AI"),
    [](std::string_view token) {
        std::cout << token << std::flush;  // Print tokens as they arrive
    }
);

auto response = future.get();
std::cout << std::endl;
```

### Multi-Turn Conversation

```cpp
// History is automatically managed
agent->chat(zoo::Message::user("My name is Alice")).get();
agent->chat(zoo::Message::user("What's my name?")).get();  // Will remember "Alice"
```

### Tool Registration and Calling

```cpp
// Define a tool function
int add(int a, int b) {
    return a + b;
}

// Register the tool with parameter names
agent->register_tool("add", "Adds two numbers together", {"a", "b"}, add);

// The model can now call this tool during inference
auto response = agent->chat(zoo::Message::user("What is 42 + 58?")).get();

if (response) {
    std::cout << "Response: " << response->text << std::endl;

    // Check if any tools were called during inference
    if (!response->tool_calls.empty()) {
        std::cout << "Tool calls made: " << response->tool_calls.size() << std::endl;
        for (const auto& tool_msg : response->tool_calls) {
            std::cout << "  " << tool_msg.content << std::endl;
        }
    }
}
```

### Error Handling

```cpp
auto response_result = agent->chat(zoo::Message::user("Hello")).get();

if (!response_result) {
    // Handle error
    zoo::Error error = response_result.error();
    switch (error.code) {
        case zoo::ErrorCode::ContextWindowExceeded:
            std::cerr << "Context full! Consider clearing history." << std::endl;
            agent->clear_history();
            break;
        case zoo::ErrorCode::InferenceFailed:
            std::cerr << "Inference failed: " << error.message << std::endl;
            break;
        default:
            std::cerr << error.to_string() << std::endl;
    }
} else {
    // Use response
    std::cout << response_result->text << std::endl;
}
```

### Custom Configuration

```cpp
zoo::Config config;
config.model_path = "models/custom-model.gguf";
config.context_size = 4096;
config.max_tokens = 256;

// Sampling parameters
config.sampling.temperature = 0.8f;
config.sampling.top_p = 0.95f;
config.sampling.top_k = 40;
config.sampling.repeat_penalty = 1.1f;

// Stop sequences
config.stop_sequences = {"\n\n", "User:"};

// Custom template
config.prompt_template = zoo::PromptTemplate::Custom;
config.custom_template = "{{role}}: {{content}}\n";

auto agent_result = zoo::Agent::create(config);
if (agent_result) {
    auto agent = std::move(*agent_result);  // std::unique_ptr<Agent>
}
```

## Architecture

Zoo-Keeper uses a three-layer architecture:

1. **Public API Layer**: Simple entry point via `zoo::Agent` class
2. **Engine Layer**: Core components (Request Queue, History Manager, Tool Registry, Tool Call Parser, Error Recovery, Agentic Loop)
3. **Backend Layer**: Abstracted llama.cpp interface with mock support for testing

### Threading Model

- **Calling Thread**: Submits `chat()` requests, receives `std::future<Response>`
- **Inference Thread**: Processes queue, executes backend, manages history

All callbacks (`on_token`) execute on the inference thread. Consumer is responsible for thread synchronization.

## Testing

Zoo-Keeper includes comprehensive unit tests using GoogleTest:

```bash
# Build tests
cmake -B build -DZOO_BUILD_TESTS=ON
cmake --build build

# Run all tests (152 tests)
ctest --test-dir build

# Run specific test suite
ctest --test-dir build -R HistoryManagerTest

# Run with verbose output
ctest --test-dir build --output-on-failure --verbose
```

Test coverage includes:
- Core types and validation
- Thread-safe request queue
- Conversation history management
- Tool registry and schema generation
- Tool call parsing and validation
- Error recovery with retry logic
- Agentic loop with tool execution
- Template rendering (Llama3, ChatML, Custom)
- Mock backend integration
- Full pipeline simulation

## Documentation

- **[Product Requirements](zoo-keeper-prd.md)**: Goals, features, and success metrics
- **[Technical Requirements](zoo-keeper-trd.md)**: Architecture, state machines, and test plan
- **[Developer Guide](CLAUDE.md)**: Build commands, architecture details, and contribution guidelines
- **API Documentation**: Generate with Doxygen (see `include/zoo/zoo.hpp` for mainpage)

## Implementation Status & Roadmap

### Phase 1: MVP (Complete)
- ✅ Asynchronous inference with `std::future`
- ✅ Request queue with producer-consumer pattern
- ✅ Conversation history management
- ✅ Template support (Llama3, ChatML, Custom)
- ✅ Streaming callbacks
- ✅ Type-safe error handling with `std::expected`
- ✅ Mock backend for testing
- ✅ Comprehensive unit test suite (115 tests)

### Phase 2: Tool System (Complete)
- ✅ Type-safe tool registration with template-based schema generation
- ✅ Tool call detection and parsing from model output
- ✅ Automatic tool execution with JSON argument handling
- ✅ Agentic loop with tool result injection
- ✅ Argument validation against JSON schema
- ✅ Error recovery with configurable retry logic
- ✅ Tool call history tracking in Response
- ✅ Comprehensive tool system tests (37 additional tests)

### Current Limitations
- **Token Counting**: Character-based estimation (4 chars ≈ 1 token)
- **Context Pruning**: Not implemented (will overflow on very long conversations)
- **KV Cache Reuse**: Cold start each turn (performance optimization pending)
- **RAG Retrieval Quality**: Baseline lexical retriever; vector ANN backend not yet integrated

### Phase 3 (Planned)
- Actual backend tokenization with caching
- Context pruning with FIFO strategy
- KV cache optimization for multi-turn efficiency
- Vector-backed RAG retrieval (SQLite + ANN)
- Advanced sampling strategies
- Performance optimizations

## Acknowledgments

- Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- Uses [tl::expected](https://github.com/TartanLlama/expected) by Sy Brand
- Uses [nlohmann/json](https://github.com/nlohmann/json) by Niels Lohmann

## Support

- **Issues**: https://github.com/yourusername/zoo-keeper/issues
- **Discussions**: https://github.com/yourusername/zoo-keeper/discussions

---

*Built with Claude Code*
