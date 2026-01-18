#pragma once

/**
 * @file zoo.hpp
 * @brief Main convenience header for Zoo-Keeper Agent Engine
 *
 * Include this single header to get access to all public Zoo-Keeper APIs.
 *
 * Zoo-Keeper is a header-only C++17 library built on top of llama.cpp
 * that functions as a complete Agent Engine for local LLM inference.
 *
 * Quick Start:
 * @code
 * #include <zoo/zoo.hpp>
 *
 * int main() {
 *     zoo::Config config;
 *     config.model_path = "path/to/model.gguf";
 *     config.context_size = 8192;
 *     config.max_tokens = 512;
 *
 *     auto agent = zoo::Agent::create(config);
 *     if (!agent) {
 *         std::cerr << "Error: " << agent.error().to_string() << std::endl;
 *         return 1;
 *     }
 *
 *     auto future = agent->chat(zoo::Message::user("Hello!"));
 *     auto response = future.get();
 *
 *     if (response) {
 *         std::cout << "Assistant: " << response->text << std::endl;
 *     } else {
 *         std::cerr << "Error: " << response.error().to_string() << std::endl;
 *     }
 *
 *     return 0;
 * }
 * @endcode
 *
 * Key Components:
 * - zoo::Agent: Main entry point for inference
 * - zoo::Config: Configuration for model, sampling, and templates
 * - zoo::Message: Conversation messages (User, Assistant, System, Tool)
 * - zoo::Response: Inference results with text, metrics, and token usage
 * - zoo::Error: Structured error handling
 *
 * Thread Safety:
 * - Agent manages its own inference thread
 * - chat() is thread-safe and returns std::future
 * - Callbacks execute on the inference thread
 *
 * Design Principles:
 * - Header-only for easy integration
 * - std::expected for composable error handling
 * - Value semantics for predictable ownership
 * - Dependency injection for testability
 */

// Core types
#include "types.hpp"

// Public API
#include "agent.hpp"

// Backend interface (for custom implementations and testing)
#include "backend/interface.hpp"

// Engine components (optional, for advanced usage)
#include "engine/history_manager.hpp"
#include "engine/template_engine.hpp"
#include "engine/request_queue.hpp"
#include "engine/agentic_loop.hpp"

/**
 * @namespace zoo
 * @brief Main namespace for Zoo-Keeper library
 *
 * All Zoo-Keeper APIs are in the `zoo` namespace.
 * Internal implementation details are in nested namespaces:
 * - zoo::backend - Backend interface and implementations
 * - zoo::engine - Internal engine components
 */

/**
 * @mainpage Zoo-Keeper Agent Engine
 *
 * @section intro Introduction
 *
 * Zoo-Keeper is a modern C++17 Agent Engine for local LLM inference,
 * built on top of llama.cpp. It provides a high-level, type-safe API
 * for building agentic AI systems.
 *
 * @section features Features
 *
 * - **Simple API**: Single entry point via zoo::Agent class
 * - **Async Inference**: Non-blocking chat() with std::future
 * - **Streaming**: Token-by-token callbacks for real-time output
 * - **History Management**: Automatic conversation tracking
 * - **Template Support**: Llama3, ChatML, and custom formats
 * - **Type Safety**: std::expected for composable error handling
 * - **Testable**: Dependency injection with MockBackend
 * - **Header-Only**: Easy integration into any project
 *
 * @section requirements Requirements
 *
 * - C++17 or later
 * - CMake 3.18+
 * - llama.cpp (included as submodule)
 * - tl::expected (included via FetchContent)
 *
 * @section building Building
 *
 * @code{.sh}
 * # Configure
 * cmake -B build -DZOO_BUILD_EXAMPLES=ON
 *
 * # Build
 * cmake --build build
 *
 * # Run examples
 * ./build/examples/simple_chat
 * @endcode
 *
 * @section example Example
 *
 * See the Quick Start code in zoo.hpp for a minimal example.
 * For more examples, see the examples/ directory.
 *
 * @section architecture Architecture
 *
 * Zoo-Keeper uses a three-layer architecture:
 *
 * 1. **Public API Layer**: zoo::Agent class
 * 2. **Engine Layer**: Request queue, history, templates, agentic loop
 * 3. **Backend Layer**: llama.cpp abstraction (IBackend interface)
 *
 * The Agent spawns an inference thread that processes requests asynchronously.
 * Calling threads submit requests via chat() and receive std::future.
 *
 * @section threading Threading Model
 *
 * - **Calling Thread**: Submits chat() requests, receives std::future
 * - **Inference Thread**: Processes queue, executes backend->generate()
 * - **Callbacks**: Execute on inference thread (user handles sync)
 *
 * @section license License
 *
 * See LICENSE file for details.
 */
