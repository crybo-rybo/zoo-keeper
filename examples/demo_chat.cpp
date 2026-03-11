/**
 * @file demo_chat.cpp
 * @brief Interactive CLI example for chatting with a locally hosted model.
 *
 * Configuration is loaded from JSON via `zoo::Config` serialization helpers,
 * and optional example tools are registered to demonstrate the agent tool loop
 * end to end.
 *
 * Usage:
 *   ./demo_chat <config.json>
 *   ./demo_chat --help
 *
 * See `examples/config.example.json` for a complete sample config.
 */

#include <zoo/core/json.hpp>
#include <zoo/zoo.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

static std::atomic<bool> g_interrupted{false};

// ============================================================================
// Example Tools
// ============================================================================

static int calculate_add(int a, int b) {
    return a + b;
}
static int calculate_subtract(int a, int b) {
    return a - b;
}
static double calculate_multiply(double a, double b) {
    return a * b;
}

static std::string get_current_time() {
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

// ============================================================================
// Config loading
// ============================================================================

/**
 * @brief Example-only configuration wrapper that extends `zoo::Config`.
 */
struct DemoConfig {
    zoo::Config zoo;
    bool tools_enabled = true;
};

/// Parses the example-only config wrapper while delegating core fields to `zoo::Config`.
static void from_json(const nlohmann::json& j, DemoConfig& config) {
    if (!j.is_object()) {
        throw std::invalid_argument("Demo config must be a JSON object");
    }

    DemoConfig parsed;
    nlohmann::json zoo_config_json = j;
    if (auto it = j.find("tools"); it != j.end()) {
        parsed.tools_enabled = it->get<bool>();
        zoo_config_json.erase("tools");
    }

    parsed.zoo = zoo_config_json.get<zoo::Config>();
    config = std::move(parsed);
}

/// Loads the demo configuration from a JSON file on disk.
static DemoConfig load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    DemoConfig config = nlohmann::json::parse(file).get<DemoConfig>();
    if (auto validation = config.zoo.validate(); !validation) {
        throw std::runtime_error("Invalid Zoo config: " + validation.error().to_string());
    }
    return config;
}

// ============================================================================
// Display helpers
// ============================================================================

/// Prints a consistent separator line for the demo CLI.
static void print_separator() {
    std::cout << std::string(60, '-') << "\n";
}

/// Prints latency and token usage metrics for the last response.
static void print_metrics(const zoo::Metrics& metrics, const zoo::TokenUsage& usage) {
    std::cout << "\n";
    print_separator();
    std::cout << "  Tokens: " << usage.prompt_tokens << " prompt + " << usage.completion_tokens
              << " completion = " << usage.total_tokens << " total\n";
    std::cout << "  Latency: " << metrics.latency_ms.count() << " ms\n";
    std::cout << "  TTFT: " << metrics.time_to_first_token_ms.count() << " ms\n";
    std::cout << "  Speed: " << std::fixed << std::setprecision(1) << metrics.tokens_per_second
              << " tok/s\n";
    print_separator();
}

/// Prints the startup banner and the active runtime configuration.
static void print_welcome(const DemoConfig& dc) {
    std::cout << "\n";
    print_separator();
    std::cout << "Zoo-Keeper Demo Chat\n";
    print_separator();
    std::cout << "  Model: " << dc.zoo.model_path << "\n";
    std::cout << "  Context: " << dc.zoo.context_size << " tokens\n";
    std::cout << "  Max tokens: "
              << (dc.zoo.max_tokens == -1 ? "unlimited" : std::to_string(dc.zoo.max_tokens))
              << "\n";
    std::cout << "  Temperature: " << dc.zoo.sampling.temperature << "\n";
    std::cout << "  GPU layers: " << dc.zoo.n_gpu_layers << "\n";
    std::cout << "  System: " << dc.zoo.system_prompt.value_or("(none)") << "\n";
    std::cout << "  Tools: " << (dc.tools_enabled ? "enabled" : "disabled") << "\n";
    print_separator();
    std::cout << "\nType a message and press Enter. Commands: /quit /clear /help\n\n";
}

/// Prints command-line usage information for the example executable.
static void print_usage(const char* prog) {
    std::cout << "Zoo-Keeper Demo Chat\n\n"
              << "Usage:\n"
              << "  " << prog << " <config.json>\n"
              << "  " << prog << " --help\n\n"
              << "Config files use the documented zoo::Config JSON shape plus one\n"
              << "example-only boolean field:\n"
              << "  tools           Enable the bundled example tools (default: true)\n\n"
              << "See examples/config.example.json and docs/configuration.md for the\n"
              << "full config contract.\n";
}

// ============================================================================
// Main
// ============================================================================

/// Records Ctrl-C so the main loop can exit cleanly.
static void signal_handler(int) {
    g_interrupted.store(true, std::memory_order_release);
}

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(argv[0]);
        return argc == 2 ? 0 : 1;
    }

    // Load config
    DemoConfig dc;
    try {
        dc = load_config(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    std::signal(SIGINT, signal_handler);

    // Create agent
    std::cout << "Loading model...\n";
    auto agent_result = zoo::Agent::create(dc.zoo);
    if (!agent_result) {
        std::cerr << "Error: " << agent_result.error().to_string() << "\n";
        return 1;
    }
    auto agent = std::move(*agent_result);

    // Register example tools
    if (dc.tools_enabled) {
        (void)agent->register_tool("add", "Add two integers", {"a", "b"}, calculate_add);
        (void)agent->register_tool("subtract", "Subtract two integers", {"a", "b"},
                                   calculate_subtract);
        (void)agent->register_tool("multiply", "Multiply two numbers", {"a", "b"},
                                   calculate_multiply);
        (void)agent->register_tool("get_time", "Get the current date and time", {},
                                   get_current_time);

        auto base_prompt = dc.zoo.system_prompt.value_or("You are a helpful AI assistant.");
        agent->set_system_prompt(agent->build_tool_system_prompt(base_prompt));
    }

    print_welcome(dc);

    // Chat loop
    std::string line;
    while (!g_interrupted.load(std::memory_order_acquire)) {
        std::cout << "You: ";
        std::cout.flush();

        if (!std::getline(std::cin, line))
            break;

        // Trim
        auto start = line.find_first_not_of(" \t\n\r");
        if (start == std::string::npos)
            continue;
        line = line.substr(start, line.find_last_not_of(" \t\n\r") - start + 1);

        // Commands
        if (line == "/quit" || line == "/exit")
            break;
        if (line == "/clear") {
            agent->clear_history();
            std::cout << "History cleared.\n\n";
            continue;
        }
        if (line == "/help") {
            std::cout << "  /quit, /exit  Exit\n"
                      << "  /clear        Clear conversation history\n"
                      << "  /help         Show commands\n"
                      << "  Ctrl+C        Stop generation\n\n";
            continue;
        }

        std::cout << "\nAssistant: ";
        std::cout.flush();
        g_interrupted.store(false, std::memory_order_release);

        auto handle = agent->chat(zoo::Message::user(line),
                                  [](std::string_view token) { std::cout << token << std::flush; });

        // Poll for Ctrl+C during generation
        while (handle.future.wait_for(std::chrono::milliseconds(100)) ==
               std::future_status::timeout) {
            if (g_interrupted.load(std::memory_order_acquire)) {
                agent->cancel(handle.id);
                break;
            }
        }

        auto result = handle.future.get();

        if (!result) {
            if (result.error().code == zoo::ErrorCode::RequestCancelled) {
                std::cout << "\n[cancelled]\n";
            } else {
                std::cerr << "\nError: " << result.error().to_string() << "\n";
            }
            continue;
        }

        std::cout << "\n";
        print_metrics(result->metrics, result->usage);
    }

    std::cout << "\nGoodbye!\n";
    return 0;
}
