/**
 * @file demo_chat.cpp
 * @brief Interactive CLI example for chatting with a locally hosted model.
 *
 * Configuration is loaded from JSON via split `zoo::ModelConfig`,
 * `zoo::AgentConfig`, and `zoo::GenerationOptions` serialization helpers, and
 * optional example tools are registered to demonstrate the agent tool loop end
 * to end.
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
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

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
 * @brief Example-only configuration wrapper around Zoo-Keeper's split configs.
 */
struct DemoConfig {
    zoo::ModelConfig model;
    zoo::AgentConfig agent;
    zoo::GenerationOptions generation;
    std::optional<std::string> system_prompt;
    bool tools_enabled = true;
};

/// Parses the example-only config wrapper while delegating nested config blocks.
static void from_json(const nlohmann::json& j, DemoConfig& config) {
    if (!j.is_object()) {
        throw std::invalid_argument("Demo config must be a JSON object");
    }

    DemoConfig parsed;
    if (auto it = j.find("model"); it != j.end()) {
        parsed.model = it->get<zoo::ModelConfig>();
    } else {
        throw std::invalid_argument("Demo config must contain a model block");
    }

    if (auto it = j.find("agent"); it != j.end()) {
        parsed.agent = it->get<zoo::AgentConfig>();
    }
    if (auto it = j.find("generation"); it != j.end()) {
        parsed.generation = it->get<zoo::GenerationOptions>();
    }
    if (auto it = j.find("system_prompt"); it != j.end()) {
        parsed.system_prompt = it->get<std::string>();
    }
    if (auto it = j.find("tools"); it != j.end()) {
        parsed.tools_enabled = it->get<bool>();
    }

    config = std::move(parsed);
}

/// Loads the demo configuration from a JSON file on disk.
static DemoConfig load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    DemoConfig config = nlohmann::json::parse(file).get<DemoConfig>();
    if (auto validation = config.model.validate(); !validation) {
        throw std::runtime_error("Invalid model config: " + validation.error().to_string());
    }
    if (auto validation = config.agent.validate(); !validation) {
        throw std::runtime_error("Invalid agent config: " + validation.error().to_string());
    }
    if (auto validation = config.generation.validate(); !validation) {
        throw std::runtime_error("Invalid generation config: " + validation.error().to_string());
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
    std::cout << "  Model: " << dc.model.model_path << "\n";
    std::cout << "  Context: " << dc.model.context_size << " tokens\n";
    std::cout << "  Max tokens: "
              << (dc.generation.max_tokens == -1 ? "unlimited"
                                                 : std::to_string(dc.generation.max_tokens))
              << "\n";
    std::cout << "  Temperature: " << dc.generation.sampling.temperature << "\n";
    std::cout << "  GPU layers: " << dc.model.n_gpu_layers << "\n";
    std::cout << "  System: " << dc.system_prompt.value_or("(none)") << "\n";
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
              << "Config files contain nested model / agent / generation blocks\n"
              << "plus two example-only fields:\n"
              << "  system_prompt   Initial system prompt applied after Agent::create()\n"
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
    auto agent_result = zoo::Agent::create(dc.model, dc.agent, dc.generation);
    if (!agent_result) {
        std::cerr << "Error: " << agent_result.error().to_string() << "\n";
        return 1;
    }
    auto agent = std::move(*agent_result);
    if (dc.system_prompt) {
        agent->set_system_prompt(*dc.system_prompt);
    }

    // Register example tools
    if (dc.tools_enabled) {
        (void)agent->register_tool("add", "Add two integers", {"a", "b"}, calculate_add);
        (void)agent->register_tool("subtract", "Subtract two integers", {"a", "b"},
                                   calculate_subtract);
        (void)agent->register_tool("multiply", "Multiply two numbers", {"a", "b"},
                                   calculate_multiply);
        (void)agent->register_tool("get_time", "Get the current date and time", {},
                                   get_current_time);

        // With native tool calling, the chat template handles tool formatting
        // automatically — no need to inject tool descriptions into the system prompt.
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

        auto handle =
            agent->chat(line, {}, [](std::string_view token) { std::cout << token << std::flush; });

        // Poll for Ctrl+C during generation
        while (!handle.ready()) {
            if (g_interrupted.load(std::memory_order_acquire)) {
                agent->cancel(handle.id());
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        auto result = handle.await_result();

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
