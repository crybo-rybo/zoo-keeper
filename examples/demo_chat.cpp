/**
 * Zoo-Keeper Demo Chat
 *
 * Interactive CLI for chatting with a locally hosted LLM.
 * Configuration is loaded from a JSON file.
 *
 * Usage:
 *   ./demo_chat <config.json>
 *   ./demo_chat --help
 *
 * Example config.json:
 *   {
 *     "model_path": "models/llama-3-8b.gguf",
 *     "context_size": 8192,
 *     "max_tokens": -1,
 *     "n_gpu_layers": -1,
 *     "system_prompt": "You are a helpful AI assistant.",
 *     "sampling": {
 *       "temperature": 0.7,
 *       "top_p": 0.9,
 *       "top_k": 40,
 *       "repeat_penalty": 1.1,
 *       "seed": -1
 *     },
 *     "stop_sequences": [],
 *     "tools": true
 *   }
 */

#include <zoo/zoo.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <ctime>

static std::atomic<bool> g_interrupted{false};

// ============================================================================
// Example Tools
// ============================================================================

static int calculate_add(int a, int b) { return a + b; }
static int calculate_subtract(int a, int b) { return a - b; }
static double calculate_multiply(double a, double b) { return a * b; }

static std::string get_current_time() {
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

// ============================================================================
// Config loading
// ============================================================================

struct DemoConfig {
    zoo::Config zoo;
    bool tools_enabled = true;
};

static DemoConfig load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    auto j = nlohmann::json::parse(file);
    DemoConfig dc;

    dc.zoo.model_path = j.at("model_path").get<std::string>();
    if (j.contains("context_size"))    dc.zoo.context_size    = j["context_size"].get<int>();
    if (j.contains("max_tokens"))      dc.zoo.max_tokens      = j["max_tokens"].get<int>();
    if (j.contains("n_gpu_layers"))    dc.zoo.n_gpu_layers    = j["n_gpu_layers"].get<int>();
    if (j.contains("use_mmap"))        dc.zoo.use_mmap        = j["use_mmap"].get<bool>();
    if (j.contains("use_mlock"))       dc.zoo.use_mlock       = j["use_mlock"].get<bool>();
    if (j.contains("system_prompt"))   dc.zoo.system_prompt   = j["system_prompt"].get<std::string>();
    if (j.contains("stop_sequences")) {
        dc.zoo.stop_sequences = j["stop_sequences"].get<std::vector<std::string>>();
    }
    if (j.contains("tools")) {
        dc.tools_enabled = j["tools"].get<bool>();
    }
    if (j.contains("max_tool_iterations")) {
        dc.zoo.max_tool_iterations = j["max_tool_iterations"].get<int>();
    }
    if (j.contains("max_tool_retries")) {
        dc.zoo.max_tool_retries = j["max_tool_retries"].get<int>();
    }
    if (j.contains("request_queue_capacity")) {
        dc.zoo.request_queue_capacity = j["request_queue_capacity"].get<size_t>();
    }

    if (j.contains("sampling")) {
        auto& s = j["sampling"];
        if (s.contains("temperature"))    dc.zoo.sampling.temperature    = s["temperature"].get<float>();
        if (s.contains("top_p"))          dc.zoo.sampling.top_p          = s["top_p"].get<float>();
        if (s.contains("top_k"))          dc.zoo.sampling.top_k          = s["top_k"].get<int>();
        if (s.contains("repeat_penalty")) dc.zoo.sampling.repeat_penalty = s["repeat_penalty"].get<float>();
        if (s.contains("repeat_last_n"))  dc.zoo.sampling.repeat_last_n  = s["repeat_last_n"].get<int>();
        if (s.contains("seed"))           dc.zoo.sampling.seed           = s["seed"].get<int>();
    }

    return dc;
}

// ============================================================================
// Display helpers
// ============================================================================

static void print_separator() {
    std::cout << std::string(60, '-') << "\n";
}

static void print_metrics(const zoo::Metrics& metrics, const zoo::TokenUsage& usage) {
    std::cout << "\n";
    print_separator();
    std::cout << "  Tokens: " << usage.prompt_tokens << " prompt + "
              << usage.completion_tokens << " completion = "
              << usage.total_tokens << " total\n";
    std::cout << "  Latency: " << metrics.latency_ms.count() << " ms\n";
    std::cout << "  TTFT: " << metrics.time_to_first_token_ms.count() << " ms\n";
    std::cout << "  Speed: " << std::fixed << std::setprecision(1)
              << metrics.tokens_per_second << " tok/s\n";
    print_separator();
}

static void print_welcome(const DemoConfig& dc) {
    std::cout << "\n";
    print_separator();
    std::cout << "Zoo-Keeper Demo Chat\n";
    print_separator();
    std::cout << "  Model: " << dc.zoo.model_path << "\n";
    std::cout << "  Context: " << dc.zoo.context_size << " tokens\n";
    std::cout << "  Max tokens: " << (dc.zoo.max_tokens == -1 ? "unlimited" : std::to_string(dc.zoo.max_tokens)) << "\n";
    std::cout << "  Temperature: " << dc.zoo.sampling.temperature << "\n";
    std::cout << "  GPU layers: " << dc.zoo.n_gpu_layers << "\n";
    std::cout << "  System: " << dc.zoo.system_prompt.value_or("(none)") << "\n";
    std::cout << "  Tools: " << (dc.tools_enabled ? "enabled" : "disabled") << "\n";
    print_separator();
    std::cout << "\nType a message and press Enter. Commands: /quit /clear /help\n\n";
}

static void print_usage(const char* prog) {
    std::cout << "Zoo-Keeper Demo Chat\n\n"
              << "Usage:\n"
              << "  " << prog << " <config.json>\n"
              << "  " << prog << " --help\n\n"
              << "The config file is a JSON object with these fields:\n\n"
              << "  model_path      (required) Path to GGUF model file\n"
              << "  context_size    Context window size (default: 8192)\n"
              << "  max_tokens      Max generation tokens, -1 = unlimited (default: -1)\n"
              << "  n_gpu_layers    GPU layers to offload, -1 = all (default: -1)\n"
              << "  use_mmap        Memory-map model file (default: true)\n"
              << "  use_mlock       Lock model in RAM (default: false)\n"
              << "  system_prompt   System prompt string\n"
              << "  stop_sequences  Array of stop strings\n"
              << "  tools           Enable example tools (default: true)\n"
              << "  sampling        Object with: temperature, top_p, top_k,\n"
              << "                  repeat_penalty, repeat_last_n, seed\n\n"
              << "Example:\n"
              << "  " << prog << " config.json\n";
}

// ============================================================================
// Main
// ============================================================================

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
        (void)agent->register_tool("subtract", "Subtract two integers", {"a", "b"}, calculate_subtract);
        (void)agent->register_tool("multiply", "Multiply two numbers", {"a", "b"}, calculate_multiply);
        (void)agent->register_tool("get_time", "Get the current date and time", {}, get_current_time);

        auto base_prompt = dc.zoo.system_prompt.value_or("You are a helpful AI assistant.");
        agent->set_system_prompt(agent->build_tool_system_prompt(base_prompt));
    }

    print_welcome(dc);

    // Chat loop
    std::string line;
    while (!g_interrupted.load(std::memory_order_acquire)) {
        std::cout << "You: ";
        std::cout.flush();

        if (!std::getline(std::cin, line)) break;

        // Trim
        auto start = line.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        line = line.substr(start, line.find_last_not_of(" \t\n\r") - start + 1);

        // Commands
        if (line == "/quit" || line == "/exit") break;
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

        auto handle = agent->chat(
            zoo::Message::user(line),
            [](std::string_view token) { std::cout << token << std::flush; }
        );

        // Poll for Ctrl+C during generation
        while (handle.future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
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
