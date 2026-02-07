/**
 * Zoo-Keeper Demo Chat Application
 *
 * Interactive CLI demonstrating the Zoo-Keeper Agent Engine with real LLM inference.
 *
 * Usage:
 *   ./demo_chat <model_path> [options]
 *
 * Options:
 *   --temperature <float>    Sampling temperature (default: 0.7)
 *   --max-tokens <int>       Max tokens to generate (default: 512)
 *   --context-size <int>     Context window size (default: 8192)
 *   --template <type>        Template type: llama3, chatml (default: llama3)
 *   --system <prompt>        System prompt
 *   --no-stream              Disable token streaming
 *   --help                   Show this help message
 */

#include "zoo/agent.hpp"
#include "zoo/types.hpp"

#include <iostream>
#include <string>
#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <future>
#include <ctime>

// Global flag for Ctrl+C handling
std::atomic<bool> g_interrupted{false};

// ============================================================================
// Example Tools
// ============================================================================

int calculate_add(int a, int b) { return a + b; }
int calculate_subtract(int a, int b) { return a - b; }
double calculate_multiply(double a, double b) { return a * b; }

std::string get_current_time() {
    auto now = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

void signal_handler(int signal) {
    if (signal == SIGINT) {
        g_interrupted = true;
        std::cout << "\n\nInterrupted. Stopping generation...\n";
    }
}

struct CLIArgs {
    std::string model_path;
    float temperature = 0.7f;
    int max_tokens = 512;
    int context_size = 8192;
    zoo::PromptTemplate template_type = zoo::PromptTemplate::Llama3;
    std::string system_prompt = "You are a helpful AI assistant.";
    bool no_stream = false;
    bool help = false;
};

void print_usage(const char* program_name) {
    std::cout << "Zoo-Keeper Demo Chat Application\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " <model_path> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --temperature <float>    Sampling temperature (default: 0.7)\n";
    std::cout << "  --max-tokens <int>       Max tokens to generate (default: 512)\n";
    std::cout << "  --context-size <int>     Context window size (default: 8192)\n";
    std::cout << "  --template <type>        Template type: llama3, chatml (default: llama3)\n";
    std::cout << "  --system <prompt>        System prompt\n";
    std::cout << "  --no-stream              Disable token streaming\n";
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " model.gguf --temperature 0.8 --max-tokens 1024\n\n";
    std::cout << "Interactive Commands:\n";
    std::cout << "  /quit, /exit    Exit the application\n";
    std::cout << "  /clear          Clear conversation history\n";
    std::cout << "  /help           Show available commands\n";
    std::cout << "  Ctrl+C          Stop current generation\n";
}

CLIArgs parse_args(int argc, char** argv) {
    CLIArgs args;

    if (argc < 2) {
        args.help = true;
        return args;
    }

    // Check if first argument is --help
    if (std::string(argv[1]) == "--help") {
        args.help = true;
        return args;
    }

    args.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            args.help = true;
            return args;
        }
        else if (arg == "--temperature" && i + 1 < argc) {
            args.temperature = std::stof(argv[++i]);
        }
        else if (arg == "--max-tokens" && i + 1 < argc) {
            args.max_tokens = std::stoi(argv[++i]);
        }
        else if (arg == "--context-size" && i + 1 < argc) {
            args.context_size = std::stoi(argv[++i]);
        }
        else if (arg == "--template" && i + 1 < argc) {
            std::string tmpl = argv[++i];
            if (tmpl == "llama3") {
                args.template_type = zoo::PromptTemplate::Llama3;
            } else if (tmpl == "chatml") {
                args.template_type = zoo::PromptTemplate::ChatML;
            } else {
                std::cerr << "Unknown template type: " << tmpl << "\n";
                args.help = true;
                return args;
            }
        }
        else if (arg == "--system" && i + 1 < argc) {
            args.system_prompt = argv[++i];
        }
        else if (arg == "--no-stream") {
            args.no_stream = true;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            args.help = true;
            return args;
        }
    }

    return args;
}

void print_separator() {
    std::cout << std::string(60, '-') << "\n";
}

void print_metrics(const zoo::Metrics& metrics, const zoo::TokenUsage& usage) {
    std::cout << "\n";
    print_separator();
    std::cout << "Metrics:\n";
    std::cout << "  Tokens: " << usage.prompt_tokens << " prompt + "
              << usage.completion_tokens << " completion = "
              << usage.total_tokens << " total\n";
    std::cout << "  Latency: " << metrics.latency_ms.count() << " ms\n";
    std::cout << "  Time to first token: " << metrics.time_to_first_token_ms.count() << " ms\n";
    std::cout << "  Speed: " << std::fixed << std::setprecision(2)
              << metrics.tokens_per_second << " tokens/sec\n";
    print_separator();
}

void print_welcome(const zoo::Config& config, bool streaming) {
    std::cout << "\n";
    print_separator();
    std::cout << "Zoo-Keeper Demo Chat\n";
    print_separator();
    std::cout << "Model: " << config.model_path << "\n";
    std::cout << "Template: " << zoo::template_to_string(config.prompt_template) << "\n";
    std::cout << "Temperature: " << config.sampling.temperature << "\n";
    std::cout << "Max Tokens: " << config.max_tokens << "\n";
    std::cout << "Context Size: " << config.context_size << "\n";
    std::cout << "System Prompt: " << config.system_prompt.value_or("(none)") << "\n";
    std::cout << "Streaming: " << (streaming ? "enabled" : "disabled") << "\n";
    print_separator();
    std::cout << "\nType your message and press Enter. Type '/quit' to exit.\n";
    std::cout << "Type '/help' for available commands.\n\n";
}

int main(int argc, char** argv) {
    // Parse arguments first before any other operations
    CLIArgs args = parse_args(argc, argv);

    if (args.help) {
        print_usage(argv[0]);
        return args.model_path.empty() ? 1 : 0;
    }

    // Setup signal handler
    std::signal(SIGINT, signal_handler);

    // Build zoo::Config from CLI args
    zoo::Config config;
    config.model_path = args.model_path;
    config.context_size = args.context_size;
    config.max_tokens = args.max_tokens;
    config.sampling.temperature = args.temperature;
    config.prompt_template = args.template_type;
    config.system_prompt = args.system_prompt;

    // Create agent
    std::cout << "Loading model...\n";
    auto agent_result = zoo::Agent::create(config);
    if (!agent_result) {
        std::cerr << "Error: " << agent_result.error().to_string() << "\n";
        return 1;
    }
    auto agent = std::move(*agent_result);  // Now a std::unique_ptr<Agent>

    // Register example tools
    agent->register_tool("add", "Add two integers", {"a", "b"}, calculate_add);
    agent->register_tool("subtract", "Subtract two integers", {"a", "b"}, calculate_subtract);
    agent->register_tool("multiply", "Multiply two numbers", {"a", "b"}, calculate_multiply);
    agent->register_tool("get_current_time", "Get the current date and time", {}, get_current_time);
    std::cout << "Registered " << agent->tool_count() << " tools.\n";

    // Print welcome
    print_welcome(config, !args.no_stream);

    // Main chat loop
    std::string line;
    while (!g_interrupted) {
        // Print prompt
        std::cout << "\nYou: ";
        std::cout.flush();

        // Read user input
        if (!std::getline(std::cin, line)) {
            break;  // EOF or error
        }

        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r"));
        line.erase(line.find_last_not_of(" \t\n\r") + 1);

        if (line.empty()) {
            continue;
        }

        // Handle commands
        if (line[0] == '/') {
            if (line == "/quit" || line == "/exit") {
                std::cout << "Goodbye!\n";
                break;
            }
            else if (line == "/clear") {
                agent->clear_history();
                std::cout << "Conversation history cleared.\n";
                continue;
            }
            else if (line == "/help") {
                std::cout << "\nAvailable commands:\n";
                std::cout << "  /quit, /exit    Exit the application\n";
                std::cout << "  /clear          Clear conversation history\n";
                std::cout << "  /help           Show this help\n";
                continue;
            }
            else {
                std::cout << "Unknown command: " << line << "\n";
                std::cout << "Type '/help' for available commands.\n";
                continue;
            }
        }

        // Print assistant prompt
        std::cout << "\nAssistant: ";
        std::cout.flush();

        // Reset interrupt flag
        g_interrupted = false;

        // Submit chat request with optional streaming callback
        std::future<zoo::Expected<zoo::Response>> future;
        if (args.no_stream) {
            // No streaming
            future = agent->chat(zoo::Message::user(line));
        } else {
            // With streaming
            auto callback = [](std::string_view token) {
                std::cout << token << std::flush;
            };
            future = agent->chat(zoo::Message::user(line), callback);
        }

        // Poll for interrupt during generation
        while (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
            if (g_interrupted) {
                agent->stop();
                std::cout << "\n[Generation stopped]\n";

                // Recreate agent for next request
                std::cout << "Reloading agent...\n";
                agent_result = zoo::Agent::create(config);
                if (!agent_result) {
                    std::cerr << "Error: " << agent_result.error().to_string() << "\n";
                    return 1;
                }
                agent = std::move(*agent_result);

                g_interrupted = false;
                break;
            }
        }

        // Get result if not interrupted
        if (!g_interrupted) {
            auto result = future.get();

            if (!result.has_value()) {
                std::cerr << "\nError: " << result.error().to_string() << "\n";
                continue;
            }

            // If not streaming, print the full response
            if (args.no_stream) {
                std::cout << result->text;
            }
            std::cout << "\n";

            // Print metrics
            print_metrics(result->metrics, result->usage);
        }
    }

    std::cout << "\n";
    return 0;
}
