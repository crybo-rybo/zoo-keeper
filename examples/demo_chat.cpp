/**
 * Zoo-Keeper Demo Chat Application
 *
 * Interactive CLI demonstrating the Zoo-Keeper Agent Engine.
 * This is a placeholder implementation that shows the intended API
 * and user experience for when the Agent class is fully implemented.
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
 *   --help                   Show this help message
 */

#include "zoo/types.hpp"
#include "zoo/engine/history_manager.hpp"
#include "zoo/engine/template_engine.hpp"
#include "zoo/backend/interface.hpp"

#include <iostream>
#include <string>

#include <csignal>
#include <atomic>
#include <chrono>
#include <iomanip>

// Global flag for Ctrl+C handling
std::atomic<bool> g_interrupted{false};

void signal_handler(int signal) {
    if (signal == SIGINT) {
        g_interrupted = true;
        std::cout << "\n\nInterrupted. Exiting gracefully...\n";
    }
}

struct DemoConfig {
    std::string model_path;
    float temperature = 0.7f;
    int max_tokens = 512;
    int context_size = 8192;
    zoo::PromptTemplate template_type = zoo::PromptTemplate::Llama3;
    std::string system_prompt = "You are a helpful AI assistant.";
    std::vector<std::string> stop_sequences;
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
    std::cout << "  --help                   Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " model.gguf --temperature 0.8 --max-tokens 1024\n\n";
    std::cout << "Interactive Commands:\n";
    std::cout << "  /quit, /exit    Exit the application\n";
    std::cout << "  /clear          Clear conversation history\n";
    std::cout << "  /help           Show available commands\n";
    std::cout << "  Ctrl+C          Exit gracefully\n";
}

DemoConfig parse_args(int argc, char** argv) {
    DemoConfig config;

    if (argc < 2) {
        config.help = true;
        return config;
    }

    config.model_path = argv[1];

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            config.help = true;
            return config;
        }
        else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        }
        else if (arg == "--max-tokens" && i + 1 < argc) {
            config.max_tokens = std::stoi(argv[++i]);
        }
        else if (arg == "--context-size" && i + 1 < argc) {
            config.context_size = std::stoi(argv[++i]);
        }
        else if (arg == "--template" && i + 1 < argc) {
            std::string tmpl = argv[++i];
            if (tmpl == "llama3") {
                config.template_type = zoo::PromptTemplate::Llama3;
            } else if (tmpl == "chatml") {
                config.template_type = zoo::PromptTemplate::ChatML;
            } else {
                std::cerr << "Unknown template type: " << tmpl << "\n";
                config.help = true;
                return config;
            }
        }
        else if (arg == "--system" && i + 1 < argc) {
            config.system_prompt = argv[++i];
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            config.help = true;
            return config;
        }
    }

    return config;
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

void print_welcome(const DemoConfig& config) {
    std::cout << "\n";
    print_separator();
    std::cout << "Zoo-Keeper Demo Chat\n";
    print_separator();
    std::cout << "Model: " << config.model_path << "\n";
    std::cout << "Template: " << zoo::template_to_string(config.template_type) << "\n";
    std::cout << "Temperature: " << config.temperature << "\n";
    std::cout << "Max Tokens: " << config.max_tokens << "\n";
    std::cout << "Context Size: " << config.context_size << "\n";
    std::cout << "System Prompt: " << config.system_prompt << "\n";
    print_separator();
    std::cout << "\nType your message and press Enter. Type '/quit' to exit.\n";
    std::cout << "Type '/help' for available commands.\n\n";
}

// Placeholder implementation until Agent class is ready
class DemoAgent {
public:
    DemoAgent(const DemoConfig& config)
        : history_(config.context_size)
        , template_engine_(config.template_type)
        , config_(config)
    {
        history_.set_system_prompt(config.system_prompt);

        // Initialize backend
        zoo::Config backend_config;
        backend_config.model_path = config.model_path;
        backend_config.context_size = config.context_size;
        backend_config.sampling.temperature = config.temperature;
        backend_config.prompt_template = config.template_type;
        backend_config.max_tokens = config.max_tokens;
        backend_config.system_prompt = config.system_prompt;
        backend_config.stop_sequences = config.stop_sequences;
        
        backend_ = zoo::backend::create_backend();
        auto init_result = backend_->initialize(backend_config);
        if (!init_result.has_value()) {
            std::cerr << "Failed to initialize backend: " << init_result.error().to_string() << "\n";
            std::exit(1);
        }

        // Warmup to suppress initial GGML metal logs
        std::cout << "Loading model and warming up backend... " << std::flush;
        
        // Redirect stdout/stderr to bit bucket
        auto old_stdout = std::cout.rdbuf();
        auto old_stderr = std::cerr.rdbuf();
        std::cout.rdbuf(nullptr);
        std::cerr.rdbuf(nullptr);

        // Simple warmup generation
        std::vector<int> warmup_tokens = {1}; // BOS
        auto warmup_result = backend_->generate(warmup_tokens, 1, {}, std::nullopt);
        if (!warmup_result.has_value()) {
            std::cerr << "Failed to warmup backend: " << warmup_result.error().to_string() << "\n";
            std::exit(1);
        }
        // Restore stdout/stderr
        std::cout.rdbuf(old_stdout);
        std::cerr.rdbuf(old_stderr);
        
        std::cout << "Done.\n" << std::flush;
    }

    zoo::Expected<zoo::Response> chat(const std::string& user_message) {
        // Add user message to history
        auto add_result = history_.add_message(zoo::Message::user(user_message));
        if (!add_result.has_value()) {
            return tl::unexpected(add_result.error());
        }

        // Render conversation
        auto rendered = template_engine_.render(history_.get_messages());
        if (!rendered.has_value()) {
            return tl::unexpected(rendered.error());
        }

        // Tokenize prompt
        auto prompt_tokens = backend_->tokenize(*rendered, true); // true = add_bos
        if (!prompt_tokens.has_value()) {
             return tl::unexpected(prompt_tokens.error());
        }

        // Metrics tracking
        auto start_time = std::chrono::steady_clock::now();
        int completion_tokens = 0;
        bool first_token = true;
        auto first_token_time = start_time;

        // Generate with streaming
        auto result_text = backend_->generate(*prompt_tokens, config_.max_tokens, config_.stop_sequences, 
            [&](std::string_view token) {
                if (first_token) {
                    first_token_time = std::chrono::steady_clock::now();
                    first_token = false;
                }
                std::cout << token << std::flush;
                completion_tokens++;
            }
        );

        if (!result_text.has_value()) {
            return tl::unexpected(result_text.error());
        }
        
        auto end_time = std::chrono::steady_clock::now();
        std::cout << "\n"; // Newline after stream

        zoo::Response response;
        response.text = *result_text;
        
        // Populate metrics
        response.usage.prompt_tokens = static_cast<int>(prompt_tokens->size());
        response.usage.completion_tokens = completion_tokens;
        response.usage.total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens;
        
        response.metrics.latency_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (!first_token) {
             response.metrics.time_to_first_token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(first_token_time - start_time);
        } else {
             response.metrics.time_to_first_token_ms = response.metrics.latency_ms;
        }
        
        if (response.metrics.latency_ms.count() > 0) {
            response.metrics.tokens_per_second = (double)completion_tokens / (response.metrics.latency_ms.count() / 1000.0);
        }

        // Add assistant response to history
        add_result = history_.add_message(zoo::Message::assistant(response.text));
        if (!add_result.has_value()) {
            return tl::unexpected(add_result.error());
        }

        return response;
    }

    void clear_history() {
        history_.clear();
        history_.set_system_prompt(config_.system_prompt);
        backend_->clear_kv_cache();
    }

    int get_message_count() const {
        return static_cast<int>(history_.get_messages().size());
    }

    bool is_context_exceeded() const {
        return history_.is_context_exceeded();
    }

private:
    zoo::engine::HistoryManager history_;
    zoo::engine::TemplateEngine template_engine_;
    DemoConfig config_;
    std::unique_ptr<zoo::backend::IBackend> backend_;
};

int main(int argc, char** argv) {
    // Parse arguments
    DemoConfig config = parse_args(argc, argv);

    // Set default stop sequences if not provided
    if (config.stop_sequences.empty()) {
        if (config.template_type == zoo::PromptTemplate::Llama3) {
            config.stop_sequences = {"<|eot_id|>", "<|end_of_text|>"};
        } else if (config.template_type == zoo::PromptTemplate::ChatML) {
            config.stop_sequences = {"<|im_end|>"};
        }
    }

    if (config.help) {
        print_usage(argv[0]);
        return config.model_path.empty() ? 1 : 0;
    }

    // Setup signal handler
    std::signal(SIGINT, signal_handler);

    // Print welcome
    print_welcome(config);

    // Create demo agent
    DemoAgent agent(config);

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
                agent.clear_history();
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

        // Check context window
        if (agent.is_context_exceeded()) {
            std::cout << "\nWarning: Context window exceeded. Consider using /clear.\n";
        }

        // Process message
        std::cout << "\nAssistant: ";
        std::cout.flush();

        auto result = agent.chat(line);

        if (!result.has_value()) {
            std::cerr << "\nError: " << result.error().to_string() << "\n";
            continue;
        }

        // Response is already printed via streaming
        // std::cout << result->text << "\n";

        // Print metrics
        print_metrics(result->metrics, result->usage);
    }

    std::cout << "\n";
    return 0;
}
