/**
 * @file demo_chat.cpp
 * @brief Interactive CLI example for the synchronous `zoo::Model` harness.
 */

#include <zoo/core/json.hpp>
#include <zoo/zoo.hpp>

#include <atomic>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

static std::atomic<bool> g_interrupted{false};

struct DemoConfig {
    zoo::ModelConfig model;
    zoo::GenerationOptions generation;
    std::optional<std::string> system_prompt;
};

static void from_json(const nlohmann::json& j, DemoConfig& config) {
    if (!j.is_object()) {
        throw std::invalid_argument("Demo config must be a JSON object");
    }

    DemoConfig parsed;
    if (auto it = j.find("model"); it != j.end()) {
        auto resolved = zoo::load_model_config(*it);
        if (!resolved) {
            throw std::invalid_argument("Failed to load model config: " +
                                        resolved.error().to_string());
        }
        parsed.model = std::move(*resolved);
    } else {
        throw std::invalid_argument("Demo config must contain a model block");
    }

    if (auto it = j.find("generation"); it != j.end()) {
        parsed.generation = it->get<zoo::GenerationOptions>();
    }
    if (auto it = j.find("system_prompt"); it != j.end()) {
        parsed.system_prompt = it->get<std::string>();
    }

    config = std::move(parsed);
}

static DemoConfig load_config(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + path);
    }

    DemoConfig config = nlohmann::json::parse(file).get<DemoConfig>();
    if (auto validation = config.model.validate(); !validation) {
        throw std::runtime_error("Invalid model config: " + validation.error().to_string());
    }
    if (auto validation = config.generation.validate(); !validation) {
        throw std::runtime_error("Invalid generation config: " + validation.error().to_string());
    }
    return config;
}

static void print_separator() {
    std::cout << std::string(60, '-') << "\n";
}

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

static void print_welcome(const DemoConfig& dc) {
    std::cout << "\n";
    print_separator();
    std::cout << "Zoo-Keeper Model Harness Chat\n";
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
    print_separator();
    std::cout << "\nType a message and press Enter. Commands: /quit /clear /help\n\n";
}

static void print_usage(const char* prog) {
    std::cout << "Zoo-Keeper Model Harness Chat\n\n"
              << "Usage:\n"
              << "  " << prog << " <config.json>\n"
              << "  " << prog << " --help\n\n"
              << "Config files contain nested model / generation blocks plus optional "
                 "system_prompt.\n";
}

static void signal_handler(int) {
    g_interrupted.store(true, std::memory_order_release);
}

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage(argv[0]);
        return argc == 2 ? 0 : 1;
    }

    DemoConfig dc;
    try {
        dc = load_config(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << "\n";
        return 1;
    }

    std::signal(SIGINT, signal_handler);

    std::cout << "Loading model...\n";
    auto model_result = zoo::Model::load(dc.model, dc.generation);
    if (!model_result) {
        std::cerr << "Error: " << model_result.error().to_string() << "\n";
        return 1;
    }
    auto model = std::move(*model_result);
    if (dc.system_prompt) {
        model->set_system_prompt(*dc.system_prompt);
    }

    print_welcome(dc);

    std::string line;
    while (!g_interrupted.load(std::memory_order_acquire)) {
        std::cout << "You: ";
        std::cout.flush();

        if (!std::getline(std::cin, line)) {
            break;
        }

        auto start = line.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) {
            continue;
        }
        line = line.substr(start, line.find_last_not_of(" \t\n\r") - start + 1);

        if (line == "/quit" || line == "/exit") {
            break;
        }
        if (line == "/clear") {
            model->clear_history();
            if (dc.system_prompt) {
                model->set_system_prompt(*dc.system_prompt);
            }
            std::cout << "History cleared.\n\n";
            continue;
        }
        if (line == "/help") {
            std::cout << "  /quit, /exit  Exit\n"
                      << "  /clear        Clear conversation history\n"
                      << "  /help         Show commands\n"
                      << "  Ctrl+C        Cancel generation\n\n";
            continue;
        }

        std::cout << "\nAssistant: ";
        std::cout.flush();
        g_interrupted.store(false, std::memory_order_release);

        auto on_token = [](std::string_view token) {
            std::cout << token << std::flush;
            return zoo::TokenAction::Continue;
        };
        auto should_cancel = [] { return g_interrupted.load(std::memory_order_acquire); };
        auto result = model->generate(line, zoo::GenerationOverride::inherit_defaults(), on_token,
                                      should_cancel);

        if (!result) {
            if (result.error().code == zoo::ErrorCode::RequestCancelled) {
                std::cout << "\n[cancelled]\n";
                g_interrupted.store(false, std::memory_order_release);
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
