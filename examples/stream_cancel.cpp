/**
 * @file stream_cancel.cpp
 * @brief Streams a response and cancels it after a short deadline.
 */

#include <zoo/zoo.hpp>

#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: stream_cancel <model.gguf>\n";
        return 1;
    }

    zoo::ModelConfig model_config;
    model_config.model_path = argv[1];
    model_config.n_gpu_layers = 0;

    zoo::GenerationOptions generation;
    generation.max_tokens = 512;

    auto model_result = zoo::Model::load(model_config, generation);
    if (!model_result) {
        std::cerr << model_result.error().to_string() << '\n';
        return 1;
    }

    auto& model = *model_result;
    const auto start = std::chrono::steady_clock::now();
    auto on_token = [](std::string_view token) {
        std::cout << token << std::flush;
        return zoo::TokenAction::Continue;
    };
    auto should_cancel = [&] { return std::chrono::steady_clock::now() - start > 250ms; };
    auto response =
        model->generate("Write a detailed travel guide for Iceland.",
                        zoo::GenerationOverride::inherit_defaults(), on_token, should_cancel);

    if (!response) {
        if (response.error().code == zoo::ErrorCode::RequestCancelled) {
            std::cout << "\n[generation cancelled]\n";
            return 0;
        }

        std::cerr << '\n' << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << "\n\nCompleted without cancellation.\n";
    return 0;
}
