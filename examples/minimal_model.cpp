/**
 * @file minimal_model.cpp
 * @brief Smallest end-to-end `zoo::Model` example.
 */

#include <zoo/zoo.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: minimal_model <model.gguf>\n";
        return 1;
    }

    zoo::ModelConfig model_config;
    model_config.model_path = argv[1];
    model_config.n_gpu_layers = 0;

    zoo::GenerationOptions generation;
    generation.max_tokens = 64;

    auto result = zoo::Model::load(model_config, generation);
    if (!result) {
        std::cerr << result.error().to_string() << '\n';
        return 1;
    }

    auto model = std::move(*result);
    model->set_system_prompt("You are a helpful AI assistant.");

    auto response = model->generate("Hello!");
    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
