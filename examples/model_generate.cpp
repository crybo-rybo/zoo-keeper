/**
 * @file model_generate.cpp
 * @brief Minimal standalone `zoo::core::Model` example.
 */

#include <zoo/zoo.hpp>

#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: model_generate <model.gguf> <prompt>\n";
        return 1;
    }

    zoo::Config config;
    config.model_path = argv[1];
    config.max_tokens = 128;
    config.n_gpu_layers = 0;

    auto model_result = zoo::core::Model::load(config);
    if (!model_result) {
        std::cerr << model_result.error().to_string() << '\n';
        return 1;
    }

    auto response = (*model_result)->generate(argv[2]);
    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
