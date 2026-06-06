/**
 * @file error_handling.cpp
 * @brief Demonstrates practical runtime error handling for `zoo::Model`.
 */

#include <zoo/zoo.hpp>

#include <iostream>

namespace {

void print_error(const zoo::Error& error) {
    switch (error.code) {
    case zoo::ErrorCode::TemplateRenderFailed:
        std::cerr << "The selected model does not expose a chat template: " << error.message
                  << '\n';
        break;
    case zoo::ErrorCode::RequestCancelled:
        std::cerr << "The request was cancelled before completion.\n";
        break;
    case zoo::ErrorCode::InferenceFailed:
        std::cerr << "Inference failed: " << error.to_string() << '\n';
        break;
    default:
        std::cerr << error.to_string() << '\n';
        break;
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: error_handling <model.gguf>\n";
        return 1;
    }

    zoo::ModelConfig model_config;
    model_config.model_path = argv[1];
    model_config.n_gpu_layers = 0;

    zoo::GenerationOptions generation;
    generation.max_tokens = 64;

    auto model_result = zoo::Model::load(model_config, generation);
    if (!model_result) {
        print_error(model_result.error());
        return 1;
    }

    auto& model = *model_result;
    auto response = model->generate("Say hello in one sentence.");
    if (!response) {
        print_error(response.error());
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
