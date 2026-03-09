/**
 * @file error_handling.cpp
 * @brief Demonstrates practical runtime error handling for `zoo::Agent`.
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

    zoo::Config config;
    config.model_path = argv[1];
    config.max_tokens = 64;
    config.n_gpu_layers = 0;

    auto agent_result = zoo::Agent::create(config);
    if (!agent_result) {
        print_error(agent_result.error());
        return 1;
    }

    auto& agent = *agent_result;
    auto handle = agent->chat(zoo::Message::user("Say hello in one sentence."));
    auto response = handle.future.get();
    if (!response) {
        print_error(response.error());
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
