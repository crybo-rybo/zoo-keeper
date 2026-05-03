/**
 * @file stream_cancel.cpp
 * @brief Streams a response and cancels it after a short delay.
 */

#include <zoo/zoo.hpp>

#include <chrono>
#include <iostream>
#include <thread>

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

    auto agent_result = zoo::Agent::create(model_config, zoo::AgentConfig{}, generation);
    if (!agent_result) {
        std::cerr << agent_result.error().to_string() << '\n';
        return 1;
    }

    auto& agent = *agent_result;
    auto handle = agent->chat("Write a detailed travel guide for Iceland.", {},
                              [](std::string_view token) { std::cout << token << std::flush; });

    std::this_thread::sleep_for(250ms);
    handle.cancel();

    auto response = handle.await_result();

    if (!response) {
        if (response.error().code == zoo::ErrorCode::RequestCancelled) {
            std::cout << "\n[request cancelled]\n";
            return 0;
        }

        std::cerr << '\n' << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << "\n\nCompleted without cancellation.\n";
    return 0;
}
