/**
 * @file minimal_agent.cpp
 * @brief Smallest end-to-end `zoo::Agent` example: load a model and complete one chat.
 *
 * Usage:
 *   ./minimal_agent <model.gguf>
 */

#include <zoo/zoo.hpp>

#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: minimal_agent <model.gguf>\n";
        return 1;
    }

    zoo::ModelConfig model;
    model.model_path = argv[1];
    model.context_size = 8192;
    model.n_gpu_layers = 0;

    zoo::AgentConfig agent_config;
    agent_config.max_history_messages = 32;

    zoo::GenerationOptions generation;
    generation.max_tokens = 256;

    auto result = zoo::Agent::create(model, agent_config, generation);
    if (!result) {
        std::cerr << result.error().to_string() << '\n';
        return 1;
    }
    auto agent_runtime = std::move(*result);

    if (auto prompt = agent_runtime->try_set_system_prompt("You are a helpful AI assistant.");
        !prompt) {
        std::cerr << prompt.error().to_string() << '\n';
        return 1;
    }

    auto handle = agent_runtime->chat(zoo::MessageView{zoo::Role::User, "Hello!"});
    auto response = handle.await_result();
    if (!response) {
        std::cerr << response.error().to_string() << '\n';
        return 1;
    }

    std::cout << response->text << '\n';
    return 0;
}
