/**
 * @file stream_cancel.cpp
 * @brief Streams a response and cancels it after a short delay.
 */

#include <zoo/zoo.hpp>

#include <chrono>
#include <atomic>
#include <iostream>
#include <thread>

using namespace std::chrono_literals;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: stream_cancel <model.gguf>\n";
        return 1;
    }

    zoo::Config config;
    config.model_path = argv[1];
    config.max_tokens = 512;
    config.n_gpu_layers = 0;

    auto agent_result = zoo::Agent::create(config);
    if (!agent_result) {
        std::cerr << agent_result.error().to_string() << '\n';
        return 1;
    }

    auto& agent = *agent_result;
    auto handle = agent->chat(
        zoo::Message::user("Write a detailed travel guide for Iceland."),
        [](std::string_view token) {
            std::cout << token << std::flush;
        }
    );

    std::atomic<bool> completed{false};
    std::thread canceller([&agent, &completed, id = handle.id] {
        std::this_thread::sleep_for(250ms);
        if (!completed.load()) {
            agent->cancel(id);
        }
    });

    auto response = handle.future.get();
    completed.store(true);
    canceller.join();

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
