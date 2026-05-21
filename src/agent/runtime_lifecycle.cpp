/**
 * @file runtime_lifecycle.cpp
 * @brief Lifecycle and shutdown handling for the internal agent runtime.
 */

#include "agent/runtime.hpp"

#include <thread>
#include <utility>

namespace zoo::internal::agent {

AgentRuntime::AgentRuntime(ModelConfig model_config, AgentConfig agent_config,
                           GenerationOptions default_generation,
                           std::unique_ptr<AgentBackend> backend)
    : model_config_(std::move(model_config)), agent_config_(agent_config),
      default_generation_options_(std::move(default_generation)), backend_(std::move(backend)),
      request_slots_(std::make_shared<RequestSlots>(agent_config_.request_queue_capacity)),
      request_mailbox_() {
    inference_thread_ = std::thread([this]() { inference_loop(); });
}

AgentRuntime::~AgentRuntime() {
    stop();
}

void AgentRuntime::stop() {
    if (!running_.load(std::memory_order_acquire)) {
        return;
    }

    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();
    if (inference_thread_.joinable()) {
        inference_thread_.join();
    }
}

bool AgentRuntime::is_running() const noexcept {
    return running_.load(std::memory_order_acquire);
}

void AgentRuntime::fail_pending(const Error& error) {
    running_.store(false, std::memory_order_release);
    request_mailbox_.shutdown();

    while (auto remaining = request_mailbox_.pop()) {
        std::visit(overloaded{
                       [&](QueuedRequest request) {
                           request_slots_->resolve_error(request.slot, request.generation, error);
                       },
                       [](Command& cmd) { resolve_command_on_shutdown(cmd); },
                   },
                   *remaining);
    }

    request_slots_->fail_all(error);
}

} // namespace zoo::internal::agent
