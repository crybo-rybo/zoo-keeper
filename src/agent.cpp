/**
 * @file agent.cpp
 * @brief Public Agent facade implementation.
 */

#include "zoo/agent.hpp"

#include "zoo/core/model.hpp"
#include "zoo/internal/agent/backend_model.hpp"
#include "zoo/internal/agent/runtime.hpp"

namespace zoo {
namespace runtime = internal::agent;

struct Agent::Impl {
    explicit Impl(const Config& cfg, std::unique_ptr<runtime::AgentBackend> owned_backend)
        : runtime(cfg, std::move(owned_backend)) {}

    runtime::AgentRuntime runtime;
};

Expected<std::unique_ptr<Agent>> Agent::create(const Config& config) {
    auto model_result = core::Model::load(config);
    if (!model_result) {
        return std::unexpected(model_result.error());
    }

    auto backend = runtime::make_model_backend(std::move(*model_result));
    auto agent_impl = std::make_unique<Impl>(config, std::move(backend));
    return std::unique_ptr<Agent>(new Agent(config, std::move(agent_impl)));
}

Agent::Agent(Config config, std::unique_ptr<Impl> impl)
    : config_(std::move(config)), impl_(std::move(impl)) {}

Agent::~Agent() = default;

RequestHandle Agent::chat(Message message,
                          std::optional<std::function<void(std::string_view)>> callback) {
    return impl_->runtime.chat(std::move(message), std::move(callback));
}

RequestHandle Agent::complete(std::vector<Message> messages,
                              std::optional<std::function<void(std::string_view)>> callback) {
    return impl_->runtime.complete(std::move(messages), std::move(callback));
}

RequestHandle Agent::extract(const nlohmann::json& output_schema, Message message,
                             std::optional<std::function<void(std::string_view)>> callback) {
    return impl_->runtime.extract(output_schema, std::move(message), std::move(callback));
}

RequestHandle Agent::extract(const nlohmann::json& output_schema, std::vector<Message> messages,
                             std::optional<std::function<void(std::string_view)>> callback) {
    return impl_->runtime.extract(output_schema, std::move(messages), std::move(callback));
}

void Agent::cancel(RequestId id) {
    impl_->runtime.cancel(id);
}

void Agent::set_system_prompt(const std::string& prompt) {
    impl_->runtime.set_system_prompt(prompt);
}

void Agent::stop() {
    impl_->runtime.stop();
}

bool Agent::is_running() const noexcept {
    return impl_->runtime.is_running();
}

std::vector<Message> Agent::get_history() const {
    return impl_->runtime.get_history();
}

void Agent::clear_history() {
    impl_->runtime.clear_history();
}

Expected<void> Agent::register_tool(tools::ToolDefinition definition) {
    return impl_->runtime.register_tool(std::move(definition));
}

Expected<void> Agent::register_tool(const std::string& name, const std::string& description,
                                    nlohmann::json schema, tools::ToolHandler handler) {
    auto definition =
        tools::detail::make_tool_definition(name, description, schema, std::move(handler));
    if (!definition) {
        return std::unexpected(definition.error());
    }
    return register_tool(std::move(*definition));
}

size_t Agent::tool_count() const noexcept {
    return impl_->runtime.tool_count();
}

std::string Agent::build_tool_system_prompt(const std::string& base_prompt) const {
    return impl_->runtime.build_tool_system_prompt(base_prompt);
}

} // namespace zoo
