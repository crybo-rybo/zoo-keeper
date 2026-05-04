/**
 * @file agent_facade.cpp
 * @brief Public Agent facade implementation.
 */

#include "zoo/agent.hpp"

#include "agent/backend_model.hpp"
#include "agent/runtime.hpp"
#include "zoo/core/model.hpp"

namespace zoo {
namespace runtime = internal::agent;

struct Agent::Impl {
    Impl(ModelConfig model_config, AgentConfig agent_config, GenerationOptions default_generation,
         std::unique_ptr<runtime::AgentBackend> owned_backend)
        : runtime(std::move(model_config), agent_config, std::move(default_generation),
                  std::move(owned_backend)) {}

    runtime::AgentRuntime runtime;
};

Expected<std::unique_ptr<Agent>> Agent::create(const ModelConfig& model_config,
                                               const AgentConfig& agent_config,
                                               const GenerationOptions& default_generation) {
    if (auto result = model_config.validate(); !result) {
        return std::unexpected(result.error());
    }
    if (auto result = agent_config.validate(); !result) {
        return std::unexpected(result.error());
    }
    if (auto result = default_generation.validate(); !result) {
        return std::unexpected(result.error());
    }

    auto model_result = core::Model::load(model_config, default_generation);
    if (!model_result) {
        return std::unexpected(model_result.error());
    }

    auto backend = runtime::make_model_backend(std::move(*model_result));
    auto agent_impl =
        std::make_unique<Impl>(model_config, agent_config, default_generation, std::move(backend));
    return std::unique_ptr<Agent>(
        new Agent(model_config, agent_config, default_generation, std::move(agent_impl)));
}

Agent::Agent(ModelConfig model_config, AgentConfig agent_config,
             GenerationOptions default_generation, std::unique_ptr<Impl> impl)
    : model_config_(std::move(model_config)), agent_config_(agent_config),
      default_generation_options_(std::move(default_generation)), impl_(std::move(impl)) {}

Agent::~Agent() = default;

RequestHandle<TextResponse> Agent::chat(std::string_view user_message,
                                        GenerationOverride generation,
                                        AsyncTokenCallback callback) {
    return impl_->runtime.chat(user_message, generation, std::move(callback));
}

RequestHandle<TextResponse> Agent::chat(MessageView message, GenerationOverride generation,
                                        AsyncTokenCallback callback) {
    return impl_->runtime.chat(message, generation, std::move(callback));
}

RequestHandle<TextResponse> Agent::complete(ConversationView messages,
                                            GenerationOverride generation,
                                            AsyncTokenCallback callback) {
    return impl_->runtime.complete(messages, generation, std::move(callback));
}

RequestHandle<ExtractionResponse> Agent::extract_stateful(const nlohmann::json& output_schema,
                                                          MessageView message,
                                                          GenerationOverride generation,
                                                          AsyncTokenCallback callback) {
    return impl_->runtime.extract(output_schema, message, generation, std::move(callback));
}

RequestHandle<ExtractionResponse> Agent::extract(const nlohmann::json& output_schema,
                                                 ConversationView messages,
                                                 GenerationOverride generation,
                                                 AsyncTokenCallback callback) {
    return impl_->runtime.extract(output_schema, messages, generation, std::move(callback));
}

void Agent::cancel(RequestId id) {
    impl_->runtime.cancel(id);
}

void Agent::set_system_prompt(std::string_view prompt) {
    impl_->runtime.set_system_prompt(prompt);
}

Expected<void> Agent::try_set_system_prompt(std::string_view prompt) {
    return impl_->runtime.try_set_system_prompt(prompt);
}

Expected<void> Agent::set_system_prompt(std::string_view prompt, std::chrono::nanoseconds timeout) {
    return impl_->runtime.set_system_prompt(prompt, timeout);
}

Expected<void> Agent::add_system_message(std::string_view message) {
    return impl_->runtime.add_system_message(message);
}

Expected<void> Agent::add_system_message(std::string_view message,
                                         std::chrono::nanoseconds timeout) {
    return impl_->runtime.add_system_message(message, timeout);
}

void Agent::stop() {
    impl_->runtime.stop();
}

bool Agent::is_running() const noexcept {
    return impl_->runtime.is_running();
}

HistorySnapshot Agent::get_history() const {
    return impl_->runtime.get_history();
}

Expected<HistorySnapshot> Agent::try_get_history() const {
    return impl_->runtime.try_get_history();
}

Expected<HistorySnapshot> Agent::get_history(std::chrono::nanoseconds timeout) const {
    return impl_->runtime.get_history(timeout);
}

void Agent::clear_history() {
    impl_->runtime.clear_history();
}

Expected<void> Agent::try_clear_history() {
    return impl_->runtime.try_clear_history();
}

Expected<void> Agent::clear_history(std::chrono::nanoseconds timeout) {
    return impl_->runtime.clear_history(timeout);
}

Expected<void> Agent::register_tool(tools::ToolDefinition definition,
                                    std::optional<std::chrono::nanoseconds> timeout) {
    return impl_->runtime.register_tool(std::move(definition), timeout);
}

Expected<void> Agent::register_tool(std::string_view name, std::string_view description,
                                    nlohmann::json schema, tools::ToolHandler handler,
                                    std::optional<std::chrono::nanoseconds> timeout) {
    std::string tool_name{name};
    std::string tool_description{description};
    auto definition =
        tools::make_tool_definition(tool_name, tool_description, schema, std::move(handler));
    if (!definition) {
        return std::unexpected(definition.error());
    }
    return register_tool(std::move(*definition), timeout);
}

Expected<void> Agent::register_tools(std::vector<tools::ToolDefinition> definitions,
                                     std::optional<std::chrono::nanoseconds> timeout) {
    return impl_->runtime.register_tools(std::move(definitions), timeout);
}

size_t Agent::tool_count() const noexcept {
    return impl_->runtime.tool_count();
}

} // namespace zoo
