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
                                        const GenerationOptions& options,
                                        AsyncTextCallback callback) {
    return impl_->runtime.chat(user_message, options, std::move(callback));
}

RequestHandle<TextResponse> Agent::chat(MessageView message, const GenerationOptions& options,
                                        AsyncTextCallback callback) {
    return impl_->runtime.chat(message, options, std::move(callback));
}

RequestHandle<TextResponse> Agent::complete(ConversationView messages,
                                            const GenerationOptions& options,
                                            AsyncTextCallback callback) {
    return impl_->runtime.complete(messages, options, std::move(callback));
}

RequestHandle<ExtractionResponse> Agent::extract(const nlohmann::json& output_schema,
                                                 std::string_view user_message,
                                                 const GenerationOptions& options,
                                                 AsyncTextCallback callback) {
    return impl_->runtime.extract(output_schema, user_message, options, std::move(callback));
}

RequestHandle<ExtractionResponse> Agent::extract(const nlohmann::json& output_schema,
                                                 MessageView message,
                                                 const GenerationOptions& options,
                                                 AsyncTextCallback callback) {
    return impl_->runtime.extract(output_schema, message, options, std::move(callback));
}

RequestHandle<ExtractionResponse> Agent::extract(const nlohmann::json& output_schema,
                                                 ConversationView messages,
                                                 const GenerationOptions& options,
                                                 AsyncTextCallback callback) {
    return impl_->runtime.extract(output_schema, messages, options, std::move(callback));
}

void Agent::cancel(RequestId id) {
    impl_->runtime.cancel(id);
}

void Agent::set_system_prompt(std::string_view prompt) {
    impl_->runtime.set_system_prompt(prompt);
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

Expected<void> Agent::register_tools(std::vector<tools::ToolDefinition> definitions) {
    return impl_->runtime.register_tools(std::move(definitions));
}

size_t Agent::tool_count() const noexcept {
    return impl_->runtime.tool_count();
}

} // namespace zoo
