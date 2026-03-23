/**
 * @file backend_model.cpp
 * @brief Concrete internal adapter from `core::Model` to `AgentBackend`.
 */

#include "zoo/internal/agent/backend_model.hpp"

#include "zoo/core/model.hpp"

namespace zoo::internal::agent {

namespace {

class ModelBackend final : public AgentBackend {
  public:
    explicit ModelBackend(std::unique_ptr<core::Model> model) : model_(std::move(model)) {}

    Expected<void> add_message(MessageView message) override {
        return model_->add_message(message);
    }

    Expected<GenerationResult> generate_from_history(const GenerationOptions& options,
                                                     TokenCallback on_token,
                                                     CancellationCallback should_cancel) override {
        auto result = model_->generate_from_history(options, on_token, should_cancel);
        if (!result) {
            return std::unexpected(result.error());
        }

        return GenerationResult{std::move(result->text), result->prompt_tokens,
                                result->tool_call_detected, std::move(result->parsed_content),
                                std::move(result->tool_calls)};
    }

    void finalize_response() override {
        model_->finalize_response();
    }
    void set_system_prompt(std::string_view prompt) override {
        model_->set_system_prompt(prompt);
    }
    HistorySnapshot get_history() const override {
        return model_->get_history();
    }
    void clear_history() override {
        model_->clear_history();
    }
    void replace_history(HistorySnapshot snapshot) override {
        model_->replace_history(std::move(snapshot));
    }
    HistorySnapshot swap_history(HistorySnapshot snapshot) override {
        return model_->swap_history(std::move(snapshot));
    }

    void trim_history(size_t max_non_system_messages) override {
        model_->trim_history(max_non_system_messages);
    }

    bool set_tool_calling(const std::vector<CoreToolInfo>& tools) override {
        return model_->set_tool_calling(tools);
    }

    bool set_schema_grammar(const std::string& grammar_str) override {
        return model_->set_schema_grammar(grammar_str);
    }

    void clear_tool_grammar() override {
        model_->clear_tool_grammar();
    }

    ParsedToolResponse parse_tool_response(std::string_view text) const override {
        auto parsed = model_->parse_tool_response(text);
        return ParsedToolResponse{std::move(parsed.content), std::move(parsed.tool_calls)};
    }

    const char* tool_calling_format_name() const noexcept override {
        return model_->tool_calling_format_name();
    }

  private:
    std::unique_ptr<core::Model> model_;
};

} // namespace

std::unique_ptr<AgentBackend> make_model_backend(std::unique_ptr<core::Model> model) {
    return std::make_unique<ModelBackend>(std::move(model));
}

} // namespace zoo::internal::agent
