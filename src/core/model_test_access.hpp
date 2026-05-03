/**
 * @file model_test_access.hpp
 * @brief Source-local access shim for Model private tests.
 */

#pragma once

#include "core/model_impl.hpp"

#include <utility>

namespace zoo::core {

struct ModelTestAccess {
    using GrammarMode = Model::Impl::GrammarMode;
    using ToolCallingState = Model::Impl::ToolCallingState;

    static std::unique_ptr<Model> make(ModelConfig model_config,
                                       GenerationOptions default_generation = GenerationOptions{}) {
        return std::unique_ptr<Model>(
            new Model(std::move(model_config), std::move(default_generation)));
    }

    static auto& chat_templates(Model& model) {
        return model.impl_->loaded_.chat_templates;
    }

    static auto& tool_state(Model& model) {
        return model.impl_->session_.tool_state;
    }

    static auto& tool_grammar_str(Model& model) {
        return model.impl_->session_.tool_grammar_str;
    }

    static auto& grammar_mode(Model& model) {
        return model.impl_->session_.grammar_mode;
    }

    static auto& messages(Model& model) {
        return model.impl_->session_.messages;
    }

    static int estimated_tokens(const Model& model) {
        return model.impl_->session_.estimated_tokens;
    }

    static int estimate_message_tokens(Model& model, const Message& message) {
        return model.estimate_message_tokens(message);
    }

    static void rollback_last_message(Model& model) {
        model.rollback_last_message();
    }

    static Expected<std::string> render_prompt_delta(Model& model) {
        return model.render_prompt_delta();
    }
};

} // namespace zoo::core
