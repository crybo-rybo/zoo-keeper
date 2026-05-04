/**
 * @file model_test_access.hpp
 * @brief Source-local access shim for Model private tests.
 */

#pragma once

#include "core/model_impl.hpp"

#include <utility>

namespace zoo::core {

struct ModelTestAccess {
    using SamplerPolicy = Model::Impl::SamplerPolicy;
    using GrammarMode = SamplerPolicy::Mode;
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

    static const auto& sampler_policy(const Model& model) {
        return model.impl_->session_.sampler_policy;
    }

    static void set_sampler_policy(Model& model, SamplerPolicy policy) {
        model.impl_->session_.sampler_policy = std::move(policy);
    }

    static auto& messages(Model& model) {
        return model.impl_->session_.messages;
    }

    static int estimated_tokens(const Model& model) {
        return model.impl_->session_.estimated_tokens;
    }

    static int estimate_message_tokens(Model& model, const Message& message) {
        return zoo::core::estimate_message_tokens(*model.impl_, message);
    }

    static void rollback_last_message(Model& model) {
        zoo::core::rollback_last_message(*model.impl_);
    }

    static Expected<std::string> render_prompt_delta(Model& model) {
        return zoo::core::render_prompt_delta(*model.impl_);
    }

    static GenerationOptions resolve_generation_options(Model& model,
                                                        const GenerationOptions& overrides) {
        return zoo::core::resolve_generation_options(*model.impl_, GenerationOverride(overrides));
    }

    static GenerationOptions resolve_generation_options(Model& model,
                                                        GenerationOverride generation) {
        return zoo::core::resolve_generation_options(*model.impl_, generation);
    }
};

} // namespace zoo::core
