/**
 * @file model.cpp
 * @brief Core lifecycle implementation for the llama.cpp-backed `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"
#include "core/backend_init.hpp"
#include "core/model_impl.hpp"

#include <llama.h>

namespace zoo::core {

void Model::initialize_global() {
    ensure_backend_initialized();
}

Model::Model(ModelConfig model_config, GenerationOptions default_generation)
    : impl_(std::make_unique<Impl>(std::move(model_config), std::move(default_generation))) {}

void Model::LlamaModelDeleter::operator()(llama_model* model) const noexcept {
    if (model) {
        llama_model_free(model);
    }
}

void Model::LlamaContextDeleter::operator()(llama_context* context) const noexcept {
    if (context) {
        llama_free(context);
    }
}

void Model::LlamaSamplerDeleter::operator()(llama_sampler* sampler) const noexcept {
    if (sampler) {
        llama_sampler_free(sampler);
    }
}

void Model::ChatTemplatesDeleter::operator()(common_chat_templates* tmpls) const noexcept {
    if (tmpls) {
        common_chat_templates_free(tmpls);
    }
}

Model::~Model() = default;

bool Model::has_tool_calling() const noexcept {
    return impl_->session_.sampler_policy.is_native_tool_call();
}

bool Model::has_schema_grammar() const noexcept {
    return impl_->session_.sampler_policy.is_schema();
}

const ModelConfig& Model::model_config() const noexcept {
    return impl_->loaded_.model_config;
}

const GenerationOptions& Model::default_generation_options() const noexcept {
    return impl_->loaded_.default_generation_options;
}

Expected<std::unique_ptr<Model>> Model::load(const ModelConfig& model_config,
                                             const GenerationOptions& default_generation) {
    if (auto result = model_config.validate(); !result) {
        return std::unexpected(result.error());
    }
    if (auto result = default_generation.validate(); !result) {
        return std::unexpected(result.error());
    }

    auto model = std::unique_ptr<Model>(new Model(model_config, default_generation));
    if (auto result = model->initialize(); !result) {
        return std::unexpected(result.error());
    }

    return model;
}

} // namespace zoo::core
