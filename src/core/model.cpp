/**
 * @file model.cpp
 * @brief Core lifecycle implementation for the llama.cpp-backed `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"

#include <chat.h>
#include <common.h>
#include <llama.h>
#include <mutex>

// ToolCallingState must be complete before ~Model() = default is instantiated.
#include "zoo/core/model_tool_calling_state.hpp"

namespace zoo::core {

static std::once_flag g_init_flag;

void Model::initialize_global() {
    std::call_once(g_init_flag, []() {
        llama_backend_init();
        ggml_backend_load_all();
    });
}

Model::Model(ModelConfig model_config, GenerationOptions default_generation)
    : model_config_(std::move(model_config)),
      default_generation_options_(std::move(default_generation)),
      active_sampling_(default_generation_options_.sampling) {}

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
