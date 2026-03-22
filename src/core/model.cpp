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

Model::Model(const Config& config) : config_(config) {}

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

Expected<std::unique_ptr<Model>> Model::load(const Config& config) {
    if (auto result = config.validate(); !result) {
        return std::unexpected(result.error());
    }

    auto model = std::unique_ptr<Model>(new Model(config));
    if (auto result = model->initialize(); !result) {
        return std::unexpected(result.error());
    }

    if (config.system_prompt) {
        model->set_system_prompt(*config.system_prompt);
    }

    return model;
}

} // namespace zoo::core
