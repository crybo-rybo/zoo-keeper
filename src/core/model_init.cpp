/**
 * @file model_init.cpp
 * @brief Backend initialization and tokenization for `zoo::core::Model`.
 */

#include "zoo/core/model.hpp"

#include <chat.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <llama.h>
#include <log.h>

namespace zoo::core {

Expected<void> Model::initialize() {
    initialize_global();

    llama_log_set(
        [](enum ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::fprintf(stderr, "%s", text);
            }
        },
        nullptr);
    common_log_pause(common_log_main());

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.n_gpu_layers;
    model_params.use_mmap = config_.use_mmap;
    model_params.use_mlock = config_.use_mlock;

    auto llama_model =
        LlamaModelHandle(llama_model_load_from_file(config_.model_path.c_str(), model_params));
    if (!llama_model) {
        return std::unexpected(Error{ErrorCode::ModelLoadFailed,
                                     "Failed to load model from path: " + config_.model_path});
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(config_.context_size);
    ctx_params.n_batch = static_cast<uint32_t>(config_.context_size);
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = -1;
    ctx_params.n_threads_batch = -1;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    // Use 8-bit KV cache to reduce memory footprint vs the upstream F16 default.
    ctx_params.type_k = GGML_TYPE_Q8_0;
    ctx_params.type_v = GGML_TYPE_Q8_0;

    auto ctx = LlamaContextHandle(llama_init_from_model(llama_model.get(), ctx_params));
    if (!ctx) {
        return std::unexpected(
            Error{ErrorCode::ContextCreationFailed, "Failed to create llama context"});
    }

    const int context_size = static_cast<int>(llama_n_ctx(ctx.get()));

    const llama_vocab* vocab = llama_model_get_vocab(llama_model.get());
    if (!vocab) {
        return std::unexpected(
            Error{ErrorCode::BackendInitFailed, "Failed to get model vocabulary"});
    }

    auto sampler = create_sampler_chain();
    if (!sampler) {
        return std::unexpected(
            Error{ErrorCode::BackendInitFailed, "Failed to create sampler chain"});
    }

    // Initialize the Jinja2 chat template system from model metadata.
    auto chat_tmpls =
        ChatTemplatesHandle(common_chat_templates_init(llama_model.get(), "").release());
    if (!chat_tmpls || !common_chat_templates_source(chat_tmpls.get())) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "Model has no chat template"});
    }

    prompt_state_ = {};

    llama_model_ = std::move(llama_model);
    ctx_ = std::move(ctx);
    sampler_ = std::move(sampler);
    context_size_ = context_size;
    vocab_ = vocab;
    chat_templates_ = std::move(chat_tmpls);

    return {};
}

Expected<std::vector<int>> Model::tokenize(const std::string& text) {
    static_assert(sizeof(int) == sizeof(llama_token));
    static_assert(alignof(int) == alignof(llama_token));

    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx_.get()), 0) == -1;
    const int32_t raw =
        llama_tokenize(vocab_, text.c_str(), text.length(), nullptr, 0, is_first, true);
    if (raw == INT32_MIN) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization overflow"});
    }
    const int n = (raw < 0) ? -raw : raw;
    if (n == 0) {
        return std::vector<int>{};
    }

    std::vector<int> tokens(n);
    if (llama_tokenize(vocab_, text.c_str(), text.length(),
                       reinterpret_cast<llama_token*>(tokens.data()), tokens.size(), is_first,
                       true) < 0) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization failed"});
    }
    return tokens;
}

} // namespace zoo::core
