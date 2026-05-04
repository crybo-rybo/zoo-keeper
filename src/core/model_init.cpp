/**
 * @file model_init.cpp
 * @brief Backend initialization and tokenization for `zoo::core::Model`.
 */

#include "core/model_impl.hpp"
#include "zoo/core/model.hpp"

#include <chat.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <llama.h>
#include <log.h>

namespace zoo::core {

Expected<void> initialize_model(Model::Impl& impl) {
    initialize_model_backend();

    llama_log_set(
        [](enum ggml_log_level level, const char* text, void*) {
            if (level >= GGML_LOG_LEVEL_WARN) {
                std::fprintf(stderr, "%s", text);
            }
        },
        nullptr);
    common_log_pause(common_log_main());

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = impl.loaded_.model_config.n_gpu_layers;
    model_params.use_mmap = impl.loaded_.model_config.use_mmap;
    model_params.use_mlock = impl.loaded_.model_config.use_mlock;

    auto llama_model = LlamaModelHandle(
        llama_model_load_from_file(impl.loaded_.model_config.model_path.c_str(), model_params));
    if (!llama_model) {
        return std::unexpected(
            Error{ErrorCode::ModelLoadFailed,
                  "Failed to load model from path: " + impl.loaded_.model_config.model_path});
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(impl.loaded_.model_config.context_size);
    ctx_params.n_batch = static_cast<uint32_t>(impl.loaded_.model_config.context_size);
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

    auto sampler = create_sampler_chain(impl);
    if (!sampler) {
        return std::unexpected(
            Error{ErrorCode::BackendInitFailed, "Failed to create sampler chain"});
    }

    // Initialize the Jinja2 chat template system from model metadata.
    auto chat_tmpls =
        ChatTemplatesHandle(common_chat_templates_init(llama_model.get(), "").release());
    if (!chat_tmpls || common_chat_templates_source(chat_tmpls.get()).empty()) {
        return std::unexpected(
            Error{ErrorCode::TemplateRenderFailed, "Model has no chat template"});
    }

    impl.session_.prompt_state = {};

    impl.loaded_.llama_model = std::move(llama_model);
    impl.session_.ctx = std::move(ctx);
    impl.session_.sampler = std::move(sampler);
    impl.loaded_.context_size = context_size;
    impl.loaded_.vocab = vocab;
    impl.loaded_.chat_templates = std::move(chat_tmpls);

    return {};
}

Expected<std::vector<int>> tokenize(Model::Impl& impl, std::string_view text) {
    static_assert(sizeof(int) == sizeof(llama_token));
    static_assert(alignof(int) == alignof(llama_token));

    const bool is_first =
        llama_memory_seq_pos_max(llama_get_memory(impl.session_.ctx.get()), 0) == -1;
    const int32_t raw =
        llama_tokenize(impl.loaded_.vocab, text.data(), text.length(), nullptr, 0, is_first, true);
    if (raw == INT32_MIN) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization overflow"});
    }
    const int n = (raw < 0) ? -raw : raw;
    if (n == 0) {
        return std::vector<int>{};
    }

    impl.session_.token_buffer.resize(static_cast<size_t>(n));
    if (llama_tokenize(impl.loaded_.vocab, text.data(), text.length(),
                       reinterpret_cast<llama_token*>(impl.session_.token_buffer.data()),
                       impl.session_.token_buffer.size(), is_first, true) < 0) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization failed"});
    }
    return impl.session_.token_buffer;
}

} // namespace zoo::core
