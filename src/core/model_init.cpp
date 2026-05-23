/**
 * @file model_init.cpp
 * @brief Backend initialization and tokenization for `zoo::core::Model`.
 */

#include "core/model_impl.hpp"
#include "zoo/core/model.hpp"

#include "core/gpu_fit.hpp"

#include <array>
#include <chat.h>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <fit.h>
#include <llama.h>
#include <log.h>
#include <mutex>
#include <vector>

namespace zoo::core {

namespace {

constexpr size_t kGpuFitMarginBytes = 1024ULL * 1024ULL * 1024ULL;
constexpr uint32_t kGpuFitMinimumContext = 4096;

std::mutex& gpu_fit_mutex() {
    static std::mutex mutex;
    return mutex;
}

Expected<void> verify_gpu_memory_fit(const Model::Impl& impl,
                                     const llama_model_params& requested_model_params,
                                     const llama_context_params& requested_context_params) {
    if (impl.loaded_.model_config.n_gpu_layers == 0 || !llama_supports_gpu_offload()) {
        return {};
    }

    llama_model_params fitted_model_params = requested_model_params;
    llama_context_params fitted_context_params = requested_context_params;

    const auto max_devices = std::max<size_t>(llama_max_devices(), 1);
    std::vector<float> tensor_split(max_devices, 0.0f);
    std::vector<size_t> margins(max_devices, kGpuFitMarginBytes);
    std::vector<llama_model_tensor_buft_override> tensor_overrides(
        std::max<size_t>(llama_max_tensor_buft_overrides(), 1), {nullptr, nullptr});

    common_params_fit_status status = COMMON_PARAMS_FIT_STATUS_ERROR;
    {
        std::lock_guard<std::mutex> lock(gpu_fit_mutex());
        status =
            common_fit_params(impl.loaded_.model_config.model_path.c_str(), &fitted_model_params,
                              &fitted_context_params, tensor_split.data(), tensor_overrides.data(),
                              margins.data(), kGpuFitMinimumContext, GGML_LOG_LEVEL_ERROR);
    }

    if (status == COMMON_PARAMS_FIT_STATUS_ERROR) {
        return std::unexpected(
            Error{ErrorCode::ModelLoadFailed, "Failed to estimate GPU memory fit for model: " +
                                                  impl.loaded_.model_config.model_path});
    }
    if (status == COMMON_PARAMS_FIT_STATUS_FAILURE ||
        !gpu_fit_preserves_requested_config(requested_model_params, fitted_model_params,
                                            requested_context_params, fitted_context_params,
                                            tensor_split, tensor_overrides)) {
        return std::unexpected(Error{
            ErrorCode::ModelLoadFailed,
            "Requested GPU offload configuration is projected to exceed available device memory",
            "Reduce n_gpu_layers or context_size, or use GgufInspector::auto_configure()."});
    }

    return {};
}

} // namespace

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

    const bool cpu_only = impl.loaded_.model_config.n_gpu_layers == 0;
    std::array<ggml_backend_dev_t, 1> cpu_only_devices{nullptr};

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = impl.loaded_.model_config.n_gpu_layers;
    model_params.use_mmap = impl.loaded_.model_config.use_mmap;
    model_params.use_mlock = impl.loaded_.model_config.use_mlock;
    if (cpu_only) {
        // llama.cpp treats nullptr as "all devices"; use an explicit empty list for CPU-only.
        model_params.devices = cpu_only_devices.data();
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(impl.loaded_.model_config.context_size);
    ctx_params.n_batch = static_cast<uint32_t>(impl.loaded_.model_config.n_batch);
    ctx_params.n_ubatch = 512;
    ctx_params.n_threads = -1;
    ctx_params.n_threads_batch = -1;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (cpu_only) {
        ctx_params.offload_kqv = false;
        ctx_params.op_offload = false;
    }
    // F16 uses more memory than Q8, but avoids KV dequant overhead in decode.
    ctx_params.type_k = GGML_TYPE_F16;
    ctx_params.type_v = GGML_TYPE_F16;

    if (auto fit = verify_gpu_memory_fit(impl, model_params, ctx_params); !fit) {
        return std::unexpected(fit.error());
    }

    auto llama_model = LlamaModelHandle(
        llama_model_load_from_file(impl.loaded_.model_config.model_path.c_str(), model_params));
    if (!llama_model) {
        return std::unexpected(
            Error{ErrorCode::ModelLoadFailed,
                  "Failed to load model from path: " + impl.loaded_.model_config.model_path});
    }

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
    if (text.size() > static_cast<size_t>(INT32_MAX - 8)) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization overflow"});
    }

    const bool is_first =
        llama_memory_seq_pos_max(llama_get_memory(impl.session_.ctx.get()), 0) == -1;

    const int32_t text_len = static_cast<int32_t>(text.size());
    // This usually avoids a count-only tokenization pass while keeping the
    // exact-size retry for unusual tokenizers.
    impl.session_.token_buffer.resize(static_cast<size_t>(text_len + 8));
    int32_t n =
        llama_tokenize(impl.loaded_.vocab, text.data(), text_len,
                       reinterpret_cast<llama_token*>(impl.session_.token_buffer.data()),
                       static_cast<int32_t>(impl.session_.token_buffer.size()), is_first, true);
    if (n == INT32_MIN) {
        return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization overflow"});
    }
    if (n < 0) {
        n = -n;
        impl.session_.token_buffer.resize(static_cast<size_t>(n));
        const int32_t filled = llama_tokenize(
            impl.loaded_.vocab, text.data(), text_len,
            reinterpret_cast<llama_token*>(impl.session_.token_buffer.data()), n, is_first, true);
        if (filled < 0) {
            return std::unexpected(Error{ErrorCode::TokenizationFailed, "Tokenization failed"});
        }
        n = filled;
    }

    if (n == 0) {
        impl.session_.token_buffer.clear();
        return std::vector<int>{};
    }

    impl.session_.token_buffer.resize(static_cast<size_t>(n));
    return impl.session_.token_buffer;
}

} // namespace zoo::core
