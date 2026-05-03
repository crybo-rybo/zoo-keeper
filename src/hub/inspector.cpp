/**
 * @file inspector.cpp
 * @brief GGUF file metadata inspection and auto-configuration.
 *
 * This file intentionally calls llama.cpp APIs directly (gguf.h and llama.h)
 * outside the core/model*.cpp convention. This is an approved exception for
 * the hub layer, which performs metadata-only reads without full model loading.
 */

#include "zoo/hub/inspector.hpp"
#include "core/backend_init.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <gguf.h>
#include <llama.h>
#include <log.h>

namespace zoo::hub {

namespace {

class ScopedLlamaLogSilencer {
  public:
    ScopedLlamaLogSilencer() {
        llama_log_get(&original_callback_, &original_user_data_);
        llama_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
    }

    ~ScopedLlamaLogSilencer() {
        llama_log_set(original_callback_, original_user_data_);
    }

  private:
    ggml_log_callback original_callback_ = nullptr;
    void* original_user_data_ = nullptr;
};

std::string read_gguf_string(const gguf_context* ctx, const char* key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        return {};
    }
    const char* val = gguf_get_val_str(ctx, id);
    return val ? std::string(val) : std::string{};
}

int32_t read_gguf_u32_as_i32(const gguf_context* ctx, const char* key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return 0;
    }
    const auto type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_UINT32) {
        return static_cast<int32_t>(gguf_get_val_u32(ctx, id));
    }
    if (type == GGUF_TYPE_INT32) {
        return gguf_get_val_i32(ctx, id);
    }
    return 0;
}

void collect_all_metadata(const gguf_context* ctx, std::map<std::string, std::string>& metadata) {
    const int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        if (!key) {
            continue;
        }

        const auto type = gguf_get_kv_type(ctx, i);
        std::string value;
        switch (type) {
        case GGUF_TYPE_STRING:
            if (const char* s = gguf_get_val_str(ctx, i)) {
                value = s;
            }
            break;
        case GGUF_TYPE_UINT32:
            value = std::to_string(gguf_get_val_u32(ctx, i));
            break;
        case GGUF_TYPE_INT32:
            value = std::to_string(gguf_get_val_i32(ctx, i));
            break;
        case GGUF_TYPE_FLOAT32:
            value = std::to_string(gguf_get_val_f32(ctx, i));
            break;
        case GGUF_TYPE_UINT64:
            value = std::to_string(gguf_get_val_u64(ctx, i));
            break;
        case GGUF_TYPE_INT64:
            value = std::to_string(gguf_get_val_i64(ctx, i));
            break;
        case GGUF_TYPE_FLOAT64:
            value = std::to_string(gguf_get_val_f64(ctx, i));
            break;
        case GGUF_TYPE_BOOL:
            value = gguf_get_val_bool(ctx, i) ? "true" : "false";
            break;
        case GGUF_TYPE_UINT8:
            value = std::to_string(gguf_get_val_u8(ctx, i));
            break;
        case GGUF_TYPE_INT8:
            value = std::to_string(gguf_get_val_i8(ctx, i));
            break;
        case GGUF_TYPE_UINT16:
            value = std::to_string(gguf_get_val_u16(ctx, i));
            break;
        case GGUF_TYPE_INT16:
            value = std::to_string(gguf_get_val_i16(ctx, i));
            break;
        default:
            value = "<array>";
            break;
        }
        metadata[key] = std::move(value);
    }
}

} // namespace

Expected<ModelInfo> GgufInspector::inspect(const std::string& file_path) {
    if (!std::filesystem::exists(file_path)) {
        return std::unexpected(
            Error{to_error_code(HubErrorCode::GgufReadFailed), "File not found: " + file_path});
    }

    // Phase 1: Raw GGUF read for KV metadata.
    gguf_init_params gguf_params{};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;

    auto* gguf_ctx = gguf_init_from_file(file_path.c_str(), gguf_params);
    if (!gguf_ctx) {
        return std::unexpected(Error{to_error_code(HubErrorCode::GgufReadFailed),
                                     "Failed to parse GGUF file: " + file_path});
    }

    ModelInfo info;
    info.file_path = std::filesystem::absolute(file_path).string();
    info.name = read_gguf_string(gguf_ctx, "general.name");
    info.architecture = read_gguf_string(gguf_ctx, "general.architecture");

    // Try architecture-specific context length keys, then generic.
    const std::string arch = info.architecture;
    if (!arch.empty()) {
        info.context_length = read_gguf_u32_as_i32(gguf_ctx, (arch + ".context_length").c_str());
        if (info.context_length == 0) {
            info.context_length = read_gguf_u32_as_i32(gguf_ctx, (arch + ".block_count").c_str());
            // block_count isn't context_length; reset if we got it wrong.
            info.context_length = 0;
        }
    }
    if (info.context_length == 0) {
        info.context_length = read_gguf_u32_as_i32(gguf_ctx, "llm.context_length");
    }

    collect_all_metadata(gguf_ctx, info.metadata);
    gguf_free(gguf_ctx);

    // Phase 2: Vocab-only model load for derived statistics.
    core::ensure_backend_initialized();

    // Suppress llama.cpp log output during inspection.
    ScopedLlamaLogSilencer silenced_logs;

    auto model_params = llama_model_default_params();
    model_params.vocab_only = true;

    auto* llama_model = llama_model_load_from_file(file_path.c_str(), model_params);
    if (llama_model) {
        info.parameter_count = llama_model_n_params(llama_model);
        info.file_size_bytes = llama_model_size(llama_model);
        info.embedding_dim = static_cast<int32_t>(llama_model_n_embd(llama_model));
        info.layer_count = static_cast<int32_t>(llama_model_n_layer(llama_model));

        // Get the training context length if we couldn't get it from raw GGUF.
        if (info.context_length == 0) {
            info.context_length = static_cast<int32_t>(llama_model_n_ctx_train(llama_model));
        }

        char desc_buf[256] = {};
        llama_model_desc(llama_model, desc_buf, sizeof(desc_buf));
        info.description = desc_buf;

        llama_model_free(llama_model);
    }

    // Extract quantization from description (typically "7B Q4_K_M" -> "Q4_K_M").
    if (info.quantization.empty() && !info.description.empty()) {
        const auto pos = info.description.find_last_of(' ');
        if (pos != std::string::npos && pos + 1 < info.description.size()) {
            const auto candidate = info.description.substr(pos + 1);
            if (candidate.size() >= 2 &&
                (candidate[0] == 'Q' || candidate[0] == 'F' || candidate[0] == 'I')) {
                info.quantization = candidate;
            }
        }
    }

    return info;
}

Expected<ModelConfig> GgufInspector::auto_configure(const ModelInfo& info) {
    if (info.file_path.empty()) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelPath, "ModelInfo has no file_path set"});
    }

    ModelConfig config;
    config.model_path = info.file_path;

    // Cap context to 8192 by default to avoid OOM on smaller systems.
    static constexpr int kDefaultMaxContext = 8192;
    if (info.context_length > 0) {
        config.context_size = std::min(info.context_length, kDefaultMaxContext);
    } else {
        config.context_size = kDefaultMaxContext;
    }

    // Offload all layers to GPU by default (-1 = all in llama.cpp).
    config.n_gpu_layers = -1;
    config.use_mmap = true;
    config.use_mlock = false;

    return config;
}

Expected<ModelConfig> GgufInspector::auto_configure(const std::string& file_path) {
    auto info = inspect(file_path);
    if (!info) {
        return std::unexpected(info.error());
    }
    return auto_configure(*info);
}

} // namespace zoo::hub
