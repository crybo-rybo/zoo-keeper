/**
 * @file gguf_inspector.cpp
 * @brief GGUF file metadata inspection and hardware-aware auto-configuration.
 */

#include "zoo/core/gguf_inspector.hpp"
#include "core/backend_init.hpp"
#include "zoo/core/system_probe.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <gguf.h>
#include <llama.h>
#include <log.h>
#include <memory>

namespace zoo::core {

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

struct GgufContextDeleter {
    void operator()(gguf_context* ctx) const noexcept {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

struct VocabOnlyModelDeleter {
    void operator()(llama_model* model) const noexcept {
        if (model != nullptr) {
            llama_model_free(model);
        }
    }
};

using GgufContextHandle = std::unique_ptr<gguf_context, GgufContextDeleter>;
using VocabOnlyModelHandle = std::unique_ptr<llama_model, VocabOnlyModelDeleter>;

GgufContextHandle make_gguf_context(const char* path, gguf_init_params params) {
    return GgufContextHandle(gguf_init_from_file(path, params));
}

VocabOnlyModelHandle load_vocab_only_model(const char* path) {
    auto params = llama_model_default_params();
    params.vocab_only = true;
    return VocabOnlyModelHandle(llama_model_load_from_file(path, params));
}

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

std::string gguf_string_value(const gguf_context* ctx, int64_t index) {
    if (const char* s = gguf_get_val_str(ctx, index)) {
        return s;
    }
    return {};
}

std::string gguf_u32_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_u32(ctx, index));
}

std::string gguf_i32_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_i32(ctx, index));
}

std::string gguf_f32_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_f32(ctx, index));
}

std::string gguf_u64_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_u64(ctx, index));
}

std::string gguf_i64_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_i64(ctx, index));
}

std::string gguf_f64_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_f64(ctx, index));
}

std::string gguf_bool_value(const gguf_context* ctx, int64_t index) {
    return gguf_get_val_bool(ctx, index) ? "true" : "false";
}

std::string gguf_u8_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_u8(ctx, index));
}

std::string gguf_i8_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_i8(ctx, index));
}

std::string gguf_u16_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_u16(ctx, index));
}

std::string gguf_i16_value(const gguf_context* ctx, int64_t index) {
    return std::to_string(gguf_get_val_i16(ctx, index));
}

std::string format_gguf_metadata_value(const gguf_context* ctx, int64_t index) {
    using Reader = std::string (*)(const gguf_context*, int64_t);
    struct Formatter {
        gguf_type type;
        Reader read;
    };

    static constexpr std::array<Formatter, 12> kFormatters{{
        {GGUF_TYPE_STRING, gguf_string_value},
        {GGUF_TYPE_UINT32, gguf_u32_value},
        {GGUF_TYPE_INT32, gguf_i32_value},
        {GGUF_TYPE_FLOAT32, gguf_f32_value},
        {GGUF_TYPE_UINT64, gguf_u64_value},
        {GGUF_TYPE_INT64, gguf_i64_value},
        {GGUF_TYPE_FLOAT64, gguf_f64_value},
        {GGUF_TYPE_BOOL, gguf_bool_value},
        {GGUF_TYPE_UINT8, gguf_u8_value},
        {GGUF_TYPE_INT8, gguf_i8_value},
        {GGUF_TYPE_UINT16, gguf_u16_value},
        {GGUF_TYPE_INT16, gguf_i16_value},
    }};

    const auto type = gguf_get_kv_type(ctx, index);
    const auto it = std::find_if(kFormatters.begin(), kFormatters.end(),
                                 [type](const Formatter& item) { return item.type == type; });
    if (it == kFormatters.end()) {
        return "<array>";
    }
    return it->read(ctx, index);
}

void collect_all_metadata(const gguf_context* ctx, std::map<std::string, std::string>& metadata) {
    const int64_t n_kv = gguf_get_n_kv(ctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char* key = gguf_get_key(ctx, i);
        if (!key) {
            continue;
        }

        metadata[key] = format_gguf_metadata_value(ctx, i);
    }
}

constexpr uint64_t kRamOverheadBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr int kFallbackTrainingContext = 8192;
constexpr int kContextHardCap = 32768;
constexpr int kContextFloor = 512;
constexpr int kDefaultBatchCap = 2048;

uint64_t per_token_kv_bytes(const ModelInfo& info) {
    if (info.layer_count <= 0 || info.embedding_dim <= 0) {
        return 0;
    }
    // KV per token = 2 (K+V) * layers * kv_dim * 2 (fp16) = 4 * layers * kv_dim.
    // For GQA, kv_dim = head_dim * kv_head_count, where head_dim = embedding_dim / head_count.
    // For MHA (or unknown head metadata), kv_dim = embedding_dim.
    uint64_t kv_dim = static_cast<uint64_t>(info.embedding_dim);
    if (info.head_count > 0 && info.kv_head_count > 0 && info.kv_head_count <= info.head_count) {
        const uint64_t head_dim =
            static_cast<uint64_t>(info.embedding_dim) / static_cast<uint64_t>(info.head_count);
        kv_dim = head_dim * static_cast<uint64_t>(info.kv_head_count);
    }
    return static_cast<uint64_t>(info.layer_count) * kv_dim * 4;
}

int compute_context_size(const ModelInfo& info, const SystemInfo& sys) {
    const int training_ctx =
        info.context_length > 0 ? info.context_length : kFallbackTrainingContext;

    int ctx_from_ram = kContextHardCap;
    const uint64_t per_token_kv = per_token_kv_bytes(info);
    if (per_token_kv > 0) {
        const uint64_t reserved = info.file_size_bytes + kRamOverheadBytes;
        const uint64_t kv_budget =
            sys.total_ram_bytes > reserved ? (sys.total_ram_bytes - reserved) / 2 : 0;
        if (kv_budget > 0) {
            const uint64_t derived = kv_budget / per_token_kv;
            ctx_from_ram = derived > static_cast<uint64_t>(kContextHardCap)
                               ? kContextHardCap
                               : static_cast<int>(derived);
        } else {
            ctx_from_ram = kContextFloor;
        }
    }

    int chosen = std::min({training_ctx, ctx_from_ram, kContextHardCap});
    return std::max(chosen, kContextFloor);
}

int compute_n_gpu_layers(const ModelInfo& info, const SystemInfo& sys) {
    if (!sys.gpu_offload_supported || sys.gpus.empty()) {
        return 0;
    }
    uint64_t total_vram = 0;
    for (const auto& gpu : sys.gpus) {
        total_vram += gpu.total_vram_bytes;
    }
    if (total_vram == 0 || info.file_size_bytes == 0) {
        return 0;
    }
    if (total_vram >= info.file_size_bytes + info.file_size_bytes / 5) {
        return -1; // Full offload with ~20% headroom.
    }
    if (info.layer_count <= 0) {
        return 0;
    }
    const uint64_t bytes_per_layer = info.file_size_bytes / static_cast<uint64_t>(info.layer_count);
    if (bytes_per_layer == 0) {
        return 0;
    }
    const int64_t fits = static_cast<int64_t>(total_vram / bytes_per_layer) - 2;
    return fits > 0 ? static_cast<int>(fits) : 0;
}

} // namespace

Expected<ModelInfo> GgufInspector::inspect(const std::string& file_path) {
    if (!std::filesystem::exists(file_path)) {
        return std::unexpected(Error{ErrorCode::GgufReadFailed, "File not found: " + file_path});
    }

    // Phase 1: Raw GGUF read for KV metadata.
    gguf_init_params gguf_params{};
    gguf_params.no_alloc = true;
    gguf_params.ctx = nullptr;

    auto gguf_ctx = make_gguf_context(file_path.c_str(), gguf_params);
    if (!gguf_ctx) {
        return std::unexpected(
            Error{ErrorCode::GgufReadFailed, "Failed to parse GGUF file: " + file_path});
    }

    ModelInfo info;
    info.file_path = std::filesystem::absolute(file_path).string();
    info.name = read_gguf_string(gguf_ctx.get(), "general.name");
    info.architecture = read_gguf_string(gguf_ctx.get(), "general.architecture");

    // Try architecture-specific context length keys, then generic.
    const std::string arch = info.architecture;
    if (!arch.empty()) {
        info.context_length =
            read_gguf_u32_as_i32(gguf_ctx.get(), (arch + ".context_length").c_str());
        if (info.context_length == 0) {
            info.context_length =
                read_gguf_u32_as_i32(gguf_ctx.get(), (arch + ".block_count").c_str());
            // block_count isn't context_length; reset if we got it wrong.
            info.context_length = 0;
        }
    }
    if (info.context_length == 0) {
        info.context_length = read_gguf_u32_as_i32(gguf_ctx.get(), "llm.context_length");
    }

    collect_all_metadata(gguf_ctx.get(), info.metadata);

    // Phase 2: Vocab-only model load for derived statistics.
    ensure_backend_initialized();

    // Suppress llama.cpp log output during inspection.
    ScopedLlamaLogSilencer silenced_logs;

    auto llama_model = load_vocab_only_model(file_path.c_str());
    if (llama_model) {
        info.parameter_count = llama_model_n_params(llama_model.get());
        info.file_size_bytes = llama_model_size(llama_model.get());
        info.embedding_dim = static_cast<int32_t>(llama_model_n_embd(llama_model.get()));
        info.layer_count = static_cast<int32_t>(llama_model_n_layer(llama_model.get()));
        // n_head / n_head_kv index hparams by layer; only safe when at least one layer exists
        // (vocab-only fixtures may have layer_count == 0).
        if (info.layer_count > 0) {
            info.head_count = static_cast<int32_t>(llama_model_n_head(llama_model.get()));
            info.kv_head_count = static_cast<int32_t>(llama_model_n_head_kv(llama_model.get()));
        }

        // Get the training context length if we couldn't get it from raw GGUF.
        if (info.context_length == 0) {
            info.context_length = static_cast<int32_t>(llama_model_n_ctx_train(llama_model.get()));
        }

        char desc_buf[256] = {};
        llama_model_desc(llama_model.get(), desc_buf, sizeof(desc_buf));
        info.description = desc_buf;
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

Expected<ModelConfig> GgufInspector::auto_configure(const ModelInfo& info, const SystemInfo& sys) {
    if (info.file_path.empty()) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelPath, "ModelInfo has no file_path set"});
    }

    ModelConfig config;
    config.model_path = info.file_path;
    config.context_size = compute_context_size(info, sys);
    config.n_batch = std::min(config.context_size, kDefaultBatchCap);
    config.n_gpu_layers = compute_n_gpu_layers(info, sys);
    config.use_mmap = true;
    config.use_mlock = sys.total_ram_bytes >= info.file_size_bytes * 5 / 2;
    return config;
}

Expected<ModelConfig> GgufInspector::auto_configure(const ModelInfo& info) {
    auto sys = SystemProbe::probe();
    if (!sys) {
        return std::unexpected(sys.error());
    }
    return auto_configure(info, *sys);
}

Expected<ModelConfig> GgufInspector::auto_configure(const std::string& file_path) {
    auto info = inspect(file_path);
    if (!info) {
        return std::unexpected(info.error());
    }
    return auto_configure(*info);
}

} // namespace zoo::core
