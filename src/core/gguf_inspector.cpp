/**
 * @file gguf_inspector.cpp
 * @brief GGUF file metadata inspection and hardware-aware auto-configuration.
 */

#include "zoo/core/gguf_inspector.hpp"
#include "zoo/core/system_probe.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <ggml.h>
#include <gguf.h>
#include <memory>
#include <string_view>

namespace zoo::core {

namespace {

struct GgufContextDeleter {
    void operator()(gguf_context* ctx) const noexcept {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

using GgufContextHandle = std::unique_ptr<gguf_context, GgufContextDeleter>;

GgufContextHandle make_gguf_context(const char* path, gguf_init_params params) {
    return GgufContextHandle(gguf_init_from_file(path, params));
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

int32_t read_arch_gguf_u32_as_i32(const gguf_context* ctx, std::string_view architecture,
                                  std::string_view suffix) {
    if (architecture.empty()) {
        return 0;
    }
    const std::string key = std::string(architecture) + "." + std::string(suffix);
    return read_gguf_u32_as_i32(ctx, key.c_str());
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

uint64_t tensor_element_count(ggml_type type, size_t tensor_size_bytes) {
    const auto block_size = ggml_blck_size(type);
    const auto type_size = ggml_type_size(type);
    if (block_size <= 0 || type_size == 0) {
        return 0;
    }
    const auto blocks = static_cast<uint64_t>(tensor_size_bytes) / static_cast<uint64_t>(type_size);
    return blocks * static_cast<uint64_t>(block_size);
}

void read_tensor_stats(const gguf_context* ctx, ModelInfo& info) {
    std::array<uint64_t, GGML_TYPE_COUNT> bytes_per_type = {};
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const auto type = gguf_get_tensor_type(ctx, i);
        const auto tensor_size = gguf_get_tensor_size(ctx, i);
        info.file_size_bytes += static_cast<uint64_t>(tensor_size);
        info.parameter_count += tensor_element_count(type, tensor_size);
        if (type >= 0 && static_cast<size_t>(type) < GGML_TYPE_COUNT) {
            bytes_per_type[static_cast<size_t>(type)] += tensor_size;
        }
    }

    // Prefer quantized types for the display name; fall back to dominant float type.
    uint64_t best = 0;
    auto dominant = static_cast<ggml_type>(GGML_TYPE_COUNT);
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto t = static_cast<ggml_type>(i);
        if (ggml_is_quantized(t) && bytes_per_type[i] > best) {
            best = bytes_per_type[i];
            dominant = t;
        }
    }
    if (dominant == static_cast<ggml_type>(GGML_TYPE_COUNT)) {
        for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
            if (bytes_per_type[i] > best) {
                best = bytes_per_type[i];
                dominant = static_cast<ggml_type>(i);
            }
        }
    }
    if (dominant != static_cast<ggml_type>(GGML_TYPE_COUNT)) {
        std::string name = ggml_type_name(dominant);
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        info.quantization = std::move(name);
    }
}

void read_gguf_metadata(const gguf_context* ctx, ModelInfo& info) {
    info.name = read_gguf_string(ctx, "general.name");
    info.architecture = read_gguf_string(ctx, "general.architecture");
    info.description = read_gguf_string(ctx, "general.description");

    if (!info.architecture.empty()) {
        info.context_length = read_arch_gguf_u32_as_i32(ctx, info.architecture, "context_length");
        info.embedding_dim = read_arch_gguf_u32_as_i32(ctx, info.architecture, "embedding_length");
        info.layer_count = read_arch_gguf_u32_as_i32(ctx, info.architecture, "block_count");
        if (info.layer_count == 0) {
            info.layer_count =
                read_arch_gguf_u32_as_i32(ctx, info.architecture, "decoder_block_count");
        }
        info.head_count = read_arch_gguf_u32_as_i32(ctx, info.architecture, "attention.head_count");
        info.kv_head_count =
            read_arch_gguf_u32_as_i32(ctx, info.architecture, "attention.head_count_kv");
    }
    if (info.context_length == 0) {
        info.context_length = read_gguf_u32_as_i32(ctx, "llm.context_length");
    }
    if (info.kv_head_count == 0 && info.head_count > 0) {
        info.kv_head_count = info.head_count;
    }
    if (info.description.empty()) {
        info.description = info.architecture;
        if (!info.description.empty() && !info.quantization.empty()) {
            info.description += " " + info.quantization;
        }
    }

    collect_all_metadata(ctx, info.metadata);
}

// Derives quantization label from the model description (e.g. "7B Q4_K_M" → "Q4_K_M").
// Requires a digit immediately after the Q/F/I prefix to avoid misclassifying
// arbitrary trailing tokens like "Foo".
void derive_quantization(ModelInfo& info) {
    if (!info.quantization.empty()) {
        return;
    }
    const auto pos = info.description.find_last_of(' ');
    if (pos == std::string::npos) {
        return;
    }
    const auto candidate = info.description.substr(pos + 1);
    if (candidate.size() < 2) {
        return;
    }
    const bool has_quant_prefix =
        std::string_view("QFI").find(candidate[0]) != std::string_view::npos;
    const bool has_digit_after = static_cast<unsigned char>(candidate[1]) >= '0' &&
                                 static_cast<unsigned char>(candidate[1]) <= '9';
    if (has_quant_prefix && has_digit_after) {
        info.quantization = candidate;
    }
}

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

// Returns the RAM that should be assumed available for the KV cache. Prefers
// MemAvailable when the platform reports it; otherwise falls back to total RAM.
uint64_t usable_ram_bytes(const SystemInfo& sys) {
    return sys.available_ram_bytes > 0 ? sys.available_ram_bytes : sys.total_ram_bytes;
}

// Sums per-GPU free VRAM, falling back to total VRAM when the backend does not
// expose a current free figure.
uint64_t aggregate_vram_bytes(const SystemInfo& sys) {
    uint64_t total = 0;
    for (const auto& gpu : sys.gpus) {
        total += gpu.free_vram_bytes > 0 ? gpu.free_vram_bytes : gpu.total_vram_bytes;
    }
    return total;
}

int compute_context_size(const ModelInfo& info, const SystemInfo& sys) {
    const int training_ctx =
        info.context_length > 0 ? info.context_length : kFallbackTrainingContext;

    int ctx_from_ram = kContextHardCap;
    const uint64_t per_token_kv = per_token_kv_bytes(info);
    if (per_token_kv > 0) {
        const uint64_t ram = usable_ram_bytes(sys);
        const uint64_t reserved = info.file_size_bytes + kRamOverheadBytes;
        const uint64_t kv_budget = ram > reserved ? (ram - reserved) / 2 : 0;
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
    const uint64_t vram = aggregate_vram_bytes(sys);
    if (vram == 0 || info.file_size_bytes == 0) {
        return 0;
    }
    if (vram >= info.file_size_bytes + info.file_size_bytes / 5) {
        return -1; // Full offload with ~20% headroom.
    }
    if (info.layer_count <= 0) {
        return 0;
    }
    const uint64_t bytes_per_layer = info.file_size_bytes / static_cast<uint64_t>(info.layer_count);
    if (bytes_per_layer == 0) {
        return 0;
    }
    // Reserve ~20% of the layers that would fit for KV cache + activations.
    // KV at 32k context can be hundreds of MB; a flat 2-layer reserve is too
    // thin for larger contexts and risks runtime OOM.
    const int64_t fits = static_cast<int64_t>(vram / bytes_per_layer);
    const int64_t headroom = std::max<int64_t>(2, fits / 5);
    const int64_t usable = fits - headroom;
    return usable > 0 ? static_cast<int>(usable) : 0;
}

} // namespace

Expected<ModelInfo> GgufInspector::inspect(const std::string& file_path) {
    std::error_code ec;
    if (!std::filesystem::exists(file_path, ec) || ec) {
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
    auto absolute = std::filesystem::absolute(file_path, ec);
    info.file_path = ec ? file_path : absolute.string();
    read_gguf_metadata(gguf_ctx.get(), info);
    read_tensor_stats(gguf_ctx.get(), info);

    derive_quantization(info);
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
    // Use available RAM (when known) so mlock doesn't starve other processes
    // already consuming a large share of memory.
    config.use_mlock = usable_ram_bytes(sys) >= info.file_size_bytes * 5 / 2;
    return config;
}

Expected<ModelConfig> GgufInspector::auto_configure(const ModelInfo& info) {
    auto sys = SystemProbe::probe();
    if (!sys) {
        return std::unexpected(sys.error());
    }
    return auto_configure(info, *sys);
}

} // namespace zoo::core
