#pragma once

#include "zoo/types.hpp"
#include "zoo/gguf_utils.hpp"
#include <string>
#include <filesystem>

namespace zoo {

/// Memory breakdown for a model+config combination.
struct MemoryEstimate {
    size_t model_weights_bytes = 0;  ///< Model weights (GGUF file mapped into memory)
    size_t kv_cache_bytes = 0;       ///< KV cache for requested context_size
    size_t compute_buffer_bytes = 0; ///< Scratch/compute buffers (estimated)
    size_t total_bytes = 0;          ///< Sum of the above

    /// Returns total_bytes as gigabytes (for display)
    double total_gb() const { return static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0); }
    double model_gb() const { return static_cast<double>(model_weights_bytes) / (1024.0 * 1024.0 * 1024.0); }
    double kv_cache_gb() const { return static_cast<double>(kv_cache_bytes) / (1024.0 * 1024.0 * 1024.0); }
};

/// Estimate memory requirements for loading a model with a given context size.
///
/// This is a fast pre-load estimate that reads only GGUF metadata (no tensor loading).
/// Useful for OOM prevention and user-facing memory requirement display.
///
/// KV cache calculation:
///   kv_cache_bytes = 2 * n_layers * n_embd * context_size * bytes_per_kv_element
///   where bytes_per_kv_element depends on kv_cache_type_k (1=F16=2, 8=Q8_0=1, 2=Q4_0=0.5)
///
/// @param model_path   Path to a .gguf model file
/// @param context_size Desired context window size in tokens
/// @param kv_type_k    KV cache key type (matches Config::kv_cache_type_k, default 1=F16)
/// @param kv_type_v    KV cache value type (matches Config::kv_cache_type_v, default 1=F16)
/// @return MemoryEstimate on success, or Error if model metadata can't be read
inline Expected<MemoryEstimate> estimate_memory(
    const std::string& model_path,
    int context_size,
    int kv_type_k = 1,
    int kv_type_v = 1
) {
    // Read GGUF metadata (fast, no tensor loading)
    auto metadata_result = read_gguf_metadata(model_path);
    if (!metadata_result) {
        return tl::unexpected(metadata_result.error());
    }
    const GgufModelInfo& meta = *metadata_result;

    MemoryEstimate est;

    // Model weights: GGUF file is mmap'd, so file size ~= memory usage
    std::error_code ec;
    auto file_size = std::filesystem::file_size(model_path, ec);
    if (ec) {
        return tl::unexpected(Error{
            ErrorCode::ModelLoadFailed,
            "Cannot determine model file size: " + model_path + " (" + ec.message() + ")"
        });
    }
    est.model_weights_bytes = static_cast<size_t>(file_size);

    // KV cache: 2 (k+v) x n_layers x n_embd x context_size x bytes_per_element
    // bytes_per_element based on ggml_type:
    //   0 = F32  -> 4 bytes
    //   1 = F16  -> 2 bytes
    //   8 = Q8_0 -> 1 byte  (approximately, actual is 32 vals per block with overhead)
    //   2 = Q4_0 -> 0.5 bytes
    auto kv_bytes_per_elem = [](int kv_type) -> double {
        switch (kv_type) {
            case 0: return 4.0;  // GGML_TYPE_F32
            case 1: return 2.0;  // GGML_TYPE_F16
            case 8: return 1.0;  // GGML_TYPE_Q8_0 (approximate)
            case 2: return 0.5;  // GGML_TYPE_Q4_0 (approximate)
            case 6: return 0.5;  // GGML_TYPE_Q4_1 (approximate)
            default: return 2.0; // Default to F16 for unknown types
        }
    };

    if (meta.n_layers > 0 && meta.n_embd > 0 && context_size > 0) {
        double k_bytes = static_cast<double>(meta.n_layers) * meta.n_embd * context_size
                       * kv_bytes_per_elem(kv_type_k);
        double v_bytes = static_cast<double>(meta.n_layers) * meta.n_embd * context_size
                       * kv_bytes_per_elem(kv_type_v);
        est.kv_cache_bytes = static_cast<size_t>(k_bytes + v_bytes);
    }

    // Compute buffers: conservative ~14% of model size (scratch memory for forward pass)
    est.compute_buffer_bytes = est.model_weights_bytes / 7; // ~14%

    est.total_bytes = est.model_weights_bytes + est.kv_cache_bytes + est.compute_buffer_bytes;
    return est;
}

} // namespace zoo
