/**
 * @file model_info.hpp
 * @brief Metadata extracted from a GGUF file without loading model weights.
 */

#pragma once

#include <cstdint>
#include <map>
#include <string>

namespace zoo::core {

/**
 * @brief Metadata extracted from a GGUF file without loading model weights.
 *
 * Populated by `GgufInspector::inspect()` and consumed by
 * `GgufInspector::auto_configure()` and the local model store.
 */
struct ModelInfo {
    std::string file_path;        ///< Absolute path to the inspected GGUF file.
    std::string name;             ///< Value of the `general.name` metadata key.
    std::string architecture;     ///< Value of the `general.architecture` metadata key.
    std::string description;      ///< Human-readable model description (e.g. "7B Q4_K_M").
    uint64_t parameter_count = 0; ///< Total number of model parameters.
    uint64_t file_size_bytes = 0; ///< Total size of all tensor data in bytes.
    int32_t embedding_dim = 0;    ///< Embedding/hidden dimension size.
    int32_t layer_count = 0;      ///< Number of transformer layers.
    int32_t head_count = 0;       ///< Number of attention heads (a.k.a. n_heads).
    int32_t kv_head_count = 0;    ///< Number of KV heads (n_kv_heads); equals head_count for MHA.
    int32_t context_length = 0;   ///< Training context length from metadata.
    std::string quantization;     ///< Quantization type extracted from description or metadata.
    std::map<std::string, std::string> metadata; ///< All raw GGUF key-value pairs as strings.

    bool operator==(const ModelInfo& other) const = default;
};

} // namespace zoo::core
