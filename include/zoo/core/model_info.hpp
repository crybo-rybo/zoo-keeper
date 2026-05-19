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
    std::string file_path;
    std::string name;         ///< Value of the `general.name` metadata key.
    std::string architecture; ///< Value of the `general.architecture` metadata key.
    std::string description;
    uint64_t parameter_count = 0;
    uint64_t file_size_bytes = 0; ///< Sum of tensor sizes, not the on-disk file size.
    int32_t embedding_dim = 0;
    int32_t layer_count = 0;
    int32_t head_count = 0;
    int32_t kv_head_count = 0;  ///< Equals head_count for MHA; smaller for GQA.
    int32_t context_length = 0; ///< Training context length, not the runtime context.
    std::string quantization;
    std::map<std::string, std::string> metadata;

    bool operator==(const ModelInfo& other) const = default;
};

} // namespace zoo::core
