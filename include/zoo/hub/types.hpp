/**
 * @file types.hpp
 * @brief Value types for the hub layer: model metadata and catalog entries.
 */

#pragma once

#include "zoo/core/types.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace zoo::hub {

/**
 * @brief Metadata extracted from a GGUF file without loading model weights.
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
    int32_t context_length = 0;   ///< Training context length from metadata.
    std::string quantization;     ///< Quantization type extracted from description or metadata.
    std::map<std::string, std::string> metadata; ///< All raw GGUF key-value pairs as strings.

    bool operator==(const ModelInfo& other) const = default;
};

/**
 * @brief Configuration for the local model store.
 */
struct ModelStoreConfig {
    std::string store_directory; ///< Root directory for model storage.
    std::string catalog_filename =
        "catalog.json"; ///< Catalog file name within the store directory.

    [[nodiscard]] Expected<void> validate() const {
        if (store_directory.empty()) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "Model store directory cannot be empty"});
        }
        if (catalog_filename.empty()) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "Catalog filename cannot be empty"});
        }
        return {};
    }

    bool operator==(const ModelStoreConfig& other) const = default;
};

/**
 * @brief A registered model in the local catalog.
 */
struct ModelEntry {
    std::string id;                   ///< Unique identifier (typically a UUID).
    std::string file_path;            ///< Absolute path to the GGUF file.
    ModelInfo info;                   ///< Cached inspection metadata.
    std::vector<std::string> aliases; ///< User-assigned short names for the model.
    std::string source_url;           ///< Download URL if fetched from HuggingFace.
    std::string huggingface_repo; ///< HuggingFace repository ID (e.g. "TheBloke/Mistral-7B-GGUF").
    std::string added_at;         ///< ISO 8601 timestamp of when the model was registered.

    bool operator==(const ModelEntry& other) const = default;
};

} // namespace zoo::hub
