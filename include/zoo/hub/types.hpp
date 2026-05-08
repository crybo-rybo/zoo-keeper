/**
 * @file types.hpp
 * @brief Value types for the hub layer: catalog entries and store configuration.
 */

#pragma once

#include "zoo/core/model_info.hpp"
#include "zoo/core/types.hpp"

#include <string>
#include <vector>

namespace zoo::hub {

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
    core::ModelInfo info;             ///< Cached inspection metadata.
    std::vector<std::string> aliases; ///< User-assigned short names for the model.
    std::string source_url;           ///< Download URL if fetched from HuggingFace.
    std::string huggingface_repo; ///< HuggingFace repository ID (e.g. "TheBloke/Mistral-7B-GGUF").
    std::string added_at;         ///< ISO 8601 timestamp of when the model was registered.

    bool operator==(const ModelEntry& other) const = default;
};

} // namespace zoo::hub
