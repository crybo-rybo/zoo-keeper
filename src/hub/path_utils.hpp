/**
 * @file path_utils.hpp
 * @brief Internal helpers for safe, deterministic hub download destinations.
 */

#pragma once

#include "zoo/core/types.hpp"

#include <filesystem>
#include <string_view>

namespace zoo::hub::detail {

inline Expected<std::filesystem::path> validate_relative_download_path(std::string_view value,
                                                                       std::string_view label) {
    const std::filesystem::path path(value);
    if (path.empty()) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelIdentifier, std::string(label) + " cannot be empty"});
    }
    if (path.is_absolute()) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelIdentifier,
                  std::string(label) + " must be relative: " + std::string(value)});
    }

    for (const auto& part : path) {
        if (part == "..") {
            return std::unexpected(
                Error{ErrorCode::InvalidModelIdentifier,
                      std::string(label) + " cannot contain '..': " + std::string(value)});
        }
    }

    return path;
}

inline Expected<std::filesystem::path>
build_download_destination(const std::filesystem::path& root, std::string_view repo_id,
                           std::string_view relative_filename) {
    auto repo = validate_relative_download_path(repo_id, "Repository ID");
    if (!repo) {
        return std::unexpected(repo.error());
    }

    auto file = validate_relative_download_path(relative_filename, "Download filename");
    if (!file) {
        return std::unexpected(file.error());
    }

    return root / *repo / *file;
}

} // namespace zoo::hub::detail
