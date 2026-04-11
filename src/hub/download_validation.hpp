/**
 * @file download_validation.hpp
 * @brief Internal validation helpers for completed Hub downloads.
 */

#pragma once

#include "zoo/core/types.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace zoo::hub::detail {

[[nodiscard]] inline Expected<void>
validate_downloaded_file(const std::filesystem::path& path,
                         std::optional<uintmax_t> expected_size_bytes = std::nullopt) {
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    if (ec) {
        return std::unexpected(Error{ErrorCode::DownloadFailed,
                                     "Cannot access downloaded model: " + path.string(),
                                     ec.message()});
    }
    if (!exists) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Downloaded model file is missing: " + path.string()});
    }

    const bool is_regular = std::filesystem::is_regular_file(path, ec);
    if (ec) {
        return std::unexpected(Error{ErrorCode::DownloadFailed,
                                     "Cannot inspect downloaded model: " + path.string(),
                                     ec.message()});
    }
    if (!is_regular) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed,
                  "Downloaded model path is not a regular file: " + path.string()});
    }

    const auto actual_size = std::filesystem::file_size(path, ec);
    if (ec) {
        return std::unexpected(Error{ErrorCode::DownloadFailed,
                                     "Cannot read downloaded model size: " + path.string(),
                                     ec.message()});
    }
    if (actual_size == 0) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Downloaded model file is empty: " + path.string()});
    }

    if (expected_size_bytes.has_value() && actual_size != *expected_size_bytes) {
        return std::unexpected(Error{ErrorCode::DownloadFailed,
                                     "Downloaded model size mismatch: " + path.string(),
                                     "expected " + std::to_string(*expected_size_bytes) +
                                         " bytes, got " + std::to_string(actual_size)});
    }

    return {};
}

} // namespace zoo::hub::detail
