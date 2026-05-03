/**
 * @file download_validation.hpp
 * @brief Internal validation helpers for completed Hub downloads.
 */

#pragma once

#include "zoo/hub/types.hpp"

#include <filesystem>
#include <string>

namespace zoo::hub::detail {

// Integrity model is delegated to llama.cpp's downloader (ETag/server-trust via
// `common_download_model`/`common_download_file_single`). This local check only
// confirms the resulting file is present and non-empty.
[[nodiscard]] inline Expected<void> validate_downloaded_file(const std::filesystem::path& path) {
    std::error_code ec;
    const bool exists = std::filesystem::exists(path, ec);
    if (ec) {
        return std::unexpected(Error{to_error_code(HubErrorCode::DownloadFailed),
                                     "Cannot access downloaded model: " + path.string(),
                                     ec.message()});
    }
    if (!exists) {
        return std::unexpected(Error{to_error_code(HubErrorCode::DownloadFailed),
                                     "Downloaded model file is missing: " + path.string()});
    }

    const bool is_regular = std::filesystem::is_regular_file(path, ec);
    if (ec) {
        return std::unexpected(Error{to_error_code(HubErrorCode::DownloadFailed),
                                     "Cannot inspect downloaded model: " + path.string(),
                                     ec.message()});
    }
    if (!is_regular) {
        return std::unexpected(
            Error{to_error_code(HubErrorCode::DownloadFailed),
                  "Downloaded model path is not a regular file: " + path.string()});
    }

    const auto actual_size = std::filesystem::file_size(path, ec);
    if (ec) {
        return std::unexpected(Error{to_error_code(HubErrorCode::DownloadFailed),
                                     "Cannot read downloaded model size: " + path.string(),
                                     ec.message()});
    }
    if (actual_size == 0) {
        return std::unexpected(Error{to_error_code(HubErrorCode::DownloadFailed),
                                     "Downloaded model file is empty: " + path.string()});
    }

    return {};
}

} // namespace zoo::hub::detail
