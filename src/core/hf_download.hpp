/**
 * @file hf_download.hpp
 * @brief Private llama-common download adapter for the core layer.
 *
 * All direct calls to llama.cpp `common_download_*` helpers live behind this
 * seam so hub code never includes llama-common headers.
 */

#pragma once

#include "zoo/core/types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace zoo::core::detail {

/// Bearer token and other options forwarded to llama-common download helpers.
struct HfDownloadOptions {
    std::string bearer_token;
};

/// Repository identifier and optional filename passed to `common_download_model`.
struct HfModelDownloadParams {
    std::string hf_repo;
    std::string hf_file;
};

/// One entry returned from the llama.cpp download cache listing.
struct HfCachedModelEntry {
    std::string repo;
    std::string tag;
};

/// Result of splitting an `owner/repo:tag` identifier.
struct HfRepoTagParts {
    std::string repo;
    std::string tag;
};

[[nodiscard]] Expected<HfRepoTagParts> split_repo_tag(const std::string& identifier);
[[nodiscard]] Expected<std::string> download_model(const HfModelDownloadParams& params,
                                                   const HfDownloadOptions& options);
[[nodiscard]] Expected<void> download_file(const std::string& url,
                                           const std::string& destination_path,
                                           const HfDownloadOptions& options);
[[nodiscard]] std::vector<HfCachedModelEntry> list_cached_models();

} // namespace zoo::core::detail
