/**
 * @file hf_download.cpp
 * @brief llama-common download wrappers returning Expected-based results.
 */

#include "core/hf_download.hpp"

#include <common.h>
#include <download.h>

#include <stdexcept>
#include <utility>

namespace zoo::core::detail {

namespace {

common_download_opts to_common_options(const HfDownloadOptions& options) {
    common_download_opts opts;
    opts.bearer_token = options.bearer_token;
    return opts;
}

Expected<void> validate_http_status(int status, const std::string& url) {
    if (status < 0) {
        return std::unexpected(Error{ErrorCode::DownloadFailed, "Download failed for: " + url});
    }
    if (status >= 400) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed,
                  "Download returned HTTP " + std::to_string(status) + " for: " + url});
    }
    return {};
}

} // namespace

Expected<HfRepoTagParts> split_repo_tag(const std::string& identifier) {
    try {
        auto [repo, tag] = common_download_split_repo_tag(identifier);
        return HfRepoTagParts{std::move(repo), std::move(tag)};
    } catch (const std::invalid_argument&) {
        return std::unexpected(Error{
            ErrorCode::InvalidModelIdentifier,
            "Repository ID must be in 'owner/repo' or 'owner/repo:tag' format: " + identifier});
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed,
                  "Failed to parse repository identifier: " + std::string(e.what())});
    }
}

Expected<std::string> download_model(const HfModelDownloadParams& params,
                                     const HfDownloadOptions& options) {
    try {
        common_params_model model_params;
        model_params.hf_repo = params.hf_repo;
        if (!params.hf_file.empty()) {
            model_params.hf_file = params.hf_file;
        }

        const auto download = common_download_model(model_params, to_common_options(options));
        if (download.model_path.empty()) {
            return std::unexpected(Error{ErrorCode::DownloadFailed,
                                         "Failed to download model from: " + params.hf_repo});
        }
        return download.model_path;
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Download error: " + std::string(e.what())});
    }
}

Expected<void> download_file(const std::string& url, const std::string& destination_path,
                             const HfDownloadOptions& options) {
    try {
        const int status =
            common_download_file_single(url, destination_path, to_common_options(options));
        return validate_http_status(status, url);
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Download error: " + std::string(e.what())});
    }
}

std::vector<HfCachedModelEntry> list_cached_models() {
    const auto cached = common_list_cached_models();
    std::vector<HfCachedModelEntry> result;
    result.reserve(cached.size());
    for (const auto& entry : cached) {
        result.push_back(HfCachedModelEntry{entry.repo, entry.tag});
    }
    return result;
}

} // namespace zoo::core::detail
