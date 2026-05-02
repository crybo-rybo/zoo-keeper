/**
 * @file huggingface.cpp
 * @brief HuggingFace Hub API client -- wraps llama.cpp llama-common download infrastructure.
 */

#include "zoo/hub/huggingface.hpp"
#include "hub/download_validation.hpp"

#include <common.h>
#include <download.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace zoo::hub {

struct HuggingFaceClient::Impl {
    Config config;

    [[nodiscard]] common_download_opts download_opts() const {
        common_download_opts opts;
        opts.bearer_token = config.token;
        return opts;
    }
};

Expected<std::unique_ptr<HuggingFaceClient>> HuggingFaceClient::create() {
    return create(Config{});
}

Expected<std::unique_ptr<HuggingFaceClient>> HuggingFaceClient::create(Config config) {
    if (auto result = config.validate(); !result) {
        return std::unexpected(result.error());
    }

    auto impl = std::make_unique<Impl>();
    impl->config = std::move(config);

    return std::unique_ptr<HuggingFaceClient>(new HuggingFaceClient(std::move(impl)));
}

HuggingFaceClient::HuggingFaceClient(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
HuggingFaceClient::~HuggingFaceClient() = default;
HuggingFaceClient::HuggingFaceClient(HuggingFaceClient&&) noexcept = default;
HuggingFaceClient& HuggingFaceClient::operator=(HuggingFaceClient&&) noexcept = default;

Expected<HuggingFaceClient::ParsedIdentifier>
HuggingFaceClient::parse_identifier(std::string_view identifier) {
    if (identifier.empty()) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelIdentifier, "Model identifier cannot be empty"});
    }

    ParsedIdentifier result;

    // Support both "::" (zoo-keeper style) and ":" (ollama/llama.cpp style) separators.
    // "::" takes precedence to avoid ambiguity with ":" tags.
    const auto double_sep = identifier.find("::");
    if (double_sep != std::string_view::npos) {
        auto repo_part = identifier.substr(0, double_sep);
        auto filename = identifier.substr(double_sep + 2);
        if (filename.empty()) {
            return std::unexpected(
                Error{ErrorCode::InvalidModelIdentifier,
                      "Empty filename after '::' in: " + std::string(identifier)});
        }
        result.filename = std::string(filename);
        try {
            auto [repo, tag] = common_download_split_repo_tag(std::string(repo_part));
            result.repo_id = std::move(repo);
            if (!tag.empty() && tag != "latest") {
                result.tag = std::move(tag);
            }
        } catch (const std::invalid_argument&) {
            return std::unexpected(
                Error{ErrorCode::InvalidModelIdentifier,
                      "Repository ID must be in 'owner/repo' or 'owner/repo:tag' format: " +
                          std::string(repo_part)});
        }
    } else {
        // Use llama.cpp's split to handle "owner/repo:tag" format.
        try {
            auto [repo, tag] = common_download_split_repo_tag(std::string(identifier));
            result.repo_id = repo;
            if (!tag.empty() && tag != "latest") {
                result.tag = std::move(tag);
            }
        } catch (const std::invalid_argument&) {
            return std::unexpected(
                Error{ErrorCode::InvalidModelIdentifier,
                      "Repository ID must be in 'owner/repo' or 'owner/repo:tag' format: " +
                          std::string(identifier)});
        }
    }

    // Validate repo ID has exactly one slash.
    const auto slash = result.repo_id.find('/');
    if (slash == std::string::npos || slash == 0 || slash == result.repo_id.size() - 1) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelIdentifier,
                  "Repository ID must be in 'owner/repo' format: " + result.repo_id});
    }
    if (result.repo_id.find('/', slash + 1) != std::string::npos) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelIdentifier,
                  "Repository ID must contain exactly one '/': " + result.repo_id});
    }

    return result;
}

Expected<std::string> HuggingFaceClient::resolve_download_url(const std::string& repo_id,
                                                              const std::string& filename) {
    return "https://huggingface.co/" + repo_id + "/resolve/main/" + filename;
}

Expected<std::string> HuggingFaceClient::download_model(const std::string& repo_id_with_tag) {
    try {
        auto parsed = parse_identifier(repo_id_with_tag);
        if (!parsed) {
            return std::unexpected(parsed.error());
        }

        common_params_model model_params;
        model_params.hf_repo = parsed->repo_id;
        if (parsed->tag) {
            model_params.hf_repo += ":" + *parsed->tag;
        }
        if (parsed->filename) {
            model_params.hf_file = *parsed->filename;
        }

        auto download = common_download_model(model_params, impl_->download_opts());
        if (download.model_path.empty()) {
            return std::unexpected(Error{ErrorCode::DownloadFailed,
                                         "Failed to download model from: " + repo_id_with_tag});
        }

        if (auto validation = detail::validate_downloaded_file(download.model_path); !validation) {
            return std::unexpected(validation.error());
        }

        return download.model_path;
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Download error: " + std::string(e.what())});
    }
}

Expected<std::string> HuggingFaceClient::download_file(const std::string& url,
                                                       const std::string& destination_path) {
    try {
        common_params_model model_params;
        model_params.url = url;
        model_params.path = destination_path;

        auto download = common_download_model(model_params, impl_->download_opts());
        if (download.model_path.empty()) {
            return std::unexpected(Error{ErrorCode::DownloadFailed, "Download failed for: " + url});
        }

        if (auto validation = detail::validate_downloaded_file(download.model_path); !validation) {
            return std::unexpected(validation.error());
        }

        return download.model_path;
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Download error: " + std::string(e.what())});
    }
}

std::vector<CachedModelInfo> HuggingFaceClient::list_cached_models() {
    auto cached = common_list_cached_models();

    std::vector<CachedModelInfo> result;
    result.reserve(cached.size());

    for (auto& entry : cached) {
        CachedModelInfo info;
        const auto slash = entry.repo.find('/');
        if (slash == std::string::npos) {
            continue;
        }
        info.user = entry.repo.substr(0, slash);
        info.model = entry.repo.substr(slash + 1);
        info.tag = entry.tag.empty() ? "latest" : std::move(entry.tag);
        info.size_bytes = 0;
        result.push_back(std::move(info));
    }

    return result;
}

} // namespace zoo::hub
