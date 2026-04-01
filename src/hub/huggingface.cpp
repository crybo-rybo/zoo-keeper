/**
 * @file huggingface.cpp
 * @brief HuggingFace Hub API client — wraps llama.cpp common download infrastructure.
 */

#include "zoo/hub/huggingface.hpp"
#include "hub/path_utils.hpp"

#include <common.h>
#include <download.h>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace zoo::hub {

struct HuggingFaceClient::Impl {
    Config config;

    [[nodiscard]] common_header_list make_headers() const {
        common_header_list headers;
        if (!config.token.empty()) {
            headers.emplace_back("Authorization", "Bearer " + config.token);
        }
        return headers;
    }

    [[nodiscard]] common_hf_file_res resolve_hf_file(const std::string& repo_id_with_tag) const {
        return common_get_hf_file(repo_id_with_tag, config.token, false, make_headers());
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
        result.repo_id = std::string(repo_part);
    } else {
        // Use llama.cpp's split to handle "owner/repo:tag" format.
        try {
            auto [repo, tag] = common_download_split_repo_tag(std::string(identifier));
            result.repo_id = repo;
            if (tag != "latest") {
                result.tag = tag;
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
        auto hf_res = impl_->resolve_hf_file(repo_id_with_tag);

        if (hf_res.ggufFile.empty()) {
            return std::unexpected(Error{ErrorCode::ModelNotFound,
                                         "No GGUF file found in repository: " + repo_id_with_tag});
        }

        const std::string url =
            "https://huggingface.co/" + hf_res.repo + "/resolve/main/" + hf_res.ggufFile;
        const std::string cache_dir = fs_get_cache_directory();
        auto dest_path =
            detail::build_download_destination(cache_dir, hf_res.repo, hf_res.ggufFile);
        if (!dest_path) {
            return std::unexpected(dest_path.error());
        }

        std::error_code ec;
        std::filesystem::create_directories(dest_path->parent_path(), ec);
        if (ec) {
            return std::unexpected(
                Error{ErrorCode::FilesystemError,
                      "Failed to create download directory: " + dest_path->parent_path().string(),
                      ec.message()});
        }

        common_params_model model_params;
        model_params.path = dest_path->string();
        model_params.url = url;
        model_params.hf_repo = hf_res.repo;
        model_params.hf_file = hf_res.ggufFile;

        bool ok =
            common_download_model(model_params, impl_->config.token, false, impl_->make_headers());
        if (!ok) {
            return std::unexpected(Error{ErrorCode::DownloadFailed,
                                         "Failed to download model from: " + repo_id_with_tag});
        }

        return dest_path->string();
    } catch (const std::exception& e) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed, "Download error: " + std::string(e.what())});
    }
}

Expected<std::string> HuggingFaceClient::download_file(const std::string& url,
                                                       const std::string& destination_path) {
    // Ensure parent directory exists.
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(destination_path).parent_path(), ec);

    int status = common_download_file_single(url, destination_path, impl_->config.token, false,
                                             impl_->make_headers());

    if (status < 0) {
        return std::unexpected(Error{ErrorCode::DownloadFailed, "Download failed for: " + url});
    }

    if (status >= 400) {
        return std::unexpected(
            Error{ErrorCode::DownloadFailed,
                  "Download returned HTTP " + std::to_string(status) + " for: " + url});
    }

    return destination_path;
}

std::vector<CachedModelInfo> HuggingFaceClient::list_cached_models() {
    auto cached = common_list_cached_models();

    std::vector<CachedModelInfo> result;
    result.reserve(cached.size());

    for (auto& entry : cached) {
        CachedModelInfo info;
        info.user = std::move(entry.user);
        info.model = std::move(entry.model);
        info.tag = std::move(entry.tag);
        info.size_bytes = entry.size;
        result.push_back(std::move(info));
    }

    return result;
}

} // namespace zoo::hub
