/**
 * @file huggingface.cpp
 * @brief HuggingFace Hub API client — policy and validation over core download adapter.
 */

#include "zoo/hub/huggingface.hpp"
#include "core/hf_download.hpp"
#include "hub/download_validation.hpp"

#include <filesystem>
#include <string>
#include <utility>

namespace zoo::hub {

namespace {

Expected<core::detail::HfModelDownloadParams>
build_model_download_params(const HuggingFaceClient::ParsedIdentifier& parsed) {
    core::detail::HfModelDownloadParams params;
    params.hf_repo = parsed.repo_id;
    if (parsed.tag) {
        params.hf_repo += ":" + *parsed.tag;
    }
    if (parsed.filename) {
        params.hf_file = *parsed.filename;
    }
    return params;
}

Expected<core::detail::HfModelDownloadParams>
build_model_download_params(const std::string& identifier) {
    auto parsed = HuggingFaceClient::parse_identifier(identifier);
    if (!parsed) {
        return std::unexpected(parsed.error());
    }
    return build_model_download_params(*parsed);
}

Expected<void> validate_explicit_filename(std::string_view filename, std::string_view identifier) {
    if (filename.empty()) {
        return std::unexpected(Error{ErrorCode::InvalidModelIdentifier,
                                     "Empty filename after '::' in: " + std::string(identifier)});
    }
    if (filename == "." || filename == ".." || filename.find('\0') != std::string_view::npos ||
        filename.find('/') != std::string_view::npos ||
        filename.find('\\') != std::string_view::npos) {
        return std::unexpected(
            Error{ErrorCode::InvalidModelIdentifier,
                  "Explicit filename must be a single path segment: " + std::string(identifier)});
    }
    return {};
}

core::detail::HfDownloadOptions download_options(const HuggingFaceClient::Config& config) {
    return core::detail::HfDownloadOptions{config.token};
}

} // namespace

struct HuggingFaceClient::Impl {
    Config config;
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

    const auto double_sep = identifier.find("::");
    if (double_sep != std::string_view::npos) {
        auto repo_part = identifier.substr(0, double_sep);
        auto filename = identifier.substr(double_sep + 2);
        if (auto valid = validate_explicit_filename(filename, identifier); !valid) {
            return std::unexpected(valid.error());
        }
        result.filename = std::string(filename);
        auto split = core::detail::split_repo_tag(std::string(repo_part));
        if (!split) {
            return std::unexpected(split.error());
        }
        result.repo_id = std::move(split->repo);
        if (!split->tag.empty() && split->tag != "latest") {
            result.tag = std::move(split->tag);
        }
    } else {
        auto split = core::detail::split_repo_tag(std::string(identifier));
        if (!split) {
            return std::unexpected(split.error());
        }
        result.repo_id = std::move(split->repo);
        if (!split->tag.empty() && split->tag != "latest") {
            result.tag = std::move(split->tag);
        }
    }

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
    auto params = build_model_download_params(repo_id_with_tag);
    if (!params) {
        return std::unexpected(params.error());
    }

    auto model_path = core::detail::download_model(*params, download_options(impl_->config));
    if (!model_path) {
        return std::unexpected(model_path.error());
    }

    if (auto validation = detail::validate_downloaded_file(*model_path); !validation) {
        return std::unexpected(validation.error());
    }

    return *model_path;
}

Expected<std::string> HuggingFaceClient::download_file(const std::string& url,
                                                       const std::string& destination_path) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(destination_path).parent_path(), ec);
    if (ec) {
        return std::unexpected(
            Error{ErrorCode::FilesystemError,
                  "Failed to create download directory: " +
                      std::filesystem::path(destination_path).parent_path().string(),
                  ec.message()});
    }

    if (auto result =
            core::detail::download_file(url, destination_path, download_options(impl_->config));
        !result) {
        return std::unexpected(result.error());
    }

    if (auto validation = detail::validate_downloaded_file(destination_path); !validation) {
        return std::unexpected(validation.error());
    }

    return destination_path;
}

std::vector<CachedModelInfo> HuggingFaceClient::list_cached_models() {
    auto cached = core::detail::list_cached_models();

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
