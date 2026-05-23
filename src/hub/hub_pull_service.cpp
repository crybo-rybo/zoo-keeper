/**
 * @file hub_pull_service.cpp
 * @brief ModelStore HuggingFace pull orchestration.
 */

#include "hub/store_internals.hpp"

#include "hub/download_validation.hpp"
#include "hub/hf_cache_paths.hpp"
#include "hub/store_catalog_io.hpp"

#include <algorithm>
#include <string>

namespace zoo::hub::detail {

namespace {

struct PulledModelSource {
    std::string local_path;
    std::string source_url;
};

std::string repo_id_with_tag(const HuggingFaceClient::ParsedIdentifier& parsed) {
    std::string repo = parsed.repo_id;
    if (parsed.tag) {
        repo += ":" + *parsed.tag;
    }
    return repo;
}

Expected<PulledModelSource> download_explicit_pull_file(HuggingFaceClient& client,
                                                        const std::string& identifier,
                                                        const std::string& repo_id,
                                                        const std::string& filename) {
    auto url = client.resolve_download_url(repo_id, filename);
    if (!url) {
        return std::unexpected(url.error());
    }

    auto result = client.download_model(identifier);
    if (!result) {
        return std::unexpected(result.error());
    }

    return PulledModelSource{std::move(*result), std::move(*url)};
}

Expected<PulledModelSource>
download_repo_snapshot(HuggingFaceClient& client,
                       const HuggingFaceClient::ParsedIdentifier& parsed) {
    auto result = client.download_model(repo_id_with_tag(parsed));
    if (!result) {
        return std::unexpected(result.error());
    }

    PulledModelSource source{std::move(*result), {}};
    if (auto url = source_url_from_hf_snapshot(parsed.repo_id, source.local_path)) {
        source.source_url = std::move(*url);
    }
    return source;
}

Expected<PulledModelSource>
download_pull_source(HuggingFaceClient& client, const std::string& identifier,
                     const HuggingFaceClient::ParsedIdentifier& parsed) {
    if (parsed.filename) {
        return download_explicit_pull_file(client, identifier, parsed.repo_id, *parsed.filename);
    }
    return download_repo_snapshot(client, parsed);
}

} // namespace

Expected<ModelEntry> HubPullService::persist_source_annotation(std::vector<ModelEntry>& entries,
                                                               const CatalogRepository& repository,
                                                               const std::string& entry_id,
                                                               std::string source_url,
                                                               std::string repo_id) {
    return mutate_catalog(
        repository, entries,
        [&entry_id, source_url = std::move(source_url),
         repo_id = std::move(repo_id)](std::vector<ModelEntry>& current) mutable {
            auto it = std::find_if(current.begin(), current.end(),
                                   [&](const ModelEntry& entry) { return entry.id == entry_id; });
            if (it == current.end()) {
                return Expected<ModelEntry>(std::unexpected(
                    Error{ErrorCode::ModelNotFound, "No model found matching: " + entry_id}));
            }

            it->source_url = std::move(source_url);
            it->huggingface_repo = std::move(repo_id);
            return Expected<ModelEntry>(*it);
        });
}

Expected<ModelEntry> HubPullService::pull(HuggingFaceClient& client, const std::string& identifier,
                                          std::vector<std::string> aliases,
                                          std::vector<ModelEntry>& entries,
                                          const CatalogRepository& repository) {
    auto parsed = HuggingFaceClient::parse_identifier(identifier);
    if (!parsed) {
        return std::unexpected(parsed.error());
    }

    auto source = download_pull_source(client, identifier, *parsed);
    if (!source) {
        return std::unexpected(source.error());
    }

    if (auto validation = validate_downloaded_file(source->local_path); !validation) {
        return std::unexpected(validation.error());
    }

    return ModelImporter::add_local_file(entries, repository, source->local_path,
                                         std::move(aliases), std::move(source->source_url),
                                         parsed->repo_id);
}

} // namespace zoo::hub::detail
