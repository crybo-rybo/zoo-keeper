/**
 * @file model_importer.cpp
 * @brief ModelStore local file import helpers.
 */

#include "hub/store_internals.hpp"

#include "hub/store_catalog_io.hpp"
#include "zoo/core/gguf_inspector.hpp"

#include <filesystem>

namespace zoo::hub::detail {

namespace {

Expected<ModelEntry> add_local_file_entry(std::vector<ModelEntry>& entries,
                                          const std::string& file_path,
                                          std::vector<std::string> aliases, std::string source_url,
                                          std::string huggingface_repo) {
    const auto abs_path = std::filesystem::absolute(file_path).string();

    if (auto result = validate_aliases_for_store(entries, aliases); !result) {
        return std::unexpected(result.error());
    }

    for (const auto& entry : entries) {
        if (entry.file_path == abs_path) {
            return std::unexpected(
                Error{ErrorCode::ModelAlreadyExists, "Model already registered: " + abs_path});
        }
    }

    auto info = core::GgufInspector::inspect(abs_path);
    if (!info) {
        return std::unexpected(info.error());
    }

    ModelEntry entry;
    entry.id = generate_id();
    entry.file_path = abs_path;
    entry.info = std::move(*info);
    entry.aliases = std::move(aliases);
    entry.added_at = now_iso8601();
    entry.source_url = std::move(source_url);
    entry.huggingface_repo = std::move(huggingface_repo);

    entries.push_back(entry);
    return entry;
}

} // namespace

Expected<ModelEntry>
ModelImporter::add_local_file(std::vector<ModelEntry>& entries, const CatalogRepository& repository,
                              const std::string& file_path, std::vector<std::string> aliases,
                              std::string source_url, std::string huggingface_repo) {
    return mutate_catalog(
        repository, entries,
        [&file_path, aliases = std::move(aliases), source_url = std::move(source_url),
         huggingface_repo = std::move(huggingface_repo)](std::vector<ModelEntry>& current) mutable {
            return add_local_file_entry(current, file_path, std::move(aliases),
                                        std::move(source_url), std::move(huggingface_repo));
        });
}

} // namespace zoo::hub::detail
