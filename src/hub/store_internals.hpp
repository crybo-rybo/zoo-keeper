/**
 * @file store_internals.hpp
 * @brief Private ModelStore persistence, lookup, import, and pull helpers.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/hub/huggingface.hpp"
#include "zoo/hub/types.hpp"

#include <span>
#include <string>
#include <vector>

namespace zoo::hub::detail {

class CatalogRepository {
  public:
    explicit CatalogRepository(ModelStoreConfig config);

    [[nodiscard]] const ModelStoreConfig& config() const noexcept;
    [[nodiscard]] std::string catalog_path() const;
    [[nodiscard]] Expected<std::vector<ModelEntry>> load() const;
    [[nodiscard]] Expected<void> save(const std::vector<ModelEntry>& entries) const;

  private:
    ModelStoreConfig config_;
};

class ModelResolver {
  public:
    [[nodiscard]] static Expected<size_t> find_index(std::span<const ModelEntry> entries,
                                                     const std::string& query);
};

class ModelImporter {
  public:
    [[nodiscard]] static Expected<ModelEntry>
    add_local_file(std::vector<ModelEntry>& entries, const CatalogRepository& repository,
                   const std::string& file_path, std::vector<std::string> aliases,
                   std::string source_url = {}, std::string huggingface_repo = {});
};

class HubPullService {
  public:
    [[nodiscard]] static Expected<ModelEntry>
    pull(HuggingFaceClient& client, const std::string& identifier, std::vector<std::string> aliases,
         std::vector<ModelEntry>& entries, const CatalogRepository& repository);

    [[nodiscard]] static Expected<ModelEntry>
    persist_source_annotation(std::vector<ModelEntry>& entries, const CatalogRepository& repository,
                              const std::string& entry_id, std::string source_url,
                              std::string repo_id);
};

} // namespace zoo::hub::detail
