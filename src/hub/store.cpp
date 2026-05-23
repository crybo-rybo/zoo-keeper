/**
 * @file store.cpp
 * @brief Local model catalog facade with JSON persistence.
 */

#include "zoo/hub/store.hpp"

#include "hub/store_catalog_io.hpp"
#include "hub/store_internals.hpp"
#include "zoo/core/gguf_inspector.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>

namespace zoo::hub {

namespace {

std::string default_store_directory() {
    std::string home;
    if (const char* h = std::getenv("HOME")) {
        home = h;
    } else {
        home = ".";
    }
    return home + "/.zoo-keeper/models";
}

} // namespace

struct ModelStore::Impl {
    detail::CatalogRepository repository;
    std::vector<ModelEntry> entries;

    Impl(detail::CatalogRepository repo, std::vector<ModelEntry> loaded_entries)
        : repository(std::move(repo)), entries(std::move(loaded_entries)) {}
};

Expected<std::unique_ptr<ModelStore>> ModelStore::open(ModelStoreConfig config) {
    if (config.store_directory.empty()) {
        config.store_directory = default_store_directory();
    }

    if (auto result = config.validate(); !result) {
        return std::unexpected(result.error());
    }

    std::error_code ec;
    std::filesystem::create_directories(config.store_directory, ec);
    if (ec) {
        return std::unexpected(Error{ErrorCode::FilesystemError,
                                     "Cannot create store directory: " + config.store_directory,
                                     ec.message()});
    }

    detail::CatalogRepository repository(std::move(config));
    auto entries = repository.load();
    if (!entries) {
        return std::unexpected(entries.error());
    }

    auto impl = std::make_unique<Impl>(std::move(repository), std::move(*entries));
    return std::unique_ptr<ModelStore>(new ModelStore(std::move(impl)));
}

ModelStore::ModelStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ModelStore::~ModelStore() = default;
ModelStore::ModelStore(ModelStore&&) noexcept = default;
ModelStore& ModelStore::operator=(ModelStore&&) noexcept = default;

Expected<ModelEntry> ModelStore::add(const std::string& file_path,
                                     std::vector<std::string> aliases) {
    return detail::ModelImporter::add_local_file(impl_->entries, impl_->repository, file_path,
                                                 std::move(aliases));
}

Expected<void> ModelStore::remove(const std::string& name_or_alias, bool delete_file) {
    std::string removed_path;
    auto result =
        detail::mutate_catalog(impl_->repository, impl_->entries,
                               [&name_or_alias, &removed_path](std::vector<ModelEntry>& current) {
                                   auto idx =
                                       detail::ModelResolver::find_index(current, name_or_alias);
                                   if (!idx) {
                                       return Expected<void>(std::unexpected(idx.error()));
                                   }

                                   removed_path = current[*idx].file_path;
                                   current.erase(current.begin() + static_cast<ptrdiff_t>(*idx));
                                   return Expected<void>{};
                               });
    if (result && delete_file) {
        std::error_code ec;
        std::filesystem::remove(removed_path, ec);
    }
    return result;
}

Expected<void> ModelStore::add_alias(const std::string& name_or_alias,
                                     const std::string& new_alias) {
    return detail::mutate_catalog(
        impl_->repository, impl_->entries,
        [&name_or_alias, &new_alias](std::vector<ModelEntry>& current) {
            auto idx = detail::ModelResolver::find_index(current, name_or_alias);
            if (!idx) {
                return Expected<void>(std::unexpected(idx.error()));
            }
            std::array<std::string, 1> aliases{new_alias};
            if (auto result = validate_aliases_for_store(current, aliases, *idx); !result) {
                return Expected<void>(std::unexpected(result.error()));
            }

            current[*idx].aliases.push_back(new_alias);
            return Expected<void>{};
        });
}

std::vector<ModelEntry> ModelStore::list() const {
    return impl_->entries;
}

Expected<ModelEntry> ModelStore::find(const std::string& query) const {
    auto idx = detail::ModelResolver::find_index(impl_->entries, query);
    if (!idx) {
        return std::unexpected(idx.error());
    }
    return impl_->entries[*idx];
}

Expected<ModelConfig> ModelStore::model_config(const std::string& name_or_alias) const {
    auto entry = find(name_or_alias);
    if (!entry) {
        return std::unexpected(entry.error());
    }
    return core::GgufInspector::auto_configure(entry->info);
}

Expected<std::unique_ptr<core::Model>>
ModelStore::load_model(const std::string& name_or_alias, const GenerationOptions& options) const {
    auto config = model_config(name_or_alias);
    if (!config) {
        return std::unexpected(config.error());
    }
    return core::Model::load(*config, options);
}

Expected<std::unique_ptr<Agent>> ModelStore::create_agent(const std::string& name_or_alias,
                                                          const AgentConfig& agent_config,
                                                          const GenerationOptions& options) const {
    auto config = model_config(name_or_alias);
    if (!config) {
        return std::unexpected(config.error());
    }
    return Agent::create(*config, agent_config, options);
}

Expected<ModelEntry> ModelStore::pull(HuggingFaceClient& client, const std::string& identifier,
                                      std::vector<std::string> aliases) {
    return detail::HubPullService::pull(client, identifier, std::move(aliases), impl_->entries,
                                        impl_->repository);
}

const ModelStoreConfig& ModelStore::config() const noexcept {
    return impl_->repository.config();
}

} // namespace zoo::hub
