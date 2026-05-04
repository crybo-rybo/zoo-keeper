/**
 * @file store.cpp
 * @brief Local model catalog implementation with JSON persistence.
 */

#include "zoo/hub/store.hpp"
#include "hub/download_validation.hpp"
#include "hub/hf_cache_paths.hpp"
#include "hub/store_internals.hpp"
#include "hub/store_json.hpp"
#include "zoo/hub/inspector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace zoo::hub {

namespace {

std::string generate_id() {
    static thread_local std::mt19937 rng(std::random_device{}());
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string id;
    id.reserve(32);
    for (int i = 0; i < 32; ++i) {
        id += kHexDigits[rng() % 16];
    }
    return id;
}

std::string now_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm buf{};
    gmtime_r(&time, &buf);
    std::ostringstream ss;
    ss << std::put_time(&buf, "%FT%TZ");
    return ss.str();
}

std::string default_store_directory() {
    std::string home;
    if (const char* h = std::getenv("HOME")) {
        home = h;
    } else {
        home = ".";
    }
    return home + "/.zoo-keeper/models";
}

bool is_blank(std::string_view value) {
    return value.find_first_not_of(" \t\n\r\f\v") == std::string_view::npos;
}

Expected<void> validate_alias_value(std::string_view alias) {
    if (is_blank(alias)) {
        return std::unexpected(Error{ErrorCode::InvalidConfig, "Alias cannot be empty"});
    }
    return {};
}

Expected<void> validate_catalog_entries(const std::vector<ModelEntry>& entries) {
    std::unordered_set<std::string> aliases;
    for (const auto& entry : entries) {
        std::unordered_set<std::string> entry_aliases;
        for (const auto& alias : entry.aliases) {
            if (auto result = validate_alias_value(alias); !result) {
                return std::unexpected(
                    Error{ErrorCode::StoreCorrupted, "Catalog contains an empty alias"});
            }
            if (!entry_aliases.insert(alias).second) {
                return std::unexpected(
                    Error{ErrorCode::StoreCorrupted,
                          "Catalog contains duplicate aliases on entry: " + entry.id});
            }
            if (!aliases.insert(alias).second) {
                return std::unexpected(
                    Error{ErrorCode::StoreCorrupted, "Catalog contains duplicate alias: " + alias});
            }
        }
    }
    return {};
}

} // namespace

Expected<void> validate_aliases_for_store(const std::vector<ModelEntry>& entries,
                                          std::span<const std::string> aliases,
                                          std::optional<size_t> skip_index = std::nullopt) {
    std::unordered_set<std::string> seen;
    for (const auto& alias : aliases) {
        if (auto result = validate_alias_value(alias); !result) {
            return std::unexpected(result.error());
        }
        if (!seen.insert(alias).second) {
            return std::unexpected(
                Error{ErrorCode::InvalidConfig, "Duplicate alias in request: " + alias});
        }
    }

    for (size_t i = 0; i < entries.size(); ++i) {
        if (skip_index.has_value() && *skip_index == i) {
            continue;
        }
        for (const auto& alias : entries[i].aliases) {
            if (seen.contains(alias)) {
                return std::unexpected(
                    Error{ErrorCode::ModelAlreadyExists, "Alias already in use: " + alias});
            }
        }
    }

    return {};
}

namespace detail {

CatalogRepository::CatalogRepository(ModelStoreConfig config) : config_(std::move(config)) {}

const ModelStoreConfig& CatalogRepository::config() const noexcept {
    return config_;
}

std::string CatalogRepository::catalog_path() const {
    return config_.store_directory + "/" + config_.catalog_filename;
}

Expected<std::vector<ModelEntry>> CatalogRepository::load() const {
    const auto path = catalog_path();
    if (!std::filesystem::exists(path)) {
        return std::vector<ModelEntry>{};
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        return std::unexpected(Error{ErrorCode::FilesystemError, "Cannot open catalog: " + path});
    }

    std::vector<ModelEntry> loaded_entries;
    try {
        auto j = nlohmann::json::parse(file);
        if (!j.is_object() || !j.contains("models") || !j["models"].is_array()) {
            return std::unexpected(
                Error{ErrorCode::StoreCorrupted, "Catalog has invalid structure: " + path});
        }
        loaded_entries = j["models"].get<std::vector<ModelEntry>>();
    } catch (const nlohmann::json::exception& e) {
        return std::unexpected(
            Error{ErrorCode::StoreCorrupted, "Failed to parse catalog: " + std::string(e.what())});
    }
    if (auto result = validate_catalog_entries(loaded_entries); !result) {
        return std::unexpected(result.error());
    }
    return loaded_entries;
}

Expected<void> CatalogRepository::save(const std::vector<ModelEntry>& entries) const {
    const auto path = catalog_path();
    const auto temp_path = path + ".tmp." + generate_id();

    nlohmann::json j;
    j["version"] = 1;
    j["models"] = entries;

    {
        std::ofstream file(temp_path, std::ios::trunc);
        if (!file.is_open()) {
            return std::unexpected(
                Error{ErrorCode::FilesystemError, "Cannot write catalog: " + temp_path});
        }
        file << j.dump(2) << "\n";
        file.flush();
        if (!file.good()) {
            std::error_code remove_ec;
            std::filesystem::remove(temp_path, remove_ec);
            return std::unexpected(
                Error{ErrorCode::FilesystemError, "Failed while writing catalog: " + temp_path});
        }
    }

    std::error_code ec;
    std::filesystem::rename(temp_path, path, ec);
    if (ec) {
        std::error_code remove_ec;
        std::filesystem::remove(temp_path, remove_ec);
        return std::unexpected(
            Error{ErrorCode::FilesystemError, "Cannot replace catalog: " + path, ec.message()});
    }
    return {};
}

Expected<size_t> ModelResolver::find_index(std::span<const ModelEntry> entries,
                                           const std::string& query) {
    // 1. Exact alias match.
    for (size_t i = 0; i < entries.size(); ++i) {
        for (const auto& alias : entries[i].aliases) {
            if (alias == query) {
                return i;
            }
        }
    }

    // 2. Exact name match.
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].info.name == query) {
            return i;
        }
    }

    // 3. Name substring match.
    for (size_t i = 0; i < entries.size(); ++i) {
        if (!entries[i].info.name.empty() &&
            entries[i].info.name.find(query) != std::string::npos) {
            return i;
        }
    }

    // 4. Path match.
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].file_path == query) {
            return i;
        }
    }

    // 5. ID match.
    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].id == query) {
            return i;
        }
    }

    return std::unexpected(Error{ErrorCode::ModelNotFound, "No model found matching: " + query});
}

Expected<ModelEntry>
ModelImporter::add_local_file(std::vector<ModelEntry>& entries, const CatalogRepository& repository,
                              const std::string& file_path, std::vector<std::string> aliases,
                              std::string source_url, std::string huggingface_repo) {
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

    auto info = GgufInspector::inspect(abs_path);
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

    if (auto result = repository.save(entries); !result) {
        entries.pop_back();
        return std::unexpected(result.error());
    }

    return entry;
}

Expected<ModelEntry> HubPullService::persist_source_annotation(std::vector<ModelEntry>& entries,
                                                               const CatalogRepository& repository,
                                                               const std::string& entry_id,
                                                               std::string source_url,
                                                               std::string repo_id) {
    auto it = std::find_if(entries.begin(), entries.end(),
                           [&](const ModelEntry& entry) { return entry.id == entry_id; });
    if (it == entries.end()) {
        return std::unexpected(
            Error{ErrorCode::ModelNotFound, "No model found matching: " + entry_id});
    }

    const std::string old_source_url = it->source_url;
    const std::string old_repo = it->huggingface_repo;
    it->source_url = std::move(source_url);
    it->huggingface_repo = std::move(repo_id);

    if (auto result = repository.save(entries); !result) {
        it->source_url = old_source_url;
        it->huggingface_repo = old_repo;
        return std::unexpected(result.error());
    }

    return *it;
}

Expected<ModelEntry> HubPullService::pull(HuggingFaceClient& client, const std::string& identifier,
                                          std::vector<std::string> aliases,
                                          std::vector<ModelEntry>& entries,
                                          const CatalogRepository& repository) {
    auto parsed = HuggingFaceClient::parse_identifier(identifier);
    if (!parsed) {
        return std::unexpected(parsed.error());
    }

    std::string repo_with_tag = parsed->repo_id;
    if (parsed->tag) {
        repo_with_tag += ":" + *parsed->tag;
    }

    std::string local_path;
    std::string source_url;
    if (parsed->filename) {
        auto url = client.resolve_download_url(parsed->repo_id, *parsed->filename);
        if (!url) {
            return std::unexpected(url.error());
        }
        source_url = *url;
        auto result = client.download_model(identifier);
        if (!result) {
            return std::unexpected(result.error());
        }
        local_path = *result;
    } else {
        auto result = client.download_model(repo_with_tag);
        if (!result) {
            return std::unexpected(result.error());
        }
        local_path = *result;

        if (auto url = detail::source_url_from_hf_snapshot(parsed->repo_id, local_path)) {
            source_url = std::move(*url);
        }
    }

    if (auto validation = detail::validate_downloaded_file(local_path); !validation) {
        return std::unexpected(validation.error());
    }

    return ModelImporter::add_local_file(entries, repository, local_path, std::move(aliases),
                                         std::move(source_url), parsed->repo_id);
}

} // namespace detail

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

    // Create the store directory if it doesn't exist.
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
    auto idx = detail::ModelResolver::find_index(impl_->entries, name_or_alias);
    if (!idx) {
        return std::unexpected(idx.error());
    }

    if (delete_file) {
        std::error_code ec;
        std::filesystem::remove(impl_->entries[*idx].file_path, ec);
        // Ignore removal errors — the file may already be gone.
    }

    auto removed = std::move(impl_->entries[*idx]);
    impl_->entries.erase(impl_->entries.begin() + static_cast<ptrdiff_t>(*idx));
    if (auto result = impl_->repository.save(impl_->entries); !result) {
        impl_->entries.insert(impl_->entries.begin() + static_cast<ptrdiff_t>(*idx),
                              std::move(removed));
        return std::unexpected(result.error());
    }
    return {};
}

Expected<void> ModelStore::add_alias(const std::string& name_or_alias,
                                     const std::string& new_alias) {
    auto idx = detail::ModelResolver::find_index(impl_->entries, name_or_alias);
    if (!idx) {
        return std::unexpected(idx.error());
    }
    std::array<std::string, 1> aliases{new_alias};
    if (auto result = validate_aliases_for_store(impl_->entries, aliases, *idx); !result) {
        return std::unexpected(result.error());
    }

    impl_->entries[*idx].aliases.push_back(new_alias);
    if (auto result = impl_->repository.save(impl_->entries); !result) {
        impl_->entries[*idx].aliases.pop_back();
        return std::unexpected(result.error());
    }
    return {};
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
    return GgufInspector::auto_configure(entry->info);
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
