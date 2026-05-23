/**
 * @file store.cpp
 * @brief Local model catalog implementation with JSON persistence.
 */

#include "zoo/hub/store.hpp"
#include "hub/download_validation.hpp"
#include "hub/hf_cache_paths.hpp"
#include "hub/store_internals.hpp"
#include "hub/store_json.hpp"
#include "zoo/core/gguf_inspector.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace zoo::hub {

namespace {

constexpr int kCatalogVersion = 1;

std::string errno_message(std::string_view action, const std::string& path, int err) {
    return std::string(action) + ": " + path + ": " + std::strerror(err);
}

class CatalogLock {
  public:
    explicit CatalogLock(int fd) noexcept : fd_(fd) {}
    ~CatalogLock() {
        if (fd_ >= 0) {
            ::flock(fd_, LOCK_UN);
            ::close(fd_);
        }
    }

    CatalogLock(const CatalogLock&) = delete;
    CatalogLock& operator=(const CatalogLock&) = delete;
    CatalogLock(CatalogLock&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    CatalogLock& operator=(CatalogLock&&) = delete;

  private:
    int fd_;
};

Expected<CatalogLock> lock_catalog(const std::string& catalog_path) {
    const auto path = catalog_path + ".lock";
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        return std::unexpected(Error{ErrorCode::FilesystemError,
                                     errno_message("Cannot open catalog lock", path, errno)});
    }
    while (::flock(fd, LOCK_EX) != 0) {
        if (errno == EINTR) {
            continue;
        }
        const int err = errno;
        ::close(fd);
        return std::unexpected(
            Error{ErrorCode::FilesystemError, errno_message("Cannot lock catalog", path, err)});
    }
    return CatalogLock(fd);
}

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

Expected<void> validate_catalog_document(const nlohmann::json& j, const std::string& path) {
    if (!j.is_object() || !j.contains("models") || !j["models"].is_array()) {
        return std::unexpected(
            Error{ErrorCode::StoreCorrupted, "Catalog has invalid structure: " + path});
    }
    const auto version = j.find("version");
    if (version == j.end() || !version->is_number_integer() || *version != kCatalogVersion) {
        return std::unexpected(
            Error{ErrorCode::StoreCorrupted, "Catalog has unsupported version: " + path});
    }
    for (const auto& entry : j["models"]) {
        if (!entry.is_object() || !entry.value("id", nlohmann::json{}).is_string() ||
            !entry.value("file_path", nlohmann::json{}).is_string() ||
            !entry.value("added_at", nlohmann::json{}).is_string() ||
            !entry.value("info", nlohmann::json{}).is_object() ||
            !entry["info"].value("file_path", nlohmann::json{}).is_string() ||
            !entry["info"].value("name", nlohmann::json{}).is_string() ||
            !entry.value("aliases", nlohmann::json{}).is_array()) {
            return std::unexpected(
                Error{ErrorCode::StoreCorrupted, "Catalog entry has invalid structure: " + path});
        }
        for (const auto& alias : entry["aliases"]) {
            if (!alias.is_string()) {
                return std::unexpected(Error{ErrorCode::StoreCorrupted,
                                             "Catalog entry has non-string alias: " + path});
            }
        }
    }
    return {};
}

Expected<std::vector<ModelEntry>> load_catalog_file(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        return std::vector<ModelEntry>{};
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        return std::unexpected(Error{ErrorCode::FilesystemError, "Cannot open catalog: " + path});
    }
    try {
        auto j = nlohmann::json::parse(file);
        if (auto result = validate_catalog_document(j, path); !result) {
            return std::unexpected(result.error());
        }
        auto entries = j["models"].get<std::vector<ModelEntry>>();
        if (auto result = validate_catalog_entries(entries); !result) {
            return std::unexpected(result.error());
        }
        return entries;
    } catch (const nlohmann::json::exception& e) {
        return std::unexpected(
            Error{ErrorCode::StoreCorrupted, "Failed to parse catalog: " + std::string(e.what())});
    }
}

Expected<void> save_catalog_file(const std::string& path, const std::vector<ModelEntry>& entries) {
    const auto temp_path = path + ".tmp." + generate_id();
    nlohmann::json j{{"version", kCatalogVersion}, {"models", entries}};
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

template <typename Mutator>
auto mutate_catalog(const detail::CatalogRepository& repository,
                    std::vector<ModelEntry>& cached_entries, Mutator&& mutator) {
    using Result = std::invoke_result_t<Mutator&, std::vector<ModelEntry>&>;
    const auto path = repository.catalog_path();

    auto lock = lock_catalog(path);
    if (!lock) {
        return Result(std::unexpected(lock.error()));
    }

    auto loaded_entries = load_catalog_file(path);
    if (!loaded_entries) {
        return Result(std::unexpected(loaded_entries.error()));
    }

    auto working_entries = std::move(*loaded_entries);
    Result result = mutator(working_entries);
    if (!result) {
        return std::move(result);
    }

    if (auto saved = save_catalog_file(path, working_entries); !saved) {
        return Result(std::unexpected(saved.error()));
    }

    cached_entries = std::move(working_entries);
    return std::move(result);
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
    auto lock = lock_catalog(path);
    if (!lock) {
        return std::unexpected(lock.error());
    }
    return load_catalog_file(path);
}

Expected<void> CatalogRepository::save(const std::vector<ModelEntry>& entries) const {
    const auto path = catalog_path();
    auto lock = lock_catalog(path);
    if (!lock) {
        return std::unexpected(lock.error());
    }
    return save_catalog_file(path, entries);
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
    if (auto url = detail::source_url_from_hf_snapshot(parsed.repo_id, source.local_path)) {
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

    if (auto validation = detail::validate_downloaded_file(source->local_path); !validation) {
        return std::unexpected(validation.error());
    }

    return ModelImporter::add_local_file(entries, repository, source->local_path,
                                         std::move(aliases), std::move(source->source_url),
                                         parsed->repo_id);
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
    std::string removed_path;
    auto result = mutate_catalog(impl_->repository, impl_->entries,
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
        // Ignore removal errors — the file may already be gone.
    }
    return result;
}

Expected<void> ModelStore::add_alias(const std::string& name_or_alias,
                                     const std::string& new_alias) {
    return mutate_catalog(impl_->repository, impl_->entries,
                          [&name_or_alias, &new_alias](std::vector<ModelEntry>& current) {
                              auto idx = detail::ModelResolver::find_index(current, name_or_alias);
                              if (!idx) {
                                  return Expected<void>(std::unexpected(idx.error()));
                              }
                              std::array<std::string, 1> aliases{new_alias};
                              if (auto result = validate_aliases_for_store(current, aliases, *idx);
                                  !result) {
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
