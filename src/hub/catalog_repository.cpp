/**
 * @file catalog_repository.cpp
 * @brief CatalogRepository persistence and catalog JSON validation.
 */

#include "hub/store_internals.hpp"

#include "hub/catalog_repository_io.hpp"
#include "hub/store_catalog_io.hpp"
#include "hub/store_json.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string_view>
#include <unordered_set>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace zoo::hub::detail {

namespace {

constexpr int kCatalogVersion = 1;

std::string errno_message(std::string_view action, const std::string& path, int err) {
    return std::string(action) + ": " + path + ": " + std::strerror(err);
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

} // namespace

CatalogLock::~CatalogLock() {
    if (fd_ >= 0) {
        ::flock(fd_, LOCK_UN);
        ::close(fd_);
    }
}

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

} // namespace zoo::hub::detail
