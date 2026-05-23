/**
 * @file catalog_repository_io.hpp
 * @brief Catalog file locking and JSON persistence declarations.
 */

#pragma once

#include "zoo/core/types.hpp"
#include "zoo/hub/types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace zoo::hub::detail {

class CatalogLock {
  public:
    explicit CatalogLock(int fd) noexcept : fd_(fd) {}
    ~CatalogLock();

    CatalogLock(const CatalogLock&) = delete;
    CatalogLock& operator=(const CatalogLock&) = delete;
    CatalogLock(CatalogLock&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    CatalogLock& operator=(CatalogLock&&) = delete;

  private:
    int fd_;
};

Expected<CatalogLock> lock_catalog(const std::string& catalog_path);
Expected<std::vector<ModelEntry>> load_catalog_file(const std::string& path);
Expected<void> save_catalog_file(const std::string& path, const std::vector<ModelEntry>& entries);

} // namespace zoo::hub::detail
