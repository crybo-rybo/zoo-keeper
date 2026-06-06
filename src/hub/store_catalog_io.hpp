/**
 * @file store_catalog_io.hpp
 * @brief Shared catalog locking, persistence, and mutation helpers for ModelStore.
 */

#pragma once

#include "hub/store_internals.hpp"
#include "zoo/hub/types.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace zoo::hub {

/**
 * @brief Rejects blank or whitespace-only alias values.
 *
 * @param alias Alias string to validate.
 * @return Empty success, or `InvalidConfig` if @p alias is blank.
 */
Expected<void> validate_alias_value(std::string_view alias);

/**
 * @brief Validates that @p aliases are non-blank, unique within the request, and
 *        not already in use by any catalog entry other than the one at @p skip_index.
 *
 * @param entries Current catalog entries to check for conflicts.
 * @param aliases Aliases to validate for the incoming operation.
 * @param skip_index If set, the catalog entry at this index is excluded from
 *        conflict checks (used when re-registering an existing entry).
 * @return Empty success, or an error identifying the first conflict.
 */
Expected<void> validate_aliases_for_store(const std::vector<ModelEntry>& entries,
                                          std::span<const std::string> aliases,
                                          std::optional<size_t> skip_index = std::nullopt);

namespace detail {

std::string generate_id();
std::string now_iso8601();

/// Atomically reloads, mutates, and persists the catalog under an exclusive lock.
template <typename Mutator>
auto mutate_catalog(const CatalogRepository& repository, std::vector<ModelEntry>& cached_entries,
                    Mutator&& mutator);

} // namespace detail
} // namespace zoo::hub

#include "hub/store_catalog_io_impl.hpp"
