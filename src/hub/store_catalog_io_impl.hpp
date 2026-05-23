/**
 * @file store_catalog_io_impl.hpp
 * @brief Inline catalog mutation helper that depends on catalog repository I/O.
 */

#pragma once

#include "hub/catalog_repository_io.hpp"

namespace zoo::hub::detail {

template <typename Mutator>
auto mutate_catalog(const CatalogRepository& repository, std::vector<ModelEntry>& cached_entries,
                    Mutator&& mutator) {
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

} // namespace zoo::hub::detail
