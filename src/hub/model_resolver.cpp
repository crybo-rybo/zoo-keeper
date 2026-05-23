/**
 * @file model_resolver.cpp
 * @brief ModelStore catalog lookup helpers.
 */

#include "hub/store_internals.hpp"

namespace zoo::hub::detail {

Expected<size_t> ModelResolver::find_index(std::span<const ModelEntry> entries,
                                           const std::string& query) {
    for (size_t i = 0; i < entries.size(); ++i) {
        for (const auto& alias : entries[i].aliases) {
            if (alias == query) {
                return i;
            }
        }
    }

    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].info.name == query) {
            return i;
        }
    }

    for (size_t i = 0; i < entries.size(); ++i) {
        if (!entries[i].info.name.empty() &&
            entries[i].info.name.find(query) != std::string::npos) {
            return i;
        }
    }

    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].file_path == query) {
            return i;
        }
    }

    for (size_t i = 0; i < entries.size(); ++i) {
        if (entries[i].id == query) {
            return i;
        }
    }

    return std::unexpected(Error{ErrorCode::ModelNotFound, "No model found matching: " + query});
}

} // namespace zoo::hub::detail
