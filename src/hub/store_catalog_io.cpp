/**
 * @file store_catalog_io.cpp
 * @brief Shared ModelStore catalog mutation helpers.
 */

#include "hub/store_catalog_io.hpp"

#include <chrono>
#include <iomanip>
#include <random>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace zoo::hub {

namespace {

bool is_blank(std::string_view value) {
    return value.find_first_not_of(" \t\n\r\f\v") == std::string_view::npos;
}

} // namespace

Expected<void> validate_alias_value(std::string_view alias) {
    if (is_blank(alias)) {
        return std::unexpected(Error{ErrorCode::InvalidConfig, "Alias cannot be empty"});
    }
    return {};
}

Expected<void> validate_aliases_for_store(const std::vector<ModelEntry>& entries,
                                          std::span<const std::string> aliases,
                                          std::optional<size_t> skip_index) {
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

} // namespace detail
} // namespace zoo::hub
