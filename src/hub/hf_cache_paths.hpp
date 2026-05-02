/**
 * @file hf_cache_paths.hpp
 * @brief Pure-logic helpers for inspecting llama.cpp's HuggingFace-style cache layout.
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace zoo::hub::detail {

// HuggingFace caches a repo "owner/name" under a folder named "models--owner--name".
inline std::string hf_cache_repo_folder(std::string_view repo_id) {
    std::string folder = "models--";
    for (const char c : repo_id) {
        if (c == '/') {
            folder += "--";
        } else {
            folder += c;
        }
    }
    return folder;
}

// Reconstruct the original "https://huggingface.co/<repo>/resolve/<commit>/<file>" URL
// from a path inside llama.cpp's HF-style cache. Returns std::nullopt when `local_path`
// does not look like a "<...>/models--owner--repo/snapshots/<commit>/<file>" path.
inline std::optional<std::string>
source_url_from_hf_snapshot(std::string_view repo_id,
                            const std::filesystem::path& local_path) {
    std::vector<std::string> parts;
    for (const auto& part : local_path.lexically_normal()) {
        parts.push_back(part.string());
    }

    const std::string repo_folder = hf_cache_repo_folder(repo_id);
    for (size_t i = 0; i + 3 < parts.size(); ++i) {
        if (parts[i] != repo_folder || parts[i + 1] != "snapshots") {
            continue;
        }

        std::filesystem::path relative;
        for (size_t j = i + 3; j < parts.size(); ++j) {
            relative /= parts[j];
        }
        if (parts[i + 2].empty() || relative.empty()) {
            return std::nullopt;
        }

        return "https://huggingface.co/" + std::string(repo_id) + "/resolve/" + parts[i + 2] + "/" +
               relative.generic_string();
    }

    return std::nullopt;
}

} // namespace zoo::hub::detail
