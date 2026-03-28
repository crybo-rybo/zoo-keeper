/**
 * @file store_json.hpp
 * @brief Internal JSON serialization for model catalog types.
 */

#pragma once

#include "zoo/hub/types.hpp"

#include <nlohmann/json.hpp>

namespace zoo::hub {

// --- ModelInfo ---

inline void to_json(nlohmann::json& j, const ModelInfo& info) {
    j = nlohmann::json{
        {"file_path", info.file_path},
        {"name", info.name},
        {"architecture", info.architecture},
        {"description", info.description},
        {"parameter_count", info.parameter_count},
        {"file_size_bytes", info.file_size_bytes},
        {"embedding_dim", info.embedding_dim},
        {"layer_count", info.layer_count},
        {"context_length", info.context_length},
        {"quantization", info.quantization},
    };
}

inline void from_json(const nlohmann::json& j, ModelInfo& info) {
    if (auto it = j.find("file_path"); it != j.end())
        it->get_to(info.file_path);
    if (auto it = j.find("name"); it != j.end())
        it->get_to(info.name);
    if (auto it = j.find("architecture"); it != j.end())
        it->get_to(info.architecture);
    if (auto it = j.find("description"); it != j.end())
        it->get_to(info.description);
    if (auto it = j.find("parameter_count"); it != j.end())
        it->get_to(info.parameter_count);
    if (auto it = j.find("file_size_bytes"); it != j.end())
        it->get_to(info.file_size_bytes);
    if (auto it = j.find("embedding_dim"); it != j.end())
        it->get_to(info.embedding_dim);
    if (auto it = j.find("layer_count"); it != j.end())
        it->get_to(info.layer_count);
    if (auto it = j.find("context_length"); it != j.end())
        it->get_to(info.context_length);
    if (auto it = j.find("quantization"); it != j.end())
        it->get_to(info.quantization);
}

// --- ModelEntry ---

inline void to_json(nlohmann::json& j, const ModelEntry& entry) {
    j = nlohmann::json{
        {"id", entry.id},
        {"file_path", entry.file_path},
        {"info", entry.info},
        {"aliases", entry.aliases},
        {"source_url", entry.source_url},
        {"huggingface_repo", entry.huggingface_repo},
        {"added_at", entry.added_at},
    };
}

inline void from_json(const nlohmann::json& j, ModelEntry& entry) {
    if (auto it = j.find("id"); it != j.end())
        it->get_to(entry.id);
    if (auto it = j.find("file_path"); it != j.end())
        it->get_to(entry.file_path);
    if (auto it = j.find("info"); it != j.end())
        it->get_to(entry.info);
    if (auto it = j.find("aliases"); it != j.end())
        it->get_to(entry.aliases);
    if (auto it = j.find("source_url"); it != j.end())
        it->get_to(entry.source_url);
    if (auto it = j.find("huggingface_repo"); it != j.end())
        it->get_to(entry.huggingface_repo);
    if (auto it = j.find("added_at"); it != j.end())
        it->get_to(entry.added_at);
}

} // namespace zoo::hub
