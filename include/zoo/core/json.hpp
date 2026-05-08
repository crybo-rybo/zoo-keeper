/**
 * @file json.hpp
 * @brief Opt-in JSON serialization helpers for public config value types.
 */

#pragma once

#include "gguf_inspector.hpp"
#include "system_probe.hpp"
#include "types.hpp"

#include <array>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace zoo::detail {

inline void require_object(const nlohmann::json& j, const char* context) {
    if (!j.is_object()) {
        throw std::invalid_argument(std::string(context) + " must be a JSON object");
    }
}

template <size_t N>
inline void reject_unknown_keys(const nlohmann::json& j, const char* context,
                                const std::array<const char*, N>& allowed_keys) {
    require_object(j, context);

    for (auto it = j.begin(); it != j.end(); ++it) {
        bool allowed = false;
        for (const char* key : allowed_keys) {
            if (it.key() == key) {
                allowed = true;
                break;
            }
        }

        if (!allowed) {
            throw std::invalid_argument("Unknown " + std::string(context) + " key: " + it.key());
        }
    }
}

} // namespace zoo::detail

namespace zoo {

inline void to_json(nlohmann::json& j, const SamplingParams& params) {
    j = nlohmann::json{{"temperature", params.temperature},
                       {"top_p", params.top_p},
                       {"top_k", params.top_k},
                       {"repeat_penalty", params.repeat_penalty},
                       {"repeat_last_n", params.repeat_last_n},
                       {"seed", params.seed}};
}

inline void from_json(const nlohmann::json& j, SamplingParams& params) {
    static constexpr std::array<const char*, 6> kAllowedKeys = {
        "temperature", "top_p", "top_k", "repeat_penalty", "repeat_last_n", "seed"};

    detail::reject_unknown_keys(j, "sampling config", kAllowedKeys);

    SamplingParams parsed;
    if (auto it = j.find("temperature"); it != j.end()) {
        it->get_to(parsed.temperature);
    }
    if (auto it = j.find("top_p"); it != j.end()) {
        it->get_to(parsed.top_p);
    }
    if (auto it = j.find("top_k"); it != j.end()) {
        it->get_to(parsed.top_k);
    }
    if (auto it = j.find("repeat_penalty"); it != j.end()) {
        it->get_to(parsed.repeat_penalty);
    }
    if (auto it = j.find("repeat_last_n"); it != j.end()) {
        it->get_to(parsed.repeat_last_n);
    }
    if (auto it = j.find("seed"); it != j.end()) {
        it->get_to(parsed.seed);
    }

    params = std::move(parsed);
}

inline void to_json(nlohmann::json& j, const ModelConfig& config) {
    j = nlohmann::json{{"model_path", config.model_path}, {"context_size", config.context_size},
                       {"n_batch", config.n_batch},       {"n_gpu_layers", config.n_gpu_layers},
                       {"use_mmap", config.use_mmap},     {"use_mlock", config.use_mlock}};
}

namespace detail {

// Applies explicit JSON overrides on top of an existing ModelConfig. Used by
// both the pure deserializer and the auto-configure resolver.
inline void apply_model_config_overrides(const nlohmann::json& j, ModelConfig& config) {
    if (auto it = j.find("context_size"); it != j.end()) {
        it->get_to(config.context_size);
    }
    if (auto it = j.find("n_batch"); it != j.end()) {
        it->get_to(config.n_batch);
    }
    if (auto it = j.find("n_gpu_layers"); it != j.end()) {
        it->get_to(config.n_gpu_layers);
    }
    if (auto it = j.find("use_mmap"); it != j.end()) {
        it->get_to(config.use_mmap);
    }
    if (auto it = j.find("use_mlock"); it != j.end()) {
        it->get_to(config.use_mlock);
    }
}

} // namespace detail

// Pure deserializer: never inspects files or probes hardware. The optional
// `auto_configure` key is recognized so it does not fail validation, but it is
// resolved only by the explicit `load_model_config()` helper below.
inline void from_json(const nlohmann::json& j, ModelConfig& config) {
    // "auto_configure" is consumed by `load_model_config`, not here. Listed so
    // strict key validation does not reject configs that opt into auto-config.
    static constexpr std::array<const char*, 7> kAllowedKeys = {
        "model_path", "context_size", "n_batch",       "n_gpu_layers",
        "use_mmap",   "use_mlock",    "auto_configure"};

    detail::reject_unknown_keys(j, "model config", kAllowedKeys);

    if (!j.contains("model_path")) {
        throw std::invalid_argument("ModelConfig JSON must contain required key: model_path");
    }

    ModelConfig parsed;
    j.at("model_path").get_to(parsed.model_path);
    detail::apply_model_config_overrides(j, parsed);

    config = std::move(parsed);
}

namespace detail {

// Inspects a GGUF file and probes the host hardware to produce a ModelConfig.
// Extracted from `load_model_config` so the entry point stays small enough to
// unit test cheaply.
inline Expected<ModelConfig> auto_configure_model_path(const std::string& model_path) {
    auto info = core::GgufInspector::inspect(model_path);
    if (!info) {
        return std::unexpected(info.error());
    }
    auto sys = core::SystemProbe::probe();
    if (!sys) {
        return std::unexpected(sys.error());
    }
    return core::GgufInspector::auto_configure(*info, *sys);
}

} // namespace detail

// Returns a `ModelConfig` from JSON, resolving `auto_configure: true` against
// the host system if requested. Explicit JSON keys override the auto-derived
// values. Performs I/O (GGUF inspection) and backend init (system probe), so
// callers should treat it as an explicit configuration step rather than pure
// parsing.
inline Expected<ModelConfig> load_model_config(const nlohmann::json& j) {
    if (!j.is_object() || !j.value("auto_configure", false)) {
        return j.get<ModelConfig>();
    }

    // Auto-configure path: skip the regular deserializer (its derived values
    // would be discarded) and resolve hardware-aware defaults from the GGUF
    // file, then layer explicit overrides on top.
    detail::require_object(j, "model config");
    if (!j.contains("model_path")) {
        throw std::invalid_argument("ModelConfig JSON must contain required key: model_path");
    }
    auto resolved = detail::auto_configure_model_path(j.at("model_path").get<std::string>());
    if (!resolved) {
        return std::unexpected(resolved.error());
    }
    detail::apply_model_config_overrides(j, *resolved);
    return *resolved;
}

inline void to_json(nlohmann::json& j, const AgentConfig& config) {
    j = nlohmann::json{{"max_history_messages", config.max_history_messages},
                       {"request_queue_capacity", config.request_queue_capacity},
                       {"max_tool_iterations", config.max_tool_iterations},
                       {"max_tool_retries", config.max_tool_retries}};
}

inline void from_json(const nlohmann::json& j, AgentConfig& config) {
    static constexpr std::array<const char*, 4> kAllowedKeys = {
        "max_history_messages", "request_queue_capacity", "max_tool_iterations",
        "max_tool_retries"};

    detail::reject_unknown_keys(j, "agent config", kAllowedKeys);

    AgentConfig parsed;
    if (auto it = j.find("max_history_messages"); it != j.end()) {
        it->get_to(parsed.max_history_messages);
    }
    if (auto it = j.find("request_queue_capacity"); it != j.end()) {
        it->get_to(parsed.request_queue_capacity);
    }
    if (auto it = j.find("max_tool_iterations"); it != j.end()) {
        it->get_to(parsed.max_tool_iterations);
    }
    if (auto it = j.find("max_tool_retries"); it != j.end()) {
        it->get_to(parsed.max_tool_retries);
    }

    config = std::move(parsed);
}

inline void to_json(nlohmann::json& j, const GenerationOptions& options) {
    j = nlohmann::json{{"sampling", options.sampling},
                       {"max_tokens", options.max_tokens},
                       {"stop_sequences", options.stop_sequences},
                       {"record_tool_trace", options.record_tool_trace}};
}

inline void from_json(const nlohmann::json& j, GenerationOptions& options) {
    static constexpr std::array<const char*, 4> kAllowedKeys = {
        "sampling", "max_tokens", "stop_sequences", "record_tool_trace"};

    detail::reject_unknown_keys(j, "generation options", kAllowedKeys);

    GenerationOptions parsed;
    if (auto it = j.find("sampling"); it != j.end()) {
        it->get_to(parsed.sampling);
    }
    if (auto it = j.find("max_tokens"); it != j.end()) {
        it->get_to(parsed.max_tokens);
    }
    if (auto it = j.find("stop_sequences"); it != j.end()) {
        it->get_to(parsed.stop_sequences);
    }
    if (auto it = j.find("record_tool_trace"); it != j.end()) {
        it->get_to(parsed.record_tool_trace);
    }

    options = std::move(parsed);
}

} // namespace zoo
