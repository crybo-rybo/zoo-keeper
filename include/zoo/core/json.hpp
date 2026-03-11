/**
 * @file json.hpp
 * @brief Opt-in JSON serialization helpers for public config value types.
 */

#pragma once

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

inline void to_json(nlohmann::json& j, const Config& config) {
    j = nlohmann::json{{"model_path", config.model_path},
                       {"context_size", config.context_size},
                       {"n_gpu_layers", config.n_gpu_layers},
                       {"use_mmap", config.use_mmap},
                       {"use_mlock", config.use_mlock},
                       {"sampling", config.sampling},
                       {"max_tokens", config.max_tokens},
                       {"stop_sequences", config.stop_sequences},
                       {"max_history_messages", config.max_history_messages},
                       {"request_queue_capacity", config.request_queue_capacity},
                       {"max_tool_iterations", config.max_tool_iterations},
                       {"max_tool_retries", config.max_tool_retries}};

    if (config.system_prompt.has_value()) {
        j["system_prompt"] = *config.system_prompt;
    }
}

inline void from_json(const nlohmann::json& j, Config& config) {
    static constexpr std::array<const char*, 13> kAllowedKeys = {"model_path",
                                                                 "context_size",
                                                                 "n_gpu_layers",
                                                                 "use_mmap",
                                                                 "use_mlock",
                                                                 "sampling",
                                                                 "max_tokens",
                                                                 "stop_sequences",
                                                                 "system_prompt",
                                                                 "max_history_messages",
                                                                 "request_queue_capacity",
                                                                 "max_tool_iterations",
                                                                 "max_tool_retries"};

    detail::reject_unknown_keys(j, "config", kAllowedKeys);

    if (!j.contains("model_path")) {
        throw std::invalid_argument("Config JSON must contain required key: model_path");
    }

    Config parsed;
    j.at("model_path").get_to(parsed.model_path);
    if (auto it = j.find("context_size"); it != j.end()) {
        it->get_to(parsed.context_size);
    }
    if (auto it = j.find("n_gpu_layers"); it != j.end()) {
        it->get_to(parsed.n_gpu_layers);
    }
    if (auto it = j.find("use_mmap"); it != j.end()) {
        it->get_to(parsed.use_mmap);
    }
    if (auto it = j.find("use_mlock"); it != j.end()) {
        it->get_to(parsed.use_mlock);
    }
    if (auto it = j.find("sampling"); it != j.end()) {
        it->get_to(parsed.sampling);
    }
    if (auto it = j.find("max_tokens"); it != j.end()) {
        it->get_to(parsed.max_tokens);
    }
    if (auto it = j.find("stop_sequences"); it != j.end()) {
        it->get_to(parsed.stop_sequences);
    }
    if (auto it = j.find("system_prompt"); it != j.end()) {
        if (it->is_null()) {
            parsed.system_prompt = std::nullopt;
        } else {
            parsed.system_prompt = it->get<std::string>();
        }
    }
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

} // namespace zoo
