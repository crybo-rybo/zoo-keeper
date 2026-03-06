#pragma once

#include "types.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct llama_chat_message;

namespace zoo::core {

// Internal backend interface for dependency injection (testing)
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual Expected<void> initialize(const Config& config) = 0;
    virtual Expected<std::vector<int>> tokenize(const std::string& text) = 0;
    virtual Expected<std::string> generate(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<std::function<void(std::string_view)>>& on_token = std::nullopt
    ) = 0;
    virtual int get_context_size() const = 0;
    virtual void clear_kv_cache() = 0;
    virtual Expected<std::string> format_prompt(const std::vector<Message>& messages) = 0;
    virtual void finalize_response(const std::vector<Message>& messages) = 0;
};

// Factory function for creating the llama.cpp backend (defined in model.cpp)
std::unique_ptr<IBackend> create_llama_backend();

/**
 * Model is a synchronous, single-threaded llama.cpp wrapper.
 *
 * It manages model loading, conversation history, prompt formatting,
 * tokenization, inference, and KV cache state. It is usable standalone
 * without tools or agents.
 *
 * Thread safety: NOT thread-safe. All calls must come from the same thread.
 * For async usage, wrap in Agent (Layer 3).
 */
class Model {
public:
    static Expected<std::unique_ptr<Model>> load(
        const Config& config,
        std::unique_ptr<IBackend> backend = nullptr
    );

    ~Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = default;
    Model& operator=(Model&&) = default;

    // Generate a response to a user message
    Expected<Response> generate(
        const std::string& user_message,
        std::optional<std::function<void(std::string_view)>> on_token = std::nullopt
    );

    // Lower-level: generate from current history state (used by Agent for tool loop)
    Expected<std::string> generate_from_history(
        std::optional<std::function<void(std::string_view)>> on_token = std::nullopt
    );

    // History management
    void set_system_prompt(const std::string& prompt);
    Expected<void> add_message(const Message& message);
    std::vector<Message> get_history() const;
    void clear_history();

    // Context info
    int context_size() const;
    int estimated_tokens() const;
    bool is_context_exceeded() const;

    // Access to backend (for Agent layer)
    IBackend& backend() { return *backend_; }
    const Config& config() const { return config_; }

private:
    Model(const Config& config, std::unique_ptr<IBackend> backend);

    Config config_;
    std::unique_ptr<IBackend> backend_;

    // History state
    std::vector<Message> messages_;
    int estimated_tokens_ = 0;
    static constexpr int kTemplateOverheadPerMessage = 8;

    // Token estimation
    int estimate_tokens(const std::string& text) const;
    Expected<void> validate_role_sequence(Role role) const;
};

} // namespace zoo::core
