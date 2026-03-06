#pragma once

#include "types.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct llama_chat_message;

namespace zoo::core {

/**
 * Model is the direct llama.cpp wrapper.
 *
 * It manages model loading, tokenization, inference, prompt formatting,
 * KV cache state, and conversation history. It is usable standalone
 * without tools or agents.
 *
 * Thread safety: NOT thread-safe. All calls must come from the same thread.
 * For async usage, wrap in Agent (Layer 3).
 */
class Model {
public:
    static Expected<std::unique_ptr<Model>> load(const Config& config);

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    // Generate a response to a user message (high-level: manages history)
    Expected<Response> generate(
        const std::string& user_message,
        std::optional<std::function<void(std::string_view)>> on_token = std::nullopt
    );

    // Result from generate_from_history
    struct GenerationResult {
        std::string text;
        int prompt_tokens = 0;
    };

    // Generate from current history state (low-level: used by Agent for tool loop)
    Expected<GenerationResult> generate_from_history(
        std::optional<std::function<void(std::string_view)>> on_token = std::nullopt
    );

    // Update internal template state after committing messages
    void finalize_response();

    // History management
    void set_system_prompt(const std::string& prompt);
    Expected<void> add_message(const Message& message);
    std::vector<Message> get_history() const;
    void clear_history();

    // Context info
    int context_size() const;
    int estimated_tokens() const;
    bool is_context_exceeded() const;
    const Config& config() const { return config_; }

private:
    explicit Model(const Config& config);

    Expected<void> initialize();
    Expected<std::vector<int>> tokenize(const std::string& text);
    Expected<std::string> run_inference(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<std::function<void(std::string_view)>>& on_token = std::nullopt
    );
    Expected<std::string> format_prompt();
    void clear_kv_cache();

    static void initialize_global();
    llama_sampler* create_sampler_chain();
    size_t find_stop_sequence(const std::string& text,
                              const std::vector<std::string>& stop_sequences) const;
    std::vector<llama_chat_message> build_llama_messages() const;
    int estimate_tokens(const std::string& text) const;

    // Config
    Config config_;

    // llama.cpp state
    llama_model* llama_model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    const llama_vocab* vocab_ = nullptr;

    int context_size_ = 0;
    const char* tmpl_ = nullptr;

    // Incremental prompt state
    int prev_len_ = 0;
    std::vector<char> formatted_;

    // History state
    std::vector<Message> messages_;
    int estimated_tokens_ = 0;
    static constexpr int kTemplateOverheadPerMessage = 8;
};

} // namespace zoo::core
