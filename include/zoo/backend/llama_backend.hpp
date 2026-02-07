#pragma once

#include "IBackend.hpp"
#include "../types.hpp"
#include <string>
#include <vector>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct llama_chat_message;

namespace zoo {
namespace backend {

/**
 * @brief Production backend implementation using llama.cpp
 *
 * This class wraps llama.cpp for model loading, tokenization, and inference.
 * It manages the lifecycle of llama_model, llama_context, and llama_sampler.
 *
 * Thread Safety: All methods are designed to be called from a single inference thread.
 * The caller is responsible for ensuring thread-safe usage.
 */
class LlamaBackend : public IBackend {
public:
    LlamaBackend();
    ~LlamaBackend() override;

    // Disable copy and move to ensure RAII cleanup works correctly
    LlamaBackend(const LlamaBackend&) = delete;
    LlamaBackend& operator=(const LlamaBackend&) = delete;
    LlamaBackend(LlamaBackend&&) = delete;
    LlamaBackend& operator=(LlamaBackend&&) = delete;

    // IBackend interface implementation
    Expected<void> initialize(const Config& config) override;
    Expected<std::vector<int>> tokenize(const std::string& text) override;
    Expected<std::string> generate(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<std::function<void(std::string_view)>>& on_token = std::nullopt
    ) override;
    int get_kv_cache_token_count() const override;
    void clear_kv_cache() override;
    Expected<std::string> format_prompt(const std::vector<Message>& messages) override;
    void finalize_response(const std::vector<Message>& messages) override;
    int get_context_size() const override;
    int get_vocab_size() const override;

private:
    /**
     * @brief Check if a stop sequence has been reached
     * @param generated_text The currently generated text
     * @param stop_sequences List of stop sequences to check
     * @return true if any stop sequence is found at the end of generated_text
     */
    bool check_stop_sequence(const std::string& generated_text,
                            const std::vector<std::string>& stop_sequences) const;

    /**
     * @brief Create and configure the sampler chain from config
     * @param config The configuration containing sampling parameters
     * @return llama_sampler* Configured sampler chain (ownership transferred to caller)
     */
    llama_sampler* create_sampler_chain(const Config& config);

    /**
     * @brief Build llama_chat_message vector from zoo::Message list
     *
     * Points directly at source data — role_to_string() returns static const char*
     * and messages are passed by const& outliving the returned vector at every call site.
     *
     * @param messages Source messages (must outlive returned vector)
     * @return std::vector<llama_chat_message> Vector of llama chat messages
     */
    std::vector<llama_chat_message> build_llama_messages(
        const std::vector<Message>& messages) const;

    // llama.cpp state (owned)
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    llama_sampler* sampler_ = nullptr;
    const llama_vocab* vocab_ = nullptr;  // Retrieved from model, not owned

    // Configuration snapshot
    int context_size_ = 0;
    int vocab_size_ = 0;
    int kv_cache_token_count_ = 0;  // Track current KV cache usage

    // Cached chat template pointer (static lifetime from model, not owned)
    const char* tmpl_ = nullptr;

    // Prompt formatting state (mirrors simple-chat.cpp pattern)
    int prev_len_ = 0;                   // Length of previously formatted text
    std::vector<char> formatted_;         // Reusable formatted prompt buffer
};

} // namespace backend
} // namespace zoo
