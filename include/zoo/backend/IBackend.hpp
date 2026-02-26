#pragma once

#include "../types.hpp"
#include <string>
#include <vector>
#include <memory>

namespace zoo {
namespace backend {

/**
 * @brief Abstract interface for LLM backend implementations
 *
 * This interface abstracts llama.cpp and enables dependency injection for testing.
 * All backend implementations must be thread-safe for the inference thread.
 *
 * Design principles:
 * - Single-threaded: All methods called from inference thread only
 * - Stateful: Backend owns llama_context and maintains KV cache state
 * - Synchronous: All operations block until complete
 */
class IBackend {
public:
    virtual ~IBackend() = default;

    /**
     * @brief Initialize the backend with configuration
     *
     * Loads model, creates context, allocates KV cache.
     * Called once during Agent construction.
     *
     * @param config Agent configuration
     * @return Expected<void> Success or initialization error
     */
    virtual Expected<void> initialize(const Config& config) = 0;

    /**
     * @brief Tokenize a string into token IDs
     *
     * @param text Input text to tokenize
     * @return Expected<std::vector<int>> Token IDs or error
     */
    virtual Expected<std::vector<int>> tokenize(const std::string& text) = 0;

    /**
     * @brief Generate completion given a prompt
     *
     * Performs synchronous inference with streaming callbacks.
     *
     * @param prompt_tokens Tokenized prompt (includes conversation history)
     * @param max_tokens Maximum tokens to generate
     * @param stop_sequences Sequences that stop generation
     * @param on_token Optional callback for each generated token (UTF-8 string)
     * @return Expected<std::string> Generated text or error
     */
    virtual Expected<std::string> generate(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<std::function<void(std::string_view)>>& on_token = std::nullopt
    ) = 0;

    /**
     * @brief Get current KV cache token count
     *
     * Returns the number of tokens currently in the KV cache.
     *
     * @return int Number of cached tokens
     */
    virtual int get_kv_cache_token_count() const = 0;

    /**
     * @brief Clear KV cache
     *
     * Removes all cached tokens, forcing re-evaluation on next generation.
     */
    virtual void clear_kv_cache() = 0;

    /**
     * @brief Format the prompt with the model's chat template
     *
     * Applies the chat template to the full message history and returns only the
     * incremental portion (new text since last call). This enables KV cache reuse
     * across turns -- only the new tokens need to be decoded.
     *
     * Maintains internal state (prev_len, formatted buffer) that tracks what has
     * already been processed. Must be paired with finalize_response() after generation.
     *
     * @param messages Full conversation history
     * @return Expected<std::string> Incremental prompt text or error
     */
    virtual Expected<std::string> format_prompt(const std::vector<Message>& messages) = 0;

    /**
     * @brief Update prompt cache state after assistant response is added
     *
     * Must be called after adding the assistant's response to history.
     * Updates internal prev_len so the next format_prompt() call returns
     * only the new user message portion.
     *
     * @param messages Full conversation history (including assistant response)
     */
    virtual void finalize_response(const std::vector<Message>& messages) = 0;

    /**
     * @brief Get model context size
     *
     * @return int Context window size in tokens
     */
    virtual int get_context_size() const = 0;

    /**
     * @brief Get model training context size from GGUF metadata
     *
     * Returns the context size the model was trained with, as recorded in its
     * GGUF metadata. This may differ from the runtime context size configured
     * in Config::context_size.
     *
     * @return int Training context size in tokens, or 0 if unavailable
     */
    virtual int get_training_context_size() const = 0;

    /**
     * @brief Get model vocabulary size
     *
     * @return int Number of tokens in vocabulary
     */
    virtual int get_vocab_size() const = 0;
};

/**
 * @brief Factory function for creating backend instances
 *
 * Returns a production LlamaBackend implementation.
 * For testing, inject MockBackend directly.
 *
 * @return std::unique_ptr<IBackend> Backend instance
 */
std::unique_ptr<IBackend> create_backend();

} // namespace backend
} // namespace zoo
