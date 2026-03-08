/**
 * @file model.hpp
 * @brief Public wrapper around llama.cpp model loading, prompting, and generation.
 */

#pragma once

#include "types.hpp"
#include <memory>
#include <vector>
#include <string>

/// Forward declarations for llama.cpp runtime types owned by `zoo::core::Model`.
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct llama_chat_message;

namespace zoo::core {

/**
 * @brief Direct llama.cpp wrapper for model lifecycle, history, and generation.
 *
 * `Model` owns the llama.cpp model, context, sampler chain, and incremental
 * chat-template state. It can be used standalone, but it is intentionally not
 * thread-safe; callers that need concurrency should use `zoo::Agent`.
 */
class Model {
public:
    /**
     * @brief Loads and initializes a model from the supplied configuration.
     *
     * @param config Runtime configuration to validate and apply.
     * @return A fully initialized model, or an error if validation or backend
     *         setup fails.
     */
    static Expected<std::unique_ptr<Model>> load(const Config& config);

    /// Releases all llama.cpp resources owned by the model.
    ~Model();
    /// Models own non-shareable backend state and cannot be copied.
    Model(const Model&) = delete;
    /// Models own non-shareable backend state and cannot be copied.
    Model& operator=(const Model&) = delete;
    /// Models remain bound to internal llama.cpp pointers and cannot be moved.
    Model(Model&&) = delete;
    /// Models remain bound to internal llama.cpp pointers and cannot be moved.
    Model& operator=(Model&&) = delete;

    /**
     * @brief Generates an assistant response for a new user message.
     *
     * This high-level entry point appends the user message, renders the prompt,
     * performs inference, appends the assistant response to history, and returns
     * usage plus timing metrics.
     *
     * @param user_message User-authored content to append before generation.
     * @param on_token Optional callback invoked for streamed token fragments.
     * @return Completed assistant response or an error.
     */
    Expected<Response> generate(
        const std::string& user_message,
        std::optional<TokenCallback> on_token = std::nullopt
    );

    /**
     * @brief Result of a low-level generation pass started from existing history.
     */
    struct GenerationResult {
        std::string text; ///< Raw generated text for the pass.
        int prompt_tokens = 0; ///< Number of prompt tokens rendered for the pass.
        bool tool_call_detected = false; ///< Whether sentinel-based grammar mode emitted a tool call.
    };

    /**
     * @brief Generates from the current history without appending a new user message.
     *
     * This lower-level entry point is used by the agent tool loop so it can
     * alternate between assistant generations and injected tool results.
     *
     * @param on_token Optional callback invoked for streamed token fragments.
     * @return Raw generation output plus prompt-token count and tool-call signal.
     */
    Expected<GenerationResult> generate_from_history(
        std::optional<TokenCallback> on_token = std::nullopt
    );

    /**
     * @brief Advances the incremental chat-template checkpoint to the current history.
     *
     * Call this after committing assistant or tool messages that should become
     * part of the stable prompt prefix for subsequent generations.
     */
    void finalize_response();

    /// Sets or replaces the leading system prompt in the tracked conversation history.
    void set_system_prompt(const std::string& prompt);
    /**
     * @brief Appends a message to history after validating role sequencing.
     *
     * @param message Message to append.
     * @return Empty success when appended, or an error if the role ordering is invalid.
     */
    Expected<void> add_message(const Message& message);
    /// Returns a copy of the current conversation history.
    std::vector<Message> get_history() const;
    /// Clears conversation history, token estimates, and cached KV state.
    void clear_history();

    /**
     * @brief Enables grammar-constrained tool calling for future generations.
     *
     * @param grammar_str GBNF grammar string rooted at `root`.
     * @return `true` when the sampler chain was rebuilt successfully.
     */
    bool set_tool_grammar(const std::string& grammar_str);
    /// Disables grammar-constrained tool calling and restores the default sampler chain.
    void clear_tool_grammar();
    /// Returns whether tool grammar constraints are currently active.
    bool has_tool_grammar() const { return grammar_active_; }

    /// Returns the configured context window size.
    int context_size() const;
    /// Returns the running estimate of tokens stored in conversation history.
    int estimated_tokens() const;
    /// Returns whether the estimated history size exceeds the configured context window.
    bool is_context_exceeded() const;
    /// Returns the immutable configuration used to construct the model.
    const Config& config() const { return config_; }

private:
    /// Constructs an uninitialized model wrapper. Call `initialize()` before use.
    explicit Model(const Config& config);

    /// Performs one-time backend and per-instance llama.cpp initialization.
    Expected<void> initialize();
    /**
     * @brief Tokenizes raw text using the loaded vocabulary.
     *
     * @param text Prompt fragment to tokenize.
     * @return Token IDs ready for llama.cpp decoding.
     */
    Expected<std::vector<int>> tokenize(const std::string& text);
    /**
     * @brief Runs prompt prefill followed by autoregressive generation.
     *
     * @param prompt_tokens Tokens representing the rendered prompt delta.
     * @param max_tokens Maximum completion length to allow for this pass.
     * @param stop_sequences Stop sequences that terminate generation when matched.
     * @param on_token Optional streaming callback invoked for visible token pieces.
     * @return The generated text for the pass.
     */
    Expected<std::string> run_inference(
        const std::vector<int>& prompt_tokens,
        int max_tokens,
        const std::vector<std::string>& stop_sequences,
        const std::optional<TokenCallback>& on_token = std::nullopt
    );
    /// Renders the current conversation history through the active chat template.
    Expected<std::string> format_prompt();
    /// Clears cached KV memory and resets incremental prompt bookkeeping.
    void clear_kv_cache();

    /// Performs process-wide llama.cpp backend initialization exactly once.
    static void initialize_global();
    /// Builds the default sampler chain from the configured sampling parameters.
    llama_sampler* create_sampler_chain();
    /// Rebuilds the sampler chain so a lazy grammar activates on `<tool_call>`.
    bool rebuild_sampler_with_grammar();
    /// Returns the length of a matching stop sequence suffix, or zero if none match.
    size_t find_stop_sequence(const std::string& text,
                              const std::vector<std::string>& stop_sequences) const;
    /// Converts tracked messages into the structure expected by llama.cpp chat templates.
    std::vector<llama_chat_message> build_llama_messages() const;
    /// Estimates token count for bookkeeping when exact prompt rendering is unavailable.
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

    // Tool grammar state
    std::string tool_grammar_str_;
    bool grammar_active_ = false;

    // Incremental prompt state
    int prev_len_ = 0;
    std::vector<char> formatted_;

    // History state
    std::vector<Message> messages_;
    int estimated_tokens_ = 0;
    static constexpr int kTemplateOverheadPerMessage = 8;
};

} // namespace zoo::core
