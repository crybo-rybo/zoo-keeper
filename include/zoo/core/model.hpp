/**
 * @file model.hpp
 * @brief Public wrapper around llama.cpp model loading, prompting, and generation.
 */

#pragma once

#include "types.hpp"
#include <memory>
#include <string>
#include <vector>

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
     * @param should_cancel Optional callback queried each decode step; returning
     *        `true` terminates generation with a `RequestCancelled` error.
     * @return Completed assistant response or an error.
     */
    Expected<Response> generate(const std::string& user_message,
                                std::optional<TokenCallback> on_token = std::nullopt,
                                CancellationCallback should_cancel = {});

    /**
     * @brief Result of a low-level generation pass started from existing history.
     */
    struct GenerationResult {
        std::string text;      ///< Raw generated text for the pass.
        int prompt_tokens = 0; ///< Number of prompt tokens rendered for the pass.
        bool tool_call_detected =
            false; ///< Whether sentinel-based grammar mode emitted a tool call.
    };

    /**
     * @brief Generates from the current history without appending a new user message.
     *
     * This lower-level entry point is used by the agent tool loop so it can
     * alternate between assistant generations and injected tool results.
     *
     * @param on_token Optional callback invoked for streamed token fragments.
     * @param should_cancel Optional callback queried each decode step; returning
     *        `true` terminates generation with a cancellation signal.
     * @return Raw generation output plus prompt-token count and tool-call signal.
     */
    Expected<GenerationResult>
    generate_from_history(std::optional<TokenCallback> on_token = std::nullopt,
                          CancellationCallback should_cancel = {});

    /**
     * @brief Advances the incremental chat-template checkpoint to the current history.
     *
     * Call this after committing assistant or tool messages that should become
     * part of the stable prompt prefix for subsequent generations.
     */
    void finalize_response();

    /**
     * @brief Sets or replaces the leading system prompt in the tracked conversation history.
     *
     * @param prompt New system prompt content.
     */
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
    void clear_tool_grammar() noexcept;
    /// Returns whether tool grammar constraints are currently active.
    bool has_tool_grammar() const noexcept {
        return grammar_active_;
    }

    /// Returns the configured context window size.
    int context_size() const noexcept;
    /// Returns the running estimate of tokens stored in conversation history.
    int estimated_tokens() const noexcept;
    /// Returns whether the estimated history size exceeds the configured context window.
    bool is_context_exceeded() const noexcept;
    /// Returns the immutable configuration used to construct the model.
    const Config& config() const noexcept {
        return config_;
    }

  private:
    struct LlamaModelDeleter {
        void operator()(llama_model* model) const noexcept;
    };
    struct LlamaContextDeleter {
        void operator()(llama_context* context) const noexcept;
    };
    struct LlamaSamplerDeleter {
        void operator()(llama_sampler* sampler) const noexcept;
    };

    using LlamaModelHandle = std::unique_ptr<llama_model, LlamaModelDeleter>;
    using LlamaContextHandle = std::unique_ptr<llama_context, LlamaContextDeleter>;
    using LlamaSamplerHandle = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;

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
    Expected<std::string> run_inference(const std::vector<int>& prompt_tokens, int max_tokens,
                                        const std::vector<std::string>& stop_sequences,
                                        const std::optional<TokenCallback>& on_token = std::nullopt,
                                        const CancellationCallback& should_cancel = {});
    /// Renders the current conversation history through the active chat template.
    Expected<std::string> render_prompt_delta();
    /// Clears cached KV memory and resets incremental prompt bookkeeping.
    void clear_kv_cache();
    /// Marks cached llama message views dirty after appending new history.
    void note_history_append() noexcept;
    /// Invalidates committed prompt state after rewriting retained history.
    void note_history_rewrite() noexcept;
    /// Resets all incremental prompt state after clearing retained history.
    void note_history_reset() noexcept;

    /// Performs process-wide llama.cpp backend initialization exactly once.
    static void initialize_global();
    /// Builds the default sampler chain from the configured sampling parameters.
    LlamaSamplerHandle create_sampler_chain();
    /// Adds penalty, top-k, top-p, and temperature samplers to an existing chain.
    void add_sampling_stages(llama_sampler* chain) const;
    /// Adds a distribution or greedy sampler as the final selection stage.
    void add_dist_sampler(llama_sampler* chain) const;
    /// Rebuilds the sampler chain so a lazy grammar activates on `<tool_call>`.
    bool rebuild_sampler_with_grammar();
    /// Returns the length of a matching stop sequence suffix, or zero if none match.
    size_t find_stop_sequence(const std::string& text,
                              const std::vector<std::string>& stop_sequences) const;
    /// Returns cached llama.cpp chat messages, rebuilding only when history has changed.
    const std::vector<llama_chat_message>& llama_messages();
    /// Estimates token count for bookkeeping when exact prompt rendering is unavailable.
    int estimate_tokens(const std::string& text) const;
    /// Trims the oldest retained conversation state to the configured history budget.
    void trim_history_to_fit();

    struct PromptState {
        int committed_prompt_len = 0;
        std::vector<char> formatted_prompt;
        std::vector<llama_chat_message> cached_llama_messages;
        bool cached_messages_dirty = true;
    };

    // Config
    Config config_;

    // llama.cpp state
    LlamaModelHandle llama_model_;
    LlamaContextHandle ctx_;
    LlamaSamplerHandle sampler_;
    const llama_vocab* vocab_ = nullptr;

    int context_size_ = 0;
    const char* tmpl_ = nullptr;

    // Tool grammar state
    std::string tool_grammar_str_;
    bool grammar_active_ = false;

    // Incremental prompt state
    PromptState prompt_state_;

    // History state
    std::vector<Message> messages_;
    int estimated_tokens_ = 0;
    static constexpr int kTemplateOverheadPerMessage = 8;
};

} // namespace zoo::core
