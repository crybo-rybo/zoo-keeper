/**
 * @file model.hpp
 * @brief Public wrapper around llama.cpp model loading, prompting, and generation.
 */

#pragma once

#include "types.hpp"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/// Forward declarations for llama.cpp runtime types owned by `zoo::core::Model`.
struct llama_model;
struct llama_context;
struct llama_sampler;
struct llama_vocab;
struct llama_chat_message;
struct common_chat_templates;

namespace zoo::core {

struct ModelTestAccess;

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
     * @param model_config Runtime model configuration to validate and apply.
     * @param default_generation Default generation policy used when a call does
     *        not override it explicitly.
     * @return A fully initialized model, or an error if validation or backend
     *         setup fails.
     */
    static Expected<std::unique_ptr<Model>>
    load(const ModelConfig& model_config,
         const GenerationOptions& default_generation = GenerationOptions{});

    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    /**
     * @brief Generates an assistant response for a new user message.
     */
    Expected<TextResponse> generate(std::string_view user_message,
                                    const GenerationOptions& options = GenerationOptions{},
                                    TokenCallback on_token = {},
                                    CancellationCallback should_cancel = {});

    /**
     * @brief Generates an assistant response for a structured inbound message.
     */
    Expected<TextResponse> generate(MessageView message,
                                    const GenerationOptions& options = GenerationOptions{},
                                    TokenCallback on_token = {},
                                    CancellationCallback should_cancel = {});

    /**
     * @brief Result of a low-level generation pass started from existing history.
     */
    struct GenerationResult {
        std::string text;      ///< Raw generated text for the pass.
        int prompt_tokens = 0; ///< Number of prompt tokens rendered for the pass.
        bool tool_call_detected =
            false;                  ///< Whether tool calling detected a tool call in the output.
        std::string parsed_content; ///< Visible content after stripping tool syntax.
        std::vector<OwnedToolCall> tool_calls; ///< Structured tool calls extracted from the output.
    };

    /**
     * @brief Generates from the current history without appending a new user message.
     *
     * @note This method does not commit the assistant turn to history.
     */
    Expected<GenerationResult>
    generate_from_history(const GenerationOptions& options = GenerationOptions{},
                          TokenCallback on_token = {}, CancellationCallback should_cancel = {});

    /**
     * @brief Advances the incremental chat-template checkpoint to the current history.
     */
    void finalize_response();

    /**
     * @brief Sets or replaces the leading system prompt in the tracked history.
     */
    void set_system_prompt(std::string_view prompt);

    /**
     * @brief Appends a message to retained history after validating role sequencing.
     */
    Expected<void> add_message(MessageView message);

    /// Returns an owning snapshot of the current conversation history.
    [[nodiscard]] HistorySnapshot get_history() const;

    /// Clears conversation history, token estimates, and cached KV state.
    void clear_history();

    void trim_history(size_t max_non_system_messages);

    /**
     * @brief Replaces the retained message history without flushing the KV cache.
     */
    void replace_history(HistorySnapshot snapshot);

    /**
     * @brief Atomically swaps retained history with a provided snapshot.
     */
    [[nodiscard]] HistorySnapshot swap_history(HistorySnapshot snapshot);

    /**
     * @brief Configures template-driven tool calling from registered tool metadata.
     */
    bool set_tool_calling(const std::vector<CoreToolInfo>& tools);

    /**
     * @brief Enables grammar-constrained schema output for future generations.
     */
    bool set_schema_grammar(const std::string& grammar_str);

    /// Disables any active grammar/tool calling and restores the default sampler chain.
    void clear_tool_grammar() noexcept;

    [[nodiscard]] bool has_tool_calling() const noexcept;
    [[nodiscard]] bool has_schema_grammar() const noexcept;

    /**
     * @brief Parses a generated text into structured content and tool calls.
     */
    struct ParsedResponse {
        std::string content;
        std::vector<OwnedToolCall> tool_calls;
    };
    [[nodiscard]] ParsedResponse parse_tool_response(std::string_view text) const;

    [[nodiscard]] const char* tool_calling_format_name() const noexcept;
    [[nodiscard]] int context_size() const noexcept;
    [[nodiscard]] int estimated_tokens() const noexcept;
    [[nodiscard]] bool is_context_exceeded() const noexcept;
    [[nodiscard]] const ModelConfig& model_config() const noexcept;
    [[nodiscard]] const GenerationOptions& default_generation_options() const noexcept;

  private:
    friend struct ModelTestAccess;

    struct Impl;

    struct LlamaModelDeleter {
        void operator()(llama_model* model) const noexcept;
    };
    struct LlamaContextDeleter {
        void operator()(llama_context* context) const noexcept;
    };
    struct LlamaSamplerDeleter {
        void operator()(llama_sampler* sampler) const noexcept;
    };
    struct ChatTemplatesDeleter {
        void operator()(common_chat_templates* tmpls) const noexcept;
    };

    using LlamaModelHandle = std::unique_ptr<llama_model, LlamaModelDeleter>;
    using LlamaContextHandle = std::unique_ptr<llama_context, LlamaContextDeleter>;
    using LlamaSamplerHandle = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;
    using ChatTemplatesHandle = std::unique_ptr<common_chat_templates, ChatTemplatesDeleter>;

    explicit Model(ModelConfig model_config, GenerationOptions default_generation);

    Expected<void> initialize();
    Expected<std::vector<int>> tokenize(std::string_view text);
    Expected<std::string> run_inference(const std::vector<int>& prompt_tokens, int max_tokens,
                                        const std::vector<std::string>& stop_sequences,
                                        TokenCallback on_token = {},
                                        CancellationCallback should_cancel = {});
    Expected<std::string> render_prompt_delta();
    void clear_kv_cache();
    void note_history_append() noexcept;
    void note_history_rewrite() noexcept;
    void note_history_reset() noexcept;
    static void initialize_global();
    LlamaSamplerHandle create_sampler_chain();
    void add_sampling_stages(llama_sampler* chain, const SamplingParams& sampling) const;
    void add_dist_sampler(llama_sampler* chain, const SamplingParams& sampling) const;
    bool rebuild_sampler_with_tool_grammar();
    bool rebuild_sampler_with_schema_grammar();
    Expected<void> ensure_grammar_sampler_for_pass();
    [[nodiscard]] std::vector<std::string>
    merge_stop_sequences(std::vector<std::string> base) const;
    [[nodiscard]] int estimate_tokens(std::string_view text) const;
    [[nodiscard]] int estimate_message_tokens(const Message& message) const;
    void trim_history_to_fit();
    void rollback_last_message() noexcept;
    [[nodiscard]] GenerationOptions
    resolve_generation_options(const GenerationOptions& overrides) const;

    std::unique_ptr<Impl> impl_;
};

} // namespace zoo::core
